import gc
import random

import numpy as np
import pytest
import torch
import torch_npu

torch_npu.npu.set_compile_mode(jit_compile=False)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def golden_recurrent_gated_delta_rule(
    query,
    key,
    value,
    state,
    beta,
    scale,
    actual_seq_lengths,
    ssm_state_indices,
    g,
    num_accepted_tokens,
):
    """Pure torch/CPU golden implementation of recurrent gated delta rule.

    Args:
        query: [T, nk, dk]
        key: [T, nk, dk]
        value: [T, nv, dv]
        state: [S, nv, dv, dk]
        beta: [T, nv]
        scale: float
        actual_seq_lengths: [batch_size] per-sequence lengths
        ssm_state_indices: [T] per-token state block index, or
            [batch_size, state_width] fixed per-request state rows
        g: [T, nv] or None
        num_accepted_tokens: [batch_size] or None

    Returns:
        (output [T, nv, dv], updated_state [S, nv, dv, dk])
    """
    q = query.to(torch.float32)
    k = key.to(torch.float32)
    v = value.to(torch.float32)
    initial_state = state.clone().to(torch.float32)
    T, n_heads_v, Dv = v.shape
    n_heads_k = q.shape[-2]
    g = torch.ones(T, n_heads_v).to(torch.float32) if g is None else g.to(torch.float32).exp()
    beta = torch.ones(T, n_heads_v).to(torch.float32) if beta is None else beta.to(torch.float32)
    o = torch.empty_like(v).to(torch.float32)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    q = q * scale

    seq_start = 0
    fixed_state_rows = ssm_state_indices.dim() == 2
    for i in range(len(actual_seq_lengths)):
        if num_accepted_tokens is None:
            init_state_idx = ssm_state_indices[i, 0] if fixed_state_rows else ssm_state_indices[seq_start]
        else:
            accepted_idx = num_accepted_tokens[i] - 1
            init_state_idx = (
                ssm_state_indices[i, accepted_idx]
                if fixed_state_rows
                else ssm_state_indices[seq_start + accepted_idx]
            )
        init_state = initial_state[init_state_idx]
        for head_id in range(n_heads_v):
            S = init_state[head_id]
            for local_idx, slot_id in enumerate(range(seq_start, seq_start + actual_seq_lengths[i])):
                q_i = q[slot_id][head_id // (n_heads_v // n_heads_k)]
                k_i = k[slot_id][head_id // (n_heads_v // n_heads_k)]
                v_i = v[slot_id][head_id]
                alpha_i = g[slot_id][head_id]
                beta_i = beta[slot_id][head_id]
                S = S * alpha_i
                x = (S * k_i.unsqueeze(-2)).sum(dim=-1)
                y = (v_i - x) * beta_i
                S_ = y[:, None] * k_i[None, :]
                S = S + S_
                state_idx = ssm_state_indices[i, local_idx] if fixed_state_rows else ssm_state_indices[slot_id]
                initial_state[state_idx][head_id] = S
                o[slot_id][head_id] = (S * q_i.unsqueeze(-2)).sum(dim=-1)
        seq_start += actual_seq_lengths[i]

    return o.to(query.dtype), initial_state.to(query.dtype)


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("mtp", [1, 2])
@pytest.mark.parametrize("headnum", [(4, 8), (8, 16), (16, 32)])
@pytest.mark.parametrize("headdim_k", [128])
@pytest.mark.parametrize("headdim_v", [128])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_recurrent_gated_delta_rule(
    batch_size,
    mtp,
    headnum,
    headdim_k,
    headdim_v,
    state_dtype,
):
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    headnum_k, headnum_v = headnum
    seq_lengths = torch.ones(batch_size, dtype=torch.int32) * mtp
    T = int(torch.sum(seq_lengths))

    state = torch.rand((T, headnum_v, headdim_v, headdim_k)).to(state_dtype)
    query = torch.nn.functional.normalize(
        torch.rand((T, headnum_k, headdim_k)),
        p=2,
        dim=-1,
    ).to(dtype)
    key = torch.nn.functional.normalize(
        torch.rand((T, headnum_k, headdim_k)),
        p=2,
        dim=-1,
    ).to(dtype)
    value = torch.rand((T, headnum_v, headdim_v)).to(dtype)
    g = torch.rand((T, headnum_v), dtype=torch.float32)
    beta = torch.rand((T, headnum_v)).to(dtype)
    ssm_state_indices = torch.arange(T, dtype=torch.int32)
    num_accepted_tokens = torch.randint(1, mtp + 1, (batch_size,), dtype=torch.int32)
    scale = headdim_k**-0.5

    out_golden, state_golden = golden_recurrent_gated_delta_rule(
        query,
        key,
        value,
        state,
        beta,
        scale,
        seq_lengths,
        ssm_state_indices,
        g,
        num_accepted_tokens,
    )
    out_golden = out_golden.to(torch.float32)
    state_golden = state_golden.to(torch.float32)

    # torch_npu op expects actual_seq_lengths = [start_pos, len1, len2, ..., lenB]
    actual_seq_lengths_npu = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            seq_lengths,
        ]
    )

    state_npu = state.npu()
    npu_out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
        query=query.npu(),
        key=key.npu(),
        value=value.npu(),
        g=g.npu(),
        beta=beta.npu(),
        state=state_npu,
        scale=scale,
        actual_seq_lengths=actual_seq_lengths_npu.npu(),
        ssm_state_indices=ssm_state_indices.npu(),
        num_accepted_tokens=num_accepted_tokens.npu(),
    )

    torch.testing.assert_close(
        npu_out.to(torch.float32).cpu(),
        out_golden,
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )
    torch.testing.assert_close(
        state_npu.to(torch.float32).cpu(),
        state_golden,
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def test_recurrent_gated_delta_rule_fixed_state_rows():
    """A shorter verification row still selects its previous accepted state."""
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    batch_size = 2
    state_width = 4
    headnum_k, headnum_v = 4, 8
    headdim_k = headdim_v = 128
    seq_lengths = torch.tensor([2, 1], dtype=torch.int32)
    T = int(seq_lengths.sum())
    state_rows = batch_size * state_width

    state = torch.rand((state_rows, headnum_v, headdim_v, headdim_k), dtype=torch.float32).to(dtype)
    query = torch.nn.functional.normalize(torch.rand((T, headnum_k, headdim_k)), p=2, dim=-1).to(dtype)
    key = torch.nn.functional.normalize(torch.rand((T, headnum_k, headdim_k)), p=2, dim=-1).to(dtype)
    value = torch.rand((T, headnum_v, headdim_v)).to(dtype)
    g = torch.rand((T, headnum_v), dtype=torch.float32)
    beta = torch.rand((T, headnum_v)).to(dtype)
    ssm_state_indices = torch.arange(state_rows, dtype=torch.int32).view(batch_size, state_width)
    # The first request's previous accepted count (3) intentionally exceeds
    # its current verification width (2).
    num_accepted_tokens = torch.tensor([3, 2], dtype=torch.int32)
    scale = headdim_k**-0.5

    out_golden, state_golden = golden_recurrent_gated_delta_rule(
        query,
        key,
        value,
        state,
        beta,
        scale,
        seq_lengths,
        ssm_state_indices,
        g,
        num_accepted_tokens,
    )
    out_golden = out_golden.to(torch.float32)
    state_golden = state_golden.to(torch.float32)

    actual_seq_lengths_npu = torch.cat([torch.zeros(1, dtype=torch.int32), seq_lengths])
    state_npu = state.npu()
    npu_out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
        query=query.npu(),
        key=key.npu(),
        value=value.npu(),
        g=g.npu(),
        beta=beta.npu(),
        state=state_npu,
        scale=scale,
        actual_seq_lengths=actual_seq_lengths_npu.npu(),
        ssm_state_indices=ssm_state_indices.npu(),
        num_accepted_tokens=num_accepted_tokens.npu(),
    )

    torch.testing.assert_close(
        npu_out.to(torch.float32).cpu(),
        out_golden,
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )
    torch.testing.assert_close(
        state_npu.to(torch.float32).cpu(),
        state_golden,
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("mtp", [1, 2])
@pytest.mark.parametrize("headnum", [(4, 8), (8, 16)])
@pytest.mark.parametrize("headdim_k", [128])
@pytest.mark.parametrize("headdim_v", [128])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_recurrent_gated_delta_rule_no_accepted(
    batch_size,
    mtp,
    headnum,
    headdim_k,
    headdim_v,
    state_dtype,
):
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    headnum_k, headnum_v = headnum
    seq_lengths = torch.ones(batch_size, dtype=torch.int32) * mtp
    T = int(torch.sum(seq_lengths))

    state = torch.rand((T, headnum_v, headdim_v, headdim_k)).to(state_dtype)
    query = torch.nn.functional.normalize(
        torch.rand((T, headnum_k, headdim_k)),
        p=2,
        dim=-1,
    ).to(dtype)
    key = torch.nn.functional.normalize(
        torch.rand((T, headnum_k, headdim_k)),
        p=2,
        dim=-1,
    ).to(dtype)
    value = torch.rand((T, headnum_v, headdim_v)).to(dtype)
    g = torch.rand((T, headnum_v), dtype=torch.float32)
    beta = torch.rand((T, headnum_v)).to(dtype)
    ssm_state_indices = torch.arange(T, dtype=torch.int32)
    scale = headdim_k**-0.5

    out_golden, state_golden = golden_recurrent_gated_delta_rule(
        query,
        key,
        value,
        state,
        beta,
        scale,
        seq_lengths,
        ssm_state_indices,
        g,
        None,
    )
    out_golden = out_golden.to(torch.float32)
    state_golden = state_golden.to(torch.float32)

    actual_seq_lengths_npu = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            seq_lengths,
        ]
    )

    state_npu = state.npu()
    npu_out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
        query=query.npu(),
        key=key.npu(),
        value=value.npu(),
        g=g.npu(),
        beta=beta.npu(),
        state=state_npu,
        scale=scale,
        actual_seq_lengths=actual_seq_lengths_npu.npu(),
        ssm_state_indices=ssm_state_indices.npu(),
    )

    torch.testing.assert_close(
        npu_out.to(torch.float32).cpu(),
        out_golden,
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )
    torch.testing.assert_close(
        state_npu.to(torch.float32).cpu(),
        state_golden,
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
