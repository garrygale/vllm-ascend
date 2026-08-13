#!/usr/bin/env python3
# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""ACL-graph probe for Domino GRU loop fusion combinations.

The service's greedy 15-step GRU loop per step is:

  ``W_sh linear -> z+s -> silu -> embed_proj2 matmul -> base+bias ->
  argmax -> gi_table[token] -> GRU cell``

This probe times the full 15-step loop (graph replay only) for every
feasible fusion combination at B = 32 / 64:

  * ``zsilu``: fuse ``z+s`` + ``silu`` into one Triton kernel,
  * ``bias_argmax``: fuse ``base+bias`` + ``argmax`` into one Triton
    kernel (per-block argmax + tiny block reduce),
  * ``gather_cell``: fuse ``gi_table[token]`` + GRU cell into one Triton
    kernel,
  * ``wsh_zsilu`` (experimental): fuse the ``W_sh`` matmul (``tl.dot``)
    + ``z+s`` + ``silu`` into one Triton kernel.  May hit the NPU UB
    limit; a FAIL row is informative.

Variants are the individual fusions and all meaningful combinations.
The per-step kernel count and us/step are printed per variant.  Every
variant is also checked for correctness against the baseline:

  * unit checks: ``zsilu`` vs ``F.silu(z+s)`` (max diff, bf16 mismatch,
    downstream argmax flips) and fused ``gather_cell`` vs
    ``cell(table[token])`` (must be bit-identical),
  * loop trace: per-step draft-token equality and final hidden-state
    max diff vs the baseline loop (eager, no graph).

Run directly on an NPU:
    python benchmarks/probe_gru_loop_fusions.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
import torch_npu
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.spec_decode.domino_gru import (
    domino_gru_cell_triton,
)
from vllm_ascend.ops.triton.triton_utils import (
    init_device_properties_triton,
)

H = 4096      # target hidden size (z dim)
G = 1024      # GRU hidden dim
M = 256       # emb_dim (correction hidden)
V = 151936    # vocab size
N_SPEC = 15
MS = (32, 64)
ITERS = 50
WARMUP = 10
DTYPE = torch.bfloat16


@triton.jit
def _zsilu_kernel(z_ptr, s_ptr, out_ptr, M, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_M)
    mask = offs < M
    z = tl.load(z_ptr + pid * M + offs, mask=mask, care_padding=False)
    s = tl.load(s_ptr + pid * M + offs, mask=mask, care_padding=False)
    x = z + s
    out = tl.sigmoid(x) * x
    tl.store(out_ptr + pid * M + offs, out.to(z_ptr.dtype.element_ty),
             mask=mask)


@triton.jit
def _bias_argmax_kernel(
    base_ptr, bias_ptr, argmax_ptr, max_ptr, V, nblocks,
    BLOCK_V: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_v = tl.program_id(1)
    offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = offs < V
    base = tl.load(base_ptr + pid_b * V + offs, mask=mask,
                   other=float("-inf"))
    bias = tl.load(bias_ptr + pid_b * V + offs, mask=mask,
                   other=float("-inf"))
    logits = base.to(tl.float32) + bias.to(tl.float32)
    logits = tl.where(mask, logits, float("-inf"))
    idx = tl.argmax(logits, axis=0)
    val = tl.max(logits, axis=0)
    tl.store(argmax_ptr + pid_b * nblocks + pid_v, pid_v * BLOCK_V + idx)
    tl.store(max_ptr + pid_b * nblocks + pid_v, val)


@triton.jit
def _block_argmax_kernel(
    argmax_ptr, max_ptr, token_ptr, nblocks, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < nblocks
    vals = tl.load(max_ptr + pid * nblocks + offs, mask=mask,
                   other=float("-inf"))
    vals = tl.where(mask, vals, float("-inf"))
    bidx = tl.argmax(vals, axis=0)
    token = tl.load(argmax_ptr + pid * nblocks + bidx)
    tl.store(token_ptr + pid, token)


@triton.jit
def _cell_gather_kernel(
    table_ptr, tokens_ptr, gh_ptr, h_ptr, h_out_ptr,
    B, G,
    stride_tok, stride_gh_b, stride_gh_g, stride_h_b, stride_h_g,
    stride_hout_b, stride_hout_g,
    BLOCK_G: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)
    offs_g = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    mask_g = offs_g < G

    token = tl.load(tokens_ptr + pid_b * stride_tok)
    gi_base = token * (3 * G)

    h_state = tl.load(
        h_ptr + pid_b * stride_h_b + offs_g * stride_h_g,
        mask=mask_g, care_padding=False,
    )
    gi_r = tl.load(table_ptr + gi_base + offs_g, mask=mask_g,
                   care_padding=False)
    gi_z = tl.load(table_ptr + gi_base + G + offs_g, mask=mask_g,
                   care_padding=False)
    gi_n = tl.load(table_ptr + gi_base + 2 * G + offs_g, mask=mask_g,
                   care_padding=False)
    gh_r = tl.load(gh_ptr + pid_b * stride_gh_b + offs_g * stride_gh_g,
                   mask=mask_g, care_padding=False)
    gh_z = tl.load(gh_ptr + pid_b * stride_gh_b + (G + offs_g) * stride_gh_g,
                   mask=mask_g, care_padding=False)
    gh_n = tl.load(
        gh_ptr + pid_b * stride_gh_b + (2 * G + offs_g) * stride_gh_g,
        mask=mask_g, care_padding=False,
    )

    r = tl.sigmoid(gi_r + gh_r)
    z = tl.sigmoid(gi_z + gh_z)
    n = 2.0 * tl.sigmoid(2.0 * (gi_n + r * gh_n)) - 1.0
    h_new = (1.0 - z) * n + z * h_state
    tl.store(
        h_out_ptr + pid_b * stride_hout_b + offs_g * stride_hout_g,
        h_new.to(h_ptr.dtype.element_ty),
        mask=mask_g,
    )


@triton.jit
def _wsh_zsilu_kernel(
    h_ptr, z_ptr, wsh_t_ptr, x_ptr, gh_ptr,
    B, G, M, N_PAD,
    BLOCK_B: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    offs_k = tl.arange(0, BLOCK_K)
    h = tl.load(
        h_ptr + offs_b[:, None] * G + offs_k[None, :],
        mask=mask_b[:, None],
        other=0.0,
    )
    for n0 in range(0, M + 3 * G, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        w = tl.load(
            wsh_t_ptr + offs_k[:, None] * N_PAD + offs_n[None, :],
            mask=offs_n[None, :] < N_PAD,
            other=0.0,
        )
        sh = tl.dot(h, w)
        z = tl.load(
            z_ptr + offs_b[:, None] * M + offs_n[None, :],
            mask=(offs_n[None, :] < M) & mask_b[:, None],
            other=0.0,
        )
        x = sh + z
        x = tl.sigmoid(x) * x
        tl.store(
            x_ptr + offs_b[:, None] * M + offs_n[None, :],
            x.to(h_ptr.dtype.element_ty),
            mask=(offs_n[None, :] < M) & mask_b[:, None],
        )
        gh_n = offs_n - M
        gh_mask = (gh_n >= 0) & (gh_n < 3 * G)
        tl.store(
            gh_ptr + offs_b[:, None] * (3 * G) + gh_n[None, :],
            sh.to(h_ptr.dtype.element_ty),
            mask=gh_mask[None, :] & mask_b[:, None],
        )


def _build_inputs(b: int, device: str, seed: int):
    torch.manual_seed(seed)
    w_sh = torch.randn(M + 3 * G, G, dtype=DTYPE, device=device) * 0.02
    w_emb2 = torch.randn(V, M, dtype=DTYPE, device=device) * 0.02
    base_logits = torch.randn(b, N_SPEC, V, dtype=DTYPE, device=device) * 0.05
    z_part = torch.randn(b, N_SPEC, M, dtype=DTYPE, device=device) * 0.05
    h = torch.zeros(1, b, G, dtype=DTYPE, device=device)
    draft_tokens = torch.empty(b, N_SPEC, dtype=torch.int64, device=device)

    n_pad = triton.next_power_of_2(M + 3 * G)
    w_sh_t = torch.zeros(G, n_pad, dtype=DTYPE, device=device)
    w_sh_t[:, : M + 3 * G] = w_sh.t().contiguous()
    return (
        w_sh,
        w_sh_t,
        w_emb2,
        base_logits,
        z_part,
        h,
        draft_tokens,
        n_pad,
    )


class _StepCtx:
    """Buffers reused across steps (capture-safe)."""

    def __init__(self, b: int, device: str):
        self.b = b
        self.device = device
        self.n_vblocks = triton.cdiv(V, 4096)
        self.n_vblocks_pad = triton.next_power_of_2(self.n_vblocks)
        self.local_argmax = torch.empty(
            b, self.n_vblocks, dtype=torch.int64, device=device
        )
        self.local_max = torch.empty(
            b, self.n_vblocks, dtype=torch.float32, device=device
        )
        self.token = torch.empty(b, dtype=torch.int64, device=device)
        self.x = torch.empty(b, M, dtype=DTYPE, device=device)
        self.gh = torch.empty(b, 3 * G, dtype=DTYPE, device=device)
        self.h_out = torch.empty(b, G, dtype=DTYPE, device=device)


def _fused_bias_argmax(ctx, base_i, bias):
    _bias_argmax_kernel[(ctx.b, ctx.n_vblocks)](
        base_i, bias, ctx.local_argmax, ctx.local_max, V, ctx.n_vblocks,
        BLOCK_V=4096, multibuffer=False,
    )
    _block_argmax_kernel[(ctx.b,)](
        ctx.local_argmax, ctx.local_max, ctx.token, ctx.n_vblocks,
        BLOCK=ctx.n_vblocks_pad, multibuffer=False,
    )
    return ctx.token


def _fused_gather_cell(ctx, table, tokens, gh, h):
    _cell_gather_kernel[(ctx.b, triton.cdiv(G, 256))](
        table, tokens, gh, h[0], ctx.h_out,
        ctx.b, G,
        1, gh.stride(0), gh.stride(1), h[0].stride(0), h[0].stride(1),
        ctx.h_out.stride(0), ctx.h_out.stride(1),
        BLOCK_G=256,
    )
    return ctx.h_out.unsqueeze(0)


def _fused_wsh_zsilu(ctx, h, z_i, w_sh_t, n_pad):
    _wsh_zsilu_kernel[(triton.cdiv(ctx.b, 16),)](
        h[0], z_i, w_sh_t, ctx.x, ctx.gh,
        ctx.b, G, M, n_pad,
        BLOCK_B=16, BLOCK_K=G, BLOCK_N=64,
    )
    return ctx.x, ctx.gh


def _fused_zsilu(ctx, z_i, s_proj):
    _zsilu_kernel[(ctx.b,)](
        z_i, s_proj, ctx.x, M, BLOCK_M=M, multibuffer=False,
    )
    return ctx.x


def _make_step(
    flags,
    ctx,
    table,
    w_sh,
    w_sh_t,
    w_emb2,
    n_pad,
    z_part,
    base_logits,
    draft_tokens,
):
    """Return a per-step function ``step(i, h) -> h`` for the given fusions."""
    use_wsh = flags.get("wsh_zsilu", False)
    use_zsilu = flags.get("zsilu", False)
    use_ba = flags.get("bias_argmax", False)
    use_gc = flags.get("gather_cell", False)

    def step(i: int, h: torch.Tensor) -> torch.Tensor:
        if use_wsh:
            x, gh = _fused_wsh_zsilu(ctx, h, z_part[:, i], w_sh_t, n_pad)
        else:
            sh = F.linear(h[0], w_sh)
            s_proj = sh[:, :M]
            gh = sh[:, M:]
            if use_zsilu:
                x = _fused_zsilu(ctx, z_part[:, i], s_proj)
            else:
                x = F.silu(z_part[:, i] + s_proj)
        bias = F.linear(x, w_emb2)
        if use_ba:
            token = _fused_bias_argmax(ctx, base_logits[:, i], bias)
        else:
            token = (base_logits[:, i] + bias).argmax(dim=-1)
        draft_tokens[:, i] = token
        if use_gc:
            return _fused_gather_cell(ctx, table, token, gh, h)
        gi = table[token]
        return domino_gru_cell_triton(gi, gh, h)

    return step


def _capture_replay_time(fn, iters: int = ITERS, warmup: int = WARMUP):
    g = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    with torch.npu.graph(g, stream=stream, capture_error_mode="global"):
        fn()
    torch.npu.synchronize()
    for _ in range(warmup):
        g.replay()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def _trace_loop(step_fn, b: int, device: str, draft_tokens: torch.Tensor):
    """Run the 15-step loop eagerly, recording hidden states per step."""
    h = torch.zeros(1, b, G, dtype=DTYPE, device=device)
    hs = [h.clone()]
    for i in range(N_SPEC):
        h = step_fn(i, h)
        hs.append(h.clone())
    return draft_tokens.clone(), hs


def _unit_checks(b: int, device: str, table, w_emb2, base_logits):
    """Direct op-level checks for the two integrated fusions."""
    print(f"B={b}: unit checks (fused vs reference):")

    z = torch.randn(b, M, dtype=DTYPE, device=device) * 0.05
    s = torch.randn(b, M, dtype=DTYPE, device=device) * 0.05
    x_fused = torch.empty(b, M, dtype=DTYPE, device=device)
    _zsilu_kernel[(b,)](z, s, x_fused, M, BLOCK_M=M, multibuffer=False)
    x_ref = F.silu(z + s)
    diff = (x_fused.float() - x_ref.float()).abs().max().item()
    mism = (x_fused != x_ref).sum().item()
    tok_f = (base_logits[:, 0] + F.linear(x_fused, w_emb2)).argmax(dim=-1)
    tok_r = (base_logits[:, 0] + F.linear(x_ref, w_emb2)).argmax(dim=-1)
    flips = (tok_f != tok_r).sum().item()
    print(
        f"  zsilu:            max_diff={diff:.6f} "
        f"bf16_mismatch={mism}/{b * M} argmax_flips={flips}/{b}",
        flush=True,
    )

    tokens = torch.randint(0, V, (b,), dtype=torch.int64, device=device)
    gh = torch.randn(b, 3 * G, dtype=DTYPE, device=device) * 0.02
    h = torch.randn(1, b, G, dtype=DTYPE, device=device) * 0.02
    h_out = torch.empty(b, G, dtype=DTYPE, device=device)
    _cell_gather_kernel[(b, triton.cdiv(G, 256))](
        table, tokens, gh, h[0], h_out,
        b, G,
        1, gh.stride(0), gh.stride(1), h[0].stride(0), h[0].stride(1),
        h_out.stride(0), h_out.stride(1),
        BLOCK_G=256,
    )
    h_fused = h_out.unsqueeze(0)
    h_ref = domino_gru_cell_triton(table[tokens], gh, h)
    diff = (h_fused.float() - h_ref.float()).abs().max().item()
    mism = (h_fused != h_ref).sum().item()
    print(
        f"  gather_cell:      max_diff={diff:.6f} "
        f"bf16_mismatch={mism}/{b * G}",
        flush=True,
    )


VARIANTS = [
    ("baseline", {}),
    ("zsilu", {"zsilu": True}),
    ("bias_argmax", {"bias_argmax": True}),
    ("gather_cell", {"gather_cell": True}),
    ("wsh_zsilu(tl.dot)", {"wsh_zsilu": True}),
    ("zsilu+gather_cell", {"zsilu": True, "gather_cell": True}),
    ("zsilu+bias_argmax", {"zsilu": True, "bias_argmax": True}),
    ("bias_argmax+gather_cell",
     {"bias_argmax": True, "gather_cell": True}),
    ("zsilu+bias_argmax+gather_cell",
     {"zsilu": True, "bias_argmax": True, "gather_cell": True}),
    ("wsh_zsilu+gather_cell",
     {"wsh_zsilu": True, "gather_cell": True}),
    ("wsh_zsilu+bias_argmax",
     {"wsh_zsilu": True, "bias_argmax": True}),
    ("wsh_zsilu+bias_argmax+gather_cell",
     {"wsh_zsilu": True, "bias_argmax": True, "gather_cell": True}),
]


def main() -> None:
    print(f"torch_npu version: {getattr(torch_npu, '__version__', 'unknown')}")
    init_device_properties_triton()
    print(f"H={H} G={G} M={M} V={V} N_SPEC={N_SPEC} B={MS} dtype={DTYPE}")

    device = "npu"
    table = torch.randn(V, 3 * G, dtype=DTYPE, device=device) * 0.02

    for b in MS:
        (
            w_sh,
            w_sh_t,
            w_emb2,
            base_logits,
            z_part,
            h,
            draft_tokens,
            n_pad,
        ) = _build_inputs(b, device, seed=0)
        ctx = _StepCtx(b, device)

        print("-" * 78)
        print(f"B={b}: full 15-step loop, graph replay:")
        _unit_checks(b, device, table, w_emb2, base_logits)
        baseline_us = None
        baseline_tokens = None
        baseline_hs = None
        results = []
        for name, flags in VARIANTS:
            step = _make_step(
                flags, ctx, table, w_sh, w_sh_t, w_emb2, n_pad,
                z_part, base_logits, draft_tokens,
            )

            def loop(h=h):
                for i in range(N_SPEC):
                    h = step(i, h)
                return h

            try:
                ms = _capture_replay_time(loop)
                us = ms * 1e3
                toks, hs = _trace_loop(step, b, device, draft_tokens)
                tok_mism = -1
                first_div = -1
                h_diff = -1.0
                if baseline_us is None:
                    baseline_us = us
                    baseline_tokens = toks
                    baseline_hs = hs
                else:
                    tok_mism = int((toks != baseline_tokens).sum().item())
                    first_div = next(
                        (
                            i
                            for i in range(N_SPEC)
                            if not torch.equal(toks[:, i], baseline_tokens[:, i])
                        ),
                        -1,
                    )
                    h_diff = max(
                        (
                            a.float() - b.float()
                        ).abs().max().item()
                        for a, b in zip(hs, baseline_hs)
                    )
                results.append(
                    (name, us, True, "", tok_mism, first_div, h_diff)
                )
            except Exception as exc:  # noqa: BLE001
                results.append(
                    (name, float("nan"), False, str(exc)[:120],
                     -1, -1, -1.0)
                )

        for name, us, ok, err, tok_mism, first_div, h_diff in results:
            if ok:
                delta = (us / baseline_us - 1.0) * 100 if baseline_us else 0.0
                tok_str = "-" if tok_mism < 0 else str(tok_mism)
                first_str = "-" if first_div < 0 else str(first_div)
                h_str = "-" if h_diff < 0 else f"{h_diff:.6f}"
                print(
                    f"  {name:34s} {us:9.2f} us/step "
                    f"({delta:+6.1f}% vs baseline) "
                    f"tok_mism={tok_str} first_div={first_str} "
                    f"h_diff={h_str}",
                    flush=True,
                )
            else:
                print(f"  {name:34s} FAIL {err}", flush=True)

    print("=" * 78)
    print("RESULT: see table above")


if __name__ == "__main__":
    main()
