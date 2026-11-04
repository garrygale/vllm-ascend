#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``dflash_config.correction_swiglu`` on the Ascend Domino fused-h walk.

SwiGLU doubles the correction fc1 to [2*emb_dim, in_dim], so the fused
tables grow with it: ``_domino_w_sh`` becomes [2M + 3G, G] and the per-step
split moves to ``sh[:, :-3G]`` / ``sh[:, -3G:]``.  These tests pin:

* the s/gh split is by trailing-3G (flag-agnostic, works at both widths);
* ``bias_and_gh`` composes silu(gate) * up with logit_scale, matching the
  two-path head ``fc2(act(fc1(z || s)))``;
* ``topk_step`` applies the same gated activation before the candidate walk;
* holders/models without the flag keep the plain SiLU path.

The methods only touch precomputed tables plus torch ops, so the holders
below run on any device.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from vllm_ascend.models.qwen3_domino import (
    AscendQwen3DominoForCausalLM,
    _correction_act,
)

M = 8          # emb_dim
G = 12         # gru hidden
V = 64         # toy vocab
SCALE = 1.3    # logit_scale != 1 pins the alpha placement


class _LP:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, layer, u):
        return F.linear(u, layer.weight) * self.scale


def _holder(swiglu: bool) -> SimpleNamespace:
    torch.manual_seed(0)
    fc2_w = torch.randn(V, M)
    model = SimpleNamespace(
        emb_dim=M,
        gru_hidden_dim=G,
        shared_emb_readout=False,
        embed_proj=[None, None, SimpleNamespace(weight=fc2_w)],
    )
    if swiglu:
        model.correction_swiglu = True
    s_rows = 2 * M if swiglu else M
    return SimpleNamespace(
        model=model,
        _ensure_domino_triton_gru=lambda: None,  # tables pre-set below
        _domino_gi_table=torch.zeros(1),
        _domino_w_sh=torch.randn(s_rows + 3 * G, G + 1),
        _domino_corr_w2=fc2_w,
        logits_processor=_LP(SCALE),
        map_draft_to_target=lambda ids: ids,
    )


def _act(model, u):
    if getattr(model, "correction_swiglu", False):
        gate, up = u.chunk(2, dim=-1)
        return F.silu(gate) * up
    return F.silu(u)


# ------------------------------------------------------------ activation
@torch.inference_mode()
def test_correction_act_gates_by_default() -> None:
    u = torch.randn(5, 2 * M)
    gate, up = u.chunk(2, dim=-1)
    torch.testing.assert_close(
        _correction_act(SimpleNamespace(correction_swiglu=True), u),
        F.silu(gate) * up,
    )
    torch.testing.assert_close(
        _correction_act(SimpleNamespace(), u[:, :M]), F.silu(u[:, :M])
    )


# ------------------------------------------------------------- s/gh split
@torch.inference_mode()
def test_u_and_gh_swiglu_splits_at_double_width() -> None:
    holder = _holder(swiglu=True)
    h = torch.randn(1, 5, G + 1)          # gru state layout [1, B, G]
    z = torch.randn(5, 2 * M)

    u, gh = AscendQwen3DominoForCausalLM.domino_optimized_u_and_gh(
        holder, h, z
    )

    sh = h[0] @ holder._domino_w_sh.t()
    assert sh.shape[-1] == 2 * M + 3 * G
    torch.testing.assert_close(u, z + sh[:, : 2 * M])
    torch.testing.assert_close(gh, sh[:, 2 * M:])


# ------------------------------------------------------------ dense bias
@torch.inference_mode()
def test_bias_and_gh_swiglu_matches_two_path_head() -> None:
    holder = _holder(swiglu=True)
    h = torch.randn(1, 5, G + 1)
    z = torch.randn(5, 2 * M)

    bias, gh = AscendQwen3DominoForCausalLM.domino_optimized_bias_and_gh(
        holder, h, z
    )

    sh = h[0] @ holder._domino_w_sh.t()
    u = z + sh[:, : -(3 * G)]
    gate, up = u.chunk(2, dim=-1)
    manual = (F.silu(gate) * up) @ holder._domino_corr_w2.t() * SCALE
    torch.testing.assert_close(bias, manual, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(gh, sh[:, -(3 * G):])


# -------------------------------------------------------------- topk step
@torch.inference_mode()
def test_topk_step_swiglu_scores_apply_scale_to_correction_only() -> None:
    torch.manual_seed(4)
    B, K = 3, 5
    holder = _holder(swiglu=True)
    w2 = holder._domino_corr_w2
    u = torch.randn(B, 2 * M)
    vals = torch.randn(B, K)
    ids = torch.stack([torch.randperm(V)[:K] for _ in range(B)])
    cand_w = AscendQwen3DominoForCausalLM.domino_topk_weights(
        holder, ids.unsqueeze(1)
    )[:, 0]

    got = AscendQwen3DominoForCausalLM.domino_topk_step(
        holder, u, vals, ids, cand_w
    )

    scores = vals + SCALE * torch.bmm(
        cand_w, _act(holder.model, u).unsqueeze(-1)
    ).squeeze(-1)
    expected = ids.gather(1, scores.argmax(dim=-1, keepdim=True)).squeeze(1)
    torch.testing.assert_close(got.to(torch.int64), expected)


# ------------------------------------------------------- flag-less models
@torch.inference_mode()
def test_silu_holders_stay_on_the_legacy_path() -> None:
    holder = _holder(swiglu=False)
    h = torch.randn(1, 5, G + 1)
    z = torch.randn(5, M)

    bias, gh = AscendQwen3DominoForCausalLM.domino_optimized_bias_and_gh(
        holder, h, z
    )

    sh = h[0] @ holder._domino_w_sh.t()
    manual = F.silu(z + sh[:, :M]) @ holder._domino_corr_w2.t() * SCALE
    torch.testing.assert_close(bias, manual, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(gh, sh[:, M:])
