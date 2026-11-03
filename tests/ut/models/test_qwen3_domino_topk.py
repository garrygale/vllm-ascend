#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``dflash_config.correction_top_k`` on the Ascend Domino draft walk.

The candidate-limited correction replaces the per-step full-vocab fc2 GEMM
(``domino_optimized_bias_and_gh`` -> ``logits_processor(embed_proj[2], ...)``)
with a gather of K fc2 rows + baddbmm + candidate argmax
(``domino_optimized_u_and_gh`` + ``domino_topk_step``).  These tests pin:

* config resolution is fail-closed (bad K / shared readout / no embed_proj);
* ``u_and_gh`` is the exact pre-vocab split of ``bias_and_gh``;
* ``topk_step`` equals the full-vocab argmax whenever the full-vocab winner
  is inside the candidate set, with ``logit_scale`` applied to the
  correction part only (base values enter unscaled, as in the dense walk
  where ``base_logits`` and the scaled bias are added).

The methods only touch precomputed tables (``_domino_w_sh``,
``_domino_corr_w2``) plus torch ops, so the holders below run on any device.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.models.qwen3_domino import (
    AscendQwen3DominoForCausalLM,
    _correction_top_k_from_config,
)

M = 8          # emb_dim
G = 12         # gru hidden
V = 64         # toy vocab
SCALE = 1.3    # logit_scale != 1 pins the alpha placement


def _holder(w_sh=None, corr_w2=None) -> SimpleNamespace:
    torch.manual_seed(0)
    model = SimpleNamespace(emb_dim=M)
    holder = SimpleNamespace(
        model=model,
        # pre-set tables short-circuit _ensure_domino_triton_gru
        _domino_gi_table=torch.zeros(1),
        _domino_w_sh=w_sh if w_sh is not None
        else torch.randn(M + 3 * G, G + 1),
        _domino_corr_w2=corr_w2 if corr_w2 is not None
        else torch.randn(V, M),
        logits_processor=SimpleNamespace(scale=SCALE),
        map_draft_to_target=lambda ids: ids,
    )
    return holder


def _config_holder(dflash_config, shared=False, embed_proj=object()):
    return SimpleNamespace(
        config=SimpleNamespace(dflash_config=dflash_config),
        shared_emb_readout=shared,
        embed_proj=embed_proj,
    )


# ------------------------------------------------------------------ config
def test_correction_top_k_absent_keeps_full_vocab_walk() -> None:
    assert _correction_top_k_from_config(_config_holder({})) is None
    assert _correction_top_k_from_config(
        _config_holder({"correction_top_k": None})
    ) is None


def test_correction_top_k_valid_value_is_returned() -> None:
    assert _correction_top_k_from_config(
        _config_holder({"correction_top_k": 128})
    ) == 128


@pytest.mark.parametrize("bad", [0, -3, 1.5, "16", True])
def test_correction_top_k_rejects_bad_values(bad) -> None:
    with pytest.raises(ValueError, match="correction_top_k"):
        _correction_top_k_from_config(_config_holder({"correction_top_k": bad}))


def test_correction_top_k_rejects_shared_readout() -> None:
    with pytest.raises(ValueError, match="shared_emb_readout"):
        _correction_top_k_from_config(
            _config_holder({"correction_top_k": 16}, shared=True)
        )


def test_correction_top_k_rejects_missing_embed_proj() -> None:
    with pytest.raises(ValueError, match="use_embed_proj"):
        _correction_top_k_from_config(
            _config_holder({"correction_top_k": 16}, embed_proj=None)
        )


# ------------------------------------------------------------- u/gh split
@torch.inference_mode()
def test_u_and_gh_is_the_pre_vocab_split_of_bias_and_gh() -> None:
    holder = _holder()
    h = torch.randn(1, 5, G + 1)          # gru state layout [1, B, G]
    z = torch.randn(5, 7)                 # z part, any width

    u, gh = AscendQwen3DominoForCausalLM.domino_optimized_u_and_gh(
        holder, h, z
    )

    sh = h[0] @ holder._domino_w_sh.t()
    torch.testing.assert_close(u, z + sh[:, :M])
    torch.testing.assert_close(gh, sh[:, M:])
    # u must be pre-SiLU: the activation belongs to the fc2 input.
    assert not torch.allclose(u, F.silu(u))


# ------------------------------------------------------------- topk step
@torch.inference_mode()
def test_topk_weights_gathers_exact_fc2_rows() -> None:
    torch.manual_seed(1)
    B, S, K = 3, 4, 6
    holder = _holder()
    w2 = holder._domino_corr_w2
    ids = torch.stack(
        [torch.randperm(V)[: K * S].view(S, K) for _ in range(B)]
    )

    got = AscendQwen3DominoForCausalLM.domino_topk_weights(holder, ids)

    torch.testing.assert_close(got, w2[ids])
    assert got.shape == (B, S, K, M)


@torch.inference_mode()
def test_topk_step_matches_full_argmax_when_winner_is_candidate() -> None:
    torch.manual_seed(3)
    B, K = 6, 8
    holder = _holder()
    w2 = holder._domino_corr_w2
    u = torch.randn(B, M)
    base = torch.randn(B, V)

    full = base + SCALE * (F.silu(u) @ w2.t())
    winner = full.argmax(dim=-1)
    vals, ids = base.topk(K, dim=-1)
    cand_w = AscendQwen3DominoForCausalLM.domino_topk_weights(
        holder, ids.unsqueeze(1)
    )[:, 0]

    got = AscendQwen3DominoForCausalLM.domino_topk_step(
        holder, u, vals, ids, cand_w
    )

    in_cand = (ids == winner.unsqueeze(1)).any(dim=1)
    assert in_cand.any(), "fixture must exercise the in-candidate rows"
    torch.testing.assert_close(
        got[in_cand].to(torch.int64), winner[in_cand]
    )
    # Out-of-candidate rows carry no exactness guarantee (the accepted
    # approximation); they only need to stay valid ids.
    assert (got >= 0).all() and (got < V).all()


@torch.inference_mode()
def test_topk_step_scores_apply_scale_to_correction_only() -> None:
    """scores = base_values + logit_scale * W2[cand] . SiLU(u)."""
    torch.manual_seed(4)
    B, K = 3, 5
    holder = _holder()
    w2 = holder._domino_corr_w2
    u = torch.randn(B, M)
    vals = torch.randn(B, K)
    ids = torch.stack(
        [torch.randperm(V)[:K] for _ in range(B)]
    )
    cand_w = AscendQwen3DominoForCausalLM.domino_topk_weights(
        holder, ids.unsqueeze(1)
    )[:, 0]

    got = AscendQwen3DominoForCausalLM.domino_topk_step(
        holder, u, vals, ids, cand_w
    )

    scores = vals + SCALE * torch.bmm(
        cand_w, F.silu(u).unsqueeze(-1)
    ).squeeze(-1)
    expected = ids.gather(
        1, scores.argmax(dim=-1, keepdim=True)
    ).squeeze(1)
    torch.testing.assert_close(got.to(torch.int64), expected)
