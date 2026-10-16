#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``dflash_config.target_tap_norm`` on the Ascend Domino draft path.

The per-tap target RMSNorm (SpecForge ``target_tap_norm``; the served Laguna
drafts use the same layout) lives in the shared vLLM fusion
``Qwen3DominoForCausalLM.combine_hidden_states``.  Ascend must keep using it
instead of forking the fusion, so this file pins that inheritance and the
resulting per-tap math through the Ascend wrapper.

Invariant chain on NPU (no Ascend-specific code is required):

* ``AscendDominoSpeculator.propose`` -> ``super().propose``
  (``vllm_ascend/worker/v2/spec_decode/domino/speculator.py``), which calls
  ``self.model.combine_hidden_states(torch.cat(aux_hidden_states, -1))``;
  same for the DFlash proposer.  The fusion therefore runs eagerly with the
  served per-tap norms.
* ``quantize_domino_model`` walks ``named_modules()`` but rewrites only
  ``LinearBase`` modules, so it never touches the ``aux_hidden_norms``
  RMSNorms, and the quantized ``fc`` (DFlash) or the fused-context
  ``hidden_norm`` + int8 quant path (Domino) see the *normalized* targets.
* The per-tap slices are strided views; that is what the shipped MLA path
  feeds ``npu_rms_norm`` (``vllm_ascend/attention/mla_v1.py``:
  ``q_c, _ = qkv_lora.split(...)`` then ``self.q_a_layernorm(q_c)``), so no
  extra ``.contiguous()`` copy is needed here.
"""

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.qwen3_domino import Qwen3DominoForCausalLM

from vllm_ascend.models.qwen3_domino import AscendQwen3DominoForCausalLM

NUM_TARGET_FEATURES = 3
TARGET_HIDDEN_SIZE = 8
NUM_DRAFT_LAYERS = 2
DRAFT_HIDDEN = 12


def _reference_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return weight * (x * torch.rsqrt(variance + eps))


def _reference_grouped_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Grouped weight, e.g. ``[D, hd]`` over ``[D, T, nkv, hd]``."""
    variance = x.pow(2).mean(-1, keepdim=True)
    pad = (1,) * (x.dim() - weight.dim())
    return weight.reshape(*weight.shape[:-1], *pad, weight.shape[-1]) * (
        x * torch.rsqrt(variance + eps)
    )


def _holder() -> SimpleNamespace:
    """The subset of Qwen3DominoModel the fusion touches, with tap norms."""
    torch.manual_seed(0)
    norms = torch.nn.ModuleList(
        [
            RMSNorm(TARGET_HIDDEN_SIZE, eps=1e-6)
            for _ in range(NUM_TARGET_FEATURES)
        ]
    )
    with torch.no_grad():
        for index, norm in enumerate(norms):
            # Distinct gains so a tap-index swap cannot pass.
            norm.weight.fill_(0.5 + index)
    model = SimpleNamespace(
        is_flare=True,
        num_target_features=NUM_TARGET_FEATURES,
        target_hidden_size=TARGET_HIDDEN_SIZE,
        aux_hidden_norms=norms,
        layer_fusion_weights=torch.randn(
            NUM_DRAFT_LAYERS, NUM_TARGET_FEATURES
        ),
        config=SimpleNamespace(num_hidden_layers=NUM_DRAFT_LAYERS),
    )
    return SimpleNamespace(model=model)


def test_ascend_wrapper_does_not_fork_the_shared_fusion() -> None:
    """The NPU draft must inherit the fusion that applies the tap norms."""
    assert (
        AscendQwen3DominoForCausalLM.combine_hidden_states
        is Qwen3DominoForCausalLM.combine_hidden_states
    )
    assert "combine_hidden_states" not in vars(AscendQwen3DominoForCausalLM)


@torch.inference_mode()
def test_ascend_fusion_normalizes_each_tap_before_the_weighted_sum() -> None:
    holder = _holder()
    hidden = torch.randn(4, NUM_TARGET_FEATURES * TARGET_HIDDEN_SIZE)
    actual = AscendQwen3DominoForCausalLM.combine_hidden_states(holder, hidden)

    taps = hidden.view(4, NUM_TARGET_FEATURES, TARGET_HIDDEN_SIZE)
    normed = torch.stack(
        [
            _reference_rms_norm(taps[:, index, :], norm.weight)
            for index, norm in enumerate(holder.model.aux_hidden_norms)
        ],
        dim=1,
    )
    fusion = torch.softmax(holder.model.layer_fusion_weights, dim=1)
    expected = torch.matmul(fusion, normed).reshape(
        4, NUM_DRAFT_LAYERS * TARGET_HIDDEN_SIZE
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def _fc_holder() -> SimpleNamespace:
    """fc variant: one shared draft-width feature, no fusion weights."""
    torch.manual_seed(0)
    norms = torch.nn.ModuleList(
        [RMSNorm(TARGET_HIDDEN_SIZE, eps=1e-6) for _ in range(NUM_TARGET_FEATURES)]
    )
    with torch.no_grad():
        for index, norm in enumerate(norms):
            norm.weight.fill_(0.5 + index)
    fc = torch.nn.Linear(NUM_TARGET_FEATURES * TARGET_HIDDEN_SIZE, DRAFT_HIDDEN)
    with torch.no_grad():
        fc.weight.normal_(0, 0.05)
    model = SimpleNamespace(
        is_flare=False,
        num_target_features=NUM_TARGET_FEATURES,
        target_hidden_size=TARGET_HIDDEN_SIZE,
        context_hidden_size=DRAFT_HIDDEN,
        aux_hidden_norms=norms,
        fc=fc,
        hidden_norm=RMSNorm(DRAFT_HIDDEN, eps=1e-6),
        layer_fusion_weights=None,
        config=SimpleNamespace(num_hidden_layers=NUM_DRAFT_LAYERS),
    )
    return SimpleNamespace(model=model)


@torch.inference_mode()
def test_ascend_fc_fusion_returns_one_shared_draft_width_feature() -> None:
    """fusion_mode='fc' (the SpecForge default) is served, not rejected."""
    holder = _fc_holder()
    hidden = torch.randn(4, NUM_TARGET_FEATURES * TARGET_HIDDEN_SIZE)
    actual = AscendQwen3DominoForCausalLM.combine_hidden_states(holder, hidden)

    taps = hidden.view(4, NUM_TARGET_FEATURES, TARGET_HIDDEN_SIZE)
    normed = torch.stack(
        [
            _reference_rms_norm(taps[:, index, :], norm.weight)
            for index, norm in enumerate(holder.model.aux_hidden_norms)
        ],
        dim=1,
    )
    expected = _reference_rms_norm(
        holder.model.fc(normed.reshape(4, -1)), holder.model.hidden_norm.weight
    )
    assert actual.shape == (4, DRAFT_HIDDEN)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@torch.inference_mode()
def test_ascend_precompute_fc_branch_writes_per_layer_kv() -> None:
    """The NPU context-KV precompute must handle the fc [T, C] layout."""
    from vllm_ascend.patch.worker.patch_qwen3_domino import (
        precompute_and_store_context_kv as ascend_precompute,
    )

    D, T, C = 2, 5, DRAFT_HIDDEN
    nkv, hd = 2, 4
    kv = nkv * hd
    torch.manual_seed(0)
    flat_weight = torch.randn(D * 2 * kv, C) * 0.1
    k_norms = [RMSNorm(hd, eps=1e-6) for _ in range(D)]
    with torch.no_grad():
        for index, norm in enumerate(k_norms):
            norm.weight.fill_(0.5 + index)
    written: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    layers = [
        SimpleNamespace(
            self_attn=SimpleNamespace(
                k_norm=k_norms[index],
                # In-place RoPE stub: the layout is what this test pins.
                rotary_emb=lambda positions, q, k: None,
            )
        )
        for index in range(D)
    ]
    def _make_recorder(index: int):
        def record(attn, k, v, cache, slot_mapping):
            written[index] = (k.clone(), v.clone())

        return record

    attn_layers = [
        SimpleNamespace(
            kv_cache=None,
            impl=SimpleNamespace(do_kv_cache_update=_make_recorder(index)),
        )
        for index in range(D)
    ]
    model = SimpleNamespace(
        is_flare=False,
        _num_attn_layers=D,
        _use_fused_context_kv=True,
        _fused_kv_quantized=False,
        context_hidden_size=C,
        hidden_norm=RMSNorm(C, eps=1e-6),
        _fused_kv_weight_flat=flat_weight,
        _fused_kv_bias_flat=None,
        _kv_size=kv,
        _head_dim=hd,
        _num_kv_heads=nkv,
        _k_norm_weights=torch.stack([n.weight.data for n in k_norms]),
        _rms_norm_eps=1e-6,
        layers=layers,
        _attn_layers=attn_layers,
    )

    context_states = torch.randn(T, C)
    ascend_precompute(
        model,
        context_states,
        torch.arange(T),
        [torch.arange(T) for _ in range(D)],
    )

    normed = _reference_rms_norm(context_states, model.hidden_norm.weight)
    expected_all = (normed @ flat_weight.t()).view(T, D, 2, nkv, hd).permute(
        1, 0, 2, 3, 4
    )
    expected_k = _reference_grouped_rms_norm(
        expected_all[:, :, 0], model._k_norm_weights
    )
    assert set(written) == {0, 1}
    for index in range(D):
        k, v = written[index]
        assert k.shape == (T, nkv, hd)
        torch.testing.assert_close(k, expected_k[index], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            v, expected_all[index, :, 1], rtol=1e-5, atol=1e-6
        )
