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


def _reference_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return weight * (x * torch.rsqrt(variance + eps))


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
