#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker patch: fused bf16 flare context-KV precompute for the Domino draft.

The 0.30 vllm-side model runs a platform-neutral per-layer precompute
(module RMSNorm + per-layer K/V projections + module k-norm + module rope,
one insert per layer).  On Ascend that is ~5 dispatches x 7 layers of small
ops per propose step.  This patch installs the 0.26-era bf16 flare fusion:

  * one shared ``hidden_norm`` over the flattened [T*D, H] context,
  * one batched GEMM (``torch.bmm``) projecting every layer's K/V through
    the stacked target-projection weights,
  * one grouped-k-norm Triton kernel (``domino_grouped_k_norm``) with
    per-layer weights, and
  * one fused RoPE call over all D*T rows (the Ascend rotary module needs
    a real key tensor, hence the cloned dummy value argument),

followed by the same per-layer cache inserts.

bf16 ONLY: the gate checks that every k/v_target_proj still carries
floating-point weights.  QAT-quantized drafts (int4/int8-packed weights
from ``quantize_domino_model``) intentionally fall back to the per-layer
path, which runs the projections through their quant methods.

TODO(port): port the quantized fused path later -- single
``npu_grouped_matmul`` over per-layer packed int4 K/V weights with the
AICore ``arange*num_ctx`` group_list (0.26 ``patch_qwen3_domino``'s
quantized branch); the activation-quant-fused hidden_norm variant rides
with it.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_domino import Qwen3DominoModel

try:
    from vllm_ascend.ops.triton.spec_decode.domino_kv_utils import (
        domino_grouped_k_norm,
    )
except Exception:  # pragma: no cover - triton unavailable on some builds
    domino_grouped_k_norm = None

logger = init_logger(__name__)

_ORIGINAL_PRECOMPUTE = Qwen3DominoModel.precompute_and_store_context_kv


def _build_bf16_fused_kv_buffers(model) -> None:
    """Stack the target K/V projection + k-norm weights for the fused GEMM.

    Sets ``_use_bf16_fused_context_kv`` on the model: True when every layer
    carries floating-point k/v_target_proj weights, False otherwise
    (QAT-quantized or heterogeneous_kv-off drafts keep the per-layer path).
    """
    model._use_bf16_fused_context_kv = False
    layers_attn = [layer.self_attn for layer in model.layers]
    for attn in layers_attn:
        for proj in (attn.k_target_proj, attn.v_target_proj):
            if proj is None or not proj.weight.data.is_floating_point():
                return

    attn0 = layers_attn[0]
    # [D, H_t, 2*kv]: per-layer [k; v] rows stacked, then transposed for bmm.
    kv_weights = [
        torch.cat(
            [attn.k_target_proj.weight.data, attn.v_target_proj.weight.data],
            dim=0,
        )
        for attn in layers_attn
    ]
    model._fused_kv_weight_T = (
        torch.stack(kv_weights, dim=0).transpose(1, 2).contiguous()
    )
    model._k_norm_weights = torch.stack(
        [attn.k_norm.weight.data for attn in layers_attn], dim=0
    ).contiguous()
    model._num_attn_layers = len(layers_attn)
    model._kv_size = attn0.kv_size
    model._head_dim = attn0.head_dim
    model._num_kv_heads = attn0.num_kv_heads
    model._rms_norm_eps = attn0.k_norm.variance_epsilon
    model._attn_layers = [layer.self_attn.attn for layer in model.layers]
    model._use_bf16_fused_context_kv = True


def precompute_and_store_context_kv(
    self,
    context_states: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
) -> None:
    """Fused bf16 flare precompute; falls back to the per-layer original
    for quantized drafts (see the module docstring TODO)."""
    if not hasattr(self, "_use_bf16_fused_context_kv"):
        _build_bf16_fused_kv_buffers(self)
    if not self._use_bf16_fused_context_kv:
        _ORIGINAL_PRECOMPUTE(
            self, context_states, context_positions, context_slot_mapping
        )
        return

    D = self._num_attn_layers
    H = self.target_hidden_size
    num_ctx = context_states.shape[0]
    kv = self._kv_size
    hd = self._head_dim
    nkv = self._num_kv_heads

    expected_width = self.config.num_hidden_layers * H
    if (
        context_states.dim() != 2
        or context_states.shape[-1] != expected_width
    ):
        raise ValueError(
            "Domino precompute expects 2D flare-fused context states of "
            f"width {expected_width}, got {tuple(context_states.shape)}"
        )

    # --- One shared hidden_norm + one batched K/V GEMM for all layers ---
    normed_context_states = self.hidden_norm(
        context_states.reshape(num_ctx * D, H)
    )
    fused = normed_context_states.view(num_ctx, D, H)
    all_kv_flat = torch.bmm(
        fused.permute(1, 0, 2).contiguous(), self._fused_kv_weight_T
    )
    # [D, T, 2, nkv, hd]; dim-2 slices are contiguous K and V.
    all_kv = all_kv_flat.view(D, num_ctx, 2, nkv, hd)
    all_k = all_kv[:, :, 0]
    all_v = all_kv[:, :, 1]

    # --- Grouped RMSNorm K ([D, T, nkv, hd]) in one kernel ---
    if domino_grouped_k_norm is not None:
        all_k_normed = domino_grouped_k_norm(
            all_k, self._k_norm_weights, self._rms_norm_eps
        )
    else:
        all_k_normed = torch.empty_like(all_k)
        for i in range(D):
            all_k_normed[i] = self.layers[i].self_attn.k_norm(all_k[i])

    # --- Fused RoPE across all layers ---
    # Ascend's rotary op requires a real key tensor (it does not accept
    # None like the CUDA/native path); the query argument is rotated in
    # place and the clone is a no-op value side.
    all_k_flat = all_k_normed.view(D * num_ctx, kv)
    positions_repeated = context_positions.repeat(D)
    tmpv = all_k_flat.clone()
    self.layers[0].self_attn.rotary_emb(positions_repeated, all_k_flat, tmpv)

    if context_slot_mapping is None:
        return  # dummy run: computation only, no cache writes

    # --- Per-layer cache insert ---
    all_k_final = all_k_flat.view(D, num_ctx, nkv, hd)
    per_layer = isinstance(context_slot_mapping, (list, tuple))
    for i in range(D):
        slot_mapping = (
            context_slot_mapping[i] if per_layer else context_slot_mapping
        )
        if slot_mapping is None:
            continue  # dummy run: skip cache ops
        attn = self._attn_layers[i]
        attn.impl.do_kv_cache_update(
            attn,
            all_k_final[i],
            all_v[i],
            attn.kv_cache,
            slot_mapping,
        )


Qwen3DominoModel.precompute_and_store_context_kv = precompute_and_store_context_kv
logger.info_once(
    "Domino bf16 fused flare context-KV precompute installed "
    "(quantized drafts fall back to the per-layer path)"
)
