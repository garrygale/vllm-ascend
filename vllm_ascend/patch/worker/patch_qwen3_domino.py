#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU implementation of Domino's fused context-KV precompute.

The vLLM main implementation uses ``ops.rms_norm`` and
``ops.rotary_embedding`` (CUDA custom ops).  On Ascend this patch uses
module-based RMSNorm, a per-layer k-norm loop, and the Ascend rotary module
with a cloned key, mirroring ``patch_qwen3_dflash.py``.

With on-the-fly W4A8 quantization, the fused K/V projection is one
``npu_grouped_matmul`` call over the per-layer packed int4 weights (3D weight
``[D, K, 2N//8]``, per-layer scales/offsets, ``split_item=2``); the packed
K+V weights are built as a single pack by ``quantize_domino_model``.

Temporary debug env vars:
  * ``VLLM_ASCEND_DOMINO_TIMING=1`` prints each precompute's wall time in us
    (with NPU sync, so it reflects device execution).
  * ``VLLM_ASCEND_DOMINO_FUSED_KV=0`` forces the per-layer fallback so the
    fused path can be benchmarked against it.
"""

import os
import time

import torch
import torch_npu

from vllm.model_executor.models.qwen3_domino import Qwen3DominoModel


def precompute_and_store_context_kv(
    self,
    context_states: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
) -> None:
    timing = os.environ.get("VLLM_ASCEND_DOMINO_TIMING", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    force_per_layer = os.environ.get(
        "VLLM_ASCEND_DOMINO_FUSED_KV", "1"
    ).strip().lower() in ("0", "false", "no", "off")
    num_ctx = context_states.shape[0] if context_states.dim() >= 1 else -1

    if timing:
        torch.npu.synchronize()
        t0 = time.perf_counter()

    try:
        if context_states.dim() != 2:
            raise ValueError(
                "Domino precompute expects 2D flare-fused context states, got "
                f"{context_states.shape}"
            )

        if not hasattr(self, "_num_attn_layers"):
            self._build_fused_kv_buffers()
        if not self._use_fused_context_kv or force_per_layer:
            self._precompute_and_store_context_kv_per_layer(
                context_states, context_positions, context_slot_mapping
            )
            return

        D = self._num_attn_layers
        H = self.target_hidden_size
        kv = self._kv_size
        hd = self._head_dim
        nkv = self._num_kv_heads

        if getattr(self, "_fused_kv_quantized", False):
            # --- Quantized fused KV projection (one grouped W4A8 GEMM) ---
            # All layers share the same hidden_norm; the [T, D, H] context is
            # flattened to [T*D, H], normalized in one module call, then
            # projected through the per-layer packed int4 K/V weights with a
            # single npu_grouped_matmul (3D weight + group_list, split_item=2).
            normed_context_states = self.hidden_norm(
                context_states.reshape(num_ctx * D, H)
            )
            fused = (
                normed_context_states.view(num_ctx, D, H)
                .permute(1, 0, 2)
                .reshape(D * num_ctx, H)
                .contiguous()
            )
            group_list = self._fused_kv_group_list
            group_list.fill_(num_ctx)
            group_list.cumsum_(0)
            all_kv_flat = torch_npu.npu_grouped_matmul(
                x=[fused],
                weight=[self._fused_kv_weight],
                antiquant_scale=[self._fused_kv_scale],
                antiquant_offset=[self._fused_kv_offset],
                group_list=group_list,
                split_item=2,
                group_type=0,
                group_list_type=0,
                output_dtype=torch.bfloat16,
            )[0].contiguous()
        else:
            # --- bf16 fused KV projection (one batched GEMM for all layers) ---
            normed_context_states = self.hidden_norm(
                context_states.reshape(num_ctx * D, H)
            )
            fused = normed_context_states.view(num_ctx, D, H)
            all_kv_flat = torch.bmm(
                fused.permute(1, 0, 2).contiguous(), self._fused_kv_weight_T
            )
            if self._fused_kv_bias is not None:
                all_kv_flat = all_kv_flat + self._fused_kv_bias.unsqueeze(1)
        # [D, T, 2, nkv, hd]; dim-2 slices are contiguous K and V.
        all_kv = all_kv_flat.view(D, num_ctx, 2, nkv, hd)
        all_k = all_kv[:, :, 0]
        all_v = all_kv[:, :, 1]

        # --- Per-layer RMSNorm K ([D, T, nkv, hd]) ---
        all_k_normed = torch.empty_like(all_k)
        for i in range(D):
            k_norm_layer = self.layers[i].self_attn.k_norm
            all_k_normed[i] = k_norm_layer(all_k[i])

        # --- Fused RoPE across all layers ---
        # Ascend's rotary op requires a real key tensor (it does not accept
        # None like the CUDA/native path); the query argument is rotated in
        # place.
        all_k_flat = all_k_normed.view(D * num_ctx, kv)
        positions_repeated = context_positions.repeat(D)
        tmpv = all_k_flat.clone()
        self.layers[0].self_attn.rotary_emb(
            positions_repeated, all_k_flat, tmpv
        )

        if context_slot_mapping is None:
            return

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
            kv_cache = attn.kv_cache
            attn.impl.do_kv_cache_update(
                attn,
                all_k_final[i],
                all_v[i],
                kv_cache,
                slot_mapping,
            )
    finally:
        if timing:
            torch.npu.synchronize()
            path = (
                "per-layer"
                if force_per_layer
                or not getattr(self, "_use_fused_context_kv", False)
                else (
                    "fused-w4a8"
                    if getattr(self, "_fused_kv_quantized", False)
                    else "fused"
                )
            )
            elapsed_us = (time.perf_counter() - t0) * 1e6
            print(
                f"[DominoKV] T={num_ctx} path={path} {elapsed_us:.1f} us",
                flush=True,
            )


Qwen3DominoModel.precompute_and_store_context_kv = (
    precompute_and_store_context_kv
)
