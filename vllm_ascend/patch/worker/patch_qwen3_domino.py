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

Debug env var:
  * ``VLLM_ASCEND_DOMINO_TIMING=1`` prints each precompute's wall time in us
    (with NPU sync, so it reflects device execution).
"""

import os
import time

import torch
import torch_npu

from vllm.model_executor.models.qwen3_domino import (
    DominoQwen3Attention,
    DominoQwen3DecoderLayer,
    Qwen3DominoModel,
)

try:
    from vllm_ascend.ops.triton.spec_decode.domino_kv_utils import (
        domino_grouped_k_norm,
    )
except Exception:  # noqa: BLE001  (triton unavailable)
    domino_grouped_k_norm = None


def _resolve_rnq_op():
    """RMSNorm + int8 dynamic quant (non-residual), from vllm-ascend's
    compiled extension; used by the fused context-KV precompute."""
    _c_ascend = getattr(torch.ops, "_C_ascend", None)
    if _c_ascend is not None and hasattr(
        _c_ascend, "npu_rms_norm_dynamic_quant"
    ):
        return _c_ascend.npu_rms_norm_dynamic_quant
    return None


_RNQ_OP = _resolve_rnq_op()


_ORIGINAL_DOMINO_LAYER_FORWARD = DominoQwen3DecoderLayer.forward
_ORIGINAL_DOMINO_MODEL_FORWARD = Qwen3DominoModel.forward


def _squeeze_scale(s: torch.Tensor) -> torch.Tensor:
    return s.squeeze(-1) if s.dim() == 2 else s


def _ascend_domino_attention_forward(
    self: DominoQwen3Attention,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    x8: torch.Tensor | None = None,
    x8s: torch.Tensor | None = None,
) -> torch.Tensor:
    """Domino attention forward with the optional fused quantized qkv.

    The fused path is attached by ``build_quantized_fused_qkv`` after
    on-the-fly quantization (W4A8 single-pack or W4A4 packed q+k+v, one
    projection call per layer).  In the all-W8A8 norm+quant fusion
    (``_use_fused_norm_quant``) the layer forward already ran
    ``npu_add_rms_norm_dynamic_quant`` and passes ``x8``/``x8s`` so the
    projection skips its own activation quant.  The fused projection also
    feeds
    ``qkv_rmsnorm_rope``, which applies q/k RMSNorm + RoPE in one kernel
    (probe: ~2x faster than split + two norms + rope).  The bf16 path keeps
    the separate q/k/v linears, since a fused bf16 projection is slower on
    NPU.
    """
    if getattr(self, "_use_fused_qkv", False):
        scheme = self._fused_qkv_scheme
        if scheme == "w4a8":
            qkv = torch_npu.npu_weight_quant_batchmatmul(
                hidden_states,
                self._fused_qkv_weight,
                antiquant_scale=self._fused_qkv_scale,
                antiquant_group_size=0,
            )
        elif scheme == "w4a4":
            x4, x4s = torch_npu.npu_dynamic_quant(
                hidden_states, dst_type=torch.quint4x2
            )
            qkv = torch_npu.npu_quant_matmul(
                x4,
                self._fused_qkv_weight,
                scale=self._fused_qkv_scale.view(-1),
                pertoken_scale=x4s.reshape(-1),
                bias=None,
                output_dtype=torch.float16,
            ).to(hidden_states.dtype)
        elif scheme == "w8a8":
            if x8 is None or x8s is None:
                x8, x8s = torch_npu.npu_dynamic_quant(hidden_states)
                if x8s.dim() == 2:
                    x8s = x8s.squeeze(1)
            qkv = torch_npu.npu_quant_matmul(
                x8,
                self._fused_qkv_weight,
                self._fused_qkv_scale,
                pertoken_scale=x8s,
                bias=None,
                output_dtype=hidden_states.dtype,
            )
        else:
            raise RuntimeError(f"unknown fused qkv scheme: {scheme}")
        q, k, v = torch.ops.vllm.qkv_rmsnorm_rope(
            input=qkv,
            cos_sin_cache=self._fused_qkv_cos_sin_cache,
            positions=positions,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            q_hidden_size=self.q_size,
            kv_hidden_size=self.kv_size,
            head_dim=self.head_dim,
            eps=self.q_norm.variance_epsilon,
            q_bias=None,
            k_bias=None,
        )
    else:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(
            q.view(
                *q_shape[:-1],
                q_shape[-1] // self.head_dim,
                self.head_dim,
            )
        ).view(q_shape)
        k = self.k_norm(
            k.view(
                *k_shape[:-1],
                k_shape[-1] // self.head_dim,
                self.head_dim,
            )
        ).view(k_shape)

        q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v)
    return self.o_proj(attn_output)


def _ascend_domino_mlp_forward(
    mlp,
    x8: torch.Tensor,
    x8s: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """W8A8 MLP with a pre-quantized (post-norm) activation.

    ``gate_up_proj`` consumes the fused norm+quant output directly; the
    SwiGLU activation is unchanged (``npu_swiglu``), and ``down_proj`` keeps
    its own per-call activation quant (nothing to fuse in front of it).
    """
    gate_up = torch_npu.npu_quant_matmul(
        x8,
        mlp.gate_up_proj.weight,
        mlp.gate_up_proj.weight_scale,
        pertoken_scale=x8s,
        bias=None,
        output_dtype=dtype,
    )
    x = mlp.act_fn(gate_up)
    x, _ = mlp.down_proj(x)
    return x


def _ascend_domino_layer_forward(
    self: DominoQwen3DecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Draft decoder layer with the optional all-W8A8 norm+quant fusion.

    When ``_use_fused_norm_quant`` is set the layer follows vLLM's
    residual-stream pattern (mathematically identical to the classic
    pre-norm block): each RMSNorm is fused with the residual add and the
    int8 activation quant, and the updated residual is returned to the model
    forward.  Otherwise it delegates to the original implementation.
    """
    if not getattr(self, "_use_fused_norm_quant", False):
        return _ORIGINAL_DOMINO_LAYER_FORWARD(
            self, positions, hidden_states
        )

    ln1 = self.input_layernorm
    ln2 = self.post_attention_layernorm
    if residual is None:
        # First layer: no accumulated residual; use the non-add fused op.
        residual = hidden_states
        x8, x8s = _RNQ_OP(
            hidden_states, ln1.weight, epsilon=ln1.variance_epsilon
        )
    else:
        out = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
            hidden_states,
            residual,
            ln1.weight,
            epsilon=ln1.variance_epsilon,
            output_mask=[True, False],
        )
        x8, x8s, residual = out[0], out[3], out[2]
    x8s = _squeeze_scale(x8s)
    attn_out = self.self_attn(
        positions, hidden_states, x8=x8, x8s=x8s
    )
    out = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
        attn_out,
        residual,
        ln2.weight,
        epsilon=ln2.variance_epsilon,
        output_mask=[True, False],
    )
    x8m, x8sm, residual = out[0], out[3], out[2]
    mlp_out = _ascend_domino_mlp_forward(
        self.mlp, x8m, _squeeze_scale(x8sm), hidden_states.dtype
    )
    return mlp_out, residual


def _ascend_domino_model_forward(
    self: Qwen3DominoModel,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Draft model forward with the optional all-W8A8 residual stream.

    With ``_use_fused_norm_quant`` the layers return ``(hidden, residual)``;
    the residual (embedding + accumulated attention outputs) is added to the
    last MLP output before ``output_proj``/final norm, mirroring vLLM's
    standard decoder loop.  Otherwise the original forward is used.
    """
    if not getattr(self, "_use_fused_norm_quant", False):
        return _ORIGINAL_DOMINO_MODEL_FORWARD(
            self, input_ids, positions, inputs_embeds
        )

    if inputs_embeds is None:
        inputs_embeds = self.embed_input_ids(input_ids)
    hidden_states = inputs_embeds
    if self.input_proj is not None:
        hidden_states = self.input_proj(hidden_states)
    residual: torch.Tensor | None = None
    for layer in self.layers:
        hidden_states, residual = layer(
            positions, hidden_states, residual
        )
    hidden_states = hidden_states + residual
    if self.output_proj is not None:
        hidden_states = self.output_proj(hidden_states)
    return self.norm(hidden_states)


def precompute_and_store_context_kv(
    self,
    context_states: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
) -> None:
    timing = os.environ.get("VLLM_ASCEND_DOMINO_TIMING", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
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
        if not self._use_fused_context_kv:
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
            fused_norm_quant = (
                getattr(self, "_use_fused_norm_quant", False)
                and self._fused_kv_scheme == "w8a8"
                and _RNQ_OP is not None
            )
            if fused_norm_quant:
                # RMSNorm is row-wise, so permute to layer-major order first
                # and fuse hidden_norm with the int8 activation quant.
                fused = (
                    context_states.view(num_ctx, D, H)
                    .permute(1, 0, 2)
                    .reshape(D * num_ctx, H)
                    .contiguous()
                )
            else:
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
            if self._fused_kv_scheme == "w4a8":
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
                )[0]
            else:  # w8a8: quantize the shared activation once, then grouped
                # int8 x int8 matmul with per-token and per-channel scales.
                if fused_norm_quant:
                    x8, x8s = _RNQ_OP(
                        fused,
                        getattr(
                            self,
                            "_hidden_norm_weight",
                            self.hidden_norm.weight.data,
                        ),
                        epsilon=getattr(
                            self,
                            "_hidden_norm_eps",
                            self.hidden_norm.variance_epsilon,
                        ),
                    )
                else:
                    x8, x8s = torch_npu.npu_dynamic_quant(fused)
                if x8s.dim() == 2:
                    x8s = x8s.squeeze(1)
                all_kv_flat = torch_npu.npu_grouped_matmul(
                    x=[x8],
                    weight=[self._fused_kv_weight],
                    scale=[self._fused_kv_scale],
                    per_token_scale=[x8s],
                    group_list=group_list,
                    split_item=2,
                    group_type=0,
                    group_list_type=0,
                    output_dtype=torch.bfloat16,
                )[0]
            all_kv_flat = all_kv_flat.contiguous()
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

        # --- Grouped RMSNorm K ([D, T, nkv, hd]) ---
        if (
            domino_grouped_k_norm is not None
            and hasattr(self, "_k_norm_weights")
        ):
            all_k_normed = domino_grouped_k_norm(
                all_k, self._k_norm_weights, self._rms_norm_eps
            )
        else:
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
                context_slot_mapping[i]
                if per_layer
                else context_slot_mapping
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
                if not getattr(self, "_use_fused_context_kv", False)
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
DominoQwen3Attention.forward = _ascend_domino_attention_forward
DominoQwen3DecoderLayer.forward = _ascend_domino_layer_forward
Qwen3DominoModel.forward = _ascend_domino_model_forward
