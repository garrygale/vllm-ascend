#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On-the-fly per-channel quantization for the Domino draft model.

The Domino checkpoint stores raw bf16 weights together with the SpecForge QAT
metadata (``dflash_config.qat_*``).  After the bf16 weights are loaded, this
module quantizes the draft linears in place with the same per-channel
symmetric math as SpecForge's ``quantize_weight``:

  * ``qat_w_bit == 4`` -> W4A8 bulk via ``npu_weight_quant_batchmatmul``,
  * ``qat_w_bit == 8`` -> W8A8 bulk via ``npu_dynamic_quant`` +
    ``npu_quant_matmul``,
  * layers listed in ``qat_w4a4_layers`` -> W4A4 via ``npu_dynamic_quant``
    (``quint4x2``) + ``npu_quant_matmul``.

``npu_weight_quant_batchmatmul`` accepts the int4-packed int32 weights in both
eager and ACL graph mode (ND layout, per-channel ``group_size=0``; validated
by ``benchmarks/probe_quant_matmul_modes.py`` D2), so W4A8 is used regardless
of eager/graph.  Only the W4A8 ``npu_quant_matmul`` combo (int8 x int32) is
unsupported on current CANN.

The Domino correction head (``prefix_gru``, ``embed_proj``/``hidden_proj``) is
never quantized; ``qat_exclude`` is honored in addition.
"""

import torch
import torch_npu

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase

ACL_FORMAT_ND = 2


def _stochastic_round(x: torch.Tensor) -> torch.Tensor:
    return torch.floor(x + torch.rand_like(x))


def quantize_weight_per_channel(
    weight: torch.Tensor,
    w_bit: int,
    stochastic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SpecForge-compatible per-channel symmetric fake-quant.

    Mirrors ``specforge.layers.wxay.quantize_weight``:
    ``scale = amax(|w|, dim=1) / qmax`` (clamped), ``w_int =
    round(w / scale).clamp(-qmax, qmax)``.  The math runs in fp32.

    Returns ``(w_int, scale)`` where ``w_int`` is fp32 in ``[-qmax, qmax]``
    and ``scale`` is ``[out_features, 1]``.
    """
    qmax = 2 ** (w_bit - 1) - 1
    w_fp32 = weight.float()
    scale = w_fp32.abs().amax(dim=1, keepdim=True) / qmax
    scale = scale.clamp(min=1e-6)
    w_scaled = w_fp32 / scale
    w_int = _stochastic_round(w_scaled) if stochastic else torch.round(w_scaled)
    w_int = w_int.clamp(-qmax, qmax)
    return w_int, scale


class DominoW4A8LinearMethod(LinearMethodBase):
    """W4A8 dynamic quant via ``npu_weight_quant_batchmatmul``.

    Per-channel mode (``antiquant_group_size == 0``) with int4-packed int32
    weights in the op layout ``[input_size, output_size // 8]`` and a 1D
    per-channel scale ``[output_size]``.  Validated on NPU in eager and ACL
    graph mode (probes ``probe_w4a8_per_channel.py`` and
    ``probe_quant_matmul_modes.py`` D2).
    """

    def create_weights(self, *args, **kwargs) -> None:
        return

    def process_weights_after_loading(self, layer) -> None:
        return

    def apply(
        self,
        layer,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = torch_npu.npu_weight_quant_batchmatmul(
            x,
            layer.weight,
            antiquant_scale=layer.weight_scale.to(x.dtype),
            antiquant_group_size=0,
        )
        if bias is not None:
            output = output + bias.to(output.dtype)
        return output


class DominoW8A8LinearMethod(LinearMethodBase):
    """W8A8 dynamic quant via ``npu_dynamic_quant`` + ``npu_quant_matmul``."""

    def create_weights(self, *args, **kwargs) -> None:
        return

    def process_weights_after_loading(self, layer) -> None:
        return

    def apply(
        self,
        layer,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x)
        if pertoken_scale.dim() == 2:
            quantized_x = quantized_x.squeeze(dim=1)
            pertoken_scale = pertoken_scale.squeeze(dim=1)
        return torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=pertoken_scale,
            bias=bias,
            output_dtype=x.dtype,
        )


class DominoW4A4LinearMethod(LinearMethodBase):
    """W4A4 dynamic quant via ``npu_dynamic_quant`` (quint4x2) +
    ``npu_quant_matmul``."""

    def create_weights(self, *args, **kwargs) -> None:
        return

    def process_weights_after_loading(self, layer) -> None:
        return

    def apply(
        self,
        layer,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dtype = x.dtype
        quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(
            x, dst_type=torch.quint4x2
        )
        pertoken_scale = pertoken_scale.reshape(-1, 1).squeeze(-1)
        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            scale=layer.weight_scale.view(-1),
            pertoken_scale=pertoken_scale,
            bias=None,
            output_dtype=torch.float16,
        ).to(dtype)
        if bias is not None:
            output = output + bias.to(dtype)
        return output


def _quantize_w4a8(
    layer,
    w_int: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    """Pack int4 weights (op layout ``[input_size, output_size // 8]``) and
    attach the per-channel W4A8 method in place."""
    w_t = w_int.to(torch.int32).t().contiguous()
    layer.weight.data = torch_npu.npu_convert_weight_to_int4pack(w_t)
    layer.register_buffer(
        "weight_scale", scale.reshape(-1).contiguous().to(torch.float32)
    )
    _set_quant_method(layer, DominoW4A8LinearMethod())


def _quantize_w8a8(
    layer,
    w_int: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    """Store int8 weights (op layout ``[input_size, output_size]``) and attach
    the W8A8 method in place."""
    layer.weight.data = w_int.to(torch.int8).t().contiguous()
    # npu_quant_matmul requires a 1D per-channel scale (non G-B/B-B mode).
    layer.register_buffer(
        "weight_scale", scale.reshape(-1).contiguous().to(torch.float32)
    )
    _set_quant_method(layer, DominoW8A8LinearMethod())


def _quantize_w4a4(
    layer,
    w_int: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    """Pack int4 weights (op layout ``[input_size // 8, output_size]``) and
    attach the W4A4 method in place."""
    packed = torch_npu.npu_convert_weight_to_int4pack(
        w_int.to(torch.int32)
    )
    layer.weight.data = packed.transpose(-1, -2)
    layer.register_buffer(
        "weight_scale", scale.reshape(-1).contiguous().to(torch.float32)
    )
    _set_quant_method(layer, DominoW4A4LinearMethod())


def _set_quant_method(layer, method) -> None:
    """Replace the layer's quant method and keep the Ascend custom-op wrapper
    in sync.

    Ascend linear layers (``Ascend*ParallelLinear``) snapshot ``quant_method``
    into their ``custom_op`` at construction time (``custom_op.update_attrs``
    in ``vllm_ascend/ops/linear.py``).  On-the-fly quantization replaces
    ``layer.quant_method`` afterwards, so without this sync the custom op
    keeps calling the old unquantized GEMM with the quantized (int32/int8)
    weight, which fails with "Tensor matB not implemented for DT_INT32/INT8".
    """
    layer.quant_method = method
    custom_op = getattr(layer, "custom_op", None)
    if custom_op is not None:
        custom_op.quant_method = method


def _is_excluded(path: str, exclude: set[str]) -> bool:
    """Match by full path, leaf name, or any ancestor segment (SpecForge
    path-aware matching)."""
    segments = set(path.split("."))
    return (
        path in exclude
        or path.rsplit(".", 1)[-1] in exclude
        or not segments.isdisjoint(exclude)
    )


def _is_w4a4_layer(path: str, w4a4: set[str]) -> bool:
    """Match ``qat_w4a4_layers`` entries; vLLM fuses gate/up into
    ``gate_up_proj`` while SpecForge lists them separately."""
    if path in w4a4:
        return True
    if path.endswith(".gate_up_proj"):
        stem = path[: -len(".gate_up_proj")]
        return stem + ".gate_proj" in w4a4 or stem + ".up_proj" in w4a4
    return False


def quantize_domino_model(model: torch.nn.Module) -> int:
    """Quantize the Domino draft linears in place per ``dflash_config``.

    Args:
        model: the Domino draft model.

    Returns the number of quantized linear layers (0 when the config does not
    request quantization).  The Domino correction head is always excluded.
    """
    dflash_config = getattr(model.config, "dflash_config", None) or {}
    qat_w_bit = dflash_config.get("qat_w_bit")
    if not qat_w_bit:
        return 0
    if qat_w_bit not in (4, 8):
        print(
            f"[DominoQuant] WARNING: unsupported qat_w_bit={qat_w_bit}; "
            "keeping bf16 weights",
            flush=True,
        )
        return 0

    qat_a_bit = dflash_config.get("qat_a_bit")
    qat_exclude = set(dflash_config.get("qat_exclude", []) or [])
    w4a4_layers = set(dflash_config.get("qat_w4a4_layers", []) or [])
    stochastic = bool(dflash_config.get("stochastic_weight", False))
    if dflash_config.get("channel_balanced"):
        print(
            "[DominoQuant] WARNING: channel_balanced is not supported yet; "
            "smooth scale is ignored",
            flush=True,
        )

    # The Domino correction head is never quantized.
    exclude = qat_exclude | {"embed_proj", "hidden_proj"}
    bulk_scheme = "W4A8" if qat_w_bit == 4 else "W8A8"
    print(
        f"[DominoQuant] quantizing draft: w_bit={qat_w_bit} a_bit={qat_a_bit} "
        f"bulk={bulk_scheme} w4a4_layers={len(w4a4_layers)} "
        f"excluded={sorted(exclude)}",
        flush=True,
    )

    count = 0
    model._fused_kv_scheme = None
    model._use_fused_qkv = False
    # The W8A8 norm+activation-quant fusion (residual-stream draft layers,
    # fused context-KV norm+quant) only applies when every quantized draft
    # linear is W8A8.  Any W4A8/W4A4 layer clears this.
    model._all_w8a8 = qat_w_bit == 8
    kv_target_pending: dict[
        int, dict[str, tuple[torch.Tensor, torch.Tensor, str]]
    ] = {}
    qkv_pending: dict[
        int, dict[str, tuple[torch.Tensor, torch.Tensor, str]]
    ] = {}
    for path, module in model.named_modules():
        if not isinstance(module, LinearBase):
            continue
        if _is_excluded(path, exclude):
            continue

        if _is_w4a4_layer(path, w4a4_layers):
            model._all_w8a8 = False
            w_int, scale = quantize_weight_per_channel(
                module.weight.data, 4, stochastic
            )
            _quantize_w4a4(module, w_int, scale)
            _maybe_record_draft_qkv(qkv_pending, path, w_int, scale, "w4a4")
        elif qat_w_bit == 4:
            w_int, scale = quantize_weight_per_channel(
                module.weight.data, 4, stochastic
            )
            _quantize_w4a8(module, w_int, scale)
            _maybe_record_draft_qkv(qkv_pending, path, w_int, scale, "w4a8")
            if path.endswith(".k_proj_target") or path.endswith(
                ".v_proj_target"
            ):
                layer_idx = int(path.split(".")[1])
                key = "k" if path.endswith(".k_proj_target") else "v"
                kv_target_pending.setdefault(layer_idx, {})[key] = (
                    w_int,
                    scale,
                    "w4a8",
                )
        else:
            w_int, scale = quantize_weight_per_channel(
                module.weight.data, 8, stochastic
            )
            _quantize_w8a8(module, w_int, scale)
            _maybe_record_draft_qkv(qkv_pending, path, w_int, scale, "w8a8")
            if path.endswith(".k_proj_target") or path.endswith(
                ".v_proj_target"
            ):
                layer_idx = int(path.split(".")[1])
                key = "k" if path.endswith(".k_proj_target") else "v"
                kv_target_pending.setdefault(layer_idx, {})[key] = (
                    w_int,
                    scale,
                    "w8a8",
                )

        count += 1

    # Build the fused context-KV buffers (single-pack of the concatenated
    # K+V matrix: int4 for W4A8, int8 for W8A8).  Concatenating two
    # separately-packed int4 tensors is NOT valid on this CANN, so this must
    # happen here while the unpacked values are still available.
    num_layers = model.config.num_hidden_layers
    if (
        len(kv_target_pending) == num_layers
        and all(
            len(pair) == 2 and all(len(entry) == 3 for entry in pair.values())
            for pair in kv_target_pending.values()
        )
    ):
        fused_weights = []
        fused_scales = []
        fused_schemes = set()
        for i in range(num_layers):
            w_int_k, scale_k, scheme_k = kv_target_pending[i]["k"]
            w_int_v, scale_v, _ = kv_target_pending[i]["v"]
            fused_int = torch.cat([w_int_k, w_int_v], dim=0)
            fused_scale = torch.cat([scale_k, scale_v]).reshape(-1)
            fused_schemes.add(scheme_k)
            if scheme_k == "w4a8":
                packed = torch_npu.npu_convert_weight_to_int4pack(
                    fused_int.to(torch.int32).t().contiguous()
                )
                packed = torch_npu.npu_format_cast(packed, ACL_FORMAT_ND)
                fused_scale = fused_scale.to(torch.bfloat16)
            elif scheme_k == "w8a8":
                packed = fused_int.to(torch.int8).t().contiguous()
                # A8W8 per-token grouped matmul (aclnnGroupedMatmulV5) only
                # allows BF16 scale with BF16 output; FLOAT scale requires
                # FLOAT16 output. Match the W4A8 branch and store BF16.
                fused_scale = fused_scale.reshape(-1).to(torch.bfloat16)
            else:
                raise RuntimeError(
                    f"unsupported fused context-KV scheme: {scheme_k}"
                )
            fused_weights.append(packed)
            fused_scales.append(fused_scale)
        if len(fused_schemes) != 1:
            raise RuntimeError(
                "mixed fused context-KV schemes: "
                f"{sorted(fused_schemes)}"
            )
        model._fused_kv_weight = torch.stack(fused_weights, dim=0)
        model._fused_kv_scale = torch.stack(fused_scales, dim=0)
        model._fused_kv_scheme = fused_schemes.pop()
        model._fused_kv_offset = (
            torch.zeros_like(model._fused_kv_scale)
            if model._fused_kv_scheme == "w4a8"
            else None
        )
        print(
            f"[DominoQuant] fused context-KV buffers built for "
            f"{num_layers} layers ({model._fused_kv_scheme})",
            flush=True,
        )

    # Build the fused draft q/k/v projection buffers (single-pack of the
    # concatenated q+k+v int4 matrix), one per layer, following each layer's
    # scheme (layer 0 is W4A4 in the current config, the rest W4A8).
    if len(qkv_pending) == num_layers and all(
        len(pair) == 3 for pair in qkv_pending.values()
    ):
        fused_qkv_weights = []
        fused_qkv_scales = []
        fused_qkv_schemes = []
        for i in range(num_layers):
            w_int_q, scale_q, scheme = qkv_pending[i]["q"]
            w_int_k, scale_k, _ = qkv_pending[i]["k"]
            w_int_v, scale_v, _ = qkv_pending[i]["v"]
            fused_int = torch.cat(
                [w_int_q, w_int_k, w_int_v], dim=0
            )
            fused_scale = torch.cat([scale_q, scale_k, scale_v])
            if scheme == "w4a8":
                packed = torch_npu.npu_convert_weight_to_int4pack(
                    fused_int.to(torch.int32).t().contiguous()
                )
                packed = torch_npu.npu_format_cast(packed, ACL_FORMAT_ND)
                fused_scale = fused_scale.reshape(-1).to(torch.bfloat16)
            elif scheme == "w8a8":
                packed = fused_int.to(torch.int8).t().contiguous()
                fused_scale = fused_scale.reshape(-1)
            else:  # w4a4
                packed = torch_npu.npu_convert_weight_to_int4pack(
                    fused_int.to(torch.int32).contiguous()
                ).transpose(-1, -2)
                fused_scale = fused_scale.reshape(-1)
            fused_qkv_weights.append(packed)
            fused_qkv_scales.append(fused_scale)
            fused_qkv_schemes.append(scheme)
        model._fused_qkv_scheme = fused_qkv_schemes
        model._fused_qkv_weight = fused_qkv_weights
        model._fused_qkv_scale = fused_qkv_scales
        model._use_fused_qkv = True
        print(
            f"[DominoQuant] fused draft qkv buffers built for "
            f"{num_layers} layers "
            f"(schemes={fused_qkv_schemes})",
            flush=True,
        )

    return count


def _maybe_record_draft_qkv(
    pending: dict,
    path: str,
    w_int: torch.Tensor,
    scale: torch.Tensor,
    scheme: str,
) -> None:
    """Record a draft q/k/v projection (not ``k_proj_target``/``v_proj_target``)
    for fused qkv buffer construction."""
    if not (
        path.endswith(".q_proj")
        or path.endswith(".k_proj")
        or path.endswith(".v_proj")
    ):
        return
    layer_idx = int(path.split(".")[1])
    key = path.rsplit(".", 1)[-1][0]  # "q" / "k" / "v"
    pending.setdefault(layer_idx, {})[key] = (w_int, scale, scheme)


def build_quantized_fused_kv_buffers(model: torch.nn.Module) -> bool:
    """Enable the quantized fused context-KV path on the Domino draft model.

    Consumes the fused W4A8/W8A8 buffers built by
    :func:`quantize_domino_model` (``_fused_kv_weight/_scale``) and fills in
    the same metadata the bf16 fused path uses (hidden-norm, k-norm, RoPE,
    attn layers, kv dims).  The per-step ``group_list`` buffer is preallocated
    and updated in place (``fill_(T)`` + ``cumsum_(0)``), so no per-call
    allocation is needed.

    Returns True when the fused quantized path is active; otherwise the
    per-layer fallback remains in use.
    """
    if getattr(model, "_fused_kv_scheme", None) not in ("w4a8", "w8a8"):
        model._use_fused_context_kv = False
        return False

    layers_attn = [layer.self_attn for layer in model.layers]
    attn0 = layers_attn[0]

    # Free the bf16 fused buffers built by the base load_weights path.
    for attr in ("_fused_kv_weight_T", "_fused_kv_bias"):
        if hasattr(model, attr):
            delattr(model, attr)

    model._hidden_norm_weight = model.hidden_norm.weight.data
    model._hidden_norm_eps = model.hidden_norm.variance_epsilon
    model._k_norm_weights = torch.stack(
        [attn.k_norm.weight.data for attn in layers_attn], dim=0
    ).contiguous()
    model._rope_head_size = attn0.rotary_emb.head_size
    model._rope_cos_sin_cache = attn0.rotary_emb.cos_sin_cache
    model._rope_is_neox = attn0.rotary_emb.is_neox_style
    model._num_attn_layers = len(layers_attn)
    model._kv_size = attn0.kv_size
    model._head_dim = attn0.head_dim
    model._num_kv_heads = attn0.num_kv_heads
    model._rms_norm_eps = attn0.k_norm.variance_epsilon
    model._attn_layers = [layer.self_attn.attn for layer in model.layers]
    model._fused_kv_group_list = torch.empty(
        model._num_attn_layers,
        dtype=torch.int64,
        device=model._fused_kv_weight.device,
    )
    model._use_fused_context_kv = True
    model._fused_kv_quantized = True
    return True


def build_quantized_fused_qkv(model: torch.nn.Module) -> bool:
    """Attach the fused draft q/k/v buffers to each attention layer.

    Only the quantized path uses the fused projection (probe: fused bf16 is
    slower on NPU, fused W4A8/W4A4 is faster).  The per-layer buffers built by
    :func:`quantize_domino_model` are copied onto each ``self_attn`` module so
    the patched attention forward can read them during the draft forward.
    """
    if not getattr(model, "_use_fused_qkv", False):
        return False
    for i, attn in enumerate(
        layer.self_attn for layer in model.layers
    ):
        attn._fused_qkv_scheme = model._fused_qkv_scheme[i]
        attn._fused_qkv_weight = model._fused_qkv_weight[i]
        attn._fused_qkv_scale = model._fused_qkv_scale[i]
        # The fused split+q/k-rmsnorm+rope Triton kernel expects a bf16
        # cos/sin cache; keep a bf16 copy (no-op when it already is bf16).
        attn._fused_qkv_cos_sin_cache = (
            attn.rotary_emb.cos_sin_cache.to(torch.bfloat16).contiguous()
        )
        attn._use_fused_qkv = True
    return True


def _rms_norm_dynamic_quant_available() -> bool:
    """Both fused norm+quant ops must be present on this CANN build."""
    npu_ns = getattr(torch.ops, "npu", None)
    _c_ascend = getattr(torch.ops, "_C_ascend", None)
    return (
        npu_ns is not None
        and hasattr(npu_ns, "npu_add_rms_norm_dynamic_quant")
        and _c_ascend is not None
        and hasattr(_c_ascend, "npu_rms_norm_dynamic_quant")
    )


def build_quantized_fused_norm_quant(model: torch.nn.Module) -> bool:
    """Enable the W8A8 norm+activation-quant fusion on the draft model.

    Requires every quantized draft linear to be W8A8 (``model._all_w8a8``),
    the fused draft qkv to be active with an all-W8A8 scheme list, and both
    fused norm+quant ops to be available on this CANN.  When enabled:

      * the decoder layers switch to the residual-stream pattern
        (``npu_add_rms_norm_dynamic_quant`` / ``npu_rms_norm_dynamic_quant``)
        and feed pre-quantized activations to the attention and gate_up
        matmuls,
      * the fused context-KV precompute fuses ``hidden_norm`` with the
        activation quant before the grouped matmul.
    """
    if (
        not getattr(model, "_all_w8a8", False)
        or not _rms_norm_dynamic_quant_available()
        or not getattr(model, "_use_fused_qkv", False)
        or any(
            scheme != "w8a8"
            for scheme in getattr(model, "_fused_qkv_scheme", [])
        )
    ):
        model._use_fused_norm_quant = False
        return False
    for layer in model.layers:
        layer._use_fused_norm_quant = True
    model._use_fused_norm_quant = True
    return True
