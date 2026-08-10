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

The Domino correction head (``prefix_gru``, ``embed_proj``/``hidden_proj``) is
never quantized; ``qat_exclude`` is honored in addition.
"""

import torch
import torch_npu

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase


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

    Weights are packed once at load time (int4 along the output dim, op layout
    ``[input_size, output_size // 8]``).  The anti-quant scale is per-channel:
    ``[1, output_size]`` with ``antiquant_group_size == input_size``.
    """

    def __init__(self, group_size: int) -> None:
        self.group_size = group_size

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
            antiquant_group_size=self.group_size,
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
    in_features: int,
) -> None:
    """Pack int4 weights and attach the W4A8 method in place."""
    w_t = w_int.to(torch.int32).t().contiguous()
    layer.weight.data = torch_npu.npu_convert_weight_to_int4pack(w_t)
    layer.register_buffer(
        "weight_scale", scale.reshape(1, -1).contiguous().to(torch.float32)
    )
    layer.quant_method = DominoW4A8LinearMethod(group_size=in_features)


def _quantize_w8a8(
    layer,
    w_int: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    """Store int8 weights (op layout ``[input_size, output_size]``) and attach
    the W8A8 method in place."""
    layer.weight.data = w_int.to(torch.int8).t().contiguous()
    layer.register_buffer("weight_scale", scale.to(torch.float32))
    layer.quant_method = DominoW8A8LinearMethod()


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
    layer.register_buffer("weight_scale", scale.to(torch.float32))
    layer.quant_method = DominoW4A4LinearMethod()


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
    for path, module in model.named_modules():
        if not isinstance(module, LinearBase):
            continue
        if _is_excluded(path, exclude):
            continue

        in_features = module.weight.shape[1]
        if _is_w4a4_layer(path, w4a4_layers):
            w_int, scale = quantize_weight_per_channel(
                module.weight.data, 4, stochastic
            )
            _quantize_w4a4(module, w_int, scale)
            scheme = "W4A4"
        elif qat_w_bit == 4:
            w_int, scale = quantize_weight_per_channel(
                module.weight.data, 4, stochastic
            )
            _quantize_w4a8(module, w_int, scale, in_features)
            scheme = "W4A8"
        else:
            w_int, scale = quantize_weight_per_channel(
                module.weight.data, 8, stochastic
            )
            _quantize_w8a8(module, w_int, scale)
            scheme = "W8A8"

        count += 1
        print(
            f"[DominoQuant] {scheme} model.{path} "
            f"weight={tuple(module.weight.shape)}",
            flush=True,
        )

    return count
