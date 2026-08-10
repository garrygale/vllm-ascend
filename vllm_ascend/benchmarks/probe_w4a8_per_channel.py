#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU probe for ``npu_weight_quant_batchmatmul`` W4A8 variants.

Validates three layouts for the Domino on-the-fly W4A8 path:

  A) int8 container ``[K, N]`` + per-channel scale, ``group_size=0``
     (first working implementation, 1 byte/weight).
  B) int4-packed int32 ``[K, N//8]`` + per-channel scale, ``group_size=0``
     (WINNER: true 4-bit storage, 0.5 byte/weight; max err 0.0236 vs bf16).
  C) int4-packed int32 + per-group scale 256 (DSpark/msModelSlim layout;
     kept as a reference only, scale orientation still needs care).

Run directly on an NPU:  ``python probe_w4a8_per_channel.py``
"""

import torch
import torch_npu


def main() -> None:
    M, K, N = 128, 2560, 1024
    x = torch.randn(M, K, dtype=torch.bfloat16, device="npu")
    w4 = torch.randint(-7, 7, (N, K), device="npu")
    scale = (torch.rand(N, device="npu") * 0.01 + 1e-4).to(torch.bfloat16)

    # Reference: per-channel dequant in fp32.
    w_deq = (w4.float() * scale.float().unsqueeze(1))
    ref_out = (x.float() @ w_deq.t()).float()

    # A) int8 container + group_size=0.
    w_i8 = w4.to(torch.int8).t().contiguous()  # [K, N]
    out_a = torch_npu.npu_weight_quant_batchmatmul(
        x, w_i8, scale, antiquant_group_size=0)
    err_a = (out_a.float() - ref_out).abs().max().item()
    print(f"A int8 perchannel     max_err={err_a:.4f}")

    # B) int4-packed int32 + group_size=0 (per-channel).
    w_t = w4.t().contiguous().to(torch.int32)  # [K, N]
    w_packed = torch_npu.npu_convert_weight_to_int4pack(w_t)  # [K, N//8]
    try:
        out_b = torch_npu.npu_weight_quant_batchmatmul(
            x, w_packed, scale, antiquant_group_size=0)
        err_b = (out_b.float() - ref_out).abs().max().item()
        print(f"B int32 perchannel    max_err={err_b:.4f}")
    except Exception as exc:  # noqa: BLE001
        print(f"B int32 perchannel    FAILED: {exc}")

    # C) int4-packed int32 + per-group 256 (fallback reference).
    g = 256
    w4g = w4.float().view(N, K // g, g)
    gscale = (w4g.abs().amax(dim=-1) / 7.0).clamp(min=1e-6)  # [N, K//g]
    w4_deq_g = (w4.float() * gscale.unsqueeze(-1).repeat(1, 1, g).view(N, K))
    ref_c = (x.float() @ w4_deq_g.t()).float()
    for label, gscale_arg in (
        ("C pergroup256 [K/g, N]", gscale.t().contiguous().to(torch.bfloat16)),
        ("C pergroup256 [N, K/g]", gscale.contiguous().to(torch.bfloat16)),
    ):
        out_c = torch_npu.npu_weight_quant_batchmatmul(
            x, w_packed, gscale_arg, antiquant_group_size=g)
        err_c = (out_c.float() - ref_c).abs().max().item()
        print(f"{label} max_err={err_c:.4f}")


if __name__ == "__main__":
    main()
