#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU probe for ``npu_quant_matmul`` dtype combos (eager + ACL graph).

Tries the three candidate quantized-matmul combos for the Domino draft:

  A) int8 x int8                    (W8A8, current graph-mode redirect)
  B) int8 x int32-packed-int4       (W4A8; the aclnnQuantMatmulV5 combo)
  C) int32-packed-int4 x int32      (W4A4, current W4A4 subset path)

Each combo is executed eagerly and inside a ``torch.npu.NPUGraph`` capture so
we can see which ones the ACL graph mode accepts on the installed CANN.

Run directly on an NPU:  ``python probe_quant_matmul_modes.py``
"""

import torch
import torch_npu


def run_eager(name: str, fn, ref):
    try:
        out = fn()
        err = (out.float() - ref.float()).abs().max().item()
        print(f"{name:28s} eager OK   max_err={err:.4f}")
    except Exception as exc:  # noqa: BLE001
        print(f"{name:28s} eager FAIL {type(exc).__name__}: {str(exc)[:200]}")


def run_graph(name: str, fn):
    try:
        stream = torch.npu.Stream()
        graph = torch.npu.NPUGraph()
        with torch.npu.stream(stream):
            graph.capture_begin()
            fn()
            graph.capture_end()
        print(f"{name:28s} graph OK")
    except Exception as exc:  # noqa: BLE001
        print(f"{name:28s} graph FAIL {type(exc).__name__}: {str(exc)[:200]}")


def main() -> None:
    print(f"torch_npu version: {getattr(torch_npu, '__version__', 'unknown')}")

    M, K, N = 128, 2560, 1024
    x = torch.randn(M, K, dtype=torch.bfloat16, device="npu")
    w8 = torch.randint(-127, 127, (N, K), device="npu")
    w4 = torch.randint(-7, 7, (N, K), device="npu")

    # Reference: bf16 matmul of dequantized weights.
    ref8 = (x.float() @ (w8.float() * 0.01).t())
    ref4 = (x.float() @ (w4.float() * 0.01).t())

    # --- A) W8A8: int8 x int8 -------------------------------------------
    xi8, xs = torch_npu.npu_dynamic_quant(x)  # int8 [M,K], per-token scale
    w8_t = w8.to(torch.int8).t().contiguous()  # [K,N]
    scale8 = (w8.float().abs().amax(dim=1) / 127.0).clamp(min=1e-6).float()

    def a8w8():
        return torch_npu.npu_quant_matmul(
            xi8, w8_t, scale8,
            pertoken_scale=xs.float().reshape(-1),
            output_dtype=torch.bfloat16,
        )

    run_eager("A int8 x int8 (W8A8)", a8w8, ref8)
    run_graph("A int8 x int8 (W8A8)", a8w8)

    # --- B) W4A8: int8 x int32-packed int4 ------------------------------
    w4_t = w4.t().contiguous().to(torch.int32)  # [K,N]
    w4_packed = torch_npu.npu_convert_weight_to_int4pack(w4_t)  # [K,N//8]
    scale4 = (w4.float().abs().amax(dim=1) / 7.0).clamp(min=1e-6).float()

    def a8w4():
        return torch_npu.npu_quant_matmul(
            xi8, w4_packed, scale4,
            pertoken_scale=xs.float().reshape(-1),
            output_dtype=torch.bfloat16,
        )

    run_eager("B int8 x int32 (W4A8)", a8w4, ref4)
    run_graph("B int8 x int32 (W4A8)", a8w4)

    # --- C) W4A4: int32-packed x int32 ----------------------------------
    x4, x4s = torch_npu.npu_dynamic_quant(x, dst_type=torch.quint4x2)
    w4_packed_n = torch_npu.npu_convert_weight_to_int4pack(
        w4.to(torch.int32)
    ).transpose(-1, -2)  # [K//8,N]

    def a4w4():
        return torch_npu.npu_quant_matmul(
            x4, w4_packed_n, scale4,
            pertoken_scale=x4s.reshape(-1),
            output_dtype=torch.float16,
        )

    run_eager("C int32 x int32 (W4A4)", a4w4, ref4)
    run_graph("C int32 x int32 (W4A4)", a4w4)


if __name__ == "__main__":
    main()
