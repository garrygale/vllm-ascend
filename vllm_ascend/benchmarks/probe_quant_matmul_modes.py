#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU probe for ``npu_quant_matmul`` dtype combos (eager + ACL graph).

Tries the candidate quantized-matmul combos for the Domino draft:

  A) int8 x int8                    (W8A8, current graph-mode redirect)
  B) int8 x int32-packed-int4       (W4A8; the aclnnQuantMatmulV5 combo)
  C) int32-packed-int4 x int32      (W4A4, current W4A4 subset path)
  D) npu_weight_quant_batchmatmul W4A8 layouts (the DSpark op):
     D1 int8 container [K,N], D2 int32-packed ND [K,N//8],
     D3 int32-packed NZ [K,N//8] (DSpark layout),
     D4 int32-packed ND passed as a transposed view (doc-recommended
     per-channel layout: storage [N//8,K], logical [K,N//8] via .t()),
     D5 int32-packed forced back to ND base format, D6 NZ passed transposed
  E) service-exact input_proj call shape (M=512, K=4096, N=2560, scale cast
     inside the call like DominoW4A8LinearMethod.apply)

The probe runs with ``allow_internal_format=True`` (set at the top of main)
to mirror the serving environment: vllm-ascend sets this in
worker/model_runner_v1.py, which changes how ops such as
``npu_convert_weight_to_int4pack`` store their output tensors.

Each combo is executed eagerly and inside a ``torch.npu.NPUGraph`` capture
under both GLOBAL and RELAXED capture modes, so we can see which ones the ACL
graph mode accepts on the installed CANN.  Some quantized ops are rejected in
GLOBAL capture mode with "supported only in the RELAXED mode", and the manual
capture_begin/capture_end probe previously leaked the capture state when a
combo failed, crashing the next op outside the graph.  Errors are printed in
full (no truncation) so they can be shared/pasted for diagnosis.

Run directly on an NPU:  ``python probe_quant_matmul_modes.py``
"""

import torch
import torch_npu

# Same value as vllm_ascend.utils.ACL_FORMAT_FRACTAL_NZ, inlined so the probe
# stays standalone and does not pull in the full vllm-ascend package.
ACL_FORMAT_FRACTAL_NZ = 29
ACL_FORMAT_ND = 2


def format_name(t: torch.Tensor) -> str:
    try:
        return str(torch_npu.get_npu_format(t))
    except Exception:  # noqa: BLE001
        return "?"


def run_eager(name: str, fn, ref):
    try:
        out = fn()
        err = (out.float() - ref.float()).abs().max().item()
        print(f"{name:28s} eager OK   max_err={err:.4f}")
    except Exception as exc:  # noqa: BLE001
        print(f"{name:28s} eager FAIL {type(exc).__name__}: {exc}")


def run_graph(name: str, fn, ref, mode: str):
    try:
        graph = torch.npu.NPUGraph()
        stream = torch.npu.Stream()
        # torch.npu.graph always calls capture_end() in __exit__, even when
        # fn() raises, so a failed combo cannot leak the capture state into
        # the rest of the probe.
        with torch.npu.graph(graph, stream=stream, capture_error_mode=mode):
            out = fn()
        graph.replay()
        torch.npu.synchronize()
        err = (out.float() - ref.float()).abs().max().item()
        print(f"{name:28s} graph[{mode:8s}] OK    max_err={err:.4f}")
    except Exception as exc:  # noqa: BLE001
        print(
            f"{name:28s} graph[{mode:8s}] FAIL "
            f"{type(exc).__name__}: {exc}"
        )


def main() -> None:
    print(f"torch_npu version: {getattr(torch_npu, '__version__', 'unknown')}")
    # Service-like storage: vllm-ascend sets allow_internal_format=True in
    # worker/model_runner_v1.py; the default (False) is what the earlier
    # probe runs used, which is why D2 passed there.
    torch.npu.config.allow_internal_format = True
    print("allow_internal_format=True (service-like)")

    M, K, N = 128, 2560, 1024
    x = torch.randn(M, K, dtype=torch.bfloat16, device="npu")
    w8 = torch.randint(-127, 127, (N, K), device="npu")
    w4 = torch.randint(-7, 7, (N, K), device="npu")

    # Reference: bf16 matmul of weights dequantized with the same per-channel
    # scales that are passed to the ops.  (An earlier version used an
    # arbitrary 0.01 scale, which made the max error ~100x larger than the
    # actual op error.)
    scale8 = (w8.float().abs().amax(dim=1) / 127.0).clamp(min=1e-6).float()
    scale4 = (w4.float().abs().amax(dim=1) / 7.0).clamp(min=1e-6).float()
    ref8 = (x.float() @ (w8.float() * scale8.unsqueeze(1)).t())
    ref4 = (x.float() @ (w4.float() * scale4.unsqueeze(1)).t())

    # --- A) W8A8: int8 x int8 -------------------------------------------
    xi8, xs = torch_npu.npu_dynamic_quant(x)  # int8 [M,K], per-token scale
    w8_t = w8.to(torch.int8).t().contiguous()  # [K,N]

    def a8w8():
        return torch_npu.npu_quant_matmul(
            xi8, w8_t, scale8,
            pertoken_scale=xs.float().reshape(-1),
            output_dtype=torch.bfloat16,
        )

    run_eager("A int8 x int8 (W8A8)", a8w8, ref8)
    run_graph("A int8 x int8 (W8A8)", a8w8, ref8, "global")
    run_graph("A int8 x int8 (W8A8)", a8w8, ref8, "relaxed")

    # --- B) W4A8: int8 x int32-packed int4 ------------------------------
    w4_t = w4.t().contiguous().to(torch.int32)  # [K,N]
    w4_packed = torch_npu.npu_convert_weight_to_int4pack(w4_t)  # [K,N//8]

    def a8w4():
        return torch_npu.npu_quant_matmul(
            xi8, w4_packed, scale4,
            pertoken_scale=xs.float().reshape(-1),
            output_dtype=torch.bfloat16,
        )

    run_eager("B int8 x int32 (W4A8)", a8w4, ref4)
    run_graph("B int8 x int32 (W4A8)", a8w4, ref4, "global")
    run_graph("B int8 x int32 (W4A8)", a8w4, ref4, "relaxed")

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
    run_graph("C int32 x int32 (W4A4)", a4w4, ref4, "global")
    run_graph("C int32 x int32 (W4A4)", a4w4, ref4, "relaxed")

    # --- D) W4A8 via npu_weight_quant_batchmatmul (the DSpark op) -------
    # x stays bf16 (the op quantizes it internally when quant_scale is
    # None), the anti-quant scale is bf16 per-channel [N] with group_size=0,
    # matching DominoW4A8LinearMethod.
    scale4_bf16 = scale4.to(torch.bfloat16)

    # D1) int8 container: one int4 value per int8 byte, [K, N].
    w4_i8 = w4.to(torch.int8).t().contiguous()

    def wqb_i8():
        return torch_npu.npu_weight_quant_batchmatmul(
            x,
            w4_i8,
            antiquant_scale=scale4_bf16,
            antiquant_group_size=0,
        )

    run_eager("D1 wqb int8 container", wqb_i8, ref4)
    run_graph("D1 wqb int8 container", wqb_i8, ref4, "global")
    run_graph("D1 wqb int8 container", wqb_i8, ref4, "relaxed")

    # D2) int32-packed ND [K, N//8] (current eager Domino layout).
    def wqb_packed_nd():
        return torch_npu.npu_weight_quant_batchmatmul(
            x,
            w4_packed,
            antiquant_scale=scale4_bf16,
            antiquant_group_size=0,
        )

    run_eager("D2 wqb int32 ND", wqb_packed_nd, ref4)
    run_graph("D2 wqb int32 ND", wqb_packed_nd, ref4, "global")
    run_graph("D2 wqb int32 ND", wqb_packed_nd, ref4, "relaxed")

    # D3) int32-packed FRACTAL_NZ [K, N//8] (DSpark layout: trans_nz before
    # packing, same order as AscendW4A8DynamicLinearMethod).
    w4_t_nz = torch_npu.npu_format_cast(w4_t, ACL_FORMAT_FRACTAL_NZ)
    w4_packed_nz = torch_npu.npu_convert_weight_to_int4pack(
        w4_t_nz
    )
    # D5) Force the packed weight back to ND base format before the call.
    w4_packed_nd = torch_npu.npu_format_cast(w4_packed, ACL_FORMAT_ND)
    print(
        "weight formats: "
        f"i8={format_name(w4_i8)} packed={format_name(w4_packed)} "
        f"nz={format_name(w4_packed_nz)} nd={format_name(w4_packed_nd)}"
    )

    def wqb_packed_nz():
        return torch_npu.npu_weight_quant_batchmatmul(
            x,
            w4_packed_nz,
            antiquant_scale=scale4_bf16,
            antiquant_group_size=0,
        )

    run_eager("D3 wqb int32 NZ", wqb_packed_nz, ref4)
    run_graph("D3 wqb int32 NZ", wqb_packed_nz, ref4, "global")
    run_graph("D3 wqb int32 NZ", wqb_packed_nz, ref4, "relaxed")

    # D4) int32-packed ND as a transposed (non-contiguous) view: storage
    # [N//8, K], logical [K, N//8] via .t().  This is the layout the doc
    # recommends for per-channel ND (the per-channel scale then runs along
    # the contiguous output-channel dimension).  The packed values are
    # identical to D2; only the strides change.
    def wqb_packed_t():
        return torch_npu.npu_weight_quant_batchmatmul(
            x,
            w4_packed.t(),
            antiquant_scale=scale4_bf16,
            antiquant_group_size=0,
        )

    run_eager("D4 wqb int32 ND transposed", wqb_packed_t, ref4)
    run_graph("D4 wqb int32 ND transposed", wqb_packed_t, ref4, "global")
    run_graph("D4 wqb int32 ND transposed", wqb_packed_t, ref4, "relaxed")

    # D5) int32-packed, explicitly cast back to ND base format.  Candidate
    # fix if the pack op returns an internal format under
    # allow_internal_format=True.
    def wqb_packed_nd_forced():
        return torch_npu.npu_weight_quant_batchmatmul(
            x,
            w4_packed_nd,
            antiquant_scale=scale4_bf16,
            antiquant_group_size=0,
        )

    run_eager("D5 wqb int32 ND forced", wqb_packed_nd_forced, ref4)
    run_graph("D5 wqb int32 ND forced", wqb_packed_nd_forced, ref4, "global")
    run_graph("D5 wqb int32 ND forced", wqb_packed_nd_forced, ref4, "relaxed")

    # D6) int32-packed NZ passed as a transposed view.  The doc requires the
    # weight to be the transposed input for NZ int32 in per-channel mode.
    def wqb_packed_nz_t():
        return torch_npu.npu_weight_quant_batchmatmul(
            x,
            w4_packed_nz.t(),
            antiquant_scale=scale4_bf16,
            antiquant_group_size=0,
        )

    run_eager("D6 wqb int32 NZ transposed", wqb_packed_nz_t, ref4)
    run_graph("D6 wqb int32 NZ transposed", wqb_packed_nz_t, ref4, "global")
    run_graph("D6 wqb int32 NZ transposed", wqb_packed_nz_t, ref4, "relaxed")

    # --- E) Service-exact input_proj call --------------------------------
    # Same dims as the failing layer (draft input_proj 4096->2560) and the
    # scale cast happens inside the closure, mirroring
    # DominoW4A8LinearMethod.apply (weight_scale.to(x.dtype)).
    M_E, K_E, N_E = 512, 4096, 2560
    x_e = torch.randn(M_E, K_E, dtype=torch.bfloat16, device="npu")
    w_e = torch.randint(-7, 7, (N_E, K_E), device="npu")
    scale_e = (w_e.float().abs().amax(dim=1) / 7.0).clamp(min=1e-6).float()
    ref_e = (x_e.float() @ (w_e.float() * scale_e.unsqueeze(1)).t())
    w_e_packed = torch_npu.npu_convert_weight_to_int4pack(
        w_e.t().contiguous().to(torch.int32)
    )

    def input_proj_call():
        return torch_npu.npu_weight_quant_batchmatmul(
            x_e,
            w_e_packed,
            antiquant_scale=scale_e.to(x_e.dtype),
            antiquant_group_size=0,
        )

    run_eager("E input_proj 512x4096", input_proj_call, ref_e)
    run_graph("E input_proj 512x4096", input_proj_call, ref_e, "global")
    run_graph("E input_proj 512x4096", input_proj_call, ref_e, "relaxed")


if __name__ == "__main__":
    main()
