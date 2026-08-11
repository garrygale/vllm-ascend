#!/usr/bin/env python3
# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""NPU probe for the W8A8 Domino MLP tail fusions.

The all-W8A8 draft MLP currently runs, per layer:

  ``npu_quant_matmul(gate_up) -> npu_swiglu -> npu_dynamic_quant ->
  npu_quant_matmul(down)``

One fusion candidate is probed against that chain on the real Domino
dims (K=2560, IM=9728):

  * ``npu_grouped_matmul_swiglu_quant_v2``: fuses the gate_up matmul,
    SwiGLU and the output quant into one aclnn op.  Measured on
    torch_npu 2.10.0.post2: works for dense E=1 up to N=10240, but
    fails for N=19456 (A8W8 ``N <= 10240`` limit), so it cannot fuse
    the Domino MLP tail (``N = 2*IM = 19456``).  The sanity matrix
    below documents this.
  * ``npu_swiglu_quant``: fuses SwiGLU + quant, but the 2.10.0 wrapper
    hard-limits ``x``'s last dim to 8192 (gate_up is 19456) and returns
    the ``127/max`` multiplier scale; not usable here.
  * ``npu_dequant_swiglu_quant``: bf16 passthrough mode (scales=None,
    activate_left=True, quant_mode=1, swiglu_mode=0) fuses SwiGLU +
    dynamic quant with the ``max/127`` scale convention, and its H limit
    (20496, 64-aligned) admits IM=9728.  Gated on CANN
    ``aclnnDequantSwigluQuantV2`` (op-plugin tests skip until 8.3.RC1).

The candidate is checked against the separate chain for numerical
agreement, ACL graph capture/replay parity, and eager/graph timing at
decode-sized token counts (M = B*15 draft tokens).  No Triton kernels
are involved in this probe.

Conclusion (2026-08-11): ``npu_dequant_swiglu_quant`` showed no
meaningful benefit on the NPU, so no MLP tail fusion is implemented in
the service; the separate chain is kept.  This probe is documentation.

Before that, a small dense/grouped sanity matrix runs the op with tiny
matrices to separate "dense (E=1) unsupported" from "A8W8 N<=10240
limit" (torch_npu 2.10.0's own A8W8 test only covers E=2, and its E=1
A4W4 test is skipped due to an outdated CANN version).

Run directly on an NPU:
    python benchmarks/probe_mlp_fusion.py
"""

from __future__ import annotations

import time

import torch
import torch_npu

# The fused aclnn op may be vendored like rms_norm_dynamic_quant.
try:
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env(include_vendor_lib=True)
except Exception:  # noqa: BLE001
    pass

K = 2560    # draft hidden size
IM = 9728   # mlp intermediate size
MS = (15, 60, 120, 240)

# Fused-vs-separate quantization noise is ~1% relative (scale/rounding
# differences), so use a relative bound with an absolute floor.
TOL_X8 = 1.0
TOL_SCALE = 1e-2
TOL_OUT_REL = 0.02
TOL_OUT_ABS = 5.0


def _squeeze_scale(s: torch.Tensor) -> torch.Tensor:
    return s.squeeze(-1) if s.dim() == 2 else s


def _build_weights(device: str):
    """Random int8 gate_up/down weights + fp32 per-channel scales."""
    w_gu_int = torch.randint(
        -128, 127, (2 * IM, K), dtype=torch.int32, device=device
    )
    w_gu = w_gu_int.to(torch.int8).t().contiguous()  # [K, 2*IM] ND
    # Realistic per-channel weight scales (typical W8A8 model magnitudes).
    s_gu = torch.rand(2 * IM, device=device) * 0.003 + 0.001

    w_down_int = torch.randint(
        -128, 127, (K, IM), dtype=torch.int32, device=device
    )
    w_down = w_down_int.to(torch.int8).t().contiguous()  # [IM, K] ND
    s_down = torch.rand(K, device=device) * 0.003 + 0.001

    return w_gu, s_gu, w_down, s_down


def _chain_sep(x8, x8s, w_gu, s_gu, w_down, s_down, dtype):
    """Current chain: gate_up matmul -> swiglu -> quant -> down matmul."""
    gate_up = torch_npu.npu_quant_matmul(
        x8, w_gu, s_gu, pertoken_scale=x8s, bias=None,
        output_dtype=dtype,
    )
    act = torch_npu.npu_swiglu(gate_up)
    d8, ds = torch_npu.npu_dynamic_quant(act)
    ds = _squeeze_scale(ds)
    down = torch_npu.npu_quant_matmul(
        d8, w_down, s_down, pertoken_scale=ds, bias=None,
        output_dtype=dtype,
    )
    return act, d8, ds, down


def _chain_deq_swiglu_quant(x8, x8s, w_gu, s_gu, w_down, s_down, dtype):
    """gate_up matmul + npu_dequant_swiglu_quant + down matmul."""
    gate_up = torch_npu.npu_quant_matmul(
        x8, w_gu, s_gu, pertoken_scale=x8s, bias=None,
        output_dtype=dtype,
    )
    d8, ds = torch_npu.npu_dequant_swiglu_quant(
        gate_up,
        weight_scale=None,
        activation_scale=None,
        bias=None,
        quant_scale=None,
        quant_offset=None,
        group_index=None,
        activate_left=True,
        quant_mode=1,
        swiglu_mode=0,
    )
    ds = _squeeze_scale(ds)
    down = torch_npu.npu_quant_matmul(
        d8, w_down, s_down, pertoken_scale=ds, bias=None,
        output_dtype=dtype,
    )
    return d8, ds, down


def _v2_sanity_case(name, e, m, k, n, device) -> bool:
    """Minimal 5D-frac v2 call to isolate E=1 vs N-limit failures."""
    x8 = torch.randint(-128, 127, (m, k), dtype=torch.int8, device=device)
    w = torch.randint(-128, 127, (e, k, n), dtype=torch.int8, device=device)
    w5d = (
        w.reshape(e, k // 16, 16, n // 32, 32)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
    )
    w_scale = torch.rand(e, n, device=device) * 0.003 + 0.001
    x_scale = torch.rand(m, device=device) * 0.04 + 0.01
    if e == 1:
        gl = torch.tensor([m], dtype=torch.int64, device=device)
    else:
        gl = torch.tensor(
            [m // 2, m], dtype=torch.int64, device=device
        )
    try:
        out, out_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
            x8,
            [w5d],
            [w_scale],
            x_scale,
            gl,
            smooth_scale=None,
            dequant_mode=0,
            dequant_dtype=torch.float32,
            group_list_type=0,
        )
        print(
            f"v2 sanity {name:34s}: OK "
            f"out={tuple(out.shape)} scale={tuple(out_scale.shape)}",
            flush=True,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        print(
            f"v2 sanity {name:34s}: FAIL "
            f"{type(exc).__name__}: {str(exc)[:300]}",
            flush=True,
        )
        return False


def _check_quant(name, d8_f, ds_f, d8_r, ds_r) -> bool:
    err_x8 = (d8_f.float() - d8_r.float()).abs().max().item()
    mismatch = (d8_f != d8_r).sum().item()
    err_s = (ds_f.float() - ds_r.float()).abs().max().item()
    ref_scale = ds_r.float().abs().max().item()
    scale_bound = max(TOL_SCALE, 0.02 * ref_scale)
    ok = err_x8 <= TOL_X8 and err_s <= scale_bound
    ratio = ds_f.float() / ds_r.float().clamp_min(1e-9)
    print(
        f"{name:46s} x8_err={err_x8:.4f} x8_mismatch={mismatch} "
        f"scale_err={err_s:.6f} (rel={err_s / max(ref_scale, 1e-9):.4%}, "
        f"bound={scale_bound:.4f}, ref_scale_max={ref_scale:.4f}, "
        f"scale_ratio[min/mean/max]="
        f"{ratio.min().item():.4f}/{ratio.mean().item():.4f}/"
        f"{ratio.max().item():.4f}) {'OK' if ok else 'FAIL'}",
        flush=True,
    )
    return ok


def _check_out(name, out_f, out_r) -> bool:
    err = (out_f.float() - out_r.float()).abs().max().item()
    ref_max = out_r.float().abs().max().item()
    ok = err <= max(TOL_OUT_ABS, TOL_OUT_REL * ref_max)
    print(
        f"{name:46s} err={err:.4f} ref_max={ref_max:.4f} "
        f"{'OK' if ok else 'FAIL'}",
        flush=True,
    )
    return ok


def _time(fn, iters: int, warmup: int, graph: bool) -> float:
    if graph:
        g = torch.npu.NPUGraph()
        stream = torch.npu.Stream()
        with torch.npu.graph(g, stream=stream, capture_error_mode="global"):
            fn()
        for _ in range(warmup):
            g.replay()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            g.replay()
        torch.npu.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main() -> None:
    print(f"torch_npu version: {getattr(torch_npu, '__version__', 'unknown')}")
    torch.npu.config.allow_internal_format = True
    print("allow_internal_format=True (service-like)")
    print(f"K={K} IM={IM} Ms={MS}")

    device = "npu"
    torch.manual_seed(0)
    dtype = torch.bfloat16

    print("-" * 60)
    print("v2 sanity matrix (dense vs grouped, N limit):")
    sanity_cases = [
        ("official a8w8 (E2 M64 K128 N64)", 2, 64, 128, 64),
        ("dense small (E1 M64 K128 N64)", 1, 64, 128, 64),
        ("dense N10240 (E1 M64 K128)", 1, 64, 128, 10240),
        ("dense N19456 (E1 M64 K128)", 1, 64, 128, 19456),
        ("dense K2560 N64 (E1 M15)", 1, 15, 2560, 64),
        ("dense K2560 N10240 (E1 M15)", 1, 15, 2560, 10240),
        ("dense K2560 N19456 (E1 M15)", 1, 15, 2560, 19456),
        ("grouped K2560 N19456 (E2 M30)", 2, 30, 2560, 19456),
    ]
    for name, e, m, k, n in sanity_cases:
        _v2_sanity_case(name, e, m, k, n, device)

    w_gu, s_gu, w_down, s_down = _build_weights(device)

    m0 = MS[0]
    x8 = torch.randint(-128, 127, (m0, K), dtype=torch.int8, device=device)
    # Realistic per-token activation scales from npu_dynamic_quant.
    x8s = torch.rand(m0, device=device) * 0.04 + 0.01

    print("-" * 60)
    print("correctness (fused vs separate, M=15):")
    all_ok = True
    act_r, d8_r, ds_r, down_r = _chain_sep(
        x8, x8s, w_gu, s_gu, w_down, s_down, dtype
    )
    try:
        d8_d, ds_d, down_d = _chain_deq_swiglu_quant(
            x8, x8s, w_gu, s_gu, w_down, s_down, dtype
        )
        all_ok &= _check_quant(
            "dequant_swiglu_quant", d8_d, ds_d, d8_r, ds_r
        )
        all_ok &= _check_out(
            "dequant_swiglu_quant down out", down_d, down_r
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"dequant_swiglu_quant: FAIL "
            f"{type(exc).__name__}: {str(exc)[:300]}",
            flush=True,
        )
        all_ok = False

    print("-" * 60)
    print("graph capture / replay parity (replay vs eager):")
    cases = [
        (
            "sep (qmatmul+swiglu+dq+down)",
            lambda: _chain_sep(
                x8, x8s, w_gu, s_gu, w_down, s_down, dtype
            ),
        ),
        (
            "dequant_swiglu_quant (+down)",
            lambda: _chain_deq_swiglu_quant(
                x8, x8s, w_gu, s_gu, w_down, s_down, dtype
            ),
        ),
    ]
    for name, fn in cases:
        for mode in ("global", "relaxed"):
            try:
                eager_out = fn()
                if not isinstance(eager_out, (tuple, list)):
                    eager_out = (eager_out,)
                g = torch.npu.NPUGraph()
                stream = torch.npu.Stream()
                with torch.npu.graph(
                    g, stream=stream, capture_error_mode=mode
                ):
                    graph_out = fn()
                if not isinstance(graph_out, (tuple, list)):
                    graph_out = (graph_out,)
                g.replay()
                torch.npu.synchronize()
                err = 0.0
                for a, b in zip(graph_out, eager_out):
                    err = max(
                        err, (a.float() - b.float()).abs().max().item()
                    )
                print(
                    f"{name:44s} graph[{mode:8s}] replay_err={err:.6f} "
                    f"{'OK' if err == 0.0 else 'FAIL'}",
                    flush=True,
                )
                all_ok &= err == 0.0
            except Exception as exc:  # noqa: BLE001
                print(
                    f"{name:44s} graph[{mode:8s}] FAIL "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                all_ok = False

    print("-" * 60)
    print("timing (ms per call, eager vs graph):")
    iters, warmup = 20, 5
    for m in MS:
        x8_m = torch.randint(
            -128, 127, (m, K), dtype=torch.int8, device=device
        )
        x8s_m = torch.rand(m, device=device) * 0.04 + 0.01
        runs = [
            (
                "sep chain",
                lambda: _chain_sep(
                    x8_m, x8s_m, w_gu, s_gu, w_down, s_down, dtype
                ),
            ),
            (
                "deq_swiglu_quant chain",
                lambda: _chain_deq_swiglu_quant(
                    x8_m, x8s_m, w_gu, s_gu, w_down, s_down, dtype
                ),
            ),
        ]
        print(f"M={m}:", flush=True)
        for name, fn in runs:
            try:
                ms_e = _time(fn, iters, warmup, graph=False)
                ms_g = _time(fn, iters, warmup, graph=True)
                print(
                    f"  {name:22s} eager={ms_e:.3f} ms "
                    f"graph={ms_g:.3f} ms",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  {name:22s} FAIL {type(exc).__name__}: {exc}",
                    flush=True,
                )

    print("=" * 60)
    print("RESULT:", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
