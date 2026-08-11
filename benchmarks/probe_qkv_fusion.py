#!/usr/bin/env python3
# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""NPU probe for fusing the Domino draft q/k/v projections.

Domino's draft attention currently runs three separate projections per layer
(``q_proj`` 2560->4096, ``k_proj``/``v_proj`` 2560->1024).  DFlash instead
uses one fused ``qkv_proj`` (2560->6144).  This probe compares, on real
Domino dims:

  * separate: 3 projection calls per layer,
  * fused:    1 projection call per layer,

for four schemes:

  * bf16 (plain ``F.linear``),
  * W4A8 (``npu_weight_quant_batchmatmul``, int32-packed int4; the fused
    weight is single-packed from the concatenated q+k+v int4 matrix),
  * W4A4 (``npu_dynamic_quant`` + ``npu_quant_matmul``),
  * mixed (layer 0 W4A4, layers 1-6 W4A8 -- the real Domino config).

Each case checks fused-vs-separate numerical agreement and times 7 sequential
layers (eager + ACL graph replay) at decode-sized token counts
(M = B*15 draft tokens).

Run directly on an NPU:
    python benchmarks/probe_qkv_fusion.py
"""

from __future__ import annotations

import time

import torch
import torch_npu

ACL_FORMAT_ND = 2

K = 2560   # draft hidden size
NQ = 4096  # 32 heads * 128
NKV = 1024  # 8 kv heads * 128
NQKV = NQ + 2 * NKV
D = 7
MS = (15, 60, 120, 240)  # B=1,4,8,16 requests * 15 draft tokens
TOLERANCE = 0.05


def _pack_w4a8(w_int: torch.Tensor) -> torch.Tensor:
    """[N, K] int4 -> [K, N//8] int32 ND (Domino W4A8 layout)."""
    packed = torch_npu.npu_convert_weight_to_int4pack(
        w_int.t().contiguous()
    )
    return torch_npu.npu_format_cast(packed, ACL_FORMAT_ND)


def _pack_w4a4(w_int: torch.Tensor) -> torch.Tensor:
    """[N, K] int4 -> [K//8, N] int32 (Domino W4A4 layout)."""
    return torch_npu.npu_convert_weight_to_int4pack(
        w_int.contiguous()
    ).transpose(-1, -2)


def _build_case(scheme: str):
    """Per-layer separate + fused weights for one scheme."""
    device = "npu"
    sep = []
    fused = None
    if scheme == "bf16":
        for _ in range(D):
            w = torch.randn(NQKV, K, dtype=torch.bfloat16, device=device)
            sep.append(
                {
                    "q": w[:NQ],
                    "k": w[NQ:NQ + NKV],
                    "v": w[NQ + NKV:],
                }
            )
        fused = None  # full W is implicit
        return sep, fused

    if scheme == "w4a8":
        fused = []
        for _ in range(D):
            w_int = torch.randint(-7, 8, (NQKV, K), dtype=torch.int32, device=device)
            scale = (torch.rand(NQKV, device=device) * 0.9 + 0.1)
            sep.append(
                {
                    "q": (
                        _pack_w4a8(w_int[:NQ]),
                        scale[:NQ].to(torch.bfloat16),
                    ),
                    "k": (
                        _pack_w4a8(w_int[NQ:NQ + NKV]),
                        scale[NQ:NQ + NKV].to(torch.bfloat16),
                    ),
                    "v": (
                        _pack_w4a8(w_int[NQ + NKV:]),
                        scale[NQ + NKV:].to(torch.bfloat16),
                    ),
                }
            )
            fused.append(
                (
                    _pack_w4a8(w_int),
                    scale.to(torch.bfloat16),
                )
            )
        return sep, fused

    if scheme == "w8a8":
        fused = []
        for _ in range(D):
            w_int = torch.randint(
                -128, 127, (NQKV, K), dtype=torch.int32, device=device
            )
            scale = (torch.rand(NQKV, device=device) * 0.9 + 0.1)
            sep.append(
                {
                    "q": (
                        w_int[:NQ].to(torch.int8).t().contiguous(),
                        scale[:NQ],
                    ),
                    "k": (
                        w_int[NQ:NQ + NKV].to(torch.int8).t().contiguous(),
                        scale[NQ:NQ + NKV],
                    ),
                    "v": (
                        w_int[NQ + NKV:].to(torch.int8).t().contiguous(),
                        scale[NQ + NKV:],
                    ),
                }
            )
            fused.append(
                (
                    w_int.to(torch.int8).t().contiguous(),
                    scale,
                )
            )
        return sep, fused

    if scheme == "w4a4":
        fused = []
        for _ in range(D):
            w_int = torch.randint(-7, 8, (NQKV, K), dtype=torch.int32, device=device)
            scale = (torch.rand(NQKV, device=device) * 0.9 + 0.1)
            sep.append(
                {
                    "q": (_pack_w4a4(w_int[:NQ]), scale[:NQ]),
                    "k": (_pack_w4a4(w_int[NQ:NQ + NKV]), scale[NQ:NQ + NKV]),
                    "v": (_pack_w4a4(w_int[NQ + NKV:]), scale[NQ + NKV:]),
                }
            )
            fused.append((_pack_w4a4(w_int), scale))
        return sep, fused

    raise ValueError(scheme)


def _proj_bf16(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x, w)


def _proj_w4a8(x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor):
    return torch_npu.npu_weight_quant_batchmatmul(
        x, packed, antiquant_scale=scale, antiquant_group_size=0
    )


def _proj_w4a4(x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor):
    x4, x4s = torch_npu.npu_dynamic_quant(x, dst_type=torch.quint4x2)
    return torch_npu.npu_quant_matmul(
        x4,
        packed,
        scale=scale.view(-1),
        pertoken_scale=x4s.reshape(-1),
        bias=None,
        output_dtype=torch.float16,
    ).to(x.dtype)


def _proj_w8a8(x: torch.Tensor, w8: torch.Tensor, scale: torch.Tensor):
    x8, x8s = torch_npu.npu_dynamic_quant(x)
    if x8s.dim() == 2:
        x8s = x8s.squeeze(1)
    return torch_npu.npu_quant_matmul(
        x8,
        w8,
        scale,
        pertoken_scale=x8s,
        bias=None,
        output_dtype=x.dtype,
    )


def _run_sep(scheme: str, x: torch.Tensor, sep) -> torch.Tensor:
    outs = []
    for l in range(D):
        if scheme == "bf16":
            q = _proj_bf16(x, sep[l]["q"])
            k = _proj_bf16(x, sep[l]["k"])
            v = _proj_bf16(x, sep[l]["v"])
        elif scheme == "w4a8":
            q = _proj_w4a8(x, *sep[l]["q"])
            k = _proj_w4a8(x, *sep[l]["k"])
            v = _proj_w4a8(x, *sep[l]["v"])
        elif scheme == "w8a8":
            q = _proj_w8a8(x, *sep[l]["q"])
            k = _proj_w8a8(x, *sep[l]["k"])
            v = _proj_w8a8(x, *sep[l]["v"])
        else:  # w4a4
            q = _proj_w4a4(x, *sep[l]["q"])
            k = _proj_w4a4(x, *sep[l]["k"])
            v = _proj_w4a4(x, *sep[l]["v"])
        outs.append(torch.cat([q, k, v], dim=-1))
    return torch.stack(outs, dim=0)


def _run_fused(scheme: str, x: torch.Tensor, fused) -> torch.Tensor:
    outs = []
    for l in range(D):
        if scheme == "bf16":
            w = fused[l]
            qkv = _proj_bf16(x, w)
        elif scheme == "w4a8":
            qkv = _proj_w4a8(x, *fused[l])
        elif scheme == "w8a8":
            qkv = _proj_w8a8(x, *fused[l])
        else:  # w4a4
            qkv = _proj_w4a4(x, *fused[l])
        outs.append(qkv)
    return torch.stack(outs, dim=0)


def _run_mixed_sep(x: torch.Tensor, sep_w4a4, sep_w4a8) -> torch.Tensor:
    outs = []
    for l in range(D):
        sep = sep_w4a4 if l == 0 else sep_w4a8
        q = _proj_w4a4(x, *sep[l]["q"]) if l == 0 else _proj_w4a8(x, *sep[l]["q"])
        k = _proj_w4a4(x, *sep[l]["k"]) if l == 0 else _proj_w4a8(x, *sep[l]["k"])
        v = _proj_w4a4(x, *sep[l]["v"]) if l == 0 else _proj_w4a8(x, *sep[l]["v"])
        outs.append(torch.cat([q, k, v], dim=-1))
    return torch.stack(outs, dim=0)


def _run_mixed_fused(x: torch.Tensor, fused_w4a4, fused_w4a8) -> torch.Tensor:
    outs = []
    for l in range(D):
        if l == 0:
            outs.append(_proj_w4a4(x, *fused_w4a4[l]))
        else:
            outs.append(_proj_w4a8(x, *fused_w4a8[l]))
    return torch.stack(outs, dim=0)


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
    print(f"K={K} NQ={NQ} NKV={NKV} NQKV={NQKV} D={D} Ms={MS}")

    device = "npu"
    torch.manual_seed(0)

    sep_bf16, _ = _build_case("bf16")
    sep_w4a8, fused_w4a8 = _build_case("w4a8")
    sep_w8a8, fused_w8a8 = _build_case("w8a8")
    sep_w4a4, fused_w4a4 = _build_case("w4a4")

    x = torch.randn(MS[0], K, dtype=torch.bfloat16, device=device)

    print("-" * 60)
    print("correctness (fused vs separate, M=15):")
    for name, out_sep, out_fused in [
        ("bf16", _run_sep("bf16", x, sep_bf16),
         _run_fused("bf16", x, [torch.cat(
             [sep_bf16[l]["q"], sep_bf16[l]["k"], sep_bf16[l]["v"]], dim=0
         ) for l in range(D)])),
        ("w4a8", _run_sep("w4a8", x, sep_w4a8),
         _run_fused("w4a8", x, fused_w4a8)),
        ("w8a8", _run_sep("w8a8", x, sep_w8a8),
         _run_fused("w8a8", x, fused_w8a8)),
        ("w4a4", _run_sep("w4a4", x, sep_w4a4),
         _run_fused("w4a4", x, fused_w4a4)),
        ("mixed", _run_mixed_sep(x, sep_w4a4, sep_w4a8),
         _run_mixed_fused(x, fused_w4a4, fused_w4a8)),
    ]:
        err = (out_sep.float() - out_fused.float()).abs().max().item()
        print(
            f"{name:8s} max_err={err:.6f} {'OK' if err <= TOLERANCE else 'FAIL'}",
            flush=True,
        )

    print("-" * 60)
    print("timing (ms per 7-layer projection pass, eager vs graph):")
    iters, warmup = 20, 5
    for m in MS:
        x_m = torch.randn(m, K, dtype=torch.bfloat16, device=device)
        runs = [
            ("bf16 sep", lambda: _run_sep("bf16", x_m, sep_bf16)),
            ("bf16 fus", lambda: _run_fused(
                "bf16", x_m,
                [torch.cat(
                    [sep_bf16[l]["q"], sep_bf16[l]["k"], sep_bf16[l]["v"]],
                    dim=0,
                ) for l in range(D)],
            )),
            ("w4a8 sep", lambda: _run_sep("w4a8", x_m, sep_w4a8)),
            ("w4a8 fus", lambda: _run_fused("w4a8", x_m, fused_w4a8)),
            ("w8a8 sep", lambda: _run_sep("w8a8", x_m, sep_w8a8)),
            ("w8a8 fus", lambda: _run_fused("w8a8", x_m, fused_w8a8)),
            ("w4a4 sep", lambda: _run_sep("w4a4", x_m, sep_w4a4)),
            ("w4a4 fus", lambda: _run_fused("w4a4", x_m, fused_w4a4)),
            ("mixed sep", lambda: _run_mixed_sep(x_m, sep_w4a4, sep_w4a8)),
            ("mixed fus", lambda: _run_mixed_fused(x_m, fused_w4a4, fused_w4a8)),
        ]
        print(f"M={m}:", flush=True)
        for name, fn in runs:
            try:
                ms_e = _time(fn, iters, warmup, graph=False)
                ms_g = _time(fn, iters, warmup, graph=True)
                print(
                    f"  {name:10s} eager={ms_e:.3f} ms graph={ms_g:.3f} ms",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  {name:10s} FAIL {type(exc).__name__}: {exc}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
