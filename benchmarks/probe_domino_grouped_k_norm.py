#!/usr/bin/env python3
# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""NPU probe for the fused grouped k-norm in the Domino precompute.

Validates ``domino_grouped_k_norm`` (one RMSNorm kernel over all 7 layers with
per-layer weights) against the per-layer ``torch_npu.npu_rms_norm`` loop,
checks graph-replay parity, and times both paths (eager + ACL graph) on real
Domino dims (D=7, nkv=8, hd=128, T=8).

Run directly on an NPU:
    python benchmarks/probe_domino_grouped_k_norm.py
"""

from __future__ import annotations

import time

import torch
import torch_npu

from vllm_ascend.ops.triton.spec_decode.domino_kv_utils import (
    domino_grouped_k_norm,
)

D, T, NKV, HD = 7, 8, 8, 128
EPS = 1e-6
TOLERANCE = 0.05


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
    print(f"D={D} T={T} NKV={NKV} HD={HD}")

    device = "npu"
    torch.manual_seed(0)
    all_k = torch.randn(D, T, NKV, HD, dtype=torch.bfloat16, device=device)
    w = torch.randn(D, HD, dtype=torch.bfloat16, device=device)

    ref = torch.empty_like(all_k)
    for l in range(D):
        normed, _ = torch_npu.npu_rms_norm(
            all_k[l].reshape(T * NKV, HD), w[l], EPS
        )
        ref[l] = normed.view(T, NKV, HD)
    torch.npu.synchronize()

    out = domino_grouped_k_norm(all_k, w, EPS)
    torch.npu.synchronize()
    err = (out.float() - ref.float()).abs().max().item()
    print(f"max_err={err:.6f} {'OK' if err <= TOLERANCE else 'FAIL'}")

    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    with torch.npu.graph(graph, stream=stream, capture_error_mode="global"):
        out_g = domino_grouped_k_norm(all_k, w, EPS)
    graph.replay()
    torch.npu.synchronize()
    replay_err = (out_g.float() - out.float()).abs().max().item()
    print(
        f"graph replay vs eager max_err={replay_err:.6f} "
        f"{'OK' if replay_err == 0.0 else 'FAIL'}"
    )

    print("-" * 60)
    print("timing (ms per call, eager vs graph):")
    iters, warmup = 20, 5

    def fused():
        return domino_grouped_k_norm(all_k, w, EPS)

    def loop():
        out_loop = torch.empty_like(all_k)
        for l in range(D):
            normed, _ = torch_npu.npu_rms_norm(
                all_k[l].reshape(T * NKV, HD), w[l], EPS
            )
            out_loop[l] = normed.view(T, NKV, HD)
        return out_loop

    for name, fn in [("fused", fused), ("loop", loop)]:
        try:
            ms_e = _time(fn, iters, warmup, graph=False)
            ms_g = _time(fn, iters, warmup, graph=True)
            print(
                f"  {name:6s} eager={ms_e:.3f} ms graph={ms_g:.3f} ms",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"  {name:6s} FAIL {type(exc).__name__}: {exc}",
                flush=True,
            )


if __name__ == "__main__":
    main()
