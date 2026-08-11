#!/usr/bin/env python3
# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""NPU probe: flare fusion einsum vs broadcast matmul.

``combine_hidden_states`` computes ``[D, T] @ [N, T, H] -> [N, D, H]``
for the Domino flare fusion.  The service now uses ``torch.matmul``
instead of ``torch.einsum`` (the generic aclnnEinsum path is slow on
NPU; specforge observed the same by avoiding einsum).  This probe times
both forms at the real Domino dims (D=7, T=9, H=4096) for
M = B*15 draft tokens, eager and in an ACL graph.

The softmax over the fusion weights is intentionally NOT inside the
timed functions: it is identical for both forms and would only dilute
the einsum-vs-matmul difference.

Run directly on an NPU:
    python benchmarks/probe_flare_fusion.py
"""

from __future__ import annotations

import time

import torch
import torch_npu

D = 7          # num draft layers
T = 9          # num target features
H = 4096       # target hidden size
MS = (15, 60, 120, 240)
ITERS = 200
WARMUP = 20


def _flare_einsum(fusion_w: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Original service form: einsum + permute + reshape."""
    fused = torch.einsum("dt,nth->dnh", fusion_w, target)
    return fused.permute(1, 0, 2).reshape(-1, D * H)


def _flare_matmul(fusion_w: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Current service form: broadcast matmul + reshape."""
    return torch.matmul(fusion_w, target).reshape(-1, D * H)


def _time(fn, graph: bool) -> float:
    if graph:
        g = torch.npu.NPUGraph()
        stream = torch.npu.Stream()
        with torch.npu.graph(g, stream=stream, capture_error_mode="global"):
            fn()
        for _ in range(WARMUP):
            g.replay()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            g.replay()
        torch.npu.synchronize()
        return (time.perf_counter() - t0) / ITERS * 1e3
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e3


def main() -> None:
    print(f"torch_npu version: {getattr(torch_npu, '__version__', 'unknown')}")
    device = "npu"
    torch.manual_seed(0)
    dtype = torch.bfloat16
    fusion_w = torch.randn(D, T, dtype=dtype, device=device)
    print(f"D={D} T={T} H={H} Ms={MS} dtype={dtype}")

    print("-" * 60)
    print("correctness (matmul vs einsum, M=15):")
    m0 = MS[0]
    target0 = torch.randn(m0, T * H, dtype=dtype, device=device).view(
        m0, T, H
    )
    w0 = torch.softmax(fusion_w, dim=1)
    y_e = _flare_einsum(w0, target0)
    y_m = _flare_matmul(w0, target0)
    err = (y_e.float() - y_m.float()).abs().max().item()
    print(
        f"max abs diff einsum vs matmul: {err:.6f} "
        f"shapes={tuple(y_e.shape)}/{tuple(y_m.shape)} "
        f"{'OK' if err == 0.0 else 'CHECK'}",
        flush=True,
    )

    print("-" * 60)
    print("timing (us per call, eager vs ACL graph):")
    for m in MS:
        target_m = torch.randn(m, T * H, dtype=dtype, device=device).view(
            m, T, H
        )
        w_m = torch.softmax(fusion_w, dim=1)
        runs = [
            ("einsum", lambda: _flare_einsum(w_m, target_m)),
            ("matmul", lambda: _flare_matmul(w_m, target_m)),
        ]
        print(f"M={m}:", flush=True)
        for name, fn in runs:
            for mode in ("eager", "graph"):
                try:
                    ms = _time(fn, mode == "graph")
                    print(
                        f"  {name:8s} {mode:6s} {ms * 1e3:8.2f} us",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"  {name:8s} {mode:6s} FAIL "
                        f"{type(exc).__name__}: {str(exc)[:200]}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
