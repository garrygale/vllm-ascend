#!/usr/bin/env python3
# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""NPU probe for the fused Domino context-KV k-norm and cache write.

Validates ``domino_grouped_k_norm`` (one RMSNorm kernel over all layers) and
``fused_kv_cache_write`` (one kernel writing K/V into all 7 layer caches)
against the per-layer references:

  * k-norm: per-layer ``torch_npu.npu_rms_norm``,
  * cache write: per-layer ``torch_npu.npu_scatter_pa_kv_cache`` (the exact op
    used by the service today) plus a plain torch flat-slot copy.

Checks numerical parity, graph-replay parity, and timings (eager + ACL graph)
on real Domino dims (D=7, nkv=8, hd=128, T=8).

Run directly on an NPU:
    python benchmarks/probe_domino_fused_kv.py
"""

from __future__ import annotations

import time

import torch
import torch_npu

from vllm_ascend.ops.triton.spec_decode.domino_kv_utils import (
    domino_grouped_k_norm,
    fused_kv_cache_write,
)

D, T, NKV, HD = 7, 8, 8, 128
EPS = 1e-6
BLOCKS, BLOCK_SIZE = 4, 16
NSLOTS = BLOCKS * BLOCK_SIZE
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


def _build(device):
    torch.manual_seed(0)
    all_k = torch.randn(D, T, NKV, HD, dtype=torch.bfloat16, device=device)
    all_v = torch.randn(D, T, NKV, HD, dtype=torch.bfloat16, device=device)
    w = torch.randn(D, HD, dtype=torch.bfloat16, device=device)
    slots = torch.randint(
        0, NSLOTS, (D, T), dtype=torch.int32, device=device
    )
    key_caches = [
        torch.zeros(BLOCKS, BLOCK_SIZE, NKV, HD, dtype=torch.bfloat16,
                    device=device)
        for _ in range(D)
    ]
    value_caches = [
        torch.zeros(BLOCKS, BLOCK_SIZE, NKV, HD, dtype=torch.bfloat16,
                    device=device)
        for _ in range(D)
    ]
    return all_k, all_v, w, slots, key_caches, value_caches


def main() -> None:
    print(f"torch_npu version: {getattr(torch_npu, '__version__', 'unknown')}")
    torch.npu.config.allow_internal_format = True
    print(f"D={D} T={T} NKV={NKV} HD={HD} slots={NSLOTS}")

    device = "npu"
    all_k, all_v, w, slots, key_caches, value_caches = _build(device)

    print("-" * 60)
    print("grouped k-norm:")
    ref = torch.empty_like(all_k)
    for l in range(D):
        normed, _ = torch_npu.npu_rms_norm(
            all_k[l].reshape(T * NKV, HD), w[l], EPS
        )
        ref[l] = normed.view(T, NKV, HD)
    out = domino_grouped_k_norm(all_k, w, EPS)
    err = (out.float() - ref.float()).abs().max().item()
    print(f"  max_err={err:.6f} {'OK' if err <= TOLERANCE else 'FAIL'}")

    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    with torch.npu.graph(graph, stream=stream, capture_error_mode="global"):
        out_g = domino_grouped_k_norm(all_k, w, EPS)
    graph.replay()
    torch.npu.synchronize()
    replay_err = (out_g.float() - out.float()).abs().max().item()
    print(
        f"  graph replay vs eager max_err={replay_err:.6f} "
        f"{'OK' if replay_err == 0.0 else 'FAIL'}"
    )

    print("-" * 60)
    print("fused cache write:")
    # Reference 1: per-layer npu_scatter_pa_kv_cache (production op).
    ref_k = [
        torch.zeros_like(key_caches[l]) for l in range(D)
    ]
    ref_v = [
        torch.zeros_like(value_caches[l]) for l in range(D)
    ]
    for l in range(D):
        torch_npu.npu_scatter_pa_kv_cache(
            key=all_k[l].contiguous(),
            value=all_v[l].contiguous(),
            key_cache=ref_k[l],
            value_cache=ref_v[l],
            slot_mapping=slots[l].contiguous(),
            cache_mode="Norm",
        )
    # Reference 2: plain flat-slot copy.
    ref2_k = [torch.zeros_like(key_caches[l]) for l in range(D)]
    ref2_v = [torch.zeros_like(value_caches[l]) for l in range(D)]
    for l in range(D):
        ref2_k[l].view(-1, NKV, HD)[slots[l]] = all_k[l]
        ref2_v[l].view(-1, NKV, HD)[slots[l]] = all_v[l]
    torch.npu.synchronize()

    scatter_vs_flat = max(
        (ref_k[l].float() - ref2_k[l].float()).abs().max().item()
        for l in range(D)
    )
    print(
        f"  reference sanity: scatter vs flat-copy max_err="
        f"{scatter_vs_flat:.6f}"
    )

    fused_ok = fused_kv_cache_write(
        all_k, all_v, key_caches, value_caches, [slots[l] for l in range(D)]
    )
    torch.npu.synchronize()
    print(f"  fused_kv_cache_write returned {fused_ok}")
    if fused_ok:
        err_prod = max(
            (key_caches[l].float() - ref_k[l].float()).abs().max().item()
            for l in range(D)
        )
        err_torch = max(
            (key_caches[l].float() - ref2_k[l].float()).abs().max().item()
            for l in range(D)
        )
        err_v = max(
            (value_caches[l].float() - ref2_v[l].float()).abs().max().item()
            for l in range(D)
        )
        print(
            f"  vs scatter_pa_kv_cache max_err={err_prod:.6f} "
            f"{'OK' if err_prod == 0.0 else 'FAIL'}"
        )
        print(
            f"  vs flat-copy K/V max_err={max(err_torch, err_v):.6f} "
            f"{'OK' if max(err_torch, err_v) == 0.0 else 'FAIL'}"
        )

        # Graph replay must reproduce the same writes.
        for l in range(D):
            key_caches[l].zero_()
            value_caches[l].zero_()
        graph = torch.npu.NPUGraph()
        stream = torch.npu.Stream()
        with torch.npu.graph(graph, stream=stream,
                             capture_error_mode="global"):
            fused_ok_g = fused_kv_cache_write(
                all_k, all_v, key_caches, value_caches,
                [slots[l] for l in range(D)],
            )
        graph.replay()
        torch.npu.synchronize()
        err_g = max(
            (key_caches[l].float() - ref2_k[l].float()).abs().max().item()
            for l in range(D)
        )
        print(
            f"  graph replay vs flat-copy max_err={err_g:.6f} "
            f"{'OK' if err_g == 0.0 and fused_ok_g else 'FAIL'}"
        )

    print("-" * 60)
    print("timing (ms per call, eager vs graph):")
    iters, warmup = 20, 5

    def k_norm_fused():
        return domino_grouped_k_norm(all_k, w, EPS)

    def k_norm_loop():
        out_loop = torch.empty_like(all_k)
        for l in range(D):
            normed, _ = torch_npu.npu_rms_norm(
                all_k[l].reshape(T * NKV, HD), w[l], EPS
            )
            out_loop[l] = normed.view(T, NKV, HD)
        return out_loop

    def cache_fused():
        for l in range(D):
            key_caches[l].zero_()
            value_caches[l].zero_()
        return fused_kv_cache_write(
            all_k, all_v, key_caches, value_caches,
            [slots[l] for l in range(D)],
        )

    def cache_loop():
        for l in range(D):
            key_caches[l].zero_()
            value_caches[l].zero_()
        for l in range(D):
            torch_npu.npu_scatter_pa_kv_cache(
                key=all_k[l].contiguous(),
                value=all_v[l].contiguous(),
                key_cache=key_caches[l],
                value_cache=value_caches[l],
                slot_mapping=slots[l].contiguous(),
                cache_mode="Norm",
            )
        return True

    for name, fn in [
        ("k-norm fused", k_norm_fused),
        ("k-norm loop", k_norm_loop),
        ("cache fused", cache_fused),
        ("cache loop", cache_loop),
    ]:
        try:
            ms_e = _time(fn, iters, warmup, graph=False)
            ms_g = _time(fn, iters, warmup, graph=True)
            print(
                f"  {name:14s} eager={ms_e:.3f} ms graph={ms_g:.3f} ms",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"  {name:14s} FAIL {type(exc).__name__}: {exc}",
                flush=True,
            )


if __name__ == "__main__":
    main()
