#!/usr/bin/env python3
# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""NPU probe for fusing Domino's q/k RMSNorm + RoPE into one kernel.

The Domino draft attention currently runs, per layer:

  * q-norm and k-norm as two separate ``npu_rms_norm`` calls,
  * one ``npu_rotary_embedding`` call for q and k.

vllm-ascend already ships ``torch.ops.vllm.qkv_rmsnorm_rope`` (Triton) which
does split + q/k RMSNorm + RoPE in a single kernel (used by minimax_m3 and
step3p5).  This probe compares it against the current three-op path on real
Domino dims (q=4096, kv=1024, head_dim=128, bf16), checks numerical parity,
and times both eager and ACL graph replay at M = B*15 draft tokens.

Run directly on an NPU:
    python benchmarks/probe_qk_norm_rope_fusion.py
"""

from __future__ import annotations

import time

import torch
import torch_npu

import vllm_ascend.ops  # noqa: F401  (registers the custom ops)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

NQ = 4096
NKV = 1024
HEAD_DIM = 128
QKV = NQ + 2 * NKV
MAX_POS = 40960
EPS = 1e-6
MS = (15, 60, 120, 240)
TOLERANCE = 0.05


def _build_cos_sin_cache(device: torch.device, dtype: torch.dtype):
    inv_freq = 1.0 / (
        1000000.0
        ** (
            torch.arange(0, HEAD_DIM, 2, dtype=torch.float32)
            / HEAD_DIM
        )
    )
    t = torch.arange(MAX_POS, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return cache.to(device=device, dtype=dtype)


def _ref_path(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
):
    """Current path: split + two npu_rms_norm + one npu_rotary_embedding."""
    m = qkv.shape[0]
    q, k, v = qkv.split([NQ, NKV, NKV], dim=-1)
    q_normed, _ = torch_npu.npu_rms_norm(
        q.view(m, -1, HEAD_DIM), q_weight, EPS
    )
    k_normed, _ = torch_npu.npu_rms_norm(
        k.view(m, -1, HEAD_DIM), k_weight, EPS
    )
    q_rope, k_rope = torch.ops.vllm.npu_rotary_embedding(
        positions,
        q_normed.view(m, NQ),
        k_normed.view(m, NKV),
        cos_sin_cache,
        HEAD_DIM,
        HEAD_DIM,
        True,
    )
    return q_rope, k_rope, v


def _fused_path(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
):
    """Fused split + q/k RMSNorm + RoPE in one Triton kernel."""
    return torch.ops.vllm.qkv_rmsnorm_rope(
        input=qkv,
        cos_sin_cache=cos_sin_cache,
        positions=positions,
        q_weight=q_weight,
        k_weight=k_weight,
        q_hidden_size=NQ,
        kv_hidden_size=NKV,
        head_dim=HEAD_DIM,
        eps=EPS,
        q_bias=None,
        k_bias=None,
    )


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
    init_device_properties_triton()
    print("allow_internal_format=True (service-like)")
    print(f"NQ={NQ} NKV={NKV} HEAD_DIM={HEAD_DIM} Ms={MS}")

    device = "npu"
    torch.manual_seed(0)
    q_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device)
    k_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device)
    cos_sin_cache = _build_cos_sin_cache(device, torch.bfloat16)

    m0 = MS[0]
    qkv = torch.randn(m0, QKV, dtype=torch.bfloat16, device=device)
    positions = torch.randint(0, MAX_POS, (m0,), dtype=torch.int64, device=device)

    print("-" * 60)
    print(f"correctness (M={m0}):")
    q_ref, k_ref, v_ref = _ref_path(
        qkv, q_weight, k_weight, cos_sin_cache, positions
    )
    q_fus, k_fus, v_fus = _fused_path(
        qkv, q_weight, k_weight, cos_sin_cache, positions
    )
    for name, a, b in [
        ("q", q_fus, q_ref),
        ("k", k_fus, k_ref),
        ("v", v_fus, v_ref),
    ]:
        err = (a.float() - b.float()).abs().max().item()
        print(
            f"{name:3s} max_err={err:.6f} {'OK' if err <= TOLERANCE else 'FAIL'}",
            flush=True,
        )

    # Graph replay must match eager bit-for-bit.
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    with torch.npu.graph(graph, stream=stream, capture_error_mode="global"):
        q_g, k_g, v_g = _fused_path(
            qkv, q_weight, k_weight, cos_sin_cache, positions
        )
    graph.replay()
    torch.npu.synchronize()
    replay_err = max(
        (q_g.float() - q_fus.float()).abs().max().item(),
        (k_g.float() - k_fus.float()).abs().max().item(),
        (v_g.float() - v_fus.float()).abs().max().item(),
    )
    print(
        f"graph replay vs eager max_err={replay_err:.6f} "
        f"{'OK' if replay_err == 0.0 else 'FAIL'}",
        flush=True,
    )

    print("-" * 60)
    print("timing (ms per call, eager vs graph):")
    iters, warmup = 20, 5
    for m in MS:
        qkv_m = torch.randn(m, QKV, dtype=torch.bfloat16, device=device)
        pos_m = torch.randint(
            0, MAX_POS, (m,), dtype=torch.int64, device=device
        )
        runs = [
            ("sep norm+rope", lambda: _ref_path(
                qkv_m, q_weight, k_weight, cos_sin_cache, pos_m
            )),
            ("fused norm+rope", lambda: _fused_path(
                qkv_m, q_weight, k_weight, cos_sin_cache, pos_m
            )),
        ]
        print(f"M={m}:", flush=True)
        for name, fn in runs:
            try:
                ms_e = _time(fn, iters, warmup, graph=False)
                ms_g = _time(fn, iters, warmup, graph=True)
                print(
                    f"  {name:16s} eager={ms_e:.3f} ms graph={ms_g:.3f} ms",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  {name:16s} FAIL {type(exc).__name__}: {exc}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
