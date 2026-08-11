#!/usr/bin/env python3
# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""NPU probe for the Domino draft sequential top-K correction strategy.

Ports DSpark's ``_sample_sequential_topk`` idea to Domino: instead of
computing the correction head over the full vocab at every sequential step
([B,256] x [256,151936] per step), take the top-K base-logit candidates once,
gather the top-K columns of the final projection weight per position, and
compute only a [B,256] x [256,K] correction per step, scattering the results
into a pre-filled ``-inf`` full-vocab buffer (the truncated distribution is
what gets sampled and recorded, preserving rejection-sampling correctness).

Measures the per-block cost of:

  * base logits (full vocab, common),
  * full correction: 15 x full-vocab correction matmul + add,
  * topk path: topk + weight-column gather + 15 x small matmul + scatter,
  * gumbel sampling: 15 x full-vocab gumbel + draft_logits recording
    (common).

Each component is timed eager and inside an ACL graph, at B=1/4/8/16 requests
(M = B*15 draft tokens), for K=32 and K=64.

Run directly on an NPU:
    python benchmarks/probe_domino_topk.py
"""

from __future__ import annotations

import time

import torch
import torch_npu

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample

V = 151936
HM = 256      # embed_proj hidden size
N_SPEC = 15
BS = (1, 4, 8, 16)
KS = (32, 64)
TOLERANCE = 0.1


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


def _check_correctness(device) -> None:
    torch.manual_seed(0)
    b = 4
    k = 32
    x_corr = torch.randn(b, HM, dtype=torch.bfloat16, device=device)
    w_fc = torch.randn(V, HM, dtype=torch.bfloat16, device=device)
    base = torch.randn(b, N_SPEC, V, dtype=torch.bfloat16, device=device)

    vals, idx = torch.topk(base, k, dim=-1)
    w_fc_t = w_fc.t().contiguous()  # [HM, V]
    corr_full = torch.nn.functional.linear(x_corr, w_fc)  # [b, V]

    max_err = 0.0
    for i in range(N_SPEC):
        wc_i = torch.index_select(
            w_fc_t, 1, idx[:, i].reshape(-1)
        ).view(HM, b, k)  # [HM, b, k]
        corr_k = torch.einsum(
            "bm,bmk->bk", x_corr, wc_i.permute(1, 0, 2)
        )
        ref = corr_full.gather(1, idx[:, i])
        max_err = max(max_err, (corr_k - ref).abs().max().item())
        # Non-candidate entries must stay -inf after scatter.
        buf = torch.full_like(base[:, i], float("-inf"))
        buf.scatter_(1, idx[:, i], vals[:, i] + corr_k)
        non_cand = buf == float("-inf")
        ok_inf = non_cand.sum().item() == b * (V - k)
        if not ok_inf:
            print("FAIL: non-candidate entries not -inf", flush=True)
            return
    print(
        f"correctness: max_err(candidate correction)={max_err:.6f} "
        f"{'OK' if max_err <= TOLERANCE else 'FAIL'}; "
        "non-candidates=-inf OK",
        flush=True,
    )


def main() -> None:
    print(f"torch_npu version: {getattr(torch_npu, '__version__', 'unknown')}")
    torch.npu.config.allow_internal_format = True
    init_device_properties_triton()
    print(f"V={V} HM={HM} N_SPEC={N_SPEC} BS={BS} KS={KS}")

    device = "npu"
    torch.manual_seed(0)
    _check_correctness(device)

    w_base = torch.randn(V, HM, dtype=torch.bfloat16, device=device)
    w_fc = torch.randn(V, HM, dtype=torch.bfloat16, device=device)
    w_fc_t = w_fc.t().contiguous()  # [HM, V]

    print("-" * 60)
    print("timing (ms per block, eager vs graph):")
    iters, warmup = 10, 3
    for b in BS:
        m = b * N_SPEC
        x_base = torch.randn(m, HM, dtype=torch.bfloat16, device=device)
        x_corr = torch.randn(b, HM, dtype=torch.bfloat16, device=device)
        base_buf = torch.empty(b, N_SPEC, V, dtype=torch.bfloat16, device=device)
        draft_logits = torch.empty(b, N_SPEC, V, dtype=torch.bfloat16, device=device)
        idx_map = torch.arange(b, dtype=torch.int64, device=device)
        temperature = torch.full((b,), 0.6, dtype=torch.float32, device=device)
        seeds = torch.arange(b, dtype=torch.int64, device=device)
        pos = torch.arange(b, dtype=torch.int64, device=device)
        step_cols = torch.arange(N_SPEC, dtype=torch.int32, device=device)

        def base_logits():
            return torch.nn.functional.linear(x_base, w_base)

        def full_corr():
            base = torch.nn.functional.linear(x_base, w_base).view(b, N_SPEC, V)
            for i in range(N_SPEC):
                base[:, i] += torch.nn.functional.linear(x_corr, w_fc)
            return base

        def topk_path(k: int):
            base = torch.nn.functional.linear(x_base, w_base).view(b, N_SPEC, V)
            vals, idx = torch.topk(base, k, dim=-1)
            base_buf.fill_(float("-inf"))
            for i in range(N_SPEC):
                wc_i = torch.index_select(
                    w_fc_t, 1, idx[:, i].reshape(-1)
                ).view(HM, b, k)  # [HM, b, k]
                corr_k = torch.einsum(
                    "bm,bmk->bk", x_corr, wc_i.permute(1, 0, 2)
                )
                base_buf[:, i].scatter_(1, idx[:, i], vals[:, i] + corr_k)
            return base_buf

        def gumbel_steps():
            base = torch.nn.functional.linear(x_base, w_base).view(b, N_SPEC, V)
            for i in range(N_SPEC):
                gumbel_sample(
                    base[:, i],
                    idx_map,
                    temperature,
                    seeds,
                    pos,
                    apply_temperature=True,
                    output_processed_logits=draft_logits,
                    output_processed_logits_col=step_cols[i],
                    use_fp64=False,
                )

        runs = [
            ("base_logits", base_logits),
            ("full_corr x15", full_corr),
            ("gumbel x15", gumbel_steps),
        ]
        for k in KS:
            runs.append((f"topk_path K={k}", lambda k=k: topk_path(k)))

        print(f"B={b} (M={m}):", flush=True)
        timings = {}
        for name, fn in runs:
            try:
                ms_e = _time(fn, iters, warmup, graph=False)
                ms_g = _time(fn, iters, warmup, graph=True)
                timings[name] = (ms_e, ms_g)
                print(
                    f"  {name:16s} eager={ms_e:.3f} ms graph={ms_g:.3f} ms",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  {name:16s} FAIL {type(exc).__name__}: {exc}",
                    flush=True,
                )

        for mode, idx in [("eager", 0), ("graph", 1)]:
            t_base = timings.get("base_logits", (0, 0))[idx]
            t_full = timings.get("full_corr x15", (0, 0))[idx]
            t_gumbel = timings.get("gumbel x15", (0, 0))[idx]
            current = t_base + t_full + t_gumbel
            for k in KS:
                t_topk = timings.get(f"topk_path K={k}", (0, 0))[idx]
                topk_total = t_base + t_topk + t_gumbel
                saving = (current - topk_total) / current * 100
                print(
                    f"  [{mode}] current={current:.3f} ms  "
                    f"topk K={k}={topk_total:.3f} ms  "
                    f"saving={saving:.1f}%",
                    flush=True,
                )


if __name__ == "__main__":
    main()
