#!/usr/bin/env python3
# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""A3 NPU self-check for the domino top-K correction walk (speed + math).

Single-file, prints its own analysis. Run on the A3 server (any env with
torch_npu importable; no vllm import needed):

    python probe_domino_topk_correction.py [--small]

Validates the serving pieces of ``dflash_config.correction_top_k`` that
cannot be tested off-device:

  1. math: the candidate-limited step (gather K fc2 rows + baddbmm + K
     argmax, logit_scale on the correction part only) equals the full-vocab
     argmax whenever the full winner is inside the base top-K candidates
     (bf16, real vocab 248320);
  2. rounding: bf16 flip rate between the dense add path
     (base + scale*silu(u)@W2^T) and the fused-accumulate candidate path on
     random logits (draft-quality-level only; greedy serving is one-hot);
  3. speed: per-step dense correction GEMM vs candidate step, top-K
     extraction over [B,6,V], and 6-step walk totals for B in {1,4,16},
     K in {16,128,1024} — eager and (if available) ACL-graph replay.

Exit code 0 iff every check passes; the tail prints a copyable summary.
"""

import argparse
import sys
import time

import torch
import torch.nn.functional as F

V = 248320
M = 256
STEPS = 6
SCALE = 1.0

CHECKS = []
SUMMARY = {}


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def timed(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    try:
        evs = [(torch.cuda.Event(True), torch.cuda.Event(True))
               for _ in range(iters)]
        for a, b in evs:
            a.record()
            fn()
            b.record()
        torch.cuda.synchronize()
        ts = sorted(a.elapsed_time(b) for a, b in evs)
        return ts[0], ts[len(ts) // 2]
    except Exception:  # noqa: BLE001 — older torch_npu: wall clock fallback
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        return ts[0], ts[len(ts) // 2]


def dense_step(u, base, w2):
    bias = F.linear(F.silu(u), w2) * SCALE
    return (base + bias).argmax(dim=-1)


def topk_gather(w2, ids_all):
    """One gather of every step's candidate rows: [B,S,K] ids -> [B,S,K,M]."""
    b, s, k = ids_all.shape
    return w2.index_select(0, ids_all.reshape(-1)).view(b, s, k, -1)


def topk_step(u, vals, ids, cand_w):
    h = F.silu(u)
    scores = vals.unsqueeze(-1).to(h.dtype).clone()
    scores.baddbmm_(cand_w, h.unsqueeze(-1), beta=1.0, alpha=SCALE)
    best = scores.squeeze(-1).argmax(dim=-1, keepdim=True)
    return ids.gather(1, best).squeeze(1)



def _setup_device():
    """npu via torch_npu (+ the cuda->npu compat shim so torch.cuda.Event /
    synchronize / CUDAGraph in this file work), or cuda for logic runs."""
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        torch_npu = None
    if torch_npu is not None and torch.npu.is_available():
        from torch_npu.contrib import transfer_to_npu  # noqa: F401

        torch.npu.set_device(0)
        print("device:", torch.npu.get_device_name(0))
        return torch.device("npu"), torch.bfloat16
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0),
              "(logic validation only; NPU numbers need torch_npu)")
        return torch.device("cuda"), torch.float16
    raise RuntimeError("no npu (torch_npu) or cuda device available")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true",
                        help="V=4096 fixture for a quick CI-style run")
    args = parser.parse_args()
    v = 4096 if args.small else V

    dev, dt = _setup_device()
    print(f"torch {torch.__version__}, V={v}")

    torch.manual_seed(7)
    w2 = (torch.randn(v, M) * 0.02).to(dev, dt)

    # ---------------------------------------------------------------- math
    for B, K in ((1, 16), (4, 128), (16, 1024)):
        u = (torch.randn(B, M) * 0.5).to(dev, dt)
        base = (torch.randn(B, v) * 2.0).to(dev, dt)
        full = (base.float() + SCALE * (F.silu(u.float()) @ w2.float().t()))
        winner = full.argmax(dim=-1).to(torch.int64)
        vals, ids = base.topk(K, dim=-1)
        cand_w = topk_gather(w2, ids.unsqueeze(1))[:, 0]
        got = topk_step(u, vals, ids, cand_w).to(torch.int64)
        in_cand = (ids == winner.unsqueeze(1)).any(dim=1)
        ok = torch.equal(got[in_cand], winner[in_cand])
        cov = in_cand.float().mean().item()
        check(f"candidate argmax exact when winner in top-{K} (B={B})", ok,
              f"in-candidate rows {int(in_cand.sum())}/{B}")

    # bf16 add-order flips between dense and fused-accumulate paths
    B, K = 4, 128
    u = (torch.randn(B, M) * 0.5).to(dev, dt)
    base = (torch.randn(B, v) * 2.0).to(dev, dt)
    d = dense_step(u, base, w2)
    vals, ids = base.topk(K, dim=-1)
    cand_w = topk_gather(w2, ids.unsqueeze(1))[:, 0]
    t = topk_step(u, vals, ids, cand_w)
    both = (ids == d.unsqueeze(1)).any(dim=1) & (ids == t.unsqueeze(1)).any(dim=1)
    flips = (d[both] != t[both]).float().mean().item() if both.any() else 0.0
    SUMMARY["bf16_flip_rate_in_candidates"] = round(flips, 6)
    check("bf16 dense-vs-candidate flip rate sane", flips < 0.05,
          f"flips {flips:.4f} on {int(both.sum())} comparable rows "
          "(rounding-level, draft-quality only)")

    # ---------------------------------------------------------------- speed
    rows = {}
    for B in (1, 4, 16):
        u = (torch.randn(B, M) * 0.5).to(dev, dt)
        base_full = (torch.randn(B, 7, v) * 2.0).to(dev, dt)
        corr_rows = torch.arange(1, 7, device=dev)
        mn, p50 = timed(lambda: dense_step(u, base_full[:, 1], w2))
        rows[("dense_step", B, None)] = (mn, p50)
        for K in (16, 128, 1024):
            mn_e, p50_e = timed(
                lambda: base_full.index_select(1, corr_rows).topk(K, dim=-1)
            )
            vals, ids = base_full.index_select(1, corr_rows).topk(K, dim=-1)
            mn_g, p50_g = timed(lambda: topk_gather(w2, ids))
            cand_w = topk_gather(w2, ids)
            mn_t, p50_t = timed(
                lambda: topk_step(u, vals[:, 0], ids[:, 0], cand_w[:, 0])
            )
            rows[("extract", B, K)] = (mn_e, p50_e)
            rows[("gather", B, K)] = (mn_g, p50_g)
            rows[("topk_step", B, K)] = (mn_t, p50_t)
        SUMMARY[f"B{B}"] = {
            "dense_step_ms_min": round(rows[("dense_step", B, None)][0], 4),
            "topk_ops_ms_min": {
                K: {
                    "extract": round(rows[("extract", B, K)][0], 4),
                    "gather": round(rows[("gather", B, K)][0], 4),
                    "step": round(rows[("topk_step", B, K)][0], 4),
                }
                for K in (16, 128, 1024)
            },
            "walk_full_ms_min": round(6 * rows[("dense_step", B, None)][0], 3),
            "walk_topk_ms_min": {
                K: round(
                    rows[("extract", B, K)][0]
                    + rows[("gather", B, K)][0]
                    + 6 * rows[("topk_step", B, K)][0],
                    3,
                )
                for K in (16, 128, 1024)
            },
        }

    print("\n=== per-op timings (min / p50 ms, bf16, eager) ===")
    for (name, B, K), (mn, p50) in sorted(rows.items()):
        tag = f"{name} B={B}" + (f" K={K}" if K else "")
        print(f"  {tag:28s} {mn:8.4f} / {p50:8.4f}")

    print("\n=== 6-step walk totals (min ms, eager; GRU cell identical in "
          "both arms, excluded) ===")
    for B in (1, 4, 16):
        s = SUMMARY[f"B{B}"]
        print(f"  B={B:2d} full {s['walk_full_ms_min']:8.3f}   "
              f"topk " + "  ".join(
                  f"K{k}={v:7.3f}" for k, v in s["walk_topk_ms_min"].items()))

    # --------------------------------------------------- optional ACL graph
    try:
        B = 4
        u = (torch.randn(B, M) * 0.5).to(dev, dt)
        base_full = (torch.randn(B, 7, v) * 2.0).to(dev, dt)
        vals, ids = base_full[:, 1:].topk(128, dim=-1)
        us = [(torch.randn(B, M) * 0.5).to(dev, dt) for _ in range(6)]

        def dense_chain():
            for j in range(6):
                dense_step(us[j], base_full[:, 1 + j], w2)

        def topk_chain():
            cw = topk_gather(w2, ids)
            for j in range(6):
                topk_step(us[j], vals[:, j], ids[:, j], cw[:, j])

        results = {}
        for name, fn in (("dense6", dense_chain), ("topk6_K128", topk_chain)):
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(s)
            with torch.cuda.graph(g):
                fn()
            results[name] = timed(lambda gr=g: gr.replay())
        SUMMARY["acl_graph_B4"] = {
            "dense_6step_ms_min": round(results["dense6"][0], 4),
            "topk_6step_K128_ms_min": round(results["topk6_K128"][0], 4),
        }
        print(f"\nACL-graph replay (B=4, 6-step chains): dense "
              f"{results['dense6'][0]:.4f} / topk128 "
              f"{results['topk6_K128'][0]:.4f} ms (min)")
    except Exception as e:  # noqa: BLE001
        print(f"\n[skip] ACL-graph section unavailable: {type(e).__name__}: {e}")

    print("\n=== copyable summary ===")
    import json
    print(json.dumps(SUMMARY, indent=1))

    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
