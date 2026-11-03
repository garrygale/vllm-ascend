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
     K in {16,128,1024} — eager and (if available) ACL-graph replay,
     including extract-inside-the-chain graphs (the serving form);
  4. automatic analysis: implied streaming rates, dispatch overhead, the
     serving-form walk accounting, extraction K-sensitivity, and a B=16
     verdict line.

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

def analyze(rows, graphs, v):
    """Print the serving-form reading of the timings collected above."""
    ana = {}
    w2_mb = v * M * 2 / 1e6
    print("\n=== analysis ===")
    print(f"W2 = {w2_mb:.1f} MB; the full walk streams 6x = {6 * w2_mb:.1f} MB "
          f"of weights per iteration regardless of B")

    print("\n[dense baseline] implied streaming rate "
          "(bytes = W2 + ~5*B*V*2 activation traffic)")
    for B in (1, 4, 16):
        d = rows.get(("dense_step", B, None))
        if d is None:
            continue
        mb = w2_mb + 5 * B * v * 2 / 1e6
        bw = mb / d[0]  # MB/ms == GB/s
        ana[f"dense_bw_B{B}_gbps"] = round(bw)
        print(f"  B={B:2d}  {d[0] * 1e3:7.1f} us/step -> {bw:6.0f} GB/s")
    d1 = rows.get(("dense_step", 1, None))
    d16 = rows.get(("dense_step", 16, None))
    if d1 and d16:
        flat = d16[0] / d1[0]
        ana["dense_B16_over_B1"] = round(flat, 2)
        print(f"  B=16/B=1 step ratio {flat:.2f} -> "
              + ("weight-stream bound" if flat < 1.3
                 else "activation traffic visible at high B"))

    if "dense6" in graphs and "topk6_K128" in graphs:
        d6 = graphs["dense6"][0]
        t6 = graphs["topk6_K128"][0]
        ana["graph_chain_ratio_B4"] = round(d6 / t6, 2)
        print(f"\n[graph replay B=4] dense6 {d6:.4f} vs topk6 {t6:.4f} ms "
              f"(incl gather, no extract) -> {d6 / t6:.2f}x; per-step "
              f"{d6 / 6 * 1e3:.1f} vs ~{t6 / 6 * 1e3:.1f} us")
        eg = rows.get(("topk_step", 4, 128))
        gg = rows.get(("gather", 4, 128))
        if eg and gg:
            eager_chain = 6 * eg[0] + gg[0]
            pol = eager_chain - t6
            ana["eager_dispatch_overhead_6steps_ms"] = round(max(pol, 0), 4)
            print(f"  eager same chain {eager_chain:.4f} ms -> "
                  f"~{max(pol, 0) * 1e3:.0f} us of that is per-op dispatch")
    if "dense6" in graphs and "topk6_ext_K128" in graphs:
        d6 = graphs["dense6"][0]
        t6 = graphs["topk6_ext_K128"][0]
        ana["graph_walk_ratio_B4_K128"] = round(d6 / t6, 2)
        print(f"  serving form (extract INSIDE the chain): dense6 {d6:.4f} vs "
              f"topk_ext_K128 {t6:.4f} ms -> {d6 / t6:.2f}x")

    print("\n[walk accounting, eager min ms] "
          "topk = extract + gather + 6 steps")
    for B in (1, 4, 16):
        d = rows.get(("dense_step", B, None))
        if d is None:
            continue
        full = 6 * d[0]
        cells = []
        for K in (16, 128, 1024):
            e = rows.get(("extract", B, K))
            g = rows.get(("gather", B, K))
            s = rows.get(("topk_step", B, K))
            if not (e and g and s):
                continue
            topk = e[0] + g[0] + 6 * s[0]
            cells.append(f"K{K} {topk:6.3f} ({full / topk:4.2f}x)")
            ana[f"walk_ratio_B{B}_K{K}"] = round(full / topk, 2)
        print(f"  B={B:2d} full {full:6.3f}   " + "   ".join(cells))

    print("\n[extraction] K sensitivity (a radix select should be near-flat)")
    for B in (1, 4, 16):
        a = rows.get(("extract", B, 16))
        c = rows.get(("extract", B, 1024))
        if a and c:
            print(f"  B={B:2d} K16 {a[0]:.3f} -> K1024 {c[0]:.3f} ms "
                  f"({c[0] / a[0]:.2f}x)")
    print("[extraction] decomposition @K=128 "
          "(row copy vs topk itself, sorted=False, strided view)")
    for B in (1, 4, 16):
        io = rows.get(("idxsel", B, None))
        tk = rows.get(("topk_only", B, 128))
        ex = rows.get(("extract", B, 128))
        un = rows.get(("extract_unsorted", B, 128))
        vw = rows.get(("extract_view", B, 128))
        if not (io and tk and ex):
            continue
        share = tk[0] / ex[0] if ex[0] else 0.0
        line = (f"  B={B:2d} idxsel {io[0]:.3f} + topk {tk[0]:.3f} "
                f"= extract {ex[0]:.3f} ms (topk is {share:.0%} of it)")
        if un:
            line += f"; sorted=False {un[0]:.3f} ({ex[0] / un[0]:.2f}x)"
            ana[f"extract_unsorted_vs_sorted_B{B}"] = round(ex[0] / un[0], 2)
        if vw:
            line += f"; strided view {vw[0]:.3f}"
            ana[f"extract_view_vs_copy_B{B}"] = round(ex[0] / vw[0], 2)
        print(line)
        ana[f"extract_topk_share_B{B}"] = round(share, 2)
        if share > 0.7:
            print("       -> aclnnTopK-bound (not the copy): a faster "
                  "selection primitive is the fix, not fusion")

    need = (rows.get(("dense_step", 16, None)), rows.get(("extract", 16, 128)),
            rows.get(("gather", 16, 128)), rows.get(("topk_step", 16, 128)))
    if all(need):
        d16e, e, g, s = need
        r = 6 * d16e[0] / (e[0] + g[0] + 6 * s[0])
        if r >= 1.5:
            msg = (f"B=16 walk ratio {r:.2f}x: correction_top_k=128 pays at "
                   f"every batch size; phase-2 fusion optional")
        elif r >= 1.3:
            msg = (f"B=16 walk ratio {r:.2f}x: pays, but extraction is eating "
                   f"the win at high B - run probe_domino_topk_fusion.py next")
        else:
            msg = (f"B=16 walk ratio {r:.2f}x: extraction-bound at high B - "
                   f"phase-2 fusion needed there; K=128 still fine for B<=8")
        ana["verdict"] = msg
        print(f"\n[verdict] {msg}")
    SUMMARY["analysis"] = ana


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
            # extract decomposition + cheap variants: is the cost the row
            # copy, the radix topk itself, or the sorted order?
            sl = base_full.index_select(1, corr_rows)
            rows[("idxsel", B, None)] = timed(
                lambda: base_full.index_select(1, corr_rows))
            rows[("topk_only", B, K)] = timed(lambda: sl.topk(K, dim=-1))
            rows[("extract_unsorted", B, K)] = timed(
                lambda: base_full.index_select(1, corr_rows).topk(
                    K, dim=-1, sorted=False))
            rows[("extract_view", B, K)] = timed(
                lambda: base_full[:, 1:].topk(K, dim=-1))
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
    graphs = {}
    try:
        B = 4
        us = [(torch.randn(B, M) * 0.5).to(dev, dt) for _ in range(6)]
        base_full = (torch.randn(B, 7, v) * 2.0).to(dev, dt)
        corr_rows = torch.arange(1, 7, device=dev)
        vals128, ids128 = base_full.index_select(1, corr_rows).topk(128, dim=-1)

        def dense_chain():
            for j in range(6):
                dense_step(us[j], base_full[:, 1 + j], w2)

        def topk_chain():
            cw = topk_gather(w2, ids128)
            for j in range(6):
                topk_step(us[j], vals128[:, j], ids128[:, j], cw[:, j])

        def topk_ext_chain(K):
            def chain():
                sl = base_full.index_select(1, corr_rows)
                vals, ids = sl.topk(K, dim=-1)
                cw = topk_gather(w2, ids)
                for j in range(6):
                    topk_step(us[j], vals[:, j], ids[:, j], cw[:, j])
            return chain

        chains = {
            "dense6": dense_chain,
            "topk6_K128": topk_chain,
            "topk6_ext_K16": topk_ext_chain(16),
            "topk6_ext_K128": topk_ext_chain(128),
        }
        for name, fn in chains.items():
            try:
                g = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        fn()
                torch.cuda.current_stream().wait_stream(s)
                with torch.cuda.graph(g):
                    fn()
                graphs[name] = timed(lambda gr=g: gr.replay())
            except Exception as e:  # noqa: BLE001
                print(f"[skip] graph {name}: {type(e).__name__}: {e}")
        if graphs:
            SUMMARY["acl_graph_B4"] = {
                k: round(v_[0], 4) for k, v_ in graphs.items()
            }
            print(f"\nACL-graph replay (B=4, 6-step chains, min ms): "
                  + "  ".join(f"{k}={v_[0]:.4f}" for k, v_ in graphs.items()))
    except Exception as e:  # noqa: BLE001
        print(f"\n[skip] ACL-graph section unavailable: {type(e).__name__}: {e}")

    analyze(rows, graphs, v)

    print("\n=== copyable summary ===")
    import json
    print(json.dumps(SUMMARY, indent=1))

    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
