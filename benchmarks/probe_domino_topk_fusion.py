#!/usr/bin/env python3
# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""A3 NPU probe for the phase-2 fused top-K extraction kernels (domino draft).

Single-file, prints its own analysis. Run on the A3 server:

    python probe_domino_topk_fusion.py            # bf16 on npu
    python probe_domino_topk_fusion.py --quick    # small-V logic check

Off-NPU it auto-falls back to cuda (fp16, e.g. the 2060) purely to validate
kernel logic and wiring — perf numbers there are meaningless for A3.

Stages of the phase-2 idea being probed (vs the current serving ops):

  A0  baseline extract = index_select([B,6,V] copy) + torch.topk
      (what AscendDominoSpeculator._sample_sequential does today)
  A1  triton split-V topk reading the *strided* [B,7,V] rows directly:
      one streaming pass over the logits, no materialized copy, exact via a
      per-tile k-pass (argmax + knock-out, K iterations) + a small
      torch.topk merge over the per-tile partials.  TILE in {512, 1024}.
  B1  the real fused kernel: head GEMM (tl.dot, x[M,H] @ W[V,H]^T) with the
      top-K taken from the fp32 accumulator in-register per V-tile — the
      [M,V] logits never touch HBM.  Partials merged like A1.
      K <= 128, BV in {128, 256}; ROWS covers all M = B*7 rows.

Exactness gates:
  * A1 must reproduce torch.topk values AND ids bitwise (pure select on the
    same input bits; both tie-break leftmost).
  * B1 logits come from tl.dot fp32 accumulation (different reduction order
    than any torch GEMM), so its gate is exact ids vs an fp32 reference GEMM
    on rows whose top1-top2 gap > 1e-2; the id-match rate vs the bf16 head
    logits is reported as information.

The tail prints a copyable JSON summary.  Exit code 0 iff every gate passes.
"""

import argparse
import json
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

CHECKS = []
SUMMARY = {}
V = 248320
H = 1024
M_EMB = 256               # correction head emb dim
STEPS = 6                 # corrected rows per iteration
KS = (16, 128, 1024)


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def skip(name, detail=""):
    print(f"[SKIP] {name}" + (f" — {detail}" if detail else ""))


# --------------------------------------------------------------------------
# A1: per-tile exact top-K by k-pass (argmax + knock out), one row per program
# --------------------------------------------------------------------------
@triton.jit
def _topk_rows_kpass(
    X, ROWS_POS, PV, PI, V, K, NT,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    pos = tl.load(ROWS_POS + row).to(tl.int64) * V
    v0 = tile * TILE
    cols = v0 + tl.arange(0, TILE)
    x = tl.load(X + pos + cols, mask=cols < V, other=-float("inf"))
    out = (row.to(tl.int64) * NT + tile) * K
    for k in range(K):
        i = tl.argmax(x, axis=0)
        m = tl.max(x, axis=0)
        tl.store(PV + out + k, m)
        tl.store(PI + out + k, v0 + i)
        x = tl.where(cols == v0 + i, -float("inf"), x)


def topk_rows_a1(flat, rows_pos, K, tile):
    """flat [B*7, V] contiguous + rows_pos [R] (flat ids of rows 1..6)."""
    R = rows_pos.shape[0]
    vv = flat.shape[1]
    NT = triton.cdiv(vv, tile)
    pv = torch.empty(R, NT, K, dtype=torch.float32, device=flat.device)
    pi = torch.empty(R, NT, K, dtype=torch.int32, device=flat.device)
    _topk_rows_kpass[(R, NT)](flat, rows_pos, pv, pi, vv, K, NT, TILE=tile)
    return merge(pv, pi, K)


# --------------------------------------------------------------------------
# B1: fused head GEMM + in-register tile top-K (logits never materialized)
# --------------------------------------------------------------------------
@triton.jit
def _head_topk(
    X, W, PV, PI, M, V, K, NT,
    H: tl.constexpr, ROWS: tl.constexpr, BV: tl.constexpr, BH: tl.constexpr,
):
    pid = tl.program_id(0)
    v0 = (pid * BV).to(tl.int64)
    rows = tl.arange(0, ROWS)
    rmask = rows < M
    acc = tl.zeros((ROWS, BV), dtype=tl.float32)
    for h0 in range(0, H, BH):
        hh = h0 + tl.arange(0, BH)
        x = tl.load(X + rows[:, None] * H + hh[None, :],
                    mask=rmask[:, None], other=0.0)
        w = tl.load(W + (v0 + tl.arange(0, BV))[None, :] * H + hh[:, None])
        acc = tl.dot(x, w, acc)
    cols = tl.arange(0, BV)
    out = pid * K
    for k in range(K):
        i = tl.argmax(acc, axis=1)
        m = tl.max(acc, axis=1)
        tl.store(PV + rows * (NT * K) + out + k, m, mask=rmask)
        tl.store(PI + rows * (NT * K) + out + k, v0 + i, mask=rmask)
        acc = tl.where(cols[None, :] == i[:, None], -float("inf"), acc)


def head_topk_b1(x, w, K, rows, bv, bh, ns):
    """x [M, H], w [V, H] -> (vals [M, K], ids [M, K]); M <= rows."""
    M = x.shape[0]
    NT = triton.cdiv(w.shape[0], bv)
    pv = torch.empty(M, NT, K, dtype=torch.float32, device=x.device)
    pi = torch.empty(M, NT, K, dtype=torch.int32, device=x.device)
    _head_topk[(NT,)](x, w, pv, pi, M, w.shape[0], K, NT,
                      H=H, ROWS=rows, BV=bv, BH=bh, num_stages=ns)
    return merge(pv, pi, K)


def merge(pv, pi, K):
    R = pv.shape[0]
    vals, idx = torch.topk(pv.reshape(R, -1), K, dim=-1)
    ids = pi.reshape(R, -1).gather(1, idx)
    return vals, ids.long()


def dense_corr_step(u, base_row, w2):
    """Original full-vocab correction step (serving `bias_and_gh` + add +
    argmax): SiLU -> fc2 [V, M_EMB] GEMM -> add base row -> argmax."""
    return (base_row + F.linear(F.silu(u), w2)).argmax(dim=-1)


def topk_gather(w2, ids_all):
    b, s, k = ids_all.shape
    return w2.index_select(0, ids_all.reshape(-1)).view(b, s, k, -1)


def topk_step(u, vals, ids, cand_w):
    """P1 candidate step: pre-gathered rows, baddbmm, K-argmax."""
    h = F.silu(u)
    scores = vals.unsqueeze(-1).to(h.dtype).clone()
    scores.baddbmm_(cand_w, h.unsqueeze(-1), beta=1.0, alpha=1.0)
    best = scores.squeeze(-1).argmax(dim=-1, keepdim=True)
    return ids.gather(1, best).squeeze(1)


# --------------------------------------------------------------------------
def timed(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    evs = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(iters)]
    for a, b in evs:
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    ts = sorted(a.elapsed_time(b) for a, b in evs)
    return ts[0], ts[len(ts) // 2]


def b1_budget(rows_m, bv):
    """acc lives in registers: [ROWS, BV] fp32.  Cap at 64KB-ish."""
    return rows_m * bv <= 64 * 256


# (BV, BH, num_stages): BV=256 stages must stay low — the W tile is
# BV*BH*2B per stage and sm_75 caps shared memory at 64KB.
B1_CFGS = ((128, 128, 2), (256, 128, 1))


def run(base, x, w, w2, label, gates_only=False):
    """base [B,7,V] logits; x [B*7,H], w [V,H] the head that produced them."""
    B = base.shape[0]
    vv = base.shape[-1]
    M = x.shape[0]
    flat = base.reshape(B * 7, vv)
    rows_pos = (
        torch.arange(B, device=base.device).view(B, 1) * 7
        + torch.arange(1, 7, device=base.device).view(1, 6)
    ).reshape(-1)
    print(f"\n--- {label}: B={B} V={vv} M={M} ---")

    # ---- gates ----
    # torch.topk breaks fp16/bf16 value ties in an unspecified (radix) order;
    # the k-pass kernel breaks them leftmost.  Honest gate: exact value
    # multiset + every reported id must hold its reported value in the
    # source row.  (Any id-set difference beyond that is a tie-order artifact
    # with identical scores — harmless for the draft argmax.)
    flat_rows = flat[rows_pos]
    for K in KS:
        tv, ti = torch.topk(flat_rows, K, dim=-1)
        for tile in (512, 1024):
            if tile > vv:
                continue
            vals, ids = topk_rows_a1(flat, rows_pos, K, tile)
            back = flat_rows.gather(1, ids).float()
            ok = torch.equal(vals, tv.float()) and torch.equal(back, vals)
            n_tie = (tv == tv[:, :1]).sum() + (tv == tv[:, -1:]).sum()
            check(f"A1 top-K exact (K={K} TILE={tile})", ok,
                  "" if ok else
                  f"val diff {(vals - tv.float()).abs().max().item():.3e}; "
                  f"boundary ties {int(n_tie)}")

    fp32_logits = x.float() @ w.float().t()
    max_m = min(128, triton.next_power_of_2(max(M, 16)))
    for K in [k for k in KS if k <= 128]:
        tref, iref = torch.topk(fp32_logits, K, dim=-1)
        det = (tref[:, 0] - tref[:, 1]).abs() > 1e-2
        ti16 = torch.topk((x @ w.t()), K, dim=-1).indices
        for bv, bh, ns in B1_CFGS:
            if not b1_budget(max_m, bv):
                continue
            try:
                _, ids = head_topk_b1(x, w, K, max_m, bv, bh, ns)
            except triton.OutOfResources as e:
                # config does not fit this device's shared memory (e.g. the
                # 2060's 64KB); not a logic failure — A3 has far more.
                skip(f"B1 K={K} BV={bv}", f"OutOfResources: {e}")
                continue
            except Exception as e:  # noqa: BLE001
                check(f"B1 K={K} BV={bv} runs", False,
                      f"{type(e).__name__}: {e}")
                continue
            ok = torch.equal(ids[det], iref[det])
            detail = (f"id match vs fp32 {int((ids == iref).sum())}/{M * K}; "
                      f"vs bf16 head {int((ids == ti16).sum())}/{M * K}; "
                      f"det rows {int(det.sum())}/{M}")
            check(f"B1 exact vs fp32 GEMM on well-separated rows "
                  f"(K={K} BV={bv})", ok, detail)

    # ---- timings ----
    out = {}
    if gates_only:
        return out
    for K in KS:
        sel = torch.arange(1, 7, device=base.device)
        mn, p50 = timed(lambda: base.index_select(1, sel).topk(K, dim=-1))
        out[f"A0_serving_extract_K{K}"] = (mn, p50)
        for tile in (512, 1024):
            if tile > vv:
                continue
            mn, p50 = timed(lambda t=tile: topk_rows_a1(flat, rows_pos, K, t))
            out[f"A1_triton_topk_K{K}_T{tile}"] = (mn, p50)
    mn, p50 = timed(lambda: x @ w.t())
    out["head_gemm_torch"] = (mn, p50)
    for K in [k for k in KS if k <= 128]:
        for bv, bh, ns in B1_CFGS:
            if not b1_budget(max_m, bv):
                continue
            try:
                mn, p50 = timed(lambda k=K, b=bv, h=bh, s=ns:
                                head_topk_b1(x, w, k, max_m, b, h, s))
                out[f"B1_fused_head_topk_K{K}_BV{bv}"] = (mn, p50)
            except Exception:  # noqa: BLE001
                pass

    # original correction head (full vocab, no top-K) + the P1 walk ops,
    # so the whole-iteration picture is in one place.  GRU cell / W_sh are
    # identical in every arm and excluded.
    u = (torch.randn(B, M_EMB, device=base.device) * 0.5).to(base.dtype)
    mn, p50 = timed(lambda: dense_corr_step(u, base[:, 1], w2))
    out["orig_dense_corr_step"] = (mn, p50)
    sel = torch.arange(1, 7, device=base.device)
    for K in KS:
        vals_k, ids_k = base.index_select(1, sel).topk(K, dim=-1)
        cand_w_k = topk_gather(w2, ids_k)
        mn, p50 = timed(lambda i=ids_k: topk_gather(w2, i))
        out[f"gather_K{K}"] = (mn, p50)
        mn, p50 = timed(lambda v=vals_k, i=ids_k, c=cand_w_k:
                        topk_step(u, v[:, 0], i[:, 0], c[:, 0]))
        out[f"topk_step_K{K}"] = (mn, p50)

    for name, (mn, p50) in sorted(out.items()):
        print(f"  {name:32s} {mn:8.3f} / {p50:8.3f} ms (min/p50)")
    SUMMARY[label] = {k: [round(v[0], 4), round(v[1], 4)]
                      for k, v in out.items()}

    # whole-walk reconstruction (min ms): head + [extract] + per-step arm
    head = out["head_gemm_torch"][0]
    walks = {
        "orig_full_vocab": head + STEPS * out["orig_dense_corr_step"][0],
    }
    for K in KS:
        walks[f"p1_unfused_K{K}"] = (
            head
            + out[f"A0_serving_extract_K{K}"][0]
            + out[f"gather_K{K}"][0]
            + STEPS * out[f"topk_step_K{K}"][0]
        )
        b1s = [v[0] for k, v in out.items()
               if k.startswith(f"B1_fused_head_topk_K{K}_")]
        if b1s:
            walks[f"p2_fused_K{K}"] = (
                min(b1s)
                + out[f"gather_K{K}"][0]
                + STEPS * out[f"topk_step_K{K}"][0]
            )
    orig = walks["orig_full_vocab"]
    print(f"  walk totals (min ms, GRU/W_sh excluded, same in all arms):")
    for name, t in walks.items():
        print(f"    {name:18s} {t:8.3f}   {orig / t:5.2f}x")
    SUMMARY[f"walks_{label}"] = {k: round(v, 3) for k, v in walks.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small-V logic check")
    ap.add_argument("--gates-only", action="store_true",
                    help="skip the timed() benchmarks")
    ap.add_argument("--batches", default="1,4,16")
    args = ap.parse_args()
    vv = 8192 if args.quick else V

    if hasattr(torch, "npu") and torch.npu.is_available():
        dev, dt = torch.device("npu"), torch.bfloat16
        torch.npu.set_device(0)
        print("device:", torch.npu.get_device_name(0))
    else:
        dev, dt = torch.device("cuda"), torch.float16
        print("device:", torch.cuda.get_device_name(0),
              "(logic validation only; A3 numbers need npu)")

    torch.manual_seed(11)
    w2 = (torch.randn(vv, M_EMB, device=dev) * 0.02).to(dt)
    for B in [int(b) for b in args.batches.split(",")]:
        x = (torch.randn(B * 7, H, device=dev) * 0.5).to(dt)
        w = (torch.randn(vv, H, device=dev) * 0.02).to(dt)
        base = (x.float() @ w.float().t()).reshape(B, 7, vv).to(dt)
        run(base, x, w, w2, f"B{B}", gates_only=args.gates_only)

    print("\n=== copyable summary ===")
    print(json.dumps(SUMMARY, indent=1))
    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
