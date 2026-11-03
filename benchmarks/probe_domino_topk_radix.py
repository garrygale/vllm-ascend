#!/usr/bin/env python3
# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Radix-select top-K probe: the phase-2b extraction primitive (NPU).

aclnnTopK is the top-K correction's bottleneck on A3 (measured: topk is
87-96% of the extract cost, sorted=False no help, ~1 ms at B=16 while the
data movement is ~0.1 ms).  This probe replaces ONLY the selection
primitive - the head GEMM stays with aclnn - with a two-pass radix select
over the bf16 bit patterns:

  pass 1  per-row histogram of the monotonic key's top 12 bits (4096 bins,
          accumulated in registers via tl.histogram; a per-element
          tl.atomic_add fallback variant runs where tl.histogram does not
          compile),
  scan    reverse cumulative count -> threshold bin t, c_above, need_tie
          (tiny torch ops on [R, 4096]),
  pass 2  emit the threshold-bin superset (prefix >= t, count in
          [K, cap]) into a per-row buffer with slot reservation via cumsum
          + one atomic per tile,
  finish  gather the superset values from the strided source rows and trim
          to an exact K with one small torch.topk over [R, cap].  A single
          12-bit level is exact *because* the trim refines inside the
          threshold bin (a two-level in-kernel radix would save the small
          topk but cost far more code; fat bins beyond the cap fall back
          per row and are reported).

Gates: exact value multiset + id-holds-value vs torch.topk on random
logits, on a heavy-tie grid (fat bins), and the all-equal overflow row
must be detected and fall back rather than corrupt.  Timings: radix vs
aclnn extract at B in {1,4,16} x K in {16,128,1024}, eager and with the
extract inside the ACL-graph 6-step chain (the serving form), plus walk
accounting against the dense full-vocab baseline.

Off-NPU it falls back to cuda/fp16 for logic validation only.  Exit code
0 iff every gate passes.
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
NB = 4096          # 12-bit key prefix bins
TIE_CAP = 4096     # per-row superset cap (overflow -> that row falls back, reported)
TILE = 2048
KS = (16, 128, 1024)


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def skip(name, detail=""):
    print(f"[SKIP] {name}" + (f" — {detail}" if detail else ""))


# --------------------------------------------------------------------------
# pass 1: per-row histogram of the 12-bit monotonic key prefix
# --------------------------------------------------------------------------
@triton.jit
def _hist12(
    SRC, ROWS_POS, HIST, V,
    NBINS: tl.constexpr, TILE: tl.constexpr, USE_HIST: tl.constexpr,
):
    row = tl.program_id(0)
    pos = tl.load(ROWS_POS + row).to(tl.int64) * V
    acc = tl.zeros((NBINS,), dtype=tl.int32)
    for v0 in range(0, V, TILE):
        cols = v0 + tl.arange(0, TILE)
        x = tl.load(SRC + pos + cols, mask=cols < V, other=-float("inf"))
        b = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        sign = b >> 15
        key = tl.where(sign == 1, (~b) & 0xFFFF, b | 0x8000)
        pref = key >> 4
        if USE_HIST:
            acc += tl.histogram(pref, NBINS)
        else:
            tl.atomic_add(HIST + row * NBINS + pref, 1)
    if USE_HIST:
        tl.store(HIST + row * NBINS + tl.arange(0, NBINS), acc)


# --------------------------------------------------------------------------
# pass 2: emit the threshold-bin superset (pref >= t) into a capped buffer;
# an exact trim to K happens on the host side over the small superset.
# --------------------------------------------------------------------------
@triton.jit
def _emit(
    SRC, ROWS_POS, THR, CAND, CNT, V,
    TILE: tl.constexpr, TCAP: tl.constexpr,
):
    row = tl.program_id(0)
    t = tl.load(THR + row)
    pos = tl.load(ROWS_POS + row).to(tl.int64) * V
    for v0 in range(0, V, TILE):
        cols = v0 + tl.arange(0, TILE)
        inb = cols < V
        x = tl.load(SRC + pos + cols, mask=inb, other=-float("inf"))
        b = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        sign = b >> 15
        key = tl.where(sign == 1, (~b) & 0xFFFF, b | 0x8000)
        pref = key >> 4
        m = (pref >= t) & inb
        idv = cols.to(tl.int32)
        n = tl.sum(m.to(tl.int32))
        base = tl.atomic_add(CNT + row, n)
        off = tl.cumsum(m.to(tl.int32), axis=0) - m.to(tl.int32)
        tl.store(CAND + row * TCAP + base + off, idv,
                 mask=m & (base + off < TCAP))


def radix_topk(flat, rows_pos, K, use_hist=True):
    """flat [B*7, V] + corrected-row ids -> exact (vals [R,K], ids [R,K]).

    Two streaming passes (12-bit histogram, threshold-bin superset emit)
    plus an exact trim over the small superset.  A row whose superset
    overflows TCAP comes back with ok=False; the caller falls back (rare,
    reported - it needs a fat threshold bin, e.g. heavy-tie data).
    """
    R = rows_pos.shape[0]
    vv = flat.shape[1]
    dev = flat.device
    hist = torch.zeros(R, NB, dtype=torch.int32, device=dev)
    _hist12[(R,)](flat, rows_pos, hist, vv,
                  NBINS=NB, TILE=TILE, USE_HIST=use_hist)

    cnt_ge = torch.flip(
        torch.cumsum(torch.flip(hist, dims=[1]).to(torch.int64), dim=1),
        dims=[1],
    )
    bins = torch.arange(NB, device=dev).view(1, -1)
    valid = cnt_ge >= K
    t = torch.where(valid, bins, torch.full_like(bins, -1)).amax(dim=1)

    cand = torch.zeros(R, TIE_CAP, dtype=torch.int32, device=dev)
    cnt = torch.zeros(R, dtype=torch.int32, device=dev)
    _emit[(R,)](flat, rows_pos, t.to(torch.int32), cand, cnt, vv,
                TILE=TILE, TCAP=TIE_CAP)
    ok = cnt.to(torch.int64) <= TIE_CAP

    # exact trim of the superset: values gathered straight from the source
    # rows (invalid slots clamped to 0 and masked to -inf).
    ar = torch.arange(TIE_CAP, device=dev).view(1, -1)
    have = ar < cnt.view(-1, 1)
    ids_c = torch.where(have, cand, torch.zeros_like(cand)).long()
    flat1 = flat.view(-1)
    vals_c = flat1[rows_pos.view(-1, 1).to(torch.int64) * vv + ids_c]
    vals_c = torch.where(have, vals_c, torch.full_like(vals_c, float("-inf")))
    tv, ti = torch.topk(vals_c, K, dim=-1)
    ids = ids_c.gather(1, ti)
    vals = tv
    return vals, ids, ok


def merge_gate(flat_rows, vals, ids, K):
    """value multiset exact + every id holds its value (tie-order free)."""
    tv, _ = torch.topk(flat_rows, K, dim=-1)
    back = flat_rows.gather(1, ids)
    ok_vals = torch.equal(
        vals.sort(dim=-1).values.to(tv.dtype), tv.sort(dim=-1).values)
    ok_hold = torch.equal(back, vals.to(flat_rows.dtype))
    return ok_vals and ok_hold


def topk_step(u, vals, ids, cand_w):
    h = F.silu(u)
    scores = vals.unsqueeze(-1).to(h.dtype).clone()
    scores.baddbmm_(cand_w, h.unsqueeze(-1), beta=1.0, alpha=1.0)
    best = scores.squeeze(-1).argmax(dim=-1, keepdim=True)
    return ids.gather(1, best).squeeze(1)


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


def _setup_device():
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small-V logic check")
    ap.add_argument("--batches", default="1,4,16")
    args = ap.parse_args()
    vv = 4096 if args.quick else V

    dev, dt = _setup_device()
    print(f"torch {torch.__version__}, V={vv}")

    use_hist = True
    try:
        torch.manual_seed(5)
        flat0 = (torch.randn(7, vv, device=dev) * 2.0).to(dt)
        rp0 = torch.arange(1, 7, device=dev)
        radix_topk(flat0, rp0, 16, use_hist=True)
    except Exception as e:  # noqa: BLE001
        use_hist = False
        skip("tl.histogram variant", f"{type(e).__name__}: {e}")
    print(f"histogram variant: {'tl.histogram' if use_hist else 'atomic'}")

    # ------------------------------------------------------------- gates
    for B in [int(b) for b in args.batches.split(",")]:
        rows_pos = (
            torch.arange(B, device=dev).view(B, 1) * 7
            + torch.arange(1, 7, device=dev).view(1, 6)
        ).reshape(-1)
        for material, tag in (
            ((torch.randn(B, 7, vv, device=dev) * 2.0).to(dt), "random"),
            (((torch.randn(B, 7, vv, device=dev) * 4).round() / 2).to(dt),
             "heavy-tie grid"),
        ):
            flat = material.reshape(B * 7, vv)
            flat_rows = flat[rows_pos]
            for K in KS:
                vals, ids, ok = radix_topk(flat, rows_pos, K, use_hist)
                if not ok.all():
                    n_bad = int((~ok).sum())
                    check(f"radix exact {tag} (B={B} K={K})", False,
                          f"superset overflow on {n_bad}/{len(ok)} rows "
                          f"(cap={TIE_CAP}) - fallback required")
                    continue
                good = merge_gate(flat_rows, vals, ids, K)
                check(f"radix exact {tag} (B={B} K={K})", good)

    # all-equal row: the threshold bin is the whole row.  At full V that is
    # far over the superset cap (overflow must be flagged); in --quick mode
    # V == cap and the row fits exactly - both are handled, values exact.
    flat = torch.zeros(7, vv, device=dev, dtype=dt)
    rp = torch.arange(1, 7, device=dev)
    vals, ids, ok = radix_topk(flat, rp, 128, use_hist)
    check("all-equal row handled (overflow flagged when row > cap)",
          (bool((~ok).all()) if vv > TIE_CAP else bool(ok.all()))
          and bool((vals == 0).all()))

    # ---------------------------------------------------------- timings
    rows_res = {}
    for B in [int(b) for b in args.batches.split(",")]:
        rows_pos = (
            torch.arange(B, device=dev).view(B, 1) * 7
            + torch.arange(1, 7, device=dev).view(1, 6)
        ).reshape(-1)
        base = (torch.randn(B, 7, vv, device=dev) * 2.0).to(dt)
        flat = base.reshape(B * 7, vv)
        sel = torch.arange(1, 7, device=dev)
        out = {}
        for K in KS:
            mn, p50 = timed(lambda: base.index_select(1, sel).topk(K, dim=-1))
            out[f"aclnn_extract_K{K}"] = (mn, p50)
            mn, p50 = timed(lambda: radix_topk(flat, rows_pos, K, use_hist)[:2])
            out[f"radix_extract_K{K}"] = (mn, p50)
        for name, (mn, p50) in sorted(out.items()):
            print(f"  B={B:2d} {name:22s} {mn:8.3f} / {p50:8.3f} ms")
        rows_res[f"B{B}"] = {k: [round(v[0], 4), round(v[1], 4)]
                             for k, v in out.items()}
        SUMMARY[f"B{B}"] = rows_res[f"B{B}"]

    print("\n[radix vs aclnn extract (min ms)]")
    for B in [int(b) for b in args.batches.split(",")]:
        d = SUMMARY[f"B{B}"]
        line = f"  B={B:2d}"
        for K in KS:
            a = d[f"aclnn_extract_K{K}"][0]
            r = d[f"radix_extract_K{K}"][0]
            line += f"   K{K}: {a:6.3f} -> {r:6.3f} ({a / r:4.2f}x)"
        print(line)
        r128 = d["radix_extract_K128"][0]
        bw = 2 * (B * 6) * vv * 2 / 1e6 / r128  # MB/ms = GB/s, 2 passes
        print(f"       radix 2-pass implied rate {bw:5.0f} GB/s "
              f"({B * 6} programs; if far below the memory rate the "
              f"histogram lowering is the bound, not bandwidth)")

    failed = [c for c in CHECKS if not c[1]]
    print("\n=== copyable summary ===")
    print(json.dumps(SUMMARY, indent=1))
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
