#!/usr/bin/env python3
"""Probe for dflash_config.out_proj_after_norm on the draft final layers.

Checks, in order:
  1. Fold parity:  norm(H_d) -> output_proj -> lm_head  vs  norm(H_d) -> fused
     (W_lm @ W_out) head, in bf16, at production-ish dims.
  2. embed_proj.0 column split [z || gru] still lines up when z is draft-width.
  3. Per-step GEMM cost, unfolded pair vs folded head, eager and NPUGraph
     replay (eager timings are dispatch-dominated on A3; trust the replay).

Run on A3:   python benchmarks/probe_out_proj_after_norm.py
CPU smoke:   python benchmarks/probe_out_proj_after_norm.py --small --cpu
"""

from __future__ import annotations

import argparse
import time

import torch

REPORT: list[str] = []


def say(line: str = "") -> None:
    print(line, flush=True)
    REPORT.append(line)


def _npu():
    return getattr(torch, "npu", None)


def pick_device(want_cpu: bool) -> torch.device:
    npu = _npu()
    if not want_cpu and npu is not None and npu.is_available():
        return torch.device("npu:0")
    if not want_cpu:
        say("[warn] torch.npu unavailable, falling back to CPU (math only)")
    return torch.device("cpu")


@torch.inference_mode()
def fold_parity(H_d, H_t, V, rows, device, dtype) -> dict:
    torch.manual_seed(0)
    W_out = torch.randn(H_t, H_d, device=device, dtype=dtype) * 0.02
    W_lm = torch.randn(V, H_t, device=device, dtype=dtype) * 0.02
    h = torch.randn(rows, H_d, device=device, dtype=dtype)

    unfolded = (h @ W_out.T) @ W_lm.T
    fused_w = torch.matmul(W_lm, W_out).to(dtype)
    folded = h @ fused_w.T

    diff = (folded.float() - unfolded.float()).abs()
    top1 = (folded.argmax(-1) == unfolded.argmax(-1)).float().mean().item()
    p = torch.softmax(unfolded.float(), -1)
    q = torch.softmax(folded.float(), -1)
    kl = (p * ((p / q).clamp_min(1e-9).log())).sum(-1).mean().item()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "unfolded_abs_mean": unfolded.float().abs().mean().item(),
        "top1_match": top1,
        "kl": kl,
    }


@torch.inference_mode()
def gru_split_parity(H_d, G, M, rows, device, dtype) -> bool:
    torch.manual_seed(1)
    z = torch.randn(rows, H_d, device=device, dtype=dtype)
    s = torch.randn(rows, G, device=device, dtype=dtype)
    w1 = torch.randn(M, H_d + G, device=device, dtype=dtype) * 0.05

    cat_path = torch.cat([z, s], dim=-1) @ w1.T
    split_path = z @ w1[:, :H_d].T + s @ w1[:, H_d:].T
    return torch.allclose(cat_path, split_path, atol=2e-2 * cat_path.abs().mean())


def _time_eager(fn, iters) -> float:
    for _ in range(5):
        fn()
    npu = _npu()
    if npu is not None:
        npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if npu is not None:
        npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


@torch.inference_mode()
def _time_graph(fn, iters) -> float | None:
    npu = _npu()
    if npu is None or not npu.is_available():
        return None
    g = npu.NPUGraph()
    static_out = fn()
    npu.synchronize()
    with npu.graph(g):
        fn()
    for _ in range(5):
        g.replay()
    npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    npu.synchronize()
    del static_out
    return (time.perf_counter() - t0) / iters * 1e6


@torch.inference_mode()
def perf(H_d, H_t, V, rows, device, dtype) -> dict:
    torch.manual_seed(2)
    h = torch.randn(rows, H_d, device=device, dtype=dtype)
    W_out = torch.randn(H_t, H_d, device=device, dtype=dtype) * 0.02
    W_lm = torch.randn(V, H_t, device=device, dtype=dtype) * 0.02
    fused_w = (W_lm.float() @ W_out.float()).to(dtype)

    def unfolded():
        return (h @ W_out.T) @ W_lm.T

    def folded():
        return h @ fused_w.T

    out = {}
    out["unfolded_eager_us"] = _time_eager(unfolded, 50)
    out["folded_eager_us"] = _time_eager(folded, 50)
    g_u = _time_graph(unfolded, 200)
    g_f = _time_graph(folded, 200)
    out["unfolded_graph_us"] = g_u
    out["folded_graph_us"] = g_f
    out["bytes_fused_head_mb"] = fused_w.numel() * fused_w.element_size() / 1e6
    out["bytes_unfolded_mb"] = (
        W_lm.numel() * W_lm.element_size() + W_out.numel() * W_out.element_size()
    ) / 1e6
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--small", action="store_true", help="tiny dims for CPU smoke")
    ap.add_argument("--rows", type=int, default=512)
    args = ap.parse_args()

    device = pick_device(args.cpu)
    dtype = torch.bfloat16 if device.type == "npu" else torch.float32
    if args.small:
        H_d, H_t, V, G, M = 32, 64, 256, 16, 8
    else:
        H_d, H_t, V, G, M = 1536, 2048, 151936, 256, 256
    rows = args.rows

    say("=== out_proj_after_norm probe ===")
    say(f"device={device} dtype={dtype} H_d={H_d} H_t={H_t} V={V} rows={rows}")

    say("")
    say("-- fold parity (bf16 two-GEMM vs folded head) --")
    p = fold_parity(H_d, H_t, V, rows, device, dtype)
    scale = max(p["unfolded_abs_mean"], 1e-9)
    for k, v in p.items():
        say(f"  {k:22s} {v:.6g}" if isinstance(v, float) else f"  {k:22s} {v}")
    verdict_p = p["mean_abs"] < 0.05 * scale and p["top1_match"] > 0.98
    say(f"  parity verdict: {'PASS' if verdict_p else 'FAIL'}")

    say("")
    say("-- embed_proj [z || gru] split at draft width --")
    ok = gru_split_parity(H_d, G, M, rows, device, dtype)
    say(f"  split parity: {'PASS' if ok else 'FAIL'}")

    say("")
    say("-- per-step GEMM cost --")
    t = perf(H_d, H_t, V, rows, device, dtype)
    for k, v in t.items():
        if v is None:
            say(f"  {k:24s} n/a (no NPUGraph)")
        else:
            say(f"  {k:24s} {v:10.1f}")
    if t["unfolded_graph_us"] and t["folded_graph_us"]:
        speed = t["unfolded_graph_us"] / t["folded_graph_us"]
        say(f"  graph speedup            {speed:10.2f}x")
        say(f"  per-step saving          {t['unfolded_graph_us'] - t['folded_graph_us']:10.1f} us")
    say(f"  head memory unfolded     {t['bytes_unfolded_mb']:10.1f} MB")
    say(f"  head memory fused        {t['bytes_fused_head_mb']:10.1f} MB")

    say("")
    say("=== COPY BELOW ===")
    for line in REPORT:
        print(line, flush=True)


if __name__ == "__main__":
    main()
