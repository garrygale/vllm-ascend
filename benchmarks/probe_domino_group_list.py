"""Probe: group_list construction for the Domino fused context-KV precompute.

The per-step fill_(T) + cumsum_(0) on the int64 group_list lands on an AI CPU
kernel (~100us per precompute step).  The serving path now uses one int64 mul
(arange[1..D] * T).  This probe measures both plus an int32 fallback, checks
numerics, and lists what the NPU profiler records so the AI CPU hop is visible.

Run on A3:  python probe_domino_group_list.py
"""

import time

import torch
import torch_npu

D = 7  # draft layers (dflare)
ITERS = 200
WARMUP = 20
T_VALUES = [96, 512, 2048, 8192]


def bench(fn, sync_each):
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    if sync_each:
        for _ in range(ITERS):
            fn()
            torch.npu.synchronize()
    else:
        for _ in range(ITERS):
            fn()
        torch.npu.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e6


def bench_variants():
    dev = "npu:0"
    g64 = torch.empty(D, dtype=torch.int64, device=dev)
    idx64 = torch.arange(1, D + 1, dtype=torch.int64, device=dev)
    g32 = torch.empty(D, dtype=torch.int32, device=dev)
    idx32 = torch.arange(1, D + 1, dtype=torch.int32, device=dev)

    print(f"{'T':>6} {'A fill+cumsum i64':>20} {'B mul i64':>20} "
          f"{'C mul i32':>20}  numerics")
    print(f"{'':>6} {'lat/pipe us':>20} {'lat/pipe us':>20} {'lat/pipe us':>20}")
    for T in T_VALUES:
        a = lambda: (g64.fill_(T), g64.cumsum_(0))
        b = lambda: torch.mul(idx64, T, out=g64)
        c = lambda: torch.mul(idx32, T, out=g32)

        a()
        ref = g64.clone()
        b()
        ok = torch.equal(g64, ref)
        c()

        print(f"{T:>6} "
              f"{bench(a, True):>9.1f}/{bench(a, False):<9.1f} "
              f"{bench(b, True):>9.1f}/{bench(b, False):<9.1f} "
              f"{bench(c, True):>9.1f}/{bench(c, False):<9.1f}  "
              f"{'OK' if ok else 'MISMATCH'}")


def profile_variants():
    try:
        from torch_npu.profiler import ProfilerActivity, profile
        acts = [ProfilerActivity.CPU, ProfilerActivity.NPU]
    except Exception as e:  # noqa: BLE001
        print(f"profiler unavailable: {e}")
        return

    dev = "npu:0"
    g64 = torch.empty(D, dtype=torch.int64, device=dev)
    idx64 = torch.arange(1, D + 1, dtype=torch.int64, device=dev)
    g32 = torch.empty(D, dtype=torch.int32, device=dev)
    idx32 = torch.arange(1, D + 1, dtype=torch.int32, device=dev)
    T = 2048

    with profile(activities=acts) as prof:
        for _ in range(10):
            g64.fill_(T)
            g64.cumsum_(0)
        for _ in range(10):
            torch.mul(idx64, T, out=g64)
        for _ in range(10):
            torch.mul(idx32, T, out=g32)
        torch.npu.synchronize()

    print("\nprofiler events (name / count / device_time us):")
    for ev in prof.key_averages():
        name = ev.key.lower()
        if not any(k in name for k in ("cumsum", "mul", "fill", "arange")):
            continue
        dt = getattr(ev, "self_device_time_total", 0.0)
        print(f"  {ev.key:<40} n={ev.count:>3}  {dt:>10.1f}  "
              f"device={getattr(ev, 'device_type', '?')}")
    print("hint: an AI CPU op shows up as an aclnn* event whose wall time is "
          "~100x its device time, or with an aicpu tag in the chrome trace "
          "(export with prof.export_chrome_trace if unsure).")


def main():
    print(f"torch={torch.__version__} torch_npu={torch_npu.__version__} "
          f"device={torch.npu.get_device_name(0)} D={D}")
    bench_variants()
    profile_variants()
    print("\nverdict rule: keep the mul path if its sync-each latency is well "
          "below fill+cumsum and numerics are OK; if int64 mul is still slow, "
          "switch group_list to int32 (needs group_list_type=1 in "
          "npu_grouped_matmul).")


if __name__ == "__main__":
    main()
