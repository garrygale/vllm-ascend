"""Probe: int8 npu_quant_matmul vs bf16 matmul at K=2048 -> N=2560.

Same logical GEMM [M, K] x [K, N] on every plausible NPU path so the draft
serving regime (M <= batch*num_query_per_req, ~1024 at 128 concurrent) shows
which one to use:

  * bf16 matmul, N=2560 (fair baseline) and N=4096 (the observed comparator)
  * npu_dynamic_quant alone (activation-quant cost, for subtraction)
  * npu_quant_matmul, int8 x int8, ND weight [K, N] — matmul only
  * npu_quant_matmul with the weight pre-cast to FRACTAL_NZ
  * npu_weight_quant_batchmatmul, int8 antiquant (W8A16, weight-only)
  * npu_weight_quant_batchmatmul, int4-packed antiquant (W4A16, the bulk
    scheme the Domino draft actually runs)

Every quantized path is checked against the bf16 reference.  A path a CANN
build rejects is reported once and skipped on the remaining sizes; a
computed summary is printed at the end.

Run on A3:  python probe_matmul_quant_paths.py
"""

import time

import torch
import torch_npu

K = 2048
N = 2560
N_REF = 4096
M_VALUES = [32, 128, 512, 1024]
ITERS = 100
WARMUP = 10
REL_ERR_PASS = 0.2
ACL_FORMAT_FRACTAL_NZ = 29
ACL_FORMAT_ND = 2


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


def per_channel(weight, qmax):
    """Domino quantize_weight_per_channel math on an [N, K] weight."""
    w32 = weight.float()
    scale = (w32.abs().amax(dim=1, keepdim=True) / qmax).clamp(min=1e-6)
    return torch.round(w32 / scale).clamp(-qmax, qmax), scale


def main():
    print(f"torch={torch.__version__} torch_npu={torch_npu.__version__} "
          f"device={torch.npu.get_device_name(0)}")
    print(f"K={K} N={N} (bf16 comparator N={N_REF}) "
          f"M sweep={M_VALUES}\n")

    dev = "npu:0"
    torch.manual_seed(0)
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)  # [N, K]
    w_t = w.t().contiguous()  # [K, N] bf16
    w2_t = ((torch.randn(N_REF, K, device=dev) * 0.02)
            .to(torch.bfloat16).t().contiguous())

    w8_nk, s8 = per_channel(w, 127)
    w8_kn = w8_nk.to(torch.int8).t().contiguous()  # [K, N], stored layout
    w4_nk, s4 = per_channel(w, 7)
    packed4 = torch_npu.npu_format_cast(
        torch_npu.npu_convert_weight_to_int4pack(
            w4_nk.t().contiguous().to(torch.int32)
        ),  # [K, N//8]
        ACL_FORMAT_ND,
    )
    scale8_f32 = s8.reshape(-1).to(torch.float32)
    scale8_bf16 = s8.reshape(-1).to(torch.bfloat16)
    scale4_bf16 = s4.reshape(-1).to(torch.bfloat16)

    nz_weight = None
    try:
        nz_weight = torch_npu.npu_format_cast(w8_kn.clone(),
                                              ACL_FORMAT_FRACTAL_NZ)
    except Exception as e:  # noqa: BLE001
        print(f"FRACTAL_NZ weight cast rejected: {e}\n")

    results = {}  # M -> {name: (lat, pipe)}
    numerics = {}  # name -> (max_abs, rel)
    dead = set()  # paths that failed; skipped on later sizes
    first = True
    for M in M_VALUES:
        x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
        x8, x8s = torch_npu.npu_dynamic_quant(x)
        if x8s.dim() == 2:
            x8s = x8s.squeeze(1)

        def quant_mm(weight):
            return lambda: torch_npu.npu_quant_matmul(
                x8, weight, scale8_f32,
                pertoken_scale=x8s, bias=None, output_dtype=torch.bfloat16,
            )

        def anti8():
            return torch_npu.npu_weight_quant_batchmatmul(
                x, w8_kn,
                antiquant_scale=scale8_bf16, antiquant_group_size=0,
            )

        def anti4():
            return torch_npu.npu_weight_quant_batchmatmul(
                x, packed4,
                antiquant_scale=scale4_bf16, antiquant_group_size=0,
            )

        rows = [
            ("bf16 N=2560", lambda: torch.matmul(x, w_t)),
            ("bf16 N=4096", lambda: torch.matmul(x, w2_t)),
            ("dyn_quant", lambda: torch_npu.npu_dynamic_quant(x)),
            ("quant ND", quant_mm(w8_kn)),
            ("anti W8A16", anti8),
            ("anti W4A16", anti4),
        ]
        if nz_weight is not None:
            rows.append(("quant NZ", quant_mm(nz_weight)))

        results[M] = {}
        outs = {}
        for name, fn in rows:
            if name in dead:
                continue
            try:
                results[M][name] = (bench(fn, True), bench(fn, False))
                outs[name] = fn()
            except Exception as e:  # noqa: BLE001
                dead.add(name)
                print(f"  {name} failed on M={M}, skipping later sizes: {e}")

        print(f"M={M}")
        for name, (lat, pipe) in results[M].items():
            print(f"  {name:<14} {lat:>8.1f} / {pipe:<8.1f} us")
        print()

        if first:
            first = False
            ref = torch.matmul(x.float(), w_t.float())
            for name in ("quant ND", "anti W8A16", "anti W4A16"):
                if name not in outs:
                    continue
                d = (outs[name].float() - ref).abs()
                rel = (d.mean() / ref.abs().mean().clamp(min=1e-6)).item()
                numerics[name] = (d.max().item(), rel)

    # ---------------- computed summary ----------------
    print("===== SUMMARY =====")
    base = "bf16 N=2560"
    quant = "quant ND"
    # only rank paths that actually compute the GEMM; dyn_quant is a
    # cost-accounting row (activation quant), not a matmul path
    gemm_names = {"bf16 N=2560", "bf16 N=4096", "quant ND", "quant NZ",
                  "anti W8A16", "anti W4A16"}
    wins = {}
    for M in M_VALUES:
        r = results[M]
        if base not in r:
            continue
        gemm_r = {k: v for k, v in r.items() if k in gemm_names}
        fastest = min(gemm_r, key=lambda k: gemm_r[k][1])
        wins[fastest] = wins.get(fastest, 0) + 1
        parts = [f"fastest GEMM={fastest} ({r[fastest][1]:.1f}us pipe)"]
        if quant in r:
            parts.append(f"quantND/bf16 x{r[quant][1] / r[base][1]:.2f}")
            if "dyn_quant" in r:
                total = r["dyn_quant"][1] + r[quant][1]
                parts.append(f"quant total(dyn+mm)/bf16 "
                             f"x{total / r[base][1]:.2f}")
        if "quant NZ" in r and quant in r:
            gain = 1 - r["quant NZ"][1] / r[quant][1]
            parts.append(f"NZ precast {gain * 100:+.0f}%")
        print(f"M={M:<5} " + " | ".join(parts))

    print()
    quant_beats = [M for M in M_VALUES
                   if quant in results[M]
                   and results[M][quant][1] < results[M][base][1]]
    if quant_beats:
        print(f"int8 quant_matmul beats bf16 same-N from M={quant_beats[0]}")
    elif wins:
        alt = max(wins, key=wins.get)
        print(f"int8 quant_matmul NEVER beats bf16 at M<={M_VALUES[-1]} "
              f"-> W8A8 linears should switch to '{alt}' or bf16")
    if dead:
        print(f"unsupported on this build: {sorted(dead)}")
    if wins:
        top = max(wins, key=wins.get)
        print(f"most-wins path across sweep: {top} "
              f"({wins[top]}/{len(M_VALUES)} sizes)")
    if numerics:
        print("numerics: " + "; ".join(
            f"{name} rel={rel:.3f} "
            f"({'PASS' if rel < REL_ERR_PASS else 'CHECK'})"
            for name, (_, rel) in numerics.items()))


if __name__ == "__main__":
    main()
