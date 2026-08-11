#!/usr/bin/env python3
# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""NPU probe for the W8A8 norm + int8-activation-quant fusions.

Domino's W8A8 draft path currently runs, per layer:

  * residual add -> RMSNorm -> ``npu_dynamic_quant`` -> fused qkv matmul,
  * residual add -> RMSNorm -> ``npu_dynamic_quant`` -> gate/up matmuls
    (gate and up quantize the same tensor twice),

and the fused context-KV precompute runs:

  * ``hidden_norm`` -> ``npu_dynamic_quant`` -> grouped matmul.

This probe validates the two fused ops that collapse those chains:

  * ``torch.ops.npu.npu_add_rms_norm_dynamic_quant`` (residual + RMSNorm +
    int8 dynamic quant in one call; output[0]=x8, output[2]=residual,
    output[3]=per-token scale, per the vllm-ascend fusion pass),
  * ``torch.ops._C_ascend.npu_rms_norm_dynamic_quant`` (RMSNorm + int8
    dynamic quant; returns ``(x8, per-token scale)``, used in dsa_v1).

Checks:

  * fused-vs-separate correctness on the real Domino dims,
  * graph capture (GLOBAL/RELAXED) + replay parity,
  * timing (eager + graph replay) at decode-sized token counts.

Run directly on an NPU:
    python benchmarks/probe_norm_quant_fusion.py
"""

from __future__ import annotations

import time

import torch
import torch_npu

EPS = 1e-6

# Draft model dims (qwen3-8b-domino-dflare.json).
K = 2560          # draft hidden size
NQ = 4096         # 32 heads * 128
NKV = 1024        # 8 kv heads * 128
NQKV = NQ + 2 * NKV
IM = 9728         # mlp intermediate size
D = 7             # draft layers
MS = (15, 60, 120, 240)  # B=1,4,8,16 requests * 15 draft tokens

# Target-side context-KV precompute dims.
H_T = 4096        # target hidden size (k_proj_target input)
KV_T = 2 * NKV    # fused K+V output size
T_CTX = (8, 16, 64, 256)

# Fused vs separate on identical math should be tight; a handful of int8
# rounding flips (norm computed slightly differently) is acceptable.
TOL_X8 = 1.0      # max abs diff on the int8 activations
TOL_SCALE = 1e-2  # max abs diff on the fp32 per-token scale
TOL_FP32 = 5.0    # output tolerance for the matmul patterns


def _rms_norm_ref(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """fp32 RMSNorm reference, output cast back to the input dtype."""
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    x_n = x_f * torch.rsqrt(var + eps)
    return (x_n * w.float()).to(x.dtype)


def _squeeze_scale(s: torch.Tensor) -> torch.Tensor:
    return s.squeeze(-1) if s.dim() == 2 else s


def _quant_ref(x: torch.Tensor):
    """npu_dynamic_quant with a 1D per-token scale."""
    x8, x8s = torch_npu.npu_dynamic_quant(x)
    return x8, _squeeze_scale(x8s)


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


def _diag_x8(name: str, x8_f, x8_ref, x8s_f, x8s_ref) -> None:
    err_x8 = (x8_f.float() - x8_ref.float()).abs().max().item()
    err_s = (x8s_f.float() - x8s_ref.float()).abs().max().item()
    mismatch = (x8_f != x8_ref).sum().item()
    ok = err_x8 <= TOL_X8 and err_s <= TOL_SCALE
    print(
        f"{name:48s} x8_err={err_x8:.4f} x8_mismatch={mismatch} "
        f"scale_err={err_s:.6f} {'OK' if ok else 'FAIL'}",
        flush=True,
    )
    return ok


def main() -> None:
    print(f"torch_npu version: {getattr(torch_npu, '__version__', 'unknown')}")
    torch.npu.config.allow_internal_format = True
    print("allow_internal_format=True (service-like)")
    print(f"K={K} NQKV={NQKV} IM={IM} D={D} H_T={H_T} KV_T={KV_T}")

    device = "npu"
    torch.manual_seed(0)

    has_add = hasattr(torch.ops.npu, "npu_add_rms_norm_dynamic_quant")
    has_rnq = hasattr(torch.ops._C_ascend, "npu_rms_norm_dynamic_quant")
    print(f"npu_add_rms_norm_dynamic_quant: {has_add}")
    print(f"npu_rms_norm_dynamic_quant: {has_rnq}")
    if not (has_add and has_rnq):
        print("missing fused ops; aborting", flush=True)
        return

    x = torch.randn(MS[0], K, dtype=torch.bfloat16, device=device)
    w_norm = (
        torch.rand(K, device=device, dtype=torch.bfloat16) * 0.5 + 0.5
    )

    print("-" * 60)
    print("correctness (fused vs separate):")
    all_ok = True

    # --- RMSNorm + dynamic quant (context-KV style, no residual) ---
    x8_ref, x8s_ref = _quant_ref(_rms_norm_ref(x, w_norm, EPS))
    x8_f, x8s_f = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(
        x, w_norm, epsilon=EPS
    )
    x8s_f = _squeeze_scale(x8s_f)
    all_ok &= _diag_x8(
        "rms_norm_dynamic_quant", x8_f, x8_ref, x8s_f, x8s_ref
    )

    # --- Residual-add + RMSNorm + dynamic quant (draft layer style) ---
    residual = torch.randn_like(x)
    out = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
        x,
        residual,
        w_norm,
        epsilon=EPS,
        output_mask=[True, False],
    )
    print(
        f"  add_rms_norm_dynamic_quant outputs: len={len(out)} "
        f"shapes={[tuple(o.shape) for o in out]} dtypes="
        f"{[o.dtype for o in out]}",
        flush=True,
    )
    sum_ref = x + residual
    x8_ref, x8s_ref = _quant_ref(_rms_norm_ref(sum_ref, w_norm, EPS))
    x8_f, x8s_f = out[0], _squeeze_scale(out[3])
    all_ok &= _diag_x8(
        "add_rms_norm_dynamic_quant", x8_f, x8_ref, x8s_f, x8s_ref
    )
    err_res = (out[2].float() - sum_ref.float()).abs().max().item()
    print(
        f"  residual output[2] vs x+residual                "
        f"err={err_res:.6f} {'OK' if err_res <= 0.05 else 'FAIL'}",
        flush=True,
    )
    all_ok &= err_res <= 0.05

    # --- Gate/up shared quant: one fused norm+quant -> two matmuls ---
    w_gate = torch.randint(
        -128, 127, (IM, K), dtype=torch.int32, device=device
    ).to(torch.int8).t().contiguous()
    w_up = torch.randint(
        -128, 127, (IM, K), dtype=torch.int32, device=device
    ).to(torch.int8).t().contiguous()
    s_gate = torch.rand(IM, device=device) * 0.9 + 0.1
    s_up = torch.rand(IM, device=device) * 0.9 + 0.1

    def _gate_up_sep(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normed = _rms_norm_ref(h, w_norm, EPS)
        g8, gs = _quant_ref(normed)
        u8, us = _quant_ref(normed)
        return (
            torch_npu.npu_quant_matmul(
                g8, w_gate, s_gate, pertoken_scale=gs, bias=None,
                output_dtype=h.dtype,
            ),
            torch_npu.npu_quant_matmul(
                u8, w_up, s_up, pertoken_scale=us, bias=None,
                output_dtype=h.dtype,
            ),
        )

    def _gate_up_fused(h: torch.Tensor, res: torch.Tensor):
        o = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
            h, res, w_norm, epsilon=EPS, output_mask=[True, False]
        )
        return (
            torch_npu.npu_quant_matmul(
                o[0], w_gate, s_gate, pertoken_scale=_squeeze_scale(o[3]),
                bias=None, output_dtype=h.dtype,
            ),
            torch_npu.npu_quant_matmul(
                o[0], w_up, s_up, pertoken_scale=_squeeze_scale(o[3]),
                bias=None, output_dtype=h.dtype,
            ),
        )

    g_ref, u_ref = _gate_up_sep(x)
    g_f, u_f = _gate_up_fused(x, torch.randn_like(x))
    err_g = (g_f.float() - g_ref.float()).abs().max().item()
    err_u = (u_f.float() - u_ref.float()).abs().max().item()
    print(
        f"  gate/up shared-quant                      "
        f"gate_err={err_g:.4f} up_err={err_u:.4f} "
        f"{'OK' if max(err_g, err_u) <= TOL_FP32 else 'FAIL'}",
        flush=True,
    )
    all_ok &= max(err_g, err_u) <= TOL_FP32

    # --- Context-KV: fused norm+quant -> grouped matmul ---
    w_kv = torch.randint(
        -128, 127, (D, H_T, KV_T), dtype=torch.int32, device=device
    ).to(torch.int8)
    s_kv = (torch.rand(D, KV_T, device=device) * 0.9 + 0.1).to(
        torch.bfloat16
    )
    w_hn = (
        torch.rand(H_T, device=device, dtype=torch.bfloat16) * 0.5 + 0.5
    )

    def _ctx_grouped(x_ctx: torch.Tensor, group_list: torch.Tensor,
                     x8_in, x8s_in) -> torch.Tensor:
        return torch_npu.npu_grouped_matmul(
            x=[x8_in],
            weight=[w_kv],
            scale=[s_kv],
            per_token_scale=[_squeeze_scale(x8s_in)],
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=0,
            output_dtype=torch.bfloat16,
        )[0].contiguous().view(D, x_ctx.shape[0] // D, KV_T)

    t = T_CTX[0]
    x_ctx = torch.randn(D * t, H_T, dtype=torch.bfloat16, device=device)
    group_list = (
        torch.arange(1, D + 1, dtype=torch.int64, device=device) * t
    )
    x8_ref, x8s_ref = _quant_ref(_rms_norm_ref(x_ctx, w_hn, EPS))
    out_ref = _ctx_grouped(x_ctx, group_list, x8_ref, x8s_ref)
    x8_f, x8s_f = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(
        x_ctx, w_hn, epsilon=EPS
    )
    out_f = _ctx_grouped(x_ctx, group_list, x8_f, x8s_f)
    err_ctx = (out_f.float() - out_ref.float()).abs().max().item()
    all_ok &= _diag_x8(
        "context-KV norm+quant", x8_f, x8_ref, x8s_f, x8s_ref
    )
    print(
        f"  context-KV grouped out vs separate       "
        f"err={err_ctx:.4f} {'OK' if err_ctx <= TOL_FP32 else 'FAIL'}",
        flush=True,
    )
    all_ok &= err_ctx <= TOL_FP32

    print("-" * 60)
    print("graph capture / replay parity (replay vs eager):")
    cases = [
        (
            "rms_norm_dynamic_quant",
            lambda: (x,),
            lambda x_in: torch.ops._C_ascend.npu_rms_norm_dynamic_quant(
                x_in, w_norm, epsilon=EPS
            ),
        ),
        (
            "add_rms_norm_dynamic_quant",
            lambda: (x, torch.randn_like(x)),
            lambda x_in, res_in: torch.ops.npu.npu_add_rms_norm_dynamic_quant(
                x_in,
                res_in,
                w_norm,
                epsilon=EPS,
                output_mask=[True, False],
            ),
        ),
        (
            "gate/up shared-quant",
            lambda: (x, torch.randn_like(x)),
            _gate_up_fused,
        ),
        (
            "context-KV norm+quant+grouped",
            lambda: (x_ctx,),
            lambda x_in: _ctx_grouped(
                x_in,
                group_list,
                *torch.ops._C_ascend.npu_rms_norm_dynamic_quant(
                    x_in, w_hn, epsilon=EPS
                ),
            ),
        ),
    ]
    for name, make_inputs, fn in cases:
        for mode in ("global", "relaxed"):
            try:
                eager_out = fn(*make_inputs())
                if not isinstance(eager_out, (tuple, list)):
                    eager_out = (eager_out,)
                graph_inputs = make_inputs()
                g = torch.npu.NPUGraph()
                stream = torch.npu.Stream()
                with torch.npu.graph(
                    g, stream=stream, capture_error_mode=mode
                ):
                    graph_out = fn(*graph_inputs)
                if not isinstance(graph_out, (tuple, list)):
                    graph_out = (graph_out,)
                g.replay()
                torch.npu.synchronize()
                err = 0.0
                for a, b in zip(graph_out, eager_out):
                    err = max(err, (a.float() - b.float()).abs().max().item())
                print(
                    f"{name:40s} graph[{mode:8s}] replay_err={err:.6f} "
                    f"{'OK' if err == 0.0 else 'FAIL'}",
                    flush=True,
                )
                all_ok &= err == 0.0
            except Exception as exc:  # noqa: BLE001
                print(
                    f"{name:40s} graph[{mode:8s}] FAIL "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                all_ok = False

    print("-" * 60)
    print("timing (ms per call, eager vs graph):")
    iters, warmup = 20, 5
    for m in MS:
        x_m = torch.randn(m, K, dtype=torch.bfloat16, device=device)
        res_m = torch.randn_like(x_m)
        runs = [
            (
                "sep add+norm+quant",
                lambda: _quant_ref(_rms_norm_ref(x_m + res_m, w_norm, EPS)),
            ),
            (
                "fus add_rms_norm_dynamic_quant",
                lambda: torch.ops.npu.npu_add_rms_norm_dynamic_quant(
                    x_m,
                    res_m,
                    w_norm,
                    epsilon=EPS,
                    output_mask=[True, False],
                ),
            ),
            (
                "sep gate/up (2x quant)",
                lambda: _gate_up_sep(x_m),
            ),
            (
                "fus gate/up (shared quant)",
                lambda: _gate_up_fused(x_m, res_m),
            ),
        ]
        print(f"M={m}:", flush=True)
        for name, fn in runs:
            try:
                ms_e = _time(fn, iters, warmup, graph=False)
                ms_g = _time(fn, iters, warmup, graph=True)
                print(
                    f"  {name:30s} eager={ms_e:.3f} ms graph={ms_g:.3f} ms",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  {name:30s} FAIL {type(exc).__name__}: {exc}",
                    flush=True,
                )

    for t in T_CTX:
        x_t = torch.randn(D * t, H_T, dtype=torch.bfloat16, device=device)
        group_list_t = (
            torch.arange(1, D + 1, dtype=torch.int64, device=device) * t
        )
        runs = [
            (
                "sep ctx norm+quant+grouped",
                lambda: _ctx_grouped(
                    x_t,
                    group_list_t,
                    *_quant_ref(_rms_norm_ref(x_t, w_hn, EPS)),
                ),
            ),
            (
                "fus ctx norm+quant+grouped",
                lambda: _ctx_grouped(
                    x_t,
                    group_list_t,
                    *torch.ops._C_ascend.npu_rms_norm_dynamic_quant(
                        x_t, w_hn, epsilon=EPS
                    ),
                ),
            ),
        ]
        print(f"T={t}:", flush=True)
        for name, fn in runs:
            try:
                ms_e = _time(fn, iters, warmup, graph=False)
                ms_g = _time(fn, iters, warmup, graph=True)
                print(
                    f"  {name:30s} eager={ms_e:.3f} ms graph={ms_g:.3f} ms",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  {name:30s} FAIL {type(exc).__name__}: {exc}",
                    flush=True,
                )

    print("=" * 60)
    print("RESULT:", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
