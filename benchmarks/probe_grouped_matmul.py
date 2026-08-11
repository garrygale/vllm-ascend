#!/usr/bin/env python3
# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""NPU probe for the Domino fused context-KV grouped W4A8 projection.

Domino projects the flare-fused context states through per-layer
``k_proj_target``/``v_proj_target``.  With on-the-fly W4A8 quantization the
fast bf16 ``torch.bmm`` fused path is unavailable, so the candidate is one
``torch_npu.npu_grouped_matmul`` call that runs all 7 layers at once:

  * single bf16 ``x`` of shape ``[D*T, K]`` (all layers share the input),
  * int32-packed int4 K+V weights (``[K, 2N//8]`` per layer, stacked or as a
    list), with per-channel bf16 ``antiquant_scale``,
  * ``antiquant_offset`` must be non-null in A16W4 mode (this CANN rejects
    nullptr); Domino quantization is symmetric, so zeros are passed,
  * a ``group_list`` (cumsum over ``T``) that splits ``x`` by layer.

Variants probed (the doc and the torch_npu tests disagree with the production
MoE usage, so we try both):

  * G1a: single 3D weight ``[D, K, 2N//8]`` + ``split_item=3`` (torch_npu
    test ``x1w1y1`` pattern),
  * G1b: same inputs + ``split_item=2`` (vllm-ascend MoE W4A8 pattern),
  * G2:  list of 7 2D weights + ``split_item=3`` (torch_npu test
    ``x1wNy1`` pattern),
  * L:   per-layer ``npu_weight_quant_batchmatmul`` fallback (D2-validated),
         included as the numeric baseline.

Also checks whether ``torch.cat``/``torch.stack`` of the packed int32 weights
works directly or needs an explicit ``npu_format_cast`` to ND first.

Each variant runs eagerly and inside ACL graph capture (GLOBAL/RELAXED) and is
compared against an fp32 torch reference.

Run directly on an NPU:
    python benchmarks/probe_grouped_matmul.py
"""

from __future__ import annotations

import time

import torch
import torch_npu

ACL_FORMAT_ND = 2

D = 7          # draft layers
T = 8          # context tokens per layer
K = 4096       # target hidden size (k_proj_target input)
N_KV = 1024    # per-projection output size (8 heads * 128)
OUT_N = 2 * N_KV
# Grouped and per-layer ops should agree tightly (same dequant math).
TOLERANCE_WQB = 0.5
# Loose bound vs the fp32 torch reference: bf16 scale rounding and matmul
# accumulation differences alone reach a few units for K=4096.
TOLERANCE_FP32 = 5.0


def _format_name(t: torch.Tensor) -> str:
    try:
        return str(torch_npu.get_npu_format(t))
    except Exception:  # noqa: BLE001
        return "?"


def _pack_int4(w_int: torch.Tensor) -> torch.Tensor:
    """Pack [N, K] int4 values into [K, N//8] int32 (Domino W4A8 layout)."""
    return torch_npu.npu_convert_weight_to_int4pack(w_int.t().contiguous())


def _diag(name: str, ok: bool, detail: str = "") -> None:
    print(
        f"{name:52s} {'OK' if ok else 'FAIL'} {detail}",
        flush=True,
    )


def _ref_projection(x: torch.Tensor, w_ints, scales) -> torch.Tensor:
    """fp32 reference: [D, T, 2*N_KV]."""
    t_size = x.shape[0] // D
    refs = []
    for l in range(D):
        fused_int = torch.cat(
            [w_ints[l][0], w_ints[l][1]], dim=0
        ).float()  # [2N, K]
        scale = (
            torch.cat([scales[l][0], scales[l][1]])
            .to(torch.bfloat16)
            .float()
        )  # bf16-rounded, like the op input
        x_l = x[l * t_size:(l + 1) * t_size].float()
        refs.append(x_l @ (fused_int * scale.unsqueeze(1)).t())
    return torch.stack(refs, dim=0)


def _grouped_call(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    scales: list[torch.Tensor],
    offsets: list[torch.Tensor],
    split_item: int,
    group_list: torch.Tensor,
) -> torch.Tensor:
    out = torch_npu.npu_grouped_matmul(
        x=[x],
        weight=weights,
        antiquant_scale=scales,
        antiquant_offset=offsets,
        group_list=group_list,
        split_item=split_item,
        group_type=0,
        group_list_type=0,
        output_dtype=torch.bfloat16,
    )[0]
    return out.contiguous().view(D, x.shape[0] // D, OUT_N)


def _wqb_fallback(x: torch.Tensor, w_cat_nd, scale_2d_bf16) -> torch.Tensor:
    """Per-layer npu_weight_quant_batchmatmul baseline (7 calls)."""
    t_size = x.shape[0] // D
    outs = []
    for l in range(D):
        outs.append(
            torch_npu.npu_weight_quant_batchmatmul(
                x[l * t_size:(l + 1) * t_size],
                w_cat_nd[l],
                antiquant_scale=scale_2d_bf16[l],
                antiquant_group_size=0,
            )
        )
    return torch.stack(outs, dim=0)


def _proj_w8a8(x: torch.Tensor, w8: torch.Tensor, scale: torch.Tensor):
    x8, x8s = torch_npu.npu_dynamic_quant(x)
    if x8s.dim() == 2:
        x8s = x8s.squeeze(1)
    return torch_npu.npu_quant_matmul(
        x8,
        w8,
        scale,
        pertoken_scale=x8s,
        bias=None,
        output_dtype=x.dtype,
    )


def _w8a8_fallback(x: torch.Tensor, w8_list, scale_2d) -> torch.Tensor:
    """Per-layer npu_dynamic_quant + npu_quant_matmul baseline (7 calls)."""
    t_size = x.shape[0] // D
    outs = []
    for l in range(D):
        outs.append(
            _proj_w8a8(
                x[l * t_size:(l + 1) * t_size], w8_list[l], scale_2d[l]
            )
        )
    return torch.stack(outs, dim=0)


def _grouped_w8a8(
    x: torch.Tensor,
    w8_3d: torch.Tensor,
    scale_2d: torch.Tensor,
    group_list: torch.Tensor,
) -> torch.Tensor:
    """True W8A8 grouped: quantize the shared activation once, one matmul."""
    x8, x8s = torch_npu.npu_dynamic_quant(x)
    if x8s.dim() == 2:
        x8s = x8s.squeeze(1)
    out = torch_npu.npu_grouped_matmul(
        x=[x8],
        weight=[w8_3d],
        # A8W8 per-token quant requires scale==BF16 when the output is BF16;
        # FLOAT scale is only valid with FLOAT16 output (aclnnGroupedMatmulV5).
        scale=[scale_2d.to(torch.bfloat16)],
        per_token_scale=[x8s],
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=0,
        output_dtype=torch.bfloat16,
    )[0]
    return out.contiguous().view(D, x.shape[0] // D, OUT_N)


def _ref_w8a8(x: torch.Tensor, w8_ints, scale_2d) -> torch.Tensor:
    """Formula-3 reference: (x8 @ w8) * weight_scale * per_token_scale."""
    t_size = x.shape[0] // D
    x8, x8s = torch_npu.npu_dynamic_quant(x)
    if x8s.dim() == 2:
        x8s = x8s.squeeze(1)
    refs = []
    for l in range(D):
        fused = torch.cat(
            [w8_ints[l][0], w8_ints[l][1]], dim=0
        ).float()  # [2N, K]
        x8_l = x8[l * t_size:(l + 1) * t_size].float()
        s_l = x8s[l * t_size:(l + 1) * t_size].unsqueeze(1)
        scale = scale_2d[l].to(torch.bfloat16).float()  # bf16-rounded, like the op input
        refs.append(
            (x8_l @ fused.t())
            * scale.unsqueeze(0)
            * s_l
        )
    return torch.stack(refs, dim=0)


def _make_variants(
    x: torch.Tensor,
    group_list: torch.Tensor,
    w_cat_nd: list[torch.Tensor],
    scale_2d_bf16: torch.Tensor,
    offset_3d: list[torch.Tensor],
    offset_list: list[torch.Tensor],
) -> list[tuple[str, callable]]:
    w_3d = torch.stack(w_cat_nd, dim=0)
    weight_3d = [w_3d]
    scale_3d = [scale_2d_bf16]
    weight_list = w_cat_nd
    scale_list = [scale_2d_bf16[l] for l in range(D)]
    return [
        ("G1a split_item=3 3D", lambda: _grouped_call(
            x, weight_3d, scale_3d, offset_3d, 3, group_list
        )),
        ("G1b split_item=2 3D", lambda: _grouped_call(
            x, weight_3d, scale_3d, offset_3d, 2, group_list
        )),
        ("G2 split_item=3 list", lambda: _grouped_call(
            x, weight_list, scale_list, offset_list, 3, group_list
        )),
        ("L per-layer wqb", lambda: _wqb_fallback(
            x, w_cat_nd, scale_2d_bf16
        )),
    ]


def _time_eager(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def _time_graph(fn, iters: int, warmup: int) -> float:
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    with torch.npu.graph(graph, stream=stream, capture_error_mode="global"):
        fn()
    for _ in range(warmup):
        graph.replay()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        graph.replay()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def _check(
    name: str,
    out: torch.Tensor,
    ref: torch.Tensor,
    ref_wqb: torch.Tensor | None = None,
) -> bool:
    err_fp32 = (out.float() - ref.float()).abs().max().item()
    rel = err_fp32 / ref.float().abs().max().item()
    ok = err_fp32 <= TOLERANCE_FP32
    err_wqb = None
    if ref_wqb is not None:
        err_wqb = (out.float() - ref_wqb.float()).abs().max().item()
        ok = ok and err_wqb <= TOLERANCE_WQB
    print(
        f"{name:34s} err_fp32={err_fp32:.4f} rel={rel:.4f} "
        f"err_wqb={'-' if err_wqb is None else f'{err_wqb:.4f}'} "
        f"{'OK' if ok else 'FAIL'}",
        flush=True,
    )
    return ok


def main() -> None:
    print(f"torch_npu version: {getattr(torch_npu, '__version__', 'unknown')}")
    # Service-like storage: vllm-ascend sets this in worker/model_runner_v1.py.
    torch.npu.config.allow_internal_format = True
    print("allow_internal_format=True (service-like)")
    print(f"D={D} T={T} K={K} N_KV={N_KV} out_N={OUT_N}")

    device = "npu"
    torch.manual_seed(0)

    # Build per-layer int4 K/V weights and per-channel scales (un-packed ints
    # are kept for the fp32 reference).
    packed_k = []
    packed_v = []
    w_ints = []
    scales = []
    for _ in range(D):
        w_int_k = torch.randint(-7, 8, (N_KV, K), dtype=torch.int32, device=device)
        w_int_v = torch.randint(-7, 8, (N_KV, K), dtype=torch.int32, device=device)
        w_ints.append((w_int_k, w_int_v))
        packed_k.append(_pack_int4(w_int_k))  # [K, N_KV//8]
        packed_v.append(_pack_int4(w_int_v))
        scales.append(
            (
                torch.rand(N_KV, device=device) * 0.9 + 0.1,
                torch.rand(N_KV, device=device) * 0.9 + 0.1,
            )
        )

    x = torch.randn(D * T, K, dtype=torch.bfloat16, device=device).contiguous()
    group_list = (
        torch.arange(1, D + 1, dtype=torch.int64, device=device) * T
    )
    ref = _ref_projection(x, w_ints, scales)

    print("packed weight formats / fused single-pack list:")
    print(
        f"packed_k[0] format={_format_name(packed_k[0])} "
        f"packed_v[0] format={_format_name(packed_v[0])}"
    )
    # Build the fused K+V packed weight per layer in ONE pack call.  Packing
    # K and V separately and concatenating the packed tensors is NOT valid on
    # this CANN (the op misreads the concatenated layout), so this is the only
    # safe construction.
    fused_w_list = []
    try:
        for l in range(D):
            fused_int = torch.cat([w_ints[l][0], w_ints[l][1]], dim=0)
            fused_w_list.append(
                torch_npu.npu_format_cast(
                    _pack_int4(fused_int), ACL_FORMAT_ND
                )
            )
        print(
            "fused single-pack x7 (pack cat([K,V]) once): "
            f"OK shapes={[tuple(w.shape) for w in fused_w_list]}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"fused single-pack x7: FAIL {type(exc).__name__}: {exc}",
            flush=True,
        )
        return
    if len(fused_w_list) != D:
        print("cannot build fused single-pack weights; aborting", flush=True)
        return
    w_cat_nd = fused_w_list  # list of [K, 2N//8] per layer

    scale_2d_bf16 = torch.stack(
        [
            torch.cat([scales[l][0], scales[l][1]]).to(torch.bfloat16)
            for l in range(D)
        ],
        dim=0,
    )  # [D, OUT_N]
    # W8A8 fused int8 K+V weights (concatenating int8 tensors is safe).
    w8_ints = []
    w8_scales = []
    for _ in range(D):
        w8_ints.append(
            (
                torch.randint(
                    -128, 127, (N_KV, K), dtype=torch.int32, device=device
                ),
                torch.randint(
                    -128, 127, (N_KV, K), dtype=torch.int32, device=device
                ),
            )
        )
        w8_scales.append(
            (
                torch.rand(N_KV, device=device) * 0.9 + 0.1,
                torch.rand(N_KV, device=device) * 0.9 + 0.1,
            )
        )
    w8_cat_nd = torch.stack(
        [
            torch.cat([wi[0], wi[1]], dim=0)
            .to(torch.int8)
            .t()
            .contiguous()
            for wi in w8_ints
        ],
        dim=0,
    )  # [D, K, 2N] int8
    scale_8_2d = torch.stack(
        [torch.cat([ws[0], ws[1]]) for ws in w8_scales],
        dim=0,
    )  # [D, 2N] fp32
    print("-" * 60)
    print("diagnostics (isolate pack semantics):")

    # Single projection, packed once (D2-style, no cat): should be tiny.
    packed_k_nd = [
        torch_npu.npu_format_cast(pk, ACL_FORMAT_ND) for pk in packed_k
    ]
    scale_k_bf16 = [scales[l][0].to(torch.bfloat16) for l in range(D)]
    out_single = torch_npu.npu_weight_quant_batchmatmul(
        x[:T],
        packed_k_nd[0],
        antiquant_scale=scale_k_bf16[0],
        antiquant_group_size=0,
    )
    ref_single = (
        x[:T].float()
        @ (
            w_ints[0][0].float()
            * scale_k_bf16[0].float().unsqueeze(1)
        ).t()
    )
    err_single = (out_single.float() - ref_single).abs().max().item()
    _diag(
        "single projection wqb (no cat) vs fp32 ref",
        err_single <= TOLERANCE_FP32,
        f"err={err_single:.4f}",
    )

    # Fused K+V packed in ONE call (no cat-of-packs): isolates the cat.
    fused_int_0 = torch.cat([w_ints[0][0], w_ints[0][1]], dim=0)  # [2N, K]
    fused_packed = _pack_int4(fused_int_0)
    fused_packed_nd = torch_npu.npu_format_cast(fused_packed, ACL_FORMAT_ND)
    out_fused1 = torch_npu.npu_weight_quant_batchmatmul(
        x[:T],
        fused_packed_nd,
        antiquant_scale=scale_2d_bf16[0],
        antiquant_group_size=0,
    )
    ref_fused0 = _ref_projection(x, w_ints, scales)[0]
    err_fused1 = (out_fused1.float() - ref_fused0).abs().max().item()
    _diag(
        "fused single-pack wqb vs fp32 ref",
        err_fused1 <= TOLERANCE_FP32,
        f"err={err_fused1:.4f}",
    )
    print("-" * 60)

    ref_wqb = _wqb_fallback(x, w_cat_nd, scale_2d_bf16)
    ref8 = _ref_w8a8(x, w8_ints, scale_8_2d)
    ref_w8 = _w8a8_fallback(
        x, [w8_cat_nd[l] for l in range(D)], scale_8_2d
    )
    offset_3d = [
        torch.zeros((D, OUT_N), dtype=torch.bfloat16, device=device)
    ]
    offset_list = [
        torch.zeros(OUT_N, dtype=torch.bfloat16, device=device)
        for _ in range(D)
    ]

    variants = _make_variants(
        x, group_list, w_cat_nd, scale_2d_bf16, offset_3d, offset_list
    )
    variants.append(
        ("G8 split_item=2 int8", lambda: _grouped_w8a8(
            x, w8_cat_nd, scale_8_2d, group_list
        ))
    )
    variants.append(
        ("L8 per-layer int8", lambda: _w8a8_fallback(
            x, [w8_cat_nd[l] for l in range(D)], scale_8_2d
        ))
    )

    print("-" * 60)
    all_ok = True
    for name, fn in variants:
        try:
            out = fn()
            is_w8 = name.startswith("G8") or name.startswith("L8")
            ref_use = ref8 if is_w8 else ref
            ref_wqb_use = (
                (None if name.startswith("L8") else ref_w8)
                if is_w8
                else (None if name.startswith("L ") else ref_wqb)
            )
            all_ok &= _check(
                f"{name} eager",
                out,
                ref_use,
                ref_wqb_use,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{name:34s} eager FAIL {type(exc).__name__}: {exc}", flush=True)
            all_ok = False
            continue

        for mode in ("global", "relaxed"):
            try:
                graph = torch.npu.NPUGraph()
                stream = torch.npu.Stream()
                with torch.npu.graph(graph, stream=stream, capture_error_mode=mode):
                    out_g = fn()
                graph.replay()
                torch.npu.synchronize()
                replay_err = (out_g.float() - out.float()).abs().max().item()
                is_w8 = name.startswith("G8") or name.startswith("L8")
                ref_use = ref8 if is_w8 else ref
                ref_wqb_use = (
                    (None if name.startswith("L8") else ref_w8)
                    if is_w8
                    else (None if name.startswith("L ") else ref_wqb)
                )
                ok = _check(
                    f"{name} graph[{mode}]",
                    out_g,
                    ref_use,
                    ref_wqb_use,
                )
                all_ok &= ok and replay_err == 0.0
                if replay_err != 0.0:
                    print(
                        f"    replay-vs-eager max_err={replay_err:.5f} (FAIL)",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"{name:34s} graph[{mode:8s}] FAIL "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                all_ok = False

    print("-" * 60)
    print("timing (ms per call, eager vs graph replay):")
    # Precompute runs per step with num_target_tokens context tokens: the
    # decode batch size (small, e.g. 1-32) or the prompt length at prefill.
    timing_ts = (1, 4, 8, 16, 64, 256)
    timing_iters = 20
    timing_warmup = 5
    for t_size in timing_ts:
        x_t = torch.randn(
            D * t_size, K, dtype=torch.bfloat16, device=device
        ).contiguous()
        group_list_t = (
            torch.arange(1, D + 1, dtype=torch.int64, device=device) * t_size
        )
        variants_t = _make_variants(
            x_t,
            group_list_t,
            w_cat_nd,
            scale_2d_bf16,
            offset_3d,
            offset_list,
        )
        variants_t.append(
            ("G8 split_item=2 int8", lambda: _grouped_w8a8(
                x_t, w8_cat_nd, scale_8_2d, group_list_t
            ))
        )
        variants_t.append(
            ("L8 per-layer int8", lambda: _w8a8_fallback(
                x_t, [w8_cat_nd[l] for l in range(D)], scale_8_2d
            ))
        )
        print(f"T={t_size}:", flush=True)
        for name, fn in variants_t:
            try:
                ms_eager = _time_eager(fn, timing_iters, timing_warmup)
                ms_graph = _time_graph(fn, timing_iters, timing_warmup)
                print(
                    f"  {name:24s} eager={ms_eager:.3f} ms "
                    f"graph={ms_graph:.3f} ms",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  {name:24s} FAIL {type(exc).__name__}: {exc}",
                    flush=True,
                )

    print("=" * 60)
    print("RESULT:", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
