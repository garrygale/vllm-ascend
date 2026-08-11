#!/usr/bin/env python3
# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""NPU probe for ``apply_top_k_top_p`` mask semantics.

Compares the NPU top-k/top-p op used by the service
(``torch_npu.npu_top_k_top_p`` on A2/A3, with the same dtype casts as
``vllm_ascend.sample.sampler._apply_top_k_top_p_torch_npu``) against vLLM's
reference semantics in
``vllm.v1.sample.ops.topk_topp_sampler.apply_top_k_top_p_pytorch``:

  * top-k keeps all logits >= the k-th largest value (ties may keep more),
  * top-p is computed by softmax over the (top-k-masked) distribution and
    keeps the smallest set whose cumulative probability is >= p, at least one,
  * when both are set, top-p runs on the top-k subset only.

The probe checks the retained-token support, per-row kept counts, and the
retained logit values, in eager mode and inside ACL graph capture (GLOBAL and
RELAXED).  It also reports tie-boundary behavior separately, since the
reference keeps all tied values while some kernels keep exactly k.

Run directly on an NPU:
    python benchmarks/probe_topk_topp.py
"""

from __future__ import annotations

import torch
import torch_npu

B, V = 8, 4096
SEEDS = (0, 1, 2)
ATOL = 0.05  # retained values must survive bf16 rounding


def _ref_apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """Reference semantics from vLLM's apply_top_k_top_p_pytorch (fp32)."""
    logits = logits.float().clone()
    if p is None and k is None:
        return logits

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        top_k_count = logits_sort.size(1) - k.to(torch.long)
        top_k_cutoff = logits_sort.gather(1, top_k_count.unsqueeze(dim=1))
        logits_sort = logits_sort.masked_fill(
            logits_sort < top_k_cutoff, -float("inf")
        )

    if p is not None:
        probs_sort = logits_sort.softmax(dim=-1, dtype=torch.float32)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False  # at least one
        logits_sort = logits_sort.masked_fill(top_p_mask, -float("inf"))

    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)


def _npu_apply(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """Service path: npu_top_k_top_p with the Ascend wrapper's dtype casts."""
    logits = logits.clone()
    if p is None and k is None:
        return logits
    if p is not None and p.dtype != logits.dtype:
        p = p.to(logits.dtype)
    if k is not None and k.dtype != torch.int32:
        k = k.to(torch.int32)
    return torch_npu.npu_top_k_top_p(logits, k=k, p=p)


def _compare(name: str, npu_out: torch.Tensor, ref: torch.Tensor) -> bool:
    npu_finite = torch.isfinite(npu_out.float())
    ref_finite = torch.isfinite(ref)

    support_ok = torch.equal(npu_finite, ref_finite)
    kept_npu = npu_finite.sum(dim=1)
    kept_ref = ref_finite.sum(dim=1)
    counts_ok = torch.equal(kept_npu, kept_ref)

    if support_ok and counts_ok:
        val_err = (npu_out.float()[npu_finite] - ref[npu_finite]).abs().max().item()
        val_ok = val_err <= ATOL
    else:
        val_err = float("nan")
        val_ok = False

    ok = support_ok and counts_ok and val_ok
    print(
        f"{name:28s} support={'OK' if support_ok else 'DIFF'} "
        f"counts={'OK' if counts_ok else 'DIFF'} "
        f"val_err={val_err:.5f} {'PASS' if ok else 'FAIL'}",
        flush=True,
    )
    if not support_ok:
        diff_rows = (npu_finite != ref_finite).any(dim=1).nonzero().flatten().tolist()
        print(f"    support diff rows={diff_rows} "
              f"npu_kept={kept_npu[diff_rows].tolist()} "
              f"ref_kept={kept_ref[diff_rows].tolist()}", flush=True)
    return ok


def _run_eager(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    ref: torch.Tensor,
) -> bool:
    npu_out = _npu_apply(logits, k, p)
    return _compare("eager", npu_out, ref)


def _run_graph(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    ref: torch.Tensor,
    mode: str,
) -> bool:
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    with torch.npu.graph(graph, stream=stream, capture_error_mode=mode):
        npu_out = _npu_apply(logits, k, p)
    graph.replay()
    torch.npu.synchronize()
    return _compare(f"graph[{mode}]", npu_out, ref)


def _run_ties_check() -> None:
    """INFO: behavior when the top-k boundary has exact ties."""
    logits = torch.randn(B, V, dtype=torch.bfloat16, device="npu")
    # Row 7: 40 identical top values, so vLLM's `>= cutoff` keeps all 40.
    logits[7, :40] = 10.0
    k = torch.full((B,), 20, dtype=torch.int32, device="npu")
    p = None
    ref = _ref_apply_top_k_top_p(logits.float(), k, None)
    npu_out = _npu_apply(logits, k, p)
    npu_kept = torch.isfinite(npu_out.float()).sum(dim=1)
    ref_kept = torch.isfinite(ref).sum(dim=1)
    same = torch.equal(npu_kept, ref_kept)
    print(
        f"TIE-BOUNDARY INFO: npu_kept={npu_kept[7].item()} "
        f"ref_kept={ref_kept[7].item()} "
        f"{'same' if same else 'different (NPU keeps exactly k, ref keeps ties)'}",
        flush=True,
    )


def main() -> None:
    print(f"torch_npu version: {getattr(torch_npu, '__version__', 'unknown')}")
    # Service-like storage: vllm-ascend sets this in worker/model_runner_v1.py.
    torch.npu.config.allow_internal_format = True
    print("allow_internal_format=True (service-like)")
    print(f"batch={B} vocab={V} seeds={SEEDS} atol={ATOL}")

    device = "npu"
    k20 = torch.full((B,), 20, dtype=torch.int32, device=device)
    k64 = torch.full((B,), 64, dtype=torch.int32, device=device)
    k_v = torch.full((B,), V, dtype=torch.int32, device=device)
    p95 = torch.full((B,), 0.95, dtype=torch.float32, device=device)
    p70 = torch.full((B,), 0.70, dtype=torch.float32, device=device)

    cases = [
        ("k=20 p=0.95", k20, p95),
        ("k=20 p=0.70", k20, p70),
        ("k=64 p=0.95", k64, p95),
        ("k=None p=0.95", None, p95),
        ("k=20 p=None", k20, None),
        ("k=V p=0.95", k_v, p95),
        ("identity", None, None),
    ]

    all_ok = True
    for seed in SEEDS:
        torch.manual_seed(seed)
        scale = torch.linspace(0.5, 3.0, B, device=device).unsqueeze(1)
        logits = (torch.randn(B, V, device=device) * scale).to(torch.bfloat16)
        for name, k, p in cases:
            # Reference uses the same bf16-quantized p as the NPU path.
            ref_p = p.to(torch.bfloat16).float() if p is not None else None
            ref = _ref_apply_top_k_top_p(logits.float(), k, ref_p)

            all_ok &= _run_eager(logits, k, p, ref)
            for mode in ("global", "relaxed"):
                try:
                    all_ok &= _run_graph(logits, k, p, ref, mode)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"{name:28s} graph[{mode:8s}] FAIL "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    all_ok = False

    print("-" * 60)
    _run_ties_check()
    print("=" * 60)
    print("RESULT:", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
