#!/usr/bin/env python3
# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""NPU probe for the probabilistic rejection-sampling ``u`` draw.

The v2 NPU rejection sampler used to hardcode ``u = 0.0`` in its non-greedy
branch, which made ``log(u) = -inf`` and therefore accepted every draft token
that had any target probability (acceptance length ~= max for temperature > 0).
This probe verifies the fix: for a single draft position the acceptance
probability must be ``min(1, p_target(x) / p_draft(x))``.

It runs the real ``rejection_sample`` from
``vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils`` with synthetic
buffers, over many seeds, in eager mode and inside ACL graph capture (GLOBAL
and RELAXED), and compares the observed acceptance fraction with theory.

Run directly on an NPU:
    python benchmarks/probe_rejection_sampling.py
"""

from __future__ import annotations

import torch
import torch_npu

from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)

N_TRIALS = 2000
VOCAB = 64
TEMPERATURE = 0.6
TOLERANCE = 0.06  # ~4 sigma at N=2000 for p in [0.15, 1.0]


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return logits.softmax(dim=-1, dtype=torch.float32)


def _logit_for_prob(prob: float) -> float:
    """Logit that gives ``prob`` for one token when the other V-1 are 0."""
    return float(torch.log(torch.tensor(prob * (VOCAB - 1) / (1 - prob))))


def _build_inputs(
    draft_token: int,
    target_logits0: torch.Tensor,
    draft_logits0: torch.Tensor | None,
) -> tuple[list[object], torch.Tensor]:
    """Build one-request/one-draft input buffers for ``rejection_sample``."""
    device = "npu"
    num_logits = 2
    bonus_logits = torch.randn(VOCAB, device=device)
    # target logits are temperature-applied (as apply_sampling_params passes
    # them to the rejection sampler), plus one bonus position.
    target_logits = torch.stack(
        [target_logits0 / TEMPERATURE, bonus_logits / TEMPERATURE]
    ).float()
    draft_logits = (
        (draft_logits0 / TEMPERATURE).float().unsqueeze(0).unsqueeze(0)
        if draft_logits0 is not None
        else None
    )
    draft_sampled = torch.zeros(num_logits, dtype=torch.int64, device=device)
    draft_sampled[1] = draft_token
    cu_num_logits = torch.tensor([0, num_logits], dtype=torch.int32, device=device)
    pos = torch.arange(num_logits, dtype=torch.int64, device=device)
    idx_mapping = torch.zeros(1, dtype=torch.int64, device=device)
    expanded_idx_mapping = torch.zeros(num_logits, dtype=torch.int64, device=device)
    expanded_local_pos = torch.arange(num_logits, dtype=torch.int64, device=device)
    temperature = torch.full((1,), TEMPERATURE, dtype=torch.float32, device=device)
    seed = torch.zeros(1, dtype=torch.int64, device=device)
    return (
        [
            target_logits,
            draft_logits,
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            seed,
            1,  # num_speculative_steps (one draft position per request)
        ],
        seed,
    )


def _expected_acceptance(
    draft_token: int,
    target_logits0: torch.Tensor,
    draft_logits0: torch.Tensor | None,
) -> float:
    p_target = _softmax(target_logits0 / TEMPERATURE)[draft_token].item()
    if draft_logits0 is None:
        # One-hot draft: q(x) = 1 for the drafted token.
        return p_target
    p_draft = _softmax(draft_logits0 / TEMPERATURE)[draft_token].item()
    return min(1.0, p_target / p_draft)


def _run_eager(args: list[object], draft_token: int, n_trials: int) -> float:
    seed_tensor = args[-1]
    accepted = 0
    for s in range(n_trials):
        seed_tensor.fill_(s)
        sampled, _ = rejection_sample(*args)
        accepted += int(sampled[0, 0].item() == draft_token)
    return accepted / n_trials


def _run_graph(
    args: list[object],
    draft_token: int,
    n_trials: int,
    mode: str,
) -> float:
    seed_tensor = args[-1]
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    with torch.npu.graph(graph, stream=stream, capture_error_mode=mode):
        sampled, _ = rejection_sample(*args)
    accepted = 0
    for s in range(n_trials):
        seed_tensor.fill_(s)
        graph.replay()
        torch.npu.synchronize()
        accepted += int(sampled[0, 0].item() == draft_token)
    return accepted / n_trials


def _check(
    name: str,
    mode: str,
    observed: float,
    expected: float,
) -> bool:
    ok = abs(observed - expected) <= TOLERANCE
    print(
        f"{name:34s} {mode:10s} observed={observed:.4f} "
        f"expected={expected:.4f} {'OK' if ok else 'FAIL'}",
        flush=True,
    )
    return ok


def main() -> None:
    print(f"torch_npu version: {getattr(torch_npu, '__version__', 'unknown')}")
    # Service-like storage: vllm-ascend sets this in worker/model_runner_v1.py.
    torch.npu.config.allow_internal_format = True
    print("allow_internal_format=True (service-like)")
    print(
        f"trials={N_TRIALS} vocab={VOCAB} temperature={TEMPERATURE} "
        f"tolerance={TOLERANCE}"
    )

    device = "npu"
    draft_token = 5
    # Logits with the draft token at a chosen probability and the rest flat.
    cases = [
        # (name, target prob at draft token, draft prob at draft token or None)
        ("probabilistic ratio 0.80", 0.20, 0.25),
        ("probabilistic ratio 0.15", 0.0375, 0.25),
        ("probabilistic ratio 1.20", 0.30, 0.25),
        ("one-hot draft", 0.20, None),
    ]

    all_ok = True
    for name, target_prob, draft_prob in cases:
        target_logits0 = torch.zeros(VOCAB, device=device)
        target_logits0[draft_token] = _logit_for_prob(target_prob)
        draft_logits0 = None
        if draft_prob is not None:
            draft_logits0 = torch.zeros(VOCAB, device=device)
            draft_logits0[draft_token] = _logit_for_prob(draft_prob)

        args, _ = _build_inputs(draft_token, target_logits0, draft_logits0)
        expected = _expected_acceptance(draft_token, target_logits0, draft_logits0)

        observed_eager = _run_eager(args, draft_token, N_TRIALS)
        all_ok &= _check(name, "eager", observed_eager, expected)

        for mode in ("global", "relaxed"):
            try:
                observed_graph = _run_graph(args, draft_token, N_TRIALS, mode)
                all_ok &= _check(name, f"graph[{mode}]", observed_graph, expected)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"{name:34s} graph[{mode:8s}] FAIL "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                all_ok = False

    print("=" * 60)
    print("RESULT:", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
