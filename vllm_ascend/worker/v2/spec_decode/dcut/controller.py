# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU cost model and prefix-preserving Domino verification decisions."""

import json
import math
import os
import statistics
import tempfile
from bisect import bisect_left
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

TABLE_VERSION = 2


@dataclass(frozen=True, order=True)
class CostKey:
    batch_size: int
    context_bucket: int
    query_tokens: int


@dataclass(frozen=True)
class Cost:
    """Steady-state speculative-step cost plus diagnostic NPU components."""

    step_ms: float
    target_ms: float
    draft_ms: float
    samples: int

    @property
    def overhead_ms(self) -> float:
        return max(0.0, self.step_ms - self.target_ms - self.draft_ms)


@dataclass(frozen=True)
class CandidateScore:
    query_tokens: int
    expected_tokens: float
    step_ms: float
    score: float
    relative_score_gain: float


@dataclass(frozen=True)
class DecisionScores:
    baseline_query_tokens: int
    baseline_expected_tokens: float
    baseline_step_ms: float
    baseline_score: float
    selected_query_tokens: int
    candidates: tuple[CandidateScore, ...]


class CostTable:
    def __init__(self, fingerprint: dict[str, Any]):
        self.fingerprint = fingerprint
        self.rows: dict[CostKey, Cost] = {}
        self.observations: dict[CostKey, list[tuple[float, float, float]]] = {}

    @classmethod
    def load(cls, path: str, fingerprint: dict[str, Any]) -> "CostTable":
        with open(path, encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict) or data.get("version") != TABLE_VERSION or data.get("fingerprint") != fingerprint:
            raise ValueError("D-Cut cost table version or model/hardware/graph configuration does not match")
        table = cls(fingerprint)
        if not isinstance(data.get("rows"), list):
            raise ValueError("D-Cut cost table rows must be a list")
        for row in data["rows"]:
            if not isinstance(row, dict):
                raise ValueError("D-Cut cost table rows must be objects")
            dimensions = [row[name] for name in ("batch_size", "context_bucket", "query_tokens", "samples")]
            if any(type(value) is not int or value <= 0 for value in dimensions):
                raise ValueError("D-Cut cost table dimensions and samples must be positive integers")
            batch, context, query, samples = dimensions
            if query < batch:
                raise ValueError("D-Cut query_tokens must include one anchor per request")
            step = float(row["step_ms"])
            target = float(row["target_ms"])
            draft = float(row["draft_ms"])
            if not math.isfinite(step) or step <= 0:
                raise ValueError("D-Cut step_ms must be finite and positive")
            if not all(math.isfinite(value) and value >= 0 for value in (target, draft)):
                raise ValueError("D-Cut diagnostic timings must be finite and nonnegative")
            key = CostKey(batch, context, query)
            if key in table.rows:
                raise ValueError("Duplicate D-Cut cost table row")
            table.rows[key] = Cost(step, target, draft, samples)
        return table

    def save(self, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "batch_size": key.batch_size,
                "context_bucket": key.context_bucket,
                "query_tokens": key.query_tokens,
                "step_ms": cost.step_ms,
                "target_ms": cost.target_ms,
                "draft_ms": cost.draft_ms,
                "overhead_ms": cost.overhead_ms,
                "samples": cost.samples,
            }
            for key, cost in sorted(self.rows.items())
        ]
        fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"version": TABLE_VERSION, "fingerprint": self.fingerprint, "rows": rows}, stream, indent=2)
                stream.write("\n")
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def observe(
        self,
        key: CostKey,
        step_ms: float,
        target_ms: float,
        draft_ms: float,
        warmup: int,
        samples: int,
    ) -> bool:
        timings = (step_ms, target_ms, draft_ms)
        if (
            key in self.rows
            or not math.isfinite(step_ms)
            or step_ms <= 0
            or not all(math.isfinite(value) and value >= 0 for value in (target_ms, draft_ms))
        ):
            return False
        observations = self.observations.setdefault(key, [])
        observations.append(timings)
        if len(observations) < warmup + samples:
            return False
        observations = observations[warmup:]
        self.rows[key] = Cost(
            statistics.median(sample[0] for sample in observations),
            statistics.median(sample[1] for sample in observations),
            statistics.median(sample[2] for sample in observations),
            len(observations),
        )
        del self.observations[key]
        return True


def context_bucket(context_length: int, buckets: tuple[int, ...]) -> int | None:
    index = bisect_left(buckets, context_length)
    return buckets[index] if index < len(buckets) else None


def query_budgets(
    limits: np.ndarray,
    ratios: tuple[float, ...],
    draft_lengths: tuple[int, ...] = (),
) -> list[int]:
    """Return paper-style total-token ratio buckets plus optional explicit depths."""
    batch = len(limits)
    full = batch + int(limits.sum())
    candidates = {max(batch, min(full, math.ceil(ratio * full))) for ratio in ratios}
    candidates.update(batch + int(np.minimum(limits, length).sum()) for length in draft_lengths)
    # Establish the full-K baseline before measuring shorter budgets.
    return [full, *sorted(candidates - {full}, reverse=True)]


def allocate_prefixes(gains: np.ndarray, limits: np.ndarray, draft_budget: int) -> np.ndarray:
    """Allocate the largest cumulative-probability gains, retaining prefixes."""
    if draft_budget < 0 or draft_budget > int(limits.sum()):
        raise ValueError("Invalid D-Cut draft budget")
    if not draft_budget:
        return np.zeros(len(limits), dtype=np.int32)
    masked = np.where(np.arange(gains.shape[1])[None, :] < limits[:, None], gains, -np.inf)
    selected = np.argsort(-masked.ravel(), kind="stable")[:draft_budget]
    return np.bincount(selected // gains.shape[1], minlength=len(limits)).astype(np.int32)


def has_viable_budget(
    limits: np.ndarray,
    table: CostTable,
    bucket: int,
    min_gain: float,
) -> bool:
    """Whether any shorter cost row can possibly clear the decision threshold."""
    return capture_gate_reason(limits, table, bucket, min_gain) is None


def capture_gate_reason(
    limits: np.ndarray,
    table: CostTable,
    bucket: int,
    min_gain: float,
) -> str | None:
    """Return why probability capture cannot help, or ``None`` when viable."""
    batch = len(limits)
    full = batch + int(limits.sum())
    baseline = table.rows.get(CostKey(batch, bucket, full))
    if baseline is None:
        return "missing_baseline_row"
    viable = any(
        key.batch_size == batch
        and key.context_bucket == bucket
        and batch <= key.query_tokens < full
        and baseline.step_ms > cost.step_ms * (1 + min_gain)
        for key, cost in table.rows.items()
    )
    return None if viable else "no_viable_budget"


def choose_caps(
    probabilities: np.ndarray,
    limits: np.ndarray,
    table: CostTable,
    bucket: int,
    min_gain: float,
) -> np.ndarray | None:
    caps, _ = _choose_caps(probabilities, limits, table, bucket, min_gain, collect_scores=False)
    return caps


def choose_caps_with_scores(
    probabilities: np.ndarray,
    limits: np.ndarray,
    table: CostTable,
    bucket: int,
    min_gain: float,
) -> tuple[np.ndarray | None, DecisionScores | None]:
    """Choose prefixes and return the score decomposition for diagnostics."""
    return _choose_caps(probabilities, limits, table, bucket, min_gain, collect_scores=True)


def _choose_caps(
    probabilities: np.ndarray,
    limits: np.ndarray,
    table: CostTable,
    bucket: int,
    min_gain: float,
    *,
    collect_scores: bool,
) -> tuple[np.ndarray | None, DecisionScores | None]:
    batch = len(limits)
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] != batch
        or np.any(limits < 0)
        or np.any(limits > probabilities.shape[1])
        or not np.all(np.isfinite(probabilities))
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        return None, None
    full = batch + int(limits.sum())
    baseline_cost = table.rows.get(CostKey(batch, bucket, full))
    if baseline_cost is None:
        return None, None
    gains = np.cumprod(probabilities.astype(np.float64), axis=1)
    mask = np.arange(gains.shape[1])[None, :] < limits[:, None]
    baseline_expected = float(batch + gains[mask].sum())
    baseline_score = baseline_expected / baseline_cost.step_ms
    best_score = baseline_score
    best_caps = limits.copy()
    selected_query = full
    candidate_scores = []
    for key, cost in sorted(table.rows.items()):
        if key.batch_size != batch or key.context_bucket != bucket or not batch <= key.query_tokens < full:
            continue
        caps = allocate_prefixes(gains, limits, key.query_tokens - batch)
        kept = np.arange(gains.shape[1])[None, :] < caps[:, None]
        expected = float(batch + gains[kept].sum())
        score = expected / cost.step_ms
        if collect_scores:
            candidate_scores.append(
                CandidateScore(
                    query_tokens=key.query_tokens,
                    expected_tokens=expected,
                    step_ms=cost.step_ms,
                    score=score,
                    relative_score_gain=score / baseline_score - 1,
                )
            )
        if score > best_score and score > baseline_score * (1 + min_gain):
            best_score, best_caps, selected_query = score, caps, key.query_tokens
    diagnostics = None
    if collect_scores:
        diagnostics = DecisionScores(
            baseline_query_tokens=full,
            baseline_expected_tokens=baseline_expected,
            baseline_step_ms=baseline_cost.step_ms,
            baseline_score=baseline_score,
            selected_query_tokens=selected_query,
            candidates=tuple(candidate_scores),
        )
    return best_caps, diagnostics


def truncate_scheduler_output(scheduler_output: Any, req_ids: list[str], caps: np.ndarray) -> Any:
    if len(req_ids) != len(caps) or len(set(req_ids)) != len(req_ids):
        raise ValueError("D-Cut caps must match unique request IDs")
    scheduled = scheduler_output.num_scheduled_tokens.copy()
    drafts = scheduler_output.scheduled_spec_decode_tokens.copy()
    removed = 0
    for req_id, cap in zip(req_ids, caps):
        old = drafts.get(req_id, [])
        if int(cap) != cap or not 0 <= cap <= len(old):
            raise ValueError("D-Cut cap must select an existing draft prefix")
        delta = len(old) - int(cap)
        if scheduled[req_id] - delta < 1:
            raise ValueError("D-Cut cannot remove the target anchor")
        if delta:
            drafts[req_id] = old[: int(cap)]
            scheduled[req_id] -= delta
            removed += delta
    if not removed:
        return scheduler_output
    return replace(
        scheduler_output,
        num_scheduled_tokens=scheduled,
        scheduled_spec_decode_tokens=drafts,
        total_num_scheduled_tokens=scheduler_output.total_num_scheduled_tokens - removed,
    )


def decode_batch_info(scheduler_output: Any, req_states: Any, buckets: tuple[int, ...]) -> tuple[list[str], int] | None:
    """Require an established, unstructured batch with one target anchor each."""
    info, _ = diagnose_batch_info(scheduler_output, req_states, buckets)
    return info


def diagnose_batch_info(
    scheduler_output: Any,
    req_states: Any,
    buckets: tuple[int, ...],
) -> tuple[tuple[list[str], int] | None, str | None]:
    """Decode a safe batch and return one precise rejection reason on failure."""
    req_ids = sorted(scheduler_output.num_scheduled_tokens)
    if not req_ids:
        return None, "empty_batch"
    if scheduler_output.scheduled_new_reqs:
        return None, "new_requests"
    if scheduler_output.scheduled_encoder_inputs:
        return None, "encoder_inputs"
    if scheduler_output.has_structured_output_requests:
        return None, "structured_output"
    if scheduler_output.preempted_req_ids:
        return None, "preempted_requests"
    if scheduler_output.scheduled_cached_reqs.resumed_req_ids:
        return None, "resumed_requests"
    if sum(scheduler_output.num_scheduled_tokens.values()) != scheduler_output.total_num_scheduled_tokens:
        return None, "scheduled_token_mismatch"
    computed = dict(
        zip(scheduler_output.scheduled_cached_reqs.req_ids, scheduler_output.scheduled_cached_reqs.num_computed_tokens)
    )
    for req_id in req_ids:
        index = req_states.req_id_to_index.get(req_id)
        if index is None:
            return None, "unknown_request"
        if req_id not in computed:
            return None, "missing_computed_tokens"
        if computed[req_id] < req_states.prefill_len.np[index]:
            return None, "prefill_incomplete"
        if (
            scheduler_output.num_scheduled_tokens[req_id]
            - len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
            != 1
        ):
            return None, "non_anchor_schedule"
    bucket = context_bucket(max(computed[req_id] for req_id in req_ids), buckets)
    if bucket is None:
        return None, "context_out_of_range"
    return (req_ids, bucket), None
