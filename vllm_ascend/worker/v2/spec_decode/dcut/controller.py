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

TABLE_VERSION = 1


@dataclass(frozen=True, order=True)
class CostKey:
    batch_size: int
    context_bucket: int
    query_tokens: int


@dataclass(frozen=True)
class Cost:
    target_ms: float
    draft_ms: float
    samples: int

    @property
    def total_ms(self) -> float:
        return self.target_ms + self.draft_ms


class CostTable:
    def __init__(self, fingerprint: dict[str, Any]):
        self.fingerprint = fingerprint
        self.rows: dict[CostKey, Cost] = {}
        self.observations: dict[CostKey, list[tuple[float, float]]] = {}

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
            target, draft = float(row["target_ms"]), float(row["draft_ms"])
            if not all(math.isfinite(value) and value > 0 for value in (target, draft)):
                raise ValueError("D-Cut cost timings must be finite and positive")
            key = CostKey(batch, context, query)
            if key in table.rows:
                raise ValueError("Duplicate D-Cut cost table row")
            table.rows[key] = Cost(target, draft, samples)
        return table

    def save(self, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "batch_size": key.batch_size,
                "context_bucket": key.context_bucket,
                "query_tokens": key.query_tokens,
                "target_ms": cost.target_ms,
                "draft_ms": cost.draft_ms,
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

    def observe(self, key: CostKey, target_ms: float, draft_ms: float, warmup: int, samples: int) -> bool:
        if key in self.rows or not all(math.isfinite(value) and value > 0 for value in (target_ms, draft_ms)):
            return False
        observations = self.observations.setdefault(key, [])
        observations.append((target_ms, draft_ms))
        if len(observations) < warmup + samples:
            return False
        observations = observations[warmup:]
        self.rows[key] = Cost(
            statistics.median(pair[0] for pair in observations),
            statistics.median(pair[1] for pair in observations),
            len(observations),
        )
        del self.observations[key]
        return True


def context_bucket(context_length: int, buckets: tuple[int, ...]) -> int | None:
    index = bisect_left(buckets, context_length)
    return buckets[index] if index < len(buckets) else None


def query_budgets(limits: np.ndarray, draft_lengths: tuple[int, ...]) -> list[int]:
    batch = len(limits)
    full = batch + int(limits.sum())
    candidates = {batch + int(np.minimum(limits, length).sum()) for length in draft_lengths}
    # Establish the full-K baseline before measuring shorter prefixes.
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


def choose_caps(
    probabilities: np.ndarray,
    limits: np.ndarray,
    table: CostTable,
    bucket: int,
    min_gain: float,
) -> np.ndarray | None:
    batch = len(limits)
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] != batch
        or np.any(limits < 0)
        or np.any(limits > probabilities.shape[1])
        or not np.all(np.isfinite(probabilities))
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        return None
    full = batch + int(limits.sum())
    baseline_cost = table.rows.get(CostKey(batch, bucket, full))
    if baseline_cost is None:
        return None
    gains = np.cumprod(probabilities.astype(np.float64), axis=1)
    mask = np.arange(gains.shape[1])[None, :] < limits[:, None]
    baseline_score = (batch + gains[mask].sum()) / baseline_cost.total_ms
    best_score = baseline_score
    best_caps = limits.copy()
    for key, cost in sorted(table.rows.items()):
        if key.batch_size != batch or key.context_bucket != bucket or not batch <= key.query_tokens < full:
            continue
        caps = allocate_prefixes(gains, limits, key.query_tokens - batch)
        kept = np.arange(gains.shape[1])[None, :] < caps[:, None]
        score = (batch + gains[kept].sum()) / cost.total_ms
        if score > best_score and score > baseline_score * (1 + min_gain):
            best_score, best_caps = score, caps
    return best_caps


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
    req_ids = sorted(scheduler_output.num_scheduled_tokens)
    if (
        not req_ids
        or scheduler_output.scheduled_new_reqs
        or scheduler_output.scheduled_encoder_inputs
        or scheduler_output.has_structured_output_requests
        or scheduler_output.preempted_req_ids
        or scheduler_output.scheduled_cached_reqs.resumed_req_ids
        or sum(scheduler_output.num_scheduled_tokens.values()) != scheduler_output.total_num_scheduled_tokens
    ):
        return None
    computed = dict(
        zip(scheduler_output.scheduled_cached_reqs.req_ids, scheduler_output.scheduled_cached_reqs.num_computed_tokens)
    )
    for req_id in req_ids:
        index = req_states.req_id_to_index.get(req_id)
        if index is None or req_id not in computed or computed[req_id] < req_states.prefill_len.np[index]:
            return None
        if (
            scheduler_output.num_scheduled_tokens[req_id]
            - len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
            != 1
        ):
            return None
    bucket = context_bucket(max(computed[req_id] for req_id in req_ids), buckets)
    return (req_ids, bucket) if bucket is not None else None
