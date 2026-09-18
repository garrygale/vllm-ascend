# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2 Domino probability transfer and optional live cost-table calibration."""

import json
import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .controller import (
    CostKey,
    CostTable,
    allocate_prefixes,
    capture_gate_reason,
    choose_caps,
    choose_caps_with_scores,
    diagnose_batch_info,
    query_budgets,
)

# Reuse the configured vLLM namespace without initializing an engine in CPU tests.
logger = logging.getLogger("vllm")


def selected_greedy_probability(logits: torch.Tensor, selected_ids: torch.Tensor) -> torch.Tensor:
    """Probability of already selected draft-vocabulary IDs."""
    if selected_ids.shape != logits.shape[:-1]:
        raise ValueError("selected_ids must match the logits batch dimensions")
    logits_fp32 = logits.float()
    selected_logits = logits_fp32.gather(-1, selected_ids.unsqueeze(-1)).squeeze(-1)
    return (selected_logits - torch.logsumexp(logits_fp32, dim=-1)).exp()


def model_fingerprint(vllm_config: Any, device_name: str) -> dict[str, Any]:
    spec = vllm_config.speculative_config
    model = vllm_config.model_config
    draft = spec.draft_model_config
    return {
        "device": device_name,
        "model": model.model,
        "draft_model": draft.model,
        "model_revision": getattr(model, "revision", None),
        "draft_revision": getattr(draft, "revision", None),
        "dtype": str(model.dtype),
        "target_quantization": getattr(model, "quantization", None),
        "target_config": model.hf_config.to_dict(),
        "draft_config": draft.hf_config.to_dict(),
        "cache_dtype": vllm_config.cache_config.cache_dtype,
        "cache_block_size": vllm_config.cache_config.block_size,
        "attention_backend": str(getattr(vllm_config.attention_config, "backend", None)),
        "async_scheduling": vllm_config.scheduler_config.async_scheduling,
        "tp_size": vllm_config.parallel_config.tensor_parallel_size,
        "block_size": spec.num_speculative_tokens,
        "max_model_len": model.max_model_len,
        "max_num_seqs": vllm_config.scheduler_config.max_num_seqs,
        "max_num_batched_tokens": vllm_config.scheduler_config.max_num_batched_tokens,
        "capture_sizes": list(vllm_config.compilation_config.cudagraph_capture_sizes or []),
        "target_graph_mode": "PIECEWISE",
        "draft_graph_mode": "NONE",
        "cost_metric": "steady_state_end_to_end_step_ms_v2",
    }


class DcutRuntime:
    def __init__(
        self,
        config: Any,
        fingerprint: dict[str, Any],
        max_num_reqs: int,
        block_size: int,
        device: torch.device,
        tp_group: Any,
        npu: Any,
    ):
        self.config = config
        self.block_size = block_size
        self.tp_group = tp_group
        self.npu = npu
        self.is_root = tp_group.rank_in_group == 0
        self.device = device
        self.table: CostTable | None = None
        self.copy_stream: Any = None
        self.copy_event: Any = None
        self.host_probs: torch.Tensor | None = None
        self.snapshot_req_ids: list[str] | None = None
        self.target_start: Any = None
        self.draft_start: Any = None
        self.draft_end: Any = None
        self.measurement_key: CostKey | None = None
        self.measurement_started_at: float | None = None
        self.current_key: CostKey | None = None
        self.current_started_at: float | None = None
        self.capture_requested = False
        self.collecting = False
        self.score_diagnostics: dict[str, Any] = {}
        self.current_performance_step_id: int | None = None
        self._next_performance_step_id = 0
        self._pending_performance_steps: dict[int, dict[str, Any]] = {}
        self._performance_totals: dict[str, float | int] = {
            "steps": 0,
            "trimmed_steps": 0,
            "full_k_steps": 0,
            "output_tokens": 0,
            "elapsed_ms": 0.0,
            "request_step_elapsed_ms": 0.0,
            "output_request_steps": 0,
            "selected_cost_table_ms": 0.0,
            "full_k_cost_table_ms": 0.0,
            "selected_request_cost_table_ms": 0.0,
            "full_k_request_cost_table_ms": 0.0,
            "selected_expected_tokens": 0.0,
            "full_k_expected_tokens": 0.0,
        }
        self.stats: dict[str, Any] = {
            "decisions": 0,
            "trimmed_tokens": 0,
            "fallbacks": 0,
            "capture_skips": 0,
            "full_k_selected": 0,
            "profiled_rows": 0,
            "fallback_reasons": {},
            "fallback_shapes": {},
            "fallback_shape_overflow": 0,
            "score_diagnostic_shape_overflow": 0,
        }
        error = None
        if self.is_root:
            try:
                fingerprint = json.loads(json.dumps(fingerprint))
                fingerprint["context_buckets"] = list(config.context_buckets)
                path = Path(config.cost_table_path)
                if path.exists():
                    try:
                        self.table = CostTable.load(str(path), fingerprint)
                    except (ValueError, KeyError, TypeError) as exc:
                        if not config.generate_cost_table:
                            raise
                        logger.warning(
                            "D-Cut cost table is stale; rebuilding it for the current configuration: %s", exc
                        )
                        self.table = CostTable(fingerprint)
                        self.table.save(str(path))
                elif config.generate_cost_table:
                    self.table = CostTable(fingerprint)
                    self.table.save(str(path))
                else:
                    raise ValueError(f"D-Cut cost table does not exist: {path}")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                error = str(exc)
        error = self.tp_group.broadcast_object(error, src=0)
        if error is not None:
            raise ValueError(error)
        if self.is_root:
            self.copy_stream = npu.Stream()
            self.copy_event = npu.Event()
            self.host_probs = torch.empty(
                (max_num_reqs, block_size), dtype=torch.float32, device="cpu", pin_memory=device.type != "cpu"
            )
            logger.info(
                "D-Cut enabled: Domino fixed K=%d, Target PIECEWISE, generate_cost_table=%s, table=%s",
                block_size,
                config.generate_cost_table,
                config.cost_table_path,
            )

    def _consume_measurement(self) -> None:
        if self.measurement_key is None or self.measurement_started_at is None or self.draft_end is None:
            return
        if not self.draft_end.query():
            # Synchronize only while generating the table. Wall time then includes
            # any device work that was still on the serving critical path.
            self.draft_end.synchronize()
        if self.copy_event is not None and not self.copy_event.query():
            # Attribute the selected-probability transfer and its required wait to
            # the step that produced it instead of shifting that cost to the next row.
            self.copy_event.synchronize()
        step_ms = (time.perf_counter() - self.measurement_started_at) * 1000
        target_ms = self.target_start.elapsed_time(self.draft_start)
        draft_ms = self.draft_start.elapsed_time(self.draft_end)
        assert self.table is not None
        try:
            completed = self.table.observe(
                self.measurement_key,
                step_ms,
                target_ms,
                draft_ms,
                self.config.profile_warmup,
                self.config.profile_samples,
            )
            if completed:
                self.table.save(self.config.cost_table_path)
                self.stats["profiled_rows"] += 1
                logger.info("D-Cut calibrated end-to-end cost row %s", self.measurement_key)
        except OSError as exc:
            logger.warning("D-Cut could not save cost table: %s", exc)
        finally:
            self.measurement_key = None
            self.measurement_started_at = None
            self.target_start = self.draft_start = self.draft_end = None

    def finish_calibration(self) -> dict[str, Any]:
        """Flush the final live sample via LLM.collective_rpc after generation."""
        if self.is_root and self.config.generate_cost_table:
            self._consume_measurement()
        stats = deepcopy(self.stats)
        if self.config.score_diagnostics:
            stats["score_diagnostics"] = self._summarize_score_diagnostics()
        if self.config.performance_diagnostics and self.is_root:
            stats["nonfallback_performance"] = self._summarize_performance_diagnostics()
        return stats

    def _start_performance_measurement(
        self,
        req_ids: list[str],
        caps: np.ndarray | None,
        bucket: int | None,
        score_payload: dict[str, Any] | None,
        started_at: float | None,
    ) -> None:
        self.current_performance_step_id = None
        if (
            not self.config.performance_diagnostics
            or not self.is_root
            or self.config.generate_cost_table
            or caps is None
            or bucket is None
            or started_at is None
        ):
            return
        batch_size = len(req_ids)
        selected_query = batch_size + int(caps.sum())
        if score_payload is None:
            return
        baseline_query = int(score_payload["baseline_query_tokens"])
        baseline_cost = float(score_payload["baseline_step_ms"])
        baseline_expected = float(score_payload["baseline_expected_tokens"])
        if selected_query == baseline_query:
            selected_cost = baseline_cost
            selected_expected = baseline_expected
        else:
            selected = next(
                (
                    candidate
                    for candidate in score_payload["candidates"]
                    if int(candidate["query_tokens"]) == selected_query
                ),
                None,
            )
            if selected is None:
                return
            selected_cost = float(selected["step_ms"])
            selected_expected = float(selected["expected_tokens"])
        step_id = self._next_performance_step_id
        self._next_performance_step_id += 1
        self._pending_performance_steps[step_id] = {
            "started_at": started_at,
            "trimmed": selected_query < baseline_query,
            "selected_cost_table_ms": selected_cost,
            "full_k_cost_table_ms": baseline_cost,
            "selected_request_cost_table_ms": selected_cost * batch_size,
            "full_k_request_cost_table_ms": baseline_cost * batch_size,
            "selected_expected_tokens": selected_expected,
            "full_k_expected_tokens": baseline_expected,
        }
        self.current_performance_step_id = step_id

    def record_step_output(self, step_id: int | None, sampled_token_ids: list[list[int]]) -> None:
        """Complete one non-fallback measurement after async output parsing."""
        if step_id is None or not self.is_root:
            return
        measurement = self._pending_performance_steps.pop(step_id, None)
        if measurement is None:
            return
        output_tokens = sum(len(tokens) for tokens in sampled_token_ids)
        output_requests = sum(bool(tokens) for tokens in sampled_token_ids)
        elapsed_ms = (time.perf_counter() - measurement["started_at"]) * 1000
        totals = self._performance_totals
        totals["steps"] += 1
        totals["trimmed_steps" if measurement["trimmed"] else "full_k_steps"] += 1
        totals["output_tokens"] += output_tokens
        totals["elapsed_ms"] += elapsed_ms
        totals["request_step_elapsed_ms"] += elapsed_ms * output_requests
        totals["output_request_steps"] += output_requests
        for name in (
            "selected_cost_table_ms",
            "full_k_cost_table_ms",
            "selected_request_cost_table_ms",
            "full_k_request_cost_table_ms",
            "selected_expected_tokens",
            "full_k_expected_tokens",
        ):
            totals[name] += measurement[name]

    def discard_step_output(self, step_id: int | None) -> None:
        if step_id is not None:
            self._pending_performance_steps.pop(step_id, None)

    def _summarize_performance_diagnostics(self) -> dict[str, Any]:
        totals = deepcopy(self._performance_totals)
        steps = int(totals["steps"])
        output_tokens = int(totals["output_tokens"])
        elapsed_ms = float(totals["elapsed_ms"])
        request_step_elapsed_ms = float(totals["request_step_elapsed_ms"])
        selected_cost = float(totals["selected_cost_table_ms"])
        full_cost = float(totals["full_k_cost_table_ms"])
        selected_request_cost = float(totals["selected_request_cost_table_ms"])
        full_request_cost = float(totals["full_k_request_cost_table_ms"])
        selected_expected = float(totals["selected_expected_tokens"])
        full_expected = float(totals["full_k_expected_tokens"])
        actual_tps = output_tokens * 1000 / elapsed_ms if elapsed_ms > 0 else 0.0
        aggregate_time_per_token = elapsed_ms / output_tokens if output_tokens > 0 else 0.0
        actual_tpot = request_step_elapsed_ms / output_tokens if output_tokens > 0 else 0.0
        selected_expected_tps = selected_expected * 1000 / selected_cost if selected_cost > 0 else 0.0
        full_expected_tps = full_expected * 1000 / full_cost if full_cost > 0 else 0.0
        selected_expected_tpot = selected_request_cost / selected_expected if selected_expected > 0 else 0.0
        full_expected_tpot = full_request_cost / full_expected if full_expected > 0 else 0.0
        nonfallback_calls = self.stats["decisions"] + self.stats["full_k_selected"]
        selection_calls = nonfallback_calls + self.stats["fallbacks"]
        return {
            "steps": steps,
            "pending_steps": len(self._pending_performance_steps),
            "trimmed_steps": int(totals["trimmed_steps"]),
            "full_k_steps": int(totals["full_k_steps"]),
            "coverage": nonfallback_calls / selection_calls if selection_calls else 0.0,
            "output_tokens": output_tokens,
            "output_request_steps": int(totals["output_request_steps"]),
            "elapsed_ms": elapsed_ms,
            "request_step_elapsed_ms": request_step_elapsed_ms,
            "output_tokens_per_second": actual_tps,
            "tpot_ms_per_token": actual_tpot,
            "aggregate_time_ms_per_output_token": aggregate_time_per_token,
            "cost_table": {
                "selected_elapsed_ms": selected_cost,
                "full_k_elapsed_ms": full_cost,
                "cost_only_latency_reduction": 1 - selected_cost / full_cost if full_cost > 0 else 0.0,
            },
            "expected": {
                "selected_tokens": selected_expected,
                "full_k_tokens": full_expected,
                "selected_tokens_per_second": selected_expected_tps,
                "full_k_tokens_per_second": full_expected_tps,
                "selected_tpot_ms_per_token": selected_expected_tpot,
                "full_k_tpot_ms_per_token": full_expected_tpot,
                "throughput_change_vs_full_k": (
                    selected_expected_tps / full_expected_tps - 1 if full_expected_tps > 0 else 0.0
                ),
                "tpot_reduction_vs_full_k": (
                    1 - selected_expected_tpot / full_expected_tpot if full_expected_tpot > 0 else 0.0
                ),
            },
        }

    def _take_probabilities(self, req_ids: list[str]) -> tuple[np.ndarray | None, str | None]:
        snapshot = self.snapshot_req_ids
        self.snapshot_req_ids = None
        if snapshot is None:
            return None, "missing_probability_snapshot"
        if len(snapshot) != len(req_ids) or set(snapshot) != set(req_ids):
            return None, "request_id_mismatch"
        if not self.copy_event.query():
            if self.config.generate_cost_table or self.config.wait_for_probs:
                self.copy_event.synchronize()
            else:
                return None, "probabilities_not_ready"
        assert self.host_probs is not None
        rows = {req_id: row for row, req_id in enumerate(snapshot)}
        probabilities = self.host_probs.numpy()[[rows[req_id] for req_id in req_ids]].copy()
        return probabilities, None

    def _record_fallback(self, reason: str, shape: tuple[int, int, int, int, int] | None) -> None:
        self.stats["fallbacks"] += 1
        reasons = self.stats["fallback_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1
        if shape is None:
            return
        batch, bucket, query, min_k, max_k = shape
        key = f"{reason}|batch={batch},context={bucket},query={query},min_k={min_k},max_k={max_k}"
        shapes = self.stats["fallback_shapes"]
        if key in shapes or len(shapes) < 64:
            shapes[key] = shapes.get(key, 0) + 1
        else:
            self.stats["fallback_shape_overflow"] += 1

    def _record_score_diagnostics(self, payload: dict[str, Any] | None) -> None:
        if payload is None:
            return
        shape = (
            f"batch={payload['batch_size']},context={payload['context_bucket']},"
            f"query={payload['baseline_query_tokens']}"
        )
        if shape not in self.score_diagnostics and len(self.score_diagnostics) >= 64:
            self.stats["score_diagnostic_shape_overflow"] += 1
            return
        entry = self.score_diagnostics.setdefault(
            shape,
            {
                "samples": 0,
                "min_gain": self.config.min_gain,
                "batch_size": payload["batch_size"],
                "baseline_query_tokens": payload["baseline_query_tokens"],
                "baseline_step_ms": payload["baseline_step_ms"],
                "baseline_expected_tokens_sum": 0.0,
                "baseline_score_sum": 0.0,
                "selected_query_counts": {},
                "best_short_query_counts": {},
                "best_short_score_gain_sum": 0.0,
                "best_short_score_gain_min": float("inf"),
                "best_short_score_gain_max": -float("inf"),
                "candidates": {},
            },
        )
        entry["samples"] += 1
        entry["baseline_expected_tokens_sum"] += payload["baseline_expected_tokens"]
        entry["baseline_score_sum"] += payload["baseline_score"]
        selected = str(payload["selected_query_tokens"])
        entry["selected_query_counts"][selected] = entry["selected_query_counts"].get(selected, 0) + 1
        candidates = payload["candidates"]
        if candidates:
            best_short = max(candidates, key=lambda candidate: candidate["score"])
            best_query = str(best_short["query_tokens"])
            entry["best_short_query_counts"][best_query] = entry["best_short_query_counts"].get(best_query, 0) + 1
            best_gain = best_short["relative_score_gain"]
            entry["best_short_score_gain_sum"] += best_gain
            entry["best_short_score_gain_min"] = min(entry["best_short_score_gain_min"], best_gain)
            entry["best_short_score_gain_max"] = max(entry["best_short_score_gain_max"], best_gain)
        baseline_expected = payload["baseline_expected_tokens"]
        baseline_step = payload["baseline_step_ms"]
        for candidate in candidates:
            query = str(candidate["query_tokens"])
            score_gain = candidate["relative_score_gain"]
            candidate_entry = entry["candidates"].setdefault(
                query,
                {
                    "samples": 0,
                    "query_tokens": candidate["query_tokens"],
                    "step_ms": candidate["step_ms"],
                    "cost_saving": 1 - candidate["step_ms"] / baseline_step,
                    "expected_tokens_sum": 0.0,
                    "expected_token_ratio_sum": 0.0,
                    "score_sum": 0.0,
                    "score_gain_sum": 0.0,
                    "score_gain_min": float("inf"),
                    "score_gain_max": -float("inf"),
                    "beats_full_k": 0,
                    "clears_min_gain": 0,
                    "selected": 0,
                },
            )
            candidate_entry["samples"] += 1
            candidate_entry["expected_tokens_sum"] += candidate["expected_tokens"]
            candidate_entry["expected_token_ratio_sum"] += candidate["expected_tokens"] / baseline_expected
            candidate_entry["score_sum"] += candidate["score"]
            candidate_entry["score_gain_sum"] += score_gain
            candidate_entry["score_gain_min"] = min(candidate_entry["score_gain_min"], score_gain)
            candidate_entry["score_gain_max"] = max(candidate_entry["score_gain_max"], score_gain)
            candidate_entry["beats_full_k"] += int(score_gain > 0)
            candidate_entry["clears_min_gain"] += int(score_gain > self.config.min_gain)
            candidate_entry["selected"] += int(payload["selected_query_tokens"] == candidate["query_tokens"])

    def _summarize_score_diagnostics(self) -> dict[str, Any]:
        result = {}
        for shape, entry in self.score_diagnostics.items():
            samples = entry["samples"]
            candidates = {}
            for query, candidate in entry["candidates"].items():
                count = candidate["samples"]
                expected_ratio = candidate["expected_token_ratio_sum"] / count
                candidates[query] = {
                    "samples": count,
                    "query_tokens": candidate["query_tokens"],
                    "avg_draft_tokens_per_request": (
                        candidate["query_tokens"] - entry["batch_size"]
                    )
                    / entry["batch_size"],
                    "step_ms": candidate["step_ms"],
                    "cost_saving": candidate["cost_saving"],
                    "avg_expected_tokens": candidate["expected_tokens_sum"] / count,
                    "avg_expected_token_ratio": expected_ratio,
                    "avg_expected_token_loss": 1 - expected_ratio,
                    # Positive means probability-weighted token loss is larger
                    # than latency savings, so this short K cannot beat full K.
                    "avg_loss_minus_cost_saving": (1 - expected_ratio) - candidate["cost_saving"],
                    "avg_score": candidate["score_sum"] / count,
                    "avg_score_gain": candidate["score_gain_sum"] / count,
                    "min_score_gain": candidate["score_gain_min"],
                    "max_score_gain": candidate["score_gain_max"],
                    "beats_full_k": candidate["beats_full_k"],
                    "clears_min_gain": candidate["clears_min_gain"],
                    "selected": candidate["selected"],
                }
            result[shape] = {
                "samples": samples,
                "min_gain": entry["min_gain"],
                "baseline": {
                    "query_tokens": entry["baseline_query_tokens"],
                    "avg_draft_tokens_per_request": (
                        entry["baseline_query_tokens"] - entry["batch_size"]
                    )
                    / entry["batch_size"],
                    "step_ms": entry["baseline_step_ms"],
                    "avg_expected_tokens": entry["baseline_expected_tokens_sum"] / samples,
                    "avg_score": entry["baseline_score_sum"] / samples,
                },
                "selected_query_counts": entry["selected_query_counts"],
                "best_short_query_counts": entry["best_short_query_counts"],
                "avg_best_short_score_gain": entry["best_short_score_gain_sum"] / samples,
                "min_best_short_score_gain": entry["best_short_score_gain_min"],
                "max_best_short_score_gain": entry["best_short_score_gain_max"],
                "candidates": candidates,
            }
        return result

    def select_caps(self, scheduler_output: Any, req_states: Any) -> tuple[list[str], np.ndarray | None]:
        """Rank zero decides, including readiness/fallback, before graph dispatch."""
        self.current_key = None
        self.current_started_at = None
        self.current_performance_step_id = None
        decision: Any = None
        step_started_at = None
        performance_payload = None
        if self.is_root:
            self._consume_measurement()
            step_started_at = time.perf_counter()
            info, batch_reason = diagnose_batch_info(scheduler_output, req_states, self.config.context_buckets)
            if info is None:
                self.snapshot_req_ids = None
                decision = ([], None, None, False, batch_reason, None, None)
            else:
                req_ids, bucket = info
                limits = np.array(
                    [len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, [])) for req_id in req_ids],
                    dtype=np.int32,
                )
                caps = None
                eligible = bool(np.all(limits <= self.block_size))
                full_query = len(req_ids) + int(limits.sum())
                shape = (len(req_ids), bucket, full_query, int(limits.min()), int(limits.max()))
                fallback_reason = None
                score_payload = None
                assert self.table is not None
                if not eligible:
                    capture = False
                    fallback_reason = "ineligible_limits"
                    self.snapshot_req_ids = None
                elif self.config.generate_cost_table:
                    capture = True
                    # Consume the preceding copy so its wait remains part of the
                    # calibrated end-to-end step, even though calibration caps do
                    # not depend on confidence.
                    self._take_probabilities(req_ids)
                    for budget in query_budgets(
                        limits,
                        self.config.candidate_ratios,
                        self.config.candidate_draft_lengths,
                    ):
                        if CostKey(len(req_ids), bucket, budget) not in self.table.rows:
                            # Balanced prefixes measure shapes without depending on confidence.
                            gains = np.broadcast_to(
                                -np.arange(self.block_size, dtype=np.float64), (len(req_ids), self.block_size)
                            )
                            caps = allocate_prefixes(gains, limits, budget - len(req_ids))
                            break
                    if caps is None:
                        fallback_reason = "calibration_complete"
                else:
                    fallback_reason = capture_gate_reason(limits, self.table, bucket, self.config.min_gain)
                    capture = fallback_reason is None
                    if capture:
                        probabilities, probability_reason = self._take_probabilities(req_ids)
                        if probabilities is None:
                            fallback_reason = probability_reason
                        else:
                            if self.config.score_diagnostics or self.config.performance_diagnostics:
                                caps, scores = choose_caps_with_scores(
                                    probabilities,
                                    limits,
                                    self.table,
                                    bucket,
                                    self.config.min_gain,
                                )
                                if scores is not None:
                                    score_payload = {
                                        "batch_size": len(req_ids),
                                        "context_bucket": bucket,
                                        "baseline_query_tokens": scores.baseline_query_tokens,
                                        "baseline_expected_tokens": scores.baseline_expected_tokens,
                                        "baseline_step_ms": scores.baseline_step_ms,
                                        "baseline_score": scores.baseline_score,
                                        "selected_query_tokens": scores.selected_query_tokens,
                                        "candidates": [
                                            {
                                                "query_tokens": candidate.query_tokens,
                                                "expected_tokens": candidate.expected_tokens,
                                                "step_ms": candidate.step_ms,
                                                "score": candidate.score,
                                                "relative_score_gain": candidate.relative_score_gain,
                                            }
                                            for candidate in scores.candidates
                                        ],
                                    }
                            else:
                                caps = choose_caps(
                                    probabilities,
                                    limits,
                                    self.table,
                                    bucket,
                                    self.config.min_gain,
                                )
                            if caps is None:
                                fallback_reason = "invalid_probabilities"
                    else:
                        # A previous shape may have requested a copy. The current
                        # cost bound proves it cannot help, so discard it without waiting.
                        self.snapshot_req_ids = None
                performance_payload = score_payload
                decision = (
                    req_ids,
                    caps.tolist() if caps is not None else None,
                    bucket,
                    capture,
                    fallback_reason,
                    shape,
                    score_payload if self.config.score_diagnostics else None,
                )
        req_ids, caps_list, bucket, capture, fallback_reason, shape, broadcast_score_payload = (
            self.tp_group.broadcast_object(decision, src=0)
        )
        if self.config.score_diagnostics:
            self._record_score_diagnostics(broadcast_score_payload)
        self.capture_requested = bool(capture)
        caps = np.array(caps_list, dtype=np.int32) if caps_list is not None else None
        if caps is None:
            self._record_fallback(fallback_reason or "unknown", shape)
        else:
            removed = sum(
                len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, [])) - int(cap)
                for req_id, cap in zip(req_ids, caps)
            )
            if removed:
                self.stats["decisions"] += 1
                self.stats["trimmed_tokens"] += removed
            else:
                self.stats["full_k_selected"] += 1
        self._start_performance_measurement(req_ids, caps, bucket, performance_payload, step_started_at)
        if fallback_reason in {"missing_baseline_row", "no_viable_budget"}:
            self.stats["capture_skips"] += 1
        if self.is_root and self.config.generate_cost_table and bucket is not None:
            total = len(req_ids) + (
                int(caps.sum())
                if caps is not None
                else sum(len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, [])) for req_id in req_ids)
            )
            key = CostKey(len(req_ids), bucket, total)
            assert self.table is not None
            if key not in self.table.rows:
                self.current_key = key
                self.current_started_at = step_started_at
        return req_ids, caps

    def begin_target(self) -> None:
        if self.current_key is not None:
            if self.current_started_at is None:
                raise RuntimeError("D-Cut cost measurement has no step start")
            self.target_start = self.npu.Event(enable_timing=True)
            self.target_start.record()

    def abort_target(self) -> None:
        if self.current_performance_step_id is not None:
            self._pending_performance_steps.pop(self.current_performance_step_id, None)
            self.current_performance_step_id = None
        self.current_key = self.measurement_key = None
        self.current_started_at = self.measurement_started_at = None
        self.target_start = self.draft_start = self.draft_end = None
        self.snapshot_req_ids = None
        self.capture_requested = False
        self.collecting = False

    def begin_proposal(self, dummy_run: bool, is_profile: bool) -> bool:
        should_capture = self.config.generate_cost_table or self.capture_requested
        self.collecting = self.is_root and should_capture and not dummy_run and not is_profile
        if self.collecting and self.current_key is not None:
            self.draft_start = self.npu.Event(enable_timing=True)
            self.draft_start.record()
        return self.collecting

    def end_proposal(self, input_batch: Any, probabilities: torch.Tensor) -> None:
        if not self.collecting:
            return
        self.collecting = False
        if self.current_key is not None:
            self.draft_end = self.npu.Event(enable_timing=True)
            self.draft_end.record()
            self.measurement_key = self.current_key
            self.measurement_started_at = self.current_started_at
            self.current_key = None
            self.current_started_at = None
        # A staging clone avoids a race with the next proposal overwriting its buffer.
        staging = probabilities[: input_batch.num_reqs].clone()
        self.copy_stream.wait_stream(self.npu.current_stream())
        assert self.host_probs is not None
        with self.npu.stream(self.copy_stream):
            self.host_probs[: input_batch.num_reqs].copy_(staging, non_blocking=True)
            if staging.device.type != "cpu":
                staging.record_stream(self.copy_stream)
            self.copy_event.record()
        self.snapshot_req_ids = list(input_batch.req_ids)
