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
        return deepcopy(self.stats)

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

    def select_caps(self, scheduler_output: Any, req_states: Any) -> tuple[list[str], np.ndarray | None]:
        """Rank zero decides, including readiness/fallback, before graph dispatch."""
        self.current_key = None
        self.current_started_at = None
        decision: Any = None
        step_started_at = None
        if self.is_root:
            self._consume_measurement()
            step_started_at = time.perf_counter()
            info, batch_reason = diagnose_batch_info(scheduler_output, req_states, self.config.context_buckets)
            if info is None:
                self.snapshot_req_ids = None
                decision = ([], None, None, False, batch_reason, None)
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
                            caps = choose_caps(probabilities, limits, self.table, bucket, self.config.min_gain)
                            if caps is None:
                                fallback_reason = "invalid_probabilities"
                    else:
                        # A previous shape may have requested a copy. The current
                        # cost bound proves it cannot help, so discard it without waiting.
                        self.snapshot_req_ids = None
                decision = (
                    req_ids,
                    caps.tolist() if caps is not None else None,
                    bucket,
                    capture,
                    fallback_reason,
                    shape,
                )
        req_ids, caps_list, bucket, capture, fallback_reason, shape = self.tp_group.broadcast_object(decision, src=0)
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
