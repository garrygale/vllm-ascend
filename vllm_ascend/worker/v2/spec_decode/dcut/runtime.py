# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2 Domino probability transfer and optional live cost-table calibration."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .controller import CostKey, CostTable, allocate_prefixes, choose_caps, decode_batch_info, query_budgets

# Reuse the configured vLLM namespace without initializing an engine in CPU tests.
logger = logging.getLogger("vllm")


def selected_greedy_probability(logits: torch.Tensor) -> torch.Tensor:
    """Selected draft-vocabulary probability before draft-to-target mapping."""
    logits_fp32 = logits.float()
    shifted = logits_fp32 - logits_fp32.amax(dim=-1, keepdim=True)
    return (-torch.logsumexp(shifted, dim=-1)).exp()


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
        self.current_key: CostKey | None = None
        self.collecting = False
        self.stats = {"decisions": 0, "trimmed_tokens": 0, "fallbacks": 0, "profiled_rows": 0}
        error = None
        if self.is_root:
            try:
                fingerprint = json.loads(json.dumps(fingerprint))
                fingerprint["context_buckets"] = list(config.context_buckets)
                path = Path(config.cost_table_path)
                if path.exists():
                    self.table = CostTable.load(str(path), fingerprint)
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
        if self.measurement_key is None or self.draft_end is None:
            return
        if not self.draft_end.query():
            # Deliberate synchronization only while generating the cost table.
            self.draft_end.synchronize()
        assert self.table is not None
        try:
            completed = self.table.observe(
                self.measurement_key,
                self.target_start.elapsed_time(self.draft_start),
                self.draft_start.elapsed_time(self.draft_end),
                self.config.profile_warmup,
                self.config.profile_samples,
            )
            if completed:
                self.table.save(self.config.cost_table_path)
                self.stats["profiled_rows"] += 1
                logger.info("D-Cut calibrated cost row %s", self.measurement_key)
        except OSError as exc:
            logger.warning("D-Cut could not save cost table: %s", exc)
        finally:
            self.measurement_key = None
            self.target_start = self.draft_start = self.draft_end = None

    def finish_calibration(self) -> dict[str, int]:
        """Flush the final live sample via LLM.collective_rpc after generation."""
        if self.is_root and self.config.generate_cost_table:
            self._consume_measurement()
        return self.stats.copy()

    def _take_probabilities(self, req_ids: list[str]) -> np.ndarray | None:
        snapshot = self.snapshot_req_ids
        self.snapshot_req_ids = None
        if snapshot is None or len(snapshot) != len(req_ids) or set(snapshot) != set(req_ids):
            return None
        if not self.copy_event.query():
            if self.config.generate_cost_table or self.config.wait_for_probs:
                self.copy_event.synchronize()
            else:
                return None
        assert self.host_probs is not None
        rows = {req_id: row for row, req_id in enumerate(snapshot)}
        return self.host_probs.numpy()[[rows[req_id] for req_id in req_ids]].copy()

    def select_caps(self, scheduler_output: Any, req_states: Any) -> tuple[list[str], np.ndarray | None]:
        """Rank zero decides, including readiness/fallback, before graph dispatch."""
        self.current_key = None
        decision: Any = None
        if self.is_root:
            self._consume_measurement()
            info = decode_batch_info(scheduler_output, req_states, self.config.context_buckets)
            if info is None:
                self.snapshot_req_ids = None
                decision = ([], None, None)
            else:
                req_ids, bucket = info
                limits = np.array(
                    [len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, [])) for req_id in req_ids],
                    dtype=np.int32,
                )
                probabilities = self._take_probabilities(req_ids)
                caps = None
                assert self.table is not None
                if np.all(limits <= self.block_size):
                    if self.config.generate_cost_table:
                        lengths = self.config.candidate_draft_lengths or tuple(
                            sorted({0, 1, min(2, self.block_size), min(4, self.block_size), self.block_size})
                        )
                        for budget in query_budgets(limits, lengths):
                            if CostKey(len(req_ids), bucket, budget) not in self.table.rows:
                                # Balanced prefixes measure shapes without using probability estimates.
                                gains = np.broadcast_to(
                                    -np.arange(self.block_size, dtype=np.float64), (len(req_ids), self.block_size)
                                )
                                caps = allocate_prefixes(gains, limits, budget - len(req_ids))
                                break
                    if caps is None and probabilities is not None:
                        caps = choose_caps(probabilities, limits, self.table, bucket, self.config.min_gain)
                decision = (req_ids, caps.tolist() if caps is not None else None, bucket)
        req_ids, caps_list, bucket = self.tp_group.broadcast_object(decision, src=0)
        caps = np.array(caps_list, dtype=np.int32) if caps_list is not None else None
        if caps is None:
            self.stats["fallbacks"] += 1
        else:
            removed = sum(
                len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, [])) - int(cap)
                for req_id, cap in zip(req_ids, caps)
            )
            if removed:
                self.stats["decisions"] += 1
                self.stats["trimmed_tokens"] += removed
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
        return req_ids, caps

    def begin_target(self) -> None:
        if self.current_key is not None:
            self.target_start = self.npu.Event(enable_timing=True)
            self.target_start.record()

    def abort_target(self) -> None:
        self.current_key = self.measurement_key = None
        self.target_start = self.draft_start = self.draft_end = None
        self.snapshot_req_ids = None
        self.collecting = False

    def begin_proposal(self, dummy_run: bool, is_profile: bool) -> bool:
        self.collecting = self.is_root and not dummy_run and not is_profile
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
            self.current_key = None
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
