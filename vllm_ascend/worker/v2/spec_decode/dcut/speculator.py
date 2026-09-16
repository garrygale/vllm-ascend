# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in probability collection around the original Domino implementation."""

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.input_batch import InputBatch

from vllm_ascend.worker.v2.spec_decode.domino.speculator import AscendDominoSpeculator

from .runtime import DcutRuntime, selected_greedy_probability


class DcutDominoSpeculator(AscendDominoSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.dcut_runtime: DcutRuntime | None = None
        self._collect_dcut_probs = False
        self.selected_probs: torch.Tensor | None = None

    def set_dcut_runtime(self, runtime: DcutRuntime) -> None:
        self.dcut_runtime = runtime
        if runtime.is_root:
            self.selected_probs = torch.empty(
                (self.max_num_reqs, self.num_speculative_steps), dtype=torch.float32, device=self.device
            )

    def _sample_step(
        self, logits_i: torch.Tensor, idx_map_i: torch.Tensor, sample_pos_i: torch.Tensor, col: int
    ) -> torch.Tensor:
        if not self._collect_dcut_probs:
            return super()._sample_step(logits_i, idx_map_i, sample_pos_i, col)
        if getattr(self, "draft_logits", None) is not None:
            raise RuntimeError("D-Cut argmax reuse requires greedy Domino sampling")
        assert self.selected_probs is not None
        # Select once in draft vocabulary space, reuse the IDs both for the
        # probability gather and for the original draft-to-target mapping.
        selected_ids = logits_i.argmax(dim=-1)
        self.selected_probs[: logits_i.shape[0], col].copy_(
            selected_greedy_probability(logits_i, selected_ids)
        )
        return self.model.map_draft_to_target(selected_ids)

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        runtime = self.dcut_runtime
        self._collect_dcut_probs = runtime is not None and runtime.begin_proposal(dummy_run, is_profile)
        try:
            # The parent retains its original input_batch and attention-wrapper behavior.
            result = super().propose(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                num_tokens_across_dp,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile=is_profile,
            )
            if self._collect_dcut_probs:
                assert runtime is not None and self.selected_probs is not None
                runtime.end_proposal(input_batch, self.selected_probs)
            return result
        except Exception:
            if runtime is not None:
                runtime.abort_target()
            raise
        finally:
            self._collect_dcut_probs = False
