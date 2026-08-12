# SPDX-License-Identifier: Apache-2.0
"""Ascend v1 Domino draft proposer.

Domino is DSpark-shaped (anchor-first block, ``num_query_per_req ==
num_speculative_tokens``) but replaces the Markov head with the
triton-table GRU correction loop.  This proposer runs on the v1 model
runner so it uses the v1 rejection sampler (``AscendRejectionSampler``)
and the v1 ``copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid``
input prep (with ``SAMPLE_FROM_ANCHOR=True``).

Unlike ``AscendDSparkProposer`` this path keeps draft graph mode enabled
and does not reject probabilistic draft sampling.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.ops.triton.spec_decode.utils import (
    copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid,
)
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer


class AscendDominoProposer(AscendDSparkProposer):
    """Domino block proposer for the v1 model runner."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        # Deliberately skip AscendDSparkProposer.__init__ (it forces eager
        # mode and rejects probabilistic draft sampling, both of which we
        # want to support on this path) and call the DFlash init directly,
        # then apply the DSpark-style buffer setup below.
        AscendDflashProposer.__init__(self, vllm_config, device, runner=runner)

        dflash_config = getattr(
            self.draft_model_config.hf_config, "dflash_config", None
        ) or {}
        self.target_hidden_size = dflash_config.get(
            "target_hidden_size",
            self.draft_model_config.hf_config.hidden_size,
        )
        self.num_draft_layers = self.draft_model_config.hf_config.num_hidden_layers

        # Anchor-as-first (N slots): Domino never uses the 1+N bonus-anchor
        # fill-in block.
        self.sample_from_anchor = True
        self.num_query_per_req = self.num_speculative_tokens
        # Domino consumes flare-fused target hidden states
        # [num_tokens, num_draft_layers * target_hidden_size].
        self.hidden_size = self.num_draft_layers * self.target_hidden_size

        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        self._dflash_hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        self.max_query_tokens = self.max_batch_size * self.num_query_per_req
        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )
        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )

        blk = 1 + self.num_speculative_tokens
        self._domino_draft_buffer = torch.zeros(
            (self.max_batch_size, blk), dtype=torch.int64, device=device
        )
        self._domino_anchor_buffer = torch.zeros(
            self.max_batch_size, dtype=torch.int64, device=device
        )

        # Per-gid block-table / slot-mapping bookkeeping (v1 self-manages
        # these; mirrors AscendDSparkProposer).
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        self._per_group_slot_mappings: dict[int, torch.Tensor] = {}
        self._per_group_kernel_block_sizes: dict[int, int] = {}
        self._per_group_block_table_buffers: dict[int, torch.Tensor] = {}
        self._per_group_query_slot_mapping_buffers: dict[int, torch.Tensor] = {}
        self._per_group_context_slot_mapping_buffers: dict[int, torch.Tensor] = {}
        self._context_slot_mapping_buffers: list[torch.Tensor | None] | None = None
        self._target_model: torch.nn.Module | None = None

    def load_model(self, model: torch.nn.Module) -> None:
        """Load the Domino draft model and share the target embed/lm_head."""
        self._target_model = model
        super().load_model(model)

    def _get_model(self) -> torch.nn.Module:
        from vllm.model_executor.models.qwen3_domino import load_domino_model

        assert self._target_model is not None
        return load_domino_model(self._target_model, self.vllm_config)

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        # The anchor (bonus) token seeds the GRU correction loop.
        n = next_token_ids.shape[0]
        self._domino_anchor_buffer[:n].copy_(next_token_ids)
        self._domino_anchor_buffer[n:].fill_(0)

        batch_size = cad.num_reqs
        num_query_total = batch_size * self.num_query_per_req
        num_sample_total = batch_size * self.num_speculative_tokens
        has_num_rejected = num_rejected_tokens_gpu is not None
        primary_gid = getattr(self, "kv_cache_gid", 0)
        self._per_group_block_table_buffers = {
            attn_group.kv_cache_group_id: self._per_group_block_tables[
                attn_group.kv_cache_group_id
            ]
            for attn_group in self.draft_attn_groups
        }
        self._context_slot_mapping_buffers = None
        self._dflash_num_context = int(cad.query_start_loc_cpu[batch_size])
        # ``target_hidden_states`` is already flare-fused by the base
        # proposer (``combine_hidden_states``), so copy it as-is.
        self._dflash_hidden_states[: self._dflash_num_context] = (
            target_hidden_states[: self._dflash_num_context]
        )

        token_indices_to_sample = torch.empty(
            num_sample_total,
            dtype=torch.int32,
            device=self.device,
        )

        draft_attn_groups = getattr(self, "draft_attn_groups", [])
        for attn_group in draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            gid_block_table = self._per_group_block_table_buffers.get(gid)
            if gid_block_table is None:
                continue
            kernel_block_size = self._per_group_kernel_block_sizes[gid]
            copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid[1, ](
                # Inputs
                next_token_ids_ptr=next_token_ids,
                target_positions_ptr=target_positions,
                context_slot_mapping_ptr=self._per_group_slot_mappings[gid],
                # Outputs
                out_input_ids_ptr=self.input_ids,
                out_context_positions_ptr=self._context_positions_buffer,
                out_query_positions_ptr=self.positions,
                out_context_slot_mapping_ptr=self._per_group_context_slot_mapping_buffers[gid],
                out_query_slot_mapping_ptr=self._per_group_query_slot_mapping_buffers[gid],
                out_token_indices_ptr=token_indices_to_sample,
                # Block table
                block_table_ptr=gid_block_table,
                block_table_stride=gid_block_table.stride(0),
                # Metadata
                query_start_loc_ptr=cad.query_start_loc,
                seq_lens_ptr=cad.seq_lens,
                num_rejected_tokens_ptr=num_rejected_tokens_gpu,
                # Scalars
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                block_size=kernel_block_size,
                num_query_per_req=self.num_query_per_req,
                num_speculative_tokens=self.num_speculative_tokens,
                total_input_tokens=self._dflash_num_context,
                batch_size=batch_size,
                HAS_NUM_REJECTED=has_num_rejected,
                SAMPLE_FROM_ANCHOR=self.sample_from_anchor,
            )
        self._context_slot_mapping_buffers = [
            self._per_group_context_slot_mapping_buffers[gidx]
            for gidx in self._layer_group_idx
        ]

        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        cad.query_start_loc = self.arange_dflash[: batch_size + 1] * self.num_query_per_req
        cad.seq_lens = effective_seq_lens + self.num_query_per_req
        cad.query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()
            * self.num_query_per_req
        ).to(torch.int32)

        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [self.num_query_per_req] * batch_size
        if hasattr(cad, "decode_token_per_req"):
            cad.decode_token_per_req = self.num_query_per_req

        cad.num_actual_tokens = num_query_total
        cad.num_input_tokens = num_query_total
        cad.max_query_len = self.num_query_per_req
        cad.max_seq_len = cad.max_seq_len + self.num_query_per_req
        cad.slot_mapping = self._per_group_query_slot_mapping_buffers[primary_gid][
            :num_query_total
        ]
        cad.positions = self.positions  # this would be sliced in attention backend
        cad.causal = False
        cad.attn_mask = None
        cad.attn_state = AscendAttentionState.ChunkedPrefill

        return num_query_total, token_indices_to_sample, cad, None
