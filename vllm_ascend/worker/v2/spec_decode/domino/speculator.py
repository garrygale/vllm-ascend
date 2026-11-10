# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, cast

import numpy as np
import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator
from vllm.v1.worker.gpu.spec_decode.domino.speculator import DominoSpeculator

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionState
from vllm_ascend.attention.mla_v1 import AscendMLABackend
from vllm_ascend.worker.v2.aclgraph_utils import _get_graph_update_backend
from vllm_ascend.worker.v2.attn_utils import (
    build_attn_metadata_factory,
    build_attn_metadata_wrapper,
)
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    prepare_dflash_inputs_factory,
)


class AscendDominoSpeculator(DominoSpeculator):
    """Ascend wrapper for the domino speculator.

    Draft attention metadata is built through the unified MRV2 factories
    (build_attn_metadata_wrapper / build_attn_metadata_factory) around
    vllm-core's _build_uniform_attn_metadata with per-group causality, and
    FULL-graph replay query lengths are re-staged through
    _update_draft_attn_metadata -- the machinery that replaced the 0.26
    draft_attn_metadata padding module. The GRU-corrected sequential sampling
    runs through the draft model's triton-table helpers.
    """

    _speculator_name = "Domino"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.input_batch: InputBatch | None = None
        self.attn_architecture: str | None = None

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if self.speculative_config.enforce_eager:
            cudagraph_mode = CUDAGraphMode.NONE
        super().init_cudagraph_manager(cudagraph_mode)
        # The Ascend graph manager is patched onto the upstream module and
        # created by super().init_cudagraph_manager without a speculator ref.
        # It needs this speculator to update full-graph params, so set it here.
        if self.query_cudagraph_manager is not None:
            self.query_cudagraph_manager.speculator = self
            self.query_cudagraph_manager.update_stream = self.update_stream

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = super().load_draft_model(
            target_model, target_attn_layer_names
        )
        # Build the triton-table eagerly now that embed/lm_head sharing is
        # done, so the one-time TP all-gather (TP>1) never runs inside ACL
        # graph capture.  ``_ensure_domino_triton_gru`` stays idempotent.
        model._ensure_domino_triton_gru()
        return model

    def set_attn(
        self,
        model_state: Any,
        kv_cache_config: Any,
        block_tables: Any,
        target_input_buffers: Any,
        target_attn_groups: Any,
    ) -> None:
        # Initialize the draft attention backend with its PCP=1 config.
        with set_current_vllm_config(self.attn_vllm_config):
            super().set_attn(
                model_state,
                kv_cache_config,
                block_tables,
                target_input_buffers,
                target_attn_groups,
            )
            self._context_slot_mappings = self._context_slot_mappings.to(torch.int32)  # type: ignore[has-type]
            # npu needs attn_backends to update full graph params in run_fullgraph.
            attn_backends: dict[str, type[AttentionBackend]] = {}
            active_layer_names = self.draft_attn_layer_names
            for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
                layer_names = kv_cache_group_spec.layer_names
                if active_layer_names is not None:
                    # Preserve cache-group order so captured graph tasks and
                    # runtime metadata stay aligned.
                    layer_names = [name for name in layer_names if name in active_layer_names]

                layer_type = cast(type[Any], AttentionLayerBase)
                attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type, layer_names)

                for layer_name in layer_names:
                    attn_backends[layer_name] = attn_layers[layer_name].get_attn_backend()

            self.attn_backends = attn_backends
            backend = _get_graph_update_backend(self.attn_groups)
            if issubclass(backend, AscendMLABackend):
                self.attn_architecture = "MLA"
            elif issubclass(backend, AscendAttentionBackend):
                self.attn_architecture = "GQA"
            else:
                self.attn_architecture = None
            dflash_speculator.prepare_dflash_inputs = prepare_dflash_inputs_factory(
                self.vllm_config.cache_config.block_size
            )

    def _prepare_draft_dcp_metadata_inputs(
        self, num_reqs: int, num_reqs_padded: int, step: int
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        # Domino drafts are GQA without DCP; the MLA host-side length fix and
        # the DCP manager do not apply.
        is_prefilling = torch.zeros(num_reqs_padded, dtype=torch.bool)
        return None, is_prefilling

    # Graph capture builds metadata from the padded request shape.
    def build_draft_attn_metadatas(self, num_reqs_padded, seq_lens_cpu_upper_bound):
        assert self.input_batch is not None
        num_tokens_padded = num_reqs_padded * self.num_query_per_req
        seq_lens_cpu, is_prefilling = self._prepare_draft_dcp_metadata_inputs(
            self.input_batch.num_reqs, num_reqs_padded, self.num_query_per_req
        )
        with (
            build_attn_metadata_wrapper(),
            build_attn_metadata_factory(
                self.input_buffers.positions,
                num_tokens_padded,
                is_prefilling=is_prefilling,
                seq_lens_cpu=seq_lens_cpu,
                attn_state=AscendAttentionState.ChunkedPrefill,
                parallel_config=self.attn_vllm_config.parallel_config,
            ),
        ):
            batch_desc = BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=num_tokens_padded,
                num_reqs=num_reqs_padded,
            )
            attn_metadata = self._build_uniform_attn_metadata(
                num_reqs=self.input_batch.num_reqs,
                batch_desc=batch_desc,
                num_query_per_req=self.num_query_per_req,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=self.num_query_per_req,
                causal=self._group_causal,
            )

        return [attn_metadata]

    def _build_attn_metadata(
        self,
        num_reqs: int,
        batch_desc: BatchExecutionDescriptor,
        query_start_loc_np: np.ndarray,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        causal: bool | dict[int, bool] = True,
        dcp_local_seq_lens: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        """Build Ascend draft metadata through the upstream attention hook."""
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        # The draft forward may process padded inputs outside FULL graphs, but
        # attention only consumes the tokens owned by real requests there.
        num_tokens = batch_desc.num_tokens if batch_desc.cg_mode == CUDAGraphMode.FULL else int(query_start_loc_np[-1])
        seq_lens_cpu, is_prefilling = self._prepare_draft_dcp_metadata_inputs(num_reqs, num_reqs_padded, step)
        with (
            build_attn_metadata_wrapper(),
            build_attn_metadata_factory(
                self.input_buffers.positions,
                num_tokens,
                is_prefilling=is_prefilling,
                seq_lens_cpu=seq_lens_cpu,
                attn_state=AscendAttentionState.ChunkedPrefill,
                parallel_config=self.attn_vllm_config.parallel_config,
            ),
        ):
            attn_metadata = super()._build_attn_metadata(
                num_reqs=num_reqs,
                batch_desc=batch_desc,
                query_start_loc_np=query_start_loc_np,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=step,
                causal=causal,
                dcp_local_seq_lens=dcp_local_seq_lens,
            )
        if batch_desc.cg_mode == CUDAGraphMode.FULL and attn_metadata is not None:
            # FULL replay uses the captured padded Q, so FIA needs cumulative
            # query lengths through every padded request slot.
            self._update_draft_attn_metadata(attn_metadata, num_reqs_padded)
        return attn_metadata

    def _update_draft_attn_metadata(self, attn_metadata, num_reqs_padded):
        """Match FULL-graph query lengths to the padded request count."""
        query_lens_list = [(i + 1) * self.num_query_per_req for i in range(num_reqs_padded)]
        for metadata in attn_metadata.values():
            decode_metadata = metadata.decode if self.attn_architecture == "MLA" else metadata
            decode_metadata.actual_seq_lengths_q = query_lens_list
        return attn_metadata

    def _sample_sequential(
        self, num_reqs: int, head_hidden: torch.Tensor
    ) -> None:
        """Triton-table GRU + fused-h zpart correction loop.

        Replaces the base GRU cell with the gi_table gather path; the
        correction head still produces full logits through aclnn matmuls, so
        sampling (gumbel/top-k/top-p) is unchanged.
        """
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]

        base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)

        prefix_len = int(getattr(self.model, "pure_draft_prefix_len", 0))
        if prefix_len > n_spec:
            raise ValueError(
                f"Domino pure_draft_prefix_len ({prefix_len}) cannot exceed "
                f"num_speculative_tokens ({n_spec})"
            )

        # Column writes into draft_tokens/prefix_ids are per-step
        # aclnnInplaceCopy ops on strided views; collect the sampled rows
        # and write each buffer once as a contiguous block instead.
        sampled_drafts = []
        anchor = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        for i in range(prefix_len):
            draft_i = self._sample_step(
                base_logits[:, i],
                idx_map[:, i],
                sample_pos[:, i],
                i,
            )
            sampled_drafts.append(draft_i)
        if prefix_len:
            prefix_ids = torch.cat(
                [anchor.view(-1, 1), torch.stack(sampled_drafts, 1)],
                dim=1,
            )
        else:
            # pure_draft_prefix_len == 0: the GRU warm-up sees the anchor
            # only (torch.stack of an empty list raises).
            prefix_ids = anchor.view(-1, 1)

        gru_hidden = self.model.domino_optimized_prefix(prefix_ids)

        sample_hidden_3d = sample_hidden.view(num_reqs, n_spec, -1)
        # Step-major layout: one explicit transpose+contiguous here replaces
        # the per-step strided slice z_part[:, i], which aclnn silently
        # contiguous-copies (the [B, emb_dim] aclnnInplaceCopy_viewCopy seen
        # before the fused GRU cell) on every correction step.
        z_part = self.model.domino_z_part(sample_hidden_3d).transpose(0, 1).contiguous()

        for i in range(prefix_len, n_spec):
            bias, gh = self.model.domino_optimized_bias_and_gh(
                gru_hidden, z_part[i]
            )
            logits_i = base_logits[:, i] + bias
            draft_i = self._sample_step(
                logits_i,
                idx_map[:, i],
                sample_pos[:, i],
                i,
            )
            sampled_drafts.append(draft_i)
            if i + 1 < n_spec:
                gru_hidden = self.model.domino_optimized_cell(
                    draft_i, gru_hidden, gh
                )
        self.draft_tokens[:num_reqs] = torch.stack(sampled_drafts, dim=1)

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
        dp_sync: Any = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        self.input_batch = input_batch
        assert self.input_batch is not None
        sync_state = dp_sync
        if dummy_run and skip_attn_for_dummy_run:
            # Profiling runs the draft with its own query token count, which
            # can differ from the target batch. Let forward_context coordinate
            # the actual draft counts instead of reusing the target DP state.
            sync_state = None
        is_prefilling = torch.from_numpy(self.input_batch.is_prefilling_np)
        with (
            build_attn_metadata_wrapper(),
            build_attn_metadata_factory(
                self.input_buffers.positions,
                self.max_num_tokens,
                is_prefilling,
                parallel_config=self.attn_vllm_config.parallel_config,
            ),
        ):
            return super().propose(
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
                sync_state,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile=is_profile,
            )
