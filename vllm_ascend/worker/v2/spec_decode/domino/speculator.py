# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, cast

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.domino.speculator import DominoSpeculator

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_wrapper

class AscendDominoSpeculator(DominoSpeculator):
    _speculator_name = "Domino"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.input_batch: InputBatch | None = None

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        super().init_cudagraph_manager(cudagraph_mode)
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
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        self._context_slot_mappings = self._context_slot_mappings.to(torch.int32)

        attn_backends: dict[str, type[AttentionBackend]] = {}
        active_layer_names = self.draft_attn_layer_names
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            layer_names = kv_cache_group_spec.layer_names
            if active_layer_names is not None:
                layer_names = list(active_layer_names.intersection(layer_names))

            layer_type = cast(type[Any], AttentionLayerBase)
            attn_layers = get_layers_from_vllm_config(
                self.vllm_config, layer_type, layer_names
            )
            for layer_name in layer_names:
                attn_backends[layer_name] = attn_layers[
                    layer_name
                ].get_attn_backend()

        self.attn_backends = attn_backends

    def build_draft_attn_metadatas(
        self, num_reqs_padded, seq_lens_cpu_upper_bound
    ):
        # vLLM 0.26.0's DFlashSpeculator._build_draft_attn_metadata does not
        # accept seq_lens_cpu_upper_bound / step. Keep this on the 0.26
        # signature unconditionally; the extra argument is accepted by the
        # caller but intentionally unused.
        num_tokens_padded = num_reqs_padded * self.num_query_per_req
        assert self.input_batch is not None
        with build_attn_metadata_wrapper():
            attn_metadata = self._build_draft_attn_metadata(
                num_reqs=self.input_batch.num_reqs,
                num_reqs_padded=num_reqs_padded,
                num_tokens_padded=num_tokens_padded,
                causal=self._group_causal,
            )
        return [attn_metadata]

    def _sample_sequential(
        self, num_reqs: int, head_hidden: torch.Tensor
    ) -> None:
        """Triton-table GRU + fused-h zpart correction loop.

        Default path (no env gate / manual fallback).  The correction head
        still produces full logits through the aclnn fc2 matmul, so sampling
        (Gumbel/top-k/top-p) is unaffected.
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

        anchor = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        prefix_ids = torch.empty(
            num_reqs,
            1 + prefix_len,
            dtype=torch.int64,
            device=self.device,
        )
        prefix_ids[:, 0] = anchor
        for i in range(prefix_len):
            draft_i = self._sample_step(
                base_logits[:, i],
                idx_map[:, i],
                sample_pos[:, i],
                i,
            )
            self.draft_tokens[:num_reqs, i] = draft_i
            prefix_ids[:, 1 + i] = draft_i

        gru_hidden = self.model.domino_optimized_prefix(prefix_ids)

        sample_hidden_3d = sample_hidden.view(num_reqs, n_spec, -1)
        z_part = self.model.domino_z_part(sample_hidden_3d)

        for i in range(prefix_len, n_spec):
            bias, gh = self.model.domino_optimized_bias_and_gh(
                gru_hidden, z_part[:, i]
            )
            logits_i = base_logits[:, i] + bias
            draft_i = self._sample_step(
                logits_i,
                idx_map[:, i],
                sample_pos[:, i],
                i,
            )
            self.draft_tokens[:num_reqs, i] = draft_i
            if i + 1 < n_spec:
                gru_hidden = self.model.domino_optimized_cell(
                    draft_i, gru_hidden, gh
                )


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
        self.input_batch = input_batch
        with build_attn_metadata_wrapper():
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
                num_tokens_across_dp,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile=is_profile,
            )
