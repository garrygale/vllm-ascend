# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.models.qwen3_domino import Qwen3DominoForCausalLM

from vllm_ascend.ops.triton.spec_decode.domino_gru import (
    domino_gru_cell_triton_gather,
)
from vllm_ascend.quantization.domino import (
    build_quantized_fused_norm_quant,
    build_quantized_fused_kv_buffers,
    build_quantized_fused_qkv,
    quantize_domino_model,
)


class AscendQwen3DominoForCausalLM(Qwen3DominoForCausalLM):
    """Ascend Domino wrapper.

    Ascend's DynamicGRU does not accept bf16, so after weight loading we create
    a persistent fp16 copy of the GRU parameters.  The base class GRU cell
    detects ``model._gru_fp16`` and casts per-step inputs to fp16 without
    mutating the bf16 parameters.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.vllm_config = vllm_config

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        super().load_weights(weights)
        self.model._gru_fp16 = {
            "weight_ih_l0": self.model.prefix_gru.weight_ih_l0.detach()
            .to(torch.float16)
            .contiguous(),
            "weight_hh_l0": self.model.prefix_gru.weight_hh_l0.detach()
            .to(torch.float16)
            .contiguous(),
        }
        self._domino_gi_table = None
        self._validate_domino_triton_gru()
        print(
            "[AscendDomino] Domino triton GRU is the default path; the "
            "gi_table is built after weight sharing",
            flush=True,
        )

        # On-the-fly per-channel quantization driven by the draft config
        # (dflash_config.qat_*): qat_w_bit selects the W4A8/W8A8 bulk scheme
        # and qat_w4a4_layers the per-layer W4A4; no qat_w_bit means bf16.
        # W4A8 (npu_weight_quant_batchmatmul, int4-packed int32) works in both
        # eager and ACL graph mode, so no eager/graph redirect is needed.
        num_quantized = quantize_domino_model(self.model)
        quant_fused = False
        if num_quantized:
            print(
                f"[AscendDomino] On-the-fly quantization applied to "
                f"{num_quantized} draft linears",
                flush=True,
            )
            quant_fused = build_quantized_fused_kv_buffers(self.model)
            if quant_fused:
                print(
                    "[AscendDomino] quantized fused context-KV "
                    f"precompute enabled ({self.model._fused_kv_scheme})",
                    flush=True,
                )
            if build_quantized_fused_qkv(self.model):
                print(
                    "[AscendDomino] fused draft qkv projections enabled "
                    "(quantized)",
                    flush=True,
                )
            if build_quantized_fused_norm_quant(self.model):
                print(
                    "[AscendDomino] W8A8 fused norm+quant enabled "
                    "(residual-stream draft layers + context-KV)",
                    flush=True,
                )

        if not quant_fused:
            # bf16 path (quantization disabled) or unsupported quantized
            # K/V scheme: rebuild the bf16 fused buffers or fall back to the
            # per-layer quantized projections.
            self.model._build_fused_kv_buffers()

    def _validate_domino_triton_gru(self) -> None:
        """Validate the triton-table path.

        Requires the ``embed_proj`` (no ``hidden_proj``) config.
        """
        if (
            self.model.embed_proj is None
            or self.model.hidden_proj is not None
        ):
            raise ValueError(
                "Domino triton GRU requires use_embed_proj=true "
                "and use_hidden_proj=false"
        )

    def _ensure_domino_triton_gru(self) -> None:
        """Precompute the triton-table cell + fused-h tensors.

        Called eagerly by ``AscendDominoSpeculator.load_draft_model`` after
        the draft ``embed_tokens`` is replaced by the target's shared
        embedding; the idempotent guard also covers any later first use.
          * ``gi_table = emb_weight @ W_ih^T`` (replaces the per-step x matmul
            with a gather); with TP>1 the per-rank shard is all-gathered into
            the full padded-vocab table so per-step gathers stay local,
          * ``W_sh = cat([W_s, W_hh], dim=0)`` (one h projection shared by the
            correction head and the GRU cell).
        """
        if self._domino_gi_table is not None:
            return

        H = self.model.target_hidden_size
        G = self.model.gru_hidden_dim
        M = self.model.emb_dim

        with torch.no_grad():
            w_ih = self.model.prefix_gru.weight_ih_l0.detach()  # [3G, H]
            w_hh = self.model.prefix_gru.weight_hh_l0.detach()  # [3G, G]
            fc1_w = self.model.embed_proj[0].weight.detach()  # [M, H+G]
            emb_w = self.model.embed_tokens.weight.detach()  # [V, H]

            gi_local = F.linear(emb_w, w_ih).contiguous()
            if get_tensor_model_parallel_world_size() > 1:
                # Local vocab shards are contiguous slices of the padded
                # vocab, so gathering in rank order reconstructs the full
                # table; per-step gathers then stay local and collective-free.
                gi_full = tensor_model_parallel_all_gather(gi_local, dim=0)
            else:
                gi_full = gi_local
            self._domino_gi_table = gi_full.contiguous()
            self._domino_w_z = fc1_w[:, :H].contiguous()
            w_s = fc1_w[:, H:].contiguous()
            self._domino_w_sh = torch.cat([w_s, w_hh], dim=0).contiguous()

        mb = (
            self._domino_gi_table.numel()
            * self._domino_gi_table.element_size()
            / 1e6
        )
        print(
            f"[AscendDomino] Domino triton GRU enabled "
            f"(gi_table {mb:.0f} MB)",
            flush=True,
        )

    def domino_z_part(self, sample_hidden: torch.Tensor) -> torch.Tensor:
        """Precompute ``z @ W_z^T`` once per block: ``[B, steps, M]``."""
        self._ensure_domino_triton_gru()
        return F.linear(sample_hidden, self._domino_w_z)

    def domino_optimized_prefix(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Run the table GRU over ``[B, 1+prefix_len]`` token ids."""
        self._ensure_domino_triton_gru()
        B = token_ids.shape[0]
        G = self.model.gru_hidden_dim
        h = torch.zeros(
            1,
            B,
            G,
            dtype=self.model.prefix_gru.weight_ih_l0.dtype,
            device=self.model.prefix_gru.weight_ih_l0.device,
        )
        for t in range(token_ids.shape[1]):
            sh = F.linear(h[0], self._domino_w_sh)
            gh = sh[:, self.model.emb_dim:]
            h = domino_gru_cell_triton_gather(
                self._domino_gi_table, token_ids[:, t], gh, h
            )
        return h

    def domino_optimized_bias_and_gh(
        self,
        h: torch.Tensor,
        z_part_i: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Correction bias + shared hidden projection for one step.

        ``sh = h @ W_sh^T`` is split into ``s_proj`` (correction) and ``gh``
        (cell), so the two per-step h matmuls become one.
        """
        self._ensure_domino_triton_gru()
        sh = F.linear(h[0], self._domino_w_sh)  # [B, M + 3G]
        s_proj = sh[:, : self.model.emb_dim]
        gh = sh[:, self.model.emb_dim:]
        x = F.silu(z_part_i + s_proj)
        bias = self.logits_processor(self.model.embed_proj[2], x)
        return bias, gh

    def domino_optimized_cell(
        self,
        token_ids: torch.Tensor,
        h: torch.Tensor,
        gh: torch.Tensor,
    ) -> torch.Tensor:
        """Table-cell step with a precomputed ``gh``."""
        self._ensure_domino_triton_gru()
        return domino_gru_cell_triton_gather(
            self._domino_gi_table, token_ids, gh, h
        )
