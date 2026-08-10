# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.models.qwen3_domino import Qwen3DominoForCausalLM

from vllm_ascend.ops.triton.spec_decode.domino_gru import (
    domino_gru_cell_triton,
)
from vllm_ascend.quantization.domino import quantize_domino_model


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
        env_value = os.environ.get("VLLM_ASCEND_DOMINO_TRITON_GRU", "").strip()
        print(
            f"[AscendDomino] load_weights: "
            f"VLLM_ASCEND_DOMINO_TRITON_GRU={env_value!r}",
            flush=True,
        )
        self.model._gru_fp16 = {
            "weight_ih_l0": self.model.prefix_gru.weight_ih_l0.detach()
            .to(torch.float16)
            .contiguous(),
            "weight_hh_l0": self.model.prefix_gru.weight_hh_l0.detach()
            .to(torch.float16)
            .contiguous(),
        }
        self._use_domino_triton_gru = False
        self._domino_gi_table = None
        if env_value.lower() in ("1", "true", "yes", "on"):
            self._validate_domino_triton_gru()
        elif env_value:
            print(
                f"[AscendDomino] WARNING: VLLM_ASCEND_DOMINO_TRITON_GRU="
                f"{env_value!r} is not a recognized truthy value; "
                f"Domino triton GRU stays disabled (use 1)",
                flush=True,
            )

        # On-the-fly per-channel quantization driven by the draft config
        # (dflash_config.qat_*).  Disable with VLLM_ASCEND_DOMINO_QUANT=0.
        # W4A8 (npu_weight_quant_batchmatmul, int4-packed int32) works in both
        # eager and ACL graph mode, so no eager/graph redirect is needed.
        quant_env = os.environ.get("VLLM_ASCEND_DOMINO_QUANT", "").strip().lower()
        if quant_env in ("0", "false", "no", "off"):
            print(
                "[AscendDomino] Domino quantization disabled by "
                "VLLM_ASCEND_DOMINO_QUANT; keeping bf16 weights",
                flush=True,
            )
        else:
            num_quantized = quantize_domino_model(self.model)
            if num_quantized:
                print(
                    f"[AscendDomino] On-the-fly quantization applied to "
                    f"{num_quantized} draft linears",
                    flush=True,
                )

        # Rebuild the fused context-KV buffers: after quantization the K/V
        # weights are no longer floating point, so this disables the fused
        # path and frees the bf16 buffers (per-layer quantized projections
        # are used instead).
        self.model._build_fused_kv_buffers()

    def _validate_domino_triton_gru(self) -> None:
        """Check the triton-table path is usable; tables build lazily.

        Only supports TP=1 and the ``embed_proj`` (no ``hidden_proj``) config.
        """
        if get_tensor_model_parallel_world_size() != 1:
            print(
                "[AscendDomino] WARNING: VLLM_ASCEND_DOMINO_TRITON_GRU "
                "requires TP=1; falling back to the manual GRU path.",
                flush=True,
            )
            return
        if getattr(self.model, "quant_config", None) is not None:
            print(
                "[AscendDomino] WARNING: VLLM_ASCEND_DOMINO_TRITON_GRU is "
                "not supported with quantized weights yet; falling back to "
                "the manual GRU path.",
                flush=True,
            )
            return
        if (
            self.model.embed_proj is None
            or self.model.hidden_proj is not None
        ):
            raise ValueError(
                "VLLM_ASCEND_DOMINO_TRITON_GRU requires use_embed_proj=true "
                "and use_hidden_proj=false"
        )
        self._use_domino_triton_gru = True
        print(
            "[AscendDomino] Domino triton GRU requested; tables will build "
            "lazily on first use",
            flush=True,
        )

    def _ensure_domino_triton_gru(self) -> None:
        """Lazily precompute the triton-table cell + fused-h tensors.

        Built on first use (not in ``load_weights``) because the draft
        ``embed_tokens`` is replaced by the target's shared embedding *after*
        weight loading; ``gi_table`` must come from the final shared table.
          * ``gi_table = emb_weight @ W_ih^T`` (replaces the per-step x matmul
            with a gather),
          * ``W_sh = cat([W_s, W_hh], dim=0)`` (one h projection shared by the
            correction head and the GRU cell).
        """
        if not self._use_domino_triton_gru:
            raise RuntimeError(
                "triton Domino GRU methods called but the path is not enabled"
            )
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

            self._domino_gi_table = F.linear(emb_w, w_ih).contiguous()
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
            f"(gi_table {mb:.0f} MB, TP=1)",
            flush=True,
        )
        self._use_domino_triton_gru = True

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
            gi = self._domino_gi_table[token_ids[:, t]]
            h = domino_gru_cell_triton(gi, gh, h)
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
        gi = self._domino_gi_table[token_ids]
        return domino_gru_cell_triton(gi, gh, h)
