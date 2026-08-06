# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
from vllm.model_executor.models.qwen3_domino import Qwen3DominoForCausalLM


class AscendQwen3DominoForCausalLM(Qwen3DominoForCausalLM):
    """Ascend Domino wrapper.

    Ascend's DynamicGRU does not accept bf16, so after weight loading we create
    a persistent fp16 copy of the GRU parameters.  The base class GRU cell
    detects ``model._gru_fp16`` and casts per-step inputs to fp16 without
    mutating the bf16 parameters.
    """

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
