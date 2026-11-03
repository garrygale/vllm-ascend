#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attribute forwarding on ``ModelWithContext``.

The domino walks branch on ``getattr(speculator.model, ..., default)`` flags.
During ACL-graph capture ``speculator.model`` is replaced by
``ModelWithContext``; a flag missing from the wrapper silently flips to the
default inside the captured graph (the original bug: shared-readout
checkpoints captured the non-shared walk).
"""

from types import SimpleNamespace

import torch
from torch import nn

from vllm_ascend.worker.v2.aclgraph_utils import ModelWithContext


class _DraftLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_emb_readout = True
        self.pure_draft_prefix_len = 1

    def embed_tokens(self, token_ids):
        return token_ids * 2


def test_shared_emb_readout_forwards_to_the_wrapped_model():
    wrapper = ModelWithContext(_DraftLike())
    assert wrapper.shared_emb_readout is True
    assert wrapper.pure_draft_prefix_len == 1


def test_unknown_attributes_fall_through_to_the_wrapped_model():
    # A flag added to the draft model later must stay visible through the
    # wrapper without extending any whitelist.
    inner = SimpleNamespace(correction_top_k=128, gru_hidden_dim=1024)
    wrapper = ModelWithContext(inner)
    assert wrapper.correction_top_k == 128
    assert wrapper.gru_hidden_dim == 1024


def test_method_forwarding_survives_the_fallback():
    wrapper = ModelWithContext(_DraftLike())
    assert wrapper.embed_tokens(torch.tensor([3])).item() == 6


def test_missing_attributes_still_raise():
    wrapper = ModelWithContext(_DraftLike())
    try:
        wrapper.definitely_not_a_flag
    except AttributeError:
        pass
    else:
        raise AssertionError("expected AttributeError for an unknown flag")


def test_false_flags_read_as_false_not_missing():
    # The regression direction: getattr(wrapper, flag, False) on a
    # shared-readout checkpoint must observe True, never the default.
    wrapper = ModelWithContext(_DraftLike())
    assert bool(getattr(wrapper, "shared_emb_readout", False)) is True
