# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm_ascend.worker.v2.spec_decode.draft_attn_metadata import (
    pad_parallel_draft_attn_metadata,
)


def test_pad_parallel_draft_attn_metadata_restores_capture_geometry():
    metadata = SimpleNamespace(
        seq_lens=torch.tensor([101, 102, 103, 0, 0, 0], dtype=torch.int32),
        seq_lens_cpu=None,
        seq_lens_list=[101, 102, 103, 0, 0, 0],
        actual_seq_lengths_q=[7, 14, 21, 21, 21, 21],
    )

    pad_parallel_draft_attn_metadata(
        {"layer": metadata},
        num_reqs=3,
        num_reqs_padded=6,
        num_query_per_req=7,
    )

    assert metadata.actual_seq_lengths_q == [7, 14, 21, 28, 35, 42]
    assert metadata.seq_lens_list == [101, 102, 103, 7, 7, 7]
    assert torch.equal(
        metadata.seq_lens,
        torch.tensor([101, 102, 103, 7, 7, 7], dtype=torch.int32),
    )
