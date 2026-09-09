# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Helpers for replaying parallel-draft attention metadata on Ascend."""

from typing import Any

import torch


def pad_parallel_draft_attn_metadata(
    attn_metadata: dict[str, Any] | None,
    *,
    num_reqs: int,
    num_reqs_padded: int,
    num_query_per_req: int,
) -> None:
    """Restore the captured dummy-row geometry for a padded draft replay.

    Upstream's parallel-draft metadata builder clamps ``query_start_loc`` at
    the live request count.  In a FULL graph replay that is smaller than the
    captured request count, so the cumulative query lengths stop at the live
    token count and the DFlash input kernel leaves the padded KV lengths at
    zero.  Capture, however, gives every dummy request ``num_query_per_req``
    query and KV tokens.

    FIA TND band mode (``sparse_mode=4``, used by non-causal sliding-window
    draft layers) derives its tiling and band geometry from both cumulative
    query lengths and per-row KV lengths.  A live request completing is
    therefore enough to make replay metadata disagree with the captured
    graph.  Full attention does not use the band geometry and tolerates the
    mismatch, which is why the corruption is window-specific.

    Rebuild the padded rows so replay matches capture exactly.  The padded
    rows point at block 0 in the persistent block-table buffers and their
    outputs are ignored by the speculator.
    """
    if attn_metadata is None or num_reqs_padded <= 0:
        return

    num_reqs = max(0, min(num_reqs, num_reqs_padded))
    padded_kv_len = max(1, num_query_per_req)
    cumulative_query_lens = [
        (i + 1) * num_query_per_req for i in range(num_reqs_padded)
    ]

    for metadata in attn_metadata.values():
        metadata.actual_seq_lengths_q = list(cumulative_query_lens)

        seq_lens_list = metadata.seq_lens_list
        if seq_lens_list is not None:
            seq_lens_list = list(seq_lens_list)
            if len(seq_lens_list) < num_reqs_padded:
                seq_lens_list.extend(
                    [padded_kv_len] * (num_reqs_padded - len(seq_lens_list))
                )
            for i in range(num_reqs, num_reqs_padded):
                seq_lens_list[i] = padded_kv_len
            metadata.seq_lens_list = seq_lens_list

        seq_lens = metadata.seq_lens
        if seq_lens is not None and seq_lens.dim() == 1:
            if seq_lens.numel() < num_reqs_padded:
                padding = seq_lens.new_full(
                    (num_reqs_padded - seq_lens.numel(),),
                    padded_kv_len,
                )
                seq_lens = torch.cat([seq_lens, padding])
            else:
                seq_lens = seq_lens[:num_reqs_padded].clone()
            seq_lens[num_reqs:num_reqs_padded] = padded_kv_len
            metadata.seq_lens = seq_lens
            metadata.seq_lens_cpu = seq_lens
