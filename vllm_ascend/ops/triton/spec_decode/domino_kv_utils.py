#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused grouped k-norm for the Domino context-KV precompute.

One RMSNorm over the last dim (head_dim) for every (layer, token, kv-head)
row, with the per-layer weight selected by the row's layer index, replacing
the per-layer k-norm loop.  Keeps the reference fp32 RMS reduction math, so
results are bit-compatible with the per-layer ``npu_rms_norm`` calls within
bf16.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _grouped_k_norm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    rows_per_layer,
    hd: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    row = tl.program_id(0)
    layer = row // rows_per_layer
    offs = tl.arange(0, BLOCK_HD)
    mask = offs < hd
    x = tl.load(x_ptr + row * hd + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    w = tl.load(
        w_ptr + layer * hd + offs, mask=mask, other=0.0
    ).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / hd
    y = x * tl.rsqrt(mean_sq + eps) * w
    tl.store(out_ptr + row * hd + offs, y.to(tl.bfloat16), mask=mask)


def domino_grouped_k_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Grouped RMSNorm over ``[D, T, nkv, hd]`` with per-layer ``[D, hd]``
    weights, in one kernel."""
    d, t, nkv, hd = x.shape
    x_flat = x.contiguous().view(d * t * nkv, hd)
    w_flat = weight.contiguous()
    out = torch.empty_like(x_flat)
    BLOCK_HD = triton.next_power_of_2(hd)
    _grouped_k_norm_kernel[(x_flat.shape[0],)](
        x_flat,
        w_flat,
        out,
        t * nkv,
        hd=hd,
        eps=eps,
        BLOCK_HD=BLOCK_HD,
    )
    return out.view(d, t, nkv, hd)

