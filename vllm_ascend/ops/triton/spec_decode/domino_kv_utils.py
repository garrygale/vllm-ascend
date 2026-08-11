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
    total_rows,
    rows_per_layer,
    hd: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    core_id = tl.program_id(0)
    core_num = tl.num_programs(0)
    rows_per_core = tl.cdiv(total_rows, core_num)
    start = core_id * rows_per_core
    end = tl.minimum(start + rows_per_core, total_rows)
    offs = tl.arange(0, BLOCK_HD)
    for row_start in tl.range(start, end, BLOCK_M):
        rows = row_start + tl.arange(0, BLOCK_M)
        mask_r = rows < total_rows
        mask = mask_r[:, None] & (offs[None, :] < hd)
        x = tl.load(
            x_ptr + rows[:, None] * hd + offs[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        layer = rows // rows_per_layer
        w = tl.load(
            w_ptr + layer[:, None] * hd + offs[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        mean_sq = tl.sum(x * x, axis=1) / hd
        y = x * tl.rsqrt(mean_sq[:, None] + eps) * w
        tl.store(
            out_ptr + rows[:, None] * hd + offs[None, :],
            y.to(tl.bfloat16),
            mask=mask,
        )


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
    props = triton.runtime.driver.active.utils.get_device_properties(
        x.device
    )
    num_cores = props.get("num_vectorcore", -1)
    if num_cores <= 0:
        num_cores = min(64, x_flat.shape[0])
    _grouped_k_norm_kernel[(num_cores,)](
        x_flat,
        w_flat,
        out,
        x_flat.shape[0],
        t * nkv,
        hd=hd,
        eps=eps,
        BLOCK_HD=BLOCK_HD,
        BLOCK_M=16,
    )
    return out.view(d, t, nkv, hd)
