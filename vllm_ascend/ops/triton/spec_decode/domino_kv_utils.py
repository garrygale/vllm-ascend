#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused grouped k-norm and multi-layer KV cache write for Domino.

The Domino context-KV precompute currently runs a per-layer k-norm loop and a
per-layer ``npu_scatter_pa_kv_cache`` write.  These two kernels replace both
loops with single Triton launches:

  * ``domino_grouped_k_norm``: one RMSNorm over the last dim (head_dim) for
    every (layer, token, kv-head) row, with the per-layer weight selected by
    the row's layer index;
  * ``fused_kv_cache_write``: one program per (layer, token) writing K and V
    into the layer's paged KV cache at its flat slot (Norm layout), assuming
    the cache tensors are contiguous ``[num_blocks, block_size, nkv, hd]``.

Both kernels keep the reference math: fp32 RMS reduction and direct slot
copy, so results are bit-compatible with the per-layer ops within bf16.
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


@triton.jit
def _fused_kv_write_kernel(
    k_all_ptr,
    v_all_ptr,
    slots_ptr,
    kc0, vc0,
    kc1, vc1,
    kc2, vc2,
    kc3, vc3,
    kc4, vc4,
    kc5, vc5,
    kc6, vc6,
    num_ctx,
    nkv_hd: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    layer = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    mask = offs < nkv_hd
    slot = tl.load(slots_ptr + layer * num_ctx + t)
    src_k = (
        k_all_ptr
        + layer * num_ctx * nkv_hd
        + t * nkv_hd
        + offs
    )
    src_v = (
        v_all_ptr
        + layer * num_ctx * nkv_hd
        + t * nkv_hd
        + offs
    )
    dst = slot * nkv_hd + offs
    k_val = tl.load(src_k, mask=mask)
    v_val = tl.load(src_v, mask=mask)
    if layer == 0:
        tl.store(kc0 + dst, k_val, mask=mask)
        tl.store(vc0 + dst, v_val, mask=mask)
    elif layer == 1:
        tl.store(kc1 + dst, k_val, mask=mask)
        tl.store(vc1 + dst, v_val, mask=mask)
    elif layer == 2:
        tl.store(kc2 + dst, k_val, mask=mask)
        tl.store(vc2 + dst, v_val, mask=mask)
    elif layer == 3:
        tl.store(kc3 + dst, k_val, mask=mask)
        tl.store(vc3 + dst, v_val, mask=mask)
    elif layer == 4:
        tl.store(kc4 + dst, k_val, mask=mask)
        tl.store(vc4 + dst, v_val, mask=mask)
    elif layer == 5:
        tl.store(kc5 + dst, k_val, mask=mask)
        tl.store(vc5 + dst, v_val, mask=mask)
    else:
        tl.store(kc6 + dst, k_val, mask=mask)
        tl.store(vc6 + dst, v_val, mask=mask)


def fused_kv_cache_write(
    k_all: torch.Tensor,
    v_all: torch.Tensor,
    key_caches: list[torch.Tensor],
    value_caches: list[torch.Tensor],
    slot_mappings: list[torch.Tensor],
) -> bool:
    """Write ``[D, T, nkv, hd]`` K/V into per-layer paged caches in one kernel.

    Returns False when the layout assumptions are not met (caller falls back
    to the per-layer ``npu_scatter_pa_kv_cache`` loop).
    """
    d, t, nkv, hd = k_all.shape
    if d != 7:
        return False
    nkv_hd = nkv * hd
    for cache in key_caches + value_caches:
        if (
            cache.ndim != 4
            or not cache.is_contiguous()
            or cache.dtype not in (torch.bfloat16, torch.float16)
            or cache.shape[2:] != (nkv, hd)
        ):
            return False
    slots = []
    for s in slot_mappings:
        if s is None or s.shape[0] < t:
            return False
        s = s[:t]
        if s.dtype != torch.int32:
            s = s.to(torch.int32)
        slots.append(s.contiguous())
    slots_stack = torch.stack(slots, dim=0)  # [D, T]
    k_flat = k_all.contiguous().view(d * t, nkv_hd)
    v_flat = v_all.contiguous().view(d * t, nkv_hd)
    BLOCK = triton.next_power_of_2(nkv_hd)
    _fused_kv_write_kernel[(t, d)](
        k_flat,
        v_flat,
        slots_stack,
        key_caches[0], value_caches[0],
        key_caches[1], value_caches[1],
        key_caches[2], value_caches[2],
        key_caches[3], value_caches[3],
        key_caches[4], value_caches[4],
        key_caches[5], value_caches[5],
        key_caches[6], value_caches[6],
        t,
        nkv_hd=nkv_hd,
        BLOCK=BLOCK,
    )
    return True
