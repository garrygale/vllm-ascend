# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton-fused Domino GRU cell for Ascend NPU.

The native DynamicGRU (aclop) cannot be captured by ACL graph, and its fp16
requirement forces dtype round-trips.  The kernel loads/stores the cell dtype
(bf16, matching specforge's eagerGRU); the gate math itself runs in fp32
(triton-ascend 3.6 restricts sigmoid/exp to fp32 inputs, and torch computes
bf16 sigmoid in fp32 internally anyway).
It is the validated winner from ``benchmarks/bench_gru_*``.  (The old
triton-ascend ``care_padding=False`` vector-core hint died with
triton-ascend 3.6; masked lanes are never stored, so plain masked loads
are correct on every version.)
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fused_gru_cell_kernel(
    gi_ptr,
    gh_ptr,
    h_ptr,
    h_out_ptr,
    B,
    G,
    stride_gi_b,
    stride_gi_g,
    stride_gh_b,
    stride_gh_g,
    stride_h_b,
    stride_h_g,
    stride_hout_b,
    stride_hout_g,
    BLOCK_G: tl.constexpr,
):
    """Fused GRU cell: gi/gh already contain the input/hidden projections.

    Loads/stores keep the cell dtype (bf16); the gate sums are cast to
    fp32 for sigmoid (triton-ascend 3.6 dtype rule), like specforge's eagerGRU:
        r = sigmoid(gi_r + gh_r)
        z = sigmoid(gi_z + gh_z)
        n = tanh(gi_n + r * gh_n)   (via 2*sigmoid(2x)-1)
        h_new = (1 - z) * n + z * h
    Masked lanes are elementwise-independent and never stored.
    """
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)
    offs_g = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    mask_g = offs_g < G

    h_state = tl.load(
        h_ptr + pid_b * stride_h_b + offs_g * stride_h_g,
        mask=mask_g,
    )
    gi_r = tl.load(
        gi_ptr + pid_b * stride_gi_b + offs_g * stride_gi_g,
        mask=mask_g,
    )
    gi_z = tl.load(
        gi_ptr + pid_b * stride_gi_b + (G + offs_g) * stride_gi_g,
        mask=mask_g,
    )
    gi_n = tl.load(
        gi_ptr + pid_b * stride_gi_b + (2 * G + offs_g) * stride_gi_g,
        mask=mask_g,
    )
    gh_r = tl.load(
        gh_ptr + pid_b * stride_gh_b + offs_g * stride_gh_g,
        mask=mask_g,
    )
    gh_z = tl.load(
        gh_ptr + pid_b * stride_gh_b + (G + offs_g) * stride_gh_g,
        mask=mask_g,
    )
    gh_n = tl.load(
        gh_ptr + pid_b * stride_gh_b + (2 * G + offs_g) * stride_gh_g,
        mask=mask_g,
    )

    # Gate math runs in fp32: triton-ascend 3.6 restricts tl.sigmoid/exp to
    # fp32/fp64 inputs (_check_dtype), so the bf16 cell sums are cast up
    # here and the result cast back at the store. This also matches
    # eagerGRU closer (torch computes bf16 sigmoid in fp32 internally).
    r = tl.sigmoid((gi_r + gh_r).to(tl.float32))
    z = tl.sigmoid((gi_z + gh_z).to(tl.float32))
    n = 2.0 * tl.sigmoid(2.0 * (gi_n + r * gh_n.to(tl.float32)).to(tl.float32)) - 1.0
    h_new = (1.0 - z) * n + z * h_state.to(tl.float32)

    tl.store(
        h_out_ptr + pid_b * stride_hout_b + offs_g * stride_hout_g,
        h_new.to(h_ptr.dtype.element_ty),
        mask=mask_g,
    )


def domino_gru_cell_triton(
    gi: torch.Tensor,
    gh: torch.Tensor,
    h: torch.Tensor,
    h_out: torch.Tensor | None = None,
    block_g: int = 256,
) -> torch.Tensor:
    """One Triton-fused GRU step.

    Args:
        gi: ``[B, 3G]`` input projection (or gathered from the precomputed
            ``gi_table``).
        gh: ``[B, 3G]`` hidden projection (shared with the correction head in
            the fused-h path).
        h: ``[1, B, G]`` current hidden state.
        h_out: optional preallocated ``[B, G]`` output buffer (capture-safe
            reuse).
        block_g: Triton BLOCK_G.

    Returns:
        ``[1, B, G]`` new hidden state in ``h``'s dtype.
    """
    B = gi.shape[0]
    G = h.shape[-1]
    if h_out is None:
        h_out = torch.empty(B, G, dtype=h.dtype, device=h.device)
    grid = (B, triton.cdiv(G, block_g))
    _fused_gru_cell_kernel[grid](
        gi,
        gh,
        h[0],
        h_out,
        B,
        G,
        gi.stride(0),
        gi.stride(1),
        gh.stride(0),
        gh.stride(1),
        h[0].stride(0),
        h[0].stride(1),
        h_out.stride(0),
        h_out.stride(1),
        BLOCK_G=block_g,
    )
    return h_out.unsqueeze(0)


@triton.jit
def _fused_gru_cell_gather_kernel(
    gi_table_ptr,
    tokens_ptr,
    gh_ptr,
    h_ptr,
    h_out_ptr,
    B,
    G,
    stride_tok,
    stride_gh_b,
    stride_gh_g,
    stride_h_b,
    stride_h_g,
    stride_hout_b,
    stride_hout_g,
    BLOCK_G: tl.constexpr,
):
    """GRU cell with the ``gi_table[token]`` gather fused in.

    Same math as ``_fused_gru_cell_kernel``; the input projection rows are
    loaded directly from the full-vocab ``gi_table`` using the sampled
    token ids, avoiding the separate ``[B, 3G]`` gather round trip.
    """
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)
    offs_g = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    mask_g = offs_g < G

    token = tl.load(tokens_ptr + pid_b * stride_tok).to(tl.int64)
    gi_base = token * (3 * G)

    h_state = tl.load(
        h_ptr + pid_b * stride_h_b + offs_g * stride_h_g,
        mask=mask_g,
    )
    gi_r = tl.load(
        gi_table_ptr + gi_base + offs_g,
        mask=mask_g,
    )
    gi_z = tl.load(
        gi_table_ptr + gi_base + G + offs_g,
        mask=mask_g,
    )
    gi_n = tl.load(
        gi_table_ptr + gi_base + 2 * G + offs_g,
        mask=mask_g,
    )
    gh_r = tl.load(
        gh_ptr + pid_b * stride_gh_b + offs_g * stride_gh_g,
        mask=mask_g,
    )
    gh_z = tl.load(
        gh_ptr + pid_b * stride_gh_b + (G + offs_g) * stride_gh_g,
        mask=mask_g,
    )
    gh_n = tl.load(
        gh_ptr + pid_b * stride_gh_b + (2 * G + offs_g) * stride_gh_g,
        mask=mask_g,
    )

    # Gate math runs in fp32: triton-ascend 3.6 restricts tl.sigmoid/exp to
    # fp32/fp64 inputs (_check_dtype), so the bf16 cell sums are cast up
    # here and the result cast back at the store. This also matches
    # eagerGRU closer (torch computes bf16 sigmoid in fp32 internally).
    r = tl.sigmoid((gi_r + gh_r).to(tl.float32))
    z = tl.sigmoid((gi_z + gh_z).to(tl.float32))
    n = 2.0 * tl.sigmoid(2.0 * (gi_n + r * gh_n.to(tl.float32)).to(tl.float32)) - 1.0
    h_new = (1.0 - z) * n + z * h_state.to(tl.float32)
    tl.store(
        h_out_ptr + pid_b * stride_hout_b + offs_g * stride_hout_g,
        h_new.to(h_ptr.dtype.element_ty),
        mask=mask_g,
    )


def domino_gru_cell_triton_gather(
    gi_table: torch.Tensor,
    token_ids: torch.Tensor,
    gh: torch.Tensor,
    h: torch.Tensor,
    h_out: torch.Tensor | None = None,
    block_g: int = 256,
) -> torch.Tensor:
    """Fused table-gather + one Triton GRU step.

    Args:
        gi_table: ``[V, 3G]`` precomputed input-projection table.
        token_ids: ``[B]`` sampled tokens (rows into ``gi_table``).
        gh: ``[B, 3G]`` hidden projection (shared with the correction head).
        h: ``[1, B, G]`` current hidden state.
        h_out: optional preallocated ``[B, G]`` output buffer.
        block_g: Triton BLOCK_G.

    Returns:
        ``[1, B, G]`` new hidden state in ``h``'s dtype.
    """
    B = token_ids.shape[0]
    G = h.shape[-1]
    if h_out is None:
        h_out = torch.empty(B, G, dtype=h.dtype, device=h.device)
    grid = (B, triton.cdiv(G, block_g))
    _fused_gru_cell_gather_kernel[grid](
        gi_table,
        token_ids,
        gh,
        h[0],
        h_out,
        B,
        G,
        token_ids.stride(0),
        gh.stride(0),
        gh.stride(1),
        h[0].stride(0),
        h[0].stride(1),
        h_out.stride(0),
        h_out.stride(1),
        BLOCK_G=block_g,
    )
    return h_out.unsqueeze(0)
