#!/usr/bin/env python3
# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""ACL-graph probe for the Domino GRU correction loop.

Times the full 15-step GRU loop (the service op chain) and each per-step
op in isolation, all inside ``torch.npu.NPUGraph`` (graph mode only), at
B = 32 / 64.  Comparing the per-op table against the full-loop time shows
how much of each step is kernel execution vs scheduling gaps between the
serialized kernels.

The op chain mirrors ``AscendDominoSpeculator._sample_sequential`` with the
default greedy draft path:

  ``F.linear(h, W_sh) -> add z_part + silu -> F.linear(x, embed_proj2) ->
  base_logits + bias -> argmax -> gi_table[token] -> domino_gru_cell_triton``

``embed_proj[2]`` is approximated by a plain ``F.linear`` (TP=1, no logits
processors), and the loop runs all 15 steps (no prefix split).  The
probabilistic draft path (``draft_sample_method="probabilistic"``) would use
``gumbel_sample`` instead of ``argmax``; the service default is greedy.

Run directly on an NPU:
    python benchmarks/probe_gru_loop_graph.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.ops.triton.spec_decode.domino_gru import (
    domino_gru_cell_triton,
)
from vllm_ascend.ops.triton.triton_utils import (
    init_device_properties_triton,
)

H = 4096      # target hidden size (z dim)
G = 1024      # GRU hidden dim
M = 256       # emb_dim (correction hidden)
V = 151936    # vocab size
N_SPEC = 15
MS = (32, 64)
ITERS = 50
WARMUP = 10
DTYPE = torch.bfloat16
def _build_inputs(b: int, device: str, seed: int):
    torch.manual_seed(seed)
    w_sh = torch.randn(M + 3 * G, G, dtype=DTYPE, device=device) * 0.02
    w_emb2 = torch.randn(V, M, dtype=DTYPE, device=device) * 0.02
    base_logits = torch.randn(b, N_SPEC, V, dtype=DTYPE, device=device) * 0.05
    z_part = torch.randn(b, N_SPEC, M, dtype=DTYPE, device=device) * 0.05
    h = torch.zeros(1, b, G, dtype=DTYPE, device=device)
    draft_tokens = torch.empty(b, N_SPEC, dtype=torch.int64, device=device)
    return (
        w_sh,
        w_emb2,
        base_logits,
        z_part,
        h,
        draft_tokens,
    )


def _step(
    i: int,
    h: torch.Tensor,
    base_logits: torch.Tensor,
    z_part: torch.Tensor,
    gi_table: torch.Tensor,
    w_sh: torch.Tensor,
    w_emb2: torch.Tensor,
    draft_tokens: torch.Tensor,
) -> torch.Tensor:
    """One service GRU step; returns the new ``[1, B, G]`` hidden state."""
    sh = F.linear(h[0], w_sh)                     # [B, M+3G]
    s_proj = sh[:, :M]
    gh = sh[:, M:]
    x = F.silu(z_part[:, i] + s_proj)             # [B, M]
    bias = F.linear(x, w_emb2)                    # [B, V]
    logits_i = base_logits[:, i] + bias           # [B, V]
    draft = logits_i.argmax(dim=-1)
    draft_tokens[:, i] = draft
    gi = gi_table[draft]                          # [B, 3G]
    return domino_gru_cell_triton(gi, gh, h)


def _full_loop(
    h: torch.Tensor,
    base_logits: torch.Tensor,
    z_part: torch.Tensor,
    gi_table: torch.Tensor,
    w_sh: torch.Tensor,
    w_emb2: torch.Tensor,
    draft_tokens: torch.Tensor,
) -> torch.Tensor:
    for i in range(N_SPEC):
        h = _step(
            i,
            h,
            base_logits,
            z_part,
            gi_table,
            w_sh,
            w_emb2,
            draft_tokens,
        )
    return h


def _capture_replay_time(fn, iters: int = ITERS, warmup: int = WARMUP):
    """Capture ``fn`` in an ACL graph, then time replays.  Returns (ms, capture_s)."""
    g = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    t0 = time.perf_counter()
    with torch.npu.graph(g, stream=stream, capture_error_mode="global"):
        fn()
    torch.npu.synchronize()
    capture_s = time.perf_counter() - t0
    for _ in range(warmup):
        g.replay()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3, capture_s


def main() -> None:
    print(f"torch_npu version: {getattr(torch_npu, '__version__', 'unknown')}")
    init_device_properties_triton()
    print(f"H={H} G={G} M={M} V={V} N_SPEC={N_SPEC} "
          f"B={MS} dtype={DTYPE}")

    device = "npu"
    gi_table = torch.randn(V, 3 * G, dtype=DTYPE, device=device) * 0.02
    print(f"gi_table: {gi_table.numel() * gi_table.element_size() / 1e6:.0f} MB")

    for b in MS:
        (
            w_sh,
            w_emb2,
            base_logits,
            z_part,
            h,
            draft_tokens,
        ) = _build_inputs(b, device, seed=0)

        # --- Per-step ops in isolation (each in its own graph) ---
        x0 = z_part[:, 0]
        sh0 = F.linear(h[0], w_sh)
        s_proj0 = sh0[:, :M]
        gh0 = sh0[:, M:]
        x_silu = F.silu(x0 + s_proj0)
        bias0 = F.linear(x_silu, w_emb2)
        logits0 = base_logits[:, 0] + bias0
        draft0 = logits0.argmax(dim=-1)
        gi0 = gi_table[draft0]

        ops = [
            ("w_sh linear [B,3330]", lambda: F.linear(h[0], w_sh)),
            ("z+silu [B,256]", lambda: F.silu(x0 + s_proj0)),
            ("embed_proj2 [B,V]", lambda: F.linear(x_silu, w_emb2)),
            ("base+bias [B,V]", lambda: base_logits[:, 0] + bias0),
            ("argmax [B,V]", lambda: logits0.argmax(dim=-1)),
            ("gi gather [B,3G]", lambda: gi_table[draft0]),
            ("gru cell [B,3G]", lambda: domino_gru_cell_triton(
                gi0, gh0, h
            )),
        ]

        print("-" * 72)
        print(f"B={b}: per-op graph replay (us/call):")
        sum_us = 0.0
        for name, fn in ops:
            ms, _ = _capture_replay_time(fn)
            us = ms * 1e3
            sum_us += us
            print(f"  {name:24s} {us:9.2f} us", flush=True)

        loop_ms, capture_s = _capture_replay_time(
            lambda: _full_loop(
                h,
                base_logits,
                z_part,
                gi_table,
                w_sh,
                w_emb2,
                draft_tokens,
            )
        )
        per_step_us = loop_ms * 1e3 / N_SPEC
        gap_us = per_step_us - sum_us
        print("-" * 72)
        print(
            f"B={b}: full 15-step loop graph replay: "
            f"{loop_ms:.3f} ms ({per_step_us:.2f} us/step), "
            f"capture wall {capture_s:.1f} s",
            flush=True,
        )
        print(
            f"B={b}: sum of isolated ops {sum_us:.2f} us/step; "
            f"gap {gap_us:.2f} us/step "
            f"({gap_us / max(per_step_us, 1e-9):.1%} of step)",
            flush=True,
        )

    print("=" * 72)
    print("RESULT: see table above")


if __name__ == "__main__":
    main()
