# Domino acceptance collapse under 32-way concurrency

Date: 2026-09-04
Status: SWA-specific replay fix applied 2026-09-09; NPU validation pending

## Context

Target service: Qwen3.6-35B-A3B + Domino draft on Ascend NPU, running
`vllm` / `vllm-ascend` on the `codex/DRAFT_qwen36_35B` branches.

- Launch: `dp=1`, `tp=2`, MoE/EP enabled, `draft_tensor_parallel_size=1`.
- Temperature 0 (greedy). Prefix caching was disabled for the later
  experiments unless stated otherwise.
- The same Domino draft checkpoint works well when serving Qwen3-8B
  (i.e. the draft model itself is not believed to be broken).
- The MRV2 Mamba/GDN support on Ascend was only ported into this branch
  on 2026-09-03 (`1852c307e fix(gdn): support Mamba/GDN layers in the
  Ascend V2 runner`).

## Symptom

Acceptance is healthy in the following configurations:

- Single long generation: stable `cum_len` around 4.27, per-position
  rates `[0.781, 0.641, 0.536, 0.430, 0.359, 0.274, 0.245]`.
- 16 concurrent workers over the full test set: no decay; server KV
  cache usage around 40%.

Acceptance collapses with 32 concurrent workers:

- Running mean acceptance length decays gradually to ~1.01 and
  per-position rates go to ~0.
- Server reports ~32 requests running, KV cache usage around 80%, and
  no `Preemptions` / `Deferred` messages.
- Reproduced on graph and eager paths.

## Experiments and results

### Prefix caching — ruled out

- Restarting with `--no-enable-prefix-caching` did not change the
  collapse.
- Re-running the same single-request prompt twice did not degrade
  acceptance.

### Max-token / sequence-length dependence

At 32 workers, increasing max tokens from 256 to 512 decreases
acceptance. Initially suspected the draft sliding-window recipe
`[3072, 2048, 512, 512, 1024, 1024, 3072]`, but:

- 16 workers with the identical checkpoint/config stays healthy, which
  rules out a pure window/sequence-length effect.
- Sliding-window config was never modified during testing.

### Triton AutoBlockify — ruled out

Applied the upstream vllm-ascend fix #15118 (disable AutoBlockify for
the MRV2 rejection kernels):

- Local commit `02a3bac76` added
  `has_auto_blockify_blacklist_op=True` to the two rejection-sampling
  kernel launches.
- The installed Triton Ascend rejects that keyword at launch:
  `keyword argument has_auto_blockify_blacklist_op was specified but
  unrecognised`.
- Reverted in `41594be5b` so the service launches again.
- Setting `TRITON_ALL_BLOCKS_PARALLEL=0` did not fix the collapse.
- The same draft works on Qwen3-8B, so a generic Triton issue is
  considered unlikely.

### Other observation

While running a single-request probe, one prompt's reported running
acceptance jumped to 8.00 (all draft positions accepted) after roughly
150 draft rounds and stayed there until the end. Not yet explained;
could be genuine degenerate repetition or a state/bookkeeping bug.

## 2026-09-05 update

The DP hang is fixed separately (see `domino_dp_hang_investigation_2026-09-04.md`),
but the acceptance issue remains under higher worker counts.

### Confirmed results

- dp=1/16: healthy, including with KV usage forced to ~80%.
- dp=1/32, full eager: decays; near-end per-position acceptance falls to ~0.
- dp=1/32 with replacement delays 0/100/500 ms: all three still decay. This
  rules out a simple finished-request/cleanup-rate race.
- dp=2/16 graph mode behaves between dp=1/16 and dp=1/32: acceptance decays,
  partially recovers, then decays again.
- The 16→32 threshold is around 28 in the tested setup and may drift.
- Same prompt (humaneval/159) is readable in a healthy single request but
  random throughout when captured during the degraded 32-worker window.

### Draft sliding-window experiment

Changing all draft sliding-window layers to full attention significantly
changes draft accuracy (as expected), but the worker-count instability
largely disappears:

- Full-attention draft remained stable through 64 workers and showed less
  decay than the sliding-window draft at 32 workers.
- A degraded full-attention sample kept the prompt prefix intact but could
  produce random digit-like continuation (e.g. `2   2   19  2  2 2     2`).

This implicates the non-causal sliding-window draft attention path rather
than a general batch-size bug in the draft backbone or target model alone.

### Debug hook status

`VLLM_DOMINO_DEBUG=1` now prints a hook-active marker and collapse-step rows
from `AscendDominoSpeculator`. Hook activation is confirmed, but per-request
rows are interleaved with the 32-worker background and have not yet been
isolated. A request-scoped/timestamped dump is the next step if needed.

### Next experiments

- Cap draft sliding windows to `<=2048` (replace the two `3072` windows).
- Keep 512/1024 sliding layers and make only the large-window layers full
  attention.
- Collect exact launch command, `max_num_seqs`, `mamba_cache_mode`,
  async-scheduling setting, and `triton.__version__`.
- Determine whether corrupted output begins only after context length exceeds
  the 2048 FIA band mask boundary.

## 2026-09-07 resolution

### Two separate triggers

1. **Non-causal draft windows above 2048.** The trained
   `[3072, 2048, 512, 512, 1024, 1024, 3072]` recipe corrupts the FIA
   non-causal band path (`sparse_mode=4` with the fixed `2048x2048` band
   mask) under sustained load, including full eager. Capping the two 3072
   layers to 2048 removes the corruption and is stable through 64 workers in
   eager mode; this is why the earlier full-attention and 2048-cap draft
   experiments changed the instability.

2. **Target FULL-graph replay with stale padded GDN/Mamba rows.** With
   windows capped to <=2048, dp=2/48 graph mode still decayed while eager
   stayed healthy. Forcing the Domino draft eager (vllm-ascend `827ca25d7`)
   did **not** fix it, proving the corruption is in the target graph, not the
   draft graph. Attention-side graph fixes (captured block tables and max
   workspace, `8050f9801`), fine-grained graph capture sizes, and
   `--no-async-scheduling` also did not fix it.

The graph bug is the same class as the MTP/GDN bug fixed by vllm-ascend
PR #15529: uniform FULL decode graphs collapse padded rows to the live
request count in `_pad_query_start_loc_for_fia`, while GDN graphs capture
metadata at request granularity. After requests finish, replayed padded
slots can still hold persistent conv/recurrent state indices from freed
blocks.

Fix (`aa66a707e`):

- `vllm_ascend/worker/v2/model_runner.py`: preserve the captured request
  shape for uniform decode FULL graphs instead of collapsing padding into a
  synthetic dummy request.
- `vllm_ascend/worker/v2/model_states/mamba_hybrid.py`: mark padded
  Mamba/GDN rows as speculative dummies
  (`num_decode_draft_tokens = num_spec`) so replay uses the same pure-spec
  GDN path as capture and refreshes/nullifies padded state rows every replay.
- `vllm_ascend/worker/v2/aclgraph_utils.py`: expose `embed_input_ids` on the
  graph wrapper.

### Validation

The previously failing graph configuration (dp=2/48 with windows <=2048) no
longer shows the acceptance decay after `aa66a707e`.

## Remaining hypotheses

1. Non-causal sliding-window draft attention (FIA `sparse_mode=4` band path)
   corrupts neighboring/stateful cache memory under high worker counts.
2. A per-sequence state bug in the MRV2 Mamba/GDN path that only appears when
   enough requests share a batch.
3. KV/draft block reuse is still involved but only when combined with the
   sliding-window draft cache path.

Upstream fixes that are absent from the current vllm/vllm-ascend
branches and may be relevant when revisiting:

- vLLM #48245: `num_output_placeholders` preemption underflow.
- vLLM #49736: GPU<->CPU syncs in MRV2 Mamba state.
- vLLM #49757: dummy runs writing Mamba state through stale block rows.
- vLLM #50327: scalar Mamba state update with int32 mappings.
- vLLM #50432: cross-block `num_accepted_tokens` race in align mode.
- vLLM #51865: uniform-decode dispatch requires all requests decoding.

## Next steps when resuming

- Run 16 workers with KV usage forced to ~80% (lower
  `--gpu-memory-utilization` or `--max-model-len`) to separate batch-size
  effects from memory-pressure/block-reuse effects.
- Sweep workers 24/28 at the current settings to find the threshold.
- Check whether generated text stays coherent while acceptance drops
  (draft-only corruption) or also degrades (target GDN state).
- Collect the exact service launch command, startup log (max_num_seqs,
  mamba cache mode, async scheduling/graphs), and
  `triton.__version__`.

## 2026-09-09 update

The earlier "resolved" status was too broad.  A checkpoint with the same
Domino weights and the same target model, but with every draft layer changed
from sliding-window attention to full attention, is stable at high
concurrency.  This isolates the remaining failure to the draft-side
sliding-window path rather than the target GDN/Mamba state or the target
graph alone.

The current branch also regressed an upstream vllm-ascend full-graph fix when
the v0.26 compatibility cleanup (`4bf5e099f`) removed
`_update_draft_attn_metadata`.  With a live request count below the captured
graph bucket, the parallel-draft metadata builder clamps cumulative query
lengths at the live count and the DFlash input kernel leaves padded KV
lengths at zero.  For example, a captured 6-request/7-token draft graph has

```text
actual_seq_lengths_q = [7, 14, 21, 28, 35, 42]
seq_lens_list        = [s0, s1, s2, s3, s4, s5]
```

but a replay with three live requests produces

```text
actual_seq_lengths_q = [7, 14, 21, 21, 21, 21]
seq_lens_list        = [s0, s1, s2, 0, 0, 0]
```

This is only reachable after a request finishes and the graph is replayed
with padding.  FIA band mode (`sparse_mode=4`, used by the non-causal
sliding-window draft layers) derives its tiling and band geometry from these
lengths, so the mismatch can corrupt the draft output.  Full attention uses
the non-band FIA path and tolerates the same padding, matching the
full-attention A/B result.

The fix restores the captured dummy-row geometry for DFlash, DSpark, and
Domino before the full-graph parameter update:

- cumulative query lengths cover the full padded graph token count;
- padded KV lengths match the captured dummy rows (`num_query_per_req`)
  instead of remaining zero;
- the padded rows continue to use block 0 and their outputs are ignored.

The independent FIA mask limit still applies: non-causal windows above 2048
are not supported by the fixed `2048x2048` band mask and must remain capped
to `<=2048` until the operator supports a larger band.

## Related separate issue

`dp > 1` with `tp=2` and EP enabled never finishes a single incoming
request (hang). Tracked separately; do not conflate with this issue.

The DP hang itself was resolved separately on 2026-09-04/05
(see `domino_dp_hang_investigation_2026-09-04.md`); it is unrelated to the
acceptance issue documented above.
