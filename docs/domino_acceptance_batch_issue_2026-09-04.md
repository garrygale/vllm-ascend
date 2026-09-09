# Domino acceptance collapse under 32-way concurrency (resolved)

Date: 2026-09-04
Status: acceptance decay fixed 2026-09-07 (`aa66a707e`); graph-mode
target-state corruption fixed in two 2026-09-09 follow-ups (fixed-row GDN
state indexing and no-spec FULL-replay reset), with NPU end-to-end
confirmation pending

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

### 2026-09-09 follow-up: normal acceptance but repeated garbage tokens

After `aa66a707e`, acceptance counters looked correct, but generated text could
become repeated garbage after roughly 60 requests, usually starting midway
through a response. This exposed a second graph-padding bug in the same
function.

`_pad_query_start_loc_for_fia` originally decided "uniform decode graph" from
total token count alone:

```text
num_tokens_padded == descriptor_num_reqs * decode_query_len
```

A mixed prefill/decode batch can satisfy that equation even when no real
request has decode query length. With descriptor 4, `decode_query_len=8`, and
real query lengths `[4, 12, 16]`, the total is 32, so the old code took the
uniform branch and produced:

```text
query_start_loc = [0, 4, 16, 32, 40]
```

The graph only has 32 input tokens, but the final padded row starts at 40.
GDN metadata then describes a different request/token topology than the
captured graph, so recurrent/conv state rows are refreshed at the wrong
positions. Because that state survives across requests, corruption appears
only after request churn and can leave acceptance counters normal while
target hidden states produce repeated garbage.

Fix (`3bc298c3e`, port of vllm-ascend PR #15707):

- require every real query length to equal `decode_query_len` before taking
  the uniform padding path;
- otherwise use the mixed-batch dummy-row layout, which gives the example
  above `query_start_loc = [0, 4, 16, 32, 32]`.

The parallel-drafting `seq_lens` override is intentionally retained: this
branch's vLLM base does not yet pass a CPU draft `seq_lens` into
`build_attn_metadata`, so removing it would replace real draft lengths with
`max_seq_len`.

The same upstream PR also fixes a second long-context layout bug in the
Qwen3.5/3.6 fused MRoPE path. The kernel reads three contiguous T/H/W
planes (`[3, T, D]`), but text-only MRV2 forwards can pass 1D positions;
indexing the cache with them yields `[T, D]`, so the H/W offsets read the
wrong cache rows. `patch_qwen3_5.py` now expands 1D positions to three
identical planes before the cache lookup. This can otherwise corrupt
attention only after positions grow, matching the late/mid-response
garbage symptom.

This follow-up still needs NPU confirmation on the long-running service test.

### 2026-09-09 follow-up: fixed-row GDN recurrent state indexing

The remaining graph-mode garbage is a state-indexing bug in the existing
AscendC recurrent GDN operator, not another padding-topology bug.

`spec_state_indices_tensor` is a fixed `[request, num_spec + 1]` table. The
operator previously received `spec_state_indices_tensor.flatten()` and indexed
it with the cumulative token offset:

```text
initial_state_idx = seq0 + num_accepted_tokens[request] - 1
output_state_idx  = seq0 + local_token_idx
```

That is correct only when every verification row has the full
`num_spec + 1` width. When the scheduler truncates a row, or graph padding
makes a row shorter, `seq0` is no longer `request * (num_spec + 1)`. The
initial-state lookup and the per-token state writes then cross into another
request's row. The builder had additionally clamped `num_accepted_tokens` to
the current row length, which hid the invalid access but selected the wrong
recurrent state after a previous step had accepted more tokens than the
current verification width.

This matches the observed failure mode: acceptance counters stay plausible,
but the persistent target GDN state is corrupted and the generated text
becomes garbage only after enough request churn for a short row to occur.

The fix ports the fixed-row semantics from the unmerged D-Cut GDN operator
(vllm-ascend PR #15207) into the existing operator instead of adding the
full new operator family:

- pass the 2-D `[B, S]` state table to `npu_recurrent_gated_delta_rule`
  instead of flattening it;
- accept `ssm_state_indices` as either legacy `[T]` or fixed `[B, S]`;
- carry `S` in tiling data and index the initial state as
  `request * S + accepted_token - 1` and output states as
  `request * S + local_token`;
- clamp accepted counts to `[1, num_spec + 1]`, not to the current row
  length;
- add NPU coverage for a `[2, 1]` verification batch whose first request has
  `num_accepted_tokens = 3`.

The legacy 1-D path is unchanged for non-speculative decode. The NPU
long-running service test is still required to confirm the fix end-to-end.

### 2026-09-09 follow-up: reset the captured spec branch on every no-spec replay

The fixed-row state indexing fix removed the acceptance decay, but graph mode
could still produce correct acceptance counters with garbage text after
request churn. The user-observed threshold was important: the failure did not
occur below 16 concurrent requests per DP rank, but appeared reliably above
it.

The remaining bug was in the Ascend GDN metadata builder's FULL-graph replay
contract. A FULL graph captured with speculative decoding contains the
speculative conv1d/recurrent tasks. At replay time those tasks consume
persistent inputs (`spec_state_indices_tensor`, `spec_query_start_loc`,
`num_accepted_tokens`, and `spec_actual_seq_lengths`) rather than rebuilding
Python-side branches. The old code reset those inputs only for a pure
non-spec decode replay. A mixed prefill/decode batch, or a batch with no
runtime draft tokens, therefore replayed the captured spec tasks with stale
metadata and advanced persistent GDN state belonging to another request.
That is why the corruption needed both graph mode and enough concurrency to
create mixed batches; below 16 requests the scheduler rarely formed one.

The fix matches upstream vllm-ascend main:

- reset the captured spec inputs whenever `num_spec_decodes == 0`, before
  prefill/decode metadata is built, instead of only inside the pure non-spec
  decode branch;
- when a dynamic-SD batch contains no runtime draft tokens at all, clear the
  spec masks instead of treating zero-draft rows as speculative rows;
- keep Domino on the base `1 + 2 * num_spec` reorder threshold. The local
  `num_spec` override was too small for the target's `1 + num_spec`
  verification width.

Regression tests cover the mixed-prefill no-spec replay, zero-draft dynamic
SD rows, and Domino's reorder threshold. NPU end-to-end confirmation is still
required.

## Remaining hypotheses

1. Non-causal sliding-window draft attention (FIA `sparse_mode=4` band path)
   corrupts neighboring/stateful cache memory under high worker counts.
2. ~~A per-sequence state bug in the MRV2 Mamba/GDN path that only appears
   when enough requests share a batch.~~ Addressed by the fixed-row state
   indexing change above; NPU confirmation pending.
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

## Related separate issue

`dp > 1` with `tp=2` and EP enabled never finishes a single incoming
request (hang). Tracked separately; do not conflate with this issue.

The DP hang itself was resolved separately on 2026-09-04/05
(see `domino_dp_hang_investigation_2026-09-04.md`); it is unrelated to the
acceptance issue documented above.
