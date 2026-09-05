# Domino + MoE target DP hang (dp>1, tp=2, EP) — resolved investigation

Date: 2026-09-04
Status: fixed — Domino DP full-graph hang resolved on MRV2

## Context

Target service: Qwen3.6-35B-A3B + Domino draft on Ascend NPU, running
`vllm` / `vllm-ascend` on the `codex/DRAFT_qwen36_35B` branches.

- Launch: `dp=2` (any `dp>1`), `tp=2`, MoE/EP enabled,
  `draft_tensor_parallel_size=1`, speculative tokens = 7.
- The V2 model runner is forced with `VLLM_USE_V2_MODEL_RUNNER=1`.
- No speculative decoding under the same DP/EP launch works fine.
- DP=1 with the same Domino draft works fine.

## Symptom

Even a single incoming request never completes:

1. `shm_broadcast`: “No available shared memory broadcast block found in
   60 seconds...”
2. Eventually: `TimeoutError: RPC call to sample_tokens timed out`.
3. Reproduced with:
   - graph and eager draft paths;
   - sync and `--no-async-scheduling`;
   - prefix caching enabled and disabled;
   - both `dp=2` and `dp>2` (user observed with `dp=2`; DP=1 works).

## Attempted fixes (all tested, hang remains)

### vllm-ascend #13600 / update-stream unification

Commit `b1d2e239d` (“[BugFix][MRV2] Unify update_stream across main and
draft to fix fullgraph deadlock”) plus local `9874dbacb` (“share main-model
update stream with Domino graphs”).

Main model and Domino graphs now use one shared `update_stream`; draft graph
managers no longer create private streams.

Result: service still hangs on a single request under dp>1.

### vllm Domino skip-DP-sync commit

Commit `ba5105d7e8` (“fix(spec_decode): skip per-step DP dispatch sync for
dense Domino drafts”).

`DominoSpeculator._skip_draft_dp_sync()` returns True; `propose()` dispatches
the dense draft locally and skips the per-step `dispatch_cg_and_sync_dp`
all-reduce for real decode and dummy runs. DFlash/DSpark retain the sync
because their draft backbones can be MoE.

Rationale: a dense Domino draft runs no cross-DP collectives itself, so the
second per-step DP all-reduce can deadlock when DP ranks are idle/active
asymmetrically.

Result: service still hangs on a single request under dp>1.

## Current understanding

### Live V2 path

- Ascend `worker.py` chooses `NPUModelRunnerV2` when
  `VLLM_USE_V2_MODEL_RUNNER` is set.
- Live runner: `vllm-ascend/vllm_ascend/worker/v2/model_runner.py`, subclass
  of `vllm/vllm/v1/worker/gpu/model_runner.py`.
- V1 machinery (`model_runner_v1.py`, Ascend `_sync_metadata_across_dp`) is
  not live and is not the fix site.

### Engine cadence

`DPEngineCoreProc.run_busy_loop()` (vllm `v1/engine/core.py`):

1. `_process_engine_step()` (real local work)
2. if nothing executed and another rank still has work: `execute_dummy_batch()`
3. `_has_global_unfinished_reqs()` — engine-level gloo all-reduce every 32
   steps only.

Workers spanning DP ranks run real or dummy main-model forwards. Every real
and dummy forward calls `dispatch_cg_and_sync_dp` on the DP cpu group and
enters MoE/EP collectives (HCCL) inside the main Qwen3.6 forward.

The stuck path is therefore collective alignment between the real rank and the
idle dummy rank, or between the EngineCore loop and the worker response
queues, rather than Domino itself (dense; no EP).

### sample_tokens timeout chain

- `shm_broadcast` warning: `vllm/distributed/device_communicators/shm_broadcast.py`
- `TimeoutError: RPC call to sample_tokens timed out`:
  `vllm/v1/executor/multiproc_executor.py` (~line 392)
- A stuck EngineCore stops draining its workers' response ring; the worker
  response lands nowhere and times out after 60s.

So the primary suspect is an EngineCore or worker collective hang, not the
HTTP/front-end path.

## Root cause

The hang was on the Domino draft **ACL full-graph** path, not in the DP
dispatch all-reduce or in the EngineCore loop.

`VLLM_DP_TRACE=1` showed that all four workers (two TP ranks per DP rank)
reached `dflash.run_fullgraph_enter` and `dflash.run_fullgraph_exit`
inside the Domino draft graph manager, but never reached the post-replay
graph-parameter update. The blocking point was the
`DFlashAclGraphManager.run_fullgraph()` context setup immediately after
`super().run_fullgraph(desc)`:

- `run_fullgraph()` built `num_tokens_across_dp` with `device=self.device`
  (an NPU tensor).
- It then entered vLLM's `set_forward_context()` with that tensor.
- vLLM's `DPMetadata.make()` treats that argument as a CPU tensor
  (`num_tokens_across_dp_cpu`) and evaluates
  `assert num_tokens_across_dp_cpu[dp_rank] == batchsize`.
- On an NPU tensor, that assert becomes a device sync immediately after an
  asynchronous ACL graph replay, and the sync never completed.

The main-model graph manager and the Eagle graph manager both create this
tensor on CPU; only the DFlash/Domino graph manager used `device=self.device`.

## Fix

Pass a CPU tensor to `set_forward_context()` in
`vllm_ascend/worker/v2/spec_decode/dflash/aclgraph.py`, matching the other
graph managers:

```python
num_tokens_across_dp = torch.full([self.speculator.dp_size], num_tokens)
```

Relevant commits:

- vllm-ascend `80a34506f` — pass CPU `num_tokens_across_dp` to draft graph
  context (the direct fix).
- vllm-ascend `7fd05c2d5` — stage-by-stage trace inside
  `DFlashAclGraphManager.run_fullgraph()` (diagnostic, can be removed after
  validation).
- vllm-ascend `1da71a0cb` — share main/draft ACL update streams and clear
  stale GDN conv state.
- vLLM `9c627fd7f9` — Domino skips the second DP dispatch all-reduce but
  reuses the main model's agreed padded batch geometry for graph dispatch.
- vLLM `2711419f15` / `5428ebd3fd` — `VLLM_DP_TRACE` instrumentation.

## Validation

With the CPU-tensor fix:

- `dp=2`, `tp=2`, MoE/EP enabled, Domino draft, MRV2 full graph: the single
  request completes and produces results; the service no longer hangs.
- The four workers now continue through
  `ascend_dflash.run_fullgraph_update_done`.

The 32-worker acceptance collapse tracked in
`docs/domino_acceptance_batch_issue_2026-09-04.md` is a separate issue and
should be re-tested after this fix.

## Notes for future debugging

`VLLM_DP_TRACE=1` remains in the branches used to validate the fix. The
diagnostic trace lines can be removed once validation is complete. Future
graph/Domino DP work should keep `num_tokens_across_dp` on CPU wherever it is
fed into vLLM's forward-context / `DPMetadata` machinery.
