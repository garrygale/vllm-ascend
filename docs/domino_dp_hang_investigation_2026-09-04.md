# Domino + MoE target DP hang (dp>1, tp=2, EP) — open investigation

Date: 2026-09-04
Status: open — reproducing and instrumenting

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

## Next step

Add an env-gated rank trace (`VLLM_DP_TRACE=1`, optional `=2` for per-dispatch
logs) at:

- `DPEngineCoreProc.run_busy_loop` and `_has_global_unfinished_reqs`
- worker `execute_model` / `sample_tokens` / `execute_dummy_batch`
- `sync_cudagraph_and_dp_padding` enter/exit
- Domino `propose()` enter/exit

Run the single-request repro with the trace enabled and wait for the hang.
The last lines from each `EngineCore_DP*` / `Worker_DP*` process should reveal
which rank entered a collective without its counterpart.

Do not conflate this issue with the separate 32-worker acceptance collapse
(see `domino_acceptance_batch_issue_2026-09-04.md`).
