# Domino on vLLM-Ascend (draftv2-030 port)

This branch carries the Domino speculative-decoding draft on top of
`v0.30.0rc2`. It pairs with the `draftv2-030` branch of the `vllm` fork
(based on `v0.30.0`); check out both at compatible commits. The migration
is documented in `migration-030/` in the workspace (commit ledger: docs
`06`/`07`; upstream delta: `04`/`05`).

Serving contract (see also `SpecForge/docs/benchmarks/domino_vllm_ascend.md`):

* `--speculative-config '{"method": "domino", "model": <draft dir>,
  "num_speculative_tokens": <block_size>, "draft_tensor_parallel_size": 1}'`
  — `num_speculative_tokens` must equal the draft's trained `block_size`
  (7 for the frozen recipe).
* The MRV2 (V2 model) runner is forced automatically for `method=domino`;
  `VLLM_USE_V2_MODEL_RUNNER=1` also works. `--enforce-eager` in the
  speculative config disables the draft/target ACL graphs for debug runs.
* The draft config's `dflash_config` carries the contract: flare fusion,
  `heterogeneous_kv: true`, per-layer `sliding_window` (all <= 2048 -- FIA
  band limit, enforced at startup), `out_proj_after_norm`, QAT
  `qat_w_bit`/`qat_w4a4_layers` (on-the-fly quantized at load), and the
  triton-table GRU requirements `use_embed_proj=true` /
  `use_hidden_proj=false`.
* Validation harness: `benchmarks/check_acceptance_service.py` (acceptance
  length over a service endpoint; this is the decay-regression gate -- see
  `domino_acceptance_batch_issue_2026-09-04.md` for the trigger classes
  and their fixes).

Deferred features (not ported; see `migration-030/06` for re-enable
recipes): HCA, VHA fold, attention sinks, DFlash2 conv, ffn_sharing
variants, shared-emb readout, fc fusion, fused-qkv / fused-context-KV /
W8A8-norm-quant perf paths, VectorCore prepare_dflash_inputs, categorical
resample.
