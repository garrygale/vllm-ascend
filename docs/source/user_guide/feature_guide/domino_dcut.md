# Domino D-Cut for Qwen3-8B

D-Cut selects how many Domino draft tokens to verify for each request. The
target uses the Ascend **V2 ModelRunner** with **PIECEWISE** graphs. Domino still
generates its checkpoint's full, fixed `block_size` on every step; its backbone
and GRU sampling loop remain eager under this graph configuration.

This implementation supports Qwen3-8B, greedy Domino draft sampling and tensor
parallelism with PP/DP/PCP/DCP sizes of one. LoRA and KV transfer are unsupported.
Prefill, mixed batches, structured outputs, resumed or preempted requests retain
the scheduled verification length.

## Configuration

Pass `dcut_config` through `additional_config`. The feature is disabled by default.

When `enabled` is absent or `false`, other D-Cut options are ignored. The worker
uses the original NPUModelRunner and AscendDominoSpeculator classes: no D-Cut
execution/sampling wrappers, probability buffers, streams, table I/O or TP
decision broadcasts are installed. Only `enabled=true` selects D-Cut subclasses.
`generate_cost_table=false` disables calibration; it does not disable D-Cut.

```json
{
  "dcut_config": {
    "enabled": true,
    "cost_table_path": "/data/domino_dcut_cost.json",
    "generate_cost_table": true,
    "profile_warmup": 2,
    "profile_samples": 5,
    "min_gain": 0.02,
    "candidate_ratios": [0.25, 0.5, 0.75, 1.0],
    "wait_for_probs": true
  }
}
```

The `generate_cost_table` switch has these behaviors:

| Value | Behavior |
| --- | --- |
| `true` | Create or resume a compatible table; explore verification lengths and collect live NPU timings. Completed rows are saved atomically. |
| `false` | Load an existing compatible table and select verification prefixes. Never collect timing samples or write the table. Missing or incompatible tables cause a startup error. |

Use these model settings alongside either mode:

```python
from transformers import AutoConfig
from vllm import LLM

draft_model = "/data/qwen3_8b_domino"  # Your Domino checkpoint.
block_size = AutoConfig.from_pretrained(draft_model).block_size
llm = LLM(
    model="Qwen/Qwen3-8B",
    speculative_config={
        "model": draft_model,
        "method": "domino",
        "draft_sample_method": "greedy",
        "num_speculative_tokens": block_size,
    },
    compilation_config={"mode": "VLLM_COMPILE", "cudagraph_mode": "PIECEWISE"},
    additional_config={"dcut_config": {
        "enabled": True,
        "cost_table_path": "/data/domino_dcut_cost.json",
        "generate_cost_table": False,
    }},
)
```

Domino selects V2 automatically in this local branch. For `vllm serve`, supply
the same dictionaries through `--speculative-config`, `--compilation-config`
and `--additional-config`.

## Calibration and performance comparison

Calibration uses real decode batches rather than startup dummy runs. A cost row
is keyed by batch size, the upper bucket of the maximum context length, and the
total number of target query tokens (one anchor plus the retained draft prefix
per request). Its authoritative cost is the median steady-state end-to-end step
latency, `step_ms`. `target_ms` and `draft_ms` are retained as diagnostic NPU
event measurements; they are not added together for decisions because that sum
omits host control, synchronization, TP broadcast and other critical-path work.

Candidate query budgets follow the paper's default ratios of 25%, 50%, 75% and
100% of the full query-token count. `candidate_ratios` can replace those ratios.
`candidate_draft_lengths` may add explicit per-request draft depths, for example
`[0, 1, 4]`; it does not replace the ratio candidates. The full scheduled length
is measured before shorter candidates for every encountered batch/context shape.

Each row discards `profile_warmup` samples and retains `profile_samples` samples.
Calibration deliberately captures selected-token probabilities and waits for NPU
events so `step_ms` includes D-Cut's probability path. A step timer starts before
the decision/controller path and is consumed at the next step after the preceding
Domino proposal event completes. This measures steady-state wall latency across
the whole step, including host and scheduler gaps. Only encountered workload
shapes are calibrated. Generate enough output tokens and repeat workloads with
representative batch sizes and context lengths before assessing inference
performance. Cost-table schema version 2 is required; regenerate older tables.

From the repository root, run the calibration helper on an Ascend machine:

```bash
python benchmarks/benchmark_domino_dcut.py \
  --draft-model /data/qwen3_8b_domino \
  --cost-table /data/domino_dcut_cost.json \
  --generate-cost-table --tp 1 --batch-size 4 \
  --context-tokens 512 --output-tokens 256 --rounds 4
```

The helper flushes the final pending timing sample using `LLM.collective_rpc`.
For direct Python use, the same callable is:

```python
def finish_dcut(worker):
    return worker.model_runner.dcut_runtime.finish_calibration()

stats = llm.collective_rpc(finish_dcut)
```

During serving, completed samples are normally consumed at the next target
execution; an unconsumed final sample does not invalidate saved rows.

Compare two fresh processes with identical model, TP, graph and batch settings:

```bash
python benchmarks/benchmark_domino_dcut.py \
  --draft-model /data/qwen3_8b_domino \
  --cost-table /data/domino_dcut_cost.json --disable-dcut

python benchmarks/benchmark_domino_dcut.py \
  --draft-model /data/qwen3_8b_domino \
  --cost-table /data/domino_dcut_cost.json --no-generate-cost-table
```

The helper reports wall time, output tokens per second and worker counters
(`decisions`, `trimmed_tokens`, `fallbacks`, `capture_skips`, `profiled_rows`).
It includes prefill time and excludes model initialization; it does not report
serving TTFT or TPOT.
Do not use calibration throughput as the steady inference result. Keep scheduler
limits and capture sizes identical between calibration and inference. Hardware,
model/draft revision, dtype, quantization, TP, graph capture sizes and scheduler
limits are fingerprinted; a mismatch requires a new table.

## Decision and fallback

Rank zero computes Domino's greedy `argmax` once, reuses the selected IDs both for
draft-to-target mapping and for gathering the selected-token probability from the
full draft vocabulary logits, and includes GRU corrections. These probabilities
are a heuristic for acceptance, not measured target acceptance probabilities.
D-Cut maximizes estimated accepted tokens per measured end-to-end `step_ms` over
calibrated query budgets. Stable allocation preserves each request's draft prefix;
`min_gain` requires a fractional improvement over the full-length estimate before
trimming. Timing buckets approximate workload cost and do not guarantee an
observed speedup.

The probability copy uses a separate stream. Before requesting it, inference
applies a capture gate using only the cost table. Capture is skipped when no
shorter calibrated budget can beat the full budget by `min_gain`, even under the
best possible acceptance probabilities. This upper-bound test is safe because a
shorter prefix cannot have greater estimated accepted-token utility than the full
prefix. It avoids probability work for flat-cost small batches and tail phases;
calibration always captures so the measured table includes that overhead.

When capture is useful, inference waits for the current proposal's probabilities
by default (`wait_for_probs=true`) so the decision can be made before graph
dispatch. This synchronization may reduce CPU/NPU overlap; measure the throughput
tradeoff on Ascend. Set `wait_for_probs=false` to test nonblocking readiness
checks; async scheduling may then cause frequent fallbacks. The full scheduled
length is retained when capture is gated off, a nonblocking copy is not ready,
the request set changed, probabilities are invalid, the full baseline row is
missing, or the context/batch has no usable cost row. Rank zero broadcasts the
decision and capture flag so all TP ranks verify identical prefixes.

Only a worker-local scheduler-output copy is shortened before graph dispatch.
Target input positions, logits indices and rejection sampling use the shortened
length; scheduler accounting uses returned token counts and keeps the original
scheduled output. The full Domino draft block is retained for the next verification step.

## Tests

CPU contract tests require NumPy, PyTorch and pytest:

```bash
pytest --confcutdir=tests/ut/spec_decode/dcut tests/ut/spec_decode/dcut -q
```

On Ascend, test actual V2 graph execution, token equality, calibration and read-only
table loading with a supplied Domino checkpoint:

```bash
pytest tests/e2e/nightly/single_node/domino_dcut \
  --domino-draft-model /data/qwen3_8b_domino --domino-dcut-tp 1 -v
```

Repeat with `--domino-dcut-tp 4` on four devices. The integration test substitutes
synthetic costs after calibration to force prefix trimming; its result validates
correctness and does not establish performance improvement.
