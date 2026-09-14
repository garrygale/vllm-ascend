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
contains batch size, the upper bucket of the maximum context length, actual
target query token count, median target-plus-sampling time and median full Domino
proposal time. Proposal timing includes context KV updates. For each encountered
batch/context, the full scheduled length is measured before shorter prefixes.
Default candidate draft lengths are the distinct values `0, 1, 2, 4, block_size`,
bounded by `block_size`; override them using `candidate_draft_lengths`.

Each row discards `profile_warmup` samples and retains `profile_samples` samples.
Calibration deliberately waits for NPU events and may reduce throughput while
exploring shorter verification lengths. Only encountered workload shapes are
calibrated. Generate enough output tokens and repeat workloads with representative
batch sizes and context lengths before assessing inference performance.

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
(`decisions`, `trimmed_tokens`, `fallbacks`, `profiled_rows`). It includes prefill
time and excludes model initialization; it does not report serving TTFT or TPOT.
Do not use calibration throughput as the steady inference result. Keep scheduler
limits and capture sizes identical between calibration and inference. Hardware,
model/draft revision, dtype, quantization, TP, graph capture sizes and scheduler
limits are fingerprinted; a mismatch requires a new table.

## Decision and fallback

Rank zero collects the greedy-selected probability from Domino's full draft
vocabulary logits, including GRU corrections, before draft-to-target ID mapping.
These probabilities are a heuristic for acceptance, not measured target
acceptance probabilities. D-Cut maximizes estimated accepted tokens per measured
target-plus-draft time over calibrated query budgets. Stable allocation preserves
each request's draft prefix; `min_gain` requires a fractional improvement over
the full-length estimate before trimming. Timing buckets approximate workload
cost and do not guarantee an observed speedup.

The probability copy uses a separate stream. Inference waits for the current
proposal's probabilities by default (`wait_for_probs=true`) so the decision can
be made before graph dispatch. This synchronization may reduce CPU/NPU overlap;
measure the throughput tradeoff on Ascend. Set `wait_for_probs=false` to test
nonblocking readiness checks; async scheduling may then cause frequent fallbacks.
The full scheduled length is retained when a nonblocking copy is not ready,
the request set changed, probabilities are invalid, the full baseline row is
missing, or the context/batch has no usable cost row. Rank zero broadcasts the
decision so all TP ranks verify identical prefixes.

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
