#!/usr/bin/env bash
# Domino acceleration benchmark wrapper (latency/throughput) for Qwen3-8B +
# Domino on Ascend NPU.  Use check-domino-acceptance.sh for acceptance.
set -e

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-15}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-results/domino}"
TRITON_GRU="${TRITON_GRU:-0}"

EXTRA_ARGS=()
if [ -n "$DRAFT_MODEL" ]; then
  EXTRA_ARGS+=(--draft-model "$DRAFT_MODEL")
fi
if [ "$TRITON_GRU" = "1" ]; then
  EXTRA_ARGS+=(--triton-gru)
fi
# Optional realistic datasets:
#   THROUGHPUT_DATASET=gsm8k THROUGHPUT_DATASET_PATH=/path/to/file.jsonl
if [ -n "${THROUGHPUT_DATASET_PATH:-}" ]; then
  EXTRA_ARGS+=(--throughput-dataset "${THROUGHPUT_DATASET:-random}")
  EXTRA_ARGS+=(--throughput-dataset-path "$THROUGHPUT_DATASET_PATH")
fi

exec python3 "$(dirname "$0")/../bench_domino.py" \
  --model "$MODEL" \
  --num-speculative-tokens "$NUM_SPEC_TOKENS" \
  --batch-size "$BATCH_SIZE" \
  --output-dir "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}" \
  "$@"
