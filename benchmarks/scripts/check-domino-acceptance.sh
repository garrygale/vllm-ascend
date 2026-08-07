#!/usr/bin/env bash
# Domino acceptance-length check against an already-running vLLM server.
set -e

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:?set SERVED_MODEL_NAME}"
DATASET="${DATASET:-gsm8k}"
DATASET_PATH="${DATASET_PATH:?set DATASET_PATH to the dataset file}"
NUM_PROMPTS="${NUM_PROMPTS:--1}"
OUTPUT_DIR="${OUTPUT_DIR:-results/domino_acceptance}"

exec python3 "$(dirname "$0")/../check_domino_acceptance.py" \
  --server-host "$SERVER_HOST" \
  --server-port "$SERVER_PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --dataset "$DATASET" \
  --dataset-path "$DATASET_PATH" \
  --num-prompts "$NUM_PROMPTS" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
