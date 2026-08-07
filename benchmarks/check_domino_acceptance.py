#!/usr/bin/env python3
# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Domino acceptance-length check against an already-running vLLM server.

You launch ``vllm serve`` with the Domino speculative config yourself; this
script only needs the host, port, and served model name.  It runs
``vllm bench serve`` on a dataset (GSM8K / HumanEval / ShareGPT / HF /
SpeedBench) and reports the spec-decode acceptance rate and length that vLLM
exposes from its server metrics.

Example:
    python benchmarks/check_domino_acceptance.py \\
        --server-host 127.0.0.1 --server-port 8000 \\
        --served-model-name Qwen/Qwen3-8B \\
        --dataset gsm8k --dataset-path /path/to/gsm8k_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone


def _to_sharegpt(path: str, fmt: str, output_dir: str) -> str:
    """Convert a GSM8K/HumanEval JSONL into a ShareGPT-format JSONL."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt = item["prompt"] if fmt == "humaneval" else item["question"]
            rows.append(
                {
                    "conversations": [
                        {"role": "user", "value": prompt},
                        {"role": "assistant", "value": ""},
                    ]
                }
            )
    out = os.path.join(output_dir, f"sharegpt_{fmt}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Converted {fmt} dataset ({len(rows)} prompts) -> {out}")
    return out


def run_cmd(cmd: list[str], log_path: str | None = None) -> str:
    print("\n+ " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    combined = (proc.stdout or "") + (proc.stderr or "")
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(combined)
    if proc.returncode != 0:
        print(combined[-6000:])
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument(
        "--dataset",
        default="gsm8k",
        choices=["speed_bench", "sharegpt", "hf", "gsm8k", "humaneval"],
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--hf-name", default=None)
    parser.add_argument("--num-prompts", type=int, default=-1)
    parser.add_argument("--output-dir", default="results/domino_acceptance")
    parser.add_argument(
        "--output-len",
        type=int,
        default=1024,
        help="generous max-tokens cap; the server stops at EOS, so this is "
        "an upper bound, not a forced length",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_name = args.dataset
    dataset_path = args.dataset_path
    if dataset_name in ("gsm8k", "humaneval"):
        dataset_path = _to_sharegpt(dataset_path, dataset_name, args.output_dir)
        dataset_name = "sharegpt"

    print(
        f"Running acceptance against {args.server_host}:{args.server_port} "
        f"(served model: {args.served_model_name}) ..."
    )
    results_json = os.path.join(args.output_dir, "acceptance_results.json")
    bench_cmd = [
        "vllm",
        "bench",
        "serve",
        "--model",
        args.served_model_name,
        "--host",
        args.server_host,
        "--port",
        str(args.server_port),
        "--dataset-name",
        dataset_name,
        "--dataset-path",
        dataset_path,
        "--num-prompts",
        str(args.num_prompts),
        "--result-filename",
        results_json,
    ]
    if dataset_name == "hf":
        bench_cmd += ["--hf-name", args.hf_name]
    if dataset_name == "sharegpt":
        bench_cmd += ["--sharegpt-output-len", str(args.output_len)]
    run_cmd(bench_cmd, log_path=os.path.join(args.output_dir, "acceptance.log"))

    result: dict = {}
    try:
        with open(results_json, encoding="utf-8") as f:
            result = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not parse acceptance results: {exc}")

    acceptance = {
        "dataset": args.dataset,
        "server_host": args.server_host,
        "server_port": args.server_port,
        "served_model_name": args.served_model_name,
        "acceptance_rate": result.get("spec_decode_acceptance_rate"),
        "acceptance_length": result.get("spec_decode_acceptance_length"),
        "num_drafts": result.get("spec_decode_num_drafts"),
        "draft_tokens": result.get("spec_decode_draft_tokens"),
        "accepted_tokens": result.get("spec_decode_accepted_tokens"),
        "requests_per_second": result.get("request_throughput"),
        "output_tokens_per_second": result.get("output_throughput"),
        "total_output_tokens": result.get("total_output_tokens"),
        "per_position_acceptance_rates": result.get(
            "spec_decode_per_position_acceptance_rates"
        ),
        "results_file": results_json,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    rate = acceptance["acceptance_rate"]
    length = acceptance["acceptance_length"]
    print(
        f"Acceptance rate: {rate if rate is not None else 'N/A'}  "
        f"Acceptance length: {length if length is not None else 'N/A'}"
    )

    summary_path = os.path.join(args.output_dir, "acceptance_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(acceptance, f, indent=2)
        f.write("\n")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
