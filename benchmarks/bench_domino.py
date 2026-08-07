#!/usr/bin/env python3
# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Domino speculative-decoding benchmark for Qwen3-8B on Ascend NPU.

Wraps the vLLM benchmark CLI (``vllm bench latency`` / ``vllm bench
throughput``) and runs each test twice: once with the bare target model
(autoregressive baseline) and once with the Domino draft model enabled via
``--speculative-config``.  Reports the acceleration ratio.

Examples:
    python benchmarks/bench_domino.py \\
        --model Qwen/Qwen3-8B \\
        --draft-model /path/to/domino/checkpoint \\
        --num-speculative-tokens 15

    # measure the optimized Triton GRU path (same env as the service)
    python benchmarks/bench_domino.py ... --triton-gru

    # realistic throughput workload from a real dataset
    python benchmarks/bench_domino.py ... \\
        --throughput-dataset humaneval \\
        --throughput-dataset-path /path/to/human-eval-v2-20210705.jsonl \\
        --throughput-output-len 128
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


def build_spec_config(args: argparse.Namespace) -> dict:
    return {
        "method": "domino",
        "model": args.draft_model,
        "num_speculative_tokens": args.num_speculative_tokens,
    }


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


def _env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.triton_gru:
        env["VLLM_ASCEND_DOMINO_TRITON_GRU"] = "1"
    return env


def run_cmd(cmd: list[str], env: dict[str, str], log_path: str | None = None):
    print("\n+ " + " ".join(cmd))
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(combined)
    if proc.returncode != 0:
        print(combined[-6000:])
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return combined


def _common_engine_args(args: argparse.Namespace) -> list[str]:
    cmd = [
        "--model",
        args.model,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--max-model-len",
        str(args.max_model_len),
    ]
    if args.load_format:
        cmd += ["--load-format", args.load_format]
    if args.eager:
        cmd += ["--enforce-eager"]
    return cmd


def run_latency(
    args: argparse.Namespace,
    spec_json: str | None,
    out_json: str,
    env: dict[str, str],
) -> dict:
    cmd = [
        "vllm",
        "bench",
        "latency",
        *_common_engine_args(args),
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--batch-size",
        str(args.batch_size),
        "--num-iters-warmup",
        str(args.num_iters_warmup),
        "--num-iters",
        str(args.num_iters),
        "--output-json",
        out_json,
    ]
    if spec_json:
        cmd += ["--speculative-config", spec_json]
    run_cmd(cmd, env, log_path=out_json + ".log")
    with open(out_json, encoding="utf-8") as f:
        return json.load(f)


def run_throughput(
    args: argparse.Namespace,
    spec_json: str | None,
    out_json: str,
    env: dict[str, str],
) -> dict:
    dataset_name = args.throughput_dataset
    dataset_path = args.throughput_dataset_path
    converted = False
    if dataset_name in ("gsm8k", "humaneval"):
        if not dataset_path:
            raise ValueError(
                f"--throughput-dataset {dataset_name} requires "
                "--throughput-dataset-path"
            )
        dataset_path = _to_sharegpt(dataset_path, dataset_name, os.path.dirname(out_json))
        dataset_name = "sharegpt"
        converted = True

    cmd = [
        "vllm",
        "bench",
        "throughput",
        *_common_engine_args(args),
        "--dataset-name",
        dataset_name,
        "--num-prompts",
        str(args.num_prompts),
        "--output-json",
        out_json,
    ]
    if dataset_name == "random":
        if args.throughput_output_len is None:
            raise ValueError(
                "--throughput-output-len is required for "
                "--throughput-dataset random"
            )
        cmd += [
            "--input-len",
            str(args.throughput_input_len),
            "--output-len",
            str(args.throughput_output_len),
        ]
    else:
        cmd += ["--dataset-path", dataset_path]
        if converted:
            output_len = (
                args.throughput_output_len
                if args.throughput_output_len is not None
                else 1024
            )
            cmd += ["--output-len", str(output_len)]
        elif args.throughput_output_len is not None:
            # Dataset-provided output lengths take precedence when available.
            cmd += ["--output-len", str(args.throughput_output_len)]
        if dataset_name == "hf":
            cmd += ["--hf-name", args.throughput_hf_name]
    if spec_json:
        cmd += ["--speculative-config", spec_json]
    run_cmd(cmd, env, log_path=out_json + ".log")
    with open(out_json, encoding="utf-8") as f:
        return json.load(f)


def _ratio(baseline: float, spec: float) -> float | None:
    if spec > 0:
        return baseline / spec
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--draft-model",
        required=True,
        help="Domino checkpoint path used to build --speculative-config",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=15,
        help="offline benchmark engine arg",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="offline benchmark engine arg",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="offline benchmark engine arg",
    )
    parser.add_argument(
        "--load-format",
        default=None,
        help="offline benchmark engine arg",
    )
    parser.add_argument(
        "--eager",
        action="store_true",
        help="offline benchmark engine arg",
    )
    parser.add_argument(
        "--triton-gru",
        action="store_true",
        help="set VLLM_ASCEND_DOMINO_TRITON_GRU=1 in benchmark processes",
    )
    parser.add_argument("--output-dir", default="results/domino")
    parser.add_argument("--no-latency", action="store_true")
    parser.add_argument("--no-throughput", action="store_true")

    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-iters-warmup", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=15)

    parser.add_argument("--throughput-input-len", type=int, default=550)
    parser.add_argument(
        "--throughput-output-len",
        type=int,
        default=None,
        help="generation budget for datasets without per-item output lengths "
        "(random, gsm8k, humaneval; default 1024 for gsm8k/humaneval).  Note "
        "offline throughput ignores EOS: every prompt generates exactly this "
        "many tokens.  Use check_domino_acceptance.py for EOS-respecting "
        "generation.",
    )
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument(
        "--throughput-dataset",
        default="random",
        choices=["random", "sharegpt", "hf", "gsm8k", "humaneval"],
        help="dataset for the throughput benchmark (gsm8k/humaneval are "
        "converted from JSONL to ShareGPT format)",
    )
    parser.add_argument("--throughput-dataset-path", default=None)
    parser.add_argument("--throughput-hf-name", default=None)

    args = parser.parse_args()

    if (
        args.throughput_dataset in ("gsm8k", "humaneval", "sharegpt", "hf")
        and not args.throughput_dataset_path
    ):
        parser.error(
            f"--throughput-dataset {args.throughput_dataset} requires "
            "--throughput-dataset-path"
        )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    env = _env(args)
    spec_json = json.dumps(build_spec_config(args))

    summary: dict = {
        "model": args.model,
        "draft_model": args.draft_model,
        "num_speculative_tokens": args.num_speculative_tokens,
        "speculative_config": json.loads(spec_json),
        "triton_gru": args.triton_gru,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if not args.no_latency:
        print("\n" + "=" * 70)
        print("LATENCY: baseline (AR) vs Domino")
        print("=" * 70)
        baseline = run_latency(
            args, None, os.path.join(output_dir, "latency_ar.json"), env
        )
        spec = run_latency(
            args, spec_json, os.path.join(output_dir, "latency_domino.json"), env
        )
        speedup = _ratio(baseline["avg_latency"], spec["avg_latency"])
        summary["latency"] = {
            "ar_avg_latency_s": baseline["avg_latency"],
            "domino_avg_latency_s": spec["avg_latency"],
            "ar_p50_s": baseline["percentiles"]["50"],
            "domino_p50_s": spec["percentiles"]["50"],
            "ar_p99_s": baseline["percentiles"]["99"],
            "domino_p99_s": spec["percentiles"]["99"],
            "speedup": speedup,
        }
        print(
            f"Avg latency: AR={baseline['avg_latency']:.4f}s  "
            f"Domino={spec['avg_latency']:.4f}s  "
            f"speedup={speedup:.2f}x" if speedup else
            f"Avg latency: AR={baseline['avg_latency']:.4f}s  "
            f"Domino={spec['avg_latency']:.4f}s  speedup=N/A"
        )

    if not args.no_throughput:
        print("\n" + "=" * 70)
        print("THROUGHPUT: baseline (AR) vs Domino")
        print("=" * 70)
        baseline = run_throughput(
            args, None, os.path.join(output_dir, "throughput_ar.json"), env
        )
        spec = run_throughput(
            args, spec_json, os.path.join(output_dir, "throughput_domino.json"), env
        )
        tok_speedup = _ratio(baseline["tokens_per_second"], spec["tokens_per_second"])
        req_speedup = _ratio(
            baseline["requests_per_second"], spec["requests_per_second"]
        )
        summary["throughput"] = {
            "dataset": args.throughput_dataset,
            "ar_tokens_per_second": baseline["tokens_per_second"],
            "domino_tokens_per_second": spec["tokens_per_second"],
            "ar_requests_per_second": baseline["requests_per_second"],
            "domino_requests_per_second": spec["requests_per_second"],
            "tokens_per_second_speedup": tok_speedup,
            "requests_per_second_speedup": req_speedup,
        }
        print(
            f"Tokens/s: AR={baseline['tokens_per_second']:.1f}  "
            f"Domino={spec['tokens_per_second']:.1f}  "
            f"speedup={tok_speedup:.2f}x"
        )
        print(
            f"Requests/s: AR={baseline['requests_per_second']:.2f}  "
            f"Domino={spec['requests_per_second']:.2f}  "
            f"speedup={req_speedup:.2f}x"
        )

    summary_path = os.path.join(output_dir, "domino_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
