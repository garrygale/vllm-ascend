#!/usr/bin/env python3
# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Measure Domino acceptance length from a running vLLM service.

The script sends prompts one at a time to the vLLM OpenAI-compatible
``/v1/completions`` endpoint.  Before and after each request it snapshots the
server's Prometheus spec-decode counters (``/metrics``) and computes that
request's acceptance length from the delta, so we get per-prompt acceptance
statistics without needing server-side per-token info in the response.

Each worker sends the next prompt immediately after the current one is
answered; ``NUM_WORKERS`` controls how many prompts are in flight in parallel.
Aggregate statistics are computed from metric snapshots around the whole run.

Edit the CONFIG block below, then run:
    python benchmarks/check_acceptance_service.py

Datasets: humaneval (JSONL with ``prompt``), gsm8k (JSONL with ``question``),
math500 (JSONL with ``problem``), mbpp (JSON array with ``prompt`` and
``test_list``), or oasst (JSONL with ``prompt``).
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests


# ---------------------------------------------------------------------------
# CONFIG - edit these before running
# ---------------------------------------------------------------------------

SERVER_IP = "127.0.0.1"
SERVER_PORT = 4144
SERVED_MODEL_NAME = "qwen35"
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = 0

DATASET = "gsm8k"  # "humaneval", "gsm8k", "math500", "mbpp" or "oasst"
DATASET_PATH = "/path/to/gsm8k_test.jsonl"

MAX_TOKENS = 1024          # generation upper bound; EOS stops earlier
NUM_PROMPTS = -1          # -1 = all prompts
OUTPUT_DIR = "results/domino_acceptance"
NUM_WORKERS = 6           # parallel workers; each sends prompts sequentially
STORE_PER_SAMPLE = False  # keep per-request rows in the summary JSON
USE_CHAT_TEMPLATE = True  # send via /v1/chat/completions so the server
                          # applies the model's chat template (matches
                          # specforge's check_acceptance --use-chat-template)
CHAT_TEMPLATE_KWARGS = {"enable_thinking": True}  # Qwen3.6-style template
REQUEST_TIMEOUT = 600     # seconds per request
METRICS_SETTLE_SECONDS = 0.2  # allow counters to flush before reading

# ---------------------------------------------------------------------------


def build_mbpp_prompt(text: str, test_list: list) -> str:
    """Standard MBPP prompt format used in the original paper."""
    tests = "\n".join(test_list)
    return (
        "You are an expert Python programmer, and here is your task: "
        f"{text} Your code should pass these tests:\n\n{tests}\n\n[BEGIN]\n"
    )


def load_prompts(path: str, dataset: str) -> list[tuple[str, str]]:
    """Return ``[(task_id, prompt), ...]`` from a dataset file."""
    if dataset == "mbpp":
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        rows = []
        for idx, item in enumerate(items):
            text = item.get("prompt") or item.get("text") or ""
            test_list = item.get("test_list") or []
            task_id = item.get("task_id", idx)
            rows.append(
                (f"mbpp_{task_id}", build_mbpp_prompt(text, test_list))
            )
        return rows

    rows = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if dataset == "humaneval":
                rows.append((item["task_id"], item["prompt"]))
            elif dataset == "gsm8k":
                rows.append((f"gsm8k_{idx}", item["question"]))
            elif dataset == "oasst":
                rows.append((f"oasst_{idx}", item["prompt"]))
            elif dataset == "math500":
                task_id = item.get("unique_id") or f"math500_{idx}"
                rows.append((task_id, item["problem"]))
            else:
                raise ValueError(f"unknown dataset: {dataset}")
    return rows


def fetch_spec_decode_metrics(base_url: str) -> dict | None:
    """Parse spec-decode counters from the Prometheus /metrics endpoint."""
    resp = requests.get(f"{base_url}/metrics", timeout=30)
    resp.raise_for_status()

    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    accepted_per_pos: dict[int, int] = {}
    found = False

    for line in resp.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("vllm:spec_decode"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        metric_name = parts[0].split("{")[0]
        if not metric_name.endswith("_total"):
            continue
        try:
            value = int(float(parts[-1]))
        except ValueError:
            continue
        found = True
        if metric_name == "vllm:spec_decode_num_drafts_total":
            num_drafts += value
        elif metric_name == "vllm:spec_decode_num_draft_tokens_total":
            num_draft_tokens += value
        elif metric_name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            label = 'position="'
            if label in line:
                start = line.index(label) + len(label)
                end = line.index('"', start)
                try:
                    pos = int(line[start:end])
                except ValueError:
                    continue
                accepted_per_pos[pos] = accepted_per_pos.get(pos, 0) + value
        elif metric_name == "vllm:spec_decode_num_accepted_tokens_total":
            num_accepted_tokens += value

    if not found:
        return None
    return {
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "accepted_per_pos": accepted_per_pos,
    }


def metric_delta(before: dict, after: dict) -> dict:
    """Delta between two metric snapshots (cumulative counters)."""
    accepted_per_pos = {}
    for pos, val in after["accepted_per_pos"].items():
        accepted_per_pos[pos] = val - before["accepted_per_pos"].get(pos, 0)
    return {
        "num_drafts": after["num_drafts"] - before["num_drafts"],
        "num_draft_tokens": (
            after["num_draft_tokens"] - before["num_draft_tokens"]
        ),
        "num_accepted_tokens": (
            after["num_accepted_tokens"] - before["num_accepted_tokens"]
        ),
        "accepted_per_pos": accepted_per_pos,
    }


def send_completion(
    base_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    use_chat_template: bool,
    chat_template_kwargs: dict,
) -> int:
    """POST /v1/completions and return the number of completion tokens."""
    if use_chat_template:
        endpoint = "/v1/chat/completions"
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stream": False,
            "chat_template_kwargs": chat_template_kwargs,
        }
    else:
        endpoint = "/v1/completions"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stream": False,
        }
    resp = requests.post(
        f"{base_url}{endpoint}", json=payload, timeout=REQUEST_TIMEOUT
    )
    resp.raise_for_status()
    data = resp.json()
    return int(data["usage"]["completion_tokens"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=DATASET,
        choices=["humaneval", "gsm8k", "oasst", "math500", "mbpp"],
    )
    parser.add_argument("--dataset-path", default=DATASET_PATH)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--num-prompts", type=int, default=NUM_PROMPTS)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--server-ip", default=SERVER_IP)
    parser.add_argument("--server-port", type=int, default=SERVER_PORT)
    parser.add_argument("--served-model-name", default=SERVED_MODEL_NAME)
    parser.add_argument(
        "--per-sample",
        action="store_true",
        default=STORE_PER_SAMPLE,
        help="store per-request rows in the summary JSON (default: off)",
    )
    parser.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=USE_CHAT_TEMPLATE,
        help="send via /v1/chat/completions so the server applies the "
        "model's chat template",
    )
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")
    if args.per_sample and args.num_workers > 1:
        parser.error(
            "--per-sample is not supported with --num-workers > 1 "
            "(per-request metric deltas are unreliable under concurrency)"
        )

    base_url = f"http://{args.server_ip}:{args.server_port}"
    prompts = load_prompts(args.dataset_path, args.dataset)
    if args.num_prompts is not None and args.num_prompts > 0:
        prompts = prompts[: args.num_prompts]
    print(
        f"Server: {base_url}  model={args.served_model_name}  "
        f"temperature={args.temperature}  "
        f"top_p={args.top_p}  top_k={args.top_k}  "
        f"chat_template={args.use_chat_template}"
    )
    print(f"Dataset: {args.dataset} ({len(prompts)} prompts)")

    before_all = fetch_spec_decode_metrics(base_url)
    if before_all is None:
        raise RuntimeError(
            "No spec-decode metrics found. Is the server running with "
            "speculative decoding and --disable-log-stats disabled?"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    per_sample = args.per_sample

    def _per_pos_rates(accepted_per_pos: dict[int, int], total_drafts: int):
        max_pos = max(accepted_per_pos, default=-1)
        return [
            accepted_per_pos.get(pos, 0) / total_drafts
            if total_drafts > 0
            else 0.0
            for pos in range(max_pos + 1)
        ]

    if per_sample:
        results: list[dict] = []
        acceptance_lengths: list[float] = []
        total_drafts = 0
        weighted_sum = 0.0
        accepted_pos_total: dict[int, int] = {}

        for i, (task_id, prompt) in enumerate(prompts):
            before = fetch_spec_decode_metrics(base_url)
            completion_tokens = send_completion(
                base_url,
                args.served_model_name,
                prompt,
                args.max_tokens,
                args.temperature,
                args.top_p,
                args.top_k,
                args.use_chat_template,
                CHAT_TEMPLATE_KWARGS,
            )
            time.sleep(METRICS_SETTLE_SECONDS)
            after = fetch_spec_decode_metrics(base_url)
            delta = metric_delta(before, after)

            num_drafts = delta["num_drafts"]
            num_accepted = delta["num_accepted_tokens"]
            acceptance_length = (
                1 + num_accepted / num_drafts if num_drafts > 0 else None
            )
            if acceptance_length is not None:
                acceptance_lengths.append(acceptance_length)
                total_drafts += num_drafts
                weighted_sum += acceptance_length * num_drafts
                for pos, val in delta["accepted_per_pos"].items():
                    accepted_pos_total[pos] = (
                        accepted_pos_total.get(pos, 0) + val
                    )

            results.append(
                {
                    "task_id": task_id,
                    "completion_tokens": completion_tokens,
                    "num_drafts": num_drafts,
                    "num_draft_tokens": delta["num_draft_tokens"],
                    "num_accepted_tokens": num_accepted,
                    "accepted_per_pos": delta["accepted_per_pos"],
                    "acceptance_length": acceptance_length,
                }
            )
            print(f"[{i + 1}/{len(prompts)}] {task_id}")

        simple_mean = (
            statistics.mean(acceptance_lengths)
            if acceptance_lengths
            else None
        )
        weighted_mean = (
            weighted_sum / total_drafts if total_drafts > 0 else None
        )
        stats = {
            "simple_mean_acceptance_length": simple_mean,
            "weighted_mean_acceptance_length": weighted_mean,
            "mean_acceptance_length": (
                weighted_mean if weighted_mean is not None else simple_mean
            ),
            "num_valid_requests": len(acceptance_lengths),
            "per_position_acceptance_rates": _per_pos_rates(
                accepted_pos_total, total_drafts
            ),
        }
    else:
        # Aggregate-only: two metric snapshots around the whole run.  The
        # draft-weighted mean equals 1 + total_accepted / total_drafts.
        work_queue: queue.Queue = queue.Queue()
        for idx, (task_id, prompt) in enumerate(prompts, start=1):
            work_queue.put((idx, task_id, prompt))
        print_lock = threading.Lock()

        def _worker(worker_id: int) -> None:
            while True:
                try:
                    idx, task_id, prompt = work_queue.get_nowait()
                except queue.Empty:
                    return
                send_completion(
                    base_url,
                    args.served_model_name,
                    prompt,
                    args.max_tokens,
                    args.temperature,
                    args.top_p,
                    args.top_k,
                    args.use_chat_template,
                    CHAT_TEMPLATE_KWARGS,
                )
                with print_lock:
                    print(f"[w{worker_id}] [{idx}/{len(prompts)}] {task_id}")

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(_worker, w)
                for w in range(args.num_workers)
            ]
            for future in futures:
                future.result()
        time.sleep(METRICS_SETTLE_SECONDS)
        after_all = fetch_spec_decode_metrics(base_url)
        delta = metric_delta(before_all, after_all)
        total_drafts = delta["num_drafts"]
        num_accepted = delta["num_accepted_tokens"]
        mean_acceptance_length = (
            1 + num_accepted / total_drafts if total_drafts > 0 else None
        )
        stats = {
            "mean_acceptance_length": mean_acceptance_length,
            "num_drafts": total_drafts,
            "num_draft_tokens": delta["num_draft_tokens"],
            "num_accepted_tokens": num_accepted,
            "num_valid_requests": len(prompts) if total_drafts > 0 else 0,
            "per_position_acceptance_rates": _per_pos_rates(
                delta["accepted_per_pos"], total_drafts
            ),
        }

    summary = {
        "server": base_url,
        "served_model_name": args.served_model_name,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "dataset": args.dataset,
        "num_requests": len(prompts),
        "num_workers": args.num_workers,
        "mean_acceptance_length": stats["mean_acceptance_length"],
        "num_valid_requests": stats["num_valid_requests"],
        "per_position_acceptance_rates": stats[
            "per_position_acceptance_rates"
        ],
        "store_per_sample": per_sample,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if per_sample:
        summary["results"] = results
        summary["simple_mean_acceptance_length"] = stats[
            "simple_mean_acceptance_length"
        ]
        summary["weighted_mean_acceptance_length"] = stats[
            "weighted_mean_acceptance_length"
        ]
    else:
        summary["num_drafts"] = stats["num_drafts"]
        summary["num_draft_tokens"] = stats["num_draft_tokens"]
        summary["num_accepted_tokens"] = stats["num_accepted_tokens"]

    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    summary_path = os.path.join(
        args.output_dir, f"acceptance_summary_{stamp}.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print("\n" + "=" * 56)
    print("Domino acceptance-length results")
    print(f"  Requests:            {len(prompts)}")
    print(f"  Valid (with drafts): {stats['num_valid_requests']}")
    if stats["mean_acceptance_length"] is not None:
        print(
            "  Mean acceptance length:            "
            f"{stats['mean_acceptance_length']:.2f}"
        )
    if per_sample:
        if stats["simple_mean_acceptance_length"] is not None:
            print(
                "  Mean acceptance length (simple):   "
                f"{stats['simple_mean_acceptance_length']:.2f}"
            )
        if stats["weighted_mean_acceptance_length"] is not None:
            print(
                "  Mean acceptance length (weighted): "
                f"{stats['weighted_mean_acceptance_length']:.2f}"
            )
    per_pos = stats["per_position_acceptance_rates"]
    if per_pos:
        print("  Per-position acceptance rates: "
              + ", ".join(f"{p:.3f}" for p in per_pos))
    print(f"  Summary: {summary_path}")
    print("=" * 56)


if __name__ == "__main__":
    main()
