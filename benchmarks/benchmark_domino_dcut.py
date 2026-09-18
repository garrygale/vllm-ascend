# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Calibrate or benchmark Qwen3-8B + Domino + V2 Target PIECEWISE."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path


DATASET_SPECS = {
    "gsm8k": ("openai/gsm8k", "main", "question"),
    "humaneval": ("openai/openai_humaneval", None, "prompt"),
}


def _load_local_rows(dataset_path: str, split: str) -> list[dict]:
    path = Path(dataset_path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8-sig") as file:
            return [json.loads(line) for line in file if line.strip()]
    if path.suffix == ".json":
        with path.open(encoding="utf-8-sig") as file:
            data = json.load(file)
        if isinstance(data, dict):
            data = data.get(split, data.get("data"))
        if not isinstance(data, list):
            raise ValueError(
                f"{dataset_path} must contain a JSON list or a '{split}'/'data' list"
            )
        return data
    raise ValueError("--dataset-path must be a .json or .jsonl file")


def load_dataset_prompts(
    dataset_name: str,
    dataset_path: str | None,
    split: str,
    num_prompts: int,
    seed: int,
) -> list[str]:
    """Load and deterministically sample benchmark prompts."""
    hf_name, hf_config, prompt_field = DATASET_SPECS[dataset_name]
    if dataset_path:
        rows = _load_local_rows(dataset_path, split)
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Loading GSM8K/HumanEval requires the 'datasets' package. "
                "Install it or pass --dataset-path with a local JSON/JSONL file."
            ) from exc
        load_args = (hf_name,) if hf_config is None else (hf_name, hf_config)
        rows = load_dataset(*load_args, split=split)

    if len(rows) < num_prompts:
        raise ValueError(
            f"dataset contains {len(rows)} rows, fewer than batch size {num_prompts}"
        )
    indices = random.Random(seed).sample(range(len(rows)), num_prompts)
    prompts = []
    for index in indices:
        row = rows[index]
        prompt = row.get(prompt_field, row.get("input"))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"dataset row {index} has no non-empty '{prompt_field}' field"
            )
        if dataset_name == "gsm8k":
            prompt = (
                "Solve the following problem step by step.\n\n"
                f"Question: {prompt.strip()}\n\nAnswer:"
            )
        prompts.append(prompt)
    return prompts


def finish_dcut(worker):
    runtime = getattr(worker.model_runner, "dcut_runtime", None)
    return runtime.finish_calibration() if runtime is not None else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model", required=True, help="Domino checkpoint path or model ID")
    parser.add_argument("--cost-table", required=True)
    parser.add_argument("--dataset", choices=DATASET_SPECS, default="gsm8k")
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional local JSON/JSONL file; otherwise download the selected Hugging Face dataset",
    )
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--generate-cost-table", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-dcut", action="store_true", help="Run the fixed-K Domino baseline")
    parser.add_argument(
        "--wait-for-probs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait for current draft probabilities; disable to test nonblocking fallback",
    )
    parser.add_argument(
        "--score-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Report cost-saving versus expected-token-loss score decomposition",
    )
    parser.add_argument(
        "--performance-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Measure actual throughput/TPOT for non-fallback D-Cut steps",
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--candidate-draft-lengths", type=int, nargs="*", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate exactly --output-tokens; disable for EOS/dynamic-batch correctness tests",
    )
    parser.add_argument("--profile-warmup", type=int, default=2)
    parser.add_argument("--profile-samples", type=int, default=5)
    args = parser.parse_args()
    if args.disable_dcut and args.generate_cost_table:
        parser.error("--generate-cost-table requires D-Cut to be enabled")
    max_num_seqs = args.max_num_seqs or args.batch_size
    if min(args.tp, args.batch_size, max_num_seqs, args.context_tokens, args.output_tokens, args.rounds) < 1:
        parser.error("TP, batch size, context/output tokens and rounds must be positive")
    if args.batch_size > max_num_seqs:
        parser.error("--max-num-seqs must be greater than or equal to --batch-size")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in the interval (0, 1]")
    if args.temperature < 0:
        parser.error("--temperature must be nonnegative")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in the interval (0, 1]")
    if args.candidate_draft_lengths is not None and any(length < 0 for length in args.candidate_draft_lengths):
        parser.error("--candidate-draft-lengths values must be nonnegative")

    dataset_prompts = load_dataset_prompts(
        args.dataset,
        args.dataset_path,
        args.dataset_split,
        args.batch_size,
        args.dataset_seed,
    )

    from transformers import AutoConfig
    from vllm import LLM, SamplingParams

    block_size = AutoConfig.from_pretrained(args.draft_model).block_size
    if args.context_tokens + args.output_tokens + block_size >= args.max_model_len:
        parser.error("--max-model-len must leave space for context, output and the full Domino block")
    if args.batch_size * (block_size + 1) > args.max_num_batched_tokens:
        parser.error("--max-num-batched-tokens must fit the full-K decode batch")
    candidate_lengths = args.candidate_draft_lengths or [3, 7, 11, block_size]
    capture_sizes = sorted(
        {
            1,
            4,
            8,
            12,
            16,
            64,
            *(args.batch_size * (length + 1) for length in candidate_lengths),
            args.batch_size * (block_size + 1),
        }
    )
    capture_sizes = [
        size for size in capture_sizes if size <= args.max_num_batched_tokens
    ]
    dcut_config = {
        "enabled": not args.disable_dcut,
        "cost_table_path": args.cost_table,
        "generate_cost_table": args.generate_cost_table,
        "wait_for_probs": args.wait_for_probs,
        "score_diagnostics": args.score_diagnostics,
        "performance_diagnostics": args.performance_diagnostics,
        "profile_warmup": args.profile_warmup,
        "profile_samples": args.profile_samples,
    }
    if args.candidate_draft_lengths is not None:
        dcut_config["candidate_draft_lengths"] = args.candidate_draft_lengths
    llm = LLM(
        model=args.target_model,
        seed=args.model_seed,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        async_scheduling=True,
        trust_remote_code=True,
        speculative_config={
            "model": args.draft_model,
            "method": "domino",
            "draft_sample_method": "greedy",
            "num_speculative_tokens": block_size,
            "draft_tensor_parallel_size": args.tp,
        },
        compilation_config={
            "mode": "VLLM_COMPILE",
            "cudagraph_mode": "PIECEWISE",
            "cudagraph_capture_sizes": capture_sizes,
        },
        additional_config={
            "enable_cpu_binding": True,
            "enable_weight_nz_layout": True,
            "dcut_config": dcut_config,
        },
    )
    tokenizer = llm.get_tokenizer()
    prompt_token_ids = [
        tokenizer.encode(prompt, add_special_tokens=False)[: args.context_tokens]
        for prompt in dataset_prompts
    ]
    if any(not tokens for tokens in prompt_token_ids):
        parser.error("the selected dataset contains a prompt that tokenizes to an empty sequence")
    prompts = [{"prompt_token_ids": tokens} for tokens in prompt_token_ids]
    prompt_lengths = [len(tokens) for tokens in prompt_token_ids]
    sampling = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.output_tokens,
        ignore_eos=args.ignore_eos,
    )
    generated = 0
    elapsed = 0.0
    output_lengths = []
    output_digest = hashlib.sha256()
    for _ in range(args.rounds):
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        elapsed += time.perf_counter() - start
        for output in outputs:
            token_ids = output.outputs[0].token_ids
            generated += len(token_ids)
            output_lengths.append(len(token_ids))
            output_digest.update(len(token_ids).to_bytes(8, "little"))
            for token_id in token_ids:
                output_digest.update(int(token_id).to_bytes(4, "little"))
    stats = llm.collective_rpc(finish_dcut)
    print(
        json.dumps(
            {
                "mode": "baseline" if args.disable_dcut else "calibration" if args.generate_cost_table else "dcut",
                "elapsed_seconds": elapsed,
                "output_tokens": generated,
                "output_tokens_per_second": generated / elapsed,
                "output_tokens_per_request_min": min(output_lengths),
                "output_tokens_per_request_mean": (
                    sum(output_lengths) / len(output_lengths)
                ),
                "output_tokens_per_request_max": max(output_lengths),
                "output_token_sha256": output_digest.hexdigest(),
                "dataset": {
                    "name": args.dataset,
                    "path": args.dataset_path,
                    "split": args.dataset_split,
                    "seed": args.dataset_seed,
                    "prompts": len(prompts),
                    "prompt_tokens_min": min(prompt_lengths),
                    "prompt_tokens_mean": sum(prompt_lengths) / len(prompt_lengths),
                    "prompt_tokens_max": max(prompt_lengths),
                },
                "benchmark_config": {
                    "batch_size": args.batch_size,
                    "rounds": args.rounds,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    "ignore_eos": args.ignore_eos,
                    "model_seed": args.model_seed,
                    "score_diagnostics": args.score_diagnostics,
                    "performance_diagnostics": args.performance_diagnostics,
                    "candidate_draft_lengths": candidate_lengths,
                    "capture_sizes": capture_sizes,
                },
                "worker_stats": stats,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
