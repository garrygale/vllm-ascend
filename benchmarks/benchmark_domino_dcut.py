# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Calibrate or benchmark Qwen3-8B + Domino + V2 Target PIECEWISE."""

import argparse
import json
import time


def finish_dcut(worker):
    runtime = getattr(worker.model_runner, "dcut_runtime", None)
    return runtime.finish_calibration() if runtime is not None else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model", required=True, help="Domino checkpoint path or model ID")
    parser.add_argument("--cost-table", required=True)
    parser.add_argument("--generate-cost-table", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-dcut", action="store_true", help="Run the fixed-K Domino baseline")
    parser.add_argument(
        "--wait-for-probs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait for current draft probabilities; disable to test nonblocking fallback",
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

    from transformers import AutoConfig
    from vllm import LLM, SamplingParams

    block_size = AutoConfig.from_pretrained(args.draft_model).block_size
    if args.context_tokens + args.output_tokens + block_size >= args.max_model_len:
        parser.error("--max-model-len must leave space for context, output and the full Domino block")
    if args.batch_size * (block_size + 1) > args.max_num_batched_tokens:
        parser.error("--max-num-batched-tokens must fit the full-K decode batch")
    dcut_config = {
        "enabled": not args.disable_dcut,
        "cost_table_path": args.cost_table,
        "generate_cost_table": args.generate_cost_table,
        "wait_for_probs": args.wait_for_probs,
        "profile_warmup": args.profile_warmup,
        "profile_samples": args.profile_samples,
    }
    if args.candidate_draft_lengths is not None:
        dcut_config["candidate_draft_lengths"] = args.candidate_draft_lengths
    llm = LLM(
        model=args.target_model,
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
            "cudagraph_capture_sizes": [1, 4, 8, 12, 16, 64, 256, 512, 768, 1024],
        },
        additional_config={
            "enable_cpu_binding": True,
            "enable_weight_nz_layout": True,
            "dcut_config": dcut_config,
        },
    )
    tokenizer = llm.get_tokenizer()
    unit = tokenizer.encode("Explain how a computer processes information. ", add_special_tokens=False)
    tokens = (unit * (args.context_tokens // len(unit) + 1))[: args.context_tokens]
    prompts = [{"prompt_token_ids": tokens} for _ in range(args.batch_size)]
    sampling = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.output_tokens,
        ignore_eos=True,
    )
    generated = 0
    elapsed = 0.0
    for _ in range(args.rounds):
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        elapsed += time.perf_counter() - start
        generated += sum(len(output.outputs[0].token_ids) for output in outputs)
    stats = llm.collective_rpc(finish_dcut)
    print(
        json.dumps(
            {
                "mode": "baseline" if args.disable_dcut else "calibration" if args.generate_cost_table else "dcut",
                "elapsed_seconds": elapsed,
                "output_tokens": generated,
                "output_tokens_per_second": generated / elapsed,
                "worker_stats": stats,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
