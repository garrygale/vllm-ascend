# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU correctness/graph replay test; synthetic costs do not measure speedup."""

import json

import pytest
from transformers import AutoConfig
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner


def finish_dcut(worker):
    runtime = getattr(worker.model_runner, "dcut_runtime", None)
    return runtime.finish_calibration() if runtime is not None else {}


def test_domino_dcut_piecewise_v2(tmp_path, request):
    draft = request.config.getoption("--domino-draft-model")
    if draft is None:
        pytest.skip("Pass --domino-draft-model with a Qwen3-8B Domino checkpoint")
    target = request.config.getoption("--domino-target-model")
    tp = request.config.getoption("--domino-dcut-tp")
    block_size = AutoConfig.from_pretrained(draft).block_size
    path = tmp_path / "cost.json"
    prompts = ["Explain how a computer processes information in detail."] * 4
    sampling = SamplingParams(temperature=0, max_tokens=192, ignore_eos=True)

    def run(enabled, generate):
        with VllmRunner(
            target,
            tensor_parallel_size=tp,
            max_model_len=1024,
            max_num_seqs=4,
            max_num_batched_tokens=max(256, 4 * (block_size + 1)),
            enable_prefix_caching=False,
            async_scheduling=True,
            speculative_config={
                "model": draft,
                "method": "domino",
                "draft_sample_method": "greedy",
                "num_speculative_tokens": block_size,
            },
            compilation_config={"mode": "VLLM_COMPILE", "cudagraph_mode": "PIECEWISE"},
            additional_config={
                "dcut_config": {
                    "enabled": enabled,
                    "cost_table_path": str(path),
                    "generate_cost_table": generate,
                    "profile_warmup": 0,
                    "profile_samples": 1,
                    "candidate_draft_lengths": [0, 1, block_size],
                    "context_buckets": [1024],
                    "wait_for_probs": True,
                }
            },
        ) as runner:
            outputs = runner.model.generate(prompts, sampling, use_tqdm=False)
            ids = [list(output.outputs[0].token_ids) for output in outputs]
            stats = runner.model.collective_rpc(finish_dcut)
        return ids, stats

    baseline, _ = run(False, False)
    calibrated, stats = run(True, True)
    assert calibrated == baseline
    assert stats[0]["profiled_rows"] >= 2
    table = json.loads(path.read_text())
    # Force trimming independently of hardware noise, preserving the real fingerprint.
    # These synthetic timings are only for correctness coverage, never a benchmark.
    for row in table["rows"]:
        anchor_only = row["query_tokens"] == row["batch_size"]
        row["target_ms"] = 0.01 if anchor_only else 1000.0
        row["draft_ms"] = 0.01 if anchor_only else 1000.0
    path.write_text(json.dumps(table))
    before = path.read_bytes()
    trimmed, stats = run(True, False)
    assert trimmed == baseline
    assert all(worker["decisions"] > 0 and worker["trimmed_tokens"] > 0 for worker in stats)
    assert path.read_bytes() == before, "generate_cost_table=false must not write the table"
