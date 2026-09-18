# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
from pathlib import Path

import pytest


BENCHMARK_PATH = (
    Path(__file__).parents[4] / "benchmarks" / "benchmark_domino_dcut.py"
)
SPEC = importlib.util.spec_from_file_location("benchmark_domino_dcut", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def test_load_gsm8k_prompts_is_deterministic(tmp_path):
    path = tmp_path / "gsm8k.json"
    path.write_text(
        json.dumps(
            [
                {"question": "What is 1 + 1?", "answer": "2"},
                {"question": "What is 2 + 2?", "answer": "4"},
                {"question": "What is 3 + 3?", "answer": "6"},
            ]
        ),
        encoding="utf-8",
    )

    first = BENCHMARK.load_dataset_prompts("gsm8k", str(path), "test", 2, 17)
    second = BENCHMARK.load_dataset_prompts("gsm8k", str(path), "test", 2, 17)

    assert first == second
    assert all(prompt.startswith("Solve the following problem") for prompt in first)
    assert all(prompt.endswith("\n\nAnswer:") for prompt in first)
    assert not any('"answer"' in prompt for prompt in first)


def test_load_humaneval_prompts_keeps_code_unchanged(tmp_path):
    rows = [
        {"prompt": "def add(a, b):\n    \"\"\"Return the sum.\"\"\"\n"},
        {"prompt": "def square(x):\n    \"\"\"Return x squared.\"\"\"\n"},
    ]
    path = tmp_path / "humaneval.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    prompts = BENCHMARK.load_dataset_prompts(
        "humaneval", str(path), "test", 2, 0
    )

    assert sorted(prompts) == sorted(row["prompt"] for row in rows)


def test_load_dataset_prompts_rejects_too_few_rows(tmp_path):
    path = tmp_path / "gsm8k.json"
    path.write_text(json.dumps([{"question": "Only one"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="fewer than batch size"):
        BENCHMARK.load_dataset_prompts("gsm8k", str(path), "test", 2, 0)


def test_load_local_json_accepts_utf8_bom(tmp_path):
    path = tmp_path / "gsm8k_bom.json"
    path.write_text(
        json.dumps(
            [
                {"question": "What is 1 + 1?"},
                {"question": "What is 2 + 2?"},
            ]
        ),
        encoding="utf-8-sig",
    )

    prompts = BENCHMARK.load_dataset_prompts(
        "gsm8k", str(path), "test", 2, 0
    )

    assert len(prompts) == 2


def test_load_dataset_prompts_accepts_input_alias(tmp_path):
    path = tmp_path / "gsm8k_input.json"
    path.write_text(
        json.dumps(
            [
                {"id": 0, "input": "What is 1 + 1?"},
                {"id": 1, "input": "What is 2 + 2?"},
            ]
        ),
        encoding="utf-8",
    )

    prompts = BENCHMARK.load_dataset_prompts(
        "gsm8k", str(path), "test", 2, 0
    )

    assert all("Question:" in prompt for prompt in prompts)
