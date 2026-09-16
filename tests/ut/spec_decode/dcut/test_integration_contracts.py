# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise production overrides with CPU doubles for upstream/NPU objects."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_controller import ScheduledBatch


def production_methods(relative_path, class_name, methods, base, namespace):
    root = Path(__file__).resolve().parents[4]
    tree = ast.parse((root / relative_path).read_text(encoding="utf-8"))
    source_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    source_class.body = [
        node for node in source_class.body if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    if not source_class.body:
        source_class.body = [ast.Pass()]
    source_class.bases = [ast.Name(id="CpuBase", ctx=ast.Load())]
    module = ast.Module(body=[source_class], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace.update(CpuBase=base, torch=torch)
    exec(compile(module, str(root / relative_path), "exec"), namespace)
    return namespace[class_name]


class CpuRunner:
    def execute_model(self, scheduler_output, **kwargs):
        self.dispatched = scheduler_output
        self.kwargs = kwargs
        if getattr(self, "fail", False):
            raise RuntimeError("target failed")
        return scheduler_output.total_num_scheduled_tokens


class CpuRuntime:
    def __init__(self):
        self.started = self.aborted = self.selected = 0

    def select_caps(self, scheduler_output, req_states):
        self.selected += 1
        return ["a", "b"], np.array([2, 0])

    def begin_target(self):
        self.started += 1

    def abort_target(self):
        self.aborted += 1


def cpu_runner(dcut_modules):
    cls = production_methods(
        "vllm_ascend/worker/v2/spec_decode/dcut/model_runner.py",
        "DcutNPUModelRunner",
        {"execute_model"},
        CpuRunner,
        {"SchedulerOutput": object, "truncate_scheduler_output": dcut_modules.controller.truncate_scheduler_output},
    )
    runner = cls()
    runner.dcut_runtime = CpuRuntime()
    runner.req_states = object()
    return runner


def test_runner_trims_before_upstream_graph_dispatch(dcut_modules):
    runner = cpu_runner(dcut_modules)
    scheduled = ScheduledBatch()
    assert runner.execute_model(scheduled) == 4
    assert scheduled.total_num_scheduled_tokens == 8
    assert runner.dispatched.num_scheduled_tokens == {"a": 3, "b": 1}
    assert runner.dcut_runtime.started == 1


@pytest.mark.parametrize("bypass", ["uninitialized", "dummy", "profile"])
def test_dcut_runner_bypasses_control_for_uninitialized_or_dummy(dcut_modules, bypass):
    runner = cpu_runner(dcut_modules)
    runtime = runner.dcut_runtime
    if bypass == "uninitialized":
        runner.dcut_runtime = None
    scheduled = ScheduledBatch()
    assert runner.execute_model(scheduled, dummy_run=bypass == "dummy", is_profile=bypass == "profile") == 8
    assert runner.dispatched is scheduled
    assert runtime.selected == runtime.started == 0


def test_target_failure_discards_pending_probability(dcut_modules):
    runner = cpu_runner(dcut_modules)
    runner.fail = True
    with pytest.raises(RuntimeError, match="target failed"):
        runner.execute_model(ScheduledBatch())
    assert runner.dcut_runtime.aborted == 1


class CpuDomino:
    def _sample_step(self, logits_i, idx_map_i, sample_pos_i, col):
        return logits_i.argmax(-1) + 10  # Nonidentity draft-to-target ID mapping.


class CorrectionModel:
    pure_draft_prefix_len = 1

    def __init__(self):
        self.logits = torch.tensor([[0.0, 2.0, 1.0], [1.0, 2.0, 0.0], [0.0, 1.0, 3.0]])
        self.bias = torch.tensor([[5.0, 0.0, 0.0]])

    def compute_draft_logits(self, hidden):
        return self.logits

    def domino_optimized_prefix(self, prefix_ids):
        self.prefix_ids = prefix_ids.clone()
        return torch.zeros(1, 1)

    def domino_z_part(self, hidden):
        return hidden

    def domino_optimized_bias_and_gh(self, hidden, z_part):
        return self.bias, hidden

    def domino_optimized_cell(self, draft_i, hidden, gh):
        return hidden

    def map_draft_to_target(self, draft_ids):
        return draft_ids + 10


def test_probability_uses_gru_corrected_logits_without_changing_full_block(dcut_modules):
    baseline_cls = production_methods(
        "vllm_ascend/worker/v2/spec_decode/domino/speculator.py",
        "AscendDominoSpeculator",
        {"_sample_sequential"},
        CpuDomino,
        {},
    )
    cls = production_methods(
        "vllm_ascend/worker/v2/spec_decode/dcut/speculator.py",
        "DcutDominoSpeculator",
        {"_sample_step"},
        baseline_cls,
        {"selected_greedy_probability": dcut_modules.runtime.selected_greedy_probability},
    )
    speculator = cls()
    speculator.num_speculative_steps = 3
    speculator.sample_indices = torch.arange(3)
    speculator.sample_idx_mapping = torch.zeros(3, dtype=torch.long)
    speculator.sample_pos = torch.arange(3)
    speculator.input_buffers = SimpleNamespace(input_ids=torch.tensor([99]))
    speculator._anchor_idx = torch.tensor([0])
    speculator.dtype = torch.float32
    speculator.device = torch.device("cpu")
    speculator.model = CorrectionModel()
    speculator.draft_tokens = torch.empty(1, 3, dtype=torch.long)
    speculator.selected_probs = torch.empty(1, 3)
    speculator._collect_dcut_probs = False
    speculator._sample_sequential(1, torch.zeros(3, 1))
    baseline = speculator.draft_tokens.clone()
    speculator._collect_dcut_probs = True
    speculator._sample_sequential(1, torch.zeros(3, 1))
    corrected = speculator.model.logits.clone()
    corrected[1:] += speculator.model.bias
    torch.testing.assert_close(speculator.selected_probs[0], corrected.softmax(-1).amax(-1))
    torch.testing.assert_close(speculator.draft_tokens, baseline)
    assert baseline.tolist() == [[11, 10, 10]]
    assert speculator.model.prefix_ids.tolist() == [[99, 11]]
