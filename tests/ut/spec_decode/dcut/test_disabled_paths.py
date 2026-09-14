# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disabled D-Cut must select the original classes and inherited methods."""

import ast
import builtins
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from test_integration_contracts import CpuDomino, CpuRunner, production_methods


def source_tree(path):
    root = Path(__file__).resolve().parents[4]
    return ast.parse((root / path).read_text(encoding="utf-8"))


def compile_function(node, namespace):
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "production_selection", "exec"), namespace)
    return namespace[node.name]


class OriginalMarker:
    def __init__(self, config, device):
        self.config = config
        self.device = device


class DcutMarker(OriginalMarker):
    pass


def import_namespace(original_module, original_class, dcut_module, dcut_class):
    imported = []
    modules = {
        original_module: SimpleNamespace(**{original_class: OriginalMarker}),
        dcut_module: SimpleNamespace(**{dcut_class: DcutMarker}),
    }

    def import_stub(name, globals=None, locals=None, fromlist=(), level=0):
        imported.append(name)
        if name in modules:
            return modules[name]
        return builtins.__import__(name, globals, locals, fromlist, level)

    return {"__builtins__": {**vars(builtins), "__import__": import_stub}}, imported


@pytest.mark.parametrize(
    "additional",
    [
        None,
        {},
        {"dcut_config": {}},
        {
            "dcut_config": {
                "enabled": False,
                "generate_cost_table": True,
                "cost_table_path": "/missing.json",
            }
        },
        {"dcut_config": {"enabled": True}},
    ],
)
def test_domino_factory_imports_dcut_only_when_enabled(additional):
    tree = source_tree("vllm_ascend/worker/v2/spec_decode/__init__.py")
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "init_speculator")
    namespace, imported = import_namespace(
        "vllm_ascend.worker.v2.spec_decode.domino.speculator",
        "AscendDominoSpeculator",
        "vllm_ascend.worker.v2.spec_decode.dcut.speculator",
        "DcutDominoSpeculator",
    )
    namespace.update(VllmConfig=object, torch=torch)
    factory = compile_function(node, namespace)
    config = SimpleNamespace(
        additional_config=additional,
        speculative_config=SimpleNamespace(
            use_dspark=lambda: False,
            use_domino=lambda: True,
        ),
    )
    device = torch.device("cpu")
    result = factory(config, device)
    enabled = bool(additional and additional.get("dcut_config", {}).get("enabled"))
    assert type(result) is (DcutMarker if enabled else OriginalMarker)
    assert result.config is config and result.device is device
    assert ("vllm_ascend.worker.v2.spec_decode.dcut.speculator" in imported) == enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_worker_selects_original_runner_when_disabled(enabled):
    tree = source_tree("vllm_ascend/worker/worker.py")
    selection = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Attribute)
        and node.test.left.attr == "enabled"
        and isinstance(node.test.left.value, ast.Attribute)
        and node.test.left.value.attr == "dcut_config"
    )
    node = ast.FunctionDef(
        name="select_runner",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="self")], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[selection],
        decorator_list=[],
    )
    namespace, imported = import_namespace(
        "unused",
        "unused",
        "vllm_ascend.worker.v2.spec_decode.dcut.model_runner",
        "DcutNPUModelRunner",
    )
    namespace.update(
        NPUModelRunnerV2=OriginalMarker,
        get_ascend_config=lambda: SimpleNamespace(
            dcut_config=SimpleNamespace(enabled=enabled),
        ),
    )
    worker = SimpleNamespace(vllm_config=object(), device=torch.device("cpu"))
    compile_function(node, namespace)(worker)
    assert type(worker.model_runner) is (DcutMarker if enabled else OriginalMarker)
    assert worker.model_runner.config is worker.vllm_config
    assert worker.model_runner.device is worker.device
    assert bool(imported) == enabled


def test_original_runner_and_domino_do_not_override_hot_paths():
    runner = production_methods(
        "vllm_ascend/worker/v2/model_runner.py", "NPUModelRunner", {"execute_model"}, CpuRunner, {}
    )
    domino = production_methods(
        "vllm_ascend/worker/v2/spec_decode/domino/speculator.py",
        "AscendDominoSpeculator",
        {"_sample_step"},
        CpuDomino,
        {},
    )
    assert runner.execute_model is CpuRunner.execute_model
    assert domino._sample_step is CpuDomino._sample_step
    for path in ("vllm_ascend/worker/v2/model_runner.py", "vllm_ascend/worker/v2/spec_decode/domino/speculator.py"):
        assert all(
            "dcut" not in node.module
            for node in ast.walk(source_tree(path))
            if isinstance(node, ast.ImportFrom) and node.module
        )


def test_original_domino_propose_preserves_arguments_wrapper_and_result():
    events = []

    @contextmanager
    def attention_wrapper():
        events.append("enter")
        yield
        events.append("exit")

    class Parent:
        def propose(self, *args, **kwargs):
            events.append("propose")
            self.forwarded = args, kwargs
            return self.result

    cls = production_methods(
        "vllm_ascend/worker/v2/spec_decode/domino/speculator.py",
        "AscendDominoSpeculator",
        {"propose"},
        Parent,
        {"Any": Any, "InputBatch": object, "build_attn_metadata_wrapper": attention_wrapper},
    )
    domino = cls()
    domino.result = object()
    args = tuple(object() for _ in range(15))
    assert domino.propose(*args, is_profile=True) is domino.result
    assert domino.input_batch is args[0]
    assert domino.forwarded == (args, {"is_profile": True})
    assert events == ["enter", "propose", "exit"]
    assert not hasattr(domino, "dcut_runtime")
