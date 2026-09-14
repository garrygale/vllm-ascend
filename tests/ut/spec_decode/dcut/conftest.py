# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated CPU contracts; no vLLM engine or NPU initialization is needed."""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def dcut_modules():
    root = Path(__file__).resolve().parents[4]
    package_name = "_ascend_dcut_cpu_tests"
    package = types.ModuleType(package_name)
    package.__path__ = [str(root / "vllm_ascend/worker/v2/spec_decode/dcut")]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location("_ascend_dcut_config_tests", root / "vllm_ascend/dcut_config.py")
    config = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = config
    spec.loader.exec_module(config)
    return types.SimpleNamespace(
        config=config,
        controller=importlib.import_module(package_name + ".controller"),
        runtime=importlib.import_module(package_name + ".runtime"),
    )
