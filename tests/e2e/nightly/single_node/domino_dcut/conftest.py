# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def pytest_addoption(parser):
    group = parser.getgroup("domino-dcut")
    group.addoption("--domino-draft-model", default=None, help="Domino checkpoint to test on Ascend")
    group.addoption("--domino-target-model", default="Qwen/Qwen3-8B")
    group.addoption("--domino-dcut-tp", type=int, default=1)
