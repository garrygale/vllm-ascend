# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from test_integration_contracts import production_methods


@pytest.mark.parametrize("mode", ["real", "dummy", "profile", "error"])
def test_opt_in_proposal_preserves_parent_call_and_runtime_cleanup(mode):
    events = []

    class Parent:
        def propose(self, *args, **kwargs):
            events.append("parent")
            self.forwarded = args, kwargs
            if mode == "error":
                raise RuntimeError("draft failed")
            return self.result

    class Runtime:
        def begin_proposal(self, dummy_run, is_profile):
            events.append("begin")
            return not dummy_run and not is_profile

        def end_proposal(self, batch, probs):
            events.append("end")
            assert batch is input_batch
            assert probs is speculator.selected_probs

        def abort_target(self):
            events.append("abort")

    cls = production_methods(
        "vllm_ascend/worker/v2/spec_decode/dcut/speculator.py",
        "DcutDominoSpeculator",
        {"propose"},
        Parent,
        {"Any": Any, "InputBatch": object},
    )
    speculator = cls()
    speculator.dcut_runtime = Runtime()
    speculator.selected_probs = torch.ones(2, 3)
    speculator.result = torch.tensor([[1, 2, 3], [4, 5, 6]])
    input_batch = SimpleNamespace(num_reqs=2)
    args = (input_batch, *(object() for _ in range(11)))
    kwargs = {"dummy_run": mode == "dummy", "is_profile": mode == "profile"}
    if mode == "error":
        with pytest.raises(RuntimeError, match="draft failed"):
            speculator.propose(*args, **kwargs)
        assert events == ["begin", "parent", "abort"]
    else:
        assert speculator.propose(*args, **kwargs) is speculator.result
        assert events == (["begin", "parent", "end"] if mode == "real" else ["begin", "parent"])
    assert speculator.forwarded == (args + (mode == "dummy", False, None), {"is_profile": mode == "profile"})
    assert not speculator._collect_dcut_probs
