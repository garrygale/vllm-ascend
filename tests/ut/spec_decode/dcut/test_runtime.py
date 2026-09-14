# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_controller import ScheduledBatch, request_states


class FakeStream:
    def wait_stream(self, stream):
        pass


class FakeEvent:
    def __init__(self, backend):
        self.backend = backend
        self.ready = True
        self.waits = 0
        self.time = 0

    def record(self):
        self.backend.time += 1
        self.time = self.backend.time

    def query(self):
        return self.ready

    def synchronize(self):
        self.waits += 1
        self.ready = True

    def elapsed_time(self, other):
        return other.time - self.time


class FakeNpu:
    def __init__(self):
        self.time = 0

    def Event(self, **kwargs):
        return FakeEvent(self)

    def Stream(self):
        return FakeStream()

    def current_stream(self):
        return FakeStream()

    def stream(self, stream):
        return nullcontext()


class FakeTp:
    def __init__(self, bus, rank):
        self.bus = bus
        self.rank_in_group = rank

    def broadcast_object(self, value, src=0):
        if self.rank_in_group == src:
            self.bus["message"] = value
        return self.bus["message"]


def runtime_for(modules, tmp_path, generate=False, bus=None, rank=0, wait_for_probs=True):
    config = modules.config.DcutConfig.from_dict(
        {
            "enabled": True,
            "cost_table_path": str(tmp_path / "cost.json"),
            "generate_cost_table": generate,
            "wait_for_probs": wait_for_probs,
            "profile_warmup": 0,
            "profile_samples": 1,
        }
    )
    if not generate and rank == 0:
        table = modules.controller.CostTable({"device": "test", "context_buckets": list(config.context_buckets)})
        table.rows[modules.controller.CostKey(2, 256, 8)] = modules.controller.Cost(10, 1, 5)
        table.rows[modules.controller.CostKey(2, 256, 4)] = modules.controller.Cost(2, 1, 5)
        table.save(config.cost_table_path)
    return modules.runtime.DcutRuntime(
        config, {"device": "test"}, 4, 3, torch.device("cpu"), FakeTp(bus if bus is not None else {}, rank), FakeNpu()
    )


def proposal(runtime, values, req_ids=("a", "b")):
    assert runtime.begin_proposal(False, False)
    runtime.end_proposal(SimpleNamespace(num_reqs=len(req_ids), req_ids=list(req_ids)), torch.tensor(values))


def test_selected_probability_matches_softmax_with_extreme_logits(dcut_modules):
    logits = torch.tensor([[1000.0, 999.0, -float("inf")], [-1000.0, -999.0, -998.0]])
    actual = dcut_modules.runtime.selected_greedy_probability(logits)
    expected = logits.softmax(-1).amax(-1)
    torch.testing.assert_close(actual, expected)


def test_transfer_reorders_requests_and_is_consumed_once(dcut_modules, tmp_path):
    runtime = runtime_for(dcut_modules, tmp_path)
    source = torch.tensor([[0.1] * 3, [0.9] * 3])
    assert runtime.begin_proposal(False, False)
    runtime.end_proposal(SimpleNamespace(num_reqs=2, req_ids=["b", "a"]), source)
    source.zero_()
    req_ids, caps = runtime.select_caps(ScheduledBatch(), request_states())
    assert req_ids == ["a", "b"]
    assert caps.tolist() == [2, 0]
    assert runtime.stats["trimmed_tokens"] == 4
    assert runtime.select_caps(ScheduledBatch(), request_states())[1] is None


def test_unready_probability_does_not_wait_in_inference(dcut_modules, tmp_path):
    runtime = runtime_for(dcut_modules, tmp_path, wait_for_probs=False)
    proposal(runtime, [[0.9] * 3, [0.1] * 3])
    runtime.copy_event.ready = False
    assert runtime.select_caps(ScheduledBatch(), request_states())[1] is None
    assert runtime.copy_event.waits == 0
    runtime.copy_event.ready = True
    assert runtime.select_caps(ScheduledBatch(), request_states())[1] is None


def test_changed_request_set_never_uses_stale_probability(dcut_modules, tmp_path):
    runtime = runtime_for(dcut_modules, tmp_path)
    proposal(runtime, [[0.9] * 3, [0.1] * 3], req_ids=("a", "other"))
    assert runtime.select_caps(ScheduledBatch(), request_states())[1] is None


def test_tp_peers_apply_root_caps_without_probability_copy(dcut_modules, tmp_path):
    bus = {}
    root = runtime_for(dcut_modules, tmp_path, bus=bus)
    peer = runtime_for(dcut_modules, tmp_path, bus=bus, rank=1)
    proposal(root, [[0.9] * 3, [0.1] * 3])
    root_ids, root_caps = root.select_caps(ScheduledBatch(), request_states())
    peer_ids, peer_caps = peer.select_caps(ScheduledBatch(), request_states())
    assert root_ids == peer_ids
    np.testing.assert_array_equal(root_caps, peer_caps)
    assert root.stats == peer.stats
    assert peer.host_probs is None
    assert not peer.begin_proposal(False, False)


def test_generation_profiles_baseline_then_shorter_budget(dcut_modules, tmp_path):
    runtime = runtime_for(dcut_modules, tmp_path, generate=True)
    req_ids, caps = runtime.select_caps(ScheduledBatch(), request_states())
    assert caps.tolist() == [3, 3]
    runtime.begin_target()
    proposal(runtime, [[0.9] * 3, [0.1] * 3])
    req_ids, caps = runtime.select_caps(ScheduledBatch(), request_states())
    assert caps.tolist() == [2, 2]
    key = dcut_modules.controller.CostKey(2, 256, 8)
    assert key in runtime.table.rows
    assert runtime.stats["profiled_rows"] == 1
    assert dcut_modules.controller.CostTable.load(runtime.config.cost_table_path, runtime.table.fingerprint).rows


def test_dummy_proposal_does_not_copy_or_profile(dcut_modules, tmp_path):
    runtime = runtime_for(dcut_modules, tmp_path, generate=True)
    assert not runtime.begin_proposal(True, False)
    assert not runtime.begin_proposal(False, True)
    assert runtime.snapshot_req_ids is None
    assert runtime.table.rows == {}


def test_missing_table_is_not_silently_accepted(dcut_modules, tmp_path):
    config = dcut_modules.config.DcutConfig.from_dict(
        {
            "enabled": True,
            "cost_table_path": str(tmp_path / "missing.json"),
        }
    )
    with pytest.raises(ValueError, match="does not exist"):
        dcut_modules.runtime.DcutRuntime(config, {}, 4, 3, torch.device("cpu"), FakeTp({}, 0), FakeNpu())


def test_final_calibration_sample_is_flushed_and_not_reprofiled(dcut_modules, tmp_path):
    runtime = runtime_for(dcut_modules, tmp_path, generate=True)
    runtime.select_caps(ScheduledBatch(), request_states())
    runtime.begin_target()
    proposal(runtime, [[0.9] * 3, [0.1] * 3])
    stats = runtime.finish_calibration()
    assert stats["profiled_rows"] == 1
    assert runtime.measurement_key is None
    assert runtime.finish_calibration() == stats
    stats["profiled_rows"] = -1
    assert runtime.stats["profiled_rows"] == 1


def test_inference_never_changes_cost_table(dcut_modules, tmp_path):
    runtime = runtime_for(dcut_modules, tmp_path)
    before = (tmp_path / "cost.json").read_bytes()
    proposal(runtime, [[0.9] * 3, [0.1] * 3])
    runtime.select_caps(ScheduledBatch(), request_states())
    runtime.begin_target()
    proposal(runtime, [[0.9] * 3, [0.1] * 3])
    runtime.finish_calibration()
    assert runtime.current_key is runtime.measurement_key is None
    assert (tmp_path / "cost.json").read_bytes() == before


def test_completed_calibration_shapes_do_not_create_timing_events(dcut_modules, tmp_path):
    runtime = runtime_for(dcut_modules, tmp_path, generate=True)
    c = dcut_modules.controller
    for query in (2, 4, 6, 8):
        runtime.table.rows[c.CostKey(2, 256, query)] = c.Cost(1, 1, 1)
    runtime.select_caps(ScheduledBatch(), request_states())
    runtime.begin_target()
    assert runtime.current_key is runtime.target_start is None


def test_default_inference_waits_for_current_probabilities(dcut_modules, tmp_path):
    runtime = runtime_for(dcut_modules, tmp_path)
    proposal(runtime, [[0.9] * 3, [0.1] * 3])
    runtime.copy_event.ready = False
    assert runtime.select_caps(ScheduledBatch(), request_states())[1].tolist() == [2, 0]
    assert runtime.copy_event.waits == 1
