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


def table_fingerprint(config):
    return {
        "device": "test",
        "context_buckets": list(config.context_buckets),
    }


def runtime_for(
    modules,
    tmp_path,
    generate=False,
    bus=None,
    rank=0,
    wait_for_probs=True,
    flat_cost=False,
    score_diagnostics=False,
    performance_diagnostics=False,
):
    config = modules.config.DcutConfig.from_dict(
        {
            "enabled": True,
            "cost_table_path": str(tmp_path / "cost.json"),
            "generate_cost_table": generate,
            "wait_for_probs": wait_for_probs,
            "score_diagnostics": score_diagnostics,
            "performance_diagnostics": performance_diagnostics,
            "profile_warmup": 0,
            "profile_samples": 1,
        }
    )
    if not generate and rank == 0:
        table = modules.controller.CostTable(table_fingerprint(config))
        table.rows[modules.controller.CostKey(2, 256, 8)] = modules.controller.Cost(11, 10, 1, 5)
        short_step = 10.9 if flat_cost else 3
        table.rows[modules.controller.CostKey(2, 256, 4)] = modules.controller.Cost(short_step, 2, 1, 5)
        table.save(config.cost_table_path)
    return modules.runtime.DcutRuntime(
        config, {"device": "test"}, 4, 3, torch.device("cpu"), FakeTp(bus if bus is not None else {}, rank), FakeNpu()
    )


def enable_capture(runtime):
    runtime.select_caps(ScheduledBatch(), request_states())
    assert runtime.capture_requested


def proposal(runtime, values, req_ids=("a", "b")):
    if not runtime.config.generate_cost_table and not runtime.capture_requested:
        enable_capture(runtime)
    assert runtime.begin_proposal(False, False)
    runtime.end_proposal(SimpleNamespace(num_reqs=len(req_ids), req_ids=list(req_ids)), torch.tensor(values))


def test_selected_probability_gathers_already_selected_ids(dcut_modules):
    logits = torch.tensor([[1000.0, 999.0, -float("inf")], [-1000.0, -999.0, -998.0]])
    selected_ids = torch.tensor([0, 1])
    actual = dcut_modules.runtime.selected_greedy_probability(logits, selected_ids)
    expected = logits.softmax(-1).gather(-1, selected_ids[:, None]).squeeze(-1)
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="batch dimensions"):
        dcut_modules.runtime.selected_greedy_probability(logits, selected_ids[:, None])


def test_transfer_reorders_requests_and_is_consumed_once(dcut_modules, tmp_path):
    runtime = runtime_for(dcut_modules, tmp_path)
    enable_capture(runtime)
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
    assert runtime.stats["fallback_reasons"]["probabilities_not_ready"] == 1
    runtime.copy_event.ready = True
    assert runtime.select_caps(ScheduledBatch(), request_states())[1] is None


def test_changed_request_set_never_uses_stale_probability(dcut_modules, tmp_path):
    runtime = runtime_for(dcut_modules, tmp_path)
    proposal(runtime, [[0.9] * 3, [0.1] * 3], req_ids=("a", "other"))
    assert runtime.select_caps(ScheduledBatch(), request_states())[1] is None
    assert runtime.stats["fallback_reasons"]["request_id_mismatch"] == 1


def test_cost_gate_skips_capture_for_flat_small_batch(dcut_modules, tmp_path):
    runtime = runtime_for(dcut_modules, tmp_path, flat_cost=True)
    runtime.snapshot_req_ids = ["a", "b"]  # Simulate a copy requested before a tail transition.
    runtime.copy_event.ready = False
    assert runtime.select_caps(ScheduledBatch(), request_states())[1] is None
    assert not runtime.capture_requested
    assert not runtime.begin_proposal(False, False)
    assert runtime.stats["capture_skips"] == 1
    assert runtime.stats["fallback_reasons"] == {"no_viable_budget": 1}
    assert runtime.stats["fallback_shapes"] == {
        "no_viable_budget|batch=2,context=256,query=8,min_k=3,max_k=3": 1
    }
    assert runtime.snapshot_req_ids is None
    assert runtime.copy_event.waits == 0


def test_missing_baseline_and_invalid_batch_have_distinct_reasons(dcut_modules, tmp_path):
    runtime = runtime_for(dcut_modules, tmp_path)
    del runtime.table.rows[dcut_modules.controller.CostKey(2, 256, 8)]
    assert runtime.select_caps(ScheduledBatch(), request_states())[1] is None
    unsafe = ScheduledBatch(scheduled_new_reqs=[object()])
    assert runtime.select_caps(unsafe, request_states())[1] is None
    assert runtime.stats["fallback_reasons"] == {
        "missing_baseline_row": 1,
        "new_requests": 1,
    }
    assert runtime.stats["capture_skips"] == 1


def test_successful_full_k_decision_is_not_a_fallback(dcut_modules, tmp_path):
    runtime = runtime_for(dcut_modules, tmp_path, score_diagnostics=True)
    key = dcut_modules.controller.CostKey(2, 256, 4)
    runtime.table.rows[key] = dcut_modules.controller.Cost(10, 9, 1, 5)
    proposal(runtime, [[0.99] * 3, [0.99] * 3])
    _, caps = runtime.select_caps(ScheduledBatch(), request_states())
    assert caps.tolist() == [3, 3]
    assert runtime.stats["full_k_selected"] == 1
    assert runtime.stats["decisions"] == 0
    assert runtime.stats["fallbacks"] == 1  # Initial capture has no preceding snapshot.
    diagnostic = runtime.finish_calibration()["score_diagnostics"]["batch=2,context=256,query=8"]
    assert diagnostic["samples"] == 1
    assert diagnostic["selected_query_counts"] == {"8": 1}
    assert diagnostic["baseline"]["avg_draft_tokens_per_request"] == 3
    short = diagnostic["candidates"]["4"]
    assert short["avg_draft_tokens_per_request"] == 1
    assert short["selected"] == 0
    assert short["beats_full_k"] == 0
    assert short["clears_min_gain"] == 0
    assert short["avg_expected_token_loss"] > short["cost_saving"]
    assert short["avg_loss_minus_cost_saving"] > 0
    assert short["avg_score_gain"] < 0


def test_nonfallback_performance_uses_actual_output_tokens(dcut_modules, tmp_path):
    runtime = runtime_for(dcut_modules, tmp_path, performance_diagnostics=True)
    proposal(runtime, [[0.9] * 3, [0.1] * 3])
    _, caps = runtime.select_caps(ScheduledBatch(), request_states())
    assert caps.tolist() == [2, 0]
    step_id = runtime.current_performance_step_id
    assert step_id is not None

    runtime.record_step_output(step_id, [[10, 11, 12], [20]])
    stats = runtime.finish_calibration()["nonfallback_performance"]

    assert stats["steps"] == 1
    assert stats["pending_steps"] == 0
    assert stats["trimmed_steps"] == 1
    assert stats["full_k_steps"] == 0
    assert stats["coverage"] == pytest.approx(0.5)
    assert stats["output_tokens"] == 4
    assert stats["output_request_steps"] == 2
    assert stats["elapsed_ms"] > 0
    assert stats["request_step_elapsed_ms"] == pytest.approx(stats["elapsed_ms"] * 2)
    assert stats["output_tokens_per_second"] == pytest.approx(4000 / stats["elapsed_ms"])
    assert stats["aggregate_time_ms_per_output_token"] == pytest.approx(stats["elapsed_ms"] / 4)
    assert stats["tpot_ms_per_token"] == pytest.approx(stats["elapsed_ms"] / 2)
    assert stats["cost_table"]["selected_elapsed_ms"] == 3
    assert stats["cost_table"]["full_k_elapsed_ms"] == 11
    assert stats["cost_table"]["cost_only_latency_reduction"] == pytest.approx(1 - 3 / 11)
    assert stats["expected"]["throughput_change_vs_full_k"] > 0
    assert stats["expected"]["selected_tpot_ms_per_token"] > 0
    assert stats["expected"]["full_k_tpot_ms_per_token"] > 0
    assert stats["expected"]["tpot_reduction_vs_full_k"] > 0


def test_fallback_is_excluded_from_nonfallback_performance(dcut_modules, tmp_path):
    runtime = runtime_for(dcut_modules, tmp_path, performance_diagnostics=True)
    runtime.select_caps(ScheduledBatch(), request_states())
    assert runtime.current_performance_step_id is None
    stats = runtime.finish_calibration()["nonfallback_performance"]
    assert stats["steps"] == 0
    assert stats["output_tokens"] == 0


def test_tp_peers_apply_root_caps_and_capture_gate(dcut_modules, tmp_path):
    bus = {}
    root = runtime_for(dcut_modules, tmp_path, bus=bus, score_diagnostics=True)
    peer = runtime_for(dcut_modules, tmp_path, bus=bus, rank=1, score_diagnostics=True)
    enable_capture(root)
    peer.select_caps(ScheduledBatch(), request_states())
    assert peer.capture_requested
    proposal(root, [[0.9] * 3, [0.1] * 3])
    root_ids, root_caps = root.select_caps(ScheduledBatch(), request_states())
    peer_ids, peer_caps = peer.select_caps(ScheduledBatch(), request_states())
    assert root_ids == peer_ids
    np.testing.assert_array_equal(root_caps, peer_caps)
    assert root.stats == peer.stats
    assert root.finish_calibration() == peer.finish_calibration()
    assert peer.host_probs is None
    assert not peer.begin_proposal(False, False)


def test_generation_rebuilds_mismatched_table(dcut_modules, tmp_path):
    config = dcut_modules.config.DcutConfig.from_dict(
        {
            "enabled": True,
            "cost_table_path": str(tmp_path / "cost.json"),
            "generate_cost_table": True,
        }
    )
    stale = dcut_modules.controller.CostTable(
        {"device": "stale", "context_buckets": list(config.context_buckets)}
    )
    stale.rows[dcut_modules.controller.CostKey(1, 256, 4)] = dcut_modules.controller.Cost(3, 1, 1, 1)
    stale.save(config.cost_table_path)

    runtime = dcut_modules.runtime.DcutRuntime(
        config, {"device": "test"}, 4, 3, torch.device("cpu"), FakeTp({}, 0), FakeNpu()
    )

    assert runtime.table.fingerprint == table_fingerprint(config)
    assert runtime.table.rows == {}
    assert dcut_modules.controller.CostTable.load(config.cost_table_path, table_fingerprint(config)).rows == {}


def test_inference_rejects_mismatched_table(dcut_modules, tmp_path):
    config = dcut_modules.config.DcutConfig.from_dict(
        {
            "enabled": True,
            "cost_table_path": str(tmp_path / "cost.json"),
            "generate_cost_table": False,
        }
    )
    stale = dcut_modules.controller.CostTable(
        {"device": "stale", "context_buckets": list(config.context_buckets)}
    )
    stale.save(config.cost_table_path)

    with pytest.raises(ValueError, match="does not match"):
        dcut_modules.runtime.DcutRuntime(
            config, {"device": "test"}, 4, 3, torch.device("cpu"), FakeTp({}, 0), FakeNpu()
        )

def test_generation_profiles_baseline_then_ratio_budget(dcut_modules, tmp_path):
    runtime = runtime_for(dcut_modules, tmp_path, generate=True)
    req_ids, caps = runtime.select_caps(ScheduledBatch(), request_states())
    assert caps.tolist() == [3, 3]
    assert runtime.capture_requested
    runtime.begin_target()
    proposal(runtime, [[0.9] * 3, [0.1] * 3])
    req_ids, caps = runtime.select_caps(ScheduledBatch(), request_states())
    assert caps.tolist() == [2, 2]
    key = dcut_modules.controller.CostKey(2, 256, 8)
    assert key in runtime.table.rows
    assert runtime.table.rows[key].step_ms > 0
    assert runtime.stats["profiled_rows"] == 1
    assert dcut_modules.controller.CostTable.load(runtime.config.cost_table_path, runtime.table.fingerprint).rows


def test_step_cost_uses_steady_state_wall_time(dcut_modules, tmp_path, monkeypatch):
    ticks = iter((10.0, 10.012))
    monkeypatch.setattr(dcut_modules.runtime.time, "perf_counter", lambda: next(ticks))
    runtime = runtime_for(dcut_modules, tmp_path, generate=True)
    runtime.select_caps(ScheduledBatch(), request_states())
    runtime.begin_target()
    proposal(runtime, [[0.9] * 3, [0.1] * 3])
    runtime.copy_event.ready = False
    runtime.finish_calibration()
    cost = runtime.table.rows[dcut_modules.controller.CostKey(2, 256, 8)]
    assert cost.step_ms == pytest.approx(12)
    assert cost.target_ms == 1
    assert cost.draft_ms == 1
    assert runtime.copy_event.waits == 1


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
    stats["fallback_reasons"]["mutated"] = 1
    assert runtime.stats["profiled_rows"] == 1
    assert "mutated" not in runtime.stats["fallback_reasons"]


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
        runtime.table.rows[c.CostKey(2, 256, query)] = c.Cost(3, 1, 1, 1)
    runtime.select_caps(ScheduledBatch(), request_states())
    runtime.begin_target()
    assert runtime.current_key is runtime.target_start is None


def test_default_inference_waits_for_current_probabilities(dcut_modules, tmp_path):
    runtime = runtime_for(dcut_modules, tmp_path)
    proposal(runtime, [[0.9] * 3, [0.1] * 3])
    runtime.copy_event.ready = False
    assert runtime.select_caps(ScheduledBatch(), request_states())[1].tolist() == [2, 0]
    assert runtime.copy_event.waits == 1
