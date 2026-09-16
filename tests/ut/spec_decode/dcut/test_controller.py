# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest


@dataclass
class ScheduledBatch:
    num_scheduled_tokens: dict = field(default_factory=lambda: {"a": 4, "b": 4})
    total_num_scheduled_tokens: int = 8
    scheduled_spec_decode_tokens: dict = field(default_factory=lambda: {"a": [11, 12, 13], "b": [-1, -1, -1]})
    scheduled_new_reqs: list = field(default_factory=list)
    scheduled_encoder_inputs: dict = field(default_factory=dict)
    has_structured_output_requests: bool = False
    preempted_req_ids: set = field(default_factory=set)
    scheduled_cached_reqs: object = field(
        default_factory=lambda: SimpleNamespace(
            req_ids=["a", "b"], num_computed_tokens=[100, 120], resumed_req_ids=set()
        )
    )


def request_states():
    return SimpleNamespace(req_id_to_index={"a": 0, "b": 1}, prefill_len=SimpleNamespace(np=np.array([20, 20])))


def make_table(controller):
    table = controller.CostTable({"device": "test"})
    table.rows[controller.CostKey(2, 256, 8)] = controller.Cost(11.0, 10.0, 1.0, 5)
    table.rows[controller.CostKey(2, 256, 4)] = controller.Cost(3.0, 2.0, 1.0, 5)
    return table


def test_optimizer_selects_high_value_prefixes(dcut_modules):
    c = dcut_modules.controller
    table = make_table(c)
    caps = c.choose_caps(np.array([[0.9] * 3, [0.1] * 3]), np.array([3, 3]), table, 256, 0.02)
    assert caps.tolist() == [2, 0]


def test_optimizer_matches_exhaustive_prefix_search(dcut_modules):
    c = dcut_modules.controller
    probs = np.array([[0.9, 0.2, 0.1], [0.8, 0.7, 0.6]])
    limits = np.array([3, 2])
    gains = np.cumprod(probs, axis=1)
    for budget in range(int(limits.sum()) + 1):
        caps = c.allocate_prefixes(gains, limits, budget)
        actual = sum(gains[i, :cap].sum() for i, cap in enumerate(caps))
        reference = max(
            sum(gains[i, :cap].sum() for i, cap in enumerate(candidate))
            for candidate in itertools.product(range(4), range(3))
            if sum(candidate) == budget
        )
        assert actual == pytest.approx(reference)
        assert int(caps.sum()) == budget
        assert np.all(caps <= limits)


def test_equal_graph_bucket_cost_keeps_all_drafts(dcut_modules):
    c = dcut_modules.controller
    table = make_table(c)
    table.rows[c.CostKey(2, 256, 4)] = table.rows[c.CostKey(2, 256, 8)]
    caps = c.choose_caps(np.full((2, 3), 0.9), np.array([3, 3]), table, 256, 0.0)
    assert caps.tolist() == [3, 3]


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_probability_falls_back(dcut_modules, bad_value):
    c = dcut_modules.controller
    probabilities = np.full((2, 3), 0.8)
    probabilities[0, 0] = bad_value
    assert c.choose_caps(probabilities, np.array([3, 3]), make_table(c), 256, 0) is None


def test_unprofiled_context_or_full_baseline_falls_back(dcut_modules):
    c = dcut_modules.controller
    table = make_table(c)
    assert c.choose_caps(np.full((2, 3), 0.9), np.array([3, 3]), table, 512, 0) is None
    del table.rows[c.CostKey(2, 256, 8)]
    assert c.choose_caps(np.full((2, 3), 0.9), np.array([3, 3]), table, 256, 0) is None


def test_capture_gate_uses_cost_upper_bound(dcut_modules):
    c = dcut_modules.controller
    limits = np.array([3, 3])
    table = make_table(c)
    assert c.has_viable_budget(limits, table, 256, min_gain=0.02)
    table.rows[c.CostKey(2, 256, 4)] = c.Cost(10.9, 9.9, 1.0, 5)
    assert not c.has_viable_budget(limits, table, 256, min_gain=0.02)
    assert not c.has_viable_budget(limits, table, 512, min_gain=0.02)


def test_prefix_truncation_preserves_original_and_scheduler_accounting(dcut_modules):
    c = dcut_modules.controller
    original = ScheduledBatch()
    truncated = c.truncate_scheduler_output(original, ["a", "b"], np.array([2, 0]))
    assert original.total_num_scheduled_tokens == 8
    assert original.scheduled_spec_decode_tokens["a"] == [11, 12, 13]
    assert truncated.scheduled_spec_decode_tokens == {"a": [11, 12], "b": []}
    assert truncated.num_scheduled_tokens == {"a": 3, "b": 1}
    assert truncated.total_num_scheduled_tokens == sum(truncated.num_scheduled_tokens.values()) == 4
    # Scheduler treats the omitted suffix as rejected; worker rejects only verified tokens.
    for req_id, accepted in (("a", 2), ("b", 0)):
        scheduler_computed = original.num_scheduled_tokens[req_id] - (
            len(original.scheduled_spec_decode_tokens[req_id]) - accepted
        )
        worker_computed = truncated.num_scheduled_tokens[req_id] - (
            len(truncated.scheduled_spec_decode_tokens[req_id]) - accepted
        )
        assert scheduler_computed == worker_computed == 1 + accepted


@pytest.mark.parametrize("caps", [[4, 0], [-1, 0], [0.5, 0]])
def test_invalid_caps_cannot_remove_anchor(dcut_modules, caps):
    with pytest.raises(ValueError):
        dcut_modules.controller.truncate_scheduler_output(ScheduledBatch(), ["a", "b"], np.array(caps))


@pytest.mark.parametrize("unsafe", ["new", "prefill", "mixed", "structured", "resumed", "preempted", "unknown"])
def test_unsafe_batch_falls_back(dcut_modules, unsafe):
    batch, states = ScheduledBatch(), request_states()
    if unsafe == "new":
        batch.scheduled_new_reqs = [object()]
    elif unsafe == "prefill":
        batch.scheduled_cached_reqs.num_computed_tokens[0] = 1
    elif unsafe == "mixed":
        batch.num_scheduled_tokens["a"] += 1
        batch.total_num_scheduled_tokens += 1
    elif unsafe == "structured":
        batch.has_structured_output_requests = True
    elif unsafe == "resumed":
        batch.scheduled_cached_reqs.resumed_req_ids = {"a"}
    elif unsafe == "preempted":
        batch.preempted_req_ids = {"a"}
    else:
        del states.req_id_to_index["a"]
    assert dcut_modules.controller.decode_batch_info(batch, states, (256, 512)) is None


def test_context_and_calibration_query_budgets(dcut_modules):
    c = dcut_modules.controller
    assert c.context_bucket(257, (256, 512)) == 512
    assert c.context_bucket(513, (256, 512)) is None
    limits = np.array([3, 2])
    assert c.query_budgets(limits, (0.25, 0.5, 0.75, 1.0)) == [7, 6, 4, 2]
    assert c.query_budgets(limits, (0.5, 1.0), (0, 2)) == [7, 6, 4, 2]


def test_table_warmup_median_and_roundtrip(dcut_modules, tmp_path):
    c = dcut_modules.controller
    table = c.CostTable({"device": "test"})
    key = c.CostKey(2, 256, 4)
    assert not table.observe(key, 99, 9, 9, warmup=1, samples=3)
    assert not table.observe(key, 3, 2, 1, warmup=1, samples=3)
    assert not table.observe(key, 100, 10, 10, warmup=1, samples=3)
    assert table.observe(key, 5, 4, 2, warmup=1, samples=3)
    assert table.rows[key] == c.Cost(5, 4, 2, 3)
    assert table.rows[key].overhead_ms == 0
    path = str(tmp_path / "table.json")
    table.save(path)
    assert c.CostTable.load(path, table.fingerprint).rows == table.rows
    with pytest.raises(ValueError, match="does not match"):
        c.CostTable.load(path, {"device": "other"})


def test_malformed_cost_row_is_rejected(dcut_modules, tmp_path):
    c = dcut_modules.controller
    table = make_table(c)
    path = tmp_path / "table.json"
    table.save(str(path))
    data = json.loads(path.read_text())
    data["rows"][0]["step_ms"] = -1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="positive"):
        c.CostTable.load(str(path), table.fingerprint)
