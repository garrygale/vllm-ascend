# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest


def vllm_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"], hidden_size=4096, num_hidden_layers=36),
            enforce_eager=False,
        ),
        speculative_config=SimpleNamespace(
            method="domino",
            draft_sample_method="greedy",
            num_speculative_tokens=3,
            draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(block_size=3)),
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="PIECEWISE"), mode=SimpleNamespace(name="VLLM_COMPILE")
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, data_parallel_size=1),
        use_v2_model_runner=True,
        kv_transfer_config=None,
        lora_config=None,
    )


@pytest.mark.parametrize(
    "values",
    [
        {"enabled": "false"},
        {"enabled": True, "cost_table_path": ""},
        {"profile_samples": 0},
        {"profile_warmup": -1},
        {"min_gain": float("nan")},
        {"min_gain": -0.1},
        {"candidate_draft_lengths": [1.5]},
        {"candidate_ratios": []},
        {"candidate_ratios": [0]},
        {"candidate_ratios": [1.1]},
        {"candidate_ratios": [float("nan")]},
        {"candidate_ratios": [True]},
        {"candidate_ratios": "invalid"},
        {"context_buckets": []},
        {"context_buckets": [0]},
        {"unknown": 1},
    ],
)
def test_invalid_config_is_rejected(dcut_modules, values):
    with pytest.raises(ValueError):
        dcut_modules.config.DcutConfig.from_dict({"enabled": True, "cost_table_path": "cost.json", **values})


def test_disabled_feature_does_not_restrict_other_models(dcut_modules):
    dcut_modules.config.DcutConfig.from_dict({}).validate_model(object())


def test_supported_config_preserves_fixed_domino_block(dcut_modules):
    config = dcut_modules.config.DcutConfig.from_dict(
        {
            "enabled": True,
            "cost_table_path": "cost.json",
            "candidate_ratios": [1, 0.5, 0.25, 0.5],
            "candidate_draft_lengths": [3, 0, 1, 1],
        }
    )
    config.validate_model(vllm_config())
    assert config.candidate_ratios == (0.25, 0.5, 1.0)
    assert config.candidate_draft_lengths == (0, 1, 3)


@pytest.mark.parametrize(
    "unsupported",
    ["model", "layers", "architecture", "method", "sampling", "block", "graph", "v1", "pp", "kv", "lora"],
)
def test_unsupported_model_or_execution_mode_is_rejected(dcut_modules, unsupported):
    config = dcut_modules.config.DcutConfig.from_dict({"enabled": True, "cost_table_path": "cost.json"})
    vconfig = vllm_config()
    if unsupported == "model":
        vconfig.model_config.hf_config.hidden_size = 2048
    elif unsupported == "layers":
        vconfig.model_config.hf_config.num_hidden_layers = 32
    elif unsupported == "architecture":
        vconfig.model_config.hf_config.architectures = ["Qwen2ForCausalLM"]
    elif unsupported == "method":
        vconfig.speculative_config.method = "dflash"
    elif unsupported == "sampling":
        vconfig.speculative_config.draft_sample_method = "probabilistic"
    elif unsupported == "block":
        vconfig.speculative_config.num_speculative_tokens = 2
    elif unsupported == "graph":
        vconfig.compilation_config.cudagraph_mode.name = "FULL_DECODE_ONLY"
    elif unsupported == "v1":
        vconfig.use_v2_model_runner = False
    elif unsupported == "pp":
        vconfig.parallel_config.pipeline_parallel_size = 2
    elif unsupported == "kv":
        vconfig.kv_transfer_config = object()
    else:
        vconfig.lora_config = object()
    with pytest.raises(ValueError):
        config.validate_model(vconfig)


@pytest.mark.parametrize("enabled", [None, False])
def test_disabled_feature_ignores_all_inactive_options(dcut_modules, enabled):
    values = {
        "generate_cost_table": True,
        "cost_table_path": None,
        "profile_samples": 0,
        "min_gain": float("nan"),
        "candidate_ratios": "invalid",
        "candidate_draft_lengths": "invalid",
        "unused_option": object(),
    }
    if enabled is not None:
        values["enabled"] = enabled
    original = values.copy()
    config = dcut_modules.config.DcutConfig.from_dict(values)
    assert config == dcut_modules.config.DcutConfig()
    config.validate_model(object())
    assert values == original
