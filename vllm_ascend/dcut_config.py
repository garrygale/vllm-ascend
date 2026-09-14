# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for Qwen3-8B Domino verification-prefix selection."""

import math
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class DcutConfig:
    enabled: bool = False
    cost_table_path: str = ""
    generate_cost_table: bool = False
    candidate_draft_lengths: tuple[int, ...] = ()
    context_buckets: tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
    profile_warmup: int = 2
    profile_samples: int = 5
    min_gain: float = 0.02
    wait_for_probs: bool = True

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "DcutConfig":
        if not isinstance(values, dict):
            raise ValueError("dcut_config must be an object")
        enabled = values.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("dcut_config.enabled must be a boolean")
        if not enabled:
            # A disabled feature must not validate or act on its inactive parameters.
            return cls()
        unknown = values.keys() - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown dcut_config fields: {sorted(unknown)}")
        values = values.copy()
        for name in ("enabled", "generate_cost_table", "wait_for_probs"):
            if name in values and not isinstance(values[name], bool):
                raise ValueError(f"dcut_config.{name} must be a boolean")
        for name in ("candidate_draft_lengths", "context_buckets"):
            if name in values:
                items = values[name]
                if not isinstance(items, (list, tuple)) or any(type(item) is not int for item in items):
                    raise ValueError(f"dcut_config.{name} must be a list of integers")
                values[name] = tuple(sorted(set(items)))
        config = cls(**values)
        if not isinstance(config.cost_table_path, str):
            raise ValueError("dcut_config.cost_table_path must be a string")
        if config.enabled and not config.cost_table_path:
            raise ValueError("D-Cut requires cost_table_path")
        if config.generate_cost_table and not config.enabled:
            raise ValueError("generate_cost_table requires D-Cut to be enabled")
        for name, minimum in (("profile_warmup", 0), ("profile_samples", 1)):
            value = getattr(config, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"dcut_config.{name} must be an integer >= {minimum}")
        if isinstance(config.min_gain, bool) or not isinstance(config.min_gain, (float, int)):
            raise ValueError("dcut_config.min_gain must be a finite nonnegative number")
        if not math.isfinite(config.min_gain) or config.min_gain < 0:
            raise ValueError("dcut_config.min_gain must be a finite nonnegative number")
        if not config.context_buckets or config.context_buckets[0] <= 0:
            raise ValueError("context_buckets must contain positive integers")
        if config.candidate_draft_lengths and config.candidate_draft_lengths[0] < 0:
            raise ValueError("candidate_draft_lengths must be nonnegative")
        return config

    def validate_model(self, vllm_config: Any) -> None:
        if not self.enabled:
            return
        model = vllm_config.model_config
        hf_config = model.hf_config
        if (
            "Qwen3ForCausalLM" not in (getattr(hf_config, "architectures", None) or [])
            or getattr(hf_config, "hidden_size", None) != 4096
            or getattr(hf_config, "num_hidden_layers", None) != 36
        ):
            raise ValueError("D-Cut currently supports only Qwen3-8B")
        spec = vllm_config.speculative_config
        if spec is None or spec.method != "domino":
            raise ValueError("D-Cut requires speculative method=domino")
        if spec.draft_sample_method != "greedy":
            raise ValueError("D-Cut currently requires greedy Domino draft sampling")
        if spec.num_speculative_tokens != getattr(spec.draft_model_config.hf_config, "block_size", None):
            raise ValueError("Domino num_speculative_tokens must equal checkpoint block_size")
        if any(length > spec.num_speculative_tokens for length in self.candidate_draft_lengths):
            raise ValueError("candidate_draft_lengths cannot exceed Domino block_size")
        compilation = vllm_config.compilation_config
        if (
            getattr(compilation.cudagraph_mode, "name", None) != "PIECEWISE"
            or getattr(compilation.mode, "name", None) != "VLLM_COMPILE"
            or model.enforce_eager
        ):
            raise ValueError("D-Cut requires VLLM_COMPILE with Target cudagraph_mode=PIECEWISE")
        if not vllm_config.use_v2_model_runner:
            raise ValueError("D-Cut requires V2 ModelRunner")
        parallel = vllm_config.parallel_config
        if any(
            getattr(parallel, name, 1) != 1
            for name in (
                "pipeline_parallel_size",
                "data_parallel_size",
                "prefill_context_parallel_size",
                "decode_context_parallel_size",
            )
        ):
            raise ValueError("D-Cut supports TP with single PP/DP/PCP/DCP stage")
        if vllm_config.kv_transfer_config is not None or vllm_config.lora_config is not None:
            raise ValueError("D-Cut does not yet support KV transfer or LoRA")
