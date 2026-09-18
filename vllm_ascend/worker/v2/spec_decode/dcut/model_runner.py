# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in runner; the disabled path uses the original NPUModelRunner."""

from typing import cast

import torch
from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.v2.model_runner import NPUModelRunner

from .controller import truncate_scheduler_output
from .runtime import DcutRuntime, model_fingerprint
from .speculator import DcutDominoSpeculator


class _DcutTrackedAsyncOutput(AsyncModelRunnerOutput):
    """Record accepted tokens after the regular async D2H parse completes."""

    def __init__(self, output: AsyncModelRunnerOutput, runtime: DcutRuntime, step_id: int):
        self.output = output
        self.runtime = runtime
        self.step_id = step_id

    def get_output(self) -> ModelRunnerOutput:
        try:
            output = self.output.get_output()
        except Exception:
            self.runtime.discard_step_output(self.step_id)
            raise
        self.runtime.record_step_output(self.step_id, output.sampled_token_ids)
        return output


class DcutNPUModelRunner(NPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        config = get_ascend_config().dcut_config
        # Platform normalization may change the requested target graph mode.
        config.validate_model(vllm_config)
        self.dcut_runtime: DcutRuntime | None = None
        self._dcut_performance_step_id: int | None = None
        super().__init__(vllm_config, device)
        self.dcut_runtime = DcutRuntime(
            config,
            model_fingerprint(vllm_config, torch.npu.get_device_name(self.device)),
            self.max_num_reqs,
            self.num_speculative_steps,
            self.device,
            get_tp_group(),
            torch.npu,
        )
        cast(DcutDominoSpeculator, self.speculator).set_dcut_runtime(self.dcut_runtime)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors=None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
    ):
        runtime = self.dcut_runtime
        performance_step_id = None
        try:
            if runtime is not None and self._dcut_performance_step_id is not None:
                # The V2 runner contract requires sample_tokens() immediately
                # after execute_model() returns None. Do not leak a measurement
                # if a caller violates that contract.
                runtime.discard_step_output(self._dcut_performance_step_id)
                self._dcut_performance_step_id = None
            if runtime is not None and not dummy_run and not is_profile:
                req_ids, caps = runtime.select_caps(scheduler_output, self.req_states)
                performance_step_id = runtime.current_performance_step_id
                if caps is not None:
                    # Preserve scheduler accounting; change only the worker view before graph dispatch.
                    scheduler_output = truncate_scheduler_output(scheduler_output, req_ids, caps)
                runtime.begin_target()
            output = super().execute_model(
                scheduler_output,
                intermediate_tensors=intermediate_tensors,
                dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                is_profile=is_profile,
            )
            if runtime is not None and performance_step_id is not None:
                if isinstance(output, AsyncModelRunnerOutput):
                    return _DcutTrackedAsyncOutput(output, runtime, performance_step_id)
                if isinstance(output, ModelRunnerOutput):
                    runtime.record_step_output(performance_step_id, output.sampled_token_ids)
                elif output is None:
                    # V2 splits forward and sampling. The normal decode path
                    # returns None here and produces AsyncModelRunnerOutput in
                    # sample_tokens(), so keep the measurement pending.
                    self._dcut_performance_step_id = performance_step_id
                else:
                    runtime.discard_step_output(performance_step_id)
            return output
        except Exception:
            self._dcut_performance_step_id = None
            if runtime is not None:
                runtime.abort_target()
            raise

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> AsyncModelRunnerOutput | ModelRunnerOutput | None:
        runtime = self.dcut_runtime
        performance_step_id = self._dcut_performance_step_id
        self._dcut_performance_step_id = None
        try:
            output = super().sample_tokens(grammar_output)
        except Exception:
            if runtime is not None:
                runtime.abort_target()
            raise

        if runtime is not None and performance_step_id is not None:
            if isinstance(output, AsyncModelRunnerOutput):
                return _DcutTrackedAsyncOutput(output, runtime, performance_step_id)
            if isinstance(output, ModelRunnerOutput):
                runtime.record_step_output(performance_step_id, output.sampled_token_ids)
            else:
                runtime.discard_step_output(performance_step_id)
        return output
