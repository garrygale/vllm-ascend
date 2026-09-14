# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in runner; the disabled path uses the original NPUModelRunner."""

from typing import cast

import torch
from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.v2.model_runner import NPUModelRunner

from .controller import truncate_scheduler_output
from .runtime import DcutRuntime, model_fingerprint
from .speculator import DcutDominoSpeculator


class DcutNPUModelRunner(NPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        config = get_ascend_config().dcut_config
        # Platform normalization may change the requested target graph mode.
        config.validate_model(vllm_config)
        self.dcut_runtime: DcutRuntime | None = None
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
        try:
            if runtime is not None and not dummy_run and not is_profile:
                req_ids, caps = runtime.select_caps(scheduler_output, self.req_states)
                if caps is not None:
                    # Preserve scheduler accounting; change only the worker view before graph dispatch.
                    scheduler_output = truncate_scheduler_output(scheduler_output, req_ids, caps)
                runtime.begin_target()
            return super().execute_model(
                scheduler_output,
                intermediate_tensors=intermediate_tensors,
                dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                is_profile=is_profile,
            )
        except Exception:
            if runtime is not None:
                runtime.abort_target()
            raise
