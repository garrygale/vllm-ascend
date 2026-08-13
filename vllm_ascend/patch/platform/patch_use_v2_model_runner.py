import vllm.envs as envs
from vllm.config.vllm import VllmConfig


def _patched_use_v2_model_runner(self) -> bool:
    """Return VLLM_USE_V2_MODEL_RUNNER env directly.

    The upstream use_v2_model_runner gate-keeps the v2 runner with
    per-model architecture whitelists, Triton availability checks, and
    feature-support inspections. On Ascend the v2 runner is controlled
    by the VLLM_USE_V2_MODEL_RUNNER environment variable, except Domino,
    whose first implementation lives only on the V2 runner.
    """
    speculative_config = getattr(self, "speculative_config", None)
    if speculative_config is not None and speculative_config.method == "domino":
        return True
    use_v2 = envs.VLLM_USE_V2_MODEL_RUNNER
    if use_v2 is not None:
        return use_v2
    return False


VllmConfig.use_v2_model_runner = property(_patched_use_v2_model_runner)
