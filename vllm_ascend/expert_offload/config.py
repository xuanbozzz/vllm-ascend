# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ConfigDict, StrictBool

from vllm_ascend.config_utils import config

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@config(config=ConfigDict(frozen=True))
class ExpertOffloadConfig:
    """Exact single-worker offload. Unknown keys and ambiguous types fail fast."""

    enabled: StrictBool = False

    def validate_layer(self, num_experts: int, top_k: int) -> None:
        if self.enabled and not 1 <= top_k <= num_experts:
            raise ValueError("expert offload requires 1 <= top_k <= num_experts")

    def validate(self, vllm_config: VllmConfig, ascend_config) -> None:
        if not self.enabled:
            return
        if not vllm_config.use_v2_model_runner:
            raise ValueError("expert offload requires Model Runner V2")
        if not vllm_config.model_config.enforce_eager:
            raise ValueError("expert offload requires enforce_eager=True")
        if vllm_config.model_config.enable_sleep_mode:
            raise ValueError("expert offload PR01 does not support sleep mode")
        if vllm_config.scheduler_config.async_scheduling:
            raise ValueError("expert offload PR01 requires async_scheduling=False")
        parallel = vllm_config.parallel_config
        for name in (
            "tensor_parallel_size",
            "data_parallel_size",
            "pipeline_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        ):
            if getattr(parallel, name) != 1:
                raise ValueError(f"expert offload PR01 requires {name}=1")
        for name in ("enable_expert_parallel", "enable_eplb", "enable_dbo"):
            if getattr(parallel, name, False):
                raise ValueError(f"expert offload PR01 does not support {name}")
        for name in ("lora_config", "kv_transfer_config"):
            if getattr(vllm_config, name, None) is not None:
                raise ValueError(f"expert offload PR01 does not support {name}")
        if vllm_config.speculative_config is not None:
            raise ValueError("expert offload does not support speculative_config yet")
        offload = vllm_config.offload_config
        if offload.uva.cpu_offload_gb or offload.prefetch.offload_group_size:
            raise ValueError("expert offload cannot be combined with generic CPU weight offload")
        if str(vllm_config.load_config.load_format) not in ("auto", "safetensors", "pt"):
            raise ValueError("expert offload PR01 supports auto, safetensors, or pt checkpoint loading only")
        if getattr(vllm_config.model_config.hf_text_config, "model_type", None) != "deepseek_v4":
            raise ValueError("expert offload PR01 supports the DeepSeek V4 W4A8 MXFP model contract only")
        for name in ("mix_placement", "enable_force_eplb", "enable_fused_mc2", "multistream_overlap_shared_expert"):
            if getattr(ascend_config, name):
                raise ValueError(f"expert offload PR01 does not support {name}")
        if ascend_config.eplb_config.dynamic_eplb:
            raise ValueError("expert offload PR01 does not support dynamic EPLB")
        if ascend_config.rl_config.enabled:
            raise ValueError("expert offload PR01 does not support RL weight reloading")
