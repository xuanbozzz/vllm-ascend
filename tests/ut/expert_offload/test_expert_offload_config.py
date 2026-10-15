# SPDX-License-Identifier: Apache-2.0
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter, ValidationError

from vllm_ascend.expert_offload.config import ExpertOffloadConfig


def make_configs():
    vc = SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(
            model="/models/target",
            enforce_eager=True,
            enable_sleep_mode=False,
            hf_text_config=SimpleNamespace(model_type="deepseek_v4"),
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False, max_num_seqs=1),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
            enable_dbo=False,
        ),
        lora_config=None,
        speculative_config=None,
        kv_transfer_config=None,
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0), prefetch=SimpleNamespace(offload_group_size=0)
        ),
        load_config=SimpleNamespace(load_format="auto"),
    )
    ac = SimpleNamespace(
        mix_placement=False,
        enable_force_eplb=False,
        enable_fused_mc2=0,
        multistream_overlap_shared_expert=False,
        eplb_config=SimpleNamespace(dynamic_eplb=False),
        rl_config=SimpleNamespace(enabled=False),
    )
    return vc, ac


def test_defaults_and_disabled_validation():
    cfg = ExpertOffloadConfig()
    assert not cfg.enabled and cfg.num_device_experts is None
    cfg.validate(None, None)
    cfg.validate_layer(0, 0)


@pytest.mark.parametrize(
    "values",
    [
        {"enabled": True},
        {"enabled": "false"},
        {"enabled": 1},
        {"num_device_experts": True},
        {"num_device_experts": 0},
        {"num_device_experts": -1},
        {"num_device_experts": 2.5},
        {"num_device_experts": "4"},
        {"num_device_expert": 4},
    ],
)
def test_schema_rejects_invalid_values(values):
    with pytest.raises(ValidationError):
        TypeAdapter(ExpertOffloadConfig).validate_python(values)


def test_nested_dict_parsing_and_immutability():
    cfg = TypeAdapter(ExpertOffloadConfig).validate_python({"enabled": True, "num_device_experts": 4})
    cfg.validate(*make_configs())
    cfg.validate_layer(8, 4)
    with pytest.raises(FrozenInstanceError):
        cfg.enabled = False


@pytest.mark.parametrize("experts,top_k", [(4, 1), (3, 1), (8, 5), (8, 0)])
def test_layer_capacity_bounds(experts, top_k):
    with pytest.raises(ValueError, match="top_k"):
        ExpertOffloadConfig(enabled=True, num_device_experts=4).validate_layer(experts, top_k)


@pytest.mark.parametrize(
    "path,value,match",
    [
        ("use_v2_model_runner", False, "Runner V2"),
        ("model_config.enable_sleep_mode", True, "sleep mode"),
        ("model_config.hf_text_config.model_type", "qwen3_moe", "DeepSeek V4"),
        ("scheduler_config.async_scheduling", True, "async_scheduling"),
        ("parallel_config.tensor_parallel_size", 2, "tensor_parallel_size"),
        ("parallel_config.data_parallel_size", 2, "data_parallel_size"),
        ("parallel_config.pipeline_parallel_size", 2, "pipeline_parallel_size"),
        ("parallel_config.prefill_context_parallel_size", 2, "prefill_context_parallel_size"),
        ("parallel_config.decode_context_parallel_size", 2, "decode_context_parallel_size"),
        ("parallel_config.enable_expert_parallel", True, "enable_expert_parallel"),
        ("parallel_config.enable_eplb", True, "enable_eplb"),
        ("parallel_config.enable_dbo", True, "enable_dbo"),
        ("lora_config", object(), "lora_config"),
        ("speculative_config", object(), "speculative_config"),
        ("kv_transfer_config", object(), "kv_transfer_config"),
        ("offload_config.uva.cpu_offload_gb", 1, "generic CPU"),
        ("offload_config.prefetch.offload_group_size", 1, "generic CPU"),
        ("load_config.load_format", "dummy", "checkpoint loading"),
    ],
)
def test_unsupported_runner_combinations(path, value, match):
    vc, ac = make_configs()
    obj = vc
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
    with pytest.raises(ValueError, match=match):
        ExpertOffloadConfig(enabled=True, num_device_experts=4).validate(vc, ac)


@pytest.mark.parametrize(
    "name", ["mix_placement", "enable_force_eplb", "enable_fused_mc2", "multistream_overlap_shared_expert"]
)
def test_unsupported_ascend_combinations(name):
    vc, ac = make_configs()
    setattr(ac, name, True)
    with pytest.raises(ValueError, match=name):
        ExpertOffloadConfig(enabled=True, num_device_experts=4).validate(vc, ac)


@pytest.mark.parametrize("name", ["dynamic_eplb", "rl"])
def test_unsupported_weight_mutation(name):
    vc, ac = make_configs()
    if name == "rl":
        ac.rl_config.enabled = True
    else:
        ac.eplb_config.dynamic_eplb = True
    with pytest.raises(ValueError):
        ExpertOffloadConfig(enabled=True, num_device_experts=4).validate(vc, ac)


@pytest.mark.parametrize("enforce_eager", [False])
def test_non_eager_execution_is_rejected(enforce_eager):
    vc, ac = make_configs()
    vc.model_config.enforce_eager = enforce_eager
    with pytest.raises(ValueError, match="enforce_eager"):
        ExpertOffloadConfig(enabled=True, num_device_experts=4).validate(vc, ac)
