# SPDX-License-Identifier: Apache-2.0
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from tests.ut.expert_offload.test_vq2a8 import ready, validate_vq, verification_config, vq_layer
from vllm_ascend.expert_offload.config import ExpertOffloadConfig
from vllm_ascend.expert_offload.expert_offload_manager import LayerOffloadBinding
from vllm_ascend.expert_offload.vq2a8 import GS32ExpertAdapter


def gs32_layer():
    layer = torch.nn.Module()
    layer.layer_name, layer.local_num_experts, layer.top_k = "gs32-target", 8, 2
    layer.activation = "silu"
    layer.moe_config = SimpleNamespace(device="cpu", is_sequence_parallel=False, has_bias=False)
    for index, name in enumerate(GS32ExpertAdapter.tensor_names):
        value = torch.tensor([2, 4, 5]).reshape(3, 1) + 10 * index
        parameter = torch.nn.Parameter(value, requires_grad=False)
        parameter.vq2a8_loaded = True
        layer.register_parameter(name, parameter)
    scheme = SimpleNamespace(
        checkpoint_format="ascend_vq2a8_gs32",
        layer_spec=SimpleNamespace(expert_ids=(2, 4, 5), layer_index=0, specs={}),
    )
    layer._expert_offload_binding = LayerOffloadBinding(
        layer, scheme, ExpertOffloadConfig(enabled=True, num_device_experts=2)
    )
    return layer


def test_gs32_eviction_preserves_all_ten_fields_and_pool_identity():
    layer, rht = gs32_layer(), vq_layer(name="rht-target")
    manager = ready(layer, rht)
    binding = layer._expert_offload_binding
    assert isinstance(binding.adapter, GS32ExpertAdapter)
    addresses = {name: value.data_ptr() for name, value in binding.resident.items()}
    assert len(addresses) == 10
    with manager.execution(has_prefill=False):
        view, ids, _ = binding.prepare(layer, torch.tensor([[5, 4]]))
    assert ids.tolist() == [[0, 1]]
    for index, name in enumerate(GS32ExpertAdapter.tensor_names):
        assert getattr(view, name).flatten().tolist() == [5 + 10 * index, 4 + 10 * index]
        assert binding.resident[name].data_ptr() == addresses[name]
    assert (
        binding.adapter.signature(binding.host_store)[0]
        != rht._expert_offload_binding.adapter.signature(rht._expert_offload_binding.host_store)[0]
    )


def test_gs32_native_store_bounds_pinned_allocations_without_changing_bytes(monkeypatch):
    layer = gs32_layer()
    binding = layer._expert_offload_binding
    store = binding.host_store
    monkeypatch.setattr("vllm_ascend.expert_offload.host_store.MAX_PINNED_ALLOCATION_BYTES", 16)
    allocations = []
    original_empty = torch.empty

    def empty(*args, **kwargs):
        assert kwargs.pop("pin_memory")
        value = original_empty(*args, **kwargs)
        allocations.append(value.nbytes)
        return value

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(get_storage_size=lambda t: t.numel(), get_npu_format=lambda t: 2)
    )
    store.finalize(layer)
    expected = {name: tensor.view(torch.uint8).reshape(3, -1).clone() for name, tensor in store.tensors.items()}
    destination = binding.adapter.materialize(store, (2, 4), "cpu")
    store.prepare_native(binding.adapter, destination, "cpu")
    assert allocations and max(allocations) <= 16
    assert not store.tensors
    for name, prepared in store.prepared.items():
        assert isinstance(prepared.data, tuple)
        torch.testing.assert_close(torch.stack(prepared.data), expected[name])


def gs32_config():
    config = verification_config()
    config.model_config.hf_config = SimpleNamespace(
        quantization_config={"format": "ascend_vq2a8_gs32"}, num_experts_per_tok=6
    )
    config.additional_config["vq2a8"] = {
        "gs32_library": "/native/gs32.so",
        "gs32_binary": "/native/mixed.o",
        "gs32_execution": "full_graph",
    }
    return config


def test_gs32_accepts_fixed_six_token_verification_with_original_draft():
    result = validate_vq(gs32_config())
    assert result["_max_device_tokens"] == 6
    assert result["gs32_execution"] == "full_graph"
    assert "ascendc_library" not in result


@pytest.mark.parametrize("failure", ["two_cards", "vq_draft", "rows", "eager_mismatch", "rht_option"])
def test_gs32_rejects_unsupported_execution_before_loading(failure):
    config = gs32_config()
    if failure == "two_cards":
        config.parallel_config.tensor_parallel_size = 2
    elif failure == "vq_draft":
        config.speculative_config.draft_model_config.quantization = "vq2a8"
    elif failure == "rows":
        config.model_config.hf_config.num_experts_per_tok = 256
    elif failure == "eager_mismatch":
        config.additional_config["vq2a8"]["gs32_execution"] = "serial"
    else:
        config.additional_config["vq2a8"]["activation_preparation"] = "packed"
    with pytest.raises(ValueError):
        validate_vq(config)


@pytest.mark.parametrize("rows", [1, 6, 36])
@pytest.mark.parametrize("prefix", [False, True])
def test_device_tensor_routes_reorder_sparse_inventory_without_host_counts(rows, prefix):
    from vllm_ascend.ops.fused_moe.vq2a8_gs32_routes import prepare_routes

    hidden = torch.arange(rows * 32, dtype=torch.float32).reshape(rows, 32).bfloat16()
    counts = torch.tensor([0, rows // 2, 0, rows - rows // 2])
    # Physical slot zero stores logical expert three, slot one stores expert one.
    experts = torch.tensor([3, 1])
    mapping = torch.tensor([-1, 1, -1, 0])
    groups = counts.cumsum(0) if prefix else counts
    output, physical_counts, slots, inverse, flags = prepare_routes(hidden, groups, int(not prefix), experts, mapping)
    assert flags.tolist() == [1]
    assert physical_counts.tolist() == [rows - rows // 2, rows // 2]
    torch.testing.assert_close(output.index_select(0, inverse), hidden, rtol=0, atol=0)
    assert slots.tolist() == [0] * (rows - rows // 2) + [1] * (rows // 2)


@pytest.mark.parametrize("bad", ["missing_expert", "invalid_slot", "overflow_count", "negative_count"])
def test_device_tensor_routes_fail_closed(bad):
    from vllm_ascend.ops.fused_moe.vq2a8_gs32_routes import prepare_routes

    hidden = torch.ones(6, 32, dtype=torch.bfloat16)
    counts, experts, mapping = torch.tensor([2, 4]), torch.tensor([0, 1]), torch.tensor([0, 1])
    if bad == "missing_expert":
        experts[1] = 1000
    elif bad == "invalid_slot":
        mapping[1] = 1000
    elif bad == "overflow_count":
        counts[0] = torch.iinfo(torch.int64).max
    else:
        counts[0] = -1
    output, physical_counts, _, _, flags = prepare_routes(hidden, counts, 1, experts, mapping)
    assert flags.tolist() == [0]
    assert not output.any() and not physical_counts.any()


def test_launcher_keeps_original_draft_quantization_separate(tmp_path):
    from tools import serve_vq2a8

    target, draft = tmp_path / "target", tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    (target / "config.json").write_text(
        json.dumps(
            {
                "num_experts_per_tok": 6,
                "quantization_config": {"quant_method": "vq2a8", "format": "ascend_vq2a8_gs32", "format_version": 1},
            }
        )
    )
    (target / "model.safetensors.index.json").write_text("{}")
    (draft / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v4", "quantization_config": {"quant_method": "fp8"}})
    )
    library, binary = tmp_path / "native.so", tmp_path / "mixed.o"
    library.touch()
    binary.touch()
    args = serve_vq2a8.parse_args(
        [
            "--model",
            str(target),
            "--draft-model",
            str(draft),
            "--num-speculative-tokens",
            "5",
            "--gs32-library",
            str(library),
            "--gs32-binary",
            str(binary),
            "--dry-run",
        ]
    )
    command = serve_vq2a8.build_command(args)
    speculative = json.loads(command[command.index("--speculative-config") + 1])
    assert speculative["quantization"] == "deepseek_v4_fp8"
    assert speculative["enforce_eager"]
    assert command[command.index("--quantization") + 1] == "vq2a8"
    compilation = json.loads(command[command.index("--compilation-config") + 1])
    assert compilation["cudagraph_capture_sizes"] == [6]
    assert compilation["cudagraph_num_of_warmups"] == 1
    assert command[command.index("--kv-cache-dtype") + 1] == "fp8"
