# SPDX-License-Identifier: Apache-2.0
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.expert_offload.config import ExpertOffloadConfig
from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager, LayerOffloadBinding, OffloadState
from vllm_ascend.expert_offload.graph import supports_descriptor
from vllm_ascend.expert_offload.vq2a8 import VQ2A8ExpertAdapter
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.quantization.vq2a8_runtime import VQ2A8Runtime

ROOT = Path(__file__).resolve().parents[3]


def vq_layer(ids=(2, 4, 5), name="target", loaded=True):
    layer = torch.nn.Module()
    layer.layer_name, layer.local_num_experts, layer.top_k = name, 8, 2
    layer.activation = "silu"
    layer.moe_config = SimpleNamespace(device="cpu", is_sequence_parallel=False, has_bias=False)
    for index, tensor_name in enumerate(VQ2A8ExpertAdapter.tensor_names):
        value = torch.tensor(ids, dtype=torch.float32).reshape(-1, 1) + index * 10
        parameter = torch.nn.Parameter(value, requires_grad=False)
        parameter.vq2a8_loaded = loaded
        layer.register_parameter(tensor_name, parameter)
    scheme = SimpleNamespace(layer_spec=SimpleNamespace(expert_ids=ids, layer_index=0, specs={}))
    layer._expert_offload_binding = LayerOffloadBinding(
        layer, scheme, ExpertOffloadConfig(enabled=True, num_device_experts=2)
    )
    return layer


def ready(*layers):
    for layer in layers:
        layer._expert_offload_binding.process_weights_after_loading(layer)
    manager = ExpertOffloadManager()
    manager.finalize_loading(torch.nn.ModuleList(layers))
    manager.get_comm = lambda *args: None
    return manager


def test_sparse_experts_transfer_every_field_and_publish_logical_mapping():
    layer = vq_layer()
    manager = ready(layer)
    binding = layer._expert_offload_binding
    assert binding.residency.logical_to_physical()[2] == 0
    with manager.execution(has_prefill=False):
        view, ids, _ = binding.prepare(layer, torch.tensor([[5, 4]]))
    assert ids.tolist() == [[0, 1]]
    for index, name in enumerate(VQ2A8ExpertAdapter.tensor_names):
        assert getattr(view, name).flatten().tolist() == [5 + index * 10, 4 + index * 10]
    # Pinned/raw host ownership is independent of the resident slot storage.
    assert binding.host_store.tensors["w13_packed_zn"].flatten().tolist() == [2, 4, 5]


def test_sparse_prefill_keeps_logical_routes_and_only_allocates_stored_rows():
    layer = vq_layer()
    manager = ready(layer)
    with manager.execution(has_prefill=True):
        view, ids, _ = layer._expert_offload_binding.prepare(layer, torch.tensor([[5, 2]]))
    assert ids.tolist() == [[5, 2]]
    assert view.w13_packed_zn.flatten().tolist() == [2, 4, 5]
    assert view.w13_packed_zn.shape[0] == 3


def test_missing_stored_expert_invalidates_manager_without_publishing():
    layer = vq_layer()
    manager = ready(layer)
    before = layer._expert_offload_binding.residency.logical_to_physical()
    with pytest.raises(KeyError), manager.execution(has_prefill=False):
        layer._expert_offload_binding.prepare(layer, torch.tensor([[7, 4]]))
    assert layer._expert_offload_binding.residency.logical_to_physical() == before
    assert manager.state is OffloadState.FAILED


def test_incomplete_stacked_checkpoint_is_rejected():
    layer = vq_layer(loaded=False)
    with pytest.raises(ValueError, match="Incomplete VQ"):
        layer._expert_offload_binding.process_weights_after_loading(layer)


def test_distinct_vq_layouts_get_distinct_pools():
    first, second = vq_layer(name="first"), vq_layer(ids=(0, 1, 2, 3), name="second")
    manager = ready(first, second)
    assert len(manager.vq_prefill_pools) == 2
    assert manager.get_prefill_pool(first._expert_offload_binding) is not manager.prefill_pool
    manager.close()
    assert not manager.vq_prefill_pools


def test_hash_inventory_bounds_verification_graph_capacity():
    layer = vq_layer(ids=(2, 4))
    binding = layer._expert_offload_binding
    assert supports_descriptor([binding], SimpleNamespace(num_tokens=6, uniform_token_count=6))


def test_full_and_resident_banks_share_fresh_validity():
    layer = torch.nn.Module()
    layer.vq2a8_layer_index = 0
    resident, full = SimpleNamespace(), SimpleNamespace()
    layer.vq2a8_weights = resident
    layer._expert_offload_binding = SimpleNamespace(adapter=SimpleNamespace(banks={"resident": resident, "full": full}))
    runtime = VQ2A8Runtime(torch.nn.ModuleList([layer]), "cpu", [0])
    resident.validity_buffer.fill_(True)
    runtime.check()
    runtime.begin()
    with pytest.raises(RuntimeError, match="rejecting"):
        runtime.check()
    full.validity_buffer.fill_(True)
    runtime.check()
    assert resident.validity_buffer.data_ptr() == full.validity_buffer.data_ptr()


def test_validity_rejects_missing_layer_owners():
    with pytest.raises(RuntimeError, match="every configured"):
        VQ2A8Runtime(torch.nn.Module(), "cpu", [0])


def isolated_method(path, class_name, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    tree = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(ROOT / path), "exec"), namespace)
    return namespace[name]


def test_verification_routes_preserve_token_weights_and_duplicates():
    method = isolated_method(
        "vllm_ascend/ops/fused_moe/token_dispatcher.py",
        "TokenDispatcherWithAllGather",
        "token_dispatch",
        {
            "torch": torch,
            "QuantType": QuantType,
            "has_lora": lambda context: False,
            "MoETokenDispatchOutput": SimpleNamespace,
            "MoEAllGatherCombineMetadata": SimpleNamespace,
        },
    )
    hidden = torch.tensor([[10.0], [20.0], [30.0]], dtype=torch.bfloat16)
    ids = torch.tensor([[2, 2], [4, 5], [5, 4]])
    weights = torch.tensor([[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])
    quant = SimpleNamespace(
        quant_type=QuantType.VQ2A8, dispatch_with_quant=False, is_quant=True, is_mxfp=False, mxfp=None
    )
    routing = SimpleNamespace(
        pertoken_scale=None,
        expert_map=None,
        global_redundant_expert_num=0,
        log2phy=None,
        apply_router_weight_on_input=False,
    )
    result = method(
        SimpleNamespace(top_k=2, lora_context=None),
        SimpleNamespace(
            quant=quant,
            routing=routing,
            hidden_states=hidden,
            topk_weights=weights,
            topk_ids=ids,
            route_slot_layout=True,
        ),
    )
    assert result.route_expert_ids.tolist() == [2, 2, 4, 5, 5, 4]
    assert result.hidden_states.flatten().tolist() == [10, 10, 20, 20, 30, 30]
    assert result.combine_metadata.expanded_row_idx.tolist() == list(range(6))
    restored = result.hidden_states.index_select(0, result.combine_metadata.expanded_row_idx.long())
    combined = (restored.float().reshape(3, 2, 1) * weights.unsqueeze(-1)).sum(1)
    torch.testing.assert_close(combined, hidden.float())


def test_chunked_device_execution_aggregates_failure_in_last_chunk():
    method = isolated_method(
        "vllm_ascend/ops/fused_moe/vq2a8.py",
        "VQ2A8ExpertWeights",
        "compute_device",
        {"torch": torch, "MAX_PROJECTION_JOBS": 6},
    )
    calls = []

    def chunk(hidden, ids, limit):
        calls.append(ids.tolist())
        return hidden * 2, None, (ids != 12).all()

    output, _, valid = method(
        SimpleNamespace(_compute_device_chunk=chunk), torch.arange(13).reshape(-1, 1), torch.arange(13), 0.0
    )
    assert [len(call) for call in calls] == [6, 6, 1]
    torch.testing.assert_close(output.flatten(), torch.arange(13) * 2)
    assert not valid


def validate_vq(config):
    path = ROOT / "vllm_ascend/quantization/configs/vq2a8_config.py"
    tree = ast.parse(path.read_text())
    node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "validate_vq2a8_config"
    )
    namespace = {"torch": torch, "Path": Path, "MAX_DECODE_GRAPH_CONTEXT": 128}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name](config)


def verification_config():
    return SimpleNamespace(
        additional_config={
            "vq2a8": {"decode_mode": "device"},
            "ascend_compilation_config": {"enable_npugraph_ex": False},
        },
        use_v2_model_runner=True,
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            quantization="vq2a8",
            enforce_eager=False,
            max_model_len=66560,
            enable_sleep_mode=False,
        ),
        speculative_config=SimpleNamespace(
            method="dspark",
            enforce_eager=True,
            enable_adaptive_verification=False,
            num_speculative_tokens=5,
            draft_model_config=SimpleNamespace(
                quantization="modelslim",
                hf_config=SimpleNamespace(model_type="deepseek_v4", architectures=["DSparkDraftModel"]),
            ),
        ),
        parallel_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(max_num_seqs=1, async_scheduling=False),
        load_config=SimpleNamespace(load_format="safetensors", device=None),
        device_config=SimpleNamespace(device="npu"),
        compilation_config=SimpleNamespace(
            mode=0,
            cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY"),
            cudagraph_capture_sizes=[6],
            max_cudagraph_capture_size=6,
            cudagraph_num_of_warmups=1,
        ),
        lora_config=None,
        kv_transfer_config=None,
        cache_config=SimpleNamespace(cpu_offload_gb=0),
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0), prefetch=SimpleNamespace(offload_group_size=0)
        ),
    )


def test_mrv2_accepts_64k_dspark_verification_with_original_draft():
    assert validate_vq(verification_config())["_max_device_tokens"] == 6


@pytest.mark.parametrize("failure", ["vq_draft", "adaptive", "async", "wrong_capture", "v1"])
def test_unsupported_verification_contracts_fail_before_loading(failure):
    config = verification_config()
    if failure == "vq_draft":
        config.speculative_config.draft_model_config.quantization = "vq2a8"
    elif failure == "adaptive":
        config.speculative_config.enable_adaptive_verification = True
    elif failure == "async":
        config.scheduler_config.async_scheduling = True
    elif failure == "wrong_capture":
        config.compilation_config.cudagraph_capture_sizes = [1]
    else:
        config.use_v2_model_runner = False
    with pytest.raises(ValueError):
        validate_vq(config)


def serve_module():
    spec = importlib.util.spec_from_file_location("vq_serve_contract", ROOT / "tools/serve_vq2a8.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_serving_command_keeps_draft_format_and_64k_chunked_prefill(tmp_path):
    target, draft = tmp_path / "target", tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    (target / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "vq2a8", "format": "ascend_vq2a8", "format_version": 1}})
    )
    (target / "model.safetensors.index.json").write_text("{}")
    (draft / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "modelslim"}}))
    module = serve_module()
    args = module.parse_args(["--model", str(target), "--draft-model", str(draft), "--num-speculative-tokens", "5"])
    command = module.build_command(args)
    assert command[command.index("--max-model-len") + 1] == "66560"
    assert command[command.index("--max-num-batched-tokens") + 1] == "2048"
    assert "--enable-chunked-prefill" in command
    assert "--no-enable-prefix-caching" in command
    compilation = json.loads(command[command.index("--compilation-config") + 1])
    assert compilation["cudagraph_capture_sizes"] == [6]
    speculative = json.loads(command[command.index("--speculative-config") + 1])
    assert speculative["model"] == str(draft.resolve())
    assert speculative["enforce_eager"]
    assert "quantization" not in speculative
    additional = json.loads(command[command.index("--additional-config") + 1])
    assert additional["expert_offload_config"] == {"enabled": True, "num_device_experts": 84}
