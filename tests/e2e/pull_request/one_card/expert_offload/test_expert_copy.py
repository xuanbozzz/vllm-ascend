# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import acl  # type: ignore[import-not-found]
import pytest
import torch
import torch_npu

from vllm_ascend.expert_offload.host_store import HostExpertStore
from vllm_ascend.expert_offload.quantization import MXFP4ExpertAdapter
from vllm_ascend.expert_offload.transfer import TorchExpertTransfer
from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import (
    AscendW4A8MXFPDSDynamicFusedMoEMethod,
    maybe_normalize_mxfp_scale_layout,
)
from vllm_ascend.utils import is_950

pytestmark = pytest.mark.skipif(not is_950(), reason="950DT MXFP4 NZ regression")
ACL_MEMCPY_DEVICE_TO_HOST = 2
NUM_EXPERTS = 6
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 256
GROUP_SIZE = 32


@pytest.fixture
def expert_bundle():
    generator = torch.Generator().manual_seed(0)
    tensors = {
        "w13_weight": torch.randint(
            256, (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE // 2), dtype=torch.uint8, generator=generator
        ),
        "w2_weight": torch.randint(
            256, (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE // 2), dtype=torch.uint8, generator=generator
        ),
        "w13_weight_scale": torch.full(
            (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE // GROUP_SIZE), 127, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu),
        "w2_weight_scale": torch.full(
            (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE // GROUP_SIZE), 127, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu),
    }
    layer = SimpleNamespace(
        local_num_experts=NUM_EXPERTS,
        layer_name="test.experts",
        weight_loader=lambda *args, **kwargs: True,
        **{name: torch.nn.Parameter(tensor, requires_grad=False) for name, tensor in tensors.items()},
    )
    store = HostExpertStore(layer, MXFP4ExpertAdapter.tensor_names)
    for name in store.names:
        shards = ("w1", "w3") if name.startswith("w13") else ("w2",)
        store.loaded[name] = {(eid, shard) for eid in range(NUM_EXPERTS) for shard in shards}
    store.finalize(layer)
    # Exercise the real post-processing without initializing a model/runtime config.
    scheme = object.__new__(AscendW4A8MXFPDSDynamicFusedMoEMethod)
    scheme.group_size = GROUP_SIZE
    return MXFP4ExpertAdapter(SimpleNamespace(activation="silu"), scheme), store


def _physical_bytes(tensor):
    torch.npu.current_stream().synchronize()
    size = torch_npu.get_storage_size(tensor) * tensor.element_size()
    result = torch.empty(size, dtype=torch.uint8, device="cpu")
    error = acl.rt.memcpy(result.data_ptr(), size, tensor.untyped_storage().data_ptr(), size, ACL_MEMCPY_DEVICE_TO_HOST)
    assert error == 0
    return result


def _gemm(x, weight, scale, counts):
    quantized, token_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
    token_scale = maybe_normalize_mxfp_scale_layout(token_scale)
    return torch_npu.npu_grouped_matmul(
        x=[quantized],
        weight=[weight],
        scale=None,
        antiquant_scale=[scale],
        per_token_scale=[token_scale],
        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        split_item=2,
        group_type=0,
        group_list=torch.tensor(counts, dtype=torch.int64, device=x.device),
        group_list_type=0,
        x_dtype=torch.float8_e4m3fn,
        weight_dtype=torch_npu.float4_e2m1fn_x2,
        output_dtype=torch.bfloat16,
    )[0]


def test_nz_full_layer_copy_preserves_addresses_and_complete_bundle(expert_bundle, monkeypatch):
    adapter, store = expert_bundle
    indices = torch.arange(NUM_EXPERTS - 1, -1, -1)
    store.tensors = {name: tensor.index_select(0, indices) for name, tensor in store.tensors.items()}
    binding = SimpleNamespace(adapter=adapter, host_store=store, num_experts=NUM_EXPERTS, device=torch.device("npu:0"))
    expected = adapter.materialize(store, range(NUM_EXPERTS), binding.device)
    prototype = adapter.materialize(store, [0], binding.device)
    store.prepare_native(adapter, prototype, binding.device)
    del prototype
    destination = adapter.allocate_pool(store, NUM_EXPERTS, binding.device)
    pointers = {name: tensor.data_ptr() for name, tensor in destination.items()}

    def forbidden_conversion(*args):
        raise AssertionError("runtime prefill must not format-cast")

    monkeypatch.setattr(adapter.scheme, "process_weights_after_loading", forbidden_conversion)
    TorchExpertTransfer.copy_experts(
        adapter, store, destination, tuple((eid, eid) for eid in range(NUM_EXPERTS)), binding.device
    )
    actual = destination
    for name in actual:
        assert actual[name].data_ptr() == pointers[name]
        assert torch.equal(_physical_bytes(actual[name]), _physical_bytes(expected[name]))
    counts = list(range(1, NUM_EXPERTS + 1))
    x = torch.randn(NUM_EXPERTS, HIDDEN_SIZE, generator=torch.Generator().manual_seed(9)).to(
        binding.device, dtype=torch.bfloat16
    )
    gate, up = _gemm(x, actual["w13_weight"], actual["w13_weight_scale"], counts).chunk(2, -1)
    hidden = torch.nn.functional.silu(gate) * up
    output = _gemm(hidden, actual["w2_weight"], actual["w2_weight_scale"], counts)
    gate, up = _gemm(x, expected["w13_weight"], expected["w13_weight_scale"], counts).chunk(2, -1)
    reference = _gemm(torch.nn.functional.silu(gate) * up, expected["w2_weight"], expected["w2_weight_scale"], counts)
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    store.close()
