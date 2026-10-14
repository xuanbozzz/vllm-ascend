# SPDX-License-Identifier: Apache-2.0
"""Verify route-slot ordering against the existing native grouped dispatcher."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.expert_offload.transfer import TorchExpertTransfer
from vllm_ascend.expert_offload.vq2a8 import VQ2A8ExpertAdapter, VQ2A8HostStore
from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
from vllm_ascend.ops.fused_moe.dataclass.router_input import MoeRouterInput
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchInput
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather
from vllm_ascend.ops.vq2a8 import VQ2A8_DTYPES, VQ2A8_FIELDS, VQ2A8ProjectionSpec, projection_shapes
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import is_950

pytestmark = pytest.mark.skipif(not is_950(), reason="Ascend950 VQ route-slot ABI")


@pytest.mark.parametrize("num_tokens", [1, 6])
def test_vq_verification_route_slots_match_native_grouped_combine(num_tokens):
    torch.npu.set_device(0)
    top_k = 6
    # Exercise duplicates, distinct token values and unequal probabilities.
    hidden = torch.arange(num_tokens * 64, dtype=torch.float32).reshape(num_tokens, 64).to("npu", torch.bfloat16)
    ids = torch.tensor([[2, 2, 4, 5, 3, 1]], device="npu", dtype=torch.int32).repeat(num_tokens, 1)
    weights = torch.tensor([[0.05, 0.1, 0.15, 0.2, 0.2, 0.3]], device="npu").repeat(num_tokens, 1)
    dispatcher = object.__new__(TokenDispatcherWithAllGather)
    dispatcher.top_k, dispatcher.num_experts_local, dispatcher.lora_context = top_k, 8, None
    inputs = MoETokenDispatchInput(
        hidden_states=hidden,
        topk_ids=ids,
        topk_weights=weights,
        routing=MoeRouterInput(
            expert_map=None, global_redundant_expert_num=0, mc2_mask=None, apply_router_weight_on_input=False
        ),
        quant=MoEQuantParams(quant_type=QuantType.VQ2A8),
        route_slot_layout=True,
    )
    route = dispatcher.token_dispatch(inputs)
    grouped = dispatcher.token_dispatch(replace(inputs, route_slot_layout=False))
    # Stand in for an expert-specific MLP, preserving BF16 rounding.
    route_output = route.hidden_states * (route.route_expert_ids[:, None] + 1).to(torch.bfloat16)
    grouped_ids = ids.flatten().sort().values
    grouped_output = grouped.hidden_states * (grouped_ids[:, None] + 1).to(torch.bfloat16)
    route_combined = dispatcher.token_combine(route_output, route.combine_metadata)
    grouped_combined = dispatcher.token_combine(grouped_output, grouped.combine_metadata)
    torch.testing.assert_close(route_combined, grouped_combined, rtol=0, atol=0)


def test_vq_native_h2d_keeps_slot_addresses_and_copies_every_field():
    torch.npu.set_device(0)
    ids = (2, 4, 5)
    specs = {"gate_up": VQ2A8ProjectionSpec(4096, 4096, 4096, 128), "down": VQ2A8ProjectionSpec(4096, 2048, 2048, 128)}
    layer = torch.nn.Module()
    layer.moe_config = SimpleNamespace(device="npu")
    originals = {}
    for prefix, kind in (("w13", "gate_up"), ("w2", "down")):
        for index, (field, dtype, shape) in enumerate(zip(VQ2A8_FIELDS, VQ2A8_DTYPES, projection_shapes(specs[kind]))):
            if field == "activation_order":
                value = torch.arange(shape[0], dtype=dtype).expand(len(ids), -1).clone()
            elif field == "rht_sign":
                value = torch.ones((len(ids), *shape), dtype=dtype)
                value[1].neg_()
            else:
                value = torch.stack([torch.full(shape, expert + index, dtype=dtype) for expert in ids])
            name = f"{prefix}_{field}"
            originals[name] = value
            parameter = torch.nn.Parameter(value.clone(), requires_grad=False)
            parameter.vq2a8_loaded = True
            layer.register_parameter(name, parameter)
    scheme = SimpleNamespace(layer_spec=SimpleNamespace(expert_ids=ids, layer_index=0, specs=specs))
    adapter = VQ2A8ExpertAdapter(layer, scheme)
    store = VQ2A8HostStore(layer, adapter)
    store.finalize(layer)
    resident = adapter.materialize(store, ids[:2], "npu")
    pointers = {name: tensor.data_ptr() for name, tensor in resident.items()}
    store.prepare_native(adapter, resident, "npu")
    TorchExpertTransfer.copy_experts(adapter, store, resident, ((5, 0), (2, 1)), "npu")
    for name, tensor in resident.items():
        assert tensor.data_ptr() == pointers[name]
        torch.testing.assert_close(tensor.cpu(), originals[name][[2, 0]], rtol=0, atol=0)
    store.close()
