# SPDX-License-Identifier: Apache-2.0
"""Offload final contiguous VQ payloads; keep native bank addresses stable."""

from dataclasses import replace
from types import SimpleNamespace

import torch

from .host_store import allocate_native_expert_bytes
from .types import ExpertTensorSpec, ExpertWeightView, PreparedExpertTensor


class VQ2A8HostStore:
    def __init__(self, layer, adapter):
        self.names = adapter.tensor_names
        self.expert_ids = tuple(adapter.scheme.layer_spec.expert_ids)
        self.rows = {expert: row for row, expert in enumerate(self.expert_ids)}
        self.num_experts = len(self.expert_ids)
        self.specs = tuple(
            ExpertTensorSpec(name, tuple(getattr(layer, name).shape), getattr(layer, name).dtype) for name in self.names
        )
        self.tensors = {}
        self.prepared = {}
        self.prototypes = {}
        self.finalized = False

    def finalize(self, layer, *, pin_memory=True):
        if self.finalized:
            raise RuntimeError("VQ host store can only finalize once")
        for spec in self.specs:
            parameter = getattr(layer, spec.name)
            if not parameter.vq2a8_loaded:
                raise ValueError(f"Incomplete VQ checkpoint: {spec.name}")
            tensor = parameter.detach()
            if tensor.device.type != "cpu" or tuple(tensor.shape) != spec.shape or tensor.dtype != spec.dtype:
                raise ValueError(f"Unexpected VQ host layout: {spec.name}")
            # Pin only the final bounded native byte blocks, retaining pageable
            # raw tensors until the layer's native store has been prepared.
            self.tensors[spec.name] = tensor.clone()
        self.pin_memory = pin_memory
        self.finalized = True

    def prepare_native(self, adapter, destination, device):
        import torch_npu

        if not self.finalized or self.prepared:
            raise RuntimeError("VQ preparation requires a finalized host store")
        for name, tensor in self.tensors.items():
            target = destination[name]
            size = tensor[0].numel() * tensor.element_size()
            if (
                not target.is_contiguous()
                or target.shape[1:] != tensor.shape[1:]
                or target.dtype != tensor.dtype
                or int(torch_npu.get_storage_size(target)) * target.element_size() != size * target.shape[0]
            ):
                raise ValueError("VQ offload requires unpadded contiguous expert blocks")
            raw_bytes = tensor.view(torch.uint8).reshape(self.num_experts, size)
            data = raw_bytes
            if self.pin_memory:
                data = allocate_native_expert_bytes(self.num_experts, size)
                for expert_id in range(self.num_experts):
                    data[expert_id].copy_(raw_bytes[expert_id])
            self.prepared[name] = PreparedExpertTensor(
                data=data,
                shape=tuple(target.shape[1:]),
                stride=tuple(target.stride()),
                dtype=target.dtype,
                device=target.device,
                npu_format=torch_npu.get_npu_format(target),
                expert_bytes=size,
            )
            self.prototypes[name] = tensor[:1].clone()
        self.tensors.clear()

    def close(self):
        self.tensors.clear()
        self.prepared.clear()
        self.prototypes.clear()


class VQ2A8ExpertAdapter:
    fields = ("packed_zn", "pair_lut", "activation_order", "weight_scale", "weight_bias", "rht_sign")
    tensor_names = tuple(f"{prefix}_{field}" for field in fields for prefix in ("w13", "w2"))

    def __init__(self, layer, scheme):
        self.scheme = scheme
        self.layer = layer
        self.banks = {}
        self.available_experts = tuple(scheme.layer_spec.expert_ids)
        layer.vq2a8_expert_ids = self.available_experts
        layer.vq2a8_layer_index = scheme.layer_spec.layer_index

    def materialize(self, host_store, expert_ids, device):
        rows = torch.tensor([host_store.rows[expert] for expert in expert_ids], dtype=torch.long)
        return {name: tensor.index_select(0, rows).to(device) for name, tensor in host_store.tensors.items()}

    def allocate_pool(self, host_store, count, device):
        prototypes = host_store.prototypes or {name: tensor[:1] for name, tensor in host_store.tensors.items()}
        return {
            name: tensor.expand(count, *tensor.shape[1:]).contiguous().to(device) for name, tensor in prototypes.items()
        }

    def signature(self, host_store):
        return ("vq2a8", host_store.specs, tuple(self.scheme.layer_spec.specs.items()))

    def make_view(self, layer, tensors, *, full_pool):
        view = ExpertWeightView(layer, tensors)
        if torch.device(layer.moe_config.device).type != "npu":
            return view
        key = "full" if full_pool else "resident"
        if key not in self.banks:
            from vllm_ascend.ops.fused_moe.vq2a8 import VQ2A8ExpertWeights

            self.scheme.ensure_native_features()
            count = tensors["w13_packed_zn"].shape[0]
            prototype = SimpleNamespace(
                **tensors,
                vq2a8_expert_ids=self.available_experts if full_pool else tuple(range(count)),
                vq2a8_num_experts=layer.local_num_experts if full_pool else count,
            )
            self.banks[key] = VQ2A8ExpertWeights.from_layer(
                prototype,
                gate_up_spec=self.scheme.layer_spec.specs["gate_up"],
                down_spec=self.scheme.layer_spec.specs["down"],
                native_options=self.scheme.native_options,
                prefill_graph_cache=self.scheme.prefill_graph_cache,
            )
        view.vq2a8_weights = self.banks[key]
        return view


class GS32ExpertAdapter(VQ2A8ExpertAdapter):
    fields = ("packed_zn", "pair_lut", "row_scale_bits", "perm", "weight_scale")
    tensor_names = tuple(f"{prefix}_{field}" for field in fields for prefix in ("w13", "w2"))

    def signature(self, host_store):
        return ("vq2a8_gs32", host_store.specs, tuple(self.scheme.layer_spec.specs.items()))

    def make_view(self, layer, tensors, *, full_pool):
        view = ExpertWeightView(layer, tensors)
        if torch.device(layer.moe_config.device).type != "npu":
            return view
        key = "full" if full_pool else "resident"
        if key not in self.banks:
            from vllm_ascend.ops.fused_moe.vq2a8_gs32 import GS32ExpertWeights

            count = tensors["w13_packed_zn"].shape[0]
            ids = self.available_experts if full_pool else tuple(range(count))
            prototype = SimpleNamespace(**tensors, vq2a8_num_experts=layer.local_num_experts if full_pool else count)
            spec = replace(self.scheme.layer_spec, expert_ids=ids)
            self.banks[key] = GS32ExpertWeights(prototype, spec, self.scheme.native_options)
        view.vq2a8_weights = self.banks[key]
        return view
