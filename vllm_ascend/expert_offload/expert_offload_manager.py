# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from enum import Enum, auto

import torch

from .host_store import HostExpertStore
from .prefill import PrefillExpertPool
from .quantization import MXFP4ExpertAdapter
from .types import ExpertWeightView


class OffloadState(Enum):
    REGISTERING = auto()
    READY = auto()
    FAILED = auto()
    CLOSED = auto()


class LayerOffloadBinding:
    """Constructed with its target layer; the runner adopts it after loading."""

    def __init__(self, layer, scheme, config):
        config.validate_layer(layer.local_num_experts, layer.top_k)
        if layer.moe_config.is_sequence_parallel or layer.moe_config.has_bias:
            raise ValueError("expert offload PR01 does not support sequence-parallel or biased experts")
        self.layer_name = layer.layer_name
        self.num_experts = layer.local_num_experts
        self.top_k = layer.top_k
        self.device = torch.device(layer.moe_config.device)
        self.adapter = MXFP4ExpertAdapter(layer, scheme)
        self.host_store = HostExpertStore(layer, self.adapter.tensor_names)
        self.manager: ExpertOffloadManager | None = None
        self.loaded = False

    def process_weights_after_loading(self, layer) -> None:
        if self.host_store.finalized:
            raise RuntimeError("expert offload does not support weight reload or repeated post-processing")
        try:
            self.host_store.finalize(layer, pin_memory=self.device.type == "npu")
            if self.device.type == "npu":
                # A single temporary expert supplies the physical layout. No per-layer slots persist.
                prototype = self.adapter.materialize(self.host_store, (0,), self.device)
                torch.npu.current_stream(self.device).synchronize()
                self.host_store.prepare_native(self.adapter, prototype, self.device)
                del prototype
            # Execution always receives an explicit full-pool view; retire raw Parameters.
            for name in self.adapter.tensor_names:
                parameter = getattr(layer, name)
                parameter.data = torch.empty(0, dtype=parameter.dtype, device="cpu")
            self.loaded = True
        except Exception:
            self.close()
            raise

    def prepare(self, layer, selected_ids):
        manager = self.manager
        if manager is None or manager.state is not OffloadState.READY or not manager.executing:
            raise RuntimeError("expert offload layer requires a ready runner-owned execution")
        try:
            tensors = manager.prefill_pool.fill(self)
            return (
                ExpertWeightView(layer, tensors),
                selected_ids,
                manager.get_comm(self, layer.moe_config, self.num_experts),
            )
        except Exception:
            manager.state = OffloadState.FAILED
            raise

    def close(self) -> None:
        self.host_store.close()
        self.loaded = False
        self.manager = None


class ExpertOffloadManager:
    """V2 runner-owned serialized eager lifecycle without global state."""

    def __init__(self):
        self.state = OffloadState.REGISTERING
        self.bindings: dict[str, LayerOffloadBinding] = {}
        self.prefill_pool = PrefillExpertPool()
        self.comm_methods = {}
        self.executing = False
        self.has_prefill = True
        self.is_profile = False

    def finalize_loading(self, model) -> None:
        if self.state is not OffloadState.REGISTERING:
            raise RuntimeError("expert offload can only finalize once")
        discovered = [
            binding
            for layer in model.modules()
            if (binding := getattr(layer, "_expert_offload_binding", None)) is not None
        ]
        try:
            for binding in discovered:
                if not binding.loaded or binding.layer_name in self.bindings:
                    raise RuntimeError("incomplete or duplicate expert offload layer registration")
                self.bindings[binding.layer_name] = binding
                binding.manager = self
                self.prefill_pool.initialize(binding)
            if not self.bindings:
                raise RuntimeError("expert offload found no supported target-model expert layers")
            self.state = OffloadState.READY
        except Exception:
            try:
                self.close()
            finally:
                # Also release layers not yet adopted when registration failed.
                for binding in discovered:
                    binding.close()
                self.state = OffloadState.FAILED
            raise

    def get_comm(self, binding, moe_config, count):
        # Lazy import avoids an initialization cycle with quantization and routed experts.
        from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl

        key = (binding.layer_name, count)
        if key not in self.comm_methods:
            config = replace(moe_config, num_experts=count, num_local_experts=count)
            self.comm_methods[key] = AllGatherCommImpl(config)
        return self.comm_methods[key]

    def begin_execution(self, *, is_profile: bool = False) -> None:
        if self.state is not OffloadState.READY or self.executing:
            raise RuntimeError("expert offload requires ready, serialized executions")
        self.executing = True
        # Conservative until V2 batch metadata proves this is decode-only.
        self.has_prefill = True
        self.is_profile = is_profile

    @contextmanager
    def execution(self, *, has_prefill: bool = True, is_profile: bool = False):
        """Own a serialized eager execution scope."""
        self.begin_execution(is_profile=is_profile)
        self.has_prefill = has_prefill
        failed = True
        try:
            yield
            failed = False
        finally:
            self.finish_execution(failed=failed)

    def finish_execution(self, *, failed: bool) -> None:
        self.executing = False
        self.has_prefill = True
        self.is_profile = False
        if failed:
            self.state = OffloadState.FAILED

    def close(self) -> None:
        try:
            for device in {binding.device for binding in self.bindings.values()}:
                if device.type == "npu":
                    torch.npu.current_stream(device).synchronize()
        finally:
            for binding in self.bindings.values():
                binding.close()
            self.bindings.clear()
            self.comm_methods.clear()
            self.prefill_pool.close()
            self.executing = False
            self.state = OffloadState.CLOSED
