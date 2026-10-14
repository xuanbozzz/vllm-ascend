# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VQ2A8 expert parameters and the standard Ascend fused-MoE adapter."""

from typing import Any

import torch

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.vq2a8 import (
    VQ2A8_DTYPES,
    VQ2A8_FIELDS,
    load_vq2a8_library,
    projection_shapes,
    require_vq2a8_features,
)

from ..base import AscendMoEScheme, QuantType, get_moe_num_logical_experts
from ..registry import register_scheme


@register_scheme("VQ2A8", "moe")
class AscendVQ2A8FusedMoEScheme(AscendMoEScheme):
    """Load final compressed tensors; routing and communication stay in FusedMoE."""

    quant_type = QuantType.VQ2A8
    supports_eplb = False

    def __init__(self, layer_spec, *, native_options=None, prefill_graph_cache=None):
        self.layer_spec = layer_spec
        self.native_options = dict(native_options or {})
        self.prefill_graph_cache = prefill_graph_cache

    def get_weight(self, num_experts, intermediate_size_per_partition, hidden_sizes, params_dtype):
        if params_dtype != torch.bfloat16:
            raise ValueError("VQ2A8 requires BF16 model activations.")
        ids = self.layer_spec.expert_ids
        if not ids or len(set(ids)) != len(ids) or any(expert < 0 or expert >= num_experts for expert in ids):
            raise ValueError("VQ2A8 stored experts must uniquely map into the logical expert inventory.")
        expected = {
            "gate_up": (2 * intermediate_size_per_partition, hidden_sizes),
            "down": (hidden_sizes, intermediate_size_per_partition),
        }
        parameters = {}
        for prefix, kind in (("w13", "gate_up"), ("w2", "down")):
            spec = self.layer_spec.specs[kind]
            if (spec.rows, spec.rht_true_columns) != expected[kind]:
                raise ValueError(f"VQ2A8 {kind} geometry does not match the unsharded MoE layer.")
            for field, dtype, shape in zip(VQ2A8_FIELDS, VQ2A8_DTYPES, projection_shapes(spec), strict=True):
                parameters[f"{prefix}_{field}"] = torch.empty((len(ids), *shape), dtype=dtype)
        return parameters

    def get_dynamic_quant_param(self, num_experts, intermediate_size_per_partition, hidden_sizes, params_dtype):
        # Input-side RHT/normalization tensors are included in get_weight():
        # they must not inherit the scalar-quantizer CHANNEL/GROUP slicing.
        return {}

    def get_weight_attrs(self, name: str) -> dict[str, Any]:
        return {
            "weight_loader": self.weight_loader,
            "vq2a8_parameter_name": name,
            "vq2a8_loaded": False,
        }

    def weight_loader(
        self, param, loaded_weight, weight_name=None, *, expert_id=None, shard_id=None, return_success=False
    ):
        """Load one complete stacked parameter from the standard checkpoint."""
        name = param.vq2a8_parameter_name
        if expert_id is not None or shard_id is not None:
            raise ValueError("VQ2A8 unified checkpoints require complete stacked parameters, not expert shards.")
        if param.vq2a8_loaded:
            raise ValueError(f"Duplicate VQ2A8 parameter {name}.")
        target = param.data
        if loaded_weight.dtype != target.dtype or loaded_weight.shape != target.shape:
            raise ValueError(f"VQ2A8 {name} shape/dtype mismatch: {loaded_weight.shape}, {loaded_weight.dtype}.")
        self._validate_field(name, loaded_weight)
        target.copy_(loaded_weight)
        param.vq2a8_loaded = True
        return True if return_success else None

    @staticmethod
    def _validate_field(name, value):
        # One-time load checks; never scan compressed weights in the hot path.
        field = name.split("_", 1)[1]
        if field == "pair_lut":
            valid = ((value & 127) != 127).all()
        elif field == "activation_order":
            expected = torch.arange(value.shape[-1], device=value.device)
            valid = (value.sort(dim=-1).values == expected).all()
        elif field == "rht_sign":
            valid = ((value == -1) | (value == 1)).all()
        elif field in ("weight_scale", "weight_bias"):
            valid = torch.isfinite(value).all()
        else:
            return
        if not bool(valid):
            raise ValueError(f"Invalid VQ2A8 {field} payload.")

    def ensure_native_features(self):
        if "ascendc_library" in self.native_options:
            load_vq2a8_library(self.native_options["ascendc_library"], self.native_options["ascendc_sha256"])
        require_vq2a8_features(
            activation_preparation=self.native_options.get("activation_preparation", "packed"),
            validity_mode=self.native_options.get("validity_mode", "torch"),
        )

    def process_weights_after_loading(self, layer):
        # Lazy to keep checkpoint/CPU validation independent of native kernels.
        from vllm_ascend.ops.fused_moe.vq2a8 import VQ2A8ExpertWeights

        for prefix in ("w13", "w2"):
            for field in VQ2A8_FIELDS:
                parameter = getattr(layer, f"{prefix}_{field}")
                if not parameter.vq2a8_loaded:
                    raise ValueError(f"Incomplete VQ2A8 load for {prefix}_{field}.")
        self.ensure_native_features()
        layer.vq2a8_expert_ids = self.layer_spec.expert_ids
        layer.vq2a8_layer_index = self.layer_spec.layer_index
        layer.vq2a8_num_experts = get_moe_num_logical_experts(layer, layer.moe_config.num_experts)
        layer.vq2a8_weights = VQ2A8ExpertWeights.from_layer(
            layer,
            gate_up_spec=self.layer_spec.specs["gate_up"],
            down_spec=self.layer_spec.specs["down"],
            native_options=self.native_options,
            prefill_graph_cache=self.prefill_graph_cache,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Consume the routes selected by AscendRoutedExperts exactly once."""
        return _EXTRA_CTX.moe_comm_method.fused_experts(
            quant_method=self,
            fused_experts_input=build_fused_experts_input(
                layer=layer,
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=layer.w13_packed_zn,
                w2=layer.w2_packed_zn,
                vq2a8=layer.vq2a8_weights,
                quant_type=self.quant_type,
                dynamic_eplb=False,
                expert_map=layer.ascend_expert_map,
                global_redundant_expert_num=layer.global_redundant_expert_num,
                mc2_mask=layer.ascend_mc2_mask,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                pertoken_scale=layer.ascend_pertoken_scale,
                activation=layer.activation,
            ),
        )
