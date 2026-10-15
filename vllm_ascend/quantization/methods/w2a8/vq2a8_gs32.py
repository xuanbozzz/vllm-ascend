# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resident GS32 tensors with the standard Ascend routing/combine pipeline."""

from dataclasses import replace
from typing import Any

import torch

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input

from ..base import AscendMoEScheme, QuantType, get_moe_num_logical_experts
from ..registry import register_scheme
from .vq2a8_gs32_ep import gs32_expert_placement
from .vq2a8_gs32_format import GS32_DTYPES, GS32_FIELDS, gs32_projection_shapes


@register_scheme("VQ2A8_GS32", "moe")
class AscendGS32FusedMoEScheme(AscendMoEScheme):
    """Keep the common VQ routing adapter, replace only weight ABI and compute."""

    checkpoint_format = "ascend_vq2a8_gs32"
    quant_type = QuantType.VQ2A8
    supports_eplb = False

    def __init__(self, layer_spec, *, native_options=None, moe_config=None, prefill_graph_cache=None):
        self.layer_spec = layer_spec
        self.native_options = dict(native_options or {})
        self.moe_config = moe_config
        self.placement = None

    def get_weight(self, num_experts, intermediate_size_per_partition, hidden_sizes, params_dtype):
        if params_dtype != torch.bfloat16:
            raise ValueError("GS32 requires BF16 activations.")
        self.placement = gs32_expert_placement(self.layer_spec.expert_ids, num_experts, self.moe_config)
        expected = {
            "gate_up": (2 * intermediate_size_per_partition, hidden_sizes),
            "down": (hidden_sizes, intermediate_size_per_partition),
        }
        result = {}
        for prefix, kind in (("w13", "gate_up"), ("w2", "down")):
            spec = self.layer_spec.specs[kind]
            if (spec.rows, spec.columns) != expected[kind]:
                raise ValueError(f"GS32 {kind} geometry does not match the unsharded MoE layer.")
            for field, dtype, shape in zip(GS32_FIELDS, GS32_DTYPES, gs32_projection_shapes(spec), strict=True):
                result[f"{prefix}_{field}"] = torch.empty((len(self.placement.local_ids), *shape), dtype=dtype)
        return result

    @staticmethod
    def _validate_field(name, value):
        field = name.split("_", 1)[1]
        if field == "pair_lut":
            valid = ((value & 127) != 127).all()
        elif field == "perm":
            valid = (value.sort(dim=-1).values == torch.arange(value.shape[-1], device=value.device)).all()
        elif field == "row_scale_bits":
            scale = value.to(torch.int32).contiguous().view(torch.float32)
            valid = ((value >= 0) & (value <= 0xFFFFFFFF)).all() & (torch.isfinite(scale) & (scale > 0)).all()
        elif field == "weight_scale":
            valid = (torch.isfinite(value) & (value > 0)).all()
        elif field == "packed_zn":
            return
        else:
            raise ValueError(f"Unknown GS32 field {field}.")
        if not bool(valid):
            raise ValueError(f"Invalid GS32 {field} payload.")

    def process_weights_after_loading(self, layer):
        from vllm_ascend.ops.fused_moe.vq2a8_gs32 import GS32ExpertWeights

        for prefix in ("w13", "w2"):
            for field in GS32_FIELDS:
                if not getattr(layer, f"{prefix}_{field}").vq2a8_loaded:
                    raise ValueError(f"Incomplete GS32 load for {prefix}_{field}.")
        layer.vq2a8_expert_ids = self.layer_spec.expert_ids
        layer.vq2a8_layer_index = self.layer_spec.layer_index
        placement = self.placement
        if placement is None:
            raise ValueError("GS32 parameters must be allocated before post-load processing.")
        layer.vq2a8_ep_size = placement.ep_size
        layer.vq2a8_ep_rank = placement.ep_rank
        layer.vq2a8_local_global_expert_ids = placement.global_ids
        layer.vq2a8_local_expert_ids = placement.local_ids
        layer.vq2a8_expert_start = placement.expert_start
        layer.vq2a8_num_experts = get_moe_num_logical_experts(layer, layer.moe_config.num_experts)
        local_spec = replace(self.layer_spec, expert_ids=placement.local_ids)
        layer.vq2a8_weights = GS32ExpertWeights(layer, local_spec, self.native_options)

    def get_dynamic_quant_param(self, num_experts, intermediate_size_per_partition, hidden_sizes, params_dtype):
        # GS32 permutation/scales are complete stacked parameters in get_weight;
        # they must not inherit scalar-quantizer CHANNEL/GROUP slicing.
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
        """Validate a global stack and load this rank's complete expert tensors."""
        name = param.vq2a8_parameter_name
        if expert_id is not None or shard_id is not None:
            raise ValueError("VQ2A8 unified checkpoints require complete stacked parameters, not expert shards.")
        if param.vq2a8_loaded:
            raise ValueError(f"Duplicate VQ2A8 parameter {name}.")
        target = param.data
        global_shape = (len(self.layer_spec.expert_ids), *target.shape[1:])
        if loaded_weight.dtype != target.dtype or loaded_weight.shape != global_shape:
            raise ValueError(f"VQ2A8 {name} shape/dtype mismatch: {loaded_weight.shape}, {loaded_weight.dtype}.")
        self._validate_field(name, loaded_weight)
        if self.placement is not None and self.placement.ep_size > 1:
            # Positions refer to stored rows in the checkpoint inventory. Expert
            # IDs need not be dense, especially in the model's hash layers.
            positions = torch.tensor(self.placement.stored_positions, dtype=torch.int64, device=loaded_weight.device)
            target.copy_(loaded_weight.index_select(0, positions))
        else:
            target.copy_(loaded_weight)
        param.vq2a8_loaded = True
        return True if return_success else None

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
