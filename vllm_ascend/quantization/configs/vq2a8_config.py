# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified VQ2A8 checkpoint parser with BF16 non-expert layers.

Like ModelSlim and compressed-tensors, this config describes a checkpoint
format. The VQ2A8 MoE computation is selected through the shared scheme
registry; its implementation lives in methods/w2a8.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import regex as re
import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm_ascend.ops.vq2a8 import VQ2A8_MAX_EXPERTS, VQ2A8ProjectionSpec, projection_shapes
from vllm_ascend.quantization.utils import is_fused_moe_layer

from ..methods import get_scheme_class

MAX_DECODE_GRAPH_CONTEXT = 128


@dataclass(frozen=True)
class VQ2A8LayerSpec:
    layer_index: int
    expert_ids: tuple[int, ...]
    specs: Mapping[str, VQ2A8ProjectionSpec]


def _parse_layers(config) -> dict[int, VQ2A8LayerSpec]:
    if (
        not isinstance(config, dict)
        or set(config) != {"quant_method", "format", "format_version", "layers"}
        or config["quant_method"] != "vq2a8"
        or config["format"] != "ascend_vq2a8"
        or type(config["format_version"]) is not int
        or config["format_version"] != 1
    ):
        raise ValueError("VQ2A8 requires the unified ascend_vq2a8 version 1 quantization_config.")
    entries = config["layers"]
    if not isinstance(entries, dict) or not entries:
        raise ValueError("VQ2A8 layers must be a nonempty object.")
    layers = {}
    for key, entry in entries.items():
        if not isinstance(key, str) or re.fullmatch(r"0|[1-9][0-9]*", key) is None:
            raise ValueError("VQ2A8 layer keys must be canonical nonnegative decimal indices.")
        if not isinstance(entry, dict) or set(entry) != {"expert_ids", "gate_up", "down"}:
            raise ValueError(f"Invalid VQ2A8 layer {key} metadata fields.")
        ids = entry["expert_ids"]
        if (
            not isinstance(ids, list)
            or not 1 <= len(ids) <= VQ2A8_MAX_EXPERTS
            or any(type(expert) is not int or not 0 <= expert < VQ2A8_MAX_EXPERTS for expert in ids)
            or ids != sorted(set(ids))
        ):
            raise ValueError(f"VQ2A8 layer {key} requires sorted unique stored expert IDs in range.")
        specs = {}
        for kind in ("gate_up", "down"):
            fields = entry[kind]
            names = {"rows", "columns", "rht_true_columns", "rht_block_size"}
            if not isinstance(fields, dict) or set(fields) != names:
                raise ValueError(f"Invalid VQ2A8 {key}.{kind} projection fields.")
            spec = VQ2A8ProjectionSpec(**fields)
            projection_shapes(spec)
            specs[kind] = spec
        if (
            specs["gate_up"].rows != 2 * specs["down"].rht_true_columns
            or specs["down"].rows != specs["gate_up"].rht_true_columns
        ):
            raise ValueError("VQ2A8 projection geometries do not form an expert MLP.")
        index = int(key)
        layers[index] = VQ2A8LayerSpec(index, tuple(ids), specs)
    if set(layers) != set(range(len(layers))):
        raise ValueError("VQ2A8 metadata must cover every model layer from zero without gaps.")
    return layers


def validate_vq2a8_config(config) -> dict:
    """Reject unsupported execution before allocating compressed parameters."""
    options = (config.additional_config or {}).get("vq2a8", {})
    allowed = {
        "ascendc_library",
        "ascendc_sha256",
        "decode_mode",
        "activation_preparation",
        "validity_mode",
        "prefill_rht",
        "prefill_batch_mode",
    }
    if not isinstance(options, dict) or set(options) - allowed:
        raise ValueError("Unknown additional_config.vq2a8 options.")
    decode_mode = options.get("decode_mode", "grouped")
    preparation = options.get("activation_preparation", "packed")
    validity_mode = options.get("validity_mode", "torch")
    if options.get("prefill_batch_mode", "jobs") not in ("jobs", "layer"):
        raise ValueError("VQ2A8 prefill_batch_mode must be jobs or layer.")
    if options.get("prefill_rht", "rowwise") not in ("rowwise", "batched", "rowwise_graph"):
        raise ValueError("VQ2A8 prefill_rht must be rowwise, batched or rowwise_graph.")
    if validity_mode not in ("torch", "fused"):
        raise ValueError("VQ2A8 validity_mode must be torch or fused.")
    if validity_mode == "fused" and (decode_mode != "device" or preparation != "sign_fused_direct"):
        raise ValueError("VQ2A8 fused validity requires device decode and sign_fused_direct preparation.")
    if decode_mode not in ("grouped", "device") or preparation not in ("packed", "sign_fused_direct"):
        raise ValueError(
            "VQ2A8 requires decode_mode=grouped/device and activation_preparation=packed/sign_fused_direct."
        )
    if decode_mode != "device" and "activation_preparation" in options:
        raise ValueError("VQ2A8 activation_preparation requires decode_mode=device.")
    use_v2 = bool(getattr(config, "use_v2_model_runner", False))
    speculative = config.speculative_config
    verify_tokens = 1
    if speculative is not None:
        draft = getattr(speculative, "draft_model_config", None)
        if (
            not use_v2
            or speculative.method != "dspark"
            or not speculative.enforce_eager
            or speculative.enable_adaptive_verification
            or draft is None
            or draft.hf_config.model_type != "deepseek_v4"
            or draft.hf_config.architectures != ["DSparkDraftModel"]
            or draft.quantization == "vq2a8"
        ):
            raise ValueError(
                "VQ2A8 speculation requires MRV2 DSpark, an original-format draft, "
                "eager drafting and fixed verification."
            )
        verify_tokens += speculative.num_speculative_tokens

    if ("ascendc_library" in options) != ("ascendc_sha256" in options):
        raise ValueError("VQ2A8 native library and SHA256 must be supplied together.")
    if "ascendc_library" in options:
        library = options["ascendc_library"]
        if not isinstance(library, str) or not Path(library).is_absolute():
            raise ValueError("VQ2A8 requires an absolute native library path.")
        if not isinstance(options["ascendc_sha256"], str) or not re.fullmatch(
            "[0-9a-f]{64}", options["ascendc_sha256"]
        ):
            raise ValueError("VQ2A8 requires the native library SHA256.")
    parallel = config.parallel_config
    for name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        if getattr(parallel, name, 1) != 1:
            raise ValueError(f"VQ2A8 currently requires {name}=1.")
    for name in ("enable_expert_parallel", "enable_eplb", "use_sequence_parallel_moe"):
        if getattr(parallel, name, False):
            raise ValueError(f"VQ2A8 does not yet support {name}.")
    if config.model_config.dtype != torch.bfloat16:
        raise ValueError("VQ2A8 requires BF16 model activations.")
    if config.load_config.load_format not in ("auto", "safetensors") or config.model_config.quantization != "vq2a8":
        raise ValueError("VQ2A8 requires --quantization vq2a8 and the standard auto/safetensors loader.")
    device = getattr(config.load_config, "device", None) or config.device_config.device
    if str(device).split(":", 1)[0] != "npu":
        raise ValueError("VQ2A8 requires parameters resident on an NPU.")
    compilation = config.compilation_config
    if getattr(compilation.mode, "value", compilation.mode) != 0:
        raise ValueError("VQ2A8 requires compilation mode NONE; native banks are not TorchDynamo inputs.")
    graph_mode = getattr(compilation.cudagraph_mode, "name", str(compilation.cudagraph_mode))
    if config.model_config.enforce_eager:
        if graph_mode not in ("NONE", "0"):
            raise ValueError("VQ2A8 eager execution requires graph mode NONE.")
    else:
        if decode_mode != "device" or graph_mode != "FULL_DECODE_ONLY":
            raise ValueError("VQ2A8 graph execution requires device decode and FULL_DECODE_ONLY.")
        if (
            getattr(config.scheduler_config, "max_num_seqs", None) != 1
            or getattr(compilation, "cudagraph_capture_sizes", None) != [verify_tokens]
            or getattr(compilation, "max_cudagraph_capture_size", None) != verify_tokens
        ):
            raise ValueError(
                "VQ2A8 decode graphs require max_num_seqs=1 and the fixed DSpark verification capture size."
            )
        if not use_v2 and not 1 <= getattr(config.model_config, "max_model_len", 0) <= MAX_DECODE_GRAPH_CONTEXT:
            raise ValueError(f"VQ2A8 decode graphs currently require max_model_len in [1,{MAX_DECODE_GRAPH_CONTEXT}].")
        if getattr(compilation, "cudagraph_num_of_warmups", 0) < 1:
            raise ValueError("VQ2A8 decode graphs require at least one warmup on the capture stream.")
        if (config.additional_config or {}).get("ascend_compilation_config", {}).get("enable_npugraph_ex", True):
            raise ValueError("VQ2A8 decode graphs require enable_npugraph_ex=false.")
    if any(getattr(config, name, None) is not None for name in ("lora_config", "kv_transfer_config")):
        raise ValueError("VQ2A8 does not yet support LoRA or KV transfer.")
    if getattr(config.scheduler_config, "async_scheduling", False):
        raise ValueError("VQ2A8 currently requires synchronous scheduling.")
    additional = config.additional_config or {}
    if any(additional.get(name) for name in ("layer_sharding", "mix_placement", "enable_flashcomm1")):
        raise ValueError("VQ2A8 requires unsharded experts without mixed placement or FlashComm.")
    offload = getattr(config, "offload_config", None)
    if (
        getattr(config.cache_config, "cpu_offload_gb", 0)
        or getattr(getattr(offload, "uva", None), "cpu_offload_gb", 0)
        or getattr(getattr(offload, "prefetch", None), "offload_group_size", 0)
        or getattr(config.model_config, "enable_sleep_mode", False)
    ):
        raise ValueError("VQ2A8 native banks require stable resident parameters; offload/sleep is unsupported.")
    result = dict(options)
    result["_max_device_tokens"] = verify_tokens
    return result


@register_quantization_config("vq2a8")
class AscendVQ2A8Config(QuantizationConfig):
    def __init__(self, config=None):
        super().__init__()
        self.layers = _parse_layers(config)
        self._prefill_graph_cache = None

    @classmethod
    def get_name(cls):
        return "vq2a8"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 0

    @classmethod
    def get_config_filenames(cls):
        return []

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def layer_spec(self, layer_index: int) -> VQ2A8LayerSpec:
        if layer_index not in self.layers:
            raise ValueError(f"VQ2A8 checkpoint has no metadata for model layer {layer_index}.")
        return self.layers[layer_index]

    def validate_model_config(self, hf_config) -> None:
        num_layers = hf_config.num_hidden_layers
        num_experts = hf_config.n_routed_experts
        hash_layers = getattr(hf_config, "num_hash_layers", 0)
        if (
            type(num_layers) is not int
            or num_layers != len(self.layers)
            or type(num_experts) is not int
            or not 1 <= num_experts <= VQ2A8_MAX_EXPERTS
            or type(hash_layers) is not int
            or not 0 <= hash_layers <= num_layers
        ):
            raise ValueError("VQ2A8 metadata disagrees with the model layer/expert inventory.")
        expected_shapes = {
            "gate_up": (2 * hf_config.moe_intermediate_size, hf_config.hidden_size),
            "down": (hf_config.hidden_size, hf_config.moe_intermediate_size),
        }
        for index, layer in self.layers.items():
            if any(expert >= num_experts for expert in layer.expert_ids):
                raise ValueError(f"VQ2A8 layer {index} has an expert outside the logical model inventory.")
            if index >= hash_layers and layer.expert_ids != tuple(range(num_experts)):
                raise ValueError("VQ2A8 non-hash layers must store every logical expert.")
            for kind, shape in expected_shapes.items():
                spec = layer.specs[kind]
                if (spec.rows, spec.rht_true_columns) != shape:
                    raise ValueError(f"VQ2A8 layer {index}.{kind} geometry disagrees with model config.")

    def get_quant_method(self, layer, prefix, tid2eid=None):
        # Keep all ordinary linears, including shared experts and the router,
        # on the same BF16 path as an unquantized Ascend model.
        if isinstance(layer, LinearBase):
            from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

            return AscendUnquantizedLinearMethod()
        if not is_fused_moe_layer(layer):
            return None
        match = re.search(r"(?:^|\.)layers\.(\d+)\.mlp\.experts(?:\.routed_experts)?$", prefix)
        if match is None:
            raise ValueError(f"VQ2A8 only supports routed expert layers, got {prefix!r}.")
        config = get_current_vllm_config()
        options = validate_vq2a8_config(config)
        self.validate_model_config(config.model_config.hf_config)
        from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod

        scheme_cls = get_scheme_class("VQ2A8", "moe")
        if scheme_cls is None:
            raise NotImplementedError("No registered VQ2A8 scheme for layer_type=moe.")

        if options.get("prefill_rht") == "rowwise_graph" and self._prefill_graph_cache is None:
            # Construct only in the worker's model-building path. This empty
            # model-owned cache allocates NPU graphs lazily on first prefill.
            from vllm_ascend.quantization.methods.w2a8.vq2a8_activation import (
                VQ2_FP8_MIN_SCALE,
                PrefillRowwiseGraphCache,
            )

            self._prefill_graph_cache = PrefillRowwiseGraphCache(min_scale=VQ2_FP8_MIN_SCALE)
        scheme = scheme_cls(
            self.layer_spec(int(match.group(1))), native_options=options, prefill_graph_cache=self._prefill_graph_cache
        )
        return AscendFusedMoEMethod(scheme, layer.moe_config, tid2eid=tid2eid)
