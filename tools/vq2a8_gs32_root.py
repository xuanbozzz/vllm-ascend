# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict GS32 HF checkpoint roots, independent of expert/kernel dispatch.

The exported base contains BF16/F32 roots and I64 hash tables, not routed
expert matrices. Names are mapped explicitly to this checkout's DeepSeek-V4
modules. Reading headers does not allocate model tensors. No global quantizer,
model registration, device initialization, or serving policy is installed here.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import regex as re
import torch
from safetensors import safe_open

_LAYER_NAME = re.compile(r"model\.layers\.(0|[1-9]\d*)\.(.+)")
_ROUTED_NAME = re.compile(r"model\.layers\.(0|[1-9]\d*)\.mlp\.experts\.(0|[1-9]\d*)\.w[123]\.weight")
_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "I64": torch.int64}
_DTYPE_BYTES = {"BF16": 2, "F32": 4, "I64": 8}


def _root_suffixes() -> dict[str, str]:
    return {
        "input_layernorm.weight": "input_layernorm.weight",
        "post_attention_layernorm.weight": "post_attention_layernorm.weight",
        "attn_hc.fn": "hc_attn_fn",
        "attn_hc.base": "hc_attn_base",
        "attn_hc.scale": "hc_attn_scale",
        "ffn_hc.fn": "hc_ffn_fn",
        "ffn_hc.base": "hc_ffn_base",
        "ffn_hc.scale": "hc_ffn_scale",
        "self_attn.q_a_proj.weight": "self_attn.wq_a.weight",
        "self_attn.q_a_norm.weight": "self_attn.q_norm.weight",
        "self_attn.q_b_proj.weight": "self_attn.wq_b.weight",
        "self_attn.kv_proj.weight": "self_attn.wkv.weight",
        "self_attn.kv_norm.weight": "self_attn.kv_norm.weight",
        "self_attn.o_a_proj.weight": "self_attn.wo_a.weight",
        "self_attn.o_b_proj.weight": "self_attn.wo_b.weight",
        "self_attn.sinks": "self_attn.attn_sink",
        "self_attn.compressor.gate_proj.weight": "self_attn.compressor.wgate.weight",
        "self_attn.compressor.kv_proj.weight": "self_attn.compressor.wkv.weight",
        "self_attn.compressor.kv_norm.weight": "self_attn.compressor.norm.weight",
        "self_attn.compressor.position_bias": "self_attn.compressor.ape",
        "self_attn.compressor.indexer.q_b_proj.weight": "self_attn.indexer.wq_b.weight",
        "self_attn.compressor.indexer.scorer.weights_proj.weight": "self_attn.indexer.weights_proj.weight",
        "self_attn.compressor.indexer.gate_proj.weight": "self_attn.indexer.compressor.wgate.weight",
        "self_attn.compressor.indexer.kv_proj.weight": "self_attn.indexer.compressor.wkv.weight",
        "self_attn.compressor.indexer.kv_norm.weight": "self_attn.indexer.compressor.norm.weight",
        "self_attn.compressor.indexer.position_bias": "self_attn.indexer.compressor.ape",
    }


def gs32_moe_root_name(name: str) -> tuple[int, str] | None:
    """Return (layer, existing router/shared dictionary key), without loading."""
    match = _LAYER_NAME.fullmatch(name)
    if match is None:
        return None
    suffixes = {
        "mlp.gate.weight": "gate.weight",
        "mlp.gate.tid2eid": "gate.tid2eid",
        "mlp.gate.e_score_correction_bias": "gate.bias",
        "mlp.shared_experts.gate_proj.weight": "shared_experts.w1.weight",
        "mlp.shared_experts.up_proj.weight": "shared_experts.w3.weight",
        "mlp.shared_experts.down_proj.weight": "shared_experts.w2.weight",
    }
    suffix = suffixes.get(match[2])
    return None if suffix is None else (int(match[1]), suffix)


def gs32_root_parameter(name: str) -> str | None:
    """Map known non-expert HF roots; None means an explicit MoE delegation.

    Unknown names, old canonical names, and routed expert weights are errors.
    This prevents silently skipping incompatible checkpoint families.
    """
    if gs32_moe_root_name(name) is not None:
        return None
    globals_ = {
        "head.weight": "lm_head.weight",
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.norm.weight": "model.norm.weight",
        "model.hc_head.hc_fn": "model.hc_head_fn",
        "model.hc_head.hc_base": "model.hc_head_base",
        "model.hc_head.hc_scale": "model.hc_head_scale",
    }
    if name in globals_:
        return globals_[name]
    match = _LAYER_NAME.fullmatch(name)
    if match is not None and match[2] in _root_suffixes():
        return f"model.layers.{match[1]}.{_root_suffixes()[match[2]]}"
    if _ROUTED_NAME.fullmatch(name):
        raise ValueError(f"GS32 base must exclude routed expert weights: {name}.")
    raise ValueError(f"Unknown GS32 root tensor: {name}.")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}.")
        result[key] = value
    return result


def _read_json(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path}.")
    return result


def gs32_root_shapes(config: Mapping) -> dict[str, tuple[int, ...]]:
    """Expected HF root inventory and exact shapes for supported V4 geometry."""
    positive = (
        "hidden_size",
        "vocab_size",
        "num_hidden_layers",
        "hc_mult",
        "q_lora_rank",
        "num_attention_heads",
        "head_dim",
        "o_groups",
        "o_lora_rank",
        "index_n_heads",
        "index_head_dim",
        "n_routed_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
    )
    for name in positive:
        if type(config.get(name)) is not int or config[name] <= 0:
            raise ValueError(f"GS32 root geometry requires positive integer {name}.")
    for name in ("num_hash_layers", "n_shared_experts"):
        if type(config.get(name)) is not int or config[name] < 0:
            raise ValueError(f"GS32 root geometry requires non-negative integer {name}.")
    h, vocab, layers = (config[name] for name in ("hidden_size", "vocab_size", "num_hidden_layers"))
    mult, qrank, heads, dim = (config[name] for name in ("hc_mult", "q_lora_rank", "num_attention_heads", "head_dim"))
    groups, orank = config["o_groups"], config["o_lora_rank"]
    experts, topk = config["n_routed_experts"], config["num_experts_per_tok"]
    ratios = config.get("compress_ratios")
    if (
        not isinstance(ratios, list)
        or len(ratios) != layers
        or any(type(ratio) is not int or ratio not in (0, 4, 128) for ratio in ratios)
    ):
        raise ValueError("GS32 roots require one compress ratio (0, 4 or 128) per layer.")
    if heads * dim % groups or topk > experts or config["num_hash_layers"] > layers:
        raise ValueError("Invalid GS32 attention/router geometry.")
    shapes = {
        "head.weight": (vocab, h),
        "model.embed_tokens.weight": (vocab, h),
        "model.norm.weight": (h,),
        "model.hc_head.hc_fn": (mult, mult * h),
        "model.hc_head.hc_base": (mult,),
        "model.hc_head.hc_scale": (1,),
    }
    for layer, ratio in enumerate(ratios):
        prefix = f"model.layers.{layer}."
        hc_rows = (2 + mult) * mult
        local = {
            "input_layernorm.weight": (h,),
            "post_attention_layernorm.weight": (h,),
            "self_attn.q_a_proj.weight": (qrank, h),
            "self_attn.q_a_norm.weight": (qrank,),
            "self_attn.q_b_proj.weight": (heads * dim, qrank),
            "self_attn.kv_proj.weight": (dim, h),
            "self_attn.kv_norm.weight": (dim,),
            "self_attn.o_a_proj.weight": (groups * orank, heads * dim // groups),
            "self_attn.o_b_proj.weight": (h, groups * orank),
            "self_attn.sinks": (heads,),
            "mlp.gate.weight": (experts, h),
        }
        for kind in ("attn", "ffn"):
            local.update(
                {f"{kind}_hc.fn": (hc_rows, mult * h), f"{kind}_hc.base": (hc_rows,), f"{kind}_hc.scale": (3,)}
            )
        if layer < config["num_hash_layers"]:
            local["mlp.gate.tid2eid"] = (vocab, topk)
        else:
            local["mlp.gate.e_score_correction_bias"] = (experts,)
        width = config["moe_intermediate_size"] * config["n_shared_experts"]
        if width:
            local.update(
                {
                    "mlp.shared_experts.gate_proj.weight": (width, h),
                    "mlp.shared_experts.up_proj.weight": (width, h),
                    "mlp.shared_experts.down_proj.weight": (h, width),
                }
            )
        if ratio:
            coeff = 2 if ratio == 4 else 1
            for suffix, shape in {
                "kv_proj.weight": (coeff * dim, h),
                "gate_proj.weight": (coeff * dim, h),
                "kv_norm.weight": (dim,),
                "position_bias": (ratio, coeff * dim),
            }.items():
                local[f"self_attn.compressor.{suffix}"] = shape
            if ratio == 4:
                index_dim, index_heads = config["index_head_dim"], config["index_n_heads"]
                for suffix, shape in {
                    "q_b_proj.weight": (index_heads * index_dim, qrank),
                    "scorer.weights_proj.weight": (index_heads, h),
                    "kv_proj.weight": (coeff * index_dim, h),
                    "gate_proj.weight": (coeff * index_dim, h),
                    "kv_norm.weight": (index_dim,),
                    "position_bias": (ratio, coeff * index_dim),
                }.items():
                    local[f"self_attn.compressor.indexer.{suffix}"] = shape
        shapes.update({prefix + suffix: shape for suffix, shape in local.items()})
    return shapes


class GS32RootCheckpoint:
    """Header-audited, pruned base checkpoint; rooted in a single index file.

    ``load_moe`` retains source dtypes. The runtime may explicitly widen router
    tensors for FP32 routing; this loader never quantizes/casts shared weights.
    ``load_parameters`` owns only non-MoE registered parameters. An optional
    streamed iterator must still cover *all* indexed roots exactly once.
    """

    def __init__(self, model_root: Path | str):
        self.root = Path(model_root).resolve(strict=True)
        self.model_config_path = self.root / "config.json"
        self.config = _read_json(self.model_config_path)
        expected = gs32_root_shapes(self.config)
        index = _read_json(self.root / "model.safetensors.index.json")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("GS32 base requires a nonempty safetensors weight_map.")
        for name in weight_map:
            gs32_root_parameter(name)
        if set(weight_map) != set(expected):
            raise ValueError(
                f"Incomplete GS32 root inventory: missing={sorted(set(expected) - set(weight_map))}, "
                f"extra={sorted(set(weight_map) - set(expected))}."
            )
        paths = {}
        for filename in weight_map.values():
            if (
                not isinstance(filename, str)
                or not filename
                or "/" in filename
                or "\\" in filename
                or ":" in filename
                or filename in (".", "..")
                or not filename.endswith(".safetensors")
            ):
                raise ValueError(f"Unsafe GS32 shard filename: {filename!r}.")
            if filename in paths:
                continue
            path = (self.root / filename).resolve(strict=True)
            if path.parent != self.root or not path.is_file():
                raise ValueError(f"GS32 shard escapes model root: {filename}.")
            paths[filename] = path
        self.inventory = {}
        self._paths = paths
        self._weight_map = dict(weight_map)
        self._file_identity = {filename: self._identity(path) for filename, path in paths.items()}
        targets = set()
        payload_bytes = total_parameters = 0
        for filename, path in sorted(paths.items()):
            with safe_open(path, framework="pt", device="cpu") as handle:
                indexed = {name for name, shard in weight_map.items() if shard == filename}
                if set(handle.keys()) != indexed:
                    raise ValueError(f"GS32 shard header/index disagree: {filename}.")
                for name in sorted(indexed):
                    view = handle.get_slice(name)
                    dtype, shape = view.get_dtype(), tuple(view.get_shape())
                    allowed = ("I64",) if name.endswith(".gate.tid2eid") else ("BF16", "F32")
                    if dtype not in allowed or shape != expected[name]:
                        raise ValueError(
                            f"GS32 root shape/dtype mismatch: {name}: {dtype} {shape}; "
                            f"expected {allowed} {expected[name]}."
                        )
                    target = gs32_root_parameter(name)
                    if target is not None:
                        if target in targets:
                            raise ValueError(f"Repeated GS32 root target: {target}.")
                        targets.add(target)
                    numel = 1
                    for dimension in shape:
                        numel *= dimension
                    total_parameters += numel
                    payload_bytes += numel * _DTYPE_BYTES[dtype]
                    self.inventory[name] = {"dtype": dtype, "shape": shape, "target": target, "shard": filename}
            self._check_file(filename)
        metadata = index.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("GS32 index metadata must be an object.")
        for field, actual in (("total_size", payload_bytes), ("total_parameters", total_parameters)):
            if field in metadata and (type(metadata[field]) is not int or metadata[field] != actual):
                raise ValueError(f"GS32 index metadata {field} does not match root tensors.")
        self.payload_bytes = payload_bytes
        self.total_parameters = total_parameters

    @staticmethod
    def _identity(path):
        info = path.stat()
        return info.st_size, info.st_mtime_ns, info.st_dev, info.st_ino

    def _check_file(self, filename):
        if self._identity(self._paths[filename]) != self._file_identity[filename]:
            raise ValueError(f"GS32 checkpoint changed after header audit: {filename}.")

    def _validate_tensor(self, name: str, value: torch.Tensor) -> None:
        spec = self.inventory[name]
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != spec["shape"]
            or value.dtype != _DTYPES[spec["dtype"]]
        ):
            raise ValueError(f"GS32 streamed root shape/dtype changed: {name}.")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite GS32 root tensor: {name}.")
        if name.endswith(".gate.tid2eid") and bool(((value < 0) | (value >= self.config["n_routed_experts"])).any()):
            raise ValueError(f"GS32 hash table expert ID is out of range: {name}.")

    def iter_tensors(self, names: Iterable[str] | None = None):
        requested = set(self.inventory) if names is None else set(names)
        if not requested <= set(self.inventory):
            raise ValueError("Requested unknown GS32 root tensor.")
        for filename, path in sorted(self._paths.items()):
            selected = sorted(name for name in requested if self._weight_map[name] == filename)
            if not selected:
                continue
            self._check_file(filename)
            with safe_open(path, framework="pt", device="cpu") as handle:
                for name in selected:
                    value = handle.get_tensor(name)
                    self._validate_tensor(name, value)
                    yield name, value
            self._check_file(filename)

    def delegated_names(self) -> set[str]:
        return {name for name, spec in self.inventory.items() if spec["target"] is None}

    def load_moe(self, layer_index: int, *, device="cpu") -> dict[str, torch.Tensor]:
        if type(layer_index) is not int or not 0 <= layer_index < self.config["num_hidden_layers"]:
            raise ValueError("GS32 MoE layer index is out of range.")
        names = {name for name in self.delegated_names() if gs32_moe_root_name(name)[0] == layer_index}
        return {
            gs32_moe_root_name(name)[1]: value.to(device=device).contiguous()
            for name, value in self.iter_tensors(names)
        }

    def load_parameters(
        self,
        parameters: Mapping[str, torch.Tensor],
        loader: Callable,
        weights: Iterable[tuple[str, torch.Tensor]] | None = None,
        *,
        allow_compressor_norm_widening: bool = False,
    ) -> tuple[set[str], dict]:
        """Load all registered roots once; reject omissions and implicit casts.

        A5 creates compressor RMSNorm weights in FP32. Its exact BF16->FP32
        widening is opt-in, applies only to those norms, and is reported.
        """
        expected_targets = {spec["target"] for spec in self.inventory.values() if spec["target"] is not None}
        if set(parameters) != expected_targets:
            raise ValueError(
                f"GS32 registered parameter inventory differs: "
                f"missing={sorted(expected_targets - set(parameters))}, "
                f"extra={sorted(set(parameters) - expected_targets)}."
            )
        if type(allow_compressor_norm_widening) is not bool:
            raise ValueError("allow_compressor_norm_widening must be bool.")
        seen, loaded, delegated, widened = set(), set(), set(), []
        for name, value in self.iter_tensors() if weights is None else weights:
            if name in seen or name not in self.inventory:
                raise ValueError(f"Unknown or repeated GS32 streamed root: {name}.")
            seen.add(name)
            self._validate_tensor(name, value)
            target = self.inventory[name]["target"]
            if target is None:
                delegated.add(name)
                continue
            param = parameters[target]
            widen = (
                allow_compressor_norm_widening
                and value.dtype == torch.bfloat16
                and param.dtype == torch.float32
                and re.fullmatch(r"model\.layers\.\d+\.self_attn\.(?:indexer\.)?compressor\.norm\.weight", target)
            )
            if tuple(param.shape) != tuple(value.shape) or (param.dtype != value.dtype and not widen):
                raise ValueError(f"GS32 parameter shape/dtype mismatch: {name} -> {target}.")
            if widen:
                value = value.float()
                widened.append(name)
            loader(param, value)
            loaded.add(target)
        if seen != set(self.inventory):
            raise ValueError(f"Incomplete GS32 streamed roots: {sorted(set(self.inventory) - seen)}.")
        return loaded, {
            "registered_parameters_loaded": len(loaded),
            "moe_root_tensors_delegated": len(delegated),
            "source_tensors_loaded": len(seen),
            "payload_bytes": self.payload_bytes,
            "compressor_norm_bf16_to_fp32": widened,
            "expert_weights_loaded": 0,
        }
