# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict, bounded reader for canonical GS32x32 BDNorm routed experts.

This is a separate format from the RHT/direct-TP1 artifacts. Opening a run
reads JSON and safetensors headers only; loading reads one compressed matrix
at a time. No dense reconstruction, device residency or kernel is implied.
The native FFN specialization requires zero weight bias and positive scales.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re
import torch
from safetensors import safe_open

from .vq2a8_checkpoint_io import read_safetensors_header

GS32_GROUP_SIZE = 32
GS32_CODEBOOK_SIZE = 16
GS32_VECTOR_LENGTH = 2
GS32_INDEX_BITS = 4
GS32_INDICES_PER_WORD = 8
GS32_MATRIX_KINDS = ("gate_up", "down")
GS32_FIELDS = ("packed_indices", "codebooks", "perm", "weight_scale", "weight_bias", "row_scale")
GS32_FORMAT = "vq2a8_gs32_bdnorm_canonical_v1"
GS32_STANDALONE_CODEC_SHA256 = "36e40cb8abdb1e97f0a40f762b04a1a93dff4a36e03fa27259d2c19d7c25de3e"
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MATRIX_NAME = re.compile(r"^(0|[1-9]\d*)\.mlp\.experts\.(0|[1-9]\d*)\.(gate_up|down)$")
_LAYER_FILE = re.compile(r"^experts_vq_layer_(0|[1-9]\d*)\.(json|safetensors)$")
_METADATA_FIELDS = frozenset(
    {
        "rows",
        "cols",
        "n_row_tiles",
        "n_col_tiles",
        "row_group_size",
        "group_size",
        "K",
        "index_bits",
        "vector_len",
        "n_vectors",
        "n_elements",
        "orig_shape",
        "norm_dim",
        "enable_perm",
        "enable_norm",
        "enable_rht",
        "residual",
        "enable_both_dim_norm",
    }
)


def _integer(value: object, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return value


def _exact(value: object, expected: object, name: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{name} must be {expected!r}, got {value!r}.")


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}.")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError(f"JSON exceeds {_MAX_JSON_BYTES} bytes: {path}.")
    try:
        result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON: {path}.") from error
    if not isinstance(result, dict):
        raise ValueError(f"JSON must contain an object: {path}.")
    return result


def _is_link(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _safe_child(root: Path, *parts: str, directory: bool = False) -> Path:
    path = root
    for part in parts:
        path = path / part
        if _is_link(path):
            raise ValueError(f"Artifact paths must not contain symlinks or junctions: {path}.")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"Artifact path escapes run root: {path}.")
    if not (resolved.is_dir() if directory else resolved.is_file()):
        raise ValueError(f"Artifact path is not a regular {'directory' if directory else 'file'}: {path}.")
    return resolved


def _signature(path: Path) -> tuple[int, int, int]:
    info = path.stat()
    return info.st_size, info.st_mtime_ns, info.st_ino


@dataclass(frozen=True)
class VQ2GS32MatrixSpec:
    """One GS32 matrix; input scales/perm are shared by both gate/up halves."""

    name: str
    layer_index: int
    expert_id: int
    kind: str
    rows: int
    columns: int

    @classmethod
    def from_dict(cls, name: str, metadata: dict[str, Any]) -> VQ2GS32MatrixSpec:
        match = _MATRIX_NAME.fullmatch(name)
        if match is None:
            raise ValueError(f"Invalid GS32 matrix name: {name!r}.")
        if not isinstance(metadata, dict) or set(metadata) != _METADATA_FIELDS:
            raise ValueError(f"{name}: metadata fields must exactly match the GS32 BDNorm schema.")
        rows = _integer(metadata["rows"], f"{name}.rows")
        columns = _integer(metadata["cols"], f"{name}.cols")
        if rows % GS32_GROUP_SIZE or columns % GS32_GROUP_SIZE:
            raise ValueError(f"{name}: rows and cols must be multiples of {GS32_GROUP_SIZE}.")
        expected = {
            "n_row_tiles": rows // GS32_GROUP_SIZE,
            "n_col_tiles": columns // GS32_GROUP_SIZE,
            "row_group_size": GS32_GROUP_SIZE,
            "group_size": GS32_GROUP_SIZE,
            "K": GS32_CODEBOOK_SIZE,
            "index_bits": GS32_INDEX_BITS,
            "vector_len": GS32_VECTOR_LENGTH,
            "n_vectors": rows * columns // GS32_VECTOR_LENGTH,
            "n_elements": rows * columns,
            "norm_dim": 0,
            "enable_perm": True,
            "enable_norm": True,
            "enable_both_dim_norm": True,
            "enable_rht": False,
            "residual": False,
        }
        for field, value in expected.items():
            _exact(metadata[field], value, f"{name}.{field}")
        shape = metadata["orig_shape"]
        if not isinstance(shape, list) or len(shape) != 2 or any(type(value) is not int for value in shape):
            raise ValueError(f"{name}.orig_shape must be a two-integer list.")
        _exact(shape, [rows, columns], f"{name}.orig_shape")
        return cls(name, int(match[1]), int(match[2]), match[3], rows, columns)

    @property
    def row_tiles(self) -> int:
        return self.rows // GS32_GROUP_SIZE

    @property
    def column_tiles(self) -> int:
        return self.columns // GS32_GROUP_SIZE

    @property
    def original_shape(self) -> tuple[int, int]:
        return self.rows, self.columns

    def expected_tensor_headers(self) -> dict[str, tuple[str, tuple[int, ...]]]:
        return {
            "packed_indices": ("I32", (self.rows * self.columns // GS32_VECTOR_LENGTH // GS32_INDICES_PER_WORD,)),
            "codebooks": ("F8_E4M3", (self.column_tiles, self.row_tiles, GS32_CODEBOOK_SIZE, GS32_VECTOR_LENGTH)),
            "perm": ("I32", (self.columns,)),
            "weight_scale": ("F32", (self.columns,)),
            "weight_bias": ("F32", (self.columns,)),
            "row_scale": ("F32", (self.rows,)),
        }

    def as_metadata(self) -> dict[str, Any]:
        """Return canonical producer metadata, e.g. for a pinned CPU oracle."""
        return {
            "rows": self.rows,
            "cols": self.columns,
            "n_row_tiles": self.row_tiles,
            "n_col_tiles": self.column_tiles,
            "row_group_size": GS32_GROUP_SIZE,
            "group_size": GS32_GROUP_SIZE,
            "K": GS32_CODEBOOK_SIZE,
            "index_bits": GS32_INDEX_BITS,
            "vector_len": GS32_VECTOR_LENGTH,
            "n_vectors": self.rows * self.columns // GS32_VECTOR_LENGTH,
            "n_elements": self.rows * self.columns,
            "orig_shape": list(self.original_shape),
            "norm_dim": 0,
            "enable_perm": True,
            "enable_norm": True,
            "enable_both_dim_norm": True,
            "enable_rht": False,
            "residual": False,
        }


@dataclass(frozen=True)
class VQ2GS32Layer:
    layer_index: int
    expert_ids: tuple[int, ...]
    metadata_path: Path
    tensor_path: Path
    specs: dict[str, VQ2GS32MatrixSpec]
    tensor_signature: tuple[int, int, int]

    def spec_for(self, expert_id: int, kind: str) -> VQ2GS32MatrixSpec:
        _integer(expert_id, "expert_id", minimum=0)
        if kind not in GS32_MATRIX_KINDS:
            raise ValueError(f"Unknown GS32 matrix kind: {kind!r}.")
        return self.specs[f"{self.layer_index}.mlp.experts.{expert_id}.{kind}"]


@dataclass(frozen=True)
class VQ2GS32Artifact:
    root: Path
    model_root: Path
    model_config_path: Path
    experts_path: Path
    model_config: dict[str, Any]
    layers: dict[int, VQ2GS32Layer]
    complete: bool
    headers_validated: bool

    def layer(self, layer_index: int) -> VQ2GS32Layer:
        _integer(layer_index, "layer_index", minimum=0)
        return self.layers[layer_index]

    def load_matrix(
        self,
        layer_index: int,
        expert_id: int,
        kind: str,
        *,
        device: torch.device | str = "cpu",
        validate_payload: bool = True,
    ) -> tuple[dict[str, torch.Tensor], VQ2GS32MatrixSpec]:
        """Read six compressed tensors for one matrix; validate on CPU before H2D.

        ``validate_payload=False`` is for trusted diagnostic callers only;
        dtype/shape checks remain mandatory. No payload cache is retained.
        """
        layer = self.layer(layer_index)
        spec = layer.spec_for(expert_id, kind)
        path = _safe_child(self.root, "experts_vq", layer.tensor_path.name)
        if _signature(path) != layer.tensor_signature:
            raise ValueError(f"GS32 tensor file changed since artifact open: {path}.")
        with safe_open(path, framework="pt", device="cpu") as handle:
            tensors = {field: handle.get_tensor(f"{spec.name}.{field}").contiguous() for field in GS32_FIELDS}
        if _signature(path) != layer.tensor_signature:
            raise ValueError(f"GS32 tensor file changed while loading: {path}.")
        _validate_tensor_shapes(tensors, spec)
        if validate_payload:
            validate_gs32_matrix_payload(tensors, spec)
        target = torch.device(device)
        if target.type != "cpu":
            tensors = {field: value.to(device=target, non_blocking=False) for field, value in tensors.items()}
        return tensors, spec


def _validate_tensor_shapes(tensors: dict[str, torch.Tensor], spec: VQ2GS32MatrixSpec) -> None:
    if set(tensors) != set(GS32_FIELDS):
        raise ValueError(f"{spec.name}: payload fields must be exactly {GS32_FIELDS}.")
    dtypes = {"I32": torch.int32, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn}
    for field, (dtype, shape) in spec.expected_tensor_headers().items():
        tensor = tensors[field]
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtypes[dtype] or tuple(tensor.shape) != shape:
            raise ValueError(f"{spec.name}.{field}: expected dtype={dtype}, shape={shape}.")
        if tensor.device.type != "cpu":
            raise ValueError(f"{spec.name}.{field}: payload validation requires CPU tensors.")


def validate_gs32_matrix_payload(tensors: dict[str, torch.Tensor], spec: VQ2GS32MatrixSpec) -> None:
    """Validate the zero-bias GS32 kernel contract without reconstructing weights."""
    _validate_tensor_shapes(tensors, spec)
    if not bool(torch.isfinite(tensors["codebooks"].float()).all()):
        raise ValueError(f"{spec.name}: codebooks contain non-finite values.")
    permutation = tensors["perm"].to(torch.int64)
    expected = torch.arange(spec.columns, dtype=torch.int64, device="cpu")
    if not torch.equal(torch.sort(permutation).values, expected):
        raise ValueError(f"{spec.name}: perm must be a bijection over [0, cols).")
    for field in ("weight_scale", "row_scale"):
        value = tensors[field]
        if not bool((torch.isfinite(value) & (value > 0)).all()):
            raise ValueError(f"{spec.name}: {field} must be finite and strictly positive.")
    # Reject negative zero too: this specialization omits the producer's
    # addition and must not silently claim an untested signed-zero contract.
    if bool((tensors["weight_bias"].contiguous().view(torch.int32) != 0).any()):
        raise ValueError(f"{spec.name}: weight_bias must contain only positive zero for this specialization.")


def validate_gs32_payload(tensors: dict[str, torch.Tensor], spec: VQ2GS32MatrixSpec) -> None:
    """Public short name used by the GS32 device-layout adapter."""
    validate_gs32_matrix_payload(tensors, spec)


def open_vq2a8_gs32_artifact(
    run_root: str | Path,
    *,
    require_complete: bool = True,
    validate_headers: bool = True,
) -> VQ2GS32Artifact:
    """Open ``run_root/{packed_model,experts_vq}`` without loading weight payloads.

    Complete means every config layer and every configured routed expert,
    including hash layers. A partial diagnostic run may omit whole layers,
    never experts or fields within a present layer. Config and JSON are not
    rewritten; legacy ``expert_dtype='fp4'`` is not treated as the VQ format.
    """
    candidate = Path(run_root).expanduser()
    if _is_link(candidate):
        raise ValueError(f"Run root must not be a symlink or junction: {candidate}.")
    root = candidate.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Run root must be a directory: {root}.")
    model_root = _safe_child(root, "packed_model", directory=True)
    experts_path = _safe_child(root, "experts_vq", directory=True)
    config_path = _safe_child(root, "packed_model", "config.json")
    config = _read_json(config_path)
    _exact(config.get("model_type"), "deepseek_v4", "config.model_type")
    layer_count = _integer(config.get("num_hidden_layers"), "config.num_hidden_layers")
    expert_count = _integer(config.get("n_routed_experts"), "config.n_routed_experts")
    hidden = _integer(config.get("hidden_size"), "config.hidden_size")
    intermediate = _integer(config.get("moe_intermediate_size"), "config.moe_intermediate_size")
    hash_layers = _integer(config.get("num_hash_layers"), "config.num_hash_layers", minimum=0)
    if hash_layers > layer_count:
        raise ValueError("config.num_hash_layers exceeds num_hidden_layers.")
    if hidden % GS32_GROUP_SIZE or intermediate % GS32_GROUP_SIZE:
        raise ValueError("Config hidden/intermediate sizes must be multiples of 32.")
    inventories: dict[str, set[int]] = {"json": set(), "safetensors": set()}
    for path in experts_path.iterdir():
        match = _LAYER_FILE.fullmatch(path.name)
        if match is None:
            if path.name.startswith("experts_vq_layer_"):
                raise ValueError(f"Invalid GS32 layer filename: {path.name}.")
            continue
        _safe_child(root, "experts_vq", path.name)
        inventories[match[2]].add(int(match[1]))
    selected = inventories["json"]
    if not selected or selected != inventories["safetensors"]:
        raise ValueError("GS32 requires matching, nonempty JSON and safetensors layer inventories.")
    expected_layers = set(range(layer_count))
    if not selected <= expected_layers or (require_complete and selected != expected_layers):
        raise ValueError("GS32 layer inventory does not match config.num_hidden_layers.")
    layers: dict[int, VQ2GS32Layer] = {}
    expert_ids = tuple(range(expert_count))
    for index in sorted(selected):
        metadata_path = _safe_child(root, "experts_vq", f"experts_vq_layer_{index}.json")
        tensor_path = _safe_child(root, "experts_vq", f"experts_vq_layer_{index}.safetensors")
        metadata = _read_json(metadata_path)
        expected_names = {f"{index}.mlp.experts.{expert}.{kind}" for expert in expert_ids for kind in GS32_MATRIX_KINDS}
        if set(metadata) != expected_names:
            raise ValueError(
                f"Layer {index}: expected every configured expert and both matrices, including hash layers."
            )
        specs = {name: VQ2GS32MatrixSpec.from_dict(name, value) for name, value in metadata.items()}
        for spec in specs.values():
            expected_shape = (2 * intermediate, hidden) if spec.kind == "gate_up" else (hidden, intermediate)
            if spec.original_shape != expected_shape:
                raise ValueError(
                    f"{spec.name}: shape {spec.original_shape} differs from config shape {expected_shape}."
                )
        signature = _signature(tensor_path)
        if validate_headers:
            headers = read_safetensors_header(tensor_path)
            expected_headers = {
                f"{spec.name}.{field}": description
                for spec in specs.values()
                for field, description in spec.expected_tensor_headers().items()
            }
            if set(headers) != set(expected_headers):
                raise ValueError(f"Layer {index}: tensor fields do not exactly match GS32 metadata.")
            for name, (dtype, shape) in expected_headers.items():
                if (headers[name].dtype, headers[name].shape) != (dtype, shape):
                    raise ValueError(f"{name}: tensor header must have dtype={dtype}, shape={shape}.")
        if _signature(tensor_path) != signature:
            raise ValueError(f"GS32 tensor file changed while reading headers: {tensor_path}.")
        layers[index] = VQ2GS32Layer(index, expert_ids, metadata_path, tensor_path, specs, signature)
    return VQ2GS32Artifact(
        root, model_root, config_path, experts_path, config, layers, selected == expected_layers, validate_headers
    )
