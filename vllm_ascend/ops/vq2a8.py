# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load the VQ2A8 native operators without enabling unrelated A5 custom ops.

The normal A5 extension build includes these operators. The original native
namespace is retained so explicitly selected standalone libraries and existing
resident banks keep their ABI. Only one provider may be loaded in a process.
"""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path

import regex as re
import torch

VQ2A8_ABI_VERSION = 1
VQ2A8_FEATURE_VERSION = 1
VQ2A8_STANDALONE_LIBRARY = "libvq2a8_ascendc_v4_v2.so"
VQ2A8_MAX_EXPERTS = 256
VQ2A8_SUPPORTED_N = 4096
VQ2A8_SUPPORTED_K = (2048, 4096)
VQ2A8_FIELDS = ("packed_zn", "pair_lut", "activation_order", "weight_scale", "weight_bias", "rht_sign")
VQ2A8_DTYPES = (torch.uint8, torch.uint8, torch.int64, torch.float32, torch.float32, torch.int8)
VQ2A8_PACKED_N_BLOCK = 32
VQ2A8_PACKED_K_BLOCK = 16
VQ2A8_CODEBOOK_K_BLOCK = 256
VQ2A8_PACKED_VALUES_PER_BYTE = 4


@dataclass(frozen=True)
class VQ2A8ProjectionSpec:
    """Only the projection geometry needed to prepare and execute a bank."""

    rows: int
    columns: int
    rht_true_columns: int
    rht_block_size: int

    def __post_init__(self) -> None:
        values = (self.rows, self.columns, self.rht_true_columns, self.rht_block_size)
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("VQ2A8 projection dimensions must be positive integers.")
        if self.rht_true_columns > self.columns:
            raise ValueError("VQ2A8 true columns cannot exceed padded columns.")
        if self.rht_block_size & (self.rht_block_size - 1) or self.columns % self.rht_block_size:
            raise ValueError("VQ2A8 RHT block size must be a power of two dividing padded columns.")


def projection_shapes(spec: VQ2A8ProjectionSpec) -> tuple[tuple[int, ...], ...]:
    """Shapes of the six compressed fields for one native expert projection."""
    n, k = spec.rows, spec.columns
    if n != VQ2A8_SUPPORTED_N or k not in VQ2A8_SUPPORTED_K:
        raise ValueError("VQ2A8 native banks require N4096 and K2048/4096.")
    return (
        (
            n // VQ2A8_PACKED_N_BLOCK,
            k // VQ2A8_PACKED_K_BLOCK,
            VQ2A8_PACKED_K_BLOCK,
            VQ2A8_PACKED_N_BLOCK // VQ2A8_PACKED_VALUES_PER_BYTE,
        ),
        (k // VQ2A8_CODEBOOK_K_BLOCK, n // VQ2A8_PACKED_N_BLOCK, VQ2A8_PACKED_N_BLOCK),
        (k,),
        (k,),
        (k,),
        (k,),
    )


def _load_builtin_library() -> None:
    # Keep device discovery and RTS initialization lazy, as in enable_custom_op.
    # This path deliberately does not change its global custom-op switch.
    from vllm_ascend.utils import AscendDeviceType, bootstrap_custom_op_env, get_ascend_device_type

    if get_ascend_device_type() != AscendDeviceType.A5:
        raise RuntimeError("VQ2A8 native operators require Ascend950.")
    if not torch.compiler.is_compiling():
        bootstrap_custom_op_env()
    importlib.import_module("vllm_ascend.vllm_ascend_C")


def get_vq2a8_bank_type():
    """Return the native bank class, loading the A5 extension when necessary."""
    native = torch.ops.vq2a8_ascendc_v4_v2
    if not hasattr(native, "abi_version"):
        try:
            _load_builtin_library()
        except (ImportError, OSError, RuntimeError) as error:
            raise RuntimeError(
                "VQ2A8 native operators are unavailable. Build the Ascend950 vllm_ascend_C extension "
                "or load an explicitly selected libvq2a8_ascendc_v4_v2.so first."
            ) from error
    return _registered_bank_type()


def _registered_bank_type():
    """Check the selected provider without importing a second native library."""
    try:
        abi = torch.ops.vq2a8_ascendc_v4_v2.abi_version()
        bank = torch.classes.vq2a8_ascendc_v4_v2.ResidentBank
    except (RuntimeError, AttributeError) as error:
        raise RuntimeError("VQ2A8 resident operators are missing; no alternate-kernel fallback is selected.") from error
    if type(abi) is not int or abi != VQ2A8_ABI_VERSION:
        raise RuntimeError("V4 v2 resident library ABI mismatch.")
    return bank


def require_vq2a8_features(
    reorder="vectorized",
    preparation="sign_fused",
    *,
    native_ops=None,
    activation_preparation="packed",
    validity_mode="torch",
) -> None:
    """Validate the native operations used by the current resident path."""
    if (reorder, preparation) != ("vectorized", "sign_fused"):
        raise ValueError("VQ2A8 core requires vectorized/sign_fused.")
    if activation_preparation not in ("packed", "sign_fused_direct"):
        raise ValueError("VQ2A8 activation preparation must be packed or sign_fused_direct.")
    if validity_mode not in ("torch", "fused"):
        raise ValueError("VQ2A8 validity mode must be torch or fused.")
    if validity_mode == "fused" and activation_preparation != "sign_fused_direct":
        raise ValueError("VQ2A8 fused validity requires sign_fused_direct preparation.")
    if native_ops is None:
        get_vq2a8_bank_type()
        native_ops = torch.ops.vq2a8_ascendc_v4_v2
    features = ("activation_reorder_version", "activation_preparation_version")
    if activation_preparation == "sign_fused_direct":
        features += ("activation_sign_strided_version",)
    if validity_mode == "fused":
        features += ("projection_validity_version",)
    for feature in features:
        try:
            version = getattr(native_ops, feature)()
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError(f"Missing VQ2A8 native feature: {feature}") from error
        if type(version) is not int or version != VQ2A8_FEATURE_VERSION:
            raise RuntimeError(f"Unsupported VQ2A8 native feature: {feature}")
    if activation_preparation == "sign_fused_direct":
        try:
            sign = native_ops.activation_sign_strided
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError("Missing VQ2A8 native feature: activation_sign_strided") from error
        if not callable(sign):
            raise RuntimeError("Missing VQ2A8 native feature: activation_sign_strided")
    if validity_mode == "fused":
        try:
            checker = native_ops.projection_validity
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError("Missing VQ2A8 native feature: projection_validity; rebuild the extension.") from error
        if not callable(checker):
            raise RuntimeError("Missing VQ2A8 native feature: projection_validity; rebuild the extension.")


def load_vq2a8_library(path, expected_sha256):
    """Load an explicitly selected standalone binary, preserving its identity."""
    if not isinstance(expected_sha256, str) or re.fullmatch("[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("V4 v2 requires an explicit SHA256 library identity.")
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.name != VQ2A8_STANDALONE_LIBRARY:
        raise ValueError("V4 v2 requires libvq2a8_ascendc_v4_v2.so, not the V1/v2/v3 library.")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected_sha256:
        raise ValueError("V4 v2 library SHA256 does not match the selected binary.")
    if str(path) not in torch.ops.loaded_libraries:
        if hasattr(torch.ops.vq2a8_ascendc_v4_v2, "abi_version"):
            raise RuntimeError("Another V4 v2 library is registered; use a fresh process.")
        torch.ops.load_library(str(path))
    _registered_bank_type()
    return {"path": str(path), "sha256": actual, "abi_version": VQ2A8_ABI_VERSION}
