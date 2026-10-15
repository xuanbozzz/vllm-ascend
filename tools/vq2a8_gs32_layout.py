# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless GS32 pair-index layout and a CPU-only producer-weight oracle.

This is not a quantizer or a native dispatch implementation. In particular,
moving BDNorm scales across GEMM does not reproduce rounded BF16 weights.
The fused consumer must implement the retained activation transforms itself.
"""

from __future__ import annotations

import numpy as np
import torch

from .vq2a8_gs32_artifact import validate_gs32_payload

GS32_LAYOUT_FORMAT = "vq2-gs32-pair-zn-v1"
GS32_SOURCE_CODEC_SHA256 = "36e40cb8abdb1e97f0a40f762b04a1a93dff4a36e03fa27259d2c19d7c25de3e"
GS32_TILE = 32
GS32_VECTOR_LENGTH = 2
GS32_PAIRS = GS32_TILE // GS32_VECTOR_LENGTH
GS32_ZN_K = 16
GS32_INDICES_PER_WORD = 8


def _indices(payload, spec):
    """Producer order: K tile, N tile, column, adjacent output-row pair."""
    words = payload["packed_indices"].numpy().view(np.uint32)
    shifts = np.arange(GS32_INDICES_PER_WORD, dtype=np.uint32) * np.uint32(4)
    indices = ((words[:, None] >> shifts) & np.uint32(15)).astype(np.uint8)
    return indices.reshape(spec.columns // GS32_TILE, spec.rows // GS32_TILE, GS32_TILE, GS32_PAIRS)


def _row_tiles(spec):
    order = np.arange(spec.rows // GS32_TILE, dtype=np.int64)
    if spec.kind == "gate_up":
        if spec.rows % 128:
            raise ValueError("Fused Gate/Up requires total output width divisible by 128.")
        # The producer stores all gate rows, then all up rows. The device's
        # output tiles hold gate32, up32, gate32, up32, ... . Pair order within
        # each group remains untouched; arbitrary VQ pairs are never split.
        order = order.reshape(2, -1).T.reshape(-1)
    elif spec.kind != "down":
        raise ValueError("GS32 kind must be gate_up or down.")
    return order


def repack_gs32_matrix(payload, spec, *, down_stage=False):
    """Convert one CPU matrix into compressed mixed-FFN consumer fields.

    The uint64 row coefficients deliberately stay in logical output order:
    Gate W2 AIC gathers gate/up scales separately using logical_hidden_offset.
    Down permutation and column scales cannot be folded into these weights.
    ``down_stage=True`` keeps natural row order even for GateUp, for reuse of
    the existing DownW2-only entry as an ordinary projection. Never pass this
    layout to the fused GateW2 stage.
    """
    validate_gs32_payload(payload, spec)
    if spec.columns % 64:
        raise ValueError("The mixed FFN consumer requires K divisible by 64.")
    order = np.arange(spec.rows // GS32_TILE) if down_stage else _row_tiles(spec)
    indices = _indices(payload, spec).transpose(1, 0, 2, 3)[order]
    indices = np.ascontiguousarray(indices).reshape(
        spec.rows // GS32_TILE, spec.columns // GS32_ZN_K, GS32_ZN_K, GS32_PAIRS
    )
    packed = np.ascontiguousarray(indices[..., 0::2] | (indices[..., 1::2] << np.uint8(4)))
    books = payload["codebooks"].view(torch.uint8).numpy().transpose(1, 0, 2, 3)[order]
    pair_lut = np.ascontiguousarray(books).reshape(spec.rows // GS32_TILE, spec.columns // GS32_TILE, 32)
    row_bits = payload["row_scale"].numpy().view(np.uint32).astype(np.uint64)
    return {
        "packed_zn": torch.from_numpy(packed),
        "pair_lut": torch.from_numpy(pair_lut),
        "row_scale_bits": torch.from_numpy(row_bits),
        "perm": payload["perm"].clone(),
        "weight_scale": payload["weight_scale"].clone(),
    }


def decode_gs32_codebook_reference(payload, spec):
    """CPU FP32 lookup before inverse permutation or either BDNorm scale.

    Mirrors the verified standalone codec's _decode_qwt, independently of the
    zN repacker. At most one matrix is decoded; no serving fallback is provided.
    """
    validate_gs32_payload(payload, spec)
    indices = torch.from_numpy(_indices(payload, spec).astype(np.int64))
    books = payload["codebooks"].float()
    weight = torch.empty((spec.rows, spec.columns), dtype=torch.float32, device="cpu")
    for column_tile in range(spec.columns // GS32_TILE):
        tile_indices = indices[column_tile].reshape(spec.rows // GS32_TILE, -1)
        vectors = torch.gather(books[column_tile], 1, tile_indices.unsqueeze(-1).expand(-1, -1, 2))
        block = vectors.reshape(spec.rows // GS32_TILE, GS32_TILE, GS32_TILE).transpose(1, 2)
        weight[:, column_tile * GS32_TILE : (column_tile + 1) * GS32_TILE] = block.reshape(spec.rows, GS32_TILE)
    return weight


def decode_gs32_weight_reference(payload, spec):
    """Producer's FP32 inverse transforms, then exactly one BF16 conversion."""
    weight = decode_gs32_codebook_reference(payload, spec)
    weight = weight[:, torch.argsort(payload["perm"].long())]
    weight = weight * payload["weight_scale"].unsqueeze(0) + payload["weight_bias"].unsqueeze(0)
    weight = weight * payload["row_scale"].unsqueeze(1)
    return weight.to(torch.bfloat16)


def prepare_gs32_activation_reference(hidden, payload, spec):
    """Unquantized FP32 input transform for a future fused consumer oracle.

    This is an algebraic scaled-codebook path, NOT the producer BF16-weight
    oracle. MXFP8 quantization, its rounding, and output scaling are separate.
    """
    validate_gs32_payload(payload, spec)
    if (
        not isinstance(hidden, torch.Tensor)
        or hidden.device.type != "cpu"
        or not hidden.is_floating_point()
        or hidden.ndim != 2
        or hidden.shape[1] != spec.columns
        or not bool(torch.isfinite(hidden.float()).all())
    ):
        raise ValueError("Reference hidden must be finite CPU [tokens, K] floating-point data.")
    return (hidden.float() * payload["weight_scale"]).index_select(1, payload["perm"].long())
