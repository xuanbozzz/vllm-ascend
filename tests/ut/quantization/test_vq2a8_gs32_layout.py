# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pair-LUT byte preservation and producer-order BDNorm CPU oracles."""

import numpy as np
import pytest
import torch

from tools.vq2a8_gs32_artifact import VQ2GS32MatrixSpec
from tools.vq2a8_gs32_layout import (
    decode_gs32_codebook_reference,
    decode_gs32_weight_reference,
    prepare_gs32_activation_reference,
    repack_gs32_matrix,
)


def fixture(kind="gate_up"):
    rows, columns = (128, 64) if kind == "gate_up" else (64, 64)
    metadata = {
        "rows": rows,
        "cols": columns,
        "n_row_tiles": rows // 32,
        "n_col_tiles": columns // 32,
        "row_group_size": 32,
        "group_size": 32,
        "K": 16,
        "index_bits": 4,
        "vector_len": 2,
        "n_vectors": rows * columns // 2,
        "n_elements": rows * columns,
        "orig_shape": [rows, columns],
        "norm_dim": 0,
        "enable_perm": True,
        "enable_norm": True,
        "enable_both_dim_norm": True,
        "enable_rht": False,
        "residual": False,
    }
    spec = VQ2GS32MatrixSpec.from_dict(f"0.mlp.experts.0.{kind}", metadata)
    # Vary every tile, column and pair: a period-16 fixture cannot expose
    # accidentally exchanged K/N tile axes or K16 half-blocks.
    index = np.random.default_rng(49231).integers(0, 16, rows * columns // 2, dtype=np.uint32)
    words = np.zeros(index.size // 8, dtype=np.uint32)
    for shift in range(8):
        words |= index[shift::8] << np.uint32(4 * shift)
    # General, non-Cartesian 16x2 pair codebooks, not four scalar levels.
    books = (torch.arange((columns // 32) * (rows // 32) * 32) * 7 % 120).to(torch.uint8)
    books[0] = 128  # Preserve FP8 negative zero byte-for-byte during repacking.
    payload = {
        "packed_indices": torch.from_numpy(words.view(np.int32)),
        "codebooks": books.reshape(columns // 32, rows // 32, 16, 2).view(torch.float8_e4m3fn),
        "perm": torch.roll(torch.arange(columns, dtype=torch.int32), 23),
        "weight_scale": torch.linspace(0.5, 1.5, columns),
        "weight_bias": torch.zeros(columns),
        "row_scale": torch.linspace(0.75, 1.75, rows),
    }
    return payload, spec, index


def literal_lookup(payload, spec, indices):
    books = payload["codebooks"].float()
    result = torch.empty(spec.rows, spec.columns)
    for row in range(spec.rows):
        for column in range(spec.columns):
            ct, rt = column // 32, row // 32
            offset = (((ct * (spec.rows // 32) + rt) * 32 + column % 32) * 16) + row % 32 // 2
            result[row, column] = books[ct, rt, int(indices[offset]), row % 2]
    return result


@pytest.mark.parametrize("kind", ["gate_up", "down"])
def test_producer_lookup_and_inverse_transforms(kind):
    payload, spec, indices = fixture(kind)
    expected = literal_lookup(payload, spec, indices)
    torch.testing.assert_close(decode_gs32_codebook_reference(payload, spec), expected, rtol=0, atol=0)
    # Independent scatter is the inverse of perm; do not reuse argsort here.
    original_columns = torch.empty_like(expected)
    original_columns[:, payload["perm"].long()] = expected
    expected_weight = (
        (original_columns * payload["weight_scale"] + payload["weight_bias"]) * payload["row_scale"][:, None]
    ).to(torch.bfloat16)
    assert torch.equal(decode_gs32_weight_reference(payload, spec).view(torch.int16), expected_weight.view(torch.int16))


@pytest.mark.parametrize("kind", ["gate_up", "down"])
def test_general_pair_lut_and_indices_survive_native_layout(kind):
    payload, spec, indices = fixture(kind)
    packed = repack_gs32_matrix(payload, spec)
    assert packed["packed_zn"].shape == (spec.rows // 32, spec.columns // 16, 16, 8)
    assert packed["pair_lut"].shape == (spec.rows // 32, spec.columns // 32, 32)
    assert packed["packed_zn"].numel() == spec.rows * spec.columns // 4
    device_bytes = packed["packed_zn"].numpy()
    nibbles = np.stack((device_bytes & 15, device_bytes >> 4), axis=-1).reshape(
        spec.rows // 32, spec.columns // 32, 32, 16
    )
    row_order = list(range(spec.rows // 32))
    if kind == "gate_up":
        half = len(row_order) // 2
        row_order = [tile for pair in zip(range(half), range(half, half * 2)) for tile in pair]
    source_books = payload["codebooks"].view(torch.uint8)
    for device_tile, source_tile in enumerate(row_order):
        for column_tile in range(spec.columns // 32):
            assert torch.equal(
                packed["pair_lut"][device_tile, column_tile].reshape(16, 2),
                source_books[column_tile, source_tile],
            )
            for column in range(32):
                for pair in range(16):
                    position = (((column_tile * (spec.rows // 32) + source_tile) * 32 + column) * 16) + pair
                    assert nibbles[device_tile, column_tile, column, pair] == indices[position]
    np.testing.assert_array_equal(
        packed["row_scale_bits"].numpy(), payload["row_scale"].numpy().view(np.uint32).astype(np.uint64)
    )
    assert torch.equal(packed["perm"], payload["perm"])
    assert torch.equal(packed["weight_scale"], payload["weight_scale"])
    assert packed["perm"].data_ptr() != payload["perm"].data_ptr()


def test_activation_transform_uses_forward_perm_after_column_scale():
    payload, spec, _ = fixture("down")
    hidden = torch.arange(2 * spec.columns).reshape(2, spec.columns).float() / 128
    transformed = prepare_gs32_activation_reference(hidden, payload, spec)
    expected = torch.stack(
        [hidden[:, int(column)] * payload["weight_scale"][int(column)] for column in payload["perm"]], dim=1
    )
    torch.testing.assert_close(transformed, expected, rtol=0, atol=0)
    # Demonstrate the real-number scale relocation, not BF16-weight equivalence.
    books = decode_gs32_codebook_reference(payload, spec).double()
    dense = books[:, torch.argsort(payload["perm"].long())] * payload["weight_scale"].double()
    dense *= payload["row_scale"].double()[:, None]
    rearranged = (hidden.double() * payload["weight_scale"].double())[:, payload["perm"].long()]
    torch.testing.assert_close(
        (rearranged @ books.T) * payload["row_scale"].double(), hidden.double() @ dense.T, rtol=1e-12, atol=1e-12
    )


def test_unsupported_bias_and_invalid_activation_fail_closed():
    payload, spec, _ = fixture()
    payload["weight_bias"][0] = 0.125
    with pytest.raises(ValueError):
        repack_gs32_matrix(payload, spec)
    payload["weight_bias"].zero_()
    for hidden in (torch.zeros(1, spec.columns + 1), torch.full((1, spec.columns), float("nan"))):
        with pytest.raises(ValueError):
            prepare_gs32_activation_reference(hidden, payload, spec)
