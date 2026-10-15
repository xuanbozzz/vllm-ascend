# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CPU numerics and explicit fake transport, never a simulated NPU PASS."""

import math

import numpy as np
import pytest
import torch

from tests.ut.quantization.test_vq2a8_gs32_layout import fixture, literal_lookup
from tools.vq2a8_gs32_layout import repack_gs32_matrix
from tools.vq2a8_gs32_reference import (
    dequantize_mxfp8_k32,
    quantize_mxfp8_k32,
    staged_swiglu,
)


def test_natural_gate_layout_preserves_every_pair_and_scale():
    payload, spec, source_indices = fixture()
    fields = repack_gs32_matrix(payload, spec, down_stage=True)
    assert not torch.equal(fields["packed_zn"], repack_gs32_matrix(payload, spec)["packed_zn"])
    decoded = FakeDownOps.decode(fields["packed_zn"].unsqueeze(0), fields["pair_lut"].unsqueeze(0))
    torch.testing.assert_close(decoded, literal_lookup(payload, spec, source_indices), rtol=0, atol=0)
    torch.testing.assert_close(fields["row_scale_bits"].to(torch.int32).view(torch.float32), payload["row_scale"])


def test_quantize_matches_independent_scale_calculation():
    generator = torch.Generator().manual_seed(397)
    data = torch.randn(9, 128, generator=generator) * torch.logspace(-8, 8, 9).unsqueeze(-1)
    rounded = data.to(torch.bfloat16).float().numpy().reshape(9, 4, 32)
    expected_scales = np.empty((9, 4), dtype=np.uint8)
    expected_scaled = np.empty_like(rounded)
    for row in range(9):
        for group in range(4):
            maximum = float(np.max(np.abs(rounded[row, group])))
            exponent = max(-126, math.ceil(math.log2(maximum / 448))) if maximum else 0
            expected_scales[row, group] = exponent + 127
            expected_scaled[row, group] = rounded[row, group] * (2.0**-exponent)
    expected_bytes = (
        torch.from_numpy(expected_scaled.reshape(9, 128))
        .to(torch.bfloat16)
        .float()
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    actual, scales = quantize_mxfp8_k32(data)
    assert torch.equal(scales, torch.from_numpy(expected_scales))
    assert torch.equal(actual, expected_bytes)
    recovered = dequantize_mxfp8_k32(actual, scales)
    assert recovered.shape == data.shape and torch.isfinite(recovered).all()


@pytest.mark.parametrize("maximum,code", [(0, 127), (224, 126), (226, 127), (448, 127), (450, 128)])
def test_quantize_scale_boundaries(maximum, code):
    values = torch.zeros(1, 32)
    values[0, 0] = maximum
    values[0, 1] = -float(maximum)
    quantized, scales = quantize_mxfp8_k32(values)
    assert scales.tolist() == [[code]]
    if maximum == 0:
        assert quantized[0, 0] == 0 and quantized[0, 1] == 128


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), torch.finfo(torch.float32).max])
def test_quantize_nonfinite_or_bf16_overflow_rejected(bad):
    with pytest.raises(ValueError):
        quantize_mxfp8_k32(torch.full((1, 32), bad))


@pytest.mark.parametrize("shape", [(0, 32), (1, 0), (2, 33), (32,)])
def test_quantize_invalid_shapes(shape):
    with pytest.raises(ValueError):
        quantize_mxfp8_k32(torch.zeros(shape))


def test_swiglu_clips_asymmetrically_and_rounds_once():
    gate_up = torch.tensor([[-20, 20, -30, 30]], dtype=torch.bfloat16)
    actual = staged_swiglu(gate_up, 10)
    gate = torch.tensor([[-20.0, 10.0]])
    up = torch.tensor([[-10.0, 10.0]])
    expected = (gate / (1 + torch.exp(-gate)) * up).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, staged_swiglu(gate_up, 0))
    for limit in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            staged_swiglu(gate_up, limit)


class FakeDownOps:
    """Explicit CPU test transport decoding the ACTUAL native packed bytes."""

    def __init__(self):
        self.launches = 0
        self.tilings = []

    def check_eager(self):
        pass

    def make_tiling(self, k, n):
        self.tilings.append((k, n))
        return torch.zeros(1600, dtype=torch.uint8)

    @staticmethod
    def decode(packed, lut):
        n, k = packed.shape[1] * 32, packed.shape[2] * 16
        bytes_ = packed[0].numpy()
        indices = np.stack((bytes_ & 15, bytes_ >> 4), -1).reshape(n // 32, k // 32, 32, 16)
        weight = torch.empty(n, k)
        for nt in range(n // 32):
            for kt in range(k // 32):
                books = lut[0, nt, kt].contiguous().view(torch.float8_e4m3fn).float().reshape(16, 2)
                tile = books[torch.from_numpy(indices[nt, kt].astype(np.int64))].permute(1, 2, 0).reshape(32, 32)
                weight[nt * 32 : (nt + 1) * 32, kt * 32 : (kt + 1) * 32] = tile
        return weight

    def project(self, data, scales, weight, lut, row_scale, group_list, mask, tiling, output):
        assert group_list.device.type == mask.device.type == tiling.device.type == "cpu"
        assert group_list.tolist() == [output.shape[0]] and mask.tolist() == [0]
        assert row_scale.dtype == torch.int64
        decoded = self.decode(weight, lut)
        coefficient = row_scale[0].to(torch.int32).view(torch.float32)
        output.copy_((dequantize_mxfp8_k32(data, scales) @ decoded.T) * coefficient)
        self.launches += 1
