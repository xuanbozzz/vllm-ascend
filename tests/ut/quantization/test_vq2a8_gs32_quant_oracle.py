# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent ml_dtypes oracle from the unchanged imported operator sources.

These are CPU numerical tests, not evidence that the NPU supports or correctly
executes the framework operations. No imported operator file is modified.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_staged import quantize_mxfp8_k32


@pytest.fixture(scope="module")
def operator_reference():
    pytest.importorskip("ml_dtypes")
    source = Path(__file__).resolve().parents[3] / "csrc/mixed_w2w4_moe_ffn/scripts/swiglu_reference.py"
    spec = importlib.util.spec_from_file_location("_gs32_untouched_operator_reference", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_original_quantizer_bytes(data, reference):
    actual_data, actual_scale = quantize_mxfp8_k32(data)
    expected_data, expected_scale = reference.quantize_h(data.numpy())
    np.testing.assert_array_equal(actual_data.numpy(), expected_data.view(np.uint8))
    np.testing.assert_array_equal(actual_scale.numpy(), expected_scale)


def test_all_finite_bf16_bit_patterns_match_original_quantizer(operator_reference):
    # Constant groups exercise every BF16 exponent/mantissa and both signs,
    # including signed zero and BF16 subnormals; comparisons include raw bits.
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16).float()
    values = values[torch.isfinite(values)]
    assert values.numel() == 65280
    data = values[:, None].expand(-1, 32).contiguous()
    assert_original_quantizer_bytes(data, operator_reference)


def test_mixed_sign_midpoints_and_scale_boundaries_match_original_quantizer(operator_reference):
    # Nonconstant groups cover E4M3 ties and small values under another
    # element's scale. The max=224/226/448/450 cases straddle E8M0 boundaries.
    values = torch.tensor(
        [
            0.0,
            -0.0,
            1 / 1024,
            -1 / 1024,
            1 / 512,
            -1 / 512,
            3 / 1024,
            -3 / 1024,
            15 / 1024,
            -15 / 1024,
            1 / 64,
            -1 / 64,
            1.0625,
            -1.0625,
            1.1875,
            -1.1875,
            1.75,
            -1.75,
            1.7578125,
            -1.7578125,
            223.0,
            -223.0,
            224.0,
            -224.0,
            225.0,
            -225.0,
            226.0,
            -226.0,
            447.0,
            -447.0,
            448.0,
            -448.0,
        ],
        dtype=torch.float32,
    )
    rows = []
    for exponent in (-120, -16, -1, 0, 1, 16, 110):
        for maximum in (224.0, 226.0, 448.0, 450.0):
            row = values.clamp(-maximum, maximum)
            row[-2], row[-1] = maximum, -maximum
            rows.append(row * (2.0**exponent))
    data = torch.stack(rows)
    assert_original_quantizer_bytes(data, operator_reference)


def test_mixed_random_finite_bf16_groups_match_original_quantizer(operator_reference):
    generator = torch.Generator().manual_seed(20260929)
    bits = torch.randint(0, 65536, (513, 128), generator=generator, dtype=torch.int32).to(torch.int16)
    data = bits.view(torch.bfloat16).float()
    # Replace only exponent-255 encodings. Quantization's explicit nonfinite
    # rejection has separate tests; this test compares finite input numerics.
    data = torch.where(torch.isfinite(data), data, torch.zeros_like(data))
    assert_original_quantizer_bytes(data, operator_reference)
