# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline GS32 numerical oracles; never used by model execution."""

import runpy
from pathlib import Path

import torch

from tools.vq2a8_gs32_layout import decode_gs32_codebook_reference, prepare_gs32_activation_reference

_math = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "vllm_ascend/quantization/methods/w2a8/vq2a8_gs32_staged.py")
)
MX_GROUP = _math["MX_GROUP"]
quantize_mxfp8_k32 = _math["quantize_mxfp8_k32"]
require_finite = _math["require_finite"]
staged_swiglu = _math["staged_swiglu"]
GS32_STAGE_LAYOUT_FORMAT = _math["GS32_STAGE_LAYOUT_FORMAT"]
del _math


def dequantize_mxfp8_k32(data, scales):
    """CPU-only independent projection oracle input decoding."""
    if data.device.type != "cpu" or scales.device.type != "cpu":
        raise ValueError("MXFP8 reference decoding is CPU-only.")
    if data.dtype != torch.uint8 or scales.dtype != torch.uint8 or data.ndim != 2:
        raise ValueError("Expected uint8 MXFP8 data and scale bytes.")
    m, k = data.shape
    if m < 1 or k < MX_GROUP or k % MX_GROUP or scales.shape != (m, k // MX_GROUP):
        raise ValueError("Invalid MXFP8 data/scale shape.")
    if bool((scales == 255).any()) or bool((scales == 0).any()):
        raise ValueError("This quantizer emits E8M0 codes 1..254 only.")
    scale_values = (scales.to(torch.int32) << 23).contiguous().view(torch.float32)
    result = data.contiguous().view(torch.float8_e4m3fn).float().reshape(m, k // MX_GROUP, MX_GROUP)
    result = (result * scale_values.unsqueeze(-1)).reshape(m, k)
    require_finite(result, "Decoded MXFP8 activation")
    return result


def staged_projection_reference(hidden, payload, spec):
    """CPU FP32 GEMM + row scale -> BF16, using quantized transformed input.

    Accumulator order may differ from hardware; this is not a bitwise GEMM
    oracle. It does not materialize or reuse BF16 reconstructed model weights.
    """
    transformed = prepare_gs32_activation_reference(hidden, payload, spec)
    data, scale = quantize_mxfp8_k32(transformed)
    activation = dequantize_mxfp8_k32(data, scale)
    codebook_weight = decode_gs32_codebook_reference(payload, spec)
    output = ((activation @ codebook_weight.T) * payload["row_scale"]).to(torch.bfloat16)
    require_finite(output, "Projection reference")
    return output


def staged_expert_reference(hidden, gate_payload, gate_spec, down_payload, down_spec, limit):
    gate_up = staged_projection_reference(hidden, gate_payload, gate_spec)
    middle = staged_swiglu(gate_up, limit)
    return staged_projection_reference(middle, down_payload, down_spec)
