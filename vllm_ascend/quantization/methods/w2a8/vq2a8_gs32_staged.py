# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical contract for the eager GS32 scaled-codebook/MXFP8 route.

This is a different numerical route from rounded BF16 reconstructed weights
and from the old rowwise W8A8 backend. CPU references are validation oracles,
never a serving fallback. Device checks deliberately synchronize in this
correctness-first implementation.
"""

from __future__ import annotations

import math

import torch

MX_GROUP = 32
FP8_MAX = 448.0
GS32_STAGE_LAYOUT_FORMAT = "vq2-gs32-down-stage-natural-zn-v1"


def require_finite(tensor, name):
    if not bool(torch.isfinite(tensor.float()).all()):
        raise ValueError(f"{name} contains NaN or Inf.")


def quantize_mxfp8_k32(data):
    """BF16 RNE -> K32 E8M0 scale -> BF16 scaled RNE -> E4M3FN RNE.

    Matches the imported operator's quantize_h convention, including zero
    groups (scale 127), exponent boundary 1.75, and saturation at +/-448.
    Returns raw uint8 data and E8M0 bytes, not an opaque platform NZ tensor.
    """
    if not isinstance(data, torch.Tensor) or not data.is_floating_point() or data.ndim != 2:
        raise ValueError("MXFP8 input must be a floating-point [M,K] tensor.")
    if data.shape[0] < 1 or data.shape[1] < MX_GROUP or data.shape[1] % MX_GROUP:
        raise ValueError("MXFP8 input requires M>0 and positive K divisible by 32.")
    require_finite(data, "MXFP8 input")
    rounded = data.to(torch.bfloat16).float()
    require_finite(rounded, "BF16-rounded MXFP8 input")
    grouped = rounded.reshape(data.shape[0], data.shape[1] // MX_GROUP, MX_GROUP)
    amax = grouped.abs().amax(dim=-1)
    bits = amax.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 255
    codes = exponent - 8 + ((bits & 0x7FFFFF) > 0x600000).to(torch.int32)
    codes = torch.where(amax == 0, 127, codes.clamp_min(1))
    inverse = ((254 - codes) << 23).contiguous().view(torch.float32)
    scaled = (grouped * inverse.unsqueeze(-1)).to(torch.bfloat16).float().clamp(-FP8_MAX, FP8_MAX)
    quantized = scaled.reshape(data.shape).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    return quantized, codes.to(torch.uint8).contiguous()


def _validate_swiglu(gate_up, limit):
    if not math.isfinite(limit) or limit < 0:
        raise ValueError("SwiGLU limit must be finite and non-negative.")
    if gate_up.dtype != torch.bfloat16 or gate_up.ndim != 2 or gate_up.shape[1] % 2:
        raise ValueError("GateUp must be a BF16 [M,2H] tensor in natural row order.")


def _swiglu_arithmetic(gate_up, limit):
    gate, up = gate_up.float().chunk(2, dim=-1)
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return (torch.nn.functional.silu(gate) * up).to(torch.bfloat16).contiguous()


def staged_swiglu_unchecked(gate_up, limit):
    """Same arithmetic; callers must check both raw input and returned output."""
    _validate_swiglu(gate_up, limit)
    return _swiglu_arithmetic(gate_up, limit)


def staged_swiglu(gate_up, limit):
    """Logical Gate-all/Up-all; FP32 clamp/SiLU/product -> one BF16 round."""
    _validate_swiglu(gate_up, limit)
    require_finite(gate_up, "GateUp output")
    hidden = _swiglu_arithmetic(gate_up, limit)
    require_finite(hidden, "SwiGLU output")
    return hidden
