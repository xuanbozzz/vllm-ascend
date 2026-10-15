# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GS32 preparation kernels; all arithmetic follows the staged MXFP8 ABI."""

import torch
from vllm.triton_utils import tl, triton

MX_GROUP = 32
FINITE_BLOCK = 1024
FLOAT32_MAX = 3.4028234663852886e38


@triton.jit
def _quantize_groups(values, BLOCK_K: tl.constexpr, GROUP: tl.constexpr):
    grouped = tl.reshape(values, (BLOCK_K // GROUP, GROUP))
    maximum = tl.max(tl.abs(grouped), axis=1)
    bits = maximum.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) & 255
    codes = exponent - 8 + ((bits & 0x7FFFFF) > 0x600000).to(tl.int32)
    codes = tl.where(maximum == 0, 127, tl.maximum(codes, 1))
    inverse = ((254 - codes) << 23).to(tl.float32, bitcast=True)
    scaled = (grouped * inverse[:, None]).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    scaled = tl.minimum(tl.maximum(scaled, -448.0), 448.0)
    return tl.reshape(scaled, (BLOCK_K,)), codes


@triton.jit
def _prepare_activation_kernel(
    hidden_ptr,
    scale_ptr,
    perm_ptr,
    slot_ptr,
    raw_gate_ptr,
    output_ptr,
    output_scale_ptr,
    flags_ptr,
    K: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_RAW_GATE: tl.constexpr,
    GROUP: tl.constexpr,
    FLOAT_MAX: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_K)
    active = columns < K
    slot = tl.load(slot_ptr + row)
    slot_valid = (slot >= 0) & (slot < EXPERTS)
    safe_slot = tl.where(slot_valid, slot, 0)
    permutation = tl.load(perm_ptr + safe_slot * K + columns, mask=active & slot_valid, other=0)
    index_valid = (permutation >= 0) & (permutation < K)
    safe_index = tl.where(index_valid, permutation, 0)
    read_mask = active & slot_valid & index_valid
    hidden = tl.load(hidden_ptr + row * K + safe_index, mask=read_mask, other=0).to(tl.float32)
    # The reference scales BEFORE permuting: both loads use the source index.
    scale = tl.load(scale_ptr + safe_slot * K + safe_index, mask=read_mask, other=0)
    transformed = hidden * scale
    rounded = transformed.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    finite = (tl.abs(hidden) <= FLOAT_MAX) & (tl.abs(scale) <= FLOAT_MAX)
    finite = finite & (tl.abs(transformed) <= FLOAT_MAX) & (tl.abs(rounded) <= FLOAT_MAX)
    valid = slot_valid & index_valid & finite
    if HAS_RAW_GATE:
        gate = tl.load(raw_gate_ptr + row * (2 * K) + columns, mask=active, other=0).to(tl.float32)
        up = tl.load(raw_gate_ptr + row * (2 * K) + K + columns, mask=active, other=0).to(tl.float32)
        valid = valid & (tl.abs(gate) <= FLOAT_MAX) & (tl.abs(up) <= FLOAT_MAX)
    # Strict callers reject invalid flags before projection; graph callers
    # defer rejection until graph completion. Both receive finite FP8 bytes.
    safe_rounded = tl.where(active & valid, rounded, 0.0)
    scaled, codes = _quantize_groups(safe_rounded, BLOCK_K, GROUP)
    fp8 = scaled.to(output_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
    tl.store(output_ptr + row * K + columns, fp8, mask=active)
    groups = tl.arange(0, BLOCK_K // GROUP)
    tl.store(output_scale_ptr + row * (K // GROUP) + groups, codes.to(tl.uint8), mask=groups < K // GROUP)
    row_valid = tl.min(tl.where(active, valid, True).to(tl.int32), axis=0)
    tl.store(flags_ptr + row, row_valid)


@triton.jit
def _finite_flags_kernel(tensor_ptr, flags_ptr, ELEMENTS: tl.constexpr, BLOCK: tl.constexpr, FLOAT_MAX: tl.constexpr):
    tile = tl.program_id(0)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(tensor_ptr + offsets, mask=offsets < ELEMENTS, other=0).to(tl.float32)
    valid = tl.min((tl.abs(values) <= FLOAT_MAX).to(tl.int32), axis=0)
    tl.store(flags_ptr + tile, valid)


@triton.jit
def _finish_flags_kernel(
    output_ptr,
    gate_flags_ptr,
    down_flags_ptr,
    route_flags_ptr,
    native_flags_ptr,
    result_ptr,
    ELEMENTS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    FLAG_BLOCK: tl.constexpr,
    ROUTE_ELEMENTS: tl.constexpr,
    NATIVE_ELEMENTS: tl.constexpr,
    FLOAT_MAX: tl.constexpr,
):
    # One writer owns the final result. Tiled scans bound UB use independently
    # of the FFN output width; no atomics or stale accumulator are involved.
    valid = tl.full((), 1, tl.int32)
    columns = tl.arange(0, BLOCK)
    for start in range(0, ELEMENTS, BLOCK):
        offsets = start + columns
        values = tl.load(output_ptr + offsets, mask=offsets < ELEMENTS, other=0).to(tl.float32)
        valid = valid & tl.min((tl.abs(values) <= FLOAT_MAX).to(tl.int32), axis=0)
    rows = tl.arange(0, FLAG_BLOCK)
    gate = tl.load(gate_flags_ptr + rows, mask=rows < ROWS, other=1)
    down = tl.load(down_flags_ptr + rows, mask=rows < ROWS, other=1)
    upstream_valid = tl.min(((gate == 1) & (down == 1)).to(tl.int32), axis=0)
    for start in range(0, ROUTE_ELEMENTS, BLOCK):
        offsets = start + columns
        flags = tl.load(route_flags_ptr + offsets, mask=offsets < ROUTE_ELEMENTS, other=1)
        upstream_valid = upstream_valid & tl.min((flags == 1).to(tl.int32), axis=0)
    for start in range(0, NATIVE_ELEMENTS, BLOCK):
        offsets = start + columns
        flags = tl.load(native_flags_ptr + offsets, mask=offsets < NATIVE_ELEMENTS, other=1)
        upstream_valid = upstream_valid & tl.min((flags == 1).to(tl.int32), axis=0)
    tl.store(result_ptr, valid & upstream_valid)


def prepare_activation(hidden, weight_scale, perm, row_slots, raw_gate=None):
    rows, width = hidden.shape
    output = torch.empty((rows, width), device=hidden.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows, width // MX_GROUP), device=hidden.device, dtype=torch.uint8)
    flags = torch.empty((rows,), device=hidden.device, dtype=torch.int32)
    _prepare_activation_kernel[(rows,)](
        hidden,
        weight_scale,
        perm,
        row_slots,
        hidden if raw_gate is None else raw_gate,
        output,
        scales,
        flags,
        K=width,
        EXPERTS=weight_scale.shape[0],
        BLOCK_K=triton.next_power_of_2(width),
        HAS_RAW_GATE=raw_gate is not None,
        GROUP=MX_GROUP,
        FLOAT_MAX=FLOAT32_MAX,
    )
    return output.view(torch.uint8), scales, flags


def finite_flags(tensor):
    count = triton.cdiv(tensor.numel(), FINITE_BLOCK)
    flags = torch.empty((count,), device=tensor.device, dtype=torch.int32)
    _finite_flags_kernel[(count,)](tensor, flags, ELEMENTS=tensor.numel(), BLOCK=FINITE_BLOCK, FLOAT_MAX=FLOAT32_MAX)
    return flags


def finish_flags(output, gate_flags, down_flags, *, route_flags=None, native_flags=None):
    result = torch.empty((1,), device=output.device, dtype=torch.int32)
    _finish_flags_kernel[(1,)](
        output,
        gate_flags,
        down_flags,
        gate_flags if route_flags is None else route_flags,
        gate_flags if native_flags is None else native_flags,
        result,
        ELEMENTS=output.numel(),
        ROWS=output.shape[0],
        BLOCK=FINITE_BLOCK,
        FLAG_BLOCK=triton.next_power_of_2(max(output.shape[0], 1)),
        ROUTE_ELEMENTS=0 if route_flags is None else route_flags.numel(),
        NATIVE_ELEMENTS=0 if native_flags is None else native_flags.numel(),
        FLOAT_MAX=FLOAT32_MAX,
    )
    return result
