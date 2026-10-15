# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture-safe GS32 route preparation with deferred device validation."""

import torch

from vllm_ascend.ops.fused_moe.vq2a8_gs32_activation import _require_npu_tensor

MAX_DECODE_ROUTED_ROWS = 8
MAX_ROUTED_ROWS = 256
MAX_ROUTE_EXPERTS = 256
NATIVE_ALIGNMENT = 32


def prepare_routes(hidden, group_list, group_type, expert_ids, logical_slots, *, allow_padding=False):
    # Fixed-shape tensor routing avoids the toolchain's Triton metadata compiler
    # failure and remains capturable: no counts or validation scalars leave device.
    rows, experts = hidden.shape[0], expert_ids.numel()
    raw = group_list.to(torch.int64)
    bounded = raw.clamp(0, rows)
    counts = bounded if group_type == 1 else bounded - torch.cat((bounded.new_zeros(1), bounded[:-1]))
    safe_counts = counts.clamp(0, rows)
    valid = ((raw >= 0) & (raw <= rows) & (counts >= 0) & (counts <= 128)).all()
    total = safe_counts.sum()
    valid = valid & ((total <= rows) if allow_padding else (total == rows))
    safe_experts = expert_ids.clamp(0, group_list.numel() - 1)
    safe_slots = logical_slots.clamp(0, experts - 1)
    slot_valid = (logical_slots >= 0) & (logical_slots < experts)
    logical = torch.arange(group_list.numel(), device=hidden.device)
    valid = (
        valid
        & (((logical_slots == -1) & (safe_counts == 0)) | (slot_valid & (expert_ids[safe_slots] == logical))).all()
    )
    physical = torch.arange(experts, device=hidden.device)
    valid = (
        valid
        & ((expert_ids >= 0) & (expert_ids < group_list.numel()) & (logical_slots[safe_experts] == physical)).all()
    )
    physical_counts = safe_counts[safe_experts]
    valid = valid & (physical_counts.sum() == total)
    ends = physical_counts.cumsum(0)
    starts = ends - physical_counts
    logical_starts = safe_counts.cumsum(0) - safe_counts
    offsets = torch.arange(rows, device=hidden.device)
    slots = (offsets[:, None] >= ends[None, :]).sum(1).clamp(0, experts - 1)
    live = offsets < total
    sources = logical_starts[safe_experts[slots]] + offsets - starts[slots]
    sources = torch.where(live, sources, offsets).clamp(0, rows - 1)
    inverse = torch.argsort(sources)
    output = torch.where((valid & live)[:, None], hidden.index_select(0, sources), torch.zeros_like(hidden))
    row_slots = torch.where(live, slots, torch.zeros_like(slots))
    return (
        output,
        torch.where(valid, physical_counts, torch.zeros_like(physical_counts)),
        row_slots,
        inverse,
        valid.int().reshape(1),
    )


def prepare(hidden, group_list, group_type, expert_ids, logical_slots, *, allow_padding=False):
    """Reorder logical dispatch rows; caller must check returned device status.

    Invalid dynamic routes produce zero counts and zero hidden values, so a
    deferred error never becomes an unchecked native pointer calculation.
    EP explicitly permits fewer live rows than capacity; unused rows become
    zero and keep an identity inverse so restoration never reads out of bounds.
    """
    for name, tensor in (
        ("hidden", hidden),
        ("group_list", group_list),
        ("expert_ids", expert_ids),
        ("logical_slots", logical_slots),
    ):
        _require_npu_tensor(tensor, name)
        if tensor.device != hidden.device:
            raise ValueError("GS32 device routes require all tensors on the same NPU.")
    if (
        hidden.dtype != torch.bfloat16
        or hidden.ndim != 2
        or not 1 <= hidden.shape[0] <= MAX_ROUTED_ROWS
        or hidden.shape[1] < NATIVE_ALIGNMENT
        or hidden.shape[1] % NATIVE_ALIGNMENT
    ):
        raise ValueError("GS32 device routes require BF16 [R,K], 1<=R<=256 and positive K divisible by 32.")
    if type(group_type) is not int or group_type not in (0, 1):
        raise ValueError("GS32 device routes group_type must be 0 (prefix) or 1 (counts).")
    if type(allow_padding) is not bool:
        raise ValueError("GS32 device routes allow_padding must be a bool.")
    if (
        group_list.dtype not in (torch.int32, torch.int64)
        or group_list.ndim != 1
        or not 1 <= group_list.numel() <= MAX_ROUTE_EXPERTS
    ):
        raise ValueError("GS32 device routes require int32/int64 group_list[L], 1<=L<=256.")
    if expert_ids.dtype != torch.int64 or expert_ids.ndim != 1 or not 1 <= expert_ids.numel() <= MAX_ROUTE_EXPERTS:
        raise ValueError("GS32 device routes require int64 expert_ids[E], 1<=E<=256.")
    if logical_slots.dtype != torch.int64 or logical_slots.shape != group_list.shape:
        raise ValueError("GS32 device routes require int64 logical_slots[L].")
    with torch.npu.device(hidden.device):
        return prepare_routes(hidden, group_list, group_type, expert_ids, logical_slots, allow_padding=allow_padding)
