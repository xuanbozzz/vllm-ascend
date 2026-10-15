# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded device routing for GS32 decode and fixed-size prefill graphs."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.fused_moe.vq2a8_gs32_routes import MAX_DECODE_ROUTED_ROWS

COPY_BLOCK = 512
MAX_EXPERT_ROWS = 128


@triton.jit
def _route_metadata_kernel(
    group_ptr,
    expert_ptr,
    logical_slot_ptr,
    counts_ptr,
    row_slots_ptr,
    gather_ptr,
    inverse_ptr,
    flags_ptr,
    ROWS: tl.constexpr,
    LOGICAL: tl.constexpr,
    EXPERTS: tl.constexpr,
    GROUP_TYPE: tl.constexpr,
    LOGICAL_BLOCK: tl.constexpr,
    EXPERT_BLOCK: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    MAX_COUNT: tl.constexpr,
    BUILD_ROWS: tl.constexpr,
    ALLOW_PADDING: tl.constexpr,
):
    logical = tl.arange(0, LOGICAL_BLOCK)
    logical_active = logical < LOGICAL
    raw = tl.load(group_ptr + logical, mask=logical_active, other=0).to(tl.int64)
    bounded = tl.minimum(tl.maximum(raw, 0), ROWS)
    value_valid = (raw >= 0) & (raw <= ROWS)
    if GROUP_TYPE == 0:
        previous = tl.load(group_ptr + logical - 1, mask=logical_active & (logical > 0), other=0).to(tl.int64)
        # Subtract only bounded values: INT64_MIN/MAX must not overflow.
        counts = bounded - tl.minimum(tl.maximum(previous, 0), ROWS)
    else:
        counts = bounded
    value_valid = value_valid & (counts >= 0) & (counts <= MAX_COUNT)
    counts = tl.where(logical_active, tl.minimum(tl.maximum(counts, 0), ROWS), 0).to(tl.int32)
    slots = tl.load(logical_slot_ptr + logical, mask=logical_active, other=-1)
    slot_valid = (slots >= 0) & (slots < EXPERTS)
    physical = tl.arange(0, EXPERT_BLOCK)
    physical_active = physical < EXPERTS
    expert = tl.load(expert_ptr + physical, mask=physical_active, other=0)
    expert_valid = (expert >= 0) & (expert < LOGICAL)
    # Keep inventory addresses affine; values from partially masked loads
    # must only feed register lookups, never subsequent GM addresses.
    # Validate original int64 values before narrowing; [-1,255] is exact FP32.
    bounded_slots = tl.where(slot_valid, slots, -1).to(tl.float32)
    bounded_experts = tl.where(expert_valid, expert, -1).to(tl.float32)
    safe_slot = tl.where(slot_valid, slots, 0).to(tl.int32)
    safe_expert = tl.where(expert_valid, expert, 0).to(tl.int32)
    back_id = tl.gather(bounded_experts, safe_slot, axis=0).to(tl.int32)
    back_slot = tl.gather(bounded_slots, safe_expert, axis=0).to(tl.int32)
    mapping_valid = ((slots == -1) & (counts == 0)) | (slot_valid & (back_id == logical))
    valid = tl.min(tl.where(logical_active, value_valid & mapping_valid, True).to(tl.int32), axis=0) == 1
    valid = valid & ((tl.sum(counts, axis=0) <= ROWS) if ALLOW_PADDING else (tl.sum(counts, axis=0) == ROWS))
    valid = valid & (
        tl.min(tl.where(physical_active, expert_valid & (back_slot == physical), True).to(tl.int32), axis=0) == 1
    )
    # Ascend gather rejects int32 sources. Counts are already bounded to
    # [0, ROWS<=256], so the FP32 roundtrip preserves every count exactly.
    physical_counts = tl.gather(counts.to(tl.float32), safe_expert, axis=0).to(tl.int32)
    physical_counts = tl.where(physical_active, physical_counts, 0)
    valid = valid & (tl.sum(physical_counts, axis=0) == (tl.sum(counts, axis=0) if ALLOW_PADDING else ROWS))
    tl.store(counts_ptr + physical, tl.where(valid, physical_counts, 0), mask=physical_active)

    if BUILD_ROWS:
        # Keep the proven decode path: at most eight row-selection iterations.
        row_offsets = tl.arange(0, ROW_BLOCK)
        slot_values = tl.full((ROW_BLOCK,), 0, tl.int32)
        source_values = tl.full((ROW_BLOCK,), 0, tl.int32)
        inverse_values = tl.where(valid, row_offsets, 0)
        remaining = physical_counts
        for row in range(ROWS):
            selected = tl.min(tl.where(physical_active & (remaining > 0), physical, EXPERTS), axis=0)
            has_owner = selected < EXPERTS
            selected = tl.where(selected < EXPERTS, selected, 0)
            owner = physical_active & (physical == selected) & has_owner
            selected_expert = tl.sum(tl.where(owner, safe_expert, 0), axis=0)
            consumed = tl.sum(tl.where(owner, physical_counts - remaining, 0), axis=0)
            source_start = tl.sum(tl.where(logical < selected_expert, counts, 0), axis=0)
            source = source_start + consumed
            slot_values = tl.where(row_offsets == row, tl.where(valid, selected, 0), slot_values)
            source_values = tl.where(
                row_offsets == row, tl.where(valid, tl.where(has_owner, source, -1), 0), source_values
            )
            # Build the inverse in registers, never as a data-dependent scatter.
            inverse_values = tl.where(valid & has_owner & (row_offsets == source), row, inverse_values)
            remaining = remaining - owner.to(tl.int32)
        tl.store(row_slots_ptr + row_offsets, slot_values, mask=row_offsets < ROWS)
        tl.store(gather_ptr + row_offsets, source_values, mask=row_offsets < ROWS)
        tl.store(inverse_ptr + row_offsets, inverse_values, mask=row_offsets < ROWS)
    tl.store(flags_ptr, valid.to(tl.int32))


@triton.jit
def _prefill_route_rows_kernel(
    group_ptr,
    expert_ptr,
    counts_ptr,
    row_slots_ptr,
    gather_ptr,
    inverse_ptr,
    flags_ptr,
    ROWS: tl.constexpr,
    LOGICAL: tl.constexpr,
    EXPERTS: tl.constexpr,
    GROUP_TYPE: tl.constexpr,
    LOGICAL_BLOCK: tl.constexpr,
    EXPERT_BLOCK: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    # One program owns one expert's disjoint output/source intervals. This
    # avoids a ROWS-long decode loop, scans, and indirect vector GM addresses.
    slot = tl.program_id(0)
    valid = tl.load(flags_ptr) == 1
    expert = tl.load(expert_ptr + slot)
    logical = tl.arange(0, LOGICAL_BLOCK)
    raw = tl.load(group_ptr + logical, mask=logical < LOGICAL, other=0).to(tl.int64)
    bounded = tl.minimum(tl.maximum(raw, 0), ROWS)
    if GROUP_TYPE == 0:
        previous = tl.load(group_ptr + logical - 1, mask=(logical < LOGICAL) & (logical > 0), other=0).to(tl.int64)
        logical_counts = bounded - tl.minimum(tl.maximum(previous, 0), ROWS)
    else:
        logical_counts = bounded
    logical_counts = tl.where(logical < LOGICAL, logical_counts, 0).to(tl.int32)
    source_start = tl.sum(tl.where(logical < expert, logical_counts, 0), axis=0)
    physical = tl.arange(0, EXPERT_BLOCK)
    physical_counts = tl.load(counts_ptr + physical, mask=physical < EXPERTS, other=0).to(tl.int32)
    physical_start = tl.sum(tl.where(physical < slot, physical_counts, 0), axis=0)
    count = tl.sum(tl.where(physical == slot, physical_counts, 0), axis=0)
    total_rows = tl.sum(physical_counts, axis=0)

    row_offsets = tl.arange(0, ROW_BLOCK)
    active = row_offsets < ROWS
    clear_invalid = (~valid) & (slot == 0)
    clear_padding = valid & (slot == 0) & (row_offsets >= total_rows)
    physical_mask = active & (
        clear_invalid
        | clear_padding
        | (valid & (row_offsets >= physical_start) & (row_offsets < physical_start + count))
    )
    source_mask = active & (
        clear_invalid | clear_padding | (valid & (row_offsets >= source_start) & (row_offsets < source_start + count))
    )
    slots = tl.where(valid & ~clear_padding, slot, 0)
    sources = tl.where(valid, tl.where(clear_padding, -1, source_start + row_offsets - physical_start), 0)
    inverse = tl.where(valid, tl.where(clear_padding, row_offsets, physical_start + row_offsets - source_start), 0)
    # Addresses remain affine even with non-power-of-two expert/row tails.
    tl.store(row_slots_ptr + row_offsets, slots, mask=physical_mask)
    tl.store(gather_ptr + row_offsets, sources, mask=physical_mask)
    tl.store(inverse_ptr + row_offsets, inverse, mask=source_mask)


@triton.jit
def _gather_hidden_kernel(
    hidden_ptr, gather_ptr, flags_ptr, output_ptr, ROWS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source = tl.load(gather_ptr + row)
    valid = (tl.load(flags_ptr) == 1) & (source >= 0) & (source < ROWS)
    source = tl.where(valid, source, 0)
    values = tl.load(hidden_ptr + source * WIDTH + columns, mask=valid & (columns < WIDTH), other=0)
    tl.store(output_ptr + row * WIDTH + columns, values, mask=columns < WIDTH)


def prepare_routes(hidden, group_list, group_type, expert_ids, logical_slots, *, allow_padding=False):
    rows, width = hidden.shape
    output = torch.empty_like(hidden)
    counts = torch.empty_like(expert_ids)
    row_slots = torch.empty((rows,), device=hidden.device, dtype=torch.int64)
    gather = torch.empty_like(row_slots)
    inverse = torch.empty_like(row_slots)
    flags = torch.empty((1,), device=hidden.device, dtype=torch.int32)
    _route_metadata_kernel[(1,)](
        group_list,
        expert_ids,
        logical_slots,
        counts,
        row_slots,
        gather,
        inverse,
        flags,
        ROWS=rows,
        LOGICAL=group_list.numel(),
        EXPERTS=expert_ids.numel(),
        GROUP_TYPE=group_type,
        LOGICAL_BLOCK=triton.next_power_of_2(group_list.numel()),
        EXPERT_BLOCK=triton.next_power_of_2(expert_ids.numel()),
        ROW_BLOCK=triton.next_power_of_2(rows),
        MAX_COUNT=MAX_EXPERT_ROWS,
        BUILD_ROWS=rows <= MAX_DECODE_ROUTED_ROWS,
        ALLOW_PADDING=allow_padding,
    )
    if rows > MAX_DECODE_ROUTED_ROWS:
        _prefill_route_rows_kernel[(expert_ids.numel(),)](
            group_list,
            expert_ids,
            counts,
            row_slots,
            gather,
            inverse,
            flags,
            ROWS=rows,
            LOGICAL=group_list.numel(),
            EXPERTS=expert_ids.numel(),
            GROUP_TYPE=group_type,
            LOGICAL_BLOCK=triton.next_power_of_2(group_list.numel()),
            EXPERT_BLOCK=triton.next_power_of_2(expert_ids.numel()),
            ROW_BLOCK=triton.next_power_of_2(rows),
        )
    _gather_hidden_kernel[(rows, triton.cdiv(width, COPY_BLOCK))](
        hidden, gather, flags, output, ROWS=rows, WIDTH=width, BLOCK=COPY_BLOCK
    )
    return output, counts, row_slots, inverse, flags
