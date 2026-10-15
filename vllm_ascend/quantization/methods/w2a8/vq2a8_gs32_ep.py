# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Linear placement of complete GS32 experts; matrix dimensions never shard."""

from dataclasses import dataclass

GS32_EP_SIZES = (1, 2, 4, 8)


@dataclass(frozen=True)
class GS32ExpertPlacement:
    ep_size: int
    ep_rank: int
    num_global_experts: int
    num_local_experts: int
    expert_start: int
    stored_positions: tuple[int, ...]
    global_ids: tuple[int, ...]
    local_ids: tuple[int, ...]


def gs32_expert_placement(expert_ids, num_experts, moe_config=None):
    """Map a global checkpoint inventory into one contiguous local EP range.

    ``num_experts`` is the local capacity supplied by RoutedExperts.create_weights.
    Sparse hash-layer inventories may contribute no resident tensors on a rank.
    The returned positions index the checkpoint stack, not the global expert IDs.
    """
    ep_size = getattr(moe_config, "ep_size", 1)
    ep_rank = getattr(moe_config, "ep_rank", 0)
    global_count = getattr(moe_config, "num_logical_experts", None)
    if global_count is None:
        global_count = getattr(moe_config, "num_experts", num_experts)
    local_count = getattr(moe_config, "num_local_experts", num_experts)
    if (
        any(type(value) is not int for value in (ep_size, ep_rank, global_count, local_count, num_experts))
        or ep_size not in GS32_EP_SIZES
        or not 0 <= ep_rank < ep_size
        or global_count <= 0
        or global_count % ep_size
        or local_count != global_count // ep_size
        or num_experts != local_count
        or getattr(moe_config, "num_experts", global_count) != global_count
    ):
        raise ValueError("GS32 requires consistent linear EP placement with 1, 2, 4 or 8 ranks and no redundancy.")
    ids = tuple(expert_ids)
    if (
        not ids
        or len(set(ids)) != len(ids)
        or any(type(expert) is not int or not 0 <= expert < global_count for expert in ids)
    ):
        raise ValueError("Invalid GS32 expert inventory.")
    start = ep_rank * local_count
    positions = tuple(index for index, expert in enumerate(ids) if start <= expert < start + local_count)
    global_ids = tuple(ids[index] for index in positions)
    return GS32ExpertPlacement(
        ep_size,
        ep_rank,
        global_count,
        local_count,
        start,
        positions,
        global_ids,
        tuple(expert - start for expert in global_ids),
    )
