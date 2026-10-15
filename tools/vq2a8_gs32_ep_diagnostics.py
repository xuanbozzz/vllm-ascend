# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flushed diagnostics for the opt-in GS32 EP native acceptance test."""

from tools.vq2a8_probe_process import emit, stage_recorder


def run_reference_projection(
    ops,
    data,
    scales,
    banks,
    output,
    synchronize,
    *,
    case_name,
    case_index,
    slot,
    ep_size,
    capacity,
):
    """Separate host/tiling submission from device completion at test boundaries."""
    rows, width = data.shape
    columns = output.shape[1]
    case = (
        f"gs32_ep_native.ep{ep_size}.capacity{capacity}.reference{case_index}.{case_name}"
        f".slot{slot}.M{rows}.K{width}.N{columns}"
    )
    emit(case, "REFERENCE", case_index=case_index, case_name=case_name, slot=slot, m=rows, k=width, n=columns)
    stage = stage_recorder(case, synchronize)
    with stage("prepare"):
        plan = ops.prepare(data, columns)
    with stage("launch"):
        ops.project_prepared(data, scales, *banks, plan, output)
