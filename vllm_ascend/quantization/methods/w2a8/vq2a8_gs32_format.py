# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serialized ABI for resident GS32 staged projections (natural output rows)."""

from dataclasses import dataclass
from pathlib import Path

import torch

GS32_FORMAT = "ascend_vq2a8_gs32"
GS32_FIELDS = ("packed_zn", "pair_lut", "row_scale_bits", "perm", "weight_scale")
GS32_DTYPES = (torch.uint8, torch.uint8, torch.int64, torch.int64, torch.float32)
GS32_SAFE_DTYPES = ("U8", "U8", "I64", "I64", "F32")
GS32_MAX_ROWS = 128


@dataclass(frozen=True)
class GS32ProjectionSpec:
    rows: int
    columns: int

    def __post_init__(self):
        if (
            type(self.rows) is not int
            or type(self.columns) is not int
            or self.rows < 32
            or self.rows % 32
            or self.columns < 64
            or self.columns % 64
        ):
            raise ValueError("GS32 staged projection requires positive N32 and K64 alignment.")


def gs32_projection_shapes(spec):
    n, k = spec.rows, spec.columns
    return ((n // 32, k // 16, 16, 8), (n // 32, k // 32, 32), (n,), (k,), (k,))


def validate_gs32_options(config, options):
    required = {"gs32_library", "gs32_binary"}

    if (
        not required.issubset(options)
        or set(options) - required - {"gs32_execution"}
        or any(not isinstance(options[key], str) or not Path(options[key]).is_absolute() for key in required)
    ):
        raise ValueError("GS32 requires absolute gs32_library/gs32_binary paths and optional gs32_execution only.")
    execution = options.get("gs32_execution", "serial")
    if execution not in ("serial", "batched", "fused", "full_graph", "full_graph_prefill"):
        raise ValueError("GS32 gs32_execution must be serial, batched, fused, full_graph or full_graph_prefill.")
    graph = execution in ("full_graph", "full_graph_prefill")
    if graph and config.model_config.enforce_eager:
        raise ValueError("GS32 graph execution requires enforce_eager=False.")
    if not graph and not config.model_config.enforce_eager:
        raise ValueError("GS32 staged kernels require eager execution; graph capture has not been validated.")
