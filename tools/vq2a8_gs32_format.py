# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load the pure checkpoint ABI without importing the Ascend plugin."""

import runpy
from pathlib import Path

_contract = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "vllm_ascend/quantization/methods/w2a8/vq2a8_gs32_format.py")
)
GS32_FORMAT = _contract["GS32_FORMAT"]
GS32_FIELDS = _contract["GS32_FIELDS"]
GS32_DTYPES = _contract["GS32_DTYPES"]
GS32_SAFE_DTYPES = _contract["GS32_SAFE_DTYPES"]
GS32ProjectionSpec = _contract["GS32ProjectionSpec"]
gs32_projection_shapes = _contract["gs32_projection_shapes"]
del _contract
