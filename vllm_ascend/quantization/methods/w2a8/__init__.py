# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W2A8 quantization methods for Ascend NPU."""

from .vq2a8 import AscendVQ2A8FusedMoEScheme
from .vq2a8_gs32 import AscendGS32FusedMoEScheme

__all__ = ["AscendVQ2A8FusedMoEScheme", "AscendGS32FusedMoEScheme"]
