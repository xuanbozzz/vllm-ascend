# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W2A8 quantization methods for Ascend NPU."""

from .vq2a8 import AscendVQ2A8FusedMoEScheme

__all__ = ["AscendVQ2A8FusedMoEScheme"]
