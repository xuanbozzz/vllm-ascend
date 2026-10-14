// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <cstdint>

namespace vq2a8_ascendc_v4_v2 {
void LaunchActivationSign(void* stream, uint32_t blocks, void* x, void* weightScale, void* weightBias,
                          void* signs, void* output, void* valid, uint32_t rows, uint32_t width);
void LaunchActivationSignStrided(void* stream, uint32_t blocks, void* x, void* weightScale, void* weightBias,
                                 void* signs, void* output, void* valid, uint32_t rows, uint32_t width,
                                 uint64_t rowStride, bool inputIsBf16);
}  // namespace vq2a8_ascendc_v4_v2
