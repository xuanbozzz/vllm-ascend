// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <cstdint>

namespace vq2a8_ascendc_v4_v2 {
constexpr uint32_t kProjectionValidityStatuses = 6;
constexpr uint32_t kProjectionValidityOutputs = 2;
constexpr uint32_t kProjectionValidityWidth = 4096;

// These arrays live on the host only. Each pointer becomes a kernel argument;
// no temporary host descriptor or device pointer table is transferred.
void LaunchProjectionValidity(void* stream, void* const* statuses, void* const* outputs, void* routeFlag, void* result,
                              uint32_t groups);
}  // namespace vq2a8_ascendc_v4_v2
