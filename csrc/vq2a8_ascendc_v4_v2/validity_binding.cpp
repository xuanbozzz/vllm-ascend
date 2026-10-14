// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <ATen/ATen.h>
#include <torch/library.h>
#include <array>
#include <cstring>
#include <vector>
#include "acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "validity_launch.h"

namespace vq2a8_ascendc_v4_v2 {
namespace {
constexpr int64_t kMaximumGroups = 6;
constexpr uintptr_t kOutputAlignment = 32;

void CheckTensor(const at::Tensor& tensor, const c10::Device& device, at::ScalarType dtype) {
  TORCH_CHECK(tensor.defined() && tensor.device() == device && tensor.scalar_type() == dtype && tensor.is_contiguous(),
              "Projection validity requires matching device/dtype and contiguous tensors");
  const auto available = tensor.storage().nbytes() / tensor.element_size();
  TORCH_CHECK(tensor.storage_offset() >= 0 && static_cast<uint64_t>(tensor.storage_offset()) <= available &&
                  static_cast<uint64_t>(tensor.numel()) <= available - tensor.storage_offset(),
              "Projection validity tensor span exceeds its storage");
}
}  // namespace

at::Tensor ProjectionValidity(at::TensorList statusList, at::TensorList outputList, const at::Tensor& routeFlag) {
  TORCH_CHECK(statusList.size() == kProjectionValidityStatuses && outputList.size() == kProjectionValidityOutputs,
              "Projection validity requires six statuses and two projection outputs");
  TORCH_CHECK(statusList[0].defined() && statusList[0].device().type() == c10::DeviceType::PrivateUse1,
              "Projection validity requires NPU tensors");
  const auto device = statusList[0].device();
  TORCH_CHECK(statusList[0].dim() == 1 && statusList[0].size(0) >= 1 && statusList[0].size(0) <= kMaximumGroups,
              "Projection validity statuses require INT32[G], G 1..6");
  const uint32_t groups = statusList[0].size(0);
  for (const auto& status : statusList) {
    CheckTensor(status, device, at::kInt);
    TORCH_CHECK(status.dim() == 1 && status.size(0) == groups, "Projection validity status shapes must match");
  }
  for (const auto& output : outputList) {
    CheckTensor(output, device, at::kBFloat16);
    TORCH_CHECK((output.dim() == 2 || (output.dim() == 3 && output.size(1) == 1)) && output.size(0) == groups &&
                    output.size(-1) == kProjectionValidityWidth,
                "Projection validity outputs require BF16[G,4096] or BF16[G,1,4096]");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(output.data_ptr()) % kOutputAlignment == 0,
                "Projection validity outputs require 32-byte alignment");
  }
  CheckTensor(routeFlag, device, at::kBool);
  TORCH_CHECK(routeFlag.dim() == 0, "Projection validity route flag must be a BOOL scalar");
  const c10_npu::OptionalNPUGuard guard(device);
  const char* soc = aclrtGetSocName();
  TORCH_CHECK(soc && std::strncmp(soc, "Ascend950", 9) == 0, "Projection validity requires Ascend950");
  auto result = at::empty({}, statusList[0].options().dtype(at::kBool));
  // TensorList is borrowed. Capture owning tensors through queued execution.
  std::vector<at::Tensor> statuses(statusList.begin(), statusList.end());
  std::vector<at::Tensor> outputs(outputList.begin(), outputList.end());
  const auto stream = c10_npu::getCurrentNPUStream();
  for (const auto* tensors : {&statuses, &outputs}) {
    for (const auto& tensor : *tensors) c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
  }
  c10_npu::NPUCachingAllocator::recordStream(routeFlag.storage().data_ptr(), stream);
  // Resolve before RunOpApi: stream() can drain the task queue and must never
  // be called inside its own callback. Outputs remain owned by that callback.
  const auto launchStream = stream.stream();
  at_npu::native::OpCommand::RunOpApi(
      "Vq2a8V4V2ProjectionValidity",
      [launchStream, statuses, outputs, routeFlag, result, groups]() -> int {
        std::array<void*, kProjectionValidityStatuses> statusPointers{};
        std::array<void*, kProjectionValidityOutputs> outputPointers{};
        for (size_t i = 0; i < statuses.size(); ++i) statusPointers[i] = statuses[i].data_ptr();
        for (size_t i = 0; i < outputs.size(); ++i) outputPointers[i] = outputs[i].data_ptr();
        LaunchProjectionValidity(launchStream, statusPointers.data(), outputPointers.data(), routeFlag.data_ptr(),
                                 result.data_ptr(), groups);
        return 0;
      },
      false);
  return result;
}
}  // namespace vq2a8_ascendc_v4_v2

TORCH_LIBRARY_FRAGMENT(vq2a8_ascendc_v4_v2, m) {
  m.def("projection_validity_version() -> int", []() -> int64_t { return 1; });
  m.def("projection_validity(Tensor[] statuses, Tensor[] outputs, Tensor route_valid) -> Tensor");
}
TORCH_LIBRARY_IMPL(vq2a8_ascendc_v4_v2, PrivateUse1, m) {
  m.impl("projection_validity", &vq2a8_ascendc_v4_v2::ProjectionValidity);
}
