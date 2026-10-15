// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Torch adapter for the DownW2 stage of mixed_w2w4_moe_ffn.
// The imported mixed kernel is intentionally not modified by this adapter.

#include <ATen/ATen.h>
#include <torch/custom_class.h>
#include <torch/library.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl_rt.h"
#if __has_include(<runtime/rt.h>)
// Preferred CANN 9.1 low-level runtime declarations. The target CANN 9.1
// installation must compile-verify this path before the adapter is accepted.
#include <runtime/rt.h>
#else
// Compatibility declaration source imported with the unmodified kernel.
// It is never copied or edited by this adapter.
#include "../mixed_w2w4_moe_ffn/hacl_rt.h"
#endif
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../mixed_w2w4_moe_ffn/components/down_w2/down_w2_tiling_builder.h"
#include "../mixed_w2w4_moe_ffn/mixed_tiling.h"

namespace vq2a8_gs32_stage {
namespace {

constexpr char kKernelName[] = "mixed_w2w4_moe_ffn";
constexpr uint64_t kGroupCount = 1;
constexpr uint64_t kMxScaleGroupSize = 32;
constexpr uint64_t kPackedK0 = 16;
constexpr uint64_t kPackedN0 = 32;
constexpr uint32_t kKernelCoreCount = 32;
constexpr uintptr_t kRequiredAlignment = 32;
constexpr int64_t kMaxProjectionRows = 128;
// Match the imported standalone Host's validated group-count range.
constexpr int64_t kMaxGroupedExperts = 256;
constexpr int64_t kRouteMaskWordBits = 32;

struct MatFP4Args {
  uint64_t group_count;
  uint64_t k;
  uint64_t n;
  uint64_t d;
  void* group_list;
  void* a;
  void* a_scale;
  void* gateup_w;
  void* gateup_table;
  void* down_w;
  void* down_table;
  void* scale_upgate;
  void* scale_down;
  void* is_expert_w4;
  void* is_down_w4;
  void* h;
  void* h_scale;
  void* c;
  void* tiling;
};

static_assert(sizeof(MatFP4Args) == 19 * sizeof(uint64_t), "unexpected mixed kernel argument ABI");
static_assert(kMixedTilingBytes == 1600, "unexpected mixed tiling ABI");
static_assert(kMixedStageDownW2 == 4, "unexpected DownW2 stage bit");

struct RegisteredKernel {
  std::vector<uint8_t> binary;
  std::string path;
  std::string kernel_name{kKernelName};
  void* binary_handle = nullptr;
  const void* stub = nullptr;
};

std::mutex g_registration_mutex;
std::shared_ptr<const RegisteredKernel> g_registered_kernel;

void CheckRt(rtError_t result, const char* operation) {
  TORCH_CHECK(result == RT_ERROR_NONE, operation, " failed with runtime error ", static_cast<int64_t>(result));
}

void CheckNotCapturing(const char* operation) {
  // The Python startup/device guard establishes the selected NPU context.
  // Use the exported query: the inline assertNotCapturing helper references
  // IsContextInitialized, which is not exported by some torch_npu builds.
  // Accept only None, so Active and Invalidated capture states both fail.
  const auto status = c10_npu::currentStreamCaptureStatusMayInitCtx();
  TORCH_CHECK(status == c10_npu::CaptureStatus::None,
              operation, " is eager-only and cannot run during NPU graph capture; status=",
              static_cast<int>(status));
}

void CheckCaptureAllowed(const char* operation) {
  const auto status = c10_npu::currentStreamCaptureStatusMayInitCtx();
  TORCH_CHECK(status == c10_npu::CaptureStatus::None || status == c10_npu::CaptureStatus::Active,
              operation, " cannot run on an invalidated NPU graph capture; status=", static_cast<int>(status));
}

std::vector<uint8_t> ReadElfBinary(const std::string& path) {
  TORCH_CHECK(!path.empty(), "vq2a8_gs32_stage.initialize requires a non-empty binary path");
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  TORCH_CHECK(stream.good(), "cannot open mixed kernel binary: ", path);
  const std::streamoff size = stream.tellg();
  TORCH_CHECK(size >= 4, "mixed kernel binary is empty or truncated: ", path);
  TORCH_CHECK(static_cast<uint64_t>(size) <= std::numeric_limits<size_t>::max(),
              "mixed kernel binary is too large for this process: ", path);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  stream.seekg(0, std::ios::beg);
  stream.read(reinterpret_cast<char*>(bytes.data()), size);
  TORCH_CHECK(stream.good(), "failed to read mixed kernel binary: ", path);
  TORCH_CHECK(bytes[0] == 0x7f && bytes[1] == 'E' && bytes[2] == 'L' && bytes[3] == 'F',
              "mixed kernel binary must be an ELF object: ", path);
  return bytes;
}

void Initialize(const std::string& binary_path) {
  CheckNotCapturing("vq2a8_gs32_stage.initialize");
  std::lock_guard<std::mutex> lock(g_registration_mutex);
  if (g_registered_kernel != nullptr) {
    TORCH_CHECK(g_registered_kernel->path == binary_path,
                "vq2a8_gs32_stage is already initialized with a different binary; start a new process");
    return;
  }

  const char* soc = aclrtGetSocName();
  TORCH_CHECK(soc != nullptr && std::strncmp(soc, "Ascend950", 9) == 0,
              "vq2a8_gs32_stage requires an initialized Ascend950 device");

  auto state = std::make_shared<RegisteredKernel>();
  state->path = binary_path;
  state->binary = ReadElfBinary(binary_path);
  rtDevBinary_t descriptor{};
  descriptor.magic = RT_DEV_BINARY_MAGIC_ELF;
  descriptor.version = 0;
  descriptor.data = state->binary.data();
  descriptor.length = state->binary.size();
  CheckRt(rtDevBinaryRegister(&descriptor, &state->binary_handle), "rtDevBinaryRegister");
  TORCH_CHECK(state->binary_handle != nullptr, "rtDevBinaryRegister returned a null binary handle");

  state->stub = reinterpret_cast<const void*>(state->kernel_name.c_str());
  CheckRt(rtFunctionRegister(state->binary_handle, state->stub, state->kernel_name.c_str(),
                             const_cast<void*>(state->stub), 0),
          "rtFunctionRegister");
  // The ELF byte vector and its function-name storage remain alive for the
  // process lifetime. There is intentionally no reset/unregister hot path.
  g_registered_kernel = std::move(state);
}

std::shared_ptr<const RegisteredKernel> RegisteredKernelSnapshot() {
  std::lock_guard<std::mutex> lock(g_registration_mutex);
  TORCH_CHECK(g_registered_kernel != nullptr,
              "vq2a8_gs32_stage.initialize(binary) must be called once during startup before project");
  return g_registered_kernel;
}

at::Tensor MakeTiling(int64_t k, int64_t n) {
  TORCH_CHECK(k > 0 && k % 64 == 0, "DownW2 K must be a positive multiple of 64");
  TORCH_CHECK(n > 0 && n % 32 == 0, "DownW2 N must be a positive multiple of 32");
  const auto blob = BuildDownW2TilingBlob(static_cast<uint64_t>(k), static_cast<uint64_t>(n));
  TORCH_CHECK(blob.size() == kMixedDownW2TilingBytes, "DownW2 tiling builder returned an unexpected ABI size");

  auto result = at::zeros({static_cast<int64_t>(kMixedTilingBytes)},
                          at::TensorOptions().device(at::kCPU).dtype(at::kByte));
  MixedTilingHeader header{
      kMixedTilingMagic,       kMixedTilingVersion,      kMixedGateW2TilingOffset,
      kMixedGateW4TilingOffset, kMixedDownW2TilingOffset, kMixedDownW4TilingOffset,
      kMixedTilingBytes,       kMixedStageDownW2,
  };
  auto* bytes = result.data_ptr<uint8_t>();
  std::memcpy(bytes, &header, sizeof(header));
  std::memcpy(bytes + kMixedDownW2TilingOffset, blob.data(), blob.size());
  return result;
}

void CheckNpuTensor(const at::Tensor& tensor, const at::Tensor& anchor, at::ScalarType dtype, int64_t rank,
                    const char* name) {
  TORCH_CHECK(tensor.defined(), name, ": undefined tensor");
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, ": an NPU tensor is required");
  TORCH_CHECK(tensor.device() == anchor.device(), name, ": all tensors must share one NPU device");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, ": incorrect dtype");
  TORCH_CHECK(tensor.dim() == rank, name, ": incorrect rank");
  TORCH_CHECK(tensor.is_contiguous(), name, ": contiguous storage is required");
  // Unified parameters contain all experts. An aligned contiguous slice may
  // start inside the allocation; the queued callback retains its base storage.
  const uint64_t offset = static_cast<uint64_t>(tensor.storage_offset()) * tensor.element_size();
  const uint64_t capacity = tensor.storage().nbytes();
  TORCH_CHECK(offset <= capacity && tensor.nbytes() <= capacity - offset,
              name, ": tensor view exceeds its base storage");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % kRequiredAlignment == 0,
              name, ": storage must be 32-byte aligned");
}

void CheckCpuMetadata(const at::Tensor& tensor, at::ScalarType dtype, int64_t rank, const char* name) {
  TORCH_CHECK(tensor.defined(), name, ": undefined tensor");
  TORCH_CHECK(tensor.device().is_cpu(), name, ": validated CPU metadata is required");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, ": incorrect dtype");
  TORCH_CHECK(tensor.dim() == rank, name, ": incorrect rank");
  TORCH_CHECK(tensor.is_contiguous(), name, ": contiguous storage is required");
  TORCH_CHECK(tensor.storage_offset() == 0, name, ": storage views/offsets are not accepted");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % kRequiredAlignment == 0,
              name, ": storage must be 32-byte aligned");
}

struct NamedRange {
  const char* name;
  uintptr_t begin;
  uintptr_t end;
};

NamedRange TensorRange(const char* name, const at::Tensor& tensor) {
  const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  const size_t bytes = tensor.nbytes();
  TORCH_CHECK(bytes > 0 && bytes <= std::numeric_limits<uintptr_t>::max() - begin,
              name, ": invalid storage byte range");
  return {name, begin, begin + bytes};
}

void CheckNoAlias(const std::vector<std::pair<const char*, at::Tensor>>& tensors) {
  std::vector<NamedRange> ranges;
  ranges.reserve(tensors.size());
  for (const auto& named : tensors) {
    ranges.push_back(TensorRange(named.first, named.second));
  }
  for (size_t left = 0; left < ranges.size(); ++left) {
    for (size_t right = left + 1; right < ranges.size(); ++right) {
      TORCH_CHECK(ranges[left].end <= ranges[right].begin || ranges[right].end <= ranges[left].begin,
                  ranges[left].name, " and ", ranges[right].name,
                  " must not alias or overlap storage");
    }
  }
}

void RecordStream(const std::vector<at::Tensor>& tensors, c10_npu::NPUStream stream) {
  for (const auto& tensor : tensors) {
    c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
  }
}

void ValidateProjectionShape(int64_t m, int64_t k, int64_t n) {
  TORCH_CHECK(m > 0 && m <= kMaxProjectionRows,
              "DownW2 projection requires 1<=M<=128 for the initial single-expert adapter");
  TORCH_CHECK(k > 0 && k % 64 == 0, "input K must be a positive multiple of 64");
  TORCH_CHECK(n > 0 && n % 32 == 0, "output N must be a positive multiple of 32");
  TORCH_CHECK(k <= std::numeric_limits<int64_t>::max() / 2, "2*K overflows the mixed kernel ABI");
}

void CheckProjectionTensors(const at::Tensor& input_bytes, const at::Tensor& input_scale,
                            const at::Tensor& packed_weight, const at::Tensor& pair_lut,
                            const at::Tensor& row_scale, const at::Tensor& output) {
  CheckNpuTensor(input_bytes, input_bytes, at::kByte, 2, "input_bytes");
  CheckNpuTensor(input_scale, input_bytes, at::kByte, 2, "input_scale");
  CheckNpuTensor(packed_weight, input_bytes, at::kByte, 5, "packed_weight");
  CheckNpuTensor(pair_lut, input_bytes, at::kByte, 4, "pair_lut");
  CheckNpuTensor(row_scale, input_bytes, at::kLong, 2, "row_scale");
  CheckNpuTensor(output, input_bytes, at::kBFloat16, 2, "output");
  const int64_t m = input_bytes.size(0);
  const int64_t k = input_bytes.size(1);
  const int64_t n = output.size(1);
  ValidateProjectionShape(m, k, n);
  TORCH_CHECK(input_scale.size(0) == m && input_scale.size(1) == k / kMxScaleGroupSize,
              "input_scale must have shape [M,K/32]");
  TORCH_CHECK(packed_weight.size(0) == static_cast<int64_t>(kGroupCount) &&
                  packed_weight.size(1) == n / kPackedN0 && packed_weight.size(2) == k / kPackedK0 &&
                  packed_weight.size(3) == kPackedK0 && packed_weight.size(4) == kPackedN0 / 4,
              "packed_weight must have shape [1,N/32,K/16,16,8]");
  TORCH_CHECK(pair_lut.size(0) == static_cast<int64_t>(kGroupCount) && pair_lut.size(1) == n / kPackedN0 &&
                  pair_lut.size(2) == k / kMxScaleGroupSize && pair_lut.size(3) == 32,
              "pair_lut must have shape [1,N/32,K/32,32]");
  TORCH_CHECK(row_scale.size(0) == static_cast<int64_t>(kGroupCount) && row_scale.size(1) == n,
              "row_scale must have shape [1,N] with FP32 bits in each INT64 low word");
  TORCH_CHECK(output.size(0) == m, "output must have shape [M,N]");
  CheckNoAlias({{"input_bytes", input_bytes}, {"input_scale", input_scale}, {"packed_weight", packed_weight},
                {"pair_lut", pair_lut}, {"row_scale", row_scale}, {"output", output}});
}

// Python owns this handle in the model instance. No constructor, Tensor
// accessor, pickle method or mutator is exported to Python: the validated
// metadata can neither be replaced nor changed between queued launches.
class ProjectionPlan : public torch::CustomClassHolder {
 public:
  ProjectionPlan(int64_t m, int64_t k, int64_t n, c10::Device device,
                 at::Tensor group_list, at::Tensor route_mask, at::Tensor tiling,
                 std::shared_ptr<const RegisteredKernel> registered)
      : m_(m), k_(k), n_(n), device_(device), group_list_(std::move(group_list)),
        route_mask_(std::move(route_mask)), tiling_(std::move(tiling)), registered_(std::move(registered)) {}

  void CheckMatches(const at::Tensor& input_bytes, const at::Tensor& output) const {
    TORCH_CHECK(input_bytes.device() == device_, "prepared projection belongs to a different NPU device");
    TORCH_CHECK(input_bytes.size(0) == m_ && input_bytes.size(1) == k_ && output.size(1) == n_,
                "prepared projection shape does not match [M,K,N]");
  }

  const at::Tensor& GroupList() const { return group_list_; }
  const at::Tensor& RouteMask() const { return route_mask_; }
  const at::Tensor& Tiling() const { return tiling_; }
  const RegisteredKernel& Kernel() const { return *registered_; }
  int64_t K() const { return k_; }
  int64_t N() const { return n_; }

 private:
  const int64_t m_;
  const int64_t k_;
  const int64_t n_;
  const c10::Device device_;
  const at::Tensor group_list_;
  const at::Tensor route_mask_;
  const at::Tensor tiling_;
  const std::shared_ptr<const RegisteredKernel> registered_;
};

c10::intrusive_ptr<ProjectionPlan> Prepare(const at::Tensor& input_bytes, int64_t n) {
  TORCH_CHECK(input_bytes.defined() && input_bytes.device().type() == c10::DeviceType::PrivateUse1,
              "input_bytes must be an NPU tensor");
  const c10_npu::OptionalNPUGuard guard(input_bytes.device());
  CheckNotCapturing("vq2a8_gs32_stage.prepare");
  CheckNpuTensor(input_bytes, input_bytes, at::kByte, 2, "input_bytes");
  const int64_t m = input_bytes.size(0);
  const int64_t k = input_bytes.size(1);
  ValidateProjectionShape(m, k, n);
  const auto registered = RegisteredKernelSnapshot();
  const auto group_list = at::full({1}, m, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  const auto route_mask = at::zeros({1}, at::TensorOptions().device(at::kCPU).dtype(at::kInt));
  const auto tiling = MakeTiling(k, n);
  // Blocking copies finish before the handle becomes visible. This allows
  // subsequent launches to use any current stream without an upload race.
  auto device_group_list = group_list.to(input_bytes.device(), at::kLong, false, true);
  auto device_route_mask = route_mask.to(input_bytes.device(), at::kInt, false, true);
  auto device_tiling = tiling.to(input_bytes.device(), at::kByte, false, true);
  return c10::make_intrusive<ProjectionPlan>(m, k, n, input_bytes.device(), std::move(device_group_list),
                                            std::move(device_route_mask), std::move(device_tiling), registered);
}

void ProjectPrepared(const at::Tensor& input_bytes, const at::Tensor& input_scale,
                     const at::Tensor& packed_weight, const at::Tensor& pair_lut, const at::Tensor& row_scale,
                     const c10::intrusive_ptr<ProjectionPlan>& plan, at::Tensor& output) {
  TORCH_CHECK(plan != nullptr, "prepared projection plan is required");
  TORCH_CHECK(input_bytes.defined() && input_bytes.device().type() == c10::DeviceType::PrivateUse1,
              "input_bytes must be an NPU tensor");
  const c10_npu::OptionalNPUGuard guard(input_bytes.device());
  CheckNotCapturing("vq2a8_gs32_stage.project_prepared");
  CheckProjectionTensors(input_bytes, input_scale, packed_weight, pair_lut, row_scale, output);
  plan->CheckMatches(input_bytes, output);
  const auto stream = c10_npu::getCurrentNPUStream();
  RecordStream({input_bytes, input_scale, packed_weight, pair_lut, row_scale, output,
                plan->GroupList(), plan->RouteMask(), plan->Tiling()}, stream);
  // Reading the handle must not drain the torch_npu task queue. RunOpApi
  // enqueues this launch after earlier producer operations on that queue.
  const auto launch_stream = stream.stream(false);
  at_npu::native::OpCommand::RunOpApi(
      "Vq2a8Gs32StageProjectPrepared",
      [plan, launch_stream, input_bytes, input_scale, packed_weight, pair_lut, row_scale, output]() -> int {
        MatFP4Args arguments{
            kGroupCount, static_cast<uint64_t>(plan->K()), static_cast<uint64_t>(2 * plan->K()),
            static_cast<uint64_t>(plan->N()), plan->GroupList().data_ptr(), input_bytes.data_ptr(),
            input_scale.data_ptr(), packed_weight.data_ptr(), pair_lut.data_ptr(), packed_weight.data_ptr(),
            pair_lut.data_ptr(), row_scale.data_ptr(), row_scale.data_ptr(), plan->RouteMask().data_ptr(),
            plan->RouteMask().data_ptr(), input_bytes.data_ptr(), input_scale.data_ptr(), output.data_ptr(),
            plan->Tiling().data_ptr(),
        };
        return static_cast<int>(rtKernelLaunch(plan->Kernel().stub, kKernelCoreCount, &arguments,
                                               sizeof(arguments), nullptr, launch_stream));
      },
      false);
}

void ValidateGroupCount(int64_t groups) {
  TORCH_CHECK(groups > 0 && groups <= kMaxGroupedExperts,
              "DownW2 grouped projection requires 1..256 resident experts");
}

void ValidateGroupCounts(const at::Tensor& counts, int64_t rows) {
  CheckCpuMetadata(counts, at::kLong, 1, "group_counts");
  ValidateGroupCount(counts.numel());
  TORCH_CHECK(rows > 0, "grouped projection must contain at least one routed row");
  int64_t total = 0;
  for (int64_t group = 0; group < counts.numel(); ++group) {
    const int64_t count = counts.data_ptr<int64_t>()[group];
    TORCH_CHECK(count >= 0 && count <= kMaxProjectionRows,
                "group_counts requires each expert row count in [0,128]");
    // Bound each addition before performing it, including adversarial counts.
    TORCH_CHECK(count <= rows - total, "group_counts exceeds routed input rows");
    total += count;
  }
  TORCH_CHECK(total == rows, "group_counts sum does not equal routed input rows");
}

class GroupedRoutes : public torch::CustomClassHolder {
 public:
  GroupedRoutes(int64_t rows, int64_t groups, c10::Device device, at::Tensor group_list)
      : rows_(rows), groups_(groups), device_(device), group_list_(std::move(group_list)) {}

  void CheckMatches(const at::Tensor& input_bytes, int64_t groups) const {
    TORCH_CHECK(input_bytes.device() == device_, "grouped routes belong to a different NPU device");
    TORCH_CHECK(input_bytes.size(0) == rows_ && groups == groups_,
                "grouped routes do not match routed row count or resident expert count");
  }

  const at::Tensor& GroupList() const { return group_list_; }

 private:
  const int64_t rows_;
  const int64_t groups_;
  const c10::Device device_;
  const at::Tensor group_list_;
};

c10::intrusive_ptr<GroupedRoutes> PrepareRoutes(const at::Tensor& anchor, const at::Tensor& counts) {
  TORCH_CHECK(anchor.defined() && anchor.device().type() == c10::DeviceType::PrivateUse1 && anchor.dim() == 2,
              "routes require an NPU [routed_rows,K] anchor");
  const c10_npu::OptionalNPUGuard guard(anchor.device());
  CheckNotCapturing("vq2a8_gs32_stage.prepare_routes");
  CheckCpuMetadata(counts, at::kLong, 1, "group_counts");
  const auto owned_counts = counts.clone();
  ValidateGroupCounts(owned_counts, anchor.size(0));
  // A route belongs to one invocation, shared by its GateUp and Down stages.
  // Blocking, forced-copy upload prevents either host mutation or another
  // stream from racing the immutable device counts.
  auto device_counts = owned_counts.to(anchor.device(), at::kLong, false, true);
  return c10::make_intrusive<GroupedRoutes>(anchor.size(0), owned_counts.numel(), anchor.device(),
                                           std::move(device_counts));
}

std::pair<at::Tensor, at::Tensor> SanitizeDeviceCounts(const at::Tensor& counts, int64_t rows,
                                                       bool allow_padding = false) {
  // Clone before validation: no caller-accessible alias of the kernel's counts
  // survives. Every operation below is device-side and recorded by capture.
  const auto owned_counts = counts.clone();
  const auto bounded = owned_counts.clamp(0, kMaxProjectionRows);
  // Sum the bounded values, never raw INT64 inputs which can overflow into a
  // seemingly valid total. At most 256 * 128 rows can reach this reduction.
  const auto total = bounded.sum();
  // EP uses fixed-capacity buffers while each rank may receive fewer rows,
  // including none. Keep strict equality for existing single-rank callers.
  const auto fits_rows = allow_padding ? total.le(rows) : total.eq(rows);
  const auto valid = owned_counts.eq(bounded).all().logical_and(fits_rows);
  auto safe_counts = at::where(valid, bounded, at::zeros_like(bounded));
  auto status = valid.to(at::kInt).reshape({1});
  return {std::move(safe_counts), std::move(status)};
}

// Explicit graph API. Unlike eager GroupedRoutes, its data is produced on the
// current stream, and is recalculated inside each replay. A different stream
// cannot use the handle without a dependency, so reject it instead of guessing.
// No constructor or metadata accessor is exposed to Python.
class DeviceRoutes : public torch::CustomClassHolder {
 public:
  DeviceRoutes(int64_t rows, int64_t groups, c10::Device device, at::Tensor group_list,
               at::Tensor status, void* stream, bool allow_padding = false)
      : rows_(rows), groups_(groups), device_(device), group_list_(std::move(group_list)),
        status_(std::move(status)), stream_(stream), allow_padding_(allow_padding) {}

  void CheckStream() const {
    TORCH_CHECK(c10_npu::getCurrentNPUStream().stream(false) == stream_,
                "device routes must be consumed on their producing NPU stream");
  }

  void CheckMatches(const at::Tensor& input_bytes, int64_t groups) const {
    TORCH_CHECK(input_bytes.device() == device_, "device routes belong to a different NPU device");
    TORCH_CHECK(input_bytes.size(0) == rows_ && groups == groups_,
                "device routes do not match routed row count or resident expert count");
    CheckStream();
  }

  c10::Device Device() const { return device_; }
  const at::Tensor& GroupList() const { return group_list_; }
  const at::Tensor& Status() const { return status_; }
  bool AllowsPadding() const { return allow_padding_; }

 private:
  const int64_t rows_;
  const int64_t groups_;
  const c10::Device device_;
  const at::Tensor group_list_;
  const at::Tensor status_;
  void* const stream_;
  const bool allow_padding_;
};

c10::intrusive_ptr<DeviceRoutes> PrepareDeviceRoutes(const at::Tensor& anchor, const at::Tensor& counts,
                                                     bool allow_padding) {
  TORCH_CHECK(anchor.defined() && anchor.device().type() == c10::DeviceType::PrivateUse1 && anchor.dim() == 2,
              "device routes require an NPU [routed_rows,K] anchor");
  const c10_npu::OptionalNPUGuard guard(anchor.device());
  CheckCaptureAllowed("vq2a8_gs32_stage.prepare_device_routes");
  CheckNpuTensor(counts, anchor, at::kLong, 1, "group_counts");
  ValidateGroupCount(counts.numel());
  TORCH_CHECK(anchor.size(0) > 0 && anchor.size(0) <= kMaxGroupedExperts * kMaxProjectionRows,
              "device routes require 1..32768 routed rows");
  auto sanitized = SanitizeDeviceCounts(counts, anchor.size(0), allow_padding);
  const auto stream = c10_npu::getCurrentNPUStream();
  RecordStream({counts, sanitized.first, sanitized.second}, stream);
  return c10::make_intrusive<DeviceRoutes>(anchor.size(0), counts.numel(), anchor.device(),
                                          std::move(sanitized.first), std::move(sanitized.second), stream.stream(false),
                                          allow_padding);
}

at::Tensor DeviceRoutesStatus(const c10::intrusive_ptr<DeviceRoutes>& routes) {
  TORCH_CHECK(routes != nullptr, "device routes are required");
  const c10_npu::OptionalNPUGuard guard(routes->Device());
  CheckCaptureAllowed("vq2a8_gs32_stage.device_routes_status");
  routes->CheckStream();
  // 1 means valid; 0 means every native group was zeroed. The caller must
  // include this result in its graph-wide validity check before using output.
  // Return an owned copy so Python mutation cannot change the internal result.
  return routes->Status().clone();
}

class GroupedProjectionPlan : public torch::CustomClassHolder {
 public:
  GroupedProjectionPlan(int64_t groups, int64_t k, int64_t n, c10::Device device,
                        at::Tensor route_mask, at::Tensor tiling,
                        std::shared_ptr<const RegisteredKernel> registered)
      : groups_(groups), k_(k), n_(n), device_(device), route_mask_(std::move(route_mask)),
        tiling_(std::move(tiling)), registered_(std::move(registered)) {}

  void CheckMatches(const at::Tensor& input_bytes, const at::Tensor& output) const {
    TORCH_CHECK(input_bytes.device() == device_, "grouped plan belongs to a different NPU device");
    TORCH_CHECK(input_bytes.size(1) == k_ && output.size(1) == n_,
                "grouped plan shape does not match [K,N]");
  }

  int64_t Groups() const { return groups_; }
  int64_t K() const { return k_; }
  int64_t N() const { return n_; }
  const at::Tensor& RouteMask() const { return route_mask_; }
  const at::Tensor& Tiling() const { return tiling_; }
  const RegisteredKernel& Kernel() const { return *registered_; }

 private:
  const int64_t groups_;
  const int64_t k_;
  const int64_t n_;
  const c10::Device device_;
  const at::Tensor route_mask_;
  const at::Tensor tiling_;
  const std::shared_ptr<const RegisteredKernel> registered_;
};

c10::intrusive_ptr<GroupedProjectionPlan> PrepareGrouped(const at::Tensor& input_bytes, int64_t n,
                                                        int64_t groups) {
  TORCH_CHECK(input_bytes.defined() && input_bytes.device().type() == c10::DeviceType::PrivateUse1,
              "input_bytes must be an NPU tensor");
  const c10_npu::OptionalNPUGuard guard(input_bytes.device());
  CheckNotCapturing("vq2a8_gs32_stage.prepare_grouped");
  CheckNpuTensor(input_bytes, input_bytes, at::kByte, 2, "input_bytes");
  ValidateGroupCount(groups);
  const int64_t k = input_bytes.size(1);
  ValidateProjectionShape(1, k, n);
  const auto registered = RegisteredKernelSnapshot();
  const int64_t mask_words = (groups + kRouteMaskWordBits - 1) / kRouteMaskWordBits;
  const auto route_mask = at::zeros({mask_words}, at::TensorOptions().device(at::kCPU).dtype(at::kInt));
  const auto tiling = MakeTiling(k, n);
  auto device_route_mask = route_mask.to(input_bytes.device(), at::kInt, false, true);
  auto device_tiling = tiling.to(input_bytes.device(), at::kByte, false, true);
  return c10::make_intrusive<GroupedProjectionPlan>(groups, k, n, input_bytes.device(),
                                                   std::move(device_route_mask), std::move(device_tiling), registered);
}

void CheckGroupedProjectionTensors(const at::Tensor& input_bytes, const at::Tensor& input_scale,
                                   const at::Tensor& packed_weight, const at::Tensor& pair_lut,
                                   const at::Tensor& row_scale, const at::Tensor& output, int64_t groups) {
  CheckNpuTensor(input_bytes, input_bytes, at::kByte, 2, "input_bytes");
  CheckNpuTensor(input_scale, input_bytes, at::kByte, 2, "input_scale");
  CheckNpuTensor(packed_weight, input_bytes, at::kByte, 5, "packed_weight");
  CheckNpuTensor(pair_lut, input_bytes, at::kByte, 4, "pair_lut");
  CheckNpuTensor(row_scale, input_bytes, at::kLong, 2, "row_scale");
  CheckNpuTensor(output, input_bytes, at::kBFloat16, 2, "output");
  const int64_t m = input_bytes.size(0);
  const int64_t k = input_bytes.size(1);
  const int64_t n = output.size(1);
  ValidateProjectionShape(1, k, n);
  TORCH_CHECK(input_scale.size(0) == m && input_scale.size(1) == k / kMxScaleGroupSize,
              "grouped input_scale must have shape [sum(M_i),K/32]");
  TORCH_CHECK(packed_weight.size(0) == groups && packed_weight.size(1) == n / kPackedN0 &&
                  packed_weight.size(2) == k / kPackedK0 && packed_weight.size(3) == kPackedK0 &&
                  packed_weight.size(4) == kPackedN0 / 4,
              "grouped packed_weight must have shape [G,N/32,K/16,16,8]");
  TORCH_CHECK(pair_lut.size(0) == groups && pair_lut.size(1) == n / kPackedN0 &&
                  pair_lut.size(2) == k / kMxScaleGroupSize && pair_lut.size(3) == 32,
              "grouped pair_lut must have shape [G,N/32,K/32,32]");
  TORCH_CHECK(row_scale.size(0) == groups && row_scale.size(1) == n,
              "grouped row_scale must have shape [G,N]");
  TORCH_CHECK(output.size(0) == m, "grouped output must have shape [sum(M_i),N]");
  CheckNoAlias({{"input_bytes", input_bytes}, {"input_scale", input_scale}, {"packed_weight", packed_weight},
                {"pair_lut", pair_lut}, {"row_scale", row_scale}, {"output", output}});
}

void ProjectGrouped(const at::Tensor& input_bytes, const at::Tensor& input_scale,
                    const at::Tensor& packed_weight, const at::Tensor& pair_lut, const at::Tensor& row_scale,
                    const c10::intrusive_ptr<GroupedRoutes>& routes,
                    const c10::intrusive_ptr<GroupedProjectionPlan>& plan, at::Tensor& output) {
  TORCH_CHECK(routes != nullptr && plan != nullptr, "grouped routes and projection plan are required");
  TORCH_CHECK(input_bytes.defined() && input_bytes.device().type() == c10::DeviceType::PrivateUse1,
              "input_bytes must be an NPU tensor");
  const c10_npu::OptionalNPUGuard guard(input_bytes.device());
  CheckNotCapturing("vq2a8_gs32_stage.project_grouped");
  CheckGroupedProjectionTensors(input_bytes, input_scale, packed_weight, pair_lut, row_scale, output, plan->Groups());
  routes->CheckMatches(input_bytes, plan->Groups());
  plan->CheckMatches(input_bytes, output);
  const auto stream = c10_npu::getCurrentNPUStream();
  RecordStream({input_bytes, input_scale, packed_weight, pair_lut, row_scale, output,
                routes->GroupList(), plan->RouteMask(), plan->Tiling()}, stream);
  const auto launch_stream = stream.stream(false);
  at_npu::native::OpCommand::RunOpApi(
      "Vq2a8Gs32StageProjectGrouped",
      [routes, plan, launch_stream, input_bytes, input_scale, packed_weight, pair_lut, row_scale, output]() -> int {
        MatFP4Args arguments{
            static_cast<uint64_t>(plan->Groups()), static_cast<uint64_t>(plan->K()),
            static_cast<uint64_t>(2 * plan->K()), static_cast<uint64_t>(plan->N()),
            routes->GroupList().data_ptr(), input_bytes.data_ptr(), input_scale.data_ptr(),
            packed_weight.data_ptr(), pair_lut.data_ptr(), packed_weight.data_ptr(), pair_lut.data_ptr(),
            row_scale.data_ptr(), row_scale.data_ptr(), plan->RouteMask().data_ptr(), plan->RouteMask().data_ptr(),
            input_bytes.data_ptr(), input_scale.data_ptr(), output.data_ptr(), plan->Tiling().data_ptr(),
        };
        return static_cast<int>(rtKernelLaunch(plan->Kernel().stub, kKernelCoreCount, &arguments,
                                               sizeof(arguments), nullptr, launch_stream));
      },
      false);
}

void ProjectGroupedDevice(const at::Tensor& input_bytes, const at::Tensor& input_scale,
                          const at::Tensor& packed_weight, const at::Tensor& pair_lut, const at::Tensor& row_scale,
                          const c10::intrusive_ptr<DeviceRoutes>& routes,
                          const c10::intrusive_ptr<GroupedProjectionPlan>& plan, at::Tensor& output) {
  TORCH_CHECK(routes != nullptr && plan != nullptr, "device routes and grouped projection plan are required");
  TORCH_CHECK(input_bytes.defined() && input_bytes.device().type() == c10::DeviceType::PrivateUse1,
              "input_bytes must be an NPU tensor");
  const c10_npu::OptionalNPUGuard guard(input_bytes.device());
  CheckCaptureAllowed("vq2a8_gs32_stage.project_grouped_device");
  CheckGroupedProjectionTensors(input_bytes, input_scale, packed_weight, pair_lut, row_scale, output, plan->Groups());
  routes->CheckMatches(input_bytes, plan->Groups());
  plan->CheckMatches(input_bytes, output);
  if (routes->AllowsPadding()) {
    // The native kernel writes only sum(M_i) leading rows. Reset all capacity
    // on the same stream, inside capture, so empty/short routes cannot reuse a
    // previous replay's output. Invalid padded routes likewise produce zeros.
    output.zero_();
  }
  const auto stream = c10_npu::getCurrentNPUStream();
  RecordStream({input_bytes, input_scale, packed_weight, pair_lut, row_scale, output,
                routes->GroupList(), routes->Status(), plan->RouteMask(), plan->Tiling()}, stream);
  const auto launch_stream = stream.stream(false);
  at_npu::native::OpCommand::RunOpApi(
      "Vq2a8Gs32StageProjectGroupedDevice",
      [routes, plan, launch_stream, input_bytes, input_scale, packed_weight, pair_lut, row_scale, output]() -> int {
        MatFP4Args arguments{
            static_cast<uint64_t>(plan->Groups()), static_cast<uint64_t>(plan->K()),
            static_cast<uint64_t>(2 * plan->K()), static_cast<uint64_t>(plan->N()),
            routes->GroupList().data_ptr(), input_bytes.data_ptr(), input_scale.data_ptr(),
            packed_weight.data_ptr(), pair_lut.data_ptr(), packed_weight.data_ptr(), pair_lut.data_ptr(),
            row_scale.data_ptr(), row_scale.data_ptr(), plan->RouteMask().data_ptr(), plan->RouteMask().data_ptr(),
            input_bytes.data_ptr(), input_scale.data_ptr(), output.data_ptr(), plan->Tiling().data_ptr(),
        };
        return static_cast<int>(rtKernelLaunch(plan->Kernel().stub, kKernelCoreCount, &arguments,
                                               sizeof(arguments), nullptr, launch_stream));
      },
      false);
}

void Project(const at::Tensor& input_bytes, const at::Tensor& input_scale, const at::Tensor& packed_weight,
             const at::Tensor& pair_lut, const at::Tensor& row_scale, const at::Tensor& group_list,
             const at::Tensor& route_mask, const at::Tensor& tiling, at::Tensor& output) {
  TORCH_CHECK(input_bytes.defined() && input_bytes.device().type() == c10::DeviceType::PrivateUse1,
              "input_bytes must be an NPU tensor");
  const c10_npu::OptionalNPUGuard guard(input_bytes.device());
  CheckNotCapturing("vq2a8_gs32_stage.project");

  CheckNpuTensor(input_bytes, input_bytes, at::kByte, 2, "input_bytes");
  CheckNpuTensor(input_scale, input_bytes, at::kByte, 2, "input_scale");
  CheckNpuTensor(packed_weight, input_bytes, at::kByte, 5, "packed_weight");
  CheckNpuTensor(pair_lut, input_bytes, at::kByte, 4, "pair_lut");
  CheckNpuTensor(row_scale, input_bytes, at::kLong, 2, "row_scale");
  CheckCpuMetadata(group_list, at::kLong, 1, "group_list");
  CheckCpuMetadata(route_mask, at::kInt, 1, "route_mask");
  CheckCpuMetadata(tiling, at::kByte, 1, "tiling");
  CheckNpuTensor(output, input_bytes, at::kBFloat16, 2, "output");

  const int64_t m = input_bytes.size(0);
  const int64_t k = input_bytes.size(1);
  const int64_t n = output.size(1);
  TORCH_CHECK(m > 0 && m <= 128, "DownW2 projection requires 1<=M<=128 for the initial single-expert adapter");
  TORCH_CHECK(k > 0 && k % 64 == 0, "input K must be a positive multiple of 64");
  TORCH_CHECK(n > 0 && n % 32 == 0, "output N must be a positive multiple of 32");
  TORCH_CHECK(k <= std::numeric_limits<int64_t>::max() / 2, "2*K overflows the mixed kernel ABI");
  TORCH_CHECK(input_scale.size(0) == m && input_scale.size(1) == k / kMxScaleGroupSize,
              "input_scale must have shape [M,K/32]");
  TORCH_CHECK(packed_weight.size(0) == static_cast<int64_t>(kGroupCount) &&
                  packed_weight.size(1) == n / kPackedN0 && packed_weight.size(2) == k / kPackedK0 &&
                  packed_weight.size(3) == kPackedK0 && packed_weight.size(4) == kPackedN0 / 4,
              "packed_weight must have shape [1,N/32,K/16,16,8]");
  TORCH_CHECK(pair_lut.size(0) == static_cast<int64_t>(kGroupCount) && pair_lut.size(1) == n / kPackedN0 &&
                  pair_lut.size(2) == k / kMxScaleGroupSize && pair_lut.size(3) == 32,
              "pair_lut must have shape [1,N/32,K/32,32]");
  TORCH_CHECK(row_scale.size(0) == static_cast<int64_t>(kGroupCount) && row_scale.size(1) == n,
              "row_scale must have shape [1,N] with FP32 bits in each INT64 low word");
  TORCH_CHECK(group_list.numel() == static_cast<int64_t>(kGroupCount),
              "group_list must have shape [1] and contain M");
  TORCH_CHECK(route_mask.numel() == 1, "route_mask must have shape [1] and contain the all-W2 value zero");
  TORCH_CHECK(tiling.numel() == static_cast<int64_t>(kMixedTilingBytes),
              "tiling must be the 1600-byte tensor returned by make_tiling(K,N)");
  TORCH_CHECK(output.size(0) == m, "output must have shape [M,N]");

  TORCH_CHECK(group_list.data_ptr<int64_t>()[0] == m, "group_list must be the validated CPU value [M]");
  TORCH_CHECK(route_mask.data_ptr<int32_t>()[0] == 0,
              "route_mask must be the validated CPU all-W2 value [0]");
  const auto expected_tiling = MakeTiling(k, n);
  TORCH_CHECK(std::memcmp(tiling.data_ptr(), expected_tiling.data_ptr(), kMixedTilingBytes) == 0,
              "tiling bytes do not match make_tiling(K,N) with stage_mask=DownW2");

  const std::vector<std::pair<const char*, at::Tensor>> npu_owners{
      {"input_bytes", input_bytes},   {"input_scale", input_scale}, {"packed_weight", packed_weight},
      {"pair_lut", pair_lut},         {"row_scale", row_scale},     {"output", output},
  };
  const std::vector<std::pair<const char*, at::Tensor>> cpu_metadata{
      {"group_list", group_list}, {"route_mask", route_mask}, {"tiling", tiling}};
  CheckNoAlias(npu_owners);
  CheckNoAlias(cpu_metadata);

  const auto registered = RegisteredKernelSnapshot();
  // Safety-first initial ABI: validate the three tiny metadata tensors on CPU,
  // then perform bounded H2D copies for every call. A future validated plan
  // handle may make these resident; this implementation makes no async
  // performance claim for the metadata path.
  auto device_group_list = group_list.to(input_bytes.device(), at::kLong, false, true);
  auto device_route_mask = route_mask.to(input_bytes.device(), at::kInt, false, true);
  auto device_tiling = tiling.to(input_bytes.device(), at::kByte, false, true);
  const auto stream = c10_npu::getCurrentNPUStream();
  RecordStream({input_bytes, input_scale, packed_weight, pair_lut, row_scale, output, device_group_list,
                device_route_mask, device_tiling},
               stream);
  // NPUStream::stream() may drain the host task queue. Resolve it before
  // entering RunOpApi so the callback never waits for its own queue slot.
  const auto launch_stream = stream.stream();
  at_npu::native::OpCommand::RunOpApi(
      "Vq2a8Gs32StageProject",
      [registered, launch_stream, input_bytes, input_scale, packed_weight, pair_lut, row_scale, group_list,
       route_mask, tiling, output, device_group_list, device_route_mask, device_tiling, k, n]() -> int {
        // Gate arguments use valid Torch-owned aliases even though stage_mask=4
        // prevents either Gate implementation from dereferencing them.
        MatFP4Args arguments{
            kGroupCount,
            static_cast<uint64_t>(k),
            static_cast<uint64_t>(2 * k),
            static_cast<uint64_t>(n),
            device_group_list.data_ptr(),
            input_bytes.data_ptr(),
            input_scale.data_ptr(),
            packed_weight.data_ptr(),
            pair_lut.data_ptr(),
            packed_weight.data_ptr(),
            pair_lut.data_ptr(),
            row_scale.data_ptr(),
            row_scale.data_ptr(),
            device_route_mask.data_ptr(),
            device_route_mask.data_ptr(),
            input_bytes.data_ptr(),
            input_scale.data_ptr(),
            output.data_ptr(),
            device_tiling.data_ptr(),
        };
        return static_cast<int>(rtKernelLaunch(registered->stub, kKernelCoreCount, &arguments, sizeof(arguments),
                                               nullptr, launch_stream));
      },
      false);
}

}  // namespace
}  // namespace vq2a8_gs32_stage

TORCH_LIBRARY(vq2a8_gs32_stage, m) {
  m.class_<vq2a8_gs32_stage::ProjectionPlan>("ProjectionPlan");
  m.class_<vq2a8_gs32_stage::GroupedRoutes>("GroupedRoutes");
  m.class_<vq2a8_gs32_stage::DeviceRoutes>("DeviceRoutes");
  m.class_<vq2a8_gs32_stage::GroupedProjectionPlan>("GroupedProjectionPlan");
  m.def("initialize(str binary) -> ()", &vq2a8_gs32_stage::Initialize);
  m.def("make_tiling(int k, int n) -> Tensor", &vq2a8_gs32_stage::MakeTiling);
  m.def("prepare(Tensor input_bytes, int n) -> __torch__.torch.classes.vq2a8_gs32_stage.ProjectionPlan");
  m.def("project_prepared(Tensor input_bytes, Tensor input_scale, Tensor packed_weight, Tensor pair_lut, "
        "Tensor row_scale, __torch__.torch.classes.vq2a8_gs32_stage.ProjectionPlan plan, "
        "Tensor(a!) output) -> ()");
  m.def("prepare_routes(Tensor anchor, Tensor counts) -> __torch__.torch.classes.vq2a8_gs32_stage.GroupedRoutes");
  m.def("prepare_device_routes(Tensor anchor, Tensor counts, bool allow_padding=False) -> "
        "__torch__.torch.classes.vq2a8_gs32_stage.DeviceRoutes");
  m.def("device_routes_status(__torch__.torch.classes.vq2a8_gs32_stage.DeviceRoutes routes) -> Tensor",
        &vq2a8_gs32_stage::DeviceRoutesStatus);
  m.def("prepare_grouped(Tensor input_bytes, int n, int groups) -> "
        "__torch__.torch.classes.vq2a8_gs32_stage.GroupedProjectionPlan");
  m.def("project_grouped(Tensor input_bytes, Tensor input_scale, Tensor packed_weight, Tensor pair_lut, "
        "Tensor row_scale, __torch__.torch.classes.vq2a8_gs32_stage.GroupedRoutes routes, "
        "__torch__.torch.classes.vq2a8_gs32_stage.GroupedProjectionPlan plan, Tensor(a!) output) -> ()");
  m.def("project_grouped_device(Tensor input_bytes, Tensor input_scale, Tensor packed_weight, Tensor pair_lut, "
        "Tensor row_scale, __torch__.torch.classes.vq2a8_gs32_stage.DeviceRoutes routes, "
        "__torch__.torch.classes.vq2a8_gs32_stage.GroupedProjectionPlan plan, Tensor(a!) output) -> ()");
  m.def("project(Tensor input_bytes, Tensor input_scale, Tensor packed_weight, Tensor pair_lut, "
        "Tensor row_scale, Tensor group_list, Tensor route_mask, Tensor tiling, Tensor(a!) output) -> ()");
}

TORCH_LIBRARY_IMPL(vq2a8_gs32_stage, PrivateUse1, m) {
  m.impl("prepare_device_routes", &vq2a8_gs32_stage::PrepareDeviceRoutes);
  m.impl("project_grouped_device", &vq2a8_gs32_stage::ProjectGroupedDevice);
  m.impl("prepare_routes", &vq2a8_gs32_stage::PrepareRoutes);
  m.impl("prepare_grouped", &vq2a8_gs32_stage::PrepareGrouped);
  m.impl("project_grouped", &vq2a8_gs32_stage::ProjectGrouped);
  m.impl("prepare", &vq2a8_gs32_stage::Prepare);
  m.impl("project_prepared", &vq2a8_gs32_stage::ProjectPrepared);
  m.impl("project", &vq2a8_gs32_stage::Project);
}
