# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only grouped projection contracts; no NPU numerical claim."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "csrc/vq2a8_gs32_stage/torch_binding.cpp"


def body(source, name):
    start = source.index(f"{name}(")
    begin = source.index("{", start)
    end = begin + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


def test_grouped_schema_retains_diagnostic_abi_and_opaque_handles():
    source = SOURCE.read_text()
    for name in ("GroupedRoutes", "GroupedProjectionPlan"):
        assert f"class {name} : public torch::CustomClassHolder" in source
        assert f'm.class_<vq2a8_gs32_stage::{name}>("{name}");' in source
    for name in ("prepare_routes", "prepare_grouped", "project_grouped", "project", "prepare", "project_prepared"):
        assert f'm.impl("{name}"' in source


def test_routes_validate_cpu_counts_and_make_one_owned_blocking_upload():
    source = SOURCE.read_text()
    prepare = body(source, "PrepareRoutes")
    assert "CheckNotCapturing(" in prepare
    assert "const auto owned_counts = counts.clone();" in prepare
    assert "ValidateGroupCounts(owned_counts, anchor.size(0))" in prepare
    assert prepare.count(".to(") == 1
    assert "owned_counts.to(anchor.device(), at::kLong, false, true)" in prepare
    checks = body(source, "ValidateGroupCounts")
    assert "CheckCpuMetadata(counts, at::kLong, 1" in checks
    assert "count >= 0 && count <= kMaxProjectionRows" in checks
    assert "count <= rows - total" in checks
    assert "total == rows" in checks


def test_grouped_plan_uses_existing_tiling_and_bitmask_word_count():
    source = SOURCE.read_text()
    prepare = body(source, "PrepareGrouped")
    assert "ValidateGroupCount(groups)" in prepare
    assert "MakeTiling(k, n)" in prepare
    assert "(groups + kRouteMaskWordBits - 1) / kRouteMaskWordBits" in prepare
    assert prepare.count(".to(") == 2
    assert "RegisteredKernelSnapshot()" in prepare


def test_grouped_launch_queues_without_build_copy_or_host_scalar_sync():
    source = SOURCE.read_text()
    project = body(source, "ProjectGrouped")
    assert "CheckGroupedProjectionTensors(" in project
    assert "routes->CheckMatches(input_bytes, plan->Groups())" in project
    assert "plan->CheckMatches(input_bytes, output)" in project
    assert "RecordStream(" in project
    assert "routes->GroupList()" in project
    assert "stream.stream(false)" in project
    assert "[routes, plan, launch_stream" in project
    assert "static_cast<uint64_t>(plan->Groups())" in project
    for forbidden in (".cpu(", ".item(", ".to(", "MakeTiling", "Synchronize", "synchronize", "stream.stream()"):
        assert forbidden not in project
    assert "kKernelCoreCount" in project
    assert "static_cast<uint64_t>(2 * plan->K())" in project


@pytest.fixture(scope="module")
def grouped_harness(tmp_path_factory):
    """Execute actual counts/handle/launch code; only NPU/ATen are doubles."""
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler is required for grouped host contracts")
    source = SOURCE.read_text()
    routes = source[
        source.index("class GroupedRoutes :") : source.index("c10::intrusive_ptr<GroupedRoutes> PrepareRoutes")
    ]
    plan = source[
        source.index("class GroupedProjectionPlan :") : source.index(
            "c10::intrusive_ptr<GroupedProjectionPlan> PrepareGrouped"
        )
    ]
    harness = r"""
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#define TORCH_CHECK(condition, ...) \
  do { if (!(condition)) throw std::runtime_error("rejected"); } while (false)
namespace torch { struct CustomClassHolder {}; }
namespace c10 {
enum class DeviceType { CPU, PrivateUse1 };
struct Device {
  int index = 0;
  DeviceType kind = DeviceType::PrivateUse1;
  DeviceType type() const { return kind; }
  bool operator==(Device other) const { return index == other.index && kind == other.kind; }
};
template <typename T> using intrusive_ptr = std::shared_ptr<T>;
template <typename T, typename... Args> intrusive_ptr<T> make_intrusive(Args&&... args) {
  return std::make_shared<T>(std::forward<Args>(args)...);
}
}
int metadata_uploads = 0;
namespace at {
enum ScalarType { kByte, kLong, kBFloat16 };
struct Tensor {
  std::vector<int64_t> shape;
  ScalarType dtype;
  c10::Device where;
  std::shared_ptr<std::vector<uint8_t>> owner;
  Tensor(std::vector<int64_t> dimensions, ScalarType type, c10::Device device = {})
      : shape(std::move(dimensions)), dtype(type), where(device),
        owner(std::make_shared<std::vector<uint8_t>>(nbytes())) {}
  bool defined() const { return owner != nullptr; }
  c10::Device device() const { return where; }
  int dim() const { return shape.size(); }
  int64_t size(size_t axis) const { return shape.at(axis); }
  int64_t numel() const { int64_t n = 1; for (auto dim : shape) n *= dim; return n; }
  size_t nbytes() const { return numel() * (dtype == kByte ? 1 : dtype == kLong ? 8 : 2); }
  void* data_ptr() const { return owner->data(); }
  template <typename T> T* data_ptr() const { return reinterpret_cast<T*>(owner->data()); }
  Tensor clone() const { Tensor result(shape,dtype,where); *result.owner = *owner; return result; }
  Tensor to(c10::Device device, ScalarType type, bool non_blocking, bool copy) const {
    TORCH_CHECK(!non_blocking && copy && type == dtype, "owned blocking copy required");
    Tensor result(shape, dtype, device);
    *result.owner = *owner;
    ++metadata_uploads;
    return result;
  }
};
}
int queue_drains = 0;
int launches = 0;
int records = 0;
int capture_status = 0;
uintptr_t launch_handle = 0;
namespace c10_npu {
struct NPUStream {
  uintptr_t handle = 100;
  void* stream(bool drain = true) const {
    if (drain) ++queue_drains;
    return reinterpret_cast<void*>(handle);
  }
};
NPUStream current;
NPUStream getCurrentNPUStream() { return current; }
struct OptionalNPUGuard { explicit OptionalNPUGuard(c10::Device) {} };
}
std::vector<std::function<int()>> callbacks;
namespace at_npu { namespace native {
struct OpCommand {
  static void RunOpApi(const char*, std::function<int()> callback, bool sync) {
    TORCH_CHECK(!sync, "must queue launch");
    callbacks.push_back(std::move(callback));
  }
};
}}
constexpr uint64_t kMxScaleGroupSize = 32;
constexpr uint64_t kPackedK0 = 16;
constexpr uint64_t kPackedN0 = 32;
constexpr uint32_t kKernelCoreCount = 32;
constexpr int64_t kMaxProjectionRows = 128;
constexpr int64_t kMaxGroupedExperts = 256;
struct MatFP4Args __ARGS__;
struct RegisteredKernel { const void* stub = reinterpret_cast<void*>(1234); };
void CheckNotCapturing(const char*) { TORCH_CHECK(capture_status == 0, "capture rejected"); }
void CheckNpuTensor(const at::Tensor& tensor, const at::Tensor& anchor,
                    at::ScalarType dtype, int rank, const char*) {
  TORCH_CHECK(tensor.defined() && tensor.device() == anchor.device() &&
              tensor.device().type() == c10::DeviceType::PrivateUse1 &&
              tensor.dtype == dtype && tensor.dim() == rank, "bad device tensor");
}
void CheckCpuMetadata(const at::Tensor& tensor, at::ScalarType dtype, int rank, const char*) {
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::CPU && tensor.dtype == dtype &&
              tensor.dim() == rank, "bad CPU metadata");
}
struct NamedRange __RANGE__;
NamedRange TensorRange(const char* name, const at::Tensor& tensor) __TENSOR_RANGE__
void CheckNoAlias(const std::vector<std::pair<const char*, at::Tensor>>& tensors) __ALIAS__
void ValidateProjectionShape(int64_t m, int64_t k, int64_t n) __SHAPE__
void ValidateGroupCount(int64_t groups) __GROUP_COUNT__
void ValidateGroupCounts(const at::Tensor& counts, int64_t rows) __COUNTS__
__ROUTES__
c10::intrusive_ptr<GroupedRoutes> PrepareRoutes(const at::Tensor& anchor, const at::Tensor& counts) __PREPARE_ROUTES__
__PLAN__
void CheckGroupedProjectionTensors(const at::Tensor& input_bytes, const at::Tensor& input_scale,
                                   const at::Tensor& packed_weight, const at::Tensor& pair_lut,
                                   const at::Tensor& row_scale, const at::Tensor& output, int64_t groups) __CHECKS__
void RecordStream(const std::vector<at::Tensor>& tensors, c10_npu::NPUStream stream) { records += tensors.size(); }
int rtKernelLaunch(const void* stub, uint32_t blocks, void* raw_args, size_t bytes, void*, void* stream) {
  auto& args = *static_cast<MatFP4Args*>(raw_args);
  TORCH_CHECK(stub && blocks == 32 && bytes == 152, "wrong launch ABI");
  TORCH_CHECK(args.group_count == 3 && args.k == 64 && args.n == 128 && args.d == 32, "wrong mixed shape");
  auto counts = static_cast<int64_t*>(args.group_list);
  TORCH_CHECK(counts[0] == 1 && counts[1] == 0 && counts[2] == 2, "route mutation or cumulative counts");
  ++launches;
  launch_handle = reinterpret_cast<uintptr_t>(stream);
  return 0;
}
void ProjectGrouped(const at::Tensor& input_bytes, const at::Tensor& input_scale,
                    const at::Tensor& packed_weight, const at::Tensor& pair_lut, const at::Tensor& row_scale,
                    const c10::intrusive_ptr<GroupedRoutes>& routes,
                    const c10::intrusive_ptr<GroupedProjectionPlan>& plan, at::Tensor& output) __PROJECT__
int main(int argc, char** argv) {
  const std::string scenario = argv[1];
  bool accepted = true;
  std::weak_ptr<GroupedRoutes> weak_routes;
  std::weak_ptr<GroupedProjectionPlan> weak_plan;
  {
    at::Tensor input({3, 64}, at::kByte), scales({3, 2}, at::kByte), weights({3, 1, 4, 16, 8}, at::kByte),
               lut({3, 1, 2, 32}, at::kByte), row({3, 32}, at::kLong), output({3, 32}, at::kBFloat16),
               mask({4}, at::kByte), tiling({1600}, at::kByte),
               counts({3}, at::kLong, {0, c10::DeviceType::CPU});
    counts.data_ptr<int64_t>()[0] = 1;
    counts.data_ptr<int64_t>()[2] = 2;
    if (scenario == "negative") counts.data_ptr<int64_t>()[0] = -1;
    if (scenario == "sum_small") counts.data_ptr<int64_t>()[2] = 1;
    if (scenario == "sum_large") counts.data_ptr<int64_t>()[2] = 3;
    if (scenario == "all_zero") { counts.data_ptr<int64_t>()[0] = 0; counts.data_ptr<int64_t>()[2] = 0; }
    if (scenario == "too_many_rows") { input = at::Tensor({130,64},at::kByte); counts.data_ptr<int64_t>()[2] = 129; }
    if (scenario == "huge_count") counts.data_ptr<int64_t>()[0] = std::numeric_limits<int64_t>::max();
    if (scenario == "too_many_groups") counts = at::Tensor({257},at::kLong,{0,c10::DeviceType::CPU});
    if (scenario == "no_groups") counts = at::Tensor({0},at::kLong,{0,c10::DeviceType::CPU});
    if (scenario == "device_counts") counts.where.kind = c10::DeviceType::PrivateUse1;
    if (scenario == "wrong_dtype") counts.dtype = at::kByte;
    if (scenario == "capture_prepare") capture_status = 1;
    try {
      auto routes = PrepareRoutes(input, counts);
      weak_routes = routes;
      // External CPU mutation must not change the already validated plan.
      std::memset(counts.data_ptr(), 0, counts.nbytes());
      auto plan = std::make_shared<GroupedProjectionPlan>(3,64,32,c10::Device{},mask,tiling,
                                                         std::make_shared<RegisteredKernel>());
      weak_plan = plan;
      if (scenario == "other_stream") c10_npu::current.handle = 200;
      if (scenario == "wrong_bank") weights = at::Tensor({2,1,4,16,8},at::kByte);
      if (scenario == "wrong_rows") { input = at::Tensor({2,64},at::kByte); scales = at::Tensor({2,2},at::kByte);
                                       output = at::Tensor({2,32},at::kBFloat16); }
      if (scenario == "wrong_device") {
        for (auto* tensor : {&input,&scales,&weights,&lut,&row,&output}) tensor->where.index = 1;
      }
      if (scenario == "alias") output.owner = input.owner;
      if (scenario == "capture_launch") capture_status = 1;
      if (scenario == "null_routes") routes.reset();
      if (scenario == "null_plan") plan.reset();
      for (int call = 0; call < 2; ++call) ProjectGrouped(input,scales,weights,lut,row,routes,plan,output);
    } catch (const std::runtime_error&) { accepted = false; }
  }
  const bool retained = !weak_routes.expired() && !weak_plan.expired();
  const auto pending = callbacks.size();
  for (const auto& callback : callbacks) TORCH_CHECK(callback() == 0, "launch failed");
  callbacks.clear();
  std::cout << "{\"accepted\":" << (accepted ? "true" : "false")
            << ",\"uploads\":" << metadata_uploads << ",\"launches\":" << launches
            << ",\"queue_drains\":" << queue_drains << ",\"records\":" << records
            << ",\"pending\":" << pending << ",\"retained\":" << (retained ? "true" : "false")
            << ",\"released\":" << ((weak_routes.expired() && weak_plan.expired()) ? "true" : "false")
            << ",\"stream\":" << launch_handle << "}";
}
"""
    for marker, actual in {
        "__ARGS__": body(source.replace("struct MatFP4Args {", "MatFP4Args() {"), "MatFP4Args"),
        "__RANGE__": body(source.replace("struct NamedRange {", "NamedRange() {"), "NamedRange"),
        "__TENSOR_RANGE__": body(source, "TensorRange"),
        "__ALIAS__": body(source, "CheckNoAlias"),
        "__SHAPE__": body(source, "ValidateProjectionShape"),
        "__GROUP_COUNT__": body(source, "ValidateGroupCount"),
        "__COUNTS__": body(source, "ValidateGroupCounts"),
        "__ROUTES__": routes,
        "__PREPARE_ROUTES__": body(source, "PrepareRoutes"),
        "__PLAN__": plan,
        "__CHECKS__": body(source, "CheckGroupedProjectionTensors"),
        "__PROJECT__": body(source, "ProjectGrouped"),
    }.items():
        harness = harness.replace(marker, actual)
    directory = tmp_path_factory.mktemp("gs32_grouped_host")
    cpp = directory / "grouped.cpp"
    executable = directory / "grouped"
    cpp.write_text(harness)
    result = subprocess.run(
        [compiler, "-std=c++17", str(cpp), "-o", str(executable)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return executable


@pytest.mark.parametrize("scenario,stream", [("same_stream", 100), ("other_stream", 200)])
def test_grouped_routes_upload_once_for_two_queued_projections(grouped_harness, scenario, stream):
    result = subprocess.run([str(grouped_harness), scenario], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {
        "accepted": True,
        "uploads": 1,
        "launches": 2,
        "queue_drains": 0,
        "records": 18,
        "pending": 2,
        "retained": True,
        "released": True,
        "stream": stream,
    }


@pytest.mark.parametrize(
    "scenario",
    [
        "negative",
        "sum_small",
        "sum_large",
        "all_zero",
        "too_many_rows",
        "huge_count",
        "too_many_groups",
        "no_groups",
        "device_counts",
        "wrong_dtype",
        "capture_prepare",
        "wrong_bank",
        "wrong_rows",
        "wrong_device",
        "alias",
        "capture_launch",
        "null_routes",
        "null_plan",
    ],
)
def test_grouped_routes_reject_invalid_inputs_before_launch(grouped_harness, scenario):
    result = subprocess.run([str(grouped_harness), scenario], capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["accepted"] is False
    assert report["launches"] == 0
    assert report["pending"] == 0
    assert report["records"] == 0
    assert report["queue_drains"] == 0
