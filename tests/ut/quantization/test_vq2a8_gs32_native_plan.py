# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source contracts only; target compilation and NPU execution remain required."""

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


def test_plan_is_opaque_and_does_not_export_mutable_metadata():
    source = SOURCE.read_text()
    assert "class ProjectionPlan : public torch::CustomClassHolder" in source
    assert 'm.class_<vq2a8_gs32_stage::ProjectionPlan>("ProjectionPlan");' in source
    assert "__torch__.torch.classes.vq2a8_gs32_stage.ProjectionPlan" in source
    # No Python constructor/accessors can expose or replace the validated tensors.
    registration = source.split("TORCH_LIBRARY(vq2a8_gs32_stage, m)", 1)[1]
    assert "def_readwrite" not in registration
    assert "def_readonly" not in registration
    assert "torch::init" not in registration
    assert 'm.impl("prepare"' in source
    assert 'm.impl("project_prepared"' in source


def test_prepare_checks_geometry_and_uploads_owned_metadata_once():
    source = SOURCE.read_text()
    prepare = body(source, "Prepare")
    assert prepare.index("OptionalNPUGuard") < prepare.index("CheckNotCapturing")
    assert "CheckNpuTensor(input_bytes" in prepare
    assert "ValidateProjectionShape(m, k, n)" in prepare
    assert "MakeTiling(k, n)" in prepare
    assert "at::full" in prepare
    assert "at::zeros" in prepare
    assert "RegisteredKernelSnapshot()" in prepare
    assert prepare.count(".to(input_bytes.device(),") == 3
    assert prepare.count("false, true)") == 3


def test_prepared_launch_rejects_wrong_geometry_device_alias_and_capture():
    source = SOURCE.read_text()
    project = body(source, "ProjectPrepared")
    assert "plan != nullptr" in project
    assert project.index("OptionalNPUGuard") < project.index("CheckNotCapturing")
    assert "plan->CheckMatches(input_bytes, output)" in project
    assert "CheckProjectionTensors(" in project
    assert "CheckNoAlias(" in body(source, "CheckProjectionTensors")
    matches = body(source, "CheckMatches")
    for check in (
        "input_bytes.device() == device_",
        "input_bytes.size(0) == m_",
        "input_bytes.size(1) == k_",
        "output.size(1) == n_",
    ):
        assert check in matches


def test_prepared_launch_has_no_metadata_build_copy_or_host_scalar_read():
    source = SOURCE.read_text()
    project = body(source, "ProjectPrepared")
    for forbidden in ("MakeTiling", ".to(", ".cpu(", ".item(", "Synchronize", "synchronize"):
        assert forbidden not in project
    assert "RecordStream(" in project
    assert "plan->GroupList()" in project
    assert "plan->RouteMask()" in project
    assert "plan->Tiling()" in project
    before, callback = project.split("at_npu::native::OpCommand::RunOpApi(", 1)
    assert "const auto launch_stream = stream.stream(false);" in before
    assert "stream.stream()" not in project
    assert "getCurrentNPUStream" not in callback
    assert ".stream()" not in callback
    assert "[plan, launch_stream" in callback
    assert "input_bytes, input_scale, packed_weight, pair_lut, row_scale, output" in callback
    assert "rtKernelLaunch(" in callback
    assert "false);" in callback


@pytest.fixture(scope="module")
def plan_harness(tmp_path_factory):
    """Compile actual plan/launch/shape/alias code with fake device boundaries.

    NPU/ATen types, recordStream, the allocator and queued runtime are doubles.
    This verifies host control flow and C++ callback ownership only.
    """
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler is required for the prepared projection host harness")
    source = SOURCE.read_text()
    class_begin = source.index("class ProjectionPlan :")
    class_end = source.index("c10::intrusive_ptr<ProjectionPlan> Prepare", class_begin)
    harness = r"""
#include <cstdint>
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
enum class DeviceType { PrivateUse1 };
struct Device {
  int index = 0;
  DeviceType type() const { return DeviceType::PrivateUse1; }
  bool operator==(Device other) const { return index == other.index; }
};
template <typename T> using intrusive_ptr = std::shared_ptr<T>;
}
namespace at {
enum ScalarType { kByte, kLong, kBFloat16 };
struct Tensor {
  std::vector<int64_t> shape;
  ScalarType dtype;
  c10::Device where;
  std::shared_ptr<std::vector<uint8_t>> owner;
  Tensor(std::vector<int64_t> dimensions, ScalarType type, c10::Device device = {})
      : shape(std::move(dimensions)), dtype(type), where(device) {
    owner = std::make_shared<std::vector<uint8_t>>(nbytes());
  }
  bool defined() const { return owner != nullptr; }
  c10::Device device() const { return where; }
  int64_t size(size_t axis) const { return shape.at(axis); }
  size_t nbytes() const {
    size_t count = dtype == kByte ? 1 : dtype == kLong ? 8 : 2;
    for (auto dim : shape) count *= dim;
    return count;
  }
  void* data_ptr() const { return owner->data(); }
};
}
int queue_drains = 0;
int launches = 0;
int recorded_owners = 0;
int capture_status = 0;
uintptr_t last_launch_stream = 0;
uintptr_t last_record_stream = 0;
namespace c10_npu {
struct NPUStream {
  uintptr_t handle = 100;
  void* stream(bool need_empty = true) const {
    if (need_empty) ++queue_drains;
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
    TORCH_CHECK(!sync, "queued launch expected");
    callbacks.push_back(std::move(callback));
  }
};
}}
constexpr uint64_t kGroupCount = 1;
constexpr uint64_t kMxScaleGroupSize = 32;
constexpr uint64_t kPackedK0 = 16;
constexpr uint64_t kPackedN0 = 32;
constexpr uint32_t kKernelCoreCount = 32;
constexpr int64_t kMaxProjectionRows = 128;
struct MatFP4Args __ARGS__;
struct RegisteredKernel { const void* stub = reinterpret_cast<void*>(1234); };
void CheckNotCapturing(const char*) { TORCH_CHECK(capture_status == 0, "capture rejected"); }
void CheckNpuTensor(const at::Tensor& tensor, const at::Tensor& anchor,
                    at::ScalarType dtype, int rank, const char*) {
  TORCH_CHECK(tensor.defined() && tensor.device() == anchor.device() &&
              tensor.dtype == dtype && tensor.shape.size() == rank, "bad tensor");
}
struct NamedRange __RANGE__;
NamedRange TensorRange(const char* name, const at::Tensor& tensor) __TENSOR_RANGE__
void CheckNoAlias(const std::vector<std::pair<const char*, at::Tensor>>& tensors) __ALIAS__
void ValidateProjectionShape(int64_t m, int64_t k, int64_t n) __SHAPE__
void CheckProjectionTensors(const at::Tensor& input_bytes, const at::Tensor& input_scale,
                            const at::Tensor& packed_weight, const at::Tensor& pair_lut,
                            const at::Tensor& row_scale, const at::Tensor& output) __CHECKS__
__PLAN__
void RecordStream(const std::vector<at::Tensor>& tensors, c10_npu::NPUStream stream) {
  recorded_owners += tensors.size();
  last_record_stream = stream.handle;
}
int rtKernelLaunch(const void* stub, uint32_t blocks, void* raw_args, size_t bytes,
                   void*, void* stream) {
  auto& args = *static_cast<MatFP4Args*>(raw_args);
  TORCH_CHECK(stub != nullptr && blocks == 32 && bytes == 152, "bad launch ABI");
  TORCH_CHECK(args.group_count == 1 && args.k == 64 && args.n == 128 && args.d == 32,
              "wrong DownW2 mixed geometry");
  TORCH_CHECK(*static_cast<uint8_t*>(args.group_list) == 1, "metadata lost");
  ++launches;
  last_launch_stream = reinterpret_cast<uintptr_t>(stream);
  return 0;
}
void ProjectPrepared(const at::Tensor& input_bytes, const at::Tensor& input_scale,
                     const at::Tensor& packed_weight, const at::Tensor& pair_lut, const at::Tensor& row_scale,
                     const c10::intrusive_ptr<ProjectionPlan>& plan, at::Tensor& output) __PROJECT__
int main(int argc, char** argv) {
  const std::string scenario = argv[1];
  at::Tensor group({1}, at::kLong), mask({1}, at::kByte), tiling({1600}, at::kByte);
  *static_cast<uint8_t*>(group.data_ptr()) = 1;
  auto plan = std::make_shared<ProjectionPlan>(1, 64, 32, c10::Device{}, group, mask, tiling,
                                               std::make_shared<RegisteredKernel>());
  std::weak_ptr<ProjectionPlan> weak = plan;
  bool accepted = true;
  {
    at::Tensor input({1, 64}, at::kByte), scale({1, 2}, at::kByte),
               weights({1, 1, 4, 16, 8}, at::kByte), lut({1, 1, 2, 32}, at::kByte),
               row({1, 32}, at::kLong), output({1, 32}, at::kBFloat16);
    if (scenario == "other_stream") c10_npu::current.handle = 200;
    if (scenario == "wrong_m") {
      input = at::Tensor({2, 64}, at::kByte);
      scale = at::Tensor({2, 2}, at::kByte);
      output = at::Tensor({2, 32}, at::kBFloat16);
    }
    if (scenario == "wrong_k") input = at::Tensor({1, 128}, at::kByte);
    if (scenario == "wrong_n") output = at::Tensor({1, 64}, at::kBFloat16);
    if (scenario == "wrong_device") {
      for (auto* tensor : {&input, &scale, &weights, &lut, &row, &output}) tensor->where.index = 1;
    }
    if (scenario == "wrong_scale") scale = at::Tensor({1, 1}, at::kByte);
    if (scenario == "wrong_dtype") input.dtype = at::kLong;
    if (scenario == "alias") output.owner = input.owner;
    if (scenario == "capture") capture_status = 1;
    if (scenario == "null_plan") plan.reset();
    try {
      for (int i = 0; i < 100; ++i) ProjectPrepared(input, scale, weights, lut, row, plan, output);
    } catch (const std::runtime_error&) { accepted = false; }
  }
  plan.reset();
  const bool retained = !weak.expired();
  const auto pending = callbacks.size();
  for (const auto& callback : callbacks) TORCH_CHECK(callback() == 0, "launch failed");
  callbacks.clear();
  std::cout << "{\"accepted\":" << (accepted ? "true" : "false")
            << ",\"pending\":" << pending << ",\"launches\":" << launches
            << ",\"queue_drains\":" << queue_drains << ",\"recorded_owners\":" << recorded_owners
            << ",\"retained_until_launch\":" << (retained ? "true" : "false")
            << ",\"released_after_launch\":" << (weak.expired() ? "true" : "false")
            << ",\"last_launch_stream\":" << last_launch_stream
            << ",\"last_record_stream\":" << last_record_stream << "}";
}
"""
    for marker, actual in {
        "__ARGS__": body(source.replace("struct MatFP4Args {", "MatFP4Args() {"), "MatFP4Args"),
        "__RANGE__": body(source.replace("struct NamedRange {", "NamedRange() {"), "NamedRange"),
        "__TENSOR_RANGE__": body(source, "TensorRange"),
        "__ALIAS__": body(source, "CheckNoAlias"),
        "__SHAPE__": body(source, "ValidateProjectionShape"),
        "__CHECKS__": body(source, "CheckProjectionTensors"),
        "__PLAN__": source[class_begin:class_end],
        "__PROJECT__": body(source, "ProjectPrepared"),
    }.items():
        harness = harness.replace(marker, actual)
    directory = tmp_path_factory.mktemp("gs32_prepared_plan_host")
    cpp = directory / "plan.cpp"
    executable = directory / "plan"
    cpp.write_text(harness)
    result = subprocess.run(
        [compiler, "-std=c++17", str(cpp), "-o", str(executable)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return executable


@pytest.mark.parametrize("scenario,stream", [("same_stream", 100), ("other_stream", 200)])
def test_real_host_plan_launch_is_queued_owned_and_does_not_drain(plan_harness, scenario, stream):
    result = subprocess.run([str(plan_harness), scenario], capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report == {
        "accepted": True,
        "pending": 100,
        "launches": 100,
        "queue_drains": 0,
        "recorded_owners": 900,
        "retained_until_launch": True,
        "released_after_launch": True,
        "last_launch_stream": stream,
        "last_record_stream": stream,
    }


@pytest.mark.parametrize(
    "scenario",
    ["wrong_m", "wrong_k", "wrong_n", "wrong_device", "wrong_scale", "wrong_dtype", "alias", "capture", "null_plan"],
)
def test_real_host_plan_rejects_invalid_launch_before_queueing(plan_harness, scenario):
    result = subprocess.run([str(plan_harness), scenario], capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["accepted"] is False
    assert report["pending"] == 0
    assert report["launches"] == 0
    assert report["recorded_owners"] == 0
    assert report["queue_drains"] == 0
