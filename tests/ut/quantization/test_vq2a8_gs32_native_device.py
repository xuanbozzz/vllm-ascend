# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual ATen sanitizer math on CPU; capture/runtime checks are host-only.

This compiles the production sanitizer and opaque route class against Torch.
It does not pretend to verify CANN, allocator recording or hardware capture.
"""

import platform
import shutil
import subprocess
import sys

import pytest
import torch
from torch.utils.cpp_extension import include_paths, library_paths

from tests.ut.quantization.test_vq2a8_gs32_native_grouped import SOURCE, body


def test_device_schema_is_separate_from_strict_eager_routes():
    source = SOURCE.read_text()
    assert 'm.class_<vq2a8_gs32_stage::DeviceRoutes>("DeviceRoutes");' in source
    for name in ("prepare_device_routes", "device_routes_status", "project_grouped_device"):
        assert f'"{name}(' in source
    for name in ("PrepareRoutes", "ProjectGrouped", "PrepareGrouped", "ProjectPrepared"):
        assert "CheckNotCapturing(" in body(source, name)
    assert "torch::init" not in source.split("TORCH_LIBRARY(vq2a8_gs32_stage, m)")[1]


def test_device_prepare_has_no_host_value_access_or_metadata_transfer():
    source = SOURCE.read_text()
    prepare = body(source, "PrepareDeviceRoutes")
    assert "CheckNpuTensor(counts, anchor, at::kLong, 1" in prepare
    assert "ValidateGroupCount(counts.numel())" in prepare
    assert "kMaxGroupedExperts * kMaxProjectionRows" in prepare
    assert "CheckCaptureAllowed(" in prepare
    sanitizer = body(source, "SanitizeDeviceCounts")
    assert sanitizer.index("counts.clone()") < sanitizer.index(".clamp(") < sanitizer.index(".sum(")
    for forbidden in (".cpu(", ".item", "data_ptr", "kCPU", "RegisteredKernelSnapshot", "synchronize"):
        assert forbidden not in prepare + sanitizer
    assert "at::where(" in sanitizer
    assert ".clone()" in body(source, "DeviceRoutesStatus")


def test_device_launch_keeps_owners_and_does_not_drain_or_upload():
    source = SOURCE.read_text()
    project = body(source, "ProjectGroupedDevice")
    assert "CheckCaptureAllowed(" in project
    assert "CheckGroupedProjectionTensors(" in project
    assert "routes->CheckMatches(input_bytes, plan->Groups())" in project
    assert "plan->CheckMatches(input_bytes, output)" in project
    assert "RecordStream(" in project
    assert "routes->GroupList()" in project
    assert "routes->Status()" in project
    assert "[routes, plan, launch_stream" in project
    assert "stream.stream(false)" in project
    assert "rtKernelLaunch(" in project
    for forbidden in (".cpu(", ".item", ".to(", "MakeTiling", "synchronize", "stream.stream()"):
        assert forbidden not in project


@pytest.fixture(scope="module")
def device_math(tmp_path_factory):
    """Compile actual ATen math/class, with only stream/capture query mocked."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for actual ATen sanitizer tests.")
    source = SOURCE.read_text()
    sanitizer = body(source, "SanitizeDeviceCounts")
    routes = source[
        source.index("class DeviceRoutes :") : source.index("c10::intrusive_ptr<DeviceRoutes> PrepareDeviceRoutes")
    ]
    directory = tmp_path_factory.mktemp("gs32-device-native")
    cpp = directory / "device.cpp"
    shared = directory / ("device.dylib" if sys.platform == "darwin" else "device.so")
    cpp.write_text(
        """
#include <ATen/ATen.h>
#include <torch/custom_class.h>
#include <torch/library.h>
#include <cstdint>
#include <utility>
int capture_state = 0;
uintptr_t current_stream = 100;
namespace c10_npu {
enum class CaptureStatus { None, Active, Invalidated };
CaptureStatus currentStreamCaptureStatusMayInitCtx() { return static_cast<CaptureStatus>(capture_state); }
struct NPUStream { void* stream(bool drain = true) const {
  TORCH_CHECK(!drain, "must not drain"); return reinterpret_cast<void*>(current_stream);
}};
NPUStream getCurrentNPUStream() { return {}; }
struct OptionalNPUGuard { explicit OptionalNPUGuard(c10::Device) {} };
}
constexpr int64_t kMaxProjectionRows = 128;
void CheckCaptureAllowed(const char* operation) """
        + body(source, "CheckCaptureAllowed")
        + "\nstd::pair<at::Tensor, at::Tensor> SanitizeDeviceCounts("
        "const at::Tensor& counts, int64_t rows, bool allow_padding = false) "
        + sanitizer
        + "\n"
        + routes
        + "\nat::Tensor DeviceRoutesStatus(const c10::intrusive_ptr<DeviceRoutes>& routes) "
        + body(source, "DeviceRoutesStatus")
        + """
c10::intrusive_ptr<DeviceRoutes> TestPrepare(const at::Tensor& counts, int64_t rows) {
  auto sanitized = SanitizeDeviceCounts(counts, rows);
  return c10::make_intrusive<DeviceRoutes>(rows, counts.numel(), counts.device(),
      std::move(sanitized.first), std::move(sanitized.second),
      c10_npu::getCurrentNPUStream().stream(false));
}
at::Tensor TestCounts(const c10::intrusive_ptr<DeviceRoutes>& routes) { return routes->GroupList().clone(); }
void TestMatches(const c10::intrusive_ptr<DeviceRoutes>& routes, const at::Tensor& anchor, int64_t groups) {
  routes->CheckMatches(anchor, groups);
}
void TestState(int64_t capture, int64_t stream) { capture_state = capture; current_stream = stream; }
std::vector<at::Tensor> TestSanitize(const at::Tensor& counts, int64_t rows, bool allow_padding) {
  auto result = SanitizeDeviceCounts(counts, rows, allow_padding);
  return {result.first, result.second};
}
TORCH_LIBRARY(gs32_device_test, m) {
  m.class_<DeviceRoutes>("DeviceRoutes");
  m.def("prepare(Tensor counts, int rows) -> __torch__.torch.classes.gs32_device_test.DeviceRoutes", &TestPrepare);
  m.def("counts(__torch__.torch.classes.gs32_device_test.DeviceRoutes routes) -> Tensor", &TestCounts);
  m.def("status(__torch__.torch.classes.gs32_device_test.DeviceRoutes routes) -> Tensor", &DeviceRoutesStatus);
  m.def("matches(__torch__.torch.classes.gs32_device_test.DeviceRoutes routes, Tensor anchor, int groups) -> ()",
        &TestMatches);
  m.def("state(int capture, int stream) -> ()", &TestState);
  m.def("sanitize(Tensor counts, int rows, bool allow_padding) -> Tensor[]", &TestSanitize);
}
"""
    )
    command = [compiler, "-std=c++17", "-shared", "-fPIC", "-O0", str(cpp), "-o", str(shared)]
    if sys.platform == "darwin":
        command.extend(["-arch", platform.machine()])
    else:
        command.append(f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}")
    command.extend(f"-I{path}" for path in include_paths())
    for path in library_paths():
        command.extend([f"-L{path}", f"-Wl,-rpath,{path}"])
    command.extend(["-ltorch", "-ltorch_cpu", "-lc10"])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    torch.ops.load_library(str(shared))
    return torch.ops.gs32_device_test


@pytest.mark.parametrize(
    "values,rows,valid",
    [
        ([1, 0, 2], 3, True),
        ([0, 0, 128], 128, True),
        ([128] * 256, 32768, True),
        ([0] * 32 + [1], 1, True),
        ([-1, 1, 3], 3, False),
        ([1, 0, 1], 3, False),
        ([1, 0, 3], 3, False),
        ([0, 0, 0], 3, False),
        ([129, 0, 0], 129, False),
        ([2**63 - 1, 2**63 - 1, 5], 3, False),
        ([-(2**63), 0, 3], 3, False),
    ],
)
def test_actual_aten_sanitizer_fails_closed_without_overflow(device_math, values, rows, valid):
    device_math.state(0, 100)
    original = torch.tensor(values, dtype=torch.int64)
    routes = device_math.prepare(original, rows)
    original.fill_(17)
    expected = torch.tensor(values, dtype=torch.int64) if valid else torch.zeros(len(values), dtype=torch.int64)
    assert torch.equal(device_math.counts(routes), expected)
    status = device_math.status(routes)
    assert status.dtype == torch.int32 and status.shape == (1,)
    assert status.tolist() == [int(valid)]
    status.fill_(1 - int(valid))
    assert device_math.status(routes).tolist() == [int(valid)], "Status must not expose mutable internal storage."


def test_device_route_status_capture_and_stream_contract(device_math):
    device_math.state(0, 100)
    routes = device_math.prepare(torch.tensor([1, 0, 2]), 3)
    device_math.state(1, 100)
    assert device_math.status(routes).tolist() == [1]
    device_math.matches(routes, torch.empty(3, 64), 3)
    for rows, groups in ((2, 3), (3, 2)):
        with pytest.raises(RuntimeError, match="routes"):
            device_math.matches(routes, torch.empty(rows, 64), groups)
    device_math.state(2, 100)
    with pytest.raises(RuntimeError, match="capture"):
        device_math.status(routes)
    device_math.state(0, 200)
    with pytest.raises(RuntimeError, match="stream"):
        device_math.status(routes)
    with pytest.raises(RuntimeError, match="stream"):
        device_math.matches(routes, torch.empty(3, 64), 3)
    device_math.state(0, 100)


@pytest.mark.parametrize("groups", [32, 64, 128])
@pytest.mark.parametrize("active", [0, 1, 128])
def test_ep_native_padding_accepts_partial_and_empty_rank(device_math, groups, active):
    """Resident expert bank sizes for 8/4/2 ranks; unused capacity stays unused."""
    counts = torch.zeros(groups, dtype=torch.int64)
    counts[-1] = active
    safe, status = device_math.sanitize(counts, 256, True)
    assert status.tolist() == [1]
    assert torch.equal(safe, counts)
    strict, strict_status = device_math.sanitize(counts, 256, False)
    assert strict_status.tolist() == [0]
    assert torch.equal(strict, torch.zeros_like(counts))


@pytest.mark.parametrize("counts", [[129, 0], [-1, 2], [2**63 - 1, 1], [128, 128]])
def test_ep_native_padding_rejects_invalid_or_over_capacity(device_math, counts):
    safe, status = device_math.sanitize(torch.tensor(counts), 128, True)
    assert status.tolist() == [0]
    assert torch.equal(safe, torch.zeros(len(counts), dtype=torch.int64))


def test_ep_native_padding_schema_and_zero_fill_are_explicit():
    source = SOURCE.read_text()
    assert "prepare_device_routes(Tensor anchor, Tensor counts, bool allow_padding=False)" in source
    prepare = body(source, "PrepareDeviceRoutes")
    assert "SanitizeDeviceCounts(counts, anchor.size(0), allow_padding)" in prepare
    project = body(source, "ProjectGroupedDevice")
    assert "if (routes->AllowsPadding())" in project
    assert project.index("CheckGroupedProjectionTensors(") < project.index("output.zero_()")
    assert project.index("output.zero_()") < project.index("OpCommand::RunOpApi(")
