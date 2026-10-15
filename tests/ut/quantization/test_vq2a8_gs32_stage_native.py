# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU/source contracts for the eager GS32 DownW2 stage adapter."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import regex as re

from tools import build_vq2a8_gs32_stage as build

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "csrc/vq2a8_gs32_stage/torch_binding.cpp"
MIXED = REPO / "csrc/mixed_w2w4_moe_ffn"


def source_text():
    return SOURCE.read_text(encoding="utf-8")


def function(source, name):
    marker = f"{name}("
    start = source.index(marker)
    begin = source.index("{", start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


def write_elf(path):
    path.write_bytes(b"\x7fELF" + bytes(range(32)))
    return path


def test_binding_reuses_only_the_existing_downw2_stage_contract():
    source = source_text()
    assert '#include "../mixed_w2w4_moe_ffn/components/down_w2/down_w2_tiling_builder.h"' in source
    assert '#include "../mixed_w2w4_moe_ffn/mixed_tiling.h"' in source
    assert "constexpr uint64_t kGroupCount = 1;" in source
    assert "constexpr uint32_t kKernelCoreCount = 32;" in source
    assert "static_assert(kMixedStageDownW2 == 4" in source
    assert "static_cast<uint64_t>(2 * k)" in function(source, "Project")
    assert "static_cast<uint64_t>(n)" in function(source, "Project")
    assert "rtKernelLaunch(registered->stub, kKernelCoreCount" in function(source, "Project")


def test_make_tiling_uses_the_original_builder_and_exact_1600_byte_header():
    source = source_text()
    body = function(source, "MakeTiling")
    assert "BuildDownW2TilingBlob" in body
    assert "kMixedTilingBytes" in body
    assert "kMixedStageDownW2" in body
    assert "kMixedDownW2TilingOffset" in body
    assert "std::memcpy(bytes + kMixedDownW2TilingOffset" in body
    mixed = (MIXED / "mixed_tiling.h").read_text(encoding="utf-8")
    assert "constexpr uint64_t kMixedTilingBytes = 1600;" in mixed
    assert "constexpr uint64_t kMixedDownW2TilingOffset = 832;" in mixed


def test_resident_expert_views_are_aligned_and_storage_bounded():
    checks = function(source_text(), "CheckNpuTensor")
    assert "storage_offset() == 0" not in checks
    assert "tensor.storage().nbytes()" in checks
    assert "tensor.nbytes()" in checks
    assert "% kRequiredAlignment == 0" in checks
    assert "tensor.storage().data_ptr()" in function(source_text(), "RecordStream")


def test_project_is_fail_closed_on_geometry_dtype_layout_alias_and_metadata():
    body = function(source_text(), "Project")
    for dtype in ("at::kByte", "at::kLong", "at::kInt", "at::kBFloat16"):
        assert dtype in body
    for shape in (
        "input_scale must have shape [M,K/32]",
        "packed_weight must have shape [1,N/32,K/16,16,8]",
        "pair_lut must have shape [1,N/32,K/32,32]",
        "row_scale must have shape [1,N]",
        "group_list must have shape [1] and contain M",
        "route_mask must have shape [1] and contain the all-W2 value zero",
        "output must have shape [M,N]",
    ):
        assert shape in body
    assert "1<=M<=128" in body
    assert "group_list.data_ptr<int64_t>()[0] == m" in body
    assert "route_mask.data_ptr<int32_t>()[0] == 0" in body
    assert "std::memcmp(tiling.data_ptr(), expected_tiling.data_ptr(), kMixedTilingBytes)" in body
    assert "CheckNoAlias(npu_owners)" in body
    assert "CheckNoAlias(cpu_metadata)" in body
    checks = function(source_text(), "CheckNpuTensor") + function(source_text(), "CheckCpuMetadata")
    assert "is_contiguous()" in checks
    assert "storage_offset() == 0" in checks
    assert "% kRequiredAlignment == 0" in checks


def test_cpu_metadata_is_validated_then_copied_and_owned_for_the_queued_launch():
    body = function(source_text(), "Project")
    assert "group_list.to(input_bytes.device(), at::kLong, false, true)" in body
    assert "route_mask.to(input_bytes.device(), at::kInt, false, true)" in body
    assert "tiling.to(input_bytes.device(), at::kByte, false, true)" in body
    assert "device_group_list.data_ptr()" in body
    assert "device_route_mask.data_ptr()" in body
    assert "device_tiling.data_ptr()" in body
    assert "device_group_list, device_route_mask, device_tiling" in body
    assert "RecordStream(" in body


def test_raw_stream_is_resolved_before_queue_callback_and_graph_is_rejected():
    body = function(source_text(), "Project")
    before_queue, callback = body.split("at_npu::native::OpCommand::RunOpApi(", 1)
    assert "CheckNotCapturing(" in before_queue
    assert before_queue.index("OptionalNPUGuard") < before_queue.index("CheckNotCapturing(")
    assert before_queue.index("CheckNotCapturing(") < before_queue.index("CheckNpuTensor(")
    assert before_queue.index("CheckNotCapturing(") < before_queue.index("group_list.to(")
    assert "const auto launch_stream = stream.stream();" in before_queue
    assert "getCurrentNPUStream" not in callback
    assert ".stream()" not in callback
    assert "[registered, launch_stream" in callback
    assert "false);" in callback
    for forbidden in ("synchronize", ".item(", "rtDeviceReset", "aclrtResetDevice"):
        assert forbidden not in body


def test_capture_guard_uses_exported_status_api_without_hidden_context_helpers_or_fallback():
    # Comments may explain the incompatible APIs; executable calls must not use them.
    source = re.sub(r"//[^\n]*|/\*.*?\*/", "", source_text(), flags=re.DOTALL)
    body = function(source, "CheckNotCapturing")
    assert "c10_npu::currentStreamCaptureStatusMayInitCtx()" in body
    assert "TORCH_CHECK(status == c10_npu::CaptureStatus::None" in body
    assert "static_cast<int>(status)" in body
    assert body.index("currentStreamCaptureStatusMayInitCtx()") < body.index("TORCH_CHECK(")
    # None-only acceptance also rejects Invalidated, unlike an Active-only boolean.
    assert not re.search(r"\b(?:if|try|catch|return)\b", body)
    for forbidden in (
        "assertNotCapturing",
        "currentStreamCaptureStatus",
        "IsContextInitialized",
        "isStreamCapturingMayInitCtx",
        "dlsym",
    ):
        assert not re.search(rf"\b{forbidden}\s*\(", source)


def test_registration_is_startup_only_idempotent_and_retains_elf_bytes():
    source = source_text()
    body = function(source, "Initialize")
    assert "CheckNotCapturing(" in body
    assert body.index("CheckNotCapturing(") < body.index("g_registration_mutex")
    assert body.index("CheckNotCapturing(") < body.index("rtDevBinaryRegister")
    assert "g_registration_mutex" in body
    assert "g_registered_kernel->path == binary_path" in body
    assert "ReadElfBinary(binary_path)" in body
    assert "rtDevBinaryRegister" in body
    assert "rtFunctionRegister" in body
    assert "g_registered_kernel = std::move(state)" in body
    assert "rtDeviceReset" not in source
    assert "rtStreamSynchronize" not in source
    assert "#if __has_include(<runtime/rt.h>)" in source


def test_torch_schema_is_exact_and_output_is_mutable():
    source = source_text()
    assert 'm.def("initialize(str binary) -> ()"' in source
    assert 'm.def("make_tiling(int k, int n) -> Tensor"' in source
    assert "Tensor(a!) output) -> ()" in source
    assert "TORCH_LIBRARY_IMPL(vq2a8_gs32_stage, PrivateUse1" in source


def test_build_tool_validates_existing_mixed_elf_and_is_host_only(tmp_path):
    kernel = write_elf(tmp_path / "mixed.o")
    evidence = build.kernel_evidence(kernel)
    assert evidence["present"] is True
    assert evidence["bytes"] == kernel.stat().st_size
    assert len(evidence["sha256"]) == 64
    project = build.cmake_project()
    assert "add_library(vq2a8_gs32_stage SHARED" in project
    assert "ascendc_library" not in project
    assert "torch_npu ascendcl runtime" in project
    assert 'target_link_options(vq2a8_gs32_stage PRIVATE "LINKER:-z,defs")' in project


def test_build_never_writes_inside_the_imported_kernel_directory():
    args = build.parse_args(["--kernel-binary", "unused.o", "--build-dir", str(MIXED / "build"), "--dry-run"])
    with pytest.raises(ValueError, match="out-of-source"):
        build.validate_options(args)


def test_build_dry_run_never_claims_device_or_graph_acceptance(tmp_path, capsys):
    kernel = write_elf(tmp_path / "mixed.o")
    result = build.main(
        [
            "--kernel-binary",
            str(kernel),
            "--build-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    assert result == 0
    output = capsys.readouterr().out
    assert '"host_extension_only": true' in output
    assert '"kernel_is_built_by_this_tool": false' in output
    assert '"device_execution_verified": false' in output
    assert '"host_schema_verified": false' in output
    assert '"graph_api_available": false' in output
    assert '"graph_execution_verified": false' in output
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("payload", [b"", b"NOPE", b"\x7fEL"])
def test_build_rejects_missing_or_non_elf_kernel(tmp_path, payload):
    kernel = tmp_path / "mixed.o"
    kernel.write_bytes(payload)
    with pytest.raises(ValueError, match="ELF"):
        build.kernel_evidence(kernel)


@pytest.mark.parametrize(
    "fault", [None, "missing_op", "wrong_class", "wrong_output_alias", "missing_class", "old_padding_schema"]
)
def test_build_schema_check_requires_typed_device_routes_and_mutable_output(fault):
    schemas = dict(build.REQUIRED_DEVICE_SCHEMAS)
    if fault == "missing_op":
        del schemas["prepare_device_routes"]
    elif fault == "wrong_class":
        schemas["project_grouped_device"] = schemas["project_grouped_device"].replace(
            ".DeviceRoutes ", ".GroupedRoutes "
        )
    elif fault == "wrong_output_alias":
        schemas["project_grouped_device"] = schemas["project_grouped_device"].replace("Tensor(a!)", "Tensor")
    elif fault == "old_padding_schema":
        schemas["prepare_device_routes"] = schemas["prepare_device_routes"].replace(", bool allow_padding=False", "")

    def find_schema(name, overload):
        assert overload == ""
        key = name.removeprefix("vq2a8_gs32_stage::")
        if key not in schemas:
            raise RuntimeError("missing op")
        return SimpleNamespace(schema=lambda: schemas[key])

    classes = {} if fault == "missing_class" else {"DeviceRoutes": object()}
    fake_torch = SimpleNamespace(
        classes=SimpleNamespace(vq2a8_gs32_stage=SimpleNamespace(**classes)),
        _C=SimpleNamespace(_dispatch_find_schema_or_throw=find_schema),
    )
    if fault:
        with pytest.raises(RuntimeError, match="schema|DeviceRoutes"):
            build.verify_registered_device_schemas(fake_torch)
    else:
        assert build.verify_registered_device_schemas(fake_torch) == schemas


@pytest.mark.parametrize("exit_code", [0, 7])
def test_build_schema_check_is_a_separate_host_only_process(tmp_path, monkeypatch, exit_code):
    library, log = tmp_path / "adapter.so", tmp_path / "schema.log"
    library.write_bytes(b"extension")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stdout"].write("schema child output\n")
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(build.subprocess, "run", run)
    if exit_code:
        with pytest.raises(RuntimeError, match="schema.*7|schema.log"):
            build.verify_library_schemas(library, log)
    else:
        build.verify_library_schemas(library, log)
    command, options = calls[0]
    assert command[0] == build.sys.executable and command[1] == "-c"
    assert "verify_loaded_library" in command[2]
    assert command[-1] == str(library.resolve())
    assert options["timeout"] == build.SCHEMA_CHECK_TIMEOUT_SECONDS
    assert options["stderr"] is subprocess.STDOUT and options["check"] is False
    assert log.read_text() == "schema child output\n"


def test_build_load_check_only_imports_and_queries_registration(tmp_path, monkeypatch):
    library = tmp_path / "adapter.so"
    library.write_bytes(b"extension")
    calls = []
    fake_torch = SimpleNamespace(ops=SimpleNamespace(load_library=lambda path: calls.append(path)))
    monkeypatch.setitem(build.sys.modules, "torch", fake_torch)
    monkeypatch.setitem(build.sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(build, "verify_registered_device_schemas", lambda module: {"checked": module is fake_torch})
    assert build.verify_loaded_library(str(library)) == {"checked": True}
    assert calls == [str(library.resolve())]


@pytest.mark.parametrize("schema_error", [False, True])
def test_build_pass_requires_schema_verification_and_does_not_claim_graph_execution(
    tmp_path, monkeypatch, schema_error
):
    kernel = write_elf(tmp_path / "mixed.o")
    cann, driver, out = tmp_path / "cann", tmp_path / "driver", tmp_path / "out"
    (cann / "include").mkdir(parents=True)
    (driver / "lib64/driver").mkdir(parents=True)
    args = build.parse_args(
        ["--kernel-binary", str(kernel), "--cann", str(cann), "--driver", str(driver), "--build-dir", str(out)]
    )
    monkeypatch.setattr(build.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        build.importlib.util, "find_spec", lambda _: SimpleNamespace(submodule_search_locations=[str(tmp_path)])
    )
    fake_torch = SimpleNamespace(__version__="test", utils=SimpleNamespace(cmake_prefix_path=str(tmp_path)))
    monkeypatch.setitem(build.sys.modules, "torch", fake_torch)

    def compile_only(command, **kwargs):
        (out / build.LIBRARY_NAME).write_bytes(b"compiled library")
        return SimpleNamespace(returncode=0)

    verified = []

    def schemas(library, log):
        verified.append((library, log))
        if schema_error:
            raise RuntimeError("schema check failed")

    monkeypatch.setattr(build.subprocess, "run", compile_only)
    monkeypatch.setattr(build, "verify_library_schemas", schemas)
    if schema_error:
        with pytest.raises(RuntimeError, match="schema"):
            build.build(args)
    else:
        assert build.build(args) == 0
    assert len(verified) == 1
    manifest = json.loads((out / "build-manifest.json").read_text())
    assert manifest["host_schema_verified"] is (not schema_error)
    assert manifest["graph_api_available"] is (not schema_error)
    assert manifest["graph_execution_verified"] is False
    assert manifest["device_execution_verified"] is False
    assert manifest["status"] == ("failed" if schema_error else "built")
