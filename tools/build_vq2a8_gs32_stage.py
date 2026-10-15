#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build only the GS32 DownW2 Torch host adapter around an existing mixed ELF."""

from __future__ import annotations

# Keep tools/bisect out of the direct-script import path.
# ruff: noqa: E402
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import hashlib
import importlib.util
import json
import platform
import re
import shutil
import subprocess
from pathlib import Path

from tools.vq2a8_live_log import LiveChildLog

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "csrc/vq2a8_gs32_stage"
BINDING = SOURCE / "torch_binding.cpp"
MIXED_SOURCE = REPO / "csrc/mixed_w2w4_moe_ffn"
TARGET = "vq2a8_gs32_stage"
LIBRARY_NAME = f"lib{TARGET}.so"
KERNEL_NAME = "mixed_w2w4_moe_ffn.o"
GENERATED_SOURCE_NAME = "host-extension-source"
SCHEMA_CHECK_TIMEOUT_SECONDS = 60
REQUIRED_DEVICE_SCHEMAS = (
    (
        "prepare_device_routes",
        "vq2a8_gs32_stage::prepare_device_routes(Tensor anchor, Tensor counts, bool allow_padding=False) -> "
        "__torch__.torch.classes.vq2a8_gs32_stage.DeviceRoutes",
    ),
    (
        "device_routes_status",
        "vq2a8_gs32_stage::device_routes_status("
        "__torch__.torch.classes.vq2a8_gs32_stage.DeviceRoutes routes) -> Tensor",
    ),
    (
        "project_grouped_device",
        "vq2a8_gs32_stage::project_grouped_device(Tensor input_bytes, Tensor input_scale, "
        "Tensor packed_weight, Tensor pair_lut, Tensor row_scale, "
        "__torch__.torch.classes.vq2a8_gs32_stage.DeviceRoutes routes, "
        "__torch__.torch.classes.vq2a8_gs32_stage.GroupedProjectionPlan plan, Tensor(a!) output) -> ()",
    ),
)


def verify_registered_device_schemas(torch_module):
    """Read registration only; never construct routes or execute an NPU op."""
    try:
        _ = torch_module.classes.vq2a8_gs32_stage.DeviceRoutes
    except (AttributeError, RuntimeError) as error:
        raise RuntimeError("GS32 schema check requires the registered DeviceRoutes class.") from error
    actual = {}
    for name, expected in REQUIRED_DEVICE_SCHEMAS:
        try:
            schema = torch_module._C._dispatch_find_schema_or_throw(f"vq2a8_gs32_stage::{name}", "")
        except RuntimeError as error:
            raise RuntimeError(f"GS32 schema check: missing {name}; rebuild the bridge .so.") from error
        actual[name] = str(schema.schema())
        if " ".join(actual[name].split()) != " ".join(expected.split()):
            raise RuntimeError(f"GS32 schema mismatch for {name}: expected {expected}, found {actual[name]}")
    return actual


def verify_loaded_library(library):
    """Run in a fresh process so an earlier .so cannot mask stale registration."""
    import torch
    import torch_npu  # noqa: F401

    torch.ops.load_library(str(Path(library).resolve(strict=True)))
    return verify_registered_device_schemas(torch)


def verify_library_schemas(library, log):
    # Import/load/query only: no initialize(), device selection, allocation,
    # kernel launch or graph capture. This can run while an NPU is occupied.
    command = [
        sys.executable,
        "-c",
        "import json, sys\n"
        "from tools.build_vq2a8_gs32_stage import verify_loaded_library\n"
        "print(json.dumps(verify_loaded_library(sys.argv[1]), sort_keys=True))\n",
        str(library.resolve()),
    ]
    print(f"GS32_STAGE_BUILD_STAGE=schema LOG={log}", flush=True)
    with log.open("w", encoding="utf-8") as stream, LiveChildLog(log, "gs32-stage-schema"):
        result = subprocess.run(
            command,
            cwd=REPO,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=SCHEMA_CHECK_TIMEOUT_SECONDS,
        )
    if result.returncode:
        raise RuntimeError(f"schema check failed ({result.returncode}); inspect {log}")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    required = (
        BINDING,
        MIXED_SOURCE / "mixed_tiling.h",
        MIXED_SOURCE / "components/down_w2/down_w2_tiling.h",
        MIXED_SOURCE / "components/down_w2/down_w2_tiling_builder.h",
    )
    missing = [path.relative_to(REPO).as_posix() for path in required if not path.is_file()]
    if missing:
        raise ValueError("GS32 stage sources missing: " + ", ".join(missing))
    return {path.relative_to(REPO).as_posix(): sha256(path) for path in required}


def kernel_evidence(path, *, require=True):
    path = path.resolve()
    if not path.is_file():
        if require:
            raise ValueError(f"Existing mixed kernel binary not found: {path}")
        return {"path": str(path), "present": False}
    size = path.stat().st_size
    with path.open("rb") as stream:
        magic = stream.read(4)
    if size < 4 or magic != b"\x7fELF":
        raise ValueError(f"Mixed kernel binary must be a non-empty ELF object: {path}")
    return {"path": str(path), "present": True, "bytes": size, "sha256": sha256(path)}


def validate_options(args):
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    directory = args.build_dir.resolve()
    if (
        directory in (REPO, SOURCE, MIXED_SOURCE)
        or directory in REPO.parents
        or SOURCE in directory.parents
        or MIXED_SOURCE in directory.parents
    ):
        raise ValueError("Choose a dedicated out-of-source build directory")
    generated_source = directory / GENERATED_SOURCE_NAME
    cmake_cache = directory / "cmake-build/CMakeCache.txt"
    if cmake_cache.is_file():
        match = re.search(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$", cmake_cache.read_text(), re.MULTILINE)
        if match and Path(match.group(1).strip()).resolve() != generated_source.resolve():
            raise ValueError("Build directory belongs to another project; do not overwrite it")
    return directory


def cmake_project():
    binding = BINDING.resolve().as_posix().replace('"', '\\"')
    return f"""# Generated by tools/build_vq2a8_gs32_stage.py; do not edit.
cmake_minimum_required(VERSION 3.16)
project(vq2a8_gs32_stage LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "Build the GS32 stage adapter on Linux with CANN 9.1 and torch_npu installed")
endif()
if(NOT IS_DIRECTORY "${{ASCEND_HOME_PATH}}" OR NOT IS_DIRECTORY "${{TORCH_NPU_PATH}}")
  message(FATAL_ERROR "ASCEND_HOME_PATH and TORCH_NPU_PATH must be existing directories")
endif()
find_package(Torch REQUIRED)
separate_arguments(TORCH_FLAGS NATIVE_COMMAND "${{TORCH_CXX_FLAGS}}")
add_library({TARGET} SHARED "{binding}")
target_compile_options({TARGET} PRIVATE ${{TORCH_FLAGS}})
target_include_directories({TARGET} PRIVATE
  ${{TORCH_INCLUDE_DIRS}}
  "${{TORCH_NPU_PATH}}/include"
  "${{ASCEND_HOME_PATH}}/include")
target_link_directories({TARGET} PRIVATE
  "${{TORCH_NPU_PATH}}/lib"
  "${{ASCEND_HOME_PATH}}/lib64"
  "${{ASCEND_HOME_PATH}}/aarch64-linux/devlib"
  "${{ASCEND_DRIVER_PATH}}/lib64/common"
  "${{ASCEND_DRIVER_PATH}}/lib64/driver")
target_link_libraries({TARGET} PRIVATE ${{TORCH_LIBRARIES}} torch_npu ascendcl runtime)
# Reject unresolved adapter symbols at link time instead of passing a build
# whose library later fails to load through torch.ops.load_library.
target_link_options({TARGET} PRIVATE "LINKER:-z,defs")
set(STAGE_BUILD_RPATH
  "${{TORCH_NPU_PATH}}/lib;${{ASCEND_HOME_PATH}}/lib64"
  "${{ASCEND_DRIVER_PATH}}/lib64/common;${{ASCEND_DRIVER_PATH}}/lib64/driver")
set_target_properties({TARGET} PROPERTIES
  BUILD_RPATH "${{STAGE_BUILD_RPATH}}"
  LIBRARY_OUTPUT_DIRECTORY "${{STAGE_OUTPUT_DIRECTORY}}")
"""


def cmake_commands(args, directory, torch_npu_path, torch_cmake):
    generated_source = directory / GENERATED_SOURCE_NAME
    cmake_build = directory / "cmake-build"
    return [
        [
            "cmake",
            "-S",
            str(generated_source),
            "-B",
            str(cmake_build),
            f"-DASCEND_HOME_PATH={args.cann.resolve()}",
            f"-DASCEND_DRIVER_PATH={args.driver.resolve()}",
            f"-DTORCH_NPU_PATH={torch_npu_path}",
            f"-DCMAKE_PREFIX_PATH={torch_cmake}",
            f"-DSTAGE_OUTPUT_DIRECTORY={directory}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        ["cmake", "--build", str(cmake_build), "--target", TARGET, "-j", str(args.jobs), "--verbose"],
    ]


def _save(path, report):
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def build(args):
    directory = validate_options(args)
    evidence = kernel_evidence(args.kernel_binary, require=not args.dry_run)
    report = {
        "status": "planned" if args.dry_run else "building",
        "implementation": "vq2a8_gs32_stage",
        "abi_version": 1,
        "source_sha256": source_hashes(),
        "kernel_binary": evidence,
        "kernel_is_built_by_this_tool": False,
        "host_extension_only": True,
        "python": sys.executable,
        "cann_9_1_compile_verified": False,
        "device_execution_verified": False,
        "host_schema_verified": False,
        "graph_api_available": False,
        "graph_execution_verified": False,
        "model_execution_verified": False,
        "performance_verified": False,
        "default_model_backend": "unchanged",
    }
    if args.dry_run:
        report["commands"] = cmake_commands(
            args,
            directory,
            "<selected-Python-torch_npu>",
            "<selected-Python-torch-cmake>",
        )
        report["deployed_kernel"] = str(directory / KERNEL_NAME)
        print(json.dumps(report, indent=2))
        return 0
    if platform.system() != "Linux":
        raise RuntimeError("Actual compilation requires Linux CANN/torch_npu; use --dry-run on Windows")

    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "build-manifest.json"
    _save(manifest, report)
    try:
        cann = args.cann.resolve(strict=True)
        driver = args.driver.resolve(strict=True)
        if not (cann / "include").is_dir():
            raise RuntimeError(f"CANN headers missing under {cann}")
        if not (driver / "lib64/driver").is_dir():
            raise RuntimeError(f"Ascend driver runtime libraries missing under {driver}")

        import torch

        spec = importlib.util.find_spec("torch_npu")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("torch_npu is not installed in the selected Python")
        torch_npu_path = Path(next(iter(spec.submodule_search_locations))).resolve()
        report["torch"] = torch.__version__
        report["torch_npu_path"] = str(torch_npu_path)

        generated_source = directory / GENERATED_SOURCE_NAME
        generated_source.mkdir(parents=True, exist_ok=True)
        generated_cmake = generated_source / "CMakeLists.txt"
        generated_cmake.write_text(cmake_project(), encoding="utf-8")
        report["generated_cmake_sha256"] = sha256(generated_cmake)
        report["commands"] = cmake_commands(
            args,
            directory,
            torch_npu_path,
            torch.utils.cmake_prefix_path,
        )
        _save(manifest, report)

        for name, command in zip(("configure", "compile"), report["commands"]):
            report["current_step"] = name
            _save(manifest, report)
            log = directory / f"{name}.log"
            print(f"GS32_STAGE_BUILD_STAGE={name} LOG={log}", flush=True)
            print("COMMAND=" + json.dumps(command), flush=True)
            with log.open("w", encoding="utf-8") as stream, LiveChildLog(log, f"gs32-stage-{name}"):
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(f"{name} failed ({result.returncode}); inspect {log}")

        library = directory / LIBRARY_NAME
        if not library.is_file() or not library.stat().st_size:
            raise RuntimeError(f"Host extension was not produced: {library}")
        if source_hashes() != report["source_sha256"]:
            raise RuntimeError("Adapter sources changed while compiling; rebuild before validation")
        if kernel_evidence(args.kernel_binary) != evidence:
            raise RuntimeError("Mixed kernel binary changed while compiling; rebuild before validation")

        report["current_step"] = "schema"
        _save(manifest, report)
        verify_library_schemas(library, directory / "schema.log")

        deployed_kernel = directory / KERNEL_NAME
        if args.kernel_binary.resolve() != deployed_kernel.resolve():
            shutil.copyfile(args.kernel_binary, deployed_kernel)
        report.update(
            status="built",
            library=str(library),
            library_sha256=sha256(library),
            deployed_kernel=str(deployed_kernel),
            deployed_kernel_sha256=sha256(deployed_kernel),
            # A successful target build verifies only declarations/linkage.
            cann_9_1_compile_verified=cann.name == "cann-9.1.0",
            cann_compile_verified=True,
            cann_path=str(cann),
            host_schema_verified=True,
            required_device_schemas=dict(REQUIRED_DEVICE_SCHEMAS),
            graph_api_available=True,
        )
        report.pop("current_step", None)
        _save(manifest, report)
        print(f"GS32_STAGE_BUILD=PASS LIBRARY={library} KERNEL={deployed_kernel} REPORT={manifest}", flush=True)
        print(
            "HOST_SCHEMA_VERIFIED=True GRAPH_API_AVAILABLE=True "
            "DEVICE_EXECUTION_VERIFIED=False GRAPH_EXECUTION_VERIFIED=False",
            flush=True,
        )
        return 0
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        _save(manifest, report)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-binary", type=Path, required=True, help="Existing mixed_w2w4_moe_ffn.o ELF")
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann-9.1.0"))
    parser.add_argument("--driver", type=Path, default=Path("/usr/local/Ascend/driver"))
    parser.add_argument("--build-dir", type=Path, default=REPO / "build/vq2a8-gs32-stage")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="print commands/hashes; do not write or compile")
    return parser.parse_args(argv)


def main(argv=None):
    try:
        return build(parse_args(argv))
    except (OSError, RuntimeError, ValueError, ImportError, subprocess.TimeoutExpired) as exc:
        print(f"GS32_STAGE_BUILD=FAIL ERROR={exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
