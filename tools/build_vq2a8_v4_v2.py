#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the standalone resident VQ2A8 operator with the installed CANN toolchain."""

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
import subprocess
from pathlib import Path

import regex as re

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "csrc/vq2a8_ascendc_v4_v2"
TARGET = "vq2a8_ascendc_v4_v2"
LIBRARY_NAME = f"lib{TARGET}.so"


ASCENDC_CMAKE_CANDIDATES = (
    "tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake",
    "compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake",
    "ascendc_devkit/tikcpp/samples/cmake/ascendc.cmake",
    "aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake",
    "x86_64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake",
    "aarch64-linux/asc/cmake/ascendc.cmake",
    "x86_64-linux/asc/cmake/ascendc.cmake",
    "tools/ascendc_kernel_cmake/ascendc.cmake",
    "cmake/asc/ascendc.cmake",
)

COMPILER_PREFIXES = (
    "aarch64-linux/ccec_compiler/bin",
    "x86_64-linux/ccec_compiler/bin",
    "compiler/ccec_compiler/bin",
    "tools/ccec_compiler/bin",
    "ascendc_devkit/ccec_compiler/bin",
    "ccec_compiler/bin",
    "bin",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def discover_cmake(cann: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        candidate = explicit.resolve(strict=True)
        if not candidate.is_file() or candidate.name != "ascendc.cmake":
            raise ValueError("--ascendc-cmake must select an installed ascendc.cmake file")
        return candidate
    for name in ASCENDC_CMAKE_CANDIDATES:
        candidate = cann / name
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(
        f"CANN ascendc.cmake was not found under {cann}. "
        "Supply --ascendc-cmake with the installed SDK file; do not download a mismatched recipe."
    )


def discover_compiler(cann: Path) -> Path:
    for prefix in COMPILER_PREFIXES:
        for name in ("bisheng", "ccec"):
            candidate = cann / prefix / name
            if candidate.is_file():
                return candidate.resolve()
    raise RuntimeError(f"CANN bisheng/ccec was not found under {cann}")


def discover_soc_config(cann: Path, soc: str) -> Path:
    # npu-smi may report only the chip family (e.g. Ascend950DT), while
    # CANN requires a full target such as Ascend950DT_9582. Never guess a SKU
    # when the selected SDK contains more than one variant of that family.
    configs: dict[str, Path] = {}
    for prefix in (
        "aarch64-linux/data/platform_config",
        "x86_64-linux/data/platform_config",
        "data/platform_config",
        "compiler/data/platform_config",
    ):
        directory = cann / prefix
        if directory.is_dir():
            for path in sorted(directory.glob("*.ini")):
                if path.is_file():
                    # The entry name is the CANN target even if the INI is a
                    # symlink to a shared config with a different basename.
                    configs.setdefault(path.stem.casefold(), directory.resolve() / path.name)
    requested = soc.casefold()
    if requested in configs:
        return configs[requested]
    if "_" not in requested:
        candidates = [path for name, path in sorted(configs.items()) if name.startswith(requested + "_")]
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            names = ", ".join(path.stem for path in candidates)
            raise RuntimeError(
                f"SoC family {soc!r} matches multiple installed CANN targets: {names}. "
                "Pass the full hardware model with --soc."
            )
    available = ", ".join(path.stem for name, path in sorted(configs.items()) if name.startswith("ascend950"))
    raise RuntimeError(
        f"SoC {soc!r} is absent from CANN platform_config under {cann}. "
        f"Installed Ascend950 targets: {available or '(none)'}. "
        "Pass the full hardware model with --soc; do not substitute another SoC."
    )


def validate_soc_config(path: Path) -> dict[str, str]:
    values = dict(re.findall(r"^\s*([A-Za-z0-9_]+)\s*=\s*([^\r\n#;]+)", path.read_text(), re.MULTILINE))
    selected = {
        key: value.strip()
        for key, value in values.items()
        if key
        in {
            "SoC_version",
            "Short_SoC_version",
            "CCEC_AIC_version",
            "CCEC_VECTOR_version",
            "cube_core_cnt",
            "vector_core_cnt",
        }
    }
    # Different CANN patch levels spell the arch values differently. An explicit
    # non-c310 architecture is a hard error; absent metadata is recorded, not guessed.
    for key in ("CCEC_AIC_version", "CCEC_VECTOR_version"):
        if key in selected and "310" not in selected[key]:
            raise RuntimeError(f"{path}: {key}={selected[key]} is not an Ascend950 c310 target")
    if "cube_core_cnt" in selected and int(selected["cube_core_cnt"]) <= 0:
        raise RuntimeError("The selected SoC has no AIC cores")
    return selected


def toolchain_probe(cann: Path, soc: str, cmake_file: Path) -> dict:
    """Read SDK provenance and run compiler identification only; no device init."""
    compiler = discover_compiler(cann)
    soc_file = discover_soc_config(cann, soc)
    soc_metadata = validate_soc_config(soc_file)
    identification = {}
    for label, flag in (("version", "--version"), ("help", "--help")):
        result = subprocess.run([str(compiler), flag], capture_output=True, text=True, timeout=30, check=False)
        if result.returncode:
            raise RuntimeError(f"Compiler {flag} failed ({result.returncode}): {result.stderr[-2000:]}")
        identification[label] = (result.stdout + result.stderr)[-32768:]
    recipe_files = [cmake_file]
    for subdir in (cmake_file.parent, cmake_file.parent / "legacy_modules"):
        for name in ("function.cmake", "bisheng_intf.cmake", "util/extract_host_stub.py", "util/merge_device_obj.py"):
            candidate = subdir / name
            if candidate.is_file():
                recipe_files.append(candidate)
    recipe_text = "\n".join(p.read_text(errors="replace") for p in recipe_files)
    if "ascendc_library" not in recipe_text:
        raise RuntimeError("The selected SDK recipe does not expose ascendc_library")
    # The actual compiler checks every used API in the next build step. This
    # textual check prevents selecting an old c220-only SDK without guessing flags.
    explicit_mix = "KERNEL_TYPE_MIX_AIC_1_2" in recipe_text
    c310 = "c310" in recipe_text
    return {
        "compiler": str(compiler),
        "compiler_identification": identification,
        "soc": soc_file.stem,
        "soc_config": str(soc_file),
        "soc_config_sha256": sha256(soc_file),
        "soc_metadata": soc_metadata,
        "recipe_sha256": {str(p): sha256(p) for p in sorted(set(recipe_files))},
        "recipe_mentions_c310": c310,
        "recipe_mentions_explicit_mix_1_2": explicit_mix,
        "api_compilation_verified": False,
    }


def source_hashes(source=SOURCE):
    required = (
        "CMakeLists.txt",
        "kernel.cpp",
        "torch_binding.cpp",
        "resident_select.cpp",
        "resident_prepare.cpp",
        "activation_kernel.cpp",
        "activation_binding.cpp",
        "activation_launch.h",
        "validity_binding.cpp",
        "validity_kernel.cpp",
        "validity_launch.h",
        "layout.h",
        "launch.h",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise ValueError("Candidate sources missing: " + ", ".join(missing))
    return {
        path.relative_to(REPO).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.rglob("*"))
        if path.is_file()
        and not {"tests", "build", "__pycache__"}.intersection(path.relative_to(source).parts)
        and (path.suffix in {".h", ".cpp", ".cmake"} or path.name == "CMakeLists.txt")
    }


def validate_options(args):
    if not re.fullmatch(r"Ascend950[A-Za-z0-9_]+", args.soc, re.IGNORECASE) or args.jobs < 1:
        raise ValueError("Require an Ascend950 SoC model or family and positive --jobs")
    directory = args.build_dir.resolve()
    if directory in (REPO, SOURCE) or directory in REPO.parents or SOURCE in directory.parents:
        raise ValueError("Choose a dedicated out-of-source build directory")
    cache = directory / "CMakeCache.txt"
    if cache.is_file():
        match = re.search(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$", cache.read_text(), re.MULTILINE)
        if match and Path(match.group(1).strip()).resolve() != SOURCE:
            raise ValueError("Build directory belongs to another project; do not overwrite the baseline")
    return directory


def cmake_commands(args, directory, recipe, npu_path, torch_cmake, *, soc=None):
    return [
        [
            "cmake",
            "-S",
            str(SOURCE),
            "-B",
            str(directory),
            f"-DASCEND_HOME_PATH={args.cann.resolve()}",
            f"-DSOC_VERSION={soc if soc is not None else args.soc}",
            f"-DVQ2A8_ASCENDC_CMAKE={recipe}",
            f"-DTORCH_NPU_PATH={npu_path}",
            f"-DCMAKE_PREFIX_PATH={torch_cmake}",
            f"-DASCEND_PYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        ["cmake", "--build", str(directory), "--target", TARGET, "-j", str(args.jobs), "--verbose"],
    ]


def build(args):
    directory = validate_options(args)
    report = {
        "status": "planned" if args.dry_run else "building",
        "implementation": "ascendc_v4_v2",
        "abi_version": 1,
        "requested_soc": args.soc,
        "soc": args.soc,
        "source_sha256": source_hashes(),
        "python": sys.executable,
        "device_execution_verified": False,
        "model_execution_verified": False,
        "performance_verified": False,
        "default_model_backend": "unchanged",
    }
    if args.dry_run:
        report["commands"] = cmake_commands(
            args,
            directory,
            args.ascendc_cmake or "<installed-CANN-ascendc.cmake>",
            "<selected-Python-torch_npu>",
            "<selected-Python-torch-cmake>",
        )
        print(json.dumps(report, indent=2))
        return 0
    if platform.system() != "Linux":
        raise RuntimeError("Actual compilation requires Linux CANN/torch_npu; use --dry-run on Windows")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "build-manifest.json"

    def save():
        manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        cann = args.cann.resolve(strict=True)
        recipe = discover_cmake(cann, args.ascendc_cmake)
        report["toolchain"] = toolchain_probe(cann, args.soc, recipe)
        report["soc"] = report["toolchain"]["soc"]
        print(f"V4_V2_BUILD_SOC REQUESTED={args.soc} RESOLVED={report['soc']}", flush=True)
        import torch

        spec = importlib.util.find_spec("torch_npu")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("torch_npu is not installed in the selected Python")
        report["torch"] = torch.__version__
        report["commands"] = cmake_commands(
            args,
            directory,
            recipe,
            next(iter(spec.submodule_search_locations)),
            torch.utils.cmake_prefix_path,
            soc=report["soc"],
        )
        for name, command in zip(("configure", "compile"), report["commands"]):
            report["current_step"] = name
            save()
            log = directory / f"{name}.log"
            print(f"V4_V2_BUILD_STAGE={name} LOG={log}", flush=True)
            print("COMMAND=" + json.dumps(command), flush=True)
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(f"{name} failed ({result.returncode}); inspect {log}")
        library = directory / LIBRARY_NAME
        if not library.is_file() or not library.stat().st_size:
            raise RuntimeError(f"Candidate library was not produced: {library}")
        if source_hashes() != report["source_sha256"]:
            raise RuntimeError("Candidate sources changed while compiling; rebuild before validation")
        report.update(
            status="built", library=str(library), library_sha256=hashlib.sha256(library.read_bytes()).hexdigest()
        )
        save()
        print(f"V4_V2_BUILD=PASS LIBRARY={library} REPORT={manifest}", flush=True)
        print("DEVICE_EXECUTION_VERIFIED=False MODEL_EXECUTION_VERIFIED=False", flush=True)
        return 0
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        save()
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--soc",
        required=True,
        help="Ascend950 model (e.g. Ascend950DT_9582), or family with exactly one installed CANN target",
    )
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann-9.1.0"))
    parser.add_argument("--ascendc-cmake", type=Path)
    parser.add_argument("--build-dir", type=Path, default=REPO / "build/vq2a8-ascendc-v4-v2")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="print commands/hashes; do not build or write files")
    return parser.parse_args(argv)


def main(argv=None):
    try:
        return build(parse_args(argv))
    except (OSError, RuntimeError, ValueError, ImportError, subprocess.TimeoutExpired) as exc:
        print(f"V4_V2_BUILD=FAIL ERROR={exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
