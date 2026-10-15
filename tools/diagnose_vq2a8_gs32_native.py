#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolate GS32 GateUp-as-DownW2 from Torch NPU using one native launch.

This diagnostic exports the SAME CPU-generated layer/expert/seed input as the
staged probe, then uses a separate ACL/runtime-only executable. No kernel is
compiled, edited or replaced. PASS covers one projection, not an FFN or model.
"""

from __future__ import annotations

# Keep tools/bisect out of the direct-script import path.
# ruff: noqa: E402
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import importlib
import json
import platform
import subprocess
import tempfile
import traceback
from pathlib import Path

import regex as re

from tools.check_vq2a8_gs32 import _cpu_modules
from tools.validate_vq2a8_gs32_stage import compare_projection, file_identity
from tools.vq2a8_live_log import LiveChildLog
from tools.vq2a8_probe_process import child_environment, parse_snapshot, terminate_child

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "csrc/vq2a8_gs32_stage/native_probe.cpp"
NATIVE_PASS = "GS32_NATIVE_EXECUTION=PASS"
MAX_DIMENSION = 65536
MAX_BUFFER_BYTES = 1 << 30


def progress(stage):
    print(f"GS32_NATIVE_DIAGNOSTIC_STAGE={stage}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--kernel-binary", type=Path, required=True)
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann-9.1.0"))
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--physical-npu", type=int, default=1)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument("--report-dir", type=Path, help="New directory only; existing paths are never reused")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan-only", action="store_true", help="No file reads, exports, compilation or execution")
    modes.add_argument("--export-only", action="store_true", help="CPU input/reference export only; no NPU or build")
    args = parser.parse_args(argv)
    if min(args.physical_npu, args.layer, args.expert) < 0:
        parser.error("NPU, layer and expert must be nonnegative")
    if not 1 <= args.tokens <= 128 or not 1 <= args.timeout_s <= 7200:
        parser.error("Require tokens 1..128 and timeout 1..7200 seconds")
    if not 0 <= args.seed < 2**63:
        parser.error("Require seed in [0,2**63)")
    args.launch_blocking = "0"
    return args


def _cpu_dependencies():
    artifact, _, layout, _ = _cpu_modules()
    staged = importlib.import_module("tools.vq2a8_gs32_reference")
    return artifact, layout, staged


def check_shape(shape):
    m, k, n = (shape[key] for key in ("m", "k", "n"))
    if any(type(value) is not int for value in (m, k, n)):
        raise ValueError("Native projection shape must contain integers")
    if not 1 <= m <= 128 or not 0 < k <= MAX_DIMENSION or not 0 < n <= MAX_DIMENSION:
        raise ValueError("Native projection dimensions outside diagnostic bounds")
    if k % 64 or n % 32:
        raise ValueError("Native projection requires K multiple of 64 and N multiple of 32")
    sizes = {
        "input_bytes.bin": m * k,
        "input_scale.bin": m * k // 32,
        "packed_weight.bin": n * k // 4,
        "pair_lut.bin": n * k // 32,
        "row_scale.bin": n * 8,
        "reference.bin": m * n * 2,
    }
    if sum(sizes.values()) > MAX_BUFFER_BYTES:
        raise ValueError("Native diagnostic tensors exceed 1 GiB")
    return sizes


def export_case(args, directory):
    import torch

    artifact_module, layout, staged = _cpu_dependencies()
    progress("cpu_artifact")
    artifact = artifact_module.open_vq2a8_gs32_artifact(args.run_root)
    payload, spec = artifact.load_matrix(args.layer, args.expert, "gate_up", device="cpu", validate_payload=True)
    shape = {"m": args.tokens, "k": spec.columns, "n": spec.rows}
    sizes = check_shape(shape)
    progress("cpu_repack")
    fields = layout.repack_gs32_matrix(payload, spec, down_stage=True)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    hidden = torch.randn(args.tokens, spec.columns, generator=generator).to(torch.bfloat16)
    progress("cpu_quant")
    prepared = (hidden.float() * fields["weight_scale"]).index_select(1, fields["perm"].long())
    data, scale = staged.quantize_mxfp8_k32(prepared)
    progress("cpu_reference")
    reference = staged.staged_projection_reference(hidden, payload, spec)
    if reference.shape != (args.tokens, spec.rows) or reference.dtype != torch.bfloat16:
        raise ValueError("Unexpected native projection reference shape/dtype")
    if not bool(torch.isfinite(reference.float()).all()):
        raise ValueError("Nonfinite CPU projection reference")
    tensors = {
        "input_bytes.bin": data,
        "input_scale.bin": scale,
        "packed_weight.bin": fields["packed_zn"],
        "pair_lut.bin": fields["pair_lut"],
        "row_scale.bin": fields["row_scale_bits"].to(torch.int64),
        "reference.bin": reference,
    }
    files = {}
    for name, tensor in tensors.items():
        if tensor.device.type != "cpu":
            raise ValueError("Native export must stay on CPU")
        raw = tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
        if len(raw) != sizes[name]:
            raise ValueError(f"{name}: export has {len(raw)} bytes; expected {sizes[name]}")
        path = Path(directory) / name
        with path.open("xb") as output:
            output.write(raw)
        files[name] = file_identity(path)
    return {
        **shape,
        "selection": [args.layer, args.expert, args.tokens, args.seed],
        "source_matrix": spec.name,
        "files": files,
        "activation_source": "CPU counterpart of the staged probe; not an export of failing NPU allocations",
        "stage_mask": 4,
        "mixed_dimensions": [1, spec.columns, 2 * spec.columns, spec.rows],
        "argument_bytes": 152,
        "block_dim": 32,
    }


def build_command(args, executable):
    sdk = args.cann.resolve()
    library_dirs = [sdk / "runtime/lib64", sdk / "lib64"]
    return [
        args.cxx,
        "-std=c++17",
        "-O2",
        "-g",
        f"-I{sdk / 'include'}",
        f"-I{sdk / (platform.machine() + '-linux') / 'include'}",
        str(SOURCE),
        *(f"-L{path}" for path in library_dirs),
        *(f"-Wl,-rpath,{path}" for path in library_dirs),
        "-lruntime",
        "-lascendcl",
        "-pthread",
        "-o",
        str(executable),
    ]


def native_command(executable, kernel, directory, shape):
    check_shape(shape)
    return [str(executable), str(kernel), str(directory), "0", *(str(shape[key]) for key in ("m", "k", "n"))]


def native_environment(args, environ=None):
    # Reuse the probe's single physical-device mapping. The C++ process calls
    # aclrtSetDevice(0); it has no Torch task queue or caching allocator.
    return child_environment(args, environ)


def validate_native_output(directory, shape, stdout, returncode):
    import torch

    if returncode != 0 or stdout.splitlines().count(NATIVE_PASS) != 1:
        raise ValueError("Native launch did not complete exactly once with its PASS receipt")
    size = check_shape(shape)["reference.bin"]
    values = []
    for name in ("output.bin", "reference.bin"):
        path = Path(directory) / name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"{name}: missing or wrong output byte length; expected {size}")
        tensor = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.bfloat16)
        values.append(tensor.reshape(shape["m"], shape["n"]))
    return compare_projection(*values, "native_gate_up")


def require_healthy_idle(snapshot, physical_npu, *, allow_busy=False):
    device_row = re.search(rf"^\s*\|\s*{physical_npu}\s*\|\s*Ascend[^|]*\|\s*([^|]+)\|", snapshot, flags=re.MULTILINE)
    health = device_row.group(1).strip() if device_row else "unknown"
    activity = parse_snapshot(snapshot, physical_npu)
    allowed_activity = activity == "idle" or (allow_busy and activity == "busy")
    if health != "OK" or not allowed_activity:
        requirement = "healthy with known activity" if allow_busy else "healthy and idle"
        raise ValueError(f"Physical NPU {physical_npu} must be {requirement}: health={health}, activity={activity}")
    return {"physical_npu": physical_npu, "logical_device": 0, "health": health, "activity": activity}


def run_logged(command, environment, log, timeout_s):
    print("COMMAND=" + json.dumps(command), flush=True)
    with log.open("x", encoding="utf-8") as output, LiveChildLog(log, "gs32-native"):
        child = subprocess.Popen(
            command, env=environment, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = child.wait(timeout=timeout_s)
        except BaseException:
            terminate_child(child)
            raise
    return code, log.read_text(encoding="utf-8", errors="replace")


def main(argv=None):
    args = parse_args(argv)
    report = {
        "status": "PLANNED",
        "scope": "one_native_downw2_gate_up_projection_without_torch_npu",
        "selection": [args.layer, args.expert, args.tokens, args.seed],
        "physical_npu": args.physical_npu,
        "native_execution_verified": False,
        "numerical_verified": False,
        "model_integration_verified": False,
        "model_dispatch_enabled": False,
        "graph_verified": False,
        "performance_verified": False,
    }
    if args.plan_only:
        print(json.dumps(report, indent=2))
        return 0
    if not args.export_only and sys.platform != "linux":
        raise RuntimeError("Native CANN execution requires Linux; use --plan-only or --export-only elsewhere")
    directory = args.report_dir
    if directory is None:
        directory = Path(tempfile.mkdtemp(prefix="vq2-gs32-native-"))
    else:
        directory = directory.resolve()
        directory.mkdir(parents=True, exist_ok=False)
    report["status"] = "FAIL"
    print(f"GS32_NATIVE_DIAGNOSTIC_DIR={directory}", flush=True)
    try:
        kernel = args.kernel_binary.resolve(strict=True)
        with kernel.open("rb") as source:
            if source.read(4) != b"\x7fELF":
                raise ValueError("Expected an existing compiled mixed kernel ELF; not rebuilding it")
        report["kernel"] = file_identity(kernel)
        report["case"] = export_case(args, directory)
        if args.export_only:
            report["status"] = "EXPORTED"
            return 0
        executable = directory / "gs32_native_probe"
        progress("build_host_only")
        report["host_source"] = file_identity(SOURCE)
        command = build_command(args, executable)
        report["build_command"] = command
        code, _ = run_logged(command, dict(os.environ), directory / "build.log", args.timeout_s)
        if code:
            raise RuntimeError(f"Diagnostic Host compilation failed ({code}); inspect build.log")
        report["host_executable"] = file_identity(executable)
        progress("device_precheck")
        snapshot = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True, timeout=20)
        (directory / "npu.log").write_text(snapshot.stdout + snapshot.stderr, encoding="utf-8")
        report["device"] = require_healthy_idle(snapshot.stdout, args.physical_npu)
        if file_identity(kernel) != report["kernel"]:
            raise RuntimeError("Kernel changed before launch; stop and inspect artifacts")
        command = native_command(executable, kernel, directory, report["case"])
        report["native_command"] = command
        progress("native_single_launch")
        code, stdout = run_logged(command, native_environment(args), directory / "native.log", args.timeout_s)
        report["native_exit_code"] = code
        report["native_execution_verified"] = code == 0 and stdout.splitlines().count(NATIVE_PASS) == 1
        progress("numeric_compare")
        report["comparison"] = validate_native_output(directory, report["case"], stdout, code)
        if file_identity(kernel) != report["kernel"]:
            raise RuntimeError("Kernel changed during validation; result is not acceptable")
        for entry in report["case"]["files"].values():
            if file_identity(entry["path"]) != entry:
                raise RuntimeError("Exported input/reference changed during validation")
        report["output"] = file_identity(directory / "output.bin")
        report["tiling"] = file_identity(directory / "tiling.bin")
        report["numerical_verified"] = True
        report["status"] = "PASS"
        return 0
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
        return 130
    except Exception as error:
        report["error"] = str(error)
        traceback.print_exc()
        return 1
    finally:
        summary = directory / "summary.json"
        summary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"GS32_NATIVE_DIAGNOSTIC={report['status']} SUMMARY={summary}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
