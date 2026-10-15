#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay one successful native GS32 case through the existing Torch bridge.

No quantization, repacking, kernel compilation or production behavior changes.
Default: synchronize the single custom launch, then check its output on CPU.
Optional NPU finite checks isolate the post-launch chain and its synchronization.
"""

from __future__ import annotations

# Keep tools/bisect out of the direct-script import path.
# ruff: noqa: E402
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import hashlib
import json
import subprocess
import tempfile
import traceback
from pathlib import Path

from tools.diagnose_vq2a8_gs32_native import check_shape, require_healthy_idle
from tools.validate_vq2a8_gs32_stage import compare_projection, file_identity
from tools.vq2a8_probe_process import child_environment, emit, run_child, stage_recorder

CASE = "gs32_torch_native_replay"
WEIGHT_FIELDS = ("packed_weight", "pair_lut", "row_scale")
DEVICE_FIELDS = ("input_bytes", "input_scale", *WEIGHT_FIELDS)
OUTPUT_CHECKS = ("cpu", "npu-synced", "npu-chained", "npu-stepped")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--kernel-binary", type=Path, required=True)
    parser.add_argument("--physical-npu", type=int, default=0)
    parser.add_argument("--allow-busy", action="store_true", help="Allow a healthy NPU already used by other processes")
    parser.add_argument("--guard-offset", action="store_true", help="Use the staged probe's leading guard expert")
    parser.add_argument("--sync-launch-stream", action="store_true", help="Compare Torch's SyncLaunchStream")
    parser.add_argument(
        "--output-check",
        choices=OUTPUT_CHECKS,
        default="cpu",
        help="CPU only, NPU finite after launch sync, directly chained, or sync after each finite operation",
    )
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.physical_npu < 0 or not 1 <= args.timeout_s <= 7200:
        parser.error("Require a nonnegative NPU index and timeout 1..7200 seconds")
    args.launch_blocking = "0"
    return args


def checked_bytes(path, identity, expected_size):
    data = path.read_bytes()
    if (
        len(data) != expected_size
        or identity["bytes"] != expected_size
        or hashlib.sha256(data).hexdigest() != identity["sha256"]
    ):
        raise ValueError(f"Native artifact identity mismatch: {path}")
    return data


def load_case(directory, kernel):
    # Lazy import keeps --plan-only and parent process independent of Torch/NPU.
    import torch

    report = json.loads((directory / "summary.json").read_text())
    if (
        report.get("status") != "PASS"
        or report.get("native_execution_verified") is not True
        or report.get("numerical_verified") is not True
    ):
        raise ValueError("Replay requires a native numerical PASS")
    checked_bytes(kernel, report["kernel"], report["kernel"]["bytes"])
    case = report["case"]
    sizes = check_shape(case)
    m, k, n = (case[key] for key in ("m", "k", "n"))
    layouts = {
        "input_bytes": (torch.uint8, (m, k)),
        "input_scale": (torch.uint8, (m, k // 32)),
        "packed_weight": (torch.uint8, (1, n // 32, k // 16, 16, 8)),
        "pair_lut": (torch.uint8, (1, n // 32, k // 32, 32)),
        "row_scale": (torch.int64, (1, n)),
        "reference": (torch.bfloat16, (m, n)),
    }
    tensors = {}
    for name, (dtype, shape) in layouts.items():
        filename = name + ".bin"
        data = checked_bytes(directory / filename, case["files"][filename], sizes[filename])
        tensors[name] = torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape)
    for name, filename, identity, dtype, shape, size in (
        ("native_output", "output.bin", report["output"], torch.bfloat16, (m, n), sizes["reference.bin"]),
        ("tiling", "tiling.bin", report["tiling"], torch.uint8, (1600,), 1600),
    ):
        data = checked_bytes(directory / filename, identity, size)
        tensors[name] = torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape)
    return report, tensors


def guard_view(value, enabled):
    import torch

    return torch.cat((torch.zeros_like(value), value), dim=0)[1:] if enabled else value


def tensor_info(name, value):
    return {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "data_ptr": hex(value.data_ptr()),
        "bytes": value.nbytes,
        "storage_ptr": hex(value.untyped_storage().data_ptr()),
        "storage_bytes": value.untyped_storage().nbytes(),
        "storage_offset": value.storage_offset(),
    }


def checked_bridge_tiling(ops, native, shape):
    import torch

    result = ops.make_tiling(shape["k"], shape["n"])
    if result.dtype != torch.uint8 or result.shape != native.shape or not torch.equal(result, native):
        raise ValueError("Torch bridge tiling differs from the successful native tiling")
    # Use Torch's allocation: a bytearray-backed frombuffer is not guaranteed
    # to meet the bridge's 32-byte CPU metadata alignment requirement.
    return result


def check_finite_steps(output, synchronize):
    import torch

    # Keep intermediates alive so the probe isolates the operation boundary,
    # without also changing allocation reuse between the individual steps.
    stage = stage_recorder(CASE, synchronize)
    with stage("output_cast"):
        converted = output.float()
    with stage("output_isfinite"):
        finite = torch.isfinite(converted)
    with stage("output_all"):
        all_finite = finite.all()
    with stage("output_scalar"):
        if not bool(all_finite):
            raise ValueError("GS32 native projection output contains NaN or Inf.")


def launch_once(ops, tensors, output, synchronize, *, output_check="cpu"):
    import torch

    if output_check not in OUTPUT_CHECKS:
        raise ValueError(f"Unknown output check: {output_check}")
    if output_check in ("npu-synced", "npu-chained"):
        from tools.vq2a8_gs32_reference import require_finite

    group = torch.tensor([output.shape[0]], dtype=torch.int64)
    mask = torch.zeros(1, dtype=torch.int32)
    stage = stage_recorder(CASE, synchronize)
    name = "custom_launch_and_finite" if output_check == "npu-chained" else "custom_launch"
    with stage(name):
        ops.project(*(tensors[name] for name in DEVICE_FIELDS), group, mask, tensors["tiling"], output)
        if output_check == "npu-chained":
            # Keep the staged probe's float/isfinite/all/bool sequence. The
            # bool itself synchronizes; do not insert a wait before this chain.
            require_finite(output, "GS32 native projection output")
    if output_check == "npu-synced":
        with stage("output_finite"):
            require_finite(output, "GS32 native projection output")
    elif output_check == "npu-stepped":
        check_finite_steps(output, synchronize)


def loaded_runtime_libraries():
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return []
    return sorted(
        {
            line.split()[-1]
            for line in maps.read_text().splitlines()
            if any(name in line for name in ("libruntime.so", "libascendcl.so", "libtorch_npu.so"))
        }
    )


def select_stream(npu, device, sync_launch):
    if not sync_launch:
        return npu.current_stream(device)
    if not hasattr(npu, "SyncLaunchStream"):
        raise RuntimeError("Installed torch_npu does not expose SyncLaunchStream; no fallback was run")
    return npu.SyncLaunchStream(device=device)


def replay_child(args):
    # Retain tensors at this level through a device failure. Exit without
    # allocator destructors submitting more events to the faulted device.
    try:
        report, cpu = load_case(args.native_report, args.kernel_binary)
        import torch
        import torch_npu

        torch.npu.set_device(0)
        device = torch.device("npu:0")
        stage = stage_recorder(CASE, torch.npu.synchronize)
        emit(
            CASE,
            "RUNTIME",
            torch=torch.__version__,
            torch_npu=torch_npu.__version__,
            allocator_config=os.environ.get("PYTORCH_NPU_ALLOC_CONF", "<unset>"),
            task_queue=os.environ.get("TASK_QUEUE_ENABLE", "<unset>"),
            libraries=loaded_runtime_libraries(),
            library=file_identity(args.library),
            kernel=report["kernel"],
            guard_offset=args.guard_offset,
            sync_launch_stream=args.sync_launch_stream,
            output_check=args.output_check,
            allow_busy=args.allow_busy,
        )
        with stage("load_bridge"):
            torch.ops.load_library(str(args.library.resolve()))
            ops = torch.ops.vq2a8_gs32_stage
            ops.initialize(str(args.kernel_binary.resolve()))
            actual_tiling = checked_bridge_tiling(ops, cpu["tiling"], report["case"])
        with stage("upload_exact_buffers"):
            tensors = {name: cpu[name].to(device) for name in DEVICE_FIELDS}
            for name in WEIGHT_FIELDS:
                tensors[name] = guard_view(tensors[name], args.guard_offset)
            tensors["tiling"] = actual_tiling
            output = torch.full(cpu["reference"].shape, float("nan"), dtype=torch.bfloat16, device=device)
        # Verify actual device bytes before the custom call, including offset
        # views. This diagnostic deliberately synchronizes and is not timed.
        with stage("verify_uploaded_bytes"):
            for name in DEVICE_FIELDS:
                if not torch.equal(tensors[name].cpu(), cpu[name]):
                    raise ValueError(f"Uploaded bytes differ from native case: {name}")
                emit(CASE, "BUFFER", **tensor_info(name, tensors[name]))
            emit(CASE, "BUFFER", **tensor_info("output", output))
        # Upload verification has synchronized all producers before changing
        # streams. launch_once synchronizes the consumer before leaving it.
        launch_stream = select_stream(torch.npu, device, args.sync_launch_stream)
        emit(CASE, "STREAM", description=str(launch_stream), libraries=loaded_runtime_libraries())
        with torch.npu.stream(launch_stream):
            launch_once(ops, tensors, output, torch.npu.synchronize, output_check=args.output_check)
        # Every mode still performs both original numerical comparisons on CPU.
        with stage("copy_output"):
            actual = output.cpu()
        results = [
            compare_projection(actual, cpu["reference"], "torch_vs_reference"),
            compare_projection(actual, cpu["native_output"], "torch_vs_native"),
        ]
        emit(
            CASE,
            "CASE_PASS",
            results=results,
            output_check=args.output_check,
            model_integration_verified=False,
            graph_verified=False,
            performance_verified=False,
        )
        return 0
    except Exception as error:
        traceback.print_exc()
        emit(CASE, "CASE_FAIL", error=str(error))
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


def child_command(args):
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--child",
        "--native-report",
        str(args.native_report.resolve()),
        "--library",
        str(args.library.resolve()),
        "--kernel-binary",
        str(args.kernel_binary.resolve()),
        "--physical-npu",
        str(args.physical_npu),
        "--output-check",
        args.output_check,
    ]
    if args.guard_offset:
        command.append("--guard-offset")
    if args.sync_launch_stream:
        command.append("--sync-launch-stream")
    if args.allow_busy:
        command.append("--allow-busy")
    return command


def main(argv=None):
    args = parse_args(argv)
    if args.child:
        return replay_child(args)
    report = {
        "status": "PLANNED",
        "scope": "one_torch_projection_of_native_export",
        "command": child_command(args),
        "model_integration_verified": False,
        "allow_busy": args.allow_busy,
    }
    if args.plan_only:
        print(json.dumps(report))
        return 0
    if sys.platform != "linux":
        raise RuntimeError("Torch replay requires Linux/Ascend; use --plan-only elsewhere")
    directory = args.report_dir
    if directory is None:
        directory = Path(tempfile.mkdtemp(prefix="vq2-gs32-torch-"))
    else:
        directory.mkdir(parents=True, exist_ok=False)
    report["status"] = "FAIL"
    try:
        snapshot = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True, timeout=20)
        report["device"] = require_healthy_idle(snapshot.stdout, args.physical_npu, allow_busy=args.allow_busy)
        result = run_child(report["command"], child_environment(args), directory / "replay.log", args.timeout_s)
        report.update(status=result["status"], result=result)
    except Exception as error:
        report["error"] = str(error)
        traceback.print_exc()
    finally:
        path = directory / "summary.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"GS32_TORCH_REPLAY={report['status']} SUMMARY={path}", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
