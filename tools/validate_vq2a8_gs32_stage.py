#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate one real GS32 expert on NPU against the staged MXFP8 CPU oracle.

This is an eager component check, not whole-model quality, graph, or performance
acceptance. The reconstructed-BF16-weight route is a different numerical route
and is deliberately not used as this probe's correctness oracle.
"""

from __future__ import annotations

# Keep tools/bisect out of the standard-library search path.
# ruff: noqa: E402
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import faulthandler
import hashlib
import json
import subprocess
import tempfile
import traceback
from contextlib import nullcontext
from pathlib import Path

import regex as re

from tools.vq2a8_probe_process import child_environment, emit, parse_snapshot, run_child, stage_recorder

CASE = "gs32_staged_real_expert"
RTOL = 0.01
ATOL = 0.01
MIN_MATCH = 0.99
MAX_TOKENS = 128
CHECKS = (
    "gate_quant_data",
    "gate_quant_scale",
    "gate_up",
    "down_quant_data",
    "down_quant_scale",
    "down_from_actual_gate",
    "ffn_end_to_end",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--kernel-binary", required=True, type=Path)
    parser.add_argument("--physical-npu", type=int, default=1)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument("--nondefault-stream", action="store_true")
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument(
        "--report-dir", type=Path, help="New output directory; defaults to a unique temporary directory"
    )
    parser.add_argument("--plan-only", action="store_true", help="No Torch/NPU import, file read, or device operation")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if min(args.physical_npu, args.layer, args.expert) < 0:
        parser.error("NPU, layer and expert must be non-negative")
    if not 1 <= args.tokens <= MAX_TOKENS or not 1 <= args.timeout_s <= 7200:
        parser.error(f"Require tokens 1..{MAX_TOKENS} and timeout 1..7200 seconds")
    if not 0 <= args.seed < 2**63 or (args.child and args.plan_only):
        parser.error("Require seed in [0, 2**63); --child cannot be combined with --plan-only")
    # Use the existing diagnostic environment policy, not a serving setting.
    args.launch_blocking = "0"
    return args


def child_command(args):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--child"]
    for flag, value in (
        ("--run-root", args.run_root.resolve()),
        ("--library", args.library.resolve()),
        ("--kernel-binary", args.kernel_binary.resolve()),
        ("--physical-npu", args.physical_npu),
        ("--layer", args.layer),
        ("--expert", args.expert),
        ("--tokens", args.tokens),
        ("--seed", args.seed),
        ("--timeout-s", args.timeout_s),
    ):
        command += [flag, str(value)]
    if args.nondefault_stream:
        command.append("--nondefault-stream")
    return command


def file_identity(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Expected a nonempty regular file: {path}")
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def compare_bytes(actual, expected, name):
    """Byte-exact activation data and E8M0 scales; no tolerance or cast."""
    import torch

    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    if actual.dtype != torch.uint8 or expected.dtype != torch.uint8 or actual.shape != expected.shape:
        raise AssertionError(f"{name}: expected matching uint8 shapes")
    different = actual.flatten() != expected.flatten()
    count = int(different.sum())
    if count:
        first = int(torch.nonzero(different, as_tuple=False)[0, 0])
        raise AssertionError(
            f"{name}: {count} differing bytes; first={first}, "
            f"actual={int(actual.flatten()[first])}, expected={int(expected.flatten()[first])}"
        )
    return {"name": name, "status": "PASS", "comparison": "byte_exact", "elements": actual.numel()}


def compare_projection(actual, expected, name):
    """Fixed BF16 tolerances, also applied per row to catch missing token rows."""
    import torch

    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    if (
        actual.dtype != torch.bfloat16
        or expected.dtype != torch.bfloat16
        or actual.shape != expected.shape
        or actual.ndim != 2
        or actual.numel() == 0
    ):
        raise AssertionError(f"{name}: expected matching nonempty 2-D BF16 outputs")
    actual, expected = actual.float(), expected.float()
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        raise AssertionError(f"{name}: nonfinite output or reference")
    close = torch.isclose(actual, expected, rtol=RTOL, atol=ATOL)
    match = float(close.double().mean())
    min_row_match = float(close.double().mean(dim=1).min())
    absolute = (actual.double() - expected.double()).abs()
    report = {
        "name": name,
        "status": "PASS" if min(match, min_row_match) >= MIN_MATCH else "FAIL",
        "comparison": "staged_mxfp8_bf16_output",
        "rtol": RTOL,
        "atol": ATOL,
        "required_match": MIN_MATCH,
        "match": match,
        "min_row_match": min_row_match,
        "max_abs_error": float(absolute.max()),
        "mean_abs_error": float(absolute.mean()),
        "max_relative_error": float((absolute / expected.double().abs().clamp_min(1e-30)).max()),
        "elements": actual.numel(),
    }
    emit(CASE, "NUMERIC", **report)
    if report["status"] != "PASS":
        raise AssertionError(
            f"{name}: overall match={match:.6%}, worst-row match={min_row_match:.6%}; "
            f"required {MIN_MATCH:.2%} with rtol={RTOL}, atol={ATOL}; thresholds are not relaxed"
        )
    return report


def run_numeric(args, artifact, expert, device, stage):
    # Imported only inside the isolated child or explicitly by CPU unit tests.
    import torch

    from tools.vq2a8_gs32_reference import (
        quantize_mxfp8_k32,
        staged_expert_reference,
        staged_projection_reference,
        staged_swiglu,
    )

    gate, gate_spec = artifact.load_matrix(args.layer, args.expert, "gate_up", device="cpu")
    down, down_spec = artifact.load_matrix(args.layer, args.expert, "down", device="cpu")
    limit = artifact.model_config["swiglu_limit"]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    hidden = torch.randn(args.tokens, gate_spec.columns, generator=generator).to(torch.bfloat16)
    records = []

    def check_quant(prefix, host_input, payload):
        # Test device-side column scaling, permutation, and quantization as well
        # as the quantizer in isolation. CPU arithmetic is never used to feed
        # the actual native projection.
        with stage(f"{prefix}_quant"):
            cpu_prepared = (host_input.float() * payload["weight_scale"]).index_select(1, payload["perm"].long())
            device_prepared = (host_input.to(device).float() * payload["weight_scale"].to(device)).index_select(
                1, payload["perm"].long().to(device)
            )
            expected_data, expected_scale = quantize_mxfp8_k32(cpu_prepared)
            actual_data, actual_scale = quantize_mxfp8_k32(device_prepared)
        records.append(compare_bytes(actual_data, expected_data, f"{prefix}_quant_data"))
        records.append(compare_bytes(actual_scale, expected_scale, f"{prefix}_quant_scale"))

    check_quant("gate", hidden, gate)
    with stage("gate_up"):
        actual_gate = expert.project("gate_up", hidden.to(device))
    expected_gate = staged_projection_reference(hidden, gate, gate_spec)
    records.append(compare_projection(actual_gate, expected_gate, "gate_up"))

    # Feed the SAME actual-gate-derived intermediate to Down on both sides.
    # Thus this check isolates the second GEMM from Gate error accumulation.
    with stage("clipped_swiglu"):
        actual_intermediate = staged_swiglu(actual_gate, limit)
    intermediate_cpu = actual_intermediate.detach().cpu()
    check_quant("down", intermediate_cpu, down)
    with stage("down_from_actual_gate"):
        actual_down = expert.project("down", actual_intermediate)
    expected_down = staged_projection_reference(intermediate_cpu, down, down_spec)
    records.append(compare_projection(actual_down, expected_down, "down_from_actual_gate"))

    # End-to-end comparison additionally covers the framework SwiGLU rounding
    # and accumulation of projection error; no per-stage reset hides it.
    with stage("ffn_end_to_end"):
        actual_ffn = expert.forward(hidden.to(device), swiglu_limit=limit)
    expected_ffn = staged_expert_reference(hidden, gate, gate_spec, down, down_spec, limit)
    records.append(compare_projection(actual_ffn, expected_ffn, "ffn_end_to_end"))
    return records


def validate_child_evidence(result, args):
    passes = [event for event in result.get("events", []) if event.get("event") == "CASE_PASS"]
    if result.get("status") != "PASS" or result.get("exit_code") != 0 or result.get("reaped") is not True:
        raise ValueError("Incomplete GS32 child completion")
    if len(passes) != 1:
        raise ValueError("Incomplete GS32 child receipt")
    receipt = passes[0]
    if (
        receipt.get("case") != CASE
        or receipt.get("device_execution_verified") is not True
        or receipt.get("graph_verified") is not False
        or receipt.get("model_integration_verified") is not False
        or receipt.get("performance_verified") is not False
        or receipt.get("model_dispatch_enabled") is not False
        or receipt.get("nondefault_stream") is not args.nondefault_stream
        or receipt.get("selection") != [args.layer, args.expert, args.tokens, args.seed]
    ):
        raise ValueError("Incomplete GS32 component-only evidence")
    for key, requested in (("library", args.library), ("kernel_binary", args.kernel_binary)):
        identity = receipt.get(key, {})
        if (
            identity.get("path") != str(requested.resolve())
            or not isinstance(identity.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]) is None
            or type(identity.get("bytes")) is not int
            or identity["bytes"] <= 0
        ):
            raise ValueError("Incomplete GS32 native-file identity")
    records = receipt.get("results", [])
    if [record.get("name") for record in records] != list(CHECKS):
        raise ValueError("Incomplete GS32 numeric stages")
    for record in records:
        if record.get("status") != "PASS" or record.get("elements", 0) <= 0:
            raise ValueError("Incomplete GS32 numeric result")
        if record["name"] in ("gate_up", "down_from_actual_gate", "ffn_end_to_end"):
            if (
                record.get("rtol") != RTOL
                or record.get("atol") != ATOL
                or record.get("required_match") != MIN_MATCH
                or not MIN_MATCH <= record.get("match", -1) <= 1
                or not MIN_MATCH <= record.get("min_row_match", -1) <= 1
            ):
                raise ValueError("Incomplete GS32 fixed-tolerance evidence")
        elif record.get("comparison") != "byte_exact":
            raise ValueError("Incomplete GS32 activation-byte evidence")
    if not any(event.get("event") == "PASS" and event.get("stage") == "final_sync" for event in result["events"]):
        raise ValueError("Incomplete GS32 final synchronization")
    return receipt


def run_case_child(args):
    faulthandler.enable()
    faulthandler.dump_traceback_later(min(60, max(1, args.timeout_s / 2)), repeat=True)
    try:
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(args.physical_npu):
            raise ValueError("Child mapping differs from selected physical NPU")
        import torch
        import torch_npu  # noqa: F401

        from tools.vq2a8_gs32_artifact import open_vq2a8_gs32_artifact
        from tools.vq2a8_gs32_runtime import GS32StagedRuntime

        identities = {"library": file_identity(args.library), "kernel_binary": file_identity(args.kernel_binary)}
        torch.npu.set_device(0)
        device = torch.device("npu:0")
        stage = stage_recorder(CASE, torch.npu.synchronize)
        with stage("artifact_and_runtime"):
            artifact = open_vq2a8_gs32_artifact(args.run_root)
            runtime = GS32StagedRuntime(artifact, args.library, args.kernel_binary, device, cache_experts=1)
            expert = runtime.get_expert(args.layer, args.expert)
        # Initialization is synchronized before crossing into another stream.
        stream_context = torch.npu.stream(torch.npu.Stream(device=device)) if args.nondefault_stream else nullcontext()
        with torch.inference_mode(), stream_context:
            results = run_numeric(args, artifact, expert, device, stage)
        with stage("final_sync"):
            pass
        emit(
            CASE,
            "CASE_PASS",
            results=results,
            selection=[args.layer, args.expert, args.tokens, args.seed],
            nondefault_stream=args.nondefault_stream,
            device_execution_verified=True,
            graph_verified=False,
            model_integration_verified=False,
            performance_verified=False,
            model_dispatch_enabled=False,
            **identities,
        )
        return 0
    except Exception as error:
        traceback.print_exc()
        emit(CASE, "CASE_FAIL", error=str(error), device_execution_verified=False, model_dispatch_enabled=False)
        return 1
    finally:
        faulthandler.cancel_dump_traceback_later()


def main(argv=None):
    args = parse_args(argv)
    if args.child:
        return run_case_child(args)
    report = {
        "scope": "one_real_expert_staged_mxfp8_eager_only",
        "status": "PLANNED",
        "command": child_command(args),
        "device_execution_verified": False,
        "graph_verified": False,
        "model_integration_verified": False,
        "performance_verified": False,
        "model_dispatch_enabled": False,
    }
    if args.plan_only:
        print(json.dumps(report, indent=2))
        return 0
    if os.name != "posix":
        raise RuntimeError("GS32 NPU validation requires Linux; use --plan-only elsewhere")
    directory = args.report_dir or Path(tempfile.mkdtemp(prefix="vq2-gs32-stage-"))
    if args.report_dir is not None:
        directory.mkdir(parents=True, exist_ok=False)
    report["status"] = "FAIL"
    try:
        snapshot = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True, timeout=20)
        (directory / "npu.log").write_text(snapshot.stdout + snapshot.stderr, encoding="utf-8")
        state = parse_snapshot(snapshot.stdout, args.physical_npu)
        if state == "unknown" or (state == "busy" and not args.allow_busy):
            report.update(status="BLOCKED", device_state=state)
        else:
            environment = child_environment(args)
            environment.setdefault("TASK_QUEUE_ENABLE", "1")
            result = run_child(child_command(args), environment, directory / "validation.log", args.timeout_s)
            report.update(status=result["status"], result=result)
            if result["status"] == "PASS":
                report["receipt"] = validate_child_evidence(result, args)
                report["device_execution_verified"] = True
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        traceback.print_exc()
    finally:
        summary = directory / "summary.json"
        summary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"GS32_STAGE_PROBE={report['status']} SUMMARY={summary}", flush=True)
    return 0 if report["status"] == "PASS" else 130 if report["status"] == "INTERRUPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
