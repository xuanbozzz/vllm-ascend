#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only GS32 framework preflight, not model or native-kernel acceptance.

Audits all metadata/headers and one selected expert's values and repacking.
Never writes checkpoints, builds kernels, initializes NPU, or enables dispatch.
"""

from __future__ import annotations

# Keep tools/bisect out of the stdlib import search path for direct execution.
# ruff: noqa: E402
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import importlib
import importlib.machinery
import json
import types
from pathlib import Path


def _cpu_modules():
    """Read local CPU siblings without initializing the vLLM/plugin package."""
    directory = Path(__file__).resolve().parent
    package_name = "_vq2a8_gs32_check_cpu"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(directory)]
        package.__spec__ = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules[package_name] = package
    elif list(sys.modules[package_name].__path__) != [str(directory)]:
        raise RuntimeError("GS32 CPU helper package is already bound to another repository.")
    return tuple(
        importlib.import_module(f"{package_name}.{module}")
        for module in ("vq2a8_gs32_artifact", "vq2a8_gs32_contract", "vq2a8_gs32_layout", "vq2a8_gs32_root")
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path, required=True, help="Directory containing packed_model/ and experts_vq/"
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument(
        "--decode-reference", action="store_true", help="Also reconstruct this expert's BF16 CPU weights"
    )
    parser.add_argument(
        "--require-native", action="store_true", help="Exit 2: this read-only CPU tool cannot certify native execution"
    )
    args = parser.parse_args(argv)
    if args.layer < 0 or args.expert < 0:
        parser.error("--layer and --expert must be non-negative")
    return args


def check_run(args):
    # Delay tensor dependencies so --help works without PyTorch/vLLM installed.
    import torch

    artifact_module, contract_module, layout_module, root_module = _cpu_modules()
    artifact = artifact_module.open_vq2a8_gs32_artifact(args.run_root)
    roots = root_module.GS32RootCheckpoint(artifact.model_root)
    if artifact.model_config != roots.config:
        raise ValueError("Model config changed between expert and root audits.")
    payloads = {}
    matrix_reports = {}
    for kind in ("gate_up", "down"):
        payload, spec = artifact.load_matrix(args.layer, args.expert, kind)
        packed = layout_module.repack_gs32_matrix(payload, spec)
        payloads[kind] = payload
        record = {
            "source_matrix": spec.name,
            "source_shape": [spec.rows, spec.columns],
            "payload_values_validated": True,
            "packed_zn_shape": list(packed["packed_zn"].shape),
            "pair_lut_shape": list(packed["pair_lut"].shape),
            "weight_and_lut_bytes": packed["packed_zn"].numel() + packed["pair_lut"].numel(),
            "row_scale_bits_dtype": str(packed["row_scale_bits"].dtype),
            "bf16_reference_decoded": False,
        }
        if args.decode_reference:
            weight = layout_module.decode_gs32_weight_reference(payload, spec)
            if not bool(torch.isfinite(weight.float()).all()):
                raise ValueError(f"BF16 reconstructed weight is not finite: {spec.name}.")
            record["bf16_reference_decoded"] = True
            # No NPU comparison or independent producer execution is implied.
            record["bf16_reference_bytes"] = weight.numel() * weight.element_size()
            del weight
        matrix_reports[kind] = record
    compatibility = contract_module.current_fused_kernel_compatibility(
        artifact.model_config, payloads["gate_up"], payloads["down"]
    )
    return {
        "scope": "metadata_headers_and_one_expert_cpu_repack_only",
        "format": artifact_module.GS32_FORMAT,
        "layout": layout_module.GS32_LAYOUT_FORMAT,
        "run_root": str(artifact.root),
        "complete_layer_inventory": artifact.complete,
        "layers": len(artifact.layers),
        "matrices": sum(len(layer.specs) for layer in artifact.layers.values()),
        "root_tensors": len(roots.inventory),
        "registered_root_tensors": len(roots.inventory) - len(roots.delegated_names()),
        "moe_root_tensors": len(roots.delegated_names()),
        "root_payload_bytes": roots.payload_bytes,
        "sampled_layer": args.layer,
        "sampled_expert": args.expert,
        "matrices_checked": matrix_reports,
        "kernel_compatibility": compatibility.as_dict(),
        "model_dispatch_enabled": False,
        "dispatch_scope": "this CPU checker does not instantiate the model",
        "full_fused_kernel_compatibility": "diagnostic only; serving uses separate DownW2 projections",
        "npu_execution_verified": False,
        "full_payload_values_verified": False,
    }


def main(argv=None):
    args = parse_args(argv)
    try:
        report = check_run(args)
    except (ValueError, KeyError, OSError) as error:
        print("GS32_FRAMEWORK_CHECK " + json.dumps({"status": "FAIL", "error": str(error)}), flush=True)
        return 1
    print("GS32_FRAMEWORK_CHECK " + json.dumps({"status": "PASS", **report}, sort_keys=True), flush=True)
    if args.require_native and not report["model_dispatch_enabled"]:
        print("GS32_NATIVE_DISPATCH=BLOCKED (data preflight is not native/model acceptance)", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
