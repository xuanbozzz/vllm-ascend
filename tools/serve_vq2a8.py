#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve DeepSeek-V4 VQ2A8 through the standard quantization and MoE pipeline."""

import argparse
import json
import math
import os
import shlex
import sys
from pathlib import Path

DEFAULT_CONTEXT = 65536 + 1024
DEEPSEEK_V4_REQUIRED_OPS = (
    "npu_hc_pre_v2",
    "npu_hc_post",
    "inplace_partial_rotary_mul",
    "compressor",
    "compressor_metadata",
    "npu_quant_lightning_indexer_v2",
    "npu_quant_lightning_indexer_v2_metadata",
    "npu_kv_quant_sparse_attn_sharedkv",
    "npu_kv_quant_sparse_attn_sharedkv_metadata",
    "kv_compress_epilog",
    "indexer_compress_epilog_v2",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--library",
        type=Path,
        help="unsupported for full-model serving; use the main extension's built-in VQ operators",
    )
    parser.add_argument("--gs32-library", type=Path)
    parser.add_argument("--gs32-binary", type=Path)
    parser.add_argument("--gs32-execution", choices=("serial", "batched", "fused", "full_graph"))
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--draft-model", type=Path, help="original-format DSpark checkpoint")
    parser.add_argument("--num-speculative-tokens", type=int, default=0)
    parser.add_argument("--num-device-experts", type=int, default=84)
    parser.add_argument("--expert-offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefix-caching", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prefill-chunk-size", type=int, default=2048)
    parser.add_argument("--kv-cache-mib", type=int, default=1024)
    parser.add_argument("--kv-cache-dtype", choices=("auto", "fp8"), default="auto")
    parser.add_argument("--memory-fraction", type=float, default=0.9)
    parser.add_argument("--reserve-gib", type=float, default=16.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--decode-mode",
        choices=("grouped", "device"),
        default="device",
        help="grouped eager baseline or device route-slot execution for fixed verification blocks",
    )
    parser.add_argument(
        "--activation-preparation",
        choices=("packed", "sign_fused_direct"),
        default="packed",
        help="activation preparation for --decode-mode device; multi-token prefill uses grouped preparation",
    )
    parser.add_argument(
        "--prefill-rht",
        choices=("rowwise", "batched", "rowwise_graph"),
        default="rowwise",
        help="grouped prefill: eager rowwise, batched RHT, or rowwise RHT/bias/quantization subgraphs",
    )
    parser.add_argument(
        "--prefill-batch-mode",
        choices=("jobs", "layer"),
        default="jobs",
        help="grouped prefill scheduling: per-six-job baseline or layer-wide preparation/packing/SwiGLU",
    )
    parser.add_argument(
        "--validity-mode",
        choices=("torch", "fused"),
        default="torch",
        help="opt-in fused projection checks; requires device decode, sign_fused_direct and rebuilt native operators",
    )
    parser.add_argument(
        "--profile-dir", type=Path, help="enable vLLM's Torch profiler controls for a separate trace run"
    )
    parser.add_argument(
        "--execution-mode",
        choices=("eager", "decode_graph"),
        default="decode_graph",
        help="MRV2 FULL_DECODE_ONLY ACLGraph; requires --decode-mode device",
    )
    check_mode = parser.add_mutually_exclusive_group()
    check_mode.add_argument("--dry-run", action="store_true")
    check_mode.add_argument(
        "--check-native-only",
        action="store_true",
        help="check native operator registration without loading model weights",
    )
    return parser.parse_args(argv)


def build_command(args):
    """Validate model assets and execution limits, then build the vLLM serve command."""
    if args.library is not None:
        raise ValueError(
            "DeepSeek-V4 serving requires the main vllm_ascend_C extension for HC and attention; "
            "omit --library. The standalone VQ library is for operator tests only."
        )
    context = DEFAULT_CONTEXT if args.max_model_len is None else args.max_model_len
    if context <= 0 or args.prefill_chunk_size <= 0:
        raise ValueError("Context and prefill chunk size must be positive.")
    if args.num_speculative_tokens < 0 or args.num_device_experts <= 0:
        raise ValueError("Invalid speculative token count or resident expert count.")
    if bool(args.num_speculative_tokens) != bool(args.draft_model):
        raise ValueError("Supply both --draft-model and a positive --num-speculative-tokens.")
    draft_quantization = None
    if args.draft_model is not None:
        draft = args.draft_model.resolve(strict=True)
        draft_config = json.loads((draft / "config.json").read_text())
        draft_quantization = (draft_config.get("quantization_config") or {}).get("quant_method")
        if draft_quantization == "vq2a8":
            raise ValueError("DSpark draft must retain its original format, not VQ2A8.")
        if draft_quantization == "fp8" and draft_config.get("model_type") == "deepseek_v4":
            # The generic FP8 backend does not implement this checkpoint's MXFP4 experts.
            draft_quantization = "deepseek_v4_fp8"
        elif draft_quantization == "modelslim":
            draft_quantization = "ascend"

    if not math.isfinite(args.memory_fraction) or not 0 < args.memory_fraction <= 1:
        raise ValueError("memory_fraction must be finite and in (0,1].")
    if not math.isfinite(args.reserve_gib) or args.reserve_gib < 1:
        raise ValueError("reserve_gib must be finite and >=1.")
    if args.kv_cache_mib <= 0 or args.kv_cache_mib > args.reserve_gib * 1024:
        raise ValueError("Positive KV cache allocation must fit inside reserve_gib.")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in [1,65535].")
    model = args.model.resolve(strict=True)
    if not (model / "config.json").is_file():
        raise ValueError("Model config.json is required.")
    additional = {
        "enable_flashcomm1": False,
        "mix_placement": False,
        "multistream_dsv4_dsa_overlap": False,
    }
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    quantization = config.get("quantization_config", {})
    if (
        not isinstance(quantization, dict)
        or quantization.get("quant_method") != "vq2a8"
        or quantization.get("format") not in ("ascend_vq2a8", "ascend_vq2a8_gs32")
        or quantization.get("format_version") != 1
        or not (model / "model.safetensors.index.json").is_file()
    ):
        raise ValueError("Select a unified VQ2A8 checkpoint (ascend_vq2a8 version 1) with a safetensors index.")
    if args.decode_mode != "device" and args.activation_preparation != "packed":
        raise ValueError("--activation-preparation sign_fused_direct requires --decode-mode device.")
    if args.execution_mode == "decode_graph" and args.decode_mode != "device":
        raise ValueError("--execution-mode decode_graph requires --decode-mode device.")
    if args.validity_mode == "fused" and (
        args.decode_mode != "device" or args.activation_preparation != "sign_fused_direct"
    ):
        raise ValueError(
            "--validity-mode fused requires --decode-mode device --activation-preparation sign_fused_direct."
        )
    additional["vq2a8"] = (
        {"decode_mode": "device", "activation_preparation": args.activation_preparation}
        if args.decode_mode == "device"
        else {}
    )
    if args.validity_mode == "fused":
        additional["vq2a8"]["validity_mode"] = args.validity_mode
    if args.prefill_rht != "rowwise":
        additional["vq2a8"]["prefill_rht"] = args.prefill_rht
    if args.prefill_batch_mode != "jobs":
        additional["vq2a8"]["prefill_batch_mode"] = args.prefill_batch_mode
    if quantization.get("format") == "ascend_vq2a8_gs32":
        if args.activation_preparation != "packed" or args.prefill_rht != "rowwise" or args.validity_mode != "torch":
            raise ValueError("RHT-specific activation/validity options cannot be used with GS32.")
        if args.gs32_library is None or args.gs32_binary is None:
            raise ValueError("GS32 requires --gs32-library and --gs32-binary.")
        native = {
            "gs32_library": args.gs32_library.resolve(strict=True),
            "gs32_binary": args.gs32_binary.resolve(strict=True),
        }
        if any(not path.is_file() for path in native.values()):
            raise ValueError("GS32 native artifacts must be regular files.")
        execution = args.gs32_execution or ("full_graph" if args.execution_mode == "decode_graph" else "serial")
        if (execution == "full_graph") != (args.execution_mode == "decode_graph"):
            raise ValueError("GS32 execution must match eager/decode_graph mode.")
        additional["vq2a8"] = {key: str(value) for key, value in native.items()}
        additional["vq2a8"]["gs32_execution"] = execution
        verify_tokens = args.num_speculative_tokens + 1
        if verify_tokens > 128 or verify_tokens * config["num_experts_per_tok"] > 256:
            raise ValueError("GS32 verification block exceeds native routed-row capacity.")
    elif args.gs32_library or args.gs32_binary or args.gs32_execution:
        raise ValueError("GS32 native options require a GS32 checkpoint.")
    if args.expert_offload:
        additional["expert_offload_config"] = {"enabled": True, "num_device_experts": args.num_device_experts}
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(model),
        "--served-model-name",
        "vq2a8",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--load-format",
        "safetensors",
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
        "--distributed-executor-backend",
        "uni",
        "--enforce-eager",
        "--compilation-config",
        json.dumps({"mode": 0, "cudagraph_mode": "NONE"}),
        "--no-async-scheduling",
        "--enable-prefix-caching" if args.prefix_caching else "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--max-num-seqs",
        "1",
        "--max-model-len",
        str(context),
        "--max-num-batched-tokens",
        str(min(context, args.prefill_chunk_size)),
        "--block-size",
        "32",
        "--kv-cache-dtype",
        # A5's GS32 attention plan uses compressed FP8 KV; upstream auto resolves to BF16.
        "fp8"
        if quantization.get("format") == "ascend_vq2a8_gs32" and args.kv_cache_dtype == "auto"
        else args.kv_cache_dtype,
        "--kv-cache-memory-bytes",
        str(args.kv_cache_mib * 1024**2),
        "--gpu-memory-utilization",
        str(args.memory_fraction),
        "--seed",
        "0",
        "--disable-log-stats",
        "--generation-config",
        "vllm",
        "--additional-config",
        json.dumps(additional),
    ]
    command.extend(["--quantization", "vq2a8"])
    if args.draft_model is not None:
        command.extend(
            [
                "--speculative-config",
                json.dumps(
                    {
                        "method": "dspark",
                        "model": str(args.draft_model.resolve()),
                        # Upstream otherwise inherits target quantization, even for another checkpoint.
                        "quantization": draft_quantization,
                        "num_speculative_tokens": args.num_speculative_tokens,
                        "enforce_eager": True,
                    }
                ),
            ]
        )
    if args.execution_mode == "decode_graph":
        command.remove("--enforce-eager")
        command[command.index("--compilation-config") + 1] = json.dumps(
            {
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [args.num_speculative_tokens + 1],
                "max_cudagraph_capture_size": args.num_speculative_tokens + 1,
                "cudagraph_num_of_warmups": 1,
            }
        )
        additional["ascend_compilation_config"] = {"enable_npugraph_ex": False}
        command[command.index("--additional-config") + 1] = json.dumps(additional)
    if args.profile_dir is not None:
        command.extend(
            [
                "--profiler-config",
                json.dumps(
                    {
                        "profiler": "torch",
                        "torch_profiler_dir": str(args.profile_dir.resolve()),
                        "torch_profiler_with_stack": False,
                    }
                ),
            ]
        )
    return command


def check_native_dependencies(activation_preparation="packed", *, validity_mode="torch"):
    """Check A5 model bindings before loading weights; this does not execute kernels."""
    # Keep CANN/vLLM initialization out of argument parsing and --dry-run.
    import torch
    import torch_npu  # noqa: F401

    from vllm_ascend.ops.vq2a8 import require_vq2a8_features

    # The default provider bootstraps the vendor environment before importing
    # the main extension, which supplies both VQ and the model's HC/DSA ops.
    if validity_mode != "torch":
        require_vq2a8_features(activation_preparation=activation_preparation, validity_mode=validity_mode)
    elif activation_preparation == "packed":
        require_vq2a8_features()
    else:
        require_vq2a8_features(activation_preparation=activation_preparation)
    missing = [
        name
        for name in DEEPSEEK_V4_REQUIRED_OPS
        if not hasattr(torch.ops._C_ascend, name)
        or not torch._C._dispatch_has_kernel_for_dispatch_key(f"_C_ascend::{name}", "PrivateUse1")
    ]
    if missing:
        raise RuntimeError(
            "The Ascend950 main extension is missing DeepSeek-V4 NPU bindings: "
            f"{', '.join(missing)}. Build/install the full vllm_ascend_C extension and its ACLNN operators. "
            "Building the standalone VQ library alone is insufficient."
        )


def main(argv=None):
    args = parse_args(argv)
    try:
        command = build_command(args)
        if args.dry_run:
            print("VLLM_USE_V2_MODEL_RUNNER=1 " + shlex.join(command))
        else:
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
            if args.validity_mode != "torch":
                check_native_dependencies(args.activation_preparation, validity_mode=args.validity_mode)
            elif args.activation_preparation == "packed":
                check_native_dependencies()
            else:
                check_native_dependencies(args.activation_preparation)
            if args.check_native_only:
                print("VQ2A8_NATIVE_CHECK=PASS (operator registration only; no device execution)")
            else:
                os.execv(command[0], command)
        return 0
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Resolve repository modules without putting tools ahead of dependencies.
    sys.path[0] = str(Path(__file__).resolve().parents[1])
    raise SystemExit(main())
