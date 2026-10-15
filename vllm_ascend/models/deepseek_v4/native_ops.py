# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load required model bindings in the process that executes DeepSeek-V4."""

import importlib

DEEPSEEK_V4_REQUIRED_OPS = (
    "npu_hc_pre_v2",
    "npu_hc_post",
    "moe_gating_top_k_hash",
    "inplace_partial_rotary_mul",
    "compressor",
    "compressor_metadata",
    "compressor_metadata_out",
    "npu_quant_lightning_indexer_v2",
    "npu_quant_lightning_indexer_v2_metadata",
    "npu_kv_quant_sparse_attn_sharedkv",
    "npu_kv_quant_sparse_attn_sharedkv_metadata",
    "kv_compress_epilog",
    "indexer_compress_epilog_v2",
)


def require_deepseek_v4_native_ops() -> str:
    """Register and check NPU bindings, without enabling generic op replacements.

    Call during model construction, after worker device selection. Imports stay
    lazy so model discovery does not initialize CANN before device visibility is
    configured. A parent-process check cannot replace this worker-local import.
    """
    import torch
    import torch_npu  # noqa: F401

    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env()
    extension = importlib.import_module("vllm_ascend.vllm_ascend_C")
    missing = [
        name
        for name in DEEPSEEK_V4_REQUIRED_OPS
        if not hasattr(torch.ops._C_ascend, name)
        or not torch._C._dispatch_has_kernel_for_dispatch_key(f"_C_ascend::{name}", "PrivateUse1")
    ]
    if missing:
        raise RuntimeError(
            f"The Ascend950 main extension {extension.__file__} is missing DeepSeek-V4 NPU bindings: "
            f"{', '.join(missing)}. Build/install the full vllm_ascend_C extension and its ACLNN operators. "
            "Building the standalone GS32 library alone is insufficient."
        )
    return extension.__file__
