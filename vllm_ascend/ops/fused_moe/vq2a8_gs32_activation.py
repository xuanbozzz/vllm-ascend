# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GS32 activation entry points with explicit eager or deferred validation."""

import torch

from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_staged import MX_GROUP, staged_swiglu_unchecked


def prepare_activation(hidden, weight_scale, perm, row_slots, raw_gate=None):
    # Lazy worker loading keeps CPU contract tests independent of Triton/NPU.
    from vllm_ascend.ops.triton.activation.vq2a8_gs32 import prepare_activation as launch

    return launch(hidden, weight_scale, perm, row_slots, raw_gate)


def finite_flags(tensor):
    from vllm_ascend.ops.triton.activation.vq2a8_gs32 import finite_flags as launch

    return launch(tensor)


def finish_activation_flags(output, gate_flags, down_flags, *, route_flags=None, native_flags=None):
    from vllm_ascend.ops.triton.activation.vq2a8_gs32 import finish_flags as launch

    return launch(output, gate_flags, down_flags, route_flags=route_flags, native_flags=native_flags)


def _require_npu_tensor(tensor, name):
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "npu" or tensor.device.index is None:
        raise ValueError(f"{name} requires an explicit NPU device; CPU execution is unsupported.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def _require_eager():
    capturing = getattr(torch.npu, "is_current_stream_capturing", None)
    if capturing is not None and capturing():
        raise RuntimeError("GS32 fused activation does not support graph capture.")


def _validate_prepare(hidden, weight_scale, perm, row_slots, raw_gate):
    tensors = {"hidden": hidden, "weight_scale": weight_scale, "perm": perm, "row_slots": row_slots}
    if raw_gate is not None:
        tensors["raw_gate"] = raw_gate
    for name, tensor in tensors.items():
        _require_npu_tensor(tensor, name)
        if tensor.device != hidden.device:
            raise ValueError("GS32 fused activation tensors must share the same device.")
    if hidden.dtype != torch.bfloat16 or hidden.ndim != 2 or hidden.shape[1] < MX_GROUP or hidden.shape[1] % MX_GROUP:
        raise ValueError("GS32 fused activation requires BF16 [M,K] with positive K divisible by 32.")
    rows, width = hidden.shape
    if (
        weight_scale.dtype != torch.float32
        or weight_scale.ndim != 2
        or weight_scale.shape[0] < 1
        or weight_scale.shape[1] != width
        or perm.dtype != torch.int64
        or perm.shape != weight_scale.shape
    ):
        raise ValueError("GS32 fused activation requires FP32 weight_scale and int64 perm with shape [E,K].")
    if row_slots.dtype != torch.int64 or row_slots.shape != (rows,):
        raise ValueError("GS32 fused activation requires int64 row_slots with shape [M].")
    if raw_gate is not None and (raw_gate.dtype != torch.bfloat16 or raw_gate.shape != (rows, 2 * width)):
        raise ValueError("GS32 fused activation raw_gate must be BF16 [M,2K].")


def prepare(hidden, weight_scale, perm, row_slots, raw_gate=None):
    _validate_prepare(hidden, weight_scale, perm, row_slots, raw_gate)
    with torch.npu.device(hidden.device):
        _require_eager()
        if hidden.shape[0] == 0:
            return hidden.new_empty(hidden.shape, dtype=torch.uint8), hidden.new_empty(
                (0, hidden.shape[1] // MX_GROUP), dtype=torch.uint8
            )
        data, scales, flags = prepare_activation(hidden, weight_scale, perm, row_slots, raw_gate)
        # Exactly one small host transfer and decision, before native launch.
        if not all(flags.cpu().tolist()):
            raise ValueError("GS32 fused activation inputs contain non-finite values or invalid indices.")
        return data, scales


def prepare_unchecked(hidden, weight_scale, perm, row_slots, raw_gate=None):
    """Return device validity that must be checked before accepting results.

    Metadata checks remain synchronous host checks. Numerical flags remain
    device tensors so a graph owner can combine them without a host transfer.
    The kernel sanitizes invalid numerical inputs before emitting FP8 bytes.
    """
    _validate_prepare(hidden, weight_scale, perm, row_slots, raw_gate)
    with torch.npu.device(hidden.device):
        if hidden.shape[0] == 0:
            return (
                hidden.new_empty(hidden.shape, dtype=torch.uint8),
                hidden.new_empty((0, hidden.shape[1] // MX_GROUP), dtype=torch.uint8),
                hidden.new_empty((0,), dtype=torch.int32),
            )
        return prepare_activation(hidden, weight_scale, perm, row_slots, raw_gate)


def finish_flags(output, gate_flags, down_flags, *, route_flags=None, native_flags=None):
    """Combine both preparation flags and output finiteness without a CPU read."""
    _require_npu_tensor(output, "GS32 output")
    if output.ndim != 2 or output.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("GS32 output requires a BF16, FP16 or FP32 [M,N] tensor.")
    for name, flags in (("gate_flags", gate_flags), ("down_flags", down_flags)):
        _require_npu_tensor(flags, name)
        if flags.device != output.device or flags.dtype != torch.int32 or flags.shape != (output.shape[0],):
            raise ValueError(f"{name} must be int32 [M] on the output device.")
    external = {"route_flags": route_flags, "native_flags": native_flags}
    for name, flags in external.items():
        if flags is not None:
            _require_npu_tensor(flags, name)
            if flags.device != output.device or flags.dtype != torch.int32 or flags.ndim != 1 or not flags.numel():
                raise ValueError(f"{name} must be a nonempty int32 vector on the output device.")
    with torch.npu.device(output.device):
        if route_flags is None and native_flags is None:
            return finish_activation_flags(output, gate_flags, down_flags)
        return finish_activation_flags(output, gate_flags, down_flags, **external)


def require_finite_fused(tensor, name):
    _require_npu_tensor(tensor, name)
    if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"{name} requires a BF16, FP16 or FP32 tensor.")
    with torch.npu.device(tensor.device):
        _require_eager()
        if tensor.numel() and not all(finite_flags(tensor).cpu().tolist()):
            raise ValueError(f"{name} contains NaN or Inf.")


def swiglu_unchecked(gate_up, limit):
    """Keep staged Torch arithmetic; the subsequent prepare checks raw GateUp."""
    return staged_swiglu_unchecked(gate_up, limit)
