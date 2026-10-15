# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in real-NPU activation bytes; no model loading or native projection.

The GS32 model option opts into Ascend950 execution and checks the checkpoint
format only. Independent CPU quantization comes from the imported operator's
NumPy/ml_dtypes reference, never the new fused implementation.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def activation_runtime(request):
    selected = request.config.getoption("--vq2a8-model", default=None)
    if not selected:
        pytest.skip("Pass a unified GS32 --vq2a8-model to opt into real Ascend activation execution.")
    model = Path(selected).resolve(strict=True)
    config = json.loads((model / "config.json").read_text())
    assert config["quantization_config"]["format"] == "ascend_vq2a8_gs32"

    import torch
    import torch_npu  # noqa: F401

    from vllm_ascend.ops.fused_moe.vq2a8_gs32_activation import prepare, require_finite_fused
    from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_staged import quantize_mxfp8_k32, staged_swiglu

    assert torch.npu.is_available(), "Explicit activation tests require a real Ascend NPU."
    # This is a required independent oracle for explicitly requested hardware
    # tests. Missing ml_dtypes is a failure, not a silently skipped comparison.
    source = Path(__file__).resolve().parents[3] / "csrc/mixed_w2w4_moe_ffn/scripts/swiglu_reference.py"
    spec = importlib.util.spec_from_file_location("_gs32_activation_independent_reference", source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    device = torch.device("npu", 0)
    torch.npu.set_device(device)
    return SimpleNamespace(
        torch=torch,
        device=device,
        prepare=prepare,
        finite=require_finite_fused,
        strict_quantize=quantize_mxfp8_k32,
        strict_swiglu=staged_swiglu,
        reference=reference,
    )


def activation_inputs(runtime, rows, width):
    torch = runtime.torch
    generator = torch.Generator(device="cpu").manual_seed(952)
    hidden = (torch.randn(rows, width, generator=generator) * 0.25).bfloat16()
    scales = torch.stack([torch.linspace(0.5, 1.5, width) * factor for factor in (1.0, 1.125, 0.875)])
    permutation = torch.stack([torch.roll(torch.arange(width, dtype=torch.int64), shift) for shift in (23, -17, 31)])
    row_slots = torch.tensor([2, 0, 1], dtype=torch.int64).repeat((rows + 2) // 3)[:rows].contiguous()
    return hidden, scales, permutation, row_slots


def compare_activation(runtime, host_inputs, *, raw_gate=None, check_strict=True):
    torch = runtime.torch
    hidden, weight_scale, permutation, row_slots = host_inputs
    # Independent source-order indexing, deliberately separate from Torch's
    # batched scale/gather and from the Triton implementation.
    host_hidden, host_scale = hidden.float().numpy(), weight_scale.numpy()
    host_perm, host_slots = permutation.numpy(), row_slots.numpy()
    transformed = np.stack(
        [host_hidden[row, host_perm[slot]] * host_scale[slot, host_perm[slot]] for row, slot in enumerate(host_slots)]
    )
    expected_data, expected_scale = runtime.reference.quantize_h(transformed)
    device_inputs = tuple(value.to(runtime.device) for value in host_inputs)
    device_raw = None if raw_gate is None else raw_gate.to(runtime.device)
    actual_data, actual_scale = runtime.prepare(*device_inputs, raw_gate=device_raw)
    torch.npu.synchronize()
    np.testing.assert_array_equal(actual_data.cpu().numpy(), expected_data.view(np.uint8))
    np.testing.assert_array_equal(actual_scale.cpu().numpy(), expected_scale)
    if check_strict:
        x, scale, perm, slots = device_inputs
        original = (x.float() * scale.index_select(0, slots)).gather(1, perm.index_select(0, slots))
        strict_data, strict_scale = runtime.strict_quantize(original)
        assert torch.equal(actual_data.cpu(), strict_data.cpu())
        assert torch.equal(actual_scale.cpu(), strict_scale.cpu())
    return actual_data, actual_scale


@pytest.mark.parametrize(
    "rows,width", [(1, 64), (3, 64), (3, 96), (6, 2048), (24, 4096), (32, 4096), (127, 64), (128, 64), (129, 64)]
)
def test_gs32_fused_activation_bytes_match_independent_cpu_and_same_npu(activation_runtime, rows, width):
    runtime = activation_runtime
    compare_activation(runtime, activation_inputs(runtime, rows, width))


def test_gs32_fused_activation_all_finite_bf16_patterns_and_signed_zero(activation_runtime):
    runtime, torch = activation_runtime, activation_runtime.torch
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16)
    values = values[torch.isfinite(values.float())]
    assert values.numel() == 65280
    hidden = values[:, None].expand(-1, 64).contiguous()
    fields = (
        hidden,
        torch.ones(1, 64, dtype=torch.float32),
        torch.arange(64, dtype=torch.int64)[None, :],
        torch.zeros(hidden.shape[0], dtype=torch.int64),
    )
    compare_activation(runtime, fields)


def test_gs32_fused_activation_mixed_ties_and_e8m0_boundaries(activation_runtime):
    runtime, torch = activation_runtime, activation_runtime.torch
    values = torch.tensor(
        [
            0.0,
            -0.0,
            1 / 1024,
            -1 / 1024,
            3 / 1024,
            -3 / 1024,
            1.0625,
            -1.0625,
            1.1875,
            -1.1875,
            1.75,
            -1.75,
            1.7578125,
            -1.7578125,
            223.0,
            -223.0,
            224.0,
            -224.0,
            225.0,
            -225.0,
            226.0,
            -226.0,
            447.0,
            -447.0,
            448.0,
            -448.0,
            15 / 1024,
            -15 / 1024,
            1 / 64,
            -1 / 64,
            1 / 512,
            -1 / 512,
        ]
    )
    rows = []
    for exponent in (-120, -16, 0, 16, 110):
        for maximum in (224.0, 226.0, 448.0, 450.0):
            row = values.clamp(-maximum, maximum)
            row[-2], row[-1] = maximum, -maximum
            rows.append((row * (2.0**exponent)).repeat(2))
    hidden = torch.stack(rows).bfloat16()
    compare_activation(
        runtime,
        (hidden, torch.ones(1, 64), torch.arange(64)[None, :], torch.zeros(len(rows), dtype=torch.int64)),
    )


@pytest.mark.parametrize("limit", [0.0, 10.0])
def test_gs32_fused_activation_down_preserves_original_swiglu_rounding(activation_runtime, limit):
    runtime, torch = activation_runtime, activation_runtime.torch
    _, scales, permutation, slots = activation_inputs(runtime, 6, 64)
    generator = torch.Generator(device="cpu").manual_seed(791)
    gate = (torch.randn(6, 128, generator=generator) * torch.arange(1, 7)[:, None] * 3).bfloat16()
    gate[:, 0], gate[:, 64] = -20, -30
    gate[:, 1], gate[:, 65] = 20, 30
    # SwiGLU remains the SAME Torch/NPU arithmetic; fusion starts after its
    # mandated BF16 round. CPU quantization then independently checks bytes.
    middle = runtime.strict_swiglu(gate.to(runtime.device), limit).cpu()
    compare_activation(runtime, (middle, scales, permutation, slots), raw_gate=gate)


@pytest.mark.parametrize(
    "fault",
    [
        "input_nan",
        "input_inf",
        "fp32_overflow",
        "bf16_round_overflow",
        "slot_negative",
        "slot_oob",
        "perm_oob",
        "swiglu_overflow",
        "raw_gate_inf",
    ],
)
def test_gs32_fused_activation_invalid_data_never_reaches_projection(activation_runtime, fault):
    runtime, torch = activation_runtime, activation_runtime.torch
    hidden, scale, perm, slots = activation_inputs(runtime, 3, 64)
    raw_gate = None
    if fault in ("input_nan", "input_inf"):
        hidden[2, 17] = float("nan") if fault == "input_nan" else float("inf")
    elif fault in ("fp32_overflow", "bf16_round_overflow"):
        hidden[2, 17] = torch.finfo(torch.bfloat16).max
        scale.fill_(2.0 if fault == "fp32_overflow" else 1.002)
    elif fault == "slot_negative":
        slots[2] = -1
    elif fault == "slot_oob":
        slots[2] = scale.shape[0]
    elif fault == "perm_oob":
        perm[1, 17] = 64
    elif fault == "swiglu_overflow":
        raw_gate = torch.full((3, 128), torch.finfo(torch.bfloat16).max, dtype=torch.bfloat16)
        gate, up = raw_gate.to(runtime.device).float().chunk(2, dim=-1)
        hidden = (torch.nn.functional.silu(gate) * up).bfloat16().cpu()
        assert not torch.isfinite(hidden).all()
    else:
        raw_gate = torch.zeros(3, 128, dtype=torch.bfloat16)
        raw_gate[2, 17] = float("inf")
        # Gate clamping could make the arithmetic finite. Its original
        # nonfinite input must remain an error instead of being hidden.
        gate, up = raw_gate.to(runtime.device).float().chunk(2, dim=-1)
        hidden = (torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)).bfloat16().cpu()
        assert torch.isfinite(hidden).all()
    device_inputs = tuple(value.to(runtime.device) for value in (hidden, scale, perm, slots))
    native_consumed = []
    with pytest.raises(ValueError, match="finite|invalid"):
        prepared = runtime.prepare(*device_inputs, raw_gate=None if raw_gate is None else raw_gate.to(runtime.device))
        native_consumed.append(prepared)
    assert native_consumed == []
    torch.npu.synchronize()
    # An error flag is local to a launch and must not poison the next request.
    compare_activation(runtime, activation_inputs(runtime, 3, 64))


def test_gs32_fused_activation_output_holes_and_empty_dispatch(activation_runtime):
    runtime, torch = activation_runtime, activation_runtime.torch
    output = torch.full((3, 128), float("nan"), dtype=torch.bfloat16, device=runtime.device)
    output.flatten()[:-1].fill_(1)
    with pytest.raises(ValueError, match="partial native output"):
        runtime.finite(output, "partial native output")
    output.fill_(1)
    runtime.finite(output, "complete native output")
    empty = tuple(value.to(runtime.device) for value in activation_inputs(runtime, 0, 64))
    data, scale = runtime.prepare(*empty)
    assert data.shape == (0, 64) and scale.shape == (0, 2)
    assert data.dtype == scale.dtype == torch.uint8


def test_gs32_fused_activation_nondefault_stream_repeated_fresh_values(activation_runtime):
    runtime, torch = activation_runtime, activation_runtime.torch
    stream = torch.npu.Stream(device=runtime.device)
    stream.wait_stream(torch.npu.current_stream(runtime.device))
    with torch.npu.stream(stream):
        for delta in (0.03125, -0.0625):
            host = list(activation_inputs(runtime, 7, 2048))
            host[0] = (host[0].float() + delta).bfloat16()
            compare_activation(runtime, tuple(host))
    torch.npu.current_stream(runtime.device).wait_stream(stream)
