# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused activation wrapper contracts; CPU tests never execute an NPU kernel."""

import importlib
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch


def activation_module():
    return importlib.import_module("vllm_ascend.ops.fused_moe.vq2a8_gs32_activation")


@dataclass(frozen=True)
class FakeDevice:
    type: str = "npu"
    index: int = 0


class NpuTaggedCpuTensor(torch.Tensor):
    """Metadata double only: storage and arithmetic still live on the CPU."""

    @property
    def device(self):
        return self.fake_device


def tagged(tensor, *, index=0):
    value = tensor.as_subclass(NpuTaggedCpuTensor)
    value.fake_device = FakeDevice(index=index)
    return value


def valid_inputs():
    return (
        tagged(torch.zeros(3, 64, dtype=torch.bfloat16)),
        tagged(torch.ones(2, 64, dtype=torch.float32)),
        tagged(torch.arange(64, dtype=torch.int64).repeat(2, 1)),
        tagged(torch.tensor([1, 0, 1], dtype=torch.int64)),
    )


class HostFlags:
    """Count the wrapper's host transfers without claiming device execution."""

    def __init__(self, values):
        self.values = values
        self.transfers = 0

    def cpu(self):
        self.transfers += 1
        return self

    def tolist(self):
        return self.values


@pytest.fixture
def wrapper(monkeypatch):
    activation = activation_module()
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(is_current_stream_capturing=lambda: False, device=lambda device: nullcontext()),
        raising=False,
    )
    return activation


def test_gs32_fused_activation_rejects_cpu_execution():
    activation = activation_module()
    with pytest.raises(ValueError, match="NPU|npu"):
        activation.prepare(
            torch.zeros(1, 64, dtype=torch.bfloat16),
            torch.ones(2, 64, dtype=torch.float32),
            torch.arange(64, dtype=torch.int64).repeat(2, 1),
            torch.zeros(1, dtype=torch.int64),
        )


@pytest.mark.parametrize(
    "index,replacement",
    [
        (0, lambda: tagged(torch.zeros(3, 64, dtype=torch.float32))),
        (0, lambda: tagged(torch.zeros(3, 63, dtype=torch.bfloat16))),
        (0, lambda: tagged(torch.zeros(3, 64, 1, dtype=torch.bfloat16))),
        (0, lambda: tagged(torch.zeros(64, 3, dtype=torch.bfloat16).T)),
        (1, lambda: tagged(torch.ones(2, 64, dtype=torch.bfloat16))),
        (1, lambda: tagged(torch.ones(2, 32, dtype=torch.float32))),
        (1, lambda: tagged(torch.ones(2, 64, dtype=torch.float32), index=1)),
        (2, lambda: tagged(torch.arange(64, dtype=torch.int32).repeat(2, 1))),
        (2, lambda: tagged(torch.arange(64, dtype=torch.int64).repeat(3, 1))),
        (3, lambda: tagged(torch.zeros(3, dtype=torch.int32))),
        (3, lambda: tagged(torch.zeros(4, dtype=torch.int64))),
        (3, lambda: tagged(torch.zeros(3, 1, dtype=torch.int64))),
    ],
)
def test_gs32_fused_activation_invalid_metadata_never_launches(wrapper, monkeypatch, index, replacement):
    args = list(valid_inputs())
    args[index] = replacement()

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid metadata reached the fusion kernel.")

    monkeypatch.setattr(wrapper, "prepare_activation", forbidden)
    with pytest.raises(ValueError):
        wrapper.prepare(*args)


@pytest.mark.parametrize(
    "raw_gate",
    [
        lambda: tagged(torch.zeros(3, 127, dtype=torch.bfloat16)),
        lambda: tagged(torch.zeros(2, 128, dtype=torch.bfloat16)),
        lambda: tagged(torch.zeros(3, 128, dtype=torch.float32)),
        lambda: tagged(torch.zeros(3, 128, dtype=torch.bfloat16), index=1),
    ],
)
def test_gs32_fused_activation_invalid_raw_gate_never_launches(wrapper, monkeypatch, raw_gate):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid raw gate metadata reached the fusion kernel.")

    monkeypatch.setattr(wrapper, "prepare_activation", forbidden)
    with pytest.raises(ValueError):
        wrapper.prepare(*valid_inputs(), raw_gate=raw_gate())


def test_gs32_fused_activation_forwards_all_buffers_and_reads_flags_once(wrapper, monkeypatch):
    args = valid_inputs()
    raw_gate = tagged(torch.zeros(3, 128, dtype=torch.bfloat16))
    data = torch.arange(3 * 64, dtype=torch.uint8).reshape(3, 64)
    scale = torch.full((3, 2), 127, dtype=torch.uint8)
    flags = HostFlags([1, 1, 1])
    calls = []

    def launch(*received, **kwargs):
        calls.append((received, kwargs))
        return data, scale, flags

    monkeypatch.setattr(wrapper, "prepare_activation", launch)
    actual_data, actual_scale = wrapper.prepare(*args, raw_gate=raw_gate)
    assert actual_data is data and actual_scale is scale
    assert flags.transfers == 1 and len(calls) == 1
    received, kwargs = calls[0]
    assert all(got is expected for got, expected in zip(received[:4], args, strict=True))
    assert (received[4] if len(received) == 5 else kwargs["raw_gate"]) is raw_gate


@pytest.mark.parametrize("flag_values", [[0, 1, 1], [1, 0, 1], [1, 1, 0]])
def test_gs32_fused_activation_bad_flag_fails_before_native_consumer(wrapper, monkeypatch, flag_values):
    flags = HostFlags(flag_values)
    monkeypatch.setattr(
        wrapper,
        "prepare_activation",
        lambda *args, **kwargs: (torch.zeros(3, 64, dtype=torch.uint8), torch.ones(3, 2, dtype=torch.uint8), flags),
    )
    consumed = []

    def pipeline():
        result = wrapper.prepare(*valid_inputs())
        consumed.append(result)

    with pytest.raises(ValueError, match="finite|invalid"):
        pipeline()
    assert consumed == [] and flags.transfers == 1


def test_gs32_fused_activation_failure_does_not_poison_next_call(wrapper, monkeypatch):
    flags = [HostFlags([1, 0, 1]), HostFlags([1, 1, 1])]

    def launch(*args, **kwargs):
        return torch.zeros(3, 64, dtype=torch.uint8), torch.full((3, 2), 127, dtype=torch.uint8), flags.pop(0)

    monkeypatch.setattr(wrapper, "prepare_activation", launch)
    with pytest.raises(ValueError):
        wrapper.prepare(*valid_inputs())
    data, scale = wrapper.prepare(*valid_inputs())
    assert data.shape == (3, 64) and scale.shape == (3, 2)
    assert flags == []


@pytest.mark.parametrize("flag_values", [[1, 1], [1, 0]])
def test_gs32_fused_activation_output_finite_flags_preserve_name_and_one_transfer(wrapper, monkeypatch, flag_values):
    flags = HostFlags(flag_values)
    output = tagged(torch.zeros(3, 128, dtype=torch.bfloat16))
    monkeypatch.setattr(wrapper, "finite_flags", lambda tensor: flags)
    if all(flag_values):
        wrapper.require_finite_fused(output, "Down native output")
    else:
        with pytest.raises(ValueError, match="Down native output"):
            wrapper.require_finite_fused(output, "Down native output")
    assert flags.transfers == 1


def test_gs32_fused_activation_capture_rejected_before_launch(wrapper, monkeypatch):
    torch.npu.is_current_stream_capturing = lambda: True

    def forbidden(*args, **kwargs):
        pytest.fail("Graph capture reached an eager-only fusion kernel.")

    monkeypatch.setattr(wrapper, "prepare_activation", forbidden)
    with pytest.raises(RuntimeError, match="capture|graph"):
        wrapper.prepare(*valid_inputs())


def test_gs32_fused_activation_empty_dispatch_does_not_launch(wrapper, monkeypatch):
    args = list(valid_inputs())
    args[0] = tagged(torch.empty(0, 64, dtype=torch.bfloat16))
    args[3] = tagged(torch.empty(0, dtype=torch.int64))

    def forbidden(*args, **kwargs):
        pytest.fail("Empty activation preparation launched a kernel.")

    monkeypatch.setattr(wrapper, "prepare_activation", forbidden)
    data, scale = wrapper.prepare(*args)
    assert data.shape == (0, 64) and scale.shape == (0, 2)
    assert data.dtype == scale.dtype == torch.uint8


@pytest.mark.parametrize("limit", [0.0, 10.0])
def test_gs32_fused_activation_unchecked_swiglu_preserves_original_arithmetic(limit):
    from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_staged import staged_swiglu, staged_swiglu_unchecked

    generator = torch.Generator().manual_seed(314)
    raw = (torch.randn(6, 128, generator=generator) * 20).bfloat16()
    gate, up = raw.float().chunk(2, dim=-1)
    if limit > 0:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    expected = (torch.nn.functional.silu(gate) * up).bfloat16().contiguous()
    assert torch.equal(staged_swiglu_unchecked(raw, limit).view(torch.int16), expected.view(torch.int16))
    assert torch.equal(staged_swiglu(raw, limit).view(torch.int16), expected.view(torch.int16))


def test_gs32_fused_activation_unchecked_requires_raw_gate_check_after_clamp():
    from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_staged import staged_swiglu, staged_swiglu_unchecked

    raw = torch.ones(1, 128, dtype=torch.bfloat16)
    raw[0, 0] = float("inf")
    assert torch.isfinite(staged_swiglu_unchecked(raw, 10)).all()
    with pytest.raises(ValueError, match="GateUp output"):
        staged_swiglu(raw, 10)


@pytest.mark.parametrize("limit", [-1.0, float("nan"), float("inf")])
def test_gs32_fused_activation_unchecked_still_validates_swiglu_metadata(limit):
    from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_staged import staged_swiglu_unchecked

    with pytest.raises(ValueError):
        staged_swiglu_unchecked(torch.ones(1, 128, dtype=torch.bfloat16), limit)
