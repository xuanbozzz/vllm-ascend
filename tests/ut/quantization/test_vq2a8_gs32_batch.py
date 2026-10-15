# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch activation preparation without changing per-expert GS32 numerics."""

from types import SimpleNamespace

import pytest
import torch

from tests.ut.quantization.test_vq2a8_gs32_layout import fixture
from tests.ut.quantization.test_vq2a8_gs32_staged import FakeDownOps
from tools.vq2a8_gs32_layout import repack_gs32_matrix
from tools.vq2a8_gs32_reference import (
    quantize_mxfp8_k32,
    staged_expert_reference,
    staged_projection_reference,
    staged_swiglu,
)


class RecordingOps(FakeDownOps):
    """CPU packed-byte oracle recording actual native launch boundaries."""

    def __init__(self):
        super().__init__()
        self.shapes = []
        self.activation_bytes = []

    def project(self, *args):
        self.shapes.append(tuple(args[-1].shape))
        self.activation_bytes.append((args[0].clone(), args[1].clone()))
        super().project(*args)


def distinct_expert_weights(ops=None, *, execution="batched"):
    from vllm_ascend.ops.fused_moe.vq2a8_gs32 import GS32ExpertWeights

    payloads = {}
    specs = {}
    layer = SimpleNamespace(vq2a8_num_experts=3)
    for kind, prefix in (("gate_up", "w13"), ("down", "w2")):
        packed_experts = []
        for slot, expert in enumerate((0, 2)):
            payload, spec, _ = fixture(kind)
            # Vary both stages independently so a slot/perm/scale mix-up
            # cannot pass by using the same fixture for every expert.
            payload["perm"] = torch.roll(payload["perm"], slot * (7 if kind == "gate_up" else -11))
            payload["weight_scale"] = payload["weight_scale"] * (1 + slot * 0.25)
            payload["row_scale"] = payload["row_scale"] * (1 - slot * 0.125)
            payload["codebooks"] = payload["codebooks"].roll(slot * 3, dims=2)
            payloads[expert, kind] = payload
            specs[kind] = spec
            packed_experts.append(repack_gs32_matrix(payload, spec, down_stage=True))
        for field in packed_experts[0]:
            values = [packed[field] for packed in packed_experts]
            if field in ("row_scale_bits", "perm"):
                values = [value.to(torch.int64) for value in values]
            setattr(layer, f"{prefix}_{field}", torch.stack(values))
    spec = SimpleNamespace(expert_ids=(0, 2), specs=specs)
    bank = GS32ExpertWeights(
        layer,
        spec,
        {"gs32_execution": execution},
        _test_ops=ops if ops is not None else RecordingOps(),
    )
    return bank, payloads, specs


def dispatched_reference(hidden, counts, payloads, specs, limit):
    expected = []
    first = 0
    for expert, count in enumerate(counts):
        # Native execution has a fixed <=128-row contract. Keep CPU GEMM
        # shapes identical while independently decoding source codebooks.
        for offset in range(0, count, 128):
            chunk = hidden[first + offset : first + min(count, offset + 128)]
            expected.append(
                staged_expert_reference(
                    chunk,
                    payloads[expert, "gate_up"],
                    specs["gate_up"],
                    payloads[expert, "down"],
                    specs["down"],
                    limit,
                )
            )
        first += count
    return torch.cat(expected)


@pytest.mark.parametrize("group_type", [0, 1])
@pytest.mark.parametrize("counts", [(2, 0, 3), (129, 0, 3)])
@pytest.mark.parametrize("limit", [None, 0.5])
def test_gs32_batch_matches_distinct_expert_reference(group_type, counts, limit):
    bank, payloads, specs = distinct_expert_weights()
    generator = torch.Generator().manual_seed(914)
    hidden = (torch.randn(sum(counts), 64, generator=generator) * 0.01).bfloat16()
    expected = dispatched_reference(hidden, counts, payloads, specs, 0.0 if limit is None else limit)
    groups = torch.tensor(counts)
    if group_type == 0:
        groups = groups.cumsum(0)
    actual, event = bank.apply(hidden, groups, group_type, limit)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert event is None
    assert all(0 < rows <= 128 for rows, _ in bank.ops.shapes)
    assert bank.ops.launches == 2 * sum((count + 127) // 128 for count in counts)


@pytest.mark.parametrize("group_type", [0, 1])
def test_gs32_batch_matches_serial_bank_for_distinct_experts_and_chunk_tail(group_type):
    serial, _, _ = distinct_expert_weights(execution="serial")
    batched, _, _ = distinct_expert_weights()
    generator = torch.Generator().manual_seed(982)
    hidden = (torch.randn(132, 64, generator=generator) * 0.01).bfloat16()
    groups = torch.tensor([129, 0, 3])
    if group_type == 0:
        groups = groups.cumsum(0)
    serial_output, serial_event = serial.apply(hidden, groups, group_type, 0.5)
    batched_output, batched_event = batched.apply(hidden, groups, group_type, 0.5)
    assert serial_event is batched_event is None
    torch.testing.assert_close(batched_output, serial_output, rtol=0, atol=0)


def test_gs32_batch_quantizes_twice_and_applies_swiglu_once(monkeypatch):
    from vllm_ascend.ops.fused_moe import vq2a8_gs32 as resident

    bank, _, _ = distinct_expert_weights()
    calls = {"quantize_mxfp8_k32": [], "staged_swiglu": []}
    for name in calls:
        function = getattr(resident, name)

        def record(*args, _name=name, _function=function, **kwargs):
            calls[_name].append(tuple(args[0].shape))
            return _function(*args, **kwargs)

        monkeypatch.setattr(resident, name, record)
    bank.apply(torch.zeros(132, 64, dtype=torch.bfloat16), torch.tensor([129, 0, 3]), 1, 10)
    assert calls["quantize_mxfp8_k32"] == [(132, 64), (132, 64)]
    assert calls["staged_swiglu"] == [(132, 128)]


def test_gs32_batch_preserves_each_native_activation_byte_and_scale():
    bank, payloads, specs = distinct_expert_weights()
    counts = (129, 0, 3)
    generator = torch.Generator().manual_seed(822)
    hidden = (torch.randn(sum(counts), 64, generator=generator) * 0.01).bfloat16()
    expected = {"gate_up": [], "down": []}
    first = 0
    for expert, count in enumerate(counts):
        for offset in range(0, count, 128):
            chunk = hidden[first + offset : first + min(count, offset + 128)]
            gate_payload = payloads[expert, "gate_up"]
            transformed = (chunk.float() * gate_payload["weight_scale"])[:, gate_payload["perm"].long()]
            expected["gate_up"].append(quantize_mxfp8_k32(transformed))
            gate = staged_projection_reference(chunk, gate_payload, specs["gate_up"])
            middle = staged_swiglu(gate, 0.5)
            down_payload = payloads[expert, "down"]
            transformed = (middle.float() * down_payload["weight_scale"])[:, down_payload["perm"].long()]
            expected["down"].append(quantize_mxfp8_k32(transformed))
        first += count
    bank.apply(hidden, torch.tensor(counts), 1, 0.5)
    actual = {"gate_up": [], "down": []}
    for (_, columns), values in zip(bank.ops.shapes, bank.ops.activation_bytes):
        actual["gate_up" if columns == specs["gate_up"].rows else "down"].append(values)
    for kind in expected:
        assert len(actual[kind]) == len(expected[kind])
        for (data, scale), (expected_data, expected_scale) in zip(actual[kind], expected[kind]):
            assert torch.equal(data, expected_data)
            assert torch.equal(scale, expected_scale)


def test_gs32_batch_completes_gate_before_swiglu_and_down(monkeypatch):
    from vllm_ascend.ops.fused_moe import vq2a8_gs32 as resident

    events = []

    class OrderedOps(RecordingOps):
        def project(self, *args):
            events.append("gate_up" if args[-1].shape[1] == 128 else "down")
            super().project(*args)

    bank, _, _ = distinct_expert_weights(OrderedOps())
    original = resident.staged_swiglu

    def record_swiglu(*args):
        events.append("swiglu")
        return original(*args)

    monkeypatch.setattr(resident, "staged_swiglu", record_swiglu)
    bank.apply(torch.zeros(132, 64, dtype=torch.bfloat16), torch.tensor([129, 0, 3]), 1, 10)
    assert events == ["gate_up"] * 3 + ["swiglu"] + ["down"] * 3


def test_gs32_batch_retains_eight_strict_finite_checks_per_layer(monkeypatch):
    from vllm_ascend.ops.fused_moe import vq2a8_gs32 as resident
    from vllm_ascend.quantization.methods.w2a8 import vq2a8_gs32_staged as stages

    names = []
    original = stages.require_finite

    def record(tensor, name):
        names.append(name)
        return original(tensor, name)

    monkeypatch.setattr(resident, "require_finite", record)
    monkeypatch.setattr(stages, "require_finite", record)
    bank, _, _ = distinct_expert_weights()
    bank.apply(torch.zeros(132, 64, dtype=torch.bfloat16), torch.tensor([129, 0, 3]), 1, 10)
    assert len(names) == 8
    assert names.count("MXFP8 input") == 2
    assert names.count("BF16-rounded MXFP8 input") == 2
    assert names.count("GateUp output") == names.count("SwiGLU output") == 1


@pytest.mark.parametrize("group_type", [0, 1])
def test_gs32_batch_empty_routes_skip_activation_and_native_work(monkeypatch, group_type):
    from vllm_ascend.ops.fused_moe import vq2a8_gs32 as resident

    bank, _, _ = distinct_expert_weights()

    def unexpected(*args, **kwargs):
        pytest.fail("Empty dispatch must not quantize or apply SwiGLU.")

    monkeypatch.setattr(resident, "quantize_mxfp8_k32", unexpected)
    monkeypatch.setattr(resident, "staged_swiglu", unexpected)
    output, event = bank.apply(
        torch.empty(0, 64, dtype=torch.bfloat16), torch.zeros(3, dtype=torch.int64), group_type, 10
    )
    assert output.shape == (0, 64)
    assert output.dtype == torch.bfloat16 and event is None
    assert bank.ops.launches == 0


@pytest.mark.parametrize("group_type,counts", [(1, [-1, 0, 3]), (1, [1, 1, 0]), (1, [1, 0, 0]), (0, [2, 1, 2])])
def test_gs32_batch_invalid_routes_fail_before_native_launch(group_type, counts):
    bank, _, _ = distinct_expert_weights()
    with pytest.raises(ValueError, match="group counts"):
        bank.apply(torch.zeros(2, 64, dtype=torch.bfloat16), torch.tensor(counts), group_type, 10)
    assert bank.ops.launches == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_gs32_batch_nonfinite_inputs_fail_closed(bad):
    bank, _, _ = distinct_expert_weights()
    hidden = torch.zeros(3, 64, dtype=torch.bfloat16)
    hidden[2, 17] = bad
    with pytest.raises(ValueError):
        bank.apply(hidden, torch.tensor([1, 0, 2]), 1, 10)


def test_gs32_batch_bf16_rounding_overflow_is_not_hidden():
    bank, _, _ = distinct_expert_weights()
    bank.fields["gate_up"]["weight_scale"].fill_(1.002)
    hidden = torch.zeros(3, 64, dtype=torch.bfloat16)
    hidden[2, 17] = torch.finfo(torch.bfloat16).max
    # Scaling is still finite FP32, but the required BF16 round overflows.
    assert torch.isfinite(hidden.float() * 1.002).all()
    assert not torch.isfinite((hidden.float() * 1.002).bfloat16()).all()
    with pytest.raises(ValueError):
        bank.apply(hidden, torch.tensor([1, 0, 2]), 1, 10)


@pytest.mark.parametrize("stage_columns", [128, 64])
def test_gs32_batch_detects_partial_writes_in_either_projection(stage_columns):
    class Partial(RecordingOps):
        def project(self, *args):
            output = args[-1]
            if output.shape[1] == stage_columns:
                complete = torch.empty_like(output)
                super().project(*args[:-1], complete)
                output.flatten()[:-1].copy_(complete.flatten()[:-1])
            else:
                super().project(*args)

    bank, _, _ = distinct_expert_weights(Partial())
    with pytest.raises(ValueError):
        bank.apply(torch.zeros(3, 64, dtype=torch.bfloat16), torch.tensor([1, 0, 2]), 1, 10)


def test_gs32_batch_bad_gate_stops_before_swiglu_and_down(monkeypatch):
    from vllm_ascend.ops.fused_moe import vq2a8_gs32 as resident

    class BadGate(RecordingOps):
        def project(self, *args):
            assert args[-1].shape[1] == 128, "Invalid gate output reached down projection."
            super().project(*args)
            args[-1][0, 0] = float("nan")

    bank, _, _ = distinct_expert_weights(BadGate())

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid gate output reached SwiGLU.")

    monkeypatch.setattr(resident, "staged_swiglu", unexpected)
    with pytest.raises(ValueError, match="output"):
        bank.apply(torch.zeros(3, 64, dtype=torch.bfloat16), torch.tensor([1, 0, 2]), 1, 10)


def test_gs32_batch_valid_call_after_failed_projection_uses_fresh_output():
    class FailOnce(RecordingOps):
        def __init__(self):
            super().__init__()
            self.corrupt_next = True

        def project(self, *args):
            super().project(*args)
            if self.corrupt_next:
                args[-1][-1, -1] = float("nan")
                self.corrupt_next = False

    bank, payloads, specs = distinct_expert_weights(FailOnce())
    generator = torch.Generator().manual_seed(127)
    hidden = (torch.randn(3, 64, generator=generator) * 0.01).bfloat16()
    groups = torch.tensor([1, 0, 2])
    with pytest.raises(ValueError, match="output"):
        bank.apply(hidden, groups, 1, 10)
    actual, _ = bank.apply(hidden, groups, 1, 10)
    expected = dispatched_reference(hidden, groups.tolist(), payloads, specs, 10)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
