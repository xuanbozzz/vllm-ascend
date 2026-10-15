# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole-forward device-route integration, using explicit CPU op doubles."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.ut.quantization.test_vq2a8_gs32_batch import dispatched_reference
from tests.ut.quantization.test_vq2a8_gs32_grouped import CpuActivation, GroupedOps, fused_bank
from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_staged import quantize_mxfp8_k32


class NoHostCounts(torch.Tensor):
    def cpu(self, *args, **kwargs):
        raise AssertionError("Device route counts must not be read on the host")


class DeviceOps(GroupedOps):
    def __init__(self):
        super().__init__()
        self.capturing = False

    def check_eager(self):
        if self.capturing:
            raise RuntimeError("Unexpected eager-only call inside capture")

    def prepare_device_routes(self, hidden, counts):
        return SimpleNamespace(counts=counts.tolist())

    def device_routes_status(self, routes):
        return torch.ones(1, dtype=torch.int32)

    def project_grouped_device(self, data, scales, weight, lut, row_scale, routes, plan, output):
        self.project_grouped(data, scales, weight, lut, row_scale, routes, output)


class DeviceActivation(CpuActivation):
    def prepare_unchecked(self, hidden, scale, perm, slots, *, raw_gate=None):
        transformed = torch.stack([(row.float() * scale[slot])[perm[slot]] for row, slot in zip(hidden, slots)])
        valid = torch.isfinite(transformed).all(1) & torch.isfinite(transformed.bfloat16()).all(1)
        if raw_gate is not None:
            valid &= torch.isfinite(raw_gate).all(1)
        safe = torch.where(valid[:, None], transformed, torch.zeros_like(transformed))
        data, scales = quantize_mxfp8_k32(safe)
        return data, scales, valid.int()

    def finish_flags(self, output, gate_flags, down_flags, *, route_flags, native_flags):
        return (
            (
                torch.isfinite(output).all()
                & gate_flags.bool().all()
                & down_flags.bool().all()
                & route_flags.bool().all()
                & native_flags.bool().all()
            )
            .int()
            .reshape(1)
        )


class DeviceRouter:
    def prepare(self, hidden, groups, kind, expert_ids, logical_slots):
        # This CPU oracle is deliberately outside the production router.
        counts = groups.as_subclass(torch.Tensor).clone()
        if kind == 0:
            counts = counts - torch.cat((counts.new_zeros(1), counts[:-1]))
        counts = counts.tolist()
        order, slots = [], []
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        for slot, expert in enumerate(expert_ids.tolist()):
            order.extend(range(offsets[expert], offsets[expert + 1]))
            slots.extend([slot] * counts[expert])
        inverse = torch.argsort(torch.tensor(order))
        return (
            hidden[order],
            torch.tensor([counts[e] for e in expert_ids]),
            torch.tensor(slots),
            inverse,
            torch.ones(1, dtype=torch.int32),
        )


def device_bank(monkeypatch, *, reverse=False):
    bank, payloads, specs = fused_bank(reversed_slots=reverse)
    bank.execution = "full_graph"
    bank.ops, bank.activation, bank.device_router = DeviceOps(), DeviceActivation(), DeviceRouter()
    bank.expert_ids = torch.tensor(sorted(bank.expert_slots, key=bank.expert_slots.get))
    bank.logical_slots = torch.full((bank.num_experts,), -1, dtype=torch.int64)
    bank.logical_slots[bank.expert_ids] = torch.arange(len(bank.expert_ids))
    bank.device_plans = {kind: object() for kind in bank.fields}
    bank.validity_buffer = torch.zeros((), dtype=torch.bool)
    bank.defer_validity, bank.capture_owners = False, []
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: bank.ops.capturing), raising=False
    )
    return bank, payloads, specs


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("group_type", [0, 1])
def test_full_graph_bank_changes_routes_without_reading_device_counts(monkeypatch, reverse, group_type):
    bank, payloads, specs = device_bank(monkeypatch, reverse=reverse)
    hidden = (torch.randn(6, 64, generator=torch.Generator().manual_seed(135)) / 32).bfloat16()
    for counts in ([6, 0, 0], [0, 0, 6], [2, 0, 4]):
        groups = torch.tensor(counts)
        if group_type == 0:
            groups = groups.cumsum(0)
        output, _ = bank.apply(hidden, groups.as_subclass(NoHostCounts), group_type, 0.5)
        expected = dispatched_reference(hidden, counts, payloads, specs, 0.5)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert bank.validity_buffer
    assert len(bank.ops.grouped) == 6


def test_capture_requires_owner_and_does_not_read_status_on_host(monkeypatch):
    bank, _, _ = device_bank(monkeypatch)
    bank.ops.capturing = True
    with pytest.raises(RuntimeError, match="owner|deferred"):
        bank.apply(torch.zeros(1, 64, dtype=torch.bfloat16), torch.tensor([1, 0, 0]), 1, 0.5)
    bank.defer_validity = True
    bank.validity_buffer = bank.validity_buffer.as_subclass(NoHostCounts)
    output, _ = bank.apply(torch.zeros(1, 64, dtype=torch.bfloat16), torch.tensor([1, 0, 0]), 1, 0.5)
    assert output.shape == (1, 64)
    assert bank.capture_owners


def test_device_bank_preserves_deferred_failure_and_recovery(monkeypatch):
    bank, _, _ = device_bank(monkeypatch)
    bank.defer_validity = True
    hidden = torch.zeros(1, 64, dtype=torch.bfloat16)
    groups = torch.tensor([1, 0, 0])
    for bad_columns in (128, 64, None):
        bank.ops.corrupt_columns = bad_columns
        bank.apply(hidden, groups, 1, 0.5)
        assert bool(bank.validity_buffer) == (bad_columns is None)
    bank.defer_validity = False
    bank.ops.corrupt_columns = 64
    with pytest.raises(ValueError, match="validity|finite"):
        bank.apply(hidden, groups, 1, 0.5)


def test_recapture_same_input_addresses_keeps_previous_native_routes(monkeypatch):
    bank, _, _ = device_bank(monkeypatch)
    bank.ops.capturing = bank.defer_validity = True
    hidden = torch.zeros(1, 64, dtype=torch.bfloat16)
    groups = torch.tensor([1, 0, 0])
    bank.apply(hidden, groups, 1, 0.5)
    bank.apply(hidden, groups, 1, 0.5)
    assert len(bank.capture_owners) == 2


def test_multi_token_graph_marks_fresh_validity_and_rejects_overflow(monkeypatch):
    bank, _, _ = device_bank(monkeypatch)
    bank.defer_validity = True
    bank.ops.capturing = True
    bank.apply(torch.zeros(36, 64, dtype=torch.bfloat16), torch.tensor([12, 0, 24]), 1, 0.5)
    assert bank.validity_buffer and len(bank.ops.grouped) == 2
    with pytest.raises(RuntimeError, match="256|shape"):
        bank.apply(torch.zeros(257, 64, dtype=torch.bfloat16), torch.tensor([128, 0, 129]), 1, 0.5)


@pytest.mark.parametrize("group_type", [0, 1])
def test_prefill_graph_bank_keeps_192_routes_and_status_on_device(monkeypatch, group_type):
    from vllm_ascend.ops.fused_moe.vq2a8_gs32 import GS32ExpertWeights

    # Exercise apply's actual dispatch independently of packed numeric kernels,
    # which are covered by the real-model and device-bank NPU tests.
    bank = GS32ExpertWeights.__new__(GS32ExpertWeights)
    bank.execution = "full_graph_prefill"
    bank.defer_validity = True
    bank.device = torch.device("cpu")
    bank.num_experts = 3
    bank.specs = {"gate_up": SimpleNamespace(columns=64)}
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: True), raising=False)
    hidden = (torch.randn(192, 64, generator=torch.Generator().manual_seed(32)) / 32).bfloat16()
    expected = torch.empty_like(hidden)
    bank._apply_device = Mock(return_value=expected)
    for counts in ([96, 0, 96], [64, 0, 128]):
        groups = torch.tensor(counts)
        if group_type == 0:
            groups = groups.cumsum(0)
        groups = groups.as_subclass(NoHostCounts)
        output, _ = bank.apply(hidden, groups, group_type, 0.5)
        assert output is expected
        args = bank._apply_device.call_args.args
        assert args[0] is hidden and args[1] is groups and args[2:] == (group_type, 0.5, True)
    assert bank._apply_device.call_count == 2
