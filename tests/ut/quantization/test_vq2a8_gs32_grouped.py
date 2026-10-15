# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped dispatch contracts with real CPU arithmetic, not NPU evidence."""

from types import SimpleNamespace

import pytest
import torch

from tests.ut.quantization.test_vq2a8_gs32_batch import (
    RecordingOps,
    dispatched_reference,
    distinct_expert_weights,
)
from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_staged import (
    quantize_mxfp8_k32,
    require_finite,
    staged_swiglu_unchecked,
)


class CpuActivation:
    """Only a flow-test substitute for the separately tested device kernel."""

    def __init__(self):
        self.phases = []

    def prepare(self, hidden, weight_scale, perm, row_slots, *, raw_gate=None):
        self.phases.append("gate" if raw_gate is None else "down")
        if raw_gate is not None:
            require_finite(raw_gate, "GateUp output")
        transformed = torch.stack(
            [(row.float() * weight_scale[slot])[perm[slot]] for row, slot in zip(hidden, row_slots.tolist())]
        )
        return quantize_mxfp8_k32(transformed)

    def swiglu_unchecked(self, gate, limit):
        return staged_swiglu_unchecked(gate, limit)

    def require_finite_fused(self, output, name):
        self.phases.append("final")
        require_finite(output, name)


class GroupedOps(RecordingOps):
    def __init__(self):
        super().__init__()
        self.routes = []
        self.grouped = []
        self.corrupt_columns = None

    def prepare_routes(self, hidden, counts):
        assert sum(counts.tolist()) == hidden.shape[0]
        route = SimpleNamespace(counts=counts.tolist())
        self.routes.append(route)
        return route

    def project_grouped(self, data, scales, weight, lut, row_scale, routes, output):
        self.grouped.append((routes, weight.shape[0], data.clone(), scales.clone()))
        first = 0
        for slot, count in enumerate(routes.counts):
            if count:
                super().project(
                    data[first : first + count],
                    scales[first : first + count],
                    weight[slot : slot + 1],
                    lut[slot : slot + 1],
                    row_scale[slot : slot + 1],
                    torch.tensor([count]),
                    torch.zeros(1, dtype=torch.int32),
                    self.make_tiling(data.shape[1], output.shape[1]),
                    output[first : first + count],
                )
            first += count
        if output.shape[1] == self.corrupt_columns:
            output[-1, -1] = float("nan")


def fused_bank(*, reversed_slots=False):
    bank, payloads, specs = distinct_expert_weights(GroupedOps())
    bank.execution = "fused"
    bank.activation = CpuActivation()
    if reversed_slots:
        bank.expert_slots = {2: 0, 0: 1}
        bank.fields = {
            kind: {field: value.flip(0) for field, value in fields.items()} for kind, fields in bank.fields.items()
        }
    return bank, payloads, specs


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("group_type", [0, 1])
@pytest.mark.parametrize("counts", [(1, 0, 2), (0, 0, 3), (32, 0, 6), (128, 0, 1)])
def test_fused_grouped_matches_reference_in_logical_dispatch_order(reverse, group_type, counts):
    bank, payloads, specs = fused_bank(reversed_slots=reverse)
    hidden = (torch.randn(sum(counts), 64, generator=torch.Generator().manual_seed(321)) * 0.01).bfloat16()
    groups = torch.tensor(counts)
    actual, event = bank.apply(hidden, groups.cumsum(0) if group_type == 0 else groups, group_type, 0.5)
    expected = dispatched_reference(hidden, counts, payloads, specs, 0.5)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert event is None
    assert len(bank.ops.grouped) == 2
    assert len(bank.ops.routes) == 1
    assert bank.ops.grouped[0][0] is bank.ops.grouped[1][0] is bank.ops.routes[0]
    assert bank.ops.routes[0].counts == ([counts[2], counts[0]] if reverse else [counts[0], counts[2]])
    assert bank.activation.phases == ["gate", "down", "final"]


def test_fused_large_expert_uses_existing_bounded_batched_path():
    bank, payloads, specs = fused_bank()
    hidden = torch.zeros(132, 64, dtype=torch.bfloat16)
    output, _ = bank.apply(hidden, torch.tensor([129, 0, 3]), 1, 10)
    torch.testing.assert_close(output, dispatched_reference(hidden, (129, 0, 3), payloads, specs, 10), rtol=0, atol=0)
    assert bank.ops.grouped == []
    assert bank.activation.phases == []
    assert bank.ops.launches == 6


def test_fused_empty_dispatch_does_not_prepare_routes_or_activation():
    bank, _, _ = fused_bank()
    output, _ = bank.apply(torch.empty(0, 64, dtype=torch.bfloat16), torch.zeros(3, dtype=torch.int64), 1, 10)
    assert output.shape == (0, 64)
    assert bank.ops.routes == bank.ops.grouped == bank.activation.phases == []


@pytest.mark.parametrize("columns,launches", [(128, 1), (64, 2)])
def test_fused_invalid_native_output_fails_and_next_call_recovers(columns, launches):
    bank, payloads, specs = fused_bank()
    bank.ops.corrupt_columns = columns
    hidden = torch.zeros(3, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        bank.apply(hidden, torch.tensor([1, 0, 2]), 1, 10)
    assert len(bank.ops.grouped) == launches
    bank.ops.corrupt_columns = None
    output, _ = bank.apply(hidden, torch.tensor([1, 0, 2]), 1, 10)
    torch.testing.assert_close(output, dispatched_reference(hidden, (1, 0, 2), payloads, specs, 10), rtol=0, atol=0)


def test_fused_invalid_activation_does_not_launch_projection():
    bank, _, _ = fused_bank()
    hidden = torch.full((1, 64), float("nan"), dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        bank.apply(hidden, torch.tensor([1, 0, 0]), 1, 10)
    assert bank.ops.grouped == []
