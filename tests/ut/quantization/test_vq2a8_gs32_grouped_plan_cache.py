# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU cache/capture regressions; injected ops do not execute NPU kernels."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tests.e2e.nightly.test_vq2a8_gs32_batched import ProjectionCapture
from tests.ut.quantization.test_vq2a8_gs32_batch import RecordingOps, distinct_expert_weights
from tests.ut.quantization.test_vq2a8_gs32_grouped import fused_bank
from vllm_ascend.ops.fused_moe.vq2a8_gs32 import _NativeStageOps


class RecordingGroupedBackend:
    """Observe dispatcher arguments and opaque-plan ownership only."""

    def __init__(self):
        self.prepared = []
        self.launched = []
        self.initialized = []
        self.routes = []
        self.fail_prepare = False

    def initialize(self, binary):
        self.initialized.append(binary)

    def prepare(self, *args):
        pytest.fail("Grouped projection unexpectedly used a single-expert plan.")

    def project_prepared(self, *args):
        pytest.fail("Grouped projection unexpectedly used the single-expert launch.")

    def prepare_routes(self, hidden, counts):
        route = object()
        self.routes.append((hidden, counts, route))
        return route

    def prepare_grouped(self, data, output_columns, groups):
        if self.fail_prepare:
            raise RuntimeError("prepare failed")
        plan = object()
        self.prepared.append((data, output_columns, groups, plan))
        return plan

    def project_grouped(self, *args):
        self.launched.append(args)


@pytest.fixture
def runtime_factory(monkeypatch, tmp_path):
    npu = SimpleNamespace(capturing=False, device=lambda device: nullcontext())
    npu.is_current_stream_capturing = lambda: npu.capturing
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    library, binary = tmp_path / "bridge.so", tmp_path / "kernel.o"
    library.touch()
    binary.touch()

    def create():
        backend = RecordingGroupedBackend()
        # Scope constructor-only doubles so subsequent CPU Tensor operations
        # retain torch.device. No real NPU registration or library load occurs.
        with monkeypatch.context() as patch:
            patch.setattr(torch, "device", lambda device: SimpleNamespace(type="npu", index=0))
            patch.setattr(torch.ops, "load_library", lambda path: None)
            patch.setattr(torch.ops, "vq2a8_gs32_stage", backend)
            runtime = _NativeStageOps(library, binary, "npu:0", execution="fused")
        assert backend.initialized == [str(binary)]
        return runtime, backend

    return create, npu


def grouped_arguments(rows=3, groups=2, columns=64, output_columns=128):
    return (
        torch.zeros(rows, columns, dtype=torch.uint8),
        torch.zeros(rows, columns // 32, dtype=torch.uint8),
        torch.zeros(groups, 1, dtype=torch.uint8),
        torch.zeros(groups, 1, dtype=torch.uint8),
        torch.zeros(groups, output_columns, dtype=torch.int64),
        object(),
        torch.empty(rows, output_columns, dtype=torch.bfloat16),
    )


def test_gs32_grouped_plan_cache_reuses_geometry_across_row_counts(runtime_factory):
    create, _ = runtime_factory
    runtime, backend = create()
    first, second = grouped_arguments(rows=3), grouped_arguments(rows=31)
    runtime.project_grouped(*first)
    runtime.project_grouped(*second)
    assert list(runtime.grouped_plans) == [(2, 64, 128)]
    assert len(backend.prepared) == 1 and len(backend.launched) == 2
    plan = runtime.grouped_plans[2, 64, 128]
    assert backend.prepared[0][3] is plan
    for actual, expected in zip(backend.launched, (first, second), strict=True):
        assert all(received is source for received, source in zip(actual[:6], expected[:6], strict=True))
        assert actual[6] is plan and actual[7] is expected[6]
    assert first[5] is not second[5], "Routes remain invocation-specific when the plan is reused."


@pytest.mark.parametrize("groups,columns,output_columns", [(3, 64, 128), (2, 128, 128), (2, 64, 64)])
def test_gs32_grouped_plan_cache_separates_each_geometry_dimension(runtime_factory, groups, columns, output_columns):
    create, _ = runtime_factory
    runtime, backend = create()
    runtime.project_grouped(*grouped_arguments())
    runtime.project_grouped(*grouped_arguments(groups=groups, columns=columns, output_columns=output_columns))
    assert set(runtime.grouped_plans) == {(2, 64, 128), (groups, columns, output_columns)}
    assert len(backend.prepared) == 2
    assert backend.launched[0][6] is not backend.launched[1][6]


def test_gs32_grouped_plan_cache_is_owned_by_each_runtime(runtime_factory):
    create, _ = runtime_factory
    left, left_backend = create()
    right, right_backend = create()
    arguments = grouped_arguments()
    left.project_grouped(*arguments)
    right.project_grouped(*arguments)
    assert left.grouped_plans is not right.grouped_plans
    assert left.grouped_plans[2, 64, 128] is not right.grouped_plans[2, 64, 128]
    assert len(left_backend.prepared) == len(right_backend.prepared) == 1
    assert left.ops is left_backend and right.ops is right_backend


@pytest.mark.parametrize("warm", [False, True])
def test_gs32_grouped_plan_cache_rejects_capture_before_prepare_or_launch(runtime_factory, warm):
    create, npu = runtime_factory
    runtime, backend = create()
    arguments = grouped_arguments()
    if warm:
        runtime.project_grouped(*arguments)
    npu.capturing = True
    with pytest.raises(RuntimeError, match="capture"):
        runtime.project_grouped(*arguments)
    with pytest.raises(RuntimeError, match="capture"):
        runtime.prepare_routes(arguments[0], torch.tensor([1, 2]))
    assert len(backend.prepared) == len(backend.launched) == int(warm)
    assert backend.routes == []


def test_gs32_grouped_plan_cache_failed_preparation_is_not_cached(runtime_factory):
    create, _ = runtime_factory
    runtime, backend = create()
    backend.fail_prepare = True
    with pytest.raises(RuntimeError, match="prepare failed"):
        runtime.project_grouped(*grouped_arguments())
    assert runtime.grouped_plans == {} and backend.launched == []
    backend.fail_prepare = False
    runtime.project_grouped(*grouped_arguments())
    assert len(runtime.grouped_plans) == len(backend.prepared) == len(backend.launched) == 1


@pytest.mark.parametrize("reversed_slots", [False, True])
@pytest.mark.parametrize("group_type", [0, 1])
@pytest.mark.parametrize("counts", [(1, 0, 2), (32, 0, 6), (0, 0, 3), (129, 0, 1)])
def test_gs32_grouped_plan_cache_capture_matches_serial_logical_experts(counts, group_type, reversed_slots):
    serial, _, _ = distinct_expert_weights(RecordingOps())
    fused, _, _ = fused_bank(reversed_slots=reversed_slots)
    baseline, candidate = ProjectionCapture(serial), ProjectionCapture(fused)
    serial.ops, fused.ops = baseline, candidate
    hidden = (torch.randn(sum(counts), 64, generator=torch.Generator().manual_seed(41)) * 0.1).bfloat16()
    groups = torch.tensor(counts, dtype=torch.int64)
    if group_type == 0:
        groups = groups.cumsum(0)
    expected, _ = serial.apply(hidden, groups, group_type, 0.5)
    actual, _ = fused.apply(hidden, groups, group_type, 0.5)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert baseline.records and baseline.records.keys() == candidate.records.keys()
    assert candidate.bank_experts == ([2, 0] if reversed_slots else [0, 2])
    for key, reference in baseline.records.items():
        captured = candidate.records[key]
        assert reference["input_shape"] == captured["input_shape"], key
        assert reference["output_shape"] == captured["output_shape"], key
        for field in ("data", "scales"):
            assert torch.equal(reference[field], captured[field]), (key, field)
    assert len(candidate.ops.grouped) == (0 if max(counts) > 128 else 2)
    assert {expert for _, expert, _ in candidate.records} == {expert for expert, count in enumerate(counts) if count}
