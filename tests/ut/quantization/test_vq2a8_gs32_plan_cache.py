# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Python plan-cache dispatch only; no NPU/runtime execution is simulated."""

import pytest
import torch


class RecordingBridge:
    def __init__(self):
        self.prepares = []
        self.prepared_launches = []
        self.serial_launches = []
        self.prepare_error = None

    def prepare(self, data, columns):
        self.prepares.append((data, columns))
        if self.prepare_error is not None:
            raise self.prepare_error
        return object()

    def project_prepared(self, *args):
        self.prepared_launches.append(args)

    def project(self, *args):
        self.serial_launches.append(args)


def native_dispatch(execution):
    from vllm_ascend.ops.fused_moe.vq2a8_gs32 import _NativeStageOps

    # Deliberately skip the constructor's device loading/registration. These
    # tests exercise the actual Python cache method with CPU argument objects.
    dispatcher = object.__new__(_NativeStageOps)
    dispatcher.execution = execution
    dispatcher.plans = {}
    dispatcher.ops = RecordingBridge()
    dispatcher.check_eager = lambda: None
    return dispatcher


def projection_arguments(rows=1, width=64, columns=128):
    return (
        torch.zeros(rows, width, dtype=torch.uint8),
        torch.zeros(rows, width // 32, dtype=torch.uint8),
        torch.zeros(1, columns // 32, width // 16, 16, 8, dtype=torch.uint8),
        torch.zeros(1, columns // 32, width // 32, 32, dtype=torch.uint8),
        torch.zeros(1, columns, dtype=torch.int64),
        torch.tensor([rows], dtype=torch.int64),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1600, dtype=torch.uint8),
        torch.empty(rows, columns, dtype=torch.bfloat16),
    )


def assert_same_arguments(actual, expected):
    assert len(actual) == len(expected)
    assert all(got is wanted for got, wanted in zip(actual, expected))


def test_gs32_plan_cache_reuses_geometry_across_distinct_activation_and_weight_tensors():
    dispatcher = native_dispatch("batched")
    first = projection_arguments()
    second = projection_arguments()
    dispatcher.project(*first)
    dispatcher.project(*second)
    assert len(dispatcher.ops.prepares) == 1
    assert dispatcher.ops.prepares[0][0] is first[0]
    assert dispatcher.ops.prepares[0][1] == 128
    assert set(dispatcher.plans) == {(1, 64, 128)}
    plan = dispatcher.plans[1, 64, 128]
    assert_same_arguments(dispatcher.ops.prepared_launches[0], (*first[:5], plan, first[-1]))
    assert_same_arguments(dispatcher.ops.prepared_launches[1], (*second[:5], plan, second[-1]))
    assert dispatcher.ops.serial_launches == []


@pytest.mark.parametrize("other_shape", [(2, 64, 128), (1, 128, 128), (1, 64, 256)])
def test_gs32_plan_cache_separates_each_geometry_dimension(other_shape):
    dispatcher = native_dispatch("batched")
    first = projection_arguments()
    second = projection_arguments(*other_shape)
    dispatcher.project(*first)
    dispatcher.project(*second)
    dispatcher.project(*first)
    assert len(dispatcher.ops.prepares) == 2
    assert set(dispatcher.plans) == {(1, 64, 128), other_shape}
    first_plan = dispatcher.plans[1, 64, 128]
    second_plan = dispatcher.plans[other_shape]
    assert first_plan is not second_plan
    assert_same_arguments(dispatcher.ops.prepared_launches[0], (*first[:5], first_plan, first[-1]))
    assert_same_arguments(dispatcher.ops.prepared_launches[1], (*second[:5], second_plan, second[-1]))
    assert_same_arguments(dispatcher.ops.prepared_launches[2], (*first[:5], first_plan, first[-1]))


def test_gs32_plan_cache_serial_keeps_legacy_arguments_and_never_prepares():
    dispatcher = native_dispatch("serial")
    dispatcher.ops.prepare_error = AssertionError("Serial mode must not prepare native plans.")
    for shape in ((1, 64, 128), (1, 64, 128), (2, 64, 128)):
        args = projection_arguments(*shape)
        dispatcher.project(*args)
        assert_same_arguments(dispatcher.ops.serial_launches[-1], args)
    assert len(dispatcher.ops.serial_launches) == 3
    assert dispatcher.ops.prepares == []
    assert dispatcher.ops.prepared_launches == []
    assert dispatcher.plans == {}


def test_gs32_plan_cache_failed_prepare_does_not_cache_or_fall_back_and_can_retry():
    dispatcher = native_dispatch("batched")
    args = projection_arguments()
    failure = RuntimeError("native metadata preparation failed")
    dispatcher.ops.prepare_error = failure
    with pytest.raises(RuntimeError) as caught:
        dispatcher.project(*args)
    assert caught.value is failure
    assert dispatcher.plans == {}
    assert dispatcher.ops.serial_launches == dispatcher.ops.prepared_launches == []
    dispatcher.ops.prepare_error = None
    dispatcher.project(*args)
    assert len(dispatcher.ops.prepares) == 2
    assert set(dispatcher.plans) == {(1, 64, 128)}
    assert len(dispatcher.ops.prepared_launches) == 1
    assert dispatcher.ops.serial_launches == []


@pytest.mark.parametrize("execution", ["serial", "batched"])
def test_gs32_plan_cache_checks_eager_before_cache_lookup_or_launch(execution):
    dispatcher = native_dispatch(execution)
    args = projection_arguments()
    dispatcher.project(*args)
    counts = (len(dispatcher.ops.prepares), len(dispatcher.ops.prepared_launches), len(dispatcher.ops.serial_launches))

    def reject_capture():
        raise RuntimeError("capture unavailable")

    dispatcher.check_eager = reject_capture
    with pytest.raises(RuntimeError, match="capture unavailable"):
        dispatcher.project(*args)
    assert counts == (
        len(dispatcher.ops.prepares),
        len(dispatcher.ops.prepared_launches),
        len(dispatcher.ops.serial_launches),
    )
