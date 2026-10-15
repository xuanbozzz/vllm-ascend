# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-NPU routing and replay, with independent Python-integer expected rows.

This suite never runs a native projection. It validates the device routing
boundary and its deferred flags; the native route sanitizer is tested separately.
"""

from types import SimpleNamespace

import pytest

from tests.e2e.nightly.test_vq2a8_gs32_activation import activation_runtime  # noqa: F401
from tests.ut.quantization.test_vq2a8_gs32_device_routes import route_reference

pytestmark = pytest.mark.filterwarnings(r"error:tl\.where with a non-boolean condition is deprecated.*:UserWarning")


@pytest.fixture(scope="module")
def route_runtime(request):
    runtime = request.getfixturevalue("activation_runtime")

    from vllm_ascend.ops.fused_moe.vq2a8_gs32_activation import finish_flags
    from vllm_ascend.ops.fused_moe.vq2a8_gs32_routes import prepare

    return SimpleNamespace(**vars(runtime), route=prepare, finish=finish_flags)


def host_fields(runtime, counts, experts, width=64, group_type=1, dtype=None):
    torch = runtime.torch
    rows, logical = sum(counts), len(counts)
    hidden = torch.arange(rows * width).reshape(rows, width).bfloat16()
    group = torch.tensor(counts, dtype=torch.int64 if dtype is None else dtype)
    if group_type == 0:
        group = group.cumsum(0).to(group.dtype)
    slots = [-1] * logical
    for slot, expert in enumerate(experts):
        slots[expert] = slot
    return hidden, group, torch.tensor(experts, dtype=torch.int64), torch.tensor(slots, dtype=torch.int64)


def enqueue(runtime, tensors, group_type):
    hidden, group, experts, slots = tensors
    return runtime.route(hidden, group, group_type, experts, slots)


def assert_route(runtime, host, group_type, results):
    torch = runtime.torch
    hidden, group, experts, slots = host
    expected = route_reference(group.tolist(), group_type, experts.tolist(), slots.tolist(), hidden.shape[0])
    physical, row_slots, source, inverse, valid = expected
    assert results[4].cpu().tolist() == [valid]
    assert results[1].cpu().tolist() == physical
    assert results[2].cpu().tolist() == row_slots
    assert results[3].cpu().tolist() == inverse
    actual = results[0].cpu()
    expected_hidden = hidden[source] if valid else torch.zeros_like(hidden)
    assert torch.equal(actual, expected_hidden)
    if valid:
        assert torch.equal(actual[inverse], hidden)


@pytest.mark.parametrize("group_type", [0, 1])
@pytest.mark.parametrize("dtype_name", ["int32", "int64"])
@pytest.mark.parametrize(
    "counts,experts",
    [
        ([1], [0]),
        ([1, 0, 2, 0], [2, 0, 3]),
        ([0, 0, 0, 8], [3, 2, 0]),
        # Exercise masked tails on both lookup tables, including unequal
        # power-of-two blocks (logical 5 -> 8, physical 3 -> 4).
        ([1, 0, 2, 0, 1], [4, 0, 2]),
        ([0, 1, 0, 2, 0, 1, 2], [6, 0, 3, 1, 5]),
        # Isolate row tails from expert tails: the original abort had both
        # ROWS=3 and EXPERTS=3, so either could have supplied its 3 -> 4 slice.
        ([1, 0, 2, 0], [2, 0, 3, 1]),  # ROWS=3, EXPERTS=4.
        ([1, 0, 3, 0], [2, 0, 3]),  # ROWS=4, EXPERTS=3.
        ([1, 0, 4, 0], [2, 0, 3, 1]),  # ROWS=5, EXPERTS=4.
        ([2, 0, 5, 0], [2, 0, 3, 1]),  # ROWS=7, EXPERTS=4.
    ],
)
def test_gs32_device_routes_match_cpu_order(route_runtime, group_type, dtype_name, counts, experts):
    runtime = route_runtime
    host = host_fields(runtime, counts, experts, group_type=group_type, dtype=getattr(runtime.torch, dtype_name))
    device = tuple(value.to(runtime.device) for value in host)
    assert_route(runtime, host, group_type, enqueue(runtime, device, group_type))


def test_gs32_device_routes_maximum_bank_and_width(route_runtime):
    runtime = route_runtime
    counts = [0] * 256
    for expert in (0, 3, 31, 95, 127, 200, 254, 255):
        counts[expert] = 1
    host = host_fields(runtime, counts, list(reversed(range(256))), width=4096)
    device = tuple(value.to(runtime.device) for value in host)
    assert_route(runtime, host, 1, enqueue(runtime, device, 1))


@pytest.mark.parametrize("group_type", [0, 1])
@pytest.mark.parametrize("dtype_name", ["int32", "int64"])
@pytest.mark.parametrize(
    "counts,experts",
    [
        ([32] * 6, [5, 3, 1, 4, 2, 0]),
        ([32] * 8, list(reversed(range(8)))),
        ([128, 0, 64], [2, 1, 0]),
        ([0, 7, 0, 31, 0, 25, 128], [6, 3, 1, 5, 0]),
        ([1] * 256, list(reversed(range(256)))),
    ],
)
def test_gs32_device_routes_prefill_matches_cpu_order(route_runtime, group_type, dtype_name, counts, experts):
    runtime = route_runtime
    host = host_fields(runtime, counts, experts, group_type=group_type, dtype=getattr(runtime.torch, dtype_name))
    device = tuple(value.to(runtime.device) for value in host)
    assert_route(runtime, host, group_type, enqueue(runtime, device, group_type))


@pytest.mark.parametrize("group_type", [0, 1])
def test_gs32_device_routes_prefill_graph_dynamic_routes_and_recovery(route_runtime, group_type):
    runtime, torch = route_runtime, route_runtime.torch
    experts = [7, 5, 3, 1, 6, 4, 2, 0]
    host = host_fields(runtime, [32] * 6 + [0, 0], experts, group_type=group_type)
    tensors = tuple(value.to(runtime.device) for value in host)
    stream = torch.npu.Stream(device=runtime.device)
    stream.wait_stream(torch.npu.current_stream(runtime.device))
    with torch.npu.stream(stream):
        for _ in range(2):
            enqueue(runtime, tensors, group_type)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            result = enqueue(runtime, tensors, group_type)
        for index, counts in enumerate(
            ([32] * 6 + [0, 0], [0, 0, 0, 0, 0, 0, 64, 128], [0, 0, 0, 0, 0, 0, 63, 129], [24] * 8)
        ):
            # The third replay is invalid although its total row count remains
            # correct. The fourth must recover with the same captured buffers.
            refreshed = list(host_fields(runtime, counts, experts, group_type=group_type))
            refreshed[0].add_(index)
            for target, source in zip(tensors, refreshed, strict=True):
                target.copy_(source)
            graph.replay()
            assert_route(runtime, refreshed, group_type, result)
        for corrupt in (True, False):
            refreshed = list(host_fields(runtime, [24] * 8, experts, group_type=group_type))
            if corrupt:
                refreshed[2][0] = 2**63 - 1
            for target, source in zip(tensors, refreshed, strict=True):
                target.copy_(source)
            graph.replay()
            assert_route(runtime, refreshed, group_type, result)
    torch.npu.current_stream(runtime.device).wait_stream(stream)


@pytest.mark.parametrize("group_type", [0, 1])
@pytest.mark.parametrize(
    "fault",
    [
        "max",
        "min",
        "negative",
        "too_many",
        "missing",
        "zero",
        "bad_id_min",
        "bad_id_max",
        "bad_id_negative",
        "bad_id_oob",
        "bad_slot_min",
        "bad_slot_max",
        "bad_slot_negative",
        "bad_slot_oob",
        "duplicate",
    ],
)
def test_gs32_device_routes_invalid_data_is_safe_and_recovers(route_runtime, group_type, fault):
    runtime = route_runtime
    host = list(host_fields(runtime, [1, 0, 2, 0], [2, 0, 3], group_type=group_type))
    if fault in ("max", "min", "negative", "too_many"):
        host[1][0] = {"max": 2**63 - 1, "min": -(2**63), "negative": -1, "too_many": 4}[fault]
    elif fault == "missing":
        host[1].copy_(runtime.torch.tensor([0, 1, 3, 3] if group_type == 0 else [0, 1, 2, 0]))
    elif fault == "zero":
        host[1].zero_()
    elif fault.startswith("bad_id_"):
        host[2][0] = {
            "bad_id_min": -(2**63),
            "bad_id_max": 2**63 - 1,
            "bad_id_negative": -1,
            "bad_id_oob": host[1].numel(),
        }[fault]
    elif fault.startswith("bad_slot_"):
        host[3][0] = {
            "bad_slot_min": -(2**63),
            "bad_slot_max": 2**63 - 1,
            "bad_slot_negative": -2,
            "bad_slot_oob": host[2].numel(),
        }[fault]
    else:
        host[2][1] = host[2][0]
    device = tuple(value.to(runtime.device) for value in host)
    results = enqueue(runtime, device, group_type)
    assert_route(runtime, host, group_type, results)
    assert results[4].cpu().tolist() == [0]
    fresh = host_fields(runtime, [1, 0, 2, 0], [2, 0, 3], group_type=group_type)
    assert_route(
        runtime, fresh, group_type, enqueue(runtime, tuple(value.to(runtime.device) for value in fresh), group_type)
    )


@pytest.mark.parametrize("group_type", [0, 1])
def test_gs32_device_routes_graph_dynamic_counts_and_status_recovery(route_runtime, group_type):
    runtime, torch = route_runtime, route_runtime.torch
    host = host_fields(runtime, [1, 0, 2, 0], [2, 0, 3], group_type=group_type)
    tensors = tuple(value.to(runtime.device) for value in host)
    stream = torch.npu.Stream(device=runtime.device)
    stream.wait_stream(torch.npu.current_stream(runtime.device))
    with torch.npu.stream(stream):
        flags = torch.ones(3, dtype=torch.int32, device=runtime.device)
        native = torch.ones(1, dtype=torch.int32, device=runtime.device)
        for _ in range(2):
            result = enqueue(runtime, tensors, group_type)
            runtime.finish(result[0], flags, flags, route_flags=result[4], native_flags=native).cpu()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            result = enqueue(runtime, tensors, group_type)
            combined = runtime.finish(result[0], flags, flags, route_flags=result[4], native_flags=native)
        for index, counts in enumerate(([1, 0, 2, 0], [0, 0, 0, 3], [0, 1, 2, 0], [3, 0, 0, 0])):
            refreshed = list(host_fields(runtime, counts, [2, 0, 3], group_type=group_type))
            refreshed[0].add_(index)
            for target, source in zip(tensors, refreshed, strict=True):
                target.copy_(source)
            native.fill_(1)
            graph.replay()
            expected_valid = int(counts[1] == 0)
            assert combined.cpu().tolist() == [expected_valid]
            assert_route(runtime, refreshed, group_type, result)
        # A native validator rejection must survive otherwise valid routes,
        # and the same captured status must recover on its next replay.
        for value in (0, 2, 1):
            native.fill_(value)
            graph.replay()
            assert combined.cpu().tolist() == [int(value == 1)]
    torch.npu.current_stream(runtime.device).wait_stream(stream)
