# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in real Ascend950 device routing and raw-kernel ACLGraph replay.

No model load, fallback, mock NPU, or CPU acceptance of native execution.
The imported stage4 binary must safely skip every group for invalid routes.
"""

from types import SimpleNamespace

import pytest

from tests.e2e.nightly.test_vq2a8_gs32_grouped import build_case, grouped_runtime  # noqa: F401


@pytest.fixture(scope="module")
def device_runtime(grouped_runtime):  # noqa: F811 - imported pytest fixture
    for name in ("prepare_device_routes", "device_routes_status", "project_grouped_device"):
        assert hasattr(grouped_runtime.ops, name), f"Rebuild the GS32 bridge: missing {name}."
    return grouped_runtime


def upload_case(runtime, groups):
    torch, device, ops = runtime.torch, runtime.device, runtime.ops
    case = build_case(groups, "edges", 64, 64)
    data, scales = case.data.to(device), case.scales.to(device)
    banks = tuple(value.to(device) for value in case.banks)
    output = torch.full((3, 64), float("nan"), dtype=torch.bfloat16, device=device)
    plan = ops.prepare_grouped(data, case.n, groups)
    counts = case.counts.to(device)
    torch.npu.synchronize()
    return SimpleNamespace(case=case, data=data, scales=scales, banks=banks, output=output, plan=plan, counts=counts)


def serial_reference(runtime, uploaded, cpu_data, cpu_counts):
    """Independent real G=1 launches for whichever rows/expert slots are routed."""
    torch, device, ops = runtime.torch, runtime.device, runtime.ops
    results, offset = [], 0
    for slot, count in enumerate(cpu_counts.tolist()):
        if not count:
            continue
        data = cpu_data[offset : offset + count].clone().to(device)
        scales = uploaded.case.scales[offset : offset + count].clone().to(device)
        output = torch.full((count, 64), float("nan"), dtype=torch.bfloat16, device=device)
        plan = ops.prepare(data, 64)
        ops.project_prepared(data, scales, *(field[slot : slot + 1] for field in uploaded.banks), plan, output)
        torch.npu.synchronize()
        results.append(output.cpu())
        offset += count
    assert offset == 3
    return torch.cat(results)


def assert_result(runtime, output, status, expected):
    torch = runtime.torch
    assert status.dtype == torch.int32 and status.shape == (1,)
    assert status.cpu().tolist() == [int(expected is not None)]
    actual = output.cpu()
    if expected is None:
        assert torch.isnan(actual.float()).all(), "Invalid counts must zero all groups and leave output untouched."
    else:
        assert torch.isfinite(actual.float()).all(), "Native output contains unwritten or nonfinite values."
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("groups", [3, 33])
@pytest.mark.parametrize("fault", ["negative", "int64_overflow", "per_group_129", "zero", "wrong_sum"])
def test_gs32_device_routes_invalid_all_zero_native_and_recovery(device_runtime, groups, fault):
    runtime = device_runtime
    torch, ops = runtime.torch, runtime.ops
    uploaded = upload_case(runtime, groups)
    invalid = uploaded.case.counts.clone()
    if fault == "negative":
        invalid[0], invalid[-1] = -1, 4
    elif fault == "int64_overflow":
        invalid[0], invalid[1], invalid[-1] = 2**63 - 1, 2**63 - 1, 5
    elif fault == "per_group_129":
        invalid[0] = 129
    elif fault == "zero":
        invalid.zero_()
    else:
        invalid[-1] = 1
    expected = serial_reference(runtime, uploaded, uploaded.case.data, uploaded.case.counts)
    for counts, reference in ((invalid, None), (uploaded.case.counts, expected)):
        uploaded.counts.copy_(counts)
        uploaded.output.fill_(float("nan"))
        routes = ops.prepare_device_routes(uploaded.data, uploaded.counts)
        # Counts must be cloned before device validation and cannot be changed
        # by this later same-stream write to the caller's allocation.
        uploaded.counts.fill_(17)
        status = ops.device_routes_status(routes)
        ops.project_grouped_device(
            uploaded.data, uploaded.scales, *uploaded.banks, routes, uploaded.plan, uploaded.output
        )
        torch.npu.synchronize()
        assert_result(runtime, uploaded.output, status, reference)
        status.fill_(1 - int(reference is not None))
        assert ops.device_routes_status(routes).cpu().tolist() == [int(reference is not None)]


@pytest.mark.parametrize("groups", [3, 33])
def test_gs32_device_routes_full_capture_replays_changed_counts_and_recovers(device_runtime, groups):
    """Capture device validation + native call; replay reads new input/count bytes.

    This is the acceptance test for raw rtKernelLaunch capture in CANN; a
    successful eager launch does not satisfy it. Graph owners stay live here.
    """
    runtime = device_runtime
    torch, ops = runtime.torch, runtime.ops
    uploaded = upload_case(runtime, groups)
    first = uploaded.case.counts.clone()
    second = first.clone()
    second[0], second[-1] = 2, 1
    invalid = first.clone()
    invalid[0], invalid[1], invalid[-1] = 2**63 - 1, 2**63 - 1, 5
    zero = torch.zeros_like(first)
    cases = [(first, 0, True), (second, 1, True), (invalid, 0, False), (zero, 0, False), (first, 2, True)]
    references = [
        serial_reference(runtime, uploaded, uploaded.case.data.roll(shift, 0), counts) if valid else None
        for counts, shift, valid in cases
    ]
    stream = torch.npu.Stream(device=runtime.device)
    stream.wait_stream(torch.npu.current_stream(runtime.device))

    def native_forward():
        uploaded.output.fill_(float("nan"))
        routes = ops.prepare_device_routes(uploaded.data, uploaded.counts)
        status = ops.device_routes_status(routes)
        ops.project_grouped_device(
            uploaded.data, uploaded.scales, *uploaded.banks, routes, uploaded.plan, uploaded.output
        )
        return routes, status

    with torch.npu.stream(stream):
        owners = native_forward()
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        owners = native_forward()
    stream.synchronize()
    pointers = (uploaded.data.data_ptr(), uploaded.counts.data_ptr(), uploaded.output.data_ptr())
    for (counts, shift, _), expected in zip(cases, references):
        with torch.npu.stream(stream):
            uploaded.data.copy_(uploaded.case.data.roll(shift, 0))
            uploaded.counts.copy_(counts)
            graph.replay()
        stream.synchronize()
        assert (uploaded.data.data_ptr(), uploaded.counts.data_ptr(), uploaded.output.data_ptr()) == pointers
        assert_result(runtime, uploaded.output, owners[1], expected)


@pytest.mark.parametrize("fault", ["cpu", "dtype", "rank", "groups0", "groups257", "rows0", "rows32769"])
def test_gs32_device_routes_metadata_errors_reject_before_native(device_runtime, fault):
    torch, ops, device = device_runtime.torch, device_runtime.ops, device_runtime.device
    anchor = torch.empty((3, 64), dtype=torch.uint8, device=device)
    counts = torch.tensor([1, 0, 2], dtype=torch.int64, device=device)
    if fault == "cpu":
        counts = counts.cpu()
    elif fault == "dtype":
        counts = counts.int()
    elif fault == "rank":
        counts = counts.reshape(1, 3)
    elif fault.startswith("groups"):
        counts = torch.empty(int(fault[6:]), dtype=torch.int64, device=device)
    else:
        anchor = torch.empty((int(fault[4:]), 64), dtype=torch.uint8, device=device)
    with pytest.raises(RuntimeError, match="NPU|dtype|rank|experts|rows"):
        ops.prepare_device_routes(anchor, counts)
    torch.npu.synchronize()


def test_gs32_device_routes_reject_cross_stream_and_untyped_counts(device_runtime):
    runtime = device_runtime
    torch, ops = runtime.torch, runtime.ops
    uploaded = upload_case(runtime, 3)
    routes = ops.prepare_device_routes(uploaded.data, uploaded.counts)
    torch.npu.synchronize()
    other = torch.npu.Stream(device=runtime.device)
    with torch.npu.stream(other), pytest.raises(RuntimeError, match="stream"):
        ops.project_grouped_device(
            uploaded.data, uploaded.scales, *uploaded.banks, routes, uploaded.plan, uploaded.output
        )
    with pytest.raises((RuntimeError, TypeError), match="DeviceRoutes"):
        ops.project_grouped_device(
            uploaded.data, uploaded.scales, *uploaded.banks, uploaded.counts, uploaded.plan, uploaded.output
        )
    torch.npu.synchronize()
