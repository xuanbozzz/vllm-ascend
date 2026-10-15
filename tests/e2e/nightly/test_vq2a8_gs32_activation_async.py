# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-NPU deferred validity and graph replay, without native projections.

GateUp and final output are supplied tensors so this isolates activation and
validity kernels. It does not claim native MoE or whole-model graph coverage.
Pass a GS32 --vq2a8-model to opt into hardware execution, as in the strict suite.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.e2e.nightly.test_vq2a8_gs32_activation import (
    activation_inputs,
    activation_runtime,  # noqa: F401
)


@pytest.fixture(scope="module")
def async_runtime(request):
    runtime = request.getfixturevalue("activation_runtime")

    from vllm_ascend.ops.fused_moe.vq2a8_gs32_activation import finish_flags, prepare_unchecked, swiglu_unchecked

    return SimpleNamespace(
        **vars(runtime), prepare_async=prepare_unchecked, finish=finish_flags, swiglu=swiglu_unchecked
    )


def host_case(runtime, rows, width, fault=None):
    torch = runtime.torch
    inputs = list(activation_inputs(runtime, rows, width))
    gate = (inputs[0].float() * 2).bfloat16()
    output = inputs[0].clone()
    if fault == "input_nan":
        inputs[0][0, 0] = float("nan")
    elif fault == "input_inf":
        inputs[0][0, 0] = float("inf")
    elif fault == "scale_overflow":
        inputs[0].fill_(torch.finfo(torch.bfloat16).max)
        inputs[1].fill_(2)
    elif fault == "slot_oob":
        inputs[3][0] = inputs[1].shape[0]
    elif fault == "perm_oob":
        inputs[2][inputs[3][0], 0] = width
    elif fault == "raw_gate_inf":
        gate[0, 0] = float("inf")
    elif fault == "output_hole":
        output[-1, -1] = float("nan")
    return (*inputs, gate, output)


def enqueue_case(runtime, tensors):
    hidden, scale, perm, slots, gate, output = tensors
    gate_data, gate_scale, gate_flags = runtime.prepare_async(hidden, scale, perm, slots)
    middle = runtime.swiglu(gate, 10.0)
    width = middle.shape[1]
    # Projection outputs are deliberate test inputs, not a simulated native
    # projection. Down preparation itself runs the production NPU kernels.
    down_scale = scale[:, :width].contiguous()
    down_perm = (perm[:, :width] % width).contiguous()
    down_data, down_scale, down_flags = runtime.prepare_async(middle, down_scale, down_perm, slots, raw_gate=gate)
    status = runtime.finish(output, gate_flags, down_flags)
    return status, gate_data, gate_scale, down_data, down_scale


def assert_finite_quantized(runtime, results):
    torch = runtime.torch
    for data in (results[1], results[3]):
        host = data.cpu().view(torch.float8_e4m3fn).float()
        assert torch.isfinite(host).all(), "Invalid input reached a non-finite native activation byte."
    for scales in (results[2], results[4]):
        host = scales.cpu()
        assert ((host >= 1) & (host <= 254)).all()


def assert_gate_bytes(runtime, host, results):
    hidden, scale, perm, slots = host[:4]
    transformed = np.stack(
        [hidden[row, perm[slot]].float().numpy() * scale[slot, perm[slot]].numpy() for row, slot in enumerate(slots)]
    )
    data, scales = runtime.reference.quantize_h(transformed)
    np.testing.assert_array_equal(results[1].cpu().numpy(), data.view(np.uint8))
    np.testing.assert_array_equal(results[2].cpu().numpy(), scales)


@pytest.mark.parametrize(
    "fault", ["input_nan", "input_inf", "scale_overflow", "slot_oob", "perm_oob", "raw_gate_inf", "output_hole"]
)
def test_gs32_activation_async_chain_rejects_and_recovers(async_runtime, fault):
    runtime, torch = async_runtime, async_runtime.torch
    stream = torch.npu.Stream(device=runtime.device)
    stream.wait_stream(torch.npu.current_stream(runtime.device))
    with torch.npu.stream(stream):
        for injected in (None, fault, None):
            host = host_case(runtime, 3, 128, injected)
            tensors = tuple(value.to(runtime.device) for value in host)
            results = enqueue_case(runtime, tensors)
            # There is no intermediate synchronization or CPU flag read.
            assert results[0].cpu().tolist() == [int(injected is None)]
            assert_finite_quantized(runtime, results)
            if injected is None:
                assert_gate_bytes(runtime, host, results)
    torch.npu.current_stream(runtime.device).wait_stream(stream)


@pytest.mark.parametrize("rows,width", [(1, 128), (8, 4096)])
def test_gs32_activation_async_graph_refreshes_flags_and_inputs(async_runtime, rows, width):
    runtime, torch = async_runtime, async_runtime.torch
    host = host_case(runtime, rows, width)
    tensors = tuple(value.to(runtime.device) for value in host)
    stream = torch.npu.Stream(device=runtime.device)
    stream.wait_stream(torch.npu.current_stream(runtime.device))
    with torch.npu.stream(stream):
        for _ in range(2):
            assert enqueue_case(runtime, tensors)[0].cpu().tolist() == [1]
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            results = enqueue_case(runtime, tensors)
        # Repeatedly refresh the SAME captured buffers and consume the SAME
        # status address, including invalid-to-valid transitions.
        for index, fault in enumerate((None, "raw_gate_inf", None, "input_nan", "output_hole", None)):
            refreshed = list(host_case(runtime, rows, width, fault))
            if fault is None:
                refreshed[0] = (refreshed[0].float() + index / 32).bfloat16()
            for target, source in zip(tensors, refreshed, strict=True):
                target.copy_(source)
            graph.replay()
            assert results[0].cpu().tolist() == [int(fault is None)]
            assert_finite_quantized(runtime, results)
            if fault is None:
                assert_gate_bytes(runtime, refreshed, results)
    torch.npu.current_stream(runtime.device).wait_stream(stream)
