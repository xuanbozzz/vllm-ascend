# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in real three-expert GS32 device-route graph acceptance.

This captures one FFN bank, not a full model. Resident weights come from three
experts in unified layer 0. Every replay updates the same input/count addresses
and checks deferred validity outside capture. No CPU emulation is used.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.e2e.nightly.test_vq2a8_gs32_batched import load_selected_layer, native_paths, route_counts

ROUTED_ROWS = 5
WARMUP_REPEATS = 2
COUNT_CASES = ((1, 1, 3), (0, 5, 0), (3, 0, 2), (5, 0, 0), (0, 0, 5))


@pytest.fixture(scope="module")
def device_bank_runtime(request):
    options = {
        name: request.config.getoption(flag, default=None)
        for name, flag in (
            ("model", "--vq2a8-model"),
            ("gs32_library", "--vq2a8-gs32-library"),
            ("gs32_binary", "--vq2a8-gs32-binary"),
        )
    }
    if not options["model"]:
        pytest.skip("Pass a unified GS32 --vq2a8-model and both native artifacts on Ascend950.")
    paths = native_paths(options)
    layer, spec, config = load_selected_layer(paths.pop("model"), layer_index=0, expert_count=3)

    import torch
    import torch_npu  # noqa: F401

    from vllm_ascend.ops.fused_moe.vq2a8_gs32 import GS32ExpertWeights
    from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32 import AscendGS32FusedMoEScheme

    assert torch.npu.is_available(), "This opt-in test requires real Ascend hardware."
    device = torch.device("npu", 0)
    torch.npu.set_device(device)
    payload_bytes = 0
    for name, value in vars(layer).copy().items():
        if isinstance(value, torch.Tensor):
            AscendGS32FusedMoEScheme._validate_field(name, value)
            payload_bytes += value.numel() * value.element_size()
            setattr(layer, name, value.to(device))
    torch.npu.synchronize()
    baselines = {
        mode: GS32ExpertWeights(layer, spec, {**paths, "gs32_execution": mode}) for mode in ("serial", "fused")
    }
    missing = next((expert for expert in range(layer.vq2a8_num_experts) if expert not in spec.expert_ids), None)
    assert missing is not None, (
        "Missing-expert acceptance requires a logical expert outside the three resident experts."
    )
    print(
        "VQ2A8_GS32_DEVICE_BANK_ARTIFACTS="
        + json.dumps(
            {
                "layer": spec.layer_index,
                "experts": spec.expert_ids,
                "resident_weight_bytes": payload_bytes,
                "native": {
                    key: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                    for key, path in paths.items()
                },
                "scope": "three real experts; not full-model or serving validation",
            }
        ),
        flush=True,
    )
    yield SimpleNamespace(
        torch=torch,
        layer=layer,
        spec=spec,
        paths=paths,
        baselines=baselines,
        device=device,
        missing_expert=missing,
        limit=config.get("swiglu_limit"),
        bank_type=GS32ExpertWeights,
    )
    torch.npu.synchronize()


def reference_output(runtime, hidden, groups, group_type, baselines=None):
    outputs = {}
    for mode, bank in (runtime.baselines if baselines is None else baselines).items():
        output, event = bank.apply(hidden, groups, group_type, runtime.limit)
        assert event is None
        outputs[mode] = output.cpu().clone()
    expected = next(iter(outputs.values()))
    for output in outputs.values():
        runtime.torch.testing.assert_close(output, expected, rtol=0, atol=0)
    return expected


class CapturedBank:
    """Own all graph addresses and perform the required external validity check."""

    def __init__(self, runtime, bank, hidden, groups, group_type):
        self.runtime = runtime
        self.bank = bank
        self.hidden = hidden.clone()
        self.groups = groups.clone()
        self.group_type = group_type
        self.replays = 0
        torch = runtime.torch
        # This scalar is allocated outside the graph, as in the model runner's
        # shared owner vector. The bank must write it on every execution.
        self.validity = torch.zeros((), dtype=torch.bool, device=runtime.device)
        bank.validity_buffer = self.validity
        bank.defer_validity = True
        for _ in range(WARMUP_REPEATS):
            self.validity.zero_()
            bank.apply(self.hidden, self.groups, group_type, runtime.limit)
            assert bool(self.validity.cpu()), "The device bank must pass eager warmup before capture."
        torch.npu.synchronize()
        self.graph = torch.npu.NPUGraph()
        self.validity.zero_()
        # Keep the default torch.npu.graph side stream. Immutable native plans
        # were prepared on the original stream and must support this handoff.
        with torch.npu.graph(self.graph):
            self.output, event = bank.apply(self.hidden, self.groups, group_type, runtime.limit)
        assert event is None
        self.addresses = (
            self.hidden.data_ptr(),
            self.groups.data_ptr(),
            self.output.data_ptr(),
            self.validity.data_ptr(),
        )

    def replay_checked(self, hidden, groups):
        torch = self.runtime.torch
        self.hidden.copy_(hidden)
        self.groups.copy_(groups)
        self.validity.zero_()
        self.graph.replay()
        self.replays += 1
        torch.npu.synchronize()
        assert self.addresses == (
            self.hidden.data_ptr(),
            self.groups.data_ptr(),
            self.output.data_ptr(),
            self.validity.data_ptr(),
        )
        if not bool(self.validity.cpu()):
            raise ValueError("GS32 graph replay failed deferred validity checks.")
        return self.output.cpu().clone()


@pytest.mark.parametrize("rows", (192, 256), ids=("i32_topk6", "i32_topk8"))
@pytest.mark.parametrize("group_type", (0, 1), ids=("prefix", "counts"))
@pytest.mark.parametrize("reverse_bank", (False, True), ids=("physical_order", "reversed_physical_order"))
def test_gs32_device_bank_prefill_graph_dynamic_routes_and_recovery(
    device_bank_runtime, rows, group_type, reverse_bank
):
    runtime = device_bank_runtime
    torch, layer, spec = runtime.torch, runtime.layer, runtime.spec
    selected_layer, selected_spec = layer, spec
    if reverse_bank:
        selected_layer = SimpleNamespace(
            **{
                name: value.flip(0).contiguous() if isinstance(value, torch.Tensor) else value
                for name, value in vars(layer).items()
            }
        )
        selected_spec = SimpleNamespace(
            layer_index=spec.layer_index, expert_ids=tuple(reversed(spec.expert_ids)), specs=spec.specs
        )
    bank = runtime.bank_type(selected_layer, selected_spec, {**runtime.paths, "gs32_execution": "full_graph_prefill"})
    baselines = {
        "batched": runtime.bank_type(layer, spec, {**runtime.paths, "gs32_execution": "batched"}),
        "fused": runtime.baselines["fused"],
    }
    generator = torch.Generator(device="cpu").manual_seed(732)
    original = (torch.randn(rows, spec.specs["gate_up"].columns, generator=generator) * 0.1).bfloat16()
    hidden = original.to(runtime.device)
    count_cases = ((rows // 3, rows // 3, rows - 2 * (rows // 3)), (128, rows - 128, 0), (0, 128, rows - 128))
    groups = route_counts(torch, layer, spec, count_cases[0], runtime.device, group_type)
    captured = CapturedBank(runtime, bank, hidden, groups, group_type)
    outputs = []
    for index, counts in enumerate(count_cases):
        groups = route_counts(torch, layer, spec, counts, runtime.device, group_type)
        fresh = (original.float() + index * 0.03125).bfloat16().to(runtime.device)
        expected = reference_output(runtime, fresh, groups, group_type, baselines)
        actual = captured.replay_checked(fresh, groups)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        outputs.append(expected)
    assert any(not torch.equal(value, outputs[0]) for value in outputs[1:]), "Replays must exercise changed outputs."

    # Preserve total rows, but exceed the native per-expert M=128 bound.
    # The captured device validator must reject safely and then recover.
    invalid = route_counts(torch, layer, spec, (129, rows - 129, 0), runtime.device, group_type)
    with pytest.raises(ValueError, match="GS32 graph replay failed deferred validity checks"):
        captured.replay_checked(fresh, invalid)
    assert not bool(captured.validity.cpu())
    torch.testing.assert_close(captured.replay_checked(fresh, groups), outputs[-1], rtol=0, atol=0)
    # Replay the original input/routes to detect stale prefill metadata or FP8.
    groups = route_counts(torch, layer, spec, count_cases[0], runtime.device, group_type)
    torch.testing.assert_close(captured.replay_checked(hidden, groups), outputs[0], rtol=0, atol=0)
    print(
        "VQ2A8_GS32_DEVICE_BANK_PREFILL_GRAPH="
        + json.dumps(
            {
                "rows": rows,
                "group_type": group_type,
                "physical_experts": selected_spec.expert_ids,
                "captures": 1,
                "replays": captured.replays,
                "invalid_cases": ["per_expert_129"],
                "status": "PASS",
                "scope": "three-expert prefill graph correctness; not full-model or serving performance",
            }
        ),
        flush=True,
    )


@pytest.mark.parametrize("group_type", (0, 1), ids=("prefix", "counts"))
@pytest.mark.parametrize("reverse_bank", (False, True), ids=("physical_order", "reversed_physical_order"))
def test_gs32_device_bank_graph_dynamic_routes_and_recovery(device_bank_runtime, group_type, reverse_bank):
    runtime = device_bank_runtime
    torch, layer, spec = runtime.torch, runtime.layer, runtime.spec
    selected_layer, selected_spec = layer, spec
    if reverse_bank:
        selected_layer = SimpleNamespace(
            **{
                name: value.flip(0).contiguous() if isinstance(value, torch.Tensor) else value
                for name, value in vars(layer).items()
            }
        )
        selected_spec = SimpleNamespace(
            layer_index=spec.layer_index, expert_ids=tuple(reversed(spec.expert_ids)), specs=spec.specs
        )
    bank = runtime.bank_type(selected_layer, selected_spec, {**runtime.paths, "gs32_execution": "full_graph"})
    generator = torch.Generator(device="cpu").manual_seed(125)
    original = (torch.randn(ROUTED_ROWS, spec.specs["gate_up"].columns, generator=generator) * 0.1).bfloat16()
    hidden = original.to(runtime.device)
    groups = route_counts(torch, layer, spec, COUNT_CASES[0], runtime.device, group_type)
    captured = CapturedBank(runtime, bank, hidden, groups, group_type)

    reference_routes = []
    for counts in COUNT_CASES:
        groups = route_counts(torch, layer, spec, counts, runtime.device, group_type)
        expected = reference_output(runtime, hidden, groups, group_type)
        actual = captured.replay_checked(hidden, groups)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        reference_routes.append(expected)
    assert any(not torch.equal(value, reference_routes[0]) for value in reference_routes[1:]), (
        "Real route changes must affect the reference, otherwise this fixture cannot detect static captured routing."
    )

    # Same shape and routes, new hidden values: neither the transformed FP8
    # bytes nor captured native output may be reused from the previous replay.
    groups = route_counts(torch, layer, spec, (3, 0, 2), runtime.device, group_type)
    for delta in (0.03125, -0.0625):
        fresh = (original.float() + delta).bfloat16().to(runtime.device)
        expected = reference_output(runtime, fresh, groups, group_type)
        torch.testing.assert_close(captured.replay_checked(fresh, groups), expected, rtol=0, atol=0)

    # Invalid dynamic values use the already captured graph. A device fault or
    # wrong exception fails the test; only the graph-external status may reject.
    missing_counts = torch.zeros(layer.vq2a8_num_experts, dtype=torch.int64)
    missing_counts[runtime.missing_expert] = ROUTED_ROWS
    if group_type == 0:
        missing_counts = missing_counts.cumsum(0)
    bad_hidden = hidden.clone()
    bad_hidden[0, 0] = float("nan")
    bad_cases = {
        "nan_hidden": (bad_hidden, groups),
        "missing_expert": (hidden, missing_counts.to(runtime.device)),
        "negative_count_or_decreasing_prefix": (
            hidden,
            route_counts(torch, layer, spec, (-1, 6, 0), runtime.device, group_type),
        ),
        "zero_count_for_nonempty_rows": (
            hidden,
            route_counts(torch, layer, spec, (0, 0, 0), runtime.device, group_type),
        ),
    }
    recovered = reference_output(runtime, hidden, groups, group_type)
    for name, (invalid_hidden, invalid_groups) in bad_cases.items():
        with pytest.raises(ValueError, match="GS32 graph replay failed deferred validity checks"):
            captured.replay_checked(invalid_hidden, invalid_groups)
        assert not bool(captured.validity.cpu()), name
        torch.testing.assert_close(captured.replay_checked(hidden, groups), recovered, rtol=0, atol=0, msg=name)
    print(
        "VQ2A8_GS32_DEVICE_BANK_GRAPH="
        + json.dumps(
            {
                "group_type": group_type,
                "physical_experts": selected_spec.expert_ids,
                "logical_experts": spec.expert_ids,
                "captures": 1,
                "replays": captured.replays,
                "invalid_cases": list(bad_cases),
                "status": "PASS",
                "scope": "three-expert device-route graph correctness; not model or serving performance",
            }
        ),
        flush=True,
    )
