# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in selected-expert GS32 equivalence and warm FFN timing on real NPUs.

Only three experts from unified layer 0 are materialized. No LLM, attention,
router, whole-model loading, graph execution or serving latency is measured.
Input-byte capture intentionally synchronizes; uncaptured comparisons and
timings separately exercise the production submission path.
"""

import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

SELECTED_EXPERTS = 3
WARMUP_REPEATS = 3
TIMING_REPEATS = 5
CASES = ((1, 1, 1), (6, 2, 24), (129, 2, 1), (0, 3, 0), (3, 0, 2), (0, 0, 0))


def native_paths(options):
    if not all(options.values()):
        raise ValueError("Model and both GS32 native artifact paths are required.")
    paths = {key: str(Path(value).resolve(strict=True)) for key, value in options.items()}
    if not all(Path(paths[key]).is_file() for key in ("gs32_library", "gs32_binary")):
        raise ValueError("Both GS32 native artifact paths must be files.")
    return paths


def load_selected_layer(model, *, layer_index=0, expert_count=SELECTED_EXPERTS, expert_positions=None):
    # Lazy imports keep test collection CPU-only until explicitly requested.
    import torch
    from safetensors import safe_open

    from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_format import (
        GS32_DTYPES,
        GS32_FIELDS,
        GS32ProjectionSpec,
        gs32_projection_shapes,
    )

    root = Path(model).resolve(strict=True)
    config = json.loads((root / "config.json").read_text())
    quant = config["quantization_config"]
    if quant["format"] != "ascend_vq2a8_gs32":
        raise ValueError("This probe requires a unified GS32 checkpoint.")
    metadata = quant["layers"][str(layer_index)]
    inventory = tuple(metadata["expert_ids"])
    positions = tuple(range(expert_count)) if expert_positions is None else tuple(expert_positions)
    if (
        not positions
        or len(set(inventory)) != len(inventory)
        or any(type(position) is not int or not 0 <= position < len(inventory) for position in positions)
        or len(set(positions)) != len(positions)
    ):
        raise ValueError("The selected layer needs enough distinct experts.")
    slices = []
    for position in positions:
        if slices and slices[-1].stop == position:
            slices[-1] = slice(slices[-1].start, position + 1)
        else:
            slices.append(slice(position, position + 1))
    specs = {kind: GS32ProjectionSpec(**metadata[kind]) for kind in ("gate_up", "down")}
    spec = SimpleNamespace(layer_index=layer_index, expert_ids=tuple(inventory[p] for p in positions), specs=specs)
    layer = SimpleNamespace(vq2a8_num_experts=config.get("n_routed_experts", max(inventory) + 1))
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    for kind, prefix in (("gate_up", "w13"), ("down", "w2")):
        for field, dtype, shape in zip(GS32_FIELDS, GS32_DTYPES, gs32_projection_shapes(specs[kind]), strict=True):
            name = f"model.layers.{layer_index}.mlp.experts.{prefix}_{field}"
            shard = (root / index[name]).resolve(strict=True)
            if not shard.is_relative_to(root):
                raise ValueError(f"Checkpoint shard escapes model root: {shard}")
            with safe_open(shard, framework="pt", device="cpu") as handle:
                sliced = handle.get_slice(name)
                if tuple(sliced.get_shape()) != (len(inventory), *shape):
                    raise ValueError(f"Unexpected complete expert geometry: {name}")
                parts = [sliced[selection] for selection in slices]
                value = (parts[0] if len(parts) == 1 else torch.cat(parts)).contiguous()
            if value.dtype != dtype:
                raise ValueError(f"Unexpected expert dtype: {name}: {value.dtype}")
            setattr(layer, f"{prefix}_{field}", value)
    return layer, spec, config


class ProjectionCapture:
    """Record actual input bytes around real native launches; never fake an op."""

    def __init__(self, bank):
        self.ops = bank.ops
        self.owners = {
            fields["packed_zn"][slot : slot + 1].data_ptr(): (kind, expert)
            for kind, fields in bank.fields.items()
            for expert, slot in bank.expert_slots.items()
        }
        self.chunks = defaultdict(int)
        self.records = {}
        self.group_routes = {}
        self.bank_experts = sorted(bank.expert_slots, key=bank.expert_slots.get)
        self.bank_kinds = {fields["packed_zn"].data_ptr(): kind for kind, fields in bank.fields.items()}

    def __getattr__(self, name):
        return getattr(self.ops, name)

    def project(self, *args):
        data, scales, weight, _, _, _, _, _, output = args
        identity = self.owners[weight.data_ptr()]
        key = (*identity, self.chunks[identity])
        self.chunks[identity] += 1
        self.records[key] = {
            "data": data.cpu().clone(),
            "scales": scales.cpu().clone(),
            "input_shape": tuple(data.shape),
            "output_shape": tuple(output.shape),
        }
        return self.ops.project(*args)

    def prepare_routes(self, hidden, counts):
        routes = self.ops.prepare_routes(hidden, counts)
        self.group_routes[id(routes)] = counts.tolist()
        return routes

    def project_grouped(self, data, scales, weight, lut, row_scale, routes, output):
        kind = self.bank_kinds[weight.data_ptr()]
        first = 0
        for expert, count in zip(self.bank_experts, self.group_routes[id(routes)], strict=True):
            if count:
                key = (kind, expert, self.chunks[kind, expert])
                self.chunks[kind, expert] += 1
                self.records[key] = {
                    "data": data[first : first + count].cpu().clone(),
                    "scales": scales[first : first + count].cpu().clone(),
                    "input_shape": (count, data.shape[1]),
                    "output_shape": (count, output.shape[1]),
                }
            first += count
        return self.ops.project_grouped(data, scales, weight, lut, row_scale, routes, output)


def compare_case(banks, hidden, group_list, group_type, limit, *, capture):
    import torch

    outputs, snapshots = {}, {}
    for mode, bank in banks.items():
        original = bank.ops
        recorder = ProjectionCapture(bank) if capture else None
        if recorder is not None:
            bank.ops = recorder
        try:
            output, _ = bank.apply(hidden, group_list, group_type, limit)
            torch.npu.synchronize()
            outputs[mode] = output.cpu().clone()
        finally:
            bank.ops = original
        if recorder is not None:
            snapshots[mode] = recorder.records
    for mode, output in outputs.items():
        torch.testing.assert_close(
            output, outputs["serial"], rtol=0, atol=0, msg=lambda message, mode=mode: f"{mode}: {message}"
        )
    if not capture:
        return
    for mode in snapshots:
        assert snapshots["serial"].keys() == snapshots[mode].keys()
        for key, baseline in snapshots["serial"].items():
            actual = snapshots[mode][key]
            assert baseline["input_shape"] == actual["input_shape"], (mode, key)
            assert baseline["output_shape"] == actual["output_shape"], (mode, key)
            for field in ("data", "scales"):
                assert torch.equal(baseline[field], actual[field]), (mode, key, field)


def route_counts(torch, layer, spec, counts, device, group_type):
    logical = torch.zeros(layer.vq2a8_num_experts, dtype=torch.int64)
    for expert, count in zip(spec.expert_ids, counts, strict=True):
        logical[expert] = count
    if group_type == 0:
        logical = logical.cumsum(0)
    return logical.to(device)


def warm_timing(torch, banks, hidden, counts, limit):
    for _ in range(WARMUP_REPEATS):
        for bank in banks.values():
            bank.apply(hidden, counts, 1, limit)
    torch.npu.synchronize()
    samples = {mode: [] for mode in banks}
    # Alternate order to reduce systematic temperature/clock drift bias.
    for repeat in range(TIMING_REPEATS):
        order = tuple(banks) if repeat % 2 == 0 else tuple(reversed(banks))
        for mode in order:
            torch.npu.synchronize()
            begin = time.perf_counter()
            banks[mode].apply(hidden, counts, 1, limit)
            torch.npu.synchronize()
            samples[mode].append((time.perf_counter() - begin) * 1000)
    medians = {mode: statistics.median(values) for mode, values in samples.items()}
    return {
        "median_ms": medians,
        "samples_ms": samples,
        "speedup_vs_batched": {mode: medians["batched"] / elapsed for mode, elapsed in medians.items()},
    }


def check_stream_cases(torch, banks, layer, spec, device, limit, *, nondefault, cases=CASES):
    generator = torch.Generator(device="cpu").manual_seed(1234)
    for counts in cases:
        host = (torch.randn(sum(counts), spec.specs["gate_up"].columns, generator=generator) * 0.1).bfloat16()
        for group_type in (0, 1):
            groups = route_counts(torch, layer, spec, counts, device, group_type)
            hidden = host.to(device)
            compare_case(banks, hidden, groups, group_type, limit, capture=True)
            # No byte capture or intermediate CPU copies on these runs.
            # Fresh values expose stale prepared-plan or activation reuse.
            for delta in (0.03125, -0.0625):
                hidden = (host.float() + delta).bfloat16().to(device)
                compare_case(banks, hidden, groups, group_type, limit, capture=False)
        print(
            "VQ2A8_GS32_BATCHED_EQUIVALENCE="
            + json.dumps({"counts": counts, "modes": list(banks), "nondefault_stream": nondefault, "status": "PASS"}),
            flush=True,
        )
        if counts in CASES[:2]:
            groups = route_counts(torch, layer, spec, counts, device, 1)
            report = warm_timing(torch, banks, hidden, groups, limit)
            print(
                "VQ2A8_GS32_BATCHED_TIMING="
                + json.dumps(
                    {**report, "counts": counts, "nondefault_stream": nondefault, "scope": "selected FFN only"}
                ),
                flush=True,
            )


def check_shared_plans_across_streams(torch, banks, layer, spec, device, limit):
    check_stream_cases(torch, banks, layer, spec, device, limit, nondefault=False)
    plans = banks["batched"].ops.plans
    default_plans = dict(plans)
    grouped_plans = dict(banks["fused"].ops.grouped_plans) if "fused" in banks else {}
    assert default_plans, "Default-stream cases must prepare real native plans."
    default_stream = torch.npu.current_stream(device)
    other_stream = torch.npu.Stream(device=device)
    other_stream.wait_stream(default_stream)
    # M=5,7,9 have not appeared on the default stream. Prepare these while
    # reusing the same bank and all previously initialized plans.
    new_counts = (5, 7, 9)
    assert not any(shape[0] in new_counts for shape in default_plans)
    with torch.npu.stream(other_stream):
        check_stream_cases(torch, banks, layer, spec, device, limit, nondefault=True, cases=(*CASES, new_counts))
    assert all(plans[key] is plan for key, plan in default_plans.items())
    assert all(any(shape[0] == rows for shape in plans) for rows in new_counts)
    nondefault_plans = {key: plan for key, plan in plans.items() if key not in default_plans}
    assert nondefault_plans
    default_stream.wait_stream(other_stream)
    # Return to the original stream and reuse the newly created handles too.
    check_stream_cases(torch, banks, layer, spec, device, limit, nondefault=False, cases=(new_counts,))
    assert all(plans[key] is plan for key, plan in nondefault_plans.items())
    if "fused" in banks:
        assert grouped_plans
        assert all(banks["fused"].ops.grouped_plans[key] is plan for key, plan in grouped_plans.items())


def check_reordered_bank(torch, banks, layer, spec, paths, device, limit):
    """Compare a physically reversed bank against the original serial bank.

    Keep logical route counts and input rows unchanged. Reversing every field
    and expert_id together forces the real grouped path to reorder input rows
    and restore output order, including leading/interior/trailing zero groups.
    """
    from vllm_ascend.ops.fused_moe.vq2a8_gs32 import GS32ExpertWeights

    reversed_layer = SimpleNamespace(
        **{
            name: value.flip(0).contiguous() if isinstance(value, torch.Tensor) else value
            for name, value in vars(layer).items()
        }
    )
    reversed_spec = SimpleNamespace(
        layer_index=spec.layer_index, expert_ids=tuple(reversed(spec.expert_ids)), specs=spec.specs
    )
    reordered = GS32ExpertWeights(reversed_layer, reversed_spec, {**paths, "gs32_execution": "fused"})
    compared = {"serial": banks["serial"], "fused_reordered": reordered}
    generator = torch.Generator(device="cpu").manual_seed(597)
    for counts in ((0, 3, 0), (3, 0, 2), (1, 2, 4)):
        host = (torch.randn(sum(counts), spec.specs["gate_up"].columns, generator=generator) * 0.1).bfloat16()
        for group_type in (0, 1):
            groups = route_counts(torch, layer, spec, counts, device, group_type)
            compare_case(compared, host.to(device), groups, group_type, limit, capture=True)
            fresh = (host.float() + 0.03125).bfloat16().to(device)
            compare_case(compared, fresh, groups, group_type, limit, capture=False)
        print(
            "VQ2A8_GS32_REORDERED_EQUIVALENCE="
            + json.dumps(
                {
                    "counts": counts,
                    "original_experts": spec.expert_ids,
                    "reordered_experts": reversed_spec.expert_ids,
                    "status": "PASS",
                }
            ),
            flush=True,
        )


def run_real_experts(request, modes):
    options = {
        name: request.config.getoption(flag)
        for name, flag in (
            ("model", "--vq2a8-model"),
            ("gs32_library", "--vq2a8-gs32-library"),
            ("gs32_binary", "--vq2a8-gs32-binary"),
        )
    }
    if not options["model"]:
        pytest.skip("Pass unified --vq2a8-model and both --vq2a8-gs32-* paths on Ascend950.")
    paths = native_paths(options)
    layer, spec, config = load_selected_layer(paths.pop("model"))

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
    print(
        "VQ2A8_GS32_BATCHED_ARTIFACTS="
        + json.dumps(
            {
                "layer": spec.layer_index,
                "experts": spec.expert_ids,
                "resident_weight_bytes": payload_bytes,
                "native": {
                    key: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                    for key, path in paths.items()
                },
                "model_integration_verified": False,
                "serving_performance_verified": False,
            }
        ),
        flush=True,
    )
    banks = {mode: GS32ExpertWeights(layer, spec, {**paths, "gs32_execution": mode}) for mode in modes}
    check_shared_plans_across_streams(torch, banks, layer, spec, device, config.get("swiglu_limit"))
    if "fused" in banks:
        check_reordered_bank(torch, banks, layer, spec, paths, device, config.get("swiglu_limit"))


def test_gs32_real_experts_batched(request):
    run_real_experts(request, ("serial", "batched"))


def test_gs32_real_experts_fused(request):
    run_real_experts(request, ("serial", "batched", "fused"))
