# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in real-NPU grouped launches, without loading model weights.

Exercise the first and last resident slots across route-mask word boundaries.
The fixture banks occupy at most a few MiB. CPU codebook reconstruction checks
the mathematical result; separate G=1 launches check grouped output bitwise.
Neither oracle uses the grouped implementation to produce expected results.
"""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def grouped_runtime(request):
    options = {
        name: request.config.getoption(flag, default=None)
        for name, flag in (
            ("model", "--vq2a8-model"),
            ("library", "--vq2a8-gs32-library"),
            ("binary", "--vq2a8-gs32-binary"),
        )
    }
    if not options["model"]:
        pytest.skip("Pass --vq2a8-model and both --vq2a8-gs32-* paths for real Ascend grouped execution.")
    assert all(options.values()), "Explicit grouped tests require both native artifact paths."
    paths = {name: Path(value).resolve(strict=True) for name, value in options.items()}
    config = json.loads((paths["model"] / "config.json").read_text())
    assert config["quantization_config"]["format"] == "ascend_vq2a8_gs32"

    # Imports follow the hardware opt-in; no CPU/fake native execution fallback.
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available(), "Explicit grouped tests require a real Ascend NPU."
    device = torch.device("npu", 0)
    torch.npu.set_device(device)
    torch.ops.load_library(str(paths["library"]))
    ops = torch.ops.vq2a8_gs32_stage
    for name in ("initialize", "prepare", "project_prepared", "prepare_routes", "prepare_grouped", "project_grouped"):
        assert hasattr(ops, name), f"Rebuild the GS32 bridge: missing {name}."
    ops.initialize(str(paths["binary"]))
    return SimpleNamespace(torch=torch, device=device, ops=ops)


def source_matrix(slot, k, n):
    """Small real producer payloads, with different values in each bank slot."""
    import torch

    from tests.ut.quantization.test_vq2a8_gs32_layout import fixture

    payload, spec, _ = fixture("gate_up" if n == 128 else "down")
    assert n == spec.rows and k in (64, 128)
    spec = replace(spec, columns=k)
    # packed_indices/codebooks both have K tiles as their outermost axis.
    payload["packed_indices"] = payload["packed_indices"].repeat(k // 64).roll(slot)
    books = payload["codebooks"].view(torch.uint8).repeat(k // 64, 1, 1, 1)
    # Avoid FP8 special encodings while distinguishing physical expert slots.
    payload["codebooks"] = ((books.to(torch.int32) + slot * 7) % 120).to(torch.uint8).view(torch.float8_e4m3fn)
    payload["perm"] = torch.roll(torch.arange(k, dtype=torch.int32), 23 + slot)
    payload["weight_scale"] = torch.linspace(0.5, 1.5, k) * (1 + slot / 512)
    payload["weight_bias"] = torch.zeros(k)
    payload["row_scale"] *= 1 + slot / 256
    return payload, spec


def build_case(groups, pattern, k, n):
    """Only active rows get CPU GEMM references; all resident slots are packed."""
    import torch

    from tools.vq2a8_gs32_layout import prepare_gs32_activation_reference, repack_gs32_matrix
    from tools.vq2a8_gs32_reference import quantize_mxfp8_k32, staged_projection_reference

    counts = torch.zeros(groups, dtype=torch.int64)
    counts[-1] = 1 if pattern == "last" else 2
    if pattern == "edges" and groups > 1:
        counts[0] = 1
    fields, inputs, scales, references, active = [], [], [], [], []
    generator = torch.Generator(device="cpu").manual_seed(7301)
    for slot, count in enumerate(counts.tolist()):
        payload, spec = source_matrix(slot, k, n)
        fields.append(repack_gs32_matrix(payload, spec, down_stage=True))
        if count:
            hidden = (torch.randn(count, k, generator=generator) / 16).bfloat16()
            transformed = prepare_gs32_activation_reference(hidden, payload, spec)
            data, scale = quantize_mxfp8_k32(transformed)
            inputs.append(data)
            scales.append(scale)
            references.append(staged_projection_reference(hidden, payload, spec))
            active.append((slot, count))
    banks = (
        torch.stack([entry["packed_zn"] for entry in fields]),
        torch.stack([entry["pair_lut"] for entry in fields]),
        torch.stack([entry["row_scale_bits"].view(torch.int64) for entry in fields]),
    )
    return SimpleNamespace(
        counts=counts,
        active=active,
        data=torch.cat(inputs),
        scales=torch.cat(scales),
        banks=banks,
        expected=torch.cat(references),
        groups=groups,
        k=k,
        n=n,
    )


def run_case(runtime, case, *, other_stream=False):
    torch, ops, device = runtime.torch, runtime.ops, runtime.device
    data, scales = case.data.to(device), case.scales.to(device)
    banks = tuple(field.to(device) for field in case.banks)
    rows = data.shape[0]
    plan = ops.prepare_grouped(data, case.n, case.groups)
    counts = case.counts.clone()
    routes = ops.prepare_routes(data, counts)
    # The handle must own a validated snapshot, not the caller's mutable tensor.
    counts.fill_(128)
    output = torch.full((rows, case.n), float("nan"), dtype=torch.bfloat16, device=device)
    repeated = torch.full_like(output, float("nan"))
    torch.npu.synchronize()
    stream = torch.npu.Stream(device=device) if other_stream else torch.npu.current_stream(device)
    with torch.npu.stream(stream):
        ops.project_grouped(data, scales, *banks, routes, plan, output)
        # Reusing route/plan owners must not mutate group offsets between calls.
        ops.project_grouped(data, scales, *banks, routes, plan, repeated)
    stream.synchronize()
    actual = output.cpu()
    assert torch.isfinite(actual.float()).all(), "A grouped row/output tile was not written."
    assert torch.equal(actual.view(torch.int16), repeated.cpu().view(torch.int16))

    serial, offset = [], 0
    for slot, count in case.active:
        # Fresh allocations keep every scale slice aligned even when K/32 <32.
        expert_data = case.data[offset : offset + count].clone().to(device)
        expert_scales = case.scales[offset : offset + count].clone().to(device)
        expert_banks = tuple(field[slot : slot + 1] for field in banks)
        expert_output = torch.full((count, case.n), float("nan"), dtype=torch.bfloat16, device=device)
        single_plan = ops.prepare(expert_data, case.n)
        ops.project_prepared(expert_data, expert_scales, *expert_banks, single_plan, expert_output)
        torch.npu.synchronize()
        serial.append(expert_output.cpu())
        offset += count
    assert offset == rows
    assert torch.equal(actual.view(torch.int16), torch.cat(serial).view(torch.int16)), "Grouped differs from G=1."
    # CPU FP32 accumulation order may differ from the native GEMM. Apply the
    # established stage-probe tolerance to every element, not just a mean.
    torch.testing.assert_close(actual.float(), case.expected.float(), rtol=0.01, atol=0.01)


@pytest.mark.parametrize("groups", [1, 3, 33, 256])
@pytest.mark.parametrize("pattern", ["last", "edges"])
@pytest.mark.parametrize("k,n", [(64, 64), (128, 128)])
def test_gs32_grouped_zero_slots_and_route_mask_boundary(grouped_runtime, groups, pattern, k, n):
    # G33 reads mask word1; G256 reads word7. Zero groups still advance bank
    # offsets and read route bits, so a single trailing live group is essential.
    run_case(grouped_runtime, build_case(groups, pattern, k, n))


def test_gs32_grouped_plan_and_owned_routes_on_nondefault_stream(grouped_runtime):
    run_case(grouped_runtime, build_case(33, "edges", 128, 128), other_stream=True)


@pytest.mark.parametrize("fault", ["float_counts", "device_counts", "negative", "wrong_sum", "per_group_129"])
def test_gs32_grouped_routes_reject_invalid_counts_before_launch(grouped_runtime, fault):
    torch, ops, device = grouped_runtime.torch, grouped_runtime.ops, grouped_runtime.device
    rows = 129 if fault == "per_group_129" else 3
    data = torch.empty((rows, 64), dtype=torch.uint8, device=device)
    counts = torch.tensor([1, 0, 2], dtype=torch.int64)
    if fault == "float_counts":
        counts = counts.float()
    elif fault == "device_counts":
        counts = counts.to(device)
    elif fault == "negative":
        counts = torch.tensor([1, -1, 3], dtype=torch.int64)
    elif fault == "wrong_sum":
        counts = torch.tensor([1, 0, 1], dtype=torch.int64)
    else:
        counts = torch.tensor([0, 0, 129], dtype=torch.int64)
    with pytest.raises(RuntimeError, match="dtype|CPU metadata|counts|count|128"):
        ops.prepare_routes(data, counts)
    torch.npu.synchronize()


def test_gs32_grouped_schema_rejects_raw_counts_in_place_of_owned_routes(grouped_runtime):
    torch, ops, device = grouped_runtime.torch, grouped_runtime.ops, grouped_runtime.device
    case = build_case(3, "edges", 64, 64)
    data, scales = case.data.to(device), case.scales.to(device)
    banks = tuple(field.to(device) for field in case.banks)
    output = torch.full((3, 64), float("nan"), dtype=torch.bfloat16, device=device)
    plan = ops.prepare_grouped(data, 64, 3)
    with pytest.raises((RuntimeError, TypeError), match="GroupedRoutes"):
        ops.project_grouped(data, scales, *banks, case.counts, plan, output)
    torch.npu.synchronize()
    assert torch.isnan(output.cpu().float()).all(), "Invalid routes must not submit a kernel."
