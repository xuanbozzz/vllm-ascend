# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-route wrapper contracts, with no emulated NPU execution."""

import ast
import importlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.ut.quantization.test_vq2a8_gs32_fused_activation import tagged


def route_kernel_source():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/activation/vq2a8_gs32_routes.py"
    module = ast.parse(source.read_text())
    kernel = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_route_metadata_kernel"
    )
    return source, kernel


def test_route_metadata_loop_only_updates_registers():
    """A compiler-workaround source contract; Ascend compilation is separate.

    Both ROWS and EXPERTS were three in the reported abort. A masked load
    audit alone misses the loop's inverse scatter and its derived address.
    """
    _, kernel = route_kernel_source()
    loops = [node for node in ast.walk(kernel) if isinstance(node, (ast.For, ast.While))]
    for loop in loops:
        accesses = [
            node.func.attr
            for node in ast.walk(loop)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tl"
            and node.func.attr in {"load", "store"}
        ]
        assert not accesses, "Row selection must not issue GM loads or stores inside the loop"


def test_route_metadata_all_addresses_are_affine():
    """Reject data-dependent metadata addresses, including inverse stores."""
    _, kernel = route_kernel_source()
    expected_offsets = {
        "group_ptr": "logical",
        "expert_ptr": "physical",
        "logical_slot_ptr": "logical",
        "counts_ptr": "physical",
        "row_slots_ptr": "row_offsets",
        "gather_ptr": "row_offsets",
        "inverse_ptr": "row_offsets",
        "flags_ptr": None,
    }
    for offset in set(expected_offsets.values()) - {None}:
        definitions = [
            node
            for node in ast.walk(kernel)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == offset for target in node.targets)
        ]
        assert len(definitions) == 1
        value = definitions[0].value
        assert isinstance(value, ast.Call) and ast.unparse(value.func) == "tl.arange"
        assert isinstance(value.args[0], ast.Constant) and value.args[0].value == 0
        assert isinstance(value.args[1], ast.Name) and value.args[1].id.endswith("_BLOCK")
    seen = set()
    for node in ast.walk(kernel):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tl"
            and node.func.attr in {"load", "store"}
        ):
            continue
        address = node.args[0]
        dependencies = {part.id for part in ast.walk(address) if isinstance(part, ast.Name)}
        pointers = dependencies & expected_offsets.keys()
        assert len(pointers) == 1
        pointer = next(iter(pointers))
        offset = expected_offsets[pointer]
        assert dependencies == ({pointer, offset} if offset else {pointer}), (
            f"{node.func.attr} of {pointer} uses a data-dependent address: {ast.unparse(address)}"
        )
        assert not any(isinstance(part, ast.Call) for part in ast.walk(address))
        assert all(isinstance(part.op, (ast.Add, ast.Sub)) for part in ast.walk(address) if isinstance(part, ast.BinOp))
        seen.add(pointer)
    assert seen == expected_offsets.keys()


@pytest.mark.parametrize("fault", ["none", "value", "mapping", "expert", "back_slot", "sum", "physical_sum"])
def test_route_validity_reductions_keep_boolean_where_condition(fault):
    """Evaluate actual validity expressions; int32 predicates must stay out of where."""
    source, kernel = route_kernel_source()
    namespace = {
        "tl": SimpleNamespace(
            where=torch.where,
            min=lambda value, axis: value.amin(dim=axis),
            sum=lambda value, axis: value.sum(dim=axis),
            int32=torch.int32,
        ),
        "logical_active": torch.tensor([True, True, False]),
        "physical_active": torch.tensor([True, True, False]),
        "value_valid": torch.tensor([True, fault != "value", False]),
        "mapping_valid": torch.tensor([True, fault != "mapping", False]),
        "expert_valid": torch.tensor([True, fault != "expert", False]),
        "physical": torch.tensor([0, 1, 2], dtype=torch.int32),
        "back_slot": torch.tensor([0, -1 if fault == "back_slot" else 1, -1], dtype=torch.int32),
        "counts": torch.tensor([1, 1 if fault == "sum" else 2, 0], dtype=torch.int32),
        "physical_counts": torch.tensor([2, 0 if fault == "physical_sum" else 1, 0], dtype=torch.int32),
        "ROWS": 3,
        "ALLOW_PADDING": False,
    }
    statements = [
        node
        for node in kernel.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "valid" for target in node.targets)
    ]
    assert statements
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(source), "exec"), namespace)
    valid = namespace["valid"]
    assert valid.dtype == torch.bool, "tl.where predicates must not inherit the int32 reduction dtype"
    assert bool(valid) == (fault == "none")
    assert torch.where(valid, 7, 0).item() == (7 if fault == "none" else 0)


@pytest.mark.parametrize("indices", [[0], [8, 0, 3, 3], list(reversed(range(9)))])
def test_route_count_gather_obeys_ascend_dtype_contract(indices):
    """Evaluate the production gather expression with the reported backend limit.

    This is a CPU dtype/exactness regression, not Triton compilation or NPU
    execution. The hardware route suite validates the complete kernel.
    """
    source, kernel = route_kernel_source()
    assignment = next(
        node
        for node in kernel.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "physical_counts" for target in node.targets)
    )
    calls = []

    def gather(src, index, axis):
        # Mirrors the dtype restriction in the user's Ascend Triton semantic.gather.
        if not (src.dtype.is_floating_point or src.dtype == torch.int8):
            raise ValueError(f"Ascend gather does not support {src.dtype}")
        calls.append(src.dtype)
        return torch.index_select(src, axis, index.long())

    counts = torch.arange(9, dtype=torch.int32)  # Every bounded count allowed by B1/top-k<=8.
    slots = torch.tensor(indices, dtype=torch.int32)
    namespace = {
        "tl": SimpleNamespace(gather=gather, float32=torch.float32, int32=torch.int32),
        "counts": counts,
        "safe_expert": slots,
    }
    actual = eval(compile(ast.Expression(assignment.value), str(source), "eval"), namespace)
    assert len(calls) == 1
    assert actual.dtype == torch.int32
    assert torch.equal(actual, counts.index_select(0, slots.long()))


@pytest.mark.parametrize("experts,logical", [([3, 0, 4], 5), ([6, 1, 4, 0, 3], 7)])
@pytest.mark.parametrize("fault", ["none", "id_min", "id_max", "slot_min", "slot_max", "id_oob", "slot_oob"])
def test_route_inverse_lookup_bounds_int64_before_gather(experts, logical, fault):
    """Check the real lookup expressions against integer indexing on CPU."""
    source, kernel = route_kernel_source()
    slots = [-1] * logical
    for slot, expert in enumerate(experts):
        slots[expert] = slot
    experts = list(experts)
    if fault.startswith("id_"):
        experts[0] = {"id_min": -(2**63), "id_max": 2**63 - 1, "id_oob": logical}[fault]
    elif fault.startswith("slot_"):
        slots[0] = {"slot_min": -(2**63), "slot_max": 2**63 - 1, "slot_oob": len(experts)}[fault]

    def gather(src, index, axis):
        assert src.dtype == torch.float32 and index.dtype == torch.int32
        assert bool(((src >= -1) & (src <= 255)).all()), "Raw int64 IDs must never be converted to FP32"
        return torch.index_select(src, axis, index.long())

    selected = {
        "slot_valid",
        "expert_valid",
        "bounded_slots",
        "bounded_experts",
        "safe_slot",
        "safe_expert",
        "back_id",
        "back_slot",
    }
    statements = [
        node
        for node in kernel.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in selected
    ]
    assert {node.targets[0].id for node in statements} == selected
    namespace = {
        "tl": SimpleNamespace(where=torch.where, gather=gather, float32=torch.float32, int32=torch.int32),
        "slots": torch.tensor(slots, dtype=torch.int64),
        "expert": torch.tensor(experts, dtype=torch.int64),
        "LOGICAL": logical,
        "EXPERTS": len(experts),
    }
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(source), "exec"), namespace)
    safe_experts = [expert if 0 <= expert < logical else -1 for expert in experts]
    safe_slots = [slot if 0 <= slot < len(experts) else -1 for slot in slots]
    assert namespace["back_id"].tolist() == [safe_experts[max(slot, 0)] for slot in safe_slots]
    assert namespace["back_slot"].tolist() == [safe_slots[max(expert, 0)] for expert in safe_experts]


def route_reference(group, group_type, expert_ids, logical_slots, rows, *, allow_padding=False):
    """Independent Python-integer oracle; it never executes device code."""
    counts = list(group) if group_type == 1 else [last - first for first, last in zip([0, *group], group)]
    valid = all(0 <= value <= rows for value in group)
    valid = valid and all(0 <= count <= 128 for count in counts)
    valid = valid and (sum(counts) <= rows if allow_padding else sum(counts) == rows)
    for logical, slot in enumerate(logical_slots):
        valid = valid and (
            (slot == -1 and counts[logical] == 0) or (0 <= slot < len(expert_ids) and expert_ids[slot] == logical)
        )
    for slot, logical in enumerate(expert_ids):
        valid = valid and 0 <= logical < len(counts) and logical_slots[logical] == slot
    if not valid:
        return [0] * len(expert_ids), [0] * rows, [0] * rows, [0] * rows, 0
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    physical = [counts[logical] for logical in expert_ids]
    source = [row for logical in expert_ids for row in range(offsets[logical], offsets[logical + 1])]
    slots = [slot for slot, count in enumerate(physical) for _ in range(count)]
    live_rows = len(source)
    inverse = [source.index(row) for row in range(live_rows)] + list(range(live_rows, rows))
    source += [-1] * (rows - live_rows)
    slots += [0] * (rows - live_rows)
    return physical, slots, source, inverse, 1


@pytest.mark.parametrize("padding", [0, 2, 5])
@pytest.mark.parametrize("valid", [True, False])
@pytest.mark.parametrize("group_type", [0, 1])
@pytest.mark.parametrize(
    "counts,experts",
    [
        ([1], [0]),
        ([0], [0]),
        ([0, 0, 0], [2, 0, 1]),
        ([1, 0, 2, 0], [2, 0, 3]),  # Both ROWS and EXPERTS are three.
        ([1, 0, 2, 0], [2, 0, 3, 1]),  # ROWS=3, EXPERTS=4.
        ([1, 0, 3, 0], [2, 0, 3]),  # ROWS=4, EXPERTS=3.
        ([1, 0, 4, 0], [2, 0, 3, 1]),
        ([2, 0, 5, 0], [2, 0, 3, 1]),
        ([0, 0, 0, 8], [3, 2, 0]),
        ([1, 0, 2, 0, 1], [4, 0, 2]),
        ([0, 1, 0, 2, 0, 1, 2], [6, 0, 3, 1, 5]),
    ],
)
def test_route_register_row_selection_matches_integer_oracle(counts, experts, group_type, valid, padding):
    """Evaluate production register arithmetic on CPU, not Triton/NPU code.

    This checks the source/inverse permutation after replacing scattered
    stores with register selection. It cannot prove backend compilation,
    device memory ordering, or graph execution; those remain NPU tests.
    """
    source, kernel = route_kernel_source()
    kernel = next(node for node in kernel.body if isinstance(node, ast.If) and ast.unparse(node.test) == "BUILD_ROWS")
    rows = min(8, max(1, sum(counts) + padding))
    logical_block = 1 << (len(counts) - 1).bit_length()
    expert_block = 1 << (len(experts) - 1).bit_length()
    row_block = 1 << (rows - 1).bit_length()
    start = next(
        index
        for index, node in enumerate(kernel.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "row_offsets" for target in node.targets)
    )
    statements = [node for node in kernel.body[start:] if isinstance(node, (ast.Assign, ast.For))]
    namespace = {
        "tl": SimpleNamespace(
            arange=lambda start, end: torch.arange(start, end, dtype=torch.int32),
            full=lambda shape, value, dtype: torch.full(shape, value, dtype=dtype),
            where=torch.where,
            sum=lambda value, axis: value.sum(dim=axis),
            min=lambda value, axis: value.amin(dim=axis),
            int32=torch.int32,
        ),
        "ROWS": rows,
        "ROW_BLOCK": row_block,
        "EXPERTS": len(experts),
        "logical": torch.arange(logical_block, dtype=torch.int32),
        "physical": torch.arange(expert_block, dtype=torch.int32),
        "physical_active": torch.arange(expert_block) < len(experts),
        "counts": torch.tensor(counts + [0] * (logical_block - len(counts)), dtype=torch.int32),
        "safe_expert": torch.tensor(experts + [0] * (expert_block - len(experts)), dtype=torch.int32),
        "physical_counts": torch.tensor(
            [counts[expert] for expert in experts] + [0] * (expert_block - len(experts)), dtype=torch.int32
        ),
        "valid": torch.tensor(valid),
    }
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(source), "exec"), namespace)
    actual = {}
    for node in kernel.body[start:]:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "store":
            continue
        pointers = {part.id for part in ast.walk(call.args[0]) if isinstance(part, ast.Name)}
        matched = pointers & {"row_slots_ptr", "gather_ptr", "inverse_ptr"}
        if not matched:
            continue
        mask = next(keyword.value for keyword in call.keywords if keyword.arg == "mask")
        selected = eval(compile(ast.Expression(mask), str(source), "eval"), namespace)
        assert selected.tolist() == [True] * rows + [False] * (row_block - rows)
        values = eval(compile(ast.Expression(call.args[1]), str(source), "eval"), namespace)
        actual[next(iter(matched))] = values[selected].tolist()

    group = counts if group_type == 1 else [sum(counts[: index + 1]) for index in range(len(counts))]
    slots = [-1] * len(counts)
    for slot, expert in enumerate(experts):
        slots[expert] = slot
    _, expected_slots, expected_source, expected_inverse, _ = route_reference(
        group, group_type, experts, slots, rows, allow_padding=True
    )
    assert actual == {
        "row_slots_ptr": expected_slots if valid else [0] * rows,
        "gather_ptr": expected_source if valid else [0] * rows,
        "inverse_ptr": expected_inverse if valid else [0] * rows,
    }


def test_gs32_device_routes_reference_reverse_bank_and_zero_experts():
    expected = ([2, 1, 0], [0, 0, 1], [1, 2, 0], [2, 0, 1], 1)
    assert route_reference([1, 0, 2, 0], 1, [2, 0, 3], [1, -1, 0, 2], 3) == expected
    assert route_reference([1, 1, 3, 3], 0, [2, 0, 3], [1, -1, 0, 2], 3) == expected


@pytest.mark.parametrize("group_type", [0, 1])
@pytest.mark.parametrize("bad", [-(2**63), 2**63 - 1, -1, 4])
def test_gs32_device_routes_reference_extremes_are_invalid(group_type, bad):
    group = [1, 1, 3, 3] if group_type == 0 else [1, 0, 2, 0]
    group[0] = bad
    assert route_reference(group, group_type, [2, 0, 3], [1, -1, 0, 2], 3)[-1] == 0


def route_inputs():
    return (
        tagged(torch.arange(3 * 64).reshape(3, 64).bfloat16()),
        tagged(torch.tensor([1, 0, 2, 0], dtype=torch.int64)),
        1,
        tagged(torch.tensor([2, 0, 3], dtype=torch.int64)),
        tagged(torch.tensor([1, -1, 0, 2], dtype=torch.int64)),
    )


@pytest.fixture
def routes(monkeypatch):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(device=lambda device: nullcontext()), raising=False)
    return importlib.import_module("vllm_ascend.ops.fused_moe.vq2a8_gs32_routes")


def test_gs32_device_routes_reject_cpu_execution(routes):
    args = list(route_inputs())
    args[0] = torch.zeros(3, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="NPU"):
        routes.prepare(*args)


@pytest.mark.parametrize("group_type,dtype", [(0, torch.int32), (0, torch.int64), (1, torch.int32), (1, torch.int64)])
def test_gs32_device_routes_forward_without_read_or_capture_query(routes, monkeypatch, group_type, dtype):
    args = list(route_inputs())
    args[1] = tagged(args[1].as_subclass(torch.Tensor).to(dtype))
    args[2] = group_type
    results = tuple(object() for _ in range(5))
    calls = []

    def launch(*received, allow_padding=False):
        assert allow_padding is False
        calls.append(received)
        return results

    monkeypatch.setattr(routes, "prepare_routes", launch)
    assert routes.prepare(*args) is results
    assert all(actual is expected for actual, expected in zip(calls[0], args, strict=True))


@pytest.mark.parametrize("rows", [9, 32, 192, 256])
def test_gs32_device_routes_accept_prefill_rows_without_host_read(routes, monkeypatch, rows):
    args = list(route_inputs())
    args[0] = tagged(torch.zeros(rows, 64, dtype=torch.bfloat16))
    calls = []
    monkeypatch.setattr(routes, "prepare_routes", lambda *received, **options: calls.append(received))
    routes.prepare(*args)
    assert len(calls) == 1
    assert all(actual is expected for actual, expected in zip(calls[0], args, strict=True))


@pytest.mark.parametrize("padding", [0, 7])
@pytest.mark.parametrize("valid", [True, False])
@pytest.mark.parametrize(
    "counts,experts",
    [
        ([32] * 6, [5, 3, 1, 4, 2, 0]),
        ([0, 0, 0], [2, 0, 1]),
        ([32] * 8, list(reversed(range(8)))),
        ([128, 0, 64], [2, 1, 0]),
        ([0, 7, 0, 31, 0, 25, 128], [6, 3, 1, 5, 0]),
        ([1] * 256, list(reversed(range(256)))),
    ],
)
def test_prefill_route_affine_stores_match_integer_oracle(counts, experts, valid, padding):
    """Check prefill row arithmetic without emulating Triton or NPU execution."""
    source, _ = route_kernel_source()
    module = ast.parse(source.read_text())
    kernel = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "_prefill_route_rows_kernel"
        ),
        None,
    )
    assert kernel is not None, "Prefill needs a parallel row builder rather than the decode row-selection loop"
    rows = min(256, max(9, sum(counts) + padding))
    row_block = 1 << (rows - 1).bit_length()
    start = next(
        index
        for index, node in enumerate(kernel.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "row_offsets" for target in node.targets)
    )
    outputs = {name: [None] * rows for name in ("row_slots_ptr", "gather_ptr", "inverse_ptr")}
    physical_counts = [counts[expert] for expert in experts]
    for slot, expert in enumerate(experts):
        namespace = {
            "tl": SimpleNamespace(arange=lambda first, last: torch.arange(first, last), where=torch.where),
            "ROWS": rows,
            "ROW_BLOCK": row_block,
            "slot": slot,
            "count": physical_counts[slot] if valid else 0,
            "physical_start": sum(physical_counts[:slot]) if valid else 0,
            "total_rows": sum(counts) if valid else 0,
            "source_start": sum(counts[:expert]),
            "valid": torch.tensor(valid),
        }
        statements = [node for node in kernel.body[start:] if isinstance(node, ast.Assign)]
        exec(compile(ast.Module(body=statements, type_ignores=[]), str(source), "exec"), namespace)
        for node in kernel.body[start:]:
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            if ast.unparse(call.func) != "tl.store":
                continue
            # Every store is a contiguous row vector; dynamic data only changes
            # masks/values, never the GM address (Ascend insert_slice regression).
            address = call.args[0]
            assert isinstance(address, ast.BinOp) and isinstance(address.op, ast.Add)
            assert isinstance(address.left, ast.Name) and address.left.id in outputs
            assert isinstance(address.right, ast.Name) and address.right.id == "row_offsets"
            mask = next(keyword.value for keyword in call.keywords if keyword.arg == "mask")
            selected = eval(compile(ast.Expression(mask), str(source), "eval"), namespace)
            values = eval(compile(ast.Expression(call.args[1]), str(source), "eval"), namespace)
            values = torch.broadcast_to(values, (row_block,))
            for row in torch.where(selected)[0].tolist():
                assert outputs[address.left.id][row] is None, "Expert row stores must never overlap"
                outputs[address.left.id][row] = int(values[row])
    slots = [-1] * len(counts)
    for slot, expert in enumerate(experts):
        slots[expert] = slot
    _, expected_slots, source_rows, inverse, _ = route_reference(counts, 1, experts, slots, rows, allow_padding=True)
    assert outputs == {
        "row_slots_ptr": expected_slots if valid else [0] * rows,
        "gather_ptr": source_rows if valid else [0] * rows,
        "inverse_ptr": inverse if valid else [0] * rows,
    }


@pytest.mark.parametrize(
    "field",
    ["rows", "width", "dtype", "group_type", "group_dtype", "group_rank", "experts", "id_dtype", "slots", "device"],
)
def test_gs32_device_routes_invalid_metadata_rejected_before_launch(routes, monkeypatch, field):
    args = list(route_inputs())
    if field == "rows":
        args[0] = tagged(torch.zeros(257, 64, dtype=torch.bfloat16))
    elif field == "width":
        args[0] = tagged(torch.zeros(3, 63, dtype=torch.bfloat16))
    elif field == "dtype":
        args[0] = tagged(torch.zeros(3, 64, dtype=torch.float32))
    elif field == "group_type":
        args[2] = 2
    elif field == "group_dtype":
        args[1] = tagged(torch.zeros(4, dtype=torch.float32))
    elif field == "group_rank":
        args[1] = tagged(torch.zeros(1, 4, dtype=torch.int64))
    elif field == "experts":
        args[3] = tagged(torch.arange(257))
    elif field == "id_dtype":
        args[3] = tagged(torch.arange(3, dtype=torch.int32))
    elif field == "slots":
        args[4] = tagged(torch.arange(3))
    else:
        args[3] = tagged(torch.arange(3), index=1)

    def forbidden(*args):
        pytest.fail("Invalid route metadata reached the NPU kernel.")

    monkeypatch.setattr(routes, "prepare_routes", forbidden)
    with pytest.raises(ValueError):
        routes.prepare(*args)


@pytest.mark.parametrize("allow_padding", [False, True])
def test_ep_route_padding_is_an_explicit_launch_option(routes, monkeypatch, allow_padding):
    calls = []
    args = route_inputs()
    monkeypatch.setattr(routes, "prepare_routes", lambda *values, **options: calls.append((values, options)))
    routes.prepare(*args, allow_padding=allow_padding)
    assert len(calls) == 1
    assert all(got is expected for got, expected in zip(calls[0][0], args, strict=True))
    assert calls[0][1] == {"allow_padding": allow_padding}


@pytest.mark.parametrize("allow_padding", [None, 0, 1, "true"])
def test_ep_route_padding_rejects_non_boolean_option(routes, monkeypatch, allow_padding):
    monkeypatch.setattr(routes, "prepare_routes", lambda *args, **kwargs: pytest.fail("Invalid mode reached launch"))
    with pytest.raises(ValueError, match="allow_padding"):
        routes.prepare(*route_inputs(), allow_padding=allow_padding)


@pytest.mark.parametrize("allow_padding", [False, True])
@pytest.mark.parametrize(
    "counts,physical_counts,rows,value_valid",
    [
        ([0, 0], [0, 0], 8, True),
        ([1, 2], [2, 1], 8, True),
        ([4, 4], [4, 4], 8, True),
        ([4, 5], [5, 4], 8, True),
        ([1, 2], [1, 1], 8, True),
        ([129, 0], [0, 129], 256, False),
    ],
)
def test_ep_route_padding_production_validity_sums(allow_padding, counts, physical_counts, rows, value_valid):
    """Execute production validity expressions on CPU, without compiler claims."""
    source, kernel = route_kernel_source()
    namespace = {
        "tl": SimpleNamespace(
            where=torch.where,
            min=lambda v, axis: v.amin(dim=axis),
            sum=lambda v, axis: v.sum(dim=axis),
            int32=torch.int32,
        ),
        "logical_active": torch.tensor([True, True]),
        "physical_active": torch.tensor([True, True]),
        "value_valid": torch.tensor([value_valid, True]),
        "mapping_valid": torch.tensor([True, True]),
        "expert_valid": torch.tensor([True, True]),
        "physical": torch.tensor([0, 1]),
        "back_slot": torch.tensor([0, 1]),
        "counts": torch.tensor(counts),
        "physical_counts": torch.tensor(physical_counts),
        "ROWS": rows,
        "ALLOW_PADDING": allow_padding,
    }
    statements = [
        node
        for node in kernel.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "valid" for target in node.targets)
    ]
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(source), "exec"), namespace)
    expected = value_valid and sum(counts) == sum(physical_counts)
    expected = expected and (sum(counts) <= rows if allow_padding else sum(counts) == rows)
    assert namespace["valid"].dtype == torch.bool
    assert bool(namespace["valid"]) == expected
