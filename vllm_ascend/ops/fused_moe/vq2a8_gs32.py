# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GS32 projections over resident unified parameters, without expert I/O.

The serial path remains the validation baseline. Batched execution preserves
native expert/chunk geometry while sharing activation work and stage checks.
"""

from pathlib import Path

import torch

from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_format import GS32_FIELDS, GS32_MAX_ROWS
from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_staged import quantize_mxfp8_k32, require_finite, staged_swiglu

NATIVE_ALIGNMENT = 32
MAX_GROUPED_EXPERTS = 256  # Imported mixed kernel host validation limit.
MAX_GRAPH_ROUTED_ROWS = 256  # Fixed MRV2 DSpark verification capacity; per-expert rows remain <=128.
MAX_PREFILL_GRAPH_ROUTED_ROWS = 256  # Fixed I32 prefill with top-k <=8.


class _NativeStageOps:
    def __init__(self, library, binary, device, *, execution="serial"):
        # Extension/runtime registration happens once during worker startup,
        # never within a queued per-projection launch callback.
        import torch_npu  # noqa: F401

        self.device = torch.device(device)
        self.execution = execution
        self.plans = {}
        self.grouped_plans = {}
        if self.device.type != "npu" or self.device.index is None:
            raise ValueError("GS32 runtime requires an explicit NPU index.")
        for path in (library, binary):
            if not Path(path).is_file():
                raise ValueError(f"GS32 native artifact is missing: {path}")
        with torch.npu.device(self.device):
            torch.ops.load_library(str(Path(library).resolve()))
            self.ops = torch.ops.vq2a8_gs32_stage
            required = ("prepare", "project_prepared") if execution != "serial" else ()
            if execution in ("fused", "full_graph", "full_graph_prefill"):
                required += ("prepare_routes", "prepare_grouped", "project_grouped")
            if execution in ("full_graph", "full_graph_prefill"):
                required += ("prepare_device_routes", "device_routes_status", "project_grouped_device")
            if not all(hasattr(self.ops, name) for name in required):
                raise RuntimeError(f"GS32 {execution} execution requires rebuilding the GS32 Torch bridge .so.")
            self.ops.initialize(str(Path(binary).resolve()))

    def make_tiling(self, k, n):
        return self.ops.make_tiling(k, n)

    def check_eager(self):
        capturing = getattr(torch.npu, "is_current_stream_capturing", None)
        if capturing is not None and capturing():
            raise RuntimeError("GS32 staged backend does not support graph capture.")

    def project(self, data, scales, weight, lut, row_scale, group_list, mask, tiling, output):
        self.check_eager()
        if self.execution == "serial":
            self.ops.project(data, scales, weight, lut, row_scale, group_list, mask, tiling, output)
            return
        key = (data.shape[0], data.shape[1], output.shape[1])
        plan = self.plans.get(key)
        if plan is None:
            plan = self.ops.prepare(data, output.shape[1])
            self.plans[key] = plan
        self.ops.project_prepared(data, scales, weight, lut, row_scale, plan, output)

    def prepare_routes(self, hidden, counts):
        self.check_eager()
        return self.ops.prepare_routes(hidden, counts)

    def project_grouped(self, data, scales, weight, lut, row_scale, routes, output):
        self.check_eager()
        key = (weight.shape[0], data.shape[1], output.shape[1])
        plan = self.grouped_plans.get(key)
        if plan is None:
            plan = self.ops.prepare_grouped(data, output.shape[1], weight.shape[0])
            self.grouped_plans[key] = plan
        self.ops.project_grouped(data, scales, weight, lut, row_scale, routes, plan, output)

    def prepare_grouped_plan(self, anchor, n, groups):
        self.check_eager()
        return self.ops.prepare_grouped(anchor, n, groups)

    def require_padding_support(self):
        schema = torch._C._dispatch_find_schema_or_throw("vq2a8_gs32_stage::prepare_device_routes", "").schema()
        if "bool allow_padding=False" not in str(schema):
            raise RuntimeError("GS32 EP graph execution requires rebuilding the GS32 Torch bridge .so for padding.")

    def prepare_device_routes(self, hidden, counts, *, allow_padding=False):
        if allow_padding:
            return self.ops.prepare_device_routes(hidden, counts, True)
        return self.ops.prepare_device_routes(hidden, counts)

    def device_routes_status(self, routes):
        return self.ops.device_routes_status(routes)

    def project_grouped_device(self, data, scales, weight, lut, row_scale, routes, plan, output):
        self.ops.project_grouped_device(data, scales, weight, lut, row_scale, routes, plan, output)


class GS32ExpertWeights:
    decode_mode = "grouped"

    def __init__(self, layer, layer_spec, options, *, _test_ops=None, _test_activation=None):
        self.execution = options.get("gs32_execution", "serial")
        if self.execution not in ("serial", "batched", "fused", "full_graph", "full_graph_prefill"):
            raise ValueError("GS32 execution must be serial, batched, fused, full_graph or full_graph_prefill.")
        self.activation = _test_activation
        if self.execution in ("fused", "full_graph", "full_graph_prefill") and self.activation is None:
            # Worker-only lazy import. Triton loads only when the mode is used.
            from vllm_ascend.ops.fused_moe import vq2a8_gs32_activation

            self.activation = vq2a8_gs32_activation
        self.specs = layer_spec.specs
        self.expert_slots = {expert: slot for slot, expert in enumerate(layer_spec.expert_ids)}
        self.num_experts = layer.vq2a8_num_experts
        self.allow_padding = getattr(layer, "vq2a8_ep_size", 1) > 1
        self.fields = {
            kind: {field: getattr(layer, f"{prefix}_{field}") for field in GS32_FIELDS}
            for kind, prefix in (("gate_up", "w13"), ("down", "w2"))
        }
        self.device = self.fields["gate_up"]["packed_zn"].device
        self.ops = (
            _test_ops
            if _test_ops is not None
            else _NativeStageOps(options["gs32_library"], options["gs32_binary"], self.device, execution=self.execution)
        )
        self.tilings = {kind: self.ops.make_tiling(spec.columns, spec.rows) for kind, spec in self.specs.items()}
        self.route_mask = torch.zeros(1, dtype=torch.int32, device="cpu")
        self.group_counts = {}
        self.defer_validity = False
        self.validity_buffer = torch.zeros((), dtype=torch.bool, device=self.device)
        self.capture_owners = []
        if self.execution in ("full_graph", "full_graph_prefill"):
            from vllm_ascend.ops.fused_moe import vq2a8_gs32_routes

            if self.allow_padding:
                self.ops.require_padding_support()
            if self.num_experts > MAX_GROUPED_EXPERTS or len(self.expert_slots) > MAX_GROUPED_EXPERTS:
                raise ValueError("GS32 full_graph requires at most 256 logical and resident experts.")
            self.device_router = vq2a8_gs32_routes
            self.expert_ids = torch.tensor(layer_spec.expert_ids, dtype=torch.int64, device=self.device)
            self.logical_slots = torch.full((self.num_experts,), -1, dtype=torch.int64, device=self.device)
            self.logical_slots[self.expert_ids] = torch.arange(len(self.expert_slots), device=self.device)
            # Registration and immutable metadata uploads finish before any
            # capture. The graph path must never lazily create a host plan.
            self.device_plans = {
                kind: self.ops.prepare_grouped_plan(
                    torch.empty((1, spec.columns), dtype=torch.uint8, device=self.device),
                    spec.rows,
                    len(self.expert_slots),
                )
                for kind, spec in self.specs.items()
                if self.expert_slots
            }

    def prepare_for_current_stream(self):
        # Native registration and immutable projection plans are built at load time.
        pass

    def _accept_validity(self, status):
        self.validity_buffer.copy_(status.reshape(()).bool())
        if not self.defer_validity and not bool(self.validity_buffer.cpu()):
            raise ValueError("GS32 device preparation or projection failed validity checks.")

    def _mark_eager_valid(self):
        self.validity_buffer.fill_(True)

    def _apply_device(self, hidden, group_list, group_list_type, limit, capturing):
        if self.allow_padding and not self.expert_slots:
            # Hash layers can have no resident experts on a rank. Keep that
            # rank in the graph/collectives and validate zero routing on device.
            self._accept_validity((group_list == 0).all())
            return hidden.new_zeros((hidden.shape[0], self.specs["down"].rows))
        route_options = {"allow_padding": True} if self.allow_padding else {}
        physical, counts, slots, inverse, route_flags = self.device_router.prepare(
            hidden, group_list, group_list_type, self.expert_ids, self.logical_slots, **route_options
        )
        routes = self.ops.prepare_device_routes(physical, counts, **route_options)
        native_flags = self.ops.device_routes_status(routes)
        retained = [physical, counts, slots, inverse, route_flags, native_flags, routes]
        value, raw_gate, stage_flags = physical, None, []
        for kind in ("gate_up", "down"):
            fields, spec = self.fields[kind], self.specs[kind]
            data, scales, flags = self.activation.prepare_unchecked(
                value, fields["weight_scale"], fields["perm"], slots, raw_gate=raw_gate
            )
            output = torch.full((hidden.shape[0], spec.rows), float("nan"), device=self.device, dtype=torch.bfloat16)
            self.ops.project_grouped_device(
                data,
                scales,
                fields["packed_zn"],
                fields["pair_lut"],
                fields["row_scale_bits"],
                routes,
                self.device_plans[kind],
                output,
            )
            retained.extend((data, scales, flags, output))
            stage_flags.append(flags)
            if kind == "gate_up":
                raw_gate = output
                value = self.activation.swiglu_unchecked(output, limit)
                retained.append(value)
        status = self.activation.finish_flags(output, *stage_flags, route_flags=route_flags, native_flags=native_flags)
        self._accept_validity(status)
        restored = output.index_select(0, inverse)
        if capturing:
            # Custom native tasks reference these addresses on every replay.
            # Keep immutable plans and per-capture routes/intermediates alive
            # with the model, even after capture's Python stack disappears.
            # Recapture can reuse input addresses while an older graph still
            # exists, so never replace a previous capture's ownership record.
            self.capture_owners.append((*retained, status, restored))
        return restored

    def _group_count(self, rows):
        count = self.group_counts.get(rows)
        if count is None:
            count = torch.tensor([rows], dtype=torch.int64, device="cpu")
            self.group_counts[rows] = count
        return count

    def _project(self, kind, expert, hidden):
        self.ops.check_eager()
        spec, fields = self.specs[kind], self.fields[kind]
        slot = self.expert_slots[expert]
        transformed = (hidden.float() * fields["weight_scale"][slot]).index_select(1, fields["perm"][slot])
        data, scales = quantize_mxfp8_k32(transformed)
        output = torch.full((hidden.shape[0], spec.rows), float("nan"), device=self.device, dtype=torch.bfloat16)
        self.ops.project(
            data,
            scales,
            fields["packed_zn"][slot : slot + 1],
            fields["pair_lut"][slot : slot + 1],
            fields["row_scale_bits"][slot : slot + 1],
            self._group_count(hidden.shape[0]),
            self.route_mask,
            self.tilings[kind],
            output,
        )
        # Required until every native shape passes the hole/finite acceptance tests.
        require_finite(output, "GS32 native projection output")
        return output

    def _project_batched(self, kind, hidden, jobs):
        spec, fields = self.specs[kind], self.fields[kind]
        # Preserve each expert's scale -> permutation arithmetic and the
        # original native M geometry. Only independent row preparation batches.
        transformed = torch.cat(
            [
                (hidden[first:last].float() * fields["weight_scale"][slot]).index_select(1, fields["perm"][slot])
                for slot, first, last in jobs
            ]
        )
        data, scales = quantize_mxfp8_k32(transformed)
        output = torch.full((hidden.shape[0], spec.rows), float("nan"), device=self.device, dtype=torch.bfloat16)
        for slot, first, last in jobs:
            chunk_scales = scales[first:last]
            # Small K fixtures can put a row view between 32-byte boundaries.
            # A contiguous view does not imply native pointer alignment.
            if chunk_scales.data_ptr() % NATIVE_ALIGNMENT:
                chunk_scales = chunk_scales.clone()
            self.ops.project(
                data[first:last],
                chunk_scales,
                fields["packed_zn"][slot : slot + 1],
                fields["pair_lut"][slot : slot + 1],
                fields["row_scale_bits"][slot : slot + 1],
                self._group_count(last - first),
                self.route_mask,
                self.tilings[kind],
                output[first:last],
            )
        # Preserve hole/NaN/Inf rejection before the next numerical stage.
        require_finite(output, f"GS32 {kind} native projection output")
        return output

    def _apply_fused(self, hidden, counts, limit):
        # The mixed kernel visits weights in physical bank order. Dispatcher
        # rows are in logical expert order, which need not match that bank.
        slot_experts = sorted(self.expert_slots, key=self.expert_slots.get)
        physical_counts = [counts[expert] for expert in slot_experts]
        inverse = None
        if slot_experts != sorted(slot_experts):
            offsets = [0]
            for count in counts:
                offsets.append(offsets[-1] + count)
            order = [row for expert in slot_experts for row in range(offsets[expert], offsets[expert + 1])]
            indices = torch.tensor(order, dtype=torch.int64, device=self.device)
            hidden = hidden.index_select(0, indices)
            inverse = torch.tensor(sorted(range(len(order)), key=order.__getitem__), device=self.device)
        row_slots = torch.tensor(
            [slot for slot, count in enumerate(physical_counts) for _ in range(count)],
            dtype=torch.int64,
            device=self.device,
        )
        routes = self.ops.prepare_routes(hidden, torch.tensor(physical_counts, dtype=torch.int64, device="cpu"))
        gate_up = self._project_grouped("gate_up", hidden, row_slots, routes)
        middle = self.activation.swiglu_unchecked(gate_up, limit)
        # Validate raw gate values as well as the rounded/quantized middle:
        # clipping must not conceal an Inf or a native partial write.
        output = self._project_grouped("down", middle, row_slots, routes, raw_gate=gate_up)
        self.activation.require_finite_fused(output, "GS32 Down output")
        return output if inverse is None else output.index_select(0, inverse)

    def _project_grouped(self, kind, hidden, row_slots, routes, *, raw_gate=None):
        fields, spec = self.fields[kind], self.specs[kind]
        data, scales = self.activation.prepare(
            hidden, fields["weight_scale"], fields["perm"], row_slots, raw_gate=raw_gate
        )
        output = torch.full((hidden.shape[0], spec.rows), float("nan"), device=self.device, dtype=torch.bfloat16)
        self.ops.project_grouped(
            data, scales, fields["packed_zn"], fields["pair_lut"], fields["row_scale_bits"], routes, output
        )
        return output

    @torch.inference_mode()
    def apply(self, hidden_states, group_list, group_list_type, swiglu_limit):
        capturing = False
        graph_rows = MAX_PREFILL_GRAPH_ROUTED_ROWS if self.execution == "full_graph_prefill" else MAX_GRAPH_ROUTED_ROWS
        if self.execution in ("full_graph", "full_graph_prefill"):
            capture_query = getattr(torch.npu, "is_current_stream_capturing", None)
            capturing = capture_query is not None and capture_query()
            if capturing and not self.defer_validity:
                raise RuntimeError("GS32 whole-graph capture requires a forward owner for deferred validity.")
            if capturing and hidden_states.shape[0] > graph_rows:
                raise RuntimeError(f"GS32 whole-graph execution supports at most {graph_rows} routed rows.")
        else:
            self.ops.check_eager()
        if (
            hidden_states.ndim != 2
            or hidden_states.dtype != torch.bfloat16
            or hidden_states.device != self.device
            or hidden_states.shape[1] != self.specs["gate_up"].columns
        ):
            raise ValueError("GS32 expects dispatched resident BF16 [tokens,hidden].")
        if (
            group_list_type not in (0, 1)
            or group_list.ndim != 1
            or group_list.dtype not in (torch.int32, torch.int64)
            or group_list.numel() != self.num_experts
        ):
            raise ValueError("GS32 group_list must have an integer count/prefix for every logical expert.")
        if self.execution in ("full_graph", "full_graph_prefill") and group_list.device != self.device:
            raise ValueError("GS32 graph group_list must be on the same device as hidden states.")
        if self.execution in ("full_graph", "full_graph_prefill") and 0 < hidden_states.shape[0] <= graph_rows:
            return self._apply_device(
                hidden_states, group_list, group_list_type, 0.0 if swiglu_limit is None else swiglu_limit, capturing
            ), None
        if capturing:
            raise RuntimeError("GS32 full_graph requires a nonempty B1 decode shape.")
        # One explicit count transfer, never one tensor.item() per expert.
        counts = group_list.cpu().tolist()
        if group_list_type == 0:
            counts = [end - start for end, start in zip(counts, [0, *counts[:-1]])]
        capacity, active_rows = hidden_states.shape[0], sum(counts)
        allow_padding = getattr(self, "allow_padding", False)
        if (
            any(count < 0 for count in counts)
            or active_rows > capacity
            or (not allow_padding and active_rows != capacity)
            or any(count and expert not in self.expert_slots for expert, count in enumerate(counts))
        ):
            raise ValueError("GS32 group counts or routed expert inventory are invalid.")
        if not active_rows:
            self._mark_eager_valid()
            return hidden_states.new_zeros((capacity, self.specs["down"].rows)), None
        # ALLGATHER EP retains a fixed dispatch capacity; only the leading
        # sum(local counts) rows belong to this rank. Never quantize the tail.
        hidden_states = hidden_states[:active_rows]

        def finish(output):
            self._mark_eager_valid()
            if active_rows == capacity:
                return output, None
            padded = output.new_zeros((capacity, output.shape[1]))
            padded[:active_rows].copy_(output)
            return padded, None

        if (
            self.execution in ("fused", "full_graph", "full_graph_prefill")
            and len(self.expert_slots) <= MAX_GROUPED_EXPERTS
            and max(counts) <= GS32_MAX_ROWS
        ):
            output = self._apply_fused(hidden_states, counts, 0.0 if swiglu_limit is None else swiglu_limit)
            return finish(output)
        if self.execution in ("batched", "fused", "full_graph", "full_graph_prefill"):
            jobs = []
            start = 0
            for expert, count in enumerate(counts):
                for offset in range(0, count, GS32_MAX_ROWS):
                    jobs.append((self.expert_slots[expert], start + offset, start + min(count, offset + GS32_MAX_ROWS)))
                start += count
            gate_up = self._project_batched("gate_up", hidden_states, jobs)
            middle = staged_swiglu(gate_up, 0.0 if swiglu_limit is None else swiglu_limit)
            output = self._project_batched("down", middle, jobs)
            return finish(output)
        output = hidden_states.new_empty((hidden_states.shape[0], self.specs["down"].rows))
        start = 0
        for expert, count in enumerate(counts):
            for offset in range(0, count, GS32_MAX_ROWS):
                first, last = start + offset, start + min(count, offset + GS32_MAX_ROWS)
                gate_up = self._project("gate_up", expert, hidden_states[first:last])
                middle = staged_swiglu(gate_up, 0.0 if swiglu_limit is None else swiglu_limit)
                output[first:last].copy_(self._project("down", expert, middle))
            start += count
        # The common combine path permits no overlap event for a fully checked eager result.
        return finish(output)
