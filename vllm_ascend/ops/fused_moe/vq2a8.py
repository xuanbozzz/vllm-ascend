# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP1 VQ experts for the common dispatch/MLP/combine pipeline.

Dispatch must preserve BF16: each expert has its own RHT signs, input scales
and bias correction. Grouped execution reads expert counts on the host. B1
device execution retains selected route slots without a count transfer. Both
check fresh validity before returning unless a forward owner explicitly takes
responsibility for checking the fixed device buffer after the whole forward.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_ascend.ops.vq2a8 import (
    VQ2A8_DTYPES,
    VQ2A8_FIELDS,
    VQ2A8_MAX_EXPERTS,
    VQ2A8_SUPPORTED_K,
    VQ2A8_SUPPORTED_N,
    VQ2A8ProjectionSpec,
    get_vq2a8_bank_type,
    projection_shapes,
    require_vq2a8_features,
)
from vllm_ascend.quantization.methods.w2a8.vq2a8_activation import (
    MAX_PREFILL_PREPARATION_ROWS,
    PackedRowwiseVQ2A8Preparation,
    RowwiseVQ2A8Preparation,
    VQ2A8ProjectionBatchPlan,
    deepseek_v4_swiglu_grouped,
    deepseek_v4_swiglu_reference,
    pack_vq2a8_projection_inputs,
)

MAX_PROJECTION_ROWS = 32
MAX_PROJECTION_JOBS = 6


class VQ2A8ExpertWeights:
    """Own immutable expert tensors and native banks on their creation stream."""

    def __init__(
        self,
        *,
        banks,
        payloads,
        specs,
        tensors,
        device,
        owner_stream,
        expert_ids=None,
        num_experts=None,
        native_options=None,
        prefill_graph_cache=None,
    ):
        self.banks = banks
        self.payloads = payloads
        self.specs = specs
        self._tensors = tuple(tensors)
        self.device = device
        self.owner_stream = owner_stream
        self._stream_banks = [(owner_stream, banks)]
        stored_experts = len(payloads["gate_up"])
        self.num_experts = stored_experts if num_experts is None else num_experts
        self.expert_ids = tuple(range(stored_experts)) if expert_ids is None else tuple(expert_ids)
        if (
            type(self.num_experts) is not int
            or self.num_experts <= 0
            or len(self.expert_ids) != stored_experts
            or len(set(self.expert_ids)) != stored_experts
            or any(type(expert) is not int or not 0 <= expert < self.num_experts for expert in self.expert_ids)
        ):
            raise ValueError("VQ2A8 stored expert IDs must uniquely map the expert axis into logical experts.")
        self._expert_slots = {expert: slot for slot, expert in enumerate(self.expert_ids)}
        self._slots = torch.arange(stored_experts, dtype=torch.int64, device=device)
        options = native_options or {}
        self.prefill_batch_mode = options.get("prefill_batch_mode", "jobs")
        if self.prefill_batch_mode not in ("jobs", "layer"):
            raise ValueError("VQ2A8 prefill_batch_mode must be jobs or layer.")
        self.prefill_layer_calls = 0
        self._preparation = RowwiseVQ2A8Preparation(
            compact=True, rht_mode=options.get("prefill_rht", "rowwise"), graph_cache=prefill_graph_cache
        )
        self.decode_mode = options.get("decode_mode", "grouped")
        self.max_device_tokens = options.get("_max_device_tokens", 1)
        self.activation_preparation = options.get("activation_preparation", "packed")
        self.validity_mode = options.get("validity_mode", "torch")
        if self.decode_mode not in ("grouped", "device"):
            raise ValueError("VQ2A8 decode_mode must be grouped or device.")
        if self.activation_preparation not in ("packed", "sign_fused_direct"):
            raise ValueError("VQ2A8 activation_preparation must be packed or sign_fused_direct.")
        if self.validity_mode not in ("torch", "fused"):
            raise ValueError("VQ2A8 validity_mode must be torch or fused.")
        self._projection_validity = None
        if self.validity_mode == "fused":
            if self.decode_mode != "device" or self.activation_preparation != "sign_fused_direct":
                raise ValueError("VQ2A8 fused validity requires device decode and sign_fused_direct preparation.")
            require_vq2a8_features(activation_preparation=self.activation_preparation, validity_mode="fused")
            self._projection_validity = torch.ops.vq2a8_ascendc_v4_v2.projection_validity
        # One fresh device flag per invocation. A forward/graph owner can
        # explicitly defer the host decision and consume last_validity.
        self.defer_validity = False
        self.validity_buffer = torch.zeros((), dtype=torch.bool, device=device)
        self._lookup = None
        self._packed_preparations = {}
        if self.decode_mode == "device":
            if any(spec.columns != spec.rht_true_columns for spec in self.specs.values()):
                raise ValueError(
                    "VQ2A8 device decode requires unpadded projections; use grouped decode for padded inputs."
                )
            lookup = torch.full((self.num_experts,), -1, dtype=torch.int64, device="cpu")
            lookup[list(self.expert_ids)] = torch.arange(stored_experts, device="cpu")
            self._lookup = lookup.to(device)
            for kind, spec in self.specs.items():
                direct = self.activation_preparation == "sign_fused_direct"
                preparation = PackedRowwiseVQ2A8Preparation(
                    compact=True, fuse_sign=direct, strided_sign=direct, direct_output=direct
                )
                # Prepare constants outside capture, separately for gate/up
                # and down even if their RHT block geometries differ.
                preparation.prepare_for_graph(device, spec.rht_block_size)
                self._packed_preparations[kind] = preparation

    @classmethod
    def from_layer(
        cls,
        layer: torch.nn.Module,
        *,
        gate_up_spec: VQ2A8ProjectionSpec,
        down_spec: VQ2A8ProjectionSpec,
        native_options=None,
        prefill_graph_cache=None,
    ) -> VQ2A8ExpertWeights:
        for config in (layer, getattr(layer, "moe_config", None)):
            if config is not None and any(getattr(config, key, 1) != 1 for key in ("tp_size", "ep_size")):
                raise ValueError("VQ2A8 grouped experts currently require TP1 without expert parallelism.")
        specs = {"gate_up": gate_up_spec, "down": down_spec}
        if gate_up_spec.rows != 2 * down_spec.rht_true_columns or down_spec.rows != gate_up_spec.rht_true_columns:
            raise ValueError("VQ2A8 gate/up and down projection geometries do not form an expert MLP.")
        anchor = layer.w13_packed_zn
        if anchor.device.type != "npu":
            raise ValueError("VQ2A8 native expert banks require NPU weights.")
        if anchor.ndim != 5 or not 1 <= anchor.shape[0] <= VQ2A8_MAX_EXPERTS:
            raise ValueError("VQ2A8 packed weights require an expert axis containing 1..256 experts.")
        num_experts = anchor.shape[0]
        tensors, payloads = [], {}
        for kind, prefix in (("gate_up", "w13"), ("down", "w2")):
            spec = specs[kind]
            n, k = spec.rows, spec.columns
            if n != VQ2A8_SUPPORTED_N or k not in VQ2A8_SUPPORTED_K:
                raise ValueError("VQ2A8 native banks require N4096 and K2048/4096.")
            block = spec.rht_block_size
            if not 0 < spec.rht_true_columns <= k or block <= 0 or block & (block - 1) or k % block:
                raise ValueError("Invalid VQ2A8 RHT geometry.")
            shapes = projection_shapes(spec)
            fields = {}
            for field, dtype, shape in zip(VQ2A8_FIELDS, VQ2A8_DTYPES, shapes):
                tensor = getattr(layer, f"{prefix}_{field}")
                if (
                    tensor.shape != (num_experts, *shape)
                    or tensor.dtype != dtype
                    or tensor.device != anchor.device
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(f"Invalid stacked VQ2A8 parameter {prefix}_{field}.")
                tensors.append(tensor)
                fields[field] = tensor.unbind(0)
            payloads[kind] = tuple(
                {field: fields[field][expert] for field in VQ2A8_FIELDS} for expert in range(num_experts)
            )
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("Initialize VQ2A8 expert banks before graph capture.")
        owner_stream = torch.npu.current_stream(anchor.device)
        bank_type = get_vq2a8_bank_type()
        banks = {
            kind: bank_type(*[[payload[field] for payload in entries] for field in VQ2A8_FIELDS])
            for kind, entries in payloads.items()
        }
        return cls(
            banks=banks,
            payloads=payloads,
            specs=specs,
            tensors=tensors,
            device=anchor.device,
            owner_stream=owner_stream,
            expert_ids=getattr(layer, "vq2a8_expert_ids", None),
            num_experts=getattr(layer, "vq2a8_num_experts", None),
            native_options=native_options,
            prefill_graph_cache=prefill_graph_cache,
        )

    @property
    def last_validity(self):
        """Alias for consumers that inspect the current fixed validity buffer."""
        return self.validity_buffer

    def prepare_for_current_stream(self):
        """Warm a bank on the current stream, retaining its immutable owners.

        Stream equality uses host metadata, never the raw launch-stream handle.
        Rebuilding only creates native pointer tables; expert storage is shared.
        Native construction validates orders and transfers tables, so cache
        misses must be resolved before capture, on the actual capture stream.
        """
        if self.device.type != "npu":
            return self.banks
        current = torch.npu.current_stream(self.device)
        for stream, banks in self._stream_banks:
            if stream == current:
                return banks
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("Warm VQ2A8 banks on the capture stream before graph capture.")
        current.wait_stream(self.owner_stream)
        bank_type = get_vq2a8_bank_type()
        banks = {
            kind: bank_type(*[[payload[field] for payload in entries] for field in VQ2A8_FIELDS])
            for kind, entries in self.payloads.items()
        }
        self._stream_banks.append((current, banks))
        return banks

    def _check_execution(self, *, device_decode=False):
        if self.device.type == "npu":
            if torch.npu.is_current_stream_capturing():
                if not device_decode:
                    raise RuntimeError("VQ2A8 grouped experts currently support eager execution only.")
                if not self.defer_validity:
                    raise RuntimeError("VQ2A8 capture requires a forward owner for deferred validity checks.")
        return self.prepare_for_current_stream()

    def _record_event(self):
        if self.device.type == "npu":
            # Lazy import avoids the comm/MLP initialization cycle. Reuse the
            # base's event policy when shared-expert overlap is disabled.
            from vllm_ascend.ops.fused_moe.moe_mlp import maybe_record_event

            return maybe_record_event()
        return self.owner_stream.record_event() if self.owner_stream is not None else None

    def _accept_validity(self, valid):
        self.validity_buffer.copy_(valid)
        if not self.defer_validity and not bool(self.validity_buffer):
            raise ValueError("VQ2A8 expert preparation or projection failed validity checks.")

    def compute_device(self, hidden_states, expert_ids, swiglu_limit):
        if hidden_states.shape[0] != expert_ids.numel() or not expert_ids.numel():
            raise ValueError("VQ2A8 device route rows and expert IDs must match and be nonempty.")
        outputs, flags, event = [], [], None
        for start in range(0, expert_ids.numel(), MAX_PROJECTION_JOBS):
            output, event, valid = self._compute_device_chunk(
                hidden_states[start : start + MAX_PROJECTION_JOBS],
                expert_ids[start : start + MAX_PROJECTION_JOBS],
                swiglu_limit,
            )
            outputs.append(output)
            flags.append(valid)
        return torch.cat(outputs), event, torch.stack(flags).all()

    def _compute_device_chunk(self, hidden_states, expert_ids, swiglu_limit):
        """Compute fixed B1 route rows, returning a fresh device validity scalar.

        No expert ID/count values are copied to the host. Output rows retain
        the selected-route order for the common token-unpermute operation.
        The caller must consume the validity scalar before accepting tokens.
        """
        banks = self._check_execution(device_decode=True)
        if (
            self.decode_mode != "device"
            or hidden_states.ndim != 2
            or hidden_states.dtype != torch.bfloat16
            or hidden_states.device != self.device
            or hidden_states.shape[1] != self.specs["gate_up"].rht_true_columns
            or expert_ids.ndim != 1
            or expert_ids.dtype not in (torch.int32, torch.int64)
            or expert_ids.device != self.device
            or not 1 <= expert_ids.numel() <= MAX_PROJECTION_JOBS
            or hidden_states.shape[0] != expert_ids.numel()
        ):
            raise ValueError("VQ2A8 device decode requires BF16 route rows and 1..6 same-device integer expert IDs.")
        ids = expert_ids.to(torch.int64)
        in_range = (ids >= 0) & (ids < self.num_experts)
        mapped = self._lookup.index_select(0, ids.clamp(0, self.num_experts - 1))
        slots = torch.where(in_range, mapped, -1).contiguous()
        valid_flags = [(slots >= 0).all()]
        raw_statuses = [] if self._projection_validity is not None else None
        projection_outputs = []
        values = hidden_states
        before_gmm2_evt = None
        for kind in ("gate_up", "down"):
            bank = banks[kind]
            scale, bias, signs, selected_valid = bank.select(slots)
            if raw_statuses is None:
                valid_flags.append((selected_valid != 0).all())
                quantized, row_scale, row_bias = self._packed_preparations[kind].packed(
                    values, scale, bias, signs, self.specs[kind], validity=valid_flags.append
                )
            else:
                raw_statuses.append(selected_valid)
                quantized, row_scale, row_bias = self._packed_preparations[kind].packed(
                    values, scale, bias, signs, self.specs[kind], raw_input_validity=raw_statuses.append
                )
            values, projected_valid = bank.project_vectorized(quantized, row_scale, row_bias, slots)
            if raw_statuses is None:
                valid_flags.append((projected_valid != 0).all() & torch.isfinite(values).all())
            else:
                raw_statuses.append(projected_valid)
                projection_outputs.append(values)
            if kind == "gate_up":
                values = deepseek_v4_swiglu_reference(values, swiglu_limit)
                before_gmm2_evt = self._record_event()
        if raw_statuses is not None:
            # Preserve exactly the existing two projection finite scans and
            # six native status checks; combine remains owned by FusedMoE.
            valid = self._projection_validity(raw_statuses, projection_outputs, valid_flags[0])
        else:
            valid = torch.stack(valid_flags).all()
        return values, before_gmm2_evt, valid

    def apply_device(self, hidden_states, expert_ids, swiglu_limit):
        output, event, valid = self.compute_device(hidden_states, expert_ids, swiglu_limit)
        self._accept_validity(valid)
        return output, event

    def _project(self, kind, expert_ids, rows, valid_flags, banks):
        spec = self.specs[kind]
        stored_ids = [self._expert_slots[expert] for expert in expert_ids]
        prepared = self._preparation.many(
            [(row, self.payloads[kind][expert], spec) for expert, row in zip(stored_ids, rows)],
            validity=valid_flags.append,
        )
        counts = [row.shape[0] for row in rows]
        quantized, scale, bias = pack_vq2a8_projection_inputs(prepared)
        slots = torch.cat([self._slots[expert : expert + 1] for expert in stored_ids])
        output, valid = banks[kind].project_vectorized(quantized, scale, bias, slots)
        valid_flags.append((valid != 0).all() & torch.isfinite(output).all())
        return [output[index, :count] for index, count in enumerate(counts)]

    def _project_layer(self, kind, stored_ids, rows, plan, slots, valid_flags, banks):
        # Preparation chunks are bounded by actual rows independently of
        # native six-job batches. Never split a job just to fill a chunk.
        prepared, requests, row_count = [], [], 0
        for expert, row in zip(stored_ids, rows):
            if requests and row_count + row.shape[0] > MAX_PREFILL_PREPARATION_ROWS:
                prepared.extend(self._preparation.many_prefill(requests, validity=valid_flags.append))
                requests, row_count = [], 0
            requests.append((row, self.payloads[kind][expert], self.specs[kind]))
            row_count += row.shape[0]
        if requests:
            prepared.extend(self._preparation.many_prefill(requests, validity=valid_flags.append))
        outputs, statuses = [], []
        for fields, selected in zip(plan.pack(prepared), slots):
            output, valid = banks[kind].project_vectorized(*fields, selected)
            outputs.append(output.flatten(0, 1))
            statuses.append(valid)
        padded = torch.cat(outputs)
        # Preserve the old check on ALL native rows, including padding. This
        # reduces launch count without accepting malformed padding outputs.
        valid_flags.append((torch.cat(statuses) != 0).all() & torch.isfinite(padded).all())
        return plan.unpack(padded)

    def _apply_layer(self, hidden_states, jobs, banks, swiglu_limit):
        counts = [count for _, _, count in jobs]
        plan = VQ2A8ProjectionBatchPlan(counts, device=self.device)
        stored_ids = [self._expert_slots[expert] for expert, _, _ in jobs]
        slots = torch.cat([self._slots[expert : expert + 1] for expert in stored_ids]).split(MAX_PROJECTION_JOBS)
        rows = [hidden_states[start : start + count] for _, start, count in jobs]
        valid_flags = []
        gate_up = self._project_layer("gate_up", stored_ids, rows, plan, slots, valid_flags, banks)
        activated = deepseek_v4_swiglu_grouped(gate_up, swiglu_limit)
        before_gmm2_evt = self._record_event()
        values = self._project_layer("down", stored_ids, activated, plan, slots, valid_flags, banks)
        self._accept_validity(torch.stack(valid_flags).all())
        self.prefill_layer_calls += 1
        return torch.cat(values), before_gmm2_evt

    def apply(
        self, hidden_states: torch.Tensor, group_list: torch.Tensor, group_list_type: int, swiglu_limit: float | None
    ) -> tuple[torch.Tensor, Any]:
        banks = self._check_execution()
        if (
            hidden_states.ndim != 2
            or hidden_states.dtype != torch.bfloat16
            or hidden_states.device != self.device
            or hidden_states.shape[1] != self.specs["gate_up"].rht_true_columns
        ):
            raise ValueError("VQ2A8 experts require dispatched BF16 [tokens,hidden] on the weight device.")
        if (
            group_list_type not in (0, 1)
            or group_list.ndim != 1
            or group_list.dtype not in (torch.int32, torch.int64)
            or group_list.numel() != self.num_experts
        ):
            raise ValueError("VQ2A8 group_list must contain one integer count or prefix sum per local expert.")
        # One explicit NPU-to-host synchronization; graph/EP support is not implied.
        counts = group_list.cpu().tolist()
        if group_list_type == 0:
            counts = [value - previous for value, previous in zip(counts, [0, *counts[:-1]])]
        if any(count < 0 for count in counts) or sum(counts) != hidden_states.shape[0]:
            raise ValueError("VQ2A8 expert counts must be nonnegative and cover every dispatched token.")
        if any(count and expert not in self._expert_slots for expert, count in enumerate(counts)):
            raise ValueError("VQ2A8 routing selected an expert absent from the stored artifact.")
        if not hidden_states.shape[0]:
            output = hidden_states.new_empty((0, self.specs["down"].rows))
            self._accept_validity(torch.ones((), dtype=torch.bool, device=self.device))
            return output, self._record_event()
        jobs, start = [], 0
        for expert, count in enumerate(counts):
            for offset in range(0, count, MAX_PROJECTION_ROWS):
                jobs.append((expert, start + offset, min(MAX_PROJECTION_ROWS, count - offset)))
            start += count
        if self.prefill_batch_mode == "layer":
            return self._apply_layer(hidden_states, jobs, banks, swiglu_limit)
        output = hidden_states.new_empty((hidden_states.shape[0], self.specs["down"].rows))
        valid_flags = []
        before_gmm2_evt = None
        for index in range(0, len(jobs), MAX_PROJECTION_JOBS):
            batch = jobs[index : index + MAX_PROJECTION_JOBS]
            expert_ids = [expert for expert, _, _ in batch]
            rows = [hidden_states[start : start + count] for _, start, count in batch]
            gate_up = self._project("gate_up", expert_ids, rows, valid_flags, banks)
            activated = deepseek_v4_swiglu_grouped(gate_up, swiglu_limit)
            before_gmm2_evt = self._record_event()
            values = self._project("down", expert_ids, activated, valid_flags, banks)
            for (_, start, count), value in zip(batch, values):
                output[start : start + count].copy_(value)
        self._accept_validity(torch.stack(valid_flags).all())
        return output, before_gmm2_evt
