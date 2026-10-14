# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VQ2A8 activation preparation, Hadamard constants and SwiGLU arithmetic.

Default prefill and packed decode retain independent one-row RHT and bias GEMVs.
Prefill can opt into batched RHT or subgraphs of the rowwise preparation.
Packed decode can fuse sign multiplication and input validation; FP8 quantization remains in PyTorch.
"""

from __future__ import annotations

import functools
import logging
import math

import torch

VQ2_FP8_MIN_SCALE = 1e-12
PACKED_GROUP_LIMIT = 6
PACKED_WIDTHS = (2048, 4096)
SIGN_FUSION_ABI = 1
STRIDED_SIGN_ABI = 1
ACTIVATION_ALIGNMENT = 32
PREFILL_GRAPH_ROWS = (6, 12, 24, 48, 96, 192)
PREFILL_GRAPH_WIDTHS = (2048, 4096)
PREFILL_GRAPH_BLOCK = 128
GRAPH_WARMUPS = 3
MAX_PREFILL_PREPARATION_ROWS = 192
MAX_PREFILL_JOB_ROWS = 32


def deepseek_v4_swiglu_reference(gate_up: torch.Tensor, swiglu_limit: float | None) -> torch.Tensor:
    """Apply the DeepSeek V4 routed-expert SwiGLU contract.

    The clamp order mirrors ``SiluAndMulWithClamp``: clamp the gate only on
    its positive side, clamp the up branch symmetrically, then evaluate SiLU
    and multiply.  Operations intentionally retain the input dtype so an NPU
    eager run exercises the same BF16 boundaries as the model path.
    """
    if not isinstance(gate_up, torch.Tensor):
        raise TypeError(f"gate_up must be a torch.Tensor, got {type(gate_up).__name__}.")
    if gate_up.ndim < 1 or gate_up.shape[-1] <= 0 or gate_up.shape[-1] % 2:
        raise ValueError(f"gate_up last dimension must be positive and even, got shape={tuple(gate_up.shape)}.")
    if swiglu_limit is not None and (
        isinstance(swiglu_limit, bool)
        or not isinstance(swiglu_limit, int | float)
        or (not math.isfinite(swiglu_limit))
        or (swiglu_limit < 0)
    ):
        raise ValueError(f"swiglu_limit must be non-negative or None, got {swiglu_limit!r}.")
    gate, up = gate_up.chunk(2, dim=-1)
    if swiglu_limit is not None and swiglu_limit > 0:
        gate = torch.clamp(gate, max=swiglu_limit)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
    return torch.nn.functional.silu(gate) * up


def deepseek_v4_swiglu_grouped(gate_up: list[torch.Tensor], swiglu_limit: float | None) -> tuple[torch.Tensor, ...]:
    """Share pointwise launches across the valid rows of one expert job batch.

    Concatenation preserves expert/row order and BF16 values. The clamp, SiLU
    and multiply still use the reference dtype and operation boundaries.
    """
    counts = [value.shape[0] for value in gate_up]
    combined = gate_up[0] if len(gate_up) == 1 else torch.cat(gate_up)
    return deepseek_v4_swiglu_reference(combined, swiglu_limit).split(counts)


def pack_vq2a8_projection_inputs(prepared):
    """Pack 1..6 prepared projections, padding only genuinely shorter jobs.

    RHT and bias GEMVs have already completed independently for each row.
    Padding operates on FP8 bytes and row metadata, with zero-filled tails.
    Even zero-length F.pad calls can allocate and launch work on the NPU.
    """
    counts = [value[0].shape[0] for value in prepared]
    max_rows = max(counts)
    quantized = torch.stack(
        [
            torch.nn.functional.pad(value[0].view(torch.uint8), (0, 0, 0, max_rows - count))
            if count < max_rows
            else value[0].view(torch.uint8)
            for value, count in zip(prepared, counts)
        ]
    ).view(torch.float8_e4m3fn)
    scale, bias = (
        torch.stack(
            [
                torch.nn.functional.pad(value[index], (0, max_rows - count)) if count < max_rows else value[index]
                for value, count in zip(prepared, counts)
            ]
        )
        for index in (1, 2)
    )
    return quantized, scale, bias


class VQ2A8ProjectionBatchPlan:
    """One invocation's row layout, reused by gate/up and down projections.

    Each native batch retains its original six-job boundary and maximum M.
    Padding gathers an explicit zero sentinel, including scale/bias zeroes.
    The plan contains row indices only, never expert weights or activations.
    """

    def __init__(self, counts, *, device):
        self.counts = tuple(counts)
        if not self.counts or any(type(count) is not int or not 1 <= count <= MAX_PREFILL_JOB_ROWS for count in counts):
            raise ValueError("Projection packing requires nonempty jobs of 1..32 rows.")
        sentinel = sum(counts)
        gather, valid_rows, shapes = [], [], []
        start = 0
        for first in range(0, len(counts), PACKED_GROUP_LIMIT):
            batch = counts[first : first + PACKED_GROUP_LIMIT]
            maximum = max(batch)
            shapes.append((len(batch), maximum))
            for count in batch:
                valid_rows.extend(range(len(gather), len(gather) + count))
                gather.extend(range(start, start + count))
                gather.extend([sentinel] * (maximum - count))
                start += count
        self.shapes = tuple(shapes)
        self.padded_rows = tuple(groups * rows for groups, rows in shapes)
        self.gather = torch.tensor(gather, dtype=torch.int64, device="cpu").to(device)
        self.valid_rows = torch.tensor(valid_rows, dtype=torch.int64, device="cpu").to(device)

    def pack(self, prepared):
        if tuple(value[0].shape[0] for value in prepared) != self.counts:
            raise ValueError("Prepared rows do not match this projection packing plan.")
        fields = []
        for field in range(3):
            values = [value[field].view(torch.uint8) if field == 0 else value[field] for value in prepared]
            zero = values[0].new_zeros((1, *values[0].shape[1:]))
            packed = torch.cat([*values, zero]).index_select(0, self.gather)
            if field == 0:
                packed = packed.view(torch.float8_e4m3fn)
            fields.append(
                [
                    value.view(*shape, *packed.shape[1:])
                    for value, shape in zip(packed.split(self.padded_rows), self.shapes)
                ]
            )
        return list(zip(*fields))

    def unpack(self, padded):
        """Gather valid output rows after the caller checks all padding too."""
        return padded.index_select(0, self.valid_rows).split(self.counts)


@functools.cache
def _sylvester_hadamard(size: int) -> torch.Tensor:
    if size <= 0 or size & size - 1:
        raise ValueError(f"Hadamard size must be a power of two, got {size}.")
    matrix = torch.tensor([[1.0]], dtype=torch.float64)
    while matrix.shape[0] < size:
        matrix = torch.cat((torch.cat((matrix, matrix), dim=1), torch.cat((matrix, -matrix), dim=1)), dim=0)
    return matrix / math.sqrt(size)


def _prepare_rowwise(signed, weight_scale, weight_bias, hadamard, min_scale):
    width = weight_scale.shape[1]
    rotated = [(row @ hadamard).reshape(1, width) for row in signed.split(1)]
    bias = torch.cat([row @ weight_bias[index] for index, row in enumerate(rotated)])
    transformed = torch.cat(rotated) * weight_scale
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = torch.clamp(transformed.abs().amax(dim=-1) / fp8_max, min=min_scale)
    quantized = torch.clamp(transformed / scale.unsqueeze(-1), -fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return quantized, scale, bias


class _PrefillGraphEntry:
    def __init__(self, rows, width, hadamard, min_scale, owner_stream):
        device = hadamard.device
        self.inputs = (
            torch.zeros(rows, width // PREFILL_GRAPH_BLOCK, PREFILL_GRAPH_BLOCK, dtype=torch.float32, device=device),
            torch.ones(rows, width, dtype=torch.float32, device=device),
            torch.zeros(rows, width, dtype=torch.float32, device=device),
        )
        self.hadamard = hadamard.clone()
        # Capture once on a side stream. Only a cache miss synchronizes; replay
        # and all refresh/output copies are ordered on the fixed owner stream.
        self.capture_stream = torch.npu.Stream(device=device)
        self.capture_stream.wait_stream(owner_stream)
        with torch.npu.stream(self.capture_stream):
            for _ in range(GRAPH_WARMUPS):
                _prepare_rowwise(*self.inputs, self.hadamard, min_scale)
        self.capture_stream.synchronize()
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph, stream=self.capture_stream):
            self.outputs = _prepare_rowwise(*self.inputs, self.hadamard, min_scale)
        owner_stream.wait_stream(self.capture_stream)

    def replay(self, signed, weight_scale, weight_bias):
        rows = signed.shape[0]
        for target, source in zip(self.inputs, (signed, weight_scale, weight_bias)):
            target[:rows].copy_(source)
        # Every active row is overwritten, including expert metadata. Inactive
        # tails are never reduced into active rows and are discarded below.
        self.graph.replay()
        quantized, scale, bias = self.outputs
        # Preserve the eager API's ownership: a later replay must not overwrite
        # results still referenced by another layer or queued native work.
        return (
            quantized[:rows].view(torch.uint8).clone().view(torch.float8_e4m3fn),
            scale[:rows].clone(),
            bias[:rows].clone(),
        )


class PrefillRowwiseGraphCache:
    """Share at most twelve graphs across one model's serial expert stages.

    The config owns this instance and explicitly passes it to each expert layer.
    No process-global state, expert IDs or expert weight tensors are cached.
    Cross-stream reuse and nested capture are rejected instead of racing buffers.
    """

    def __init__(self, *, min_scale):
        self.min_scale = min_scale
        self._entries = {}
        self._device = None
        self._owner_stream = None
        self.replays = 0

    @property
    def captures(self):
        return len(self._entries)

    def run(self, signed, weight_scale, weight_bias, hadamard):
        if signed.device.type != "npu":
            raise ValueError("VQ2A8 rowwise_graph preparation requires an NPU.")
        return self._run_on_stream(signed, weight_scale, weight_bias, hadamard, torch.npu.current_stream(signed.device))

    def _run_on_stream(self, signed, weight_scale, weight_bias, hadamard, stream):
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("VQ2A8 prefill subgraphs cannot run inside another graph capture.")
        if signed.ndim != 3 or weight_scale.ndim != 2:
            raise ValueError("Invalid VQ2A8 prefill graph input ranks.")
        rows, width = weight_scale.shape
        if (
            not 1 <= rows <= PREFILL_GRAPH_ROWS[-1]
            or width not in PREFILL_GRAPH_WIDTHS
            or signed.shape != (rows, width // PREFILL_GRAPH_BLOCK, PREFILL_GRAPH_BLOCK)
            or weight_bias.shape != weight_scale.shape
            or hadamard.shape != (PREFILL_GRAPH_BLOCK, PREFILL_GRAPH_BLOCK)
            or any(
                value.dtype != torch.float32 or value.device != signed.device
                for value in (signed, weight_scale, weight_bias, hadamard)
            )
            or any(value.requires_grad for value in (signed, weight_scale, weight_bias, hadamard))
        ):
            raise ValueError("Prefill graphs require FP32 K2048/4096, block128, 1..192 rows without gradients.")
        if self._owner_stream is not None and (self._device != signed.device or self._owner_stream != stream):
            raise RuntimeError("VQ2A8 prefill graph cache requires its original device and stream.")
        bucket = next(count for count in PREFILL_GRAPH_ROWS if rows <= count)
        key = (width, bucket)
        entry = self._entries.get(key)
        if entry is None:
            # Only publish a successfully captured entry; failures propagate.
            entry = _PrefillGraphEntry(bucket, width, hadamard, self.min_scale, stream)
            self._device, self._owner_stream = signed.device, stream
            self._entries[key] = entry
            logging.getLogger(__name__).info("VQ2A8_PREFILL_GRAPH_CAPTURE width=%d rows=%d", width, bucket)
        result = entry.replay(signed, weight_scale, weight_bias)
        self.replays += 1
        return result


class RowwiseVQ2A8Preparation:
    """Prepare expert activations with a bounded cache of Hadamard constants."""

    def __init__(self, *, compact=False, validity=None, rht_mode="rowwise", graph_cache=None):
        if rht_mode not in ("rowwise", "batched", "rowwise_graph"):
            raise ValueError("VQ2A8 RHT mode must be rowwise, batched or rowwise_graph.")
        self.rht_mode = rht_mode
        self.graph_cache = graph_cache
        if rht_mode == "rowwise_graph" and graph_cache is None:
            self.graph_cache = PrefillRowwiseGraphCache(min_scale=VQ2_FP8_MIN_SCALE)
        self._key = None
        self._hadamard = None
        self._graph_key = None
        self.compact = compact
        # Grouped expert execution may combine validity checks before returning.
        # Still scan every input: no cached or skipped validity decisions.
        self.validity = validity

    def _validate(self, valid):
        if self.validity is not None:
            self.validity(valid)
        elif not bool(valid):
            raise ValueError("Invalid activation/weight_scale/weight_bias (non-finite) or rht_sign (not -1/+1).")

    def rows(self, hidden, payload, spec):
        if self.compact or self.rht_mode != "rowwise":
            return self.many([(hidden, payload, spec)])[0]
        prepared = []
        for row in hidden.split(1):
            if spec.columns != spec.rht_true_columns:
                row = torch.nn.functional.pad(row, (0, spec.columns - spec.rht_true_columns))
            with torch.device("cpu"):
                prepared.append(
                    self(row, payload["weight_scale"], payload["weight_bias"], payload["rht_sign"], spec.rht_block_size)
                )
        if not prepared:
            raise ValueError("AscendC preparation requires at least one row.")
        if len(prepared) == 1:
            return tuple(value.contiguous() for value in prepared[0])
        return tuple(torch.cat(values, dim=0).contiguous() for values in zip(*prepared))

    def many(self, requests, *, validity=None):
        """Batch pointwise work, optionally batch RHT, and keep bias GEMV one-row.

        All inputs are revalidated on every call, in one collective decision.
        There is no cached sign/scale/bias validity or transformed weight copy.
        At most six projections of <=32 rows share a same-width workspace.
        """
        return self._many(requests, validity=validity, max_jobs=PACKED_GROUP_LIMIT)

    def many_prefill(self, requests, *, validity=None):
        """Prepare up to 192 actual rows across an entire prefill stage."""
        return self._many(requests, validity=validity, max_jobs=MAX_PREFILL_PREPARATION_ROWS)

    def _many(self, requests, *, validity, max_jobs):
        if not 1 <= len(requests) <= max_jobs:
            raise ValueError(f"Grouped preparation requires 1..{max_jobs} projections.")
        if any(hidden.ndim != 2 for hidden, _, _ in requests):
            raise ValueError("Grouped preparation requires rank-two activation rows.")
        if sum(hidden.shape[0] for hidden, _, _ in requests) > MAX_PREFILL_PREPARATION_ROWS:
            raise ValueError("Grouped preparation requires at most 192 actual rows.")
        first, _, spec = requests[0]
        width, block = spec.columns, spec.rht_block_size
        values, counts = [], []
        for hidden, payload, current in requests:
            if (
                (current.columns, current.rht_true_columns, current.rht_block_size)
                != (width, spec.rht_true_columns, block)
                or hidden.ndim != 2
                or not 1 <= hidden.shape[0] <= 32
                or hidden.shape[1] != current.rht_true_columns
                or hidden.device != first.device
                or not 0 < current.rht_true_columns <= width
            ):
                raise ValueError("Grouped preparation requires matching geometry/device and 1..32 rows per projection.")
            counts.append(hidden.shape[0])
            for row in hidden.split(1):
                if width != current.rht_true_columns:
                    row = torch.nn.functional.pad(row, (0, width - current.rht_true_columns))
                self._check_metadata(row, payload["weight_scale"], payload["weight_bias"], payload["rht_sign"], block)
                values.append((row, payload["weight_scale"], payload["weight_bias"], payload["rht_sign"]))
        # One conversion instead of one allocation/launch per row. Mixed input
        # dtypes retain the old conversion-before-concatenation contract.
        compact = self.compact and len({value[0].dtype for value in values}) == 1
        x = torch.cat([value[0] if compact else value[0].float() for value in values]).float()
        weight_scale, weight_bias, rht_sign = (torch.stack([value[i] for value in values]) for i in (1, 2, 3))
        valid = torch.isfinite(x).all() & torch.isfinite(weight_scale).all() & torch.isfinite(weight_bias).all()
        # Equality with -1/+1 is exact in int8; no widening buffer is needed.
        signs = rht_sign if self.compact else rht_sign.to(torch.int16)
        valid = valid & ((signs == -1) | (signs == 1)).all()
        if validity is None:
            self._validate(valid)
        else:
            validity(valid)
        self._ensure_hadamard(first.device, block)
        signed = x.reshape(-1, width // block, block) * rht_sign.float().reshape(-1, width // block, block)
        if self.rht_mode == "rowwise_graph":
            quantized, scale, bias = self.graph_cache.run(signed, weight_scale, weight_bias, self._hadamard)
            return list(zip(quantized.split(counts), scale.split(counts), bias.split(counts)))
        # Batching only RHT passed the 950PR probe. Keep it opt-in until full
        # model acceptance; GEMM rounding can depend on backend and geometry.
        if self.rht_mode == "batched":
            rotated = list((signed @ self._hadamard).reshape(-1, width).split(1))
        else:
            rotated = [(row @ self._hadamard).reshape(1, width) for row in signed.split(1)]
        # Do not batch the bias GEMV: the 950PR probe detected rounding changes.
        bias = torch.cat([row @ values[i][2] for i, row in enumerate(rotated)])
        transformed = torch.cat(rotated) * weight_scale
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = torch.clamp(transformed.abs().amax(dim=-1) / fp8_max, min=VQ2_FP8_MIN_SCALE)
        quantized = torch.clamp(transformed / scale.unsqueeze(-1), -fp8_max, fp8_max).to(torch.float8_e4m3fn)
        return list(zip(quantized.split(counts), scale.split(counts), bias.split(counts)))

    def __call__(self, activation, weight_scale, weight_bias, rht_sign, rht_block_size):
        self._check_metadata(activation, weight_scale, weight_bias, rht_sign, rht_block_size)
        width = activation.shape[1]
        x = activation.float()
        valid = torch.isfinite(x).all() & torch.isfinite(weight_scale).all() & torch.isfinite(weight_bias).all()
        signs = rht_sign.to(torch.int16)
        valid = valid & ((signs == -1) | (signs == 1)).all()
        self._validate(valid)

        self._ensure_hadamard(activation.device, rht_block_size)
        blocks = x.reshape(1, width // rht_block_size, rht_block_size)
        blocks = blocks * rht_sign.float().reshape(width // rht_block_size, rht_block_size)
        rotated = (blocks @ self._hadamard).reshape(1, width)
        bias_correction = rotated @ weight_bias.float()
        transformed = rotated * weight_scale.float().unsqueeze(0)
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = torch.clamp(transformed.abs().amax(dim=-1) / fp8_max, min=VQ2_FP8_MIN_SCALE)
        quantized = torch.clamp(transformed / scale.unsqueeze(-1), -fp8_max, fp8_max).to(torch.float8_e4m3fn)
        return quantized, scale, bias_correction

    @staticmethod
    def _check_metadata(activation, weight_scale, weight_bias, rht_sign, rht_block_size):
        if activation.ndim != 2 or activation.shape[0] != 1 or activation.shape[1] <= 0:
            raise ValueError("AscendC preparation requires exactly one nonempty activation row.")
        width = activation.shape[1]
        if (
            type(rht_block_size) is not int
            or rht_block_size <= 0
            or rht_block_size & (rht_block_size - 1)
            or width % rht_block_size
        ):
            raise ValueError("rht_block_size must be a positive power of two dividing the activation width.")
        for name, tensor, dtype in (
            ("weight_scale", weight_scale, torch.float32),
            ("weight_bias", weight_bias, torch.float32),
            ("rht_sign", rht_sign, torch.int8),
        ):
            if tensor.shape != (width,) or tensor.dtype != dtype or tensor.device != activation.device:
                raise ValueError(f"{name} must be {dtype}[{width}] on {activation.device}.")

    def prepare_for_graph(self, device, rht_block_size):
        """Initialize and freeze one projection's constant before capture.

        Use a separate preparation object per projection geometry. This does
        not cache validity or alter the original row-wise RHT arithmetic.
        """
        if type(rht_block_size) is not int or rht_block_size <= 0 or rht_block_size & (rht_block_size - 1):
            raise ValueError("Graph preparation requires a positive power-of-two RHT block.")
        self._ensure_hadamard(torch.device(device), rht_block_size)
        self._graph_key = self._key

    def _ensure_hadamard(self, device, rht_block_size):
        key = (device, rht_block_size)
        if self._graph_key is not None and key != self._graph_key:
            raise RuntimeError("Graph preparation geometry changed; constants cannot initialize during capture/replay.")
        if self._key != key:
            # Never generate the FP64 Sylvester matrix on the default NPU.
            with torch.device("cpu"):
                hadamard = _sylvester_hadamard(rht_block_size)
            self._hadamard = hadamard.to(device=device, dtype=torch.float32)
            self._key = key


class PackedRowwiseVQ2A8Preparation(RowwiseVQ2A8Preparation):
    """Prepare one activation row per selected expert without Python repacking."""

    def __init__(
        self, *, compact=False, validity=None, fuse_sign=False, native_ops=None, strided_sign=False, direct_output=False
    ):
        super().__init__(compact=compact, validity=validity)
        if (strided_sign and not fuse_sign) or (direct_output and not strided_sign):
            raise ValueError("Direct output requires strided sign fusion; strided sign requires fuse_sign.")
        self.fuse_sign = fuse_sign
        self.strided_sign = strided_sign
        self.direct_output = direct_output
        self._sign = None
        if not fuse_sign:
            return
        native = native_ops if native_ops is not None else torch.ops.vq2a8_ascendc_v4_v2
        try:
            version = native.activation_preparation_version()
            sign = native.activation_sign
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError(
                "Packed sign fusion requires V4/v2 activation_sign ABI 1; no unfused fallback is selected implicitly."
            ) from error
        if type(version) is not int or version != SIGN_FUSION_ABI:
            raise RuntimeError(f"Unsupported packed sign fusion ABI {version}; require {SIGN_FUSION_ABI}.")
        if strided_sign:
            try:
                version = native.activation_sign_strided_version()
                sign = native.activation_sign_strided
            except (AttributeError, RuntimeError) as error:
                raise RuntimeError("Strided sign fusion requires a rebuilt library; no implicit fallback.") from error
            if type(version) is not int or version != STRIDED_SIGN_ABI:
                raise RuntimeError(f"Unsupported strided sign ABI {version}; require {STRIDED_SIGN_ABI}.")
        self._sign = sign

    def packed(self, hidden, weight_scale, weight_bias, signs, spec, *, validity=None, raw_input_validity=None):
        """Return contiguous ``(FP8 rows, row scales, row biases)``.

        ``hidden`` and every metadata tensor contain one row for each selected
        expert.  Only the bounded M=1 decode geometry is accepted.  General
        prefill and padded widths continue through the inherited ``many`` path.
        """

        groups, width, block = self._check_packed(hidden, weight_scale, weight_bias, signs, spec)
        if raw_input_validity is not None and validity is not None:
            raise ValueError("Raw input validity and reduced validity consumers are mutually exclusive.")
        if raw_input_validity is not None and (not self.fuse_sign or not callable(raw_input_validity)):
            raise ValueError("Raw input validity requires native sign fusion and a callable consumer.")
        # Preserve the original FP32 materialization unless native strided
        # reads were explicitly selected, including expanded BF16 gate/up rows.
        if self.strided_sign:
            self._check_strided_hidden(hidden)
            x = hidden
        else:
            x = hidden.to(dtype=torch.float32).contiguous()
        self._ensure_hadamard(hidden.device, block)

        if self.fuse_sign:
            signed, input_valid = self._sign(x, weight_scale, weight_bias, signs)
            if raw_input_validity is not None:
                raw_input_validity(input_valid)
            else:
                valid = input_valid.all()
        else:
            valid = torch.isfinite(x).all() & torch.isfinite(weight_scale).all()
            valid = valid & torch.isfinite(weight_bias).all()
            valid = valid & ((signs == -1) | (signs == 1)).all()
            signed = x * signs.float()
        if raw_input_validity is None:
            if validity is None:
                self._validate(valid)
            else:
                validity(valid)

        signed_blocks = signed.reshape(groups, width // block, block)
        rotated = torch.empty((groups, width), dtype=torch.float32, device=hidden.device)
        bias = torch.empty(groups, dtype=torch.float32, device=hidden.device)
        for group_index in range(groups):
            # Keep these as independent one-row GEMMs.  A batched GEMM is not
            # byte-equivalent to RowwiseVQ2A8Preparation on Ascend 950.
            if self.direct_output:
                # Keep identical input ranks and GEMM geometry; only the output
                # destinations differ. NPU byte equivalence needs device tests.
                row = rotated[group_index : group_index + 1]
                torch.matmul(
                    signed_blocks[group_index : group_index + 1],
                    self._hadamard,
                    out=row.reshape(1, width // block, block),
                )
                torch.matmul(row, weight_bias[group_index], out=bias[group_index : group_index + 1])
            else:
                row = (signed_blocks[group_index : group_index + 1] @ self._hadamard).reshape(1, width)
                rotated[group_index : group_index + 1].copy_(row)
                row_bias = row @ weight_bias[group_index]
                bias[group_index : group_index + 1].copy_(row_bias)

        transformed = rotated * weight_scale
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = torch.clamp(transformed.abs().amax(dim=-1) / fp8_max, min=VQ2_FP8_MIN_SCALE)
        quantized = torch.clamp(transformed / scale.unsqueeze(-1), -fp8_max, fp8_max).to(torch.float8_e4m3fn)
        return quantized.contiguous(), scale.contiguous(), bias.contiguous()

    @staticmethod
    def _check_strided_hidden(hidden):
        if hidden.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("Strided sign input requires BF16 or FP32.")
        row_stride, column_stride = hidden.stride()
        if column_stride != 1 or (row_stride != 0 and row_stride < hidden.shape[1]):
            raise ValueError("Strided sign input requires unit columns and expanded or non-overlapping rows.")
        if hidden.data_ptr() % ACTIVATION_ALIGNMENT or (
            hidden.shape[0] > 1 and (row_stride * hidden.element_size()) % ACTIVATION_ALIGNMENT
        ):
            raise ValueError("Strided sign input base and row stride require 32-byte alignment.")

    @staticmethod
    def _check_packed(hidden, weight_scale, weight_bias, signs, spec):
        if hidden.ndim != 2 or not 1 <= hidden.shape[0] <= PACKED_GROUP_LIMIT:
            raise ValueError("Packed preparation requires 1..6 activation rows, one per selected expert.")
        groups, width = hidden.shape
        if width not in PACKED_WIDTHS:
            raise ValueError("Packed preparation supports activation width 2048 or 4096 only.")
        try:
            geometry = (spec.columns, spec.rht_true_columns, spec.rht_block_size)
        except AttributeError as error:
            raise ValueError("Packed preparation requires columns, rht_true_columns, and rht_block_size.") from error
        block = geometry[2]
        if geometry[:2] != (width, width):
            raise ValueError("Packed preparation does not support padded or mismatched activation geometry.")
        if type(block) is not int or block <= 0 or block & (block - 1) or width % block:
            raise ValueError("rht_block_size must be a positive power of two dividing the activation width.")
        for name, tensor, dtype in (
            ("weight_scale", weight_scale, torch.float32),
            ("weight_bias", weight_bias, torch.float32),
            ("signs", signs, torch.int8),
        ):
            if (
                tensor.shape != (groups, width)
                or tensor.dtype != dtype
                or tensor.device != hidden.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"{name} must be contiguous {dtype}[{groups},{width}] on {hidden.device}.")
        return groups, width, block
