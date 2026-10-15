# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager-only GS32 native projections with a single bounded expert cache.

Two launches per expert FFN, no dense weight fallback and no changes to the
imported mixed kernel. Cold expert I/O, repacking, transfers and finite checks
are intentional bring-up costs. This is not a latency baseline.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch

from .vq2a8_gs32_layout import repack_gs32_matrix
from .vq2a8_gs32_reference import quantize_mxfp8_k32, require_finite, staged_swiglu

MAX_STAGE_TOKENS = 128


class _NativeStageOps:
    def __init__(self, library, binary, device):
        # Extension/runtime registration happens once during worker startup,
        # never within a queued per-projection launch callback.
        import torch_npu  # noqa: F401

        self.device = torch.device(device)
        if self.device.type != "npu" or self.device.index is None:
            raise ValueError("GS32 runtime requires an explicit NPU index.")
        for path in (library, binary):
            if not Path(path).is_file():
                raise ValueError(f"GS32 native artifact is missing: {path}")
        with torch.npu.device(self.device):
            torch.ops.load_library(str(Path(library).resolve()))
            self.ops = torch.ops.vq2a8_gs32_stage
            self.ops.initialize(str(Path(binary).resolve()))

    def make_tiling(self, k, n):
        return self.ops.make_tiling(k, n)

    def check_eager(self):
        capturing = getattr(torch.npu, "is_current_stream_capturing", None)
        if capturing is not None and capturing():
            raise RuntimeError("GS32 staged backend does not support graph capture.")

    def project(self, *args):
        self.check_eager()
        self.ops.project(*args)


class GS32StagedExpert:
    """A single compressed expert resident on the runtime's device."""

    def __init__(self, runtime, layer, expert):
        self.runtime = runtime
        self.fields = {}
        self.specs = {}
        for kind in ("gate_up", "down"):
            payload, spec = runtime.artifact.load_matrix(layer, expert, kind, device="cpu", validate_payload=True)
            fields = repack_gs32_matrix(payload, spec, down_stage=True)
            # uint64 is only an offline ABI description. Every coefficient
            # uses its low 32 bits, so signed int64 transfers are lossless.
            fields["row_scale_bits"] = fields["row_scale_bits"].to(torch.int64).unsqueeze(0)
            fields["packed_zn"] = fields["packed_zn"].unsqueeze(0)
            fields["pair_lut"] = fields["pair_lut"].unsqueeze(0)
            fields["perm"] = fields["perm"].to(torch.int64)
            self.fields[kind] = {key: value.to(runtime.device, non_blocking=False) for key, value in fields.items()}
            for key in ("packed_zn", "pair_lut", "row_scale_bits"):
                value = self.fields[kind][key]
                # A leading guard expert exercises the unified loader's nonzero
                # storage offset. Finite/CPU oracle comparisons still cover all outputs.
                self.fields[kind][key] = torch.cat((torch.zeros_like(value), value), dim=0)[1:]

            self.specs[kind] = spec
        gate, down = self.specs["gate_up"], self.specs["down"]
        if gate.rows != 2 * down.columns or gate.columns != down.rows:
            raise ValueError("GS32 expert GateUp/Down dimensions do not form an FFN.")
        # Cold loads are blocking, and each projection below checks finite
        # output before returning. Cache eviction cannot race unfinished work.

    @torch.inference_mode()
    def project(self, kind, hidden):
        if kind not in self.fields:
            raise ValueError("Projection kind must be gate_up or down.")
        self.runtime.ops.check_eager()
        spec, fields = self.specs[kind], self.fields[kind]
        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.device != self.runtime.device
            or hidden.dtype != torch.bfloat16
            or hidden.ndim != 2
            or not 1 <= hidden.shape[0] <= MAX_STAGE_TOKENS
            or hidden.shape[1] != spec.columns
        ):
            raise ValueError("GS32 projection requires device BF16 [M,K] with 1<=M<=128 and matching K.")
        require_finite(hidden, "GS32 projection input")
        transformed = (hidden.float() * fields["weight_scale"]).index_select(1, fields["perm"])
        input_bytes, input_scale = quantize_mxfp8_k32(transformed)
        # Native bridge validates these CPU values before copying them to
        # device. Never trust arbitrary device-side tiling or routing bytes.
        group_list = torch.tensor([hidden.shape[0]], dtype=torch.int64, device="cpu")
        route_mask = torch.zeros(1, dtype=torch.int32, device="cpu")
        # A 99%-match tolerance alone could miss a completely unwritten N32
        # tile at N=4096. Poison first: finite checks must catch every hole.
        output = torch.full(
            (hidden.shape[0], spec.rows), float("nan"), dtype=torch.bfloat16, device=self.runtime.device
        )
        self.runtime.ops.project(
            input_bytes,
            input_scale,
            fields["packed_zn"],
            fields["pair_lut"],
            fields["row_scale_bits"],
            group_list,
            route_mask,
            self.runtime.tiling(spec.columns, spec.rows),
            output,
        )
        require_finite(output, "GS32 native projection output")
        return output

    @torch.inference_mode()
    def forward(self, hidden, swiglu_limit):
        gate_up = self.project("gate_up", hidden)
        return self.project("down", staged_swiglu(gate_up, swiglu_limit))


class GS32StagedRuntime:
    """One cache across all layers; opt-in fake ops only for CPU unit tests."""

    def __init__(self, artifact, library, binary, device, *, cache_experts=8, _test_ops=None):
        if type(cache_experts) is not int or not 1 <= cache_experts <= 256:
            raise ValueError("cache_experts must be an integer in [1,256].")
        if _test_ops is None:
            # Register the NPU device type before constructing torch.device.
            import torch_npu  # noqa: F401

        self.device = torch.device(device)
        if _test_ops is None and self.device.type != "npu":
            raise ValueError("GS32 production runtime requires NPU; there is no CPU fallback.")
        if not artifact.complete or not artifact.headers_validated:
            raise ValueError("GS32 runtime requires a complete, header-validated artifact.")
        self.artifact = artifact
        self.cache_experts = cache_experts
        self.ops = _test_ops if _test_ops is not None else _NativeStageOps(library, binary, self.device)
        self._experts = OrderedDict()
        self._tilings = {}

    def tiling(self, k, n):
        key = (k, n)
        if key not in self._tilings:
            self._tilings[key] = self.ops.make_tiling(k, n)
        return self._tilings[key]

    def get_expert(self, layer, expert):
        self.ops.check_eager()
        if type(layer) is not int or type(expert) is not int:
            raise ValueError("GS32 layer and expert indices must be integers.")
        if expert not in self.artifact.layer(layer).expert_ids:
            raise ValueError("GS32 expert is absent from the artifact.")
        key = (layer, expert)
        if key not in self._experts:
            # Evict BEFORE allocating the replacement, retaining the bound.
            if len(self._experts) >= self.cache_experts:
                self._experts.popitem(last=False)
            self._experts[key] = GS32StagedExpert(self, layer, expert)
        self._experts.move_to_end(key)
        return self._experts[key]

    def expert(self, layer, expert, hidden):
        limit = float(self.artifact.model_config["swiglu_limit"])
        return self.get_expert(layer, expert).forward(hidden, limit)
