# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed GS32 checks for the imported *current* mixed FFN ABI.

This module does not dispatch a kernel and does not certify a model for
serving.  It only answers whether one Gate/Up and Down payload satisfy the
mathematical preconditions that can be represented by the current raw fused
kernel ABI.  Torch ownership, current-stream use, task-queue lifetime, graph
capture/replay and real-Ascend950 execution remain separate acceptance gates.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .vq2a8_gs32_artifact import GS32_FIELDS, VQ2GS32MatrixSpec, validate_gs32_payload

_MISSING = object()


@dataclass(frozen=True)
class CurrentFusedNumericalRoute:
    """Numerics implemented by the imported raw fused kernel.

    The two explicit ``False`` fields are part of the public report so a
    caller cannot accidentally present this route as either reference path.
    """

    name: str
    gate_up_input: str
    intermediate: str
    down_input: str
    equivalent_to_bf16_reconstructed_weights: bool
    equivalent_to_original_rowwise_a8_w8_experiment: bool


CURRENT_FUSED_NUMERICAL_ROUTE = CurrentFusedNumericalRoute(
    name="current_raw_k32_mxfp8_intermediate",
    gate_up_input=("GS32 Gate/Up permutation and input-column scale must be applied before the raw A/A_scale inputs"),
    intermediate="unclipped SwiGLU followed by K32 MXFP8 activation quantization",
    down_input="no GS32 Down permutation or input-column scaling is representable by the current raw ABI",
    equivalent_to_bf16_reconstructed_weights=False,
    equivalent_to_original_rowwise_a8_w8_experiment=False,
)


_UNVERIFIED_RUNTIME_CONTRACTS = (
    "Torch adapter tensor ownership and device guard",
    "current NPU stream launch",
    "task-queue strong-owner lifetime",
    "non-default-stream execution",
    "graph capture and replay with stable addresses",
    "real Ascend950 kernel accuracy and performance",
)


@dataclass(frozen=True)
class CurrentFusedKernelCompatibility:
    """Structured result for the current raw ABI's mathematical gate."""

    abi_math_prerequisites_met: bool
    model_ready: bool
    blockers: tuple[str, ...]
    required_preprocessing: tuple[str, ...]
    unverified_runtime_contracts: tuple[str, ...]
    numerical_route: CurrentFusedNumericalRoute

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report for diagnostic CLIs."""
        return asdict(self)


def _config_value(model_config: object, name: str) -> object:
    if isinstance(model_config, Mapping):
        return model_config.get(name, _MISSING)
    return getattr(model_config, name, _MISSING)


def _display(value: object) -> str:
    return "<missing>" if value is _MISSING else repr(value)


def _positive_multiple_of_32(model_config: object, name: str, blockers: list[str]) -> int | None:
    value = _config_value(model_config, name)
    if type(value) is not int or value <= 0 or value % 32:
        blockers.append(
            f"CONFIG_{name.upper()}: {name} must be a positive integer divisible by 32, got {_display(value)}."
        )
        return None
    return value


def _payload_spec(kind: str, hidden: int, intermediate: int) -> VQ2GS32MatrixSpec:
    rows, columns = (2 * intermediate, hidden) if kind == "gate_up" else (hidden, intermediate)
    return VQ2GS32MatrixSpec(f"0.mlp.experts.0.{kind}", 0, 0, kind, rows, columns)


def _check_repack_geometry(hidden: int | None, intermediate: int | None, blockers: list[str]) -> None:
    """Mirror every geometry restriction imposed by ``repack_gs32_matrix``."""
    if hidden is not None and hidden % 64:
        blockers.append(
            "GATE_UP_REPACK_K_UNSUPPORTED: hidden_size is Gate/Up K and must be divisible by 64 for the "
            f"current mixed-FFN layout, got {hidden}."
        )
    if intermediate is not None:
        gate_up_rows = 2 * intermediate
        if gate_up_rows % 128:
            blockers.append(
                "GATE_UP_REPACK_N_UNSUPPORTED: 2 * moe_intermediate_size is Gate/Up N and must be divisible "
                f"by 128 for interleaved gate/up tiles, got {gate_up_rows}."
            )
        if intermediate % 64:
            blockers.append(
                "DOWN_REPACK_K_UNSUPPORTED: moe_intermediate_size is Down K and must be divisible by 64 for "
                f"the current mixed-FFN layout, got {intermediate}."
            )


def _validate_payload(
    payload: object,
    kind: str,
    hidden: int | None,
    intermediate: int | None,
    blockers: list[str],
) -> bool:
    prefix = kind.upper()
    if not isinstance(payload, Mapping):
        blockers.append(f"{prefix}_PAYLOAD: payload must be a mapping containing exactly {GS32_FIELDS}.")
        return False
    if set(payload) != set(GS32_FIELDS):
        missing = sorted(set(GS32_FIELDS) - set(payload))
        extra = sorted(set(payload) - set(GS32_FIELDS))
        blockers.append(f"{prefix}_PAYLOAD_FIELDS: exact six-field payload required; missing={missing}, extra={extra}.")
        return False
    non_cpu = [
        field
        for field in GS32_FIELDS
        if not isinstance(payload[field], torch.Tensor) or payload[field].device.type != "cpu"
    ]
    if non_cpu:
        blockers.append(f"{prefix}_PAYLOAD_CPU: validation requires CPU tensors for fields {non_cpu}.")
        return False
    if hidden is None or intermediate is None:
        blockers.append(f"{prefix}_PAYLOAD_SHAPE: payload shape cannot be certified without valid model geometry.")
        return False
    try:
        validate_gs32_payload(dict(payload), _payload_spec(kind, hidden, intermediate))
    except (TypeError, ValueError) as error:
        blockers.append(f"{prefix}_PAYLOAD_INVALID: {error}")
        return False
    return True


def _is_identity_permutation(value: torch.Tensor) -> bool:
    expected = torch.arange(value.numel(), dtype=value.dtype, device="cpu")
    return torch.equal(value, expected)


def _is_exact_unit_scale(value: torch.Tensor) -> bool:
    return torch.equal(value, torch.ones_like(value))


def current_fused_kernel_compatibility(
    model_config: object,
    gate_payload: object,
    down_payload: object,
) -> CurrentFusedKernelCompatibility:
    """Check only what the current raw fused ABI can represent.

    Gate/Up ``perm`` and ``weight_scale`` may be handled by a separate,
    graph-safe activation-preparation step before the raw ``A/A_scale`` input.
    Down transforms occur between SwiGLU and Down quantization, so the current
    ABI is compatible only when Down ``perm`` is identity and Down
    ``weight_scale`` is exactly one.  Its SwiGLU is unclipped, hence
    ``swiglu_limit`` must be explicitly zero.

    A passing result is not a serving approval: ``model_ready`` deliberately
    remains false until every item in ``unverified_runtime_contracts`` is
    accepted independently on real hardware.
    """
    blockers: list[str] = []
    required_preprocessing: list[str] = []

    model_type = _config_value(model_config, "model_type")
    if model_type != "deepseek_v4":
        blockers.append(f"CONFIG_MODEL_TYPE: model_type must be 'deepseek_v4', got {_display(model_type)}.")
    hidden = _positive_multiple_of_32(model_config, "hidden_size", blockers)
    intermediate = _positive_multiple_of_32(model_config, "moe_intermediate_size", blockers)
    _check_repack_geometry(hidden, intermediate, blockers)

    limit = _config_value(model_config, "swiglu_limit")
    if type(limit) not in (int, float) or not math.isfinite(float(limit)) or float(limit) != 0.0:
        blockers.append(
            "SWIGLU_CLIPPING_UNSUPPORTED: current raw ABI implements unclipped SwiGLU; "
            f"swiglu_limit must be explicitly 0, got {_display(limit)}."
        )

    gate_valid = _validate_payload(gate_payload, "gate_up", hidden, intermediate, blockers)
    down_valid = _validate_payload(down_payload, "down", hidden, intermediate, blockers)

    if gate_valid:
        gate_perm = gate_payload["perm"]
        gate_scale = gate_payload["weight_scale"]
        if not _is_identity_permutation(gate_perm) or not _is_exact_unit_scale(gate_scale):
            required_preprocessing.append(
                "Gate/Up: before raw A/A_scale quantization, multiply logical columns by weight_scale, then "
                "gather them by perm; this is external to the current fused ABI."
            )

    if down_valid:
        if not _is_identity_permutation(down_payload["perm"]):
            blockers.append(
                "DOWN_PERM_UNSUPPORTED: Down perm must be identity; the required post-SwiGLU gather is absent "
                "from the current raw ABI."
            )
        if not _is_exact_unit_scale(down_payload["weight_scale"]):
            blockers.append(
                "DOWN_COLUMN_SCALE_UNSUPPORTED: Down weight_scale must be exactly all ones; the required "
                "post-SwiGLU input-column scaling is absent from the current raw ABI."
            )

    return CurrentFusedKernelCompatibility(
        abi_math_prerequisites_met=not blockers,
        model_ready=False,
        blockers=tuple(blockers),
        required_preprocessing=tuple(required_preprocessing),
        unverified_runtime_contracts=_UNVERIFIED_RUNTIME_CONTRACTS,
        numerical_route=CURRENT_FUSED_NUMERICAL_ROUTE,
    )


def require_current_fused_compatibility(
    model_config: object,
    gate_payload: object,
    down_payload: object,
) -> CurrentFusedKernelCompatibility:
    """Return the mathematical report or raise with every fail-closed blocker."""
    report = current_fused_kernel_compatibility(model_config, gate_payload, down_payload)
    if not report.abi_math_prerequisites_met:
        details = "\n".join(f"  - {blocker}" for blocker in report.blockers)
        raise ValueError(
            "GS32 payload is incompatible with the current raw mixed-FFN ABI:\n"
            f"{details}\n"
            "No kernel dispatch was attempted. A mathematical pass would still not certify model readiness; "
            "Torch/current-stream/queue/graph and real-Ascend950 gates remain unverified."
        )
    return report


__all__ = [
    "CURRENT_FUSED_NUMERICAL_ROUTE",
    "CurrentFusedKernelCompatibility",
    "CurrentFusedNumericalRoute",
    "current_fused_kernel_compatibility",
    "require_current_fused_compatibility",
]
