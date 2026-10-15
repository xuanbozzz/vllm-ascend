# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common MoE dispatch ordering against an independent decoded-byte oracle."""

from types import SimpleNamespace

import pytest
import torch

from tests.ut.quantization.test_vq2a8_gs32_layout import fixture
from tests.ut.quantization.test_vq2a8_gs32_staged import FakeDownOps
from tools.vq2a8_gs32_layout import repack_gs32_matrix
from tools.vq2a8_gs32_reference import staged_expert_reference


def weights(ops=None):
    from vllm_ascend.ops.fused_moe.vq2a8_gs32 import GS32ExpertWeights

    pairs = {kind: fixture(kind)[:2] for kind in ("gate_up", "down")}
    layer = SimpleNamespace(vq2a8_num_experts=3)
    for kind, prefix in (("gate_up", "w13"), ("down", "w2")):
        payload, spec = pairs[kind]
        for field, value in repack_gs32_matrix(payload, spec, down_stage=True).items():
            if field in ("row_scale_bits", "perm"):
                value = value.to(torch.int64)
            setattr(layer, f"{prefix}_{field}", value.unsqueeze(0).repeat(2, *([1] * value.ndim)))
    spec = SimpleNamespace(expert_ids=(0, 2), specs={kind: pair[1] for kind, pair in pairs.items()})
    return GS32ExpertWeights(layer, spec, {}, _test_ops=ops or FakeDownOps()), pairs


@pytest.mark.parametrize("group_type,counts", [(0, [2, 2, 3]), (1, [2, 0, 1])])
def test_resident_routing_uses_stored_slots_and_preserves_token_order(group_type, counts):
    bank, pairs = weights()
    hidden = (torch.randn(3, 64) * 0.01).bfloat16()
    expected = staged_expert_reference(hidden, *pairs["gate_up"], *pairs["down"], 10)
    actual, _ = bank.apply(hidden, torch.tensor(counts), group_type, 10)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert bank.ops.launches == 4
    assert not hasattr(bank, "artifact")  # All tensors come from the unified loader.
    assert bank.decode_mode == "grouped"


def test_resident_missing_route_and_bad_counts_fail_closed():
    bank, _ = weights()
    hidden = torch.zeros(1, 64, dtype=torch.bfloat16)
    for counts in ([0, 1, 0], [-1, 0, 2], [0, 0, 0]):
        with pytest.raises(ValueError):
            bank.apply(hidden, torch.tensor(counts), 1, 10)
    assert bank.ops.launches == 0


def test_stage_partial_output_is_detected():
    class Partial(FakeDownOps):
        def project(self, *args):
            args[-1][0, 0] = 0

    bank, _ = weights(Partial())
    with pytest.raises(ValueError, match="output"):
        bank.apply(torch.zeros(1, 64, dtype=torch.bfloat16), torch.tensor([1, 0, 0]), 1, 10)


def test_stage_chunks_large_expert_groups():
    bank, _ = weights()
    output, _ = bank.apply(torch.zeros(129, 64, dtype=torch.bfloat16), torch.tensor([129, 0, 0]), 1, 10)
    assert output.shape == (129, 64)
    assert torch.isfinite(output).all()
    assert bank.ops.launches == 4
