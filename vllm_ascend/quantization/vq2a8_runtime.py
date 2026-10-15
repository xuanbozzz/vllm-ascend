# SPDX-License-Identifier: Apache-2.0
"""Runner-owned VQ validity, shared by resident and full-pool expert banks."""

import torch


class VQ2A8Runtime:
    def __init__(self, model, device, expected_layers=None):
        self.owners = []
        self.layer_indices = []
        layer_indices = set()
        for layer in model.modules():
            weights = getattr(layer, "vq2a8_weights", None)
            if weights is None:
                continue
            index = layer.vq2a8_layer_index
            if index in layer_indices:
                raise RuntimeError(f"Duplicate VQ validity owner for layer {index}")
            layer_indices.add(index)
            binding = getattr(layer, "_expert_offload_binding", None)
            variants = [weights]
            if binding is not None:
                variants.extend(getattr(binding.adapter, "banks", {}).values())
            variants = tuple({id(value): value for value in variants}.values())
            self.owners.append(variants)
            self.layer_indices.append(index)
        if expected_layers is not None and layer_indices != set(expected_layers):
            raise RuntimeError("VQ validity must cover every configured target expert layer")
        self.validity = torch.zeros(len(self.owners), dtype=torch.bool, device=device)
        for variants, flag in zip(self.owners, self.validity.unbind(), strict=True):
            for weights in variants:
                weights.validity_buffer = flag
                weights.defer_validity = True

    def begin(self):
        self.validity.zero_()

    def warm(self):
        for variants in self.owners:
            for weights in variants:
                weights.prepare_for_current_stream()

    def check(self):
        if self.owners and not bool(self.validity.all()):
            failed = [index for index, valid in zip(self.layer_indices, self.validity.cpu().tolist()) if not valid]
            raise RuntimeError(f"VQ2A8 forward failed validity checks in layers {failed}; rejecting execution output")
