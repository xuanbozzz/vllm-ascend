#!/usr/bin/env python3
"""Generate one mixed W2/W4 GateUp+SwiGLUQuant+Down test case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ml_dtypes
import numpy as np

from swiglu_reference import dequantize_h, interleave_gate_up, quantize_h, swiglu
from w2_layout import (
    CODEBOOK_K,
    CODEBOOK_N,
    build_pair_lut,
    pack_zn_2bit_codes,
    reshape_codes_by_codebook,
    restore_codes_from_codebook_groups,
)
from w4_layout import decode_e2m1, expand_mx_scale, pack_zn_fp4, validate_conversion


MAX_GROUP_COUNT = 256
MX_SCALE_GROUP = 32
E4M3FN_MAX = np.float32(448.0)
E8M0_EXP_BIAS = 127
FP8 = ml_dtypes.float8_e4m3fn
BF16 = ml_dtypes.bfloat16
STAGE_GATE_W2 = 1 << 0
STAGE_GATE_W4 = 1 << 1
STAGE_DOWN_W2 = 1 << 2
STAGE_DOWN_W4 = 1 << 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate mixed 2+2, 2+4 and 4+4 expert inputs and final BF16 C golden"
    )
    parser.add_argument("G", type=int, help="expert count, 1..256")
    parser.add_argument("K", type=int, help="GateUp reduction K, multiple of 64")
    parser.add_argument("N", type=int, help="Gate+Up total width, multiple of 128")
    parser.add_argument("D", type=int, help="Down output width, multiple of 32")
    parser.add_argument("--group-list", required=True, help="comma-separated non-cumulative M_i")
    parser.add_argument("--is-expert-w4", required=True, help="comma-separated GateUp 0/1 routes")
    parser.add_argument("--is-down-w4", required=True, help="comma-separated Down 0/1 routes")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def parse_int_list(text: str, count: int, name: str) -> np.ndarray:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != count or any(not part for part in parts):
        raise ValueError(f"{name} must contain exactly {count} comma-separated integers")
    try:
        return np.asarray([int(part) for part in parts], dtype=np.int64)
    except ValueError as error:
        raise ValueError(f"invalid {name}: {text!r}") from error


def validate_case(g: int, k: int, n: int, d: int, group_list: np.ndarray,
                  expert_w4: np.ndarray, down_w4: np.ndarray) -> None:
    if not 1 <= g <= MAX_GROUP_COUNT:
        raise ValueError(f"G must be in [1,{MAX_GROUP_COUNT}], got {g}")
    if k <= 0 or k % 64:
        raise ValueError(f"K must be a positive multiple of 64, got {k}")
    if n <= 0 or n % 128:
        raise ValueError(f"N must be a positive multiple of 128, got {n}")
    if d <= 0 or d % 32:
        raise ValueError(f"D must be a positive multiple of 32, got {d}")
    if np.any(group_list < 0) or int(group_list.sum()) == 0:
        raise ValueError("group_list must be non-negative and contain at least one token")
    if np.any((expert_w4 != 0) & (expert_w4 != 1)):
        raise ValueError("is_expert_w4 entries must be 0 or 1")
    if np.any((down_w4 != 0) & (down_w4 != 1)):
        raise ValueError("is_down_w4 entries must be 0 or 1")
    invalid = np.flatnonzero((expert_w4 == 1) & (down_w4 == 0))
    if invalid.size:
        raise ValueError(f"invalid 4+2 route at experts {invalid.tolist()}")


def pack_route_mask(values: np.ndarray) -> np.ndarray:
    words = np.zeros((values.size + 31) // 32, dtype=np.uint32)
    for index, value in enumerate(values):
        if int(value):
            words[index >> 5] |= np.uint32(1) << np.uint32(index & 31)
    return words


def compute_stage_mask(group_list: np.ndarray, expert_w4: np.ndarray,
                       down_w4: np.ndarray) -> int:
    stage_mask = 0
    for group_m, gate_route, down_route in zip(group_list, expert_w4, down_w4):
        if int(group_m) == 0:
            continue
        stage_mask |= STAGE_GATE_W4 if int(gate_route) else STAGE_GATE_W2
        stage_mask |= STAGE_DOWN_W4 if int(down_route) else STAGE_DOWN_W2
    return stage_mask


def quantize_mxfp8(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped = np.ascontiguousarray(data, dtype=np.float32).reshape(data.shape[0], -1, MX_SCALE_GROUP)
    amax = np.max(np.abs(grouped), axis=-1)
    ratio = np.maximum(amax / E4M3FN_MAX, np.finfo(np.float32).tiny)
    exponent = np.ceil(np.log2(ratio)).astype(np.int32)
    exponent = np.where(amax == 0.0, 0, exponent)
    exponent = np.clip(exponent, -126, 127)
    scale = np.exp2(exponent.astype(np.float32)).astype(np.float32)
    normalized = np.clip(grouped / scale[..., None], -E4M3FN_MAX, E4M3FN_MAX)
    fp8 = np.ascontiguousarray(normalized.reshape(data.shape).astype(FP8))
    codes = np.ascontiguousarray((exponent + E8M0_EXP_BIAS).astype(np.uint8))
    dequant = fp8.astype(np.float32).reshape(grouped.shape) * scale[..., None]
    return fp8, codes, np.ascontiguousarray(dequant.reshape(data.shape))


def random_e8m0_scale(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    exponent = rng.integers(-4, -1, size=shape, dtype=np.int16)
    return np.ascontiguousarray(exponent + E8M0_EXP_BIAS, dtype=np.uint8)


def make_w2_weight(rng: np.random.Generator, out_n: int, red_k: int,
                   gate_up_interleave: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_groups = out_n // CODEBOOK_N
    k_groups = red_k // CODEBOOK_K
    scales = rng.uniform(0.0625, 0.25, size=(n_groups, k_groups, 1)).astype(np.float32)
    base = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
    levels = np.ascontiguousarray((scales * base).astype(FP8))
    codes = rng.integers(0, 4, size=(out_n, red_k), dtype=np.uint8)

    device_codes = interleave_gate_up(codes, axis=0) if gate_up_interleave else codes
    device_levels = interleave_gate_up(levels, axis=0, block_size=1) if gate_up_interleave else levels
    packed = pack_zn_2bit_codes(device_codes)
    table = build_pair_lut(device_levels).reshape(n_groups, k_groups, 32)

    grouped_codes = reshape_codes_by_codebook(codes)
    grouped_weight = np.take_along_axis(
        levels[..., None, None, :], grouped_codes[..., None], axis=-1
    )[..., 0]
    dequant = restore_codes_from_codebook_groups(grouped_weight).astype(np.float32)
    return np.ascontiguousarray(packed), np.ascontiguousarray(table), np.ascontiguousarray(dequant)


def make_w4_weight(rng: np.random.Generator, out_n: int, red_k: int,
                   gate_up_interleave: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    codes = rng.integers(0, 16, size=(out_n, red_k), dtype=np.uint8)
    scale = random_e8m0_scale(rng, (out_n, red_k // MX_SCALE_GROUP))
    device_codes = interleave_gate_up(codes, axis=0) if gate_up_interleave else codes
    device_scale = interleave_gate_up(scale, axis=0) if gate_up_interleave else scale
    packed = pack_zn_fp4(device_codes)
    dequant = decode_e2m1(codes) * expand_mx_scale(scale)
    return np.ascontiguousarray(packed), np.ascontiguousarray(device_scale), np.ascontiguousarray(dequant)


def append_array(handle, value: np.ndarray) -> int:
    contiguous = np.ascontiguousarray(value)
    contiguous.tofile(handle)
    return int(contiguous.nbytes)


def generate(g: int, k: int, n: int, d: int, group_list: np.ndarray,
             expert_w4: np.ndarray, down_w4: np.ndarray, seed: int) -> None:
    validate_case(g, k, n, d, group_list, expert_w4, down_w4)
    validate_conversion()
    total_m = int(group_list.sum())
    hidden = n // 2
    input_dir = Path("input")
    output_dir = Path("output")
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    activation_source = rng.standard_normal((total_m, k), dtype=np.float32)
    activation, activation_scale, activation_f32 = quantize_mxfp8(activation_source)
    # uint64 stores a normalized FP32 coefficient in its low 32 bits.
    # W4 routes ignore the corresponding coefficient but keep their [G,N] row.
    supported_scales = np.asarray([0.5, 0.75, 1.0, 1.25, 1.5, 2.0], dtype=np.float32)
    scale_upgate_f32 = rng.choice(supported_scales, size=(g, n))
    scale_down_f32 = rng.choice(supported_scales, size=(g, d))
    scale_upgate = scale_upgate_f32.view(np.uint32).astype(np.uint64)
    scale_down = scale_down_f32.view(np.uint32).astype(np.uint64)
    golden_gateup = np.empty((total_m, n), dtype=np.float32)
    gate_w_offsets, gate_table_offsets = [0], [0]

    with (input_dir / "input_gateup_w.bin").open("wb") as weight_file, \
         (input_dir / "input_gateup_table.bin").open("wb") as table_file:
        row_begin = 0
        for expert_id, group_m_value in enumerate(group_list):
            if int(expert_w4[expert_id]):
                packed, table, dequant_weight = make_w4_weight(rng, n, k, True)
            else:
                packed, table, dequant_weight = make_w2_weight(rng, n, k, True)
            gate_w_offsets.append(gate_w_offsets[-1] + append_array(weight_file, packed))
            gate_table_offsets.append(gate_table_offsets[-1] + append_array(table_file, table))

            group_m = int(group_m_value)
            row_end = row_begin + group_m
            if group_m:
                golden_gateup[row_begin:row_end] = (
                    activation_f32[row_begin:row_end] @ dequant_weight.T
                )
                if not int(expert_w4[expert_id]):
                    golden_gateup[row_begin:row_end] = (
                        golden_gateup[row_begin:row_end] * scale_upgate_f32[expert_id]
                    ).astype(BF16).astype(np.float32)
            row_begin = row_end

    h, h_scale = quantize_h(swiglu(golden_gateup))
    h_f32 = dequantize_h(h, h_scale)
    golden_down = np.empty((total_m, d), dtype=np.float32)
    down_w_offsets, down_table_offsets = [0], [0]

    with (input_dir / "input_down_w.bin").open("wb") as weight_file, \
         (input_dir / "input_down_table.bin").open("wb") as table_file:
        row_begin = 0
        for expert_id, group_m_value in enumerate(group_list):
            if int(down_w4[expert_id]):
                packed, table, dequant_weight = make_w4_weight(rng, d, hidden, False)
            else:
                packed, table, dequant_weight = make_w2_weight(rng, d, hidden, False)
            down_w_offsets.append(down_w_offsets[-1] + append_array(weight_file, packed))
            down_table_offsets.append(down_table_offsets[-1] + append_array(table_file, table))

            group_m = int(group_m_value)
            row_end = row_begin + group_m
            if group_m:
                golden_down[row_begin:row_end] = h_f32[row_begin:row_end] @ dequant_weight.T
                if not int(down_w4[expert_id]):
                    golden_down[row_begin:row_end] *= scale_down_f32[expert_id]
            row_begin = row_end

    golden_c = np.ascontiguousarray(golden_down.astype(BF16))
    expert_mask = pack_route_mask(expert_w4)
    down_mask = pack_route_mask(down_w4)
    stage_mask = compute_stage_mask(group_list, expert_w4, down_w4)
    group_list.astype(np.int64, copy=False).tofile(input_dir / "input_group_list.bin")
    activation.tofile(input_dir / "input_a.bin")
    activation_scale.tofile(input_dir / "input_a_scale.bin")
    scale_upgate.tofile(input_dir / "input_scale_upgate.bin")
    scale_down.tofile(input_dir / "input_scale_down.bin")
    expert_mask.tofile(input_dir / "input_is_expert_w4.bin")
    down_mask.tofile(input_dir / "input_is_down_w4.bin")
    golden_down.tofile(output_dir / "golden_c_fp32.bin")
    golden_c.tofile(output_dir / "golden_c.bin")

    metadata = {
        "experts": g,
        "group_list": group_list.tolist(),
        "group_list_semantics": "non-cumulative M_i",
        "total_m": total_m,
        "k": k,
        "n": n,
        "hidden": hidden,
        "d": d,
        "is_expert_w4": expert_w4.tolist(),
        "is_down_w4": down_w4.tolist(),
        "valid_routes": {"00": "W2+W2", "01": "W2+W4", "11": "W4+W4"},
        "route_mask_layout": "uint32 little-endian words; expert i is bit (i&31) of word (i>>5)",
        "route_mask_words": int(expert_mask.size),
        "active_stage_mask": stage_mask,
        "gateup_weight_offsets": gate_w_offsets,
        "gateup_table_offsets": gate_table_offsets,
        "down_weight_offsets": down_w_offsets,
        "down_table_offsets": down_table_offsets,
        "a_shape": [total_m, k],
        "a_dtype": "float8_e4m3fn",
        "a_scale_shape": [total_m, k // 32],
        "a_scale_dtype": "float8_e8m0",
        "c_shape": [total_m, d],
        "c_dtype": "bfloat16",
        "seed": seed,
    }
    (input_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    routes = [f"{int(expert_w4[i])}{int(down_w4[i])}" for i in range(g)]
    print(f"group_list: {group_list.tolist()} (T={total_m})")
    print(f"routes:     {routes}")
    print(f"stage mask: 0x{stage_mask:x} (GateW2/W4,DownW2/W4 bits 0..3)")
    print(f"mask bytes: {expert_mask.nbytes} each")
    print(f"Gate W/table bytes: {gate_w_offsets[-1]}/{gate_table_offsets[-1]}")
    print(f"Down W/table bytes: {down_w_offsets[-1]}/{down_table_offsets[-1]}")
    print(f"Golden C:   {golden_c.shape} BF16, {golden_c.nbytes} bytes")


if __name__ == "__main__":
    args = parse_args()
    group_list = parse_int_list(args.group_list, args.G, "group-list")
    expert_w4 = parse_int_list(args.is_expert_w4, args.G, "is-expert-w4")
    down_w4 = parse_int_list(args.is_down_w4, args.G, "is-down-w4")
    generate(args.G, args.K, args.N, args.D, group_list, expert_w4, down_w4, args.seed)
