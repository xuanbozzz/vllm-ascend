#!/usr/bin/env python3
"""比较 grouped matmul 的 BF16 输出与 CPU FP32 Golden。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


MAX_GROUP_COUNT = 256


def bf16_to_fp32(raw: np.ndarray) -> np.ndarray:
    return (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)


def round_fp32_to_bf16(data: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(data, dtype=np.float32).view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return bf16_to_fp32(((bits + rounding_bias) >> np.uint32(16)).astype(np.uint16))


def load_fp32_exact(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    expected = int(np.prod(shape))
    data = np.fromfile(path, dtype=np.float32)
    if data.size != expected:
        raise ValueError(f"{path} 包含 {data.size} 个 FP32 元素，预期 {expected} 个")
    return data.reshape(shape)


def load_bf16_exact(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    expected = int(np.prod(shape))
    raw = np.fromfile(path, dtype=np.uint16)
    if raw.size != expected:
        raise ValueError(f"{path} 包含 {raw.size} 个 BF16 元素，预期 {expected} 个")
    return bf16_to_fp32(raw).reshape(shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较最终 Down BF16 output_c.bin 与 FP32 Golden；位置参数为 G D")
    parser.add_argument("G", type=int, help=f"组/专家数，范围 1..{MAX_GROUP_COUNT}")
    parser.add_argument("D", type=int, help="Down 输出维 D")
    parser.add_argument("--group-list", default="input/input_group_list.bin", help="非累计 int64 M_i 文件")
    parser.add_argument("--output", default="output/output_c.bin")
    parser.add_argument("--golden", default="output/golden_c_fp32.bin")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--required-match-rate", type=float, default=99, help="满足 np.isclose 的最低百分比")
    parser.add_argument("--show", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.G <= MAX_GROUP_COUNT:
        raise ValueError(f"G 必须位于 [1,{MAX_GROUP_COUNT}]，实际为 {args.G}")
    if args.D <= 0:
        raise ValueError(f"D 必须为正数，实际为 {args.D}")

    group_list = np.fromfile(Path(args.group_list), dtype=np.int64)
    if group_list.shape != (args.G,):
        raise ValueError(f"{args.group_list} 包含 {group_list.size} 个 int64，预期 {args.G} 个")
    if np.any(group_list < 0):
        raise ValueError(f"group_list[i] 必须为非负整数，实际为 {group_list.tolist()}")
    total_m = int(group_list.sum())
    if total_m == 0:
        raise ValueError("group_list 至少需要包含一个 token 行")

    shape = (total_m, args.D)
    output = load_bf16_exact(Path(args.output), shape)
    golden_fp32 = load_fp32_exact(Path(args.golden), shape)
    if not np.isfinite(output).all() or not np.isfinite(golden_fp32).all():
        output_bad = int((~np.isfinite(output)).sum())
        golden_bad = int((~np.isfinite(golden_fp32)).sum())
        raise ValueError(f"检测到 NaN/Inf：output={output_bad}，golden={golden_bad}")

    # 主要验证 L0C FP32 结果经过 Fixpipe BF16 舍入后的正确性。
    golden_bf16 = round_fp32_to_bf16(golden_fp32)
    close = np.isclose(output, golden_bf16, rtol=args.rtol, atol=args.atol, equal_nan=False)
    absolute_error = np.abs(output - golden_bf16)
    relative_error = absolute_error / np.maximum(np.abs(golden_bf16), args.atol)
    fp32_absolute_error = np.abs(output - golden_fp32)
    match_rate = float(close.mean() * 100.0)

    print(f"group_list: {group_list.tolist()} (non-cumulative M_i)")
    print(f"shape: {shape}, output: BF16, golden: FP32 -> BF16")
    print(f"overall match rate: {match_rate:.6f}% (required {args.required_match_rate:.6f}%)")
    print(f"vs BF16 golden: max abs error={absolute_error.max():.8e}, mean abs error={absolute_error.mean():.8e}, "
          f"max relative error: {relative_error.max():.8e}")
    print(f"vs FP32 golden: max abs error={fp32_absolute_error.max():.8e}, "
          f"mean abs error={fp32_absolute_error.mean():.8e}")

    row_begin = 0
    for group_id, group_m_value in enumerate(group_list):
        group_m = int(group_m_value)
        row_end = row_begin + group_m
        if group_m == 0:
            print(f"group {group_id}: M_i=0, skipped")
        else:
            group_abs = absolute_error[row_begin:row_end]
            group_match = float(close[row_begin:row_end].mean() * 100.0)
            print(f"group {group_id}: M_i={group_m}, match={group_match:.6f}%, max_abs={group_abs.max():.8e}, "
                  f"mean_abs={group_abs.mean():.8e}")
        row_begin = row_end

    mismatch_flat = np.flatnonzero(~close)
    if mismatch_flat.size:
        order = mismatch_flat[np.argsort(absolute_error.reshape(-1)[mismatch_flat])[::-1]]
        group_ends = np.cumsum(group_list)
        print(f"largest {min(args.show, order.size)} mismatches:")
        for rank, flat_index in enumerate(order[:args.show], start=1):
            token_id, column = np.unravel_index(flat_index, shape)
            group_id = int(np.searchsorted(group_ends, token_id, side="right"))
            group_begin = 0 if group_id == 0 else int(group_ends[group_id - 1])
            print(f"{rank:3d}: group={group_id}, local_m={token_id - group_begin}, token={token_id}, n={column}, "
                  f"output={output[token_id, column]:.8e}, golden_bf16={golden_bf16[token_id, column]:.8e}, "
                  f"golden_fp32={golden_fp32[token_id, column]:.8e}, "
                  f"abs={absolute_error[token_id, column]:.8e}")

    if match_rate < args.required_match_rate:
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
