"""MatMul1 + SwiGLU + 单路 MXFP8 量化的 CPU 参考。

MatMul1 的逻辑列顺序为 [Gate, Up]；设备权重按 Gate32、Up32 交错存放。
SwiGLU 使用 BF16 Gate/Up，FP32 计算后舍入 BF16；每 32 个 H 共用一个 E8M0 scale。
"""

from __future__ import annotations

from pathlib import Path

import ml_dtypes
import numpy as np


FP8 = ml_dtypes.float8_e4m3fn
BF16 = ml_dtypes.bfloat16
MX_SCALE_GROUP = 32
E4M3FN_MAX = np.float32(448.0)


def interleave_gate_up(data: np.ndarray, axis: int, block_size: int = 32) -> np.ndarray:
    """把指定轴的 [Gate,Up] 两半改为 Gate块、Up块交错；块内次序不变。

    对逻辑权重使用 block_size=32；对已经按 N32 分组的 LUT 使用 block_size=1。
    两者必须使用相同的 N32 块次序，K16 码表分组保持不变。
    """
    count = data.shape[axis]
    if block_size <= 0 or count % (2 * block_size):
        raise ValueError("Gate/Up 两半必须分别包含完整的块")
    blocks = np.arange(count).reshape(2, count // (2 * block_size), block_size)
    order = blocks.transpose(1, 0, 2).reshape(-1)
    return np.ascontiguousarray(np.take(data, order, axis=axis))


def finite_fp32(data: np.ndarray, name: str) -> np.ndarray:
    values = np.ascontiguousarray(data, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} 包含 NaN/Inf")
    return values


def round_bf16(data: np.ndarray) -> np.ndarray:
    """BF16 RNE 舍入后转回 FP32，便于 CPU 继续计算。"""
    with np.errstate(over="ignore", invalid="ignore"):
        values = finite_fp32(data, "BF16 input").astype(BF16).astype(np.float32)
    return finite_fp32(values, "BF16 rounded value")


def swiglu(z: np.ndarray) -> np.ndarray:
    """输入逻辑 [M,2H] FP32 MatMul1 结果，内部保留随路 BF16 舍入边界。"""
    if z.ndim != 2 or z.shape[1] == 0 or z.shape[1] % 64:
        raise ValueError("MatMul1 结果必须为 [M,N]，N 是 64 的正整数倍")
    gate, up = np.split(round_bf16(z), 2, axis=1)
    # 大负 gate 的 exp 可以溢出为 +Inf，此时 gate / Inf 为 -0，仍是有效结果。
    with np.errstate(over="ignore", under="ignore"):
        h = (gate / (np.float32(1.0) + np.exp(-gate))) * up
    return finite_fp32(h, "SwiGLU output")


def quantize_h(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """H -> BF16 -> 动态 E8M0 -> BF16 乘倒数 -> E4M3；不做高低两路拆分。"""
    data = round_bf16(data)
    if data.ndim != 2 or data.shape[1] == 0 or data.shape[1] % MX_SCALE_GROUP:
        raise ValueError("H 必须为二维矩阵，列数是 32 的正整数倍")
    grouped = data.reshape(data.shape[0], data.shape[1] // MX_SCALE_GROUP, MX_SCALE_GROUP)
    amax = np.max(np.abs(grouped), axis=-1)

    # 448 = 1.75 * 2^8。直接比较 BF16 指数/尾数，避免 log2 在边界选错档。
    bits = np.ascontiguousarray(amax).view(np.uint32)
    exponent = ((bits >> 23) & np.uint32(255)).astype(np.int32)
    codes = exponent - 8 + ((bits & np.uint32(0x7FFFFF)) > np.uint32(0x600000))
    codes = np.maximum(codes, 1)  # 限制倒数在 BF16 可表示的有限范围内。
    codes = np.where(amax == 0, 127, codes)
    inverse = np.ldexp(np.ones_like(amax), 127 - codes)

    # 设备在 BF16 寄存器中做乘法，保留这一舍入，再进行 FP8 RNE/SAT 转换。
    scaled = round_bf16(grouped * inverse[..., None])
    scaled = np.clip(scaled, -E4M3FN_MAX, E4M3FN_MAX)
    h_fp8 = np.ascontiguousarray(scaled.reshape(data.shape).astype(FP8))
    return h_fp8, np.ascontiguousarray(codes.astype(np.uint8))


def dequantize_h(h: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if h.ndim != 2 or h.shape[1] == 0 or h.shape[1] % MX_SCALE_GROUP:
        raise ValueError("H FP8 形状必须为 [M,H]，H 是 32 的正整数倍")
    expected = (h.shape[0], h.shape[1] // MX_SCALE_GROUP)
    if scale.shape != expected or scale.dtype != np.uint8:
        raise ValueError(f"H scale 必须为 {expected} uint8")
    if np.any(scale == 255):
        raise ValueError("E8M0 0xFF 表示 NaN，不能作为有效 scale")
    h_fp8 = h.view(FP8) if h.dtype == np.uint8 else h.astype(FP8, copy=False)
    values = finite_fp32(h_fp8, "H FP8")
    scale_values = np.ldexp(np.ones(expected, dtype=np.float32), scale.astype(np.int32) - 127)
    with np.errstate(over="ignore", invalid="ignore"):
        output = values.reshape(expected + (MX_SCALE_GROUP,)) * scale_values[..., None]
    return finite_fp32(output.reshape(h.shape), "H dequantized")


def read_exact(path: Path, dtype, shape: tuple[int, ...]) -> np.ndarray:
    expected_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"{path} 包含 {actual_bytes} 字节，预期 {expected_bytes} 字节")
    return np.fromfile(path, dtype=dtype).reshape(shape)
