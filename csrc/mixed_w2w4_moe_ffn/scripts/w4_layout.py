"""MXFP4 E2M1 packed-zN layout and bit-exact reference helpers."""

from __future__ import annotations

import numpy as np


ZN_K0 = 16
ZN_N0 = 32
MX_SCALE_GROUP = 32
FP4_LUT_HALF_BYTES = 256
FP4_LUT_BYTES = 2 * FP4_LUT_HALF_BYTES

# E2M1 code order is s|ee|m.  Code 0x8 is negative zero.
E2M1_VALUES = np.asarray(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

E2M1_TO_E4M3_BITS = np.asarray(
    [0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C,
     0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC],
    dtype=np.uint8,
)


def validate_fp4_codes(codes: np.ndarray) -> None:
    if codes.dtype != np.uint8:
        raise TypeError(f"codes must be uint8, got {codes.dtype}")
    if codes.size and int(codes.max()) > 15:
        raise ValueError("E2M1 code must be in [0,15]")


def pack_nibbles(codes: np.ndarray) -> np.ndarray:
    """Pack adjacent E2M1 codes; the earlier element occupies the low nibble."""
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    validate_fp4_codes(codes)
    if codes.shape[-1] % 2:
        raise ValueError("the last dimension must be even")
    return np.ascontiguousarray(codes[..., 0::2] | (codes[..., 1::2] << np.uint8(4)), dtype=np.uint8)


def unpack_nibbles(packed: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    out = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.uint8)
    out[..., 0::2] = packed & np.uint8(0x0F)
    out[..., 1::2] = packed >> np.uint8(4)
    return out


def pack_zn_fp4(codes: np.ndarray) -> np.ndarray:
    """Convert logical ``[...,N,K]`` codes to ``[...,N/32,K/16,16,16]`` packed zN."""
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    validate_fp4_codes(codes)
    if codes.ndim < 2:
        raise ValueError("codes must have at least N and K dimensions")

    n, k = codes.shape[-2:]
    if n % ZN_N0:
        raise ValueError(f"N must be divisible by {ZN_N0}")
    if k % ZN_K0:
        raise ValueError(f"K must be divisible by {ZN_K0}")

    prefix = codes.shape[:-2]
    dims = tuple(range(len(prefix)))
    logical_kn = np.swapaxes(codes, -2, -1)
    zn = logical_kn.reshape(prefix + (k // ZN_K0, ZN_K0, n // ZN_N0, ZN_N0))
    zn = zn.transpose(dims + (len(prefix) + 2, len(prefix), len(prefix) + 1, len(prefix) + 3))
    return pack_nibbles(zn)


def unpack_zn_fp4(packed: np.ndarray) -> np.ndarray:
    """Restore logical ``[...,N,K]`` E2M1 codes from packed zN."""
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    if packed.ndim < 4:
        raise ValueError("packed zN must end in [N/32,K/16,16,16]")
    if packed.shape[-2:] != (ZN_K0, ZN_N0 // 2):
        raise ValueError(f"packed zN must end in [{ZN_K0},{ZN_N0 // 2}]")

    prefix = packed.shape[:-4]
    dims = tuple(range(len(prefix)))
    n_blocks, k_blocks = packed.shape[-4:-2]
    zn = unpack_nibbles(packed)
    logical_kn = zn.transpose(dims + (len(prefix) + 1, len(prefix) + 2, len(prefix), len(prefix) + 3))
    logical_kn = logical_kn.reshape(prefix + (k_blocks * ZN_K0, n_blocks * ZN_N0))
    return np.ascontiguousarray(np.swapaxes(logical_kn, -2, -1))


def decode_e2m1(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.uint8)
    validate_fp4_codes(codes)
    return E2M1_VALUES[codes]


def encode_e2m1_as_e4m3(codes: np.ndarray) -> np.ndarray:
    """Convert E2M1 codes to equal-valued E4M3 bytes, preserving signed zero."""
    codes = np.asarray(codes, dtype=np.uint8)
    validate_fp4_codes(codes)
    return np.ascontiguousarray(E2M1_TO_E4M3_BITS[codes], dtype=np.uint8)


def build_fp4_to_fp8_lut() -> np.ndarray:
    """Return [low[256], high[256]] bytes indexed by the full packed byte."""
    packed = np.arange(FP4_LUT_HALF_BYTES, dtype=np.uint8)
    return np.concatenate((
        encode_e2m1_as_e4m3(packed & np.uint8(0x0F)),
        encode_e2m1_as_e4m3(packed >> np.uint8(4)),
    ))


def gather_packed_fp4_to_e4m3(packed: np.ndarray) -> np.ndarray:
    """Reference for two B8 gathers followed by a low/high interleaved store."""
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    table = build_fp4_to_fp8_lut()
    out = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.uint8)
    out[..., 0::2] = table[:FP4_LUT_HALF_BYTES][packed]
    out[..., 1::2] = table[FP4_LUT_HALF_BYTES:][packed]
    return out


def decode_e4m3fn(bits: np.ndarray) -> np.ndarray:
    """Decode finite E4M3FN bit patterns to float32."""
    bits = np.asarray(bits, dtype=np.uint8)
    sign = np.where((bits & np.uint8(0x80)) != 0, -1.0, 1.0).astype(np.float32)
    exponent = ((bits >> np.uint8(3)) & np.uint8(0x0F)).astype(np.int32)
    mantissa = (bits & np.uint8(0x07)).astype(np.float32)

    subnormal = (mantissa / np.float32(8.0)) * np.float32(2.0 ** -6)
    normal = (np.float32(1.0) + mantissa / np.float32(8.0)) * np.exp2(exponent - 7).astype(np.float32)
    value = np.where(exponent == 0, subnormal, normal).astype(np.float32)
    if np.any((exponent == 15) & (mantissa == 7)):
        value = np.where((exponent == 15) & (mantissa == 7), np.nan, value)
    return sign * value


def decode_e8m0(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.uint8)
    if np.any(codes == np.uint8(0xFF)):
        raise ValueError("0xFF is reserved and is not emitted by this generator")
    return np.exp2(codes.astype(np.int32) - 127).astype(np.float32)


def expand_mx_scale(scale_codes: np.ndarray) -> np.ndarray:
    """Expand one E8M0 scale per K32 group along the final K dimension."""
    return np.repeat(decode_e8m0(scale_codes), MX_SCALE_GROUP, axis=-1)


def validate_conversion() -> None:
    codes = np.arange(16, dtype=np.uint8)
    converted = decode_e4m3fn(encode_e2m1_as_e4m3(codes))
    expected = decode_e2m1(codes)
    if not np.array_equal(converted, expected) or not np.array_equal(np.signbit(converted), np.signbit(expected)):
        raise AssertionError("E2M1 -> E4M3 conversion check failed")

    packed = np.arange(FP4_LUT_HALF_BYTES, dtype=np.uint8)
    actual = decode_e4m3fn(gather_packed_fp4_to_e4m3(packed))
    expected = decode_e2m1(unpack_nibbles(packed))
    if not np.array_equal(actual, expected) or not np.array_equal(np.signbit(actual), np.signbit(expected)):
        raise AssertionError("packed E2M1 -> E4M3 gather check failed")
