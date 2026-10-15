# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded safetensors reads and no-overwrite checkpoint publication."""

import ctypes
import errno
import hashlib
import json
import math
import os
import stat
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_file(path: Path) -> None:
    # Windows _commit/fsync requires a writable file descriptor. This helper
    # only handles newly-written output files, never the canonical source.
    with path.open("rb+") as stream:
        os.fsync(stream.fileno())


def _sync_directory(path: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _publish_directory(staging: Path, output: Path) -> None:
    if os.name == "nt":
        # Windows rename already refuses an existing destination directory.
        os.rename(staging, output)
        return
    if sys.platform != "linux":
        raise OSError("Atomic no-replace publication is supported on Linux and Windows only.")
    # POSIX rename may overwrite a concurrently-created EMPTY directory.
    # Use Linux RENAME_NOREPLACE instead of an exists()/rename() race; fail
    # closed if the filesystem/libc does not provide this primitive.
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        rename = libc.renameat2
    except AttributeError as error:
        raise OSError(errno.ENOTSUP, "libc.renameat2 is required for no-replace publication") from error
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    at_fdcwd, rename_noreplace = -100, 1
    if rename(at_fdcwd, os.fsencode(staging), at_fdcwd, os.fsencode(output), rename_noreplace) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(output))


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _no_symlinks(path):
    """Reject symlinks/junctions before resolution, including ancestor aliases."""
    absolute = Path(os.path.abspath(path.expanduser()))
    for part in (absolute, *absolute.parents):
        if part.is_symlink():
            raise ValueError(f"Prepack paths must not contain symlinks: {part}")
        try:
            attributes = getattr(part.lstat(), "st_file_attributes", 0)
        except FileNotFoundError:
            continue
        if attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise ValueError(f"Prepack paths must not contain filesystem junctions/reparse points: {part}")
    return absolute


def _snapshot(paths):
    return {str(path): _sha256(path) for path in paths}


def _assert_unchanged(snapshot):
    for filename, expected in snapshot.items():
        path = _no_symlinks(Path(filename))
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"Source or conversion code changed; refusing to publish: {path}")


_MAX_SAFETENSORS_HEADER_SIZE = 100000000
_SAFETENSORS_DTYPES = {"F8_E4M3": 1, "F32": 4, "I8": 1, "I32": 4, "U8": 1}


@dataclass(frozen=True)
class VQ2TensorHeader:
    """One tensor entry from a safetensors header."""

    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]


def read_safetensors_header(tensor_path: str | Path) -> dict[str, VQ2TensorHeader]:
    """Read and structurally validate a safetensors header without payload I/O."""
    path = Path(tensor_path)
    file_size = path.stat().st_size
    with path.open("rb") as tensor_file:
        length_bytes = tensor_file.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"Truncated safetensors length prefix: {path}.")
        header_size = struct.unpack("<Q", length_bytes)[0]
        if header_size > _MAX_SAFETENSORS_HEADER_SIZE:
            raise ValueError(
                f"Safetensors header size {header_size} exceeds the "
                f"{_MAX_SAFETENSORS_HEADER_SIZE}-byte safety limit: {path}."
            )
        if header_size > file_size - 8:
            raise ValueError(f"Safetensors header size {header_size} exceeds file size {file_size}: {path}.")
        header_bytes = tensor_file.read(header_size)
    try:
        raw_header = json.loads(header_bytes, object_pairs_hook=_metadata_object_without_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid safetensors JSON header: {path}.") from error
    if not isinstance(raw_header, dict):
        raise ValueError(f"Safetensors header must be an object: {path}.")
    data_size = file_size - 8 - header_size
    headers: dict[str, VQ2TensorHeader] = {}
    occupied_ranges: list[tuple[int, int, str]] = []
    for name, entry in raw_header.items():
        if name == "__metadata__":
            if not isinstance(entry, dict) or any(
                (not isinstance(key, str) or not isinstance(value, str) for key, value in entry.items())
            ):
                raise ValueError(f"Invalid safetensors __metadata__: {path}.")
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid safetensors entry for {name!r}: {path}.")
        if set(entry) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"Safetensors entry {name!r} must contain exactly dtype, shape and data_offsets: {path}.")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if dtype not in _SAFETENSORS_DTYPES:
            raise ValueError(f"Unsupported safetensors dtype {dtype!r} for {name!r}.")
        if not isinstance(shape, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in shape
        ):
            raise ValueError(f"Invalid safetensors shape for {name!r}: {shape!r}.")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in offsets)
        ):
            raise ValueError(f"Invalid safetensors offsets for {name!r}: {offsets!r}.")
        start, end = offsets
        if start < 0 or end < start or end > data_size:
            raise ValueError(f"Out-of-range safetensors offsets for {name!r}: {offsets!r}, data size={data_size}.")
        expected_bytes = math.prod(shape) * _SAFETENSORS_DTYPES[dtype]
        if end - start != expected_bytes:
            raise ValueError(
                f"Tensor {name!r} stores {end - start} bytes, expected {expected_bytes} for "
                f"dtype={dtype}, shape={shape}."
            )
        headers[name] = VQ2TensorHeader(dtype, tuple(shape), (start, end))
        occupied_ranges.append((start, end, name))
    previous_end = 0
    previous_name = "<data-start>"
    for start, end, name in sorted(occupied_ranges):
        if start < previous_end:
            raise ValueError(f"Safetensors data for {name!r} overlaps {previous_name!r}: {path}.")
        if start > previous_end:
            raise ValueError(
                f"Safetensors data has a gap before {name!r}: expected offset {previous_end}, got {start}: {path}."
            )
        previous_end = end
        previous_name = name
    if previous_end != data_size:
        raise ValueError(f"Safetensors data has {data_size - previous_end} trailing bytes: {path}.")
    return headers


def _metadata_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}.")
        result[key] = value
    return result
