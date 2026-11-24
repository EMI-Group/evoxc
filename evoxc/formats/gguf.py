"""GGUF v2/v3 reader — container parsing with block dequantization.

Layout (all integers little-endian)::

    [0, 4)          magic, ASCII "GGUF"
    [4, 8)          u32 version (2 or 3)
    [8, 16)         u64 tensor_count
    [16, 24)        u64 metadata_kv_count
    metadata        `kv_count` × (string key, u32 value type, typed value)
    tensor infos    `tensor_count` × (
                        string name, u32 n_dims, u64 dims[n_dims],
                        u32 ggml type id, u64 data offset)
    <padding>       zero bytes up to the next `alignment` boundary
    data section    tensor payloads at their offsets (relative to here)

A string is ``u64 length`` followed by ``length`` UTF-8 bytes.  GGUF stores
dimensions fastest-axis-first (GGML ``ne[0]`` first), so this reader exposes the
reversed tuple as the numpy row-major shape.  ``general.alignment`` metadata
(u32, default 32) sets the data-section alignment and must be a power of two.

Metadata value types: UINT8/INT8/UINT16/INT16/UINT32/INT32/FLOAT32/BOOL
(0–7), STRING (8), ARRAY (9: u32 element type + u64 count + values; nested
arrays are rejected), UINT64/INT64/FLOAT64 (10–12).

Validation (binding): magic/version, bounds-checked string lengths, metadata
type ids, a power-of-two alignment, ``n_dims <= 4``, known ggml type ids, an
element count divisible by the quantized block size, every tensor payload
inside the data section, and no two non-empty payloads overlapping.  Unknown
or unimplemented ggml type ids raise :class:`~evoxc.core.PersistenceError`
naming the id (and the GGML name where known).

Supported GGML types (ids): F32 (0), F16 (1), Q4_0 (2), Q4_1 (3), Q5_0 (6),
Q5_1 (7), Q8_0 (8), Q8_1 (9), Q2_K (10), Q3_K (11), Q4_K (12), Q5_K (13),
Q6_K (14), BF16 (30).  All other ids (including Q8_K, the IQ family, TQ, and
the integer/F64 types) are explicit v1 deferrals.

Access semantics: ``tensor(name)`` dequantizes block-quantized types to a fresh
dense ``float32`` array; F32/F16 return a zero-copy view in the tensor's own
dtype; BF16 is widened to ``float32`` (numpy has no bfloat16).  ``raw(name)``
returns the stored payload as a 1-D ``uint8`` view.  With ``mmap=True`` views
are read-only; with ``mmap=False`` every returned array is an independent
writable copy.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from ..core import PersistenceError, Tensor
from . import mmio
from .tensorinfo import TensorInfo

__all__ = ["GGUFFile", "load_gguf"]

_MAGIC = b"GGUF"
_SUPPORTED_VERSIONS = (2, 3)
_DEFAULT_ALIGNMENT = 32
_GGML_MAX_DIMS = 4

# --- metadata value types --------------------------------------------------
_UINT8, _INT8, _UINT16, _INT16 = 0, 1, 2, 3
_UINT32, _INT32, _FLOAT32, _BOOL = 4, 5, 6, 7
_STRING, _ARRAY = 8, 9
_UINT64, _INT64, _FLOAT64 = 10, 11, 12

#: value type id -> (struct format, byte size); BOOL is read as u8 and Python bool.
_SCALARS: Dict[int, Tuple[str, int]] = {
    _UINT8: ("<B", 1),
    _INT8: ("<b", 1),
    _UINT16: ("<H", 2),
    _INT16: ("<h", 2),
    _UINT32: ("<I", 4),
    _INT32: ("<i", 4),
    _FLOAT32: ("<f", 4),
    _UINT64: ("<Q", 8),
    _INT64: ("<q", 8),
    _FLOAT64: ("<d", 8),
}

_VALUE_TYPE_NAMES: Dict[int, str] = {
    _UINT8: "UINT8",
    _INT8: "INT8",
    _UINT16: "UINT16",
    _INT16: "INT16",
    _UINT32: "UINT32",
    _INT32: "INT32",
    _FLOAT32: "FLOAT32",
    _BOOL: "BOOL",
    _STRING: "STRING",
    _ARRAY: "ARRAY",
    _UINT64: "UINT64",
    _INT64: "INT64",
    _FLOAT64: "FLOAT64",
}


def _f(path: str, message: str) -> PersistenceError:
    return PersistenceError(f"{path}: {message}")


# ---------------------------------------------------------------------------
# dequantization (numpy, block layouts from ggml)
#
# Every dequantizer takes a ``(n_blocks, type_size)`` uint8 array of stored
# blocks and returns a ``(n_blocks, block_size)`` float32 array in element
# order.  Element order matters: the caller reshapes the flat result to the
# tensor's numpy shape.
# ---------------------------------------------------------------------------


def _f16_column(blocks: np.ndarray, start: int) -> np.ndarray:
    """Read the little-endian float16 scale at byte ``start`` as (n, 1)."""
    return blocks[:, start : start + 2].copy().view("<f2").astype(np.float32).reshape(-1, 1)


def _nibbles(qs: np.ndarray) -> np.ndarray:
    """Split packed 4-bit values (n, 16) into the block order (low16, high16)."""
    return np.concatenate([qs & 0x0F, qs >> 4], axis=1)


def _dequant_bf16(blocks: np.ndarray) -> np.ndarray:
    """BF16 words -> float32 (``u16 << 16`` reinterpreted, exact)."""
    words = blocks.view("<u2").astype(np.uint32)
    return (words << 16).view(np.float32)


def _dequant_q4_0(blocks: np.ndarray) -> np.ndarray:
    d = _f16_column(blocks, 0)
    q = _nibbles(blocks[:, 2:18]).astype(np.int16) - 8
    return q.astype(np.float32) * d


def _dequant_q4_1(blocks: np.ndarray) -> np.ndarray:
    d = _f16_column(blocks, 0)
    m = _f16_column(blocks, 2)
    q = _nibbles(blocks[:, 4:20]).astype(np.float32)
    return q * d + m


def _q5_quants(blocks: np.ndarray, qh_start: int, qs_start: int) -> np.ndarray:
    """5-bit quants as uint16: low nibbles + 4 high bits (n, 32)."""
    qh = blocks[:, qh_start]
    for byte in range(1, 4):
        qh = qh.astype(np.uint32) | (blocks[:, qh_start + byte].astype(np.uint32) << (8 * byte))
    ql = _nibbles(blocks[:, qs_start : qs_start + 16]).astype(np.uint16)
    bits = ((qh[:, None] >> np.arange(32, dtype=np.uint32)[None, :]) & 1).astype(np.uint16)
    return ql | (bits << 4)


def _dequant_q5_0(blocks: np.ndarray) -> np.ndarray:
    d = _f16_column(blocks, 0)
    q = _q5_quants(blocks, 2, 6).astype(np.int16) - 16
    return q.astype(np.float32) * d


def _dequant_q5_1(blocks: np.ndarray) -> np.ndarray:
    d = _f16_column(blocks, 0)
    m = _f16_column(blocks, 2)
    q = _q5_quants(blocks, 4, 8).astype(np.float32)
    return q * d + m


def _dequant_q8_0(blocks: np.ndarray) -> np.ndarray:
    d = _f16_column(blocks, 0)
    q = blocks[:, 2:34].view(np.int8).astype(np.float32)
    return q * d


def _dequant_q8_1(blocks: np.ndarray) -> np.ndarray:
    # Layout: f16 d, f16 s (the dot-product pre-sum, unused here), int8 qs[32].
    d = _f16_column(blocks, 0)
    q = blocks[:, 4:36].view(np.int8).astype(np.float32)
    return q * d


def _dequant_q2_k(blocks: np.ndarray) -> np.ndarray:
    n = blocks.shape[0]
    scales = blocks[:, 0:16]
    qs = blocks[:, 16:80]
    d = _f16_column(blocks, 80)
    dmin = _f16_column(blocks, 82)
    dl = (d * (scales & 0x0F).astype(np.float32)).reshape(n, 16, 1)
    ml = (dmin * (scales >> 4).astype(np.float32)).reshape(n, 16, 1)
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)
    q = (qs.reshape(n, 2, 1, 32) >> shifts) & 0x03
    q = q.reshape(n, 16, 16).astype(np.float32)
    return (dl * q - ml).reshape(n, 256)


def _dequant_q3_k(blocks: np.ndarray) -> np.ndarray:
    n = blocks.shape[0]
    hmask = blocks[:, 0:32]
    qs = blocks[:, 32:96]
    scales_raw = blocks[:, 96:108]
    d = _f16_column(blocks, 108)
    # 6-bit scales: low nibbles in bytes 0..7, high 2 bits packed in bytes 8..11.
    low = scales_raw[:, 0:8].reshape(n, 1, 8) >> np.array([0, 4], dtype=np.uint8).reshape(1, 2, 1)
    low = low.reshape(n, 16)
    high = scales_raw[:, 8:12].reshape(n, 1, 4) >> np.array(
        [0, 2, 4, 6], dtype=np.uint8
    ).reshape(1, 4, 1)
    high = high.reshape(n, 16)
    scales = ((low & 0x0F) | ((high & 0x03) << 4)).astype(np.int8) - 32
    dl = (d * scales.astype(np.float32)).reshape(n, 16, 1)
    shift = np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)
    ql = (qs.reshape(n, 2, 1, 32) >> shift) & 0x03
    ql = ql.reshape(n, 16, 16)
    qh = (hmask.reshape(n, 1, 1, 32) >> np.arange(8, dtype=np.uint8).reshape(1, 1, 8, 1)) & 0x01
    qh = qh.reshape(n, 16, 16)
    # ggml: value = ql - 4 when the high bit is CLEAR, else ql.
    q = ql.astype(np.int8) - ((qh ^ 1).astype(np.int8) << 2)
    return (dl * q.astype(np.float32)).reshape(n, 256)


def _k_scale_min(scales: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Unpack Q4_K/Q5_K's 12 scale/min bytes into (scale (n,8), min (n,8))."""
    n = scales.shape[0]
    packed = scales.reshape(n, 3, 4)
    d, m, m_d = packed[:, 0, :], packed[:, 1, :], packed[:, 2, :]
    scale = np.concatenate([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], axis=1)
    minimum = np.concatenate([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], axis=1)
    return scale.reshape(n, 8), minimum.reshape(n, 8)


def _dequant_q4_k(blocks: np.ndarray) -> np.ndarray:
    n = blocks.shape[0]
    d = _f16_column(blocks, 0)
    dmin = _f16_column(blocks, 2)
    scale, minimum = _k_scale_min(blocks[:, 4:16])
    qs = blocks[:, 16:144]
    ds = (d * scale.astype(np.float32)).reshape(n, 8, 1)
    dm = (dmin * minimum.astype(np.float32)).reshape(n, 8, 1)
    q = (qs.reshape(n, 4, 1, 32) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)) & 0x0F
    q = q.reshape(n, 8, 32).astype(np.float32)
    return (ds * q - dm).reshape(n, 256)


def _dequant_q5_k(blocks: np.ndarray) -> np.ndarray:
    n = blocks.shape[0]
    d = _f16_column(blocks, 0)
    dmin = _f16_column(blocks, 2)
    scale, minimum = _k_scale_min(blocks[:, 4:16])
    qh = blocks[:, 16:48]
    qs = blocks[:, 48:176]
    ds = (d * scale.astype(np.float32)).reshape(n, 8, 1)
    dm = (dmin * minimum.astype(np.float32)).reshape(n, 8, 1)
    ql = (qs.reshape(n, 4, 1, 32) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)) & 0x0F
    ql = ql.reshape(n, 8, 32)
    bits = (
        (qh.reshape(n, 1, 1, 32) >> np.arange(8, dtype=np.uint8).reshape(1, 1, 8, 1)) & 0x01
    ).reshape(n, 8, 32)
    q = (ql | (bits << 4)).astype(np.float32)
    return (ds * q - dm).reshape(n, 256)


def _dequant_q6_k(blocks: np.ndarray) -> np.ndarray:
    n = blocks.shape[0]
    ql = blocks[:, 0:128]
    qh = blocks[:, 128:192]
    scales = blocks[:, 192:208].view(np.int8).astype(np.float32)
    d = _f16_column(blocks, 208)
    dd = (d * scales).reshape(n, 16, 1)
    low = ((ql.reshape(n, 2, 1, 64) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)) & 0x0F)
    low = low.reshape(n, 8, 32)
    high = (
        (qh.reshape(n, 2, 1, 32) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1))
        & 0x03
    ).reshape(n, 8, 32)
    q = (low | (high << 4)).astype(np.int8) - 32
    q = q.reshape(n, 16, 16).astype(np.float32)
    return (dd * q).reshape(n, 256)


class _GgmlType(NamedTuple):
    name: str
    block_size: int
    type_size: int
    dtype: Optional[np.dtype]  # natural numpy dtype for unquantized types
    dequant: Optional[Callable[[np.ndarray], np.ndarray]]
    dtype_name: str  # TensorInfo.dtype


def _plain(name: str, dtype: str) -> _GgmlType:
    dt = np.dtype(dtype)
    return _GgmlType(name, 1, dt.itemsize, dt, None, dt.name)


def _quant(name: str, block: int, size: int, fn: Callable) -> _GgmlType:
    return _GgmlType(name, block, size, None, fn, name)


_GGML_TYPES: Dict[int, _GgmlType] = {
    0: _plain("F32", "float32"),
    1: _plain("F16", "float16"),
    2: _quant("Q4_0", 32, 18, _dequant_q4_0),
    3: _quant("Q4_1", 32, 20, _dequant_q4_1),
    6: _quant("Q5_0", 32, 22, _dequant_q5_0),
    7: _quant("Q5_1", 32, 24, _dequant_q5_1),
    8: _quant("Q8_0", 32, 34, _dequant_q8_0),
    9: _quant("Q8_1", 32, 36, _dequant_q8_1),
    10: _quant("Q2_K", 256, 84, _dequant_q2_k),
    11: _quant("Q3_K", 256, 110, _dequant_q3_k),
    12: _quant("Q4_K", 256, 144, _dequant_q4_k),
    13: _quant("Q5_K", 256, 176, _dequant_q5_k),
    14: _quant("Q6_K", 256, 210, _dequant_q6_k),
    30: _GgmlType("BF16", 1, 2, None, _dequant_bf16, "BF16"),
}

#: GGML type id -> name, for error messages about *unsupported* ids.
_GGML_TYPE_NAMES: Dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4",
    41: "Q1_0", 42: "Q2_0", 43: "F8_E4M3",
}


class _Entry:
    """One validated tensor info record plus its resolved data-section offset."""

    __slots__ = ("info", "ggml", "offset")

    def __init__(self, info: TensorInfo, ggml: _GgmlType, offset: int):
        self.info = info
        self.ggml = ggml
        self.offset = offset


class GGUFFile:
    """An open, validated GGUF v2/v3 file (metadata + tensor infos + data).

    Use :func:`load_gguf` (or the constructor), then :meth:`tensor_names`,
    :meth:`info`, :meth:`tensor`, :meth:`raw`.  The object owns a file
    handle/mapping: use it as a context manager or call :meth:`close`.
    """

    def __init__(self, path, *, mmap: bool = True):
        self.path = os.fspath(path)
        self._mmap = bool(mmap)
        self._file = mmio.MappedFile(self.path, mmap=self._mmap)
        self.version = 0
        self.metadata: Dict[str, Any] = {}
        self.alignment = _DEFAULT_ALIGNMENT
        self._entries: Dict[str, _Entry] = {}
        self._data_start = 0
        self._data_len = 0
        try:
            self._parse()
        except BaseException:
            self._file.close()
            raise

    # -- parsing ------------------------------------------------------------
    def _parse(self) -> None:
        f = self._file
        if f.size < 24:
            raise _f(self.path, f"file is {f.size} bytes — too short for a GGUF header")
        if f.read(0, 4, "GGUF magic") != _MAGIC:
            raise _f(
                self.path,
                f"not a GGUF file (magic {f.read(0, 4, 'GGUF magic')!r}, expected b'GGUF')",
            )
        self.version = f.u32(4, "GGUF version")
        if self.version not in _SUPPORTED_VERSIONS:
            raise _f(
                self.path,
                f"unsupported GGUF version {self.version}; this reader supports "
                f"versions {_SUPPORTED_VERSIONS}",
            )
        tensor_count = f.u64(8, "GGUF tensor count")
        kv_count = f.u64(16, "GGUF metadata count")
        pos = 24
        self._guard_count(kv_count, 8 + 4, pos, "metadata entries")
        self._guard_count(tensor_count, 8 + 4 + 4 + 8, pos, "tensor infos")

        metadata: Dict[str, Any] = {}
        for _ in range(kv_count):
            key, pos = self._read_string(pos, "metadata key")
            if key in metadata:
                raise _f(self.path, f"duplicate metadata key {key!r} at offset {pos}")
            value_type = f.u32(pos, f"metadata value type for {key!r}")
            pos += 4
            value, pos = self._read_value(value_type, pos, key)
            metadata[key] = value
        self.metadata = metadata
        self.alignment = self._resolve_alignment(metadata)

        entries: Dict[str, _Entry] = {}
        for _ in range(tensor_count):
            name, pos = self._read_string(pos, "tensor name")
            if name in entries:
                raise _f(self.path, f"duplicate tensor name {name!r} at offset {pos}")
            n_dims = f.u32(pos, f"tensor {name!r} dim count")
            pos += 4
            if n_dims > _GGML_MAX_DIMS:
                raise _f(
                    self.path,
                    f"tensor {name!r} has {n_dims} dimensions; GGUF allows at "
                    f"most {_GGML_MAX_DIMS}",
                )
            dims = []
            for _dim in range(n_dims):
                dims.append(f.u64(pos, f"tensor {name!r} dimension"))
                pos += 8
            type_id = f.u32(pos, f"tensor {name!r} ggml type")
            pos += 4
            offset = f.u64(pos, f"tensor {name!r} data offset")
            pos += 8
            entries[name] = self._build_entry(name, dims, type_id, offset)
        self._entries = entries

        data_start = mmio.align_up(pos, self.alignment)
        if data_start > f.size:
            if tensor_count:
                raise _f(
                    self.path,
                    f"tensor info section ends at {pos} and alignment "
                    f"{self.alignment} puts the data section at {data_start}, "
                    f"beyond the {f.size}-byte file",
                )
            # A metadata-only GGUF (no tensors, e.g. a tokenizer-only file)
            # may end right after the last metadata value: the alignment
            # padding only exists when a data section follows.
            data_start = f.size
        self._data_start = data_start
        self._data_len = f.size - data_start
        self._check_layout()

    def _guard_count(self, count: int, min_bytes: int, pos: int, what: str) -> None:
        remaining = self._file.size - pos
        if count * min_bytes > remaining:
            raise _f(
                self.path,
                f"GGUF header declares {count} {what}, which cannot fit in the "
                f"remaining {remaining} bytes",
            )

    def _read_string(self, pos: int, what: str) -> Tuple[str, int]:
        f = self._file
        length = f.u64(pos, f"{what} length")
        pos += 8
        if length > f.size - pos:
            raise _f(
                self.path,
                f"{what} length {length} at offset {pos - 8} runs past the "
                f"{f.size}-byte file",
            )
        raw = f.read(pos, length, what)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _f(self.path, f"{what} at offset {pos} is not valid UTF-8: {exc}") from exc
        return text, pos + length

    def _read_value(self, value_type: int, pos: int, key: str) -> Tuple[Any, int]:
        f = self._file
        if value_type == _STRING:
            return self._read_string(pos, f"metadata value {key!r}")
        if value_type == _ARRAY:
            element_type = f.u32(pos, f"array element type for {key!r}")
            pos += 4
            count = f.u64(pos, f"array element count for {key!r}")
            pos += 8
            if element_type == _ARRAY:
                raise _f(
                    self.path,
                    f"metadata {key!r} is an array of arrays, which GGUF forbids",
                )
            if element_type not in _VALUE_TYPE_NAMES:
                raise _f(
                    self.path,
                    f"metadata {key!r} uses unknown array element type "
                    f"{element_type} at offset {pos - 4}",
                )
            if element_type == _STRING:
                min_size = 8
            elif element_type == _BOOL:
                min_size = 1
            else:
                min_size = _SCALARS[element_type][1]
            self._guard_count(count, min_size, pos, f"elements of array {key!r}")
            values = []
            for _ in range(count):
                value, pos = self._read_value(element_type, pos, key)
                values.append(value)
            return values, pos
        if value_type == _BOOL:
            return bool(f.u8(pos, f"metadata value {key!r}")), pos + 1
        spec = _SCALARS.get(value_type)
        if spec is None:
            raise _f(
                self.path,
                f"metadata {key!r} uses unknown GGUF value type {value_type} "
                f"at offset {pos - 4}",
            )
        fmt, size = spec
        return f.unpack(fmt, pos, f"metadata value {key!r}")[0], pos + size

    def _resolve_alignment(self, metadata: Dict[str, Any]) -> int:
        alignment = metadata.get("general.alignment", _DEFAULT_ALIGNMENT)
        if isinstance(alignment, bool) or not isinstance(alignment, int):
            raise _f(
                self.path,
                f"general.alignment must be an integer, got "
                f"{type(alignment).__name__}",
            )
        if alignment <= 0 or (alignment & (alignment - 1)) != 0:
            raise _f(
                self.path,
                f"general.alignment must be a positive power of two, got {alignment}",
            )
        return alignment

    def _build_entry(
        self, name: str, dims: List[int], type_id: int, offset: int
    ) -> _Entry:
        ggml = _GGML_TYPES.get(type_id)
        if ggml is None:
            known = _GGML_TYPE_NAMES.get(type_id)
            label = f"{type_id} ({known})" if known else f"{type_id} (unknown)"
            raise _f(
                self.path,
                f"tensor {name!r} uses unsupported ggml type id {label}; "
                f"supported: {[t.name for t in _GGML_TYPES.values()]}",
            )
        numel = 1
        for dim in dims:
            numel *= int(dim)
        if numel % ggml.block_size != 0:
            raise _f(
                self.path,
                f"tensor {name!r} has {numel} elements ({dims}) with ggml type "
                f"{ggml.name}, not a multiple of its block size "
                f"{ggml.block_size}",
            )
        nbytes = (numel // ggml.block_size) * ggml.type_size
        info = TensorInfo(
            name=name,
            dtype=ggml.dtype_name,
            shape=tuple(reversed(dims)),
            nbytes=nbytes,
            source_dtype=ggml.name,
        )
        return _Entry(info, ggml, offset)

    def _check_layout(self) -> None:
        ranged = []
        for name, entry in self._entries.items():
            end = entry.offset + entry.info.nbytes
            if end > self._data_len:
                raise _f(
                    self.path,
                    f"tensor {name!r} claims data range [{entry.offset}, "
                    f"{end}) but the data section is only {self._data_len} bytes",
                )
            if entry.info.nbytes:
                ranged.append((entry.offset, end, name))
        ranged.sort()
        prev_end = 0
        prev_name: Optional[str] = None
        for begin, end, name in ranged:
            if begin < prev_end:
                raise _f(
                    self.path,
                    f"tensor {name!r} data range [{begin}, {end}) overlaps "
                    f"tensor {prev_name!r} ending at {prev_end}",
                )
            prev_end, prev_name = end, name

    # -- access -------------------------------------------------------------
    def tensor_names(self) -> List[str]:
        """Tensor names in header order."""
        return list(self._entries)

    def info(self, name: str) -> TensorInfo:
        """Descriptor (dtype/shape/stored byte size) for ``name``."""
        return self._entry(name).info

    def raw(self, name: str) -> np.ndarray:
        """The stored payload of ``name`` as a 1-D ``uint8`` array.

        No dequantization is applied.  With ``mmap=True`` this is a read-only
        view over the file; with ``mmap=False`` it is an independent copy.
        """
        entry = self._entry(name)
        return self._payload(entry, copy=not self._mmap)

    def tensor(self, name: str) -> Tensor:
        """Materialize ``name`` as a host :class:`~evoxc.core.Tensor`.

        F32/F16 return a view in their own dtype (zero-copy under mmap), BF16
        is widened to float32 and every block-quantized type is dequantized to
        a fresh dense float32 array.
        """
        entry = self._entry(name)
        payload = self._payload(entry, copy=False)
        ggml = entry.ggml
        if ggml.dequant is None:
            array = payload.view(ggml.dtype).reshape(entry.info.shape)
            if not self._mmap:
                array = np.array(array, copy=True)
            return Tensor(array)
        nblocks = entry.info.nbytes // ggml.type_size
        blocks = payload.reshape(nblocks, ggml.type_size)
        dense = ggml.dequant(blocks).reshape(entry.info.shape)
        return Tensor(np.ascontiguousarray(dense))

    def close(self) -> None:
        """Release the file handle/mapping (idempotent, never raises)."""
        self._file.close()

    def __contains__(self, name) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __enter__(self) -> "GGUFFile":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"GGUFFile({self.path!r}, version={self.version}, "
            f"tensors={len(self._entries)}, metadata_keys={len(self.metadata)}, "
            f"alignment={self.alignment})"
        )

    # -- internals ----------------------------------------------------------
    def _entry(self, name: str) -> _Entry:
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(
                f"{self.path}: no tensor named {name!r} "
                f"(available: {sorted(self._entries)})"
            )
        return entry

    def _payload(self, entry: _Entry, copy: bool) -> np.ndarray:
        return self._file.byte_view(
            self._data_start + entry.offset,
            entry.info.nbytes,
            f"tensor {entry.info.name!r}",
            copy=copy,
        )


def load_gguf(path, *, mmap: bool = True) -> GGUFFile:
    """Open and validate a GGUF v2/v3 file.

    Args:
        path: Path to the ``.gguf`` file.
        mmap: ``True`` (default) memory-maps the file; ``False`` reads eagerly
            and returns independent writable arrays.

    Returns:
        A validated :class:`GGUFFile`.  Close it (or use it as a context
        manager) when done.
    """
    return GGUFFile(path, mmap=mmap)
