"""Independent byte-level builders and reference decoders for ``tests/formats``.

Nothing in this module imports ``evoxc``.  Every container is encoded here from
the published format description (the safetensors JSON-header layout and the
GGUF v2/v3 header/table layout) and every GGUF block layout is decoded with a
straightforward per-element loop written from the GGML block definitions, so the
assertions in ``test_safetensors.py`` / ``test_gguf.py`` cross-check the reader
instead of re-running its own vectorized code.

Two deliberate simplifications keep the reference values **bit-exact**:

* block quantized payloads are random bytes whose f16 scale fields are
  overwritten with small powers of two, so every scale/dot product is exact in
  float32 and the dequantized values can be compared with ``assert_array_equal``;
* the reference quantizers (Q4_0/Q5_0/Q8_0) follow the GGML reference rounding
  and are only used for approximate round-trip sanity checks.

Import from tests as ``from tests.formats._builders import ...``.
"""
from __future__ import annotations

import json
import struct

import numpy as np

# ---------------------------------------------------------------------------
# safetensors
# ---------------------------------------------------------------------------

#: safetensors dtype tag -> numpy dtype used for the payload bytes.
SAFETENSORS_DTYPES = {
    "F64": "<f8",
    "F32": "<f4",
    "F16": "<f2",
    "BF16": "<u2",
    "I64": "<i8",
    "I32": "<i4",
    "I16": "<i2",
    "I8": "i1",
    "U64": "<u8",
    "U32": "<u4",
    "U16": "<u2",
    "U8": "u1",
    "BOOL": "?",
    "F8_E4M3": "u1",
    "F8_E5M2": "u1",
}

#: the tags the writer accepts (BF16 / F8 are read-only in v1).
SAFETENSORS_WRITABLE_TAGS = tuple(
    tag for tag in SAFETENSORS_DTYPES if tag not in ("BF16", "F8_E4M3", "F8_E5M2")
)


def encode_safetensors_header(header) -> bytes:
    """Encode a header object exactly as the canonical writer does."""
    return json.dumps(
        header, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def raw_safetensors(header_bytes: bytes, data: bytes = b"", *, length=None) -> bytes:
    """Wrap pre-encoded header bytes + data (``length`` overrides the u64 field)."""
    declared = len(header_bytes) if length is None else int(length)
    return struct.pack("<Q", declared) + header_bytes + data


def build_safetensors(entries, metadata=None, *, pad=True, data_length=None) -> bytes:
    """Encode a safetensors container from already-validated pieces.

    ``entries`` is a sequence of dicts with ``name``, ``dtype`` (tag), ``shape``
    (row-major) and ``data`` (payload ``bytes`` or a numpy array), plus an
    optional explicit ``data_offsets`` pair.  Without explicit offsets the
    payloads are laid out back to back from 0 with no padding (the reader
    validates extents, not alignment).  ``pad`` pads the header with ASCII
    spaces to an 8-byte multiple; ``data_length`` truncates/sizes the data
    buffer independently of the declared offsets.
    """
    header = {}
    if metadata is not None:
        header["__metadata__"] = dict(metadata)
    payloads = []
    cursor = 0
    for entry in entries:
        data = entry["data"]
        if not isinstance(data, (bytes, bytearray, memoryview)):
            data = np.ascontiguousarray(data).tobytes()
        data = bytes(data)
        offsets = entry.get("data_offsets")
        if offsets is None:
            offsets = [cursor, cursor + len(data)]
        # Keep explicit offsets VERBATIM (a corrupt-file test writes non-ints).
        begin, end = offsets[0], offsets[1]
        header[entry["name"]] = {
            "dtype": entry["dtype"],
            "shape": [int(d) for d in entry["shape"]],
            "data_offsets": [begin, end],
        }
        payloads.append((int(begin), data))
        cursor = max(cursor, int(end))
    header_bytes = encode_safetensors_header(header)
    if pad:
        header_bytes += b" " * ((-len(header_bytes)) % 8)
    buffer = bytearray(cursor if data_length is None else int(data_length))
    for begin, data in payloads:
        buffer[begin : begin + len(data)] = data
    return raw_safetensors(header_bytes, bytes(buffer))


def parse_safetensors(blob: bytes):
    """Return ``(header, data_start, data)`` parsed straight from container bytes."""
    (header_len,) = struct.unpack_from("<Q", blob, 0)
    header = json.loads(blob[8 : 8 + header_len].decode("utf-8"))
    return header, 8 + header_len, blob[8 + header_len :]


# ---------------------------------------------------------------------------
# GGUF: type ids and block geometry
# ---------------------------------------------------------------------------

GGML_TYPE_IDS = {
    "F32": 0,
    "F16": 1,
    "Q4_0": 2,
    "Q4_1": 3,
    "Q5_0": 6,
    "Q5_1": 7,
    "Q8_0": 8,
    "Q8_1": 9,
    "Q2_K": 10,
    "Q3_K": 11,
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
    "BF16": 30,
}

#: ggml type id -> (block_size, type_size) for the types this reader supports.
GGML_BLOCK = {
    0: (1, 4),
    1: (1, 2),
    2: (32, 18),
    3: (32, 20),
    6: (32, 22),
    7: (32, 24),
    8: (32, 34),
    9: (32, 36),
    10: (256, 84),
    11: (256, 110),
    12: (256, 144),
    13: (256, 176),
    14: (256, 210),
    30: (1, 2),
}

# metadata value type ids (GGUF spec)
UINT8, INT8, UINT16, INT16 = 0, 1, 2, 3
UINT32, INT32, FLOAT32, BOOL = 4, 5, 6, 7
STRING, ARRAY = 8, 9
UINT64, INT64, FLOAT64 = 10, 11, 12

_SCALAR_FMT = {
    UINT8: "<B",
    INT8: "<b",
    UINT16: "<H",
    INT16: "<h",
    UINT32: "<I",
    INT32: "<i",
    FLOAT32: "<f",
    UINT64: "<Q",
    INT64: "<q",
    FLOAT64: "<d",
}


class Encoded:
    """Pre-encoded value bytes, for building deliberately invalid containers."""

    __slots__ = ("raw",)

    def __init__(self, raw: bytes):
        self.raw = bytes(raw)


def encode_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def encode_value(type_id: int, value) -> bytes:
    """Encode one metadata value (arrays as ``(element_type_id, values)``)."""
    if isinstance(value, Encoded):
        return value.raw
    if type_id == BOOL:
        return bytes([1 if value else 0])
    if type_id == STRING:
        return encode_string(str(value))
    if type_id == ARRAY:
        element_type, values = value
        out = struct.pack("<I", element_type) + struct.pack("<Q", len(values))
        return out + b"".join(encode_value(element_type, item) for item in values)
    return struct.pack(_SCALAR_FMT[type_id], value)


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def ne_dims(shape) -> list:
    """GGUF stores dimensions fastest-axis-first, so reverse the numpy shape."""
    return [int(d) for d in reversed(tuple(shape))]


class BuiltGGUF:
    """A built container plus the layout facts tests want to assert on."""

    __slots__ = ("data", "infos_end", "data_start", "tensor_offsets")

    def __init__(self, data, infos_end, data_start, tensor_offsets):
        self.data = data
        self.infos_end = infos_end
        self.data_start = data_start
        self.tensor_offsets = tensor_offsets


def build_gguf_detailed(
    *,
    metadata=(),
    tensors=(),
    version=3,
    alignment=None,
    declared_tensor_count=None,
    declared_kv_count=None,
    pad_data=True,
    trailing=b"",
) -> BuiltGGUF:
    """Build a GGUF v2/v3 container byte by byte.

    Args:
        metadata: ``(key, value_type_id, value)`` triples, written in order.
        tensors: dicts with ``name``, ``ne`` (fastest-axis-first dims),
            ``type_id``, ``payload`` (bytes) and an optional ``offset``.
        version: header version (2 or 3; anything else builds an invalid file).
        alignment: data-section alignment; ``None`` uses the 32-byte default and
            writes no ``general.alignment`` key, an int writes that key too.
        declared_tensor_count / declared_kv_count: override the header counts.
        pad_data: pad the data section to ``alignment`` (skipped automatically
            when there are no tensors, the metadata-only layout).
        trailing: bytes appended after the data section.
    """
    metadata = list(metadata)
    if alignment is not None and not any(
        key == "general.alignment" for key, _type, _value in metadata
    ):
        metadata.append(("general.alignment", UINT32, int(alignment)))
    tensor_count = len(tensors) if declared_tensor_count is None else declared_tensor_count
    kv_count = len(metadata) if declared_kv_count is None else declared_kv_count
    header = b"GGUF" + struct.pack("<I", version)
    header += struct.pack("<Q", tensor_count) + struct.pack("<Q", kv_count)
    for key, type_id, value in metadata:
        header += encode_string(key) + struct.pack("<I", type_id)
        header += encode_value(type_id, value)

    infos = bytearray()
    offsets = {}
    cursor = 0
    payloads = []
    for tensor in tensors:
        dims = [int(d) for d in tensor["ne"]]
        payload = bytes(tensor["payload"])
        offset = int(tensor.get("offset", cursor))
        infos += encode_string(tensor["name"])
        infos += struct.pack("<I", len(dims))
        for dim in dims:
            infos += struct.pack("<Q", dim)
        infos += struct.pack("<I", int(tensor["type_id"]))
        infos += struct.pack("<Q", offset)
        offsets[tensor["name"]] = offset
        payloads.append((offset, payload))
        cursor = max(cursor, offset + len(payload))

    infos_end = len(header) + len(infos)
    effective = 32 if alignment is None else int(alignment)
    if tensors and pad_data:
        data_start = align_up(infos_end, effective)
    else:
        data_start = infos_end
    # A metadata-only file may still be padded when the caller asks for it.
    if not tensors and pad_data:
        data_start = align_up(infos_end, effective)

    data = bytearray(cursor)
    for offset, payload in payloads:
        data[offset : offset + len(payload)] = payload
    padding = b"\x00" * (data_start - infos_end)
    blob = bytes(header) + bytes(infos) + padding + bytes(data) + bytes(trailing)
    return BuiltGGUF(blob, infos_end, data_start, offsets)


def build_gguf(**kwargs) -> bytes:
    """The container bytes only (see :func:`build_gguf_detailed`)."""
    return build_gguf_detailed(**kwargs).data


# ---------------------------------------------------------------------------
# GGUF: payload generation
# ---------------------------------------------------------------------------

#: ggml id -> f16-scale field offsets and base values (powers of two => exact).
_SCALE_FIELDS = {
    2: ((0, 1.0),),
    3: ((0, 1.0), (2, 0.5)),
    6: ((0, 1.0),),
    7: ((0, 1.0), (2, 0.5)),
    8: ((0, 1.0),),
    9: ((0, 1.0), (2, 0.5)),
    10: ((80, 0.5), (82, 0.25)),
    11: ((108, 0.5),),
    12: ((0, 0.5), (2, 0.25)),
    13: ((0, 0.5), (2, 0.25)),
    14: ((208, 0.5),),
}


def _write_f16(blocks: np.ndarray, column: int, values) -> None:
    encoded = np.asarray(values, dtype="<f2").view(np.uint8).reshape(-1, 2)
    blocks[:, column : column + 2] = encoded


def quant_blocks(type_id: int, nblocks: int, seed: int = 0) -> np.ndarray:
    """Random ``(nblocks, type_size)`` uint8 blocks with exact f16 scales."""
    _, type_size = GGML_BLOCK[type_id]
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, size=(nblocks, type_size), dtype=np.uint8)
    for column, base in _SCALE_FIELDS[type_id]:
        values = [base * 2.0 ** (b % 3) for b in range(nblocks)]
        _write_f16(blocks, column, values)
    return blocks


def tensor_payload(type_id: int, shape, seed: int = 0) -> bytes:
    """Encode ``shape`` (numpy row-major) worth of payload for ``type_id``."""
    shape = tuple(int(d) for d in shape)
    numel = 1
    for dim in shape:
        numel *= dim
    rng = np.random.default_rng(seed)
    if type_id == 0:
        return rng.standard_normal(numel).astype("<f4").tobytes()
    if type_id == 1:
        return (rng.standard_normal(numel) * 4).astype("<f2").tobytes()
    if type_id == 30:
        floats = rng.standard_normal(numel).astype("<f4")
        return (floats.view("<u4") >> 16).astype("<u2").tobytes()
    block, _type_size = GGML_BLOCK[type_id]
    if numel % block:
        raise ValueError(f"{numel} elements is not a multiple of {block}")
    return quant_blocks(type_id, numel // block, seed).tobytes()


# ---------------------------------------------------------------------------
# GGUF: independent reference dequantizers (one element loop per layout)
# ---------------------------------------------------------------------------


def _f16(word: bytes, offset: int) -> float:
    return float(np.frombuffer(word[offset : offset + 2], dtype="<f2")[0])


def _i8(value: int) -> int:
    return value - 256 if value > 127 else value


def _deq_q4_0(block: bytes):
    d = _f16(block, 0)
    qs = block[2:18]
    out = [0.0] * 32
    for j in range(16):
        out[j] = d * ((qs[j] & 0x0F) - 8)
        out[j + 16] = d * ((qs[j] >> 4) - 8)
    return out


def _deq_q4_1(block: bytes):
    d = _f16(block, 0)
    m = _f16(block, 2)
    qs = block[4:20]
    out = [0.0] * 32
    for j in range(16):
        out[j] = d * (qs[j] & 0x0F) + m
        out[j + 16] = d * (qs[j] >> 4) + m
    return out


def _deq_q5_common(block: bytes, d: float, m: float, qh_off: int, qs_off: int):
    qh = block[qh_off : qh_off + 4]
    qs = block[qs_off : qs_off + 16]
    out = [0.0] * 32
    for j in range(16):
        xh0 = (qh[j // 8] >> (j % 8)) & 1
        xh1 = (qh[(j + 16) // 8] >> ((j + 16) % 8)) & 1
        out[j] = d * ((qs[j] & 0x0F) | (xh0 << 4)) + m
        out[j + 16] = d * ((qs[j] >> 4) | (xh1 << 4)) + m
    return out


def _deq_q5_0(block: bytes):
    d = _f16(block, 0)
    out = _deq_q5_common(block, d, 0.0, 2, 6)
    return [value - 16 * d for value in out]


def _deq_q5_1(block: bytes):
    d = _f16(block, 0)
    m = _f16(block, 2)
    return _deq_q5_common(block, d, m, 4, 8)


def _deq_q8_0(block: bytes):
    d = _f16(block, 0)
    qs = [_i8(value) for value in block[2:34]]
    return [d * value for value in qs]


def _deq_q8_1(block: bytes):
    # f16 d, f16 s (the pre-computed dot sum, not part of the values), int8 qs.
    d = _f16(block, 0)
    qs = [_i8(value) for value in block[4:36]]
    return [d * value for value in qs]


def _deq_q2_k(block: bytes):
    scales = block[0:16]
    qs = block[16:80]
    d = _f16(block, 80)
    dmin = _f16(block, 82)
    out = [0.0] * 256
    index = 0
    for half in range(2):
        q = qs[half * 32 : (half + 1) * 32]
        shift = 0
        for j in range(4):
            scale = scales[half * 8 + 2 * j]
            dl = d * (scale & 0x0F)
            ml = dmin * (scale >> 4)
            for l in range(16):
                out[index] = dl * ((q[l] >> shift) & 0x03) - ml
                index += 1
            scale = scales[half * 8 + 2 * j + 1]
            dl = d * (scale & 0x0F)
            ml = dmin * (scale >> 4)
            for l in range(16):
                out[index] = dl * ((q[l + 16] >> shift) & 0x03) - ml
                index += 1
            shift += 2
    return out


def _q3_scale(raw: bytes, index: int) -> int:
    """Unpack one 6-bit Q3_K scale (stored value is biased by 32).

    Layout: bytes 0..7 hold the 16 low nibbles (byte j = scales j and j+8),
    bytes 8..11 hold the high 2 bits bit-sliced, so tmp byte k carries scales
    k, k+4, k+8 and k+12 (bits 0-1, 2-3, 4-5, 6-7 respectively).
    """
    group, offset = divmod(index, 4)  # which 2-bit field / which tmp byte
    if group < 2:
        low = raw[4 * group + offset] & 0x0F
    else:
        low = raw[4 * (group - 2) + offset] >> 4
    high = (raw[8 + offset] >> (2 * group)) & 0x03
    return (low | (high << 4)) - 32


def _deq_q3_k(block: bytes):
    hmask = block[0:32]
    qs = block[32:96]
    raw_scales = block[96:108]
    d = _f16(block, 108)
    out = [0.0] * 256
    index = 0
    for half in range(2):
        q = qs[half * 32 : (half + 1) * 32]
        shift = 0
        for j in range(4):
            bit = 1 << (half * 4 + j)
            dl = d * _q3_scale(raw_scales, half * 8 + 2 * j)
            for l in range(16):
                ql = (q[l] >> shift) & 0x03
                out[index] = dl * (ql - (0 if hmask[l] & bit else 4))
                index += 1
            dl = d * _q3_scale(raw_scales, half * 8 + 2 * j + 1)
            for l in range(16):
                ql = (q[l + 16] >> shift) & 0x03
                out[index] = dl * (ql - (0 if hmask[l + 16] & bit else 4))
                index += 1
            shift += 2
    return out


def _k_scale_min(raw: bytes, index: int):
    """Q4_K/Q5_K ``get_scale_min_k4``: unpack one (scale, min) byte pair."""
    if index < 4:
        return raw[index] & 0x3F, raw[index + 4] & 0x3F
    scale = (raw[index + 4] & 0x0F) | (((raw[index - 4] >> 6) & 0x03) << 4)
    minimum = (raw[index + 4] >> 4) | (((raw[index] >> 6) & 0x03) << 4)
    return scale, minimum


def _deq_q4_k(block: bytes):
    d = _f16(block, 0)
    dmin = _f16(block, 2)
    raw_scales = block[4:16]
    qs = block[16:144]
    out = [0.0] * 256
    for group in range(4):
        q = qs[group * 32 : (group + 1) * 32]
        scale, minimum = _k_scale_min(raw_scales, 2 * group)
        dl, ml = d * scale, dmin * minimum
        for l in range(32):
            out[group * 64 + l] = dl * (q[l] & 0x0F) - ml
        scale, minimum = _k_scale_min(raw_scales, 2 * group + 1)
        dl, ml = d * scale, dmin * minimum
        for l in range(32):
            out[group * 64 + 32 + l] = dl * (q[l] >> 4) - ml
    return out


def _deq_q5_k(block: bytes):
    d = _f16(block, 0)
    dmin = _f16(block, 2)
    raw_scales = block[4:16]
    qh = block[16:48]
    qs = block[48:176]
    out = [0.0] * 256
    for group in range(4):
        q = qs[group * 32 : (group + 1) * 32]
        scale, minimum = _k_scale_min(raw_scales, 2 * group)
        dl, ml = d * scale, dmin * minimum
        for l in range(32):
            value = (q[l] & 0x0F) + (16 if qh[l] & (1 << (2 * group)) else 0)
            out[group * 64 + l] = dl * value - ml
        scale, minimum = _k_scale_min(raw_scales, 2 * group + 1)
        dl, ml = d * scale, dmin * minimum
        for l in range(32):
            value = (q[l] >> 4) + (16 if qh[l] & (1 << (2 * group + 1)) else 0)
            out[group * 64 + 32 + l] = dl * value - ml
    return out


def _deq_q6_k(block: bytes):
    ql = block[0:128]
    qh = block[128:192]
    scales = [_i8(value) for value in block[192:208]]
    d = _f16(block, 208)
    out = [0.0] * 256
    for half in range(2):
        base = half * 128
        ql_half = ql[half * 64 : (half + 1) * 64]
        qh_half = qh[half * 32 : (half + 1) * 32]
        for l in range(32):
            is_ = l // 16
            q1 = ((ql_half[l] & 0x0F) | (((qh_half[l] >> 0) & 0x03) << 4)) - 32
            q2 = ((ql_half[l + 32] & 0x0F) | (((qh_half[l] >> 2) & 0x03) << 4)) - 32
            q3 = ((ql_half[l] >> 4) | (((qh_half[l] >> 4) & 0x03) << 4)) - 32
            q4 = ((ql_half[l + 32] >> 4) | (((qh_half[l] >> 6) & 0x03) << 4)) - 32
            base_scale = half * 8 + is_
            out[base + l] = d * scales[base_scale + 0] * q1
            out[base + l + 32] = d * scales[base_scale + 2] * q2
            out[base + l + 64] = d * scales[base_scale + 4] * q3
            out[base + l + 96] = d * scales[base_scale + 6] * q4
    return out


def _deq_bf16(word: bytes):
    (value,) = struct.unpack("<H", word)
    return float(np.frombuffer(struct.pack("<I", value << 16), dtype="<f4")[0])


_REF_DEQUANT = {
    2: _deq_q4_0,
    3: _deq_q4_1,
    6: _deq_q5_0,
    7: _deq_q5_1,
    8: _deq_q8_0,
    9: _deq_q8_1,
    10: _deq_q2_k,
    11: _deq_q3_k,
    12: _deq_q4_k,
    13: _deq_q5_k,
    14: _deq_q6_k,
}


def reference_values(type_id: int, payload: bytes, shape):
    """The dense array ``GGUFFile.tensor()`` must produce for this payload."""
    shape = tuple(int(d) for d in shape)
    numel = 1
    for dim in shape:
        numel *= dim
    if type_id == 0:
        return np.frombuffer(payload, dtype="<f4", count=numel).reshape(shape)
    if type_id == 1:
        return np.frombuffer(payload, dtype="<f2", count=numel).reshape(shape)
    if type_id == 30:
        words = np.frombuffer(payload, dtype="<u2", count=numel)
        widened = (words.astype(np.uint32) << 16).view(np.float32)
        return widened.reshape(shape)
    block, type_size = GGML_BLOCK[type_id]
    raw = np.frombuffer(payload, dtype=np.uint8)
    nblocks = numel // block
    out = np.empty(numel, dtype=np.float32)
    dequant = _REF_DEQUANT[type_id]
    for index in range(nblocks):
        chunk = raw[index * type_size : (index + 1) * type_size].tobytes()
        out[index * block : (index + 1) * block] = dequant(chunk)
    return out.reshape(shape)


# ---------------------------------------------------------------------------
# GGUF: reference quantizers (round-trip sanity only)
# ---------------------------------------------------------------------------


def _f16_bytes(value: float) -> bytes:
    return np.asarray([value], dtype="<f2").tobytes()


def quantize_q4_0(values) -> bytes:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size % 32:
        raise ValueError("Q4_0 needs a multiple of 32 elements")
    out = bytearray()
    for start in range(0, flat.size, 32):
        block = flat[start : start + 32]
        amax = float(np.max(np.abs(block))) if block.size else 0.0
        d = float(np.float16(amax / 7.0)) if amax else 0.0
        inverse = 0.0 if d == 0.0 else 1.0 / d
        qs = bytearray(16)
        for j in range(16):
            low = min(15, int(float(block[j]) * inverse + 8.5))
            high = min(15, int(float(block[j + 16]) * inverse + 8.5))
            qs[j] = low | (high << 4)
        out += _f16_bytes(d) + qs
    return bytes(out)


def quantize_q5_0(values) -> bytes:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size % 32:
        raise ValueError("Q5_0 needs a multiple of 32 elements")
    out = bytearray()
    for start in range(0, flat.size, 32):
        block = flat[start : start + 32]
        amax = float(np.max(np.abs(block))) if block.size else 0.0
        d = float(np.float16(amax / 15.0)) if amax else 0.0
        inverse = 0.0 if d == 0.0 else 1.0 / d
        packed = [min(31, int(float(entry) * inverse + 16.5)) for entry in block]
        qs = bytearray(16)
        for j in range(16):
            qs[j] = (packed[j] & 0x0F) | ((packed[j + 16] & 0x0F) << 4)
        qh = 0
        for j in range(16):
            qh |= ((packed[j] >> 4) & 1) << j
            qh |= ((packed[j + 16] >> 4) & 1) << (j + 16)
        out += _f16_bytes(d) + struct.pack("<I", qh) + qs
    return bytes(out)


def quantize_q8_0(values) -> bytes:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size % 32:
        raise ValueError("Q8_0 needs a multiple of 32 elements")
    out = bytearray()
    for start in range(0, flat.size, 32):
        block = flat[start : start + 32]
        amax = float(np.max(np.abs(block))) if block.size else 0.0
        d = float(np.float16(amax / 127.0)) if amax else 0.0
        inverse = 0.0 if d == 0.0 else 1.0 / d
        quants = np.clip(np.rint(block * inverse), -127, 127).astype(np.int8)
        out += _f16_bytes(d) + quants.tobytes()
    return bytes(out)


REFERENCE_QUANTIZERS = {
    2: quantize_q4_0,
    6: quantize_q5_0,
    8: quantize_q8_0,
}
