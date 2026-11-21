"""Tests for ``evoxc.formats.gguf`` — GGUF v2/v3 container + dequantization.

The binding contract is ``evoxc/formats/CONTEXT.md``.  Every fixture is built
byte-by-byte by ``tests/formats/_builders.py`` (an encoder written from the GGUF
format description) and every quantized value is checked against that module's
INDEPENDENT per-element reference dequantizer — nothing here reuses evoxc's own
vectorized block code, so a transposed nibble order or a wrong `qh` bit position
fails the comparison.  f16 scales in the generated blocks are small powers of
two, which makes every reference value exact in float32, so the type table is
pinned with ``assert_array_equal`` (bit-exact) rather than a tolerance.

Covered: all 14 supported ggml types, metadata value types (scalars, bools,
strings and arrays of each incl. an empty array), alignment variants, metadata-
only and multi-tensor files, mmap vs eager-copy semantics, the documented
rejections (bad magic/version, absurd counts, unknown value/type ids, payloads
outside or overlapping the data section, ``n_dims`` > 4, non-block-multiple
element counts, bad alignment, string-length overflow, nested arrays) and
reference-quantizer round trips for Q4_0/Q5_0/Q8_0.
"""
from __future__ import annotations

import glob
import os
import struct

import numpy as np
import pytest

from evoxc.core import PersistenceError, Tensor
from evoxc.formats import GGUFFile, TensorInfo, load_gguf
from tests.formats._builders import (
    ARRAY,
    BOOL,
    FLOAT32,
    FLOAT64,
    GGML_BLOCK,
    GGML_TYPE_IDS,
    INT8,
    INT16,
    INT32,
    INT64,
    REFERENCE_QUANTIZERS,
    STRING,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    Encoded,
    align_up,
    build_gguf,
    build_gguf_detailed,
    ne_dims,
    reference_values,
    tensor_payload,
)

#: ggml id -> the canonical dtype name ``TensorInfo.dtype`` must report.
CANONICAL_DTYPE = {
    0: "float32",
    1: "float16",
    30: "BF16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
}

SUPPORTED = sorted(GGML_TYPE_IDS.items(), key=lambda item: item[1])

_REAL_GGUF_PATTERNS = (
    "~/Source/*/stories15M-q4_0.gguf",
    "~/Source/*/*/stories15M-q4_0.gguf",
    "~/models/*.gguf",
    "~/gguf/*.gguf",
    "~/models/*/*.gguf",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _tensor(name, type_id, shape, *, seed=0, offset=None):
    payload = tensor_payload(type_id, shape, seed)
    entry = {
        "name": name,
        "ne": ne_dims(shape),
        "type_id": type_id,
        "payload": payload,
    }
    if offset is not None:
        entry["offset"] = offset
    return entry, payload


def _write(tmp_path, blob, name="m.gguf"):
    path = tmp_path / name
    path.write_bytes(blob)
    return path


@pytest.fixture
def file_of(tmp_path):
    """Build + load one container; closes it at teardown."""

    opened = []

    def factory(*, metadata=(), tensors=(), version=3, alignment=None, mmap=False):
        blob = build_gguf(metadata=metadata, tensors=tensors, version=version, alignment=alignment)
        handle = load_gguf(_write(tmp_path, blob), mmap=mmap)
        opened.append(handle)
        return handle

    yield factory
    for handle in opened:
        handle.close()


# ---------------------------------------------------------------------------
# supported ggml types
# ---------------------------------------------------------------------------


class TestSupportedTypes:
    @pytest.mark.parametrize("name,type_id", SUPPORTED, ids=[n for n, _ in SUPPORTED])
    def test_known_value_and_descriptor(self, file_of, name, type_id):
        block, type_size = GGML_BLOCK[type_id]
        shape = (2, 3) if block == 1 else (2, block)
        entry, payload = _tensor("w", type_id, shape)
        f = file_of(tensors=[entry])

        info = f.info("w")
        assert isinstance(info, TensorInfo)
        assert info.name == "w"
        assert info.source_dtype == name
        assert info.dtype == CANONICAL_DTYPE[type_id]
        assert info.shape == shape
        numel = 1
        for dim in shape:
            numel *= dim
        assert info.numel == numel
        assert info.nbytes == (numel // block) * type_size

        tensor = f.tensor("w")
        assert isinstance(tensor, Tensor)
        expected = reference_values(type_id, payload, shape)
        np.testing.assert_array_equal(tensor.numpy(), expected)
        assert tensor.numpy().shape == info.shape

    @pytest.mark.parametrize("type_id", [0, 1, 8, 12, 14, 30])
    def test_multi_dim_shapes_are_reversed(self, file_of, type_id):
        block, _ = GGML_BLOCK[type_id]
        shape = (3, 2, 1) if block == 1 else (3, 2, block)
        entry, payload = _tensor("w", type_id, shape)
        f = file_of(tensors=[entry])
        assert f.info("w").shape == shape
        np.testing.assert_array_equal(
            f.tensor("w").numpy(), reference_values(type_id, payload, shape)
        )

    def test_scalar_rank_zero_tensor(self, file_of):
        entry, payload = _tensor("w", 0, ())
        f = file_of(tensors=[entry])
        assert f.info("w").shape == ()
        assert f.info("w").rank == 0
        np.testing.assert_array_equal(f.tensor("w").numpy(), reference_values(0, payload, ()))

    @pytest.mark.parametrize("type_id", [2, 6, 8, 10, 12, 14])
    def test_quantized_nbytes_is_the_stored_payload_size(self, file_of, type_id):
        block, type_size = GGML_BLOCK[type_id]
        shape = (5, block)
        entry, payload = _tensor("w", type_id, shape)
        f = file_of(tensors=[entry])
        info = f.info("w")
        assert info.nbytes == (5 * block // block) * type_size == len(payload)
        assert f.raw("w").nbytes == info.nbytes


class TestRawAndViews:
    @pytest.mark.parametrize("type_id", [0, 14])
    def test_raw_is_the_stored_bytes(self, file_of, type_id):
        block, _ = GGML_BLOCK[type_id]
        shape = (4,) if block == 1 else (4, block)
        entry, payload = _tensor("rawme", type_id, shape)
        f = file_of(tensors=[entry])
        raw = f.raw("rawme")
        assert isinstance(raw, np.ndarray)
        assert raw.dtype == np.uint8
        assert raw.ndim == 1
        assert raw.tobytes() == payload

    def test_f32_and_f16_are_views_in_their_own_dtype(self, file_of):
        f32, _ = _tensor("f32", 0, (4,))
        f16, _ = _tensor("f16", 1, (4,))
        f = file_of(tensors=[f32, f16], mmap=True)
        for name, dtype in (("f32", "float32"), ("f16", "float16")):
            host = f.tensor(name).numpy()
            assert host.dtype == np.dtype(dtype)
            assert not host.flags.owndata  # zero-copy view of the mapped file
            assert not host.flags.writeable
            with pytest.raises(ValueError):
                host[0] = 0

    def test_bf16_widens_to_float32(self, file_of):
        entry, payload = _tensor("bf", 30, (4,))
        f = file_of(tensors=[entry])
        widened = f.tensor("bf").numpy()
        assert widened.dtype == np.dtype("float32")
        np.testing.assert_array_equal(widened, reference_values(30, payload, (4,)))
        # exact widening: each float32 is the u16 word shifted into the high half
        words = np.frombuffer(payload, dtype="<u2")
        np.testing.assert_array_equal(
            widened.view(np.uint32) >> 16, words.astype(np.uint32)
        )

    @pytest.mark.parametrize("type_id", [2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    def test_quantized_types_dequantize_to_a_fresh_float32_array(self, file_of, type_id):
        block, _ = GGML_BLOCK[type_id]
        entry, _payload = _tensor("q", type_id, (2, block))
        f = file_of(tensors=[entry])
        dense = f.tensor("q").numpy()
        assert dense.dtype == np.dtype("float32")
        assert dense.flags.writeable
        original = float(dense[0, 0])
        dense[0, 0] = original + 1.0  # a fresh array: never aliases the file
        assert float(f.tensor("q").numpy()[0, 0]) == original


# ---------------------------------------------------------------------------
# mmap vs eager copy
# ---------------------------------------------------------------------------


class TestMmapSemantics:
    def test_mmap_views_are_read_only(self, file_of):
        entry, _ = _tensor("w", 0, (4,))
        raw_entry, _ = _tensor("q", 8, (32,))
        f = file_of(tensors=[entry, raw_entry], mmap=True)
        host = f.tensor("w").numpy()
        assert not host.flags.writeable
        assert not host.flags.owndata
        with pytest.raises(ValueError):
            host[0] = 1.0
        raw = f.raw("w")
        assert not raw.flags.writeable
        with pytest.raises(ValueError):
            raw[0] = 0
        # a dequantized payload is always a fresh writable array
        assert f.tensor("q").numpy().flags.writeable

    def test_mmap_false_returns_independent_writable_copies(self, tmp_path):
        entry, payload = _tensor("w", 0, (4,))
        path = _write(tmp_path, build_gguf(tensors=[entry]))
        before = path.read_bytes()
        with load_gguf(path, mmap=False) as f:
            first = f.tensor("w").numpy()
            second = f.tensor("w").numpy()
            assert first.flags.writeable and first.flags.owndata
            assert first.tobytes() == payload
            first[0] = -1.0
            assert second[0] != -1.0
            raw = f.raw("w")
            assert raw.flags.writeable and raw.flags.owndata
            raw[0] = 0xFF
            assert f.raw("w")[0] != 0xFF
        assert path.read_bytes() == before

    def test_zero_element_tensors(self, file_of):
        plain, _ = _tensor("empty_f32", 0, (0,))
        quant, _ = _tensor("empty_q4", 2, (0,))
        f = file_of(tensors=[plain, quant])
        assert f.info("empty_f32").shape == (0,)
        assert f.info("empty_f32").nbytes == 0
        assert f.info("empty_q4").shape == (0,)
        assert f.info("empty_q4").nbytes == 0
        assert f.tensor("empty_f32").numpy().shape == (0,)
        assert f.tensor("empty_q4").numpy().shape == (0,)
        assert f.tensor("empty_q4").numpy().dtype == np.dtype("float32")
        assert f.raw("empty_q4").size == 0

    def test_tensor_info_shape_matches_materialized_shape(self, file_of):
        entry, _ = _tensor("q", 12, (2, 3, 256))
        f = file_of(tensors=[entry])
        assert f.tensor("q").numpy().shape == f.info("q").shape == (2, 3, 256)


# ---------------------------------------------------------------------------
# metadata KV table
# ---------------------------------------------------------------------------


class TestMetadata:
    @pytest.mark.parametrize(
        "type_id,value",
        [
            (UINT8, 200),
            (INT8, -7),
            (UINT16, 65530),
            (INT16, -300),
            (UINT32, 4000000000),
            (INT32, -2000000000),
            (FLOAT32, 2.25),
            (FLOAT64, -1.5e300),
            (UINT64, 2**63 + 5),
            (INT64, -(2**40)),
            (BOOL, True),
            (STRING, "héllo wörld"),
        ],
        ids=[
            "u8",
            "i8",
            "u16",
            "i16",
            "u32",
            "i32",
            "f32",
            "f64",
            "u64",
            "i64",
            "bool",
            "string",
        ],
    )
    def test_scalar_value_types(self, file_of, type_id, value):
        f = file_of(metadata=[("k", type_id, value)])
        stored = f.metadata["k"]
        assert stored == value
        if type_id == BOOL:
            assert isinstance(stored, bool)
        elif type_id == STRING:
            assert isinstance(stored, str)
        elif type_id == FLOAT32 or type_id == FLOAT64:
            assert isinstance(stored, float)
        else:
            assert isinstance(stored, int) and not isinstance(stored, bool)

    @pytest.mark.parametrize(
        "element_type,values",
        [
            (UINT8, [0, 1, 255]),
            (INT8, [-128, 0, 127]),
            (UINT16, [0, 65535]),
            (INT16, [-32768, 7]),
            (UINT32, [0, 2**32 - 1]),
            (INT32, [-2**31, 42]),
            (FLOAT32, [0.5, -2.25, 8.0]),
            (FLOAT64, [1.5, -1e100]),
            (UINT64, [0, 2**64 - 1]),
            (INT64, [-2**63, 9]),
            (BOOL, [True, False, True]),
            (STRING, ["alpha", "", "β"]),
        ],
        ids=["u8", "i8", "u16", "i16", "u32", "i32", "f32", "f64", "u64", "i64", "bool", "str"],
    )
    def test_array_value_types(self, file_of, element_type, values):
        f = file_of(metadata=[("arr", ARRAY, (element_type, values))])
        stored = f.metadata["arr"]
        assert stored == list(values)
        assert isinstance(stored, list)
        if element_type == STRING:
            assert all(isinstance(item, str) for item in stored)
        elif element_type == BOOL:
            assert all(isinstance(item, bool) for item in stored)

    def test_empty_arrays(self, file_of):
        f = file_of(
            metadata=[
                ("ints", ARRAY, (UINT32, [])),
                ("strs", ARRAY, (STRING, [])),
                ("bools", ARRAY, (BOOL, [])),
            ]
        )
        assert f.metadata == {"ints": [], "strs": [], "bools": []}

    def test_metadata_with_no_entries(self, file_of):
        entry, _ = _tensor("w", 0, (2,))
        f = file_of(metadata=[], tensors=[entry])
        assert f.metadata == {}
        assert f.alignment == 32

    def test_version_and_repr(self, file_of):
        entry, _ = _tensor("w", 0, (2,))
        for version in (2, 3):
            f = file_of(tensors=[entry], version=version)
            assert f.version == version
            assert isinstance(f, GGUFFile)
            assert "version=%d" % version in repr(f)

    def test_nested_arrays_are_rejected(self, tmp_path):
        raw_value = struct.pack("<I", ARRAY) + struct.pack("<Q", 2)
        blob = build_gguf(metadata=[("bad", ARRAY, Encoded(raw_value))])
        _rejects(tmp_path, blob, "'bad'", "array of arrays")

    def test_duplicate_metadata_key_is_rejected(self, tmp_path):
        blob = build_gguf(
            metadata=[("k", UINT32, 1), ("k", UINT32, 2)]
        )
        _rejects(tmp_path, blob, "duplicate metadata key 'k'")

    def test_duplicate_tensor_name_is_rejected(self, tmp_path):
        first, _ = _tensor("w", 0, (1,))
        second, _ = _tensor("w", 0, (1,), offset=4)
        blob = build_gguf(tensors=[first, second])
        _rejects(tmp_path, blob, "duplicate tensor name 'w'")


# ---------------------------------------------------------------------------
# alignment and layout
# ---------------------------------------------------------------------------


class TestAlignment:
    @pytest.mark.parametrize("alignment", [1, 16, 32, 64, 256])
    def test_alignment_variants(self, tmp_path, alignment):
        first, payload_a = _tensor("a", 0, (3,))
        second, payload_b = _tensor("b", 1, (5,), seed=1)
        built = build_gguf_detailed(
            tensors=[first, second],
            alignment=alignment,
            metadata=[("general.alignment", UINT32, alignment)],
        )
        assert built.data_start % alignment == 0
        assert built.data_start == align_up(built.infos_end, alignment)
        path = _write(tmp_path, built.data, f"align{alignment}.gguf")
        with load_gguf(path) as f:
            assert f.alignment == alignment
            assert f.raw("a").tobytes() == payload_a
            assert f.raw("b").tobytes() == payload_b
            np.testing.assert_array_equal(
                f.tensor("b").numpy(), reference_values(1, payload_b, (5,))
            )

    def test_default_alignment_without_the_key(self, tmp_path):
        first, _ = _tensor("a", 0, (2,))
        built = build_gguf_detailed(tensors=[first])
        assert built.data_start == align_up(built.infos_end, 32)
        with load_gguf(_write(tmp_path, built.data)) as f:
            assert f.alignment == 32

    def test_metadata_only_file_without_data_padding(self, tmp_path):
        blob = build_gguf(
            metadata=[
                ("tokenizer.ggml.model", STRING, "llama"),
                ("tokenizer.ggml.tokens", ARRAY, (STRING, ["a", "b", "c"])),
            ]
        )
        with load_gguf(_write(tmp_path, blob)) as f:
            assert len(f) == 0
            assert f.tensor_names() == []
            assert f.metadata["tokenizer.ggml.tokens"] == ["a", "b", "c"]
            assert "GGUFFile" in repr(f)

    def test_metadata_only_file_with_padding_is_accepted_too(self, tmp_path):
        # The padding is optional for a tensors-free file, not forbidden.
        built = build_gguf_detailed(
            metadata=[("tokenizer.ggml.model", STRING, "llama")],
            alignment=64,
            pad_data=True,
        )
        assert built.data_start == align_up(built.infos_end, 64)
        assert built.data_start > built.infos_end
        with load_gguf(_write(tmp_path, built.data)) as f:
            assert len(f) == 0
            assert f.alignment == 64
            assert f.metadata["tokenizer.ggml.model"] == "llama"

    def test_per_tensor_offsets_need_not_be_aligned(self, tmp_path):
        # F16 (2 bytes) at 0, F32 at 3: valid, just unaligned.
        first, _ = _tensor("a", 1, (1,))
        second, payload_b = _tensor("b", 0, (1,), offset=3)
        built = build_gguf_detailed(tensors=[first, second])
        assert built.tensor_offsets["b"] == 3
        with load_gguf(_write(tmp_path, built.data)) as f:
            np.testing.assert_array_equal(
                f.tensor("b").numpy(), reference_values(0, payload_b, (1,))
            )

    def test_multi_tensor_file_and_name_order(self, tmp_path):
        entries = []
        payloads = {}
        for index, (name, type_id, shape) in enumerate(
            [("z.weight", 0, (2, 2)), ("a.bias", 12, (256,)), ("m.scale", 1, (4,))]
        ):
            entry, payload = _tensor(name, type_id, shape, seed=index)
            entries.append(entry)
            payloads[name] = payload
        with load_gguf(_write(tmp_path, build_gguf(tensors=entries))) as f:
            assert f.tensor_names() == ["z.weight", "a.bias", "m.scale"]
            assert list(f) == f.tensor_names()
            assert len(f) == 3
            assert "a.bias" in f and "nope" not in f
            for name, payload in payloads.items():
                assert f.raw(name).tobytes() == payload
            with pytest.raises(KeyError, match="no tensor named 'nope'"):
                f.info("nope")
            with pytest.raises(KeyError, match="no tensor named 'nope'"):
                f.tensor("nope")
            with pytest.raises(KeyError, match="no tensor named 'nope'"):
                f.raw("nope")

    def test_context_manager_and_close(self, tmp_path):
        entry, payload = _tensor("w", 0, (2,))
        path = _write(tmp_path, build_gguf(tensors=[entry]))
        with load_gguf(path) as f:
            assert f.raw("w").tobytes() == payload
        with pytest.raises(PersistenceError, match="file is closed"):
            f.tensor("w")
        f.close()  # idempotent, never raises


# ---------------------------------------------------------------------------
# rejections
# ---------------------------------------------------------------------------


def _rejects(tmp_path, blob, *fragments):
    path = _write(tmp_path, blob)
    with pytest.raises(PersistenceError) as excinfo:
        load_gguf(path)
    message = str(excinfo.value)
    assert str(path) in message
    for fragment in fragments:
        assert fragment in message, message
    return message


def _patched(blob, fmt, offset, value):
    buf = bytearray(blob)
    struct.pack_into(fmt, buf, offset, value)
    return bytes(buf)


class TestRejections:
    def test_too_short(self, tmp_path):
        _rejects(tmp_path, b"GGUF", "too short for a GGUF header")

    def test_bad_magic(self, tmp_path):
        blob = build_gguf()
        _rejects(tmp_path, _patched(blob, "4s", 0, b"GGUX"), "not a GGUF file", "GGUX")

    @pytest.mark.parametrize("version", [1, 4])
    def test_unsupported_version(self, tmp_path, version):
        blob = _patched(build_gguf(), "<I", 4, version)
        _rejects(tmp_path, blob, f"unsupported GGUF version {version}")

    def test_absurd_tensor_count(self, tmp_path):
        blob = _patched(build_gguf(), "<Q", 8, 10**9)
        _rejects(tmp_path, blob, "tensor infos", "cannot fit in the remaining")

    def test_absurd_metadata_count(self, tmp_path):
        blob = _patched(build_gguf(), "<Q", 16, 10**9)
        _rejects(tmp_path, blob, "metadata entries", "cannot fit in the remaining")

    def test_unknown_metadata_value_type(self, tmp_path):
        entry, _ = _tensor("w", 0, (1,))
        blob = build_gguf(metadata=[("bad", UINT32, 1)], tensors=[entry])
        key_end = 24 + 8 + len(b"bad")
        _rejects(tmp_path, _patched(blob, "<I", key_end, 99), "'bad'", "unknown GGUF value type 99")

    def test_unknown_metadata_array_element_type(self, tmp_path):
        raw_value = struct.pack("<I", 99) + struct.pack("<Q", 1)
        blob = build_gguf(metadata=[("arr", ARRAY, Encoded(raw_value))])
        _rejects(tmp_path, blob, "'arr'", "unknown array element type 99")

    @pytest.mark.parametrize(
        "type_id,label", [(15, "Q8_K"), (16, "IQ2_XXS"), (999, "unknown")]
    )
    def test_unsupported_ggml_type_ids(self, tmp_path, type_id, label):
        blob = build_gguf(
            tensors=[{"name": "w", "ne": [32], "type_id": type_id, "payload": b"\x00" * 18}]
        )
        _rejects(tmp_path, blob, "'w'", f"unsupported ggml type id {type_id} ({label})")

    def test_payload_beyond_the_data_section(self, tmp_path):
        entry, _ = _tensor("w", 0, (4,))
        blob = build_gguf(tensors=[entry])
        _rejects(tmp_path, blob[:-4], "'w'", "data section is only 12 bytes")

    def test_overlapping_payloads(self, tmp_path):
        first = {"name": "a", "ne": [1], "type_id": 0, "payload": b"\x00" * 4, "offset": 0}
        second = {"name": "b", "ne": [1], "type_id": 0, "payload": b"\x00" * 4, "offset": 2}
        blob = build_gguf(tensors=[first, second])
        _rejects(tmp_path, blob, "'b'", "overlaps tensor 'a'")

    def test_too_many_dimensions(self, tmp_path):
        blob = build_gguf(
            tensors=[{"name": "w", "ne": [2, 2, 2, 2, 2], "type_id": 0, "payload": b"\x00" * 128}]
        )
        _rejects(tmp_path, blob, "'w'", "5 dimensions", "at most 4")

    def test_element_count_not_a_block_multiple(self, tmp_path):
        blob = build_gguf(
            tensors=[{"name": "w", "ne": [31], "type_id": 2, "payload": b"\x00" * 18}]
        )
        _rejects(tmp_path, blob, "'w'", "31 elements", "block size 32")

    @pytest.mark.parametrize("alignment", [0, 3, 12])
    def test_bad_alignment(self, tmp_path, alignment):
        blob = build_gguf(metadata=[("general.alignment", UINT32, alignment)])
        _rejects(tmp_path, blob, "positive power of two", str(alignment))

    def test_non_integer_alignment(self, tmp_path):
        blob = build_gguf(metadata=[("general.alignment", STRING, "32")])
        _rejects(tmp_path, blob, "general.alignment must be an integer", "str")

    def test_string_length_overflow(self, tmp_path):
        blob = build_gguf(metadata=[("k", UINT32, 1)])
        _rejects(tmp_path, _patched(blob, "<Q", 24, 10**6), "metadata key length", "runs past")

    def test_tensor_name_length_overflow(self, tmp_path):
        entry, _ = _tensor("w", 0, (1,))
        blob = build_gguf(tensors=[entry])
        offset = 24  # first tensor-info string starts right after the header
        _rejects(tmp_path, _patched(blob, "<Q", offset, 10**6), "tensor name length", "runs past")

    def test_invalid_utf8_string(self, tmp_path):
        # key length 2 followed by two bytes that are not valid UTF-8
        header = b"GGUF" + struct.pack("<I", 3)
        header += struct.pack("<Q", 0) + struct.pack("<Q", 1)  # 0 tensors, 1 KV
        key = struct.pack("<Q", 2) + b"\xff\xfe"
        value = struct.pack("<I", UINT32) + struct.pack("<I", 1)
        _rejects(tmp_path, header + key + value, "metadata key", "not valid UTF-8")

    def test_tensor_info_section_beyond_the_file(self, tmp_path):
        entry, _ = _tensor("w", 0, (4,))
        built = build_gguf_detailed(tensors=[entry], alignment=256)
        # cut the whole data section and the padding
        _rejects(
            tmp_path,
            built.data[: built.infos_end],
            "tensor info section ends at",
            "beyond the",
        )


# ---------------------------------------------------------------------------
# reference quantizer round trips
# ---------------------------------------------------------------------------


class TestQuantizedRoundTrip:
    @pytest.mark.parametrize("type_id", [2, 6, 8], ids=["Q4_0", "Q5_0", "Q8_0"])
    def test_quantize_dequantize_round_trip(self, tmp_path, type_id):
        block, type_size = GGML_BLOCK[type_id]
        rng = np.random.default_rng(7)
        values = (rng.standard_normal((3, block)) * 2.0).astype(np.float32)
        payload = REFERENCE_QUANTIZERS[type_id](values)
        assert len(payload) == 3 * type_size
        blob = build_gguf(
            tensors=[
                {
                    "name": "w",
                    "ne": ne_dims((3, block)),
                    "type_id": type_id,
                    "payload": payload,
                }
            ]
        )
        with load_gguf(_write(tmp_path, blob)) as f:
            stored = f.tensor("w").numpy()
        assert stored.shape == (3, block)
        # bit-exact against the independent reference dequantizer
        np.testing.assert_array_equal(stored, reference_values(type_id, payload, (3, block)))
        # and close to the values that were quantized
        tolerance = {2: 0.5, 6: 0.25, 8: 0.05}[type_id] * float(np.max(np.abs(values)))
        np.testing.assert_allclose(stored, values, atol=tolerance, rtol=0.0)

    def test_quantized_bit_pattern_is_not_a_float_view(self, tmp_path):
        """A quantized payload must never be reinterpreted as dense floats."""
        entry, payload = _tensor("q", 2, (32,))
        with load_gguf(_write(tmp_path, build_gguf(tensors=[entry]))) as f:
            assert f.info("q").nbytes == 18
            assert f.tensor("q").numpy().size == 32
            assert f.raw("q").tobytes() == payload


# ---------------------------------------------------------------------------
# real checkpoint (host-dependent, never downloaded)
# ---------------------------------------------------------------------------


def _find_real_gguf():
    for pattern in _REAL_GGUF_PATTERNS:
        hits = sorted(glob.glob(os.path.expanduser(pattern)))
        for hit in hits:
            if os.path.isfile(hit) and os.path.getsize(hit) > 0:
                return hit
    return None


class TestRealCheckpoint:
    def test_real_gguf_checkpoint(self):
        found = _find_real_gguf()
        if found is None:
            pytest.skip(
                "no .gguf checkpoint present on this host "
                "(host guard; nothing is downloaded)"
            )
        with load_gguf(found) as f:
            assert f.version in (2, 3)
            assert f.alignment in (1, 2, 4, 8, 16, 32, 64, 128, 256)
            names = f.tensor_names()
            assert names
            for name in names[:3]:  # keep the dequantization work small
                info = f.info(name)
                host = f.tensor(name).numpy()
                assert host.shape == info.shape
                assert host.dtype in (np.dtype("float32"), np.dtype("float16"))
                assert f.raw(name).nbytes == info.nbytes
                assert np.isfinite(host).all()
