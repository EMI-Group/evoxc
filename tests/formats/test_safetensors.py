"""Tests for ``evoxc.formats.safetensors`` — the safetensors tensor half.

Covers the binding contract in ``evoxc/formats/CONTEXT.md``:

* round-trips for every writable dtype (incl. 0-d, zero-element and multi-dim
  shapes), metadata, multi-tensor files;
* ``mmap=True`` zero-copy READ-ONLY views vs ``mmap=False`` independent writable
  copies, and the documented zero-element exception (a fresh writable array in
  both modes);
* an independent parse of the writer's container layout (u64-LE header length,
  8-byte-aligned header, 8-byte-aligned tensor starts) and the reader's
  acceptance of a non-8-multiple header length;
* ``TensorInfo`` fields, name ordering, ``in``/``len``/iteration, context
  manager/``close()``, ``as_dict()``;
* atomic writes (a failed write never touches the target file or leaves a
  partial file behind) and the write-side rejections;
* the eleven documented corruption rejections, each naming the file and the
  offending tensor/offset.

Every fixture is built synthetically in ``tmp_path`` by the byte-level builders
in ``tests/formats/_builders.py``; a real checkpoint is used only when one
already exists on the host.
"""
from __future__ import annotations

import dataclasses
import glob
import json
import os
import struct

import numpy as np
import pytest

from evoxc.core import Device, DeviceError, PersistenceError, Tensor
from evoxc.formats import (
    SafetensorsFile,
    TensorInfo,
    load_safetensors,
    save_safetensors,
)
from tests.formats._builders import (
    build_safetensors,
    parse_safetensors,
    raw_safetensors,
)

#: every dtype the writer accepts, with its safetensors tag and numpy name.
WRITABLE_DTYPES = [
    ("float64", "F64"),
    ("float32", "F32"),
    ("float16", "F16"),
    ("int64", "I64"),
    ("int32", "I32"),
    ("int16", "I16"),
    ("int8", "I8"),
    ("uint64", "U64"),
    ("uint32", "U32"),
    ("uint16", "U16"),
    ("uint8", "U8"),
    ("bool", "BOOL"),
]

_CHECKPOINT_PATTERNS = (
    "~/Source/*/tiny-random-LlamaForCausalLM/model.safetensors",
    "~/Source/*/*/tiny-random-LlamaForCausalLM/model.safetensors",
    "~/models/*/model.safetensors",
    "~/hf/*/model.safetensors",
    "~/.cache/huggingface/hub/models--*/snapshots/*/model.safetensors",
)


# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------


def sample_array(dtype, shape=(2, 3)):
    """A small array whose values are exactly representable in ``dtype``."""
    shape = tuple(shape)
    numel = 1
    for dim in shape:
        numel *= dim
    flat = np.arange(numel)
    if dtype == "bool":
        return (flat % 3 == 0).reshape(shape)
    if dtype.startswith("float"):
        flat = flat / 4.0 if flat.size else flat
    return flat.astype(dtype).reshape(shape)


@pytest.fixture
def path(tmp_path):
    return tmp_path / "weights.safetensors"


def _write(path, tensors, metadata=None):
    save_safetensors(path, tensors, metadata=metadata)
    return path


def _blob(path):
    return path.read_bytes()


def _load_blob(tmp_path, blob, name="corrupt.safetensors"):
    bad = tmp_path / name
    bad.write_bytes(blob)
    return bad


def _valid_blob(name="w", dtype="F32", shape=(2, 2), values=None):
    """A minimal valid container (one F32 tensor by default)."""
    values = sample_array("float32", shape) if values is None else values
    return build_safetensors(
        [{"name": name, "dtype": dtype, "shape": shape, "data": values}]
    )


# ---------------------------------------------------------------------------
# round-trips
# ---------------------------------------------------------------------------


class TestRoundTrip:
    @pytest.mark.parametrize("dtype,tag", WRITABLE_DTYPES)
    def test_every_writable_dtype(self, tmp_path, dtype, tag):
        arr = sample_array(dtype)
        path = _write(tmp_path / "w.safetensors", {"w": arr})
        with load_safetensors(path, mmap=False) as f:
            assert f.tensor_names() == ["w"]
            t = f.tensor("w")
            assert isinstance(t, Tensor)
            assert t.dtype == arr.dtype == np.dtype(dtype)
            assert t.shape == arr.shape
            np.testing.assert_array_equal(t.numpy(), arr)
            info = f.info("w")
            assert info.dtype == np.dtype(dtype).name
            assert info.source_dtype == tag
            assert info.shape == arr.shape
            assert info.nbytes == arr.nbytes
            assert info.numel == arr.size
            assert info.rank == arr.ndim

    @pytest.mark.parametrize(
        "shape", [(), (0,), (2, 0), (3, 0, 2), (2, 3), (1, 1, 4), (2, 2, 2, 2)]
    )
    def test_shapes_including_scalar_and_empty(self, tmp_path, shape):
        arr = sample_array("float32", shape)
        path = _write(tmp_path / "w.safetensors", {"w": arr})
        f = load_safetensors(path, mmap=False)
        try:
            t = f.tensor("w")
            assert t.shape == shape
            assert f.info("w").shape == shape
            assert f.info("w").numel == arr.size
            np.testing.assert_array_equal(t.numpy(), arr)
        finally:
            f.close()

    def test_metadata_round_trip(self, tmp_path):
        arr = sample_array("int32")
        meta = {"format": "pt", "note": "hello", "empty": ""}
        path = _write(tmp_path / "m.safetensors", {"w": arr}, metadata=meta)
        with load_safetensors(path) as f:
            assert f.metadata == meta

    @pytest.mark.parametrize("metadata", [None, {}])
    def test_absent_or_empty_metadata_is_empty_dict(self, tmp_path, metadata):
        path = _write(tmp_path / "m.safetensors", {"w": sample_array("float32")}, metadata)
        with load_safetensors(path) as f:
            assert f.metadata == {}
        header, _, _ = parse_safetensors(_blob(path))
        assert "__metadata__" not in header

    def test_multi_tensor_file(self, tmp_path):
        tensors = {
            "beta": np.arange(4, dtype=np.int16),
            "alpha": np.linspace(0, 1, 6, dtype=np.float32),
            "gamma.scalar": np.array(2.5, dtype=np.float64),
            "delta.empty": np.zeros((0, 3), dtype=np.uint8),
        }
        path = _write(tmp_path / "multi.safetensors", tensors, metadata={"k": "v"})
        f = load_safetensors(path)
        try:
            # the writer sorts names, and tensor_names() preserves header order
            assert f.tensor_names() == sorted(tensors)
            assert list(f) == sorted(tensors)
            assert len(f) == len(tensors)
            for name in tensors:
                assert name in f
                np.testing.assert_array_equal(f.tensor(name).numpy(), tensors[name])
            loaded = f.as_dict()
            assert set(loaded) == set(tensors)
            assert all(isinstance(v, Tensor) for v in loaded.values())
            assert f.metadata == {"k": "v"}
        finally:
            f.close()

    def test_metadata_only_container(self, tmp_path):
        path = tmp_path / "meta.safetensors"
        path.write_bytes(build_safetensors([], metadata={"a": "b"}))
        with load_safetensors(path) as f:
            assert len(f) == 0
            assert f.tensor_names() == []
            assert f.as_dict() == {}
            assert f.metadata == {"a": "b"}

    def test_load_accepts_pathlib_and_returns_file_object(self, tmp_path):
        path = _write(tmp_path / "w.safetensors", {"w": sample_array("float32")})
        f = load_safetensors(path)
        try:
            assert isinstance(f, SafetensorsFile)
            assert f.path == os.fspath(path)
        finally:
            f.close()

    def test_tensor_input_is_accepted(self, tmp_path):
        arr = sample_array("float32")
        path = _write(tmp_path / "w.safetensors", {"w": Tensor(arr)})
        with load_safetensors(path, mmap=False) as f:
            np.testing.assert_array_equal(f.tensor("w").numpy(), arr)

    def test_non_contiguous_input_is_serialized_densely(self, tmp_path):
        arr = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]  # (4, 3) strided
        assert not arr.flags.c_contiguous
        path = _write(tmp_path / "w.safetensors", {"w": arr})
        with load_safetensors(path, mmap=False) as f:
            np.testing.assert_array_equal(f.tensor("w").numpy(), arr)

    def test_scalar_shape_is_not_promoted(self, tmp_path):
        # np.ascontiguousarray would silently turn a 0-d input into shape (1,).
        path = _write(tmp_path / "w.safetensors", {"w": np.array(7, dtype=np.int32)})
        with load_safetensors(path) as f:
            assert f.info("w").shape == ()
            assert f.tensor("w").numpy().shape == ()


# ---------------------------------------------------------------------------
# mmap vs eager copy semantics
# ---------------------------------------------------------------------------


class TestMmapSemantics:
    def test_mmap_gives_zero_copy_read_only_views(self, path):
        arr = sample_array("float32")
        _write(path, {"w": arr})
        with load_safetensors(path, mmap=True) as f:
            t = f.tensor("w")
            host = t.numpy()
            assert host is t.numpy()  # zero-copy: the tensor's own ndarray
            assert not host.flags.writeable
            assert not host.flags.owndata
            with pytest.raises(ValueError):
                host[0, 0] = 1.0
            np.testing.assert_array_equal(host, arr)

    def test_mmap_false_gives_independent_writable_copies(self, path):
        arr = sample_array("float32")
        _write(path, {"w": arr})
        before = _blob(path)
        with load_safetensors(path, mmap=False) as f:
            first = f.tensor("w").numpy()
            second = f.tensor("w").numpy()
            assert first.flags.writeable and first.flags.owndata
            assert first is not second
            first[0, 0] = 99.0
            assert second[0, 0] == arr[0, 0]
            assert f.tensor("w").numpy()[0, 0] == arr[0, 0]
        assert _blob(path) == before
        with load_safetensors(path) as f:
            np.testing.assert_array_equal(f.tensor("w").numpy(), arr)

    @pytest.mark.parametrize("mmap", [True, False])
    def test_zero_element_tensor_is_fresh_writable_array(self, path, mmap):
        # Documented exception: an empty tensor cannot alias file memory.
        _write(path, {"empty": np.zeros((0, 4), dtype=np.float32)})
        with load_safetensors(path, mmap=mmap) as f:
            host = f.tensor("empty").numpy()
            assert host.shape == (0, 4)
            assert host.flags.writeable
            assert host.flags.owndata

    def test_as_dict_follows_the_same_copy_semantics(self, path):
        _write(path, {"a": sample_array("float32"), "b": sample_array("int32")})
        with load_safetensors(path, mmap=False) as f:
            loaded = f.as_dict()
        assert all(v.numpy().flags.writeable for v in loaded.values())


# ---------------------------------------------------------------------------
# container layout
# ---------------------------------------------------------------------------


class TestContainerLayout:
    def test_writer_emits_canonical_layout(self, tmp_path):
        tensors = {
            "a": np.arange(3, dtype=np.float32),  # 12 bytes -> pads to 16
            "b": np.arange(4, dtype=np.uint8),  # 4 bytes
            "c": np.zeros((2, 2), dtype=np.float64),  # 32 bytes
        }
        path = _write(tmp_path / "layout.safetensors", tensors, metadata={"m": "1"})
        blob = _blob(path)
        (header_len,) = struct.unpack_from("<Q", blob, 0)
        assert header_len % 8 == 0
        header, data_start, data = parse_safetensors(blob)
        assert data_start == 8 + header_len
        assert len(data) == len(blob) - data_start
        entries = {k: v for k, v in header.items() if k != "__metadata__"}
        assert set(entries) == set(tensors)
        running = 0
        for name in sorted(entries):
            begin, end = entries[name]["data_offsets"]
            assert begin % 8 == 0
            assert begin == ((running + 7) // 8) * 8
            assert end - begin == tensors[name].nbytes
            running = end
        # payload bytes are the raw little-endian array bytes, in order
        assert data[entries["a"]["data_offsets"][0] : entries["a"]["data_offsets"][1]] == (
            tensors["a"].astype("<f4").tobytes()
        )

    def test_reader_accepts_non_multiple_of_eight_header_length(self, tmp_path):
        values = sample_array("float32", (2, 2))
        header = {
            "w": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, values.nbytes],
            }
        }
        header_bytes = json.dumps(header, separators=(",", ":")).encode()
        while len(header_bytes) % 8 == 0:  # trailing JSON whitespace is legal
            header_bytes += b" "
        assert len(header_bytes) % 8 != 0
        path = tmp_path / "odd.safetensors"
        path.write_bytes(raw_safetensors(header_bytes, values.astype("<f4").tobytes()))
        with load_safetensors(path) as f:
            assert f.tensor("w").numpy().shape == (2, 2)
            np.testing.assert_array_equal(f.tensor("w").numpy(), values)

    def test_zero_length_extents_may_share_a_range(self, tmp_path):
        # The overlap rule applies to non-empty extents only, so two empty
        # tensors may both claim [0, 0].
        blob = build_safetensors(
            [
                {"name": "a", "dtype": "F32", "shape": [0], "data": b"",
                 "data_offsets": [0, 0]},
                {"name": "b", "dtype": "F32", "shape": [0, 0], "data": b"",
                 "data_offsets": [0, 0]},
            ]
        )
        path = tmp_path / "empty.safetensors"
        path.write_bytes(blob)
        with load_safetensors(path) as f:
            assert f.tensor_names() == ["a", "b"]
            assert f.tensor("b").numpy().shape == (0, 0)

    def test_header_order_is_preserved(self, tmp_path):
        blob = build_safetensors(
            [
                {"name": "z", "dtype": "U8", "shape": [2], "data": b"\x01\x02"},
                {"name": "a", "dtype": "U8", "shape": [2], "data": b"\x03\x04"},
            ]
        )
        path = tmp_path / "order.safetensors"
        path.write_bytes(blob)
        with load_safetensors(path) as f:
            assert f.tensor_names() == ["z", "a"]

    def test_info_descriptor_fields(self, tmp_path):
        arr = np.zeros((3, 5), dtype=np.float16)
        path = _write(tmp_path / "w.safetensors", {"w": arr})
        with load_safetensors(path) as f:
            info = f.info("w")
        assert isinstance(info, TensorInfo)
        assert (info.name, info.dtype, info.source_dtype) == ("w", "float16", "F16")
        assert info.shape == (3, 5)
        assert info.nbytes == arr.nbytes
        assert info.numel == 15
        assert info.rank == 2
        assert "w" in str(info) and "float16" in str(info)

    def test_tensor_info_is_frozen(self):
        info = TensorInfo(name="w", dtype="float32", shape=(2,), nbytes=8, source_dtype="F32")
        with pytest.raises(dataclasses.FrozenInstanceError):
            info.name = "other"
        with pytest.raises(dataclasses.FrozenInstanceError):
            info.shape = (3,)

    def test_unknown_names_raise_key_error(self, path):
        _write(path, {"w": sample_array("float32")})
        with load_safetensors(path) as f:
            with pytest.raises(KeyError, match="no tensor named 'nope'"):
                f.info("nope")
            with pytest.raises(KeyError, match="no tensor named 'nope'"):
                f.tensor("nope")

    def test_close_is_idempotent_and_further_reads_fail(self, path):
        _write(path, {"w": sample_array("float32")})
        f = load_safetensors(path)
        alive = f.tensor("w").numpy()  # keep a view alive across close()
        f.close()
        f.close()  # idempotent, never raises
        with pytest.raises(PersistenceError, match="file is closed"):
            f.tensor("w")
        np.testing.assert_array_equal(alive, sample_array("float32"))

    def test_context_manager_closes(self, path):
        _write(path, {"w": sample_array("float32")})
        with load_safetensors(path) as f:
            assert f.tensor_names() == ["w"]
        with pytest.raises(PersistenceError, match="file is closed"):
            f.tensor("w")


# ---------------------------------------------------------------------------
# atomic writes
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    def test_failed_write_preserves_existing_file(self, path):
        _write(path, {"w": np.arange(4, dtype=np.float32)})
        before = _blob(path)
        with pytest.raises(PersistenceError, match="cannot represent"):
            save_safetensors(path, {"w": np.zeros(2, dtype=np.complex64)})
        assert _blob(path) == before
        assert sorted(p.name for p in path.parent.iterdir()) == [path.name]
        with load_safetensors(path) as f:
            np.testing.assert_array_equal(f.tensor("w").numpy(), np.arange(4, dtype=np.float32))

    def test_failed_write_on_bad_metadata_preserves_existing_file(self, path):
        _write(path, {"w": np.arange(2, dtype=np.float32)})
        before = _blob(path)
        with pytest.raises(PersistenceError, match="__metadata__ entries must be strings"):
            save_safetensors(path, {"w": np.arange(2, dtype=np.float32)}, metadata={"n": 1})
        assert _blob(path) == before
        assert sorted(p.name for p in path.parent.iterdir()) == [path.name]

    def test_missing_directory_leaves_no_partial_file(self, tmp_path):
        target = tmp_path / "absent" / "w.safetensors"
        with pytest.raises(PersistenceError, match="cannot create temporary file"):
            save_safetensors(target, {"w": np.arange(2, dtype=np.float32)})
        assert not target.exists()
        assert not target.parent.exists()

    def test_successful_write_replaces_the_file(self, path):
        _write(path, {"old": np.arange(3, dtype=np.float32)})
        _write(path, {"new": np.arange(2, dtype=np.int32)})
        with load_safetensors(path) as f:
            assert f.tensor_names() == ["new"]
        assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


# ---------------------------------------------------------------------------
# write-side rejections
# ---------------------------------------------------------------------------


class TestWriteRejections:
    @pytest.mark.parametrize(
        "value",
        [
            np.zeros(2, dtype=np.complex64),
            np.zeros(2, dtype=np.complex128),
            np.zeros(2, dtype=np.longdouble),
            np.array([object()], dtype=object),
            np.array(["a", "bc"]),
        ],
        ids=["complex64", "complex128", "longdouble", "object", "str"],
    )
    def test_unsupported_dtypes(self, tmp_path, value):
        with pytest.raises(PersistenceError, match="cannot represent"):
            save_safetensors(tmp_path / "w.safetensors", {"w": value})
        assert not (tmp_path / "w.safetensors").exists()

    def test_non_string_metadata_value(self, tmp_path):
        with pytest.raises(PersistenceError, match="must be strings"):
            save_safetensors(
                tmp_path / "w.safetensors", {"w": np.zeros(2)}, metadata={"k": 5}
            )

    def test_non_string_metadata_key(self, tmp_path):
        with pytest.raises(PersistenceError, match="must be strings"):
            save_safetensors(
                tmp_path / "w.safetensors", {"w": np.zeros(2)}, metadata={7: "v"}
            )

    def test_non_mapping_inputs(self, tmp_path):
        with pytest.raises(TypeError, match="mapping of tensor name"):
            save_safetensors(tmp_path / "w.safetensors", [("w", np.zeros(2))])
        with pytest.raises(TypeError, match="metadata must be a mapping"):
            save_safetensors(
                tmp_path / "w.safetensors", {"w": np.zeros(2)}, metadata=[("k", "v")]
            )

    def test_non_string_tensor_name(self, tmp_path):
        with pytest.raises(TypeError, match="names must be strings"):
            save_safetensors(tmp_path / "w.safetensors", {1: np.zeros(2)})

    def test_reserved_metadata_tensor_name(self, tmp_path):
        with pytest.raises(PersistenceError, match="reserved"):
            save_safetensors(tmp_path / "w.safetensors", {"__metadata__": np.zeros(2)})

    def test_non_cpu_tensor_raises_device_error(self, tmp_path):
        tensor = Tensor(_FakeCudaPayload((2, 2), np.zeros((2, 2), dtype=np.float32)))
        with pytest.raises(DeviceError, match="host memory"):
            save_safetensors(tmp_path / "w.safetensors", {"w": tensor})
        assert not (tmp_path / "w.safetensors").exists()

    def test_bf16_and_f8_are_read_only_in_v1(self, tmp_path):
        """Reading BF16/F8 works but writing back cannot restore the tag."""
        raw16 = np.array([0x3F80, 0x4000], dtype="<u2")
        raw8 = np.array([0x7E, 0x01], dtype=np.uint8)
        blob = build_safetensors(
            [
                {"name": "bf", "dtype": "BF16", "shape": [2], "data": raw16.tobytes()},
                {"name": "f8", "dtype": "F8_E4M3", "shape": [2], "data": raw8.tobytes()},
            ]
        )
        source = tmp_path / "src.safetensors"
        source.write_bytes(blob)
        with load_safetensors(source) as f:
            assert f.info("bf").dtype == "BF16"
            assert f.info("bf").source_dtype == "BF16"
            assert f.tensor("bf").dtype == np.dtype("uint16")
            np.testing.assert_array_equal(f.tensor("bf").numpy(), raw16)
            assert f.info("f8").dtype == "F8_E4M3"
            assert f.info("f8").source_dtype == "F8_E4M3"
            assert f.tensor("f8").dtype == np.dtype("uint8")
            rewritten = {"bf": f.tensor("bf"), "f8": f.tensor("f8")}
        out = tmp_path / "out.safetensors"
        save_safetensors(out, rewritten)
        with load_safetensors(out) as f:
            # the raw storage words are indistinguishable, so the tag degrades
            assert f.info("bf").source_dtype == "U16"
            assert f.info("f8").source_dtype == "U8"
            np.testing.assert_array_equal(f.tensor("bf").numpy(), raw16)


class _FakeCudaPayload:
    """Duck-typed non-cpu device payload (the ``core.Tensor`` payload protocol)."""

    def __init__(self, shape, values):
        self.shape = tuple(shape)
        self.dtype = np.dtype(np.float32)
        self.device = Device("cuda", 0)
        self._values = values

    def to_host(self):  # pragma: no cover - the writer must never get here
        raise AssertionError("no implicit device-to-host transfer is allowed")


# ---------------------------------------------------------------------------
# corruption rejections (the eleven documented cases)
# ---------------------------------------------------------------------------


class TestCorruption:
    def _rejects(self, tmp_path, blob, *fragments):
        bad = _load_blob(tmp_path, blob)
        with pytest.raises(PersistenceError) as excinfo:
            load_safetensors(bad, mmap=False)
        message = str(excinfo.value)
        assert str(bad) in message
        for fragment in fragments:
            assert fragment in message, message
        return message

    def test_short_file(self, tmp_path):
        self._rejects(tmp_path, b"\x00\x00\x00\x00", "too short")

    def test_header_length_overflow(self, tmp_path):
        self._rejects(
            tmp_path,
            raw_safetensors(b"{}", b"", length=10**6),
            "does not fit in a",
        )

    def test_header_length_zero(self, tmp_path):
        self._rejects(tmp_path, raw_safetensors(b"", b""), "header length is 0")

    def test_invalid_json(self, tmp_path):
        self._rejects(tmp_path, raw_safetensors(b"{not json"), "invalid safetensors JSON")

    def test_non_object_header(self, tmp_path):
        self._rejects(tmp_path, raw_safetensors(b"[1, 2]"), "must be a JSON object")

    def test_non_object_entry(self, tmp_path):
        self._rejects(
            tmp_path,
            raw_safetensors(b'{"w": [1, 2]}'),
            "must be a JSON object",
        )

    def test_unknown_dtype_names_its_offset(self, tmp_path):
        blob = build_safetensors(
            [{"name": "w", "dtype": "F99", "shape": [2], "data": b"\x00" * 8}]
        )
        self._rejects(tmp_path, blob, "'w'", "F99", "at data offset 0")

    def test_byte_extent_mismatch(self, tmp_path):
        blob = build_safetensors(
            [
                {
                    "name": "w",
                    "dtype": "F32",
                    "shape": [2],
                    "data": b"\x00" * 8,
                    "data_offsets": [0, 4],
                }
            ]
        )
        self._rejects(tmp_path, blob, "'w'", "needing 8 bytes", "span 4 bytes")

    def test_out_of_bounds_offsets(self, tmp_path):
        blob = build_safetensors(
            [
                {
                    "name": "w",
                    "dtype": "F32",
                    "shape": [16],
                    "data": b"\x00" * 16,
                    "data_offsets": [0, 64],
                }
            ],
            data_length=16,
        )
        self._rejects(tmp_path, blob, "'w'", "data buffer is only 16 bytes")

    def test_negative_dimension(self, tmp_path):
        blob = build_safetensors(
            [{"name": "w", "dtype": "F32", "shape": [-1], "data": b""}]
        )
        self._rejects(tmp_path, blob, "'w'", "invalid dimension -1")

    def test_reversed_offsets(self, tmp_path):
        blob = build_safetensors(
            [
                {
                    "name": "w",
                    "dtype": "F32",
                    "shape": [2],
                    "data": b"\x00" * 16,
                    "data_offsets": [8, 4],
                }
            ]
        )
        self._rejects(tmp_path, blob, "'w'", "invalid data offsets [8, 4]")

    def test_overlapping_tensors(self, tmp_path):
        blob = build_safetensors(
            [
                {
                    "name": "a",
                    "dtype": "F32",
                    "shape": [2],
                    "data": b"\x00" * 12,
                    "data_offsets": [0, 8],
                },
                {
                    "name": "b",
                    "dtype": "F32",
                    "shape": [2],
                    "data": b"",
                    "data_offsets": [4, 12],
                },
            ]
        )
        self._rejects(tmp_path, blob, "'b'", "overlaps tensor 'a'")

    def test_non_integer_offsets(self, tmp_path):
        blob = build_safetensors(
            [
                {
                    "name": "w",
                    "dtype": "F32",
                    "shape": [2],
                    "data": b"\x00" * 8,
                    "data_offsets": [0.0, 8.0],
                }
            ]
        )
        self._rejects(tmp_path, blob, "'w'", "invalid 'data_offsets'")

    def test_non_string_metadata(self, tmp_path):
        blob = build_safetensors(
            [{"name": "w", "dtype": "U8", "shape": [1], "data": b"\x00"}],
            metadata={"k": 5},
        )
        self._rejects(tmp_path, blob, "__metadata__ entries must be strings")

    def test_metadata_not_an_object(self, tmp_path):
        header = json.dumps({"__metadata__": ["k", "v"]}).encode()
        self._rejects(tmp_path, raw_safetensors(header), "must be a JSON object")

    def test_truncated_data_buffer(self, tmp_path):
        blob = _valid_blob(shape=(4,), values=np.zeros(4, dtype=np.float32))
        self._rejects(tmp_path, blob[:-4], "data buffer is only 12 bytes")


# ---------------------------------------------------------------------------
# real checkpoint (host-dependent, never downloaded)
# ---------------------------------------------------------------------------


def _find_checkpoint():
    for pattern in _CHECKPOINT_PATTERNS:
        hits = sorted(glob.glob(os.path.expanduser(pattern)))
        if hits:
            return hits[0]
    return None


class TestRealCheckpoint:
    def test_real_safetensors_checkpoint_consistency(self):
        found = _find_checkpoint()
        if found is None:
            pytest.skip(
                "no safetensors checkpoint present on this host "
                "(host guard; nothing is downloaded)"
            )
        with load_safetensors(found) as f:
            assert len(f) > 0
            for name in f.tensor_names():
                info = f.info(name)
                host = f.tensor(name).numpy()
                assert host.shape == info.shape
                assert host.nbytes == info.nbytes == info.numel * host.dtype.itemsize
                if info.dtype in ("BF16", "F8_E4M3", "F8_E5M2"):
                    # raw storage words: uint16 / uint8 views
                    assert host.dtype in (np.dtype("uint16"), np.dtype("uint8"))
                else:
                    assert host.dtype.name == info.dtype
                if host.size:
                    assert np.isfinite(host).any() or info.dtype.startswith(
                        ("int", "uint", "bool")
                    )
