"""safetensors reader/writer — the tensor half of the safetensors container.

Layout (all integers little-endian)::

    [0, 8)                        u64  header length N
    [8, 8 + N)                    UTF-8 JSON header (N is a multiple of 8;
                                  unused bytes are ASCII spaces)
    [8 + N, file_size)            data buffer: the raw tensor bytes

The JSON header is one object mapping each tensor name to
``{"dtype": <tag>, "shape": [dims...], "data_offsets": [begin, end]}`` plus an
optional ``"__metadata__"`` entry holding a ``{str: str}`` map.  Tensor offsets
are byte offsets into the data buffer, i.e. relative to ``8 + N``, and
``end - begin`` must equal ``numel * itemsize``.

Validation (binding): the magic is a length, so every entry is checked against
the file before a view exists — header length in bounds, JSON object shape,
known dtype tag, non-negative integer dims, exactly two integer data offsets,
``0 <= begin <= end <= data_len``, the byte extent matching the shape, and no
two non-empty tensor extents overlapping.  Any violation raises
:class:`~evoxc.core.PersistenceError` naming the file, the tensor and the byte
offset.  Nothing is repaired or truncated.

Reading: ``mmap=True`` (default) returns **zero-copy read-only** numpy views
over the mapped file; ``mmap=False`` reads eagerly and returns independent
writable arrays.  Every returned value is a :class:`~evoxc.core.Tensor` on
``Device("cpu", 0)`` (numpy host memory).

Writing: canonical layout — tensors in sorted-name order, every tensor start
8-byte aligned inside the data buffer, the JSON header padded with spaces to
an 8-byte multiple, and an atomic temp-file + ``os.replace``.  Accepted inputs
are :class:`~evoxc.core.Tensor`, numpy arrays and memmaps (anything
``np.asarray`` accepts); dtypes the format cannot represent (``complex64``,
``float128``, ``object``, strings, ...) raise
:class:`~evoxc.core.PersistenceError`.  ``BF16``/``F8_E4M3``/``F8_E5M2`` are
read-only in v1: they surface as raw ``uint16``/``uint8`` views (evoxc has no
bf16/f8 dtype), so writing them back would need an explicit tag override,
which this API does not take.
"""
from __future__ import annotations

import json
import os
import struct
import tempfile
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..core import PersistenceError, Tensor
from . import mmio
from .tensorinfo import TensorInfo

__all__ = ["SafetensorsFile", "load_safetensors", "save_safetensors"]

#: safetensors tag -> (numpy dtype of the returned view, canonical dtype name).
_DTYPES: Dict[str, Tuple[np.dtype, str]] = {
    "F64": (np.dtype("float64"), "float64"),
    "F32": (np.dtype("float32"), "float32"),
    "F16": (np.dtype("float16"), "float16"),
    # evoxc has no bfloat16/f8 dtype: surface the stored words verbatim.
    "BF16": (np.dtype("uint16"), "BF16"),
    "I64": (np.dtype("int64"), "int64"),
    "I32": (np.dtype("int32"), "int32"),
    "I16": (np.dtype("int16"), "int16"),
    "I8": (np.dtype("int8"), "int8"),
    "U64": (np.dtype("uint64"), "uint64"),
    "U32": (np.dtype("uint32"), "uint32"),
    "U16": (np.dtype("uint16"), "uint16"),
    "U8": (np.dtype("uint8"), "uint8"),
    "BOOL": (np.dtype("bool"), "bool"),
    "F8_E4M3": (np.dtype("uint8"), "F8_E4M3"),
    "F8_E5M2": (np.dtype("uint8"), "F8_E5M2"),
}

#: numpy dtype name -> safetensors tag (the writer's accepted set).  BF16 /
#: F8_* are absent on purpose: numpy cannot express them, and their raw
#: uint16/uint8 storage words are indistinguishable from U16/U8 here.
_NUMPY_TO_TAG: Dict[str, str] = {
    "float64": "F64",
    "float32": "F32",
    "float16": "F16",
    "int64": "I64",
    "int32": "I32",
    "int16": "I16",
    "int8": "I8",
    "uint64": "U64",
    "uint32": "U32",
    "uint16": "U16",
    "uint8": "U8",
    "bool": "BOOL",
}

_HEADER_BOUNDARY = 8


def _f(path: str, message: str) -> PersistenceError:
    return PersistenceError(f"{path}: {message}")


class _Entry:
    """One validated header entry: descriptor + resolved data-buffer range."""

    __slots__ = ("info", "begin", "end")

    def __init__(self, info: TensorInfo, begin: int, end: int):
        self.info = info
        self.begin = begin
        self.end = end


class SafetensorsFile:
    """An open, validated safetensors file.

    Use :func:`load_safetensors` (or the constructor) to open it, then
    :meth:`tensor_names`, :meth:`info`, :meth:`tensor` / :meth:`as_dict`.
    The object owns a file handle/mapping: use it as a context manager or call
    :meth:`close` when done.  Zero-copy views remain valid after :meth:`close`
    only while the view itself is alive.
    """

    def __init__(self, path, *, mmap: bool = True):
        self.path = os.fspath(path)
        self._mmap = bool(mmap)
        self._file = mmio.MappedFile(self.path, mmap=self._mmap)
        self.metadata: Dict[str, str] = {}
        self._entries: Dict[str, _Entry] = {}
        try:
            self._parse()
        except BaseException:
            self._file.close()
            raise

    # -- parsing ------------------------------------------------------------
    def _parse(self) -> None:
        f = self._file
        if f.size < 8:
            raise _f(self.path, f"file is {f.size} bytes — too short for a safetensors header length")
        (header_len,) = f.unpack("<Q", 0, "safetensors header length")
        if header_len == 0:
            raise _f(self.path, "safetensors header length is 0")
        if header_len > f.size - 8:
            raise _f(
                self.path,
                f"safetensors header length {header_len} does not fit in a "
                f"{f.size}-byte file",
            )
        raw_header = f.read(8, header_len, "safetensors JSON header")
        try:
            header = json.loads(raw_header.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _f(self.path, f"invalid safetensors JSON header: {exc}") from exc
        if not isinstance(header, dict):
            raise _f(
                self.path,
                f"safetensors header must be a JSON object, got {type(header).__name__}",
            )
        self._data_start = 8 + header_len
        self._data_len = f.size - self._data_start
        for name, value in header.items():
            if name == "__metadata__":
                self._parse_metadata(value)
                continue
            self._entries[name] = self._parse_entry(name, value)
        self._check_no_overlap()

    def _parse_metadata(self, value) -> None:
        if not isinstance(value, dict):
            raise _f(
                self.path,
                f"safetensors __metadata__ must be a JSON object, got {type(value).__name__}",
            )
        clean: Dict[str, str] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not isinstance(item, str):
                raise _f(
                    self.path,
                    "safetensors __metadata__ entries must be strings, got "
                    f"{key!r}: {item!r}",
                )
            clean[key] = item
        self.metadata = clean

    def _parse_entry(self, name: str, value) -> _Entry:
        if not isinstance(name, str):
            raise _f(self.path, f"safetensors tensor name {name!r} is not a string")
        if not isinstance(value, dict):
            raise _f(
                self.path,
                f"safetensors entry {name!r} must be a JSON object, got "
                f"{type(value).__name__}",
            )
        offsets = value.get("data_offsets")
        begin = end = None
        if (
            isinstance(offsets, (list, tuple))
            and len(offsets) == 2
            and all(isinstance(o, int) and not isinstance(o, bool) for o in offsets)
        ):
            begin, end = int(offsets[0]), int(offsets[1])
        dtype_tag = value.get("dtype")
        if not isinstance(dtype_tag, str) or dtype_tag not in _DTYPES:
            where = (
                f"at data offset {begin}"
                if isinstance(begin, int)
                else "with no valid data_offsets"
            )
            raise _f(
                self.path,
                f"tensor {name!r} uses unsupported safetensors dtype "
                f"{dtype_tag!r} {where}; supported: {sorted(_DTYPES)}",
            )
        shape_raw = value.get("shape")
        if not isinstance(shape_raw, (list, tuple)):
            raise _f(
                self.path,
                f"tensor {name!r} has no list 'shape' (got {shape_raw!r})",
            )
        shape: List[int] = []
        for dim in shape_raw:
            if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
                raise _f(
                    self.path,
                    f"tensor {name!r} has invalid dimension {dim!r} in shape "
                    f"{list(shape_raw)!r}",
                )
            shape.append(int(dim))
        if begin is None:
            raise _f(
                self.path,
                f"tensor {name!r} has invalid 'data_offsets' {offsets!r} "
                "(expected [begin, end])",
            )
        if begin < 0 or end < begin:
            raise _f(
                self.path,
                f"tensor {name!r} has invalid data offsets [{begin}, {end}]",
            )
        if end > self._data_len:
            raise _f(
                self.path,
                f"tensor {name!r} claims data_offsets [{begin}, {end}] but the "
                f"data buffer is only {self._data_len} bytes",
            )
        np_dtype, canonical = _DTYPES[dtype_tag]
        numel = 1
        for dim in shape:
            numel *= dim
        expected = numel * np_dtype.itemsize
        if end - begin != expected:
            raise _f(
                self.path,
                f"tensor {name!r} has shape {tuple(shape)} {canonical} needing "
                f"{expected} bytes but data_offsets [{begin}, {end}] span "
                f"{end - begin} bytes",
            )
        info = TensorInfo(
            name=name,
            dtype=canonical,
            shape=tuple(shape),
            nbytes=end - begin,
            source_dtype=dtype_tag,
        )
        return _Entry(info, begin, end)

    def _check_no_overlap(self) -> None:
        ranged = sorted(
            (entry.begin, entry.end, name)
            for name, entry in self._entries.items()
            if entry.end > entry.begin
        )
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
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(
                f"{self.path}: no tensor named {name!r} "
                f"(available: {sorted(self._entries)})"
            )
        return entry.info

    def tensor(self, name: str) -> Tensor:
        """Materialize ``name`` as a host :class:`~evoxc.core.Tensor`.

        With ``mmap=True`` this is a zero-copy read-only view; with
        ``mmap=False`` it is an independent writable copy.
        """
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(
                f"{self.path}: no tensor named {name!r} "
                f"(available: {sorted(self._entries)})"
            )
        np_dtype, _ = _DTYPES[entry.info.source_dtype]
        array = self._file.array(
            np_dtype,
            entry.info.shape,
            self._data_offset(entry),
            entry.info.nbytes,
            f"tensor {name!r}",
            copy=not self._mmap,
        )
        return Tensor(array)

    def as_dict(self) -> Dict[str, Tensor]:
        """All tensors as ``{name: Tensor}`` (see :meth:`tensor`)."""
        return {name: self.tensor(name) for name in self._entries}

    def close(self) -> None:
        """Release the file handle/mapping (idempotent, never raises)."""
        self._file.close()

    def __contains__(self, name) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __enter__(self) -> "SafetensorsFile":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"SafetensorsFile({self.path!r}, tensors={len(self._entries)}, "
            f"metadata_keys={sorted(self.metadata)})"
        )

    def _data_offset(self, entry: _Entry) -> int:
        """Absolute file offset of ``entry``'s first payload byte."""
        return self._data_start + entry.begin


def load_safetensors(path, *, mmap: bool = True) -> SafetensorsFile:
    """Open and validate a safetensors file.

    Args:
        path: Path to the ``.safetensors`` file.
        mmap: ``True`` (default) memory-maps the file and returns zero-copy
            read-only views from :meth:`SafetensorsFile.tensor`; ``False``
            reads eagerly and returns independent writable arrays.

    Returns:
        A validated :class:`SafetensorsFile`.  Close it (or use it as a
        context manager) when done.
    """
    return SafetensorsFile(path, mmap=mmap)


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------


def _as_array(path: str, name: str, value) -> Tuple[np.ndarray, str]:
    """Normalize one writer input to a contiguous little-endian array + tag."""
    if isinstance(value, Tensor):
        # Explicit host access: a non-cpu payload raises core.DeviceError here
        # (no implicit device→host transfer inside the writer).
        value = value.numpy()
    try:
        array = np.asarray(value)
    except Exception as exc:  # noqa: BLE001 - normalized below
        raise _f(path, f"tensor {name!r} cannot be read as a numpy array: {exc}") from exc
    tag = _NUMPY_TO_TAG.get(array.dtype.name)
    if tag is None:
        raise _f(
            path,
            f"tensor {name!r} has dtype {array.dtype.name!r}, which safetensors "
            f"cannot represent; supported: {sorted(_NUMPY_TO_TAG)}",
        )
    if array.dtype.itemsize > 1:
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    if not array.flags.c_contiguous:
        # NB: np.ascontiguousarray promotes 0-d arrays to shape (1,), which
        # would silently change a scalar tensor's shape.
        array = np.ascontiguousarray(array)
    return array, tag


def save_safetensors(
    path,
    tensors: Mapping[str, object],
    *,
    metadata: Optional[Mapping[str, str]] = None,
) -> None:
    """Write ``tensors`` to ``path`` in canonical safetensors layout.

    Args:
        path: Destination path (written atomically via temp file +
            :func:`os.replace`).
        tensors: Mapping of tensor name to :class:`~evoxc.core.Tensor`, numpy
            array/memmap (or anything ``np.asarray`` accepts).
        metadata: Optional ``{str: str}`` header metadata.

    Raises:
        TypeError: ``tensors``/``metadata`` are not mappings, or a tensor name
            is not a string.
        PersistenceError: A dtype the format cannot represent, non-string
            metadata, or an I/O failure.
    """
    path = os.fspath(path)
    if not isinstance(tensors, Mapping):
        raise TypeError(
            f"save_safetensors expects a mapping of tensor name to array, got "
            f"{type(tensors).__name__}"
        )
    clean_metadata: Dict[str, str] = {}
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise TypeError(
                f"save_safetensors metadata must be a mapping of str to str, "
                f"got {type(metadata).__name__}"
            )
        for key, value in metadata.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise _f(
                    path,
                    "safetensors __metadata__ entries must be strings, got "
                    f"{key!r}: {value!r}",
                )
            clean_metadata[key] = value

    arrays: Dict[str, Tuple[np.ndarray, str]] = {}
    for name in sorted(tensors):
        if not isinstance(name, str):
            raise TypeError(
                f"save_safetensors tensor names must be strings, got {name!r}"
            )
        if name == "__metadata__":
            raise _f(path, "tensor name '__metadata__' is reserved by the format")
        arrays[name] = _as_array(path, name, tensors[name])

    header: Dict[str, object] = {}
    if clean_metadata:
        header["__metadata__"] = clean_metadata
    offset = 0
    for name in sorted(arrays):
        array, tag = arrays[name]
        offset = mmio.align_up(offset, _HEADER_BOUNDARY)
        nbytes = int(array.nbytes)
        header[name] = {
            "dtype": tag,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    try:
        header_bytes = json.dumps(
            header, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise _f(path, f"cannot serialize safetensors header: {exc}") from exc
    pad = mmio.align_up(len(header_bytes), _HEADER_BOUNDARY) - len(header_bytes)
    header_bytes += b" " * pad

    directory = os.path.dirname(os.path.abspath(path))
    prefix = os.path.basename(path) + ".tmp-"
    try:
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=prefix)
    except OSError as exc:
        raise _f(path, f"cannot create temporary file: {exc}") from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(struct.pack("<Q", len(header_bytes)))
            handle.write(header_bytes)
            written = 8 + len(header_bytes)
            for name in sorted(arrays):
                array, _ = arrays[name]
                begin, _end = header[name]["data_offsets"]  # type: ignore[index]
                data_start = 8 + len(header_bytes) + begin
                if data_start > written:
                    handle.write(b"\x00" * (data_start - written))
                    written = data_start
                if array.nbytes:
                    # reshape(-1) is a view for our contiguous array and makes
                    # the memoryview cast valid for 0-d (scalar) tensors too.
                    handle.write(memoryview(array.reshape(-1)).cast("B"))
                written += int(array.nbytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException as exc:
        try:
            os.unlink(tmp_path)
        except OSError:  # pragma: no cover - environment dependent
            pass
        if isinstance(exc, OSError):
            raise _f(path, f"cannot write safetensors file: {exc}") from exc
        raise
