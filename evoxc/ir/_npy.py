"""Private ndarray container handling for IR serialization.

The on-disk payload stores each constant as base64 of an ``np.save`` container.
The in-process native envelope stores the same array inline as base64 of its raw C-order little-endian payload.
This module owns both forms plus the dtype memo tables the serializer shares.
It is private to ``evoxc.ir``: no public API is defined here.
"""

from __future__ import annotations

import ast
import base64
import io
import struct

import numpy as np

from evoxc.core import VerificationError

#: Attribute-value marker for numpy array payloads.
_NDARRAY_TAG = "__evoxc_ndarray__"

#: Magic + format-version prefix of an ``np.save`` (v1.0) container.
_NPY_MAGIC = b"\x93NUMPY\x01\x00"

#: Memo tables for the two conversions that repeat once per encoded value.
#: Both are keyed on immutable values, so sharing is safe; they are capped to
#: keep a pathological caller from growing them without bound.
_DTYPE_NAME_CACHE: dict = {}
_DTYPE_CACHE: dict = {}


def _is_int(value) -> bool:
    """True for plain ints (bools excluded)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _dtype_name(dtype_obj) -> str:
    """The ``dtype.name`` string of ``dtype_obj``, memoized (the name is not cached)."""
    try:
        return _DTYPE_NAME_CACHE[dtype_obj]
    except KeyError:
        pass
    name = np.dtype(dtype_obj).name
    if len(_DTYPE_NAME_CACHE) < 256:
        _DTYPE_NAME_CACHE[dtype_obj] = name
    return name


def _dtype_from_name(name: str) -> np.dtype:
    """Parse a ``dtype.name`` string, memoized."""
    try:
        return _DTYPE_CACHE[name]
    except KeyError:
        pass
    dtype_obj = np.dtype(name)
    if len(_DTYPE_CACHE) < 256:
        _DTYPE_CACHE[name] = dtype_obj
    return dtype_obj


# ---------------------------------------------------------------------------
# ``np.save`` container (the on-disk payload's constants table)
# ---------------------------------------------------------------------------


def _npy_bytes(array: np.ndarray) -> bytes:
    """The ``np.save(..., allow_pickle=False)`` container bytes of ``array``.

    A fieldless, non-object dtype uses a hand-built v1.0 header (same magic,
    version, header padding and C-order payload as ``np.save``) because the
    general NumPy path costs ~100 us per array; anything else falls back to
    ``np.save`` itself so the accepted input set is unchanged.
    """
    dtype_obj = array.dtype
    if (
        dtype_obj.hasobject
        or dtype_obj.fields is not None
        or dtype_obj.subdtype is not None
    ):
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        return buffer.getvalue()
    header = "{'descr': %r, 'fortran_order': False, 'shape': %r, }" % (
        dtype_obj.str,
        tuple(array.shape),
    )
    header += " " * ((-(10 + len(header) + 1)) % 64) + "\n"
    return (
        _NPY_MAGIC
        + struct.pack("<H", len(header))
        + header.encode("latin1")
        + array.tobytes(order="C")
    )


def _parse_npy(raw: bytes) -> np.ndarray:
    """Parse a ``.npy`` v1/v2/v3 container (C order, no pickles) into an ndarray.

    This is the read counterpart of :func:`_npy_bytes` and accepts exactly what
    ``np.load(..., allow_pickle=False)`` accepts for the constant table: the
    magic + version, a header length and a dict header, then a raw payload of
    the recorded dtype and shape.
    """
    try:
        if len(raw) < 10 or raw[:6] != b"\x93NUMPY":
            raise ValueError("missing .npy magic")
        major = raw[6]
        if major == 1:
            header_len = struct.unpack("<H", raw[8:10])[0]
            start = 10
        elif major in (2, 3):
            if len(raw) < 12:
                raise ValueError("truncated .npy header")
            header_len = struct.unpack("<I", raw[8:12])[0]
            start = 12
        else:
            raise ValueError(f"unsupported .npy version {major}")
        header_end = start + header_len
        if header_end > len(raw):
            raise ValueError("truncated .npy header")
        header = ast.literal_eval(raw[start:header_end].decode("latin1"))
        if not isinstance(header, dict):
            raise ValueError("malformed .npy header")
        if header.get("fortran_order") is not False:
            raise ValueError("Fortran-order arrays are not supported")
        descr = header.get("descr")
        shape = header.get("shape")
        if not isinstance(descr, (str, list)) or not isinstance(shape, tuple):
            raise ValueError("malformed .npy header")
        if not all(
            isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0
            for dim in shape
        ):
            raise ValueError("malformed .npy shape")
        dtype_obj = np.dtype(descr)
        if dtype_obj.hasobject:
            raise ValueError("object arrays are not supported")
        expected = 1
        for dim in shape:
            expected *= dim
        flat = np.frombuffer(raw, dtype=dtype_obj, offset=header_end)
        if flat.size != expected:
            raise ValueError(
                f"payload holds {flat.size} items, header says {expected}"
            )
        view = flat.reshape(shape)
    except (ValueError, TypeError, SyntaxError, struct.error) as exc:
        raise VerificationError(f"constant payload is corrupt: {exc}") from exc
    return np.array(view, dtype=dtype_obj, order="C", copy=True)


def _npy_to_b64(array: np.ndarray) -> str:
    """Base64 of the ``np.save`` bytes of ``array``."""
    return base64.b64encode(_npy_bytes(array)).decode("ascii")


def _npy_from_b64(data_b64) -> np.ndarray:
    """Rebuild an ndarray from base64 ``np.save`` bytes."""
    if not isinstance(data_b64, str):
        raise VerificationError("constant 'data_b64' must be a base64 string")
    try:
        raw = base64.b64decode(data_b64.encode("ascii"), validate=True)
    except ValueError as exc:
        raise VerificationError(f"constant 'data_b64' is not valid base64: {exc}") from exc
    return _parse_npy(raw)


# ---------------------------------------------------------------------------
# Inline payload (the native envelope body)
# ---------------------------------------------------------------------------


def _raw_array_to_b64(array: np.ndarray) -> str:
    """Base64 of ``array``'s raw C-order little-endian payload (native inline form)."""
    converted = array.astype(
        np.dtype(_dtype_name(array.dtype)).newbyteorder("<"), copy=False
    )
    return base64.b64encode(converted.tobytes(order="C")).decode("ascii")


def _encode_inline_ndarray(array: np.ndarray) -> dict:
    """Encode ``array`` as a self-contained ``{"__evoxc_ndarray__": payload}``."""
    return {
        _NDARRAY_TAG: {
            "dtype": _dtype_name(array.dtype),
            "shape": [int(dim) for dim in array.shape],
            "data_b64": _raw_array_to_b64(array),
        }
    }


def _decode_inline_ndarray(inner) -> np.ndarray:
    """Decode one self-contained ndarray payload into an owning native-order array."""
    if not isinstance(inner, dict):
        raise VerificationError(f"invalid inline ndarray payload {inner!r}")
    name = inner.get("dtype")
    shape = inner.get("shape")
    data_b64 = inner.get("data_b64")
    if not isinstance(name, str):
        raise VerificationError(f"invalid inline ndarray payload {inner!r}: bad 'dtype'")
    if not isinstance(shape, (list, tuple)) or not all(_is_int(dim) for dim in shape):
        raise VerificationError(f"invalid inline ndarray payload {inner!r}: bad 'shape'")
    if not isinstance(data_b64, str):
        raise VerificationError(
            f"invalid inline ndarray payload {inner!r}: bad 'data_b64'"
        )
    try:
        dtype_obj = _dtype_from_name(name)
    except TypeError as exc:
        raise VerificationError(
            f"invalid inline ndarray dtype name {name!r}: {exc}"
        ) from exc
    try:
        raw = base64.b64decode(data_b64.encode("ascii"), validate=True)
    except ValueError as exc:
        raise VerificationError(
            f"inline ndarray 'data_b64' is not valid base64: {exc}"
        ) from exc
    try:
        flat = np.frombuffer(raw, dtype=dtype_obj.newbyteorder("<"))
        view = flat.reshape(tuple(int(dim) for dim in shape))
    except ValueError as exc:
        raise VerificationError(
            f"malformed inline ndarray payload for dtype {name!r} and shape "
            f"{list(shape)!r}: {exc}"
        ) from exc
    return np.array(view, dtype=dtype_obj, order="C", copy=True)


# ---------------------------------------------------------------------------
# Marker encoding / decoding
# ---------------------------------------------------------------------------


def _encode_ndarray(array: np.ndarray, constants: dict, inline: bool = False) -> dict:
    """Store ``array`` in the constants table (or inline it); return the marker."""
    if inline:
        return _encode_inline_ndarray(array)
    key = f"c{len(constants)}"
    constants[key] = {
        "dtype": _dtype_name(array.dtype),
        "shape": [int(dim) for dim in array.shape],
        "data_b64": _npy_to_b64(array),
    }
    return {_NDARRAY_TAG: key}


def _decode_ndarray_ref(encoded, constants: dict) -> np.ndarray:
    """Resolve a ``{"__evoxc_ndarray__": ...}`` marker to its ndarray.

    The marker holds either a constants-table key (the public payload) or a
    self-contained inline payload (the native envelope body).
    """
    if not isinstance(encoded, dict) or set(encoded.keys()) != {_NDARRAY_TAG}:
        raise VerificationError(f"invalid ndarray reference {encoded!r}")
    key = encoded[_NDARRAY_TAG]
    if isinstance(key, dict):
        return _decode_inline_ndarray(key)
    entry = constants.get(key)
    if not isinstance(entry, dict):
        raise VerificationError(
            f"constant '{key}' is referenced but not present in the constants table"
        )
    try:
        array = _npy_from_b64(entry["data_b64"])
    except (KeyError, TypeError) as exc:
        raise VerificationError(f"constant '{key}' is malformed: {exc}") from exc
    recorded_shape = entry.get("shape")
    if not isinstance(recorded_shape, list) or not all(
        _is_int(dim) for dim in recorded_shape
    ):
        raise VerificationError(f"constant '{key}' has no valid recorded 'shape'")
    if _dtype_name(array.dtype) != entry.get("dtype") or tuple(
        int(d) for d in array.shape
    ) != tuple(recorded_shape):
        raise VerificationError(
            f"constant '{key}': payload does not match its recorded dtype/shape"
        )
    return array
