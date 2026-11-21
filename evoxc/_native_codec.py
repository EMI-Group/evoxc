"""Private bridge between a Python ``ir.Module`` and the native ``evoxc-native-ir`` envelope.

This module is the in-process wire seam the optional Rust/PyO3 extension
(``evoxc._evoxc_native``, loaded by :mod:`evoxc._native`) speaks: an
``ir.Module`` is converted into a plain nested ``dict`` of Python builtins and
strings, and an envelope returned by native code is converted back into a
fully wired, ``ir.verify``-clean ``ir.Module``.

It is deliberately **not** the persistence format (``evoxc.persist`` owns the
on-disk format) and it imports only ``evoxc.core``, ``evoxc.ir``, NumPy and
the standard library.

The envelope is ``evoxc-native-ir`` version 1 (``ENVELOPE_FORMAT``), documented
in ``native/CONTEXT.md`` ("Wire format: the envelope").  Its body is the
``evoxc-ir`` payload produced by :func:`evoxc.ir.serialize_module` — same
``module``/``functions``/``ops`` layout, same value ids, same dim encodings —
with two deliberate differences:

* the discriminator is ``{"format": "evoxc-native-ir", "version": 1}`` instead
  of ``{"format": "evoxc-ir", "version": ir.IR_FORMAT_VERSION}``, and there is
  no ``sha256`` key (the envelope is an in-process value, not a stored file);
* ndarray attribute payloads are **self-contained** rather than hoisted into a
  ``constants`` table: an attribute value is
  ``{"__evoxc_ndarray__": {"dtype": name, "shape": [ints], "data_b64": str}}``
  where ``data_b64`` is base64 of the array's raw C-order little-endian bytes
  (the Python payload instead points at ``constants[key]``, whose ``data_b64``
  is a base64 ``npy`` file).

Type and dim encodings (shared with the Python payload)::

    Type = {"dtype": str, "shape": [Dim, ...]}      # dtype = numpy dtype.name
    Dim  = {"int": n}
         | {"dim": name, "size": n?}                # symbolic Dim, optional size
         | {"expr": {"op": "add"|"sub"|"mul"|"floordiv"|"mod"|"min"|"max",
                     "args": [Dim, Dim]}}
         | None                                     # runtime-dynamic

Tuples that must survive the round trip encode as ``{"__evoxc_tuple__": [...]}``
and are passed through VERBATIM (contents are still walked, so nested array
payloads convert); native code decodes every tuple through its own tag-directed
path.

Conversion (mirrors ``native/evoxc-compiler/tests/parity_transform/convert.py``):

* :func:`encode_module` serializes with :func:`evoxc.ir.serialize_module`,
  drops ``sha256`` and the ``constants`` table, inlines every ndarray payload
  and stamps the native discriminator.  The input module is only read (never
  mutated, no counter advance).
* :func:`decode_module` reverses it: inline payloads are hoisted back into a
  ``constants`` table (``npy`` base64, the form :mod:`evoxc.ir.serialize`
  uses), the Python discriminator and version are restored, the ``sha256`` is
  recomputed, and :func:`evoxc.ir.deserialize_module` rebuilds and re-verifies
  the module (original ids, parent pointers and ``uses`` preserved).

Every malformed envelope is a loud ``ValueError`` from this module, or the
``PersistenceError``/``VerificationError`` raised by
:func:`evoxc.ir.deserialize_module` for body violations — never a silent
coercion.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from numbers import Integral
from typing import Any

import numpy as np

from evoxc import ir
from evoxc.core import Dim, DimExpr

__all__ = ["ENVELOPE_FORMAT", "encode_module", "decode_module", "encode_type", "decode_type"]

#: Envelope version this codec reads and writes (the native ``VERSION``).
ENVELOPE_FORMAT = 1

#: Wire discriminator of the native envelope.
_NATIVE_FORMAT = "evoxc-native-ir"

#: Wire discriminator of the Python IR payload this codec converts to/from.
_PYTHON_FORMAT = "evoxc-ir"

#: Top-level keys that belong to the envelope body (everything else is metadata
#: a caller may carry through untouched).
_ENVELOPE_KEYS = frozenset(
    {"format", "version", "module", "functions", "ops", "constants", "sha256"}
)

#: Envelope body keys carried into the python payload on decode; the
#: discriminators, a ``constants`` table and a ``sha256`` are rebuilt, and any
#: unrelated top-level key is ignored.
_BODY_KEYS = ("module", "functions", "ops")

#: Attribute-value marker for numpy arrays (self-contained in the native form).
_NDARRAY_TAG = "__evoxc_ndarray__"

#: Attribute-value marker preserving tuples through the JSON round-trip.
_TUPLE_TAG = "__evoxc_tuple__"


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------


def _require_key(mapping: dict, key: str, where: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{where}: missing required key {key!r}")
    return mapping[key]


def _is_int(value: Any) -> bool:
    """True for plain ints (bools excluded)."""
    return isinstance(value, Integral) and not isinstance(value, bool)


# ---------------------------------------------------------------------------
# Type (ValueType) encoding
# ---------------------------------------------------------------------------


def encode_type(value_type: ir.ValueType) -> dict:
    """Encode an :class:`ir.ValueType` as ``{"dtype": name, "shape": [dim, ...]}``.

    ``shape`` uses the native dim encodings (:func:`_encode_dim`).
    """
    if not isinstance(value_type, ir.ValueType):
        raise TypeError(
            f"encode_type expects an ir.ValueType, got {type(value_type).__name__}"
        )
    return {
        "dtype": value_type.dtype.name,
        "shape": [_encode_dim(dim) for dim in value_type.shape],
    }


def decode_type(payload: Any) -> ir.ValueType:
    """Decode a native ``{"dtype": ..., "shape": [...]}`` dict into an ``ir.ValueType``."""
    if not isinstance(payload, dict):
        raise ValueError(
            f"malformed value type: expected a dict, got {type(payload).__name__}"
        )
    dtype_name = _require_key(payload, "dtype", "value type")
    shape = _require_key(payload, "shape", "value type")
    if not isinstance(dtype_name, str):
        raise ValueError(
            f"malformed value type: 'dtype' must be a string, got {dtype_name!r}"
        )
    if not isinstance(shape, (list, tuple)):
        raise ValueError(
            f"malformed value type: 'shape' must be a list, got {shape!r}"
        )
    try:
        dtype = np.dtype(dtype_name)
    except TypeError as exc:
        raise ValueError(f"invalid dtype name {dtype_name!r}: {exc}") from exc
    return ir.ValueType(dtype, tuple(_decode_dim(dim) for dim in shape))


def _encode_dim(dim: Any) -> Any:
    """Encode one shape dimension in the native form (``int | None | Dim | DimExpr``)."""
    if dim is None:
        return None
    if isinstance(dim, bool):
        raise TypeError(f"a shape dimension cannot be a bool: {dim!r}")
    if isinstance(dim, Dim):
        encoded: dict = {"dim": dim.name}
        if dim.size is not None:
            encoded["size"] = int(dim.size)
        return encoded
    if isinstance(dim, DimExpr):
        return {
            "expr": {
                "op": dim.op,
                "args": [_encode_dim(dim.left), _encode_dim(dim.right)],
            }
        }
    if isinstance(dim, Integral):
        return {"int": int(dim)}
    raise TypeError(
        f"cannot encode shape dimension {dim!r}: expected int, None, Dim or DimExpr, "
        f"got {type(dim).__name__}"
    )


def _decode_dim(raw: Any) -> Any:
    """Decode one shape dimension from its native form."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        if "expr" in raw:
            expr = raw["expr"]
            if not isinstance(expr, dict):
                raise ValueError(f"malformed dimension: 'expr' must be a dict, got {expr!r}")
            op = expr.get("op")
            args = expr.get("args")
            if not isinstance(op, str):
                raise ValueError(f"malformed dimension: expr op must be str, got {op!r}")
            if not isinstance(args, (list, tuple)) or len(args) != 2:
                raise ValueError(
                    f"malformed dimension: expr 'args' must be a 2-element list, got {args!r}"
                )
            try:
                return DimExpr(op, _decode_dim(args[0]), _decode_dim(args[1]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"malformed dimension: invalid expr {raw!r}: {exc}") from exc
        if "dim" in raw:
            name = raw["dim"]
            size = raw.get("size")
            if not isinstance(name, str):
                raise ValueError(f"malformed dimension: 'dim' must be a str, got {name!r}")
            if size is not None and not _is_int(size):
                raise ValueError(f"malformed dimension: 'size' must be an int, got {size!r}")
            return Dim(name, None if size is None else int(size))
        if "int" in raw:
            value = raw["int"]
            if not _is_int(value):
                raise ValueError(f"malformed dimension: 'int' must hold an int, got {value!r}")
            return int(value)
    raise ValueError(
        f"malformed dimension encoding {raw!r}: expected null, "
        "{'int': n}, {'dim': name, 'size': n} or {'expr': {'op': ..., 'args': [...]}}"
    )


# ---------------------------------------------------------------------------
# ndarray payloads
# ---------------------------------------------------------------------------


def _array_to_inline(array: np.ndarray) -> dict:
    """Encode an ndarray as the native inline payload (raw little-endian bytes)."""
    name = array.dtype.name
    little_endian = np.dtype(name).newbyteorder("<")
    converted = array.astype(little_endian, copy=False)
    return {
        "dtype": name,
        "shape": [int(size) for size in array.shape],
        "data_b64": base64.b64encode(converted.tobytes(order="C")).decode("ascii"),
    }


def _inline_to_array(inner: Any, where: str) -> np.ndarray:
    """Decode one native inline payload into an owning, native-order ndarray."""
    if not isinstance(inner, dict):
        raise ValueError(
            f"{where}: inline ndarray payload must be a dict, got {type(inner).__name__}"
        )
    name = _require_key(inner, "dtype", f"{where} ndarray")
    shape = _require_key(inner, "shape", f"{where} ndarray")
    data_b64 = _require_key(inner, "data_b64", f"{where} ndarray")
    if not isinstance(name, str):
        raise ValueError(f"{where}: ndarray 'dtype' must be a string, got {name!r}")
    if not isinstance(shape, (list, tuple)) or not all(_is_int(size) for size in shape):
        raise ValueError(f"{where}: ndarray 'shape' must be a list of ints, got {shape!r}")
    if not isinstance(data_b64, str):
        raise ValueError(f"{where}: ndarray 'data_b64' must be a string, got {data_b64!r}")
    try:
        dtype = np.dtype(name)
    except TypeError as exc:
        raise ValueError(f"{where}: invalid dtype name {name!r}: {exc}") from exc
    try:
        raw = base64.b64decode(data_b64.encode("ascii"), validate=True)
    except ValueError as exc:
        raise ValueError(f"{where}: ndarray 'data_b64' is not valid base64: {exc}") from exc
    try:
        flat = np.frombuffer(raw, dtype=dtype.newbyteorder("<"))
        view = flat.reshape(tuple(int(size) for size in shape))
    except ValueError as exc:
        raise ValueError(
            f"{where}: malformed ndarray payload for dtype {name!r} and shape "
            f"{list(shape)!r}: {exc}"
        ) from exc
    # ``np.array(..., copy=True)`` normalizes byte order to native, makes the
    # array owning and writable, and keeps C order (including 0-d shape ()).
    return np.array(view, dtype=dtype, order="C", copy=True)


def _constants_array(entry: Any, where: str) -> np.ndarray:
    """Decode one python-payload ``constants`` entry (base64 ``npy`` bytes)."""
    if not isinstance(entry, dict):
        raise ValueError(
            f"{where}: constants entry must be a dict, got {type(entry).__name__}"
        )
    name = _require_key(entry, "dtype", f"{where} constants entry")
    shape = _require_key(entry, "shape", f"{where} constants entry")
    data_b64 = _require_key(entry, "data_b64", f"{where} constants entry")
    if not isinstance(data_b64, str):
        raise ValueError(f"{where}: constants 'data_b64' must be a string, got {data_b64!r}")
    try:
        raw = base64.b64decode(data_b64.encode("ascii"), validate=True)
    except ValueError as exc:
        raise ValueError(f"{where}: constants 'data_b64' is not valid base64: {exc}") from exc
    try:
        with io.BytesIO(raw) as buffer:
            array = np.load(buffer, allow_pickle=False)
    except (ValueError, EOFError) as exc:
        raise ValueError(f"{where}: constants payload is corrupt: {exc}") from exc
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{where}: constants payload is not an ndarray")
    if array.dtype.name != name or tuple(int(d) for d in array.shape) != tuple(
        int(d) for d in shape
    ):
        raise ValueError(
            f"{where}: constants payload does not match its recorded dtype/shape "
            f"({name!r}, {list(shape)!r})"
        )
    return array


def _array_to_npy_b64(array: np.ndarray) -> str:
    """Base64 of the ``np.save`` bytes of ``array`` (the python payload form)."""
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Payload rewriting (python payload <-> native envelope body)
# ---------------------------------------------------------------------------


def _to_native_value(value: Any, constants: dict, where: str) -> Any:
    """Recursively rewrite one payload value, inlining ndarray payloads."""
    if isinstance(value, list):
        return [_to_native_value(item, constants, where) for item in value]
    if isinstance(value, dict):
        keys = set(value.keys())
        if keys == {_NDARRAY_TAG}:
            key = value[_NDARRAY_TAG]
            if not isinstance(key, str):
                raise ValueError(
                    f"{where}: unexpected already-inline ndarray payload {value!r}"
                )
            if key not in constants:
                raise ValueError(
                    f"{where}: ndarray attribute references missing constants key {key!r}"
                )
            return {_NDARRAY_TAG: _array_to_inline(_constants_array(constants[key], where))}
        if keys == {_TUPLE_TAG}:
            return {
                _TUPLE_TAG: [
                    _to_native_value(item, constants, where) for item in value[_TUPLE_TAG]
                ]
            }
        return {
            str(key): _to_native_value(item, constants, where)
            for key, item in value.items()
        }
    return value


def _from_native_value(value: Any, constants: dict, where: str) -> Any:
    """Recursively rewrite one native value, hoisting inline arrays into ``constants``."""
    if isinstance(value, list):
        return [_from_native_value(item, constants, where) for item in value]
    if isinstance(value, dict):
        keys = set(value.keys())
        if keys == {_NDARRAY_TAG}:
            inner = value[_NDARRAY_TAG]
            if isinstance(inner, str):
                raise ValueError(
                    f"{where}: unexpected constants reference {value!r} in a native "
                    "envelope (payloads are inline)"
                )
            array = _inline_to_array(inner, where)
            key = f"c{len(constants)}"
            constants[key] = {
                "dtype": array.dtype.name,
                "shape": [int(size) for size in array.shape],
                "data_b64": _array_to_npy_b64(array),
            }
            return {_NDARRAY_TAG: key}
        if keys == {_TUPLE_TAG}:
            return {
                _TUPLE_TAG: [
                    _from_native_value(item, constants, where)
                    for item in value[_TUPLE_TAG]
                ]
            }
        return {
            str(key): _from_native_value(item, constants, where)
            for key, item in value.items()
        }
    return value


def _sha256(payload: dict) -> str:
    """The python payload's canonical integrity hash (mirrors ``ir.serialize``)."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Module encoding / decoding
# ---------------------------------------------------------------------------


def encode_module(module: ir.Module) -> dict:
    """Encode ``module`` as an ``evoxc-native-ir`` v1 envelope dict.

    Read-only: the module is never mutated (no counter advance, no ``uses``
    edits).  The module must verify (``ir.serialize_module`` verifies first).
    """
    if not isinstance(module, ir.Module):
        raise TypeError(f"encode_module expects an ir.Module, got {type(module).__name__}")
    payload = ir.serialize_module(module)
    constants = payload.get("constants", {})
    if not isinstance(constants, dict):
        raise ValueError("malformed IR payload: 'constants' must be a dict")
    body = {
        key: value
        for key, value in payload.items()
        if key not in ("sha256", "constants")
    }
    envelope = _to_native_value(body, constants, "payload")
    envelope["format"] = _NATIVE_FORMAT
    envelope["version"] = ENVELOPE_FORMAT
    return envelope


def decode_module(envelope: Any) -> ir.Module:
    """Rebuild a fully wired ``ir.Module`` from an ``evoxc-native-ir`` envelope.

    The result preserves original op/value ids, attributes, types, locations
    and parent/use pointers; ``ir.verify`` accepts it whenever it accepted the
    module the transform started from.  Module id counters are reset past every
    decoded id.

    Raises:
        ValueError: For a malformed envelope (wrong format/version, a
            constants-style ndarray reference, a bad inline payload, ...).
        evoxc.core.PersistenceError: Wrong python discriminator/version or a
            sha256 mismatch in the rebuilt payload.
        evoxc.core.VerificationError: Structural violations found by
            ``evoxc.ir.deserialize_module``.
    """
    if not isinstance(envelope, dict):
        raise ValueError(
            f"malformed native envelope: expected a dict, got {type(envelope).__name__}"
        )
    envelope_format = envelope.get("format")
    if envelope_format != _NATIVE_FORMAT:
        raise ValueError(
            f"unsupported native envelope format {envelope_format!r} "
            f"(this codec accepts only {_NATIVE_FORMAT!r})"
        )
    envelope_version = envelope.get("version")
    if envelope_version != ENVELOPE_FORMAT:
        raise ValueError(
            f"unsupported native envelope version {envelope_version!r} "
            f"(this codec accepts only {ENVELOPE_FORMAT})"
        )
    body = {
        key: envelope[key] for key in _BODY_KEYS if key in envelope
    }
    constants: dict = {}
    payload = _from_native_value(body, constants, "envelope")
    payload["format"] = _PYTHON_FORMAT
    payload["version"] = ir.IR_FORMAT_VERSION
    payload["constants"] = constants
    payload["sha256"] = _sha256(payload)
    return ir.deserialize_module(payload)
