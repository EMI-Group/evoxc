"""Private bridge between a Python ``ir.Module`` and the native ``evoxc-native-ir`` envelope.

This module is the in-process wire seam the optional Rust/PyO3 extension
(``evoxc._evoxc_native``, loaded by :mod:`evoxc._native`) speaks: an
``ir.Module`` is converted into a plain nested ``dict`` of Python builtins and
strings, and an envelope returned by native code is converted back into a
fully wired, ``ir.verify``-clean ``ir.Module``.

It is deliberately **not** the persistence format (``evoxc.persist`` owns the
on-disk format) and it imports only ``evoxc.core``, ``evoxc.ir``, NumPy and the
standard library.

The envelope is ``evoxc-native-ir`` version 1 (``ENVELOPE_FORMAT``), documented
in ``native/CONTEXT.md`` ("Wire format: the envelope").  Its body is the
``evoxc-ir`` payload layout — same ``module``/``functions``/``ops`` layout, same
value ids, same dim encodings — with two deliberate differences:

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
and are passed through VERBATIM (contents are still walked natively); native
code decodes every tuple through its own tag-directed path.

Conversion (mirrors ``native/evoxc-compiler/tests/parity_transform/convert.py``):

* :func:`encode_module` serializes through
  :func:`evoxc.ir.serialize._serialize_native_payload`, which verifies the
  module (exactly like the public ``ir.serialize_module``), emits every ndarray
  inline and skips the integrity hash — no second walk of the payload.  The
  input module is only read (never mutated, no counter advance).
* :func:`decode_module` validates the envelope discriminator/version and hands
  the body straight to :func:`evoxc.ir.serialize._deserialize_envelope_body`,
  which rebuilds the module from the self-contained inline payloads (no
  constants-table hoist, no re-encode) and re-verifies it.  Original ids,
  parent pointers and ``uses`` are preserved.
* :func:`decode_native_module` is the TRUSTED fast decoder for an envelope the
  native engine just produced from a module this runtime encoded (the
  transform seam): same discriminator/version check, then
  :func:`evoxc.ir._native_decode.build_module_fast` builds the IR objects directly
  — no generic ``_require_*`` validation, no sha256 and no ``ir.verify``.
  The result is structurally identical; ``EVOXC_NATIVE_DECODE_CHECK=1`` turns
  the verification cross-check back on for development.

Every malformed envelope is a loud error from this module or the
``VerificationError`` raised by the rebuild/``verify`` for body violations —
never a silent coercion.
"""

from __future__ import annotations

import os
from numbers import Integral
from typing import Any

import numpy as np

from evoxc import ir
from evoxc.core import Dim, DimExpr
from evoxc.ir import _native_decode, serialize as _ir_serialize

__all__ = [
    "ENVELOPE_FORMAT",
    "encode_module",
    "decode_module",
    "decode_native_module",
    "encode_type",
    "decode_type",
]

#: Envelope version this codec reads and writes (the native ``VERSION``).
ENVELOPE_FORMAT = 1

#: Development-only cross-check for :func:`decode_native_module`: when set (and
#: not ``0``/empty) at import time, the fast decode is followed by ``ir.verify``.
_VERIFY_NATIVE_DECODE = os.environ.get("EVOXC_NATIVE_DECODE_CHECK", "") not in (
    "",
    "0",
    "false",
    "False",
)

#: Wire discriminator of the native envelope.
_NATIVE_FORMAT = "evoxc-native-ir"

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
# Module encoding / decoding
# ---------------------------------------------------------------------------


def encode_module(module: ir.Module) -> dict:
    """Encode ``module`` as an ``evoxc-native-ir`` v1 envelope dict.

    Read-only: the module is never mutated (no counter advance, no ``uses``
    edits), and it must verify (the serializer verifies it before emitting the
    envelope, exactly as the public ``ir.serialize_module`` does).
    """
    if not isinstance(module, ir.Module):
        raise TypeError(f"encode_module expects an ir.Module, got {type(module).__name__}")
    payload = _ir_serialize._serialize_native_payload(module)
    envelope = {
        key: value
        for key, value in payload.items()
        if key not in ("format", "version", "constants", "sha256")
    }
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
        ValueError: For a malformed envelope discriminator/version.
        evoxc.core.VerificationError: Structural violations found while
            rebuilding the body or by ``ir.verify``.
    """
    return _ir_serialize._deserialize_envelope_body(_validated_envelope_body(envelope))


def decode_native_module(
    envelope: Any, *, verify_module: bool | None = None
) -> ir.Module:
    """Rebuild an ``ir.Module`` from a TRUSTED native envelope (fast path).

    This is the hot-path decoder for an envelope produced in-process by the
    native transform engine from a module this runtime encoded moments
    earlier — it constructs the IR objects directly (no generic
    ``_require_*`` validation, no sha256, no serialize/deserialize round trip)
    and only checks the envelope discriminator/version.  The result is
    structurally identical to :func:`decode_module`'s (same ids, attributes,
    ordered complete ``uses``, parent pointers and advanced id counters).

    ``verify_module`` defaults to the ``EVOXC_NATIVE_DECODE_CHECK`` environment
    flag (read once at import, off by default), which is the development-only
    cross-check: it runs ``ir.verify`` over the fast-decoded module.
    """
    module = _native_decode.build_module_fast(_validated_envelope_body(envelope))
    if verify_module is None:
        verify_module = _VERIFY_NATIVE_DECODE
    if verify_module:
        ir.verify(module)
    return module


def _validated_envelope_body(envelope: Any) -> dict:
    """Check the envelope discriminator/version and return its body mapping."""
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
    return {key: envelope[key] for key in _BODY_KEYS if key in envelope}
