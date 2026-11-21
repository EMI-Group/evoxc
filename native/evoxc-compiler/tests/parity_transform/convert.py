"""Convert between the Python evoxc IR payload and the native envelope.

Two wire formats exist for the same IR:

* ``evoxc/ir/serialize.py`` (``format: "evoxc-ir"``) — the Python payload.
  An ndarray attribute is ``{"__evoxc_ndarray__": "<constants key>"}`` and the
  top-level ``constants`` table maps that key to
  ``{"dtype", "shape", "data_b64"}`` where ``data_b64`` decodes to ``np.save``
  bytes (NPY file format).  A ``sha256`` key covers the canonical body.
* ``evoxc-native-ir`` (``native/CONTEXT.md``) — the Rust envelope.  The
  payloads are self-contained: the same marker holds the payload inline as
  ``{"dtype": str, "shape": [ints], "data_b64": str}`` where ``data_b64`` is
  base64 of the array's RAW C-ORDER LITTLE-ENDIAN bytes (no NPY header).

Everything else (dims, types, locations, block/op layout) is byte-compatible,
and tuple markers are passed through VERBATIM — the converter never flattens a
``{"__evoxc_tuple__": [...]}`` marker, so native code decodes every tuple-valued
attribute through its real tag-directed path.  The conversion therefore only
rewrites ndarray payloads and the two top-level discriminators
(``format``/``version`` plus ``sha256``/``constants``).
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import pathlib

import numpy as np

#: Repository root (this file lives at native/evoxc-compiler/tests/parity_transform/).
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
#: The installed PyO3 extension built by ``scripts/build-native.sh``.
NATIVE_SO = REPO_ROOT / "evoxc" / "_evoxc_native.abi3.so"

_NDARRAY_TAG = "__evoxc_ndarray__"
_TUPLE_TAG = "__evoxc_tuple__"

#: Dtype names accepted as byte-order-clean round trips.
_OK_BYTEORDERS = {"=", "|", "<"}


def load_native():
    """Load ``evoxc._evoxc_native`` from the built ``.so`` (never ``import evoxc._evoxc_native``).

    ``import evoxc._evoxc_native`` would only work if the extension were
    importable as part of the package; loading the file directly keeps the
    harness independent of how the package is installed, and matches
    ``native/evoxc-python/verify_transform.py``.
    """
    if not NATIVE_SO.is_file():
        raise FileNotFoundError(
            f"missing native extension {NATIVE_SO}; build it with "
            "`source scripts/dev-env.sh && scripts/build-native.sh --debug`"
        )
    spec = importlib.util.spec_from_file_location("evoxc._evoxc_native", NATIVE_SO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# ndarray payloads
# ---------------------------------------------------------------------------
def _npy_to_array(entry: dict) -> np.ndarray:
    """Decode one Python ``constants`` entry (``np.save`` bytes) to an ndarray."""
    raw = base64.b64decode(entry["data_b64"].encode("ascii"), validate=True)
    with io.BytesIO(raw) as buffer:
        array = np.load(buffer, allow_pickle=False)
    if array.dtype.name != entry["dtype"]:
        raise ValueError(
            f"constant payload dtype {array.dtype.name!r} does not match the "
            f"recorded {entry['dtype']!r}"
        )
    if tuple(int(d) for d in array.shape) != tuple(int(d) for d in entry["shape"]):
        raise ValueError(
            f"constant payload shape {array.shape!r} does not match the "
            f"recorded {tuple(entry['shape'])!r}"
        )
    return array


def _array_to_inline(array: np.ndarray) -> dict:
    """Encode an ndarray as the native inline payload (raw little-endian bytes)."""
    if array.dtype.byteorder not in _OK_BYTEORDERS:
        raise ValueError(
            f"cannot convert a {array.dtype.byteorder}-byte-order array "
            f"({array.dtype!r}) to the native little-endian payload"
        )
    native = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return {
        "dtype": array.dtype.name,
        "shape": [int(dim) for dim in array.shape],
        "data_b64": base64.b64encode(
            np.ascontiguousarray(native).tobytes(order="C")
        ).decode("ascii"),
    }


def _inline_to_array(inner: dict) -> np.ndarray:
    """Decode one native inline payload back to an ndarray."""
    dtype = np.dtype(inner["dtype"])
    if dtype.byteorder not in _OK_BYTEORDERS:
        raise ValueError(f"native payload carries a non-little-endian dtype {dtype!r}")
    dtype = dtype.newbyteorder("<")
    raw = base64.b64decode(inner["data_b64"].encode("ascii"), validate=True)
    array = np.frombuffer(raw, dtype=dtype)
    shape = tuple(int(d) for d in inner["shape"])
    expected = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if array.size != expected:
        raise ValueError(
            f"native payload has {array.size} elements but shape {shape} needs "
            f"{expected}"
        )
    return array.reshape(shape)


# ---------------------------------------------------------------------------
# Python payload -> native envelope
# ---------------------------------------------------------------------------
def _to_native_value(value, constants: dict):
    """Recursively rewrite one payload value, inlining ndarray payloads."""
    if isinstance(value, list):
        return [_to_native_value(item, constants) for item in value]
    if isinstance(value, dict):
        keys = set(value.keys())
        if keys == {_NDARRAY_TAG}:
            key = value[_NDARRAY_TAG]
            if not isinstance(key, str):
                raise ValueError(
                    f"unexpected already-inline ndarray payload {value!r} in a "
                    "Python IR payload"
                )
            if key not in constants:
                raise ValueError(
                    f"ndarray attribute references missing constants key {key!r}"
                )
            return {_NDARRAY_TAG: _array_to_inline(_npy_to_array(constants[key]))}
        if keys == {_TUPLE_TAG}:
            return {
                _TUPLE_TAG: [
                    _to_native_value(item, constants) for item in value[_TUPLE_TAG]
                ]
            }
        return {str(key): _to_native_value(item, constants) for key, item in value.items()}
    return value


def to_native(module) -> dict:
    """Convert a Python ``ir.Module`` into an ``evoxc-native-ir`` envelope."""
    from evoxc import ir

    payload = ir.serialize_module(module)
    body = {key: value for key, value in payload.items() if key not in ("sha256", "constants")}
    envelope = _to_native_value(body, payload.get("constants", {}))
    envelope["format"] = "evoxc-native-ir"
    envelope["version"] = 1
    return envelope


# ---------------------------------------------------------------------------
# native envelope -> Python payload / Module
# ---------------------------------------------------------------------------
def _from_native_value(value, constants: dict):
    """Recursively rewrite one native value, hoisting inline arrays into ``constants``."""
    if isinstance(value, list):
        return [_from_native_value(item, constants) for item in value]
    if isinstance(value, dict):
        keys = set(value.keys())
        if keys == {_NDARRAY_TAG}:
            inner = value[_NDARRAY_TAG]
            if isinstance(inner, str):
                raise ValueError(
                    f"unexpected constants reference {value!r} in a native envelope"
                )
            array = _inline_to_array(inner)
            key = f"c{len(constants)}"
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            constants[key] = {
                "dtype": array.dtype.name,
                "shape": [int(dim) for dim in array.shape],
                "data_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            }
            return {_NDARRAY_TAG: key}
        if keys == {_TUPLE_TAG}:
            return {
                _TUPLE_TAG: [
                    _from_native_value(item, constants) for item in value[_TUPLE_TAG]
                ]
            }
        return {str(key): _from_native_value(item, constants) for key, item in value.items()}
    return value


def _sha256(payload: dict) -> str:
    """The Python IR payload's canonical integrity hash."""
    try:  # prefer the owning implementation when the package is importable
        from evoxc.ir.serialize import _sha256 as python_sha256

        return python_sha256(payload)
    except Exception:  # pragma: no cover - replicated form below
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(body.encode("utf-8")).hexdigest()


def to_python_payload(envelope: dict) -> dict:
    """Convert a native envelope into an ``evoxc-ir`` payload dict."""
    body = {
        key: value
        for key, value in envelope.items()
        if key not in ("sha256", "constants", "transform")
    }
    constants: dict = {}
    payload = _from_native_value(body, constants)
    payload["format"] = "evoxc-ir"
    payload["version"] = 1
    payload["constants"] = constants
    payload["sha256"] = _sha256(payload)
    return payload


def from_native(envelope: dict):
    """Convert a native envelope back into a Python ``ir.Module`` (re-verified)."""
    from evoxc import ir

    return ir.deserialize_module(to_python_payload(envelope))
