"""IR serialization: versioned, self-describing JSON-able payloads.

``serialize_module``/``deserialize_module`` implement the core of the
``.evoxcgraph`` format: the payload dict returned here is what ``evoxc.persist``
wraps in its own container (magic header + JSON metadata + integrity hash).
The IR payload itself is already self-describing and integrity-checked —
a SHA-256 over the canonical JSON of its content — so any wrapper stays dumb.

Payload schema (``serialize_module`` result):

    {
      "format": "evoxc-ir",                 # discriminator
      "version": 1,                       # IR_FORMAT_VERSION
      "module": {"name": str, "metadata": {...}},
      "functions": [
        {
          "name": str,
          "input_types":  [type, ...],    # type = {"dtype": str, "shape": [dim, ...]}
          "output_types": [type, ...],    # informational; recomputed on load
                                          # from the return terminator
          "metadata": {...},              # extension: free-form JSON-able notes
          "block": {
            "arguments": [{"id": int, "type": type}, ...],
            "ops": [{"id": int, "ref": str}, ...]     # ref into "ops" table
          }
        }, ...
      ],
      "ops": [
        {
          "id": int,
          "name": str,
          "operands": [value_id, ...],
          "results":  [{"id": int, "type": type}, ...],
          "attributes": {...},            # ndarray attrs become {"__evoxc_ndarray__": "k"}
          "regions":  [region, ...],      # same block encoding as functions
          "location": {"file","line","col","code_snippet"} | null
        }, ...
      ],
      "constants": {"<k>": {"dtype": str, "shape": [int, ...], "data_b64": str}},
      "sha256": "<hex digest>"
    }

Encoding notes (binding):
- ``dim``: ``{"int": n}`` | ``{"dim": "name"}`` (symbolic Dim; a known size is
  carried as an extra ``"size": n`` field — ``{"dim": "name", "size": n}``) |
  ``{"expr": {"op": "<name>", "args": [dim, dim]}}`` (compound DimExpr;
  ``args`` is the binary ``[left, right]`` pair of ``core.DimExpr``) |
  ``null`` (runtime-dynamic). Decoding rebuilds ``Dim``/``DimExpr`` objects,
  preserving symbolic identity (names, sizes, expression trees).
- ``dtype``: numpy ``dtype.name`` string (bool/int8/.../float64/complex128).
- Constant payloads: base64 of ``np.save`` bytes — the only numpy involvement.
  Any attribute value that is a numpy array is stored in the ``constants``
  table as ``{"dtype": dtype.name, "shape": [int, ...], "data_b64": base64}``
  and the attribute value becomes ``{"__evoxc_ndarray__": "<k>"}`` (the table
  key). On load the payload is decoded and checked against the recorded
  dtype/shape (mismatch -> VerificationError).
- Tuples are wrapped as ``{"__evoxc_tuple__": [...]}`` so the tuple/list
  distinction survives the JSON round-trip (``json`` itself flattens tuples
  to arrays); attribute sequence tags decode back to the original shape.
- ``runtime_call``/``block_call`` ``result_specs``: encoded as a list of type
  dicts ``{"dtype": str, "shape": [dim, ...]}`` (one per declared result);
  on decode every entry is NORMALIZED back to a ``ValueType`` instance (the
  in-memory form the Builder produces), so ``verify``'s in-memory comparison
  of ``result_specs`` vs results still passes after a round-trip.
- Block op lists reference the FLAT ``ops`` table via ``"ref"``: the op id as
  a decimal STRING (``{"id": int, "ref": str(op_id)}`` — the string is the
  authoritative reference; the int mirrors it). Operands reference values by
  their module-unique int ids; values are defined inline by ops (``results``)
  and blocks (``arguments``).
- A region with a single block (v1 shape) is encoded as that block's encoding
  directly; a multi-block region is encoded as ``{"blocks": [block, ...]}``
  (reserved for future control-flow forms; the decoder accepts both).
- The sha256 is computed over ``json.dumps(payload_without_sha256,
  sort_keys=True, separators=(",", ":"))`` (canonical form), and recomputed on
  load: mismatch raises ``VerificationError``; unknown format/version raises
  ``PersistenceError`` (both owned by ``evoxc.core``) — never a silent
  re-derivation. ``serialize_module`` runs ``verify`` first; a failed module
  never serializes.
"""

from __future__ import annotations

import hashlib
import json
from itertools import count

import numpy as np

from evoxc.core import (
    DTypeError,
    Dim,
    DimExpr,
    PersistenceError,
    TensorSpec,
    VerificationError,
    dtype,
)

from ._npy import (
    _NDARRAY_TAG,
    _decode_ndarray_ref,
    _dtype_from_name,
    _dtype_name,
    _encode_ndarray,
    _is_int,
)
from .block import Block
from .function import Function
from .location import Location
from .module import Module
from .op import Op
from .op_defs import ATTR_ANY, ATTR_NDARRAY, ATTR_SHAPE, opdef
from .region import Region
from .types import ValueType
from .value import Use, Value
from .verify import verify
from .version import IR_FORMAT_VERSION

__all__ = ["IR_FORMAT_VERSION", "serialize_module", "deserialize_module"]

#: Wire discriminator of the payload.
IR_FORMAT = "evoxc-ir"

#: Attribute-value marker preserving tuples through the JSON round-trip.
_TUPLE_TAG = "__evoxc_tuple__"

#: Memo tables for the two conversions that repeat once per encoded value.
_ATTR_SCHEMA_CACHE: dict = {}
_VALUE_TYPE_CACHE: dict = {}


def _attr_schema(op_name: str) -> dict:
    """The ``OpDef`` attribute schema of ``op_name`` as ``{attr: AttrSpec}``.

    Rebuilt whenever the registered ``OpDef`` object changes identity, so a
    runtime ``register_opdef`` that replaces a definition is never masked.
    """
    definition = opdef(op_name)
    cached = _ATTR_SCHEMA_CACHE.get(op_name)
    if cached is not None and cached[0] is definition:
        return cached[1]
    schema = {spec.name: spec for spec in definition.attributes}
    _ATTR_SCHEMA_CACHE[op_name] = (definition, schema)
    return schema


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _require(mapping, key: str, where: str):
    """Return ``mapping[key]`` or raise ``VerificationError``."""
    if not isinstance(mapping, dict) or key not in mapping:
        raise VerificationError(f"malformed IR payload: '{key}' is missing from {where}")
    return mapping[key]


def _require_list(mapping, key: str, where: str) -> list:
    value = _require(mapping, key, where)
    if not isinstance(value, list):
        raise VerificationError(
            f"malformed IR payload: '{key}' in {where} must be a list, "
            f"got {type(value).__name__}"
        )
    return value


def _require_dict(mapping, key: str, where: str) -> dict:
    value = _require(mapping, key, where)
    if not isinstance(value, dict):
        raise VerificationError(
            f"malformed IR payload: '{key}' in {where} must be a dict, "
            f"got {type(value).__name__}"
        )
    return value


def _require_str(mapping, key: str, where: str) -> str:
    value = _require(mapping, key, where)
    if not isinstance(value, str):
        raise VerificationError(
            f"malformed IR payload: '{key}' in {where} must be a str, "
            f"got {type(value).__name__}"
        )
    return value


def _require_int(mapping, key: str, where: str) -> int:
    value = _require(mapping, key, where)
    if not _is_int(value):
        raise VerificationError(
            f"malformed IR payload: '{key}' in {where} must be an int, "
            f"got {value!r}"
        )
    return value


# ---------------------------------------------------------------------------
# Integrity hash
# ---------------------------------------------------------------------------


def _canonical_json(payload: dict) -> str:
    """The canonical JSON form the integrity hash covers."""
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"IR payload is not canonical JSON: {exc}") from exc


def _sha256(payload: dict) -> str:
    """Hex SHA-256 of the canonical JSON of ``payload`` (no sha256 key)."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Dim / type / location encoding
# ---------------------------------------------------------------------------


def _encode_dim(dim) -> dict | None:
    """Encode one shape dim: int | Dim | DimExpr | None (runtime-dynamic)."""
    if dim is None:
        return None
    if isinstance(dim, Dim):
        encoded = {"dim": dim.name}
        if dim.size is not None:
            encoded["size"] = dim.size
        return encoded
    if isinstance(dim, DimExpr):
        return {
            "expr": {
                "op": dim.op,
                "args": [_encode_dim(dim.left), _encode_dim(dim.right)],
            }
        }
    if _is_int(dim):
        return {"int": dim}
    raise VerificationError(f"cannot serialize shape dimension {dim!r}")


def _decode_dim(encoded) -> int | Dim | DimExpr | None:
    """Decode one shape dim back to int | Dim | DimExpr | None."""
    if encoded is None:
        return None
    if isinstance(encoded, dict):
        if "expr" in encoded:
            expr = encoded["expr"]
            if not isinstance(expr, dict) or not isinstance(expr.get("op"), str):
                raise VerificationError(f"malformed DimExpr encoding {encoded!r}")
            args = expr.get("args")
            if not isinstance(args, list) or len(args) != 2:
                raise VerificationError(
                    f"malformed DimExpr encoding {encoded!r}: 'args' must be a "
                    "2-element list"
                )
            try:
                return DimExpr(expr["op"], _decode_dim(args[0]), _decode_dim(args[1]))
            except (ValueError, TypeError) as exc:
                raise VerificationError(
                    f"malformed DimExpr encoding {encoded!r}: {exc}"
                ) from exc
        if "dim" in encoded:
            name = encoded["dim"]
            if not isinstance(name, str):
                raise VerificationError(f"malformed Dim encoding {encoded!r}")
            size = encoded.get("size")
            if size is not None and not _is_int(size):
                raise VerificationError(f"malformed Dim encoding {encoded!r}")
            return Dim(name, size=size)
        if "int" in encoded:
            value = encoded["int"]
            if _is_int(value):
                return value
            raise VerificationError(f"malformed dim encoding {encoded!r}")
    raise VerificationError(f"invalid dim encoding {encoded!r}")


def _encode_type(value_type: ValueType) -> dict:
    """Encode a ``ValueType`` as ``{"dtype": name, "shape": [dim, ...]}``."""
    if not isinstance(value_type, ValueType):
        raise VerificationError(
            f"expected a ValueType to encode, got {type(value_type).__name__}"
        )
    return {
        "dtype": _dtype_name(value_type.dtype),
        "shape": [_encode_dim(dim) for dim in value_type.shape],
    }


def _decode_type(encoded) -> ValueType:
    """Decode a type dict back to a ``ValueType``."""
    if not isinstance(encoded, dict):
        raise VerificationError(f"invalid type encoding {encoded!r}")
    dtype_name = encoded.get("dtype")
    if not isinstance(dtype_name, str):
        raise VerificationError(f"invalid type encoding {encoded!r}: missing 'dtype'")
    try:
        dtype_obj = _dtype_from_name(dtype_name)
    except TypeError as exc:
        raise VerificationError(
            f"invalid type encoding {encoded!r}: bad dtype name {dtype_name!r}"
        ) from exc
    shape_data = encoded.get("shape")
    if not isinstance(shape_data, list):
        raise VerificationError(f"invalid type encoding {encoded!r}: missing 'shape'")
    # Static shapes repeat heavily across a large module; a frozen ValueType is
    # immutable, so one instance can serve every value with the same type.
    for dim in shape_data:
        if not (
            isinstance(dim, dict)
            and len(dim) == 1
            and "int" in dim
            and _is_int(dim["int"])
        ):
            return ValueType(dtype_obj, tuple(_decode_dim(dim) for dim in shape_data))
    key = (dtype_name, tuple(dim["int"] for dim in shape_data))
    cached = _VALUE_TYPE_CACHE.get(key)
    if cached is not None:
        return cached
    value_type = ValueType(dtype_obj, key[1])
    if len(_VALUE_TYPE_CACHE) < 4096:
        _VALUE_TYPE_CACHE[key] = value_type
    return value_type


def _encode_location(location) -> dict | None:
    """Encode an ``Op.location`` (None stays null)."""
    if location is None:
        return None
    return {
        "file": location.file,
        "line": location.line,
        "col": location.col,
        "code_snippet": location.code_snippet,
    }


def _decode_location(encoded) -> Location | None:
    """Decode a location dict back to a ``Location`` (None stays None)."""
    if encoded is None:
        return None
    if not isinstance(encoded, dict):
        raise VerificationError(f"invalid location encoding {encoded!r}")
    file = encoded.get("file")
    line = encoded.get("line")
    col = encoded.get("col")
    snippet = encoded.get("code_snippet")
    if (
        not isinstance(file, str)
        or not _is_int(line)
        or not _is_int(col)
        or (snippet is not None and not isinstance(snippet, str))
    ):
        raise VerificationError(f"invalid location encoding {encoded!r}")
    return Location(file=file, line=line, col=col, code_snippet=snippet)


# ---------------------------------------------------------------------------
# Generic attribute-value encoding (JSON-able scalars, containers, specials)
# ---------------------------------------------------------------------------


def _encode_generic(value, constants: dict, inline: bool = False):
    """Recursively encode a JSON-able attribute value.

    Plain scalars pass through; numpy scalars normalize to Python scalars;
    tuples are wrapped in a marker so they survive JSON as tuples; numpy
    arrays go to the constants table (or the inline envelope form); ``Dim``/
    ``DimExpr`` use the dim encoding; ``ValueType`` uses the type encoding.
    """
    if isinstance(value, (np.bool_, np.integer, np.floating, np.str_)):
        value = value.item()
    if value is None or isinstance(value, (bool, str)) or _is_int(value):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, np.ndarray):
        return _encode_ndarray(value, constants, inline)
    if isinstance(value, (Dim, DimExpr)):
        return _encode_dim(value)
    if isinstance(value, ValueType):
        return _encode_type(value)
    if isinstance(value, tuple):
        return {
            _TUPLE_TAG: [
                _encode_generic(item, constants, inline) for item in value
            ]
        }
    if isinstance(value, list):
        return [_encode_generic(item, constants, inline) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _encode_generic(item, constants, inline)
            for key, item in value.items()
        }
    raise VerificationError(
        f"attribute value {value!r} is not serializable (JSON-able scalars/"
        "containers, Dim/DimExpr, ValueType, or numpy arrays only)"
    )


def _decode_generic(value, constants: dict):
    """Recursively decode an attribute value (inverse of ``_encode_generic``).

    Recognizes the ndarray and tuple markers; ``{"dim": ...}`` / ``{"expr":
    ...}`` dicts decode back to ``Dim``/``DimExpr`` (int dims never appear in
    generic positions — they exist only inside shape/type lists, decoded
    schema-driven). ``ValueType``s are only restored in ``result_specs``
    (handled by the attribute schema dispatch); elsewhere they round-trip as
    plain dicts.
    """
    if value is None or isinstance(value, (bool, str)) or _is_int(value):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, list):
        return [_decode_generic(item, constants) for item in value]
    if isinstance(value, dict):
        keys = set(value.keys())
        if keys == {_NDARRAY_TAG}:
            return _decode_ndarray_ref(value, constants)
        if keys == {_TUPLE_TAG}:
            return tuple(_decode_generic(item, constants) for item in value[_TUPLE_TAG])
        if "dim" in value and keys <= {"dim", "size"}:
            return _decode_dim(value)
        if keys == {"expr"}:
            return _decode_dim(value)
        return {
            str(key): _decode_generic(item, constants) for key, item in value.items()
        }
    raise VerificationError(f"cannot decode serialized value {value!r}")


# ---------------------------------------------------------------------------
# result_specs (runtime_call / block_call)
# ---------------------------------------------------------------------------


def _encode_spec(spec) -> dict:
    """Encode one declared result spec as a type dict.

    Accepts the Builder's in-memory forms — ``ValueType`` | ``TensorSpec`` |
    ``{"dtype": ..., "shape": ...}`` — and always emits the type-dict wire
    form; decoding normalizes everything back to ``ValueType``.
    """
    if isinstance(spec, ValueType):
        return _encode_type(spec)
    if isinstance(spec, TensorSpec):
        return _encode_type(ValueType(spec.dtype, tuple(spec.shape)))
    if isinstance(spec, dict):
        try:
            return _encode_type(ValueType(dtype(spec["dtype"]), tuple(spec["shape"])))
        except (KeyError, TypeError, DTypeError) as exc:
            raise VerificationError(
                f"cannot interpret result spec {spec!r} — expected a ValueType, "
                "a TensorSpec, or a {'dtype': ..., 'shape': ...} dict"
            ) from exc
    raise VerificationError(
        f"cannot interpret result spec {spec!r} — expected a ValueType, a "
        "TensorSpec, or a {'dtype': ..., 'shape': ...} dict"
    )


def _decode_spec(encoded) -> ValueType:
    """Decode one wire result spec back to a ``ValueType``."""
    return _decode_type(encoded)


# ---------------------------------------------------------------------------
# Attribute dicts (schema-driven)
# ---------------------------------------------------------------------------


def _encode_attrs(op: Op, constants: dict, inline: bool = False) -> dict:
    """Encode an op's attributes per its ``OpDef`` schema."""
    schema = _attr_schema(op.name)
    encoded = {}
    for key, value in op.attributes.items():
        spec = schema.get(key)
        if spec is None:
            # Unreachable after verify; kept as a clear guard.
            raise VerificationError(f"op '{op.name}': unknown attribute '{key}'")
        if spec.type == ATTR_NDARRAY:
            if not isinstance(value, np.ndarray):
                raise VerificationError(
                    f"op '{op.name}': attribute '{key}' must be a numpy array"
                )
            encoded[key] = _encode_ndarray(value, constants, inline)
        elif spec.type == ATTR_SHAPE:
            encoded[key] = [_encode_dim(dim) for dim in value]
        elif spec.type == ATTR_ANY and key == "result_specs":
            encoded[key] = [_encode_spec(spec) for spec in value]
        else:
            encoded[key] = _encode_generic(value, constants, inline)
    return encoded


def _decode_attrs(op_name: str, encoded_attrs, constants: dict) -> dict:
    """Decode an op's attribute dict per its ``OpDef`` schema."""
    if not isinstance(encoded_attrs, dict):
        raise VerificationError(f"op '{op_name}': attributes must be a dict")
    try:
        schema = _attr_schema(op_name)
    except KeyError as exc:
        raise VerificationError(f"unknown op '{op_name}'") from exc
    return _decode_attrs_with_schema(op_name, schema, encoded_attrs, constants)


def _decode_attrs_with_schema(
    op_name: str, schema: dict, encoded_attrs: dict, constants: dict
) -> dict:
    """Decode attribute values against an already-resolved ``OpDef`` schema."""
    decoded = {}
    for key, value in encoded_attrs.items():
        spec = schema.get(key)
        if spec is None:
            raise VerificationError(f"op '{op_name}': unknown attribute '{key}'")
        if spec.type == ATTR_NDARRAY:
            decoded[key] = _decode_ndarray_ref(value, constants)
        elif spec.type == ATTR_SHAPE:
            if not isinstance(value, list):
                raise VerificationError(
                    f"op '{op_name}': attribute '{key}' must be a list of dims"
                )
            decoded[key] = tuple(_decode_dim(dim) for dim in value)
        elif spec.type == ATTR_ANY and key == "result_specs":
            if not isinstance(value, list):
                raise VerificationError(
                    f"op '{op_name}': attribute '{key}' must be a list of specs"
                )
            decoded[key] = tuple(_decode_spec(spec) for spec in value)
        else:
            decoded[key] = _decode_generic(value, constants)
    return decoded


# ---------------------------------------------------------------------------
# Blocks / regions
# ---------------------------------------------------------------------------


def _encode_block(block: Block) -> dict:
    """Encode one block: arguments plus references into the flat ops table."""
    return {
        "arguments": [
            {"id": arg.id, "type": _encode_type(arg.type)} for arg in block.arguments
        ],
        "ops": [{"id": op.id, "ref": str(op.id)} for op in block.ops],
    }


def _encode_region(region: Region) -> dict:
    """Encode one region: the block encoding (v1 single-block shape), or a
    ``{"blocks": [...]}`` wrapper for multi-block regions."""
    if len(region.blocks) == 1:
        return _encode_block(region.blocks[0])
    return {"blocks": [_encode_block(block) for block in region.blocks]}


def _parse_ref(ref, where: str) -> int:
    """Parse a block op-list ``"ref"`` (decimal string of the op id)."""
    if not isinstance(ref, str):
        raise VerificationError(
            f"{where}: op ref must be a string of the op id, got {ref!r}"
        )
    try:
        return int(ref)
    except ValueError:
        raise VerificationError(
            f"{where}: op ref {ref!r} is not an integer op id"
        ) from None


def _build_block(data, ops_by_id: dict, values_by_id: dict, where: str) -> Block:
    """Rebuild one block from its encoding (arguments + op refs)."""
    if not isinstance(data, dict):
        raise VerificationError(f"malformed IR payload: {where} must be a dict")
    block = Block()
    arguments = []
    for i, entry in enumerate(_require_list(data, "arguments", where)):
        if not isinstance(entry, dict):
            raise VerificationError(f"{where}: argument {i} entry must be a dict")
        vid = _require_int(entry, "id", f"{where} argument {i}")
        value_type = _decode_type(_require(entry, "type", f"{where} argument {i}"))
        arguments.append(Value(id=vid, type=value_type, owner=block, index=i))
    block.arguments = tuple(arguments)
    for argument in arguments:
        values_by_id[argument.id] = argument
    for entry in _require_list(data, "ops", where):
        if not isinstance(entry, dict):
            raise VerificationError(f"{where}: op entry must be a dict")
        ref = _require(entry, "ref", where)
        op_id = _parse_ref(ref, where)
        if "id" in entry and _is_int(entry["id"]) and entry["id"] != op_id:
            raise VerificationError(
                f"{where}: op entry id {entry['id']} does not match its ref '{ref}'"
            )
        op = ops_by_id.get(op_id)
        if op is None:
            raise VerificationError(
                f"{where}: op ref '{ref}' does not reference an op in the ops table"
            )
        block.append(op)
    return block


def _build_region(
    data, ops_by_id: dict, values_by_id: dict, parent, where: str
) -> Region:
    """Rebuild one region (block encoding, or a ``{"blocks": [...]}`` list)."""
    region = Region()
    region.parent = parent
    if isinstance(data, dict) and "blocks" in data:
        blocks_data = data["blocks"]
        if not isinstance(blocks_data, list) or not blocks_data:
            raise VerificationError(
                f"{where}: region 'blocks' must be a non-empty list"
            )
        for block_data in blocks_data:
            region.append_block(
                _build_block(block_data, ops_by_id, values_by_id, where)
            )
        return region
    region.append_block(_build_block(data, ops_by_id, values_by_id, where))
    return region


# ---------------------------------------------------------------------------
# Serialization (module -> payload)
# ---------------------------------------------------------------------------


def _encode_op(op: Op, constants: dict, inline: bool = False) -> dict:
    """Encode one op into the flat ops table entry."""
    return {
        "id": op.id,
        "name": op.name,
        "operands": [operand.id for operand in op.operands],
        "results": [
            {"id": result.id, "type": _encode_type(result.type)}
            for result in op.results
        ],
        "attributes": _encode_attrs(op, constants, inline),
        "regions": [_encode_region(region) for region in op.regions],
        "location": _encode_location(op.location),
    }


def _collect_ops(
    block: Block, ops_data: list, constants: dict, inline: bool = False
) -> None:
    """Depth-first walk appending every op to the flat table."""
    for op in block.ops:
        ops_data.append(_encode_op(op, constants, inline))
        for region in op.regions:
            for nested in region.blocks:
                _collect_ops(nested, ops_data, constants, inline)


def _encode_function(function: Function, constants: dict, inline: bool = False) -> dict:
    """Encode one function entry (block refs resolve against the ops table)."""
    return {
        "name": function.name,
        "input_types": [_encode_type(t) for t in function.input_types],
        "output_types": [_encode_type(t) for t in function.output_types],
        "metadata": _encode_generic(function.metadata, constants, inline),
        "block": _encode_block(function.entry_block),
    }


def serialize_module(module: Module) -> dict:
    """Serialize ``module`` into the self-describing payload dict above.

    Requires a verified module (call ``verify`` first); serialization performs
    no semantic changes.

    Raises:
        VerificationError: If the module fails ``verify``.
    """
    return _serialize_module(module)


def _serialize_module(
    module: Module,
    *,
    inline_arrays: bool = False,
    include_sha256: bool = True,
) -> dict:
    """The shared serializer behind :func:`serialize_module` and the native seam.

    ``inline_arrays`` emits self-contained ndarray payloads instead of a
    ``constants`` table (the ``evoxc-native-ir`` envelope body), and
    ``include_sha256=False`` skips the integrity hash for an in-process
    consumer that never stores the payload.  The module is always verified.
    """
    if not isinstance(module, Module):
        raise TypeError(
            f"serialize_module() expects an ir.Module, got {type(module).__name__}"
        )
    verify(module)
    constants: dict = {}
    functions_data = [
        _encode_function(function, constants, inline_arrays)
        for function in module.functions
    ]
    ops_data: list = []
    for function in module.functions:
        for block in function.region.blocks:
            _collect_ops(block, ops_data, constants, inline_arrays)
    payload = {
        "format": IR_FORMAT,
        "version": IR_FORMAT_VERSION,
        "module": {
            "name": module.name,
            "metadata": _encode_generic(module.metadata, constants, inline_arrays),
        },
        "functions": functions_data,
        "ops": ops_data,
        "constants": constants,
    }
    if include_sha256:
        payload["sha256"] = _sha256(payload)
    return payload


# ---------------------------------------------------------------------------
# Deserialization (payload -> module)
# ---------------------------------------------------------------------------


def _build_op(data, values_by_id: dict, constants: dict) -> Op:
    """Rebuild one op (phase 1: identity, attrs, results; wiring later)."""
    if not isinstance(data, dict):
        raise VerificationError("malformed IR payload: ops table entries must be dicts")
    op_id = _require_int(data, "id", "op entry")
    name = _require_str(data, "name", "op entry")
    op = Op(
        name=name,
        id=op_id,
        attributes=_decode_attrs(name, data.get("attributes", {}), constants),
        location=_decode_location(data.get("location")),
    )
    results = []
    for i, entry in enumerate(_require_list(data, "results", f"op '{name}'")):
        if not isinstance(entry, dict):
            raise VerificationError(f"op '{name}': result {i} entry must be a dict")
        vid = _require_int(entry, "id", f"op '{name}' result {i}")
        value_type = _decode_type(_require(entry, "type", f"op '{name}' result {i}"))
        results.append(Value(id=vid, type=value_type, owner=op, index=i))
    op.results = tuple(results)
    for result in op.results:
        values_by_id[result.id] = result
    return op


def _build_function(
    data, ops_by_id: dict, values_by_id: dict, constants: dict
) -> Function:
    """Rebuild one function (region + entry block + signature)."""
    where = "function entry"
    name = _require_str(data, "name", where)
    input_types = tuple(
        _decode_type(t) for t in _require_list(data, "input_types", where)
    )
    block = _build_block(
        _require(data, "block", where), ops_by_id, values_by_id, f"function '{name}'"
    )
    region = Region()
    region.append_block(block)  # wires block.parent = region
    function = Function(
        name=name,
        input_types=input_types,
        region=region,
        metadata=_decode_generic(data.get("metadata", {}), constants),
    )
    return function


def _fast_forward_counters(
    module: Module, ops_by_id: dict, values_by_id: dict
) -> None:
    """Advance the module's id counters past every payload id.

    ``Module._op_ids``/``_value_ids`` are ``itertools.count`` starting at 0;
    replacing them with counters starting at ``max_id + 1`` guarantees a later
    ``Builder.create`` on the deserialized module never collides ids.
    """
    max_op_id = max(ops_by_id, default=-1)
    max_value_id = max(values_by_id, default=-1)
    module._op_ids = count(max_op_id + 1)
    module._value_ids = count(max_value_id + 1)


def _build_module(body: dict) -> Module:
    """Rebuild the full module structure from a (hash-verified) payload body.

    Phase order: (1) all ops of the flat table + their result values (ids,
    types, attributes, locations); (2) function blocks (arguments + op refs,
    wiring ``op.parent``); (3) nested op regions; (4) operands + ``Use``
    bookkeeping; (5) id-counter fast-forward. Original ids are preserved
    everywhere.
    """
    if not isinstance(body, dict):
        raise VerificationError("malformed IR payload: body must be a dict")
    module_info = _require_dict(body, "module", "payload")
    functions_data = _require_list(body, "functions", "payload")
    ops_data = _require_list(body, "ops", "payload")
    constants = body.get("constants", {})
    if not isinstance(constants, dict):
        raise VerificationError("malformed IR payload: 'constants' must be a dict")
    module = Module(
        name=_require_str(module_info, "name", "module"),
        metadata=_decode_generic(module_info.get("metadata", {}), constants),
        version=_require_int(body, "version", "payload"),
    )
    ops_by_id: dict = {}
    values_by_id: dict = {}
    for entry in ops_data:
        op = _build_op(entry, values_by_id, constants)
        ops_by_id[op.id] = op
    for entry in functions_data:
        module.add_function(
            _build_function(entry, ops_by_id, values_by_id, constants)
        )
    for entry in ops_data:
        op_id = _require_int(entry, "id", "op entry")
        op = ops_by_id[op_id]
        regions_data = entry.get("regions", [])
        if not isinstance(regions_data, list):
            raise VerificationError(
                f"op '{op.name}': 'regions' must be a list of region encodings"
            )
        op.regions = tuple(
            _build_region(
                region_data,
                ops_by_id,
                values_by_id,
                op,
                f"op '{op.name}' region {i}",
            )
            for i, region_data in enumerate(regions_data)
        )
    for entry in ops_data:
        op_id = _require_int(entry, "id", "op entry")
        op = ops_by_id[op_id]
        operands = []
        for i, vid in enumerate(_require_list(entry, "operands", f"op '{op.name}'")):
            if not _is_int(vid):
                raise VerificationError(
                    f"op '{op.name}': operand {i} must be a value id (int), "
                    f"got {vid!r}"
                )
            operand = values_by_id.get(vid)
            if operand is None:
                raise VerificationError(
                    f"op '{op.name}': operand {i} references undefined value {vid}"
                )
            operands.append(operand)
            operand.add_use(Use(op, i))
        op.operands = tuple(operands)
    _fast_forward_counters(module, ops_by_id, values_by_id)
    return module


def deserialize_module(payload: dict) -> Module:
    """Rebuild a ``Module`` from a payload produced by ``serialize_module``.

    Validates ``format``/``version``, recomputes and compares the sha256,
    rebuilds the IR objects (with their ORIGINAL ids, parent pointers, and
    operand uses), and runs ``verify`` on the result.

    Raises:
        PersistenceError: Unknown format discriminator or version.
        VerificationError: Integrity hash mismatch or structural violation.
    """
    if not isinstance(payload, dict):
        raise PersistenceError(
            f"IR payload must be a dict, got {type(payload).__name__}"
        )
    payload_format = payload.get("format")
    if payload_format != IR_FORMAT:
        raise PersistenceError(
            f"unknown IR payload format {payload_format!r} "
            f"(expected {IR_FORMAT!r})"
        )
    payload_version = payload.get("version")
    if payload_version != IR_FORMAT_VERSION:
        raise PersistenceError(
            f"unsupported IR payload version {payload_version!r} "
            f"(this runtime knows {IR_FORMAT_VERSION})"
        )
    recorded = payload.get("sha256")
    if not isinstance(recorded, str):
        raise VerificationError("IR payload is missing its 'sha256' integrity hash")
    body = {key: value for key, value in payload.items() if key != "sha256"}
    if _sha256(body) != recorded:
        raise VerificationError("IR payload integrity check failed: sha256 mismatch")
    module = _build_module(body)
    verify(module)
    return module


def _serialize_native_payload(module: Module) -> dict:
    """Serialize ``module`` for the native envelope: inline arrays, no hash.

    The module is verified exactly as :func:`serialize_module` does — the
    verification is part of the contract, not an optimisation the caller may
    assume away.  The result carries ``format``/``version``/``module``/
    ``functions``/``ops`` and an EMPTY ``constants`` table (every ndarray
    payload is self-contained), and no ``sha256`` (an in-process envelope is
    never stored, so there is nothing to checksum).
    """
    return _serialize_module(module, inline_arrays=True, include_sha256=False)


def _deserialize_envelope_body(body: dict) -> Module:
    """Rebuild a module from an ``evoxc-native-ir`` body (inline arrays).

    The caller (the private native codec) has already validated the envelope
    discriminator and version, so only the IR version is restored here: the
    body is rebuilt with :func:`_build_module` and re-verified exactly like
    :func:`deserialize_module`.  No sha256 is recomputed (an in-process
    envelope carries none).

    Raises:
        VerificationError: For any body violation or a failed ``verify``.
    """
    if not isinstance(body, dict):
        raise VerificationError("malformed IR payload: body must be a dict")
    body = dict(body)
    body["version"] = IR_FORMAT_VERSION
    module = _build_module(body)
    verify(module)
    return module
