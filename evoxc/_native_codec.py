"""Private v1 codec between a Python ``ir.Module`` and the neutral native envelope.

This module is the in-process bridge the optional Rust/PyO3 extension
(``evoxc._evoxc_native``, loaded by :mod:`evoxc._native`) speaks: an
``ir.Module`` is converted to a plain nested ``dict`` of Python builtins,
bytes and NumPy arrays' raw payloads, and a returned envelope is converted
back to a fully wired, ``ir.verify``-clean ``ir.Module``.

It is deliberately **not** the persistence format (``evoxc.persist`` owns the
on-disk format) and it imports only ``evoxc.core``, ``evoxc.ir``, NumPy and
the standard library.

Envelope format (``ENVELOPE_FORMAT = 1``)::

    # ValueType
    {"dtype": "float32", "shape": [dim, ...]}

    # dim: int | None | ["dim", name, size_or_None] | ["expr", op, left, right]
    #      op in add|sub|mul|floordiv|mod|min|max; left/right are dims recursively

    # module
    {
      "format": 1,
      "name": "main",
      "functions": [
        {"name": "main", "input_types": [type, ...], "blocks": [block, ...]},
      ],
      "constants": [ {"dtype": "float32", "shape": [2, 3], "data": b"..."} ],
      "value_types": {"<value id>": type, ...},
    }

    # block
    {
      "arguments": [value_id, ...],
      "ops": [
        {"id": int, "name": "add", "operands": [value_id, ...], "results": [value_id, ...],
         "attributes": {name: attr_value, ...},
         "regions": [ [block, ...], ... ],
         "location": {"file": str, "line": int, "column": int} | None },
      ],
    }

Attribute values (JSON-safe forms; tagged only when JSON cannot represent them
unambiguously)::

    bool | int | float | str | None   -> the Python value
    tuple                             -> {"__evoxc__": "tuple", "items": [...]}
    list                              -> {"__evoxc__": "list", "items": [...]}
    ir.ValueType                      -> {"__evoxc__": "type", "dtype": ..., "shape": [...]}
    dtype name (a str)                -> the plain string (never tagged)
    numpy.dtype                       -> {"__evoxc__": "dtype", "name": "float32"}
    Dim                               -> {"__evoxc__": "dim", "name": ..., "size": ...}
    DimExpr                           -> {"__evoxc__": "expr", "op": ..., "left": ..., "right": ...}
    numpy.ndarray                     -> {"__evoxc__": "const_ref", "index": i}
    dict                              -> {"__evoxc__": "dict", "items": [[key, value], ...]}

Design decisions (binding for this private v1 format):

1. Shape dims use the positional forms (``["dim", name, size]`` /
   ``["expr", op, left, right]``); attribute-level ``Dim``/``DimExpr`` use the
   tagged dict forms. Both decode back to equal frozen objects.
2. ``ATTR_DTYPE`` attribute values are plain name strings because
   ``ir.Builder`` already normalized them to ``np.dtype(...).name``; the tagged
   ``dtype`` form is used only for real ``np.dtype`` objects encountered in an
   attribute payload (e.g. ``ATTR_ANY``) and decodes to ``np.dtype(name)``.
   A plain ``str`` is never tagged.
3. ``Location.code_snippet`` and ``Module.metadata``/``Function.metadata`` are
   NOT part of envelope format 1 (the contract fixes the fields): decode sets
   ``code_snippet=None`` and ``metadata={}``. This is by design, not a bug.
   Likewise the envelope carries no module-format version: decode stamps the
   current ``ir.IR_FORMAT_VERSION``.
4. Every ndarray attribute becomes an entry in ``constants`` (identical
   ``(dtype name, shape, bytes)`` payloads are deduplicated) referenced by
   ``{"__evoxc__": "const_ref", "index": i}``. Raw bytes are C-contiguous
   LITTLE-ENDIAN ``dtype.itemsize`` bytes; ``dtype`` is the NumPy name (byte
   order dropped) and ``shape`` a list of ints. Decode builds an owning,
   writable, NATIVE-byte-order, C-contiguous array, so a non-native-endian
   input array decodes value-equal with the same dtype name and shape.
5. ``bool`` is checked before ``int`` (``bool`` is an ``int`` subclass); ints
   and floats round-trip as their Python types; tuple-vs-list survives via the
   tags; dicts are tagged with ordered ``[key, value]`` pairs whose keys are
   encoded through the same codec (non-str keys are allowed).
6. Any unsupported attribute value type (``np.generic`` scalar, ``set``, a
   custom object, ...) raises ``TypeError`` naming the op, the attribute and
   the value type — never a silent coercion.
7. ``encode_module`` never mutates the module: it only reads (no counter
   advance, no ``uses`` edits).
8. ``decode_module`` fully rewires the IR: result ``Value.owner``/``index``,
   block-argument ``Value.owner``/``index``, ``Op.parent``, ``Region.parent``
   (function body vs owning op), ``Function.parent``, every ``Value.uses``
   entry (including repeated operands), and the module's id counters (reset to
   ``max_id + 1``) so later ``Builder`` allocations cannot collide.
9. ``value_types`` (keyed by ``str(value_id)``) is the authoritative type table
   on decode: every block argument of every block (including nested regions)
   and every op result must be present there.

An ``ir.ValueType`` attribute (the ``result_specs`` payload of
``runtime_call``/``block_call``/``external_call``) cannot be represented by
the base attribute grammar, so it uses the tagged ``"type"`` form: without it
a verified module containing those ops could not round-trip.
"""

from __future__ import annotations

import itertools
from numbers import Integral
from typing import Any

import numpy as np

from evoxc import ir
from evoxc.core import Dim, DimExpr

__all__ = ["ENVELOPE_FORMAT", "encode_module", "decode_module", "encode_type", "decode_type"]

#: Envelope format version this codec reads and writes.
ENVELOPE_FORMAT = 1

_TAG = "__evoxc__"

_DIMEXPR_OPS = frozenset({"add", "sub", "mul", "floordiv", "mod", "min", "max"})


# ---------------------------------------------------------------------------
# Type (ValueType) encoding
# ---------------------------------------------------------------------------


def encode_type(value_type: ir.ValueType) -> dict:
    """Encode an :class:`ir.ValueType` as ``{"dtype": name, "shape": [...]}``."""
    if not isinstance(value_type, ir.ValueType):
        raise TypeError(
            f"encode_type expects an ir.ValueType, got {type(value_type).__name__}"
        )
    return {
        "dtype": value_type.dtype.name,
        "shape": [_encode_dim(dim) for dim in value_type.shape],
    }


def decode_type(payload: Any) -> ir.ValueType:
    """Decode a ``{"dtype": ..., "shape": [...]}`` dict into an ``ir.ValueType``."""
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
    """Encode one shape dimension (``int | None | Dim | DimExpr``)."""
    if dim is None:
        return None
    if isinstance(dim, bool):
        raise TypeError(f"a shape dimension cannot be a bool: {dim!r}")
    if isinstance(dim, Integral):
        return int(dim)
    if isinstance(dim, Dim):
        return ["dim", dim.name, None if dim.size is None else int(dim.size)]
    if isinstance(dim, DimExpr):
        return ["expr", dim.op, _encode_dim(dim.left), _encode_dim(dim.right)]
    raise TypeError(
        f"cannot encode shape dimension {dim!r}: expected int, None, Dim or DimExpr, "
        f"got {type(dim).__name__}"
    )


def _decode_dim(raw: Any) -> Any:
    """Decode one shape dimension from its positional envelope form."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError(f"malformed dimension: bool is not a valid dim ({raw!r})")
    if isinstance(raw, Integral):
        return int(raw)
    if isinstance(raw, (list, tuple)):
        if len(raw) == 3 and raw[0] == "dim":
            name, size = raw[1], raw[2]
            if not isinstance(name, str):
                raise ValueError(f"malformed dimension: name must be str, got {name!r}")
            if size is not None and (
                isinstance(size, bool) or not isinstance(size, Integral)
            ):
                raise ValueError(
                    f"malformed dimension: size must be int or None, got {size!r}"
                )
            return Dim(name, None if size is None else int(size))
        if len(raw) == 4 and raw[0] == "expr":
            op = raw[1]
            if not isinstance(op, str):
                raise ValueError(f"malformed dimension: expr op must be str, got {op!r}")
            if op not in _DIMEXPR_OPS:
                raise ValueError(
                    f"malformed dimension: unknown DimExpr op {op!r} "
                    f"(expected one of {sorted(_DIMEXPR_OPS)})"
                )
            return DimExpr(op, _decode_dim(raw[2]), _decode_dim(raw[3]))
    raise ValueError(f"malformed dimension encoding: {raw!r}")


# ---------------------------------------------------------------------------
# Constants (ndarray attribute payloads)
# ---------------------------------------------------------------------------


class _ConstantTable:
    """Accumulates deduplicated ndarray payloads while encoding a module."""

    def __init__(self) -> None:
        self.entries: list[dict] = []
        self._index: dict[tuple[str, tuple[int, ...], bytes], int] = {}

    def add(self, array: np.ndarray) -> int:
        """Record ``array`` and return its ``constants`` index (dedup by payload)."""
        name = array.dtype.name
        little_endian = np.dtype(name).newbyteorder("<")
        contiguous = np.ascontiguousarray(array, dtype=little_endian)
        data = contiguous.tobytes(order="C")
        shape = tuple(int(size) for size in array.shape)
        key = (name, shape, data)
        index = self._index.get(key)
        if index is None:
            index = len(self.entries)
            self.entries.append(
                {"dtype": name, "shape": list(shape), "data": data}
            )
            self._index[key] = index
        return index


def _decode_constant(entry: Any) -> np.ndarray:
    """Decode one ``constants`` entry into an owning native-order ndarray."""
    if not isinstance(entry, dict):
        raise ValueError(
            f"malformed constant payload: expected a dict, got {type(entry).__name__}"
        )
    name = _require_key(entry, "dtype", "constant")
    shape = _require_key(entry, "shape", "constant")
    data = _require_key(entry, "data", "constant")
    if not isinstance(name, str):
        raise ValueError(f"malformed constant payload: 'dtype' must be str, got {name!r}")
    if not isinstance(shape, (list, tuple)) or not all(
        isinstance(size, Integral) and not isinstance(size, bool) for size in shape
    ):
        raise ValueError(
            f"malformed constant payload: 'shape' must be a list of ints, got {shape!r}"
        )
    if isinstance(data, bytearray):
        data = bytes(data)
    if not isinstance(data, bytes):
        raise ValueError(
            f"malformed constant payload: 'data' must be bytes, got {type(data).__name__}"
        )
    try:
        dtype = np.dtype(name)
    except TypeError as exc:
        raise ValueError(f"invalid dtype name {name!r}: {exc}") from exc
    little_endian = dtype.newbyteorder("<")
    try:
        flat = np.frombuffer(data, dtype=little_endian)
        view = flat.reshape(tuple(int(size) for size in shape))
    except ValueError as exc:
        raise ValueError(
            f"malformed constant payload for dtype {name!r} and shape "
            f"{list(shape)!r}: {exc}"
        ) from exc
    # ``np.array(..., copy=True)`` normalizes byte order to native, makes the
    # array owning and writable, and keeps C order (including 0-d shape ()).
    return np.array(view, dtype=dtype, order="C", copy=True)


# ---------------------------------------------------------------------------
# Attribute encoding
# ---------------------------------------------------------------------------


def _encode_attr(value: Any, op_name: str, attr_name: str, constants: _ConstantTable) -> Any:
    """Encode one attribute value; anything unrepresentable is a loud ``TypeError``."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        raise _unsupported_attr(op_name, attr_name, value)
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return value
    if isinstance(value, tuple):
        return {
            _TAG: "tuple",
            "items": [
                _encode_attr(item, op_name, attr_name, constants) for item in value
            ],
        }
    if isinstance(value, list):
        return {
            _TAG: "list",
            "items": [
                _encode_attr(item, op_name, attr_name, constants) for item in value
            ],
        }
    if isinstance(value, ir.ValueType):
        encoded = {_TAG: "type"}
        encoded.update(encode_type(value))
        return encoded
    if isinstance(value, np.ndarray):
        return {_TAG: "const_ref", "index": constants.add(value)}
    if isinstance(value, np.dtype):
        return {_TAG: "dtype", "name": value.name}
    if isinstance(value, Dim):
        return {_TAG: "dim", "name": value.name, "size": value.size}
    if isinstance(value, DimExpr):
        return {
            _TAG: "expr",
            "op": value.op,
            "left": _encode_attr(value.left, op_name, attr_name, constants),
            "right": _encode_attr(value.right, op_name, attr_name, constants),
        }
    if isinstance(value, dict):
        return {
            _TAG: "dict",
            "items": [
                [
                    _encode_attr(key, op_name, attr_name, constants),
                    _encode_attr(item, op_name, attr_name, constants),
                ]
                for key, item in value.items()
            ],
        }
    raise _unsupported_attr(op_name, attr_name, value)


def _unsupported_attr(op_name: str, attr_name: str, value: Any) -> TypeError:
    return TypeError(
        f"cannot encode attribute '{attr_name}' of op '{op_name}': unsupported "
        f"value type {type(value).__name__} ({value!r})"
    )


def _decode_attr(value: Any, constants: list[np.ndarray], where: str) -> Any:
    """Decode one attribute value from its envelope form."""
    if value is None or isinstance(value, (bool, Integral, float, str)):
        return value
    if isinstance(value, dict):
        tag = value.get(_TAG)
        if tag is None:
            raise ValueError(f"{where}: untagged dict attribute value {value!r}")
        if tag == "tuple":
            return tuple(
                _decode_attr(item, constants, where)
                for item in _require_items(value, where)
            )
        if tag == "list":
            return [
                _decode_attr(item, constants, where)
                for item in _require_items(value, where)
            ]
        if tag == "type":
            return decode_type(value)
        if tag == "const_ref":
            index = _require_key(value, "index", f"{where} const_ref")
            if isinstance(index, bool) or not isinstance(index, Integral):
                raise ValueError(f"{where}: const_ref index must be int, got {index!r}")
            if not 0 <= int(index) < len(constants):
                raise ValueError(
                    f"{where}: const_ref index {index} is out of range "
                    f"(the envelope has {len(constants)} constant(s))"
                )
            return constants[int(index)]
        if tag == "dtype":
            name = _require_key(value, "name", f"{where} dtype")
            if not isinstance(name, str):
                raise ValueError(f"{where}: dtype name must be str, got {name!r}")
            try:
                return np.dtype(name)
            except TypeError as exc:
                raise ValueError(f"{where}: invalid dtype name {name!r}: {exc}") from exc
        if tag == "dim":
            name = _require_key(value, "name", f"{where} dim")
            size = value.get("size")
            if not isinstance(name, str):
                raise ValueError(f"{where}: dim name must be str, got {name!r}")
            if size is not None and (
                isinstance(size, bool) or not isinstance(size, Integral)
            ):
                raise ValueError(f"{where}: dim size must be int or None, got {size!r}")
            return Dim(name, None if size is None else int(size))
        if tag == "expr":
            op = _require_key(value, "op", f"{where} expr")
            left = _require_key(value, "left", f"{where} expr")
            right = _require_key(value, "right", f"{where} expr")
            if not isinstance(op, str):
                raise ValueError(f"{where}: expr op must be str, got {op!r}")
            return DimExpr(
                op,
                _decode_attr(left, constants, where),
                _decode_attr(right, constants, where),
            )
        if tag == "dict":
            pairs = _require_items(value, where)
            decoded: dict = {}
            for pair in pairs:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError(
                        f"{where}: dict attribute entry must be a [key, value] pair, "
                        f"got {pair!r}"
                    )
                decoded[_decode_attr(pair[0], constants, where)] = _decode_attr(
                    pair[1], constants, where
                )
            return decoded
        raise ValueError(f"{where}: unknown attribute tag {tag!r}")
    raise ValueError(
        f"{where}: untagged attribute value of type {type(value).__name__} ({value!r})"
    )


def _require_items(value: dict, where: str) -> list:
    items = _require_key(value, "items", where)
    if not isinstance(items, (list, tuple)):
        raise ValueError(f"{where}: 'items' must be a list, got {items!r}")
    return list(items)


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------


def _require_key(mapping: dict, key: str, where: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{where}: missing required key {key!r}")
    return mapping[key]


def _as_id(value: Any, where: str) -> int:
    """Validate a module-unique integer id (bools rejected)."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{where}: expected an int id, got {value!r}")
    return int(value)


# ---------------------------------------------------------------------------
# Module encoding
# ---------------------------------------------------------------------------


def encode_module(module: ir.Module) -> dict:
    """Encode ``module`` as the neutral in-process envelope dict.

    Read-only: the module is never mutated (no counter advance, no ``uses``
    edits).
    """
    if not isinstance(module, ir.Module):
        raise TypeError(f"encode_module expects an ir.Module, got {type(module).__name__}")
    constants = _ConstantTable()
    value_types: dict[str, dict] = {}
    functions = []
    for function in module.functions:
        blocks = [
            _encode_block(block, constants, value_types)
            for block in function.region.blocks
        ]
        functions.append(
            {
                "name": function.name,
                "input_types": [encode_type(t) for t in function.input_types],
                "blocks": blocks,
            }
        )
    return {
        "format": ENVELOPE_FORMAT,
        "name": module.name,
        "functions": functions,
        "constants": constants.entries,
        "value_types": value_types,
    }


def _encode_block(block: ir.Block, constants: _ConstantTable, value_types: dict) -> dict:
    arguments = []
    for argument in block.arguments:
        value_types[str(argument.id)] = encode_type(argument.type)
        arguments.append(argument.id)
    ops = []
    for op in block.ops:
        attributes = {
            name: _encode_attr(value, op.name, name, constants)
            for name, value in op.attributes.items()
        }
        regions = [
            [_encode_block(inner, constants, value_types) for inner in region.blocks]
            for region in op.regions
        ]
        ops.append(
            {
                "id": op.id,
                "name": op.name,
                "operands": [operand.id for operand in op.operands],
                "results": [result.id for result in op.results],
                "attributes": attributes,
                "regions": regions,
                "location": _encode_location(op.location),
            }
        )
        for result in op.results:
            value_types[str(result.id)] = encode_type(result.type)
    return {"arguments": arguments, "ops": ops}


def _encode_location(location: ir.Location | None) -> dict | None:
    if location is None:
        return None
    return {"file": location.file, "line": location.line, "column": location.col}


# ---------------------------------------------------------------------------
# Module decoding
# ---------------------------------------------------------------------------


class _Decoder:
    """Rebuilds and wires an ``ir.Module`` from an envelope."""

    def __init__(self, value_types: dict[int, ir.ValueType], constants: list[np.ndarray]):
        self.value_types = value_types
        self.constants = constants
        self.values: dict[int, ir.Value] = {}
        self.defined: set[int] = set()
        self.op_ids: set[int] = set()

    # -- values ---------------------------------------------------------

    def get_value(self, value_id: Any, where: str) -> ir.Value:
        """Return the ``Value`` for ``value_id``, creating an unwired one if needed."""
        vid = _as_id(value_id, where)
        existing = self.values.get(vid)
        if existing is not None:
            return existing
        value_type = self.value_types.get(vid)
        if value_type is None:
            raise ValueError(
                f"{where}: value id {vid} has no entry in the envelope's "
                "'value_types' table"
            )
        value = ir.Value(id=vid, type=value_type, owner=None, index=0)
        self.values[vid] = value
        return value

    def define_value(self, value_id: Any, owner: Any, index: int, where: str) -> ir.Value:
        """Bind a value to its defining op/block, rejecting duplicate ids."""
        vid = _as_id(value_id, where)
        if vid in self.defined:
            raise ValueError(f"{where}: duplicate value id {vid}")
        value = self.get_value(vid, where)
        value.owner = owner
        value.index = index
        self.defined.add(vid)
        return value

    # -- structure ------------------------------------------------------

    def decode_block(self, data: dict, region: ir.Region, where: str) -> ir.Block:
        if not isinstance(data, dict):
            raise ValueError(f"{where}: block must be a dict, got {type(data).__name__}")
        argument_ids = _require_key(data, "arguments", where)
        ops_data = _require_key(data, "ops", where)
        if not isinstance(argument_ids, (list, tuple)):
            raise ValueError(f"{where}: 'arguments' must be a list, got {argument_ids!r}")
        if not isinstance(ops_data, (list, tuple)):
            raise ValueError(f"{where}: 'ops' must be a list, got {ops_data!r}")
        block = ir.Block(arguments=(), ops=[])
        block.arguments = tuple(
            self.define_value(vid, block, index, f"{where} argument {index}")
            for index, vid in enumerate(argument_ids)
        )
        for index, op_data in enumerate(ops_data):
            block.ops.append(self.decode_op(op_data, block, f"{where} op {index}"))
        region.append_block(block)  # wires block.parent
        return block

    def decode_op(self, data: dict, parent: ir.Block, where: str) -> ir.Op:
        if not isinstance(data, dict):
            raise ValueError(f"{where}: op must be a dict, got {type(data).__name__}")
        name = _require_key(data, "name", where)
        if not isinstance(name, str):
            raise ValueError(f"{where}: op 'name' must be a string, got {name!r}")
        where = f"{where} ('{name}')"
        op_id = _as_id(_require_key(data, "id", where), f"{where} id")
        if op_id in self.op_ids:
            raise ValueError(f"{where}: duplicate op id {op_id}")
        self.op_ids.add(op_id)
        operand_ids = _require_list(data, "operands", where)
        result_ids = _require_list(data, "results", where)
        attributes_data = data.get("attributes", {})
        if not isinstance(attributes_data, dict):
            raise ValueError(f"{where}: 'attributes' must be a dict, got {attributes_data!r}")
        attributes = {}
        for attr_name, attr_value in attributes_data.items():
            if not isinstance(attr_name, str):
                raise ValueError(f"{where}: attribute name must be a string, got {attr_name!r}")
            attributes[attr_name] = _decode_attr(
                attr_value, self.constants, f"{where} attribute '{attr_name}'"
            )
        regions_data = data.get("regions", [])
        if not isinstance(regions_data, (list, tuple)):
            raise ValueError(f"{where}: 'regions' must be a list, got {regions_data!r}")
        op = ir.Op(
            name=name,
            id=op_id,
            operands=(),
            attributes=attributes,
            regions=(),
            results=(),
            location=_decode_location(data.get("location"), where),
        )
        regions = []
        for region_index, blocks_data in enumerate(regions_data):
            region_where = f"{where} region {region_index}"
            if not isinstance(blocks_data, (list, tuple)):
                raise ValueError(
                    f"{region_where}: region encoding must be a list of blocks, "
                    f"got {blocks_data!r}"
                )
            region = ir.Region(blocks=[], parent=op)
            for block_index, block_data in enumerate(blocks_data):
                self.decode_block(
                    block_data, region, f"{region_where} block {block_index}"
                )
            regions.append(region)
        op.regions = tuple(regions)
        op.results = tuple(
            self.define_value(vid, op, index, f"{where} result {index}")
            for index, vid in enumerate(result_ids)
        )
        op.operands = tuple(
            self.get_value(vid, f"{where} operand {index}")
            for index, vid in enumerate(operand_ids)
        )
        op.parent = parent
        return op

    def decode_function(self, data: Any) -> ir.Function:
        if not isinstance(data, dict):
            raise ValueError(f"function entry must be a dict, got {type(data).__name__}")
        name = _require_key(data, "name", "function")
        if not isinstance(name, str):
            raise ValueError(f"function: 'name' must be a string, got {name!r}")
        input_types_data = _require_key(data, "input_types", f"function '{name}'")
        blocks_data = _require_key(data, "blocks", f"function '{name}'")
        if not isinstance(input_types_data, (list, tuple)):
            raise ValueError(
                f"function '{name}': 'input_types' must be a list, got {input_types_data!r}"
            )
        if not isinstance(blocks_data, (list, tuple)):
            raise ValueError(
                f"function '{name}': 'blocks' must be a list, got {blocks_data!r}"
            )
        region = ir.Region(blocks=[])
        for block_index, block_data in enumerate(blocks_data):
            self.decode_block(
                block_data, region, f"function '{name}' block {block_index}"
            )
        return ir.Function(
            name=name,
            input_types=tuple(decode_type(t) for t in input_types_data),
            region=region,
            metadata={},
            parent=None,
        )

    # -- final wiring ---------------------------------------------------

    def rebuild_uses(self, module: ir.Module) -> None:
        """Rebuild every ``Value.uses`` from the decoded operand lists."""
        for value in self.values.values():
            value.uses.clear()
        for op in _iter_ops(module):
            for operand_index, operand in enumerate(op.operands):
                if operand.id not in self.values:
                    raise ValueError(
                        f"op '{op.name}' (id {op.id}) operand {operand_index} "
                        f"references undefined value {operand.id}"
                    )
                operand.uses.append(ir.Use(op, operand_index))

    def check_all_defined(self) -> None:
        """Every referenced value must have exactly one definition."""
        for vid, value in self.values.items():
            if vid not in self.defined:
                raise ValueError(
                    f"envelope value id {vid} is referenced but never defined as a "
                    "block argument or op result"
                )

    def reset_counters(self, module: ir.Module) -> None:
        """Fast-forward module id counters past the decoded ids."""
        max_op = max(self.op_ids, default=-1)
        max_value = max(self.defined, default=-1)
        module._op_ids = itertools.count(max_op + 1)
        module._value_ids = itertools.count(max_value + 1)


def _iter_ops(module: ir.Module):
    """Yield every op in the module, recursing into nested regions."""
    def walk(region: ir.Region):
        for block in region.blocks:
            for op in block.ops:
                yield op
                for nested in op.regions:
                    yield from walk(nested)

    for function in module.functions:
        yield from walk(function.region)


def _require_list(data: dict, key: str, where: str) -> list:
    value = _require_key(data, key, where)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{where}: '{key}' must be a list, got {value!r}")
    return list(value)


def _decode_location(raw: Any, where: str) -> ir.Location | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: 'location' must be a dict or None, got {raw!r}")
    file = _require_key(raw, "file", f"{where} location")
    line = _require_key(raw, "line", f"{where} location")
    column = _require_key(raw, "column", f"{where} location")
    if not isinstance(file, str):
        raise ValueError(f"{where}: location file must be a string, got {file!r}")
    for label, entry in (("line", line), ("column", column)):
        if isinstance(entry, bool) or not isinstance(entry, Integral):
            raise ValueError(
                f"{where}: location {label} must be an int, got {entry!r}"
            )
    # ``code_snippet`` is not part of envelope format 1 (by design).
    return ir.Location(file, int(line), int(column), code_snippet=None)


def decode_module(envelope: Any) -> ir.Module:
    """Rebuild a fully wired ``ir.Module`` from an envelope dict.

    The result preserves original op/value ids, attributes, types, locations
    and parent/use pointers; ``ir.verify`` accepts it whenever it accepted the
    module that produced the envelope. The module id counters are reset past
    every decoded id.

    Raises:
        ValueError: For a malformed envelope (unknown ``format``, missing or
            duplicate ids, missing keys, bad dtype names, ...).
    """
    if not isinstance(envelope, dict):
        raise ValueError(
            f"malformed native envelope: expected a dict, got {type(envelope).__name__}"
        )
    envelope_format = envelope.get("format")
    if envelope_format != ENVELOPE_FORMAT:
        raise ValueError(
            f"unsupported native envelope format {envelope_format!r} "
            f"(this codec accepts only {ENVELOPE_FORMAT})"
        )
    name = _require_key(envelope, "name", "envelope")
    functions_data = _require_key(envelope, "functions", "envelope")
    constants_data = _require_key(envelope, "constants", "envelope")
    types_data = _require_key(envelope, "value_types", "envelope")
    if not isinstance(name, str):
        raise ValueError(f"envelope: 'name' must be a string, got {name!r}")
    if not isinstance(functions_data, (list, tuple)):
        raise ValueError(f"envelope: 'functions' must be a list, got {functions_data!r}")
    if not isinstance(constants_data, (list, tuple)):
        raise ValueError(f"envelope: 'constants' must be a list, got {constants_data!r}")
    if not isinstance(types_data, dict):
        raise ValueError(f"envelope: 'value_types' must be a dict, got {types_data!r}")

    constants = [_decode_constant(entry) for entry in constants_data]
    value_types: dict[int, ir.ValueType] = {}
    for key, type_payload in types_data.items():
        if isinstance(key, bool):
            raise ValueError(f"envelope: value_types key {key!r} is not an int id")
        try:
            vid = int(key)
        except (TypeError, ValueError):
            raise ValueError(
                f"envelope: value_types key {key!r} is not an int id"
            ) from None
        if vid in value_types:
            raise ValueError(f"envelope: duplicate value_types entry for id {vid}")
        value_types[vid] = decode_type(type_payload)

    decoder = _Decoder(value_types, constants)
    module = ir.Module(name=name, version=ir.IR_FORMAT_VERSION)
    for index, function_data in enumerate(functions_data):
        if not isinstance(function_data, dict):
            raise ValueError(
                f"envelope function {index}: expected a dict, "
                f"got {type(function_data).__name__}"
            )
        module.add_function(decoder.decode_function(function_data))

    decoder.check_all_defined()
    decoder.rebuild_uses(module)
    decoder.reset_counters(module)
    return module
