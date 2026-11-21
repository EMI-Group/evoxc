"""Trusted fast decode of the ``evoxc-native-ir`` envelope (the native seam).

The native transform engine returns its result in-process as an
``evoxc-native-ir`` envelope produced from a module this runtime encoded
moments earlier, so its structure is known valid: no sha256, no generic
``_require_*`` validation and no post-decode ``verify`` are needed.  This
module rebuilds that envelope into a fully wired :class:`evoxc.ir.Module` on
that trusted path.

:func:`build_module_fast` is the hot-path sibling of
``evoxc.ir.serialize._build_module``: it constructs every IR object
positionally, appends ``Use`` records directly (a trusted envelope has no
duplicate uses), memoizes types/locations by envelope-object identity and
advances the module id counters past every decoded id -- preserving original
op/value ids, attributes, locations, parent pointers and ordered, complete
``uses``.

The validated path stays in ``evoxc.ir.serialize`` (``deserialize_module`` /
``_deserialize_envelope_body``); ``evoxc._native_codec.decode_module`` keeps
using it for any envelope of uncertain provenance.

Private to ``evoxc.ir``: nothing public is defined here.
"""

from __future__ import annotations

from evoxc.core import Dim, DimExpr

from ._npy import _dtype_from_name
from .block import Block
from .function import Function
from .module import Module
from .op import Op
from .region import Region
from .serialize import (
    _VALUE_TYPE_CACHE,
    _attr_schema,
    _decode_attrs_with_schema,
    _decode_generic,
    _decode_location,
    _fast_forward_counters,
)
from .types import ValueType
from .value import Use, Value
from .version import IR_FORMAT_VERSION

__all__ = ["build_module_fast"]

#: Memo tables for the trusted fast decode (frozen, structurally equal values).
_DIM_CACHE: dict = {}
_DIMEXPR_CACHE: dict = {}

# ---------------------------------------------------------------------------
# Trusted fast decode helpers (the native transform envelope)
# ---------------------------------------------------------------------------
#
# The native transform envelope is produced in-process by the Rust engine from
# a module this runtime encoded moments earlier, so its structure is known
# valid: no sha256, no generic ``_require_*`` validation and no post-decode
# ``verify`` on this path.  The helpers below are the memoized construction
# primitives `build_module_fast` uses.  ``Dim``, ``DimExpr``, ``Location``
# and ``ValueType`` are frozen dataclasses, so structurally equal instances
# are interchangeable and safe to share across decodes.


def _decode_dim_fast(encoded) -> int | Dim | DimExpr | None:
    """``_decode_dim`` for a trusted envelope, memoizing symbolic dims."""
    if encoded is None:
        return None
    if "int" in encoded:
        return encoded["int"]
    if "dim" in encoded:
        key = (encoded["dim"], encoded.get("size"))
        dim = _DIM_CACHE.get(key)
        if dim is None:
            dim = Dim(key[0], size=key[1])
            if len(_DIM_CACHE) < 4096:
                _DIM_CACHE[key] = dim
        return dim
    expr = encoded["expr"]
    key = (
        expr["op"],
        _decode_dim_fast(expr["args"][0]),
        _decode_dim_fast(expr["args"][1]),
    )
    dim_expr = _DIMEXPR_CACHE.get(key)
    if dim_expr is None:
        dim_expr = DimExpr(key[0], key[1], key[2])
        if len(_DIMEXPR_CACHE) < 4096:
            _DIMEXPR_CACHE[key] = dim_expr
    return dim_expr


def _decode_type_fast(encoded) -> ValueType:
    """``_decode_type`` for a trusted envelope (same ``_VALUE_TYPE_CACHE``)."""
    dims = tuple(_decode_dim_fast(dim) for dim in encoded["shape"])
    key = (encoded["dtype"], dims)
    cached = _VALUE_TYPE_CACHE.get(key)
    if cached is not None:
        return cached
    value_type = ValueType(_dtype_from_name(key[0]), dims)
    if len(_VALUE_TYPE_CACHE) < 4096:
        _VALUE_TYPE_CACHE[key] = value_type
    return value_type


#: Shared empty constants table (the native envelope inlines every array).
_NO_CONSTANTS: dict = {}


def _build_region_fast(
    data, ops_by_id: dict, values_by_id: dict, type_memo: dict, parent
) -> Region:
    """``_build_region`` for a trusted envelope (no generic validation)."""
    region = Region()
    region.parent = parent
    blocks_data = data["blocks"] if "blocks" in data else (data,)
    for block_data in blocks_data:
        block = Block()
        arguments = []
        argument_index = 0
        for argument_entry in block_data["arguments"]:
            value_type = argument_entry["type"]
            key = id(value_type)
            decoded_type = type_memo.get(key)
            if decoded_type is None:
                decoded_type = _decode_type_fast(value_type)
                type_memo[key] = decoded_type
            value = Value(argument_entry["id"], decoded_type, block, argument_index)
            values_by_id[value.id] = value
            arguments.append(value)
            argument_index += 1
        block.arguments = tuple(arguments)
        for entry in block_data["ops"]:
            op = ops_by_id[entry["id"]]
            op.parent = block
            block.ops.append(op)
        region.append_block(block)
    return region


def build_module_fast(body: dict) -> Module:
    """Rebuild a module from a TRUSTED ``evoxc-native-ir`` body (fast path).

    This is the hot-path sibling of ``evoxc.ir.serialize._build_module``: it accepts the
    in-process envelope the native extension returns (produced from a module
    this runtime encoded moments earlier), constructs every IR object
    positionally, appends ``Use`` records directly (a trusted envelope has no
    duplicate uses), memoizes types/locations by envelope-object identity and
    skips the generic ``_require_*`` validation, the sha256 and ``verify``.
    The caller (``evoxc._native_codec``) owns the discriminator/version check
    and the optional debug-time ``verify``.

    The result is structurally identical to :func:`_build_module`'s: original
    ids, attributes, types, locations, parent pointers, ordered complete
    ``uses`` and id counters advanced past every decoded id.
    """
    module_info = body["module"]
    functions_data = body["functions"]
    ops_data = body["ops"]
    constants = body.get("constants")
    if not isinstance(constants, dict):
        constants = _NO_CONSTANTS
    module = Module(
        name=module_info["name"],
        metadata=_decode_generic(module_info.get("metadata") or {}, constants),
        version=IR_FORMAT_VERSION,
    )
    ops_by_id: dict = {}
    values_by_id: dict = {}
    type_memo: dict = {}
    location_memo: dict = {}
    schema_memo: dict = {}
    _value = Value
    _op = Op
    # Phase 1: the flat ops table — identity, attributes, locations, results.
    for entry in ops_data:
        op_id = entry["id"]
        name = entry["name"]
        schema = schema_memo.get(name)
        if schema is None:
            schema = _attr_schema(name)
            schema_memo[name] = schema
        encoded_attributes = entry["attributes"]
        if encoded_attributes:
            attributes = _decode_attrs_with_schema(
                name, schema, encoded_attributes, constants
            )
        else:
            attributes = {}
        encoded_location = entry["location"]
        if encoded_location is None:
            location = None
        else:
            key = id(encoded_location)
            location = location_memo.get(key)
            if location is None:
                location = _decode_location(encoded_location)
                location_memo[key] = location
        op = _op(name, op_id)
        op.attributes = attributes
        op.location = location
        results = []
        result_index = 0
        for result_entry in entry["results"]:
            value_type = result_entry["type"]
            key = id(value_type)
            decoded_type = type_memo.get(key)
            if decoded_type is None:
                decoded_type = _decode_type_fast(value_type)
                type_memo[key] = decoded_type
            value = _value(result_entry["id"], decoded_type, op, result_index)
            values_by_id[value.id] = value
            results.append(value)
            result_index += 1
        op.results = tuple(results)
        ops_by_id[op_id] = op
    # Phase 2: function signatures, entry blocks (arguments + op refs).
    # A trusted envelope's block op entry mirrors the op id in ``"ref"``.
    for entry in functions_data:
        input_types = tuple(
            _decode_type_fast(value_type) for value_type in entry["input_types"]
        )
        block_data = entry["block"]
        block = Block()
        arguments = []
        argument_index = 0
        for argument_entry in block_data["arguments"]:
            value_type = argument_entry["type"]
            key = id(value_type)
            decoded_type = type_memo.get(key)
            if decoded_type is None:
                decoded_type = _decode_type_fast(value_type)
                type_memo[key] = decoded_type
            value = _value(argument_entry["id"], decoded_type, block, argument_index)
            values_by_id[value.id] = value
            arguments.append(value)
            argument_index += 1
        block.arguments = tuple(arguments)
        for op_entry in block_data["ops"]:
            op_id = op_entry.get("id")
            if op_id is None:  # the native writer always emits it, mirrors "ref"
                op_id = int(op_entry["ref"])
            op = ops_by_id[op_id]
            op.parent = block
            block.ops.append(op)
        region = Region()
        region.append_block(block)
        module.add_function(
            Function(
                entry["name"],
                input_types,
                region,
                _decode_generic(entry.get("metadata") or {}, constants),
            )
        )
    # Phase 3: nested op regions (none in the common arithmetic graph).
    for entry in ops_data:
        regions_data = entry["regions"]
        if not regions_data:
            continue
        op = ops_by_id[entry["id"]]
        op.regions = tuple(
            _build_region_fast(region_data, ops_by_id, values_by_id, type_memo, op)
            for region_data in regions_data
        )
    # Phase 4: operands + ordered, complete use lists (appended directly: a
    # trusted envelope has no duplicate uses, so `Value.add_use`'s dedup index
    # would only cost an index build per value).
    for entry in ops_data:
        op = ops_by_id[entry["id"]]
        operands = []
        operand_index = 0
        for value_id in entry["operands"]:
            operand = values_by_id[value_id]
            operands.append(operand)
            operand.uses.append(Use(op, operand_index))
            operand_index += 1
        op.operands = tuple(operands)
    _fast_forward_counters(module, ops_by_id, values_by_id)
    return module


