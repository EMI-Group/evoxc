"""Extension-rule bridge: serve Python rules to the native transform engine.

The native engine owns the op rewrite, but a handful of ops have rules that
live in PYTHON (custom blocks via `block:<name>`, external kernels via
`external:<name>`, sparse ops via `sparse_*`). For those the engine calls back
into :func:`extension_rule` with a neutral request dict and expects either
``None`` (no rule — the engine then raises its explicit `TransformError`) or a
response dict describing a straight-line subgraph that computes the rewritten
values:

    {"module": <evoxc-native-ir envelope>, "entry": <function name>,
     "output_ids": [<value id>, ...],
     "output_axes": [0|None, ...] | omitted}

Request keys (`native/CONTEXT.md` "Extension-rule bridge"): ``kind``
(``"batching"`` | ``"jvp"`` | ``"vjp"``), ``op_name``, ``rule_key``,
``attributes``, ``operand_types``, ``result_types``, ``axes``, ``tangents``,
``cotangent_type``.

Eligible `rule_key`s only (`block:*`, `external:*`, `sparse_*`, ...): the
registry lookup mirrors `batching._resolve_batching_rule` — the key the engine
computed from the op name and its name/block-name attribute. Lookup is LATE
(every call), so rules registered after import are picked up.

Batching rules are served faithfully: the scratch function's block arguments
are the op's REWRITTEN operands (exactly what native's `inline_function`
supplies), and the proxy op carries the original name/attributes plus
detached result values of the request's result types (rules read
`op.results[0].type`, `op.name`, `op.attributes`).

AD kinds: the request carries only tangent FLAGS / a cotangent TYPE — the
actual tangent/cotangent SSA values are not passed and cannot be bound into
the inlined subgraph — so this bridge returns ``None`` for them (the engine
then raises its explicit non-differentiable error). It is forward-compatible:
if a request ever carries ``tangent_values`` / ``cotangent_values`` (lists of
value ids aligned with the operands / results), those are bound as trailing
block arguments and the rule is invoked faithfully.
"""

from __future__ import annotations

import base64
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from evoxc import _native_codec, core, ir
from evoxc.trace import with_builder
from evoxc.transforms import batching
from evoxc.transforms._metadata import UNMAPPED, MappedAxes

#: Attribute schema tags (see `evoxc/ir/op_defs/__init__.py`).
_ATTR_ANY = "any"
_ATTR_DTYPE = "dtype"
_ATTR_NDARRAY = "ndarray"
_ATTR_SHAPE = "shape"
_TUPLE_TAG = "__evoxc_tuple__"


def extension_rule(request: Mapping[str, Any]):
    """The callback the native engine invokes; see the module docstring."""
    kind = (request or {}).get("kind")
    if kind == "batching":
        return _batching_response(request)
    if kind in ("jvp", "vjp"):
        return _derivative_response(request, kind)
    return None


# --- batching ---------------------------------------------------------------


def _batching_response(request: Mapping[str, Any]):
    rule_key = request.get("rule_key") or ""
    rule = batching.batching_rules.get(rule_key)
    if rule is None:
        return None
    op_name = request["op_name"]
    operand_types = _types(request.get("operand_types"))
    result_types = _types(request.get("result_types"))
    axes = tuple(_mapped_axes(entry) for entry in (request.get("axes") or ()))
    if len(axes) != len(operand_types):
        axes = tuple(UNMAPPED for _ in operand_types)
    builder, block, proxy = _scratch(op_name, request, operand_types, result_types)
    with with_builder(builder):
        new_values, new_axes = rule(proxy, tuple(block.arguments), axes)
    new_values, new_axes = batching._validate_rule_result(proxy, new_values, new_axes)
    builder.set_terminator(block, "return", operands=tuple(new_values))
    response = {
        "module": _native_codec.encode_module(builder.module),
        "entry": builder.module.main.name,
        "output_ids": [value.id for value in new_values],
    }
    output_axes = _output_axes(new_axes)
    if output_axes is not None:
        response["output_axes"] = output_axes
    return response


# --- derivative rules (forward-compatible) ---------------------------------


def _derivative_response(request: Mapping[str, Any], kind: str):
    # Known native protocol gap (see the module docstring): the request does
    # not carry the tangent/cotangent SSA values, so an inlined subgraph
    # cannot bind them. Returning None makes the engine raise its explicit
    # non-differentiable `TransformError`. Forward-compatible: a request that
    # ever carries `tangent_values` / `cotangent_values` is served below.
    values_key = "tangent_values" if kind == "jvp" else "cotangent_values"
    if not request.get(values_key):
        return None
    return _bound_derivative_response(request, kind, values_key)  # pragma: no cover


# --- scratch module / proxy op ---------------------------------------------


def _types(raw) -> Tuple["ir.ValueType", ...]:
    return tuple(_native_codec.decode_type(item) for item in (raw or ()))


def _mapped_axes(entry) -> MappedAxes:
    if entry is None:
        return UNMAPPED
    return MappedAxes((int(entry),))


def _output_axes(new_axes) -> Optional[list]:
    """`output_axes` for the response, or None to let native derive them.

    Native accepts only ``0`` / ``None`` per output (non-leading axes are
    deferred). A rule that returns a MULTI-axis value ((0, 1) after nested
    vmap) cannot express that, so the key is omitted and native derives the
    mappedness from the leading batch dims of the result types.
    """
    if any(mapped.count > 1 for mapped in new_axes):
        return None
    return [None if mapped.count == 0 else 0 for mapped in new_axes]


def _scratch(op_name, request, operand_types, result_types):
    """A one-function scratch module + the proxy op for one rule invocation."""
    builder = ir.Builder()
    builder.build_module(name=f"{op_name}__evoxc_rule")
    function = builder.build_function(
        name=f"{op_name}__evoxc_rule", input_types=tuple(operand_types)
    )
    block = function.entry_block
    attributes = _decode_attributes(op_name, request.get("attributes") or {})
    results = tuple(
        _detached_value(value_type, index)
        for index, value_type in enumerate(result_types)
    )
    proxy = ir.Op(
        name=op_name,
        id=-1,
        operands=tuple(block.arguments),
        attributes=attributes,
        results=results,
    )
    return builder, block, proxy


def _detached_value(value_type: "ir.ValueType", index: int) -> "ir.Value":
    """An SSA value that belongs to no block — the proxy op's result.

    Rules only read `.type` off these (the batching contract: rules build on
    the OPERANDS they receive, never on `op.results`). Ids are negative so
    they can never alias a value of the scratch module.
    """
    return ir.Value(
        id=-1 - index, type=value_type, owner=None, index=index
    )


# --- attribute decoding -----------------------------------------------------


def _decode_attributes(op_name: str, raw: Mapping[str, Any]):
    schema = {}
    try:
        for spec in ir.opdef(op_name).attributes:
            schema[spec.name] = spec.type
    except Exception:  # pragma: no cover - unknown op name: generic decode
        schema = {}
    decoded = {}
    for key, value in raw.items():
        tag = schema.get(key)
        if tag == _ATTR_NDARRAY:
            decoded[key] = _decode_ndarray(value)
        elif tag == _ATTR_SHAPE and isinstance(value, list):
            decoded[key] = tuple(_decode_dim(item) for item in value)
        elif tag == _ATTR_ANY and key == "result_specs":
            decoded[key] = tuple(
                _native_codec.decode_type(item) for item in _tuple_items(value)
            )
        elif tag == _ATTR_DTYPE and isinstance(value, str):
            decoded[key] = core.dtype(value)
        else:
            decoded[key] = _decode_generic(value)
    return decoded


def _tuple_items(value):
    if isinstance(value, Mapping) and _TUPLE_TAG in value:
        return list(value[_TUPLE_TAG])
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _decode_generic(value):
    if isinstance(value, Mapping) and _TUPLE_TAG in value:
        return tuple(_decode_generic(item) for item in value[_TUPLE_TAG])
    if isinstance(value, list):
        return [_decode_generic(item) for item in value]
    return value


def _decode_dim(value):
    if isinstance(value, Mapping):
        if "dim" in value:
            name = value["dim"]
            size = value.get("size")
            return core.Dim(name, size) if size is not None else core.Dim(name)
        decoder = getattr(_native_codec, "_decode_dim", None)
        if decoder is not None:  # pragma: no cover - DimExpr round trip
            return decoder(value)
    return value


def _decode_ndarray(value):
    """Decode the wire array form (`dtype`/`shape`/`data` base64) to numpy."""
    if isinstance(value, np.ndarray):
        return value
    if not isinstance(value, Mapping):
        return value
    dtype = value.get("dtype")
    shape = value.get("shape")
    data = value.get("data")
    if dtype is None or data is None:
        return value
    try:
        array = np.frombuffer(base64.b64decode(data), dtype=np.dtype(dtype))
    except Exception:  # pragma: no cover - unexpected payload: leave as-is
        return value
    if shape is not None:
        extents = [item if isinstance(item, int) else _decode_dim(item) for item in shape]
        if all(isinstance(item, int) for item in extents):
            array = array.reshape(tuple(extents))
    return array
