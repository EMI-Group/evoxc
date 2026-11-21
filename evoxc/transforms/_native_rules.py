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
``tangent_values`` (jvp), ``cotangent_type``, ``cotangent_values`` (vjp).

Eligible `rule_key`s only (`block:*`, `external:*`, `sparse_*`, ...): the
registry lookup uses `batching.batching_rules` under the key the engine
computed from the op name and its name/block-name attribute. Lookup is LATE
(every call), so rules registered after import are picked up.

Batching rules are served faithfully: the scratch function's block arguments
are the op's REWRITTEN operands (exactly what native's `inline_function`
supplies), and the proxy op carries the original name/attributes plus
detached result values of the request's result types (rules read
`op.results[0].type`, `op.name`, `op.attributes`).

AD kinds are served the same way. Native passes the dual SSA values
themselves — ``tangent_values`` (one entry per operand, operand order) for
``jvp`` and ``cotangent_values`` (one entry per result, result order) for
``vjp`` — so the scratch function's block arguments are, in order: the op's
operands (n), its result proxies (r), then the NON-ZERO duals in list order.
A dual argument's type is its operand's (jvp) / result's (vjp) type; a zero
slot contributes no argument and is bound as `autodiff.ZeroTangent`.
The response's ``output_ids`` holds exactly one entry per operand (jvp) /
result (vjp), in order, and is NULLABLE: a ``None``/`autodiff.ZeroTangent`
entry is sent as ``None`` (native's structurally-zero dual), so the scratch
subgraph returns only its real contributions and no zero-materialization op is
emitted — native materializes zeros only where a zero dual reaches a graph
output.
"""

from __future__ import annotations

import base64
import threading
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from evoxc import _native_codec, core, ir
from evoxc.trace import with_builder
from evoxc.transforms import autodiff, batching
from evoxc.transforms._metadata import UNMAPPED, MappedAxes

#: Attribute schema tags (see `evoxc/ir/op_defs/__init__.py`).
_ATTR_ANY = "any"
_ATTR_DTYPE = "dtype"
_ATTR_NDARRAY = "ndarray"
_ATTR_SHAPE = "shape"
_TUPLE_TAG = "__evoxc_tuple__"

#: The exception a Python rule raised during the current native transform call.
#: The engine re-types every bridge failure as a `TransformError` (its message
#: text is preserved but a rule's own error class is not), so `call_native`
#: restores this original afterwards — a portable-decomposition failure keeps
#: raising `BlockError`, exactly as it did when the rule ran in Python.
_RULE_FAILURE = threading.local()


def reset_rule_failure() -> None:
    """Forget the previous call's rule failure (called before a native call)."""
    _RULE_FAILURE.error = None


def take_rule_failure() -> Optional[BaseException]:
    """Return (and clear) the exception the last native call's rules raised."""
    error = getattr(_RULE_FAILURE, "error", None)
    _RULE_FAILURE.error = None
    return error


def extension_rule(request: Mapping[str, Any]):
    """The callback the native engine invokes; see the module docstring."""
    try:
        kind = (request or {}).get("kind")
        if kind == "batching":
            return _batching_response(request)
        if kind in ("jvp", "vjp"):
            return _derivative_response(request, kind)
        return None
    except BaseException as exc:  # re-raised: recorded for `take_rule_failure`
        _RULE_FAILURE.error = exc
        raise


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


# --- derivative rules -------------------------------------------------------


def _derivative_response(request: Mapping[str, Any], kind: str):
    """Serve a Python JVP/VJP rule to the native AD engine.

    ``jvp`` uses the registered JVP rule when there is one and otherwise
    derives the JVP from the registered VJP rule (the adjoint trick in
    `autodiff._jvp_from_vjp`); ``vjp`` requires a registered VJP rule. No rule
    at all returns ``None`` so the engine raises its explicit
    `TransformError`.
    """
    rule_key = request.get("rule_key") or ""
    if kind == "jvp":
        rule = autodiff.jvp_rules.get(rule_key)
        if rule is None and rule_key not in autodiff.vjp_rules:
            return None
    else:
        rule = autodiff.vjp_rules.get(rule_key)
        if rule is None:
            return None

    op_name = request["op_name"]
    operand_types = _types(request.get("operand_types"))
    result_types = _types(request.get("result_types"))
    # The duals are aligned with the operands (jvp) / results (vjp); a dual
    # block argument carries the type of its primal and zero slots contribute
    # no argument (native's `extension_arguments` order).
    primal_types = operand_types if kind == "jvp" else result_types
    dual_key = "tangent_values" if kind == "jvp" else "cotangent_values"
    dual_ids = request.get(dual_key)
    if dual_ids is None:
        dual_ids = (None,) * len(primal_types)
    dual_ids = tuple(dual_ids)
    if len(dual_ids) != len(primal_types):
        raise core.TransformError(
            f"{kind} rule for op '{rule_key}': the native request carries "
            f"{len(dual_ids)} dual value(s) for {len(primal_types)} "
            f"{'operand' if kind == 'jvp' else 'result'}(s)"
        )
    dual_types = tuple(
        primal_types[index]
        for index, value in enumerate(dual_ids)
        if value is not None
    )

    builder, block, proxy, dual_args = _derivative_scratch(
        op_name, request, operand_types, result_types, dual_types
    )
    duals = _bind_duals(dual_ids, dual_args)
    with with_builder(builder):
        if kind == "jvp":
            if rule is not None:
                outputs = rule(proxy, duals)
            else:
                outputs = autodiff._jvp_from_vjp(proxy, duals, builder)
        else:
            outputs = rule(proxy, duals, proxy.operands)

    primals = proxy.results if kind == "jvp" else proxy.operands
    outputs = _validate_derivative_output(rule_key, kind, outputs, len(primals))
    # Only the real contributions enter the scratch block; a zero slot becomes
    # a `None` output id (native's structurally-zero dual), so no zeros op is
    # emitted here.
    real = tuple(entry for entry in outputs if isinstance(entry, ir.Value))
    builder.set_terminator(block, "return", operands=real)
    return {
        "module": _native_codec.encode_module(builder.module),
        "entry": builder.module.main.name,
        "output_ids": [
            None if entry is None or isinstance(entry, autodiff.ZeroTangent)
            else entry.id
            for entry in outputs
        ],
    }


def _bind_duals(dual_ids: Sequence[Any], dual_args: Sequence["ir.Value"]) -> Tuple[Any, ...]:
    """One entry per operand (jvp) / result (vjp): the dual value or `ZeroTangent`."""
    duals = []
    position = 0
    for entry in dual_ids:
        if entry is None:
            duals.append(autodiff.ZeroTangent())
        else:
            duals.append(dual_args[position])
            position += 1
    return tuple(duals)


def _validate_derivative_output(rule_key: str, kind: str, outputs, expected: int):
    """Validate a rule's return with the sweep's (pinned) `TransformError` wording."""
    noun, other = ("tangents", "result") if kind == "jvp" else ("cotangents", "operand")
    if not isinstance(outputs, (tuple, list)):
        raise core.TransformError(
            f"{kind} rule for op '{rule_key}' did not return a tuple of "
            f"{noun}, got {type(outputs).__name__}"
        )
    if len(outputs) != expected:
        raise core.TransformError(
            f"{kind} rule for op '{rule_key}' returned {len(outputs)} "
            f"{noun}, expected {expected} (one per {other})"
        )
    entry_name = "tangent" if kind == "jvp" else "cotangent"
    for entry in outputs:
        if entry is not None and not isinstance(entry, (ir.Value, autodiff.ZeroTangent)):
            raise core.TransformError(
                f"{kind} rule for op '{rule_key}' returned an invalid "
                f"{entry_name} entry {entry!r}: entries must be "
                f"ir.Value | None | ZeroTangent"
            )
    return tuple(outputs)


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


def _derivative_scratch(op_name, request, operand_types, result_types, dual_types):
    """A scratch module for one AD rule: operands, result proxies, then duals.

    Block-argument order is exactly native's `extension_arguments`: the op's
    rewritten operands (n), its result proxies (r), then the non-zero duals in
    list order. The proxy op exposes the operands as `.operands` and the result
    proxies as `.results` (rules read `.type` off both).
    """
    builder = ir.Builder()
    builder.build_module(name=f"{op_name}__evoxc_rule")
    function = builder.build_function(
        name=f"{op_name}__evoxc_rule",
        input_types=tuple(operand_types) + tuple(result_types) + tuple(dual_types),
    )
    block = function.entry_block
    attributes = _decode_attributes(op_name, request.get("attributes") or {})
    operand_count = len(operand_types)
    result_count = len(result_types)
    proxy = ir.Op(
        name=op_name,
        id=-1,
        operands=tuple(block.arguments[:operand_count]),
        attributes=attributes,
        results=tuple(block.arguments[operand_count:operand_count + result_count]),
    )
    dual_args = tuple(block.arguments[operand_count + result_count:])
    return builder, block, proxy, dual_args


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
