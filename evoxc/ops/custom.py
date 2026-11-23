"""Custom compiler escape hatches: ``custom_call`` and ``inline_stablehlo``.

The two graph-side escape hatches that hand work to the EXTERNAL compiler
instead of to evoxc's own backends:

- ``custom_call`` declares an opaque call to a NAMED StableHLO custom-call
  target (``call_target_name``) plus an optional opaque
  ``backend_config`` string and an ``api_version``. The target is expected to
  be registered with the compiler that lowers the graph (IREE/XLA custom-call
  targets), never with evoxc.
- ``inline_stablehlo`` declares a raw StableHLO/MLIR snippet (``body``)
  emitted verbatim as this op's lowering — the last-resort escape hatch when
  no ``custom_call`` target exists.

Contrast with ``external_call`` (``evoxc.ops.external``): an external kernel
is a Python callable evoxc itself dispatches at run time, while these two ops
are pure COMPILER declarations — evoxc never executes, interprets or validates
the target/snippet, it only carries it into the emitted program.

Binding rules (see this directory's ``CONTEXT.md``, "Custom-call /
raw-StableHLO ops"):

- Both ops are OPAQUE and declare their outputs through ``result_specs`` —
  their result types resolve from that attribute (no ``shape_fn``), and
  ``ir.verify`` requires ``result_specs`` to equal the op result types.
- Both are ``callback`` effect: an opaque external call/snippet is never
  reordered, duplicated or eliminated. ``has_side_effect=True`` additionally
  declares OBSERVABLE effects so the compiler must preserve them.
- The target name / backend config / snippet is STATIC — a Python str that
  specializes the graph. It is validated by the COMPILER, never by evoxc
  (which does not parse compiler flags or MLIR).
- Neither the numpy interpreter nor the native CPU engine implements these
  ops: they are only compilable on compiler backends that declare the
  custom-op capability (``Capabilities.custom_ops``), and everywhere else the
  failure is an explicit ``BackendError`` — never a silent fallback.
"""
from __future__ import annotations

import re
from typing import Tuple, Union

from evoxc import core
from evoxc import ir

from . import _utils

__all__ = ["custom_call", "inline_stablehlo"]

# ``inline_stablehlo`` snippet token patterns. The trailing
# ``(?![0-9A-Za-z_])`` lookahead keeps ``%in0x`` from being read as ``%in0``
# (and likewise for ``%out``). The assignment pattern accepts both the plain
# ``%out0 = ...`` form and the explicit-type ``%out0:2 = ...`` form.
_IN_REF_RE = re.compile(r"%in(\d+)(?![0-9A-Za-z_])")
_OUT_REF_RE = re.compile(r"%out(\d+)(?![0-9A-Za-z_])")
_OUT_ASSIGN_RE = re.compile(
    r"^\s*%out(\d+)(?![0-9A-Za-z_])\s*(?::\d+)?\s*=", re.MULTILINE
)


def _result_specs(op_name, result):
    """Normalize ``result`` into ``(single, specs)`` or raise ``TypeError``."""
    if isinstance(result, core.TensorSpec):
        return True, (result,)
    if isinstance(result, (tuple, list)):
        if not result or not all(
            isinstance(spec, core.TensorSpec) for spec in result
        ):
            raise TypeError(
                f"{op_name}: result must be a TensorSpec or a non-empty "
                "tuple/list of TensorSpecs"
            )
        return False, tuple(result)
    raise TypeError(
        f"{op_name}: result must be a TensorSpec or a tuple/list of "
        f"TensorSpecs, got {type(result).__name__}"
    )


def _location_suffix(loc) -> str:
    """`` (at file:line:col)`` for a real call site, ``""`` otherwise."""
    if loc.line > 0 and loc.file != "<unknown>":
        return f" (at {loc})"
    return ""


def custom_call(
    call_target_name, *operands, result, backend_config="", api_version=1,
    has_side_effect=False,
) -> Union["core.SymbolicTensor", Tuple["core.SymbolicTensor", ...]]:
    """Call a named StableHLO custom-call target at compile time.

    Builds a ``custom_call`` IR op (effect ``callback``) carrying the target
    name, the declared result specs, the opaque backend configuration, the
    custom-call API version and the side-effect flag as op attributes. The
    target is declared by NAME only — graph artifacts carry no kernel code.

    The target must be one the external compiler knows (an IREE/XLA
    custom-call target), and the compiler that lowers the graph — not evoxc —
    validates the target name, the ``backend_config`` payload and the
    operand/result types: evoxc neither parses compiler flags nor registry
    contents and never emits a guess.

    The op compiles only on a compiler backend that declares the custom-op
    capability (``Capabilities.custom_ops``); the numpy interpreter and the
    native CPU engine reject it with an explicit ``BackendError`` — never a
    silent fallback.

    Args:
        call_target_name: The StableHLO custom-call target symbol (non-empty
            str). STATIC — it specializes the graph.
        *operands: ``SymbolicTensor`` or Python scalar operands of the call
            (a scalar is auto-promoted to its own 0-d constant operand).
        result: A ``core.TensorSpec`` (single output) or a non-empty
            tuple/list of ``TensorSpec`` (multiple outputs) declaring the
            call's result contract.
        backend_config: Opaque backend-specific configuration string passed
            through with the call (empty = none).
        api_version: The StableHLO custom-call API version; must be >= 1
            (the value constraint is enforced by ``ir.verify``).
        has_side_effect: ``True`` declares observable side effects, so the
            compiler must not reorder or eliminate the call.

    Returns:
        A single ``SymbolicTensor`` when ``result`` is a ``TensorSpec``, else
        a tuple of ``SymbolicTensor`` (one per spec).

    Raises:
        core.TraceError: no active trace; or a concrete ``Tensor`` operand.
        TypeError: ``call_target_name`` is not a non-empty str;
            ``backend_config`` is not a str; ``api_version`` is not an int
            (bool rejected); ``has_side_effect`` is not a bool; ``result`` is
            not a TensorSpec or a non-empty tuple/list of TensorSpecs.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    if not isinstance(call_target_name, str) or not call_target_name:
        kind = (
            type(call_target_name).__name__
            if not isinstance(call_target_name, str)
            else "empty str"
        )
        raise TypeError(
            f"custom_call: call_target_name must be a non-empty str, got {kind}"
        )
    if not isinstance(backend_config, str):
        raise TypeError(
            "custom_call: backend_config must be a str, got "
            f"{type(backend_config).__name__}"
        )
    if isinstance(api_version, bool) or not isinstance(api_version, int):
        raise TypeError(
            "custom_call: api_version must be an int, got "
            f"{type(api_version).__name__}"
        )
    if not isinstance(has_side_effect, bool):
        raise TypeError(
            "custom_call: has_side_effect must be a bool, got "
            f"{type(has_side_effect).__name__}"
        )
    single, specs = _result_specs("custom_call", result)
    # The IR requires result_specs to be a sequence of ValueType instances
    # EXACTLY equal to the op's result types (ir.verify enforces this).
    result_specs = tuple(
        ir.ValueType(dtype=spec.dtype, shape=tuple(spec.shape))
        for spec in specs
    )
    op_operands = tuple(
        _utils.as_operand(operand, location=loc).value for operand in operands
    )
    op = builder.create(
        "custom_call",
        operands=op_operands,
        attributes={
            "call_target_name": call_target_name,
            "result_specs": result_specs,
            "backend_config": backend_config,
            "api_version": api_version,
            "has_side_effect": has_side_effect,
        },
        location=loc,
    )
    results = tuple(
        core.SymbolicTensor(
            value=value,
            dtype=value.type.dtype,
            shape=value.type.shape,
            location=loc,
        )
        for value in op.results
    )
    return results[0] if single else results


def inline_stablehlo(
    body, *operands, result, has_side_effect=False,
) -> Union["core.SymbolicTensor", Tuple["core.SymbolicTensor", ...]]:
    """Emit a raw StableHLO/MLIR snippet verbatim as this op's lowering.

    Builds an ``inline_stablehlo`` IR op (effect ``callback``) carrying the
    snippet, the declared result specs and the side-effect flag. The snippet
    is the last-resort compiler escape hatch: evoxc emits it verbatim and
    never parses or validates it beyond the ``%in``/``%out`` token contract
    below. It compiles only on a compiler backend that declares the custom-op
    capability (``Capabilities.custom_ops``); the numpy interpreter and the
    native CPU engine reject it with an explicit ``BackendError`` — never a
    silent fallback.

    SSA contract of ``body``:

    - Operand *i* (0-based, in the caller's positional order — a Python
      scalar counts as its own operand) is available to the snippet as
      ``%in<i>``. Any ``%in<i>`` with ``i`` greater than or equal to the
      operand count raises ``core.TraceError``. No ``%in`` reference is
      required.
    - Result *i* MUST be defined by an assignment to ``%out<i>`` — plain
      ``%out<i> = ...`` or explicit-type ``%out<i>:<type> = ...`` — and every
      declared result index must be assigned at least once (otherwise
      ``core.TraceError``). A ``%out<i>`` reference or assignment with ``i``
      greater than or equal to the declared result count raises
      ``core.TraceError``.
    - The declared result types must equal ``result_specs`` in order and
      dtype; THE COMPILER validates that — evoxc never parses MLIR.

    Args:
        body: The raw StableHLO/MLIR snippet (non-empty str), emitted
            verbatim. STATIC — it specializes the graph.
        *operands: ``SymbolicTensor`` or Python scalar operands, bound to
            ``%in0``, ``%in1``, … in caller order.
        result: A ``core.TensorSpec`` (single output) or a non-empty
            tuple/list of ``TensorSpec`` (multiple outputs); result *i* is
            ``%out<i>``.
        has_side_effect: ``True`` declares observable side effects, so the
            compiler must not reorder or eliminate the snippet.

    Returns:
        A single ``SymbolicTensor`` when ``result`` is a ``TensorSpec``, else
        a tuple of ``SymbolicTensor`` (one per spec).

    Raises:
        core.TraceError: no active trace; a concrete ``Tensor`` operand; an
            ``%in<i>`` operand reference out of range; an ``%out<i>`` result
            reference or assignment out of range; or a declared result index
            that the snippet never assigns.
        TypeError: ``body`` is not a non-empty str; ``has_side_effect`` is
            not a bool; ``result`` is not a TensorSpec or a non-empty
            tuple/list of TensorSpecs.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    if not isinstance(body, str) or not body:
        kind = type(body).__name__ if not isinstance(body, str) else "empty str"
        raise TypeError(
            f"inline_stablehlo: body must be a non-empty str, got {kind}"
        )
    if not isinstance(has_side_effect, bool):
        raise TypeError(
            "inline_stablehlo: has_side_effect must be a bool, got "
            f"{type(has_side_effect).__name__}"
        )
    single, specs = _result_specs("inline_stablehlo", result)

    # Token-level snippet semantics (never a parse of the MLIR itself).
    suffix = _location_suffix(loc)
    for match in _IN_REF_RE.finditer(body):
        index = int(match.group(1))
        if index >= len(operands):
            raise core.TraceError(
                f"inline_stablehlo: %in{index} is out of range "
                f"({len(operands)} operand(s) available){suffix}"
            )
    for match in _OUT_REF_RE.finditer(body):
        index = int(match.group(1))
        if index >= len(specs):
            raise core.TraceError(
                f"inline_stablehlo: %out{index} is out of range "
                f"({len(specs)} result(s) declared){suffix}"
            )
    assigned = {int(match.group(1)) for match in _OUT_ASSIGN_RE.finditer(body)}
    for index in range(len(specs)):
        if index not in assigned:
            raise core.TraceError(
                f"inline_stablehlo: declared result %out{index} is never "
                f"assigned ({len(assigned)} of {len(specs)} results "
                f"assigned){suffix}"
            )

    # The IR requires result_specs to be a sequence of ValueType instances
    # EXACTLY equal to the op's result types (ir.verify enforces this).
    result_specs = tuple(
        ir.ValueType(dtype=spec.dtype, shape=tuple(spec.shape))
        for spec in specs
    )
    op_operands = tuple(
        _utils.as_operand(operand, location=loc).value for operand in operands
    )
    op = builder.create(
        "inline_stablehlo",
        operands=op_operands,
        attributes={
            "body": body,
            "result_specs": result_specs,
            "has_side_effect": has_side_effect,
        },
        location=loc,
    )
    results = tuple(
        core.SymbolicTensor(
            value=value,
            dtype=value.type.dtype,
            shape=value.type.shape,
            location=loc,
        )
        for value in op.results
    )
    return results[0] if single else results
