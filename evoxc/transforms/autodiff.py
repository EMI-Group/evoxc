"""Automatic-differentiation rule registries and shared helpers.

Holds the JVP/VJP rule registries, the `ZeroTangent` sentinel and the VJP→JVP
adjoint derivation (`_jvp_from_vjp`).
The native (Rust) engine owns every AD sweep (`grad` / `jvp` / `vjp`) and every
builtin derivative rule; it reaches these registries through the extension-rule
bridge in `_native_rules.py`, which serves the Python rules for `block:*`,
`external:*` and `sparse_*` ops.
All AD transforms are graph→graph: the result graphs contain only ordinary
`evoxc.ops` ops (backends need no autodiff runtime support). Rule signatures are
binding — see `./CONTEXT.md` "Rule-call signatures" and "AD semantics".
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from evoxc import core, ir, ops
from evoxc.core import TransformError

# Binding rule signatures (see ./CONTEXT.md):
#   jvp: rule(op, tangents) -> output_tangents
#     tangents: tuple of ir.Value | None | ZeroTangent aligned with
#     `op.operands` (None == ZeroTangent == zero tangent); primal values are
#     available as `op.operands` / `op.results`. Returns a tuple aligned with
#     `op.results`.
#   vjp: rule(op, cotangents, primals) -> input_cotangents
#     cotangents: tuple of ir.Value | None | ZeroTangent aligned with
#     `op.results`; primals: tuple of the op's operand values. Returns a tuple
#     of ir.Value | None | ZeroTangent aligned with `op.operands` (None /
#     ZeroTangent = zero gradient).
JVPRule = Callable[[ir.Op, Tuple[Optional[ir.Value], ...]], Tuple[ir.Value, ...]]
VJPRule = Callable[
    [ir.Op, Tuple[Optional[ir.Value], ...], Tuple[ir.Value, ...]],
    Tuple[Optional[ir.Value], ...],
]

#: Op-def-name → rule. Custom blocks register under `block:<block_name>` (done
#: by `BlockOp.jvp_rule/vjp_rule(fn)` in `evoxc/block`). Builtin rules are
#: registered by `rules.py` at import time.
jvp_rules: Dict[str, JVPRule] = {}
vjp_rules: Dict[str, VJPRule] = {}


class ZeroTangent:
    """Sentinel for an all-zero tangent/cotangent that need not materialize.

    Rules return this for undifferentiated values (e.g. `stop_gradient`,
    boolean/int-producing ops). The native engine treats it as a structurally
    zero dual: the extension-rule bridge sends a `None` output id for it, and
    the engine materializes a zeros value only where a zero dual reaches a
    graph output.
    """

    __slots__ = ()
    _instance: Optional["ZeroTangent"] = None

    def __new__(cls) -> "ZeroTangent":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "ZeroTangent"


def register_jvp_rule(op_name: str, fn: JVPRule) -> None:
    """Register (or replace) the JVP rule for `op_name`.

    Custom blocks register under the `block:<block_name>` namespace; that is
    exactly what `BlockOp.jvp_rule(fn)` does in `evoxc/block`.
    """
    if not isinstance(op_name, str) or not op_name:
        raise ValueError("op_name must be a non-empty string")
    jvp_rules[op_name] = fn


def register_vjp_rule(op_name: str, fn: VJPRule) -> None:
    """Register (or replace) the VJP rule for `op_name`.

    Custom blocks register under the `block:<block_name>` namespace; that is
    exactly what `BlockOp.vjp_rule(fn)` does in `evoxc/block`.
    """
    if not isinstance(op_name, str) or not op_name:
        raise ValueError("op_name must be a non-empty string")
    vjp_rules[op_name] = fn


def get_jvp_rule(op_name: str) -> Optional[JVPRule]:
    """The registered JVP rule for `op_name`, or `None` (never raises)."""
    return jvp_rules.get(op_name)


def get_vjp_rule(op_name: str) -> Optional[VJPRule]:
    """The registered VJP rule for `op_name`, or `None` (never raises)."""
    return vjp_rules.get(op_name)


def require_vjp_rule(op_name: str) -> VJPRule:
    """Like `get_vjp_rule`, but raises `TransformError` naming the op."""
    rule = vjp_rules.get(op_name)
    if rule is None:
        raise TransformError(
            f"grad/vjp: no VJP rule for {_rule_display(op_name)}. "
            f"{_rule_hint(op_name, 'register_vjp_rule', 'vjp_rule').capitalize()}; "
            f"there is no silent fallback."
        )
    return rule


# ---------------------------------------------------------------------------
# Rule conventions (binding for `rules.py`, `evoxc/block` and
# `evoxc/external_rules.py` — they code against this).
#
# PROXY-OP CONVENTION: a rule is invoked as `rule(proxy, ...)` where `proxy` is
# a fresh `ir.Op` dataclass instance that is NEVER inserted into any block. It
# carries the original op's `name`/`id`/`attributes` (a shallow-copied dict;
# leaf objects such as constant arrays are shared, read-only) and `location`,
# but its `operands` and `results` are the transformed graph's SSA values (same
# types as the originals). `op.opdef`/`op.effect`/`op.result` resolve normally
# (by name). Rules must treat the proxy as read-only and must NOT follow
# `value.owner`/`value.defining_op` off proxy values.
# ---------------------------------------------------------------------------


def _rule_name(op: ir.Op) -> str:
    """Registry key for `op` (block calls are keyed ``block:<block_name>``,
    external calls keyed ``external:<name>``)."""
    if op.name == "block_call":
        return f"block:{op.attributes['block_name']}"
    if op.name == "external_call":
        return f"external:{op.attributes['name']}"
    return op.name


def _rule_display(key: str) -> str:
    """Message display for a missing rule keyed `key`.

    External-call keys name the `external_call` op (UX/tests pin the op
    name) with the key in parentheses; block keys and plain op names keep
    the key itself (which names the block / the op).
    """
    if key.startswith("external:"):
        return f"op 'external_call' (key '{key}')"
    return f"op '{key}'"


def _rule_hint(key: str, register_fn: str, block_method: str) -> str:
    """Registration-advice clause for a missing rule keyed `key`.

    Adaptive to the namespaced key prefixes: an `external:<name>` key points
    at the external-kernel handle (`evoxc.register_external_kernel`) and a
    `block:<name>` key at the `BlockOp` decorators; plain op-name keys keep
    the generic registry hint (`register_fn`).
    """
    if key.startswith("external:"):
        name = key[len("external:"):]
        return (
            f"register one via the external-kernel handle returned by "
            f"evoxc.register_external_kernel('{name}', fn) "
            f"(.batching_rule/.vjp_rule/.jvp_rule) or register a portable "
            f"decomposition"
        )
    if key.startswith("block:"):
        return (
            f"register one with {register_fn}('{key}', fn) "
            f"(custom blocks: BlockOp.{block_method})"
        )
    return f"register one with {register_fn}('{key}', fn)"


def _sym(value: ir.Value) -> "core.SymbolicTensor":
    """Wrap an `ir.Value` as a `core.SymbolicTensor` (dtype/shape from its type)."""
    return core.SymbolicTensor(
        value=value, dtype=value.type.dtype, shape=value.type.shape
    )


def _is_real(tangent) -> bool:
    """True iff `tangent` is a real `ir.Value` (not None/ZeroTangent)."""
    return isinstance(tangent, ir.Value)


def _combine(entries) -> Optional[ir.Value]:
    """Combine accumulated cotangent entries: `None` for no entries, the
    single entry as-is, or the add-reduced `ir.Value` of multiple entries
    (built with `evoxc.ops.add` — the canonical cotangent-accumulation op)."""
    if not entries:
        return None
    if len(entries) == 1:
        return entries[0]
    total = _sym(entries[0])
    for other in entries[1:]:
        total = ops.add(total, _sym(other))
    return total.value


def _jvp_from_vjp(proxy: ir.Op, tangents, builder: ir.Builder):
    """Derive the JVP of `proxy` from its registered VJP rule (adjoint trick).

    The VJP rule builds the linear map L: c ↦ Jᵀc. Instantiate it with
    c = the op's (rebuilt) primal results — any well-typed c works, since the
    adjoint of L is independent of the concrete c — then apply L's transpose
    (J) to the tangents with a SECOND reverse sweep over the ops L emitted
    (the rule's emissions). Cotangents only propagate along paths that reach
    a ct slot (a `proxy.results` value): paths to primals are second-order
    terms and are dropped, which also keeps the graph free of dead
    second-order ops (the decomposition's own vjp rules never run).

    Returns a tuple aligned with `proxy.results` of
    `ir.Value | None | ZeroTangent` — the output tangents J·v.
    """
    if all(t is None or isinstance(t, ZeroTangent) for t in tangents):
        return (ZeroTangent(),) * len(proxy.results)

    key = _rule_name(proxy)
    vjp_rule = require_vjp_rule(key)
    before = {op.id for op in builder.current_block.ops}
    l_outputs = vjp_rule(proxy, tuple(proxy.results), proxy.operands)
    emitted = [op for op in builder.current_block.ops if op.id not in before]
    if not isinstance(l_outputs, (tuple, list)):
        raise TransformError(
            f"vjp rule for op '{key}' did not return a tuple of cotangents, "
            f"got {type(l_outputs).__name__}"
        )
    if len(l_outputs) != len(proxy.operands):
        raise TransformError(
            f"vjp rule for op '{key}' returned {len(l_outputs)} cotangents, "
            f"expected {len(proxy.operands)} (one per operand)"
        )
    for entry in l_outputs:
        if entry is not None and not isinstance(entry, (ir.Value, ZeroTangent)):
            raise TransformError(
                f"vjp rule for op '{key}' returned an invalid cotangent entry "
                f"{entry!r}: entries must be ir.Value | None | ZeroTangent"
            )

    # Second reverse sweep = the adjoint application J·v (transpose of L):
    # seed the tangents on L's outputs, then sweep back. Keep cotangents only
    # on the ct slots and on values downstream of them (the only paths that
    # reach J·v); every other operand's cotangent is a second-order term.
    ct_ids = {result.id for result in proxy.results}
    on_ct_path = set(ct_ids)
    for op in emitted:
        if any(operand.id in on_ct_path for operand in op.operands):
            on_ct_path.update(result.id for result in op.results)

    acc2: Dict[int, list] = {}
    for u, v in zip(l_outputs, tangents):
        if _is_real(u) and _is_real(v):
            acc2.setdefault(u.id, []).append(v)
    for op in reversed(emitted):
        if op.name == "constant":
            continue  # zero operands — nothing to propagate backward
        if not any(acc2.get(result.id) for result in op.results):
            continue  # no seeded result — off the ct path (second-order)
        per_result = tuple(_combine(acc2.get(result.id)) for result in op.results)
        vjp_fn = require_vjp_rule(_rule_name(op))
        in_cts = vjp_fn(op, per_result, op.operands)
        for operand, in_ct in zip(op.operands, in_cts):
            if _is_real(in_ct) and operand.id in on_ct_path:
                acc2.setdefault(operand.id, []).append(in_ct)

    # Output tangents: accumulated cotangents on the ct slots (the primal
    # results); missing -> structurally zero.
    out_tangents = []
    for result in proxy.results:
        entries = acc2.get(result.id)
        out_tangents.append(_combine(entries) if entries else ZeroTangent())
    return tuple(out_tangents)
