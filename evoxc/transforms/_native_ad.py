"""Native delegation for `grad` / `jvp` / `vjp`.

The Rust engine (`evoxc._evoxc_native`, reached through `evoxc._native`) is the
single authority for the AD sweep and every builtin derivative rule. Python
keeps only frontend validation/normalization (argnum, tangent/cotangent and
shape checking — in `grad.py`/`jvp.py`/`vjp.py`) and the `Graph`
reconstruction here: the decoded module's block arguments and `return`
operands are wrapped back into the documented input/output trees.

The wire contract (`native/CONTEXT.md` "Transforms"):

* ``grad`` — config ``{"argnums": [int, ...]}`` over flattened tensor inputs;
  result inputs are the original inputs, result outputs one gradient per
  selected index (selector order).
* ``jvp`` — config ``{"tangent_inputs": [int, ...]}`` (flattened input indices
  that get an explicit tangent block argument); result inputs are the original
  inputs followed by those tangents, outputs the primal outputs followed by one
  tangent per tensor output.
* ``vjp`` — config ``{"cotangent_inputs": [int, ...]}`` (flattened OUTPUT
  indices); the others must be rank-0 and are seeded in-graph with a scalar
  one; outputs the primal outputs followed by one cotangent per tensor input.

NATIVE ORDERING WORKAROUND — multi-tangent ``jvp``/``vjp``. The native sweep
currently assigns the TYPE of each explicit tangent/cotangent block argument in
an unordered map's iteration order rather than in selector order, so a request
with two or more shape-distinct extras intermittently produces a mistyped graph
(a run-time shape error, or a silently wrong result when the types coincide).
`_selects_in_order` verifies the decoded block-argument types against the
selector and the call is retried a bounded number of times; only when every
attempt is misordered does `sweep_graph` take the documented Python fallback.
The native fix is to build the extra arguments by iterating the selector list
in order (not the map's keys), which removes both this retry and the retry cap.

TEMPORARY FALLBACK — extension-ruled AD. The native derivative request carries
only tangent FLAGS / a cotangent TYPE, not the tangent/cotangent SSA values, so
an op whose rule lives in Python (`block:*`, `external:*`, and the sparse IR
category — `sparse_*`, `sparse_dot_dense`, `dense_dot_sparse`) cannot be served
by the extension bridge. Such graphs route to the Python sweep in
`autodiff.py` (`forward_sweep` / `reverse_sweep`), the documented fallback. The
one-line native change that removes the exception: add
``tangent_values``/``cotangent_values`` to `ExtensionRequest` and append those
value ids to ``extension_arguments`` after operands + results — then bind the
optional trailing block args the handler already accepts and this fallback can
be deleted.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from evoxc import core, ir
from evoxc.trace import Graph, StaticValue
from evoxc.transforms import _native_bridge
from evoxc.transforms.autodiff import forward_sweep, jvp_rules, reverse_sweep, vjp_rules
from evoxc.transforms.batching import _rule_key

#: Ops whose derivative rules live in Python (`block:<name>` / `external:<name>`
#: keys, or the `sparse_*` / sparse-category op names — the sparse module also
#: registers `sparse_dot_dense` / `dense_dot_sparse`, which carry the `sparse`
#: IR category rather than the name prefix).
_EXTENSION_OPS = frozenset({"block_call", "external_call"})
_EXTENSION_SPARSE_CATEGORY = "sparse"

#: Attempts allowed to absorb the native multi-extra ordering bug (see the
#: module docstring); each call is an independent draw, so a mismatch that
#: survives this many is treated as permanent and falls back to Python.
_ORDER_ATTEMPTS = 64


def _needs_python_ad(graph: Graph) -> bool:
    """True iff `graph` needs the documented Python AD fallback.

    That is the case when the entry function contains an extension-territory
    op (`block_call`, `external_call`, or any `sparse`-category op) whose rule
    key has a registered Python derivative rule — the only ops the native extension
    bridge cannot serve today (see the module docstring). Everything else —
    the whole builtin op surface — goes through the native engine.
    """
    registered = set(jvp_rules) | set(vjp_rules)
    if not registered:
        return False
    for op in _entry_block(graph).ops:
        if not _is_extension_op(op):
            continue
        try:
            key = _rule_key(op)
        except (KeyError, AttributeError):  # malformed op: let the engine report
            continue
        if key in registered:
            return True
    return False


def _is_extension_op(op: "ir.Op") -> bool:
    """True for ops whose derivative rules are Python-registered by design."""
    if op.name in _EXTENSION_OPS:
        return True
    try:
        return ir.opdef(op.name).category == _EXTENSION_SPARSE_CATEGORY
    except KeyError:  # unknown op: the engine reports it
        return False


# ---------------------------------------------------------------------------
# Public compositions (called by grad.py / jvp.py / vjp.py)
# ---------------------------------------------------------------------------


def grad_graph(graph: Graph, selected: Sequence[int], single: bool) -> Graph:
    """Native ``grad``: `graph` → a gradient graph for `selected` tensor inputs.

    `selected` must already be validated/normalized by `grad._normalize_argnums`
    (in order); `single` says whether the caller spelled a bare int `argnums`
    (one bare gradient output) or an int sequence (`None` included — a tuple of
    gradients).
    """
    selected = tuple(selected)
    module, transform = _native_bridge.call_native(
        "grad", graph.module, {"argnums": list(selected)}
    )
    _native_bridge._check_transform_meta(module, transform)
    input_ids = _input_ids(transform)
    if len(input_ids) != len(graph.tensor_specs):
        raise core.TransformError(
            "native grad: the transformed function has "
            f"{len(input_ids)} block args, but the source graph has "
            f"{len(graph.tensor_specs)} tensor inputs"
        )
    gradients = _return_operands(module, "grad")
    if len(gradients) != len(selected):
        raise core.TransformError(
            "native grad: the transformed function returns "
            f"{len(gradients)} gradient(s), but {len(selected)} were selected"
        )
    if single:
        output_tree = core.TreeSpec(type=None)
    else:
        output_tree = core.TreeSpec(
            type=tuple,
            children=tuple(core.TreeSpec(type=None) for _ in selected),
        )
    return type(graph)(
        module,
        graph.input_specs,
        graph.tensor_specs,
        output_tree,
        graph.static_values,
        (),
        _input_source_locations(graph, input_ids),
    )


def sweep_graph(graph: Graph, kind: str, flat_specs: Sequence[Any]) -> Graph:
    """Native ``jvp`` / ``vjp``: `graph` → the sweep graph.

    `flat_specs` is the flat extra-spec tuple aligned with the flattened tensor
    inputs (jvp tangents) or outputs (vjp cotangents): a `core.TensorSpec`
    entry becomes an explicit extra block arg, `None` is a static zero seed.
    Both trees are built exactly like the Python sweep's (see
    `autodiff._sweep`), so the documented input/output shape is preserved.
    """
    if kind not in ("jvp", "vjp"):
        raise ValueError(f"sweep_graph: unknown AD kind {kind!r}")
    flat_specs = tuple(flat_specs)
    selected = [index for index, spec in enumerate(flat_specs) if spec is not None]
    config_key = "tangent_inputs" if kind == "jvp" else "cotangent_inputs"
    primal_count = len(graph.tensor_specs)
    # A single extra argument has no ordering to get wrong, so no retry (and a
    # native inference error there is genuine, surfaced verbatim).
    retryable = len(selected) > 1
    module = None
    last_error = None
    for _attempt in range(_ORDER_ATTEMPTS if retryable else 1):
        try:
            candidate, transform = _native_bridge.call_native(
                kind, graph.module, {config_key: selected}
            )
            _native_bridge._check_transform_meta(candidate, transform)
        except (
            core.ShapeError,
            core.DTypeError,
            core.VerificationError,
            core.BackendError,
        ) as exc:
            # A misordered build infers its ops from the wrong extra type and
            # usually fails right here (e.g. `dot` contracting dims).
            last_error = exc
            continue
        if not retryable or _selects_in_order(
            candidate, primal_count, selected, flat_specs
        ):
            module = candidate
            break
    if module is None:
        if last_error is not None and not retryable:
            raise last_error
        # Every attempt was misordered: the documented Python sweep is the
        # semantically identical last resort (see the module docstring).
        if kind == "jvp":
            return forward_sweep(graph, flat_specs)
        return reverse_sweep(graph, flat_specs)

    input_ids = _input_ids(transform)

    tensor_specs = tuple(graph.tensor_specs) + tuple(
        spec for spec in flat_specs if spec is not None
    )

    # Input tree: (original input tree, flat extra tuple tree); `None` entries
    # become recorded `None` static leaves (exactly `autodiff._sweep`).
    _, extra_input_tree = core.flatten(flat_specs)
    input_tree = core.TreeSpec(
        type=tuple, children=(graph.input_specs, extra_input_tree)
    )
    static_values = list(graph.static_values)
    base = graph.input_specs.num_leaves
    for index, spec in enumerate(flat_specs):
        if spec is None:
            static_values.append(
                StaticValue(
                    index=base + index,
                    path=(1, index),
                    value=None,
                    kind=type(None).__qualname__,
                )
            )

    # Output tree: (original output tree, flat extra outputs tree).
    primal_outputs = _return_operands(graph.module, kind)
    outputs = _return_operands(module, kind)
    extra_values = outputs[len(primal_outputs):]
    expected_extra = len(primal_outputs) if kind == "jvp" else primal_count
    if len(extra_values) != expected_extra:
        raise core.TransformError(
            f"native {kind}: the transformed function returns "
            f"{len(extra_values)} extra output(s), expected {expected_extra}"
        )
    _, extra_output_tree = core.flatten(
        tuple(_symbolic(value) for value in extra_values)
    )
    output_tree = core.TreeSpec(
        type=tuple, children=(graph.output_tree, extra_output_tree)
    )

    return type(graph)(
        module,
        input_tree,
        tensor_specs,
        output_tree,
        tuple(static_values),
        graph.output_static_values,
        _input_source_locations(graph, input_ids[:primal_count]),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry_block(graph: Graph) -> "ir.Block":
    return graph.module.main.entry_block


def _selects_in_order(
    module: "ir.Module",
    primal_count: int,
    selected: Sequence[int],
    flat_specs: Sequence[Any],
) -> bool:
    """True iff each extra block argument's declared type is its selector's.

    Verifies the native ordering workaround (module docstring): the explicit
    tangent/cotangent block arguments must be typed in SELECTOR order, so
    position ``k`` must carry the dtype/shape of ``flat_specs[selected[k]]``.
    Entries are aligned on rank and dtype; a `None` (runtime-dynamic) dim on
    either side is unchecked, matching `TensorSpec` semantics.
    """
    arguments = module.main.entry_block.arguments
    if len(arguments) != primal_count + len(selected):
        return False
    for position, index in enumerate(selected):
        expected = flat_specs[index]
        actual = arguments[primal_count + position].type
        if expected is None or expected.dtype != actual.dtype:
            return False
        if len(expected.shape) != len(actual.shape):
            return False
        if not all(
            _same_dim(want, got) for want, got in zip(expected.shape, actual.shape)
        ):
            return False
    return True


def _same_dim(expected: Any, actual: Any) -> bool:
    """Structural dim comparison; a `None` (dynamic) dim matches anything."""
    if expected is None or actual is None:
        return True
    try:
        return bool(expected == actual)
    except Exception:  # pragma: no cover - uncomparable dims are a mismatch
        return False


def _return_operands(module: "ir.Module", kind: str) -> Tuple["ir.Value", ...]:
    terminator = module.main.entry_block.terminator
    if terminator is None:
        raise core.ShapeError(
            f"native {kind}: the transformed function has no `return` terminator"
        )
    return tuple(terminator.operands)


def _input_ids(transform: Mapping[str, Any]) -> List[int]:
    """The transformed function's block-argument value ids (metadata, verbatim)."""
    return [int(item) for item in (transform.get("input_ids") or ())]


def _input_source_locations(
    graph: Graph, input_ids: Sequence[int]
) -> Dict[int, Any]:
    """Carry the source inputs' locations onto the rebuilt block arguments."""
    locations: Dict[int, Any] = {}
    source_block = _entry_block(graph)
    for old_argument, new_id in zip(source_block.arguments, input_ids):
        location = graph.source_locations.get(old_argument.id)
        if location is not None:
            locations[new_id] = location
    return locations


def _symbolic(value: "ir.Value") -> "core.SymbolicTensor":
    """Wrap an `ir.Value` as a `core.SymbolicTensor` (dtype/shape from its type)."""
    return core.SymbolicTensor(
        value=value, dtype=value.type.dtype, shape=value.type.shape
    )
