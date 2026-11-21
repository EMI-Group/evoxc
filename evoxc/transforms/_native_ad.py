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

Derivative rules that live in Python (`block:*`, `external:*`, the sparse
ops) reach the engine through the extension-rule bridge
(`evoxc/transforms/_native_rules.py`): the engine hands the handler the dual
SSA values as well as the tangent flags / cotangent type, inlines the returned
subgraph and uses every output id as a real dual contribution. There is no
Python AD fallback and no retry: each call reaches native exactly once and a
native error (a missing rule, a shape/dtype inference failure) surfaces
verbatim.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from evoxc import core, ir
from evoxc.trace import Graph, StaticValue
from evoxc.transforms import _native_bridge


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
    Both trees are built exactly like the documented sweep shape, so the
    input/output structure of the result graph is unchanged.
    """
    if kind not in ("jvp", "vjp"):
        raise ValueError(f"sweep_graph: unknown AD kind {kind!r}")
    flat_specs = tuple(flat_specs)
    selected = [index for index, spec in enumerate(flat_specs) if spec is not None]
    config_key = "tangent_inputs" if kind == "jvp" else "cotangent_inputs"
    primal_count = len(graph.tensor_specs)
    module, transform = _native_bridge.call_native(
        kind, graph.module, {config_key: selected}
    )
    _native_bridge._check_transform_meta(module, transform)
    input_ids = _input_ids(transform)

    tensor_specs = tuple(graph.tensor_specs) + tuple(
        spec for spec in flat_specs if spec is not None
    )

    # Input tree: (original input tree, flat extra tuple tree); `None` entries
    # become recorded `None` static leaves (exactly the sweep's convention).
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
    source_block = graph.module.main.entry_block
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
