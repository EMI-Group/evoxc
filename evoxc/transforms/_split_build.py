"""The CONSTRUCTION half of the `split_by_inputs` machinery: the two graphs.

`split_by_inputs` is the FIRST structural (non-numeric) pass in this package:
it slices the graph DATA STRUCTURE into a `prepare` half (the fixed inputs'
contribution) and an `apply` half (the varying inputs on top of a prepared
boundary). It carries NO numerical authority — no derivatives, no batching,
no computation — so it lives in Python rather than in the Rust engine (see
`./CONTEXT.md`, "split_by_inputs").

THIS MODULE AND `_split_deps.py` ARE ONE PASS, NOT TWO INDEPENDENT PASSES.
`_split_deps.py` takes the source graph apart (gates, dependence masks, needed
set, boundary selection, the validated partition — its steps 1-4 and 4b);
everything here turns those read-only facts into NEW objects: pruned input
trees, cloned single-function modules and the two `Graph`s, then the
`SplitPlan`. The two halves are re-exported together by `_split_core.py`, and
the step numbers continue from `_split_deps.py`:

5. `prune_input_tree` / `rebuild_static_values` — the pruned input trees and
   the re-indexed static-value records for each half.
6. `clone_module` — one fresh single-function module per half, cloned op by
   op with a value-id remap (types carried VERBATIM so `ir.verify` agrees
   with the carried result types).
7. `build_prepare` / `build_apply` — the two `Graph`s.
8. `_prepare_ops` / `_apply_ops` / `_unused_varying_leaves` — the per-half
   SOURCE op lists `partition` (`_split_deps.py`) records.
9. `split_graph` — the whole composition returning a `SplitPlan`.

Never mutates the source graph, its module, or any source `ValueType`/`Dim`:
values are cloned and types are shared by reference.
"""

from __future__ import annotations

import collections
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from evoxc import core
from evoxc import ir
from evoxc.trace._tree import _iter_leaf_paths
from evoxc.trace.graph import Graph, StaticValue
from evoxc.trace.trace import _SymbolicLeaf, _TensorSpecLeaf, _flatten_trace

from evoxc.transforms._split_deps import (
    _CONSTANT_OP,
    _loc,
    _op_mask,
    _registered_base,
    _value_types,
    partition,
)
from evoxc.transforms._split_types import BoundaryValue, SplitPlan


# ---------------------------------------------------------------------------
# 5. pruned input trees + static records
# ---------------------------------------------------------------------------


def _is_fixed_arity_node(node: "core.TreeSpec") -> bool:
    """True for container nodes whose child ARITY cannot change.

    A namedtuple, a dataclass and a registered pytree node are rebuilt from a
    fixed constructor signature, so dropping SOME of their children would
    produce an object the run-time `core.unflatten` cannot rebuild. Plain
    tuple/list/dict nodes have no such constraint.
    """
    if _registered_base(node.type) is not None:
        return True
    node_type = node.type
    if not isinstance(node_type, type):
        return False
    if issubclass(node_type, tuple) and hasattr(node_type, "_fields"):
        return True
    import dataclasses

    return dataclasses.is_dataclass(node_type)


def _pruned_node_data(node: "core.TreeSpec", kept_keys: Sequence[Any]) -> Any:
    """The `node_data` for a PARTIALLY pruned container node.

    `dict` subclasses record their (sorted) keys — ``(default_factory, keys)``
    for a `defaultdict`, whose factory is preserved. Every other container
    keeps its `node_data` (``None`` for tuple/list nodes; fixed-arity nodes
    never reach here — they raise first).
    """
    node_type = node.type
    if isinstance(node_type, type) and issubclass(node_type, dict):
        if issubclass(node_type, collections.defaultdict):
            factory = None if node.node_data is None else node.node_data[0]
            return (factory, list(kept_keys))
        return list(kept_keys)
    return node.node_data


def prune_input_tree(
    tree_spec: "core.TreeSpec", keep_tree_leaf
) -> Tuple[Optional["core.TreeSpec"], Tuple[int, ...]]:
    """Prune a graph input tree to the tensor leaves `keep_tree_leaf` accepts.

    `keep_tree_leaf(leaf_spec) -> bool` is consulted for TENSOR leaves only
    (leaf specs whose type is the trace `_TensorSpecLeaf` marker), once per
    tensor leaf in pre-order, so it sees tensor-input positions in
    `tensor_specs` order. STATIC leaves are always kept (a static value is
    part of the specialization, never a boundary input) and their leaf
    `TreeSpec`s are reused verbatim.

    Returns ``(new_spec, kept_tree_leaf_positions)``: `new_spec` is the pruned
    tree (`None` when nothing survives) and the positions are the ORIGINAL
    tree's flat leaf positions of the kept leaves, in the new tree's
    pre-order. A container is rebuilt whenever ANY child was rebuilt — never
    reused verbatim just because its direct child COUNT is unchanged, so a
    per-leaf prune INSIDE a plain tuple/list/dict really drops the leaf. A
    container whose subtree keeps no leaf is dropped; a fixed-arity node
    (namedtuple / dataclass / registered pytree node) that would keep only
    SOME of its children raises `core.TransformError` naming the pytree path
    — its arity is fixed, so suggest a plain tuple/dict.
    """
    position = [0]

    def walk(node: "core.TreeSpec", prefix: Tuple[Any, ...]):
        if not node.children:
            if node.num_leaves == 0:
                return None, ()  # empty container: no leaves to keep
            index = position[0]
            position[0] += 1
            if node.type is _TensorSpecLeaf:
                return (node, (index,)) if keep_tree_leaf(node) else (None, ())
            return node, (index,)  # static leaf: always kept
        if node.num_leaves == 0:
            return None, ()  # a container of empty containers
        kept: List[Tuple[Any, "core.TreeSpec"]] = []
        positions: Tuple[int, ...] = ()
        rebuilt = False
        for index, child in enumerate(node.children):
            key = (
                node.node_data[index]
                if isinstance(node.type, type) and issubclass(node.type, dict)
                else index
            )
            new_child, child_positions = walk(child, prefix + (key,))
            if new_child is not None:
                kept.append((key, new_child))
                positions += child_positions
                # A DESCENDANT may have been pruned even when the direct child
                # count is unchanged, so only an unchanged child OBJECT proves
                # this subtree is untouched (see the docstring's contract).
                if new_child is not child:
                    rebuilt = True
        if not kept:
            return None, ()
        if not rebuilt and len(kept) == len(node.children):
            return node, positions  # nothing pruned: reuse the node verbatim
        if _is_fixed_arity_node(node):
            raise core.TransformError(
                f"split_by_inputs: cannot prune the fixed-arity pytree node "
                f"{getattr(node.type, '__name__', node.type)!r} at pytree path "
                f"{core.format_path(prefix)} — its constructor arity is fixed, "
                f"so dropping only some of its children cannot be rebuilt at "
                f"run time. Use a plain tuple/list/dict for splittable "
                f"containers."
            )
        new_node = core.TreeSpec(
            type=node.type,
            children=tuple(child for _, child in kept),
            context=node.context,
            node_data=_pruned_node_data(node, [key for key, _ in kept]),
        )
        return new_node, positions

    return walk(tree_spec, ())


def rebuild_static_values(
    source_graph: Graph,
    kept_tree_positions: Sequence[int],
    new_tree: Optional["core.TreeSpec"],
    *,
    extra_leaves: int = 0,
) -> Tuple[StaticValue, ...]:
    """Rebuild `StaticValue` records for a pruned tree.

    The kept source records (matched by their ORIGINAL flat leaf index) keep
    their `value` and `kind` verbatim and gain their NEW flat index and pytree
    path in `new_tree` (paths from `_iter_leaf_paths`). A mismatch between the
    kept position count and the new tree's leaf count means the prune helper
    and this rebuild disagree — an internal bug, reported as
    `core.VerificationError`.

    `extra_leaves` says how many leaves `new_tree` carries BEYOND the kept
    ones, all of them AFTER them (the `apply` signature appends the boundary
    tuple to the pruned varying tree, so the kept leaves stay a prefix and
    their new indices and paths are unaffected).
    """
    static_by_position = {record.index: record for record in source_graph.static_values}
    paths = list(_iter_leaf_paths(new_tree)) if new_tree is not None else []
    expected = len(kept_tree_positions) + int(extra_leaves)
    if len(paths) != expected:
        raise core.VerificationError(
            f"split_by_inputs: the pruned input tree has {len(paths)} leaves "
            f"but {len(kept_tree_positions)} source leaf positions were kept "
            f"(+{int(extra_leaves)} appended leaves) — prune/rebuild "
            f"bookkeeping disagree"
        )
    records: List[StaticValue] = []
    for new_index, old_position in enumerate(kept_tree_positions):
        source_record = static_by_position.get(old_position)
        if source_record is None:
            continue
        records.append(
            StaticValue(
                index=new_index,
                path=paths[new_index],
                value=source_record.value,
                kind=source_record.kind,
            )
        )
    return tuple(records)


# ---------------------------------------------------------------------------
# 6. cloning
# ---------------------------------------------------------------------------


def _clone_region(
    builder: "ir.Builder", region: "ir.Region", remap: Dict[int, "ir.Value"]
) -> "ir.Region":
    """Raise: nested regions can never reach the clone loop in v1.

    `gates` refuses every region-bearing op, so a region here is an internal
    inconsistency. Kept total (it raises rather than mis-cloning) so a future
    caller cannot silently copy a region outside the gate's knowledge.
    """
    del builder, region, remap
    raise core.TransformError(
        "split_by_inputs: cannot clone a nested region (control-flow regions "
        "are not splittable in v1)"
    )


def _clone_op(
    builder: "ir.Builder", op: "ir.Op", remap: Dict[int, "ir.Value"]
) -> "ir.Op":
    """Clone one op into the builder, remapping its operands and results.

    Types are carried VERBATIM (`tuple(value.type for value in op.results)`)
    so `ir.verify`, which re-derives result types, agrees exactly; attributes
    are shallow-copied (an ndarray constant payload stays shared by
    reference).
    """
    regions = tuple(_clone_region(builder, region, remap) for region in op.regions)
    try:
        operands = tuple(remap[value.id] for value in op.operands)
    except KeyError:
        raise core.TransformError(
            f"split_by_inputs: cannot clone op '{op.name}'{_loc(op)}: an "
            f"operand is neither a cloned input nor an already-cloned result"
        ) from None
    new_op = builder.create(
        op.name,
        operands=operands,
        attributes=dict(op.attributes),
        result_types=tuple(value.type for value in op.results),
        location=op.location,
        regions=regions,
    )
    for new_value, old_value in zip(new_op.results, op.results):
        remap[old_value.id] = new_value
    return new_op


def clone_module(
    op_list: Sequence["ir.Op"],
    input_types: Sequence["ir.ValueType"],
    seed_ids: Sequence[int],
    output_ids: Sequence[int],
) -> Tuple["ir.Module", Dict[int, "ir.Value"]]:
    """Build one fresh single-function module from `op_list`.

    `input_types` types the new entry-block arguments (in order) and
    `seed_ids` names, position by position, the SOURCE value id each new
    argument stands for (a source block argument, or a boundary value remapped
    to a prepared argument). The returned `remap` covers every cloned value.
    `output_ids` are the SOURCE ids of the `return` operands.

    Returns ``(module, remap)``; the module's function/module are both named
    ``"main"`` (the v1 `Graph` convention).
    """
    builder = ir.Builder()
    module = builder.build_module(name="main")
    function = builder.build_function(
        name="main", input_types=tuple(input_types)
    )
    block = function.entry_block
    arguments = block.arguments
    if len(arguments) != len(seed_ids):
        raise core.TransformError(
            f"split_by_inputs: {len(arguments)} new block arguments but "
            f"{len(seed_ids)} seed value ids"
        )
    remap: Dict[int, "ir.Value"] = {}
    for source_id, argument in zip(seed_ids, arguments):
        if source_id in remap:
            raise core.TransformError(
                f"split_by_inputs: source value %{source_id} is seeded twice "
                f"into the cloned signature"
            )
        remap[source_id] = argument
    for op in op_list:
        _clone_op(builder, op, remap)
    try:
        returns = tuple(remap[value_id] for value_id in output_ids)
    except KeyError as exc:
        raise core.TransformError(
            f"split_by_inputs: output value %{exc.args[0]} is not produced by "
            f"the cloned program"
        ) from None
    builder.set_terminator(block, "return", operands=returns)
    return module, remap


# ---------------------------------------------------------------------------
# 7. the two graphs
# ---------------------------------------------------------------------------


def _keep_predicate(positions: Sequence[int]):
    """A tensor-leaf predicate for `prune_input_tree` matching `positions`.

    The predicate is consulted once per tensor leaf in pre-order, so a
    counter maps tensor leaves onto `tensor_specs` positions (== block-argument
    positions).
    """
    wanted = set(int(position) for position in positions)
    seen = [-1]

    def keep(_leaf_spec: "core.TreeSpec") -> bool:
        seen[0] += 1
        return seen[0] in wanted

    return keep


def _symbolic_tree(module: "ir.Module") -> "core.TreeSpec":
    """The flat-`tuple` output tree for a cloned module's `return` operands."""
    _, tree = _flatten_trace(
        tuple(
            _SymbolicLeaf(
                core.SymbolicTensor(
                    value=value, dtype=value.type.dtype, shape=value.type.shape
                )
            )
            for value in module.main.entry_block.terminator.operands
        )
    )
    return tree


def build_prepare(
    source: Graph,
    prepare_ops: Sequence["ir.Op"],
    boundary: Sequence[BoundaryValue],
    fixed_positions: Sequence[int],
) -> Graph:
    """Build the `prepare` graph: the FIXED inputs → the boundary tuple.

    Signature: the fixed `core.TensorSpec` OBJECTS (source order, so their
    symbolic `Dim`s survive by identity); the input tree is the source tree
    pruned to the fixed tensor leaves (statics preserved and re-indexed); the
    output tree is a FLAT `tuple` root with one `_SymbolicLeaf` per boundary
    value, so ``evoxc.run(prepare, *fixed_args)`` returns a flat tuple of
    prepared tensors.
    """
    entry_block = source.module.main.entry_block
    arguments = entry_block.arguments
    ordered = sorted(int(position) for position in fixed_positions)

    pruned_tree, kept_positions = prune_input_tree(
        source.input_specs, _keep_predicate(ordered)
    )
    if pruned_tree is None:
        raise core.TransformError(
            "split_by_inputs: the pruned `prepare` input tree has no leaves"
        )
    input_types = tuple(arguments[position].type for position in ordered)
    seed_ids = tuple(arguments[position].id for position in ordered)
    output_ids = tuple(value.source_value_id for value in boundary)
    module, remap = clone_module(prepare_ops, input_types, seed_ids, output_ids)

    source_locations = {}
    for position in ordered:
        location = source.source_locations.get(arguments[position].id)
        if location is not None:
            source_locations[remap[arguments[position].id].id] = location

    return Graph(
        module,
        pruned_tree,
        tuple(source.tensor_specs[position] for position in ordered),
        _symbolic_tree(module),
        rebuild_static_values(source, kept_positions, pruned_tree),
        (),
        source_locations,
    )


def build_apply(
    source: Graph,
    apply_ops: Sequence["ir.Op"],
    boundary: Sequence[BoundaryValue],
    varying_positions: Sequence[int],
) -> Graph:
    """Build the `apply` graph: varying inputs + the prepared tuple → outputs.

    Signature: the VARYING source `core.TensorSpec` objects first, then the
    boundary specs — so the call shape is
    ``evoxc.run(apply, varying_tree, prepared_tuple)``. The input tree is a
    synthetic 2-child `tuple` root: element 0 is the source input tree pruned
    to the varying tensor leaves (statics preserved and re-indexed), element 1
    the flat tuple of boundary specs. The output tree (and the output static
    records) are the SOURCE graph's, verbatim.
    """
    entry_block = source.module.main.entry_block
    arguments = entry_block.arguments
    ordered = sorted(int(position) for position in varying_positions)

    varying_tree, kept_positions = prune_input_tree(
        source.input_specs, _keep_predicate(ordered)
    )
    if varying_tree is None:
        raise core.TransformError(
            "split_by_inputs: the pruned `apply` varying input tree has no "
            "leaves"
        )
    boundary_specs = tuple(value.spec for value in boundary)
    _, boundary_tree = _flatten_trace(boundary_specs)
    synthetic_tree = core.TreeSpec(
        type=tuple, children=(varying_tree, boundary_tree)
    )

    input_types = tuple(arguments[position].type for position in ordered) + (
        _boundary_value_types(entry_block, boundary)
    )
    seed_ids = tuple(arguments[position].id for position in ordered) + tuple(
        value.source_value_id for value in boundary
    )
    output_ids = tuple(
        value.id for value in entry_block.terminator.operands
    )
    module, remap = clone_module(apply_ops, input_types, seed_ids, output_ids)

    source_locations = {}
    for position in ordered:
        location = source.source_locations.get(arguments[position].id)
        if location is not None:
            source_locations[remap[arguments[position].id].id] = location
    for value in boundary:
        if value.location is None:
            continue
        source_locations[remap[value.source_value_id].id] = value.location

    tensor_specs = tuple(
        source.tensor_specs[position] for position in ordered
    ) + boundary_specs
    return Graph(
        module,
        synthetic_tree,
        tensor_specs,
        source.output_tree,
        rebuild_static_values(
            source, kept_positions, synthetic_tree, extra_leaves=len(boundary_specs)
        ),
        source.output_static_values,
        source_locations,
    )


def _boundary_value_types(
    entry_block: "ir.Block", boundary: Sequence[BoundaryValue]
) -> Tuple["ir.ValueType", ...]:
    """The SOURCE `ir.ValueType` of each boundary value (verbatim)."""
    types_by_id = _value_types(entry_block)
    try:
        return tuple(
            types_by_id[value.source_value_id] for value in boundary
        )
    except KeyError as exc:
        raise core.TransformError(
            f"split_by_inputs: boundary value %{exc.args[0]} does not exist in "
            f"the source graph"
        ) from None


# ---------------------------------------------------------------------------
# 8. op partition
# ---------------------------------------------------------------------------


def _prepare_ops(
    needed_ops: Sequence["ir.Op"],
    masks: Dict[int, int],
    boundary: Sequence[BoundaryValue],
) -> Tuple["ir.Op", ...]:
    """The `prepare` op list: fixed-only producers of the boundary, transitively.

    Backward closure from every boundary value that is an op result, through
    FIXED-only (`mask == 0`) needed ops. `constant` ops are included when
    they feed the closure (a fixed-only constant used only by `prepare` is
    cheaper here than shipping it across). Entry-block order is preserved.
    """
    producers = {
        result.id: op for op in needed_ops for result in op.results
    }
    selected: Set[int] = set()
    worklist = [value.source_value_id for value in boundary]
    while worklist:
        value_id = worklist.pop()
        producer = producers.get(value_id)
        if producer is None:
            continue  # a pass-through input: nothing to compute
        if id(producer) in selected:
            continue
        if _op_mask(producer, masks) != 0:
            continue  # mixed: `apply` computes it
        selected.add(id(producer))
        for operand in producer.operands:
            worklist.append(operand.id)
    return tuple(op for op in needed_ops if id(op) in selected)


def _apply_ops(
    needed_ops: Sequence["ir.Op"],
    masks: Dict[int, int],
    boundary: Sequence[BoundaryValue],
    output_ids: Sequence[int],
) -> Tuple["ir.Op", ...]:
    """The `apply` op list: every mixed needed op plus the required constants.

    A mixed (`mask != 0`) needed op always belongs to `apply`. A needed
    `constant` op is pulled in when one of its results is a `return` operand
    or an operand of a selected op — constants are never boundary values, so
    cloning the (static) payload is how `apply` sees them. Constants have no
    operands, so one fixpoint pass suffices. Entry-block order is preserved.
    """
    boundary_ids = {value.source_value_id for value in boundary}
    by_id = {id(op): op for op in needed_ops}
    producers = {result.id: op for op in needed_ops for result in op.results}
    selected: Set[int] = {
        id(op) for op in needed_ops if _op_mask(op, masks) != 0
    }

    def select_constant(value_id: int) -> bool:
        """Add the `constant` producer of `value_id` to `apply`; True if added."""
        producer = producers.get(value_id)
        if (
            producer is None
            or producer.name != _CONSTANT_OP
            or id(producer) in selected
        ):
            return False
        selected.add(id(producer))
        return True

    changed = True
    while changed:
        changed = False
        for op_id in list(selected):
            for operand in by_id[op_id].operands:
                if operand.id in boundary_ids:
                    continue
                producer = operand.defining_op
                if producer is None:
                    continue  # a varying input: seeded directly
                if id(producer) in selected:
                    continue
                if select_constant(operand.id):
                    changed = True
                    continue
                raise core.TransformError(
                    f"split_by_inputs: op '{by_id[op_id].name}' consumes a "
                    f"fixed-only value produced by '{producer.name}'"
                    f"{_loc(producer)} that is not a boundary value — the "
                    f"boundary selection is inconsistent"
                )
        for value_id in output_ids:
            if value_id in boundary_ids:
                continue
            if select_constant(value_id):
                changed = True
    return tuple(op for op in needed_ops if id(op) in selected)


def _unused_varying_leaves(
    needed_ops: Sequence["ir.Op"],
    masks: Dict[int, int],
    varying_positions: Sequence[int],
) -> Tuple[int, ...]:
    """Varying tensor positions whose bit appears in no needed op's mask.

    A varying input consumed only by dead ops is KEPT in `apply`'s signature
    (signature stability) but does not influence the outputs; reporting it
    makes that visible.
    """
    used = 0
    for op in needed_ops:
        used |= _op_mask(op, masks)
    return tuple(
        position for position in varying_positions if not (used >> position) & 1
    )


# ---------------------------------------------------------------------------
# 9. the composition
# ---------------------------------------------------------------------------


def split_graph(graph: Graph, fixed_positions: Sequence[int]) -> SplitPlan:
    """Split a traced `Graph` into a `prepare` / `apply` pair.

    `fixed_positions` are flat TENSOR-input indices (into `graph.tensor_specs`)
    that stay in `prepare`; every other tensor input moves to `apply`. Both
    halves must be non-trivial (`core.TransformError` otherwise). The source
    graph is never mutated.

    `partition` (the validated input + op split) comes from `_split_deps.py`:
    the two modules are the read and construction halves of ONE pass.
    """

    part = partition(graph, fixed_positions)
    prepare = build_prepare(graph, part.prepare_ops, part.boundary, part.fixed)
    apply = build_apply(graph, part.apply_ops, part.boundary, part.varying)
    return SplitPlan(
        graph,
        prepare,
        apply,
        part.fixed,
        part.varying,
        part.boundary,
        unused_varying_leaves=part.unused_varying_leaves,
        prepare_op_ids=frozenset(id(op) for op in part.prepare_ops),
        apply_op_ids=frozenset(id(op) for op in part.apply_ops),
    )
