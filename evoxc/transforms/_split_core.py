"""Machinery for the `split_by_inputs` structural graph split.

`split_by_inputs` is the FIRST structural (non-numeric) pass in this package:
it slices the graph DATA STRUCTURE into a `prepare` half (the fixed inputs'
contribution) and an `apply` half (the varying inputs on top of a prepared
boundary). It carries NO numerical authority — no derivatives, no batching,
no computation — so it lives in Python here rather than in the Rust engine
(see `./CONTEXT.md`, "split_by_inputs").

The pipeline implemented here, all of it pure graph manipulation:

1. `gates` — refuse what v1 cannot split (multi-function modules, nested
   control-flow regions, effectful ops: running an effectful op ONCE in
   `prepare` is not the same as it not depending on the varying inputs).
2. `dependence_masks` — one Python-int bitset per SSA value recording WHICH
   varying tensor inputs it depends on (`0` = fixed-only); straight-line SSA
   makes this a single forward pass.
3. `needed_set` — backward reachability from the `return` operands.
4. `select_boundary` — the values that cross from `prepare` to `apply`, in a
   deterministic order (see the function's rationale comments).
5. `prune_input_tree` / `rebuild_static_values` — the pruned input trees and
   the re-indexed static-value records for each half.
6. `clone_module` — one fresh single-function module per half, cloned op by
   op with a value-id remap (types carried VERBATIM so `ir.verify` agrees
   with the carried result types).
7. `build_prepare` / `build_apply` — the two `Graph`s, and `split_graph` —
   the whole composition returning a `SplitPlan`.

Never mutates the source graph, its module, or any source `ValueType`/`Dim`:
values are cloned and types are shared by reference.
"""

from __future__ import annotations

import collections
import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from evoxc import core
from evoxc import ir
from evoxc.core import tree as _core_tree
from evoxc.trace._tree import _iter_leaf_paths
from evoxc.trace.graph import Graph, StaticValue
from evoxc.trace.trace import _SymbolicLeaf, _TensorSpecLeaf, _flatten_trace

from evoxc.transforms._split_types import BoundaryValue, SplitPlan

__all__ = [
    "gates",
    "dependence_masks",
    "needed_set",
    "select_boundary",
    "prune_input_tree",
    "rebuild_static_values",
    "clone_module",
    "build_prepare",
    "build_apply",
    "split_graph",
]

#: Effect kinds a splittable op may declare. `read` (block_call / rank /
#: world_size) is allowed: it observes metadata, not tensor data, so running
#: it in `prepare` cannot make the boundary depend on the varying inputs.
_ALLOWED_EFFECTS = frozenset({ir.EFFECT_PURE, ir.EFFECT_READ})

#: The op name whose results are NEVER a boundary value (see
#: `select_boundary`): constants are cloned into `apply` instead.
_CONSTANT_OP = "constant"

#: The registered-custom-node table of core's pytrees (mutated in place by
#: `register_pytree_node`, so this alias stays live) — the same MRO walk
#: convention as `evoxc/transforms/vectorize.py`.
_PYTREE_NODE_REGISTRY = _core_tree._PYTREE_NODE_REGISTRY


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------


def _loc(op: "ir.Op") -> str:
    """``" at file:line:col"`` for an op with a source location, else ``""``."""
    location = op.location
    return "" if location is None else f" at {location}"


def _entry_block(module: "ir.Module") -> "ir.Block":
    """The entry block of a module that is ALREADY known to hold one
    function (see `gates`)."""
    return module.main.entry_block


def _op_mask(op: "ir.Op", masks: Dict[int, int]) -> int:
    """The dependence mask of an op (its results share one mask)."""
    mask = 0
    for result in op.results:
        mask |= masks.get(result.id, 0)
    return mask


def _op_is_needed(op: "ir.Op", needed: Set[int]) -> bool:
    """True iff any of the op's results is backward-reachable."""
    return any(result.id in needed for result in op.results)


def _value_types(entry_block: "ir.Block") -> Dict[int, "ir.ValueType"]:
    """Map every SSA value id in the block to its `ir.ValueType` (VERBATIM —
    the type objects are carried into the cloned modules by reference)."""
    types: Dict[int, "ir.ValueType"] = {
        argument.id: argument.type for argument in entry_block.arguments
    }
    for op in entry_block.ops:
        for result in op.results:
            types[result.id] = result.type
    return types


def _registered_base(spec_type: Any) -> Optional[type]:
    """The first type in `spec_type`'s MRO registered as a pytree node."""
    if not isinstance(spec_type, type):
        return None
    for base in spec_type.__mro__:
        if base in _PYTREE_NODE_REGISTRY:
            return base
    return None


# ---------------------------------------------------------------------------
# 1. gates
# ---------------------------------------------------------------------------


def _gate_op(op: "ir.Op") -> None:
    """Refuse one op that cannot be split across the boundary.

    Region-bearing control flow (`if`/`while`, i.e. `evoxc.cond` /
    `while_loop` / `scan`) is blocked because the pass is a straight-line
    structural slice in v1. Effectful ops are blocked because splitting them
    across `prepare`/`apply` would change how often (and where) they run —
    and running an effectful op exactly ONCE in `prepare` is NOT the same as
    it not depending on the varying inputs (collectives, `runtime_call`,
    `external_call`, `custom_call` and `inline_stablehlo` all land here).
    """
    if op.regions:
        raise core.TransformError(
            f"split_by_inputs: control-flow regions are not yet splittable — "
            f"op '{op.name}'{_loc(op)} carries {len(op.regions)} nested "
            f"region(s) (cond/while_loop/scan are blocked in v1)"
        )
    effect = op.effect
    if effect in _ALLOWED_EFFECTS:
        return
    raise core.TransformError(
        f"split_by_inputs: op '{op.name}'{_loc(op)} has effect '{effect}' "
        f"and cannot be split into a prepare/apply pair: running an effectful "
        f"op once in `prepare` is not the same as it not depending on the "
        f"varying inputs (only '{ir.EFFECT_PURE}' and '{ir.EFFECT_READ}' ops "
        f"are splittable in v1)"
    )


def _iter_ops(block: "ir.Block"):
    """Yield every op of `block` and, recursively, of its nested regions.

    Straight-line blocks are the only ones that survive `_gate_op`, but the
    walk stays total so a nested op can never escape the effect check (the
    region-bearing op itself is refused first).
    """
    for op in block.ops:
        yield op
        for region in op.regions:
            for inner_block in region.blocks:
                yield from _iter_ops(inner_block)


def gates(module: "ir.Module") -> "ir.Block":
    """Validate that a module can be split; return its entry block.

    Raises `core.TransformError` when the module does not hold exactly ONE
    function with ONE block, when any op carries nested regions, when any op
    is effectful (see `_gate_op`) — in the entry block or in any nested
    `op.regions` — or when the entry block has no `return` terminator.
    Messages name the offending op and its source location.
    """
    if len(module.functions) != 1:
        raise core.TransformError(
            f"split_by_inputs: graph modules with {len(module.functions)} "
            f"functions are not supported in v1 (expected exactly one)"
        )
    function = module.functions[0]
    if len(function.region.blocks) != 1:
        raise core.TransformError(
            f"split_by_inputs: multi-block function regions are not supported "
            f"in v1 ({len(function.region.blocks)} blocks)"
        )
    block = function.entry_block
    for op in _iter_ops(block):
        _gate_op(op)
    if block.terminator is None:
        raise core.TransformError(
            "split_by_inputs: the entry block has no `return` terminator"
        )
    return block


# ---------------------------------------------------------------------------
# 2. dependence masks
# ---------------------------------------------------------------------------


def dependence_masks(
    entry_block: "ir.Block",
    n_tensor_inputs: int,
    varying_positions: Sequence[int],
) -> Tuple[Dict[int, int], Tuple["ir.Op", ...]]:
    """Per-SSA-value dependence bitsets over the VARYING tensor inputs.

    Bit ``k`` of a value's mask is set iff the value depends on the tensor
    input at block-argument position ``k``. Block argument ``i`` starts with
    bit ``i`` set iff that tensor input is varying, else ``0`` (fixed-only);
    an op's mask is the OR of its operands' masks (straight-line SSA, one
    forward pass — every operand is already defined).

    Returns ``(masks, ops)`` where ``ops`` is the ordered NON-TERMINATOR op
    list. Raises `core.TransformError` when the block's argument count does
    not match `n_tensor_inputs` or an operand is not defined before its use.
    """
    varying = set(int(position) for position in varying_positions)
    arguments = entry_block.arguments
    if len(arguments) != n_tensor_inputs:
        raise core.TransformError(
            f"split_by_inputs: the entry block has {len(arguments)} block "
            f"arguments but the graph records {n_tensor_inputs} tensor inputs"
        )
    masks: Dict[int, int] = {}
    for argument in arguments:
        masks[argument.id] = (1 << argument.index) if argument.index in varying else 0
    ops: List["ir.Op"] = []
    for op in entry_block.ops:
        if op.is_terminator:
            continue
        mask = 0
        for operand in op.operands:
            try:
                mask |= masks[operand.id]
            except KeyError:
                raise core.TransformError(
                    f"split_by_inputs: op '{op.name}'{_loc(op)} uses value "
                    f"%{operand.id} before it is defined (the entry block must "
                    f"be straight-line SSA)"
                ) from None
        for result in op.results:
            masks[result.id] = mask
        ops.append(op)
    return masks, tuple(ops)


# ---------------------------------------------------------------------------
# 3. liveness
# ---------------------------------------------------------------------------


def needed_set(entry_block: "ir.Block", masks: Dict[int, int]) -> Set[int]:
    """The set of value ids backward-reachable from the `return` operands.

    A single reverse pass: an op whose result is needed makes all its
    operands needed. Ops whose results are all outside the set are dead and
    are dropped from both halves (they can only be `pure`/`read` — see
    `gates` — so dropping them cannot drop an observable effect).

    ``masks`` is accepted so callers can hand the whole analysis product
    along unchanged; reachability does not consult it.
    """
    del masks  # accepted for call-site symmetry only (see docstring)
    terminator = entry_block.terminator
    if terminator is None:
        raise core.TransformError(
            "split_by_inputs: the entry block has no `return` terminator"
        )
    needed: Set[int] = {value.id for value in terminator.operands}
    ops = [op for op in entry_block.ops if not op.is_terminator]
    for op in reversed(ops):
        if any(result.id in needed for result in op.results):
            for operand in op.operands:
                needed.add(operand.id)
    return needed


# ---------------------------------------------------------------------------
# 4. boundary selection
# ---------------------------------------------------------------------------


def select_boundary(
    source: Graph, masks: Dict[int, int], needed: Set[int]
) -> Tuple[BoundaryValue, ...]:
    """The values that must cross from `prepare` to `apply`, in order.

    A FIXED-only value (``mask == 0``) must be materialized by `prepare` iff
    `apply` still needs its VALUE at run time, i.e. iff it is returned
    directly by the source graph or consumed by a MIXED (``mask != 0``) op.
    A fixed-only value consumed only by fixed-only ops stays inside
    `prepare` — the downstream fixed-only value is the one that crosses.

    Order (deterministic):

    1. fixed-only pass-through INPUT arguments that cross, in block-argument
       order (rationale: an input has no producer, so `prepare` can only
       forward it — and forwarding keeps the source `TensorSpec` object);
    2. results of needed fixed-only NON-`constant` ops that cross, in
       entry-block op order (rationale: `prepare` computes them, so the
       boundary is the computed value, not its operands).

    `constant` results are NEVER bundled: a constant payload is cloned into
    `apply` verbatim (it is static data with no fixed/varying dependence, so
    shipping it through the boundary would only add an output/input pair for
    something `apply` already carries).
    """
    entry_block = source.module.main.entry_block
    terminator = entry_block.terminator
    return_ids = {value.id for value in terminator.operands}

    def crosses(value: "ir.Value") -> bool:
        """True iff `apply` needs this fixed-only value as a boundary tensor."""
        if value.id in return_ids:
            return True
        for use in value.uses:
            owner = use.owner
            if owner.is_terminator:
                continue
            if not _op_is_needed(owner, needed):
                continue
            if _op_mask(owner, masks) != 0:
                return True
        return False

    boundary: List[BoundaryValue] = []
    # Rule 1 — pass-through fixed inputs, in block-argument order.
    for argument in entry_block.arguments:
        if masks[argument.id] != 0:
            continue
        if not crosses(argument):
            continue
        boundary.append(
            BoundaryValue(
                argument.id,
                source.tensor_specs[argument.index],
                defining_op=None,
                location=source.source_locations.get(argument.id),
                is_input=True,
            )
        )
    # Rule 2 — results of needed fixed-only non-constant ops, in op order.
    for op in entry_block.ops:
        if op.is_terminator or not _op_is_needed(op, needed):
            continue
        if op.name == _CONSTANT_OP:
            continue
        if _op_mask(op, masks) != 0:
            continue
        for result in op.results:
            if not crosses(result):
                continue
            boundary.append(
                BoundaryValue(
                    result.id,
                    core.TensorSpec(shape=tuple(result.type.shape), dtype=result.type.dtype),
                    defining_op=op.name,
                    location=op.location,
                    is_input=False,
                )
            )
    return tuple(boundary)


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
    pre-order. A container whose subtree keeps no leaf is dropped; a
    fixed-arity node (namedtuple / dataclass / registered pytree node) that
    would keep only SOME of its children raises `core.TransformError` naming
    the pytree path — its arity is fixed, so suggest a plain tuple/dict.
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
        if not kept:
            return None, ()
        if len(kept) == len(node.children):
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
    """
    if not isinstance(graph, Graph):
        raise TypeError(
            f"split_graph expects a traced evoxc.Graph, got "
            f"{type(graph).__name__}"
        )
    count = len(graph.tensor_specs)
    fixed = sorted(set(int(position) for position in fixed_positions))
    if not fixed:
        raise core.TransformError(
            "split_by_inputs: no tensor input is fixed — every input would "
            "move to `apply`, so there is nothing to prepare (mark at least "
            "one tensor leaf as fixed)"
        )
    if any(position < 0 or position >= count for position in fixed):
        raise core.TransformError(
            f"split_by_inputs: fixed tensor-leaf index out of range for a graph "
            f"with {count} tensor inputs: {fixed}"
        )
    varying = [position for position in range(count) if position not in set(fixed)]
    if not varying:
        raise core.TransformError(
            f"split_by_inputs: all {count} tensor inputs are fixed — `apply` "
            f"would have no varying input, so there is nothing to split "
            f"(leave at least one tensor leaf varying)"
        )

    entry_block = gates(graph.module)
    masks, ops = dependence_masks(entry_block, count, varying)
    needed = needed_set(entry_block, masks)
    boundary = select_boundary(graph, masks, needed)
    needed_ops = tuple(op for op in ops if _op_is_needed(op, needed))

    output_ids = tuple(
        value.id for value in entry_block.terminator.operands
    )
    prepare_ops = _prepare_ops(needed_ops, masks, boundary)
    apply_ops = _apply_ops(needed_ops, masks, boundary, output_ids)

    prepare = build_prepare(graph, prepare_ops, boundary, fixed)
    apply = build_apply(graph, apply_ops, boundary, varying)
    return SplitPlan(
        graph,
        prepare,
        apply,
        fixed,
        varying,
        boundary,
        unused_varying_leaves=_unused_varying_leaves(needed_ops, masks, varying),
    )
