"""The READ half of the `split_by_inputs` machinery: what the graph says.

`split_by_inputs` is the FIRST structural (non-numeric) pass in this package:
it slices the graph DATA STRUCTURE into a `prepare` half (the fixed inputs'
contribution) and an `apply` half (the varying inputs on top of a prepared
boundary). It carries NO numerical authority — no derivatives, no batching,
no computation — so it lives in Python rather than in the Rust engine (see
`./CONTEXT.md`, "split_by_inputs").

THIS MODULE AND `_split_build.py` ARE ONE PASS, NOT TWO INDEPENDENT PASSES.
This file only READS the source graph and computes masks / ids / sets;
`_split_build.py` turns those facts into pruned trees, cloned modules and the
two `Graph`s. The two halves are re-exported together by `_split_core.py`, and
the step numbers below continue in `_split_build.py` (5-9). Nothing here
clones, builds, verifies or mutates anything:

1. `gates` — refuse what v1 cannot split (multi-function modules, nested
   control-flow regions, effectful ops: running an effectful op ONCE in
   `prepare` is not the same as it not depending on the varying inputs).
2. `dependence_masks` — one Python-int bitset per SSA value recording WHICH
   varying tensor inputs it depends on (`0` = fixed-only); straight-line SSA
   makes this a single forward pass.
3. `needed_set` — backward reachability from the `return` operands.
4. `select_boundary` — the values that cross from `prepare` to `apply`, in a
   deterministic order (see the function's rationale comments);
   then 4b, `partition` / `op_id_partition` — the ONE validated derivation the
   two halves and every report are built from: the input partition, the
   boundary and the per-half SOURCE op lists (the op lists themselves are
   selected in `_split_build.py`, which is why `partition` reaches for them
   lazily).

Never mutates the source graph, its module, or any source `ValueType`/`Dim`.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from evoxc import core
from evoxc import ir
from evoxc.core import tree as _core_tree
from evoxc.trace.graph import Graph

from evoxc.transforms._split_types import BoundaryValue

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
# 4b. the validated partition (the op lists are selected in `_split_build.py`)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Partition:
    """One validated split: the input partition plus the op partition.

    The single derivation `split_graph` builds its two graphs from and
    `op_id_partition` reports on, so a plan's recorded op provenance can never
    drift from the halves it built.
    """

    fixed: Tuple[int, ...]
    varying: Tuple[int, ...]
    boundary: Tuple[BoundaryValue, ...]
    prepare_ops: Tuple["ir.Op", ...]
    apply_ops: Tuple["ir.Op", ...]
    unused_varying_leaves: Tuple[int, ...]


def partition(graph: Graph, fixed_positions: Sequence[int]) -> _Partition:
    """Validate the input partition and run the whole op partition.

    `fixed_positions` are flat TENSOR-input indices (into `graph.tensor_specs`)
    that stay in `prepare`; every other tensor input moves to `apply`. Both
    halves must be non-trivial (`core.TransformError` otherwise). The source
    graph is never mutated.
    """

    # Imported HERE, not at module scope: the op-list selection is the
    # construction half's, and `_split_build` imports this module (masks,
    # boundary, `partition` itself) at module scope — a module-level import of
    # it here would be a cycle. The two modules are one pass; see the module
    # docstring.
    from evoxc.transforms._split_build import (
        _apply_ops,
        _prepare_ops,
        _unused_varying_leaves,
    )

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
    return _Partition(
        fixed=tuple(fixed),
        varying=tuple(varying),
        boundary=boundary,
        prepare_ops=_prepare_ops(needed_ops, masks, boundary),
        apply_ops=_apply_ops(needed_ops, masks, boundary, output_ids),
        unused_varying_leaves=_unused_varying_leaves(needed_ops, masks, varying),
    )


def op_id_partition(
    graph: Graph, fixed_positions: Sequence[int]
) -> Tuple[frozenset, frozenset]:
    """The SOURCE op-id sets each half of the split emits, ``(prepare, apply)``.

    `clone_module` clones every emitted source op exactly once, so a set's size
    IS that half's entry-block op count. A `constant` op whose result feeds
    BOTH a fixed-only producer of the boundary and a mixed op is emitted by
    both halves, so its id appears in both sets — and ONCE in their union,
    which is what keeps `SplitPlan.summary()["dropped_ops"]` from going
    negative. Both sets come from the SAME `partition` call, so their ids are
    comparable.
    """

    part = partition(graph, fixed_positions)
    return (
        frozenset(id(op) for op in part.prepare_ops),
        frozenset(id(op) for op in part.apply_ops),
    )
