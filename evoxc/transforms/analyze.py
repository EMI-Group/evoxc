"""Public `analyze_inputs` — the per-input facts behind `split_by_inputs`.

`analyze_inputs` answers, for ONE traced graph, "what does each tensor input
cost me?": how many ops consume it, how many ops transitively depend on it,
which outputs it influences, and how many bytes it would take to materialize
the boundary if that input were the FIXED half of a split.

Pure inspection: it never mutates the graph, never traces, never executes and
never imports a backend (only `evoxc.core` / `evoxc.trace` / this package, per
the import contract in `./CONTEXT.md`).
Every fact is computed with the SAME `_split_core` helpers the split itself
uses (`gates` / `dependence_masks` / `needed_set` / `select_boundary`), so the
report can never drift from the transform it explains.

Axis note (binding): `InputInfo.index` is a TENSOR-input index (the
``graph.tensor_specs`` order — the axis `fixed_leaves` uses), while
`InputInfo.path` is rendered over the flat TREE-leaf positions of
``graph.input_specs``, which also counts the static leaves.

Read `./CONTEXT.md` ("split_by_inputs") for the mask semantics, the runtime
call shapes and the correctness law; `split.py` is the pass itself.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from evoxc import core
from evoxc.trace import Graph
from evoxc.trace._tree import _iter_leaf_paths

from evoxc.transforms import _split_core
from evoxc.transforms._split_types import InputAnalysis, InputInfo

__all__ = ["analyze_inputs", "InputAnalysis", "InputInfo"]


def analyze_inputs(graph) -> InputAnalysis:
    """Report the per-tensor-input facts of a traced graph.

    Args:
        graph: a traced `evoxc.Graph` (`evoxc.trace`'s result). Anything else
            raises `TypeError`.

    Returns:
        An `InputAnalysis`: one `InputInfo` per tensor input in
        ``graph.tensor_specs`` order, the graph-level counts, and whether
        `evoxc.split_by_inputs` accepts the graph (`blocked_reason` carries the
        gate's message verbatim when it does not).

    On a BLOCKED graph the per-op facts (`direct_consumers` / `op_count` /
    `outputs`) are still reported — the gate blocks the SPLIT, not the
    counting — while the boundary fields are ``None``. A boundary field is
    also ``None`` when fixing that input ALONE is not a legal split (a graph
    with a single tensor input, where "that one input fixed" means "all
    fixed"). Fixing an input alone that crosses nothing is a LEGAL split and
    reports ``0`` bytes over ``0`` values.
    """
    if not isinstance(graph, Graph):
        raise TypeError(
            "analyze_inputs expects a traced evoxc.Graph (call evoxc.trace "
            f"first), got {type(graph).__name__}"
        )

    specs = graph.tensor_specs
    n_inputs = len(specs)
    output_count = graph.output_tree.num_leaves - len(graph.output_static_values)
    blocked_reason = _blocked_reason(graph.module)
    paths = _tensor_input_paths(graph)
    block = _analysis_block(graph.module)

    if block is None:
        # No single readable function: the gate already explains why and there
        # is nothing to walk, so report the inputs with zeroed facts.
        infos = tuple(
            InputInfo(index, _path_at(paths, index), specs[index])
            for index in range(n_inputs)
        )
        return InputAnalysis(
            graph,
            infos,
            op_count=0,
            output_count=output_count,
            splittable=blocked_reason is None,
            blocked_reason=blocked_reason,
        )

    ops = tuple(op for op in block.ops if not op.is_terminator)
    # Bits are per tensor input (`bit i` ⇔ "depends on tensor input i"), which
    # is exactly the all-varying configuration `dependence_masks` expresses.
    masks, _ = _split_core.dependence_masks(
        block, n_inputs, range(n_inputs)
    )

    direct = _direct_consumers(block, ops, n_inputs)
    reachable = _reachable_ops(ops, masks, n_inputs)
    influenced = _influenced_outputs(graph, block, masks, n_inputs, output_count)
    boundary_count, boundary_bytes = _boundary_alone(
        graph, block, masks, n_inputs, blocked_reason
    )

    infos = tuple(
        InputInfo(
            index,
            _path_at(paths, index),
            specs[index],
            direct_consumers=direct[index],
            op_count=reachable[index],
            outputs=influenced[index],
            boundary_bytes=boundary_bytes[index],
            boundary_count=boundary_count[index],
        )
        for index in range(n_inputs)
    )
    return InputAnalysis(
        graph,
        infos,
        op_count=len(ops),
        output_count=output_count,
        splittable=blocked_reason is None,
        blocked_reason=blocked_reason,
    )


# ---------------------------------------------------------------------------
# graph seams
# ---------------------------------------------------------------------------


def _analysis_block(module) -> Optional[Any]:
    """The entry block to walk, or ``None`` when the module has no single
    readable function (the gate explains that case)."""
    if len(module.functions) != 1:
        return None
    blocks = module.functions[0].region.blocks
    return blocks[0] if blocks else None


def _blocked_reason(module) -> Optional[str]:
    """The split gate's message verbatim, or ``None`` when splittable."""
    try:
        _split_core.gates(module)
    except core.TransformError as exc:
        return str(exc)
    return None


def _tensor_input_paths(graph: Graph) -> List[str]:
    """Rendered pytree paths of the TENSOR inputs, in tensor-input order.

    Every flat leaf of ``graph.input_specs`` in pre-order, minus the STATIC
    leaf positions (``{record.index for record in graph.static_values}``) —
    the same filter `SplitPlan.text()` applies.
    """
    static_positions = {record.index for record in graph.static_values}
    return [
        core.format_path(path)
        for position, path in enumerate(_iter_leaf_paths(graph.input_specs))
        if position not in static_positions
    ]


def _path_at(paths: List[str], index: int) -> str:
    return paths[index] if index < len(paths) else "?"


# ---------------------------------------------------------------------------
# per-input facts
# ---------------------------------------------------------------------------


def _direct_consumers(block, ops, n_inputs: int) -> List[int]:
    """Distinct non-terminator entry-block ops taking each block arg."""
    position_by_id = {
        argument.id: index for index, argument in enumerate(block.arguments)
    }
    counts = [0] * n_inputs
    for op in ops:
        seen = set()
        for operand in op.operands:
            position = position_by_id.get(operand.id)
            if position is None or position in seen:
                continue
            seen.add(position)
            counts[position] += 1
    return counts


def _reachable_ops(ops, masks, n_inputs: int) -> List[int]:
    """Ops whose dependence mask carries each input's bit."""
    counts = [0] * n_inputs
    for op in ops:
        mask = _split_core._op_mask(op, masks)
        while mask:
            bit = mask & -mask
            counts[bit.bit_length() - 1] += 1
            mask ^= bit
    return counts


def _influenced_outputs(
    graph: Graph, block, masks, n_inputs: int, output_count: int
) -> List[List[int]]:
    """Tensor-output leaf indices each input influences.

    Static output leaves are recorded in ``graph.output_static_values`` and are
    NOT emitted as `return` operands, so operand ``k`` IS tensor-output index
    ``k`` (the same convention `trace` builds the terminator with).
    """
    terminator = block.terminator
    operands = () if terminator is None else tuple(terminator.operands)
    tensor_return_ids = [value.id for value in operands[:output_count]]

    outputs: List[List[int]] = [[] for _ in range(n_inputs)]
    for output_index, value_id in enumerate(tensor_return_ids):
        mask = masks.get(value_id, 0)
        while mask:
            bit = mask & -mask
            outputs[bit.bit_length() - 1].append(output_index)
            mask ^= bit
    return outputs


def _boundary_alone(
    graph: Graph, block, masks, n_inputs: int, blocked_reason
) -> Tuple[List[Optional[int]], List[Optional[int]]]:
    """What fixing each input ALONE would cost: ``(counts, bytes)`` per input.

    Both entries are ``None`` when the graph is not splittable (the gate's
    concern, reported separately) or when fixing that input alone is not a
    legal split — i.e. it would leave no varying input, which happens only in
    a graph with fewer than two tensor inputs.

    The sizing reproduces the ``fixed_leaves=(i,)`` call exactly:
    `dependence_masks` with every OTHER tensor input varying, `needed_set` from
    the shared reachability pass, then `select_boundary` — the same helpers
    `split_graph` runs, so the numbers cannot drift.
    """
    bytes_per_input: List[Optional[int]] = [None] * n_inputs
    count_per_input: List[Optional[int]] = [None] * n_inputs
    if blocked_reason is not None or n_inputs < 2:
        return count_per_input, bytes_per_input

    needed = _split_core.needed_set(block, masks)
    for index in range(n_inputs):
        varying = [other for other in range(n_inputs) if other != index]
        masks_i, _ = _split_core.dependence_masks(block, n_inputs, varying)
        values = _split_core.select_boundary(graph, masks_i, needed)
        count_per_input[index] = len(values)
        bytes_per_input[index] = _sum_nbytes(values)
    return count_per_input, bytes_per_input


def _sum_nbytes(values) -> Optional[int]:
    """Summed static boundary size; ``None`` when any value's size is unknown."""
    total = 0
    for value in values:
        if value.nbytes is None:
            return None
        total += value.nbytes
    return total
