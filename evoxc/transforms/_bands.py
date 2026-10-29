"""Banded ("clustered") merge strategy for the `batch_functions` transform.

The flat merge (`bands=None`) evaluates every function on ONE batched function
axis: programs that differ are merged with `select` muxes and stacked constants,
so the number of emitted ops stays small but the WASTED work grows with N —
every row of a divergent position computes all of its variants.
The banded strategy splits the population into CONTIGUOUS bands and merges each
band independently, then glues the per-band outputs together with one
`concatenate` (plus one row re-permutation when the bands are not in function
order) — trading DISPATCHES for wasted work:

* within a band only that band's functions merge, so a `switch_ops` class
  computes one variant per DISTINCT variant among the band's functions instead
  of one per variant in the whole population;
* across bands nothing is shared, so a position that is shared by functions in
  different bands is recomputed per band, and every band pays one input
  slice/gather per mapped input plus (collectively) one output concatenation.

Partition rules (`normalize_bands` / `even_partition` / `plan_auto_partition`):

* ``bands=None`` — the flat merge (byte-identical to the pre-`bands` output);
* ``bands=K`` (``1 <= K <= N``) — K contiguous bands, sizes as even as
  possible, LARGER bands first (``N=10, K=3`` → ``(0,1,2,3) (4,5,6) (7,8,9)``);
* ``bands="auto"`` — the cheapest candidate partition under the static cost
  model below: the flat one, the FINGERPRINT partition (programs with the same
  structural fingerprint — "programs that would share every class" — in one
  band, bands in first-appearance order), the fingerprint partition with all
  SINGLETON groups coalesced into one trailing misc band, and the GEOMETRIC
  LADDER (`ladder_partitions`: ``k`` in ``AUTO_BAND_LADDER`` =
  ``(2, 4, 8, 16, 32, 64)`` clipped to ``2..N-1``, plus ``k = N``).
  Flat is candidate 0, so the chosen cost can never exceed the flat cost.
* `bands=1` and an auto-resolved single band delegate to the flat emitter, so
  their graph is BYTE-IDENTICAL to `bands=None` (only the report differs).

Static cost model (units: "element-equivalents" — dispatch and arithmetic are
commensurate because a dispatch is what an element-op costs when the op is
small)::

    DISPATCH_COST  = 512.0   # one emitted op == dispatching 512 element-ops
    BAND_OVERHEAD  = 8192.0  # per band: input slice/gather + concat piece + probe
    CONSTANT_COST  = 128.0   # materializing one constant
    CATEGORY_WEIGHTS = {"linalg": 12, "reduction": 2, "sorting": 2, "sparse": 4}
    DEFAULT_CATEGORY_WEIGHT = 1
    SYMBOLIC_DIM_FALLBACK = 1  # a Dim/DimExpr/None dim counts as 1 element

A class costs ``DISPATCH_COST + work(op) * rows`` per distinct VARIANT plus
``DISPATCH_COST + elements * band_size`` per extra variant (the `select` mux,
per-row by construction), where ``rows`` is the band size for a class whose
VARIANT OPS read ``rows``-many rows — one that reaches a mapped input, a SWITCH
class (a muxed value is mapped) or a STACKED constant (``(n, *payload_shape)``)
— and 1 otherwise, and ``work(op) = static result elements * category weight``.
Row-dependence PROPAGATES through operand slots (`_class_row_dependence`), so a
shared-input population (``axes=None``) whose per-program constants are its
only per-row difference prices its whole chain per row instead of once.
`DISPATCH_COST = 512` is the measured crossover: a dispatch costs
roughly what 512 element-ops cost, so banding pays off once a divergent
position's per-row element count is in that range and never helps for deep
elementwise chains over small vectors (see `BatchReport.cost_estimate` /
`flat_cost_estimate` for the model's own verdict on a real merge).
`BAND_OVERHEAD = 8192` covers everything an extra band emits on top of its own
classes — the per-mapped-input slice/gather view, one output concat piece and the
per-band bookkeeping — measured at ~2-3 dispatches on a 1024-element row
(one dispatch ≈ 5 us on numpy ≈ 4000 element-equivalents), and it is the knob
that keeps the model from splitting a population whose flat merge already shares
its work (`bands="auto"` then stays flat, which measured faster in every such
case: a 12-program switched population 0.15 vs 0.39 ms, a 10-program one-op
population 0.13 vs 0.22 ms).
The model exists only to CHOOSE a partition for `bands="auto"`; it is static
program metadata (shapes/ops), never a measurement, and it is not used for the
explicit `K` path.

v1 restriction: ``out_axes=None`` is rejected when `bands` is not ``None`` (a
per-band value cannot be one unmapped output) — see `check_banded_out_entries`.

Everything stays STATIC: the function axis is a Python `int` (the population
size), every band size is an `int`, and no symbolic `Dim` is ever introduced —
banded merges are ordinary graphs that compiler backends lower unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from evoxc import core
from evoxc import ir
from evoxc import ops
from evoxc.ops._utils import _suppress_location_capture, get_location
from evoxc.trace import Graph, with_builder
from evoxc.transforms import _emit
from evoxc.transforms import _merge
from evoxc.transforms._emit import BatchReport
from evoxc.transforms._merge import build_merge_dag
from evoxc.transforms._metadata import MappedAxes, UNMAPPED

# --- public spellings (consumed by `batch_functions`) -----------------------

#: `bands=None` — the single-band flat merge.
BAND_STRATEGY_FLAT = "flat"

#: An explicit band count `K`.
BAND_STRATEGY_EXPLICIT = "explicit"

#: The automatic partition (`BAND_STRATEGY_AUTO` chosen by the cost model).
BAND_STRATEGY_AUTO = "auto"

#: The `bands="auto"` spelling.
AUTO = "auto"

#: Static cost model — see the module docstring for the rationale.
DISPATCH_COST = 512.0
BAND_OVERHEAD = 8192.0
CONSTANT_COST = 128.0
CATEGORY_WEIGHTS = {
    "linalg": 12.0,
    "reduction": 2.0,
    "sorting": 2.0,
    "sparse": 4.0,
}
DEFAULT_CATEGORY_WEIGHT = 1.0
SYMBOLIC_DIM_FALLBACK = 1

#: The geometric band counts the ``"auto"`` planner tries on top of the
#: structural (fingerprint) candidates, ascending (clipped to ``2..n-1``, with
#: ``k = n`` appended by `ladder_partitions`).
AUTO_BAND_LADDER: Tuple[int, ...] = (2, 4, 8, 16, 32, 64)


# --- band validation --------------------------------------------------------


def normalize_bands(bands: Any, n: int) -> Tuple[str, Optional[int]]:
    """Normalize `bands` into ``(strategy, K)``.

    Args:
        bands: `None` (flat), an `int` ``K`` with ``1 <= K <= n`` (K contiguous
            bands), or the string ``"auto"``.
        n: The population size (``len(fns)``).

    Returns:
        ``("flat", None)``, ``("explicit", K)`` or ``("auto", None)``.

    Raises:
        TypeError: `bands` is a `bool` (a `bool` is an `int` subclass — never a
            band count), or neither `None`/`int`/`str`.
        core.TransformError: an unknown string, or a K outside ``1..n``.
    """
    if bands is None:
        return BAND_STRATEGY_FLAT, None
    if isinstance(bands, bool):
        raise TypeError(
            "batch_functions: bands must be None, an int K with 1 <= K <= N, "
            f"or {AUTO!r} — booleans are not accepted (got {bands!r}); pass an "
            "int band count or "
            f"{AUTO!r}"
        )
    if isinstance(bands, int):
        if bands < 1 or bands > n:
            raise core.TransformError(
                f"batch_functions: bands={bands} is out of range for {n} "
                "function(s) — the band count K must satisfy 1 <= K <= N; use "
                "bands=None for the flat merge"
            )
        return BAND_STRATEGY_EXPLICIT, bands
    if isinstance(bands, str):
        if bands == AUTO:
            return BAND_STRATEGY_AUTO, None
        raise core.TransformError(
            f"batch_functions: unknown bands value {bands!r}; accepted values "
            f"are None (flat merge), an int K with 1 <= K <= N, or {AUTO!r}"
        )
    raise TypeError(
        "batch_functions: bands must be None, an int K with 1 <= K <= N, or "
        f"{AUTO!r}, got {type(bands).__name__}"
    )


def check_banded_out_entries(out_entries: Sequence[Optional[int]]) -> None:
    """Reject ``out_axes=None`` on a banded merge (a v1 restriction).

    A band produces a per-band value for every tensor output; those per-band
    values are concatenated along the function axis, so every output leaf must
    carry the function axis.

    Raises:
        core.TransformError: any tensor output leaf has an `out_axes` entry of
            `None`.
    """
    for leaf_index, entry in enumerate(out_entries):
        if entry is None:
            raise core.TransformError(
                "batch_functions: out_axes=None is not supported with bands= "
                "(a per-band value cannot be a single unmapped output) — v1 "
                "restriction of the banded merge; use out_axes=0 for every "
                f"tensor output, or bands=None (offending output leaf: "
                f"{leaf_index})"
            )


def even_partition(n: int, k: int) -> Tuple[Tuple[int, ...], ...]:
    """``n`` indices in `k` contiguous bands, sizes as even as possible, larger first.

    ``even_partition(10, 3) == ((0, 1, 2, 3), (4, 5, 6), (7, 8, 9))``,
    ``even_partition(1, 1) == ((0,),)``, ``even_partition(7, 7)`` = 7 singletons.

    Raises:
        core.TransformError: ``k`` is not in ``1..n`` (defensive — the public
            boundary validates it in `normalize_bands`).
    """
    if k < 1 or k > n:
        raise core.TransformError(
            f"batch_functions: cannot split {n} function(s) into {k} band(s) — "
            "the band count K must satisfy 1 <= K <= N"
        )
    base, extra = divmod(n, k)
    bands: List[Tuple[int, ...]] = []
    position = 0
    for index in range(k):
        size = base + (1 if index < extra else 0)
        bands.append(tuple(range(position, position + size)))
        position += size
    return tuple(bands)

# --- structural fingerprints ------------------------------------------------


def program_fingerprint(graph: Graph, groups: Sequence[Any]) -> tuple:
    """The program's full structural fingerprint (the "would share every class" key).

    Walks the entry block in program order and records, per op,
    ``(identity, operand_refs, result_types)`` plus the terminator's operand
    refs, where an operand ref is ``("in", leaf_index)`` for a block argument
    or ``("op", position)`` for a value produced by an earlier op of the same
    program. Constants fingerprint as ``("constant",)`` whatever their payload
    (their payloads stack into one per-row constant) and `switch_ops` group
    members as ``("group", key)`` — exactly the pairs of programs whose class
    DAG would collapse to one class per position.

    Raises:
        core.TransformError: an operand is neither a block argument nor a value
            produced earlier in the same block (not single-block SSA).
    """
    block = graph.module.functions[0].entry_block
    arg_ids = {id(argument): i for i, argument in enumerate(block.arguments)}
    positions: Dict[int, int] = {}
    items: List[tuple] = []

    def refs_of(operands, op_name: str) -> tuple:
        refs = []
        for operand in operands:
            index = arg_ids.get(id(operand))
            if index is not None:
                refs.append(("in", index))
                continue
            position = positions.get(id(operand))
            if position is None:
                raise core.TransformError(
                    "batch_functions: internal error — the operand of "
                    f"'{op_name}' is neither a block argument nor a value "
                    "produced earlier in the entry block"
                )
            refs.append(("op", position))
        return tuple(refs)

    for op in block.ops:
        if _merge._opdef(op.name).is_terminator:
            continue
        variant = _merge.variant_id(op)
        items.append(
            (
                _merge._identity(op, variant, groups),
                refs_of(op.operands, op.name),
                _merge._result_types(op),
            )
        )
        for result in op.results:
            positions[id(result)] = len(items) - 1
    terminator = block.terminator
    return (tuple(items), refs_of(terminator.operands, terminator.name))


def fingerprint_partition(
    graphs: Sequence[Graph], groups: Sequence[Any]
) -> Tuple[Tuple[int, ...], ...]:
    """Group program indices by structural fingerprint (first-appearance order)."""
    buckets: Dict[tuple, List[int]] = {}
    for index, graph in enumerate(graphs):
        buckets.setdefault(program_fingerprint(graph, groups), []).append(index)
    return tuple(tuple(rows) for rows in buckets.values())

# --- the static cost model --------------------------------------------------


def _static_elements(result_types: Sequence["ir.ValueType"]) -> float:
    """Element count of a result-type tuple (summed; symbolic dims count as 1)."""
    total = 0.0
    for value_type in result_types:
        count = 1.0
        for dim in value_type.shape:
            if isinstance(dim, int) and not isinstance(dim, bool):
                count *= dim
            else:
                count *= SYMBOLIC_DIM_FALLBACK
        total += count
    return total


def op_work(op: "ir.Op") -> float:
    """One variant's per-row element work (result elements × category weight)."""
    weight = CATEGORY_WEIGHTS.get(
        _merge._opdef(op.name).category, DEFAULT_CATEGORY_WEIGHT
    )
    return _static_elements(_merge._result_types(op)) * weight


def _class_row_dependence(
    dag: "_merge.MergeDAG", entries: Sequence[Optional[int]]
) -> "Tuple[Dict[int, bool], Dict[int, bool]]":
    """Per-class row-dependence flags (see `band_cost`).

    Returns ``(operand_rows, value_rows)``:

    * ``operand_rows[class_id]`` — the class's VARIANT ops work on that many
      rows, i.e. one of its operand slots is row-dependent;
    * ``value_rows[class_id]`` — the class's EMITTED VALUE carries the function
      axis (that many rows of real work for every consumer).

    Both propagate through operand slots, and both are seeded by three sources:

    * a MAPPED input leaf (`entries` entry of ``0``);
    * a SWITCH class (``len(info.variants) >= 2``) — its value is muxed per row
      by static select masks, so the emitted value is mapped even when every
      operand is unmapped (`_emit._align_switch_candidate` broadcasts the
      candidates), and a consumer then works on `size` rows;
    * a constant class the emitter STACKS (`_emit.constant_payloads_agree` is
      ``False`` — the payloads differ across the contributing functions):
      `_emit._Merged.emit_constant` builds a ``(size, *payload_shape)`` per-row
      constant, so every consumer of it recomputes its value on every row of
      the band. This is what makes a SHARED-INPUT population (`axes=None`, the
      GP/tree-evaluation shape, where per-program constants are the only
      per-row difference) price its work per row instead of once.
    """
    operand_rows: Dict[int, bool] = {}
    value_rows: Dict[int, bool] = {}
    for class_id in dag.order:
        info = dag.classes[class_id]
        program_index, op = info.representative
        operand_dependent = False
        for operand in op.operands:
            ref = dag.ref_of(program_index, operand)
            if ref[0] == "input":
                if entries[ref[1]] is not None:
                    operand_dependent = True
                    break
            elif value_rows.get(ref[0], False):
                operand_dependent = True
                break
        if op.name == "constant":
            stacked = not _emit.constant_payloads_agree(
                _emit.constant_payloads(info)
            )
            operand_rows[class_id] = stacked
            value_rows[class_id] = stacked
        else:
            operand_rows[class_id] = operand_dependent
            value_rows[class_id] = operand_dependent or len(info.variants) >= 2
    return operand_rows, value_rows


def band_cost(
    graphs: Sequence[Graph],
    entries: Sequence[Optional[int]],
    groups: Sequence[Any],
    alignment: str = _merge.DEFAULT_ALIGNMENT,
) -> float:
    """The model's cost of merging `graphs` as ONE band.

    `alignment` is the `batch_functions` class-alignment spelling the merge will
    use, so the model prices the class DAG the emitter actually builds.
    """
    dag = build_merge_dag(graphs, groups, alignment)
    operand_rows, value_rows = _class_row_dependence(dag, entries)
    size = len(graphs)
    total = 0.0
    for class_id in dag.reachable:
        info = dag.classes[class_id]
        op = info.representative[1]
        if op.name == "constant":
            total += CONSTANT_COST
            continue
        rows = size if operand_rows.get(class_id, False) else 1
        for variant in info.variants:
            _program, variant_op = info.representative_for(variant)
            total += DISPATCH_COST + op_work(variant_op) * rows
        extra = len(info.variants) - 1
        if extra:
            # A switch mux (mask + `select`, plus one broadcast per unmapped
            # candidate) is per-row by construction.
            total += extra * (
                DISPATCH_COST + _static_elements(_merge._result_types(op)) * size
            )
    return total


def partition_cost(
    graphs: Sequence[Graph],
    partition: Sequence[Sequence[int]],
    entries: Sequence[Optional[int]],
    groups: Sequence[Any],
    alignment: str = _merge.DEFAULT_ALIGNMENT,
    memo: Optional[Dict[Tuple[int, ...], float]] = None,
) -> float:
    """The model's cost of a whole partition (bands + per-band overhead).

    `memo` caches per-band costs by band rows (valid for one population): the
    auto candidates often share bands (e.g. the fingerprint partition and a
    ladder step), and a band cost is a class-DAG build.
    """
    total = 0.0
    for rows in partition:
        key = tuple(rows)
        cost = memo.get(key) if memo is not None else None
        if cost is None:
            cost = band_cost(
                [graphs[index] for index in key], entries, groups, alignment
            )
            if memo is not None:
                memo[key] = cost
        total += cost
    if len(partition) > 1:
        total += BAND_OVERHEAD * len(partition)
    return total


def ladder_partitions(n: int) -> List[Tuple[Tuple[int, ...], ...]]:
    """The geometric even-partition candidates of the ``"auto"`` ladder.

    ``even_partition(n, k)`` for ``k`` in `AUTO_BAND_LADDER` clipped to
    ``2..n-1`` (so the flat merge is never a duplicate and the all-singleton
    partition is always last), plus ``k = n`` itself.
    """
    counts = [k for k in AUTO_BAND_LADDER if 2 <= k <= n - 1]
    counts.append(n)
    return [even_partition(n, k) for k in dict.fromkeys(counts)]


def plan_auto_partition(
    graphs: Sequence[Graph],
    entries: Sequence[Optional[int]],
    groups: Sequence[Any],
    alignment: str = _merge.DEFAULT_ALIGNMENT,
) -> Tuple[Tuple[Tuple[int, ...], ...], float, float]:
    """Choose the cheapest candidate partition (`bands="auto"`).

    Candidates, in priority order (earlier wins a tie): the FLAT partition, the
    fingerprint partition, the fingerprint partition with all singleton groups
    coalesced into ONE trailing misc band (omitted when there are no
    singletons), then the geometric ladder `ladder_partitions` — ``k`` in
    {2, 4, 8, 16, 32, 64} ∩ ``2..n-1`` and ``k = n``, ascending (a tie between
    two ladder steps goes to the smaller band count, i.e. the cheaper glue).
    The ladder is what makes the measured optimum of a DIVERGENT population
    reachable: its fingerprints are all distinct, so the structural candidates
    degenerate to {flat, k = n} while the measured argmin sits in between (or
    at ``k = n``). Duplicates are dropped preserving order.

    Returns:
        ``(partition, chosen_cost, flat_cost)``.

    Raises:
        core.TransformError: the model's chosen cost exceeds the flat cost —
            impossible by construction (flat is candidate 0), so this would be
            an internal bookkeeping bug.
    """
    n = len(graphs)
    buckets = fingerprint_partition(graphs, groups)
    singleton_rows = tuple(
        index for rows in buckets if len(rows) == 1 for index in rows
    )
    coalesced = tuple(rows for rows in buckets if len(rows) > 1) + (
        (singleton_rows,) if singleton_rows else ()
    )
    candidates: List[Tuple[Tuple[int, ...], ...]] = [
        (tuple(range(n)),),
        buckets,
        coalesced,
        *ladder_partitions(n),
    ]
    unique: List[Tuple[Tuple[int, ...], ...]] = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    memo: Dict[Tuple[int, ...], float] = {}
    costs = [
        partition_cost(graphs, cand, entries, groups, alignment, memo)
        for cand in unique
    ]
    best = 0
    for index in range(1, len(costs)):
        if costs[index] < costs[best]:
            best = index
    if costs[best] > costs[0]:
        raise core.TransformError(
            "batch_functions: internal error — the automatic partition costs "
            f"{costs[best]} > the flat partition's {costs[0]}"
        )
    return unique[best], costs[best], costs[0]

# --- band-local input views -------------------------------------------------


class _BandInputs:
    """Band-local views of the merged function-axis block args.

    A band is a SUBSET of the function rows, so an op of the band reads row
    ``rows[k]`` of a MAPPED input at band row ``k``. Every ``(band, input
    leaf)`` view is materialized LAZILY, ONCE — the first time that band's
    emission resolves the block argument — and recorded in the band's
    `ValueEnv`, so a graph slicing an input from many ops emits one slice.
    An UNMAPPED (shared) input needs no view at all: the block arg itself is
    shared by every band (`UNMAPPED` metadata).

    The view is a `slice` when the band is contiguous AND the arg's trailing
    dims are static ints, else a `gather` with ONE cached int32 index constant
    (non-contiguous or symbolic trailing dims).
    """

    def __init__(
        self,
        block_args: Sequence[ir.Value],
        entries: Sequence[Optional[int]],
        rows: Sequence[int],
        state: "_emit._Merged",
        new_block: ir.Block,
        call_site: Any,
    ) -> None:
        self._block_args = block_args
        self._entries = entries
        self._rows = tuple(rows)
        self._state = state
        self._new_block = new_block
        self._call_site = call_site
        self._cache: Dict[int, ir.Value] = {}
        self._index_value: Optional[ir.Value] = None
        first = self._rows[0]
        self._contiguous = self._rows == tuple(
            range(first, first + len(self._rows))
        )

    def __getitem__(self, leaf_index: int) -> ir.Value:
        """The band-local value of tensor input leaf `leaf_index`."""
        cached = self._cache.get(leaf_index)
        if cached is not None:
            return cached
        arg = self._block_args[leaf_index]
        if self._entries[leaf_index] is None:
            value, axes = arg, UNMAPPED
        else:
            value = self._materialize(arg)
            axes = MappedAxes((0,))
        self._cache[leaf_index] = value
        self._state.env.set(value, axes)
        return value

    def _materialize(self, arg: ir.Value) -> ir.Value:
        """Emit the band view of one MAPPED block arg."""
        shape = tuple(arg.type.shape)
        if not shape:
            raise core.TransformError(
                "batch_functions: internal error — a mapped input block "
                "argument has rank 0"
            )
        start = len(self._new_block.ops)
        try:
            if self._contiguous and all(
                isinstance(dim, int) and not isinstance(dim, bool) for dim in shape
            ):
                first = self._rows[0]
                value = ops.slice(
                    _emit._sym(arg),
                    start=(first,) + (0,) * (len(shape) - 1),
                    lengths=(len(self._rows),) + shape[1:],
                    strides=1,
                ).value
            else:
                value = ops.gather(
                    _emit._sym(arg), _emit._sym(self._index()), axis=0
                ).value
        except core.ShapeError as exc:
            raise core.TransformError(
                f"batch_functions: cannot read band rows {self._rows} from a "
                f"mapped input of shape {shape}: {exc}"
            ) from exc
        _emit._stamp_locations(self._new_block, start, self._call_site)
        return value

    def _index(self) -> ir.Value:
        """The band's cached int32 row-index constant (gather path only)."""
        value = self._index_value
        if value is None:
            start = len(self._new_block.ops)
            value = ops.constant(
                core.tensor(np.asarray(self._rows, dtype=np.int32))
            ).value
            _emit._stamp_locations(self._new_block, start, self._call_site)
            self._index_value = value
        return value

# --- the banded driver ------------------------------------------------------


def _combine_bands(
    bands: Sequence[Tuple[Sequence[int], Tuple[ir.Value, ...], "_emit._Merged"]],
    out_entries: Sequence[Optional[int]],
    n_functions: int,
    new_block: ir.Block,
    call_site: Any,
) -> Tuple[ir.Value, ...]:
    """Concatenate the per-band outputs into one function-axis output per leaf.

    Every band value is mapped on its leading axis (band size), so the pieces
    concatenate along axis 0; when the bands are not in function order the
    concatenated rows are re-permuted ONCE with a shared static index constant
    so that output row g is function g.
    """
    helper_start = len(new_block.ops)
    concat_order = [index for rows, _values, _state in bands for index in rows]
    inverse: Optional[ir.Value] = None
    if concat_order != list(range(n_functions)):
        inverse_values = [0] * n_functions
        for position, global_index in enumerate(concat_order):
            inverse_values[global_index] = position
        inverse = ops.constant(
            core.tensor(np.asarray(inverse_values, dtype=np.int32))
        ).value
    outputs: List[ir.Value] = []
    for leaf_index in range(len(out_entries)):
        pieces: List[ir.Value] = []
        for rows, values, state in bands:
            value = values[leaf_index]
            if not state.env.get(value).is_mapped:
                value = _emit._broadcast(
                    value, (len(rows),) + value.type.shape
                )
            pieces.append(value)
        try:
            combined = (
                pieces[0]
                if len(pieces) == 1
                else ops.concatenate([_emit._sym(p) for p in pieces], axis=0).value
            )
            if inverse is not None:
                combined = ops.gather(
                    _emit._sym(combined), _emit._sym(inverse), axis=0
                ).value
        except core.ShapeError as exc:
            raise core.TransformError(
                "batch_functions: cannot combine the per-band values of output "
                f"leaf {leaf_index} into one function-axis output: {exc}"
            ) from exc
        outputs.append(combined)
    _emit._stamp_locations(new_block, helper_start, call_site)
    return tuple(outputs)


def emit_banded_graph(
    graphs: Sequence[Graph],
    partition: Sequence[Sequence[int]],
    entries: Sequence[Optional[int]],
    out_entries: Sequence[Optional[int]],
    groups: Sequence[Any],
    strategy: str,
    cost_estimate: Optional[float] = None,
    flat_cost_estimate: Optional[float] = None,
    alignment: str = _merge.DEFAULT_ALIGNMENT,
) -> Graph:
    """Build the merged graph of a multi-band merge (and attach `batch_report`).

    Args:
        graphs: The traced programs being merged (identical signatures).
        partition: Contiguous bands (each a tuple of GLOBAL function indices);
            their concatenation is a permutation of ``range(len(graphs))``.
        entries: One ``0``/``None`` per TENSOR input leaf.
        out_entries: One ``0``/``None`` per TENSOR output leaf (``None`` was
            already rejected by `check_banded_out_entries`).
        groups: The `switch_ops` merge groups (declaration order).
        strategy: The report's `band_strategy` (``"explicit"``/``"auto"``).
        cost_estimate: The model's estimate for `partition`, when the model ran.
        flat_cost_estimate: The model's estimate for the flat partition, when
            the model ran.
        alignment: The `batch_functions` class-alignment spelling — every band's
            class DAG is built with it (per-band class DAGs align exactly like
            the flat one; `SwitchRecord.rows_per_variant` stays in GLOBAL
            function indices because it is mapped through the band's row map,
            which is independent of the class assignment).

    Returns:
        The merged `Graph`, with `graph.batch_report` set.

    Raises:
        core.TransformError: an internal bookkeeping violation or a shape
            problem while slicing/gathering/concatenating band values.
    """
    graph0 = graphs[0]
    n_functions = len(graphs)
    if sorted(index for rows in partition for index in rows) != list(
        range(n_functions)
    ):
        raise core.TransformError(
            "batch_functions: internal error — the band partition "
            f"{partition!r} is not a permutation of 0..{n_functions - 1}"
        )

    builder, _module, new_block, new_specs, _input_axes = _emit.new_merged_module(
        graph0, entries, n_functions
    )
    source_locations: Dict[int, Any] = {}
    for old_arg, new_arg in zip(
        graph0.module.functions[0].entry_block.arguments, new_block.arguments
    ):
        location = graph0.source_locations.get(old_arg.id)
        if location is not None:
            source_locations[new_arg.id] = location

    call_site = get_location()
    bands: List[Tuple[Tuple[int, ...], Tuple[ir.Value, ...], "_emit._Merged"]] = []
    switched: List[Any] = []
    value_classes = shared_classes = stacked = muxes = 0
    with _suppress_location_capture(), with_builder(builder):
        for rows in partition:
            band_graphs = [graphs[index] for index in rows]
            band_dag = build_merge_dag(band_graphs, groups, alignment)
            band_size = len(rows)
            state = _emit._Merged(
                band_graphs,
                band_dag,
                band_size,
                groups,
                row_map=tuple(rows),
            )
            state.new_block = new_block
            state.call_site = call_site
            state.band_args = _BandInputs(
                new_block.arguments, entries, rows, state, new_block, call_site
            )
            for class_id in band_dag.reachable:
                info = band_dag.classes[class_id]
                if info.representative[1].name == "constant":
                    state.emit_constant(info)
                else:
                    state.emit_class(info)
            outputs = state.emit_outputs(out_entries)
            bands.append((tuple(rows), outputs, state))
            source_locations.update(state.source_locations)
            value_classes += len(band_dag.reachable)
            shared_classes += sum(
                1
                for class_id in band_dag.reachable
                if len(band_dag.classes[class_id].program_indices) >= 2
            )
            stacked += state.stacked_constants
            muxes += state.output_muxes
            switched.extend(state.switched)
        output_values = _combine_bands(
            bands, out_entries, n_functions, new_block, call_site
        )

    graph = _emit.finish_merged_graph(
        graph0,
        builder,
        new_block,
        new_specs,
        output_values,
        source_locations,
    )
    graph.batch_report = BatchReport(
        n_functions=n_functions,
        value_classes=value_classes,
        shared_classes=shared_classes,
        switched=tuple(switched),
        stacked_constants=stacked,
        output_muxes=muxes,
        emitted_ops=len(new_block.ops) - 1,
        bands=len(partition),
        band_sizes=tuple(len(rows) for rows in partition),
        band_strategy=strategy,
        cost_estimate=cost_estimate,
        flat_cost_estimate=flat_cost_estimate,
    )
    return graph
