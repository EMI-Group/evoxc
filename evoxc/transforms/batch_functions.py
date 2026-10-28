"""Public `batch_functions` — merge N same-signature programs onto one graph.

`batch_functions(fns, axes=0, out_axes=0, *, switch_ops=None)` merges a LIST of
N traced `Graph`s (or of plain callables/`Defn`s) with IDENTICAL input and
output signatures but DIFFERENT implementations into ONE `Graph` whose leading
axis is a **static "function axis"** of length N: row i evaluates function i.

Binding semantics (see `./CONTEXT.md` "Batch functions"): for any run-time
inputs, ``run(merged, *inputs)`` row i equals ``run(graphs[i], *rowwise_i)`` —
a mapped (`axes` entry ``0``) input supplies row i of its leading axis, an
unmapped (``None``) input is shared by every function, static leaves are shared
(validated), and output row i is function i's output.

Emission strategy (never a silent Python-loop fallback):

* positions where all functions run the SAME op become ONE batched op (exactly
  `vectorize` speed — with identical programs the result contains no `select`
  at all);
* positions with the same skeleton but different ops inside a merge group
  (`switch_ops`) become one op per variant plus a static per-row `select` mask
  cascade (the muxed value is ALWAYS mapped on the function axis, whatever the
  mappedness of the shared inputs);
* every other divergence is still CORRECT via `select`-muxing where the values
  diverge (constant stacking resp. output muxing).

Banding (`bands=`): the merge can split the population into CONTIGUOUS bands
and merge each band independently, gluing the per-band outputs with one
`concatenate` (plus a static row re-permutation when the bands are not in
function order).
Banding trades DISPATCHES for wasted work: within a band only that band's
functions share positions, so a divergent position computes one op per DISTINCT
variant of the band instead of one per variant of the whole population, while
work shared only across bands is recomputed per band and each band pays its
input slicing.
`bands=None` (the default) is the single-band flat merge and stays
BYTE-IDENTICAL to the pre-`bands` output; see `_bands` for the partition rules
and the static cost model behind `bands="auto"`.

The returned `Graph` carries `graph.batch_report` — a `BatchReport` describing
what the merge did (`BatchReport` / `SwitchRecord` are re-exported here from
`_emit`, where the report is assembled).
"""

from __future__ import annotations

import collections.abc
from typing import Any, List, Optional, Sequence, Tuple

from evoxc import core
from evoxc import ir
from evoxc.trace import Graph, trace
from evoxc.trace._tree import _iter_leaf_paths, _static_equal
from evoxc.transforms import batching
from evoxc.transforms import _bands
from evoxc.transforms._emit import BatchReport, SwitchRecord, emit_merged_graph
from evoxc.transforms._merge import (
    MergeDAG,
    MergeGroup,
    _opdef,
    build_merge_dag,
    parse_switch_groups,
)
from evoxc.transforms._wrappers import TransformCallable
from evoxc.transforms.vectorize import (
    _is_int,
    _normalize_axes,
    _normalize_axis_entries,
)
from evoxc.transforms.vmap import _derive_unvectorized_args_and_entries

__all__ = ["batch_functions", "BatchReport", "SwitchRecord"]

_BATCH_FUNCTIONS_CALLABLE_DOC = """\
`batch_functions(fns)` applied to a sequence of callables/`Defn`s returns a
TransformCallable: calling it with a structure of TensorSpecs (mapped inputs
include the leading function axis) + static values traces EVERY function
exactly once with the same stripped specs and returns the merged Graph, whose
leading axis of length len(fns) is a static "function axis".

Equivalent to: strip the leading function-axis dim from each mapped spec, trace
each function, then merge the resulting graphs (mapped inputs/outputs keep a
static leading dim of length len(fns))."""


# --- public entry point -----------------------------------------------------


def batch_functions(
    fns: Sequence[Any],
    axes: Any = 0,
    out_axes: Any = 0,
    *,
    switch_ops: Any = None,
    bands: Any = None,
) -> Any:
    """Merge N same-signature programs into one graph with a function axis.

    Args:
        fns: A non-empty ``list``/``tuple`` (any `collections.abc.Sequence`
            except `str`/`bytes`) of traced `Graph`s **or** of plain
            callables/`Defn`s — never mixed. Every program must have the same
            input AND output tree structure and per-leaf dtype/shape/device,
            plus identical recorded static leaves; the implementations may
            differ freely.
        axes: `int 0` or `None`, or a pytree thereof matching each function's
            input tree. Entry `0` makes that input leaf's leading axis the
            FUNCTION axis (the merged input gains a static leading dim of
            length ``len(fns)``); `None` shares the input unmapped across every
            function. Normalization is exactly `vectorize`'s: a bare int/None
            only when the graph has exactly ONE tensor input (or a registered
            pytree node in the tree), recorded static positions must be `None`,
            entries must be ``{0, None}`` and non-zero ints raise
            `core.TransformError` (v1 deferral).
        out_axes: pytree matching the output tree, entries ``{0, None}``. `0`
            (the default) keeps the leading function axis on the output; `None`
            is allowed only when the merged output is NOT mapped (else
            `core.TransformError` — a batch axis is never silently dropped).
            Supported cases need no rearrangement ops beyond an explicit
            broadcast of an unmapped output under `out_axes=0`.
        switch_ops: op-switching policy. `None`/`False` (default) switches
            nothing (only exact-structure sharing + value muxing); `True` uses
            the default merge groups (the IR categories
            ``{"elementwise", "comparison"}``); any other iterable (not `str`)
            gives explicit groups, each element either a category name (`str`,
            validated against ``{o.category for o in ir.all_opdefs()}``) or an
            iterable of op names (validated against the IR registry). Ops of one
            group at the same graph position become one op per variant plus a
            static per-row `select` mask cascade.
        bands: the band ("cluster") strategy — how the N functions are
            PARTITIONED before merging. `None` (default) merges them all on one
            function axis (byte-identical to the pre-`bands` output); an `int`
            `K` with ``1 <= K <= N`` merges K contiguous bands whose sizes are
            as even as possible, larger bands first (``N=10, K=3`` →
            ``(4, 3, 3)`` rows ``(0-3) (4-6) (7-9)``); ``"auto"`` lets the
            static cost model choose among the flat partition, the structural
            fingerprint partition (programs that would share every class) and
            the fingerprint partition with all singletons coalesced into one
            trailing misc band (see `_bands`). Banding trades dispatches for
            wasted work: a divergent position emits one op per DISTINCT variant
            of its band instead of one per variant of the whole population, at
            the cost of re-computing cross-band sharing plus one input slice
            and (collectively) one output concatenation per band. `K=1` and an
            auto-resolved single band emit the flat graph. Validated here, so a
            bad value raises before any tracing (callable path included).
            v1 restriction: `bands` other than `None` requires `out_axes=0` for
            every tensor output (a per-band value cannot be one unmapped
            output) — `core.TransformError` otherwise.

    Returns:
        * `Graph` when given `Graph`s — the merged graph, with
          ``graph.batch_report`` (a `BatchReport`) describing the merge.
        * `TransformCallable` (`kind="batch_functions"`) when given callables —
          calling it with TensorSpecs returns the merged `Graph`; every wrapped
          function is traced exactly once per call and concrete `Tensor` args
          raise `core.TraceError` (transforms never execute).

    Raises:
        TypeError: `fns` is not a non-empty sequence, mixes Graphs with
            callables, holds an unsupported element, `switch_ops` is
            malformed, or `bands` is a bool / an unsupported type.
        core.TraceError: mismatched input/output tree structure or static
            leaves between functions.
        core.DTypeError / core.ShapeError / core.DeviceError: per-leaf dtype /
            shape / device mismatch between functions.
        core.TransformError: invalid `axes`/`out_axes` entries, an unknown
            `bands` value or a band count outside ``1..N``, `bands` combined
            with an `out_axes=None` tensor output, an op without a batching
            rule in any program (naming the function index), region-bearing
            control flow, a registered output pytree node with static leaves
            (v1 deferral), or conflicting switch-variant result types.

    Example:
        g0 = evoxc.trace(lambda x: evoxc.sin(x), spec)
        g1 = evoxc.trace(lambda x: evoxc.cos(x), spec)
        merged = evoxc.transforms.batch_functions([g0, g1], switch_ops=True)
        y = evoxc.evaluate(...)  # row 0 == sin(x), row 1 == cos(x)
    """
    items, kind = _validate_fns(fns)
    # `bands` is validated BEFORE any tracing (the callable path included), so a
    # bad band count can never leave half-traced programs behind.
    _bands.normalize_bands(bands, len(items))
    groups = parse_switch_groups(switch_ops)
    if kind == "graph":
        try:
            normalized = _normalize_axes(items[0], axes)
        except core.TransformError as exc:
            raise core.TransformError(
                "batch_functions: invalid axes: " + str(exc)
            ) from exc
        entries = batching._tensor_input_axes(items[0], normalized)
        out_entries = _normalize_out_axes(items[0], out_axes)
        return _merge_graphs(items, entries, out_entries, groups, bands)
    return TransformCallable(
        build=lambda *args: _batch_callables(
            items, args, axes, out_axes, groups, bands
        ),
        kind="batch_functions",
        doc=_BATCH_FUNCTIONS_CALLABLE_DOC,
    )


# --- callable path ----------------------------------------------------------


def _batch_callables(fns, args, axes, out_axes, groups, bands=None) -> Graph:
    """`TransformCallable` body: strip → trace every fn once → merge.

    Axes are normalized against the BATCHED argument specs and the leading
    function-axis dim is stripped from each mapped spec BEFORE tracing (the
    traced graphs therefore carry the per-function specs — running
    `vectorize._normalize_axes` on them afterwards would mis-rank-check an
    `axes=0` on a scalar-per-function input).
    """
    stripped, entries = _derive_unvectorized_args_and_entries(
        args, axes, "batch_functions"
    )
    graphs = [trace(fn, *stripped) for fn in fns]
    out_entries = _normalize_out_axes(graphs[0], out_axes)
    return _merge_graphs(graphs, entries, out_entries, groups, bands)


# --- fns validation ---------------------------------------------------------


def _validate_fns(fns) -> Tuple[List[Any], str]:
    """Validate `fns`; return ``(items, kind)`` with kind ``graph``/``callable``.

    `fns` must be a non-empty `collections.abc.Sequence` (a `str`/`bytes` is
    rejected — it is a sequence of characters, never a program list) holding
    either ONLY traced `Graph`s or ONLY callables/`Defn`s; mixing the two kinds
    is a `TypeError` naming both kinds and an index.
    """
    if isinstance(fns, (str, bytes, bytearray)) or not isinstance(
        fns, collections.abc.Sequence
    ):
        raise TypeError(
            "batch_functions: fns must be a non-empty sequence (list/tuple) of "
            "traced Graphs or of callables/@evoxc.defn functions, got "
            f"{type(fns).__name__}"
        )
    items = list(fns)
    if not items:
        raise TypeError(
            "batch_functions: fns must be a non-empty sequence of traced "
            "Graphs or of callables — got an empty sequence"
        )
    is_graph = [isinstance(item, Graph) for item in items]
    if all(is_graph):
        return items, "graph"
    if any(is_graph):
        graph_index = is_graph.index(True)
        other_index = next(i for i, flag in enumerate(is_graph) if not flag)
        raise TypeError(
            "batch_functions: fns must not mix traced Graphs with callables — "
            f"element {graph_index} is a Graph while element {other_index} is "
            f"a {type(items[other_index]).__name__}; pass one kind only"
        )
    for index, item in enumerate(items):
        if not (callable(item) or getattr(item, "__evoxc_defn__", False)):
            raise TypeError(
                f"batch_functions: fns[{index}] must be a traced Graph or a "
                f"callable/@evoxc.defn function, got {type(item).__name__}"
            )
    return items, "callable"


# --- cross-function contract validation -------------------------------------


def _program_function(graph: Graph, index: int) -> ir.Function:
    """The graph's single entry function (v1 program-shape guard)."""
    try:
        function = batching._entry_function(graph)
    except core.TransformError as exc:
        raise core.TransformError(
            f"batch_functions: function {index}: {exc}"
        ) from exc
    block = function.entry_block
    terminator = block.terminator
    if terminator is None or terminator.name != "return":
        raise core.TransformError(
            f"batch_functions: function {index}: the entry block has no "
            "'return' terminator (v1 graphs terminate with 'return')"
        )
    if len(block.arguments) != len(graph.tensor_specs):
        raise core.TransformError(
            f"batch_functions: function {index}: the graph function has "
            f"{len(block.arguments)} block args but {len(graph.tensor_specs)} "
            "tensor input specs"
        )
    return function


def _tensor_paths(tree_spec, static_records) -> List[Tuple[Any, ...]]:
    """Pytree paths of the TENSOR leaves of `tree_spec`, in flat leaf order."""
    paths = list(_iter_leaf_paths(tree_spec))
    static_positions = {record.index for record in static_records}
    return [
        path
        for position, path in enumerate(paths)
        if position not in static_positions
    ]


def _check_static_leaves(
    expected: Sequence[Any], got: Sequence[Any], index: int, what: str
) -> None:
    """Static leaves must match function 0 in both kind and value."""
    for record0, record in zip(expected, got):
        if record.kind == record0.kind and _static_equal(
            record.value, record0.value
        ):
            continue
        raise core.TraceError(
            f"batch_functions: function {index}: static {what} leaf at pytree "
            f"path {core.format_path(record0.path)} differs from function 0 "
            f"(function 0 records {record0.value!r} of kind {record0.kind!r}, "
            f"function {index} passes {record.value!r} of kind "
            f"{record.kind!r}); static values specialize the graph and must "
            "match across all merged functions"
        )


def _check_contract(graphs: Sequence[Graph]) -> None:
    """Validate that every function shares function 0's full signature.

    Compares each function ``k >= 1`` against function 0: identical input and
    output pytree STRUCTURE (`core.TraceError`, with the first mismatch path),
    identical tensor-leaf count, per-leaf dtype (`core.DTypeError`), shape
    (`core.ShapeError`) and device (`core.DeviceError`), and identical recorded
    static input/output leaves (type + value; ndarray leaves via
    `np.array_equal` semantics — `core.TraceError`). The merged input specs are
    function 0's `TensorSpec`s (name/device preserved): only shapes may change,
    and only for mapped leaves.
    """
    graph0 = graphs[0]
    input_paths = _tensor_paths(graph0.input_specs, graph0.static_values)
    output_paths = _tensor_paths(graph0.output_tree, graph0.output_static_values)
    output_types0 = graph0.module.functions[0].entry_block.terminator.operands
    for index, graph in enumerate(graphs[1:], start=1):
        function = _program_function(graph, index)
        mismatch = core.first_mismatch_path(
            graph.input_specs, graph0.input_specs, leaf_vs_empty_is_mismatch=True
        )
        if mismatch is not None:
            raise core.TraceError(
                f"batch_functions: function {index}: the input tree structure "
                f"differs from function 0 — first mismatch at pytree path "
                f"{core.format_path(mismatch)}"
            )
        if len(graph.tensor_specs) != len(graph0.tensor_specs):
            raise core.TraceError(
                f"batch_functions: function {index}: {len(graph.tensor_specs)} "
                f"tensor inputs but function 0 has {len(graph0.tensor_specs)}"
            )
        for leaf, (spec0, spec) in enumerate(
            zip(graph0.tensor_specs, graph.tensor_specs)
        ):
            path = core.format_path(input_paths[leaf])
            if spec.dtype != spec0.dtype:
                raise core.DTypeError(
                    f"batch_functions: function {index}: input leaf {leaf} at "
                    f"pytree path {path} has dtype {spec.dtype} but function 0 "
                    f"has {spec0.dtype}"
                )
            if tuple(spec.shape) != tuple(spec0.shape):
                raise core.ShapeError(
                    f"batch_functions: function {index}: input leaf {leaf} at "
                    f"pytree path {path} has shape {tuple(spec.shape)} but "
                    f"function 0 has {tuple(spec0.shape)}"
                )
            if spec.device != spec0.device:
                raise core.DeviceError(
                    f"batch_functions: function {index}: input leaf {leaf} at "
                    f"pytree path {path} lives on device {spec.device} but "
                    f"function 0 has {spec0.device}"
                )
        _check_static_leaves(graph0.static_values, graph.static_values, index, "input")

        mismatch = core.first_mismatch_path(
            graph.output_tree, graph0.output_tree, leaf_vs_empty_is_mismatch=True
        )
        if mismatch is not None:
            raise core.TraceError(
                f"batch_functions: function {index}: the output tree structure "
                f"differs from function 0 — first mismatch at pytree path "
                f"{core.format_path(mismatch)}"
            )
        outputs = function.entry_block.terminator.operands
        if len(outputs) != len(output_types0):
            raise core.TraceError(
                f"batch_functions: function {index}: the graph returns "
                f"{len(outputs)} tensor values but function 0 returns "
                f"{len(output_types0)}"
            )
        for leaf, (value0, value) in enumerate(zip(output_types0, outputs)):
            path = core.format_path(output_paths[leaf])
            if value.type.dtype != value0.type.dtype:
                raise core.DTypeError(
                    f"batch_functions: function {index}: output leaf {leaf} at "
                    f"pytree path {path} has dtype {value.type.dtype} but "
                    f"function 0 has {value0.type.dtype}"
                )
            if tuple(value.type.shape) != tuple(value0.type.shape):
                raise core.ShapeError(
                    f"batch_functions: function {index}: output leaf {leaf} at "
                    f"pytree path {path} has shape {tuple(value.type.shape)} "
                    f"but function 0 has {tuple(value0.type.shape)}"
                )
        _check_static_leaves(
            graph0.output_static_values, graph.output_static_values, index, "output"
        )


def _check_ops(graphs: Sequence[Graph]) -> None:
    """Every op in every program must be batchable (or a `constant`).

    Region-bearing control flow (`cond`/`while_loop`/`scan`) and every op
    without a registered batching rule (looked up through
    `batching._rule_key`, so `block:<name>` / `external:<name>` namespacing
    works) is a v1 deferral of `batch_functions` — an explicit
    `core.TransformError` naming the op and the function index, never a silent
    Python-loop fallback.
    """
    for index, graph in enumerate(graphs):
        block = graph.module.functions[0].entry_block
        for op in block.ops:
            if _opdef(op.name).is_terminator:
                continue
            if op.regions:
                raise core.TransformError(
                    f"batch_functions: cannot batch op '{op.name}' from "
                    f"function {index}: region-bearing control-flow ops "
                    "(cond/while_loop/scan) are not supported by "
                    "batch_functions in v1 — v1 deferral of batch_functions"
                )
            if op.name == "constant":
                continue
            key = batching._rule_key(op)
            if key in batching.batching_rules:
                continue
            raise core.TransformError(
                f"batch_functions: cannot batch op '{op.name}' from function "
                f"{index}: no batching rule registered for key '{key}' — v1 "
                "deferral of batch_functions (runtime_call/external_call, "
                "collectives, sparse/random/sorting/linalg ops and block calls "
                "without their namespaced rules are not batchable); register a "
                f"rule with register_batching_rule('{key}', fn) or drop the op "
                "from the merged programs — there is no silent Python-loop "
                "fallback"
            )


def _check_output_nodes(graphs: Sequence[Graph]) -> None:
    """Registered output pytree nodes with static leaves are a v1 deferral.

    `batch_functions` prepends a static function axis to every mapped output,
    which would go stale for node metadata derived from tensor shapes (e.g. a
    sparse value's `dense_shape`). `register_batched_aux_remap` handles this
    for `vectorize`; `batch_functions` deliberately does NOT attempt batched aux
    remapping in v1.
    """

    def walk(spec) -> bool:
        if batching._lookup_batched_aux_remap(spec) is not None:
            return True
        return any(walk(child) for child in spec.children)

    for index, graph in enumerate(graphs):
        if walk(graph.output_tree):
            raise core.TransformError(
                "batch_functions: function "
                f"{index} returns a registered pytree node with static leaves "
                "(e.g. a sparse tensor) — batched output-aux remapping is a v1 "
                "deferral of batch_functions; densify the output or stay on "
                "evoxc.vmap"
            )


# --- out_axes normalization -------------------------------------------------


def _normalize_out_axes(
    graph: Graph, out_axes
) -> List[Optional[int]]:
    """Normalize `out_axes` into one ``0``/``None`` per TENSOR output leaf.

    Mirrors the front half of `vmap._rearrange_outputs`: the pytree must match
    the graph's output structure (first-mismatch-path `core.TransformError`
    otherwise), static output leaves must be `None`, entries must be
    ``{0, None}`` (non-zero ints are the v1 non-leading-axes deferral), and a
    bare int/None is accepted only with at most ONE tensor output. No
    rearrangement ops are emitted — the caller only reads the entries.
    """
    static_positions = {record.index for record in graph.output_static_values}
    tensor_positions = set(range(graph.output_tree.num_leaves)) - static_positions
    if out_axes is None or _is_int(out_axes):
        if len(tensor_positions) > 1:
            raise core.TransformError(
                "batch_functions: a bare int/None out_axes applies only when "
                f"the graph has at most ONE tensor output; got "
                f"{len(tensor_positions)} tensor outputs — pass a pytree "
                "matching the output structure instead"
            )
        entries = [
            out_axes if index in tensor_positions else None
            for index in range(graph.output_tree.num_leaves)
        ]
    else:
        axes_leaves, axes_spec = core.flatten(out_axes)
        mismatch = core.first_mismatch_path(
            axes_spec, graph.output_tree, leaf_vs_empty_is_mismatch=True
        )
        if mismatch is not None:
            raise core.TransformError(
                "batch_functions: the out_axes pytree does not match the "
                f"output structure — first mismatch at pytree path "
                f"{core.format_path(mismatch)}; out_axes must be a pytree with "
                "the same container structure as the outputs (0 keeps the "
                "leading function axis, None requires the output unmapped)"
            )
        entries = list(axes_leaves)
    # No rank range-check for outputs: entry 0 on a rank-0 unmapped output is
    # the legal "broadcast to the function axis" case, not an error.
    normalized = _normalize_axis_entries(
        entries, tensor_positions, None, "batch_functions"
    )
    return [normalized[position] for position in sorted(tensor_positions)]


# --- merge driver -----------------------------------------------------------


def _merge_graphs(
    graphs: Sequence[Graph],
    entries: Sequence[Optional[int]],
    out_entries: Sequence[Optional[int]],
    groups: Sequence[MergeGroup],
    bands: Any = None,
) -> Graph:
    """Validate the merge and build the merged graph (the two-phase core).

    Validation runs FIRST and in the documented order (program shape, the
    cross-function contract, per-op batchability, output pytree nodes), so a
    banded merge reports exactly the errors a flat one would.

    Then the partition is resolved: ``bands=None`` is the flat emitter;
    ``bands=K`` is `_bands.even_partition`; ``bands="auto"`` is
    `_bands.plan_auto_partition` under the static cost model. A SINGLE-band
    partition (`K=1`, or an auto resolution that keeps every function together)
    delegates to the flat emitter too, so its graph is byte-identical to
    `bands=None` — only the report's band fields differ.
    """
    for index, graph in enumerate(graphs):
        _program_function(graph, index)
    _check_contract(graphs)
    _check_ops(graphs)
    _check_output_nodes(graphs)
    n_functions = len(graphs)
    strategy, band_count = _bands.normalize_bands(bands, n_functions)
    if strategy == _bands.BAND_STRATEGY_FLAT:
        dag: MergeDAG = build_merge_dag(graphs, groups)
        return emit_merged_graph(graphs, dag, entries, out_entries, groups)
    # A per-band value cannot be a single unmapped output; checked before the
    # partition so `bands=1` obeys the same contract as any other band count.
    _bands.check_banded_out_entries(out_entries)
    cost_estimate = flat_cost_estimate = None
    if strategy == _bands.BAND_STRATEGY_AUTO:
        partition, cost_estimate, flat_cost_estimate = _bands.plan_auto_partition(
            graphs, entries, groups
        )
    else:
        partition = _bands.even_partition(n_functions, band_count)
    if len(partition) == 1:
        dag = build_merge_dag(graphs, groups)
        graph = emit_merged_graph(graphs, dag, entries, out_entries, groups)
        report = graph.batch_report
        report.bands = 1
        report.band_sizes = (n_functions,)
        report.band_strategy = strategy
        report.cost_estimate = cost_estimate
        report.flat_cost_estimate = flat_cost_estimate
        return graph
    return _bands.emit_banded_graph(
        graphs,
        partition,
        entries,
        out_entries,
        groups,
        strategy,
        cost_estimate,
        flat_cost_estimate,
    )
