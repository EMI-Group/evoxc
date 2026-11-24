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

The merge is the **native engine's** (`evoxc._evoxc_native.batch_functions`,
reached through `_native_batch`); Python keeps ONLY the frontend work:

* `fns` / `axes` / `out_axes` / `switch_ops` / `bands` / `alignment` validation
  (all of it BEFORE any tracing or native call, with the pinned error wording);
* the callable path (`TransformCallable`, strip → trace each function once);
* the cross-function contract and the program-shape / output-pytree checks,
  whose messages carry the pytree paths only Python knows;
* `Graph` reconstruction from the decoded merged module (input specs, trees,
  static values, input source locations) and `graph.batch_report` rehydration.

The merge strategy itself lives in ``native/evoxc-compiler/src/batch_functions``
(never a silent Python-loop fallback):

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
BYTE-IDENTICAL to the pre-`bands` output; the partition rules and the static
cost model behind `bands="auto"` are the native engine's.

`alignment=` chooses how a `switch_ops` position's class is picked — the only
positions where two programs can place DIFFERENT ops at one skeleton.
`"variant"` (the default) lets a variant an existing class already holds win
over the occurrence-position class (sibling branches like `(x+1) * (x-1)` vs
`(x-1) * (x+1)` then get one class per op instead of two mixed classes);
`"occurrence"` keeps the k-th distinct variant of a program at the k-th
occurrence class, i.e. the byte-identical legacy output; `"auto"` is
variant-first with the occurrence fallback plus a last-resort reuse of any other
free class so it never creates more classes than `"occurrence"`.
It is a PURE PERFORMANCE heuristic, so every mode produces bit-identical
numbers; constants are never aligned (they stack by occurrence, which is what
folds N payloads into one stacked constant).

The returned `Graph` carries `graph.batch_report` — a `BatchReport` describing
what the merge did (`BatchReport` / `SwitchRecord` live in `_report` and are
re-exported here; the native engine reports the numbers and this module
rehydrates them into the dataclasses). `graph.source_locations` carries the
contributing INPUT values' locations forward (the merged module's own ops carry
the contributing ops' locations, stamped by the native engine).
"""

from __future__ import annotations

import collections.abc
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evoxc import core
from evoxc import ir
from evoxc.trace import Graph, trace
from evoxc.trace._tree import _iter_leaf_paths, _static_equal
from evoxc.transforms import batching
from evoxc.transforms import _native_batch, _native_rules
from evoxc.transforms._report import BatchReport, SwitchRecord, report_from_native
from evoxc.transforms._wrappers import TransformCallable
from evoxc.transforms.vectorize import (
    _is_int,
    _normalize_axes,
    _normalize_axis_entries,
)
from evoxc.transforms.vmap import _derive_unvectorized_args_and_entries

__all__ = ["batch_functions", "BatchReport", "SwitchRecord"]

#: The three accepted `alignment` spellings, in preference order.
ALIGNMENT_VARIANT = "variant"
ALIGNMENT_OCCURRENCE = "occurrence"
ALIGNMENT_AUTO = "auto"
ALIGNMENTS: Tuple[str, ...] = (
    ALIGNMENT_VARIANT,
    ALIGNMENT_OCCURRENCE,
    ALIGNMENT_AUTO,
)

#: The default `alignment` policy (the `"variant"` heuristic — a pure
#: performance choice; every mode is row-correct).
DEFAULT_ALIGNMENT = ALIGNMENT_VARIANT

#: The merge groups `switch_ops=True` opts into (IR category names).
DEFAULT_SWITCH_CATEGORIES: Tuple[str, ...] = ("elementwise", "comparison")

BAND_STRATEGY_FLAT = "flat"
BAND_STRATEGY_EXPLICIT = "explicit"
BAND_STRATEGY_AUTO = "auto"
AUTO = "auto"

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
    alignment: Any = DEFAULT_ALIGNMENT,
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
            trailing misc band. Banding trades dispatches for wasted work: a
            divergent position emits one op per DISTINCT variant of its band
            instead of one per variant of the whole population, at the cost of
            re-computing cross-band sharing plus one input slice and
            (collectively) one output concatenation per band. `K=1` and an
            auto-resolved single band emit the flat graph. Validated here, so a
            bad value raises before any tracing (callable path included).
            v1 restriction: `bands` other than `None` requires `out_axes=0` for
            every tensor output (a per-band value cannot be one unmapped
            output) — `core.TransformError` otherwise.
        alignment: how a `switch_ops` position's CLASS is chosen — i.e. which
            emitted op a program's op joins when two programs put different ops
            at one skeleton. `"variant"` (default) = a variant an EXISTING class
            of that skeleton already holds (and which does not yet hold this
            program) wins over the occurrence-position class, so sibling
            branches such as ``(x+1) * (x-1)`` vs ``(x-1) * (x+1)`` get one
            class per op instead of two mixed classes; `"occurrence"` = the k-th
            distinct variant of a program at a skeleton joins the k-th class
            created for that skeleton (the byte-identical legacy output);
            `"auto"` = variant-first with the occurrence-position fallback, plus
            a last-resort reuse of any other free class of the skeleton, so it
            never creates more classes than `"occurrence"`. Case-insensitive.
            Only switch-group positions are aligned: non-switched positions have
            one variant per skeleton by construction and CONSTANTS stack by
            occurrence (payload deliberately
            outside the identity), which is what folds N distinct payloads into
            ONE stacked constant. The class assignment is a PURE PERFORMANCE
            heuristic — every assignment is row-correct because a class's
            operands are shared by slot equality — so `alignment` can never
            change the numbers, only the op/select count. Composes with `bands`
            (per-band class DAGs use the same alignment).

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
            malformed, `bands` is a bool / an unsupported type, or `alignment`
            is not a string.
        core.TraceError: mismatched input/output tree structure or static
            leaves between functions.
        core.DTypeError / core.ShapeError / core.DeviceError: per-leaf dtype /
            shape / device mismatch between functions.
        core.TransformError: invalid `axes`/`out_axes` entries, an unknown
            `bands` value or a band count outside ``1..N``, an unknown
            `alignment` spelling, `bands` combined
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
    # `bands` and `alignment` are validated BEFORE any tracing (the callable
    # path included), so a bad band count or alignment spelling can never leave
    # half-traced programs behind.
    _normalize_bands(bands, len(items))
    alignment = _normalize_alignment(alignment)
    switch_spec = _normalize_switch_ops(switch_ops)
    if kind == "graph":
        try:
            normalized = _normalize_axes(items[0], axes)
        except core.TransformError as exc:
            raise core.TransformError(
                "batch_functions: invalid axes: " + str(exc)
            ) from exc
        entries = batching._tensor_input_axes(items[0], normalized)
        out_entries = _normalize_out_axes(items[0], out_axes)
        return _merge_graphs(
            items, entries, out_entries, switch_spec, bands, alignment
        )
    return TransformCallable(
        build=lambda *args: _batch_callables(
            items, args, axes, out_axes, switch_spec, bands, alignment
        ),
        kind="batch_functions",
        doc=_BATCH_FUNCTIONS_CALLABLE_DOC,
    )


# --- callable path ----------------------------------------------------------


def _batch_callables(
    fns, args, axes, out_axes, switch_spec, bands=None, alignment=DEFAULT_ALIGNMENT
) -> Graph:
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
    return _merge_graphs(
        graphs, entries, out_entries, switch_spec, bands, alignment
    )


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


# --- `bands` / `alignment` / `switch_ops` validation (before any tracing) ----


def _normalize_bands(bands: Any, n: int) -> Tuple[str, Optional[int]]:
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


def _check_banded_out_entries(out_entries: Sequence[Optional[int]]) -> None:
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


def _normalize_alignment(alignment: Any) -> str:
    """Normalize `alignment` into one of `ALIGNMENTS` (case-insensitive,
    whitespace-stripped), rejecting anything else LOUDLY.

    Raises:
        TypeError: `alignment` is not a string.
        core.TransformError: `alignment` is not an accepted spelling.
    """
    if not isinstance(alignment, str):
        raise TypeError(
            "batch_functions: alignment must be a string, one of "
            f"{list(ALIGNMENTS)}; got {type(alignment).__name__}"
        )
    normalized = alignment.strip().lower()
    if normalized not in ALIGNMENTS:
        raise core.TransformError(
            f"batch_functions: unknown alignment {alignment!r}; accepted forms "
            f"are {list(ALIGNMENTS)}"
        )
    return normalized


def _op_categories() -> frozenset:
    """The set of IR op category names (`{o.category for o in all_opdefs()}`)."""
    return frozenset(definition.category for definition in ir.all_opdefs())


def _validate_switch_category(category: str) -> str:
    """Validate one category merge group against the IR op registry."""
    if not isinstance(category, str) or not category:
        raise TypeError(
            "batch_functions: switch_ops category entries must be non-empty "
            f"strings, got {category!r}"
        )
    valid = _op_categories()
    if category not in valid:
        raise core.TransformError(
            f"batch_functions: unknown switch_ops category {category!r}; "
            f"valid categories are {sorted(valid)}"
        )
    return category


def _validate_switch_names(names) -> List[str]:
    """Validate one explicit op-name merge group against the IR registry."""
    try:
        items = list(names)
    except TypeError as exc:
        raise TypeError(
            "batch_functions: a switch_ops entry must be a category name "
            "(str) or an iterable of op names, got "
            f"{type(names).__name__}"
        ) from exc
    if not items:
        raise core.TransformError(
            "batch_functions: empty switch_ops merge group — an explicit "
            "group must name at least one op"
        )
    for name in items:
        if not isinstance(name, str):
            raise TypeError(
                "batch_functions: switch_ops op-name entries must be strings, "
                f"got {name!r}"
            )
        if not ir.has_opdef(name):
            raise core.TransformError(
                f"batch_functions: unknown op name {name!r} in a switch_ops "
                "merge group; names must be canonical IR op names "
                "(see ir.op_names())"
            )
    return items


def _normalize_switch_ops(switch_ops) -> Any:
    """Validate `switch_ops` and return the native config form.

    * ``None`` / ``False`` → `None` (no merge groups, no op switching);
    * ``True`` → `True` (the default groups, one per
      `DEFAULT_SWITCH_CATEGORIES` entry — ``elementwise`` and ``comparison``);
    * any other iterable (a `str`/`bytes` is rejected) → a `list` of merge
      groups, each a validated category name (`str`) or a validated `list` of
      IR op names — exactly the shape the native engine's config accepts.

    Raises:
        TypeError: `switch_ops` is a str/bytes or not iterable, or an element
            is neither a str nor an iterable of str.
        core.TransformError: an unknown category, an empty/unknown op-name
            group.
    """
    if switch_ops is None or switch_ops is False:
        return None
    if switch_ops is True:
        return True
    if isinstance(switch_ops, (str, bytes, bytearray)):
        raise TypeError(
            "batch_functions: switch_ops must be None, False, True, or an "
            "iterable of merge groups (category names / iterables of op "
            f"names); got {type(switch_ops).__name__}"
        )
    try:
        items = list(switch_ops)
    except TypeError as exc:
        raise TypeError(
            "batch_functions: switch_ops must be None, False, True, or an "
            "iterable of merge groups (category names / iterables of op "
            f"names); got {type(switch_ops).__name__}"
        ) from exc
    groups: List[Any] = []
    for item in items:
        if isinstance(item, str):
            groups.append(_validate_switch_category(item))
        elif isinstance(item, (bytes, bytearray)):
            raise TypeError(
                "batch_functions: switch_ops entries must be a category name "
                f"(str) or an iterable of op names, got {type(item).__name__}"
            )
        else:
            groups.append(_validate_switch_names(item))
    return groups


# --- cross-function contract validation -------------------------------------


#: Memo of ``op name -> ir.OpDef`` — avoids the per-op lazy `evoxc.ir.op_defs`
#: import inside ``Op.opdef``/``Op.is_terminator``. Op defs are registered once
#: at `evoxc.ir` import time and never re-registered, so the mapping is
#: immutable.
_OPDEF_CACHE: Dict[str, Any] = {}


def _opdef(name: str) -> Any:
    """The registered `ir.OpDef` for `name`, memoized (raises `KeyError` if absent)."""
    definition = _OPDEF_CACHE.get(name)
    if definition is None:
        definition = _OPDEF_CACHE[name] = ir.opdef(name)
    return definition


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


_NATIVE_BATCHABLE: frozenset[str] | None = None


def _native_batchable() -> frozenset[str]:
    """The native engine's builtin batching rule keys (lazily, once)."""
    global _NATIVE_BATCHABLE
    if _NATIVE_BATCHABLE is None:
        from evoxc import _native

        if not _native.available():
            _NATIVE_BATCHABLE = frozenset()
        else:
            _NATIVE_BATCHABLE = frozenset(_native.batching_rule_keys())
    return _NATIVE_BATCHABLE


def _check_ops(graphs: Sequence[Graph]) -> None:
    """Every op in every program must be batchable (or a `constant`).

    Region-bearing control flow (`cond`/`while_loop`/`scan`) and every op
    without a registered batching rule (looked up through
    `batching._rule_key`, so `block:<name>` / `external:<name>` namespacing
    works) is a v1 deferral of `batch_functions` — an explicit
    `core.TransformError` naming the op and the function index, never a silent
    Python-loop fallback. A key is available when the Python registry holds it
    OR the native engine's compiled-in builtin set does, so a native-only rule
    can never be rejected by a stale mirror; host-namespace rules
    (`block:*` / `external:*` / `sparse_*`) live only in the Python registry.

    The native engine runs the same check with the same wording; it stays here
    because it must fire BEFORE the banded-`out_axes` restriction (the standard
    merge validation always wins over a band-strategy restriction), and because
    it names the Python-registered rules the engine only sees through the
    extension bridge.
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
            if key in batching.batching_rules or key in _native_batchable():
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

    The pytree must match the graph's output structure (first-mismatch-path
    `core.TransformError` otherwise), static output leaves must be `None`,
    entries must be ``{0, None}`` (non-zero ints are the v1 non-leading-axes
    deferral), and a bare int/None is accepted only with at most ONE tensor
    output. No rearrangement ops are emitted — the caller only reads the
    entries.
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
    switch_spec: Any,
    bands: Any = None,
    alignment: str = DEFAULT_ALIGNMENT,
) -> Graph:
    """Validate the merge, run the native merge, rebuild the merged `Graph`.

    Validation runs FIRST and in the documented order (program shape, the
    cross-function contract, per-op batchability, output pytree nodes), then the
    banded-`out_axes` restriction, so every rejection is the Python-side one
    with its pinned message — before the native engine is asked to do anything.

    The engine (`evoxc._evoxc_native.batch_functions`, through
    `_native_batch.batch_functions`) receives the encoded programs plus the
    already-normalized config (``axes`` / ``out_axes`` one entry per tensor
    leaf, the validated ``switch_ops`` spec, ``bands``, the canonical
    ``alignment`` and the Python extension-rule bridge) and returns the merged
    module plus its 12-key report. Python then rebuilds the `Graph`: input specs
    are function 0's (a MAPPED leaf gains a STATIC leading ``int`` function
    axis, never a symbolic `Dim`; an UNMAPPED leaf reuses its source spec
    object), output tree / static values are function 0's, input source
    locations are carried positionally, and `batch_report` is rehydrated into
    the public dataclasses.
    """
    for index, graph in enumerate(graphs):
        _program_function(graph, index)
    _check_contract(graphs)
    _check_ops(graphs)
    _check_output_nodes(graphs)
    n_functions = len(graphs)
    strategy, _band_count = _normalize_bands(bands, n_functions)
    if strategy != BAND_STRATEGY_FLAT:
        # A per-band value cannot be a single unmapped output; checked after
        # the standard merge validation and before the partition, so `bands=1`
        # obeys the same contract as any other band count.
        _check_banded_out_entries(out_entries)

    config: Dict[str, Any] = {
        "axes": list(entries),
        "out_axes": list(out_entries),
        "switch_ops": switch_spec,
        "bands": bands,
        "alignment": alignment,
        "extension_rule": _native_rules.extension_rule,
    }
    result = _native_batch.batch_functions(
        [graph.module for graph in graphs], config
    )
    module = result["module"]
    report = report_from_native(result["batch_report"])
    if report.n_functions != n_functions:
        raise core.TransformError(
            "native batch_functions: the report describes "
            f"{report.n_functions} function(s) but {n_functions} were merged"
        )
    graph = _rebuild_merged_graph(
        graphs[0], module, entries, n_functions, result.get("input_ids")
    )
    graph.batch_report = report
    return graph


def _rebuild_merged_graph(
    graph0: Graph,
    module: ir.Module,
    entries: Sequence[Optional[int]],
    n_functions: int,
    input_ids: Optional[Sequence[int]] = None,
) -> Graph:
    """Wrap the decoded merged module back into a `Graph`.

    Input specs are derived from function 0's specs (never re-built from the
    decoded types, so symbolic row dims and spec identity survive): a mapped
    leaf gains a STATIC leading ``n_functions`` dim, an unmapped leaf reuses the
    source spec object unchanged (dtype/device/name preserved throughout).
    Input source locations are carried positionally onto the merged block
    arguments (``input_ids`` when the engine reports them, otherwise the
    decoded block arguments — the merged block args ARE the tensor inputs, in
    order).
    """
    function = module.main
    block = function.entry_block
    if len(block.arguments) != len(graph0.tensor_specs):
        raise core.TransformError(
            "native batch_functions: the merged function has "
            f"{len(block.arguments)} block args but the input graph has "
            f"{len(graph0.tensor_specs)} tensor inputs"
        )
    new_specs: List[core.TensorSpec] = []
    for spec, entry in zip(graph0.tensor_specs, entries):
        if entry is None:
            new_specs.append(spec)
            continue
        new_specs.append(
            core.TensorSpec(
                shape=(n_functions, *spec.shape),
                dtype=spec.dtype,
                device=spec.device,
                name=spec.name,
            )
        )
    if input_ids is None:
        input_ids = [argument.id for argument in block.arguments]
    elif len(input_ids) != len(block.arguments):
        raise core.TransformError(
            "native batch_functions: the merge metadata names "
            f"{len(input_ids)} input ids but the merged function has "
            f"{len(block.arguments)} block args"
        )
    source_locations: Dict[int, Any] = {}
    source_block = graph0.module.main.entry_block
    for old_argument, new_id in zip(source_block.arguments, input_ids):
        location = graph0.source_locations.get(old_argument.id)
        if location is not None:
            source_locations[new_id] = location
    return Graph(
        module,
        graph0.input_specs,
        new_specs,
        graph0.output_tree,
        graph0.static_values,
        graph0.output_static_values,
        source_locations,
    )
