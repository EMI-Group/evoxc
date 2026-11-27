"""Public `split_by_inputs` — split a graph at its input structure.

`split_by_inputs` slices ONE traced graph into TWO ordinary graphs, at the
boundary between inputs that stay FIXED across calls and inputs that VARY
per call:

* ``prepare`` — the fixed inputs' contribution. It computes, once, every
  value the varying half still needs, and returns them as a FLAT tuple of
  tensors (the *boundary*).
* ``apply`` — the varying inputs evaluated ON TOP of that prepared boundary.

Both halves are ordinary `Graph`s of ordinary ops, so every backend runs them
with no transform-specific support.

This is the FIRST structural (non-numeric) pass in `evoxc.transforms`: it
carries NO numerical authority (no derivatives, no batching, no computation)
and only slices the graph DATA STRUCTURE, so it lives in Python rather than in
the Rust engine (see `./CONTEXT.md`, "split_by_inputs").

Exact expansion (nothing hidden):

1. a callable / `Defn` input is traced exactly ONCE per call
   (``evoxc.trace(fn, *specs)``); a traced `Graph` input is used as-is;
2. the graph is GATED — v1 refuses region-bearing control flow
   (`cond` / `while_loop` / `scan`) and every effectful op (`collective`,
   `callback`, `custom_call`, …), see `_split_core.gates`;
3. dependence marking — each SSA value gets a bitset of the varying tensor
   inputs it depends on (`_split_core.dependence_masks`), and dead ops are
   dropped by backward reachability from the `return` operands
   (`_split_core.needed_set`);
4. boundary selection — the fixed-only values the varying half still consumes
   (`_split_core.select_boundary`, deterministic order);
5. `prepare` and `apply` are built as fresh modules by CLONING the needed ops
   (`_split_core.build_prepare` / `_split_core.build_apply`); the source graph
   is never mutated.

Runtime call shapes (binding):

.. code-block:: python

    plan = evoxc.split_by_inputs(graph, fixed={...})
    prepared = evoxc.run(prepare_executable, *fixed_args)   # a flat tuple
    outputs = evoxc.run(apply_executable, varying_tree, prepared)

``evoxc.run``'s root argument IS the args tuple, so `apply`'s signature is
``((varying inputs, ...), prepared_tuple)`` — see `SplitPlan.apply`'s
docstring for the exact input tree.

CORRECTNESS LAW: ``apply(prepare(f), v)`` agrees with ``f(f, v)`` under the
backend's normal numerical contract (the ops are the SAME ops, cloned, in the
same order). The boundary is a SNAPSHOT of the fixed half's contribution: it
is **not** re-derived per call, so changing a fixed input means re-running
`prepare` — the two graphs carry no per-turn state and no invalidation logic.

Worked example — a fitness graph ``f(dataset, params, candidate)`` where
``params`` is fixed and a candidate changes every evaluation::

    def fitness(dataset, params, candidate):
        return evoxc.sum(evoxc.tanh(evoxc.dot(dataset, candidate) * params))

    graph = evoxc.trace(
        fitness,
        evoxc.TensorSpec((768, 32), evoxc.float32),    # dataset  (varies)
        evoxc.TensorSpec((32,), evoxc.float32),        # params   (FIXED)
        evoxc.TensorSpec((32,), evoxc.float32),        # candidate(varies)
    )
    plan = evoxc.split_by_inputs(graph, fixed_leaves=[1])   # params fixed

    def stage(half):
        return evoxc.load(evoxc.compile(evoxc.lower(half, backend="numpy"),
                                       backend="numpy"),
                          backend="numpy")

    prepare, apply = stage(plan.prepare), stage(plan.apply)   # once per shape set
    prepared = evoxc.run(prepare, params)          # 1-tuple: the boundary tensors
    score = evoxc.run(apply, (dataset, candidate), prepared)

`prepare` is fed ONLY the fixed inputs (the input tree pruned to the fixed
tensor leaves, statics preserved), while `apply` takes the synthetic 2-child
root ``(varying inputs, prepared tuple)`` — so a re-run of `prepare` is what a
changed fixed input requires, and the backend/device in `lower`/`compile`/
`load` is the caller's choice (the example pins ``"numpy"`` only for
concreteness).

v1 limits (one line): region-bearing control flow and effectful ops are
blocked, there is no per-turn state tracking (a changed fixed input needs a
new `prepare` run), and `apply` takes the synthetic two-element signature
``(varying inputs, prepared tuple)``.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from evoxc import core
from evoxc.trace import Graph
from evoxc.trace import trace as trace_fn
from evoxc.trace._tree import _iter_leaf_paths

from evoxc.transforms import _split_core
from evoxc.transforms._split_core import _registered_base
from evoxc.transforms._split_types import SplitCallable, SplitPlan
from evoxc.transforms.vectorize import _broadcast_registered_axes, _is_int

__all__ = ["split_by_inputs"]


_SPLIT_CALLABLE_DOC = """Trace the wrapped function ONCE per call and split it.

``sc(*specs)`` — a structure of `core.TensorSpec`s (and static values)
matching the wrapped function's inputs — traces the function with
``evoxc.trace`` and returns the `SplitPlan` (see `evoxc.split_by_inputs`).
Concrete `core.Tensor` arguments raise `core.TraceError`: building a plan
never executes anything.
"""


def split_by_inputs(graph_or_fn, fixed=None, *, fixed_leaves=None):
    """Split a graph into a `prepare` / `apply` pair at its input structure.

    Args:
        graph_or_fn: a traced `evoxc.Graph` (split immediately → `SplitPlan`)
            or a plain callable / `@evoxc.defn` function (→ `SplitCallable`,
            the package's spec-callable convention: the function is traced
            exactly once per call and then split). Anything else raises
            `TypeError`.
        fixed: a pytree mask mirroring the graph's INPUT structure with
            ``True``/``False``/``None`` at tensor-leaf positions — ``True``
            keeps that tensor input in `prepare` (FIXED), ``False``/``None``
            moves it to `apply` (varying). A bare ``True``/``False``
            broadcasts across ALL tensor leaves; over a registered pytree
            node a single entry broadcasts across the node's leaves. A
            non-``None`` entry at a STATIC (non-tensor) leaf raises
            `core.TransformError` — a static value always specializes the
            graph and is never a boundary input. Exactly one of ``fixed`` /
            ``fixed_leaves`` must be given.
        fixed_leaves: a sequence of flat TENSOR-input indices (into
            ``graph.tensor_specs`` order) to keep in `prepare`. Non-int
            entries raise `TypeError`; an out-of-range or duplicate index
            raises `core.TransformError`.

    Returns:
        A `SplitPlan` (graph input) or a `SplitCallable` (callable input)
        whose ``__call__`` returns a `SplitPlan`. With NO fixed tensor leaf,
        or with ALL tensor leaves fixed, `core.TransformError` is raised —
        both halves must be non-trivial. A mask whose structure does not
        mirror the inputs raises `core.TransformError` naming the first
        mismatch pytree path; a graph with control-flow regions or an
        effectful op raises `core.TransformError` naming the op (and its
        source location).

    The expansion, the runtime call shapes (``run(prepare, *fixed_args)`` →
    flat tuple; ``run(apply, varying_tree, prepared_tuple)`` → the source
    output tree) and the correctness law are documented in this module's
    docstring; the source graph is never mutated.
    """
    if (fixed is None) == (fixed_leaves is None):
        raise TypeError(
            "split_by_inputs expects exactly ONE of `fixed` (a pytree "
            "bool/None mask mirroring the inputs) or `fixed_leaves` (a "
            "sequence of flat tensor-input indices); got "
            f"{'neither' if fixed is None else 'both'}"
        )
    # Cheap argument-kind validation up front, even for the callable path
    # (the range/duplicate checks need the traced graph and run per call).
    leaf_entries: Optional[Tuple[int, ...]] = (
        None if fixed_leaves is None else _leaf_index_tuple(fixed_leaves)
    )

    if isinstance(graph_or_fn, Graph):
        positions = _resolve_positions(graph_or_fn, fixed, leaf_entries)
        return _split_core.split_graph(graph_or_fn, positions)
    if callable(graph_or_fn):

        def trace_and_split(*specs) -> SplitPlan:
            graph = trace_fn(graph_or_fn, *specs)
            positions = _resolve_positions(graph, fixed, leaf_entries)
            return _split_core.split_graph(graph, positions)

        return SplitCallable(trace_and_split, doc=_SPLIT_CALLABLE_DOC)
    raise TypeError(
        "split_by_inputs expects a traced evoxc.Graph or a callable/@evoxc.defn "
        f"function, got {type(graph_or_fn).__name__} (call evoxc.trace first to "
        "split an already-built graph)"
    )


# ---------------------------------------------------------------------------
# fixed-leaf resolution (mask -> tensor positions, or explicit indices)
# ---------------------------------------------------------------------------


def _resolve_positions(
    graph: Graph, fixed: Any, leaf_entries: Optional[Tuple[int, ...]]
) -> Tuple[int, ...]:
    """The fixed TENSOR-input positions, from either argument form."""
    if leaf_entries is not None:
        return _positions_from_leaves(graph, leaf_entries)
    return _positions_from_mask(graph, fixed)


def _leaf_index_tuple(fixed_leaves: Any) -> Tuple[int, ...]:
    """Validate the argument KIND of `fixed_leaves` (a sequence of ints)."""
    if isinstance(fixed_leaves, (str, bytes)):
        raise TypeError(
            "split_by_inputs: `fixed_leaves` must be a sequence of flat "
            f"tensor-input indices, got a {type(fixed_leaves).__name__}"
        )
    try:
        entries = tuple(fixed_leaves)
    except TypeError:
        raise TypeError(
            "split_by_inputs: `fixed_leaves` must be a sequence of flat "
            f"tensor-input indices, got {type(fixed_leaves).__name__}"
        ) from None
    for entry in entries:
        if not _is_int(entry):
            raise TypeError(
                "split_by_inputs: `fixed_leaves` entries must be plain ints "
                f"(indices into the graph's tensor inputs), got {entry!r} of "
                f"type {type(entry).__name__}"
            )
    return entries


def _positions_from_leaves(
    graph: Graph, entries: Sequence[int]
) -> Tuple[int, ...]:
    """Validate explicit tensor-input indices (range, duplicates; sorted)."""
    count = len(graph.tensor_specs)
    seen = set()
    for entry in entries:
        if entry < 0 or entry >= count:
            raise core.TransformError(
                f"split_by_inputs: fixed tensor-leaf index {entry} is out of "
                f"range for a graph with {count} tensor inputs (valid indices "
                f"are 0..{count - 1})"
            )
        if entry in seen:
            raise core.TransformError(
                f"split_by_inputs: duplicate fixed tensor-leaf index {entry} "
                f"in fixed_leaves"
            )
        seen.add(entry)
    return tuple(sorted(seen))


def _tree_has_registered_node(tree_spec: "core.TreeSpec") -> bool:
    """True iff the input tree contains a registered pytree node anywhere."""
    if _registered_base(tree_spec.type) is not None:
        return True
    return any(_tree_has_registered_node(child) for child in tree_spec.children)


def _positions_from_mask(graph: Graph, mask: Any) -> Tuple[int, ...]:
    """Flatten + validate the `fixed` mask into fixed tensor-input positions.

    The mask is indexed by TREE leaf position (pre-order over
    ``graph.input_specs``); the returned positions index
    ``graph.tensor_specs``. The conversion counts the non-static tree
    positions before each tensor leaf — static positions are exactly
    ``{record.index for record in graph.static_values}``.
    """
    n_leaves = graph.input_specs.num_leaves
    static_positions = {record.index for record in graph.static_values}
    tensor_positions = set(range(n_leaves)) - static_positions

    if isinstance(mask, bool) or mask is None:
        # A bare entry broadcasts across ALL tensor leaves (static leaves get
        # None: a static value is never a boundary input).
        entries = [
            mask if isinstance(mask, bool) and index in tensor_positions else None
            for index in range(n_leaves)
        ]
    else:
        mask_leaves, mask_spec = core.flatten(mask)
        if _tree_has_registered_node(graph.input_specs):
            entries, mismatch = _broadcast_registered_axes(
                mask_spec,
                graph.input_specs,
                mask_leaves,
                tensor_positions,
                "split_by_inputs",
            )
        else:
            mismatch = core.first_mismatch_path(
                mask_spec, graph.input_specs, leaf_vs_empty_is_mismatch=True
            )
            entries = list(mask_leaves) if mismatch is None else []
        if mismatch is not None:
            raise core.TransformError(
                "split_by_inputs: the `fixed` mask does not match the graph's "
                f"input structure — first mismatch at pytree path "
                f"{core.format_path(mismatch)}; `fixed` must be a pytree with "
                "the same container structure as the inputs (True keeps a "
                "tensor leaf in `prepare`, False/None leaves it varying, and "
                "None is required at static leaves)"
            )

    paths = list(_iter_leaf_paths(graph.input_specs))
    fixed = []
    tensor_index = -1
    for index, entry in enumerate(entries):
        if index in static_positions:
            if entry is not None:
                raise core.TransformError(
                    f"split_by_inputs: the `fixed` mask entry at pytree path "
                    f"{core.format_path(paths[index])} applies to a STATIC "
                    f"input leaf ({entry!r}); a static value is always part of "
                    f"the graph's specialization, never a boundary input — use "
                    f"None there"
                )
            continue
        tensor_index += 1
        if entry is None:
            continue
        if not isinstance(entry, bool):
            raise core.TransformError(
                f"split_by_inputs: the `fixed` mask entry at pytree path "
                f"{core.format_path(paths[index])} (tensor input "
                f"{tensor_index}) must be True, False or None, got {entry!r} "
                f"of type {type(entry).__name__}"
            )
        if entry:
            fixed.append(tensor_index)
    return tuple(fixed)
