"""Public `vmap` — function-side sugar over `vectorize`.

`vmap` uses exactly the same batching machinery as `vectorize`; the
equivalence contract (binding, tested) lives in `./CONTEXT.md`:

* `vmap(graph, in_axes, out_axes=0)` ≡ `vectorize(graph, in_axes)` followed by
  an output-axis rearrangement step (ordinary transpose / size-one axis
  insertion ops) driven by `out_axes`; with `out_axes=0` both produce
  identical IR.
* `vmap(fn_or_defn, in_axes, out_axes)` returns a `TransformCallable`:
  `vmap(f)(*batched_specs)` ≡ rearrange(vectorize(evoxc.trace(f,
  *strip(batched_specs)), in_axes), out_axes), where `strip` removes the
  leading mapped dim from each mapped spec (dtype/device/name preserved).
  The wrapped function is traced exactly ONCE per call.
"""

from __future__ import annotations

import re

from evoxc import core
from evoxc.trace import Graph, trace
from evoxc.trace.graph import _normalize_leaf_types
from evoxc.transforms._wrappers import TransformCallable
from evoxc.transforms._native_bridge import vmap_graph as _native_vmap_graph
from evoxc.transforms.batching import _tensor_input_axes, with_batch_depth
from evoxc.transforms.vectorize import (
    _broadcast_registered_axes,
    _is_int,
    _normalize_axis_entries,
    _normalize_axes,
    _registered_node_tensor_positions,
    _tree_has_registered_node,
    vectorize,
)

_VMAP_CALLABLE_DOC = """\
`vmap(f)` applied to a callable/`Defn` returns a TransformCallable:
calling it with a structure of TensorSpecs (mapped inputs include the leading
batch dim) + static values returns the vectorized Graph, tracing `f` exactly
once. Equivalent to: trace `f` with the leading mapped dim stripped from each
mapped spec, then vectorize, then rearrange outputs per out_axes.

`vmap(tf)` applied to another TransformCallable (composition — nested vmap,
`vmap(grad(f))`, ...) also returns a TransformCallable: calling it invokes
the inner callable with the stripped specs (which traces its own wrapped
function exactly once) and vectorizes the returned Graph with the outer
axes, then rearranges outputs per out_axes — each level adds one leading
mapped dim."""

#: Reserved name pattern of the fresh batch dims `vectorize_graph` introduces
#: (ONE shared `Dim` per pass, named from the active pass depth — `"batch"`
#: at depth 0, `"batch_1"` at depth 1, ... — see
#: `batching._batched_input_specs`). These are the ONLY dims mapped-output
#: detection keys on (by identity); the `_\d+` suffix stays accepted for dims
#: named `batch_1` in pre-existing or user-built graphs.
_BATCH_DIM_NAME = re.compile(r"^batch(_\d+)?$")


def vmap(fn_or_graph, in_axes=0, out_axes=0):
    """Vectorize a function or graph over a leading axis (function-side sugar).

    Args:
        fn_or_graph: a `Graph` (vectorized directly), or a callable/`Defn`
            (returned as a `TransformCallable`; see below).
        in_axes: `int`, `None`, or a pytree thereof matching the inputs — the
            axis of each input that becomes the vectorized (leading) axis;
            `None` leaves an input unmapped. v1 supports entries in {None, 0};
            other ints raise `core.TransformError` (deferred).
        out_axes: `int`, `None`, or a pytree thereof matching the outputs —
            where each mapped output axis should end up. `0` keeps it leading
            (an unmapped output gets an explicitly requested size-one axis at
            0); `None` requires the output to be unmapped, otherwise
            `core.TransformError` (axis mismatch). v1: {None, 0}.

    Returns:
        * `Graph` when given a `Graph` — the vectorized graph with outputs
          rearranged per `out_axes`.
        * `TransformCallable` when given a callable/`Defn` — `vmap(f)(*specs)`
          returns the vectorized `Graph`; `f` is invoked exactly once, in
          graph-building mode. Passing concrete tensors raises `TraceError`
          (transforms never execute; build the returned graph explicitly).
        * `TransformCallable` when given another `TransformCallable`
          (composition — `vmap(vmap(f))`, `vmap(grad(f))`, ...): calling the
          result invokes the inner callable with the stripped specs (it traces
          its wrapped fn exactly once), then vectorizes the returned graph
          with the outer `in_axes` and rearranges per `out_axes`. Nested
          levels compose mechanically, each adding one leading mapped dim.

    Unsupported ops without a batching rule raise `core.TransformError`
    naming the op — never a silent fallback. Nested vmap is supported when the
    required batching rules exist (each level adds one leading mapped dim).
    """
    if isinstance(fn_or_graph, Graph):
        return _vmap_graph(fn_or_graph, in_axes, out_axes)

    if callable(fn_or_graph) or getattr(fn_or_graph, "__evoxc_defn__", False):
        return TransformCallable(
            build=lambda *args: _vmap_fn(fn_or_graph, args, in_axes, out_axes),
            kind="vmap",
            doc=_VMAP_CALLABLE_DOC,
        )

    raise TypeError("vmap expects an evoxc.Graph, a callable, or an evoxc.Defn")


def _vmap_fn(fn, args, in_axes, out_axes) -> Graph:
    """Build the `TransformCallable` result for one `vmap(f)(*args)` call.

    Plain callable/`Defn`: trace `fn` exactly once with unvectorized specs
    derived from `args` (strip the leading mapped dim from each mapped spec
    per `in_axes`), then vectorize and rearrange outputs per `out_axes`.

    `TransformCallable` (`vmap` of another transform — e.g. `vmap(vmap(f))`,
    `vmap(grad(f))`): composition. The inner callable already maps stripped
    specs to a `Graph`, so it is INVOKED with the stripped argument structure
    (tracing its own wrapped fn exactly once) instead of being traced here;
    the outer `in_axes` is then applied to that graph via `vectorize` — the
    inner graph's input tree has the same structure as the stripped args, so
    `in_axes` re-normalizes cleanly — followed by the usual output
    rearrangement. Depth-3+ nesting recurses through the same path (each
    nested `vmap` layer is itself a `TransformCallable`).
    """
    unvectorized = _derive_unvectorized_args(args, in_axes)
    if isinstance(fn, TransformCallable):
        # Composition (nested vmap): the inner callable runs one level deeper
        # so its vectorize pass names its batch dim `batch_1`, `batch_2`, ...
        # while the outer pass below runs at the parent depth and keeps the
        # current level's name — the numpy interpreter binds symbolic dims BY
        # NAME, so same-named dims from different nesting levels would
        # collide at run time (unequal extents raise a misleading ShapeError).
        with with_batch_depth():
            inner_graph = fn(*unvectorized)
        return _vmap_graph(inner_graph, in_axes, out_axes)
    graph = trace(fn, *unvectorized)
    return _vmap_graph(graph, in_axes, out_axes)


def _vmap_graph(graph: Graph, in_axes, out_axes) -> Graph:
    """`vmap`'s graph composition: validate/normalize, then let native rewrite.

    Python keeps every frontend validation (input axes normalization, output
    axes structure/`{0, None}` rules) with the historical message wording;
    the op rewrite AND the output rearrangement (leading transpose / size-one
    axis insertion / the "a batch axis is never silently dropped" error) are
    the native engine's (see `_native_bridge.vmap_graph`).
    """
    tensor_entries = _tensor_input_axes(graph, _normalize_axes(graph, in_axes))
    out_entries = _normalize_out_axes(graph, out_axes)
    return _native_vmap_graph(graph, tensor_entries, out_entries)


def _derive_unvectorized_args(args, in_axes):
    """Given the batched specs/static values `args` and `in_axes`, return the
    underlying (unvectorized) spec structure for tracing the wrapped function:
    mapped entries drop their leading dim (dtype/device/name preserved);
    unmapped entries pass through; static values stay static.

    `in_axes` follows the same normalization rules as
    `vectorize._normalize_axes` (pytree matching `args` — with registered
    pytree-node leaf broadcast, e.g. a sparse-tensor spec node — bare int/None
    only with exactly one tensor spec or a registered-node-containing
    argument tree; v1 entries {None, 0}; static leaves None).
    Concrete `core.Tensor` leaves raise `core.TraceError` (transforms never
    execute; `TransformCallable` performs the same check at its boundary).
    """
    return _derive_unvectorized_args_and_entries(args, in_axes, "vmap")[0]


def _derive_unvectorized_args_and_entries(args, in_axes, what="vmap"):
    """`_derive_unvectorized_args` + the per-tensor-leaf axes entries.

    Returns ``(unvectorized_args, entries)`` where `entries` holds one ``0``
    (mapped) / ``None`` (unmapped) per TENSOR spec leaf, in
    `graph.tensor_specs` order — the pre-normalized form `batch_functions`
    needs (it cannot re-normalize against the traced graph: tracing STRIPS the
    leading dim, so an `axes=0` on a scalar-per-function input would fail the
    rank check). `what` prefixes the error messages ("vmap" by default).
    """
    args_leaves, args_spec = core.flatten(args)
    for index, leaf in enumerate(args_leaves):
        if isinstance(leaf, core.Tensor):
            raise core.TraceError(
                f"calling an evoxc.{what} result with concrete Tensors is not "
                f"supported (argument leaf {index}): transforms build Graphs "
                "from TensorSpecs and never execute. Pass TensorSpecs to "
                "obtain the transformed Graph, then build/run it explicitly "
                "(evoxc.build / evoxc.run)."
            )
    # `core.flatten` records a `TensorSpec` (an evoxc dataclass) as a leaf VALUE
    # but classifies the leaf *spec* as an empty container, so `core.unflatten`
    # would try to rebuild it as a dataclass node — normalize the leaf types
    # first (the same trick evoxc/trace/graph.py uses for output trees).
    args_spec = _normalize_leaf_types(args_spec)
    tensor_positions = {
        index
        for index, leaf in enumerate(args_leaves)
        if isinstance(leaf, core.TensorSpec)
    }
    ranks = {position: args_leaves[position].rank for position in tensor_positions}
    has_registered = _tree_has_registered_node(args_spec)

    if in_axes is None or _is_int(in_axes):
        if len(tensor_positions) != 1 and not has_registered:
            raise core.TransformError(
                f"{what}: a bare int/None in_axes applies only when the "
                "arguments have exactly ONE tensor spec; got "
                f"{len(tensor_positions)} tensor specs — pass a pytree "
                "matching the argument structure instead"
            )
        entries = [
            in_axes if index in tensor_positions else None
            for index in range(len(args_leaves))
        ]
    else:
        axes_leaves, axes_spec = core.flatten(in_axes)
        if has_registered:
            entries, mismatch = _broadcast_registered_axes(
                axes_spec, args_spec, axes_leaves, tensor_positions, what
            )
        else:
            mismatch = core.first_mismatch_path(
                axes_spec, args_spec, leaf_vs_empty_is_mismatch=True
            )
            entries = list(axes_leaves) if mismatch is None else None
        if mismatch is not None:
            raise core.TransformError(
                f"{what}: the in_axes pytree does not match the argument "
                f"structure — first mismatch at pytree path "
                f"{core.format_path(mismatch)}; in_axes must be a pytree with "
                "the same container structure as the arguments (0 maps a "
                "tensor spec's leading axis, None leaves it unmapped)"
            )

    normalized = _normalize_axis_entries(
        entries, tensor_positions, ranks, what
    )
    tensor_entries = [normalized[position] for position in sorted(tensor_positions)]
    # Tensor leaves UNDER a registered pytree node (e.g. a sparse-tensor
    # node's indices/values leaves) keep their shape UNCHANGED when mapped:
    # their leading dim is NOT a batch dim (sparse COO indices are
    # (None, ndim) — dim 0 is the runtime-dynamic nnz). The batch dim is
    # prepended later by vectorize (`_batched_input_specs`), and the
    # output-side dense_shape remap is `register_batched_aux_remap`'s job.
    under_registered = _registered_node_tensor_positions(
        args_spec, tensor_positions
    )
    unvectorized_leaves = []
    for index, leaf in enumerate(args_leaves):
        if (
            index in tensor_positions
            and normalized[index] == 0
            and index not in under_registered
        ):
            spec = leaf
            unvectorized_leaves.append(
                core.TensorSpec(
                    shape=spec.shape[1:],
                    dtype=spec.dtype,
                    device=spec.device,
                    name=spec.name,
                )
            )
        else:
            unvectorized_leaves.append(leaf)
    return core.unflatten(unvectorized_leaves, args_spec), tensor_entries


# --- output-axis rearrangement ----------------------------------------------


def _collect_batch_dims(graph: Graph):
    """The fresh batch `Dim` objects this vectorization introduced.

    `vectorize_graph` prepends ONE fresh symbolic batch `Dim` (named from the
    active pass depth — `"batch"` at depth 0, `"batch_1"` at depth 1, ...)
    shared by all mapped input specs of the pass (unmapped specs are reused
    unchanged), so the batch dims of THIS vectorization are exactly the
    leading entries of the input specs whose names follow that reserved
    pattern. Collected as objects — nested vmap levels create their OWN fresh
    Dims with level-distinct names, and only the current level's objects must
    count (output detection compares by identity, `is`). An empty list (no
    mapped inputs) makes every output unmapped.
    """
    batch_dims = []
    for spec in graph.tensor_specs:
        if not spec.shape:
            continue
        leading = spec.shape[0]
        if isinstance(leading, core.Dim) and _BATCH_DIM_NAME.match(leading.name):
            batch_dims.append(leading)
    return batch_dims


def _is_batch_entry(entry, batch_dims) -> bool:
    """True when one shape entry IS a batch dim (identity) or a `DimExpr`
    built purely from batch dims and ints — e.g. the `max(batch, batch)` a
    symbolic broadcast of mapped operands from DIFFERENT vmap levels produces
    (distinct Dim objects never compare equal, so their broadcast stays a max
    DimExpr; within one pass the shared dim compares equal and no max arises).
    Such an entry still IS the mapped leading axis."""
    if isinstance(entry, core.Dim):
        return any(entry is dim for dim in batch_dims)
    if isinstance(entry, core.DimExpr):
        return _dims_in_batch(entry, batch_dims)
    return False


def _dims_in_batch(expr, batch_dims) -> bool:
    if isinstance(expr, core.Dim):
        return any(expr is dim for dim in batch_dims)
    if isinstance(expr, core.DimExpr):
        return _dims_in_batch(expr.left, batch_dims) and _dims_in_batch(
            expr.right, batch_dims
        )
    return True  # int factors are fine


def _leading_batch_count(shape, batch_dims) -> int:
    """The length of the leading contiguous prefix of `shape` made of batch
    dims (identity-checked). > 0 ⇒ the value is mapped."""
    count = 0
    for entry in shape:
        if _is_batch_entry(entry, batch_dims):
            count += 1
        else:
            break
    return count


def _normalize_out_axes(graph: Graph, out_axes):
    """Validate/normalize `out_axes` against the graph's OUTPUT structure.

    Returns one ``0``/``None`` entry per TENSOR output leaf (output-leaf
    order). This is the Python half of `vmap`'s rearrangement: the native
    engine applies it (leading transpose / size-one axis insertion / the "a
    batch axis is never silently dropped" error, which needs the vectorized
    module) while the structural and v1 restrictions keep their historical
    wording here (pytree of 0|None; a bare 0/None only with at most one
    tensor output; static output leaves must be None; v1 entries {None, 0}).
    """
    static_positions = {record.index for record in graph.output_static_values}
    tensor_positions = set(range(graph.output_tree.num_leaves)) - static_positions
    has_registered = _tree_has_registered_node(graph.output_tree)

    if out_axes is None or _is_int(out_axes):
        # A bare entry applies to the single tensor output; with ZERO tensor
        # outputs there is nothing to rearrange and the bare form (e.g. the
        # out_axes=0 default) is a no-op. Multiple tensor outputs need a
        # pytree (a bare entry would be ambiguous) — UNLESS the output tree
        # contains a registered pytree node (e.g. a sparse-tensor node): the
        # bare entry then applies to ALL its tensor leaves (statics → None).
        if len(tensor_positions) > 1 and not has_registered:
            raise core.TransformError(
                "vmap: a bare int/None out_axes applies only when the graph "
                f"has at most ONE tensor output; got {len(tensor_positions)} "
                "tensor outputs — pass a pytree matching the output "
                "structure instead"
            )
        entries = [
            out_axes if index in tensor_positions else None
            for index in range(graph.output_tree.num_leaves)
        ]
    else:
        axes_leaves, axes_spec = core.flatten(out_axes)
        if has_registered:
            entries, mismatch = _broadcast_registered_axes(
                axes_spec, graph.output_tree, axes_leaves, tensor_positions,
                "vmap",
            )
        else:
            mismatch = core.first_mismatch_path(
                axes_spec, graph.output_tree, leaf_vs_empty_is_mismatch=True
            )
            entries = list(axes_leaves) if mismatch is None else None
        if mismatch is not None:
            raise core.TransformError(
                "vmap: the out_axes pytree does not match the output "
                f"structure — first mismatch at pytree path "
                f"{core.format_path(mismatch)}; out_axes must be a pytree with "
                "the same container structure as the outputs (0 keeps a "
                "mapped axis leading / inserts a size-one axis for an "
                "unmapped output, None requires the output unmapped)"
            )
    # No rank range-check for outputs: entry 0 on a rank-0 unmapped output is
    # the legal size-one-axis insertion, not an error.
    return _normalize_axis_entries(entries, tensor_positions, None, "vmap")
