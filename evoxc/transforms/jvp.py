"""Public `jvp` — forward-mode Jacobian-vector products.

`jvp` is graph→graph: the result is a `Graph` of ordinary ops mapping
(primal inputs, tangent inputs) → (primal outputs, tangent outputs). The native
(Rust) AD engine owns the forward sweep and every builtin JVP rule; Python keeps
the tangent-tree validation/normalization and the `Graph` reconstruction.

The documented Python fallback (`autodiff.forward_sweep`) runs only for graphs
whose derivative rules live in Python (`block:*`/`external:*`/`sparse_*`) — the
native extension request cannot carry the tangent SSA values (see
`_native_ad.py`). Semantics are binding — see `./CONTEXT.md` "AD semantics".
"""

from __future__ import annotations

from evoxc.trace import Graph, trace
from evoxc.transforms import _native_ad
from evoxc.transforms._wrappers import TransformCallable
from evoxc.transforms.autodiff import forward_sweep
from evoxc.transforms.grad import _normalize_extra_pytree


def jvp(fn_or_graph, tangents):
    """Forward-mode differentiation: push tangents through the computation.

    Args:
        fn_or_graph: a `Graph`, or a callable/`Defn` (see Returns).
        tangents: a pytree of `TensorSpec` (or `None` = zero tangent) matching
            the **input** structure; each tangent's shape/dtype must match its
            primal spec. Tangents are runtime tensors, so they become explicit
            inputs of the result graph.

    Returns:
        * `Graph` when given a `Graph`: inputs = primal inputs followed by
          tangent inputs (flattened order); outputs = `(primal_outputs,
          tangent_outputs)` — a 2-tuple of trees.
        * `TransformCallable` when given a callable/`Defn`:
          `jvp(f, tangents)(*specs)` traces `f` exactly once and returns that
          same graph (the callable's specs describe the primal inputs only).
          Concrete tensors raise `TraceError` (transforms never execute).

    Ops without a JVP rule (`runtime_call`, collectives, custom blocks without
    a registered rule) raise `core.TransformError` naming the op/block —
    never a silent fallback. `stop_gradient` yields a zero tangent.
    """
    if isinstance(fn_or_graph, Graph):
        return _jvp_graph(fn_or_graph, tangents)

    if callable(fn_or_graph) or getattr(fn_or_graph, "__evoxc_defn__", False):
        return TransformCallable(
            build=lambda *args: _jvp_fn(fn_or_graph, args, tangents),
            kind="jvp",
        )

    raise TypeError("jvp expects an evoxc.Graph, a callable, or an evoxc.Defn")


def _jvp_graph(graph: Graph, tangents) -> Graph:
    """Validate/normalize the tangent spec tree against the graph inputs, then
    run the forward sweep (native by default, the documented Python fallback for
    extension-ruled ops). The result already has the required input/output
    structure: inputs = 2-tuple (original input tree, flat tangent tuple tree),
    outputs = 2-tuple (primal outputs with the original output tree, flat
    tangent tensors aligned with the flattened tensor outputs)."""
    flat_tangents = _normalize_extra_pytree(
        tangents,
        graph.input_specs,
        graph.static_values,
        graph.tensor_specs,
        transform="jvp",
        kind="tangent",
    )
    if _native_ad._needs_python_ad(graph):
        return forward_sweep(graph, flat_tangents)
    return _native_ad.sweep_graph(graph, "jvp", flat_tangents)


def _jvp_fn(fn, args, tangents) -> Graph:
    """`jvp(f, tangents)(*specs)` — trace `fn` once via `evoxc.trace(fn, *args)`,
    then apply `_jvp_graph` with the stored tangents."""
    return _jvp_graph(trace(fn, *args), tangents)
