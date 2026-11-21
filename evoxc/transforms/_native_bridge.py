"""Shared plumbing for the native (Rust) transform engine.

`evoxc.vectorize` / `evoxc.vmap` keep ONLY frontend validation/normalization
and `Graph` reconstruction in Python; the graph rewrite itself is the native
single authority (`evoxc._evoxc_native`, reached through the typed façade in
`evoxc/_native.py`). The same helpers serve the AD half (`grad`/`jvp`/`vjp`)
and any later transform that delegates to the engine.

Two entry points:

* :func:`call_native` — run one native transform on an `ir.Module` and return
  the transformed module plus its `"transform"` metadata dict
  (``{"kind", "output_ids", "input_ids", "batch_dim"}``).
* :func:`rebuild_graph` — wrap the transformed module back into a
  :class:`evoxc.trace.Graph`, carrying the source graph's input/output tree
  skeletons, static values and input source locations.

:func:`vectorize_graph` / :func:`vmap_graph` are the two batching
compositions the public frontends call (they do the Python-side normalization
of the config, never the op rewrite).

The engine's `batch_dim` naming: the Python nesting state (`batching.
_batch_depth`) is passed explicitly as ``config["batch_dim_name"]`` so the
transformed module NEVER carries a stale ``metadata["batch_depth"]`` (depth
names, not metadata, are the source of truth — a graph transformed twice must
use the CURRENT depth).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from evoxc import _native, core, ir
from evoxc.transforms import _native_rules, batching
from evoxc.transforms._metadata import MappedAxes

#: Transform kinds that create a fresh batch dim (they honour `batch_dim_name`).
_BATCHING_KINDS = frozenset({"vectorize", "vmap"})


def call_native(
    kind: str,
    module: "ir.Module",
    config: Optional[Mapping[str, Any]] = None,
) -> Tuple["ir.Module", Dict[str, Any]]:
    """Run one native transform on `module`; return `(module, transform_dict)`.

    `config` carries the documented transform-config keys (`axes`,
    `out_axes`, `argnums`, `tangent_inputs`, `cotangent_inputs`,
    `batch_dim_name`, `extension_rule`). Two keys are supplied here when the
    caller omits them:

    * ``batch_dim_name`` — the name of the CURRENT Python pass depth's batch
      dim (`batching._batch_dim()`), for the batching kinds only.
    * ``extension_rule`` — the Python extension-rule callback
      (`_native_rules.extension_rule`) that serves `block:*` / `external:*` /
      `sparse_*` rules to the engine.

    Encoding/decoding of the envelope happens inside the façade
    (`evoxc._native.transform`), which also maps native failures onto
    `core.TransformError`.
    """
    resolved: Dict[str, Any] = dict(config or {})
    if kind in _BATCHING_KINDS:
        resolved.setdefault("batch_dim_name", batching._batch_dim().name)
    resolved.setdefault("extension_rule", _native_rules.extension_rule)
    _native_rules.reset_rule_failure()
    try:
        result = _native.transform(kind, module, resolved)
    except Exception as exc:
        # The engine wraps a bridge failure in its own `TransformError`; restore
        # the rule's original exception (same message, own class) instead.
        failure = _native_rules.take_rule_failure()
        if failure is not None and failure is not exc:
            raise failure from exc
        raise
    return result["module"], result["transform"]


def _return_operand_ids(module: "ir.Module") -> list:
    block = module.main.entry_block
    terminator = block.terminator
    if terminator is None:
        return []
    return [value.id for value in terminator.operands]


def _check_transform_meta(module: "ir.Module", transform: Mapping[str, Any]) -> None:
    """Cross-check the transform metadata against the decoded module.

    The metadata is the Python side's only description of "what changed";
    a divergence means the envelope round trip lost something, so it fails
    loudly instead of producing a silently wrong `Graph`.
    """
    expected = list(transform.get("output_ids") or ())
    actual = _return_operand_ids(module)
    if expected != actual:
        raise core.TransformError(
            "native transform: the transform metadata's output ids "
            f"{expected} do not match the transformed module's return "
            f"operands {actual}"
        )


def rebuild_graph(
    source_graph: "Any",
    module: "ir.Module",
    transform: Mapping[str, Any],
    *,
    input_specs: Optional["core.TreeSpec"] = None,
    output_tree: Optional["core.TreeSpec"] = None,
    output_static_values: Optional[tuple] = None,
) -> "Any":
    """Wrap a transformed module back into a `Graph` (batching semantics).

    Args:
        source_graph: the graph the transform was applied to (its trees,
            static values and input source locations are carried forward).
        module: the decoded transformed module.
        transform: the native `"transform"` metadata dict.
        input_specs: override for the input tree skeleton (defaults to the
            source graph's — batching never changes the structure).
        output_tree: override for the output tree skeleton (defaults to the
            source graph's, with registered-node static leaves remapped for
            the pass's batch dim). AD transforms that CHANGE the output
            structure must pass their own tree.
        output_static_values: override for the output static records
            (defaults to the remapped source records).

    Batching contract: the number of tensor output leaves is unchanged, so
    the source tree is reused verbatim (`_remap_output_aux` handles
    registered pytree output nodes whose static leaves gain the batch dim).
    Mapped input specs are derived from the transformed block-argument types
    (so every mapped input shares the batch `Dim` identity the codec
    produced); unmapped inputs reuse the source spec object unchanged.
    """
    function = module.main
    block = function.entry_block
    transform = dict(transform or {})
    _check_transform_meta(module, transform)

    input_ids = list(transform.get("input_ids") or ())
    if len(input_ids) != len(source_graph.tensor_specs) or len(
        block.arguments
    ) != len(input_ids):
        raise core.TransformError(
            "native transform: the transformed function has "
            f"{len(block.arguments)} block args / the metadata names "
            f"{len(input_ids)} input ids, but the source graph has "
            f"{len(source_graph.tensor_specs)} tensor inputs"
        )

    batch_name = transform.get("batch_dim")
    batch_dim = None
    if batch_name:
        # The envelope round trip rebuilds every `Dim` from its name, so the
        # decoded types hold DISTINCT-but-equal `Dim` objects. One pass = ONE
        # batch dim (object identity is the documented contract — mapped-output
        # detection compares the batch `Dim` by identity), so unify them.
        batch_dim = _unify_batch_dim(block, batch_name)

    # Mapped input specs are DERIVED from the source specs, not re-built from
    # the decoded types: the source spec's shape objects (e.g. a symbolic row
    # `Dim("B")`) are the graph's, so they must survive by identity, and every
    # mapped spec must share ONE batch `Dim` object (a per-spec decode would
    # give distinct-but-equal dims).
    new_specs = []
    for old_spec, argument in zip(source_graph.tensor_specs, block.arguments):
        value_type = argument.type
        if batch_dim is not None and len(value_type.shape) == len(old_spec.shape) + 1:
            new_specs.append(
                core.TensorSpec(
                    shape=(batch_dim,) + tuple(old_spec.shape),
                    dtype=old_spec.dtype,
                    device=old_spec.device,
                    name=old_spec.name,
                )
            )
        else:
            new_specs.append(old_spec)

    # Carry the input source locations forward: `source_locations` is keyed by
    # old input value id, so map old arg id -> new arg id positionally.
    source_locations: Dict[int, Any] = {}
    source_block = source_graph.module.main.entry_block
    for old_argument, new_id in zip(source_block.arguments, input_ids):
        location = source_graph.source_locations.get(old_argument.id)
        if location is not None:
            source_locations[new_id] = location

    batch_name = transform.get("batch_dim")
    if output_tree is None:
        if batch_name:
            if batch_dim is None:
                batch_dim = _find_batch_dim(block, batch_name)
            output_tree, remapped_statics = batching._remap_output_aux(
                source_graph, batch_dim
            )
            if output_static_values is None:
                output_static_values = remapped_statics
        else:
            output_tree = source_graph.output_tree
    if output_static_values is None:
        output_static_values = source_graph.output_static_values
    if input_specs is None:
        input_specs = source_graph.input_specs

    return type(source_graph)(
        module,
        input_specs,
        tuple(new_specs),
        output_tree,
        source_graph.static_values,
        output_static_values,
        source_locations,
    )


def _unify_batch_dim(block: "ir.Block", batch_name: str) -> Optional["core.Dim"]:
    """Give every `Dim(batch_name)` in the module the SAME `Dim` object.

    Returns that canonical object (the first mapped argument's leading dim),
    or None when the module carries no such dim (an all-unmapped pass).
    """
    canonical = None
    for argument in block.arguments:
        for entry in argument.type.shape:
            if isinstance(entry, core.Dim) and entry.name == batch_name:
                canonical = entry
                break
        if canonical is not None:
            break
    if canonical is None:
        return None

    def _values():
        yield from block.arguments
        for op in block.ops:
            yield from op.results
            for region in op.regions:
                for inner in region.blocks:
                    yield from inner.arguments
                    for inner_op in inner.ops:
                        yield from inner_op.results

    for value in _values():
        value_type = value.type
        shape = getattr(value_type, "shape", None)
        if not shape:
            continue
        new_shape = tuple(
            canonical if isinstance(entry, core.Dim) and entry.name == batch_name else entry
            for entry in shape
        )
        if any(a is not b for a, b in zip(new_shape, shape)):
            try:
                value_type.shape = new_shape
            except Exception:  # frozen ValueType
                object.__setattr__(value_type, "shape", new_shape)
    return canonical


def _find_batch_dim(block: "ir.Block", batch_name: str) -> "core.Dim":
    """The batch `Dim` object the transformed module carries.

    The codec rebuilds dims from the envelope, so the mapped input types hold
    the SAME `Dim` objects the rebuilt specs use — return that object (not a
    freshly constructed one), keeping object identity for the output-aux
    remap. Falls back to a fresh `core.Dim(batch_name)`.
    """
    for argument in block.arguments:
        shape = argument.type.shape
        if shape:
            leading = shape[0]
            if isinstance(leading, core.Dim) and leading.name == batch_name:
                return leading
    return core.Dim(batch_name)  # pragma: no cover - defensive


def input_axes_config(graph, tensor_axes) -> Dict[str, Optional[int]]:
    """The native ``axes`` config for a graph: one entry per block argument.

    `tensor_axes` holds one ``0``/``None`` per TENSOR input leaf (the output of
    `batching._tensor_input_axes`); the block arguments ARE the tensor inputs,
    in order, so the config is a plain value-id → axis map.
    """
    block = graph.module.main.entry_block
    if len(block.arguments) != len(tensor_axes):
        raise core.TransformError(
            "vectorize: the graph function has "
            f"{len(block.arguments)} block args but {len(tensor_axes)} "
            "tensor input specs"
        )
    return {str(argument.id): entry for argument, entry in zip(block.arguments, tensor_axes)}


def vectorize_graph(graph, axes) -> "Any":
    """The native `vectorize` composition (graceful shim over `call_native`)."""
    tensor_axes = batching._tensor_input_axes(graph, axes)
    module, transform = call_native(
        "vectorize", graph.module, {"axes": input_axes_config(graph, tensor_axes)}
    )
    return rebuild_graph(graph, module, transform)


def vmap_graph(graph, tensor_entries, out_entries) -> "Any":
    """The native `vmap` composition: vectorize + output-axis rearrangement.

    `tensor_entries` are the per-tensor-input ``0``/``None`` entries and
    `out_entries` the per-TENSOR-output ``0``/``None`` entries (both already
    Python-validated: v1 supports only ``{0, None}``). Native owns the
    rearrangement (a leading transpose / a size-one axis insertion) and the
    "a batch axis is never silently dropped" error.
    """
    block = graph.module.main.entry_block
    if len(block.arguments) != len(tensor_entries):
        raise core.TransformError(
            "vectorize: the graph function has "
            f"{len(block.arguments)} block args but {len(tensor_entries)} "
            "tensor input specs"
        )
    config = {
        "axes": input_axes_config(graph, tensor_entries),
        "out_axes": list(out_entries),
    }
    module, transform = call_native("vmap", graph.module, config)
    return rebuild_graph(graph, module, transform)
