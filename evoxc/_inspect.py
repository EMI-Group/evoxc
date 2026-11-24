"""evoxc._inspect — private inspection rendering for the pipeline wrappers.

The readability half of ``evoxc.pipeline``: the pure rendering helpers behind
``Executable.__repr__`` / ``Executable.summary()`` / ``Executable.plan_text()``
and their ``BoundExecutable`` mirrors. Every function here is a pure read over
data the wrappers already hold — the ``Signature``, the backend executable's
already-built metadata, the backend name, the device, the binding count and the
reduced (unbound) input specs. Nothing here traces, lowers, compiles, loads,
runs, transfers or mutates anything, no fast-path cache is populated, and an
inspection snapshot never raises.

Import rule (binding): ``evoxc.core`` and the standard library only.
``evoxc.pipeline`` imports this module, so the dependency is one-way — plain
values flow in, rendered strings/dicts flow out, and no wrapper instance is
ever passed. The trace-time leaf markers used to render a signature tree are
therefore supplied by the caller (``tensor_leaf_types``), and every helper the
run/bind/build fast paths use stays in ``pipeline``.
"""

from __future__ import annotations

import dataclasses

from evoxc import core


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def inspection_str(value) -> str:
    """``str(value)`` for one-line inspection lines; ``"None"`` when absent."""
    return "None" if value is None else str(value)


def inspection_device(value) -> str:
    """Compact ``kind:index`` device spelling for one-line inspection lines.

    ``core.Device`` values render as ``cpu:0`` / ``cuda:0`` (the ``kind:index``
    parsing form ``EVOXC_DEVICE`` accepts); any other value renders as
    ``str(value)`` and an absent device as ``"None"``. Display only — the
    summary's ``"device"`` key keeps the canonical ``str(device)``.
    """
    if value is None:
        return "None"
    kind = getattr(value, "kind", None)
    index = getattr(value, "index", None)
    if (
        isinstance(kind, str)
        and isinstance(index, int)
        and not isinstance(index, bool)
    ):
        return f"{kind}:{index}"
    return str(value)


def backend_device(backend_executable):
    """The backend executable's ``device`` attribute, or ``None``.

    Inspection only: a backend executable that exposes no ``device`` (or a
    missing one) renders as ``None`` instead of raising.
    """
    return getattr(backend_executable, "device", None)


def spec_summary(spec) -> dict:
    """JSON-safe one-entry description of a tensor ``core.TensorSpec``.

    ``shape`` renders each dim through ``str`` (so symbolic ``Dim``/``DimExpr``
    stay JSON-safe), ``dtype`` through ``str``, and ``device`` is ``None``
    when the spec pins none.
    """
    device = getattr(spec, "device", None)
    return {
        "shape": [str(dim) for dim in getattr(spec, "shape", ())],
        "dtype": str(getattr(spec, "dtype", None)),
        "device": None if device is None else str(device),
    }


def signature_node_summary(spec, tensor_leaf_types=((), ())) -> str:
    """User-facing one-line description of ONE signature ``core.TreeSpec`` node.

    ``core.describe_node(spec)`` for container nodes and for every non-tensor
    leaf (static positions keep their Python type name verbatim), but a
    CHILDLESS tensor leaf renders a readable role name instead of leaking the
    private trace-time marker class into an inspection dict:
    ``_TensorSpecLeaf`` / ``core.TensorSpec`` → ``"tensor input leaf"``,
    ``_SymbolicLeaf`` / ``core.SymbolicTensor`` → ``"tensor output leaf"``,
    and a plain ``type=None`` leaf (grad/vjp-built output trees) →
    ``"tensor leaf"``.

    ``tensor_leaf_types`` is the ``(input_types, output_types)`` pair of leaf
    marker types the caller (``evoxc.pipeline``, which owns those markers)
    supplies — this module imports ``core`` only.

    Display only: ``core.describe_node`` itself keeps its wording unchanged,
    because that text is bound into structure-mismatch error messages.
    """
    input_types, output_types = tensor_leaf_types
    if not spec.children and spec.num_leaves == 1:
        leaf_type = spec.type
        if any(leaf_type is marker for marker in input_types):
            return "tensor input leaf"
        if any(leaf_type is marker for marker in output_types):
            return "tensor output leaf"
        if leaf_type is None:
            return "tensor leaf"
    return core.describe_node(spec)


def tree_summary(tree, tensor_leaf_types=((), ())):
    """``signature_node_summary`` (one-line tree description) or ``None``."""
    if tree is None:
        return None
    return signature_node_summary(tree, tensor_leaf_types)


def capabilities_summary(capabilities) -> dict:
    """JSON-safe dict of a backend's ``Capabilities`` dataclass fields.

    The ``dtypes`` frozenset is reported as ``"dtype_count": len(...)`` (the
    concrete dtype set is an implementation detail of a chip); any other set
    value (``rng_bit_generator``) is reported as a sorted list; every remaining
    field is reported as its plain value.
    """
    summary: dict = {}
    for capability in dataclasses.fields(capabilities):
        value = getattr(capabilities, capability.name, None)
        if capability.name == "dtypes":
            summary["dtype_count"] = len(value) if value is not None else 0
        elif isinstance(value, (set, frozenset)):
            summary[capability.name] = sorted(str(item) for item in value)
        else:
            summary[capability.name] = value
    return summary


def artifact_required_ops(backend_executable) -> list:
    """The loaded artifact's ``required_custom_ops`` as a list, ``[]`` when the
    backend executable exposes no artifact (or no attribute of that name).

    Pure read of the artifact metadata the ``compile`` stage recorded — never
    resolves the ops, never touches the registry.
    """
    try:
        artifact = getattr(backend_executable, "artifact", None)
        ops = getattr(artifact, "required_custom_ops", None)
    except Exception:  # noqa: BLE001 - inspection must never raise
        return []
    return [str(name) for name in (ops or ())]


def bindings_reduced_input_specs(signature, bindings, walk_leaves, is_tensor_leaf):
    """Tensor specs a caller must still supply after ``bindings``, or ``None``
    when the cheap derivation is unavailable.

    Derives the REDUCED (unbound) input signature by dropping the
    ``signature.input_specs`` entries whose flat leaf index is bound — the
    same pre-order flat leaf indexing ``bind``/``run`` use. ``walk_leaves``
    and ``is_tensor_leaf`` are ``evoxc.pipeline``'s own tree walker/leaf
    classifier (passed in so this module keeps its ``core``-only import rule).
    Pure inspection: never traces, validates a binding value, or touches a
    plan; a missing/malformed signature yields ``None`` so callers fall back
    to the wrapped executable's full signature.
    """
    try:
        input_specs = tuple(getattr(signature, "input_specs", None) or ())
        input_tree = getattr(signature, "input_tree", None)
        if signature is None or input_tree is None:
            return None
        bound = set(bindings)
        if not bound:
            return input_specs
        kept = []
        tensor_index = 0
        for leaf_spec, leaf_index, _path in walk_leaves(input_tree):
            if not is_tensor_leaf(leaf_spec):
                continue
            spec = input_specs[tensor_index]
            tensor_index += 1
            if leaf_index not in bound:
                kept.append(spec)
        return tuple(kept)
    except (AttributeError, IndexError, KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Executable rendering
# ---------------------------------------------------------------------------


def executable_repr(*, backend, device, signature) -> str:
    """One-line inspection line for a loaded pipeline executable.

    E.g. ``<Executable backend=numpy device=cpu:0 inputs=1 outputs=1>``;
    an unknown backend name or device renders as ``None``.
    """
    input_specs = getattr(signature, "input_specs", None) or ()
    output_specs = getattr(signature, "output_specs", None) or ()
    return (
        f"<Executable backend={inspection_str(backend)} "
        f"device={inspection_device(device)} "
        f"inputs={len(input_specs)} outputs={len(output_specs)}>"
    )


def executable_summary(
    *,
    backend,
    device,
    backend_executable,
    signature,
    capabilities=None,
    tensor_leaf_types=((), ()),
) -> dict:
    """JSON-safe inspection snapshot of one loaded pipeline executable.

    All values are plain data the wrapper already holds: ``backend`` (name or
    ``None``), ``device``, the ``backend_executable`` (for its ``functions``
    and its artifact metadata), the ``signature``, the already-rendered
    ``capabilities`` summary (``None`` when the backend cannot be resolved)
    and the caller-supplied ``tensor_leaf_types``. A missing signature yields
    zeroed counts, ``None`` trees and empty ``input_specs``/``output_specs`` —
    it never raises.
    """
    input_specs = tuple(getattr(signature, "input_specs", None) or ())
    output_specs = tuple(getattr(signature, "output_specs", None) or ())
    static_values = tuple(getattr(signature, "static_values", None) or ())
    functions = getattr(backend_executable, "functions", None) or ()
    input_tree = getattr(signature, "input_tree", None)
    output_tree = getattr(signature, "output_tree", None)
    return {
        "backend": backend,
        "device": None if device is None else str(device),
        "functions": [str(name) for name in functions],
        "input_count": len(input_specs),
        "output_count": len(output_specs),
        "input_specs": [spec_summary(spec) for spec in input_specs],
        "output_specs": [spec_summary(spec) for spec in output_specs],
        "input_tree": tree_summary(input_tree, tensor_leaf_types),
        "output_tree": tree_summary(output_tree, tensor_leaf_types),
        "static_value_count": len(static_values),
        "required_ops": artifact_required_ops(backend_executable),
        "capabilities": capabilities,
    }


def plan_text(backend_executable):
    """The backend executable's optional textual plan, or ``None``.

    Delegates to the backend executable's callable ``plan_text`` (the numpy
    interpreter implements it); a backend without one returns ``None``.
    Rendering a plan executes nothing and populates no cache.
    """
    render = getattr(backend_executable, "plan_text", None)
    if not callable(render):
        return None
    return render()


# ---------------------------------------------------------------------------
# BoundExecutable rendering
# ---------------------------------------------------------------------------


def bound_executable_repr(
    *, backend, device, signature, bound_count, reduced_input_specs
) -> str:
    """One-line inspection line for a bound pipeline executable.

    E.g. ``<BoundExecutable backend=numpy device=cpu:0 inputs=1 outputs=1
    bound=1>``; ``inputs`` counts the UNBOUND tensor inputs (the ones a caller
    still passes to ``run``), falling back to the full input count when the
    reduced derivation is unavailable.
    """
    output_specs = getattr(signature, "output_specs", None) or ()
    if reduced_input_specs is None:
        input_count = len(getattr(signature, "input_specs", None) or ())
    else:
        input_count = len(reduced_input_specs)
    return (
        f"<BoundExecutable backend={inspection_str(backend)} "
        f"device={inspection_device(device)} "
        f"inputs={input_count} outputs={len(output_specs)} "
        f"bound={bound_count}>"
    )


def bound_executable_summary(snapshot, *, bound_count, reduced_input_specs):
    """Add a bound executable's ``bound_count`` and REDUCED input side to an
    already-delegated ``Executable.summary()`` snapshot.

    ``input_count`` and ``input_specs`` describe only the inputs a caller
    still supplies whenever ``reduced_input_specs`` is available (while
    ``input_tree`` stays the full recorded tree); otherwise the delegated
    snapshot is returned unchanged.
    """
    snapshot["bound_count"] = bound_count
    if reduced_input_specs is not None:
        snapshot["input_count"] = len(reduced_input_specs)
        snapshot["input_specs"] = [spec_summary(spec) for spec in reduced_input_specs]
    return snapshot
