"""Callback resolver for the Rust CPU engine's host boundaries.

A ``NativeProgram`` runs entirely inside Rust except at three fronts, where the
engine calls back into Python through the resolver object handed to
``NativeProgram.run(flat_inputs, resolver)``::

    call_block(name, args, static_args=None) -> list[numpy.ndarray]
    call_external(name, args)                -> list[numpy.ndarray]
    call_runtime(name, args)                 -> list[numpy.ndarray]

``name`` is the op's static identifier (``block_call``'s ``block_name``,
``external_call``'s ``name``, ``runtime_call``'s ``callback`` id); ``args`` is
the operand list, positionally in IR order, as host numpy arrays. Each method
returns a LIST of numpy arrays, one per declared result, in declaration order —
the engine validates them against the op's declared result types (count /
dtype / shape), exactly as the numpy interpreter does for its own dispatch.

Resolution mirrors the numpy reference interpreter (``kernels/custom.py``), so
a graph that runs on one runs on the other:

- ``call_block``: ``evoxc.block.registry.get_impl(name, "native")``; the impl
  is called ``impl(*arrays, **static_args)`` where the ``static_args`` VALUES
  are the JSON-safe ``{"kind": ..., "value": ...}`` payloads recorded on the
  op — NOT decoded Python values, exactly like the numpy impl contract
  (``tests/block/CONTEXT.md``), so one block impl serves both backends. Portable decompositions were
  already inlined at ``lower()`` time (``inline_portables(...,
  keep_backend_impls="native")``), so a block reaching run time MUST have a
  registered ``native`` impl — a block carrying only another backend's impl
  (e.g. ``numpy``) raises ``core.BackendError`` naming block and backend
  (never a silent cross-backend fallback). ``static_args`` is the
  ``block_call`` op's JSON-able static-kwarg dict; the engine passes it as the
  third argument (see this directory's CONTEXT.md — the FFI contract), and an
  engine build that omits it is an explicit ``core.BackendError`` naming the
  block whenever the impl provably requires the block's declared attributes.
- ``call_external``: ``evoxc.external.get_external_kernel_entry(name,
  "native")`` — the per-backend ``native`` slot with the default (``None``)
  slot fallback. The registry is never serialized, so a graph/artifact
  requires the same registrations in the running process; an unknown name
  raises ``core.BackendError`` pointing at
  ``evoxc.register_external_kernel``. A slot registered
  ``device_resident=True`` is rejected explicitly: this backend hands kernels
  host numpy arrays and cannot honor device-resident semantics (register a
  host-mode slot, or run on a device backend such as iree).
- ``call_runtime``: ``evoxc.ops.constant._get_callback(id)`` — the callback
  registry the numpy interpreter's ``resolve_callback`` uses. Unknown ids
  raise ``core.BackendError`` naming the id; callbacks are never embedded in
  artifacts.

Import acyclicity (binding): this module imports ``evoxc.core`` and numpy at
top level; ``evoxc.block`` / ``evoxc.external`` / ``evoxc.ops`` are imported
INSIDE the methods (backends may import ``evoxc.ops`` only inside function
bodies).
"""
from __future__ import annotations

import inspect
from typing import Any, Mapping

import numpy as np

from evoxc import core

from ..external_validate import normalize_results

__all__ = ["NativeCallbackResolver"]


def _label(kind: str, name: Any) -> str:
    """The error label for one boundary (``op 'block_call' (block 'x')``)."""
    return f"native {kind} boundary (name {name!r})"


def _to_arrays(args: Any, label: str) -> list:
    """Normalize the engine's operand payload into a list of host ndarrays.

    The FFI contract passes a list of numpy arrays; a single array or a
    ``core.Tensor`` is accepted for robustness, and 0-d/NumPy-scalar entries
    are coerced with ``np.asarray`` (the engine may hand back a scalar for a
    0-d value). Nothing else is guessed — a non-array-like operand raises
    ``core.BackendError`` naming the boundary.
    """
    if isinstance(args, (core.Tensor, np.ndarray, np.generic)):
        entries = [args]
    elif isinstance(args, (list, tuple)):
        entries = list(args)
    else:
        raise core.BackendError(
            f"{label}: operands must be a list of numpy arrays, got "
            f"{type(args).__name__}"
        )
    arrays = []
    for index, entry in enumerate(entries):
        if isinstance(entry, core.Tensor):
            arrays.append(entry.numpy())
        elif isinstance(entry, np.ndarray):
            arrays.append(entry)
        elif isinstance(entry, np.generic):
            arrays.append(np.asarray(entry))
        else:
            raise core.BackendError(
                f"{label}: operand {index} is {type(entry).__name__}, expected "
                "a numpy array (the native engine passes host arrays to "
                "callback boundaries)"
            )
    return arrays


def _returns(result: Any, label: str) -> list:
    """Normalize a kernel/impl/callback return into a list of host ndarrays.

    Reuses the shared external-kernel normalization (``normalize_results``:
    ndarray / ``core.Tensor`` / tuple/list, anything else a ``BackendError``
    naming the boundary), then materializes every ``core.Tensor`` entry as its
    host array. The engine validates count/dtype/shape against the op's
    declared result types.
    """
    arrays = []
    for entry in normalize_results(result, label):
        if isinstance(entry, core.Tensor):
            arrays.append(entry.numpy())
        elif isinstance(entry, np.ndarray):
            arrays.append(entry)
        else:
            raise core.BackendError(
                f"{label}: kernel returned {type(entry).__name__}, expected "
                "numpy arrays (the native engine consumes host arrays)"
            )
    return arrays


def _static_args(static_args: Any) -> dict:
    """The block impl's static-kwarg dict from the engine's third argument.

    Accepts the ``static_args`` mapping directly or an op-attributes mapping
    carrying it, so either serialization of the ``block_call`` attributes
    works. ``None``/absent means "no static kwargs".
    """
    if static_args is None:
        return {}
    if not isinstance(static_args, Mapping):
        raise core.BackendError(
            "native block_call boundary: static_args must be a mapping of "
            f"static kwargs, got {type(static_args).__name__}"
        )
    nested = static_args.get("static_args")
    if isinstance(nested, Mapping):
        return dict(nested)
    return dict(static_args)


def _declared_attribute_names(block: Any) -> tuple[str, ...]:
    """The static-attribute names a block DECLARES (schema order)."""
    schema = getattr(block, "attribute_schema", None)
    if not isinstance(schema, Mapping):
        return ()
    return tuple(schema)


def _impl_needs_static_args(impl: Any, block: Any, operands: int) -> bool:
    """True only when ``impl`` provably cannot run without the declared kwargs.

    An engine build that calls ``call_block(name, args)`` without the third
    ``static_args`` argument would otherwise make the impl raise a bare
    ``TypeError``/``KeyError`` about a missing kwarg, which reads like a user
    bug rather than FFI drift. Only the UNAMBIGUOUS case is reported: the impl
    binds the operands positionally once the block's DECLARED attribute names
    are supplied as keywords, and does not bind them without. An impl whose
    signature cannot be introspected (builtins, C callables), one that accepts
    ``**kwargs``, or a block that declares no attributes is never reported —
    nothing is guessed and no legitimate call is rejected.
    """
    names = _declared_attribute_names(block)
    if not names:
        return False
    try:
        signature = inspect.signature(impl)
    except (TypeError, ValueError):  # non-introspectable callable
        return False
    placeholders = [None] * operands
    try:
        signature.bind(*placeholders)
    except TypeError:
        pass
    else:
        return False  # callable with the operands alone
    try:
        signature.bind(*placeholders, **dict.fromkeys(names))
    except TypeError:
        return False  # not explained by the declared attributes
    return True


class NativeCallbackResolver:
    """Resolves the Rust CPU engine's Python callback boundaries.

    Stateless: one instance may be shared by every run of one executable (the
    backend creates it once at ``load``). Methods are called BY THE ENGINE
    during ``NativeProgram.run``; they are not a user-facing API.

    This class intentionally does not implement the ``Executable``/``Backend``
    protocols — it is the boundary adapter only.
    """

    def call_block(self, name: Any, args: Any, static_args: Any = None) -> list:
        """Dispatch a ``block_call`` op to the block's registered ``native`` impl.

        ``static_args`` (the op's static-kwarg dict) must be supplied by the
        engine as the third argument; blocks whose impl requires the declared
        static kwargs get an explicit ``core.BackendError`` naming the block
        when it is missing rather than a bare ``TypeError`` from the impl.
        """
        label = _label("block_call", name)
        if not isinstance(name, str) or not name:
            raise core.BackendError(
                f"{label}: block name must be a non-empty string"
            )
        from evoxc.block import BlockError
        from evoxc.block import registry as block_registry

        try:
            block = block_registry.get_block(name)
        except BlockError as exc:
            raise core.BackendError(
                f"op 'block_call': unknown block {name!r} — declare it with "
                "evoxc.block(...) before building graphs that call it"
            ) from exc
        impl = block_registry.get_impl(name, "native")
        if impl is None:
            raise core.BackendError(
                f"op 'block_call': no native implementation registered for "
                f"block {name!r} — portable decompositions are inlined at "
                "lower() time, so a block reaching the engine must have a "
                "registered 'native' impl (BlockOp.impl('native', fn)); "
                "register one, or run the graph on the backend that owns the "
                "impl (numpy/iree/...)"
            )
        kwargs = _static_args(static_args)
        arrays = _to_arrays(args, label)
        if static_args is None and _impl_needs_static_args(impl, block, len(arrays)):
            declared = list(_declared_attribute_names(block))
            raise core.BackendError(
                f"{label}: the native engine called call_block without its "
                "third static_args argument, but the impl requires the "
                f"block's declared attributes {declared} — block static "
                "kwargs travel as call_block(name, args, static_args); "
                "rebuild the extension from a matching native/ tree"
            )
        result = impl(*arrays, **kwargs)
        return _returns(result, label)

    def call_external(self, name: Any, args: Any) -> list:
        """Dispatch an ``external_call`` op through the ``evoxc.external`` registry."""
        label = _label("external_call", name)
        if not isinstance(name, str) or not name:
            raise core.BackendError(
                f"{label}: external kernel name must be a non-empty string"
            )
        from evoxc.external import get_external_kernel_entry

        entry = get_external_kernel_entry(name, "native")
        if entry is None:
            raise core.BackendError(
                f"op 'external_call': no external kernel registered under name "
                f"{name!r} — register it with "
                "evoxc.register_external_kernel(name, callable) in this "
                "process before running graphs that call it (kernels are "
                "never embedded in artifacts)"
            )
        kernel, device_resident = entry
        if device_resident:
            raise core.BackendError(
                f"op 'external_call': kernel {name!r} is registered "
                "device_resident=True for the native backend — the native "
                "backend passes host numpy arrays and cannot honor "
                "device-resident semantics; register a host-mode native slot "
                "(evoxc.register_external_kernel(name, fn, backend='native')) "
                "or run the kernel on a device backend like iree"
            )
        arrays = _to_arrays(args, label)
        result = kernel(*arrays)
        return _returns(result, label)

    def call_runtime(self, name: Any, args: Any) -> list:
        """Dispatch a ``runtime_call`` op to its registered Python callback."""
        label = _label("runtime_call", name)
        if not isinstance(name, str) or not name:
            raise core.BackendError(
                f"{label}: runtime_call callback id must be a non-empty string"
            )
        from evoxc.ops.constant import _get_callback

        callback = _get_callback(name)
        if callback is None:
            raise core.BackendError(
                f"op 'runtime_call': no callback registered under id {name!r} "
                "— artifacts containing runtime_call require the same "
                "callback registrations in the process at run time "
                "(callbacks are never embedded in artifacts)"
            )
        arrays = _to_arrays(args, label)
        result = callback(*arrays)
        return _returns(result, label)
