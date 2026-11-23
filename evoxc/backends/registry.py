"""Backend registry: register / unregister / get (+ lazy backend discovery).

The default backend, ``numpy_backend`` (a ``NumpyBackend`` instance), is
registered by ``evoxc.backends.numpy`` at import time. ``evoxc.lower``/``compile``/
``load`` default to it (defaulting is pipeline's job, not the registry's).

Optional adapters are NOT imported here or at ``evoxc``/``evoxc.backends``
import time. ``get(name)`` auto-activates them ON FIRST USE: on a registry
miss it imports the adapter module named by ``OPTIONAL_ADAPTERS`` and calls
its ``register()`` (which probes its dependency — a missing dependency raises
``core.BackendError`` with an actionable hint, propagated unchanged), then
retries the lookup.

Optional adapters:

- ``iree`` / ``xla`` / ``tvm`` — the StableHLO-consuming compiler adapters
  (``adapters/``); iree/tvm probe their pip package at ``register()``, xla's
  probe is a per-call PJRT plugin.
- ``native`` — the in-process Rust CPU execution engine (``native/``); its
  ``register()`` probes the ``evoxc._evoxc_native`` extension for the
  ``native_compile`` entry point (built from the ``native/`` Cargo workspace
  by ``scripts/build-native.sh``) and raises ``core.BackendError`` with build
  instructions when it is absent — never a silent fallback to numpy.

Third-party backends can ship as entry points in the ``evoxc.backends``
group. On a miss — AFTER ``OPTIONAL_ADAPTERS`` — ``get()`` scans that group
once per process and loads every entry point not already registered:

- a loaded object that is a ``Backend`` instance is registered as-is;
- a loaded object that is a MODULE with a ``register()`` function: that
  function is called (it registers the backend(s) itself);
- any other callable is called; when it returns a ``Backend``, that instance
  is registered.

Entry points are loaded in ``entry_point.name`` order (deterministic), at
most once per process (``_reset_entry_point_cache()`` is a private test hook).
A failing entry point raises ``core.BackendError`` naming it and chaining the
underlying exception — never swallowed. Registration is process-global and
last-registration-wins only through the explicit ``register(..., replace=True)``
of a DIFFERENT instance.
"""
from __future__ import annotations

from typing import Any

from evoxc.core import (
    BackendError,
    _note_backend_transfer_preference,  # internal cross-module contract (Tensor.to routing)
)

from .backend import Backend

__all__ = ["register", "unregister", "get", "OPTIONAL_ADAPTERS"]

_registry: dict[str, Backend] = {}

#: Optional adapters: backend name -> adapter module path.
#: Modules implement a ``register()`` function that checks the compiler
#: dependency and calls ``registry.register(instance)`` (idempotent). They
#: are imported ONLY inside ``get()`` — never at package import time.
OPTIONAL_ADAPTERS: dict[str, str] = {
    "iree": "evoxc.backends.adapters.iree",
    "xla": "evoxc.backends.adapters.xla",
    "tvm": "evoxc.backends.adapters.tvm",
    "native": "evoxc.backends.native",
}

#: The entry-point group third-party backend distributions publish into.
ENTRY_POINT_GROUP = "evoxc.backends"

#: Set once the entry-point group has been scanned (the scan is process-wide).
_entry_points_loaded = False


def register(backend: Backend, *, replace: bool = False) -> Backend:
    """Register a backend under ``backend.name``.

    Idempotent for the same instance (imports may re-execute); a DIFFERENT
    instance under an existing name raises ``BackendError`` unless
    ``replace=True`` (then the new instance takes the name — the supported
    way to install a customized adapter instance, e.g. an iree adapter
    pointed at a custom-built ``iree`` module). A missing or non-string name
    raises ``BackendError``.

    Returns the backend (convenience for chaining).
    """
    if not isinstance(backend, Backend):
        raise BackendError(
            f"register() expects a Backend instance, got {type(backend).__name__}"
        )
    name = backend.name
    if not isinstance(name, str) or not name:
        raise BackendError(
            f"backend of type {type(backend).__name__} must declare a non-empty `name`"
        )
    existing = _registry.get(name)
    if existing is not None and existing is not backend and not replace:
        raise BackendError(
            f"backend name {name!r} is already registered by "
            f"{type(existing).__name__} — pass replace=True to install "
            f"{type(backend).__name__} under the same name"
        )
    _registry[name] = backend
    return backend


def unregister(name: str) -> Backend:
    """Remove and return the backend registered under ``name``.

    Unknown names raise ``BackendError`` (never a silent no-op).
    """
    existing = _registry.pop(name, None)
    if existing is None:
        known = ", ".join(sorted(_registry)) or "(none)"
        raise BackendError(
            f"unknown backend {name!r}; registered backends: {known}"
        )
    return existing


def _reset_entry_point_cache() -> None:
    """TEST HOOK (private, not exported from ``evoxc``): allow a re-scan.

    Clears the once-per-process entry-point flag so the next ``get()`` miss
    re-reads ``importlib.metadata.entry_points(group="evoxc.backends")``.
    """
    global _entry_points_loaded
    _entry_points_loaded = False


def _load_entry_points() -> None:
    """Scan the ``evoxc.backends`` entry-point group, once per process.

    Each entry point is loaded and registered per the module-docstring
    contract; any failure raises ``core.BackendError`` naming the entry point
    and chaining the original exception.
    """
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True
    import importlib.metadata

    for entry_point in sorted(
        importlib.metadata.entry_points(group=ENTRY_POINT_GROUP),
        key=lambda ep: ep.name,
    ):
        try:
            loaded: Any = entry_point.load()
            if isinstance(loaded, Backend):
                register(loaded)
            elif callable(getattr(loaded, "register", None)):
                loaded.register()
            elif callable(loaded):
                produced = loaded()
                if isinstance(produced, Backend):
                    register(produced)
            else:
                raise BackendError(
                    "entry point loaded an object that is neither a Backend, "
                    "a module with register(), nor a callable"
                )
        except Exception as exc:  # noqa: BLE001 - re-raised as BackendError
            raise BackendError(
                f"failed to load evoxc.backends entry point "
                f"{entry_point.name!r} ({entry_point.value!r}): {exc}"
            ) from exc


def get(name: str) -> Backend:
    """Look up a backend by name; activate optional adapters / entry points.

    Registered backends resolve directly. On a miss, an optional compiler
    adapter named in ``OPTIONAL_ADAPTERS`` is imported and its
    ``register()`` is called (which probes the compiler dependency and
    registers the instance — a missing dependency raises
    ``core.BackendError`` with a pip-install hint, propagated UNCHANGED);
    then, still on a miss, the ``evoxc.backends`` ENTRY POINT group is
    scanned once per process (see ``_load_entry_points``) and the lookup is
    retried. Unknown names raise ``BackendError`` listing the registered
    backends.

    On every successful lookup the backend is ALSO recorded as the
    PREFERRED device-transfer backend for the device kinds it serves
    (``_note_backend_transfer_preference`` — see ``evoxc.core.tensor``):
    ``Tensor.to`` then resolves that backend's own per-backend provider
    slot. Adapters register their providers under per-backend slots in
    ``register()`` and NEVER overwrite each other's or the flat default
    slot, so the active transfer provider follows the last ``get()``
    deterministically regardless of import order (no last-wins clobbering).
    Module import alone never changes the preference.
    """
    backend = _registry.get(name)
    if backend is not None:
        _note_backend_transfer_preference(backend.name)
        return backend
    if name in OPTIONAL_ADAPTERS:
        import importlib

        module = importlib.import_module(OPTIONAL_ADAPTERS[name])
        module.register()
        backend = _registry.get(name)
        if backend is not None:
            _note_backend_transfer_preference(backend.name)
            return backend
    _load_entry_points()
    backend = _registry.get(name)
    if backend is not None:
        _note_backend_transfer_preference(backend.name)
        return backend
    known = ", ".join(sorted(_registry)) or "(none)"
    raise BackendError(
        f"unknown backend {name!r}; registered backends: {known} "
        f"(optional adapters: {', '.join(sorted(OPTIONAL_ADAPTERS))}; "
        f"third-party backends are discovered from the "
        f"{ENTRY_POINT_GROUP!r} entry-point group)"
    )
