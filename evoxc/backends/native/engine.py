"""Thin FFI seam onto the in-process Rust CPU execution engine (crate ``evoxc-cpu``).

This is the ONLY place in ``evoxc`` that talks to the extension's CPU-engine
surface. Everything the ``native`` backend needs from the Rust side goes
through the functions below, so the integration stays thin and auditable:

    native_compile(envelope: dict) -> NativeProgram
        # envelope = the ``evoxc-native-ir`` module envelope ``transform()``
        # consumes (``evoxc._native_codec.encode_module``)
        # errors raise NativeError with kind
        # VerificationError/ShapeError/DTypeError/BackendError/General
    NativeProgram.run(flat_inputs: list[numpy.ndarray], resolver=None)
        -> list[numpy.ndarray]
    NativeProgram.entry() -> str
    NativeProgram.plan_text() -> str
    NativeProgram.stats() -> dict
        # ops, dispatched_ops, blocks, plan_entries, compile_ms,
        # unsupported ops
    NativeProgram.output_types() -> list[dict]   # encoded ValueType dicts
    NativeProgram.required_ops() -> dict
        # {"blocks": [...], "external": [...], "runtime_calls": [...]}

The extension is an OPTIONAL, lazily loaded seam (``evoxc._native`` owns
discovery and loading; ``import evoxc`` never imports it). Every entry point
here therefore starts from :func:`engine_module`, which raises an actionable
``core.BackendError`` — never a silent fallback to another backend — when the
extension is absent, unloadable, or built without the ``native_compile``
entry point (i.e. the Rust ``evoxc-cpu`` crate has not landed yet).

Error mapping is deliberately the SAME implementation the ``evoxc._native``
façade uses (``_native._translate``): only a ``NativeError`` whose ``kind`` is
a known string is translated onto the public ``evoxc.core`` error family, with
the message preserved verbatim; anything else re-raises unchanged.

Import acyclicity (binding, see ``../CONTEXT.md``): this module imports
``evoxc.core`` at top level; ``evoxc.ir`` / ``evoxc._native`` /
``evoxc._native_codec`` are imported INSIDE function bodies so that importing
the ``native`` backend package never loads the extension.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from evoxc import core

__all__ = [
    "BACKEND_NAME",
    "ENGINE_EXTRA",
    "ENGINE_VERSION_KEY",
    "SUPPORTED_DTYPES",
    "engine_available",
    "engine_module",
    "engine_version",
    "encode_module",
    "compile_module",
    "program_entry",
    "program_run",
    "program_plan_text",
    "program_stats",
    "program_required_ops",
    "program_output_types",
    "unsupported_ops",
]

#: Registry / artifact backend name spelled by this package.
BACKEND_NAME = "native"

#: Runtime-dependency key recorded in a ``native`` artifact.
ENGINE_VERSION_KEY = "evoxc-native"

#: Optional extra that carries the compiled extension (informational only —
#: the extension ships INSIDE the ``evoxc`` wheel as
#: ``evoxc/_evoxc_native.abi3.so``; it is never a pip extra of its own).
ENGINE_EXTRA = "evoxc"

#: The ``NativeProgram`` methods the backend needs. ``run`` is mandatory
#: (the export surface cannot work without it); ``entry`` is required for the
#: ``functions`` attribute. The remaining accessors are diagnostics and are
#: resolved lazily with an explicit ``BackendError`` when absent.
_REQUIRED_METHODS = ("run", "entry")

_REQUIRED_ATTRS = ("native_compile",)

_REMEDY = (
    "the evoxc native extension does not expose the Rust CPU execution engine "
    "(missing {missing}). Build it with `source scripts/dev-env.sh && "
    "scripts/build-native.sh` (the extension lands at "
    "`evoxc/_evoxc_native.abi3.so` from the `native/` Cargo workspace, whose "
    "`evoxc-cpu` crate provides `native_compile`), or set EVOXC_NATIVE_LIB to "
    "a built extension. The `native` backend executes graphs on that Rust "
    "engine; it never falls back to another backend — use backend='numpy' (or "
    "another backend) explicitly for the pure-Python interpreter."
)


def _supported_dtypes() -> frozenset:
    """The 14 dtypes the backend declares (the engine's supported set).

    Matches the engine's dtype surface as agreed with the FFI contract: the
    numeric widths evoxc's IR supports plus ``bool``. Narrower than the numpy
    reference backend only if the engine reports so — an op whose result
    dtype falls outside this set is rejected at ``lower()`` with a
    ``core.BackendError`` (never silently coerced).
    """
    import numpy as np

    return frozenset(
        np.dtype(dtype)
        for dtype in (
            np.bool_,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.float16,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        )
    )


SUPPORTED_DTYPES = _supported_dtypes()


# ---------------------------------------------------------------------------
# Extension lookup
# ---------------------------------------------------------------------------


def engine_module():
    """Return the loaded extension module, or raise an actionable BackendError.

    Importing ``evoxc._native`` here (inside the body) keeps the extension off
    the ``import evoxc`` / ``import evoxc.backends`` path. ``_native.require``
    raises ``ImportError`` (with build instructions) when no extension can be
    loaded; that is re-raised as ``core.BackendError`` because every caller is
    a backend staging method whose contract is explicit backend errors.

    A loaded extension WITHOUT ``native_compile`` (the Rust CPU engine crate
    not built yet) is a distinct, equally explicit failure: the module is
    present but this backend cannot work, and the message names the missing
    entry point rather than pretending the dependency is absent.
    """
    from evoxc import _native

    try:
        module = _native.require()
    except ImportError as exc:
        raise core.BackendError(f"{_REMEDY.format(missing='the extension itself')} Loader error: {exc}") from exc
    missing = [attr for attr in _REQUIRED_ATTRS if not hasattr(module, attr)]
    if missing:
        raise core.BackendError(
            _REMEDY.format(missing=", ".join(missing))
        )
    return module


def engine_available() -> bool:
    """True iff the extension is loadable AND exposes ``native_compile``.

    Never raises (used by ``register()``'s probe and by docs/tests): a missing
    extension, a broken one, or one built without the CPU engine all report
    ``False``. The actionable error comes from the staging methods via
    :func:`engine_module`.
    """
    try:
        engine_module()
    except core.BackendError:
        return False
    return True


def engine_version() -> str:
    """The extension's reported version (recorded in ``native`` artifacts)."""
    from evoxc import _native

    try:
        return _native.version()
    except ImportError as exc:
        raise core.BackendError(f"{_REMEDY.format(missing='the extension itself')} Loader error: {exc}") from exc


def _translate(module, exc: BaseException) -> None:
    """Re-raise a ``NativeError`` as its public ``evoxc.core`` counterpart.

    Delegates to ``evoxc._native._translate`` — the ONE mapping implementation
    for ``NativeError.kind`` (message verbatim, chained; any other exception
    or unknown kind re-raised unchanged). Reusing it guarantees the CPU engine
    and the transform/infer entry points report identically.
    """
    from evoxc import _native

    _native._translate(module, exc)  # always raises


# ---------------------------------------------------------------------------
# Program lifecycle
# ---------------------------------------------------------------------------


def encode_module(module) -> dict:
    """Encode an ``ir.Module`` into the ``evoxc-native-ir`` envelope dict.

    Wraps ``evoxc._native_codec.encode_module`` — the envelope the transform
    entry points already use, which verifies the module before emitting (a
    ``core.VerificationError`` surfaces as-is). Encoding is pure: the same
    module always yields an equivalent envelope, so a caller may compute it
    once and hand it to :func:`compile_module` to avoid a second encode.
    """
    from evoxc import _native_codec

    return _native_codec.encode_module(module)


def compile_module(module, envelope: Mapping | None = None) -> Any:
    """Compile an ``ir.Module`` into a native ``NativeProgram``.

    Encodes the module with :func:`encode_module` (the ``evoxc-native-ir``
    envelope the transform entry points already use — it verifies the module
    before emitting) and calls ``native_compile``. A ``NativeError`` is
    translated onto the ``evoxc.core`` family; the result is validated against
    the ``NativeProgram`` surface the backend relies on.

    ``envelope``: an ALREADY-ENCODED envelope for ``module``, used verbatim
    instead of re-encoding. The caller owns the invariant that the envelope
    belongs to this module (the backend only forwards one it attached to the
    very payload object it staged) — a stale envelope would compile a
    different graph, so never pass one across a payload change.

    ``lower()`` calls this as its capability probe (the engine is the
    authority on which ops it implements) and ``load()`` calls it to build
    the runnable program; both get the same explicit errors.
    """
    native = engine_module()
    if envelope is None:
        envelope = encode_module(module)
    try:
        program = native.native_compile(envelope)
    except Exception as exc:  # noqa: BLE001 — translated only for NativeError
        _translate(native, exc)
        raise  # unreachable; keeps linters honest
    return _validate_program(program)


def _validate_program(program: Any) -> Any:
    """Validate the object ``native_compile`` returned against the FFI surface.

    A plain object with the documented methods is expected. A malformed
    return (wrong type, missing ``run``/``entry``) is a contract violation and
    raises ``core.BackendError`` naming what is missing — never a late
    ``AttributeError`` from inside a run.
    """
    missing = [
        method for method in _REQUIRED_METHODS if not callable(getattr(program, method, None))
    ]
    if missing:
        raise core.BackendError(
            "native_compile returned an object that does not implement the "
            f"NativeProgram contract (missing {', '.join(missing)}; got "
            f"{type(program).__name__})"
        )
    return program


def _optional_attr(program: Any, name: str):
    """Fetch a diagnostic accessor, failing explicitly when the engine lacks it."""
    accessor = getattr(program, name, None)
    if not callable(accessor):
        raise core.BackendError(
            f"the native CPU engine's NativeProgram does not expose {name}() — "
            "it is part of the documented FFI surface (see "
            "evoxc/backends/native/CONTEXT.md); rebuild the extension from a "
            "matching native/ tree"
        )
    return accessor


def program_run(
    program: Any,
    flat_inputs: Iterable[Any],
    resolver: Any = None,
    native: Any = None,
):
    """``NativeProgram.run(flat_inputs, resolver)`` with error translation.

    ``flat_inputs`` are numpy arrays (host data — the engine's v1 contract);
    ``resolver`` is the callback resolver object (``call_block`` /
    ``call_external`` / ``call_runtime``, see ``resolver.py``) or ``None`` for
    a graph with no callback boundaries.

    ``native``: an ALREADY-RESOLVED extension module (the caller's cached
    :func:`engine_module` result), used instead of re-resolving it on every
    call — the hot path of a device-free ``run``. ``None`` resolves it here,
    with the same explicit ``core.BackendError`` when it is unavailable.
    """
    if native is None:
        native = engine_module()
    try:
        return program.run(
            flat_inputs if isinstance(flat_inputs, list) else list(flat_inputs),
            resolver,
        )
    except Exception as exc:  # noqa: BLE001 — translated only for NativeError
        _translate(native, exc)
        raise  # unreachable; keeps linters honest


def program_entry(program: Any) -> str:
    """The engine's entry-function name."""
    entry = program.entry()
    if not isinstance(entry, str) or not entry:
        raise core.BackendError(
            "the native CPU engine's NativeProgram.entry() returned "
            f"{entry!r} where a non-empty function name was expected"
        )
    return entry


def program_plan_text(program: Any) -> str:
    """Human-readable native dispatch plan (debugging aid)."""
    text = _optional_attr(program, "plan_text")()
    if not isinstance(text, str):
        raise core.BackendError(
            "the native CPU engine's NativeProgram.plan_text() returned "
            f"{type(text).__name__} where a string was expected"
        )
    return text


def program_stats(program: Any) -> dict:
    """The engine's compile/plan statistics dict (debugging aid)."""
    stats = _optional_attr(program, "stats")()
    if not isinstance(stats, Mapping):
        raise core.BackendError(
            "the native CPU engine's NativeProgram.stats() returned "
            f"{type(stats).__name__} where a mapping was expected"
        )
    return dict(stats)


def program_required_ops(program: Any):
    """The engine's required-ops report (``blocks``/``external``/``runtime_calls``).

    Returns ``None`` when the engine build predates this accessor, so a
    diagnostics gap never blocks staging — the authoritative requirement set
    is always derived from the serialized module by ``compile()``.
    """
    accessor = getattr(program, "required_ops", None)
    if not callable(accessor):
        return None
    required = accessor()
    if not isinstance(required, Mapping):
        raise core.BackendError(
            "the native CPU engine's NativeProgram.required_ops() returned "
            f"{type(required).__name__} where a mapping was expected"
        )
    return {
        str(key): tuple(value) if isinstance(value, (list, tuple)) else value
        for key, value in required.items()
    }


def program_output_types(program: Any):
    """The engine's encoded output ``ValueType`` dicts (``None`` when absent)."""
    accessor = getattr(program, "output_types", None)
    if not callable(accessor):
        return None
    types = accessor()
    if not isinstance(types, (list, tuple)):
        raise core.BackendError(
            "the native CPU engine's NativeProgram.output_types() returned "
            f"{type(types).__name__} where a list was expected"
        )
    return list(types)


def unsupported_ops(stats: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract the engine's unimplemented-op report from a stats mapping.

    The FFI contract names the key ``unsupported ops``; the tolerant lookup
    accepts the underscore spelling (and either a list of strings or a list of
    records carrying an ``op``/``name`` field) so an early engine build cannot
    make the backend report "no unsupported ops" by spelling drift — the
    check that consumes this is the explicit-op-support gate at ``lower()``.
    """
    raw: Any = ()
    for key in ("unsupported_ops", "unsupported ops", "unsupported"):
        value = stats.get(key)
        if value is not None:
            raw = value
            break
    names: list[str] = []
    for entry in raw or ():
        if isinstance(entry, str):
            names.append(entry)
        elif isinstance(entry, Mapping):
            name = entry.get("op", entry.get("name"))
            if isinstance(name, str):
                names.append(name)
            else:
                names.append(repr(dict(entry)))
        else:
            names.append(repr(entry))
    return tuple(sorted(set(names)))
