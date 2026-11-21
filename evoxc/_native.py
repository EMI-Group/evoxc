"""Optional Rust/PyO3 extension loader and typed façade.

``evoxc._evoxc_native`` is the compiled numerical core being built under
``native/`` (dtypes/dims/shapes, op schemas and shape inference, the graph IR,
the batching and AD engines). It is an **optional** acceleration seam:

* ``import evoxc`` never imports or requires it;
* the extension is discovered lazily on first use — the ``EVOXC_NATIVE_LIB``
  path when set, otherwise the first match of
  ``_evoxc_native.abi3.so`` / ``_evoxc_native*.so`` / ``*.pyd`` / ``*.dylib``
  in the package directory (deterministic sorted order);
* loading is serialized by one module lock, with the loaded module and any
  load failure cached, so ``available()`` is stable across calls;
* a missing or broken extension produces one actionable error naming
  ``scripts/build-native.sh`` (and its ``source scripts/dev-env.sh``
  prerequisite), the expected ``evoxc/_evoxc_native.abi3.so`` file, and the
  ``EVOXC_NATIVE_LIB`` override, chaining the underlying loader error.

The façade converts evoxc's own IR objects to and from the neutral envelope in
:mod:`evoxc._native_codec` and maps the extension's ``NativeError.kind`` onto
the public ``evoxc.core`` error family (message preserved verbatim). Nothing
in evoxc delegates to this seam yet; ``ir/inference.py`` and ``transforms``
will flip onto it in a later pass.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys
import threading
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

from evoxc import core, ir

from . import _native_codec

__all__ = [
    "NATIVE_LIB_ENV",
    "EXTENSION_MODULE_NAME",
    "available",
    "require",
    "version",
    "infer",
    "transform",
]

#: Environment variable naming an explicit extension path (``.so``/``.pyd``).
NATIVE_LIB_ENV = "EVOXC_NATIVE_LIB"

#: Fully qualified name the extension is imported under.
EXTENSION_MODULE_NAME = "evoxc._evoxc_native"

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

#: Discovery patterns, in deterministic priority order.
_DISCOVERY_PATTERNS = (
    "_evoxc_native.abi3.so",
    "_evoxc_native*.so",
    "_evoxc_native*.pyd",
    "_evoxc_native*.dylib",
)

_EXPECTED_FILE = os.path.join("evoxc", "_evoxc_native.abi3.so")

_REMEDY = (
    "the evoxc native extension is not available. Build it with "
    "`source scripts/dev-env.sh && scripts/build-native.sh` "
    f"(it produces {_EXPECTED_FILE}), or set {NATIVE_LIB_ENV} to the path of "
    "a built extension (.so/.pyd)."
)

#: ``NativeError.kind`` -> public ``evoxc.core`` error class.
_KIND_TO_ERROR = {
    "ShapeError": core.ShapeError,
    "VerificationError": core.VerificationError,
    "DTypeError": core.DTypeError,
    "TransformError": core.TransformError,
    "BackendError": core.BackendError,
    "DeviceError": core.DeviceError,
    "TraceError": core.TraceError,
}

_LOCK = threading.RLock()
_MODULE: ModuleType | None = None
_LOAD_ERROR: ImportError | None = None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _discover_paths() -> list[str]:
    """Candidate extension paths (the env override, or sorted package matches)."""
    override = os.environ.get(NATIVE_LIB_ENV)
    if override:
        return [override]
    found: list[str] = []
    seen: set[str] = set()
    for pattern in _DISCOVERY_PATTERNS:
        for path in sorted(glob.glob(os.path.join(_PACKAGE_DIR, pattern))):
            if path not in seen:
                seen.add(path)
                found.append(path)
    return found


def _load_extension() -> ModuleType:
    """Discover and import the extension, raising ``ImportError`` with context."""
    override = os.environ.get(NATIVE_LIB_ENV)
    paths = _discover_paths()
    if override and not os.path.isfile(override):
        raise ImportError(
            f"{NATIVE_LIB_ENV} is set to {override!r}, but that file does not exist"
        )
    if not paths:
        raise ImportError(
            f"no native extension was found in {_PACKAGE_DIR!r} "
            f"(looked for {', '.join(_DISCOVERY_PATTERNS)})"
        )
    path = paths[0]
    spec = importlib.util.spec_from_file_location(EXTENSION_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"could not create an import spec for the native extension at {path!r}"
        )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError:
        raise
    except Exception as exc:  # a broken .so must not crash the interpreter
        raise ImportError(f"failed to load the native extension from {path!r}: {exc}") from exc
    sys.modules[EXTENSION_MODULE_NAME] = module
    return module


def _get_module() -> ModuleType:
    """Return the loaded extension (double-checked, cached; raises ``ImportError``)."""
    global _MODULE, _LOAD_ERROR
    module = _MODULE
    if module is not None:
        return module
    with _LOCK:
        if _MODULE is not None:
            return _MODULE
        if _LOAD_ERROR is not None:
            raise _LOAD_ERROR
        try:
            module = _load_extension()
        except ImportError as exc:
            _LOAD_ERROR = exc
            raise
        except Exception as exc:  # available() must never raise
            error = ImportError(
                f"failed to load the evoxc native extension: {exc}"
            )
            _LOAD_ERROR = error
            raise error from exc
        _MODULE = module
        return module


def _reset() -> None:
    """Forget the cached module and load failure (private; for tests)."""
    global _MODULE, _LOAD_ERROR
    with _LOCK:
        _MODULE = None
        _LOAD_ERROR = None


def available() -> bool:
    """True if the native extension can be loaded; never raises and is cached."""
    try:
        _get_module()
    except ImportError:
        return False
    return True


def require() -> ModuleType:
    """Return the loaded extension or raise an actionable ``ImportError``."""
    try:
        return _get_module()
    except ImportError as exc:
        raise ImportError(f"{_REMEDY} Loader error: {exc}") from exc


def version() -> str:
    """The extension's reported version string."""
    return require().version()


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def _translate(module: ModuleType, exc: BaseException) -> None:
    """Re-raise a ``NativeError`` as its public ``evoxc.core`` counterpart.

    Only exceptions that are instances of this extension's ``NativeError`` and
    whose ``kind`` is known are translated (message verbatim, chained). Any
    other exception or unknown kind is re-raised unchanged — never swallowed.
    """
    native_error = getattr(module, "NativeError", None)
    if native_error is None or not isinstance(exc, native_error):
        raise exc
    kind = getattr(exc, "kind", None)
    error_cls = _KIND_TO_ERROR.get(kind) if isinstance(kind, str) else None
    if error_cls is None:
        raise exc
    raise error_cls(str(exc)) from exc


# ---------------------------------------------------------------------------
# Typed façade
# ---------------------------------------------------------------------------


def _encode_types(value_types: Iterable[ir.ValueType]) -> list[dict]:
    return [_native_codec.encode_type(value_type) for value_type in value_types]


def _decode_types(payload: Any) -> list[ir.ValueType]:
    if not isinstance(payload, (list, tuple)):
        raise TypeError(
            f"the native extension returned {type(payload).__name__} where a list "
            "of value types was expected"
        )
    return [_native_codec.decode_type(entry) for entry in payload]


def _encode_module_context(module: Mapping[str, Any] | None) -> dict | None:
    """Encode the optional ``infer`` module context (function signatures)."""
    if module is None:
        return None
    if not isinstance(module, Mapping):
        raise TypeError(
            f"infer: 'module' must be a mapping, got {type(module).__name__}"
        )
    functions = module.get("functions")
    if functions is None:
        return dict(module)
    encoded = []
    for entry in functions:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise TypeError(
                "infer: 'module' functions must be (name, [ValueType, ...]) pairs, "
                f"got {entry!r}"
            )
        name, value_types = entry
        encoded.append((name, _encode_types(value_types)))
    context = dict(module)  # preserve every other key untouched
    context["functions"] = encoded
    return context


def infer(
    op_name: str,
    operand_types: Sequence[ir.ValueType],
    attributes: dict[str, Any],
    result_types: Sequence[ir.ValueType] | None = None,
    module: Mapping[str, Any] | None = None,
) -> list[ir.ValueType]:
    """Infer result types through the native core.

    Operands (and ``result_types`` when given) are encoded as neutral
    ``ValueType`` dicts; ``attributes`` pass through verbatim. ``module`` may
    supply ``{"functions": [(name, [ValueType, ...]), ...]}`` whose value types
    are encoded (other keys pass through, ``None``/absent untouched).
    """
    if not isinstance(op_name, str):
        raise TypeError(f"infer: 'op_name' must be a string, got {type(op_name).__name__}")
    if not isinstance(attributes, dict):
        raise TypeError(
            f"infer: 'attributes' must be a dict, got {type(attributes).__name__}"
        )
    operand_payload = _encode_types(operand_types)
    result_payload = None if result_types is None else _encode_types(result_types)
    module_payload = _encode_module_context(module)

    native = require()
    try:
        raw = native.infer(
            op_name, operand_payload, attributes, result_payload, module_payload
        )
    except Exception as exc:  # translated only for a known NativeError.kind
        _translate(native, exc)
        raise  # unreachable; keeps linters/type-checkers honest
    return _decode_types(raw)


def transform(kind: str, module: ir.Module, config: dict[str, Any]) -> dict[str, Any]:
    """Run a native graph transform over ``module``.

    Encodes the module with :func:`evoxc._native_codec.encode_module`, calls
    the extension, and decodes ``result["module"]`` (and ``output_types`` when
    present) back to IR objects. Every other key/value — including a callable
    ``config["extension_rule"]``, which is never encoded — is preserved
    verbatim in a NEW dict.
    """
    if not isinstance(kind, str):
        raise TypeError(f"transform: 'kind' must be a string, got {type(kind).__name__}")
    if not isinstance(module, ir.Module):
        raise TypeError(
            f"transform: 'module' must be an ir.Module, got {type(module).__name__}"
        )
    if not isinstance(config, dict):
        raise TypeError(
            f"transform: 'config' must be a dict, got {type(config).__name__}"
        )
    envelope = _native_codec.encode_module(module)

    native = require()
    try:
        result = native.transform(kind, envelope, config)
    except Exception as exc:  # translated only for a known NativeError.kind
        _translate(native, exc)
        raise  # unreachable; keeps linters/type-checkers honest
    if not isinstance(result, Mapping):
        raise TypeError(
            f"the native extension's transform returned {type(result).__name__} "
            "where a mapping was expected"
        )
    output = dict(result)
    if "module" in output:
        if not isinstance(output["module"], Mapping):
            raise TypeError(
                "the native extension's transform result 'module' must be an "
                f"envelope mapping, got {type(output['module']).__name__}"
            )
        output["module"] = _native_codec.decode_module(output["module"])
    if output.get("output_types") is not None:
        output["output_types"] = _decode_types(output["output_types"])
    return output
