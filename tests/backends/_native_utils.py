"""Shared helpers for the ``native`` (Rust CPU engine) backend test suites.

NOT collected (underscore prefix; import package-qualified as
``from tests.backends import _native_utils as nu``).

The ``numpy`` backend is the reference oracle everywhere: a graph is evaluated
twice — once on ``backend="numpy"`` (the pure-Python interpreter) and once on
``backend="native"`` (the in-process Rust engine) — and the values compared with
a per-dtype tolerance. Exact equality for bool/integer, a defensible
``rtol``/``atol`` per float width (see :data:`TOLERANCES`).

The engine is OPTIONAL. Most suites gate on :data:`ENGINE_AVAILABLE`; the one
test that must run without it is the missing-extension error pin in
``test_native_backend.py`` (see :func:`without_engine`).
"""
from __future__ import annotations

import contextlib
import os
from typing import Any

import numpy as np
import pytest

import evoxc
from evoxc import core
from evoxc.backends.native import engine

__all__ = [
    "ENGINE_AVAILABLE",
    "SKIP_REASON",
    "TOLERANCES",
    "as_numpy",
    "assert_parity",
    "assert_tree_parity",
    "build_native",
    "dtype_tolerance",
    "evaluate_native",
    "evaluate_numpy",
    "evaluate_pair",
    "run_native",
    "specs_for",
    "without_engine",
]

#: Skip reason used by every suite gated on the compiled extension.
SKIP_REASON = (
    "evoxc._evoxc_native is not built — run `source scripts/dev-env.sh && "
    "uv sync --locked --all-packages --extra dev && scripts/build-native.sh`"
)

#: True when the extension loads AND exposes ``native_compile``.
ENGINE_AVAILABLE = engine.engine_available()

#: Per-float-width comparison budgets, measured against the numpy reference.
#: f16 arithmetic is genuinely lossy (~3 decimal digits), f32 keeps ~1e-7, f64
#: ~1e-15; the budgets below leave a 10-100x margin over observed drift so a
#: legitimate reassociation/reduction-order difference never flakes while a
#: real numerical bug still fails.
TOLERANCES: dict[str, tuple[float, float]] = {
    "float16": (2e-3, 2e-3),
    "float32": (1e-5, 1e-6),
    "float64": (1e-12, 1e-13),
    "complex64": (1e-5, 1e-6),
    "complex128": (1e-12, 1e-13),
}


def dtype_tolerance(dtype: Any) -> tuple[float, float]:
    """``(rtol, atol)`` for ``dtype``; exact (0, 0) for bool/integer."""
    name = np.dtype(dtype).name
    return TOLERANCES.get(name, (0.0, 0.0))


def as_numpy(value: Any) -> Any:
    """Recursively replace every ``core.Tensor`` leaf with its host ndarray."""
    if isinstance(value, core.Tensor):
        return value.numpy()
    if isinstance(value, tuple):
        return tuple(as_numpy(entry) for entry in value)
    if isinstance(value, list):
        return [as_numpy(entry) for entry in value]
    if isinstance(value, dict):
        return {key: as_numpy(entry) for key, entry in value.items()}
    return value


def assert_parity(
    got: Any,
    expected: Any,
    *,
    dtype: Any = None,
    label: str = "",
) -> None:
    """Assert one array pair matches within its dtype's budget.

    dtype and shape must match exactly (a dtype mismatch is a real bug);
    values are compared with :func:`dtype_tolerance` of the ARRAY dtype unless
    ``dtype`` overrides it.
    """
    got = np.asarray(got)
    expected = np.asarray(expected)
    where = f"{label}: " if label else ""
    assert got.dtype == expected.dtype, (
        f"{where}dtype {got.dtype} != reference {expected.dtype}"
    )
    assert got.shape == expected.shape, (
        f"{where}shape {got.shape} != reference {expected.shape}"
    )
    rtol, atol = dtype_tolerance(dtype if dtype is not None else expected.dtype)
    np.testing.assert_allclose(
        got,
        expected,
        rtol=rtol,
        atol=atol,
        equal_nan=expected.dtype.kind in "fc",
        err_msg=f"{where}native backend disagrees with the numpy reference",
    )


def assert_tree_parity(got: Any, expected: Any, *, label: str = "") -> None:
    """Assert a pytree of arrays matches, leaf by leaf, with exact structure."""
    got_leaves, got_tree = core.tree_flatten(got)
    expected_leaves, expected_tree = core.tree_flatten(expected)
    assert got_tree == expected_tree, (
        f"{label}: output structure {got_tree} != reference {expected_tree}"
    )
    assert len(got_leaves) == len(expected_leaves)
    for index, (actual, reference) in enumerate(zip(got_leaves, expected_leaves)):
        assert_parity(actual, reference, label=f"{label}[{index}]")


def evaluate_native(fn_or_graph: Any, *args: Any) -> Any:
    """Evaluate on the native backend; returns host-ndarray leaves."""
    return as_numpy(evoxc.evaluate(fn_or_graph, *args, backend="native"))


def evaluate_numpy(fn_or_graph: Any, *args: Any) -> Any:
    """Evaluate on the numpy reference backend; returns host-ndarray leaves."""
    return as_numpy(evoxc.evaluate(fn_or_graph, *args, backend="numpy"))


def evaluate_pair(fn_or_graph: Any, *args: Any) -> Any:
    """Evaluate on both backends, assert parity, and return the native tree."""
    got = evaluate_native(fn_or_graph, *args)
    expected = evaluate_numpy(fn_or_graph, *args)
    assert_tree_parity(got, expected, label=getattr(fn_or_graph, "__name__", "graph"))
    return got


def specs_for(*arrays: Any) -> tuple[core.TensorSpec, ...]:
    """``TensorSpec``s matching concrete host arrays (for ``evoxc.build``)."""
    return tuple(
        core.TensorSpec(shape=tuple(np.shape(array)), dtype=np.asarray(array).dtype)
        for array in arrays
    )


def build_native(fn: Any, *specs: Any, **options: Any) -> Any:
    """``evoxc.build`` staged on the native backend."""
    return evoxc.build(fn, *specs, backend="native", **options)


def run_native(executable: Any, *args: Any) -> Any:
    """``evoxc.run`` a native executable, returning host-ndarray leaves."""
    return as_numpy(evoxc.run(executable, *args))


@contextlib.contextmanager
def without_engine(*, path: str | None = None):
    """Make the extension unloadable for the duration of the block.

    Points ``EVOXC_NATIVE_LIB`` at a path that does not exist AND clears the
    loader's cached module/failure (``evoxc._native._reset()``), so
    ``_native.require`` takes the real failure path instead of returning the
    already-loaded module. Everything is restored afterwards, including the
    loader cache — this is the ONE test seam that exercises the
    missing-extension contract, and it must not leak state into other tests.

    ``path`` overrides the bogus path (default: a non-existent file next to a
    temp directory that is never created).
    """
    from evoxc import _native

    bogus = path or os.path.join(
        os.path.dirname(os.path.abspath(evoxc.__file__)), "_does_not_exist_native.so"
    )
    with _native._LOCK:
        saved_module = _native._MODULE
        saved_error = _native._LOAD_ERROR
    saved_env = os.environ.get(_native.NATIVE_LIB_ENV)
    _native._reset()
    os.environ[_native.NATIVE_LIB_ENV] = bogus
    try:
        yield bogus
    finally:
        if saved_env is None:
            os.environ.pop(_native.NATIVE_LIB_ENV, None)
        else:
            os.environ[_native.NATIVE_LIB_ENV] = saved_env
        with _native._LOCK:
            _native._MODULE = saved_module
            _native._LOAD_ERROR = saved_error


def require_engine() -> None:
    """Skip the calling test when the extension is absent (helper form)."""
    if not ENGINE_AVAILABLE:  # pragma: no cover - environment guard
        pytest.skip(SKIP_REASON)
