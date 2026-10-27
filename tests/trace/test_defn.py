"""Tests for `@evoxc.defn` — the graph-definition marker decorator.

Contract under test (`evoxc/trace/defn.py` + `evoxc/trace/CONTEXT.md`):
- `@evoxc.defn` (bare or with options) returns an `evoxc.Defn` that stores the
  ORIGINAL function object — it does NOT wrap, so metadata is naturally
  preserved.
- A `Defn` is NEVER callable: direct calls always raise `evoxc.TraceError`
  directing to `evoxc.trace(defn, *specs)` / `evoxc.evaluate(defn, *args)`.
- Idempotent: `evoxc.defn(existing_defn)` returns the same object.
- `evoxc.trace` accepts a Defn and produces the same Graph as tracing the
  plain function.

Constants inside traced functions MUST be `evoxc.constant(evoxc.tensor(v,
dtype=d))` — never captured concrete Tensors (see spec-compliance tests).
"""

import inspect

import numpy as np
import pytest

import evoxc


# --- decorator forms --------------------------------------------------------


def test_bare_decorator_returns_defn_marking_both():
    @evoxc.defn
    def f(x):
        return evoxc.add(x, x)

    assert isinstance(f, evoxc.Defn)
    assert f.__evoxc_defn__
    assert f.fn.__evoxc_defn__


def test_defn_stores_the_original_function_object():
    def f(x):
        return evoxc.add(x, x)

    d = evoxc.defn(f)

    assert d.fn is f


def test_options_form_stores_options_and_bare_defaults_to_empty():
    def f(x):
        return evoxc.add(x, x)

    bare = evoxc.defn(f)
    assert bare.options == {}

    @evoxc.defn(compiler_hint="iree")
    def g(x):
        return evoxc.add(x, x)

    assert g.options == {"compiler_hint": "iree"}


# --- idempotence ------------------------------------------------------------


def test_defn_of_defn_is_idempotent():
    def f(x):
        return evoxc.add(x, x)

    d = evoxc.defn(f)
    again = evoxc.defn(d)

    assert again is d


def test_defn_fn_is_always_the_unwrapped_plain_function():
    def f(x):
        return evoxc.add(x, x)

    d = evoxc.defn(f)
    again = evoxc.defn(d)  # unwraps — never nests Defns

    assert d.fn is f
    assert again.fn is f
    assert not isinstance(d.fn, evoxc.Defn)
    assert not isinstance(again.fn, evoxc.Defn)


# --- metadata preservation (defn does NOT wrap) -----------------------------


def test_defn_preserves_function_metadata():
    def f(x):
        """Add one to every element."""
        return evoxc.add(x, evoxc.constant(evoxc.tensor(1.0, dtype=evoxc.float32)))

    d = evoxc.defn(f)

    assert d.fn.__name__ == f.__name__
    assert d.fn.__doc__ == f.__doc__
    assert inspect.signature(d.fn) == inspect.signature(f)


def test_defn_repr_contains_function_name():
    @evoxc.defn
    def add_one(x):
        return evoxc.add(x, evoxc.constant(evoxc.tensor(1.0, dtype=evoxc.float32)))

    assert "add_one" in repr(add_one)


# --- calling a Defn always raises (no implicit eager) -----------------------

_CALL_CASES = [
    pytest.param((), id="no-args"),
    pytest.param((evoxc.TensorSpec((3,), evoxc.float32),), id="tensor-spec"),
    pytest.param(
        (np.array([1.0, 2.0, 3.0], dtype=np.float32),), id="concrete-ndarray"
    ),
]


@pytest.mark.parametrize("args", _CALL_CASES)
@pytest.mark.parametrize("needle", ["evoxc.trace", "evoxc.evaluate"])
def test_calling_defn_always_raises_traceerror_directing_to_staging(args, needle):
    @evoxc.defn
    def f(x):
        return evoxc.add(x, x)

    with pytest.raises(evoxc.TraceError, match=needle):
        f(*args)


# --- tracing a Defn equals tracing the plain fn -----------------------------


def test_trace_defn_matches_trace_plain_fn():
    @evoxc.defn
    def f(x):
        return evoxc.add(x, evoxc.constant(evoxc.tensor(1.0, dtype=evoxc.float32)))

    spec = evoxc.TensorSpec((3,), evoxc.float32)
    graph_from_defn = evoxc.trace(f, spec)
    graph_from_fn = evoxc.trace(f.fn, spec)

    assert evoxc.ir.serialize_module(graph_from_defn.module) == evoxc.ir.serialize_module(
        graph_from_fn.module
    )
