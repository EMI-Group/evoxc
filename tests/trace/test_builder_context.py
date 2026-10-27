"""Tests for the active-builder context (`current_builder` / `with_builder` /
`builder_stack`).

Contract under test (`evoxc/trace/builder.py` + `evoxc/trace/CONTEXT.md`):
- Outside any trace, `current_builder()` raises `evoxc.TraceError`
  ("No active trace") — this is the hook `evoxc.ops` uses, so op functions
  called outside a trace fail the same way.
- Inside `evoxc.trace(fn, ...)`, `current_builder()` returns the ONE
  `evoxc.ir.Builder` serving the whole trace: the same instance across ops
  in the traced fn AND inside control-flow regions (`cond`/`while_loop`
  callables) — `control_flow.py` swaps the builder's insertion point, it
  does not create new builders.
- `with_builder` (alias `builder_stack`) is a nestable LIFO context
  manager usable standalone.

NOTE (contract gap): the contract is silent about calling `evoxc.trace`
inside a traced function — no assertion is made for nested traces here.
"""

import numpy as np
import pytest

import evoxc
from evoxc.trace import builder_stack, current_builder, with_builder


# --- no trace active --------------------------------------------------------


def test_current_builder_raises_outside_trace():
    with pytest.raises(evoxc.TraceError, match="No active trace"):
        current_builder()


def test_op_outside_trace_raises_no_active_trace():
    """Ops query `current_builder()` — a SymbolicTensor from a finished
    trace cannot be used outside one."""
    captured = []

    def fn(x):
        captured.append(x)
        return evoxc.add(x, x)

    evoxc.trace(fn, evoxc.TensorSpec((2,), evoxc.float32))

    assert len(captured) == 1
    assert isinstance(captured[0], evoxc.SymbolicTensor)
    with pytest.raises(evoxc.TraceError, match="No active trace"):
        evoxc.add(captured[0], captured[0])


# --- active builder during a trace ------------------------------------------


def test_current_builder_inside_trace_is_stable_shared_builder():
    builders = []

    def fn(x):
        builders.append(current_builder())
        y = evoxc.add(x, x)
        builders.append(current_builder())
        return y

    evoxc.trace(fn, evoxc.TensorSpec((2,), evoxc.float32))

    assert len(builders) == 2
    assert isinstance(builders[0], evoxc.ir.Builder)
    # Same object on two separate ops — the contextvar stack is stable
    # for the duration of the trace.
    assert builders[0] is builders[1]


# --- manual with_builder installation ---------------------------------------


def test_with_builder_installs_and_restores():
    b = evoxc.ir.Builder()

    with pytest.raises(evoxc.TraceError, match="No active trace"):
        current_builder()

    with with_builder(b) as entered:
        assert entered is b
        assert current_builder() is b

    # Stack fully restored on exit.
    with pytest.raises(evoxc.TraceError, match="No active trace"):
        current_builder()


def test_with_builder_nesting_is_lifo():
    b1 = evoxc.ir.Builder()
    b2 = evoxc.ir.Builder()

    with with_builder(b1):
        assert current_builder() is b1
        with with_builder(b2):
            assert current_builder() is b2
        assert current_builder() is b1

    with pytest.raises(evoxc.TraceError, match="No active trace"):
        current_builder()


def test_builder_stack_is_alias_of_with_builder():
    assert builder_stack is with_builder


# --- ops route into nested control-flow regions on the SAME builder ---------


def test_while_loop_regions_use_the_same_builder_instance():
    trace_builders = []
    region_builders = []

    @evoxc.defn
    def f(x):
        trace_builders.append(current_builder())

        def cond_fn(state):
            region_builders.append(("cond", current_builder()))
            i, acc = state
            return evoxc.less(i, evoxc.constant(evoxc.tensor(3, dtype=evoxc.int32)))

        def body_fn(state):
            region_builders.append(("body", current_builder()))
            i, acc = state
            return (
                evoxc.add(i, evoxc.constant(evoxc.tensor(1, dtype=evoxc.int32))),
                evoxc.add(acc, x),
            )

        return evoxc.while_loop(
            cond_fn,
            body_fn,
            (
                evoxc.constant(evoxc.tensor(0, dtype=evoxc.int32)),
                evoxc.constant(evoxc.zeros((2,), evoxc.float32)),
            ),
        )

    evoxc.trace(f, evoxc.TensorSpec((2,), evoxc.float32))

    # cond_fn and body_fn each ran once at trace time.
    assert [where for where, _ in region_builders] == ["cond", "body"]
    for where, builder in region_builders:
        assert isinstance(builder, evoxc.ir.Builder)
        assert builder is trace_builders[0]


def test_while_loop_runs_end_to_end(run_graph, as_numpy):
    @evoxc.defn
    def f(x):
        def cond_fn(state):
            i, acc = state
            return evoxc.less(i, evoxc.constant(evoxc.tensor(3, dtype=evoxc.int32)))

        def body_fn(state):
            i, acc = state
            return (
                evoxc.add(i, evoxc.constant(evoxc.tensor(1, dtype=evoxc.int32))),
                evoxc.add(acc, x),
            )

        return evoxc.while_loop(
            cond_fn,
            body_fn,
            (
                evoxc.constant(evoxc.tensor(0, dtype=evoxc.int32)),
                evoxc.constant(evoxc.zeros((2,), evoxc.float32)),
            ),
        )

    graph = evoxc.trace(f, evoxc.TensorSpec((2,), evoxc.float32))
    result = as_numpy(run_graph(graph, np.array([10.0, 20.0], dtype=np.float32)))

    i, acc = result
    assert i == 3
    np.testing.assert_allclose(acc, np.array([30.0, 60.0], dtype=np.float32))
