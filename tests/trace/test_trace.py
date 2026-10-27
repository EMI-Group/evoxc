"""Contract tests for the core tracer `evoxc.trace(fn_or_defn, *specs) -> Graph`.

Covers the 7-step trace algorithm documented in `evoxc/trace/CONTEXT.md`:
flat tracing, symbolic/`None` dims, structured (pytree) I/O, static-value
snapshotting/specialization, closure-capture rules, plain-fn acceptance,
zero-IO graphs, invalid input/output leaves, output-static re-insertion,
SymbolicTensor purity, and no-caching freshness.

The evoxc package is the system under test (read-only from here). Only the
container *skeleton* of `Graph.input_specs` is asserted — trace trees record
TensorSpec/SymbolicTensor leaves via private marker types (`_TensorSpecLeaf` /
`_SymbolicLeaf`) by design (see the "Tree leaf markers" note in the trace
CONTEXT.md).
"""

import dataclasses
import enum
from typing import NamedTuple

import numpy as np
import pytest

import evoxc


# --- module-level pytree containers used as structured inputs -----------------


class PairNT(NamedTuple):
    a: object
    b: object


@dataclasses.dataclass
class PairDC:
    w: object
    z: object


class Mode(enum.Enum):
    A = 1
    B = 2


# --- helpers ------------------------------------------------------------------


def _op_names(graph: evoxc.Graph) -> list:
    """Op names of the entry function body, in program order."""
    return [op.name for op in graph.module.main.entry_block.ops]


def _skeleton(spec):
    """Reduce a `evoxc.core.TreeSpec` to its container skeleton.

    Leaf types are deliberately ignored (trace-time leaf types are private
    markers); container nodes keep (type, node_data, children).
    """
    if not spec.children:
        return "leaf"
    node_data = spec.node_data
    if node_data is not None:
        node_data = tuple(node_data)
    return (spec.type, node_data, tuple(_skeleton(child) for child in spec.children))


def _symbolic_tensor() -> evoxc.SymbolicTensor:
    """A fresh SymbolicTensor wrapping a traced graph's block argument."""
    graph = evoxc.trace(lambda x: x, evoxc.TensorSpec((3,), evoxc.float32))
    arg = graph.module.main.entry_block.arguments[0]
    return evoxc.SymbolicTensor(value=arg, dtype=evoxc.float32, shape=(3,))


# --- 1. basic trace ------------------------------------------------------------


def test_basic_trace_flat_fn(run_graph, as_numpy):
    def f(x):
        return evoxc.add(x, evoxc.constant(evoxc.tensor(1.0, dtype=evoxc.float32)))

    spec = evoxc.TensorSpec((3,), evoxc.float32)
    g = evoxc.trace(f, spec)

    assert isinstance(g, evoxc.Graph)
    assert g.tensor_specs == (spec,)
    assert isinstance(g.module, evoxc.ir.Module)
    main = g.module.main
    assert isinstance(main, evoxc.ir.Function)
    assert main.name == "main"
    assert len(main.entry_block.arguments) == 1
    assert _op_names(g) == ["constant", "add", "return"]

    g.verify()

    result = run_graph(g, np.ones(3, np.float32))
    np.testing.assert_array_equal(as_numpy(result), np.ones(3, np.float32) + 1)


# --- 2. symbolic and runtime-dynamic dims --------------------------------------


def test_symbolic_dims_preserved_and_accepted_at_run(run_graph, as_numpy):
    b_dim, h_dim = evoxc.Dim("B"), evoxc.Dim("H")
    spec = evoxc.TensorSpec((b_dim, h_dim), evoxc.float32)

    def f(x):
        return evoxc.add(x, x)

    g = evoxc.trace(f, spec)
    traced_shape = g.tensor_specs[0].shape
    assert traced_shape[0] is b_dim
    assert traced_shape[1] is h_dim

    g.verify()
    result = run_graph(g, np.ones((2, 3), np.float32))
    assert as_numpy(result).shape == (2, 3)


def test_none_dims_runtime_dynamic(run_graph, as_numpy):
    def f(x):
        return evoxc.add(x, x)

    g = evoxc.trace(f, evoxc.TensorSpec((None, 3), evoxc.float32))
    for leading in (1, 5):
        result = run_graph(g, np.zeros((leading, 3), np.float32))
        assert as_numpy(result).shape == (leading, 3)


@pytest.mark.parametrize("bad_shape", [(5,), (5, 3, 2)], ids=["rank-1", "rank-3"])
def test_none_dim_rank_mismatch_raises(run_graph, bad_shape):
    def f(x):
        return evoxc.add(x, x)

    g = evoxc.trace(f, evoxc.TensorSpec((None, 3), evoxc.float32))
    with pytest.raises(evoxc.ShapeError, match="rank mismatch"):
        run_graph(g, np.zeros(bad_shape, np.float32))


# --- 3. structured (pytree) inputs ---------------------------------------------


def test_structured_inputs_round_trip(run_graph, as_numpy):
    x_spec = evoxc.TensorSpec((2,), evoxc.float32)
    y = {
        "a": evoxc.TensorSpec((2,), evoxc.float32),
        "dc": PairDC(evoxc.TensorSpec((3,), evoxc.float32), evoxc.TensorSpec((4,), evoxc.float32)),
        "l": [evoxc.TensorSpec((8,), evoxc.float32), evoxc.TensorSpec((9,), evoxc.float32)],
        "n": PairNT(evoxc.TensorSpec((3,), evoxc.float32), evoxc.TensorSpec((2,), evoxc.float32)),
        "t": (
            evoxc.TensorSpec((5,), evoxc.float32),
            (evoxc.TensorSpec((6,), evoxc.float32), evoxc.TensorSpec((7,), evoxc.float32)),
        ),
    }

    def f(x, y):
        return (evoxc.add(x, y["a"]), {"neg": evoxc.negate(y["n"].a)})

    g = evoxc.trace(f, x_spec, y)

    # The recorded input tree matches the container skeleton of the traced
    # structure (leaf types are private markers by design — skeleton only).
    mirror = {
        "a": None,
        "dc": PairDC(None, None),
        "l": [None, None],
        "n": PairNT(None, None),
        "t": (None, (None, None)),
    }
    _, mirror_tree = evoxc.core.flatten((None, mirror))
    assert g.input_specs.type is tuple  # the specs tuple is the root pytree
    assert _skeleton(g.input_specs) == _skeleton(mirror_tree)

    # tensor_specs = flat pre-order leaf order, dict keys sorted
    # (x, a, dc.w, dc.z, l[0], l[1], n.a, n.b, t[0], t[1][0], t[1][1]).
    assert [s.shape for s in g.tensor_specs] == [
        (2,),
        (2,),
        (3,),
        (4,),
        (8,),
        (9,),
        (3,),
        (2,),
        (5,),
        (6,),
        (7,),
    ]

    # One recorded source location per tensor input.
    assert len(g.source_locations) == 11
    assert all(
        isinstance(location, evoxc.ir.Location)
        for location in g.source_locations.values()
    )

    g.verify()

    yin = {
        "a": np.ones(2, np.float32),
        "dc": PairDC(np.ones(3, np.float32), np.ones(4, np.float32)),
        "l": [np.ones(8, np.float32), np.ones(9, np.float32)],
        "n": PairNT(np.full(3, 5.0, np.float32), np.ones(2, np.float32)),
        "t": (
            np.ones(5, np.float32),
            (np.ones(6, np.float32), np.ones(7, np.float32)),
        ),
    }
    summed, rest = run_graph(g, np.ones(2, np.float32), yin)
    np.testing.assert_allclose(as_numpy(summed), np.full(2, 2.0, np.float32))
    np.testing.assert_allclose(as_numpy(rest["neg"]), np.full(3, -5.0, np.float32))


# --- 4. static values: snapshot, specialization, validation --------------------


def test_static_int_specializes_graph():
    def f(x, k):
        return evoxc.add(x, evoxc.constant(evoxc.tensor(k, dtype=evoxc.float32)))

    spec = evoxc.TensorSpec((2,), evoxc.float32)
    g1 = evoxc.trace(f, spec, 1)
    g2 = evoxc.trace(f, spec, 2)

    assert len(g1.static_values) == 1
    record = g1.static_values[0]
    assert record.index == 1
    assert record.path == (1,)
    assert record.value == 1
    assert record.kind == "int"

    serialized1 = evoxc.ir.serialize_module(g1.module)
    serialized2 = evoxc.ir.serialize_module(g2.module)
    assert serialized1 != serialized2
    assert serialized1["constants"] != serialized2["constants"]


@pytest.mark.parametrize(
    ("flag", "present", "absent"),
    [(True, "multiply", "negate"), (False, "negate", "multiply")],
)
def test_static_python_if_specializes(flag, present, absent):
    def f(x, flag):
        if flag:
            return evoxc.multiply(x, x)
        return evoxc.negate(x)

    g = evoxc.trace(f, evoxc.TensorSpec((2,), evoxc.float32), flag)
    names = _op_names(g)
    assert present in names
    assert absent not in names
    g.verify()


STATIC_CASES = [
    pytest.param(1.5, "float", id="float"),
    pytest.param("hello", "str", id="str"),
    pytest.param(None, "NoneType", id="none"),
    pytest.param(1 + 2j, "complex", id="complex"),
    pytest.param(True, "bool", id="bool"),
    pytest.param(slice(0, 2), "slice", id="slice"),
    pytest.param(Mode.B, "Mode", id="enum"),
    # np.dtype kind name is numpy-version-dependent (dtype vs Float32DType)
    pytest.param(np.dtype("float32"), None, id="dtype"),
]


@pytest.mark.parametrize(("value", "expected_kind"), STATIC_CASES)
def test_static_value_types_accepted(value, expected_kind):
    def f(x, s):
        return evoxc.multiply(x, 2.0)

    g = evoxc.trace(f, evoxc.TensorSpec((2,), evoxc.float32), value)
    assert len(g.static_values) == 1
    record = g.static_values[0]
    assert record.index == 1
    assert record.path == (1,)
    assert record.value == value
    if expected_kind is not None:
        assert record.kind == expected_kind
    g.verify()


def test_static_value_run_mismatch_raises(run_graph):
    def f(x, k):
        return evoxc.add(x, evoxc.constant(evoxc.tensor(k, dtype=evoxc.float32)))

    g = evoxc.trace(f, evoxc.TensorSpec((2,), evoxc.float32), 1)
    with pytest.raises(evoxc.TraceError, match="graph was specialized on"):
        run_graph(g, np.ones(2, np.float32), 7)
    # A wrong static TYPE at run time is rejected the same way.
    with pytest.raises(evoxc.TraceError, match="graph was specialized on"):
        run_graph(g, np.ones(2, np.float32), 1.0)


# --- 5. closure capture ---------------------------------------------------------


@pytest.mark.parametrize("closure", [2, 3.5], ids=["int", "float"])
def test_closure_scalar_captured_and_snapshotted(closure, run_graph, as_numpy):
    def f(x):
        return evoxc.multiply(x, closure)

    g = evoxc.trace(f, evoxc.TensorSpec((3,), evoxc.float32))
    assert "constant" in _op_names(g)
    result = run_graph(g, np.ones(3, np.float32))
    np.testing.assert_allclose(as_numpy(result), np.ones(3, np.float32) * closure)


def test_closure_captured_tensor_raises():
    w = evoxc.tensor(np.ones(3, np.float32))

    def f(x):
        return evoxc.add(x, w)

    with pytest.raises(evoxc.TraceError, match="Concrete Tensor"):
        evoxc.trace(f, evoxc.TensorSpec((3,), evoxc.float32))


# --- 6. plain functions ----------------------------------------------------------


def test_plain_fn_equivalent_to_defn():
    def body(x):
        return evoxc.multiply(x, x)

    decorated = evoxc.defn(body)  # the SAME fn object, wrapped
    spec = evoxc.TensorSpec((2,), evoxc.float32)

    g_defn = evoxc.trace(decorated, spec)
    g_plain = evoxc.trace(body, spec)
    g_unwrapped = evoxc.trace(decorated.fn, spec)

    assert isinstance(g_defn, evoxc.Graph)
    assert isinstance(g_plain, evoxc.Graph)
    assert evoxc.ir.serialize_module(g_defn.module) == evoxc.ir.serialize_module(
        g_plain.module
    )
    assert evoxc.ir.serialize_module(g_defn.module) == evoxc.ir.serialize_module(
        g_unwrapped.module
    )


# --- 7. zero-IO graphs -----------------------------------------------------------


def test_zero_input_static_output(run_graph):
    def f():
        return 3

    g = evoxc.trace(f)
    assert len(g.tensor_specs) == 0
    assert len(g.static_values) == 0
    assert len(g.output_static_values) == 1
    record = g.output_static_values[0]
    assert record.index == 0
    assert record.path == ()
    assert record.value == 3
    assert record.kind == "int"

    g.verify()
    assert run_graph(g) == 3


def test_zero_tensor_output_consuming_input(run_graph):
    def f(x):
        _ = evoxc.add(x, x)  # builds IR …
        return None  # … but yields no tensor result

    g = evoxc.trace(f, evoxc.TensorSpec((3,), evoxc.float32))
    terminator = g.module.main.entry_block.terminator
    assert terminator.name == "return"
    assert terminator.operands == ()
    g.verify()
    assert run_graph(g, np.ones(3, np.float32)) is None


def test_zero_args_tensor_output(run_graph, as_numpy):
    def f():
        return evoxc.constant(evoxc.zeros((2,), evoxc.float32))

    g = evoxc.trace(f)
    assert len(g.tensor_specs) == 0
    g.verify()
    result = run_graph(g)
    np.testing.assert_array_equal(as_numpy(result), np.zeros(2, np.float32))


# --- 8. error cases --------------------------------------------------------------

BAD_SPEC_CASES = [
    pytest.param(
        evoxc.tensor(np.ones(3, np.float32)), id="concrete-tensor"
    ),
    pytest.param(_symbolic_tensor(), id="symbolic-tensor"),
    pytest.param(object(), id="unknown-object"),
]


@pytest.mark.parametrize("bad_spec", BAD_SPEC_CASES)
def test_invalid_input_spec_raises(bad_spec):
    def f(x):
        return x

    with pytest.raises(evoxc.TraceError) as excinfo:
        evoxc.trace(f, bad_spec)
    # every error names the pytree path of the offending leaf
    assert "pytree path [0]" in str(excinfo.value)


def test_ndarray_input_spec_is_static():
    """A raw ndarray spec leaf IS a static value — accepted at trace time and
    recorded in `static_values` (never a tensor input)."""

    def f(x, arr):
        return x

    arr = np.ones(3, np.float32)
    g = evoxc.trace(f, evoxc.TensorSpec((3,), evoxc.float32), arr)
    assert len(g.tensor_specs) == 1  # the TensorSpec only — not the ndarray
    (record,) = g.static_values
    assert record.kind == "ndarray"
    np.testing.assert_array_equal(record.value, arr)


BAD_OUTPUT_CASES = [
    pytest.param(
        lambda x: evoxc.tensor(np.ones(3, np.float32)),
        "Invalid trace output at pytree path",
        id="concrete-tensor",
    ),
    pytest.param(
        lambda x: evoxc.TensorSpec((3,), evoxc.float32),
        "TensorSpec cannot be returned",
        id="tensorspec",
    ),
    pytest.param(
        lambda x: object(),
        "Invalid trace output at pytree path",
        id="unknown-object",
    ),
]


@pytest.mark.parametrize(("bad_output_fn", "match"), BAD_OUTPUT_CASES)
def test_invalid_output_raises(bad_output_fn, match):
    with pytest.raises(evoxc.TraceError, match=match):
        evoxc.trace(bad_output_fn, evoxc.TensorSpec((3,), evoxc.float32))


def test_wrong_arg_count_raises_type_error():
    def f(a, b):
        return evoxc.add(a, b)

    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        evoxc.trace(f, evoxc.TensorSpec((3,), evoxc.float32))


def test_non_callable_raises_type_error():
    with pytest.raises(TypeError, match="not callable"):
        evoxc.trace(42)


# --- 9. output static values re-inserted -----------------------------------------


def test_output_static_values_reinserted(run_graph, as_numpy):
    def f(x):
        return (evoxc.add(x, x), "meta", 3.5)

    g = evoxc.trace(f, evoxc.TensorSpec((2,), evoxc.float32))
    assert [(r.index, r.path, r.value) for r in g.output_static_values] == [
        (1, (1,), "meta"),
        (2, (2,), 3.5),
    ]

    doubled, meta, half = run_graph(g, np.ones(2, np.float32))
    np.testing.assert_allclose(as_numpy(doubled), np.full(2, 2.0, np.float32))
    assert meta == "meta"
    assert half == 3.5

    # unflatten_outputs re-inserts the statics at the recorded positions too
    rebuilt = g.unflatten_outputs([evoxc.tensor(np.full(2, 2.0, np.float32))])
    assert isinstance(rebuilt[0], evoxc.Tensor)
    assert rebuilt[1] == "meta"
    assert rebuilt[2] == 3.5


# --- 10. SymbolicTensor purity ----------------------------------------------------


def test_symbolic_tensor_has_no_concrete_accessors(run_graph, as_numpy):
    def f(x):
        assert not hasattr(x, "numpy")
        assert not hasattr(x, "__dlpack__")
        assert not hasattr(x, "data_ptr")
        return evoxc.multiply(x, 2.0)

    g = evoxc.trace(f, evoxc.TensorSpec((3,), evoxc.float32))
    result = run_graph(g, np.ones(3, np.float32))
    np.testing.assert_allclose(as_numpy(result), np.full(3, 2.0, np.float32))


# --- 11. fresh graphs, no caching --------------------------------------------------


def test_trace_not_cached():
    def f(x):
        return evoxc.negate(x)

    spec = evoxc.TensorSpec((3,), evoxc.float32)
    g1 = evoxc.trace(f, spec)
    g2 = evoxc.trace(f, spec)

    assert g1 is not g2
    assert g1.module is not g2.module
    # Deterministic tracing: structurally identical modules serialize alike.
    assert evoxc.ir.serialize_module(g1.module) == evoxc.ir.serialize_module(g2.module)
