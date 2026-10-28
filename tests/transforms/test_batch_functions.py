"""Tests for `evoxc.batch_functions` — merging N programs onto one function axis.

Contract under test (binding): `evoxc/transforms/CONTEXT.md` § "Batch
functions". `batch_functions(fns, axes=0, out_axes=0, *, switch_ops=None)`
merges a non-empty sequence of N traced `Graph`s with IDENTICAL input/output
signatures but DIFFERENT implementations into ONE `Graph` whose leading axis is
a STATIC "function axis" of length N: row i evaluates function i.

Section A (correctness) asserts, per scenario, that ``run(merged, *inputs)``
row i equals ``run(graphs[i], *rowwise_i)`` — a mapped (`axes` entry ``0``)
input supplies row i of its leading axis, an unmapped (``None``) input is
shared by every function, static leaves are shared. Divergence is still
CORRECT without op switching (constant stacking / output muxing); different
topologies merge via output muxing.

Section B (structural) asserts the cost model and the graph-level guarantees:
identical programs emit ZERO `select`s (exactly one `vectorize` pass), shared
subexpressions appear exactly once, a switch group emits one op per variant
plus a `select` mask cascade, the function axis is a STATIC python `int`
(never a symbolic `Dim`), the merged graph is an ORDINARY registered-op graph
that stages through the explicit pipeline and persists through
`Graph.save`/`load`, and the transform is deterministic and never mutates its
input graphs.

Execution is explicit staging only (`lower → compile → load → run`) — numpy
backend, CPU, small shapes.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pytest

import evoxc
from evoxc import core
from evoxc import ir

F32 = evoxc.float32

# ---------------------------------------------------------------------------
# helpers (module level — no fixtures, no conftest changes)
# ---------------------------------------------------------------------------


def run_graph(graph, *args):
    """Explicit staging pipeline: lower → compile → load → run."""
    return evoxc.run(evoxc.load(evoxc.compile(evoxc.lower(graph))), *args)


def to_np_tree(value):
    """Recursively replace `evoxc.Tensor` leaves with numpy arrays.

    `run_graph` returns a bare `Tensor` for one output, a `tuple` for
    multi-output functions, and reconstructs dict / namedtuple / list output
    trees — all of them are handled here.
    """
    if isinstance(value, core.Tensor):
        return value.numpy()
    if isinstance(value, dict):
        return {key: to_np_tree(item) for key, item in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):  # namedtuple
        return type(value)(*(to_np_tree(item) for item in value))
    if isinstance(value, list):
        return [to_np_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_np_tree(item) for item in value)
    return value


def entry_ops(graph):
    """The merged graph's entry-block ops (the `return` terminator included)."""
    return list(graph.module.main.entry_block.ops)


def all_ops(graph):
    """Every op in every block/region of the graph's module."""
    found = []

    def walk(block):
        for op in block.ops:
            found.append(op)
            for region in op.regions:
                for nested in region.blocks:
                    walk(nested)

    walk(graph.module.main.entry_block)
    return found


def op_names(graph):
    """Entry-block op names, in emission order (terminator included)."""
    return [op.name for op in entry_ops(graph)]


def emitted_count(graph):
    """Ops emitted in the merged entry block (`return` excluded)."""
    return len(entry_ops(graph)) - 1


def merged(fns, **kwargs):
    """`evoxc.batch_functions(...)` plus the caller's `verify()` obligation.

    `batch_functions` deliberately does NOT run `ir.verify` (staging stays
    explicit), so EVERY merged graph in this file goes through this helper and
    is therefore verified.
    """
    graph = evoxc.batch_functions(fns, **kwargs)
    assert isinstance(graph, evoxc.Graph)
    graph.verify()  # raises core.VerificationError on bad IR
    return graph


#: The elementwise op family used to build "same skeleton, different op"
#: programs. All four live in the IR `elementwise` category, so they are
#: interchangeable under `switch_ops` categories.
ELEMENTWISE_OPS = (
    ("add", evoxc.add),
    ("subtract", evoxc.subtract),
    ("multiply", evoxc.multiply),
    ("divide", evoxc.divide),
)


def _with_constant(op_fn, constant):
    """A fresh closure per constant (never a late-binding loop variable)."""

    def fn(x):
        return op_fn(x, constant)

    return fn


def elementwise_programs(n_functions, spec):
    """`n_functions` same-signature programs, each applying ONE different
    elementwise op with its own (strictly positive) constant payload."""
    graphs = []
    for index in range(n_functions):
        _name, op_fn = ELEMENTWISE_OPS[index % len(ELEMENTWISE_OPS)]
        graphs.append(evoxc.trace(_with_constant(op_fn, float(index + 1)), spec))
    return graphs


def row_reference(graphs, row_args):
    """Stack each program's OWN run on its row of the mapped inputs."""
    return np.stack(
        [
            to_np_tree(run_graph(graph, *args))
            for graph, args in zip(graphs, row_args)
        ]
    )


def assert_rowwise(out, reference, rtol=1e-5, atol=1e-6):
    """Row i of a merged output equals the stacked per-program reference."""
    out_np = to_np_tree(out)
    reference = np.asarray(reference, dtype=out_np.dtype)
    assert out_np.shape == reference.shape
    np.testing.assert_allclose(out_np, reference, rtol=rtol, atol=atol)


def assert_tree_rowwise(out, reference, rtol=1e-5, atol=1e-6):
    """`assert_rowwise` over structured (dict / list / tuple) outputs."""
    if isinstance(out, dict):
        assert isinstance(reference, dict)
        assert set(out) == set(reference)
        for key in out:
            assert_tree_rowwise(out[key], reference[key], rtol, atol)
        return
    if isinstance(out, (list, tuple)) and not isinstance(out, core.Tensor):
        assert isinstance(reference, (list, tuple))
        assert len(out) == len(reference)
        for actual, expected in zip(out, reference):
            assert_tree_rowwise(actual, expected, rtol, atol)
        return
    assert_rowwise(out, reference, rtol, atol)


def shape_dims(graph):
    """Every static-shape entry of the merged input specs and value types."""
    dims = []
    for spec in graph.tensor_specs:
        dims.extend(spec.shape)
    for op in all_ops(graph):
        for value in op.results:
            dims.extend(value.type.shape)
    return dims


def output_type(graph):
    """The result type of the FIRST tensor output leaf of the merged graph."""
    return graph.module.main.entry_block.terminator.operands[0].type


#: The four `switch_ops` spellings exercised throughout (factories so no
#: mutable list/dict is shared between parametrized runs).
SWITCH_SPELLINGS = [
    pytest.param(lambda: None, id="switch-none"),
    pytest.param(lambda: True, id="switch-true"),
    pytest.param(lambda: ["elementwise"], id="switch-category"),
    pytest.param(
        lambda: [{"add", "subtract", "multiply", "divide"}], id="switch-names"
    ),
]

#: The spellings that actually ENABLE merging, paired with the `SwitchRecord`
#: group name they produce (a category name, or the sorted ","-join of an
#: explicit op-name set — the WHOLE declared set, not just the used variants).
SWITCHING_SPELLINGS = [
    pytest.param(lambda: True, "elementwise", id="switch-true"),
    pytest.param(lambda: ["elementwise"], "elementwise", id="switch-category"),
    pytest.param(
        lambda: [{"add", "subtract", "multiply", "divide"}],
        "add,divide,multiply,subtract",
        id="switch-names",
    ),
]


class NT(NamedTuple):
    """A namedtuple input pytree."""

    a: object
    b: object


@dataclass
class DC:
    """A dataclass input pytree."""

    a: object
    b: object


# ---------------------------------------------------------------------------
# A. Correctness contract — row i of the merged run == function i's own run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("switch_ops", SWITCH_SPELLINGS)
@pytest.mark.parametrize("n_functions", [2, 3, 5])
def test_row_i_equals_function_i(n_functions, switch_ops):
    """N programs differing in ONE elementwise op, every switch spelling."""
    spec = evoxc.TensorSpec((3,), F32)
    graphs = elementwise_programs(n_functions, spec)
    graph = merged(graphs, switch_ops=switch_ops())

    rng = np.random.default_rng(11)
    xs = (rng.normal(size=(n_functions, 3)) + 5.0).astype(np.float32)

    out = run_graph(graph, xs)
    assert_rowwise(out, row_reference(graphs, [(xs[i],) for i in range(n_functions)]))

    report = graph.batch_report
    assert isinstance(report, evoxc.BatchReport)
    assert report.n_functions == n_functions


def test_public_surface():
    """`batch_functions` / `BatchReport` / `SwitchRecord` are top-level exports."""
    assert evoxc.batch_functions is evoxc.transforms.batch_functions
    assert evoxc.BatchReport is evoxc.transforms.BatchReport
    assert evoxc.SwitchRecord is evoxc.transforms.SwitchRecord

    spec = evoxc.TensorSpec((3,), F32)
    graph = merged(elementwise_programs(2, spec), switch_ops=True)
    report = graph.batch_report
    assert isinstance(report, evoxc.BatchReport)
    assert isinstance(report.switched[0], evoxc.SwitchRecord)
    assert set(report.as_dict()) == {
        "n_functions",
        "value_classes",
        "shared_classes",
        "switched",
        "stacked_constants",
        "output_muxes",
        "emitted_ops",
    }


@pytest.mark.parametrize("shape", [(), (3,), (2, 3)])
def test_rank0_rank1_rank2_mapped_inputs(shape):
    """A map of the leading axis works for rank-0 / rank-1 / rank-2 rows.

    The GRAPH path normalizes `axes` against the traced (per-function) specs,
    so a rank-0 per-function input cannot be axis-mapped (`axes=0` is out of
    range) — its function axis only exists on the BATCHED spec, which is the
    spec-callable path's job (see the callable test). Rank 0 is therefore
    covered here through `axes=None`.
    """
    spec = evoxc.TensorSpec(shape, F32)
    graphs = elementwise_programs(3, spec)
    axis = None if shape == () else 0
    graph = merged(graphs, axes=axis, switch_ops=True)

    rng = np.random.default_rng(12)
    if shape == ():
        shared = np.array(-1.5, dtype=np.float32)
        out = run_graph(graph, shared)
        reference = row_reference(graphs, [(shared,)] * 3)
    else:
        xs = (rng.normal(size=(3,) + shape) + 5.0).astype(np.float32)
        out = run_graph(graph, xs)
        reference = row_reference(graphs, [(xs[i],) for i in range(3)])
    assert_rowwise(out, reference)
    # Per-function shape with a leading function axis of length 3.
    assert output_type(graph).shape == (3,) + shape


def test_rank0_output_via_sum():
    """A rank-0 per-function OUTPUT gets the function axis: (N,)."""
    spec = evoxc.TensorSpec((3,), F32)

    def f_scaled(x):
        return evoxc.sum(evoxc.multiply(x, 2.0))

    def f_shifted(x):
        return evoxc.sum(evoxc.add(x, 1.0))

    graphs = [evoxc.trace(f_scaled, spec), evoxc.trace(f_shifted, spec)]
    graph = merged(graphs, switch_ops=True)

    xs = np.random.default_rng(13).normal(size=(2, 3)).astype(np.float32)
    out = run_graph(graph, xs)
    assert_rowwise(out, row_reference(graphs, [(xs[i],) for i in range(2)]))
    assert output_type(graph).shape == (2,)


class TestInputSignatures:
    """Multi-input, pytree, static-leaf and multi-output signatures."""

    def test_multi_input_mixed_axes(self):
        """`axes=(0, None)`: the first input is rowwise, the second shared."""
        spec = evoxc.TensorSpec((3,), F32)
        weight = evoxc.TensorSpec((3,), F32)

        def f_product(x, w):
            return evoxc.sum(evoxc.multiply(x, w))

        def f_shift(x, w):
            return evoxc.sum(evoxc.add(x, w))

        graphs = [
            evoxc.trace(f_product, spec, weight),
            evoxc.trace(f_shift, spec, weight),
        ]
        graph = merged(graphs, axes=(0, None), switch_ops=True)

        rng = np.random.default_rng(14)
        xs = rng.normal(size=(2, 3)).astype(np.float32)
        w = rng.normal(size=(3,)).astype(np.float32)
        out = run_graph(graph, xs, w)
        assert_rowwise(
            out, row_reference(graphs, [(xs[0], w), (xs[1], w)])
        )
        assert len(graph.tensor_specs) == 2
        assert graph.tensor_specs[0].shape == (2, 3)
        assert graph.tensor_specs[1].shape == (3,)  # shared: no function axis

    def test_multi_input_all_mapped(self):
        """`axes=(0, 0)`: both inputs are rowwise on the function axis."""
        spec = evoxc.TensorSpec((3,), F32)

        def f_add(x, y):
            return evoxc.add(x, y)

        def f_subtract(x, y):
            return evoxc.subtract(x, y)

        graphs = [
            evoxc.trace(f_add, spec, spec),
            evoxc.trace(f_subtract, spec, spec),
        ]
        graph = merged(graphs, axes=(0, 0), switch_ops=True)

        rng = np.random.default_rng(15)
        xs = rng.normal(size=(2, 3)).astype(np.float32)
        ys = rng.normal(size=(2, 3)).astype(np.float32)
        out = run_graph(graph, xs, ys)
        assert_rowwise(
            out, row_reference(graphs, [(xs[0], ys[0]), (xs[1], ys[1])])
        )
        assert [spec.shape for spec in graph.tensor_specs] == [(2, 3), (2, 3)]

    def test_dict_input(self):
        """A dict pytree input: `axes=({"a": 0, "b": 0}, None)`."""
        spec = evoxc.TensorSpec((3,), F32)

        def f_add(d, w):
            return evoxc.add(d["a"], d["b"])

        def f_multiply(d, w):
            return evoxc.multiply(d["a"], d["b"])

        graphs = [
            evoxc.trace(f_add, {"a": spec, "b": spec}, spec),
            evoxc.trace(f_multiply, {"a": spec, "b": spec}, spec),
        ]
        graph = merged(
            graphs, axes=({"a": 0, "b": 0}, None), switch_ops=True
        )

        rng = np.random.default_rng(16)
        a = rng.normal(size=(2, 3)).astype(np.float32)
        b = rng.normal(size=(2, 3)).astype(np.float32)
        w = rng.normal(size=(3,)).astype(np.float32)
        out = run_graph(graph, {"a": a, "b": b}, w)
        assert_rowwise(
            out,
            row_reference(
                graphs,
                [({"a": a[0], "b": b[0]}, w), ({"a": a[1], "b": b[1]}, w)],
            ),
        )
        assert graph.tensor_specs[0].shape == (2, 3)

    @pytest.mark.parametrize("container", [list, tuple], ids=["list", "tuple"])
    def test_sequence_input(self, container):
        """A list / tuple pytree input with per-position axes."""
        spec = evoxc.TensorSpec((3,), F32)

        def f_add(p):
            return evoxc.add(p[0], p[1])

        def f_multiply(p):
            return evoxc.multiply(p[0], p[1])

        graphs = [
            evoxc.trace(f_add, container([spec, spec])),
            evoxc.trace(f_multiply, container([spec, spec])),
        ]
        axes = ([0, 0],) if container is list else ((0, 0),)
        graph = merged(graphs, axes=axes, switch_ops=True)

        rng = np.random.default_rng(17)
        a = rng.normal(size=(2, 3)).astype(np.float32)
        b = rng.normal(size=(2, 3)).astype(np.float32)
        out = run_graph(graph, container([a, b]))
        assert_rowwise(
            out,
            row_reference(
                graphs,
                [
                    (container([a[0], b[0]]),),
                    (container([a[1], b[1]]),),
                ],
            ),
        )

    def test_namedtuple_input(self):
        """A namedtuple input pytree is reconstructed at run time."""
        spec = evoxc.TensorSpec((3,), F32)

        def f_add(p):
            return evoxc.add(p.a, p.b)

        def f_multiply(p):
            return evoxc.multiply(p.a, p.b)

        graphs = [
            evoxc.trace(f_add, NT(spec, spec)),
            evoxc.trace(f_multiply, NT(spec, spec)),
        ]
        graph = merged(graphs, axes=(NT(0, 0),), switch_ops=True)

        rng = np.random.default_rng(18)
        a = rng.normal(size=(2, 3)).astype(np.float32)
        b = rng.normal(size=(2, 3)).astype(np.float32)
        out = run_graph(graph, NT(a, b))
        assert_rowwise(
            out,
            row_reference(
                graphs, [(NT(a[0], b[0]),), (NT(a[1], b[1]),)]
            ),
        )
        assert type(to_np_tree(out)) is np.ndarray

    def test_dataclass_input(self):
        """A dataclass input pytree is reconstructed at run time."""
        spec = evoxc.TensorSpec((3,), F32)

        def f_add(p):
            return evoxc.add(p.a, p.b)

        def f_subtract(p):
            return evoxc.subtract(p.a, p.b)

        graphs = [
            evoxc.trace(f_add, DC(spec, spec)),
            evoxc.trace(f_subtract, DC(spec, spec)),
        ]
        graph = merged(graphs, axes=(DC(0, 0),), switch_ops=True)

        rng = np.random.default_rng(19)
        a = rng.normal(size=(2, 3)).astype(np.float32)
        b = rng.normal(size=(2, 3)).astype(np.float32)
        out = run_graph(graph, DC(a, b))
        assert_rowwise(
            out,
            row_reference(
                graphs, [(DC(a[0], b[0]),), (DC(a[1], b[1]),)]
            ),
        )

    def test_multi_output_tuple(self):
        """Two tensor outputs ⇒ `out_axes=(0, 0)` and a rowwise tuple result."""
        spec = evoxc.TensorSpec((3,), F32)

        def f_shift(x):
            return evoxc.add(x, 1.0), evoxc.multiply(x, 2.0)

        def f_drop(x):
            return evoxc.subtract(x, 1.0), evoxc.multiply(x, 3.0)

        graphs = [evoxc.trace(f_shift, spec), evoxc.trace(f_drop, spec)]
        graph = merged(graphs, out_axes=(0, 0), switch_ops=True)

        rng = np.random.default_rng(20)
        xs = rng.normal(size=(2, 3)).astype(np.float32)
        out = run_graph(graph, xs)
        assert isinstance(out, tuple) and len(out) == 2

        references = [to_np_tree(run_graph(g, xs[i])) for i, g in enumerate(graphs)]
        expected = tuple(
            np.stack([references[i][position] for i in range(2)])
            for position in range(2)
        )
        assert_tree_rowwise(out, expected)
        assert [value.type.shape for value in
                graph.module.main.entry_block.terminator.operands] == [
            (2, 3),
            (2, 3),
        ]

    def test_mixed_static_leaves(self):
        """Static leaves sit beside tensor leaves (and are shared, never mapped)."""
        spec = evoxc.TensorSpec((3,), F32)

        def f_multiply(x, k):
            return evoxc.multiply(x, k)

        def f_add(x, k):
            return evoxc.add(x, k)

        graphs = [evoxc.trace(f_multiply, spec, 2.0), evoxc.trace(f_add, spec, 2.0)]
        graph = merged(graphs, axes=(0, None), switch_ops=True)

        xs = np.random.default_rng(21).normal(size=(2, 3)).astype(np.float32)
        out = run_graph(graph, xs, 2.0)
        assert_rowwise(
            out, row_reference(graphs, [(xs[0], 2.0), (xs[1], 2.0)])
        )
        assert len(graph.tensor_specs) == 1  # the static leaf is not a tensor
        assert graph.tensor_specs[0].shape == (2, 3)


def test_different_topologies_merge_via_output_muxing():
    """`sin(x)` vs `exp(exp(x))` share no ops — the output is muxed."""
    spec = evoxc.TensorSpec((), F32)

    def f_sin(x):
        return evoxc.sin(x)

    def f_double_exp(x):
        return evoxc.exp(evoxc.exp(x))

    graphs = [evoxc.trace(f_sin, spec), evoxc.trace(f_double_exp, spec)]
    graph = merged(graphs, axes=None)

    x = np.array(0.5, dtype=np.float32)
    out = run_graph(graph, x)
    assert_rowwise(out, row_reference(graphs, [(x,), (x,)]))

    report = graph.batch_report
    assert report.output_muxes == 1
    assert report.switched == ()  # the skeletons differ: nothing to switch
    assert report.shared_classes == 0
    assert op_names(graph).count("select") == 1


def test_per_program_constants_are_stacked():
    """Differing constant payloads ⇒ ONE stacked constant, no select for it."""
    spec = evoxc.TensorSpec((), F32)

    def f_one(x):
        return evoxc.add(x, 1.0)

    def f_two(x):
        return evoxc.add(x, 2.0)

    graphs = [evoxc.trace(f_one, spec), evoxc.trace(f_two, spec)]
    graph = merged(graphs, axes=None)

    x = np.array(0.25, dtype=np.float32)
    out = run_graph(graph, x)
    assert_rowwise(out, row_reference(graphs, [(x,), (x,)]))

    report = graph.batch_report
    assert report.stacked_constants == 1
    names = op_names(graph)
    assert names.count("constant") == 1  # the stacked (N,)-shaped constant
    assert "select" not in names  # the constant itself is not muxed


def test_identical_constants_share_one_constant():
    """Identical payloads ⇒ ONE shared constant and no stacking."""
    spec = evoxc.TensorSpec((), F32)

    def f_one(x):
        return evoxc.add(x, 1.0)

    graphs = [evoxc.trace(f_one, spec), evoxc.trace(f_one, spec)]
    graph = merged(graphs, axes=None)

    x = np.array(0.25, dtype=np.float32)
    assert_rowwise(
        run_graph(graph, x), row_reference(graphs, [(x,), (x,)])
    )

    report = graph.batch_report
    assert report.stacked_constants == 0
    names = op_names(graph)
    assert names.count("constant") == 1
    assert "select" not in names


UNARY_FAMILY = (
    ("sin", evoxc.sin),
    ("cos", evoxc.cos),
    ("exp", evoxc.exp),
    ("tanh", evoxc.tanh),
)
BINARY_FAMILY = (
    ("add", evoxc.add),
    ("subtract", evoxc.subtract),
    ("multiply", evoxc.multiply),
    ("divide", evoxc.divide),
)


def _random_program(rng):
    """A GP-style random program: a chain of unary / constant-binary steps."""
    steps = []
    for _ in range(int(rng.integers(1, 4))):  # depth 1..3
        if rng.random() < 0.5:
            index = int(rng.integers(len(UNARY_FAMILY)))
            steps.append(("unary", UNARY_FAMILY[index]))
        else:
            index = int(rng.integers(len(BINARY_FAMILY)))
            constant = float(np.round(rng.normal(), 2))
            steps.append(("binary", BINARY_FAMILY[index], constant))
    return steps


def _build_program(steps):
    """Turn a step list into a callable over ONE rank-1 input."""

    def fn(x):
        value = x
        for step in steps:
            if step[0] == "unary":
                value = step[1][1](value)
            else:
                value = step[1][1](value, step[2])
        return value

    return fn


@pytest.mark.parametrize("switch_ops", [None, True], ids=["plain", "switch"])
def test_random_program_population(switch_ops):
    """A 16-program random population (add/sub/mul/div/sin/cos/exp/tanh)."""
    rng = np.random.default_rng(7)
    spec = evoxc.TensorSpec((6,), F32)
    programs = [_random_program(rng) for _ in range(16)]
    graphs = [evoxc.trace(_build_program(steps), spec) for steps in programs]

    graph = merged(graphs, switch_ops=switch_ops)
    xs = rng.uniform(-0.5, 0.5, size=(16, 6)).astype(np.float32)

    out = run_graph(graph, xs)
    reference = row_reference(graphs, [(xs[i],) for i in range(16)])
    assert out.numpy().shape == (16, 6)
    assert np.isfinite(reference).all()
    # Same float32 ops in the same order — the merge must not perturb a bit.
    np.testing.assert_array_equal(out.numpy(), reference)

    report = graph.batch_report
    assert report.n_functions == 16
    if switch_ops is None:
        assert report.switched == ()
    else:
        assert report.switched  # at least one merged elementwise position


# ---------------------------------------------------------------------------
# B. Structural properties
# ---------------------------------------------------------------------------


class TestStructuralProperties:
    def test_identical_programs_emit_no_select(self):
        """`[g, g]` costs exactly one vectorize pass: no selects at all."""
        spec = evoxc.TensorSpec((3,), F32)

        def squaring(x):
            return evoxc.multiply(x, x)

        graph0 = evoxc.trace(squaring, spec)
        graph = merged([graph0, graph0])

        names = op_names(graph)
        assert "select" not in names
        # Each original op appears exactly once (one batched op, not two).
        assert names == op_names(graph0)
        assert emitted_count(graph) == emitted_count(graph0)
        assert graph.batch_report.stacked_constants == 0
        assert graph.batch_report.output_muxes == 0

        xs = np.random.default_rng(30).normal(size=(2, 3)).astype(np.float32)
        out = run_graph(graph, xs)
        assert_rowwise(out, row_reference([graph0, graph0], [(xs[0],), (xs[1],)]))

        # ... and equals `vmap(graph0, 0)` on the same stacked input.
        vmapped = run_graph(evoxc.vmap(graph0, 0), xs)
        np.testing.assert_array_equal(out.numpy(), to_np_tree(vmapped))

    def test_identical_programs_with_a_constant_share_one_reshape(self):
        """`x + 1.0` twice: constant + add shared, plus vectorize's reshape."""
        spec = evoxc.TensorSpec((3,), F32)
        graph0 = evoxc.trace(lambda x: x + 1.0, spec)
        graph = merged([graph0, graph0])

        names = op_names(graph)
        assert names == ["constant", "reshape", "add", "return"]
        assert "select" not in names
        assert emitted_count(graph) == 3
        assert graph.batch_report.value_classes == 2
        assert graph.batch_report.shared_classes == 2

        xs = np.random.default_rng(31).normal(size=(2, 3)).astype(np.float32)
        assert_rowwise(
            run_graph(graph, xs),
            row_reference([graph0, graph0], [(xs[0],), (xs[1],)]),
        )

    def test_shared_subexpression_appears_once(self):
        """`sin(x)` feeding `+1` vs `*2`: ONE `sin` op, one shared class."""
        spec = evoxc.TensorSpec((3,), F32)

        def f_shift(x):
            return evoxc.sin(x) + 1.0

        def f_scale(x):
            return evoxc.sin(x) * 2.0

        graphs = [evoxc.trace(f_shift, spec), evoxc.trace(f_scale, spec)]
        graph = merged(graphs)

        assert op_names(graph).count("sin") == 1
        assert graph.batch_report.shared_classes >= 1

        xs = np.random.default_rng(32).normal(size=(2, 3)).astype(np.float32)
        assert_rowwise(
            run_graph(graph, xs),
            row_reference(graphs, [(xs[0],), (xs[1],)]),
        )

    @pytest.mark.parametrize("switch_ops, expected_group", SWITCHING_SPELLINGS)
    def test_switch_group_emits_one_op_per_variant(self, switch_ops, expected_group):
        """add/subtract/multiply switch: one op per variant + select cascade."""
        spec = evoxc.TensorSpec((3,), F32)

        def f_add(x):
            return evoxc.add(x, 1.0)

        def f_subtract(x):
            return evoxc.subtract(x, 2.0)

        def f_multiply(x):
            return evoxc.multiply(x, 3.0)

        graphs = [
            evoxc.trace(f_add, spec),
            evoxc.trace(f_subtract, spec),
            evoxc.trace(f_multiply, spec),
        ]
        graph = merged(graphs, switch_ops=switch_ops())

        report = graph.batch_report
        assert isinstance(report.switched, tuple) and len(report.switched) == 1
        record = report.switched[0]
        assert isinstance(record, evoxc.SwitchRecord)
        assert record.variants == ("add", "subtract", "multiply")
        assert record.rows_per_variant == ((0,), (1,), (2,))
        assert record.group == expected_group

        names = op_names(graph)
        for variant in record.variants:
            assert names.count(variant) == 1
        # One select per extra variant (variant 0 is the accumulator seed).
        assert names.count("select") == len(record.variants) - 1
        assert report.output_muxes == 0

        xs = np.random.default_rng(33).normal(size=(3, 3)).astype(np.float32)
        assert_rowwise(
            run_graph(graph, xs),
            row_reference(graphs, [(xs[i],) for i in range(3)]),
        )

    def test_every_merged_graph_verifies(self):
        """`graph.verify()` passes on every merged graph of this file's shapes."""
        spec = evoxc.TensorSpec((3,), F32)

        def f_sin(x):
            return evoxc.sin(x)

        def f_cos(x):
            return evoxc.cos(x)

        def f_exp_exp(x):
            return evoxc.exp(evoxc.exp(x))

        cases = [
            (elementwise_programs(3, spec), {}),
            ([evoxc.trace(f_sin, spec), evoxc.trace(f_cos, spec)], {}),
            ([evoxc.trace(f_sin, spec), evoxc.trace(f_exp_exp, spec)], {}),
            (elementwise_programs(3, spec), {"switch_ops": True}),
            (elementwise_programs(3, spec), {"axes": None, "switch_ops": True}),
        ]
        for graphs, kwargs in cases:
            merged(graphs, **kwargs).verify()

    @pytest.mark.parametrize("axes", [0, None], ids=["mapped", "shared"])
    @pytest.mark.parametrize("shape", [(3,), (2, 3)])
    def test_function_axis_is_a_static_int(self, axes, shape):
        """The function axis is a static `int` N; NO symbolic dims anywhere."""
        n_functions = 4
        spec = evoxc.TensorSpec(shape, F32)
        graphs = elementwise_programs(n_functions, spec)
        graph = merged(graphs, axes=axes, switch_ops=True)

        expected_input = (n_functions,) + shape if axes == 0 else shape
        assert tuple(graph.tensor_specs[0].shape) == expected_input
        if axes == 0:
            leading = graph.tensor_specs[0].shape[0]
            assert leading == n_functions
            assert type(leading) is int  # a python int, never a core.Dim

        merged_output = output_type(graph)
        assert merged_output.shape[0] == n_functions
        assert type(merged_output.shape[0]) is int
        assert merged_output.shape == (n_functions,) + shape

        dims = shape_dims(graph)
        assert dims  # anti-vacuity
        assert all(isinstance(dim, int) or dim is None for dim in dims), (
            f"symbolic dims leaked into the merged graph: {dims}"
        )
        for dim in dims:
            assert not isinstance(dim, (core.Dim, core.DimExpr))

    @pytest.mark.parametrize("n_variants", [3, 4])
    @pytest.mark.parametrize(
        "shape", [(), (3,), (2, 3)], ids=["rank0", "rank1", "rank2"]
    )
    def test_shared_input_switch_broadcasts_along_the_function_axis(
        self, n_variants, shape
    ):
        """`axes=None` + `switch_ops=True`: shared candidates get the axis."""
        spec = evoxc.TensorSpec(shape, F32)
        graphs = elementwise_programs(n_variants, spec)
        graph = merged(graphs, axes=None, switch_ops=True)

        if shape == ():
            shared = np.array(0.75, dtype=np.float32)
        else:
            rng = np.random.default_rng(34)
            shared = (rng.normal(size=shape) + 4.0).astype(np.float32)
        out = run_graph(graph, shared)
        assert_rowwise(out, row_reference(graphs, [(shared,)] * n_variants))
        assert to_np_tree(out).shape == (n_variants,) + shape
        assert output_type(graph).shape == (n_variants,) + shape

        record = graph.batch_report.switched[0]
        assert len(record.variants) == n_variants
        assert record.rows_per_variant == tuple((i,) for i in range(n_variants))

    def test_rows_sharing_a_variant_are_grouped(self):
        """Two programs picking the SAME variant land in one `rows_per_variant`."""
        spec = evoxc.TensorSpec((), F32)

        def f_add_one(x):
            return evoxc.add(x, 1.0)

        def f_add_two(x):
            return evoxc.add(x, 2.0)

        def f_multiply(x):
            return evoxc.multiply(x, 3.0)

        graphs = [
            evoxc.trace(f_add_one, spec),
            evoxc.trace(f_add_two, spec),
            evoxc.trace(f_multiply, spec),
        ]
        graph = merged(graphs, axes=None, switch_ops=True)

        record = graph.batch_report.switched[0]
        assert record.variants == ("add", "multiply")
        assert record.rows_per_variant == ((0, 1), (2,))

        x = np.array(1.5, dtype=np.float32)
        assert_rowwise(run_graph(graph, x), row_reference(graphs, [(x,)] * 3))

    def test_switched_value_consumed_downstream_stays_mapped(self):
        """`y = switched(x); return y * 1` — the muxed value is treated as mapped.

        Every constant payload is `1.0`, so the ONLY diverging position is the
        switched elementwise op; the outer `multiply` is a SHARED class and
        consumes the already-muxed value (that is why `output_muxes` is 0).
        """
        spec = evoxc.TensorSpec((), F32)

        def f_add(x):
            return evoxc.multiply(evoxc.add(x, 1.0), 1.0)

        def f_multiply(x):
            return evoxc.multiply(evoxc.multiply(x, 1.0), 1.0)

        graphs = [evoxc.trace(f_add, spec), evoxc.trace(f_multiply, spec)]
        graph = merged(graphs, axes=None, switch_ops=True)

        report = graph.batch_report
        assert report.switched == (
            evoxc.SwitchRecord(
                group="elementwise",
                variants=("add", "multiply"),
                rows_per_variant=((0,), (1,)),
            ),
        )
        assert report.value_classes == 3
        assert report.shared_classes == 3
        # The downstream op is shared and consumes the ALREADY mapped mux.
        assert report.output_muxes == 0

        x = np.array(2.0, dtype=np.float32)
        assert_rowwise(run_graph(graph, x), row_reference(graphs, [(x,), (x,)]))

    def test_two_merges_of_the_same_graphs_are_identical(self):
        """Deterministic: the same inputs give the same serialized module."""
        spec = evoxc.TensorSpec((3,), F32)
        graphs = elementwise_programs(3, spec)

        first, second = [merged(graphs, switch_ops=True) for _ in range(2)]
        assert ir.serialize_module(first.module) == ir.serialize_module(
            second.module
        )

    def test_input_graphs_are_not_mutated(self):
        """The transform builds a NEW graph; the inputs are untouched."""
        spec = evoxc.TensorSpec((3,), F32)
        graphs = elementwise_programs(3, spec)
        before = [ir.serialize_module(graph.module) for graph in graphs]

        merged(graphs, switch_ops=True)

        after = [ir.serialize_module(graph.module) for graph in graphs]
        assert before == after

    def test_merged_graph_contains_only_registered_ops(self):
        """Every op — entry block and regions — resolves to an opdef."""
        spec = evoxc.TensorSpec((3,), F32)
        graphs = elementwise_programs(3, spec)
        graph = merged(graphs, switch_ops=True)

        registered = {definition.name for definition in ir.all_opdefs()}
        ops = all_ops(graph)
        assert ops  # anti-vacuity
        for op in ops:
            assert op.opdef is not None, f"unregistered op {op.name!r} in the IR"
            assert op.name in registered

    def test_merged_graph_stages_and_persists(self, tmp_path):
        """Explicit staging, save/load round-trip, and NO persisted metadata."""
        spec = evoxc.TensorSpec((3,), F32)
        graphs = elementwise_programs(3, spec)
        graph = merged(graphs, switch_ops=True)

        xs = np.random.default_rng(35).normal(size=(3, 3)).astype(np.float32)
        expected = row_reference(graphs, [(xs[i],) for i in range(3)])

        # Explicit staging — the documented composition `evoxc.build` performs.
        lowered = evoxc.lower(graph)
        compiled = evoxc.compile(lowered)
        executable = evoxc.load(compiled)
        assert_rowwise(evoxc.run(executable, xs), expected)

        # `evoxc.build`/`evoxc.evaluate` are CALLABLE shorthands (they trace);
        # a Graph stages through the explicit pipeline above, and they say so.
        with pytest.raises(TypeError, match="explicit lower/compile/load pipeline"):
            evoxc.build(graph)
        with pytest.raises(TypeError, match="explicit lower/compile/load pipeline"):
            evoxc.evaluate(graph, xs)

        # Persistence: the merge report is transform metadata, not graph data.
        path = tmp_path / "merged.evoxcgraph"
        graph.save(str(path))
        loaded = evoxc.Graph.load(str(path))
        loaded.verify()
        # `batch_report` is a plain instance attribute and is NEVER persisted
        # (it describes how the graph was built, not what it computes).
        assert hasattr(graph, "batch_report")
        assert not hasattr(loaded, "batch_report")
        assert ir.serialize_module(loaded.module) == ir.serialize_module(graph.module)
        assert_rowwise(run_graph(loaded, xs), expected)

    def test_spec_callable_path(self):
        """Callable fns → `TransformCallable(kind="batch_functions")`."""
        calls = []

        def f_add(x):
            calls.append("add")
            return evoxc.add(x, 1.0)

        def f_multiply(x):
            calls.append("multiply")
            return evoxc.multiply(x, 2.0)

        transform = evoxc.batch_functions([f_add, f_multiply])
        assert transform.kind == "batch_functions"

        batched = evoxc.TensorSpec((2, 3), F32)
        # Same source line: the merge's own mask/select ops carry the
        # `batch_functions` CALL-SITE location, so a serialized comparison is
        # only location-stable when both calls happen on the same line.
        first, second = [transform(batched) for _ in range(2)]

        # Traced exactly ONCE per function per call, and a FRESH graph each call.
        assert calls == ["add", "multiply", "add", "multiply"]
        assert first is not second
        assert first.module is not second.module
        assert ir.serialize_module(first.module) == ir.serialize_module(
            second.module
        )

        assert first.tensor_specs[0].shape == (2, 3)
        assert output_type(first).shape == (2, 3)

        xs = np.random.default_rng(36).normal(size=(2, 3)).astype(np.float32)
        reference = row_reference(
            [evoxc.trace(f_add, evoxc.TensorSpec((3,), F32)),
             evoxc.trace(f_multiply, evoxc.TensorSpec((3,), F32))],
            [(xs[0],), (xs[1],)],
        )
        assert_rowwise(run_graph(first, xs), reference)

        # Concrete tensors are not accepted — transforms never execute.
        with pytest.raises(core.TraceError):
            transform(evoxc.zeros((2, 3), dtype=F32))
