"""Native (Rust CPU engine) FEATURE INTEGRATIONS — the frontend features that
must keep working when a graph runs on ``backend="native"``.

Contract: ``../../evoxc/backends/native/CONTEXT.md`` (capability matrix:
``dynamic_shapes`` / ``collectives`` / ``runtime_calls`` / ``external_calls`` /
``custom_blocks`` all ``True``, ``sparse_ops`` ``False``) plus the frontend
contracts it leans on — ``evoxc.cond`` / ``while_loop`` / ``scan``, pytree I/O
and static specialization, symbolic dims, ``grad`` / ``vmap``, external
kernels, custom ``native`` block impls (portables inlined at ``lower()``),
single-rank collectives and the sparse deferral.

Oracle: the numpy interpreter is the numerical reference for the SAME graph
(``nu.evaluate_pair`` asserts per-dtype parity with exact dtype/shape
agreement); where an independent formula is unambiguous the numbers are ALSO
pinned against an explicit ``np.*`` or hand-written loop reference.

Staging: one executable is reused where the point is staging
(``nu.build_native`` + ``nu.run_native``); transformed graphs are ``Graph``
objects that ``evoxc.evaluate`` will not stage, so ``_stage`` runs the explicit
``lower`` → ``compile`` → ``load`` pipeline per backend.

Deliberately NOT touched (pinned elsewhere): ``nan_to_num`` with default
attributes and ``random.split``/``random_n``. Conventions: fixed seeds, tiny
shapes, CPU only, no network, no GPU.
"""
import dataclasses
import statistics
import time
from typing import NamedTuple

import numpy as np
import pytest

import evoxc
from evoxc import core
from tests.backends import _native_utils as nu

pytestmark = pytest.mark.skipif(not nu.ENGINE_AVAILABLE, reason=nu.SKIP_REASON)


# ---------------------------------------------------------------------------
# Deterministic inputs and small shared helpers
# ---------------------------------------------------------------------------


def _lin(shape, lo=-1.5, hi=1.5, dtype=np.float32):
    """Deterministic linearly-spaced array over ``[lo, hi]`` (no RNG)."""
    count = int(np.prod(shape))
    return np.linspace(lo, hi, count).reshape(shape).astype(dtype)


def _normal(shape, seed, dtype=np.float32):
    """Fixed-seed normal values (through float64, then cast)."""
    return np.random.default_rng(seed).normal(size=shape).astype(dtype)


def _signed_pair(shape):
    """Two deterministic arrays with clearly positive / negative reductions."""
    magnitude = np.abs(_lin(shape, -1.5, 1.5)) + 0.25
    return magnitude.astype(np.float32), (-magnitude).astype(np.float32)


def _stage(graph, backend):
    """Explicit ``lower`` → ``compile`` → ``load`` for an already-traced graph.

    Transformed graphs (``grad``/``vmap``) are ``Graph`` objects, which
    ``evoxc.evaluate`` refuses to stage — the pipeline must be explicit.
    """
    lowered = evoxc.lower(graph, backend=backend)
    artifact = evoxc.compile(lowered, backend=backend)
    return evoxc.load(artifact, backend=backend)


def _run_pair(graph, *args):
    """Run one traced graph on both backends, assert parity, return native."""
    got = nu.as_numpy(evoxc.run(_stage(graph, "native"), *args))
    expected = nu.as_numpy(evoxc.run(_stage(graph, "numpy"), *args))
    nu.assert_tree_parity(got, expected, label="graph")
    return got


def _build_and_run_pair(fn, specs, args):
    """Stage ``fn`` for BOTH backends and compare — the explicit-staging form.

    ``evoxc.evaluate`` derives specs from concrete tensors only, so a graph
    with a static (Python) argument needs ``evoxc.build`` from explicit specs.
    """
    got = nu.as_numpy(evoxc.run(nu.build_native(fn, *specs), *args))
    expected = nu.as_numpy(
        evoxc.run(evoxc.build(fn, *specs, backend="numpy"), *args)
    )
    nu.assert_tree_parity(got, expected, label="graph")
    return got


# ---------------------------------------------------------------------------
# 1. cond — a graph-input predicate, both polarities on ONE executable
# ---------------------------------------------------------------------------


@evoxc.defn
def _cond_double_or_shift(x):
    return evoxc.cond(
        evoxc.sum(x) > 0.0,
        lambda v: v * 2.0,
        lambda v: v + 1.0,
        x,
    )


@evoxc.defn
def _cond_reduction_branches(x):
    """Both branches reduce — the cond result is a SCALAR."""
    return evoxc.cond(
        evoxc.sum(x) > 0.0,
        lambda v: evoxc.sum(v * v),
        lambda v: evoxc.sum(v) - 100.0,
        x,
    )


@evoxc.defn
def _cond_dict_operands(inp):
    """A dict of two tensors passed positionally as the operand pytree."""
    a, b = inp["a"], inp["b"]
    return evoxc.cond(
        evoxc.sum(a) > 0.0,
        lambda p: p["a"] * 2.0 + p["b"],
        lambda p: p["a"] - p["b"],
        {"a": a, "b": b},
    )


class TestCond:
    def test_both_polarities_on_one_executable(self):
        positive, negative = _signed_pair((3, 4))
        executable = nu.build_native(_cond_double_or_shift, *nu.specs_for(positive))

        taken = nu.run_native(executable, positive)
        nu.assert_tree_parity(
            taken,
            nu.evaluate_numpy(_cond_double_or_shift, positive),
            label="cond true branch",
        )
        np.testing.assert_allclose(taken, positive * 2.0, rtol=1e-5, atol=1e-6)

        taken = nu.run_native(executable, negative)
        nu.assert_tree_parity(
            taken,
            nu.evaluate_numpy(_cond_double_or_shift, negative),
            label="cond false branch",
        )
        np.testing.assert_allclose(taken, negative + 1.0, rtol=1e-5, atol=1e-6)

    def test_cond_nesting_a_reduction(self):
        positive, negative = _signed_pair((2, 3))
        for values, expected in (
            (positive, np.sum(positive * positive)),
            (negative, np.sum(negative) - 100.0),
        ):
            got = nu.evaluate_pair(_cond_reduction_branches, values)
            assert np.shape(got) == ()  # a scalar, not a vector
            np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    def test_cond_pytree_operands(self):
        whole = _lin((4,), -1.0, 1.0)
        ones = np.ones((4,), np.float32)
        nu.evaluate_pair(_cond_dict_operands, {"a": whole, "b": ones})
        nu.evaluate_pair(_cond_dict_operands, {"a": -whole, "b": ones})

    def test_python_bool_predicate_is_a_trace_error(self):
        """Control flow over a Python value is a *Python* value decision —
        ``if``/``while``, never a graph ``cond`` op."""

        def bad(x):
            return evoxc.cond(True, lambda v: v * 2.0, lambda v: v, x)

        with pytest.raises(core.TraceError, match="pred must be a core.SymbolicTensor"):
            evoxc.trace(bad, core.TensorSpec((3,), np.float32))


# ---------------------------------------------------------------------------
# 2. while_loop — single carry, tuple carry with a float-reduction condition,
#    zero iterations, and a dict pytree carry
# ---------------------------------------------------------------------------


@evoxc.defn
def _while_single_carry(x):
    return evoxc.while_loop(lambda c: evoxc.sum(c) > 0.0, lambda c: c - 1.0, x)


@evoxc.defn
def _while_three_carries(x, y):
    def cond(carried):
        return evoxc.max(carried[1]) > 0.5  # reads a reduction of the FLOAT carry

    def body(carried):
        return (carried[0] + 1.0, carried[1] * 0.5, carried[2] - 1.0)

    return evoxc.while_loop(cond, body, (x, y, evoxc.sum(x)))


@evoxc.defn
def _while_never_entered(x):
    return evoxc.while_loop(lambda c: evoxc.sum(c) < 0.0, lambda c: c - 1.0, x)


@evoxc.defn
def _while_dict_carry(x, y):
    def cond(carried):
        return evoxc.sum(carried["x"]) > 2.0

    def body(carried):
        return {"x": carried["x"] * 0.5, "y": carried["y"] + 1.0}

    return evoxc.while_loop(cond, body, {"x": x, "y": y})


class TestWhileLoop:
    def test_single_carry(self):
        x = _lin((4,), 0.5, 4.5)  # sum ~ 10 → ten exact iterations
        got = nu.evaluate_pair(_while_single_carry, x)

        reference = x.copy()
        while np.sum(reference) > 0.0:
            reference = reference - 1.0
        np.testing.assert_allclose(got, reference, rtol=0, atol=0)

    def test_three_carries_with_a_float_reduction_condition(self):
        x = _lin((3,), -1.0, 1.0)
        y = np.array([3.0, 1.0, 0.25], np.float32)  # 3 → 1.5 → 0.75 → 0.375
        got = nu.evaluate_pair(_while_three_carries, x, y)

        ref_x, ref_y = x.copy(), y.copy()
        ref_z = np.asarray(np.sum(x), np.float32)  # 0-d, like the graph scalar
        while np.max(ref_y) > 0.5:  # halving is exact in binary
            ref_x = ref_x + 1.0
            ref_y = ref_y * 0.5
            ref_z = np.asarray(ref_z - 1.0)  # keep a 0-d ARRAY leaf
        nu.assert_tree_parity(got, (ref_x, ref_y, ref_z), label="three carries")

    def test_zero_iterations_returns_the_carry_unchanged(self):
        x = _lin((3, 3), 0.25, 4.0)  # sum > 0, so the body never runs
        got = nu.evaluate_pair(_while_never_entered, x)
        np.testing.assert_array_equal(got, x)

    def test_dict_pytree_carry(self):
        x = _lin((4,), 2.0, 10.0)  # sum = 24 → 8 → 4 → 2 (stop)
        y = np.zeros((4,), np.float32)
        got = nu.evaluate_pair(_while_dict_carry, x, y)

        ref_x, ref_y = x.copy(), y.copy()
        while np.sum(ref_x) > 2.0:
            ref_x, ref_y = ref_x * 0.5, ref_y + 1.0
        nu.assert_tree_parity(
            got, {"x": ref_x, "y": ref_y}, label="dict carry"
        )


# ---------------------------------------------------------------------------
# 3. scan — a cumsum scan returning both leaves, and a static-length prefix
# ---------------------------------------------------------------------------


@evoxc.defn
def _scan_cumsum(init, xs):
    return evoxc.scan(lambda c, v: (c + v, v * 2.0), init, xs)


@evoxc.defn
def _scan_final_carry(init, xs):
    return evoxc.scan(lambda c, v: (c + v, c), init, xs)[0]


@evoxc.defn
def _scan_prefix(init, xs):
    return evoxc.scan(lambda c, v: (c + v, v), init, xs, length=2)


class TestScan:
    def test_cumulative_sum_returns_both_leaves(self):
        xs = _lin((4, 3), -1.0, 1.0)
        init = np.zeros((3,), np.float32)
        final, ys = nu.evaluate_pair(_scan_cumsum, init, xs)

        nu.assert_tree_parity(ys, xs * 2.0, label="scan outputs")
        np.testing.assert_allclose(final, xs.sum(axis=0), rtol=1e-5, atol=1e-5)

    def test_final_carry_matches_the_sum(self):
        xs = _lin((5, 2), -2.0, 2.0)
        init = np.full((2,), 0.5, np.float32)
        got = nu.evaluate_pair(_scan_final_carry, init, xs)
        np.testing.assert_allclose(got, init + xs.sum(axis=0), rtol=1e-5, atol=1e-5)

    def test_static_length_prefix(self):
        xs = _lin((4, 3), -1.0, 1.0)
        init = np.zeros((3,), np.float32)
        final, ys = nu.evaluate_pair(_scan_prefix, init, xs)

        assert ys.shape == (2, 3)  # only the first two steps ran
        nu.assert_tree_parity(ys, xs[:2], label="prefix outputs")
        np.testing.assert_allclose(final, xs[:2].sum(axis=0), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. Pytree inputs AND outputs, plus static specialization
# ---------------------------------------------------------------------------


class _Pair(NamedTuple):
    y: object
    z: object


@evoxc.defn
def _tree_program(inp, scale: int):
    """Nested input ``{"a", "w", "s": (t,)}`` → nested output with a
    namedtuple leaf, specialized by a STATIC ``int`` argument."""
    reduced = evoxc.sum(inp["a"] * inp["w"], axes=1) + inp["s"][0]
    return {
        "y": reduced,
        "aux": (reduced * float(scale), _Pair(reduced - 1.0, {"k": reduced + 2.0})),
    }


@dataclasses.dataclass
class _Config:
    """A fully-static config object (every field is a static leaf)."""

    scale: float
    bias: float


@evoxc.defn
def _config_program(x, cfg: _Config):
    return x * cfg.scale + cfg.bias


def _tree_specs(scale=3):
    return (
        {
            "a": core.TensorSpec((2, 3), np.float32),
            "w": core.TensorSpec((3,), np.float32),
            "s": (core.TensorSpec((2,), np.float32),),
        },
        scale,
    )


def _tree_args():
    return {
        "a": _normal((2, 3), seed=11),
        "w": _normal((3,), seed=12),
        "s": (_normal((2,), seed=13),),
    }


class TestPytrees:
    def test_nested_input_and_output_structure(self):
        args = _tree_args()
        executable = nu.build_native(_tree_program, *_tree_specs())

        raw = evoxc.run(executable, args, 3)  # NOT as_numpy: check the types
        assert isinstance(raw, dict)
        assert isinstance(raw["aux"], tuple) and len(raw["aux"]) == 2
        assert isinstance(raw["aux"][1], _Pair)
        assert isinstance(raw["aux"][1].z, dict) and set(raw["aux"][1].z) == {"k"}

        numpy_executable = evoxc.build(_tree_program, *_tree_specs(), backend="numpy")
        raw_numpy = evoxc.run(numpy_executable, args, 3)
        assert type(raw) is type(raw_numpy)
        assert type(raw["aux"]) is type(raw_numpy["aux"])
        assert type(raw["aux"][1]) is type(raw_numpy["aux"][1])

        nu.assert_tree_parity(nu.as_numpy(raw), nu.as_numpy(raw_numpy))

        # Independent formulas for the leaf values.
        reference = (args["a"] * args["w"]).sum(axis=1) + args["s"][0]
        np.testing.assert_allclose(raw["y"].numpy(), reference, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            raw["aux"][0].numpy(), reference * 3.0, rtol=1e-5, atol=1e-6
        )

    def test_static_leaf_specializes_the_graph(self):
        args = _tree_args()
        executable = nu.build_native(_tree_program, *_tree_specs())

        base = nu.run_native(executable, args, 3)
        nu.assert_tree_parity(
            base,
            _build_and_run_pair(_tree_program, _tree_specs(), (args, 3)),
            label="static 3",
        )
        # A different static value is a DIFFERENT graph — never a silent
        # re-specialization of the already-built executable.
        with pytest.raises(core.TraceError, match="graph was specialized on 3"):
            evoxc.run(executable, args, 4)

        # A second static value stages a second executable; only the
        # static-derived branch changes, the tensor math is identical.
        scaled = _build_and_run_pair(_tree_program, _tree_specs(4), (args, 4))
        nu.assert_tree_parity(scaled["y"], base["y"], label="static y")
        np.testing.assert_allclose(
            scaled["aux"][0], scaled["y"] * 4.0, rtol=1e-5, atol=1e-6
        )

    def test_static_dataclass_config(self):
        x = _lin((5,), -1.0, 1.0)
        cfg = _Config(scale=2.0, bias=0.5)
        specs = (core.TensorSpec((5,), np.float32), cfg)

        got = nu.as_numpy(evoxc.run(nu.build_native(_config_program, *specs), x, cfg))
        reference = nu.as_numpy(
            evoxc.run(evoxc.build(_config_program, *specs, backend="numpy"), x, cfg)
        )
        nu.assert_tree_parity(got, reference, label="dataclass config")
        np.testing.assert_allclose(got, x * 2.0 + 0.5, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# 5. Dynamic (symbolic) dims — ONE executable, several run-time sizes
# ---------------------------------------------------------------------------


_DIM_B = core.Dim("B")


@evoxc.defn
def _symbolic_matmul(x, w):
    return evoxc.tanh(evoxc.dot(x, w))


@evoxc.defn
def _symbolic_concat(x):
    return evoxc.concatenate((x, x * 2.0), axis=1)


class TestDynamicDims:
    def test_one_executable_two_sizes(self):
        w = _lin((3, 2), -1.0, 1.0)
        executable = nu.build_native(
            _symbolic_matmul,
            core.TensorSpec((_DIM_B, 3), np.float32),
            core.TensorSpec((3, 2), np.float32),
        )
        # The EXECUTABLE's own spec keeps the symbolic dim — one compile.
        assert core.Dim("B") in executable.signature.input_specs[0].shape

        for batch in (2, 5):
            x = _lin((batch, 3), -1.0, 1.0)
            got = nu.run_native(executable, x, w)
            assert got.shape == (batch, 2)
            nu.assert_tree_parity(
                got,
                nu.evaluate_numpy(_symbolic_matmul, x, w),
                label=f"B={batch}",
            )
            np.testing.assert_allclose(got, np.tanh(x @ w), rtol=1e-5, atol=1e-6)

    def test_symbolic_output_dim(self):
        executable = nu.build_native(
            _symbolic_concat, core.TensorSpec((_DIM_B, 3), np.float32)
        )
        assert core.Dim("B") in executable.signature.output_specs[0].shape

        for batch in (2, 4):
            x = _lin((batch, 3), -1.0, 1.0)
            got = nu.run_native(executable, x)
            assert got.shape == (batch, 6)
            nu.assert_tree_parity(
                got, nu.evaluate_numpy(_symbolic_concat, x), label=f"out B={batch}"
            )
            np.testing.assert_allclose(
                got, np.concatenate((x, x * 2.0), axis=1), rtol=1e-6, atol=1e-6
            )


# ---------------------------------------------------------------------------
# 6. Graph transforms on the native backend: grad and vmap
# ---------------------------------------------------------------------------


def _quadratic(x):
    return evoxc.sum(x * x)


def _mlp_loss(params, x):
    hidden = evoxc.tanh(evoxc.dot(x, params["w1"]))
    logits = evoxc.dot(hidden, params["w2"])
    return evoxc.sum(logits * logits) / 2.0


def _row_energy(v):
    return evoxc.sum(v * v)


def _weighted_row_sum(w, v):
    return evoxc.sum(w * v)


class TestTransforms:
    def test_grad_of_a_quadratic(self):
        x = _normal((3, 4), seed=21)
        graph = evoxc.grad(_quadratic, argnums=0)(core.TensorSpec((3, 4), np.float32))

        got = _run_pair(graph, x)
        assert got.shape == (3, 4)
        np.testing.assert_allclose(got, 2.0 * x, rtol=1e-5, atol=1e-6)

    def test_grad_of_an_mlp_loss_per_weight_tree(self):
        params = {
            "w1": _normal((3, 4), seed=22),
            "w2": _normal((4, 1), seed=23),
        }
        x = _normal((2, 3), seed=24)
        specs = (
            {
                "w1": core.TensorSpec((3, 4), np.float32),
                "w2": core.TensorSpec((4, 1), np.float32),
            },
            core.TensorSpec((2, 3), np.float32),
        )
        graph = evoxc.grad(_mlp_loss, argnums=(0, 1))(*specs)

        got = _run_pair(graph, params, x)
        assert isinstance(got, tuple) and len(got) == 2

        hidden = np.tanh(x @ params["w1"])
        logits = hidden @ params["w2"]
        delta = (logits @ params["w2"].T) * (1.0 - hidden * hidden)
        np.testing.assert_allclose(got[0], x.T @ delta, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(got[1], hidden.T @ logits, rtol=1e-4, atol=1e-5)

    def test_vmap_of_a_row_reduction(self):
        x = _normal((3, 4), seed=25)
        graph = evoxc.vmap(_row_energy, in_axes=0)(core.TensorSpec((3, 4), np.float32))

        got = _run_pair(graph, x)
        assert got.shape == (3,)
        np.testing.assert_allclose(got, (x * x).sum(axis=1), rtol=1e-5, atol=1e-6)

    def test_vmap_mapped_and_unmapped_operands(self):
        w = _normal((3, 4), seed=26)
        v = _normal((4,), seed=27)
        graph = evoxc.vmap(_weighted_row_sum, in_axes=(0, None))(
            core.TensorSpec((3, 4), np.float32), core.TensorSpec((4,), np.float32)
        )

        got = _run_pair(graph, w, v)
        assert got.shape == (3,)
        np.testing.assert_allclose(got, (w * v).sum(axis=1), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 7. External kernels — the per-backend ``native`` host slot
# ---------------------------------------------------------------------------

_feat_ext_calls = {"n": 0}


def _feat_ext_kernel(a, b):
    """The kernel itself (numpy in / numpy out — the host contract)."""
    _feat_ext_calls["n"] += 1
    return np.tanh(a + b) * 1.5


evoxc.register_external_kernel("feat_ext_host", _feat_ext_kernel, backend="native")
evoxc.register_external_kernel("feat_ext_host", _feat_ext_kernel, backend="numpy")
evoxc.register_external_kernel(
    "feat_ext_device", _feat_ext_kernel, backend="native", device_resident=True
)


@evoxc.defn
def _external_graph(x, y):
    z = evoxc.external_call(
        "feat_ext_host", x, y, result=evoxc.TensorSpec((4,), evoxc.float32)
    )
    return evoxc.sum(z) + y


@evoxc.defn
def _external_device_graph(x, y):
    return evoxc.external_call(
        "feat_ext_device", x, y, result=evoxc.TensorSpec((4,), evoxc.float32)
    )


@evoxc.defn
def _external_unregistered_graph(x):
    return evoxc.external_call(
        "feat_ext_unregistered", x, result=evoxc.TensorSpec((4,), evoxc.float32)
    )


class TestExternalKernels:
    def test_host_slot_runs_the_kernel(self):
        x = _lin((4,), -1.0, 1.0)
        y = np.array([0.5, -0.5, 1.0, -1.0], np.float32)

        before = _feat_ext_calls["n"]
        executable = nu.build_native(_external_graph, *nu.specs_for(x, y))
        assert _feat_ext_calls["n"] == before  # never at staging time

        got = nu.run_native(executable, x, y)
        assert _feat_ext_calls["n"] == before + 1  # exactly once, at run time
        nu.assert_tree_parity(
            got, nu.evaluate_numpy(_external_graph, x, y), label="external host"
        )
        # Independent reference: the kernel body, reduced, then the trailing add.
        np.testing.assert_allclose(
            got, np.sum(np.tanh(x + y) * 1.5) + y, rtol=1e-5, atol=1e-6
        )

    def test_device_resident_slot_is_rejected(self):
        x = _lin((4,), -1.0, 1.0)
        y = np.ones((4,), np.float32)
        executable = nu.build_native(_external_device_graph, *nu.specs_for(x, y))
        with pytest.raises(core.BackendError, match="device_resident"):
            nu.run_native(executable, x, y)

    def test_unregistered_name_raises_at_run_time(self):
        x = _lin((4,), -1.0, 1.0)
        executable = nu.build_native(_external_unregistered_graph, *nu.specs_for(x))

        with pytest.raises(core.BackendError) as excinfo:
            nu.run_native(executable, x)
        message = str(excinfo.value)
        assert "feat_ext_unregistered" in message
        assert "evoxc.register_external_kernel" in message


# ---------------------------------------------------------------------------
# 8. Custom blocks — a registered ``native`` impl, portables inlined
# ---------------------------------------------------------------------------

_feat_block_calls = {"native": 0, "numpy": 0}


feat_block_scaled = evoxc.block(
    "feat_block_scaled",
    inputs=[evoxc.TensorSpec((4,), evoxc.float32)],
    outputs=[evoxc.TensorSpec((4,), evoxc.float32)],
    attributes={"scale": float},
)


@feat_block_scaled.impl("native")
def _feat_block_scaled_native(x, **static):
    _feat_block_calls["native"] += 1
    return np.asarray(x) * static["scale"]["value"] + 1.0


@feat_block_scaled.impl("numpy")
def _feat_block_scaled_numpy(x, **static):
    _feat_block_calls["numpy"] += 1
    return np.asarray(x) * static["scale"]["value"] + 1.0


feat_block_numpy_only = evoxc.block(
    "feat_block_numpy_only",
    inputs=[evoxc.TensorSpec((4,), evoxc.float32)],
    outputs=[evoxc.TensorSpec((4,), evoxc.float32)],
)


@feat_block_numpy_only.impl("numpy")
def _feat_block_numpy_only_impl(x):
    return np.asarray(x) * 3.0


feat_block_portable = evoxc.block(
    "feat_block_portable",
    inputs=[evoxc.TensorSpec((4,), evoxc.float32)],
    outputs=[evoxc.TensorSpec((4,), evoxc.float32)],
)


@feat_block_portable.portable
@evoxc.defn
def _feat_block_portable_impl(
    x: evoxc.TensorSpec((4,), evoxc.float32),
) -> evoxc.TensorSpec((4,), evoxc.float32):
    return x * 4.0


@evoxc.defn
def _block_graph(x):
    return feat_block_scaled(x, 2.5) + 0.5


@evoxc.defn
def _numpy_only_block_graph(x):
    return feat_block_numpy_only(x)


@evoxc.defn
def _portable_block_graph(x):
    return feat_block_portable(x) + 1.0


class TestCustomBlocks:
    def test_native_impl_dispatches_through_the_resolver(self):
        x = _lin((4,), -1.0, 1.0)
        _feat_block_calls["native"] = 0
        _feat_block_calls["numpy"] = 0

        executable = nu.build_native(_block_graph, core.TensorSpec((4,), np.float32))
        assert _feat_block_calls["native"] == 0  # impls never run at staging

        got = nu.run_native(executable, x)
        assert _feat_block_calls["native"] == 1
        assert _feat_block_calls["numpy"] == 0  # no cross-backend fallback
        np.testing.assert_allclose(got, x * 2.5 + 1.5, rtol=1e-6, atol=1e-6)

        # The artifact self-describes the custom op it needs.
        assert "feat_block_scaled" in executable.summary()["required_ops"]
        assert "feat_block_scaled" in (
            executable.backend_executable.artifact.required_custom_ops
        )

        # The numpy backend dispatches to ITS own registered impl.
        expected = nu.evaluate_numpy(_block_graph, x)
        assert _feat_block_calls["numpy"] == 1
        nu.assert_tree_parity(got, expected, label="block native vs numpy")

    def test_block_with_only_a_numpy_impl_fails_at_lower(self):
        graph = evoxc.trace(
            _numpy_only_block_graph, core.TensorSpec((4,), np.float32)
        )
        with pytest.raises(core.BackendError) as excinfo:
            evoxc.lower(graph, backend="native")
        message = str(excinfo.value)
        assert "feat_block_numpy_only" in message
        assert "native impl" in message

    def test_portable_only_block_is_inlined(self):
        x = _lin((4,), -1.0, 1.0)
        executable = nu.build_native(
            _portable_block_graph, core.TensorSpec((4,), np.float32)
        )
        # Inlined at lower() ⇒ the engine needs no custom impl for it.
        assert "feat_block_portable" not in executable.summary()["required_ops"]

        got = nu.run_native(executable, x)
        np.testing.assert_allclose(got, x * 4.0 + 1.0, rtol=1e-6, atol=1e-6)
        nu.assert_tree_parity(
            got, nu.evaluate_numpy(_portable_block_graph, x), label="portable"
        )


# ---------------------------------------------------------------------------
# 9. Collectives — single-rank identity semantics
# ---------------------------------------------------------------------------


@evoxc.defn
def _coll_all_reduce(x):
    return evoxc.dist.all_reduce(x)


@evoxc.defn
def _coll_all_gather(x):
    return evoxc.dist.all_gather(x)


@evoxc.defn
def _coll_reduce_scatter(x):
    return evoxc.dist.reduce_scatter(x)


@evoxc.defn
def _coll_broadcast(x):
    return evoxc.dist.broadcast(x)


@evoxc.defn
def _coll_rank_world_size(x):
    return evoxc.sum(x) + evoxc.dist.rank() + evoxc.dist.world_size()


class TestCollectives:
    @pytest.mark.parametrize(
        "fn",
        [_coll_all_reduce, _coll_all_gather, _coll_reduce_scatter, _coll_broadcast],
        ids=["all_reduce", "all_gather", "reduce_scatter", "broadcast"],
    )
    def test_single_rank_identity(self, fn):
        x = _lin((2, 3), -1.0, 1.0)
        got = nu.evaluate_pair(fn, x)
        # world_size == 1 ⇒ every collective is the identity for one rank.
        nu.assert_tree_parity(got, x, label="single rank identity")

    def test_rank_and_world_size_graph_scalars(self):
        x = _lin((3,), -1.0, 1.0)
        got = nu.evaluate_pair(_coll_rank_world_size, x)
        assert np.shape(got) == ()
        np.testing.assert_allclose(
            got, np.float32(np.sum(x)) + np.float32(1.0), rtol=1e-5, atol=1e-6
        )


# ---------------------------------------------------------------------------
# 10. Sparse rejection — numpy-backend-only by contract
# ---------------------------------------------------------------------------


def _sparse_sum_graph():
    spec = evoxc.sparse.SparseTensorSpec(
        core.TensorSpec((None, 2), evoxc.int64),
        core.TensorSpec((None,), evoxc.float32),
        dense_shape=(3, 4),
    )

    @evoxc.defn
    def sparse_sum(s):
        return evoxc.sum(evoxc.sparse.to_dense(s))

    return evoxc.trace(sparse_sum, spec)


class TestSparseRejection:
    def test_sparse_op_defers_with_an_actionable_error(self):
        graph = _sparse_sum_graph()
        with pytest.raises(core.BackendError) as excinfo:
            evoxc.lower(graph, backend="native")
        message = str(excinfo.value)
        assert "sparse_to_dense" in message  # names the offending op
        assert "backend='numpy'" in message  # names the supported route


# ---------------------------------------------------------------------------
# 11. Loose performance sanity — a ~250-op chain, native <= 5x numpy
# ---------------------------------------------------------------------------

_CHAIN_SIZE = 64
_CHAIN_STEPS = 50


@evoxc.defn
def _long_chain(x):
    y = x
    for _ in range(_CHAIN_STEPS):
        y = evoxc.tanh(evoxc.multiply(y, 1.0009) + 0.001)
    return evoxc.sum(y) + evoxc.max(evoxc.reshape(y, (-1,)))


def _median_run_seconds(executable, array, *, warmup=3, repeats=20):
    """Median wall-clock of ``run`` after warmup (this backend runs sync)."""
    for _ in range(warmup):
        evoxc.run(executable, array)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        evoxc.run(executable, array)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


class TestPerformanceSanity:
    def test_native_is_within_five_times_the_numpy_engine(self):
        """A LOOSE sanity bound, not a benchmark: the Rust engine must not be
        pathologically slower than the Python interpreter on a long dispatch
        chain (no absolute-time assertions — host load varies)."""
        x = _normal((_CHAIN_SIZE, _CHAIN_SIZE), seed=31)
        spec = core.TensorSpec((_CHAIN_SIZE, _CHAIN_SIZE), np.float32)

        numpy_executable = evoxc.build(_long_chain, spec, backend="numpy")
        native_executable = evoxc.build(_long_chain, spec, backend="native")

        numpy_median = _median_run_seconds(numpy_executable, x)
        native_median = _median_run_seconds(native_executable, x)

        nu.assert_tree_parity(
            nu.as_numpy(evoxc.run(native_executable, x)),
            nu.as_numpy(evoxc.run(numpy_executable, x)),
            label="chain",
        )

        ops = native_executable.backend_executable.stats()["ops"]
        assert native_median <= 5.0 * numpy_median, (
            f"native median {native_median * 1e3:.3f} ms is more than 5x the "
            f"numpy median {numpy_median * 1e3:.3f} ms on a {ops}-dispatch "
            f"({_CHAIN_SIZE}x{_CHAIN_SIZE} f32) chain"
        )
        assert ops >= 200  # the chain really is a long dispatch sequence
