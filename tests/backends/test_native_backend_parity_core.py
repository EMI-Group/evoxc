"""Native (Rust CPU engine) vs numpy parity: elementwise / comparison / reduction ops.

The ``native`` backend executes an ordinary ``Graph`` on the in-process Rust
engine (``evoxc-cpu``); the ``numpy`` interpreter is the numerical reference for
the same graph. Every case here traces ONE graph and evaluates it on BOTH
backends through :func:`tests.backends._native_utils.evaluate_pair`, which
asserts per-dtype parity (exact for bool/integer, ``rtol``/``atol`` per float
width) AND exact dtype/shape agreement — a dtype drift is a real bug, never a
tolerance question.

Scope matches the objective: binary + unary elementwise, comparisons, the
backend's declared 14-dtype surface, broadcasting, reductions (incl. dtype
promotion and NaN propagation), argmax/argmin (tie + NaN semantics), the
cumulative ops and one fused chain staged through ``evoxc.build`` and reused
across input sets. Deferred/unsupported ops are NOT guessed at here: the
engine's own ``lower()`` gate owns that contract (see
``evoxc/backends/native/CONTEXT.md``).

Conventions (see ``tests/backends/CONTEXT.md``): fixed seeds, small shapes, no
timing, CPU only, no eager numerical reimplementation — the numpy backend is
the oracle.
"""
import numpy as np
import pytest

import evoxc
from evoxc import core
from evoxc.backends.native import engine
from tests.backends import _native_utils as nu

pytestmark = pytest.mark.skipif(not nu.ENGINE_AVAILABLE, reason=nu.SKIP_REASON)


# ---------------------------------------------------------------------------
# Helpers: graph factories + deterministic inputs
# ---------------------------------------------------------------------------


def _binary(op):
    """A two-input defn calling ``op`` (parametrization-friendly factory)."""

    @evoxc.defn
    def fn(a, b):
        return op(a, b)

    return fn


def _unary(op):
    """A one-input defn calling ``op``."""

    @evoxc.defn
    def fn(x):
        return op(x)

    return fn


def _reduce(op, axes, keepdims):
    """A defn reducing ``x`` by ``op``; ``axes``/``keepdims`` are static values."""

    @evoxc.defn
    def fn(x):
        return op(x, axes=axes, keepdims=keepdims)

    return fn


def _arg(op, axis, keepdims):
    """A defn taking an arg-extremum of ``x`` (static ``axis``/``keepdims``)."""

    @evoxc.defn
    def fn(x):
        return op(x, axis=axis, keepdims=keepdims)

    return fn


def _signed(shape=(3, 4), seed=2):
    """Fixed-seed normal float32 values spanning both signs."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape).astype(np.float32)


def _unit(shape=(3, 4), seed=0):
    """Values strictly inside ``(-1, 1)`` — safe for tan and log1p."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.9, 0.9, size=shape).astype(np.float32)


def _positive(shape=(3, 4), seed=1):
    """Strictly positive values — the domain of log/log1p/sqrt."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.5, 3.0, size=shape).astype(np.float32)


def _ints(shape=(3, 4), low=-3, high=4, dtype=np.int32, seed=3):
    """Fixed-seed small integers (no accumulation overflow)."""
    rng = np.random.default_rng(seed)
    return rng.integers(low, high, size=shape).astype(dtype)


def _comparison_pair():
    """Float32 operands carrying NaN and both signed zeros."""
    a = np.array([[np.nan, 0.0, -0.0, 1.0, -1.0, 2.5]], dtype=np.float32)
    b = np.array([[np.nan, -0.0, 0.0, -1.0, 1.0, 2.5]], dtype=np.float32)
    return a, b


# The backend's declared dtype surface (``engine.SUPPORTED_DTYPES``), by name.
_DTYPE_NAMES = [
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]


def _sample(dtype_name, shape=(3, 4), seed=3):
    """A deterministic array of ``dtype_name`` with overflow-safe magnitudes."""
    dtype = np.dtype(dtype_name)
    rng = np.random.default_rng(seed)
    if dtype.kind == "b":
        return rng.random(shape) > 0.5
    if dtype.kind in "iu":
        low = 0 if dtype.kind == "u" else -7
        return rng.integers(low, 8, size=shape).astype(dtype)
    if dtype.kind == "f":
        return rng.normal(size=shape).astype(dtype)
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)


# ---------------------------------------------------------------------------
# 1. Binary elementwise ops
# ---------------------------------------------------------------------------


class TestElementwiseBinary:
    @pytest.mark.parametrize(
        "name, op",
        [
            ("add", evoxc.add),
            ("subtract", evoxc.subtract),
            ("multiply", evoxc.multiply),
            ("maximum", evoxc.maximum),
            ("minimum", evoxc.minimum),
        ],
    )
    def test_float_binary(self, name, op):
        a, b = _signed(seed=10), _signed(seed=11)
        got = nu.evaluate_pair(_binary(op), a, b)
        assert got.dtype == np.float32
        assert got.shape == (3, 4)

    @pytest.mark.parametrize("name, op", [("divide", evoxc.divide), ("power", evoxc.power)])
    def test_float_divide_and_power(self, name, op):
        """Positive base / non-zero divisor keeps the case in the op's domain."""
        a, b = _positive(seed=12), _positive(seed=13)
        got = nu.evaluate_pair(_binary(op), a, b)
        assert got.dtype == np.float32

    @pytest.mark.parametrize("divisor_sign", [1.0, -1.0])
    def test_float_remainder_is_floormod(self, divisor_sign):
        """``remainder`` takes the sign of the DIVISOR (numpy floormod, not fmod)."""
        a = np.array([[-4.5, -4.5, 4.5, 4.5, -1.0, 1.0]], dtype=np.float32)
        b = divisor_sign * np.array([[3.0, 0.5, 3.0, 0.5, 2.0, 2.0]], dtype=np.float32)
        got = nu.evaluate_pair(_binary(evoxc.remainder), a, b)
        np.testing.assert_array_equal(got, np.remainder(a, b))
        # fmod differs on the negative-mixed pairs: prove the native path is floormod.
        assert not np.array_equal(got, np.fmod(a, b))

    def test_integer_remainder_mixed_signs(self):
        a = np.array([[-7, -1, 0, 1, 7, -7]], dtype=np.int32)
        b = np.array([[3, 3, 3, 3, 3, -3]], dtype=np.int32)
        got = nu.evaluate_pair(_binary(evoxc.remainder), a, b)
        assert got.dtype == np.int32
        np.testing.assert_array_equal(got, np.remainder(a, b))

    @pytest.mark.parametrize(
        "name, op, np_op",
        [
            ("bitwise_and", evoxc.bitwise_and, np.bitwise_and),
            ("bitwise_or", evoxc.bitwise_or, np.bitwise_or),
            ("bitwise_xor", evoxc.bitwise_xor, np.bitwise_xor),
        ],
    )
    def test_integer_bitwise(self, name, op, np_op):
        a, b = _ints(seed=14), _ints(seed=15)
        got = nu.evaluate_pair(_binary(op), a, b)
        assert got.dtype == np.int32
        np.testing.assert_array_equal(got, np_op(a, b))

    @pytest.mark.parametrize(
        "name, op, np_op",
        [
            ("bitwise_left_shift", evoxc.bitwise_left_shift, np.left_shift),
            ("bitwise_right_shift", evoxc.bitwise_right_shift, np.right_shift),
        ],
    )
    def test_integer_shifts(self, name, op, np_op):
        a = _ints(low=-8, high=9, seed=16)
        b = _ints(low=0, high=4, seed=17)
        got = nu.evaluate_pair(_binary(op), a, b)
        assert got.dtype == np.int32
        np.testing.assert_array_equal(got, np_op(a, b))

    @pytest.mark.parametrize(
        "name, op, np_op",
        [("logical_and", evoxc.logical_and, np.logical_and), ("logical_or", evoxc.logical_or, np.logical_or)],
    )
    def test_boolean_logical(self, name, op, np_op):
        rng = np.random.default_rng(18)
        a, b = rng.random((3, 4)) > 0.5, rng.random((3, 4)) > 0.5
        got = nu.evaluate_pair(_binary(op), a, b)
        assert got.dtype == np.bool_
        np.testing.assert_array_equal(got, np_op(a, b))

    def test_logical_not(self):
        rng = np.random.default_rng(19)
        x = rng.random((3, 4)) > 0.5
        got = nu.evaluate_pair(_unary(evoxc.logical_not), x)
        assert got.dtype == np.bool_
        np.testing.assert_array_equal(got, np.logical_not(x))


# ---------------------------------------------------------------------------
# 2. Unary elementwise ops
# ---------------------------------------------------------------------------


class TestElementwiseUnary:
    @pytest.mark.parametrize(
        "name, op, inputs",
        [
            ("negate", evoxc.negate, _signed),
            ("abs", evoxc.abs, _signed),
            ("exp", evoxc.exp, _unit),
            ("log", evoxc.log, _positive),
            ("log1p", evoxc.log1p, _unit),
            ("sqrt", evoxc.sqrt, _positive),
            ("square", evoxc.square, _unit),
            ("sin", evoxc.sin, _unit),
            ("cos", evoxc.cos, _unit),
            ("tan", evoxc.tan, _unit),
            ("tanh", evoxc.tanh, _unit),
            ("sigmoid", evoxc.sigmoid, _unit),
            ("relu", evoxc.relu, _signed),
            ("erf", evoxc.erf, _unit),
            ("gelu", evoxc.gelu, _unit),
            ("sign", evoxc.sign, _signed),
            ("round", evoxc.round, _signed),
            ("ceil", evoxc.ceil, _signed),
            ("floor", evoxc.floor, _signed),
            ("stop_gradient", evoxc.stop_gradient, _signed),
        ],
    )
    def test_float_unary(self, name, op, inputs):
        x = inputs()
        got = nu.evaluate_pair(_unary(op), x)
        assert got.dtype == np.float32
        assert got.shape == x.shape

    def test_activation_formulas(self):
        """The numpy backend is the oracle, but pin the formulas for two composites.

        The engine's ``tanh``/``exp`` are its own implementations, so they are
        compared with the documented float32 budget (:func:`nu.dtype_tolerance`)
        rather than bit-exactly: measured drift vs ``np.tanh`` is <= 2 ulp
        (5.96e-8 abs, 1.19e-7 rel) — well inside rtol=1e-5 / atol=1e-6.
        """
        rtol, atol = nu.dtype_tolerance(np.float32)
        x = _signed(seed=20)
        tanh = nu.evaluate_pair(_unary(evoxc.tanh), x)
        np.testing.assert_allclose(tanh, np.tanh(x), rtol=rtol, atol=atol)
        sigmoid = nu.evaluate_pair(_unary(evoxc.sigmoid), x)
        np.testing.assert_allclose(sigmoid, 1.0 / (1.0 + np.exp(-x)), rtol=rtol, atol=atol)

    def test_cast_float_to_int_truncates(self):
        x = np.array([[-0.9, -0.5, -0.1, 0.5], [1.5, 2.5, 3.7, -3.7]], dtype=np.float32)
        got = nu.evaluate_pair(_unary(lambda v: evoxc.cast(v, np.int32)), x)
        assert got.dtype == np.int32
        # astype truncates toward zero (never rounds) on both backends.
        np.testing.assert_array_equal(got, np.trunc(x).astype(np.int32))

    def test_cast_int_to_float_exact(self):
        x = _ints(low=-64, high=65, seed=21)
        got = nu.evaluate_pair(_unary(lambda v: evoxc.cast(v, np.float32)), x)
        assert got.dtype == np.float32
        np.testing.assert_array_equal(got, x.astype(np.float32))


# ---------------------------------------------------------------------------
# 3. Comparisons
# ---------------------------------------------------------------------------


class TestComparisons:
    @pytest.mark.parametrize(
        "name, op, np_op",
        [
            ("equal", evoxc.equal, np.equal),
            ("not_equal", evoxc.not_equal, np.not_equal),
            ("less", evoxc.less, np.less),
            ("less_equal", evoxc.less_equal, np.less_equal),
            ("greater", evoxc.greater, np.greater),
            ("greater_equal", evoxc.greater_equal, np.greater_equal),
        ],
    )
    def test_comparison_ops(self, name, op, np_op):
        a, b = _comparison_pair()
        got = nu.evaluate_pair(_binary(op), a, b)
        # assert_parity checks dtype equality; pin the bool contract explicitly.
        assert got.dtype == np.bool_
        assert got.shape == a.shape
        np.testing.assert_array_equal(got, np_op(a, b))

    def test_nan_and_signed_zero_semantics(self):
        a, b = _comparison_pair()
        equal = nu.evaluate_pair(_binary(evoxc.equal), a, b)
        np.testing.assert_array_equal(equal, [[False, True, True, False, False, True]])
        not_equal = nu.evaluate_pair(_binary(evoxc.not_equal), a, b)
        np.testing.assert_array_equal(not_equal, [[True, False, False, True, True, False]])
        # NaN is unordered: every ordering comparison against it is False.
        for op in (evoxc.less, evoxc.less_equal, evoxc.greater, evoxc.greater_equal):
            got = nu.evaluate_pair(_binary(op), a, b)
            assert got[0, 0] == np.bool_(False)


# ---------------------------------------------------------------------------
# 4. The declared 14-dtype surface
# ---------------------------------------------------------------------------


def _arithmetic_graph():
    @evoxc.defn
    def fn(a, b):
        return evoxc.multiply(evoxc.add(a, b), a)

    return fn


def _boolean_graph():
    @evoxc.defn
    def fn(a, b):
        return evoxc.logical_or(evoxc.logical_and(a, b), evoxc.logical_not(a))

    return fn


class TestDtypeSurface:
    def test_supported_dtype_set_is_exactly_the_fourteen(self):
        names = {np.dtype(dtype).name for dtype in engine.SUPPORTED_DTYPES}
        assert names == set(_DTYPE_NAMES)

    @pytest.mark.parametrize("dtype_name", _DTYPE_NAMES)
    def test_add_then_multiply_per_dtype(self, dtype_name):
        a = _sample(dtype_name, seed=30)
        b = _sample(dtype_name, seed=31)
        # numpy's own bool arithmetic is not the evoxc contract: bool uses logic.
        fn = _boolean_graph() if np.dtype(dtype_name).kind == "b" else _arithmetic_graph()
        got = nu.evaluate_pair(fn, a, b)
        assert got.dtype == np.dtype(dtype_name)
        assert got.shape == a.shape

    @pytest.mark.parametrize(
        "a_dtype, b_dtype",
        [(np.int32, np.int32), (np.int32, np.float32), (np.int8, np.int64), (np.float16, np.float32)],
    )
    def test_divide_promotion_matches_reference(self, a_dtype, b_dtype):
        """True division promotes identically on both backends (never truncates)."""
        a = _sample(np.dtype(a_dtype).name, seed=32)
        b = np.abs(_sample(np.dtype(b_dtype).name, seed=33)) + 1
        got = nu.evaluate_pair(_binary(evoxc.divide), a, b)
        assert got.dtype.kind == "f"


# ---------------------------------------------------------------------------
# 5. Broadcasting
# ---------------------------------------------------------------------------


class TestBroadcasting:
    @pytest.mark.parametrize(
        "name, op, np_op, shape_a, shape_b",
        [
            ("add_3x1_1x4", evoxc.add, np.add, (3, 1), (1, 4)),
            ("add_scalar_2x3", evoxc.add, np.add, (), (2, 3)),
            ("multiply_2x1x4_3x1", evoxc.multiply, np.multiply, (2, 1, 4), (3, 1)),
            ("multiply_4_2x3x4", evoxc.multiply, np.multiply, (4,), (2, 3, 4)),
            ("less_3x1_1x4", evoxc.less, np.less, (3, 1), (1, 4)),
        ],
    )
    def test_broadcast_binary(self, name, op, np_op, shape_a, shape_b):
        rng = np.random.default_rng(40)
        a = rng.normal(size=shape_a).astype(np.float32)
        b = rng.normal(size=shape_b).astype(np.float32)
        got = nu.evaluate_pair(_binary(op), a, b)
        assert got.shape == np.broadcast_shapes(shape_a, shape_b)
        np.testing.assert_allclose(got, np_op(a, b), rtol=0, atol=0)

    def test_broadcast_keeps_both_operand_shapes(self):
        """A (2,1,4) x (3,1) broadcast is (2,3,4) and does not collapse a dim."""
        a, b = _signed((2, 1, 4), seed=41), _signed((3, 1), seed=42)
        got = nu.evaluate_pair(_binary(evoxc.add), a, b)
        assert got.shape == (2, 3, 4)
        np.testing.assert_allclose(got, a + b, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 6. Reductions
# ---------------------------------------------------------------------------

_REDUCTIONS = [
    ("reduce_sum", evoxc.reduce_sum, np.sum),
    ("reduce_max", evoxc.reduce_max, np.max),
    ("reduce_min", evoxc.reduce_min, np.min),
    ("reduce_prod", evoxc.reduce_prod, np.prod),
    ("reduce_mean", evoxc.reduce_mean, np.mean),
]

_AXIS_CASES = [
    (None, False),
    (0, False),
    ((1,), False),
    ((0, 2), False),
    (None, True),
    (0, True),
    ((0, 2), True),
]


class TestReductions:
    @pytest.mark.parametrize("axes, keepdims", _AXIS_CASES)
    @pytest.mark.parametrize("name, op, np_op", _REDUCTIONS)
    def test_reduce_rank3(self, name, op, np_op, axes, keepdims):
        x = np.random.default_rng(50).normal(size=(2, 3, 4)).astype(np.float32)
        got = nu.evaluate_pair(_reduce(op, axes, keepdims), x)
        # The numpy backend's values are the contract (asserted by evaluate_pair);
        # the SHAPE and the raw numpy values are pinned here too — every one of
        # these five reductions agrees with numpy element-for-element on f32.
        reference = np_op(x, axis=axes, keepdims=keepdims)
        assert got.shape == np.shape(reference)
        np.testing.assert_allclose(got, reference, rtol=1e-6, atol=1e-6)

    def test_sum_api_spelling_matches_reduce_sum(self):
        x = np.random.default_rng(51).normal(size=(3, 4)).astype(np.float32)
        spelled = nu.evaluate_pair(_reduce(evoxc.sum, (1,), False), x)
        direct = nu.evaluate_pair(_reduce(evoxc.reduce_sum, (1,), False), x)
        nu.assert_parity(spelled, direct, label="evoxc.sum vs evoxc.reduce_sum")

    @pytest.mark.parametrize("dtype_name", ["bool", "int8", "int32", "int64", "uint8", "uint16"])
    def test_integer_and_bool_reduction_promotion(self, dtype_name):
        """Integer/bool sum-prod-mean promote exactly like the numpy backend."""
        x = _sample(dtype_name, seed=52)
        for op in (evoxc.reduce_sum, evoxc.reduce_prod, evoxc.reduce_mean, evoxc.reduce_max):
            got = nu.evaluate_pair(_reduce(op, None, False), x)
            assert got.shape == ()
        # bool sums to a count (integer) and means to a float, like numpy; an
        # unsigned input keeps an UNSIGNED accumulator (uint8/uint16 -> uint64).
        total = nu.evaluate_pair(_reduce(evoxc.reduce_sum, (0,), False), x)
        assert total.dtype.kind in "iu"
        mean = nu.evaluate_pair(_reduce(evoxc.reduce_mean, (0,), False), x)
        assert mean.dtype.kind == "f"

    @pytest.mark.parametrize("op, np_op", [(evoxc.reduce_max, np.max), (evoxc.reduce_min, np.min), (evoxc.reduce_mean, np.mean)])
    @pytest.mark.parametrize("axes", [None, 0, 1])
    def test_nan_inputs(self, op, np_op, axes):
        """NaN propagates through max/min/mean identically to the numpy backend."""
        x = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float32)
        got = nu.evaluate_pair(_reduce(op, axes, False), x)
        reference = np_op(x, axis=axes)
        np.testing.assert_array_equal(np.isnan(got), np.isnan(reference))
        np.testing.assert_array_equal(got, reference)

    def test_keepdims_preserves_rank(self):
        x = np.random.default_rng(53).normal(size=(2, 3, 4)).astype(np.float32)
        got = nu.evaluate_pair(_reduce(evoxc.reduce_sum, None, True), x)
        assert got.shape == (1, 1, 1)
        assert got.ndim == x.ndim


# ---------------------------------------------------------------------------
# 7. argmax / argmin
# ---------------------------------------------------------------------------

_TIE_INPUT = np.array(
    [[1, 3, 3, 2], [0, 5, 5, 5], [7, 7, 1, 7]],
    dtype=np.int32,
)


class TestArgExtrema:
    @pytest.mark.parametrize("axis, keepdims", [(None, False), (0, False), (1, False), (0, True), (1, True), (None, True)])
    @pytest.mark.parametrize("name, op, np_op", [("argmax", evoxc.argmax, np.argmax), ("argmin", evoxc.argmin, np.argmin)])
    def test_first_occurrence_ties(self, name, op, np_op, axis, keepdims):
        got = nu.evaluate_pair(_arg(op, axis, keepdims), _TIE_INPUT)
        assert got.dtype == np.int64
        # numpy (and evoxc) resolve ties to the FIRST extreme index.
        np.testing.assert_array_equal(got, np_op(_TIE_INPUT, axis=axis, keepdims=keepdims))

    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("name, op", [("argmax", evoxc.argmax), ("argmin", evoxc.argmin)])
    def test_nan_bearing_input_matches_reference(self, name, op, axis):
        """NaN is an extreme for evoxc's arg-extrema (first NaN wins) — same as
        the numpy backend; this is deliberately NOT ``np.nanargmax``'s skip rule."""
        x = np.array([[1.0, np.nan, 3.0], [np.nan, 2.0, 0.0]], dtype=np.float32)
        got = nu.evaluate_pair(_arg(op, axis, False), x)
        assert got.dtype == np.int64
        # reference: the same graph on the numpy backend, evaluated explicitly.
        expected = nu.evaluate_numpy(_arg(op, axis, False), x)
        np.testing.assert_array_equal(got, expected)

    def test_reduction_argmax_axis_none_is_flat_index(self):
        got = nu.evaluate_pair(_arg(evoxc.argmax, None, False), _TIE_INPUT)
        assert got.shape == ()
        assert int(got) == int(np.argmax(_TIE_INPUT))


# ---------------------------------------------------------------------------
# 8. cumsum / cumprod
# ---------------------------------------------------------------------------


class TestCumulative:
    @pytest.mark.parametrize("name, op, np_op", [("cumsum", evoxc.cumsum, np.cumsum), ("cumprod", evoxc.cumprod, np.cumprod)])
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("dtype", [np.int32, np.float32])
    def test_cumulative_ops(self, name, op, np_op, axis, dtype):
        x = _ints(low=-3, high=4, seed=60) if dtype is np.int32 else _unit(seed=61)
        got = nu.evaluate_pair(_unary(lambda v: op(v, axis=axis)), x)
        assert got.dtype == x.dtype
        assert got.shape == x.shape
        np.testing.assert_allclose(got, np_op(x, axis=axis), rtol=0, atol=0)

    def test_cumsum_axis0_matches_manual_accumulation(self):
        x = np.array([[1.0, -2.0], [3.0, 4.0], [0.5, -1.0]], dtype=np.float32)
        got = nu.evaluate_pair(_unary(lambda v: evoxc.cumsum(v, axis=0)), x)
        np.testing.assert_allclose(
            got,
            np.stack([x[0], x[0] + x[1], x[0] + x[1] + x[2]]),
            rtol=0,
            atol=0,
        )


# ---------------------------------------------------------------------------
# 9. Fused chain staged once through evoxc.build, reused for two input sets
# ---------------------------------------------------------------------------


def _fused_chain():
    @evoxc.defn
    def fn(a, b):
        h = evoxc.tanh(evoxc.dot(a, b))
        return evoxc.sigmoid(evoxc.reduce_sum(h, axes=(1,)))

    return fn


class TestFusedChainExecutableReuse:
    def test_one_executable_two_input_sets(self):
        fn = _fused_chain()
        first = (_signed((3, 4), seed=70), _signed((4, 5), seed=71))
        second = (_signed((3, 4), seed=72), _signed((4, 5), seed=73))

        specs = nu.specs_for(*first)
        assert all(isinstance(spec, core.TensorSpec) for spec in specs)
        executable = nu.build_native(fn, *specs)
        assert executable.backend == "native"

        # The SAME compiled executable serves both disjoint input sets.
        for label, (a, b) in (("first", first), ("second", second)):
            got = nu.run_native(executable, a, b)
            assert got.dtype == np.float32
            assert got.shape == (3,)
            nu.assert_parity(got, nu.evaluate_numpy(fn, a, b), label=f"{label} vs numpy backend")
            nu.assert_parity(got, nu.evaluate_native(fn, a, b), label=f"{label} vs native oracle")
            # Independent formula pin (no evoxc code): sigmoid(row-sum(tanh(a @ b))).
            reference = 1.0 / (1.0 + np.exp(-np.tanh(a @ b).sum(axis=1)))
            np.testing.assert_allclose(
                got, reference, rtol=1e-5, atol=1e-6, err_msg=f"{label}: fused-chain formula"
            )

    def test_chain_with_mixed_output_kinds(self):
        """One graph, one output tuple: reduction + cumulative + comparison."""

        @evoxc.defn
        def fn(a, b):
            product = evoxc.multiply(a, b)
            return (
                evoxc.reduce_sum(product, axes=0),
                evoxc.cumsum(product, axis=1),
                evoxc.greater(product, a),
            )

        a, b = _signed(seed=74), _signed(seed=75)
        got = nu.evaluate_pair(fn, a, b)
        assert len(got) == 3
        assert got[0].dtype == np.float32 and got[0].shape == (4,)
        assert got[1].dtype == np.float32 and got[1].shape == (3, 4)
        assert got[2].dtype == np.bool_ and got[2].shape == (3, 4)
