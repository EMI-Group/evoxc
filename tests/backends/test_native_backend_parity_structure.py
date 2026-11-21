"""Native (Rust CPU engine) vs numpy parity: structure / indexing / sorting / linalg.

The ``native`` backend executes an ordinary ``Graph`` on the in-process Rust
engine (``evoxc-cpu``); the ``numpy`` interpreter is the numerical reference for
the SAME graph. Every case traces ONE graph and evaluates it on BOTH backends
through :func:`tests.backends._native_utils.evaluate_pair`, which asserts
per-dtype parity (exact for bool/integer, ``rtol``/``atol`` per float width) AND
exact dtype/shape agreement — a dtype drift is a real bug, never a tolerance
question.

Where an independent reference is unambiguous the result is ALSO pinned against
an explicit ``np.*`` computation (``np.reshape``/``np.transpose``/``np.take``/
``np.put_along_axis``/``np.sort(kind="stable")``/``np.matmul``/``np.linalg.solve``
and a hand-written cross-correlation loop for ``conv``), so the file pins the
NUMBERS, not merely agreement between two backends.

Documented evoxc semantics honoured here (see ``tests/backends/CONTEXT.md``):
``slice`` lengths are SPANS (``limit = start + length``, strides must be
positive), ``dot`` requires rank >= 2, ``scatter`` is numpy ``put_along_axis``
on a copy (last write wins for duplicate indices; the input is never mutated),
descending ``sort``/``argsort`` is the flip of the stable ascending result.
Deferred/unsupported ops are not guessed at: the engine's own ``lower()`` gate
reports them as an explicit ``core.BackendError``.

Conventions: fixed seeds, small shapes, no timing, CPU only, no network, no
eager numerical reimplementation — the numpy backend is the oracle.
"""
import numpy as np
import pytest

import evoxc
from tests.backends import _native_utils as nu

pytestmark = pytest.mark.skipif(not nu.ENGINE_AVAILABLE, reason=nu.SKIP_REASON)


# ---------------------------------------------------------------------------
# Deterministic inputs and formula references
# ---------------------------------------------------------------------------


def _lin(shape, lo=-1.5, hi=1.5, dtype=np.float32):
    """Deterministic linearly-spaced array over ``[lo, hi]`` (no RNG)."""
    count = int(np.prod(shape))
    return np.linspace(lo, hi, count).reshape(shape).astype(dtype)


def _ints(shape, lo=-3, hi=4, dtype=np.int32):
    """Deterministic small integers in ``[lo, hi)`` (no accumulation overflow)."""
    count = int(np.prod(shape))
    return (np.arange(count) % (hi - lo) + lo).reshape(shape).astype(dtype)


def _normal(shape, seed, dtype=np.float32):
    """Fixed-seed normal values (rounded through float64 then cast)."""
    return np.random.default_rng(seed).normal(size=shape).astype(dtype)


def _unary(op):
    """A one-input ``defn`` calling ``op`` (parametrization-friendly factory)."""

    @evoxc.defn
    def fn(x):
        return op(x)

    return fn


def _padding_pairs(padding, ndim):
    """Normalize ``padding`` to one ``(before, after)`` pair per spatial axis."""
    if isinstance(padding, tuple) and len(padding) == 2 and not isinstance(
        padding[0], (tuple, list)
    ):
        return [padding] * ndim  # symmetric (lo, hi) on every spatial axis
    return [tuple(entry) for entry in padding]


def _cross_correlate(x, w, *, strides=1, padding=(0, 0)):
    """Hand-written NCHW cross-correlation loop — NO evoxc code.

    Matches the numpy kernel's semantics (``tests/backends/CONTEXT.md``): the
    input is padded with ``padding`` zeros per spatial axis and every output
    entry is ``sum_{ci, ki} x_pad[n, ci, i*stride + ki] * w[co, ci, ki]``.
    Computed in float64 and cast back so the float32 budget covers only the
    native engine's own reduction order.
    """
    dtype = np.asarray(x).dtype
    x = np.asarray(x, np.float64)
    w = np.asarray(w, np.float64)
    ndim = x.ndim - 2
    strides = (strides,) * ndim if isinstance(strides, int) else tuple(strides)
    pads = _padding_pairs(padding, ndim)
    padded = np.pad(x, [(0, 0), (0, 0)] + pads, mode="constant")
    batch, c_in = x.shape[:2]
    c_out = w.shape[0]
    assert w.shape[1] == c_in
    kshape = w.shape[2:]
    out_spatial = [
        (padded.shape[2 + i] - kshape[i]) // strides[i] + 1 for i in range(ndim)
    ]
    out = np.zeros((batch, c_out, *out_spatial), np.float64)
    for n in range(batch):
        for co in range(c_out):
            for out_index in np.ndindex(*out_spatial):
                acc = 0.0
                for ci in range(c_in):
                    for k_index in np.ndindex(*kshape):
                        src = tuple(
                            out_index[i] * strides[i] + k_index[i]
                            for i in range(ndim)
                        )
                        acc += padded[(n, ci) + src] * w[(co, ci) + k_index]
                out[(n, co) + tuple(out_index)] = acc
    return out.astype(dtype)


# --- 1. reshape / transpose / broadcast / select ---


class TestReshapeTranspose:
    @pytest.mark.parametrize(
        "source_shape, target",
        [
            ((4, 5), (5, -1)),
            ((4, 5), (2, -1)),
            ((2, 3, 4), (24,)),
            ((2, 3, 4), (2, 3, 4)),
        ],
        ids=["wildcard-last", "wildcard-last-2", "flatten", "identity"],
    )
    def test_reshape_matches_numpy(self, source_shape, target):
        x = _normal(source_shape, seed=1)
        got = nu.evaluate_pair(_unary(lambda t: evoxc.reshape(t, target)), x)
        np.testing.assert_array_equal(got, np.reshape(x, target))

    def test_reshape_rank3_to_rank2(self):
        x = _normal((2, 3, 4), seed=2)
        got = nu.evaluate_pair(_unary(lambda t: evoxc.reshape(t, (2, -1))), x)
        assert got.shape == (2, 12)
        np.testing.assert_array_equal(got, x.reshape(2, 12))

    @pytest.mark.parametrize("permutation", [None, (2, 0, 1), (1, 2, 0)])
    def test_transpose_default_and_explicit(self, permutation):
        x = _normal((2, 3, 4), seed=3)
        if permutation is None:
            fn = _unary(evoxc.transpose)
            reference = np.transpose(x)
        else:
            fn = _unary(lambda t: evoxc.transpose(t, permutation))
            reference = np.transpose(x, permutation)
        got = nu.evaluate_pair(fn, x)
        np.testing.assert_array_equal(got, reference)

    @pytest.mark.parametrize(
        "shape, target",
        [((3,), (2, 3)), ((2, 1), (2, 3)), ((), (2, 3))],
        ids=["vector", "column", "scalar"],
    )
    def test_broadcast_matches_numpy(self, shape, target):
        if shape == ():
            # A graph 0-d scalar: reduce an input, then broadcast the result.

            @evoxc.defn
            def fn(t):
                return evoxc.broadcast(evoxc.sum(t), target)

            source = _lin((4,))
            got = nu.evaluate_pair(fn, source)
            np.testing.assert_array_equal(got, np.broadcast_to(np.sum(source), target))
            return
        x = _lin(shape)
        got = nu.evaluate_pair(_unary(lambda t: evoxc.broadcast(t, target)), x)
        np.testing.assert_array_equal(got, np.broadcast_to(x, target))

    def test_select_broadcast_mask(self):
        pred = np.array([[True], [False], [True], [False]])
        x = _lin((4, 5))
        y = _lin((4, 5), -2.0, -1.0)
        got = nu.evaluate_pair(lambda p, a, b: evoxc.select(p, a, b), pred, x, y)
        np.testing.assert_array_equal(got, np.where(pred, x, y))

    def test_select_dtype_promotion(self):
        pred = np.array([True, False, True])
        on_true = np.array([1, 2, 3], np.int32)
        on_false = np.array([0.5, 1.5, 2.5], np.float32)
        got = nu.evaluate_pair(
            lambda p, a, b: evoxc.select(p, a, b), pred, on_true, on_false
        )
        assert got.dtype == np.result_type(on_true, on_false)
        np.testing.assert_array_equal(got, np.where(pred, on_true, on_false))


# --- 2. slice — evoxc lengths are SPANS (limit = start + length) ---


class TestSlice:
    @pytest.mark.parametrize(
        "start, lengths, strides",
        [
            ((0, 0), (2, 3), (1, 1)),
            ((1, 0), (3, 2), (1, 2)),
            ((0, 1), (4, 4), (2, 1)),
            ((1, 1), (2, 2), (1, 1)),
        ],
        ids=["origin", "strided", "row-stride2", "offset"],
    )
    def test_slice_start_length_stride(self, start, lengths, strides):
        x = _lin((4, 5))
        fn = _unary(lambda t: evoxc.slice(t, start, lengths, strides))
        got = nu.evaluate_pair(fn, x)
        # limit = start + length (SPAN semantics), NOT start + length * stride.
        reference = x[
            tuple(slice(s, s + length, stride)
                  for s, length, stride in zip(start, lengths, strides))
        ]
        np.testing.assert_array_equal(got, reference)
        # The span bounds the slice; a stride thins it to ceil(length / stride)
        # entries — the extent the graph declares (never length * stride).
        assert got.shape == reference.shape
        assert all(
            extent == -(length // -stride)
            for extent, length, stride in zip(got.shape, lengths, strides)
        )

    def test_slice_negative_start_counts_from_the_end(self):
        x = _lin((4, 5))
        got = nu.evaluate_pair(_unary(lambda t: evoxc.slice(t, -2, 2)), x)
        np.testing.assert_array_equal(got, x[2:4, 3:5])

    def test_slice_negative_stride_is_rejected(self):
        # The IR slice op takes positive strides only; the ops layer says so
        # explicitly (never a silent sign flip). A descending view is a flip.
        @evoxc.defn
        def fn(t):
            return evoxc.slice(t, 0, (2, 2), -1)

        x = _lin((4, 5))
        with pytest.raises(evoxc.ShapeError, match="strides must be positive"):
            evoxc.evaluate(fn, x, backend="numpy")

    def test_descending_slice_via_static_getitem(self):
        x = _lin((4, 5))

        @evoxc.defn
        def fn(t):
            return t[::-1, ::2]

        got = nu.evaluate_pair(fn, x)
        np.testing.assert_array_equal(got, x[::-1, ::2])

    def test_slice_rank3_per_axis(self):
        x = _normal((3, 4, 5), seed=5)
        got = nu.evaluate_pair(
            _unary(lambda t: evoxc.slice(t, (1, 0, 2), (2, 3, 2), (1, 2, 1))), x
        )
        np.testing.assert_array_equal(got, x[1:3, 0:3:2, 2:4])


# --- 3. gather ---


class TestGather:
    @pytest.mark.parametrize("axis", [0, 1])
    def test_gather_index_vector(self, axis):
        x = _lin((4, 5))
        indices = np.array([2, 0, 2, 1], np.int64)
        got = nu.evaluate_pair(
            lambda t, i: evoxc.gather(t, i, axis=axis), x, indices
        )
        np.testing.assert_array_equal(got, np.take(x, indices, axis=axis))

    @pytest.mark.parametrize(
        "x_shape, indices, axis",
        [
            ((3, 2), np.array([-1, 0, -3], np.int64), 0),  # negative indices
            ((2, 3), np.array([[0, 2], [1, 0]], np.int64), 1),  # multi-dim index
            ((3, 2), np.array(1, np.int64), 0),  # 0-d index (rank drop)
        ],
        ids=["negative", "multi-dim", "zero-dim"],
    )
    def test_gather_index_shapes(self, x_shape, indices, axis):
        x = _lin(x_shape)
        got = nu.evaluate_pair(
            lambda t, i: evoxc.gather(t, i, axis=axis), x, indices
        )
        reference = np.take(x, indices, axis=axis)
        assert got.shape == reference.shape
        np.testing.assert_array_equal(got, reference)

    def test_gather_non_identity_permutation(self):
        values = np.array([3.5, -1.0, 2.25, 0.0, 1.0], np.float32)
        permutation = np.argsort(values)
        got = nu.evaluate_pair(
            lambda t, i: evoxc.gather(t, i, axis=0), values, permutation
        )
        # A sort permutation turns the value vector into its sorted self.
        np.testing.assert_array_equal(got, values[permutation])
        np.testing.assert_array_equal(got, np.sort(values))


# --- 4. scatter — put-along-axis on a COPY (last write wins on duplicates) ---


def _scatter_reference(x, indices, updates, axis):
    """``np.put_along_axis`` on a copy, with indices rank-normalized like the
    kernel (``tests/backends/CONTEXT.md`` documented semantics)."""
    out = np.array(x, copy=True)
    index = np.asarray(indices)
    rank = x.ndim
    if index.ndim == 0:
        index = index.reshape((1,) * rank)
    elif index.ndim < rank:
        index = index.reshape(
            (1,) * axis + index.shape + (1,) * (rank - axis - 1)
        )
    np.put_along_axis(out, index, np.asarray(updates), axis=axis)
    return out


class TestScatter:
    @pytest.mark.parametrize("axis", [0, 1])
    def test_scatter_update_slices(self, axis):
        x = _lin((4, 2) if axis == 0 else (2, 4))
        indices = np.array([2, 0], np.int64)
        updates = _lin((2, 2), -2.0, -1.0)
        got = nu.evaluate_pair(
            lambda t, i, u: evoxc.scatter(t, i, u, axis=axis), x, indices, updates
        )
        np.testing.assert_array_equal(
            got, _scatter_reference(x, indices, updates, axis)
        )

    @pytest.mark.parametrize("axis", [0, 1])
    def test_scatter_duplicate_indices_match_the_numpy_backend(self, axis):
        # Duplicate indices are the interesting case; the reference backend
        # defines the semantics (put-along-axis: the LAST write wins), so the
        # assertion is parity FIRST and the np reference as confirmation.
        if axis == 0:
            x = np.zeros((3, 2), np.float32)
            indices = np.array([1, 1, 0], np.int64)
            updates = np.arange(6, dtype=np.float32).reshape(3, 2)
        else:
            x = np.zeros((2, 3), np.float32)
            indices = np.array([2, 2], np.int64)
            updates = np.array([[5.0, 6.0], [7.0, 8.0]], np.float32)
        got = nu.evaluate_pair(
            lambda t, i, u: evoxc.scatter(t, i, u, axis=axis), x, indices, updates
        )
        np.testing.assert_array_equal(
            got, _scatter_reference(x, indices, updates, axis)
        )

    def test_scatter_negative_indices(self):
        x = np.zeros((3, 2), np.float32)
        indices = np.array([-1, 0, -2], np.int64)
        updates = _lin((3, 2), 1.0, 2.0)
        got = nu.evaluate_pair(
            lambda t, i, u: evoxc.scatter(t, i, u, axis=0), x, indices, updates
        )
        np.testing.assert_array_equal(
            got, _scatter_reference(x, indices, updates, 0)
        )

    def test_scatter_never_mutates_its_input(self):
        x = np.zeros((2, 3), np.float32)
        before = x.copy()
        indices = np.array([2, 0], np.int64)
        updates = _lin((2, 2))
        nu.evaluate_pair(
            lambda t, i, u: evoxc.scatter(t, i, u, axis=1), x, indices, updates
        )
        np.testing.assert_array_equal(x, before)


# --- 5. concatenate / pad ---


class TestConcatenatePad:
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("count", [2, 3])
    def test_concatenate(self, axis, count):
        shape = (2, 3) if axis == 0 else (3, 2)
        arrays = [_lin(shape, -1.5 + i, 1.5 - i) for i in range(count)]

        @evoxc.defn
        def fn(*args):
            return evoxc.concatenate(tuple(args), axis=axis)

        got = nu.evaluate_pair(fn, *arrays)
        np.testing.assert_array_equal(got, np.concatenate(arrays, axis=axis))

    def test_concatenate_dtype_promotion(self):
        a = np.array([1, 2, 3], np.int32)
        b = np.array([0.5, 1.5, 2.5], np.float32)
        got = nu.evaluate_pair(lambda x, y: evoxc.concatenate((x, y), axis=0), a, b)
        assert got.dtype == np.result_type(a, b)
        np.testing.assert_array_equal(got, np.concatenate((a, b), axis=0))

    @pytest.mark.parametrize(
        "config, value",
        [
            (((1, 2), (0, 1)), 3.0),
            ((1, 2), -1.0),
            (((0, 0), (2, 1)), 0.5),
        ],
        ids=["pairs", "symmetric", "interior-axis"],
    )
    def test_pad_matches_numpy(self, config, value):
        x = _lin((2, 3))

        @evoxc.defn
        def fn(t):
            return evoxc.pad(t, config, value=value)

        got = nu.evaluate_pair(fn, x)
        pairs = [
            (entry, entry) if isinstance(entry, int) else entry for entry in config
        ]
        np.testing.assert_array_equal(
            got, np.pad(x, pairs, mode="constant", constant_values=value)
        )

    def test_pad_integer_dtype_and_rank3(self):
        x = _ints((2, 2, 2), lo=-2, hi=3)
        got = nu.evaluate_pair(
            _unary(lambda t: evoxc.pad(t, (0, (1, 0), 2), value=-5)), x
        )
        np.testing.assert_array_equal(
            got,
            np.pad(x, [(0, 0), (1, 0), (2, 2)], mode="constant", constant_values=-5),
        )


# --- 6. tril / triu ---


class TestTrilTriu:
    @pytest.mark.parametrize(
        "op, np_op", [(evoxc.tril, np.tril), (evoxc.triu, np.triu)]
    )
    @pytest.mark.parametrize("k", [0, 1, -1, 2])
    @pytest.mark.parametrize("shape", [(4, 4), (3, 5)])
    def test_triangular_mask(self, op, np_op, k, shape):
        x = _lin(shape)
        got = nu.evaluate_pair(_unary(lambda t: op(t, k=k)), x)
        np.testing.assert_array_equal(got, np_op(x, k=k))

    @pytest.mark.parametrize("op, np_op", [(evoxc.tril, np.tril), (evoxc.triu, np.triu)])
    def test_integer_dtype(self, op, np_op):
        x = _ints((3, 5), lo=-4, hi=5)
        got = nu.evaluate_pair(_unary(lambda t: op(t, k=-1)), x)
        assert got.dtype == np.int32
        np.testing.assert_array_equal(got, np_op(x, k=-1))

    def test_batched(self):
        x = _normal((2, 3, 3), seed=6)
        got = nu.evaluate_pair(_unary(evoxc.tril), x)
        np.testing.assert_array_equal(got, np.tril(x))


# --- 7. sort / argsort ---


_TIE_HEAVY = np.array(
    [[2, 1, 2, 1, 0], [1, 1, 2, 2, 2], [0, 2, 0, 1, 1]], np.int32
)


class TestSortArgsort:
    @pytest.mark.parametrize("axis", [-1, 0, 1])
    def test_sort_ascending(self, axis):
        x = _normal((3, 5), seed=7)
        got = nu.evaluate_pair(_unary(lambda t: evoxc.sort(t, axis=axis)), x)
        np.testing.assert_array_equal(got, np.sort(x, axis=axis))

    @pytest.mark.parametrize("axis", [-1, 0, 1])
    def test_sort_descending_is_the_stable_ascending_flip(self, axis):
        got = nu.evaluate_pair(
            _unary(lambda t: evoxc.sort(t, axis=axis, descending=True)), _TIE_HEAVY
        )
        np.testing.assert_array_equal(
            got, np.flip(np.sort(_TIE_HEAVY, axis=axis, kind="stable"), axis=axis)
        )

    def test_sort_stable_ties(self):
        got = nu.evaluate_pair(
            _unary(lambda t: evoxc.sort(t, axis=1, stable=True)), _TIE_HEAVY
        )
        np.testing.assert_array_equal(
            got, np.sort(_TIE_HEAVY, axis=1, kind="stable")
        )

    def test_sort_1d_tie_heavy(self):
        values = _TIE_HEAVY[0]
        got = nu.evaluate_pair(_unary(lambda t: evoxc.sort(t, stable=True)), values)
        assert got.shape == values.shape
        np.testing.assert_array_equal(
            got, np.sort(values, kind="stable")
        )

    def test_sort_bool(self):
        mask = np.array(
            [[True, False, True], [False, False, True], [True, True, False]]
        )
        got = nu.evaluate_pair(_unary(lambda t: evoxc.sort(t, axis=1)), mask)
        assert got.dtype == np.bool_
        np.testing.assert_array_equal(got, np.sort(mask, axis=1))

    @pytest.mark.parametrize("axis", [-1, 0, 1])
    @pytest.mark.parametrize("stable", [False, True])
    def test_argsort_ascending(self, axis, stable):
        got = nu.evaluate_pair(
            _unary(lambda t: evoxc.argsort(t, axis=axis, stable=stable)),
            _TIE_HEAVY,
        )
        assert got.dtype == np.int64
        if stable:
            # With stable=True the tie order is fully specified.
            np.testing.assert_array_equal(
                got, np.argsort(_TIE_HEAVY, axis=axis, kind="stable")
            )
        # Independent invariant: the permutation induces the sorted order.
        np.testing.assert_array_equal(
            np.take_along_axis(_TIE_HEAVY, got, axis=axis),
            np.sort(_TIE_HEAVY, axis=axis, kind="stable"),
        )

    @pytest.mark.parametrize("axis", [-1, 0, 1])
    def test_argsort_descending(self, axis):
        got = nu.evaluate_pair(
            _unary(lambda t: evoxc.argsort(t, axis=axis, descending=True, stable=True)),
            _TIE_HEAVY,
        )
        np.testing.assert_array_equal(
            got,
            np.flip(np.argsort(_TIE_HEAVY, axis=axis, kind="stable"), axis=axis),
        )
        # The permutation's values are the descending order.
        np.testing.assert_array_equal(
            np.take_along_axis(_TIE_HEAVY, got, axis=axis),
            np.flip(np.sort(_TIE_HEAVY, axis=axis, kind="stable"), axis=axis),
        )

    def test_argsort_1d_matches_numpy(self):
        values = np.array([3, 1, 2, 1, 3, 0], np.int64)
        got = nu.evaluate_pair(_unary(lambda t: evoxc.argsort(t, stable=True)), values)
        np.testing.assert_array_equal(got, np.argsort(values, kind="stable"))
        np.testing.assert_array_equal(values[got], np.sort(values))


# --- 8. tile / flip / roll / diag / diagonal / nan_to_num ---


class TestStructural:
    @pytest.mark.parametrize(
        "reps", [(2, 3), (1, 2), (2,), (3, 1, 2)], ids=["2x3", "1x2", "1d-reps", "promote"]
    )
    def test_tile(self, reps):
        x = _lin((2, 3)) if len(reps) <= 2 else _lin((3,))
        got = nu.evaluate_pair(_unary(lambda t: evoxc.tile(t, reps)), x)
        np.testing.assert_array_equal(got, np.tile(x, reps))

    @pytest.mark.parametrize("axes", [0, 1, -1, (0, 1), None],
                             ids=["axis0", "axis1", "axis-1", "both", "all"])
    def test_flip(self, axes):
        x = _normal((3, 4), seed=8)
        got = nu.evaluate_pair(_unary(lambda t: evoxc.flip(t, axes)), x)
        np.testing.assert_array_equal(got, np.flip(x, axis=axes))

    @pytest.mark.parametrize(
        "shift, axis",
        [(1, 0), (-1, 1), (-2, None), (3, None), ((1, -1), (0, 1))],
        ids=["axis0", "axis1-neg", "flat-neg", "flat-big", "per-axis"],
    )
    def test_roll(self, shift, axis):
        x = _normal((3, 4), seed=9)
        got = nu.evaluate_pair(_unary(lambda t: evoxc.roll(t, shift, axis=axis)), x)
        np.testing.assert_array_equal(got, np.roll(x, shift, axis=axis))

    @pytest.mark.parametrize(
        "shape",
        [(3,), (4, 4), (3, 5)],
        ids=["vector-to-matrix", "matrix-to-diagonal", "nonsquare-diagonal"],
    )
    def test_diag_round_trip(self, shape):
        x = (
            np.array([1.0, -2.0, 0.5], np.float32)
            if len(shape) == 1
            else _normal(shape, seed=10)
        )
        got = nu.evaluate_pair(_unary(evoxc.diag), x)
        np.testing.assert_array_equal(got, np.diag(x))
        assert got.shape == np.diag(x).shape

    @pytest.mark.parametrize("offset", [0, 1, -1])
    def test_diagonal_offset(self, offset):
        x = _normal((4, 4), seed=12)
        got = nu.evaluate_pair(_unary(lambda t: evoxc.diagonal(t, offset=offset)), x)
        np.testing.assert_array_equal(got, np.diagonal(x, offset=offset))

    def test_diagonal_explicit_axes(self):
        x = _normal((2, 3, 4), seed=13)
        got = nu.evaluate_pair(
            _unary(lambda t: evoxc.diagonal(t, offset=1, axis1=1, axis2=2)), x
        )
        np.testing.assert_array_equal(
            got, np.diagonal(x, offset=1, axis1=1, axis2=2)
        )

    def test_nan_to_num_explicit_replacements(self):
        x = np.array(
            [[np.nan, np.inf, -np.inf, 1.0], [0.0, -0.0, 2.0, np.nan]], np.float32
        )
        got = nu.evaluate_pair(
            _unary(lambda t: evoxc.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)),
            x,
        )
        np.testing.assert_array_equal(
            got, np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        )

    # BUG(evoxc): the native engine rejects a NULL attribute value while
    # decoding the `nan_to_num` op — the DEFAULT call (and any call that leaves
    # `posinf`/`neginf` at their `None` default) fails at `lower()` with
    # `VerificationError: op 'nan_to_num': attribute "posinf" must be a number
    # or null, got null`, whereas the numpy reference backend accepts it.
    # Minimal repro:
    #   evoxc.evaluate(lambda t: evoxc.nan_to_num(t),
    #                  np.array([np.nan, np.inf], np.float32), backend="native")
    # The assertions encode the DOCUMENTED contract (numpy's nan_to_num
    # defaults: NaN -> 0.0, +inf -> dtype max, -inf -> dtype min), so this pin
    # XPASSes — failing loudly — the day the native side is fixed.
    @pytest.mark.xfail(
        strict=True,
        reason="BUG(evoxc): native engine rejects a null `posinf`/`neginf` "
        "attribute, so default nan_to_num fails at lower()",
    )
    @pytest.mark.parametrize("kwargs", [{}, {"nan": 2.0}], ids=["defaults", "nan-only"])
    def test_nan_to_num_default_infinities_match_numpy(self, kwargs):
        x = np.array([[np.nan, np.inf, -np.inf, 1.0]], np.float32)
        got = nu.evaluate_pair(_unary(lambda t: evoxc.nan_to_num(t, **kwargs)), x)
        np.testing.assert_array_equal(got, np.nan_to_num(x, **kwargs))


# --- 9. dot / matmul ---


class TestDot:
    @pytest.mark.parametrize(
        "a_shape, b_shape, dtype",
        [
            ((3, 4), (4, 2), np.float32),
            ((2, 3, 4), (2, 4, 5), np.float32),
            ((2, 3, 4), (4, 5), np.float32),
            ((3, 4), (2, 4, 5), np.float32),
            ((1, 3, 4), (2, 4, 5), np.float32),
            ((3, 4), (4, 2), np.float64),
        ],
        ids=[
            "rank2",
            "batched",
            "broadcast-batch",
            "lhs-rank2-rhs-rank3",
            "size1-batch",
            "float64",
        ],
    )
    def test_dot_matches_numpy_matmul(self, a_shape, b_shape, dtype):
        a = _normal(a_shape, seed=14, dtype=dtype)
        b = _normal(b_shape, seed=15, dtype=dtype)
        got = nu.evaluate_pair(lambda x, y: evoxc.dot(x, y), a, b)
        assert got.dtype == dtype
        assert got.shape == np.matmul(a, b).shape
        nu.assert_parity(got, np.matmul(a, b), label="dot")

    def test_dot_integer_is_exact(self):
        a = _ints((3, 4), lo=-3, hi=4)
        b = _ints((4, 2), lo=-3, hi=4)
        got = nu.evaluate_pair(lambda x, y: evoxc.dot(x, y), a, b)
        assert got.dtype == np.int32
        np.testing.assert_array_equal(got, np.matmul(a, b))

    def test_matmul_sugar_equals_dot(self):
        a = _normal((2, 3, 4), seed=26)
        b = _normal((2, 4, 5), seed=27)
        got_matmul = nu.evaluate_pair(lambda x, y: evoxc.matmul(x, y), a, b)
        got_dot = nu.evaluate_pair(lambda x, y: evoxc.dot(x, y), a, b)
        np.testing.assert_array_equal(got_matmul, got_dot)
        np.testing.assert_array_equal(got_matmul, np.matmul(a, b))


# --- 10. conv (cross-correlation, NCHW; padding per spatial axis) ---


class TestConv:
    @pytest.mark.parametrize(
        "x_shape, w_shape, kwargs",
        [
            ((2, 1, 6), (1, 1, 3), {"padding": ((1, 0),)}),
            ((2, 2, 8), (3, 2, 3), {"strides": 2}),
            ((1, 1, 5, 5), (1, 1, 3, 3), {"padding": ((1, 0), (0, 1))}),
            ((2, 2, 4, 4), (3, 2, 2, 2), {}),
        ],
        ids=["1d-padded", "1d-strided", "2d-padded", "2d-multichannel"],
    )
    def test_conv_matches_a_hand_written_cross_correlation(
        self, x_shape, w_shape, kwargs
    ):
        x = _normal(x_shape, seed=28)
        w = _normal(w_shape, seed=29)
        got = nu.evaluate_pair(lambda a, b: evoxc.conv(a, b, **kwargs), x, w)
        reference = _cross_correlate(x, w, **kwargs)
        assert got.shape == reference.shape
        nu.assert_parity(got, reference, label="conv")

    def test_conv_same_padding(self):
        # SAME on (1, 2, 7) with a 3-wide kernel: out = ceil(7 / 1) = 7, so the
        # total pad is (7 - 1) + 3 - 7 = 2 ⇒ one zero on each side.
        x = _normal((1, 2, 7), seed=30)
        w = _normal((3, 2, 3), seed=31)
        got = nu.evaluate_pair(lambda a, b: evoxc.conv(a, b, padding="SAME"), x, w)
        assert got.shape == (1, 3, 7)
        nu.assert_parity(got, _cross_correlate(x, w, padding=(1, 1)), label="conv SAME")


# --- 11. solve ---


_POSITIVE_DEFINITE = np.array(
    [[4.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 2.0]], np.float64
)


class TestSolve:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_solve_3x3_matrix_rhs(self, dtype):
        a = _POSITIVE_DEFINITE.astype(dtype)
        b = _normal((3, 2), seed=38, dtype=dtype)
        got = nu.evaluate_pair(lambda x, r: evoxc.solve(x, r), a, b)
        assert got.dtype == dtype and got.shape == (3, 2)
        nu.assert_parity(got, np.linalg.solve(a, b), label="solve")
        # Residual pin: A @ x must reproduce b within the dtype's budget.
        rtol, atol = nu.dtype_tolerance(dtype)
        np.testing.assert_allclose(a @ got, b, rtol=rtol, atol=atol)

    def test_solve_batched(self):
        # evoxc requires the rhs to be (..., n, k) for a batched `a`.
        a = np.stack([_POSITIVE_DEFINITE, _POSITIVE_DEFINITE + np.eye(3)])
        b = _normal((2, 3, 2), seed=40, dtype=np.float64)
        got = nu.evaluate_pair(lambda x, r: evoxc.solve(x, r), a, b)
        assert got.shape == (2, 3, 2)
        nu.assert_parity(got, np.linalg.solve(a, b), label="solve batched")
        np.testing.assert_allclose(
            np.matmul(a, got), b, rtol=1e-12, atol=1e-13
        )


# --- 12. Combined structure chain staged ONCE and reused for two input sets ---


def _structure_chain():
    @evoxc.defn
    def fn(a, b):
        ordered = evoxc.sort(a, axis=-1)  # (3, 4) row-sorted values
        merged = evoxc.add(ordered, b)  # (3, 4)
        # reshape AFTER a transpose: rank-2 → rank-1 flat view.
        flat = evoxc.reshape(evoxc.transpose(merged), (-1,))  # (12,)
        # gather AFTER sort, indexed by argsort's permutation of the flat view.
        picked = evoxc.gather(flat, evoxc.argsort(flat, axis=0), axis=0)
        # concatenate of reductions (4 + 3 entries ⇒ one axis-0 join).
        head = evoxc.concatenate(
            (
                evoxc.reduce_sum(merged, axes=0),
                evoxc.reduce_sum(merged, axes=1),
            ),
            axis=0,
        )
        return picked, head

    return fn


class TestStructureChainExecutableReuse:
    def test_one_executable_two_input_sets(self):
        fn = _structure_chain()
        first = (_normal((3, 4), seed=41), _normal((3, 4), seed=42))
        second = (_normal((3, 4), seed=43), _normal((3, 4), seed=44))

        executable = nu.build_native(fn, *nu.specs_for(*first))
        assert executable.backend == "native"

        for label, (a, b) in (("first", first), ("second", second)):
            got_picked, got_head = nu.run_native(executable, a, b)
            expected = nu.evaluate_numpy(fn, a, b)
            nu.assert_tree_parity((got_picked, got_head), expected, label=label)

            # Independent formulas: the argsort∘gather round trip is the sorted
            # array itself, so picked = sort(reshape(transpose(sort(a) + b))).
            merged = np.sort(a, axis=-1) + b
            np.testing.assert_allclose(
                got_picked, np.sort(merged.T.reshape(-1)), rtol=1e-5, atol=1e-6
            )
            np.testing.assert_allclose(
                got_head,
                np.concatenate((merged.sum(axis=0), merged.sum(axis=1))),
                rtol=1e-5,
                atol=1e-6,
            )

    def test_chain_shapes_and_dtypes(self):
        fn = _structure_chain()
        a = _normal((3, 4), seed=45)
        b = _normal((3, 4), seed=46)
        got_picked, got_head = nu.evaluate_pair(fn, a, b)
        assert got_picked.shape == (12,) and got_picked.dtype == np.float32
        assert got_head.shape == (7,) and got_head.dtype == np.float32
