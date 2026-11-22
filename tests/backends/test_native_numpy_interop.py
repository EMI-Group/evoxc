"""Native backend ↔ NumPy interop — the bidirectional host-data contract.

The ``native`` backend executes graphs on the in-process Rust CPU engine
(``evoxc-cpu``) over HOST numpy data: inputs are ``core.Tensor``s backed by
numpy arrays that ``evoxc.from_numpy`` wraps zero-copy, and outputs are fresh
writable C-contiguous numpy arrays wrapped back in ``core.Tensor``s (see
``../../evoxc/backends/native/CONTEXT.md``). This file pins that boundary
through the PUBLIC API only — ``evoxc.from_numpy`` → ``evoxc.build`` /
``evoxc.run`` (and the shared ``_native_utils`` staging helpers) with
``backend="native"`` — never through engine internals:

1. Every dtype the backend declares (bool, int8/16/32/64, uint8/16/32/64,
   float16/32/64, complex64/complex128) round-trips with its dtype and shape
   intact, and the result is a FRESH, writable, C-contiguous, owning array
   that matches both an explicit ``np.*`` reference and the numpy interpreter.
2. Memory LAYOUT is never silently misread: a Fortran-order array, a
   transposed view and a negatively-strided slice each give the same values as
   their ``np.ascontiguousarray`` copy, and ``run`` never mutates the caller's
   buffer (including the base of a view).
3. Read-only inputs (``arr.setflags(write=False)``) work, and stay read-only.
4. 0-d and zero-length arrays round-trip (shape, dtype and values); the
   rank-2 zero-length case is a strict-xfail ``BUG(evoxc)`` pin — the native
   marshalling cannot take a byte view of an empty higher-rank array (see
   ``TestScalarAndEmpty::test_zero_length_matrix_round_trips`` and the
   ``Known Issues`` entry in ``./CONTEXT.md``).
5. No aliasing: mutating an output array never reaches the input, and two
   consecutive runs produce identical results.
6. A two-call chain — the output of one native ``run`` fed back in through
   ``evoxc.from_numpy`` / the ``core.Tensor`` itself / ``evoxc.bind`` —
   equals a single numpy pass of the equivalent graph.

The numpy interpreter is the reference throughout (``nu.evaluate_pair`` /
``nu.assert_parity``: exact dtype/shape agreement, a per-width float budget
otherwise), and every layout/dtype case is ALSO pinned against an explicit
``np.*`` computation so the numbers — not merely backend agreement — are
fixed. The engine is OPTIONAL: the module-level gate below skips the whole
file when ``evoxc._evoxc_native`` is absent (the missing-extension error
contract lives in ``test_native_backend_missing_engine.py``).
Conventions: fixed values (no RNG), tiny shapes, CPU only, no network, no
timing.
"""
from __future__ import annotations

import numpy as np
import pytest

import evoxc
from evoxc import core
from tests.backends import _native_utils as nu

pytestmark = pytest.mark.skipif(not nu.ENGINE_AVAILABLE, reason=nu.SKIP_REASON)

#: Every dtype the native backend declares (``engine.SUPPORTED_DTYPES``), in a
#: readable order — pinned here so a dropped dtype is a test failure.
DTYPES = (
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

# ---------------------------------------------------------------------------
# Graphs under test (all value-preserving or simply verifiable by hand)
# ---------------------------------------------------------------------------


@evoxc.defn
def _reshape_3x2(x):
    """``(2, 3)`` → ``(3, 2)``: a pure relabel, so values must be untouched."""
    return evoxc.reshape(x, (3, 2))


@evoxc.defn
def _double(x):
    """Elementwise ``x + x`` — every element is read once, in storage order."""
    return evoxc.add(x, x)


@evoxc.defn
def _quadruple(x):
    """``double(double(x))`` — the single-pass reference for the run chain."""
    doubled = evoxc.add(x, x)
    return evoxc.add(doubled, doubled)


@evoxc.defn
def _reshape_scalar(x):
    """Reshape to rank 0 (one element, shape ``()``)."""
    return evoxc.reshape(x, ())


@evoxc.defn
def _reshape_empty_1d(x):
    """Reshape a zero-length vector to itself."""
    return evoxc.reshape(x, (0,))


@evoxc.defn
def _reshape_empty_2d(x):
    """Reshape a zero-length matrix to itself."""
    return evoxc.reshape(x, (0, 3))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _host_values(dtype, shape=(2, 3)):
    """Deterministic small host values of ``dtype`` (no RNG).

    Values stay inside a narrow range for every dtype (no integer overflow in
    ``_double``, no f16 precision surprises) and complex values carry a
    non-zero imaginary part so a dropped imaginary lane cannot pass.
    """
    dtype = np.dtype(dtype)
    count = int(np.prod(shape))
    if dtype.kind == "b":
        values = np.arange(count) % 2 == 0
    elif dtype.kind == "u":
        values = np.arange(count) % 5
    elif dtype.kind == "i":
        values = np.arange(count) % 7 - 3
    elif dtype.kind == "c":
        values = np.linspace(-1.5, 1.5, count) + 1j * (np.arange(count) % 3 - 1)
    else:
        values = np.linspace(-1.5, 1.5, count)
    return np.ascontiguousarray(values.reshape(shape).astype(dtype, copy=False))


def _matrix(shape=(4, 3), *, count=None, dtype=np.float32):
    """A deterministic float32 matrix with distinct, ordered entries."""
    if count is None:
        count = int(np.prod(shape))
    return np.linspace(-12.0, 12.0, count).reshape(shape).astype(dtype)


def _run(executable, array):
    """``evoxc.run`` with the explicit ``evoxc.from_numpy`` host-data wrap."""
    return evoxc.run(executable, evoxc.from_numpy(array))


def _spec_for(array, *, name=None):
    """The ``TensorSpec`` matching a concrete host array."""
    return core.TensorSpec(tuple(np.shape(array)), np.asarray(array).dtype, name=name)


def _assert_fresh_host_array(array, reference):
    """A native output array is fresh, writable, C-contiguous and unaliased."""
    assert isinstance(array, np.ndarray), f"expected an ndarray, got {type(array)}"
    assert array.flags.c_contiguous, "a native output array must be C-contiguous"
    assert array.flags.writeable, "a native output array must be writable"
    assert array.flags.owndata, "a native output array must own its bytes"
    assert not np.shares_memory(array, reference), (
        "a native output array must not alias the caller's input buffer"
    )


def _snapshot(array):
    """A value snapshot of ``array`` plus its base buffer (when it has one)."""
    base = array.base
    return array.copy(), None if base is None else np.array(base, copy=True)


def _assert_unchanged(array, snapshot):
    """The caller's array — and the base buffer behind a view — is untouched."""
    before, base_before = snapshot
    np.testing.assert_array_equal(np.asarray(array), before)
    if base_before is not None:
        np.testing.assert_array_equal(np.asarray(array.base), base_before)


# ---------------------------------------------------------------------------
# 1. Every dtype round-trips
# ---------------------------------------------------------------------------


class TestDtypeRoundTrip:
    @pytest.mark.parametrize("dtype", DTYPES, ids=[np.dtype(d).name for d in DTYPES])
    def test_every_dtype_round_trips(self, dtype):
        dtype = np.dtype(dtype)
        values = _host_values(dtype)
        executable = nu.build_native(_reshape_3x2, _spec_for(values))

        out = _run(executable, values)

        assert isinstance(out, core.Tensor)
        assert out.dtype == dtype
        assert out.shape == (3, 2)
        assert out.device == core.Device("cpu", 0)

        array = out.numpy()
        _assert_fresh_host_array(array, values)
        expected = np.reshape(values, (3, 2))
        assert array.dtype == dtype
        assert array.shape == (3, 2)
        # A reshape never touches the values: pin them EXACTLY.
        np.testing.assert_array_equal(array, expected)

        # ... and the numpy interpreter produces the identical result.
        nu.assert_parity(nu.evaluate_pair(_reshape_3x2, values), expected)

    @pytest.mark.parametrize("dtype", DTYPES, ids=[np.dtype(d).name for d in DTYPES])
    def test_every_dtype_reads_back_through_from_numpy(self, dtype):
        """The host array handed in is the exact array ``from_numpy`` wrapped."""
        values = _host_values(np.dtype(dtype))
        tensor = evoxc.from_numpy(values)
        assert isinstance(tensor, core.Tensor)
        assert tensor.dtype == np.dtype(dtype)
        assert tensor.shape == values.shape
        assert tensor.device == core.Device("cpu", 0)
        # from_numpy is the documented zero-copy wrap of host data.
        assert np.shares_memory(tensor.numpy(), values)


# ---------------------------------------------------------------------------
# 2. Layout: non-contiguous / Fortran / negative-stride inputs are never
#    silently misread, and the caller's buffer is never written
# ---------------------------------------------------------------------------


class TestLayout:
    def _check(self, executable, view):
        """Run ``view`` and its C-contiguous copy; both equal ``2 * values``."""
        contiguous = np.ascontiguousarray(view)
        expected = contiguous * np.float32(2)

        out = _run(executable, view)
        array = out.numpy()
        _assert_fresh_host_array(array, view)
        np.testing.assert_array_equal(array, expected)

        # The C-contiguous copy of the same logical values agrees.
        copied = _run(executable, contiguous).numpy()
        np.testing.assert_array_equal(array, copied)

        # The numpy interpreter agrees on the very same view.
        nu.assert_parity(nu.evaluate_numpy(_double, view), expected)
        return array

    def test_fortran_order_array(self):
        values = _matrix()
        fortran = np.asfortranarray(values)
        assert fortran.flags.f_contiguous
        assert not fortran.flags.c_contiguous
        np.testing.assert_array_equal(fortran, values)

        snapshot = _snapshot(fortran)
        executable = nu.build_native(_double, _spec_for(fortran))
        array = self._check(executable, fortran)

        np.testing.assert_array_equal(array, values * np.float32(2))
        _assert_unchanged(fortran, snapshot)

    def test_transposed_view(self):
        backing = _matrix((2, 3, 4), count=24)
        view = backing.transpose(1, 0, 2)  # (3, 2, 4): neither C- nor F-order
        assert view.base is not None
        assert not view.flags.c_contiguous
        assert not view.flags.f_contiguous

        snapshot = _snapshot(view)
        executable = nu.build_native(_double, _spec_for(view))
        array = self._check(executable, view)

        np.testing.assert_array_equal(
            array, np.ascontiguousarray(view) * np.float32(2)
        )
        _assert_unchanged(view, snapshot)

    def test_negatively_strided_slice(self):
        backing = _matrix((4, 6), count=24)
        view = backing[::-1, ::2]  # (4, 3), negative stride on axis 0
        assert view.base is not None
        assert any(stride < 0 for stride in view.strides)
        assert not view.flags.c_contiguous

        snapshot = _snapshot(view)
        executable = nu.build_native(_double, _spec_for(view))
        array = self._check(executable, view)

        np.testing.assert_array_equal(
            array, np.ascontiguousarray(view) * np.float32(2)
        )
        _assert_unchanged(view, snapshot)

    def test_a_raw_ndarray_is_accepted_and_not_mutated(self):
        """``evoxc.run`` wraps a raw ndarray via ``from_numpy`` (documented)."""
        values = _matrix()
        snapshot = _snapshot(values)
        executable = nu.build_native(_double, _spec_for(values))

        out = evoxc.run(executable, values)  # no explicit from_numpy
        array = out.numpy()
        _assert_fresh_host_array(array, values)
        np.testing.assert_array_equal(array, values * np.float32(2))
        _assert_unchanged(values, snapshot)


# ---------------------------------------------------------------------------
# 3. Read-only inputs
# ---------------------------------------------------------------------------


class TestReadOnlyInput:
    def test_read_only_input_runs_and_stays_read_only(self):
        values = _matrix()
        values.setflags(write=False)
        assert not values.flags.writeable
        before = values.copy()

        executable = nu.build_native(_double, _spec_for(values))
        first = _run(executable, values).numpy()
        second = _run(executable, values).numpy()

        expected = before * np.float32(2)
        _assert_fresh_host_array(first, values)
        np.testing.assert_array_equal(first, expected)
        np.testing.assert_array_equal(second, expected)
        np.testing.assert_array_equal(values, before)
        assert not values.flags.writeable, "run must not flip the caller's flags"


# ---------------------------------------------------------------------------
# 4. 0-d and zero-length arrays
# ---------------------------------------------------------------------------


class TestScalarAndEmpty:
    def test_rank_zero_array_round_trips(self):
        values = np.array(2.5, dtype=np.float32)
        assert values.shape == ()

        executable = nu.build_native(_reshape_scalar, _spec_for(values))
        out = _run(executable, values)

        assert out.shape == ()
        assert out.dtype == np.dtype(np.float32)
        array = out.numpy()
        _assert_fresh_host_array(array, values)
        assert array.shape == ()
        assert array[()] == np.float32(2.5)
        nu.assert_parity(nu.evaluate_pair(_reshape_scalar, values), values)

    def test_zero_length_vector_round_trips(self):
        values = np.zeros((0,), dtype=np.int32)

        executable = nu.build_native(_reshape_empty_1d, _spec_for(values))
        out = _run(executable, values)

        assert out.shape == (0,)
        assert out.dtype == np.dtype(np.int32)
        array = out.numpy()
        assert array.shape == (0,)
        assert array.size == 0
        assert array.flags.c_contiguous
        assert array.flags.writeable
        nu.assert_parity(nu.evaluate_pair(_reshape_empty_1d, values), values)

    # BUG(evoxc): the native engine's array marshalling takes the byte view with
    # `memoryview.cast("B")`, which numpy refuses for ANY view whose SHAPE
    # contains a zero — so a rank >= 2 zero-length array fails on BOTH the input
    # path (`array_to_buffer`) and the output path (`buffer_to_array`), while the
    # numpy interpreter runs the same graph. Rank-0 and rank-1 empty arrays work
    # (pinned above). Minimal repro:
    #   np.zeros((0, 3), np.float32).data.cast("B")
    #     -> TypeError: memoryview: cannot cast view with zeros in shape or strides
    #   evoxc.evaluate(_reshape_empty_2d, np.zeros((0, 3), np.float32),
    #                  backend="native")
    #     -> TypeError: flat_inputs[0]: cannot take a byte view (...)
    #   (and a (0,) input reshaped to (0, 3) fails on the OUTPUT side:
    #    "run: output 0: memoryview: cannot cast view with zeros in shape or strides")
    # Delete this marker (and the matching Known Issues entry in
    # ``tests/backends/CONTEXT.md``) once ``native/evoxc-python/src/cpu.rs``
    # handles empty higher-rank views.
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "BUG(evoxc): native marshalling cannot byte-view a rank >= 2 "
            "zero-length array"
        ),
    )
    def test_zero_length_matrix_round_trips(self):
        values = np.zeros((0, 3), dtype=np.float64)

        executable = nu.build_native(_reshape_empty_2d, _spec_for(values))
        out = _run(executable, values)

        assert out.shape == (0, 3)
        assert out.dtype == np.dtype(np.float64)
        array = out.numpy()
        assert array.shape == (0, 3)
        assert array.size == 0
        assert array.flags.c_contiguous
        assert array.flags.writeable
        nu.assert_parity(nu.evaluate_pair(_reshape_empty_2d, values), values)


# ---------------------------------------------------------------------------
# 5. No aliasing between inputs and outputs
# ---------------------------------------------------------------------------


class TestNoAliasing:
    def test_mutating_an_output_never_reaches_the_input(self):
        values = _matrix()
        snapshot = _snapshot(values)
        executable = nu.build_native(_double, _spec_for(values))

        array = _run(executable, values).numpy()
        np.testing.assert_array_equal(array, values * np.float32(2))
        array.fill(-1)  # a writable output: the caller may do this

        _assert_unchanged(values, snapshot)
        # A later run is unaffected by the previous output's contents.
        again = _run(executable, values).numpy()
        np.testing.assert_array_equal(again, values * np.float32(2))
        assert not np.shares_memory(array, again)

    def test_two_consecutive_runs_are_identical(self):
        values = _matrix()
        executable = nu.build_native(_double, _spec_for(values))

        first = _run(executable, values)
        second = _run(executable, values)

        assert first is not second
        assert first.data is not second.data
        assert not np.shares_memory(first.numpy(), second.numpy())
        nu.assert_parity(first.numpy(), second.numpy())
        np.testing.assert_array_equal(first.numpy(), second.numpy())


# ---------------------------------------------------------------------------
# 6. Chained runs: one native output is the next native input
# ---------------------------------------------------------------------------


class TestChainedRuns:
    SPEC_NAME = "x"

    def _executable(self):
        values = _matrix()
        executable = nu.build_native(_double, _spec_for(values, name=self.SPEC_NAME))
        return values, executable

    def _expected(self, values):
        """The numpy single pass of the equivalent two-``_double`` graph."""
        return nu.evaluate_numpy(_quadruple, values)

    def test_output_feeds_the_next_run_through_from_numpy(self):
        values, executable = self._executable()
        expected = self._expected(values)
        np.testing.assert_array_equal(expected, values * np.float32(4))

        first = _run(executable, values)
        second = _run(executable, first.numpy())  # explicit host round trip
        third = evoxc.run(executable, evoxc.from_numpy(first.numpy()))

        np.testing.assert_array_equal(second.numpy(), expected)
        np.testing.assert_array_equal(third.numpy(), expected)
        # Feeding the output Tensor back in directly is equally valid.
        fourth = evoxc.run(executable, first)
        np.testing.assert_array_equal(fourth.numpy(), expected)

    def test_bind_supplies_the_first_output(self):
        values, executable = self._executable()
        expected = self._expected(values)

        first = _run(executable, values)
        bound = evoxc.bind(executable, **{self.SPEC_NAME: first})
        np.testing.assert_array_equal(evoxc.run(bound).numpy(), expected)
        # The bound chain is repeatable and never recompiles the graph.
        np.testing.assert_array_equal(evoxc.run(bound).numpy(), expected)
        np.testing.assert_array_equal(first.numpy(), values * np.float32(2))


# ---------------------------------------------------------------------------
# 7. The shared skip gate
# ---------------------------------------------------------------------------


def test_engine_gate_is_the_shared_helper_flag():
    """This module runs only where the extension built; the gate is `nu`'s.

    The skip-guard correctness pin: an absent extension skips every test in
    this file through the shared ``ENGINE_AVAILABLE``/``SKIP_REASON`` pair
    rather than failing collection or raising from a fixture.
    """
    assert nu.ENGINE_AVAILABLE is True
    assert isinstance(nu.SKIP_REASON, str) and nu.SKIP_REASON
