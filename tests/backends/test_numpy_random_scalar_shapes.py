"""Rank-0 (``shape=()``) random sampling on the numpy interpreter (+ native parity).

Contract under test (``evoxc/backends/numpy/kernels/CONTEXT.md``, "Scalar-shape
contract (binding)"): a kernel that COMPUTES its result must hand
``core.Tensor`` an ``np.ndarray``, so a ``shape=()`` (rank-0) op normalizes with
``np.asarray(...)``. numpy 2.x 0-d arithmetic returns a bare numpy SCALAR, and a
scalar exposes ``.shape`` / ``.dtype`` / ``.device`` (``'cpu'``), so without the
normalization it falls into ``core.Tensor``'s device-payload branch and raises
``TypeError: payload.device must be a core.Device or None, got str``. The four
computed rank-0 cases are ``random_uniform``, ``random_randint`` and BOTH
branches of ``random_normal`` (the f32 Box–Muller fast path and the exact f64
math that f64/f16 take); ``random_permutation`` / ``random_multinomial`` /
``random_key_mix`` already build real arrays (``permutation(n=0)`` is an empty
``(0,)`` array, NEVER a scalar) and are pinned here for that reason.

What is asserted:
* a ``shape=()`` draw is a 0-d ``core.Tensor`` on ``Device("cpu", 0)`` whose
  ``.numpy()`` is a genuine 0-d ``np.ndarray`` of the declared dtype;
* the scalar draw equals element 0 of the ``(1,)`` draw from the same key —
  the shape only changes how many stream values are materialized, not the
  stream itself (true for all six ops under all three algorithms);
* numpy-vs-native parity of the scalar draw: BIT-EXACT (``np.array_equal``)
  for ``uniform``/``randint``/``normal`` in f16 and f64, and for f32 ``normal``
  under the two algorithms whose f32 Box–Muller path is bit-exact against the
  numpy kernel; the ``philox4x32_10`` f32 ``normal`` leg uses the DOCUMENTED
  fast-path budget ``rtol=1e-6`` (see ``test_iree_normal_f32_fastpath_parity``
  and ``evoxc/ops/random.py``: "maxdiff ~1e-6 vs the f64 numpy kernel") so a
  libm ``log``/``cos`` drift can never flake it — it measures ~0.0 here;
* ``permutation`` at ``n=0`` / ``n=1`` numpy-vs-native, exact.

Keys are graph VALUES, so every key is created INSIDE the traced function
(``evoxc.random.key(seed, algorithm=…)`` builds a graph Constant there); a
concrete key ``Tensor`` cannot be a trace input. CPU only, fixed seed, no
timing, no network. The native legs skip when the extension is not built.
"""
from __future__ import annotations

import numpy as np
import pytest

import evoxc
from evoxc import core

from tests.backends import _native_utils as nu

#: The three canonical algorithms (``evoxc.ir.op_defs.random.ALGORITHMS``).
ALGORITHMS = ("splitmix64", "threefry2x32", "philox4x32_10")

#: Seed of every key in this module (deterministic, stateless draws).
SEED = 11

#: The f32 ``normal`` Box–Muller fast-path budget (documented drift ~1e-6).
F32_NORMAL_TOL = 1e-6

#: ``(kind, dtype, params)`` — the rank-0 cases the scalar-shape contract
#: covers: three float widths for uniform/normal plus the int32/int64
#: ``randint`` spans (the int64 high is past 2**32, so the float scaling is
#: exercised where a 32-bit counter would have wrapped).
SCALAR_CASES = [
    pytest.param("uniform", "float16", dict(low=-1.0, high=2.0), id="uniform-f16"),
    pytest.param("uniform", "float32", dict(low=-1.0, high=2.0), id="uniform-f32"),
    pytest.param("uniform", "float64", dict(low=-1.0, high=2.0), id="uniform-f64"),
    pytest.param("normal", "float16", dict(mean=1.0, std=2.0), id="normal-f16"),
    pytest.param("normal", "float32", dict(mean=1.0, std=2.0), id="normal-f32"),
    pytest.param("normal", "float64", dict(mean=1.0, std=2.0), id="normal-f64"),
    pytest.param("randint", "int32", dict(low=-7, high=13), id="randint-i32"),
    pytest.param(
        "randint", "int64", dict(low=0, high=2**40), id="randint-i64-wide"
    ),
]


def _sampler(kind, algorithm, dtype, shape, **params):
    """A ``defn`` drawing one ``shape``-shaped sample from a key of ``algorithm``."""

    @evoxc.defn
    def sample():
        key = evoxc.random.key(SEED, algorithm=algorithm)
        if kind == "uniform":
            return evoxc.random.uniform(key, shape, dtype=dtype, **params)
        if kind == "normal":
            return evoxc.random.normal(key, shape, dtype=dtype, **params)
        return evoxc.random.randint(key, shape, dtype=dtype, **params)

    return sample


def _draw(kind, algorithm, dtype, shape, backend="numpy", **params):
    """Evaluate the rank-0 (or length-1) sample and return the ``Tensor``."""
    fn = _sampler(kind, algorithm, dtype, shape, **params)
    return evoxc.evaluate(fn, backend=backend)


# ---------------------------------------------------------------------------
# the scalar-shape contract itself (numpy backend — no extension required)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("kind, dtype, params", SCALAR_CASES)
def test_rank0_draw_is_zero_d_cpu_tensor(algorithm, kind, dtype, params):
    """``shape=()`` returns a 0-d ``core.Tensor`` on cpu:0 — never a numpy scalar."""
    out = _draw(kind, algorithm, dtype, (), **params)

    assert isinstance(out, core.Tensor), (
        f"{kind} shape=() returned {type(out).__name__}, not a core.Tensor "
        "(a bare numpy scalar would have hit core.Tensor's device-payload branch)"
    )
    assert out.shape == (), f"{kind} shape=() produced shape {out.shape}"
    assert out.dtype == np.dtype(dtype), f"{kind} dtype {out.dtype} != {dtype}"
    assert out.device == core.Device("cpu", 0), (
        f"{kind} shape=() landed on {out.device}, expected cpu:0"
    )

    host = out.numpy()
    # The regression this pins: without ``np.asarray`` the payload was a numpy
    # SCALAR (``TypeError: payload.device ... got str`` at wrap time); a 0-d
    # ndarray is the only accepted shape here.
    assert type(host) is np.ndarray, (
        f"{kind} shape=() .numpy() returned {type(host).__name__}, not an ndarray"
    )
    assert host.shape == () and host.ndim == 0, f"{kind} .numpy() shape {host.shape}"
    assert host.dtype == np.dtype(dtype)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("kind, dtype, params", SCALAR_CASES)
def test_rank0_draw_matches_first_element_of_length_one(
    algorithm, kind, dtype, params
):
    """A scalar draw is stream value 0 — identical to ``(1,)``[0] from the same key."""
    scalar = _draw(kind, algorithm, dtype, (), **params)
    vector = _draw(kind, algorithm, dtype, (1,), **params)

    assert vector.shape == (1,)
    assert scalar.shape == ()
    scalar_host = np.asarray(scalar.numpy())
    vector_host = np.asarray(vector.numpy())
    assert scalar_host.shape == ()
    assert vector_host.shape == (1,)
    # The dtype rules are shape-independent, so element 0 is the same value.
    assert np.array_equal(scalar_host, vector_host[0]), (
        f"{kind} {algorithm} {dtype}: scalar draw {scalar_host!r} != (1,)[0] "
        f"{vector_host[0]!r}"
    )


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_rank0_draw_is_deterministic_across_runs(algorithm):
    """A scalar draw is stateless: the same graph re-run is bit-identical."""
    executable = evoxc.build(
        _sampler("uniform", algorithm, "float32", (), low=-1.0, high=2.0),
        backend="numpy",
    )
    first = evoxc.run(executable).numpy()
    second = evoxc.run(executable).numpy()
    assert first.shape == ()
    assert np.array_equal(np.asarray(first), np.asarray(second))


# ---------------------------------------------------------------------------
# permutation at the small-n boundary: real (empty) arrays, never scalars
# ---------------------------------------------------------------------------
def _permutation_sampler(n, algorithm="splitmix64"):
    """The permutation graph (``n`` is static, so one graph per ``n``)."""

    @evoxc.defn
    def permute():
        key = evoxc.random.key(SEED, algorithm=algorithm)
        return evoxc.random.permutation(key, n)

    return permute


def _permutation(n, algorithm="splitmix64"):
    return evoxc.evaluate(_permutation_sampler(n, algorithm), backend="numpy")


@pytest.mark.parametrize("n, expected", [(0, []), (1, [0])], ids=["n0", "n1"])
def test_permutation_small_n_is_a_real_array(n, expected):
    """``permutation(n=0)`` is an EMPTY ``(0,)`` array and ``n=1`` is ``[0]``."""
    out = _permutation(n)
    host = out.numpy()
    assert type(host) is np.ndarray, (
        f"permutation(n={n}) .numpy() returned {type(host).__name__}"
    )
    assert out.shape == (len(expected),)
    assert out.device == core.Device("cpu", 0)
    assert host.dtype == np.int32
    assert host.tolist() == expected


# ---------------------------------------------------------------------------
# numpy-vs-native parity of the rank-0 draw
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not nu.ENGINE_AVAILABLE, reason=nu.SKIP_REASON)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("kind, dtype, params", SCALAR_CASES)
def test_rank0_draw_matches_native(algorithm, kind, dtype, params):
    """The native engine's scalar draw agrees with the numpy reference."""
    reference = _draw(kind, algorithm, dtype, (), **params)
    native = _draw(kind, algorithm, dtype, (), backend="native", **params)

    assert isinstance(native, core.Tensor)
    assert native.shape == reference.shape == ()
    assert native.dtype == reference.dtype

    got = np.asarray(native.numpy())
    expected = np.asarray(reference.numpy())
    assert got.shape == expected.shape == ()
    if kind == "normal" and dtype == "float32" and algorithm == "philox4x32_10":
        # Documented f32 Box–Muller fast-path budget (measured drift ~0.0 here,
        # so this is pure headroom against libm log/cos, never a widened assert).
        np.testing.assert_allclose(
            got,
            expected,
            rtol=F32_NORMAL_TOL,
            atol=0.0,
            err_msg=f"{kind} {algorithm} {dtype}: native disagrees with numpy",
        )
    else:
        np.testing.assert_array_equal(
            got,
            expected,
            err_msg=f"{kind} {algorithm} {dtype}: native is not bit-exact with numpy",
        )


@pytest.mark.skipif(not nu.ENGINE_AVAILABLE, reason=nu.SKIP_REASON)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("n", [0, 1])
def test_permutation_small_n_matches_native(algorithm, n):
    """``permutation(n=0)`` / ``permutation(n=1)`` are exact numpy-vs-native."""
    sampler = _permutation_sampler(n, algorithm)
    reference = evoxc.evaluate(sampler, backend="numpy")
    native = evoxc.evaluate(sampler, backend="native")

    assert native.shape == reference.shape
    assert native.dtype == reference.dtype
    np.testing.assert_array_equal(
        np.asarray(native.numpy()),
        np.asarray(reference.numpy()),
        err_msg=f"permutation(n={n}) {algorithm}: native is not bit-exact with numpy",
    )
