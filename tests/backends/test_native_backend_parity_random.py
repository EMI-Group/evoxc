"""Key-based functional RNG (``evoxc.random``) on the ``native`` backend.

Parity of the six ``random_*`` graph ops against the ``numpy`` reference
interpreter (``evoxc/backends/numpy/kernels/random.py`` is the word-stream
reference — see the "Key-based functional RNG" section of
``../../evoxc/CONTEXT.md`` and ``../../evoxc/ops/CONTEXT.md``).

Contract under test (``../../native/evoxc-cpu/CONTEXT.md``, "Random is
BIT-EXACT with ``../evoxc/backends/numpy/kernels/random.py`` for splitmix64,
threefry2x32 and philox4x32_10 (key handling, salt folding, counter layout,
uniform/randint/permutation/multinomial/normal, including the f32 Box–Muller
normal fast path)"): the Rust CPU engine implements all SIX random ops, so a
deferral here would be a real finding — it is only COMPILER backends that
defer ``random_multinomial``.

Exactness expectations, per that contract:

* ``random_uniform`` / ``random_randint`` / ``random_permutation`` /
  ``random_multinomial`` and f64 ``random_normal`` — BIT-EXACT vs numpy
  (asserted with ``np.testing.assert_array_equal``).
* f32 ``random_normal`` — the DOCUMENTED f32 Box–Muller fast-path budget
  ``RANDOM_NORMAL_TOL = dict(rtol=1e-4, atol=1e-5)``, taken verbatim from
  ``tests/backends/test_iree_random_algorithms_parity.py:99`` (matching
  ``test_iree_normal_f32_fastpath_parity.py:61`` and
  ``test_iree_emitters_parity.py:557``), which documents the measured drift
  as ~4.8e-7 (pure libm ``log``/``cos`` drift, ~20x headroom) and is the
  budget ``evoxc/ops/random.py`` describes as "a documented f32 Box–Muller
  fast path, maxdiff ~1e-6 vs the f64 numpy kernel" (see also
  ``../../evoxc/backends/numpy/CONTEXT.md`` and
  ``../../evoxc/backends/adapters/PER-ADAPTER-NOTES.md:55``). f32 ``normal``
  is still DETERMINISTIC: the same graph run twice is bit-identical.

Keys are graph VALUES: a concrete ``evoxc.random.key(7)`` Tensor cannot be a
trace input, so every key is created INSIDE the traced function (the
``key()`` creator builds a graph Constant there) — the pattern below.

CPU only, fixed seeds, tiny shapes, no timing, no network.

``random.split``/``split_n`` stamp the SplitMix64 golden-gamma salt (and its
multiples, masked to 64 bits) as an unsigned 64-bit attribute; the native
envelope carries those salts and decodes them back to the same words, so those
legs are bit-exact vs numpy like every other op.
"""
import numpy as np
import pytest

import evoxc
from tests.backends import _native_utils as nu

pytestmark = pytest.mark.skipif(not nu.ENGINE_AVAILABLE, reason=nu.SKIP_REASON)

#: The three canonical algorithms (``evoxc.ir.op_defs.random.ALGORITHMS``).
ALGORITHMS = ("splitmix64", "threefry2x32", "philox4x32_10")

#: Documented f32 ``normal`` Box–Muller fast-path budget — see the module
#: docstring for the source (``test_iree_random_algorithms_parity.py:99``;
#: measured drift ~4.8e-7, so this is ~20x headroom, never a widened assert).
RANDOM_NORMAL_TOL = dict(rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _exact_pair(fn, *args, label=""):
    """Evaluate on both backends; assert the arrays are BIT-IDENTICAL."""
    got = np.asarray(nu.evaluate_native(fn, *args))
    expected = np.asarray(nu.evaluate_numpy(fn, *args))
    where = f"{label}: " if label else ""
    assert got.dtype == expected.dtype, (
        f"{where}dtype {got.dtype} != reference {expected.dtype}"
    )
    assert got.shape == expected.shape, (
        f"{where}shape {got.shape} != reference {expected.shape}"
    )
    np.testing.assert_array_equal(
        got,
        expected,
        err_msg=f"{where}native backend is not bit-exact with numpy",
    )
    return got


def _key_layout(algorithm):
    """``(shape, dtype)`` of the algorithm's key — from the concrete creator."""
    key = evoxc.random.key(0, algorithm=algorithm)
    return tuple(key.shape), key.dtype


def _uniform(algorithm, shape, dtype, low=0.0, high=1.0):
    @evoxc.defn
    def uniform():
        key = evoxc.random.key(7, algorithm=algorithm)
        return evoxc.random.uniform(
            key, shape=shape, low=low, high=high, dtype=dtype
        )

    return uniform


def _randint(algorithm, shape, low, high, dtype):
    @evoxc.defn
    def randint():
        key = evoxc.random.key(7, algorithm=algorithm)
        return evoxc.random.randint(key, shape, low, high, dtype=dtype)

    return randint


def _permutation(algorithm, n):
    @evoxc.defn
    def permutation():
        key = evoxc.random.key(7, algorithm=algorithm)
        return evoxc.random.permutation(key, n)

    return permutation


def _normal(algorithm, shape, dtype, mean=0.0, std=1.0):
    @evoxc.defn
    def normal():
        key = evoxc.random.key(7, algorithm=algorithm)
        return evoxc.random.normal(
            key, shape=shape, mean=mean, std=std, dtype=dtype
        )

    return normal


def _split(algorithm):
    @evoxc.defn
    def split():
        key = evoxc.random.key(7, algorithm=algorithm)
        first, second = evoxc.random.split(key)
        return first, second

    return split


def _split_n(algorithm, n):
    @evoxc.defn
    def split_n():
        key = evoxc.random.key(7, algorithm=algorithm)
        return tuple(evoxc.random.split_n(key, n))

    return split_n


def _first_subkey_uniform(algorithm):
    @evoxc.defn
    def first_subkey_uniform():
        key = evoxc.random.key(7, algorithm=algorithm)
        first, _second = evoxc.random.split(key)
        return evoxc.random.uniform(first, shape=(8,), dtype="float32")

    return first_subkey_uniform


def _multinomial(algorithm, num_samples=5):
    @evoxc.defn
    def multinomial(probs):
        key = evoxc.random.key(7, algorithm=algorithm)
        return evoxc.random.multinomial(key, probs, num_samples)

    return multinomial


def _composition(algorithm):
    @evoxc.defn
    def composition():
        key = evoxc.random.key(7, algorithm=algorithm)
        uniform = evoxc.random.uniform(key, shape=(4,), dtype="float64")
        normal = evoxc.random.normal(key, shape=(4,), dtype="float64")
        product = evoxc.multiply(uniform, normal)
        return evoxc.add(product, 1.0), evoxc.reduce_sum(product, axes=None)

    return composition


# ---------------------------------------------------------------------------
# random_uniform — bit-exact (f32, f64, rank-2), determinism, executable reuse
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_uniform_float32_bit_exact(algorithm):
    out = _exact_pair(
        _uniform(algorithm, (8,), "float32"), label=f"uniform f32 {algorithm}"
    )
    assert out.dtype == np.dtype("float32")
    assert out.shape == (8,)
    # A unit-interval draw never leaves [0, 1) — true on BOTH backends, so the
    # bit-exact assertion above cannot be satisfied by degenerate garbage.
    assert ((out >= 0.0) & (out < 1.0)).all()


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_uniform_float64_bit_exact(algorithm):
    out = _exact_pair(
        _uniform(algorithm, (8,), "float64", low=-2.0, high=5.0),
        label=f"uniform f64 {algorithm}",
    )
    assert out.dtype == np.dtype("float64")
    assert ((out >= -2.0) & (out < 5.0)).all()


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_uniform_rank2_shape_bit_exact(algorithm):
    out = _exact_pair(
        _uniform(algorithm, (2, 3), "float32"), label=f"uniform (2,3) {algorithm}"
    )
    assert out.shape == (2, 3)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_uniform_cross_run_determinism(algorithm):
    """Two separate ``evoxc.evaluate`` calls on the same graph, same bytes."""
    fn = _uniform(algorithm, (8,), "float32")
    first = np.asarray(nu.evaluate_native(fn))
    second = np.asarray(nu.evaluate_native(fn))
    assert first.tobytes() == second.tobytes()
    # ... and it is the numpy stream, not just a stable wrong one.
    assert first.tobytes() == np.asarray(nu.evaluate_numpy(fn)).tobytes()


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_uniform_same_executable_two_runs(algorithm):
    """One compiled executable, two ``run`` calls ⇒ identical values."""
    fn = _uniform(algorithm, (8,), "float32")
    executable = nu.build_native(fn)
    first = np.asarray(nu.run_native(executable))
    second = np.asarray(nu.run_native(executable))
    assert first.tobytes() == second.tobytes()
    np.testing.assert_array_equal(first, np.asarray(nu.evaluate_numpy(fn)))


# ---------------------------------------------------------------------------
# random_randint — bit-exact, in range, int32 and int64
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize(
    "low, high, dtype",
    ((0, 10, "int32"), (-5, 5, "int32"), (0, 10, "int64"), (-5, 5, "int64")),
)
def test_randint_bit_exact(algorithm, low, high, dtype):
    out = _exact_pair(
        _randint(algorithm, (6,), low, high, dtype),
        label=f"randint[{low},{high}) {dtype} {algorithm}",
    )
    assert out.dtype == np.dtype(dtype)
    assert out.shape == (6,)
    # ``high`` is exclusive — every draw is a real value in the bound.
    assert ((out >= low) & (out < high)).all()


# ---------------------------------------------------------------------------
# random_permutation — bit-exact, a true permutation of arange(n)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_permutation_bit_exact(algorithm):
    out = _exact_pair(
        _permutation(algorithm, 8), label=f"permutation {algorithm}"
    )
    assert out.dtype == np.dtype("int32")
    assert out.shape == (8,)
    np.testing.assert_array_equal(np.sort(out), np.arange(8))


# ---------------------------------------------------------------------------
# random_normal — f64 bit-exact; f32 within the documented fast-path budget
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_normal_float64_bit_exact(algorithm):
    out = _exact_pair(
        _normal(algorithm, (6,), "float64"), label=f"normal f64 {algorithm}"
    )
    assert out.dtype == np.dtype("float64")
    assert out.shape == (6,)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("std", (1.0, 0.5))
def test_normal_float32_within_documented_budget(algorithm, std):
    """f32 output takes the documented Box–Muller fast path (~1 ulp drift).

    Budget: ``RANDOM_NORMAL_TOL`` — see the module docstring for its source
    and the measured ~4.8e-7 drift it documents.
    """
    got = np.asarray(
        nu.evaluate_native(_normal(algorithm, (8,), "float32", std=std))
    )
    expected = np.asarray(
        nu.evaluate_numpy(_normal(algorithm, (8,), "float32", std=std))
    )
    assert got.dtype == expected.dtype == np.dtype("float32")
    assert got.shape == expected.shape == (8,)
    np.testing.assert_allclose(
        got,
        expected,
        rtol=RANDOM_NORMAL_TOL["rtol"],
        atol=RANDOM_NORMAL_TOL["atol"],
        err_msg=(
            "f32 normal drifted past the documented Box–Muller fast-path "
            f"budget {RANDOM_NORMAL_TOL} ({algorithm}, std={std})"
        ),
    )


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_normal_float32_rank2_within_documented_budget(algorithm):
    got = np.asarray(
        nu.evaluate_native(_normal(algorithm, (4, 3), "float32", std=0.5))
    )
    expected = np.asarray(
        nu.evaluate_numpy(_normal(algorithm, (4, 3), "float32", std=0.5))
    )
    assert got.shape == expected.shape == (4, 3)
    np.testing.assert_allclose(
        got,
        expected,
        rtol=RANDOM_NORMAL_TOL["rtol"],
        atol=RANDOM_NORMAL_TOL["atol"],
    )


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_normal_float32_deterministic_across_runs(algorithm):
    """The fast path is a pure function: two separate runs, identical bytes."""
    fn = _normal(algorithm, (8,), "float32")
    first = np.asarray(nu.evaluate_native(fn))
    second = np.asarray(nu.evaluate_native(fn))
    assert first.tobytes() == second.tobytes()


# ---------------------------------------------------------------------------
# random_split / random_split_n — numpy-reference layout, native parity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_split_subkey_layout_matches_algorithm(algorithm):
    """``split`` yields 2 subkeys with the algorithm's declared layout.

    The layout is read back from the algorithm's own concrete key (the single
    source of truth: splitmix64 → rank-0 int64, threefry2x32 → (2,) int32,
    philox4x32_10 → (4,) int32) instead of being hardcoded — the numpy
    reference must agree with it.
    """
    key_shape, key_dtype = _key_layout(algorithm)
    subkeys = nu.evaluate_numpy(_split(algorithm))
    assert len(subkeys) == 2
    for index, subkey in enumerate(subkeys):
        subkey = np.asarray(subkey)
        assert subkey.shape == key_shape, (
            f"{algorithm} subkey {index} shape {subkey.shape} != key shape "
            f"{key_shape}"
        )
        assert subkey.dtype == np.dtype(key_dtype)
    # The two subkeys are decorrelated states, not the same key twice.
    assert np.asarray(subkeys[0]).tobytes() != np.asarray(subkeys[1]).tobytes()
    # Deterministic and stateless: a second evaluation draws the same pair.
    again = nu.evaluate_numpy(_split(algorithm))
    for index in range(2):
        np.testing.assert_array_equal(
            np.asarray(again[index]), np.asarray(subkeys[index])
        )


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_split_n_yields_n_subkeys(algorithm):
    key_shape, key_dtype = _key_layout(algorithm)
    three = nu.evaluate_numpy(_split_n(algorithm, 3))
    assert len(three) == 3
    for subkey in three:
        assert np.asarray(subkey).shape == key_shape
        assert np.asarray(subkey).dtype == np.dtype(key_dtype)
    # ``split_n(k, 3)[:2]`` is exactly ``split(k)``: salt 0 and the golden
    # salt, in that order (both are ``random_key_mix(key, salt)``).
    pair = nu.evaluate_numpy(_split(algorithm))
    for index in range(2):
        np.testing.assert_array_equal(
            np.asarray(three[index]), np.asarray(pair[index])
        )
    # n=0 is the empty split.
    assert nu.evaluate_numpy(_split_n(algorithm, 0)) == ()


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_split_matches_numpy_on_native(algorithm):
    got = nu.evaluate_native(_split(algorithm))
    expected = nu.evaluate_numpy(_split(algorithm))
    assert len(got) == len(expected) == 2
    for index, (g, e) in enumerate(zip(got, expected)):
        np.testing.assert_array_equal(
            np.asarray(g), np.asarray(e), err_msg=f"subkey {index} {algorithm}"
        )


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_first_subkey_sampling_matches_numpy_on_native(algorithm):
    fn = _first_subkey_uniform(algorithm)
    got = np.asarray(nu.evaluate_native(fn))
    expected = np.asarray(nu.evaluate_numpy(fn))
    np.testing.assert_array_equal(got, expected)


def _high_index_subkey_uniform(algorithm, n=8):
    """Sample from the LAST (high-index) subkey of ``split_n(k, n)``.

    ``split_n`` stamps ``i * GOLDEN mod 2**64`` salts; the high indices are
    the ones whose unmasked value used to overflow 64 bits, so this is the
    regression that proves the masked salt reaches the native engine intact.
    """

    @evoxc.defn
    def high_index_subkey_uniform():
        key = evoxc.random.key(7, algorithm=algorithm)
        subkeys = evoxc.random.split_n(key, n)
        return evoxc.random.uniform(subkeys[n - 1], shape=(8,), dtype="float32")

    return high_index_subkey_uniform


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_high_index_split_n_subkey_sampling_matches_numpy_on_native(algorithm):
    """``split_n(k, 8)``'s high-index subkey draws bit-identically on native."""
    fn = _high_index_subkey_uniform(algorithm)
    got = np.asarray(nu.evaluate_native(fn))
    expected = np.asarray(nu.evaluate_numpy(fn))
    assert got.shape == expected.shape == (8,)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_split_n_full_key_tree_matches_numpy_on_native(algorithm):
    """All 8 ``split_n`` subkeys agree with numpy bit-for-bit."""
    fn = _split_n(algorithm, 8)
    got = nu.evaluate_native(fn)
    expected = nu.evaluate_numpy(fn)
    assert len(got) == len(expected) == 8
    for index, (g, e) in enumerate(zip(got, expected)):
        np.testing.assert_array_equal(
            np.asarray(g), np.asarray(e), err_msg=f"subkey {index} {algorithm}"
        )


# ---------------------------------------------------------------------------
# random_multinomial — the native engine implements it (only COMPILER
# backends defer it), so this is bit-exact, never a BackendError
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_multinomial_bit_exact(algorithm):
    probs = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    got = np.asarray(
        nu.evaluate_native(_multinomial(algorithm), probs)
    )
    expected = np.asarray(
        nu.evaluate_numpy(_multinomial(algorithm), probs)
    )
    assert got.dtype == expected.dtype == np.dtype("int32")
    assert got.shape == expected.shape == (5,)
    np.testing.assert_array_equal(got, expected)
    # Drawn INDICES over a 3-element distribution.
    assert ((got >= 0) & (got < 3)).all()


# ---------------------------------------------------------------------------
# cross-algorithm independence (sanity, not a parity pin)
# ---------------------------------------------------------------------------
def test_same_seed_different_algorithms_differ():
    streams = {}
    for algorithm in ALGORITHMS:
        fn = _uniform(algorithm, (4,), "float32")
        native = np.asarray(nu.evaluate_native(fn))
        np.testing.assert_array_equal(native, np.asarray(nu.evaluate_numpy(fn)))
        streams[algorithm] = native.tobytes()
    assert len(set(streams.values())) == len(ALGORITHMS), (
        "one seed must give three distinct streams across the three algorithms"
    )


# ---------------------------------------------------------------------------
# composition — a sampled uniform fed through arithmetic with a normal draw
# and a reduction, on both backends
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_composition_uniform_normal_reduce_matches_numpy(algorithm):
    got = nu.evaluate_native(_composition(algorithm))
    expected = nu.evaluate_numpy(_composition(algorithm))
    nu.assert_tree_parity(got, expected, label=f"composition {algorithm}")
    # Both leaves are f64 and every input stream is bit-exact, so the result
    # is bit-exact too (elementwise mul/add preserve the operand bit patterns).
    for index, (g, e) in enumerate(zip(got, expected)):
        np.testing.assert_array_equal(
            np.asarray(g), np.asarray(e), err_msg=f"composition leaf {index}"
        )
