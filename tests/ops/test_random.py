"""evoxc.random — key-based functional RNG graph-op tests.

The contract under test (binding — see ``evoxc/ops/random.py`` and
``evoxc/CONTEXT.md`` section "evoxc.random"):

- Keys are rank-0 int64 tensors holding a 64-bit state. Same key + same
  operands ⇒ BIT-IDENTICAL values, repeatable across separate evaluate runs.
- Ops NEVER consume/mutate a key; ``split``/``split_n`` derive decorrelated
  keys (SplitMix64 with salts). Different op kinds sharing one key draw
  decorrelated streams (fixed per-op salts).
- dtype rules: uniform/normal → floating (default float32); randint/
  permutation → integer (default int32); multinomial output int32.
- ``shape`` accepts int / Dim / DimExpr (runtime-dynamic None rejected);
  low/high/mean/std may be Python scalars or symbolic tensors broadcastable
  against shape; the result shape is the broadcast.
- ``key()`` is a polymorphic creator (concrete Tensor outside a trace,
  Constant op inside); every other function is a graph op (TraceError
  outside a trace; concrete Tensor operands raise TraceError).
- Backends: numpy interpreter (all ops); StableHLO export is v1 for every
  random op EXCEPT ``multinomial`` (the cumulative-search decomposition is
  not wired in v1 — compiler backends reject it with an explicit
  BackendError naming ``random_multinomial``); transforms have no rules for
  random ops → TransformError.
"""
import importlib
from contextlib import contextmanager

import numpy as np
import pytest

import evoxc
from evoxc import core
import evoxc.numpy as enp

from tests.ops.conftest import ops_of, run_numpy

# --- helpers -----------------------------------------------------------------


def _i64(v):
    """A static int64 scalar constant (the verified while_loop pattern)."""
    return evoxc.constant(evoxc.tensor(np.array(v, dtype=np.int64)))


def _exec(fn, *specs):
    """Explicit pipeline (trace → lower → compile → load → run) for graphs
    with symbolic dims, which ``evoxc.evaluate`` cannot stage (it derives fully
    static specs from concrete args)."""
    graph = evoxc.trace(fn, *specs)
    return evoxc.load(evoxc.compile(evoxc.lower(graph)))


@contextmanager
def _compiler_adapter_registered_without_compiler(name):
    """Expose the REAL optional compiler adapter ``name`` through the registry
    without probing (or importing) its compiler dependency.

    ``registry.get(name)`` auto-activates an optional adapter by importing its
    module and calling ``register()``, which probes the compiler packages
    (``check_available``) and raises ``core.BackendError`` with a pip-install
    hint when they are absent. The adapter MODULE itself imports no compiler
    at module scope (the binding heavy-import rule: ``iree`` /
    ``iree.compiler`` / ``iree.runtime`` are imported lazily inside function
    bodies), so its singleton and ``Capabilities`` exist in a compiler-free
    environment; only that probe is environment-dependent.

    Installing the real singleton by direct registry assignment (save/restore
    of the previous entry) therefore bypasses ONLY the probe: the adapter's
    OWN shared ``CompilerBackend.lower`` path — verify → capability pre-check
    → portable inlining → StableHLO export — runs identically with or without
    the compiler packages installed, and ``lower`` never compiles. The
    previous entry is restored in ``finally``, so the helper is
    order-independent: it works whether or not a real ``"iree"`` backend is
    already registered in this process, and never leaks the assignment to
    another test. Tests run sequentially, so no cross-test interference.

    Mirrors the ``iree_stub_registered`` helpers in
    ``tests/pipeline_env_defaults_test.py`` / ``tests/test_pipeline_options.py``
    (same direct-registry-assignment save/restore style) — here with the real
    adapter singleton instead of a stub.
    """
    from evoxc.backends import registry as _registry

    module = importlib.import_module(_registry.OPTIONAL_ADAPTERS[name])
    backend = getattr(module, f"{name}_backend")
    table = _registry._registry
    previous = table.get(name)
    table[name] = backend
    try:
        yield backend
    finally:
        if previous is None:
            table.pop(name, None)
        else:
            table[name] = previous



# --- 1. determinism: same key ⇒ bit-identical draws -------------------------


@pytest.mark.parametrize(
    "op_name",
    ["uniform", "normal", "randint", "permutation", "multinomial", "split"],
)
def test_cross_evaluate_determinism(op_name):
    """Two separate evaluate calls with the same key give bit-identical
    draws for every op kind (split: identical derived key pair)."""
    seed = np.array(123, dtype=np.int64)

    def build():
        @evoxc.defn
        def f(key):
            if op_name == "uniform":
                return evoxc.random.uniform(key, (5, 3))
            if op_name == "normal":
                return evoxc.random.normal(key, (5, 3))
            if op_name == "randint":
                return evoxc.random.randint(key, (7,), 0, 10)
            if op_name == "permutation":
                return evoxc.random.permutation(key, 6)
            if op_name == "multinomial":
                probs = np.array([0.25, 0.25, 0.5], dtype=np.float32)
                return evoxc.random.multinomial(key, evoxc.constant(evoxc.tensor(probs)), 20)
            a, b = evoxc.random.split(key)
            return a, b

        return f

    f = build()
    r1 = run_numpy(f, seed)
    r2 = run_numpy(f, seed)
    if isinstance(r1, tuple):
        assert all(np.array_equal(a, b) for a, b in zip(r1, r2))
    else:
        assert np.array_equal(r1, r2)


def test_determinism_across_staging_paths():
    """The explicit pipeline and evoxc.evaluate produce the same draws."""

    @evoxc.defn
    def f(key):
        return evoxc.random.uniform(key, (4, 2))

    seed = np.array(7, dtype=np.int64)
    via_eval = run_numpy(f, seed)
    exe = _exec(f, evoxc.TensorSpec((), evoxc.int64))
    via_pipeline = evoxc.run(exe, evoxc.random.key(7)).numpy()
    assert np.array_equal(via_eval, via_pipeline)


def test_different_keys_differ():
    """Different keys ⇒ different draws (overwhelmingly likely)."""

    @evoxc.defn
    def f(key):
        return evoxc.random.uniform(key, (8,))

    a = run_numpy(f, np.array(1, dtype=np.int64))
    b = run_numpy(f, np.array(2, dtype=np.int64))
    assert not np.array_equal(a, b)


def test_op_kinds_share_one_key_draw_decorrelated_streams():
    """Two op kinds sharing one key draw from decorrelated streams (per-op
    salts): the uniform and normal outputs for the same key must differ."""

    @evoxc.defn
    def f(key):
        return evoxc.random.uniform(key, (8,)), evoxc.random.normal(key, (8,))

    u, n = run_numpy(f, np.array(1, dtype=np.int64))
    assert not np.array_equal(u, n)


# --- 2. split / split_n semantics -------------------------------------------


def test_split_reproducible_and_distinct():
    """split(key) is reproducible; the two derived keys differ from each
    other and from the parent key."""

    @evoxc.defn
    def f(key):
        a, b = evoxc.random.split(key)
        return a, b

    seed = np.array(99, dtype=np.int64)
    a1, b1 = run_numpy(f, seed)
    a2, b2 = run_numpy(f, seed)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    assert not np.array_equal(a1, b1)  # derived keys are decorrelated
    assert a1.item() != 99 and b1.item() != 99  # distinct from the parent


def test_split_n_distinct_and_deterministic():
    """split_n(key, n) yields n pairwise-distinct keys, deterministically."""

    @evoxc.defn
    def f(key):
        return evoxc.random.split_n(key, 5)

    keys = run_numpy(f, np.array(99, dtype=np.int64))
    assert len(keys) == 5
    values = [k.item() for k in keys]
    assert len(set(values)) == 5  # pairwise distinct
    keys2 = run_numpy(f, np.array(99, dtype=np.int64))
    assert all(np.array_equal(a, b) for a, b in zip(keys, keys2))


def test_split_n_two_matches_split():
    """split_n(key, 2) ≡ split(key) (both are random_key_mix sugar)."""

    @evoxc.defn
    def f(key):
        return evoxc.random.split(key)

    @evoxc.defn
    def g(key):
        return evoxc.random.split_n(key, 2)

    seed = np.array(5, dtype=np.int64)
    a, b = run_numpy(f, seed)
    c, d = run_numpy(g, seed)
    assert np.array_equal(a, c) and np.array_equal(b, d)


def test_split_derived_keys_drive_independent_draws():
    """Samples drawn from split-derived keys are decorrelated: two uniform
    draws from split(key) differ from two draws sharing one key's stream."""

    @evoxc.defn
    def f(key):
        k1, k2 = evoxc.random.split(key)
        return evoxc.random.uniform(k1, (6,)), evoxc.random.uniform(k2, (6,))

    u1, u2 = run_numpy(f, np.array(42, dtype=np.int64))
    assert not np.array_equal(u1, u2)


# --- 3. distribution sanity (loose tolerances) ------------------------------


def test_uniform_range_and_mean():
    @evoxc.defn
    def f(key):
        return evoxc.random.uniform(key, (10000,), 0.0, 1.0)

    u = run_numpy(f, np.array(1, dtype=np.int64))
    assert u.shape == (10000,)
    assert u.dtype == np.float32
    assert u.min() >= 0.0 and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 0.05
    assert abs(u.std() - 1.0 / np.sqrt(12.0)) < 0.05


def test_uniform_range_custom_low_high():
    @evoxc.defn
    def f(key):
        return evoxc.random.uniform(key, (10000,), 2.0, 5.0)

    u = run_numpy(f, np.array(1, dtype=np.int64))
    assert u.min() >= 2.0 and u.max() < 5.0
    assert abs(u.mean() - 3.5) < 0.05


def test_normal_mean_and_std():
    @evoxc.defn
    def f(key):
        return evoxc.random.normal(key, (10000,), 0.0, 1.0)

    z = run_numpy(f, np.array(1, dtype=np.int64))
    assert z.dtype == np.float32
    assert abs(z.mean()) < 0.05
    assert abs(z.std() - 1.0) < 0.1


def test_normal_affine_transform():
    @evoxc.defn
    def f(key):
        return evoxc.random.normal(key, (10000,), 3.0, 2.0)

    z = run_numpy(f, np.array(1, dtype=np.int64))
    assert abs(z.mean() - 3.0) < 0.1
    assert abs(z.std() - 2.0) < 0.2


def test_randint_range():
    @evoxc.defn
    def f(key):
        return evoxc.random.randint(key, (20000,), 0, 10)

    r = run_numpy(f, np.array(1, dtype=np.int64))
    assert r.dtype == np.int32
    assert r.min() >= 0 and r.max() <= 9  # high exclusive
    # all values in the range appear
    assert set(np.unique(r)) == set(range(10))
    assert abs(r.mean() - 4.5) < 0.2


def test_multinomial_indices_in_range():
    @evoxc.defn
    def f(key, probs):
        return evoxc.random.multinomial(key, probs, 10000)

    p = np.array([0.1, 0.2, 0.7], dtype=np.float32)
    idx = run_numpy(f, np.array(1, dtype=np.int64), p)
    assert idx.dtype == np.int32
    assert set(np.unique(idx)) <= {0, 1, 2}
    # the 0.7 mass dominates the empirical frequency
    assert (idx == 2).mean() > 0.6


def test_multinomial_zero_samples():
    @evoxc.defn
    def f(key, probs):
        return evoxc.random.multinomial(key, probs, 0)

    idx = run_numpy(f, np.array(1, dtype=np.int64), np.array([0.5, 0.5], np.float32))
    assert idx.shape == (0,)


# --- 4. permutation ----------------------------------------------------------


def test_permutation_static_n_is_a_shuffle():
    @evoxc.defn
    def f(key):
        return evoxc.random.permutation(key, 7)

    perm = run_numpy(f, np.array(1, dtype=np.int64))
    assert perm.dtype == np.int32
    assert perm.shape == (7,)
    assert np.array_equal(np.sort(perm), np.arange(7))


def test_permutation_symbolic_n():
    """n as a symbolic rank-0 tensor input: the result is a shuffle of
    0..n-1 with runtime-dynamic length."""

    @evoxc.defn
    def f(key, n):
        return evoxc.random.permutation(key, n)

    perm = run_numpy(f, np.array(1, dtype=np.int64), np.array(6, dtype=np.int64))
    assert perm.shape == (6,)
    assert np.array_equal(np.sort(perm), np.arange(6))


def test_permutation_dtype_and_zero():
    @evoxc.defn
    def f(key):
        return evoxc.random.permutation(key, 4, dtype=evoxc.int64)

    perm = run_numpy(f, np.array(1, dtype=np.int64))
    assert perm.dtype == np.int64
    assert np.array_equal(np.sort(perm), np.arange(4))

    @evoxc.defn
    def g(key):
        return evoxc.random.permutation(key, 0)

    empty = run_numpy(g, np.array(1, dtype=np.int64))
    assert empty.shape == (0,)


def test_permutation_static_n_traced_shape_is_static():
    """A static Python int n traces to a STATIC (n,) output shape — so
    evoxc.cond branch unification works for static-size populations; a
    symbolic rank-0 n input stays runtime-dynamic (None,)."""

    @evoxc.defn
    def f_static(key):
        return evoxc.random.permutation(key, 7)

    static_spec = evoxc.lower(
        evoxc.trace(f_static, evoxc.TensorSpec((), "int64"))
    ).signature.output_specs[0]
    assert static_spec.shape == (7,)

    @evoxc.defn
    def f_symbolic(key, n):
        return evoxc.random.permutation(key, n)

    dynamic_spec = evoxc.lower(
        evoxc.trace(f_symbolic, evoxc.TensorSpec((), "int64"), evoxc.TensorSpec((), "int64"))
    ).signature.output_specs[0]
    assert dynamic_spec.shape == (None,)


# --- 5. symbolic operands ----------------------------------------------------


def test_uniform_symbolic_high_columnwise():
    """uniform(key, (num_sample, m), 0.0, bound) with bound an explicit
    graph input of shape (m,): values within [0, bound) column-wise and the
    broadcast result shape (num_sample, m)."""
    m = core.Dim("m")

    @evoxc.defn
    def f(key, bound):
        return evoxc.random.uniform(key, (50, m), 0.0, bound)

    exe = _exec(f, evoxc.TensorSpec((), evoxc.int64), evoxc.TensorSpec((m,), evoxc.float32))
    bound = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = evoxc.run(exe, evoxc.random.key(1), bound)
    assert out.shape == (50, 3)
    assert out.dtype == np.float32
    assert (out.numpy() >= 0.0).all()
    assert (out.numpy() < bound[None, :]).all()


def test_uniform_symbolic_low_scalar():
    """A rank-0 symbolic low operand is broadcast against shape."""

    @evoxc.defn
    def f(key, lo):
        return evoxc.random.uniform(key, (4,), lo, 1.0)

    out = run_numpy(f, np.array(1, dtype=np.int64), np.array(0.25, dtype=np.float32))
    assert (out >= 0.25).all() and (out < 1.0).all()


def test_randint_symbolic_low_high():
    """randint with symbolic low/high operands: values in [low, high)."""

    @evoxc.defn
    def f(key, lo, hi):
        return evoxc.random.randint(key, (1000,), lo, hi)

    out = run_numpy(
        f, np.array(2, dtype=np.int64), np.array(0, dtype=np.int64),
        np.array(10, dtype=np.int64),
    )
    assert out.min() >= 0 and out.max() <= 9


def test_randint_symbolic_high_columnwise():
    """randint(key, (num, m), 0, highs) with highs of shape (m,): values in
    [0, highs) column-wise (broadcast result shape)."""
    m = core.Dim("m")

    @evoxc.defn
    def f(key, highs):
        return evoxc.random.randint(key, (1000, m), 0, highs)

    exe = _exec(f, evoxc.TensorSpec((), evoxc.int64), evoxc.TensorSpec((m,), evoxc.int32))
    highs = np.array([2, 5], dtype=np.int32)
    out = evoxc.run(exe, evoxc.random.key(4), highs)
    v = out.numpy()
    assert v.shape == (1000, 2)
    assert (v[:, 0] < 2).all() and (v[:, 1] < 5).all()
    assert (v >= 0).all()


# --- 6. dtype rules ----------------------------------------------------------


@pytest.mark.parametrize(
    ("fn", "spec"),
    [
        (lambda key: evoxc.random.uniform(key, (2,), dtype=evoxc.int32),
         evoxc.TensorSpec((), evoxc.int64)),
        (lambda key: evoxc.random.normal(key, (2,), dtype=evoxc.int64),
         evoxc.TensorSpec((), evoxc.int64)),
    ],
    ids=["uniform-int", "normal-int"],
)
def test_float_ops_reject_integer_dtype(fn, spec):
    with pytest.raises(core.DTypeError, match="floating dtype"):
        evoxc.trace(fn, spec)


@pytest.mark.parametrize(
    ("fn", "spec"),
    [
        (lambda key: evoxc.random.randint(key, (2,), 0, 5, dtype=evoxc.float32),
         evoxc.TensorSpec((), evoxc.int64)),
        (lambda key: evoxc.random.permutation(key, 4, dtype=evoxc.float32),
         evoxc.TensorSpec((), evoxc.int64)),
    ],
    ids=["randint-float", "permutation-float"],
)
def test_int_ops_reject_float_dtype(fn, spec):
    with pytest.raises(core.DTypeError, match="integer dtype"):
        evoxc.trace(fn, spec)


def test_multinomial_input_must_be_float():
    @evoxc.defn
    def f(key, p):
        return evoxc.random.multinomial(key, p, 3)

    with pytest.raises(core.DTypeError, match="floating dtype"):
        evoxc.evaluate(f, evoxc.random.key(1), np.array([1, 1], dtype=np.int32))


def test_default_output_dtypes():
    @evoxc.defn
    def f(key, probs):
        return (
            evoxc.random.uniform(key, (2,)),
            evoxc.random.randint(key, (2,), 0, 3),
            evoxc.random.permutation(key, 3),
            evoxc.random.multinomial(key, probs, 3),
        )

    u, r, p, mn = run_numpy(
        f, np.array(1, dtype=np.int64), np.array([0.5, 0.5], np.float32)
    )
    assert u.dtype == np.float32
    assert r.dtype == np.int32
    assert p.dtype == np.int32
    assert mn.dtype == np.int32


def test_explicit_dtypes_respected():
    @evoxc.defn
    def f(key):
        return (
            evoxc.random.uniform(key, (2,), dtype=evoxc.float64),
            evoxc.random.permutation(key, 4, dtype=evoxc.int64),
        )

    u, p = run_numpy(f, np.array(1, dtype=np.int64))
    assert u.dtype == np.float64
    assert p.dtype == np.int64


# --- 7. explicitness ---------------------------------------------------------


def test_random_ops_raise_traceerror_outside_trace():
    key = evoxc.random.key(1)
    calls = [
        lambda: evoxc.random.uniform(key, (2,)),
        lambda: evoxc.random.normal(key, (2,)),
        lambda: evoxc.random.randint(key, (2,), 0, 5),
        lambda: evoxc.random.permutation(key, 4),
        lambda: evoxc.random.split(key),
        lambda: evoxc.random.split_n(key, 2),
    ]
    for call in calls:
        with pytest.raises(core.TraceError):
            call()


def test_multinomial_outside_trace():
    with pytest.raises(core.TraceError):
        evoxc.random.multinomial(
            evoxc.random.key(1), np.array([0.5, 0.5], dtype=np.float32), 3
        )


def test_concrete_tensor_key_inside_trace_raises():
    with pytest.raises(core.TraceError, match="concrete Tensor key"):
        evoxc.trace(
            lambda: evoxc.random.uniform(evoxc.tensor(np.array(1, np.int64)), (2,))
        )


def test_concrete_tensor_operand_inside_trace_raises():
    @evoxc.defn
    def f(key):
        return evoxc.random.uniform(key, (2,), evoxc.tensor(np.array(0.0)), 1.0)

    with pytest.raises(core.TraceError, match="concrete Tensor operand"):
        evoxc.evaluate(f, evoxc.random.key(1))


def test_key_outside_trace_is_concrete_tensor():
    k = evoxc.random.key(123)
    assert isinstance(k, core.Tensor)
    assert k.dtype == np.dtype("int64")
    assert k.shape == ()
    assert k.numpy().item() == 123


def test_key_masks_seed_to_64_bits():
    # two's complement 64-bit masking: negative seeds round-trip, values
    # beyond 2^64 wrap, 2^63 becomes the most negative int64.
    assert evoxc.random.key(-1).numpy().item() == -1
    assert evoxc.random.key(2**64 + 5).numpy().item() == 5
    assert evoxc.random.key(2**63).numpy().item() == -(2**63)


def test_key_rejects_non_int_seed():
    with pytest.raises(TypeError, match="seed must be a Python int"):
        evoxc.random.key(1.5)
    with pytest.raises(TypeError, match="seed must be a Python int"):
        evoxc.random.key("1")


def test_key_inside_trace_builds_constant():
    @evoxc.defn
    def f():
        return evoxc.random.key(42)

    graph = evoxc.trace(f)
    assert [op.name for op in ops_of(graph)] == ["constant", "return"]
    assert evoxc.evaluate(f).numpy().item() == 42


def test_concrete_key_usable_as_explicit_graph_input():
    """key(seed) outside a trace returns a concrete Tensor that can be
    passed as an explicit graph input and run through the pipeline."""
    key = evoxc.random.key(123)

    @evoxc.defn
    def f(key):
        return evoxc.random.uniform(key, (3,))

    exe = _exec(f, evoxc.TensorSpec((), evoxc.int64))
    out1 = evoxc.run(exe, key).numpy()
    out2 = evoxc.run(exe, evoxc.random.key(123)).numpy()
    assert np.array_equal(out1, out2)


def test_key_spec_must_be_rank0_int64():
    for spec in (evoxc.TensorSpec((), evoxc.int32), evoxc.TensorSpec((2,), evoxc.int64)):
        with pytest.raises(core.ShapeError, match="rank-0 int64 tensor"):
            evoxc.trace(lambda key: evoxc.random.uniform(key, (2,)), spec)


def test_randint_operands_must_be_int():
    with pytest.raises(TypeError, match="random.randint low"):
        evoxc.trace(
            lambda key: evoxc.random.randint(key, (2,), 0.5, 2),
            evoxc.TensorSpec((), evoxc.int64),
        )


# --- 8. random inside while_loop (key threaded through state) ----------------


def test_random_inside_while_loop():
    """Random ops are pure graph ops: they work inside while_loop bodies
    with the key threaded through the loop state via split, and the whole
    loop is deterministic across evaluate calls."""

    @evoxc.defn
    def f():
        init = (_i64(0), _i64(7), enp.zeros((2,), evoxc.float32))

        def cond_fn(state):
            i, _k, _acc = state
            return i < _i64(4)

        def body_fn(state):
            i, k, _acc = state
            k1, k2 = evoxc.random.split(k)
            return (i + 1, k2, evoxc.random.uniform(k1, (2,)))

        return evoxc.while_loop(cond_fn, body_fn, init)

    o1 = evoxc.evaluate(f)
    o2 = evoxc.evaluate(f)
    assert o1[0].numpy().item() == 4  # four iterations ran
    acc1, acc2 = o1[2].numpy(), o2[2].numpy()
    assert np.array_equal(acc1, acc2)  # deterministic across runs
    assert (acc1 >= 0.0).all() and (acc1 < 1.0).all()


def test_while_loop_key_threading_changes_stream():
    """Threading a DIFFERENT initial key through the same loop gives a
    different draw (the loop consumes the key stream)."""

    @evoxc.defn
    def f(seed):
        init = (_i64(0), seed, enp.zeros((2,), evoxc.float32))

        def cond_fn(state):
            i, _k, _acc = state
            return i < _i64(4)

        def body_fn(state):
            i, k, _acc = state
            k1, k2 = evoxc.random.split(k)
            return (i + 1, k2, evoxc.random.uniform(k1, (2,)))

        return evoxc.while_loop(cond_fn, body_fn, init)

    o1 = evoxc.evaluate(f, evoxc.random.key(1))
    o2 = evoxc.evaluate(f, evoxc.random.key(2))
    assert not np.array_equal(o1[2].numpy(), o2[2].numpy())


# --- 9. validation errors ----------------------------------------------------


def test_randint_static_low_ge_high_valueerror():
    with pytest.raises(ValueError, match="high must be greater than low"):
        evoxc.trace(
            lambda key: evoxc.random.randint(key, (2,), 5, 2),
            evoxc.TensorSpec((), evoxc.int64),
        )


def test_randint_symbolic_low_ge_high_runtime_valueerror():
    @evoxc.defn
    def f(key, lo, hi):
        return evoxc.random.randint(key, (2,), lo, hi)

    with pytest.raises(ValueError, match="span <= 0"):
        evoxc.evaluate(
            f, evoxc.random.key(1), np.array(5, dtype=np.int64),
            np.array(2, dtype=np.int64),
        )


def test_permutation_negative_static_n():
    with pytest.raises(ValueError, match="n must be >= 0"):
        evoxc.trace(
            lambda key: evoxc.random.permutation(key, -1),
            evoxc.TensorSpec((), evoxc.int64),
        )


def test_permutation_rejects_non_int_n():
    with pytest.raises(TypeError, match="n must be a Python int"):
        evoxc.trace(
            lambda key: evoxc.random.permutation(key, 2.5),
            evoxc.TensorSpec((), evoxc.int64),
        )


def test_shape_rejects_runtime_dynamic_none():
    with pytest.raises(core.TraceError, match="runtime-dynamic None"):
        evoxc.trace(
            lambda key: evoxc.random.uniform(key, (2, None)),
            evoxc.TensorSpec((), evoxc.int64),
        )


def test_shape_rejects_non_shape_type():
    with pytest.raises(core.TraceError, match="int or a tuple/list"):
        evoxc.trace(
            lambda key: evoxc.random.uniform(key, 2.5),
            evoxc.TensorSpec((), evoxc.int64),
        )


def test_split_n_validation():
    with pytest.raises(ValueError, match="n must be >= 0"):
        evoxc.trace(
            lambda key: evoxc.random.split_n(key, -1),
            evoxc.TensorSpec((), evoxc.int64),
        )
    with pytest.raises(TypeError, match="n must be a Python int"):
        evoxc.trace(
            lambda key: evoxc.random.split_n(key, 1.5),
            evoxc.TensorSpec((), evoxc.int64),
        )


@pytest.mark.parametrize(
    ("probs", "match"),
    [
        (np.array([-0.5, 1.5], dtype=np.float32), "non-negative"),
        (np.array([0.3, 0.3], dtype=np.float32), "sum to 1"),
        (np.array([], dtype=np.float32), "non-empty"),
    ],
    ids=["negative", "sum-not-1", "empty"],
)
def test_multinomial_input_validation(probs, match):
    @evoxc.defn
    def f(key, p):
        return evoxc.random.multinomial(key, p, 10)

    with pytest.raises(ValueError, match=match):
        evoxc.evaluate(f, evoxc.random.key(1), probs)


def test_multinomial_input_ndim():
    @evoxc.defn
    def f(key, p):
        return evoxc.random.multinomial(key, p, 3)

    with pytest.raises(core.ShapeError, match="1-D"):
        evoxc.evaluate(f, evoxc.random.key(1), np.ones((2, 2), np.float32))


def test_multinomial_num_samples_validation():
    @evoxc.defn
    def f(key, p):
        return evoxc.random.multinomial(key, p, -1)

    with pytest.raises(ValueError, match="num_samples must be >= 0"):
        evoxc.evaluate(f, evoxc.random.key(1), np.array([0.5, 0.5], np.float32))

    @evoxc.defn
    def g(key, p):
        return evoxc.random.multinomial(key, p, 2.5)

    with pytest.raises(TypeError, match="num_samples must be a Python int"):
        evoxc.evaluate(g, evoxc.random.key(1), np.array([0.5, 0.5], np.float32))


def test_multinomial_concrete_input_inside_trace_raises():
    @evoxc.defn
    def f(key):
        return evoxc.random.multinomial(
            key, evoxc.tensor(np.array([0.5, 0.5], np.float32)), 3
        )

    with pytest.raises(core.TraceError, match="concrete Tensor input"):
        evoxc.evaluate(f, evoxc.random.key(1))


# --- 10. transforms have no rules for random ops ------------------------------


def test_vmap_of_random_op_raises_transformerror():
    """vmap maps the batch axis into the random op's operands; there is no
    batching rule in v1, so the error is an explicit TransformError (never a
    silent Python-loop fallback)."""

    @evoxc.defn
    def f(key, bound):
        return evoxc.random.uniform(key, (3,), 0.0, bound)

    with pytest.raises(core.TransformError, match="no batching rule for op 'random_uniform'"):
        evoxc.vmap(f, in_axes=(None, 0))(
            evoxc.TensorSpec((), evoxc.int64), evoxc.TensorSpec((2, 3), evoxc.float32)
        )


# --- 11. backend coverage: numpy only in v1 -----------------------------------


def test_stablehlo_export_rejects_random_op():
    """The StableHLO exporter names the offending random op in its explicit
    v1 BackendError (no compiler installation needed — the check is in the
    exporter). ``random_multinomial`` is the only random op still deferred
    in v1 — the other five export (see tests/backends/
    test_iree_emitters_parity.py for their iree parity)."""

    @evoxc.defn
    def f(key):
        probs = evoxc.constant(
            evoxc.tensor(np.array([0.25, 0.25, 0.5], dtype=np.float32))
        )
        return evoxc.random.multinomial(key, probs, 5)

    graph = evoxc.trace(f, evoxc.TensorSpec((), evoxc.int64))
    with pytest.raises(core.BackendError) as excinfo:
        evoxc.backends.stablehlo.export(graph)
    msg = str(excinfo.value)
    assert "op 'random_multinomial'" in msg
    assert "not supported in v1" in msg


def test_compiler_backend_lower_rejects_random_op():
    """Adapter lowering rejects the one still-deferred random op
    (``random_multinomial``) via the shared stablehlo export path before
    any compiler runs (pure capability pre-check).

    Environment-INDEPENDENT: the real iree adapter singleton is installed
    into ``evoxc.backends.registry`` for the duration of the test, bypassing
    only the env-dependent ``register()``/``check_available()`` probe — no
    compiler is imported, invoked or required, and nothing is skipped, so the
    assertion is strict with OR without the IREE packages installed.
    """

    @evoxc.defn
    def f(key):
        probs = evoxc.constant(
            evoxc.tensor(np.array([0.25, 0.25, 0.5], dtype=np.float32))
        )
        return evoxc.random.multinomial(key, probs, 5)

    graph = evoxc.trace(f, evoxc.TensorSpec((), evoxc.int64))
    with _compiler_adapter_registered_without_compiler("iree"):
        with pytest.raises(core.BackendError) as excinfo:
            evoxc.lower(graph, backend="iree")
    message = str(excinfo.value)
    assert "op 'random_multinomial'" in message
    assert "not supported in v1" in message
