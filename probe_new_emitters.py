"""Probe: exercise all new StableHLO emissions through iree (llvm-cpu) vs numpy."""
import sys
import numpy as np
import evoxc

OUT = {}
FAIL = []


def _to_np(v):
    if isinstance(v, evoxc.Tensor):
        return np.asarray(v.numpy())
    return np.asarray(v)


def check(name, got, want, exact=False, tol=1e-5):
    # Pytree-aware: tuple outputs (e.g. random.split's key pair) compare
    # leaf-by-leaf; Tensor leaves convert to ndarrays.
    g = evoxc.tree_map(_to_np, got)
    w = evoxc.tree_map(_to_np, want)
    g_leaves, w_leaves = evoxc.tree_leaves(g), evoxc.tree_leaves(w)
    if len(g_leaves) != len(w_leaves):
        FAIL.append(f"{name}: leaf count {len(g_leaves)} != {len(w_leaves)}")
        return
    ok = True
    maxdiff = 0.0
    for gp, wp in zip(g_leaves, w_leaves):
        if gp.shape != wp.shape:
            FAIL.append(f"{name}: shape {gp.shape} != {wp.shape}")
            return
        if exact:
            leaf_ok = np.array_equal(gp, wp)
        else:
            leaf_ok = np.allclose(gp, wp, rtol=tol, atol=tol)
        maxdiff = max(
            maxdiff, float(np.max(np.abs(gp.astype(np.float64) - wp.astype(np.float64))))
        )
        ok = ok and leaf_ok
    msg = f"{name}: {'EXACT' if exact else 'allclose'} {'OK' if ok else 'FAIL'} maxdiff={maxdiff}"
    print(msg)
    if not ok:
        FAIL.append(msg)


def run_iree(fn, *specs):
    exe = evoxc.build(fn, *specs, backend="iree")
    return exe


def run_cuda(fn, *specs):
    exe = evoxc.build(fn, *specs, backend="iree", device=evoxc.core.Device("cuda", 5), target_backends=["cuda"])
    return exe


def cmp(name, fn, *specs, args=(), exact=False, tol=1e-5, backend="iree"):
    try:
        exe = run_iree(fn, *specs) if backend == "iree" else run_cuda(fn, *specs)
        want = evoxc.evaluate(fn, *args)
        got = evoxc.run(exe, *args)
        check(name, got.numpy() if isinstance(got, evoxc.Tensor) else got,
              want.numpy() if isinstance(want, evoxc.Tensor) else want, exact=exact, tol=tol)
    except Exception as e:
        import traceback
        traceback.print_exc()
        FAIL.append(f"{name}: EXCEPTION {e!r}")


rng = np.random.default_rng(7)

# ---------------- gather ----------------
P, D, K = 8, 5, 4
pop = rng.standard_normal((P, D)).astype(np.float32)
idx0 = np.array([3, 1, 6, 1], dtype=np.int64)
idx1 = np.array([2, 0, 4], dtype=np.int64)   # axis=1 indices
neg = np.array([-1, 2], dtype=np.int64)


def g0(x, i):
    return evoxc.gather(x, i, axis=0)


def g1(x, i):
    return evoxc.gather(x, i, axis=1)


def g1s(x):
    return evoxc.gather(x, evoxc.constant(evoxc.core.Tensor(np.array([2, 0, 4], dtype=np.int64))), axis=1)


def g0s(x):
    return evoxc.gather(x, evoxc.constant(evoxc.core.Tensor(np.array([3, 1, 6, 1], dtype=np.int64))), axis=0)


cmp("gather_axis0_rt", g0, evoxc.TensorSpec((P, D), evoxc.float32), evoxc.TensorSpec((K,), evoxc.int64), args=(pop, idx0), exact=True)
cmp("gather_axis0_static", g0s, evoxc.TensorSpec((P, D), evoxc.float32), args=(pop,), exact=True)
cmp("gather_axis1_rt", g1, evoxc.TensorSpec((P, D), evoxc.float32), evoxc.TensorSpec((3,), evoxc.int64), args=(pop, idx1), exact=True)
cmp("gather_axis1_static", g1s, evoxc.TensorSpec((P, D), evoxc.float32), args=(pop,), exact=True)
cmp("gather_neg", g0, evoxc.TensorSpec((P, D), evoxc.float32), evoxc.TensorSpec((2,), evoxc.int64), args=(pop, neg), exact=True)


def g_scalar(x, i):
    return evoxc.gather(x, i, axis=0)


cmp("gather_0d", g_scalar, evoxc.TensorSpec((P, D), evoxc.float32), evoxc.TensorSpec((), evoxc.int64), args=(pop, np.array(3, dtype=np.int64)), exact=True)

# ---------------- sort / argsort ----------------
x1 = np.array([5.0, 1.0, 5.0, 3.0, 2.0], dtype=np.float32)
x2 = rng.standard_normal((3, 6)).astype(np.float32)
x1b = np.array([True, False, True, False], dtype=np.bool_)


def s1(x):
    return evoxc.sort(x, axis=0)


def s1d(x):
    return evoxc.sort(x, axis=0, descending=True)


def s2(x):
    return evoxc.sort(x, axis=1)


def a1(x):
    return evoxc.argsort(x, axis=0)


def a1d(x):
    return evoxc.argsort(x, axis=0, descending=True)


def a2(x):
    return evoxc.argsort(x, axis=1)


def sb(x):
    return evoxc.argsort(x, axis=0)


cmp("sort_1d", s1, evoxc.TensorSpec((5,), evoxc.float32), args=(x1,), exact=True)
cmp("sort_1d_desc", s1d, evoxc.TensorSpec((5,), evoxc.float32), args=(x1,), exact=True)
cmp("sort_2d_axis1", s2, evoxc.TensorSpec((3, 6), evoxc.float32), args=(x2,), exact=True)
cmp("argsort_1d_stable", a1, evoxc.TensorSpec((5,), evoxc.float32), args=(x1,), exact=True)
cmp("argsort_1d_desc", a1d, evoxc.TensorSpec((5,), evoxc.float32), args=(x1,), exact=True)
cmp("argsort_2d_axis1", a2, evoxc.TensorSpec((3, 6), evoxc.float32), args=(x2,), exact=True)
cmp("argsort_bool", sb, evoxc.TensorSpec((4,), np.bool_), args=(x1b,), exact=True)

# ---------------- argmax / argmin ----------------
am = np.array([3.0, 1.0, 3.0, 2.0], dtype=np.float32)
a2d = rng.standard_normal((3, 7)).astype(np.float32)


def mx(x):
    return evoxc.argmax(x, axis=0)


def mn(x):
    return evoxc.argmin(x, axis=0)


def mxk(x):
    return evoxc.argmax(x, axis=1, keepdims=True)


def mxflat(x):
    return evoxc.argmax(x)


cmp("argmax_1d", mx, evoxc.TensorSpec((4,), evoxc.float32), args=(am,), exact=True)
cmp("argmin_1d", mn, evoxc.TensorSpec((4,), evoxc.float32), args=(am,), exact=True)
cmp("argmax_2d_keepdims", mxk, evoxc.TensorSpec((3, 7), evoxc.float32), args=(a2d,), exact=True)
cmp("argmax_flat", mxflat, evoxc.TensorSpec((3, 7), evoxc.float32), args=(a2d,), exact=True)


def mx0(x):
    return evoxc.argmax(x, axis=0)


cmp("argmax_2d_axis0", mx0, evoxc.TensorSpec((3, 7), evoxc.float32), args=(a2d,), exact=True)

# ---------------- tile ----------------
t = np.arange(6, dtype=np.float32).reshape(2, 3)


def t1(x):
    return evoxc.tile(x, (2,))


def t2(x):
    return evoxc.tile(x, (2, 2))


def t3(x):
    return evoxc.tile(x, (2, 1))


def t4(x):
    return evoxc.tile(x, (2, 2, 1))


cmp("tile_rightalign", t1, evoxc.TensorSpec((2, 3), evoxc.float32), args=(t,), exact=True)
cmp("tile_2x2", t2, evoxc.TensorSpec((2, 3), evoxc.float32), args=(t,), exact=True)
cmp("tile_2x1", t3, evoxc.TensorSpec((2, 3), evoxc.float32), args=(t,), exact=True)
cmp("tile_promote", t4, evoxc.TensorSpec((2, 3), evoxc.float32), args=(t,), exact=True)

# ---------------- scatter ----------------
sc = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
si = np.array([4, 1, 3], dtype=np.int64)
su = np.array([7.0, 8.0, 9.0], dtype=np.float32)
mat = np.arange(12, dtype=np.float32).reshape(3, 4)
mupd = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)


def sc1(x, i, u):
    return evoxc.scatter(x, i, u, axis=0)


def sc2(x, i, u):
    return evoxc.scatter(x, i, u, axis=0)


cmp("scatter_1d", sc1, evoxc.TensorSpec((5,), evoxc.float32), evoxc.TensorSpec((3,), evoxc.int64), evoxc.TensorSpec((3,), evoxc.float32), args=(sc, si, su), exact=True)


def scrow(x, i, u):
    return evoxc.scatter(x, i, u, axis=0)


cmp("scatter_row_0d", scrow, evoxc.TensorSpec((3, 4), evoxc.float32), evoxc.TensorSpec((), evoxc.int64), evoxc.TensorSpec((4,), evoxc.float32), args=(mat, np.array(1, dtype=np.int64), mupd), exact=True)


def scrow2(x, i, u):
    return evoxc.scatter(x, i, u, axis=0)


idx2 = np.array([2, 0], dtype=np.int64)
upd2 = np.array([[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]], dtype=np.float32)
cmp("scatter_row_multi", scrow2, evoxc.TensorSpec((3, 4), evoxc.float32), evoxc.TensorSpec((2,), evoxc.int64), evoxc.TensorSpec((2, 4), evoxc.float32), args=(mat, idx2, upd2), exact=True)


def sccol(x, i, u):
    return evoxc.scatter(x, i, u, axis=1)


icol = np.array([1, 3], dtype=np.int64)
# Frontend rule: updates = x.shape[:axis] + indices.shape + x.shape[axis+1:]
# = (3,) + (2,) + () = (3, 2).
ucol = np.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=np.float32)
cmp("scatter_axis1_col", sccol, evoxc.TensorSpec((3, 4), evoxc.float32), evoxc.TensorSpec((2,), evoxc.int64), evoxc.TensorSpec((3, 2), evoxc.float32), args=(mat, icol, ucol), exact=True)

# ---------------- shifts ----------------
a64 = np.array([1, -8, 255], dtype=np.int64)
u64 = np.array([1, 2**63, 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)


def shl(x, s):
    return evoxc.bitwise_left_shift(x, s)


def shr(x, s):
    return evoxc.bitwise_right_shift(x, s)


def shr_u64(x, s):
    one = evoxc.constant(evoxc.core.Tensor(np.array([1], dtype=np.int64)))
    mask = evoxc.subtract(evoxc.bitwise_left_shift(one, evoxc.subtract(64, s)), 1)
    return evoxc.bitwise_and(evoxc.bitwise_right_shift(x, s), mask)


cmp("shift_left_i64", shl, evoxc.TensorSpec((3,), evoxc.int64), evoxc.TensorSpec((), evoxc.int64), args=(a64, np.array(3, dtype=np.int64)), exact=True)
cmp("shift_right_i64_arith", shr, evoxc.TensorSpec((3,), evoxc.int64), evoxc.TensorSpec((), evoxc.int64), args=(a64, np.array(3, dtype=np.int64)), exact=True)
cmp("shift_right_u64_logical", shr_u64, evoxc.TensorSpec((3,), evoxc.int64), evoxc.TensorSpec((), evoxc.int64),
    args=(u64.view(np.int64), np.array(3, dtype=np.int64)), exact=True)
# The mask trick must equal numpy's native uint64 shift, bit for bit.
try:
    exe = run_iree(shr_u64, evoxc.TensorSpec((3,), evoxc.int64), evoxc.TensorSpec((), evoxc.int64))
    got_u64 = evoxc.run(exe, u64.view(np.int64), np.array(3, dtype=np.int64)).numpy().view(np.uint64)
    check("shift_right_u64_numpy_parity", got_u64, np.right_shift(u64, 3), exact=True)
except Exception as e:
    import traceback
    traceback.print_exc()
    FAIL.append(f"shift_right_u64_numpy_parity: EXCEPTION {e!r}")

# ---------------- random (SplitMix64 parity, BIT-EXACT) ----------------
def rkey():
    k = evoxc.random.key(42)
    return evoxc.random.split(k)


def runiform(k):
    return evoxc.random.uniform(k, (4, 5), low=-1.0, high=3.0)


def rnormal(k):
    return evoxc.random.normal(k, (4, 5), mean=2.0, std=0.5)


def rrandint(k):
    return evoxc.random.randint(k, (4, 5), low=-10, high=10)


def rperm(k):
    return evoxc.random.permutation(k, 12)


def rperm64(k):
    return evoxc.random.permutation(k, 12, dtype=evoxc.int64)


cmp("random_key_mix", rkey, args=(), exact=True)
cmp("random_uniform", runiform, evoxc.TensorSpec((), evoxc.int64), args=(np.array(42, dtype=np.int64),), exact=True)
cmp("random_normal", rnormal, evoxc.TensorSpec((), evoxc.int64), args=(np.array(42, dtype=np.int64),), exact=True)
cmp("random_randint", rrandint, evoxc.TensorSpec((), evoxc.int64), args=(np.array(42, dtype=np.int64),), exact=True)
cmp("random_permutation", rperm, evoxc.TensorSpec((), evoxc.int64), args=(np.array(42, dtype=np.int64),), exact=True)
cmp("random_permutation_i64", rperm64, evoxc.TensorSpec((), evoxc.int64), args=(np.array(42, dtype=np.int64),), exact=True)

# split_n + uniform chained (PSO-style stream)
def rchain(k):
    ks = evoxc.random.split_n(k, 3)
    return evoxc.random.uniform(ks[0], (4, 3)) + evoxc.random.normal(ks[1], (4, 3)) + evoxc.random.randint(ks[2], (4, 3), low=0, high=5)


cmp("random_chain", rchain, evoxc.TensorSpec((), evoxc.int64), args=(np.array(42, dtype=np.int64),), exact=False, tol=1e-5)

print()
if FAIL:
    print("FAILURES:")
    for f in FAIL:
        print(" -", f)
    sys.exit(1)
print("ALL PROBES PASSED")
