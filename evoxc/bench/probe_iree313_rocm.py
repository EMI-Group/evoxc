"""IREE 3.13.0 / ROCm re-validation probe for the shapes 3.11.0 broke.

The 3.11.0 ROCm record (``evoxc/backends/adapters/CONTEXT.md`` Known Issues 8f/10 and
``PER-ADAPTER-NOTES.md``) pinned four defects:

  * a ``while_loop`` whose carries are initialized from a graph constant and whose
    condition compares an INT32 counter -> the HIP HAL ABORTS at run
    (``hip/event_semaphore.c:786: ABORTED ... hal.fence.await``);
  * ``evoxc.scan`` (same family: a constant carry init) -> ABORT,
    ``hal.device.queue.alloca`` / ``hal.fence.await``;
  * ``iree-compile`` SIGSEGV (exit -11) on a ``while`` NESTED in a ``cond``
    (``IREE::Stream::AffinityAnalysis::run()`` -- a compile-time crash);
  * the MO/NSGA2-class while shape (>=3 loop-carried values, a ``select`` in the
    body, an input-derived ``(n, n)`` i1 dominance matrix): ``iree-compile`` SIGSEGV
    NONDETERMINISTICALLY (~85-100 % of runs, so a single green compile proves
    nothing).

This probe re-measures each class on the currently pinned iree-base (3.13.0),
plus the ordinary elementwise/reduce graph and an SO-algorithm-style loop-heavy
population step, on ``llvm-cpu`` and ``rocm``.  Every graph is a
``@evoxc.defn``-style traced function built ONLY on evoxc's public API; parity is
against a hand-written numpy oracle (not the evoxc numpy backend), so the check is
independent of the frontend.

Per cell it records: lower/compile/load ms, device residency (output device kinds,
the ``.numpy()`` refusal and its exact message), max-abs and scale-relative parity
error, warm per-call time (``run`` + ``Executable.synchronize()``, median and best
of ``--reps`` after ``--warmup`` warmups) and, for the two compile-crash classes,
the per-repeat compile outcome (the crash count).

Every graph is built and run in its OWN subprocess: a failed HIP device poisons the
process (and often hangs at teardown), and a compiler SIGSEGV would take the probe
down with it, so the parent only aggregates child exit codes -- ``-11`` is a real
SIGSEGV, not a retried-away failure.

Results for the recorded run: ``evoxc/bench/iree313_rocm_results.json`` and the
narrative ``evoxc/bench/iree313_rocm_findings.md``.

Usage (needs iree-base compiler+runtime, and a ROCm device for the rocm target):

    EVOXC_DISABLE_LOCATIONS=1 python evoxc/bench/probe_iree313_rocm.py \
        --targets llvm-cpu,rocm --seed 0 --out evoxc/bench/iree313_rocm_results.json

    # a subset of cells / targets
    python evoxc/bench/probe_iree313_rocm.py --cells mo_class,cond_while --targets llvm-cpu
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np

import evoxc
import evoxc.numpy as enp

F32 = evoxc.float32
I32 = evoxc.int32
HOST = evoxc.Device("cpu", 0)

# --------------------------------------------------------------------------
# graph geometry (fixed, so every recorded cell is comparable across runs)
# --------------------------------------------------------------------------
ELEMENTWISE_SHAPE = (256, 128, 64)      # x (N,K) @ w (K,M) + b
WHILE_WIDTH = 256                       # carried f32 vector width
WHILE_STEPS = 5
SCAN_STEPS, SCAN_WIDTH = 32, 128
SO_POP, SO_DIM, SO_STEPS = 10000, 100, 4
COND_WIDTH = 64
COND_WHILE_STEPS = 3
MO_N, MO_OBJECTIVES = 32, 2


def _rng(seed, offset):
    """Per-cell RNG stream (``--seed`` + a fixed cell offset => determinism)."""
    return np.random.default_rng(seed + offset)


# --------------------------------------------------------------------------
# graph builders -- each returns a dict of fn / specs / args / ref (+ metadata)
# --------------------------------------------------------------------------
def build_elementwise(seed):
    """(a) y = relu(x @ w + b) then a full reduce -- the ordinary numeric path."""
    n, k, m = ELEMENTWISE_SHAPE
    rng = _rng(seed, 1)
    x = rng.standard_normal((n, k), dtype=np.float32)
    w = (rng.standard_normal((k, m), dtype=np.float32) * 0.1).astype(np.float32)
    b = (rng.standard_normal((m,), dtype=np.float32) * 0.1).astype(np.float32)

    def fn(x, w, b):
        return evoxc.sum(evoxc.relu(evoxc.add(evoxc.dot(x, w), b)))

    ref = [np.float32(np.maximum(x @ w + b, 0.0).sum())]
    return {
        "fn": fn,
        "specs": (
            evoxc.TensorSpec((n, k), F32),
            evoxc.TensorSpec((k, m), F32),
            evoxc.TensorSpec((m,), F32),
        ),
        "args": [x, w, b],
        "ref": ref,
        "graph": f"y = relu(x @ w + b); sum(y)  x{(n, k)} w{(k, m)} b{(m,)}",
    }


def build_while_i32_counter(seed):
    """(b) the 3.11 ABORT class: INT32 counter carry, constant-initialized carries.

    Three carries (i32 counter, f32 scalar accumulator, f32 vector) -- the counter
    is a graph constant (``enp.zeros``, the documented 3.11 failing init) and the
    condition compares it against a Python int literal (widened in the cond
    region, the exact form `repro_minimal.py --variant abort` used).
    """
    rng = _rng(seed, 2)
    x = rng.standard_normal(WHILE_WIDTH, dtype=np.float32)

    def fn(x):
        init = (enp.zeros((), I32), enp.zeros((), F32), x)

        def cond(carry):
            counter, _acc, _values = carry
            return evoxc.less(counter, WHILE_STEPS)

        def body(carry):
            counter, acc, values = carry
            one = enp.full((), 1, I32)
            new_values = evoxc.tanh(evoxc.add(values, 0.1))
            return (
                evoxc.add(counter, one),
                evoxc.add(acc, evoxc.sum(new_values)),
                new_values,
            )

        counter, acc, values = evoxc.while_loop(cond, body, init)
        return (counter, acc, values)

    values = x.copy()
    acc = np.float32(0.0)
    for _ in range(WHILE_STEPS):
        values = np.tanh(values + np.float32(0.1))
        acc = np.float32(acc + values.sum(dtype=np.float32))
    return {
        "fn": fn,
        "specs": (evoxc.TensorSpec((WHILE_WIDTH,), F32),),
        "args": [x],
        "ref": [np.int32(WHILE_STEPS), acc, values],
        "graph": (
            f"while_loop, {WHILE_STEPS} iters, carries (i32 counter init from a graph "
            f"constant, f32 scalar acc, f32 {(WHILE_WIDTH,)}) -- cond less(i32, literal)"
        ),
    }


def build_scan(seed):
    """(c) the 3.11 ABORT class: ``evoxc.scan`` with a CONSTANT carry init."""
    rng = _rng(seed, 3)
    xs = rng.standard_normal((SCAN_STEPS, SCAN_WIDTH), dtype=np.float32)

    def fn(xs):
        init = enp.zeros((SCAN_WIDTH,), F32)

        def step(carry, x):
            new_carry = evoxc.tanh(evoxc.add(evoxc.multiply(carry, 0.9), x))
            return new_carry, evoxc.sum(new_carry)

        carry, ys = evoxc.scan(step, init, xs)
        return (carry, evoxc.sum(ys))

    carry = np.zeros(SCAN_WIDTH, dtype=np.float32)
    steps = []
    for index in range(SCAN_STEPS):
        carry = np.tanh(carry * np.float32(0.9) + xs[index])
        steps.append(np.float32(carry.sum(dtype=np.float32)))
    return {
        "fn": fn,
        "specs": (evoxc.TensorSpec((SCAN_STEPS, SCAN_WIDTH), F32),),
        "args": [xs],
        "ref": [
            carry,
            np.float32(np.sum(np.array(steps, dtype=np.float32), dtype=np.float32)),
        ],
        "graph": (
            f"scan over {SCAN_STEPS} steps of a {(SCAN_WIDTH,)} carry "
            f"(constant init), stacked per-step sums"
        ),
    }


def build_so_step(seed):
    """(d) the evox SO-algorithm step shape inside a loop: (10000, 100) population.

    Each iteration: per-row score (reduce over the row axis), rank-0 ``argmin``, a
    data-dependent ``gather`` of the best row, a broadcast mean shift toward it and
    an i1 NaN guard turned into a ``select`` -- plus a carried scalar accumulator,
    i.e. the "10000x100-ish population step inside a while loop" geometry that
    Known Issue 13 measured on ROCm.
    """
    rng = _rng(seed, 4)
    pop = rng.standard_normal((SO_POP, SO_DIM), dtype=np.float32)
    noise = rng.random(SO_POP, dtype=np.float32)

    def scores_of(p):
        return evoxc.sum(evoxc.multiply(p, p), axes=1)

    def fn(pop, noise):
        init = (enp.zeros((), I32), pop, enp.zeros((), F32))

        def cond(carry):
            counter, _p, _acc = carry
            return evoxc.less(counter, SO_STEPS)

        def body(carry):
            counter, p, acc = carry
            one = enp.full((), 1, I32)
            scores = scores_of(p)
            index = evoxc.argmin(scores)
            best_row = evoxc.gather(p, enp.expand_dims(index, 0), axis=0)
            lr = enp.multiply(enp.expand_dims(noise, 1), enp.full((), 0.01, F32))
            candidate = evoxc.add(
                p, evoxc.multiply(evoxc.subtract(best_row, p), lr)
            )
            finite = evoxc.equal(scores, scores)          # i1 NaN guard
            candidate = enp.where(enp.expand_dims(finite, 1), candidate, best_row)
            return (
                evoxc.add(counter, one),
                candidate,
                evoxc.add(acc, evoxc.min(scores)),
            )

        counter, p, acc = evoxc.while_loop(cond, body, init)
        return (counter, p, acc, scores_of(p))

    p = pop.copy()
    acc = np.float32(0.0)
    for _ in range(SO_STEPS):
        scores = (p * p).sum(axis=1, dtype=np.float32)
        best_row = p[int(np.argmin(scores)): int(np.argmin(scores)) + 1]
        lr = (noise[:, None] * np.float32(0.01)).astype(np.float32)
        candidate = p + (best_row - p) * lr
        finite = np.isfinite(scores)
        candidate = np.where(finite[:, None], candidate, best_row)
        acc = np.float32(acc + scores.min())
        p = candidate.astype(np.float32)
    return {
        "fn": fn,
        "specs": (
            evoxc.TensorSpec((SO_POP, SO_DIM), F32),
            evoxc.TensorSpec((SO_POP,), F32),
        ),
        "args": [pop, noise],
        "ref": [np.int32(SO_STEPS), p, acc, (p * p).sum(axis=1, dtype=np.float32)],
        "graph": (
            f"while_loop, {SO_STEPS} iters over a ({SO_POP}, {SO_DIM}) f32 population: "
            "row scores -> rank-0 argmin -> gather(best row) -> masked mean shift, "
            "carries (i32 counter, population, f32 accumulator)"
        ),
    }


def build_cond_while(seed):
    """(e) the 3.11 COMPILE-SIGSEGV class: a ``while_loop`` nested in a ``cond``."""
    rng = _rng(seed, 5)
    x = rng.standard_normal(COND_WIDTH, dtype=np.float32)
    pred_true = np.array(True, dtype=np.bool_)
    pred_false = np.array(False, dtype=np.bool_)

    def fn(x, pred):
        def true_fn(x):
            init = (enp.zeros((), I32), x)

            def cond(carry):
                counter, _values = carry
                return evoxc.less(counter, COND_WHILE_STEPS)

            def body(carry):
                counter, values = carry
                one = enp.full((), 1, I32)
                return evoxc.add(counter, one), evoxc.tanh(evoxc.add(values, 0.25))

            _counter, values = evoxc.while_loop(cond, body, init)
            return values

        def false_fn(x):
            return evoxc.negate(x)

        return evoxc.cond(pred, true_fn, false_fn, x)

    looped = x.copy()
    for _ in range(COND_WHILE_STEPS):
        looped = np.tanh(looped + np.float32(0.25))
    return {
        "fn": fn,
        "specs": (
            evoxc.TensorSpec((COND_WIDTH,), F32),
            evoxc.TensorSpec((), np.bool_),
        ),
        "args": [x, pred_true],
        "ref": [looped],
        "alt": [
            {"label": "predFalse", "args": [x, pred_false], "ref": [-x]},
        ],
        "graph": (
            f"cond(pred, while_loop x{COND_WHILE_STEPS} over {(COND_WIDTH,)}, negate) "
            "-- a while nested in a cond region"
        ),
    }


def build_mo_class(seed):
    """(f) the MO/NSGA2 tell shape: ``non_dominate_rank`` (verbatim evox kernel).

    Input-derived ``(n, n)`` i1 dominance matrix, in-body reductions, a
    bool->int32 cast chain and THREE loop-carried values with a ``select`` in the
    body -- the exact shape whose ``iree-compile`` SIGSEGV was nondeterministic on
    3.10/3.11 (Known Issue 10).  Mirrors
    ``tests/backends/test_while_init_rewrite.py::_non_dominate_rank``.
    """
    rng = _rng(seed, 7)
    x = np.abs(rng.standard_normal((MO_N, MO_OBJECTIVES)).astype(np.float32))

    def fn(x):
        n = x.shape[0]
        x_i = enp.expand_dims(x, 1)
        x_j = enp.expand_dims(x, 0)
        le = evoxc.less_equal(x_i, x_j)
        lt = evoxc.less(x_i, x_j)
        le_all = evoxc.min(le, axes=2)
        lt_any = evoxc.max(lt, axes=2)
        dominates = evoxc.logical_and(le_all, lt_any)
        zero_i32 = enp.full((), 0, I32)
        one_i32 = enp.full((), 1, I32)
        count = evoxc.sum(enp.astype(dominates, I32), axes=0)
        rank = enp.zeros((n,), I32)

        def cond_fn(carry):
            c, _r, _cur = carry
            return evoxc.greater_equal(evoxc.max(c), zero_i32)

        def body_fn(carry):
            c, r, cur = carry
            front = evoxc.equal(c, zero_i32)
            r = enp.where(front, cur, r)                       # select in the body
            front_dom = evoxc.sum(
                enp.astype(
                    evoxc.logical_and(dominates, enp.expand_dims(front, 1)), I32
                ),
                axes=0,
            )
            c = evoxc.subtract(c, front_dom)
            c = evoxc.subtract(c, enp.astype(front, I32))
            return (c, r, evoxc.add(cur, one_i32))

        _c, rank, _cur = evoxc.while_loop(cond_fn, body_fn, (count, rank, zero_i32))
        return rank

    le_all = (x[:, None, :] <= x[None, :, :]).all(axis=2)
    lt_any = (x[:, None, :] < x[None, :, :]).any(axis=2)
    dominates = le_all & lt_any
    count = dominates.sum(axis=0).astype(np.int32)
    rank = np.zeros(MO_N, dtype=np.int32)
    current = 0
    while count.max() >= 0:
        front = count == 0
        if not front.any():
            break
        rank[front] = current
        count = (
            count
            - dominates[front].sum(axis=0).astype(np.int32)
            - front.astype(np.int32)
        )
        current += 1
    return {
        "fn": fn,
        "specs": (evoxc.TensorSpec((MO_N, MO_OBJECTIVES), F32),),
        "args": [x],
        "ref": [rank],
        "graph": (
            f"non_dominate_rank n={MO_N}: input-derived ({MO_N}, {MO_N}) i1 dominance "
            "matrix, in-body reductions, bool->i32 cast chain, 3 carried values "
            "(count, rank, current), select in the body"
        ),
    }


CELLS = {
    "elementwise": build_elementwise,
    "while_i32_counter": build_while_i32_counter,
    "scan": build_scan,
    "so_step": build_so_step,
    "cond_while": build_cond_while,
    "mo_class": build_mo_class,
}

#: cells whose 3.11 failure mode was a COMPILE crash -> compiled N times per target
COMPILE_REPEATS = {"cond_while": 5, "mo_class": 10}


# --------------------------------------------------------------------------
# helpers shared by the child worker
# --------------------------------------------------------------------------
def _device_for(target, index):
    return HOST if target == "llvm-cpu" else evoxc.Device("rocm", index)


def _as_array(value):
    """``np.ascontiguousarray`` promotes 0-d arrays, which breaks 0-d specs."""
    array = np.asarray(value)
    return np.ascontiguousarray(array) if array.ndim else array


def _place(args, device):
    """Host arrays -> device tensors (evoxc never transfers implicitly)."""
    return [evoxc.tensor(_as_array(a)).to(device) for a in args]


def _host_leaves(outputs):
    return [leaf.to(HOST).numpy() for leaf in evoxc.tree_leaves(outputs)]


def _parity(got_leaves, refs):
    """Per-leaf max-abs and scale-relative error against the numpy oracle."""
    per_leaf = []
    for got, want in zip(got_leaves, refs):
        got = np.asarray(got)
        want = np.asarray(want)
        if got.shape != want.shape:
            per_leaf.append({"shape_mismatch": True,
                             "got_shape": list(got.shape),
                             "ref_shape": list(want.shape)})
            continue
        got64 = got.astype(np.float64)
        want64 = want.astype(np.float64)
        diff = np.abs(got64 - want64)
        scale = float(np.max(np.abs(want64))) if want64.size else 0.0
        per_leaf.append({
            "shape": list(got.shape),
            "dtype": str(got.dtype),
            "max_abs_err": float(diff.max()) if diff.size else 0.0,
            "scale": scale,
            "max_rel_err": (float(diff.max()) / scale) if scale else 0.0,
            "bit_exact": bool(np.array_equal(got, want)),
        })
    numeric = [leaf for leaf in per_leaf if "max_abs_err" in leaf]
    return {
        "leaves": per_leaf,
        "max_abs_err": max((leaf["max_abs_err"] for leaf in numeric), default=None),
        "max_rel_err": max((leaf["max_rel_err"] for leaf in numeric), default=None),
        "all_bit_exact": bool(numeric) and all(
            leaf["bit_exact"] for leaf in numeric
        ) and len(numeric) == len(per_leaf),
    }


def _residency(outputs):
    """Where the outputs physically live + whether ``.numpy()`` refuses."""
    leaves = evoxc.tree_leaves(outputs)
    first = leaves[0]
    record = {
        "output_devices": [str(leaf.device) for leaf in leaves],
        "device_kinds": sorted({leaf.device.kind for leaf in leaves}),
    }
    try:
        first.numpy()
    except Exception as exc:  # noqa: BLE001 - the refusal IS the observation
        record["numpy_refused"] = True
        record["numpy_refusal"] = f"{type(exc).__name__}: {exc}"
    else:
        record["numpy_refused"] = False
    return record


def _phase(**payload):
    """Emit one machine-readable phase line (the parent parses them)."""
    print("PHASE " + json.dumps(payload), flush=True)


def _finish(code=0):
    sys.stdout.flush()
    sys.stderr.flush()
    # A failed HIP device frequently hangs at interpreter teardown; leave now that
    # every phase line is on the pipe (same discipline as the reference repros).
    os._exit(code)


# --------------------------------------------------------------------------
# child worker: one graph, one target, one process
# --------------------------------------------------------------------------
def run_child(args):
    cell = CELLS[args.cell](args.seed)
    device = _device_for(args.target, args.device_index)
    _phase(phase="start", cell=args.cell, target=args.target,
           graph=cell["graph"], device=str(device))
    try:
        started = time.perf_counter()
        graph = evoxc.trace(cell["fn"], *cell["specs"])
        trace_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        lowered = evoxc.lower(graph, backend="iree",
                              target_backends=[args.target])
        lower_ms = (time.perf_counter() - started) * 1000.0
        _phase(phase="lower", trace_ms=trace_ms, lower_ms=lower_ms)

        started = time.perf_counter()
        artifact = evoxc.compile(lowered, backend="iree",
                                 target_backends=[args.target])
        compile_ms = (time.perf_counter() - started) * 1000.0
        _phase(phase="compile", compile_ms=compile_ms)
        if args.compile_only:
            _finish(0)

        started = time.perf_counter()
        executable = evoxc.load(artifact, backend="iree", device=device)
        load_ms = (time.perf_counter() - started) * 1000.0
        _phase(phase="load", load_ms=load_ms)

        inputs = _place(cell["args"], device)
        outputs = evoxc.run(executable, *inputs)
        executable.synchronize()
        _phase(phase="run", residency=_residency(outputs),
               parity=_parity(_host_leaves(outputs), cell["ref"]))

        for alternative in cell.get("alt", ()):
            alt_inputs = _place(alternative["args"], device)
            alt_outputs = evoxc.run(executable, *alt_inputs)
            executable.synchronize()
            _phase(phase="alt", label=alternative["label"],
                   residency=_residency(alt_outputs),
                   parity=_parity(_host_leaves(alt_outputs), alternative["ref"]))

        samples = []
        for _ in range(args.warmup + args.reps):
            started = time.perf_counter()
            evoxc.run(executable, *inputs)
            executable.synchronize()
            samples.append((time.perf_counter() - started) * 1000.0)
        warm = samples[args.warmup:]
        _phase(phase="timing",
               warm_ms_median=statistics.median(warm),
               warm_ms_best=min(warm),
               warm_samples=warm,
               warmup_samples=samples[:args.warmup])
    except Exception as exc:  # noqa: BLE001 - the probe records, never retries
        _phase(phase="error", error=f"{type(exc).__name__}: {exc}",
               traceback=traceback.format_exc()[-4000:])
        _finish(1)
    _finish(0)


# --------------------------------------------------------------------------
# parent: provenance, child orchestration, aggregation
# --------------------------------------------------------------------------
def _version(distribution):
    try:
        from importlib.metadata import version
        return version(distribution)
    except Exception:  # noqa: BLE001 - metadata absent is fine, record it
        return None


def _rocm_version():
    for path in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-rocm"):
        try:
            with open(path) as handle:
                return handle.read().strip()
        except OSError:
            continue
    return None


def _gfx_arch():
    try:
        proc = subprocess.run(["rocminfo"], capture_output=True, text=True,
                              timeout=60)
        found = sorted(set(re.findall(r"\b(gfx[0-9a-f]{3,})\b", proc.stdout)))
        return ",".join(found) or None
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return None


def _gpu_name(device_index):
    try:
        import iree.runtime as rt
        devices = rt.get_driver("hip").query_available_devices()
        entry = devices[device_index]
        return f"{entry.get('name')} (iree hip device_id {entry.get('device_id')})"
    except Exception:  # noqa: BLE001
        return None


def _meta(args):
    return {
        "probe": "evoxc/bench/probe_iree313_rocm.py",
        "purpose": (
            "re-validate on iree-base 3.13.0 the loop/scan/while-in-cond/MO-class "
            "shapes recorded as broken on 3.11.0 (adapters CONTEXT Known Issues "
            "8f/10, PER-ADAPTER-NOTES Known Issue 10)"
        ),
        "date": datetime.date.today().isoformat(),
        "host": f"{platform.node()} / {platform.platform()} / "
                f"{platform.processor() or 'cpu'} ({os.cpu_count()} cpus)",
        "gpu": _gpu_name(args.device_index),
        "gfx_arch": _gfx_arch(),
        "rocm": _rocm_version(),
        "iree_base_compiler": _version("iree-base-compiler"),
        "iree_base_runtime": _version("iree-base-runtime"),
        "numpy": np.__version__,
        "evoxc": getattr(evoxc, "__version__", None),
        "python": platform.python_version(),
        "seed": args.seed,
        "command": " ".join([os.path.basename(sys.executable)] + sys.argv),
        "cells": list(CELLS),
        "targets": args.targets.split(","),
        "shared_host_caveat": (
            "the ROCm GPU is SHARED with other work on this box: timings are a "
            "single sequential process per cell (never two timers at once) but "
            "must be read as indicative, not isolated-hardware numbers; "
            "correctness/compile-crash results are unaffected"
        ),
        "timing": (
            "each sample = evoxc.run + Executable.synchronize(); warm = median and "
            f"best of {args.reps} samples after {args.warmup} warmups"
        ),
        "parity": (
            "per leaf, max |got - ref| and max |got - ref| / max|ref| against a "
            "hand-written numpy oracle (independent of the evoxc numpy backend); "
            "'all_bit_exact' over every leaf"
        ),
        "compile_repeats": (
            "the two compile-crash cells are compiled compile_repeats times per "
            "target in SEPARATE processes; a non-zero exit is recorded verbatim "
            "(-11 = SIGSEGV, -6 = SIGABRT), never retried away"
        ),
        "env": {key: os.environ[key] for key in
                ("EVOXC_DISABLE_LOCATIONS", "EVOXC_IREE_COMPILE_ARGS",
                 "EVOXC_IREE_RUNTIME_ARGS") if key in os.environ},
        "host_caveat_note": "one timing process at a time; EVOXC_DISABLE_LOCATIONS=1",
    }


def _run_child_process(args, cell, target, compile_only):
    """Spawn one child; return its parsed phases + exit code + tail output."""
    command = [
        sys.executable, os.path.abspath(__file__),
        "--child", "--cell", cell, "--target", target,
        "--seed", str(args.seed), "--device-index", str(args.device_index),
        "--reps", str(args.reps), "--warmup", str(args.warmup),
    ]
    if compile_only:
        command.append("--compile-only")
    started = time.perf_counter()
    try:
        proc = subprocess.run(command, capture_output=True, text=True,
                              timeout=args.timeout, env=os.environ.copy())
        stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        code = "timeout"
    wall_ms = (time.perf_counter() - started) * 1000.0
    phases = []
    for line in stdout.splitlines():
        if line.startswith("PHASE "):
            try:
                phases.append(json.loads(line[len("PHASE "):]))
            except json.JSONDecodeError:
                continue
    merged = {}
    for phase in phases:
        merged.setdefault("phases", []).append(phase)
        merged[phase["phase"]] = {k: v for k, v in phase.items() if k != "phase"}
    record = {
        "target": target,
        "cell": cell,
        "compile_only": compile_only,
        "returncode": code,
        "exit_signal": (-code) if isinstance(code, int) and code < 0 else None,
        "crashed": (code != 0),
        "wall_ms": wall_ms,
        "phases": merged.get("phases", []),
    }
    for key in ("lower", "compile", "load", "run", "timing", "error", "alt"):
        if key in merged:
            record[key] = merged[key]
    if code != 0:
        record["failure"] = {
            "returncode": code,
            "signal": record["exit_signal"],
            "stderr_tail": (stderr or "")[-3000:],
            "stdout_tail": (stdout or "")[-2000:],
        }
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="IREE 3.13.0 ROCm re-validation probe (see module docstring).")
    parser.add_argument("--targets", default="llvm-cpu,rocm",
                        help="comma-separated: llvm-cpu,rocm")
    parser.add_argument("--cells", default=",".join(CELLS),
                        help=f"comma-separated subset of: {','.join(CELLS)}")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="per-child-process timeout in seconds")
    parser.add_argument("--out", default="evoxc/bench/iree313_rocm_results.json")
    # child-worker switches (not for direct use)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cell", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--target", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--compile-only", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.child:
        run_child(args)
        return 0

    cells = [name.strip() for name in args.cells.split(",") if name.strip()]
    unknown = [name for name in cells if name not in CELLS]
    if unknown:
        parser.error(f"unknown cell(s) {unknown}; choose from {sorted(CELLS)}")
    targets = [name.strip() for name in args.targets.split(",") if name.strip()]

    results = {"_meta": _meta(args), "runs": {}}
    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=1)

    for target in targets:
        for cell in cells:
            key = f"{target}|{cell}"
            print(f"=== {key}", flush=True)
            cell_record = _run_child_process(args, cell, target, compile_only=False)
            print(f"  measure exit={cell_record['returncode']} "
                  f"wall={cell_record['wall_ms']:.0f}ms", flush=True)
            if "compile" in cell_record:
                print(f"  compile {cell_record['compile']['compile_ms']:.0f} ms "
                      f"lower {cell_record['lower']['lower_ms']:.0f} ms", flush=True)
            if "timing" in cell_record:
                print(f"  warm median {cell_record['timing']['warm_ms_median']:.3f} ms "
                      f"best {cell_record['timing']['warm_ms_best']:.3f} ms", flush=True)
            if "run" in cell_record:
                residency = cell_record["run"]["residency"]
                print(f"  devices {residency['device_kinds']} "
                      f"numpy_refused={residency['numpy_refused']} "
                      f"parity max_abs={cell_record['run']['parity']['max_abs_err']}",
                      flush=True)
            if cell_record["crashed"]:
                print(f"  FAILURE {cell_record.get('failure', {}).get('stderr_tail', '')[-400:]}",
                      flush=True)

            repeats = COMPILE_REPEATS.get(cell, 0)
            if repeats:
                repeat_records = []
                for index in range(repeats):
                    one = _run_child_process(args, cell, target, compile_only=True)
                    repeat_records.append(one)
                    print(f"  repeat {index + 1}/{repeats} exit={one['returncode']} "
                          f"compile_ms={one.get('compile', {}).get('compile_ms')}",
                          flush=True)
                ok = [r for r in repeat_records if r["returncode"] == 0]
                cell_record["compile_repeats"] = {
                    "n": repeats,
                    "ok": len(ok),
                    "crashed": len(repeat_records) - len(ok),
                    "signals": sorted({r["exit_signal"] for r in repeat_records
                                       if r["exit_signal"] is not None}),
                    "compile_ms": [r.get("compile", {}).get("compile_ms")
                                   for r in repeat_records],
                    "records": repeat_records,
                }
                print(f"  repeats: {len(ok)}/{repeats} ok, signals "
                      f"{cell_record['compile_repeats']['signals']}", flush=True)

            results["runs"][key] = cell_record
            with open(args.out, "w") as handle:
                json.dump(results, handle, indent=1)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
