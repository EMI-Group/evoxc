"""Graph-split decision evidence probe (deterministic, public API only).

Answers one decision question about the structural transform
``evoxc.split_by_inputs`` — *does ONE ``prepare`` plus N ``apply`` runs beat N
unsplit runs at the N a real application uses, and where does the boundary
cost hurt?* — by timing three ways of expressing the SAME computation:

* **unsplit** — one graph over every input, ``build`` once and run N times
  (per-generation cost ``build / N + run``);
* **split** — ``split_by_inputs(graph, fixed_leaves=...)``, i.e. one
  ``prepare`` graph over the fixed inputs plus one ``apply`` graph over the
  varying inputs and the prepared boundary (per-generation cost
  ``(build(prepare) + build(apply) + run(prepare)) / N + run(apply)``);
* **hand-written** — the same two halves written BY HAND as two ordinary
  traced functions (``prepare_fixed(X, W1) -> h`` then
  ``apply_varying(h, W2)``), with the same accounting. This is the user's
  alternative to the transform, so it measures whether the automatic split is
  free or costs something.

Three workload families, each a plausible application shape:

* **A — fixed precompute (expected win).** A GP/NEAT-shaped fitness graph:
  fixed ``dataset (N, 64)`` and ``W1 (64, 64)``, varying candidate ``W2 (64,)``.
  ``prepare`` normalizes the dataset and computes ``tanh(Xn @ W1)`` (the
  expensive half); ``apply`` computes ``mean(tanh_out @ W2)``.
  Swept over ``--sizes`` (dataset rows) and ``--reuse`` counts.
* **B — boundary-dominated (expected loss).** Fixed ``X (16384, 64)`` taken
  almost as-is (one elementwise scale), so ``prepare`` produces a ~4 MiB
  boundary and ``apply`` is a trivial op — the case where materializing and
  re-reading the boundary costs more than recomputing.
* **C — state-variable shape.** Fixed ``(dataset, W1)``, varying ``(state, W2)``
  with a loop-free recurrence step: ``prepare`` hoists the state-INDEPENDENT
  ``dataset @ W1`` while ``W1`` itself crosses the boundary for the mixed
  ``state @ W1`` term, so the transform still removes per-turn work.

Each cell records the boundary (values, bytes, per-value specs), both halves'
op counts and graph digests, build/prepare/apply/unsplit timings with their
min/median/max spread, the amortized per-generation cost at every reuse count
with its speedup (from the medians, plus a min-based cross-check), the analytic
crossover reuse count, and the competing strategy: for family A a hand-written
BATCHED graph over 64 candidates (mathematically the graph ``vmap`` over a
candidate axis expands to) compared against 64 applies and 64 unsplit runs.

``vmap`` itself is NOT measured: every builtin transform delegates to the Rust
engine in ``evoxc._evoxc_native`` and this worktree has no built extension, so
the probe records ``"vmap": {"status": "unavailable", ...}`` and measures the
hand-batched graph as the documented stand-in (its result is checked against 64
separate applies, so the stand-in is validated, not assumed).

Measurement protocol — the host is shared and its load swings by several times
over a run, so all timed stages are **interleaved**: every repetition measures
each formulation once, in the same order, so a contention episode is spread
across formulations instead of landing entirely on one of them. Every number is
``time.perf_counter`` min/median/max over ``--reps`` after one untimed warmup
per call shape, one timing process at a time, with the load average recorded —
indicative, not isolated-hardware, numbers. ``EVOXC_DISABLE_LOCATIONS=1`` is
forced before ``evoxc`` is imported (bench policy: location capture is per-node
trace overhead that is not part of what is compared).

``--threads`` (default 1) pins the BLAS/OpenMP pools before numpy is imported:
this host reports more CPUs to Python than ``nproc`` says are usable, so the
ambient pools oversubscribe it and a BLAS-bound cell comes out bimodal. The
requested and effective values are recorded in ``_meta.threading``.

Run it explicitly (evidence tooling — ``evoxc`` never imports this module and
pytest never collects it)::

    source scripts/dev-env.sh
    PYTHONPATH=. .venv/bin/python evoxc/bench/probe_split_by_inputs.py \\
        --seed 0 --reps 7 --out evoxc/bench/split_by_inputs_results.json

Artefacts: ``evoxc/bench/split_by_inputs_results.json`` (``_meta`` plus one
entry per ``--label`` under ``runs``) and the narrated
``evoxc/bench/split_by_inputs_findings.md``.
A re-run with a new ``--label`` PRESERVES every other label in the file.
A genuine correctness failure (a half's output diverging from the numpy
oracle, a non-cpu output device on the numpy backend, a failed
``SplitPlan.verify()``) exits nonzero; an unavailable backend or the missing
native extension is recorded as data and does not fail the run.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time

REPS_DEFAULT = 7
REUSE_DEFAULT = "1,4,16,64"  # N_apply values the amortization is reported at
SIZES_DEFAULT = "512,4096"  # dataset rows for families A and C
BOUNDARY_SIZE_DEFAULT = 16384  # dataset rows for family B (~4 MiB boundary)
BATCH_CANDIDATES = 64  # candidates in the batched (vmap stand-in) comparison
PARITY_TOL = 1e-5  # float32 formulations differ only by reassociation

SHARED_HOST_CAVEAT = (
    "the host is shared and runs in slow-moving fast/slow modes — the SAME "
    "call samples anywhere from ~1.4 to ~59 ms in one 21-repetition cell — so "
    "a formulation's min and median are mode statistics, not its cost: all "
    "timed stages are interleaved per repetition, the amortized table is "
    "reported THREE ways (medians, mins, and the median of per-repetition "
    "ratios with the mode cancelled), the load average is recorded at both "
    "ends, one timing process runs at a time, and the ratios are the claim "
    "while the absolute milliseconds are indicative rather than "
    "isolated-hardware measurements"
)

LIMITS = (
    "numpy interpreter backend only (host memory on Device('cpu', 0)); no "
    "device transfers, no compiled backend and no multi-process measurement "
    "are part of this record; the boundary is re-read from host memory, so a "
    "device-resident boundary would pay an explicit upload the numpy backend "
    "does not model"
)


# ---------------------------------------------------------------------------
# small timing / formatting helpers
# ---------------------------------------------------------------------------
def _ms(value):
    return round(value, 3)


def _ratio(value):
    return round(value, 2) if value is not None and math.isfinite(value) else None


def _spread(samples):
    """min / median / max of a sample list, rounded for the record."""
    return {
        "min_ms": _ms(min(samples)),
        "median_ms": _ms(statistics.median(samples)),
        "max_ms": _ms(max(samples)),
    }


def _timed_once(fn):
    """Run ``fn`` once; return (result, elapsed_ms)."""
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1e3


def _nproc():
    """What ``nproc`` reports (the usable CPU count), or ``None``."""
    import subprocess

    try:
        return int(subprocess.check_output(["nproc"], text=True).strip())
    except Exception:  # noqa: BLE001 - provenance only
        return None


def _loadavg():
    try:
        return [round(value, 2) for value in os.getloadavg()]
    except OSError:
        return None


def _cpu_model():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _versions():
    def _dist(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    from evoxc import _native
    import evoxc as _evoxc_module

    return {
        "python": sys.version.split()[0],
        "numpy": _dist("numpy"),
        "evoxc": _dist("evoxc") or getattr(_evoxc_module, "__version__", None),
        "native_extension": _native.version() if _native.available() else None,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
    }


def _repo_relative(path):
    if not path:
        return path
    try:
        here = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        return os.path.relpath(os.path.abspath(path), here)
    except ValueError:
        return os.path.abspath(path)


def _write_json(path, label, run, meta):
    """Write/merge ``{"_meta": meta, "runs": {label: run, ...}}``."""
    document = {"_meta": meta, "runs": {}}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            existing = json.load(handle)
        document["runs"] = dict(existing.get("runs") or {})
        document["_meta"] = dict(existing.get("_meta") or {})
    document["_meta"] = meta
    document["runs"][label] = run
    if path:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return document


def _max_abs_error(got, want):
    import numpy as np

    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    if got.shape != want.shape:
        raise AssertionError(f"shape mismatch: {got.shape} != {want.shape}")
    return float(np.max(np.abs(got - want)))


def _ops(graph):
    """Non-terminator entry-block op count (the plan's op-count convention)."""
    return len(graph.module.main.entry_block.ops) - 1


# ---------------------------------------------------------------------------
# workload families (deterministic; public API only)
# ---------------------------------------------------------------------------
def _workload_a(evoxc, np, rows, seed):
    """Family A — fixed precompute: GP/NEAT-shaped fitness over one candidate."""
    rng = np.random.default_rng(seed)
    f32 = evoxc.float32

    @evoxc.defn
    def fitness(dataset, W1, W2):
        mu = evoxc.mean(dataset, axes=0)
        sd = evoxc.std(dataset, axes=0)
        normalized = (dataset - mu) / (sd + 1e-6)
        hidden = evoxc.tanh(evoxc.matmul(normalized, W1))
        return evoxc.mean(evoxc.matmul(hidden, W2))

    @evoxc.defn
    def hand_prepare(dataset, W1):
        mu = evoxc.mean(dataset, axes=0)
        sd = evoxc.std(dataset, axes=0)
        normalized = (dataset - mu) / (sd + 1e-6)
        return evoxc.tanh(evoxc.matmul(normalized, W1))

    @evoxc.defn
    def hand_apply(hidden, W2):
        return evoxc.mean(evoxc.matmul(hidden, W2))

    @evoxc.defn
    def batched(dataset, W1, W2s):
        # The graph `vmap(fitness, in_axes=(None, None, 0))` expands to: the
        # fixed half stays unbatched (shared), the candidate axis is a real op
        # dimension, and the reduction runs per candidate.
        mu = evoxc.mean(dataset, axes=0)
        sd = evoxc.std(dataset, axes=0)
        normalized = (dataset - mu) / (sd + 1e-6)
        hidden = evoxc.tanh(evoxc.matmul(normalized, W1))
        return evoxc.mean(evoxc.matmul(hidden, evoxc.transpose(W2s)), axes=0)

    dataset = rng.standard_normal((rows, 64)).astype(np.float32)
    W1 = (rng.standard_normal((64, 64)) * 0.1).astype(np.float32)
    W2 = rng.standard_normal((64,)).astype(np.float32)
    W2s = rng.standard_normal((BATCH_CANDIDATES, 64)).astype(np.float32)

    def oracle(values):
        hidden = np.tanh(
            ((values[0] - values[0].mean(axis=0))
             / (values[0].std(axis=0) + np.float32(1e-6))) @ values[1]
        )
        return np.asarray(np.mean(hidden @ values[2], dtype=np.float32),
                          dtype=np.float32)

    return {
        "workload": "A_fixed_precompute",
        "label": "fixed precompute (GP/NEAT fitness)",
        "expectation": "win",
        "rows": rows,
        "shape": (f"dataset ({rows}, 64) float32 fixed + W1 (64, 64) float32 "
                  f"fixed; candidate W2 (64,) float32 varying"),
        "fn": fitness,
        "specs": (evoxc.TensorSpec((rows, 64), f32),
                  evoxc.TensorSpec((64, 64), f32),
                  evoxc.TensorSpec((64,), f32)),
        "fixed_leaves": [0, 1],
        "arrays": (dataset, W1, W2),
        "hand_prepare": hand_prepare,
        "hand_prepare_specs": (evoxc.TensorSpec((rows, 64), f32),
                               evoxc.TensorSpec((64, 64), f32)),
        "hand_apply": hand_apply,
        "hand_apply_specs": (evoxc.TensorSpec((rows, 64), f32),
                             evoxc.TensorSpec((64,), f32)),
        "hand_apply_indices": [2],  # prepared + candidate W2
        "batched": batched,
        "batched_specs": (evoxc.TensorSpec((rows, 64), f32),
                          evoxc.TensorSpec((64, 64), f32),
                          evoxc.TensorSpec((BATCH_CANDIDATES, 64), f32)),
        "batched_arrays": (dataset, W1, W2s),
        "oracle": oracle,
    }


def _workload_b(evoxc, np, rows, seed):
    """Family B — boundary-dominated: cheap fixed op, large boundary."""
    rng = np.random.default_rng(seed)
    f32 = evoxc.float32

    @evoxc.defn
    def boundary_graph(X, W2):
        return evoxc.mean((X * 3.0) * W2)

    @evoxc.defn
    def hand_prepare(X):
        return X * 3.0

    @evoxc.defn
    def hand_apply(scaled, W2):
        return evoxc.mean(scaled * W2)

    X = rng.standard_normal((rows, 64)).astype(np.float32)
    W2 = rng.standard_normal((64,)).astype(np.float32)

    def oracle(values):
        scaled = values[0] * np.float32(3.0)
        return np.asarray(np.mean(scaled * values[1], dtype=np.float32),
                          dtype=np.float32)

    return {
        "workload": "B_boundary_dominated",
        "label": "boundary-dominated (cheap fixed op, large boundary)",
        "expectation": "loss",
        "rows": rows,
        "shape": (f"X ({rows}, 64) float32 fixed taken with one elementwise "
                  f"scale; W2 (64,) float32 varying"),
        "fn": boundary_graph,
        "specs": (evoxc.TensorSpec((rows, 64), f32),
                  evoxc.TensorSpec((64,), f32)),
        "fixed_leaves": [0],
        "arrays": (X, W2),
        "hand_prepare": hand_prepare,
        "hand_prepare_specs": (evoxc.TensorSpec((rows, 64), f32),),
        "hand_apply": hand_apply,
        "hand_apply_specs": (evoxc.TensorSpec((rows, 64), f32),
                             evoxc.TensorSpec((64,), f32)),
        "hand_apply_indices": [1],  # prepared + varying W2
        "batched": None,
        "batched_specs": None,
        "batched_arrays": None,
        "oracle": oracle,
    }


def _workload_c(evoxc, np, rows, seed):
    """Family C — state-variable shape: loop-free recurrence step."""
    rng = np.random.default_rng(seed)
    f32 = evoxc.float32

    @evoxc.defn
    def step(dataset, W1, state, W2):
        base = evoxc.matmul(dataset, W1)
        turn = evoxc.tanh(evoxc.matmul(state, W1))
        return evoxc.mean((base + turn) * W2)

    @evoxc.defn
    def hand_prepare(dataset, W1):
        return evoxc.matmul(dataset, W1)

    @evoxc.defn
    def hand_apply(base, W1, state, W2):
        turn = evoxc.tanh(evoxc.matmul(state, W1))
        return evoxc.mean((base + turn) * W2)

    dataset = rng.standard_normal((rows, 64)).astype(np.float32)
    W1 = (rng.standard_normal((64, 64)) * 0.1).astype(np.float32)
    state = (rng.standard_normal((rows, 64)) * 0.5).astype(np.float32)
    W2 = rng.standard_normal((64,)).astype(np.float32)

    def oracle(values):
        base = values[0] @ values[1]
        turn = np.tanh(values[2] @ values[1])
        return np.asarray(np.mean((base + turn) * values[3], dtype=np.float32),
                          dtype=np.float32)

    return {
        "workload": "C_state_variable",
        "label": "state-variable recurrence step",
        "expectation": "win",
        "rows": rows,
        "shape": (f"dataset ({rows}, 64) float32 + W1 (64, 64) float32 fixed; "
                  f"state ({rows}, 64) float32 + W2 (64,) float32 varying"),
        "fn": step,
        "specs": (evoxc.TensorSpec((rows, 64), f32),
                  evoxc.TensorSpec((64, 64), f32),
                  evoxc.TensorSpec((rows, 64), f32),
                  evoxc.TensorSpec((64,), f32)),
        "fixed_leaves": [0, 1],
        "arrays": (dataset, W1, state, W2),
        "hand_prepare": hand_prepare,
        "hand_prepare_specs": (evoxc.TensorSpec((rows, 64), f32),
                               evoxc.TensorSpec((64, 64), f32)),
        "hand_apply": hand_apply,
        "hand_apply_specs": (evoxc.TensorSpec((rows, 64), f32),
                             evoxc.TensorSpec((64, 64), f32),
                             evoxc.TensorSpec((rows, 64), f32),
                             evoxc.TensorSpec((64,), f32)),
        "hand_apply_indices": [1, 2, 3],  # prepared + W1 + state + W2
        "batched": None,
        "batched_specs": None,
        "batched_arrays": None,
        "oracle": oracle,
    }


# ---------------------------------------------------------------------------
# interleaved stage measurement
# ---------------------------------------------------------------------------
def _trace_all(evoxc, fns, reps):
    """Interleaved trace: {name: (fn, specs)} -> ({name: graph}, {name: spread})."""
    for fn, specs in fns.values():
        evoxc.trace(fn, *specs)  # untimed warmup: the first trace is lazy init
    samples = {name: [] for name in fns}
    graphs = {}
    for _ in range(reps):
        for name, (fn, specs) in fns.items():
            graphs[name], elapsed = _timed_once(lambda: evoxc.trace(fn, *specs))
            samples[name].append(elapsed)
    return graphs, {name: _spread(values) for name, values in samples.items()}


def _build_all(evoxc, graphs, backend, reps):
    """Interleaved ``lower -> compile -> load``: {name: (executable, spread)}."""
    for graph in graphs.values():
        evoxc.load(evoxc.compile(evoxc.lower(graph, backend=backend),
                                 backend=backend), backend=backend)
    samples = {name: [] for name in graphs}
    executables = {}
    for _ in range(reps):
        for name, graph in graphs.items():
            start = time.perf_counter()
            lowered = evoxc.lower(graph, backend=backend)
            artifact = evoxc.compile(lowered, backend=backend)
            executables[name] = evoxc.load(artifact, backend=backend)
            samples[name].append((time.perf_counter() - start) * 1e3)
    return {name: (executables[name], _spread(samples[name]))
            for name in graphs}


def _time_all(calls, reps):
    """Interleaved run timing: {name: callable} -> ({name: spread}, [rep dict]).

    Each repetition times EVERY call shape once, in the same order, and the
    per-repetition row is kept as well: the run times its own slow-moving host
    mode, so a ratio taken inside one repetition is the comparable quantity.
    """
    for call in calls.values():
        call()  # untimed warmup per call shape
    samples = {name: [] for name in calls}
    per_rep = []
    for _ in range(reps):
        row = {}
        for name, call in calls.items():
            elapsed = _timed_once(call)[1]
            samples[name].append(elapsed)
            row[name] = elapsed
        per_rep.append(row)
    return {name: _spread(values) for name, values in samples.items()}, per_rep


# ---------------------------------------------------------------------------
# plan reporting
# ---------------------------------------------------------------------------
def _boundary_rows(plan):
    rows = []
    for value in plan.boundary:
        rows.append({
            "is_input": bool(value.is_input),
            "defining_op": value.defining_op,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "nbytes": value.nbytes,
        })
    return rows


def _amortized(unsplit_total, split_total, hand_total, runs, reuse_counts):
    """Per-generation cost per reuse count, plus the analytic crossover."""
    rows = {}
    for reuse in reuse_counts:
        unsplit = unsplit_total / reuse + runs["unsplit"]
        split = ((split_total + runs["prepare"]) / reuse) + runs["apply"]
        hand = ((hand_total + runs["hand_prepare"]) / reuse) + runs["hand_apply"]
        rows[str(reuse)] = {
            "unsplit_ms": _ms(unsplit),
            "split_ms": _ms(split),
            "hand_written_ms": _ms(hand),
            "speedup_unsplit_over_split": _ratio(unsplit / split),
            "speedup_unsplit_over_hand_written": _ratio(unsplit / hand),
            "speedup_hand_written_over_split": _ratio(hand / split),
        }
    # Analytic crossover from the measured terms. With U = unsplit_total,
    # S = split_total, P = prepare run and A = apply run, the split wins when
    #   (S + P)/k + A  <  U/k + U_run   i.e.  (S + P - U)/k < U_run - A.
    numerator = split_total + runs["prepare"] - unsplit_total
    denominator = runs["unsplit"] - runs["apply"]
    analytic = None
    if denominator > 0:
        analytic = (1 if numerator <= 0
                    else int(math.ceil(numerator / denominator)))
    elif numerator < 0:
        analytic = 1  # apply alone is already at least as cheap as one run
    measured = None
    for reuse in reuse_counts:
        if rows[str(reuse)]["speedup_unsplit_over_split"] >= 1.0:
            measured = reuse
            break
    return rows, analytic, measured


def _amortized_ratio(unsplit_total, split_total, hand_total, per_rep,
                     reuse_counts):
    """Per-generation ratios from PER-REPETITION pairs, plus the crossover.

    Each repetition contributes one sample of every call shape, so
    ``unsplit_ms / split_ms`` computed inside that repetition divides two
    measurements taken in the same host mode; the median of those ratios is
    therefore the comparison that survives a bimodal shared host, where two
    formulations' separate medians can land in different modes.
    The build totals (``*_total``) are the same measured medians the other two
    tables use — they are one-off staging costs, not per-turn work.
    """
    gathered = {str(reuse): {"split": [], "hand": []} for reuse in reuse_counts}
    for row in per_rep:
        for reuse in reuse_counts:
            unsplit = unsplit_total / reuse + row["unsplit"]
            split = ((split_total + row["prepare"]) / reuse) + row["apply"]
            hand = ((hand_total + row["hand_prepare"]) / reuse) + row["hand_apply"]
            gathered[str(reuse)]["split"].append(unsplit / split)
            gathered[str(reuse)]["hand"].append(unsplit / hand)
    rows = {}
    for reuse in reuse_counts:
        key = str(reuse)
        rows[key] = {
            "speedup_unsplit_over_split":
                _ratio(statistics.median(gathered[key]["split"])),
            "speedup_unsplit_over_hand_written":
                _ratio(statistics.median(gathered[key]["hand"])),
            "pairs": len(gathered[key]["split"]),
        }
    measured = None
    for reuse in reuse_counts:
        if rows[str(reuse)]["speedup_unsplit_over_split"] >= 1.0:
            measured = reuse
            break
    return rows, measured


# ---------------------------------------------------------------------------
# one cell
# ---------------------------------------------------------------------------
def measure_cell(evoxc, np, case, backend, reps, reuse_counts, summary):
    """Measure one workload cell: split, unsplit, hand-written, batched."""
    key = f"{case['workload']}|{case['rows']}"
    arrays = case["arrays"]
    fixed_count = len(case["fixed_leaves"])

    # --- trace every function for this cell, interleaved
    fns = {
        "source": (case["fn"], case["specs"]),
        "hand_prepare": (case["hand_prepare"], case["hand_prepare_specs"]),
        "hand_apply": (case["hand_apply"], case["hand_apply_specs"]),
    }
    if case["batched"] is not None:
        fns["batched"] = (case["batched"], case["batched_specs"])
    graphs, trace_spreads = _trace_all(evoxc, fns, reps)
    graph = graphs["source"]
    source_digest = graph.digest()
    source_ops = _ops(graph)

    # --- the transform itself (cheap, CPU-light, not interleaved with stages)
    plan = None
    split_samples = []
    for _ in range(reps):
        plan, elapsed = _timed_once(
            lambda: evoxc.split_by_inputs(graph, fixed_leaves=case["fixed_leaves"])
        )
        split_samples.append(elapsed)
    split_ms = statistics.median(split_samples)
    source_unchanged = graph.digest() == source_digest
    if not source_unchanged:
        summary["failures"].append(f"{key}: source graph digest changed by the split")
    try:
        plan.verify()
    except Exception as exc:  # noqa: BLE001 - a wiring-law failure is a failure
        summary["failures"].append(f"{key}: SplitPlan.verify() {exc}")

    report = plan.summary()
    boundary_bytes = report["boundary_bytes"]
    boundary = _boundary_rows(plan)

    # --- stage every graph, interleaved
    build_graphs = {
        "prepare": plan.prepare,
        "apply": plan.apply,
        "source": graph,
        "hand_prepare": graphs["hand_prepare"],
        "hand_apply": graphs["hand_apply"],
    }
    if case["batched"] is not None:
        build_graphs["batched"] = graphs["batched"]
    staged = _build_all(evoxc, build_graphs, backend, reps)
    # `staged` maps name -> (executable, spread)
    prepare_exe, prepare_stats = staged["prepare"]
    apply_exe, apply_stats = staged["apply"]
    unsplit_exe, unsplit_stats = staged["source"]
    hand_prepare_exe, hand_prepare_stats = staged["hand_prepare"]
    hand_apply_exe, hand_apply_stats = staged["hand_apply"]
    prepare_build = prepare_stats["median_ms"]
    apply_build = apply_stats["median_ms"]
    unsplit_build = unsplit_stats["median_ms"]
    hand_prepare_build = hand_prepare_stats["median_ms"]
    hand_apply_build = hand_apply_stats["median_ms"]

    split_total = (trace_spreads["source"]["median_ms"] + split_ms
                   + prepare_build + apply_build)
    unsplit_total = trace_spreads["source"]["median_ms"] + unsplit_build
    hand_total = (trace_spreads["hand_prepare"]["median_ms"]
                  + trace_spreads["hand_apply"]["median_ms"]
                  + hand_prepare_build + hand_apply_build)

    # --- tensors and the observable call shapes
    tensors = [evoxc.from_numpy(array) for array in arrays]
    fixed_args = tensors[:fixed_count]
    varying = tuple(tensors[fixed_count:])

    prepared = evoxc.run(prepare_exe, *fixed_args)  # flat tuple of boundary tensors
    hand_prepared = evoxc.run(hand_prepare_exe, *fixed_args)
    # The hand-written `apply` takes the prepared value first, then whichever
    # SOURCE inputs it still needs (indices into `arrays`), which for family C
    # is the fixed matrix W1 the split ships across its boundary.
    hand_apply_args = (hand_prepared,) + tuple(
        tensors[index] for index in case["hand_apply_indices"])

    unsplit_output = evoxc.run(unsplit_exe, *tensors)
    split_output = evoxc.run(apply_exe, varying, prepared)
    hand_output = evoxc.run(hand_apply_exe, *hand_apply_args)

    # --- parity: every formulation against a plain numpy oracle
    want = case["oracle"](arrays)
    parity = {
        "unsplit": _max_abs_error(unsplit_output.numpy(), want),
        "split": _max_abs_error(split_output.numpy(), want),
        "hand_written": _max_abs_error(hand_output.numpy(), want),
    }
    for name, error in parity.items():
        if not (error <= PARITY_TOL):
            summary["failures"].append(
                f"{key}: {name} parity {error:g} > {PARITY_TOL:g}")
    device = str(unsplit_output.device)
    if backend == "numpy" and device != "Device(kind='cpu', index=0)":
        summary["failures"].append(
            f"{key}: numpy output on {device}, expected cpu:0")
    prepared_device = str(prepared[0].device) if prepared else device
    if backend == "numpy" and prepared_device != "Device(kind='cpu', index=0)":
        summary["failures"].append(
            f"{key}: prepare output on {prepared_device}, expected cpu:0")

    # --- timed runs, interleaved across every call shape
    calls = {
        "unsplit": lambda: evoxc.run(unsplit_exe, *tensors),
        "prepare": lambda: evoxc.run(prepare_exe, *fixed_args),
        "apply": lambda: evoxc.run(apply_exe, varying, prepared),
        "hand_prepare": lambda: evoxc.run(hand_prepare_exe, *fixed_args),
        "hand_apply": lambda: evoxc.run(hand_apply_exe, *hand_apply_args),
    }
    batched_exe = staged["batched"][0] if case["batched"] is not None else None
    batched_tensors = None
    if batched_exe is not None:
        batched_tensors = [evoxc.from_numpy(array)
                          for array in case["batched_arrays"]]
        calls["batched"] = lambda: evoxc.run(batched_exe, *batched_tensors)
    run_blocks, run_per_rep = _time_all(calls, reps)

    runs = {name: run_blocks[name]["median_ms"]
            for name in ("unsplit", "prepare", "apply",
                         "hand_prepare", "hand_apply")}
    amortized, crossover_analytic, crossover_measured = _amortized(
        unsplit_total, split_total, hand_total, runs, reuse_counts)
    # Two cross-checks on the medians, because this host samples the same call
    # anywhere from ~1.4 to ~59 ms: the min-based table, and the
    # per-repetition ratio table (see _meta.accounting) which cancels the mode.
    amortized_min, _, _ = _amortized(
        unsplit_total, split_total, hand_total,
        {name: block["min_ms"] for name, block in run_blocks.items()},
        reuse_counts)
    amortized_ratio, crossover_ratio = _amortized_ratio(
        unsplit_total, split_total, hand_total, run_per_rep, reuse_counts)
    unsplit_med = runs["unsplit"]
    prepare_med = runs["prepare"]
    apply_med = runs["apply"]

    # --- the competing strategy: one batched run for BATCH_CANDIDATES
    batch_block = None
    if batched_exe is not None:
        candidate_count = int(case["batched_specs"][-1].shape[0])
        batched_run = run_blocks["batched"]
        batched_total = (trace_spreads["batched"]["median_ms"]
                         + staged["batched"][1]["median_ms"])
        batched_output = calls["batched"]()
        split_64 = split_total + prepare_med + candidate_count * apply_med
        unsplit_64 = unsplit_total + candidate_count * unsplit_med
        batched_64 = batched_total + batched_run["median_ms"]
        candidates = np.asarray(case["batched_arrays"][-1])
        applies = np.stack([
            float(evoxc.run(
                apply_exe, (evoxc.from_numpy(candidates[position]),), prepared
            ).numpy())
            for position in range(candidate_count)
        ])
        stand_in_parity = _max_abs_error(batched_output.numpy(), applies)
        if not (stand_in_parity <= PARITY_TOL):
            summary["failures"].append(
                f"{key}: batched stand-in parity {stand_in_parity:g} > "
                f"{PARITY_TOL:g} against {candidate_count} applies")
        batch_block = {
            "status": "measured as a hand-written batched graph",
            "note": (
                "the graph vmap(fitness, in_axes=(None, None, 0)) expands to; "
                "vmap itself is unavailable in this worktree (see "
                "_meta.vmap), so this stand-in is validated against "
                f"{candidate_count} separate applies"
            ),
            "candidates": candidate_count,
            "batched_ops": _ops(graphs["batched"]),
            "build": {"trace_ms": trace_spreads["batched"]["median_ms"],
                      "staging": staged["batched"][1],
                      "total_ms": _ms(batched_total)},
            "run": batched_run,
            "parity_max_abs_err_vs_64_applies": stand_in_parity,
            "totals_for_candidates": {
                "split_64_applies_ms": _ms(split_64),
                "unsplit_64_runs_ms": _ms(unsplit_64),
                "batched_one_run_ms": _ms(batched_64),
                "speedup_batched_over_split_64": _ratio(split_64 / batched_64),
                "speedup_batched_over_unsplit_64": _ratio(unsplit_64 / batched_64),
            },
            "per_candidate_ms": {
                "split_64_applies": _ms(split_64 / candidate_count),
                "unsplit_64_runs": _ms(unsplit_64 / candidate_count),
                "batched_one_run": _ms(batched_64 / candidate_count),
            },
        }

    cell = {
        "workload": case["workload"],
        "label": case["label"],
        "expectation": case["expectation"],
        "rows": case["rows"],
        "shape": case["shape"],
        "backend": backend,
        "fixed_leaves": list(report["fixed_leaves"]),
        "varying_leaves": list(report["varying_leaves"]),
        "boundary_values": report["boundary_values"],
        "boundary_bytes": boundary_bytes,
        "boundary": boundary,
        "source_ops": source_ops,
        "prepare_ops": report["prepare_ops"],
        "apply_ops": report["apply_ops"],
        "dropped_ops": report["dropped_ops"],
        "unused_varying_leaves": list(report["unused_varying_leaves"]),
        "source_digest": source_digest,
        "prepare_digest": report["prepare_digest"],
        "apply_digest": report["apply_digest"],
        "source_digest_unchanged_by_split": source_unchanged,
        "split_summary": report,
        "hand_written": {
            "prepare_ops": _ops(graphs["hand_prepare"]),
            "apply_ops": _ops(graphs["hand_apply"]),
            "prepare_digest": graphs["hand_prepare"].digest(),
            "apply_digest": graphs["hand_apply"].digest(),
            "ops_match_split": (
                _ops(graphs["hand_prepare"]) == report["prepare_ops"]
                and _ops(graphs["hand_apply"]) == report["apply_ops"]
            ),
        },
        "build": {
            "trace_ms": trace_spreads["source"]["median_ms"],
            "split_transform_ms": _ms(split_ms),
            "prepare_ms": _ms(prepare_build),
            "apply_ms": _ms(apply_build),
            "hand_prepare_trace_ms": trace_spreads["hand_prepare"]["median_ms"],
            "hand_apply_trace_ms": trace_spreads["hand_apply"]["median_ms"],
            "hand_prepare_ms": _ms(hand_prepare_build),
            "hand_apply_ms": _ms(hand_apply_build),
            "unsplit_staging_ms": _ms(unsplit_build),
            "split_total_ms": _ms(split_total),
            "unsplit_total_ms": _ms(unsplit_total),
            "hand_written_total_ms": _ms(hand_total),
        },
        "trace_spread": trace_spreads,
        "build_spread": {
            "prepare": prepare_stats,
            "apply": apply_stats,
            "unsplit": unsplit_stats,
            "hand_prepare": hand_prepare_stats,
            "hand_apply": hand_apply_stats,
        },
        "run": run_blocks,
        "run_samples": {name: [_ms(rep[name]) for rep in run_per_rep]
                        for name in run_blocks},
        "boundary_overhead_ms_one_turn": _ms(
            prepare_med + apply_med - unsplit_med),
        "amortized": amortized,
        "amortized_from_min": amortized_min,
        "amortized_per_repetition_ratio": amortized_ratio,
        "crossover_reuse_analytic": crossover_analytic,
        "crossover_reuse_measured": crossover_measured,
        "crossover_reuse_from_per_repetition_ratio": crossover_ratio,
        "reuse_counts": list(reuse_counts),
        "parity_max_abs_err": parity,
        "batched_vmap_stand_in": batch_block,
        "output_device": device,
        "prepared_device": prepared_device,
        "reps": reps,
    }
    _print_cell(cell)
    return cell


def _ratio_row(rows, key):
    """Format a per-reuse ratio row as ``1:0.84 4:2.46 16:6.89 64:13.42``."""
    return " ".join(f"{reuse}:{row[key]:.2f}" for reuse, row in rows.items())


def _print_cell(cell):
    print(f"\n[{cell['workload']}] rows={cell['rows']} backend={cell['backend']}")
    print(f"  boundary {cell['boundary_values']} value(s), "
          f"{cell['boundary_bytes']} bytes; ops "
          f"source={cell['source_ops']} prepare={cell['prepare_ops']} "
          f"apply={cell['apply_ops']} (hand "
          f"{cell['hand_written']['prepare_ops']}/"
          f"{cell['hand_written']['apply_ops']})")
    build = cell["build"]
    print(f"  build: trace {build['trace_ms']:.3f} split "
          f"{build['split_transform_ms']:.3f} prepare {build['prepare_ms']:.3f} "
          f"apply {build['apply_ms']:.3f} | split_total "
          f"{build['split_total_ms']:.3f} unsplit_total "
          f"{build['unsplit_total_ms']:.3f} hand_total "
          f"{build['hand_written_total_ms']:.3f}")
    run = cell["run"]
    print(f"  run(median): unsplit {run['unsplit']['median_ms']:.3f} prepare "
          f"{run['prepare']['median_ms']:.3f} apply "
          f"{run['apply']['median_ms']:.3f} | hand "
          f"{run['hand_prepare']['median_ms']:.3f}"
          f"+{run['hand_apply']['median_ms']:.3f}")
    print(f"  {'reuse':>6}{'unsplit':>10}{'split':>10}{'hand':>10}"
          f"{'x_split':>9}{'x_hand':>9}")
    for reuse, row in cell["amortized"].items():
        print(f"  {reuse:>6}{row['unsplit_ms']:>10.3f}{row['split_ms']:>10.3f}"
              f"{row['hand_written_ms']:>10.3f}"
              f"{row['speedup_unsplit_over_split']:>9.2f}"
              f"{row['speedup_unsplit_over_hand_written']:>9.2f}")
    print(f"  crossover reuse: analytic {cell['crossover_reuse_analytic']} "
          f"measured {cell['crossover_reuse_measured']}")
    print(f"  per-repetition ratio (mode-cancelling): "
          f"x_split {_ratio_row(cell['amortized_per_repetition_ratio'], 'speedup_unsplit_over_split')}"
          f" x_hand {_ratio_row(cell['amortized_per_repetition_ratio'], 'speedup_unsplit_over_hand_written')}"
          f" -> crossover {cell['crossover_reuse_from_per_repetition_ratio']}")
    batch = cell["batched_vmap_stand_in"]
    if batch:
        totals = batch["totals_for_candidates"]
        print(f"  batched stand-in ({batch['candidates']} candidates): "
              f"one run {totals['batched_one_run_ms']:.3f} vs "
              f"{batch['candidates']} applies "
              f"{totals['split_64_applies_ms']:.3f} vs "
              f"{batch['candidates']} unsplit runs "
              f"{totals['unsplit_64_runs_ms']:.3f} (x"
              f"{totals['speedup_batched_over_split_64']:.2f} / x"
              f"{totals['speedup_batched_over_unsplit_64']:.2f})")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _parse_ints(raw):
    chunks = [raw] if isinstance(raw, str) else raw
    return [int(item.strip()) for chunk in chunks
            for item in chunk.split(",") if item.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the deterministic inputs and candidate "
                             "population (default 0)")
    parser.add_argument("--reps", type=int, default=REPS_DEFAULT,
                        help=f"repetitions per measurement, min/median/max "
                             f"(default {REPS_DEFAULT})")
    parser.add_argument("--reuse", default=REUSE_DEFAULT,
                        help=f"comma-separated N_apply reuse counts "
                             f"(default {REUSE_DEFAULT})")
    parser.add_argument("--sizes", default=SIZES_DEFAULT,
                        help=f"comma-separated dataset row counts for "
                             f"families A and C (default {SIZES_DEFAULT})")
    parser.add_argument("--boundary-size", type=int,
                        default=BOUNDARY_SIZE_DEFAULT,
                        help=f"dataset row count for family B "
                             f"(default {BOUNDARY_SIZE_DEFAULT})")
    parser.add_argument("--workloads", default="a,b,c",
                        help="comma-separated family letters (default a,b,c)")
    parser.add_argument("--threads", type=int, default=1,
                        help="pin the BLAS/OpenMP thread pools to this many "
                             "threads before numpy is imported (0 = leave the "
                             "ambient environment alone); see _meta.threading")
    parser.add_argument("--backends", nargs="+", default=["numpy"],
                        help="backend names, comma- or space-separated "
                             "(default numpy)")
    parser.add_argument("--out", default=None,
                        help="JSON results path; runs.<label> is written while "
                             "other labels are preserved")
    parser.add_argument("--label", default=None,
                        help="label of this run under runs.<label> in --out")
    parser.add_argument("--json", action="store_true",
                        help="print the full result document to stdout")
    args = parser.parse_args(argv)
    if args.reps < 1:
        parser.error("--reps must be >= 1")
    if args.threads < 0:
        parser.error("--threads must be >= 0 (0 leaves the ambient pools)")
    reuse_counts = _parse_ints(args.reuse)
    if not reuse_counts or any(count < 1 for count in reuse_counts):
        parser.error("--reuse must be a non-empty list of positive integers")
    sizes = _parse_ints(args.sizes)
    if not sizes or any(size < 1 for size in sizes):
        parser.error("--sizes must be a non-empty list of positive integers")
    if args.boundary_size < 1:
        parser.error("--boundary-size must be a positive integer")
    families = [item.strip().lower() for item in args.workloads.split(",")
                if item.strip()]
    if not families or any(family not in {"a", "b", "c"} for family in families):
        parser.error("--workloads must name families from {a, b, c}")
    requested_backends = [item.strip() for chunk in args.backends
                          for item in chunk.split(",") if item.strip()]
    if not requested_backends:
        parser.error("--backends must name at least one backend")
    label = args.label or f"seed{args.seed}"

    # Bench policy: location capture is per-node trace overhead, not part of
    # what this probe compares. Set BEFORE evoxc is imported.
    os.environ.setdefault("EVOXC_DISABLE_LOCATIONS", "1")
    # Thread-pool control. This host reports more CPUs to Python
    # (os.cpu_count / sched_getaffinity) than `nproc` says are usable, so the
    # ambient numpy/OpenBLAS pool oversubscribes it and a BLAS-bound cell comes
    # out bimodal, the same work sampling anywhere from a few to tens of
    # milliseconds. Pinning the pools puts every formulation on the same
    # 1-thread footing; the effective values are recorded in _meta.threading.
    thread_vars = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    threading = {
        "requested": args.threads,
        "pinned": args.threads > 0,
        "cpu_count_reported": os.cpu_count(),
        "cpu_affinity": (len(os.sched_getaffinity(0))
                         if hasattr(os, "sched_getaffinity") else None),
        "nproc_reported": _nproc(),
        "effective_env": {},
        "why": (
            "this host reports more CPUs to Python than the cgroup actually "
            "grants (os.cpu_count vs nproc), so the ambient BLAS/OpenMP pools "
            "oversubscribe it and a BLAS-bound cell comes out bimodal, the "
            "same work sampling anywhere from a few to tens of milliseconds; "
            "the pools are pinned so two formulations are compared on the "
            "same footing, and the effective values are recorded here"
        ),
    }
    if args.threads > 0:
        for name in thread_vars:
            os.environ[name] = str(args.threads)
    threading["effective_env"] = {name: os.environ.get(name)
                                  for name in thread_vars}
    import numpy as np
    import evoxc
    import evoxc.backends as backends

    versions = _versions()
    loadavg_start = _loadavg()
    print("== split_by_inputs decision probe ==")
    print(f"seed={args.seed} reps={args.reps} reuse={reuse_counts} "
          f"sizes={sizes} boundary_size={args.boundary_size} "
          f"workloads={families} backends={requested_backends} label={label}")
    print(f"python={versions['python']} numpy={versions['numpy']} "
          f"evoxc={versions['evoxc']} native={versions['native_extension']}")
    print(f"platform={versions['platform']} cpus={versions['cpu_count']} "
          f"cpu={versions['cpu_model']}")
    print(f"EVOXC_DISABLE_LOCATIONS={os.environ['EVOXC_DISABLE_LOCATIONS']}")
    print(f"threads={args.threads} (nproc {threading['nproc_reported']}, "
          f"os.cpu_count {threading['cpu_count_reported']}, affinity "
          f"{threading['cpu_affinity']}) env={threading['effective_env']}")
    print(f"loadavg(start)={loadavg_start}")

    summary = {"failures": []}
    available, unavailable = [], {}
    for backend in requested_backends:
        try:
            backends.get(backend)
        except Exception as exc:  # noqa: BLE001 - a missing backend is data
            reason = f"{type(exc).__name__}: {exc}"
            unavailable[backend] = reason
            print(f"BACKEND_UNAVAILABLE: {backend} ({reason})")
            summary["failures"].append(
                f"backend {backend!r} unavailable: {reason}")
            continue
        available.append(backend)

    vmap_status = {"status": "unavailable", "reason": None}
    try:
        from evoxc import _native
        if _native.available():
            vmap_status = {"status": "available",
                           "reason": "native extension "
                                     f"{_native.version()} is loadable"}
        else:
            vmap_status["reason"] = (
                "the Rust extension evoxc._evoxc_native is not built in this "
                "worktree and every builtin transform (vmap included) "
                "delegates to it with no Python fallback; the batched "
                "comparison is a hand-written graph validated against 64 "
                "separate applies"
            )
    except Exception as exc:  # noqa: BLE001 - data, not a failure
        vmap_status["reason"] = f"{type(exc).__name__}: {exc}"
    print(f"vmap: {vmap_status['status']}")

    run = {
        "label": label,
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
        "command": "python evoxc/bench/probe_split_by_inputs.py "
                   f"--seed {args.seed} --reps {args.reps} "
                   f"--reuse {','.join(str(c) for c in reuse_counts)} "
                   f"--sizes {','.join(str(s) for s in sizes)} "
                   f"--boundary-size {args.boundary_size} "
                   f"--workloads {','.join(families)} "
                   f"--threads {args.threads} "
                   f"--backends {','.join(requested_backends)}"
                   + (f" --out {_repo_relative(args.out)}" if args.out else ""),
        "versions": versions,
        "seed": args.seed,
        "reps": args.reps,
        "reuse_counts": reuse_counts,
        "sizes": sizes,
        "boundary_size": args.boundary_size,
        "batch_candidates": BATCH_CANDIDATES,
        "backends": requested_backends,
        "available_backends": available,
        "unavailable_backends": unavailable,
        "vmap": vmap_status,
        "threading": threading,
        "protocol": (
            "every timed stage is interleaved per repetition (trace, build and "
            "run each measure all formulations once per repetition, in the "
            "same order) so a contention episode on this shared host is spread "
            "across formulations; each call shape gets one untimed warmup"
        ),
        "env": {"EVOXC_DISABLE_LOCATIONS":
                os.environ["EVOXC_DISABLE_LOCATIONS"]},
        "accounting": {
            "unsplit_per_generation_ms":
                "unsplit_total_ms / N_apply + unsplit run (median)",
            "split_per_generation_ms":
                "(split_total_ms + prepare run median) / N_apply + apply run "
                "(median)",
            "hand_written_per_generation_ms":
                "(hand_written_total_ms + hand prepare run median) / N_apply + "
                "hand apply run median",
            "unsplit_total_ms":
                "trace(source) + lower+compile+load(source)",
            "split_total_ms":
                "trace(source) + split_by_inputs + lower+compile+load("
                "prepare) + lower+compile+load(apply)",
            "hand_written_total_ms":
                "trace(hand prepare) + trace(hand apply) + "
                "lower+compile+load(both)",
            "crossover_reuse_analytic":
                "ceil((split_total_ms + prepare_run - unsplit_total_ms) / "
                "(unsplit_run - apply_run)), i.e. the k above which "
                "(split_total + prepare_run)/k + apply_run drops below "
                "unsplit_total/k + unsplit_run",
            "crossover_reuse_measured":
                "the first N_apply in --reuse whose measured speedup >= 1.00",
            "amortized_from_min":
                "the same table recomputed from each call shape's min sample, "
                "as a cross-check on the medians",
            "amortized_per_repetition_ratio":
                "THE PRIMARY TABLE: for every repetition the ratio "
                "unsplit_per_generation / split_per_generation (and / "
                "hand_written_per_generation) is computed from that "
                "repetition's own samples and the reported number is the "
                "median of those --reps ratios; one repetition times every "
                "call shape, so a host mode shared by the whole repetition "
                "cancels, while two formulations' separate medians can land in "
                "different modes (see _meta.shared_host_caveat)",
            "crossover_reuse_from_per_repetition_ratio":
                "the first N_apply in --reuse whose median per-repetition "
                "speedup >= 1.00",
            "run_samples":
                "every raw per-repetition sample of every call shape, in "
                "issue order, so the mode mixture behind each table is "
                "auditable rather than hidden behind min/median/max",
        },
        "loadavg_start": loadavg_start,
        "cells": {},
        "failures": summary["failures"],
        "status": "PASS",
    }

    builders = {"a": _workload_a, "b": _workload_b, "c": _workload_c}
    for backend in available:
        for family in families:
            if family == "b":
                cases = [_workload_b(evoxc, np, args.boundary_size, args.seed)]
            else:
                cases = [builders[family](evoxc, np, rows, args.seed)
                         for rows in sizes]
            for case in cases:
                key = f"{case['workload']}|{case['rows']}"
                run["cells"][key] = measure_cell(
                    evoxc, np, case, backend, args.reps, reuse_counts, summary)
    if not available:
        print("\nno available backend: nothing measured")

    run["loadavg_end"] = _loadavg()
    run["failures"] = summary["failures"]
    run["status"] = "PASS" if not summary["failures"] else "FAIL"

    meta = {
        "probe": "evoxc/bench/probe_split_by_inputs.py",
        "date_utc": run["date_utc"],
        "platform": versions["platform"],
        "machine": versions["machine"],
        "cpu_model": versions["cpu_model"],
        "cpu_count": versions["cpu_count"],
        "python": versions["python"],
        "numpy": versions["numpy"],
        "evoxc": versions["evoxc"],
        "native_extension": versions["native_extension"],
        "seed": args.seed,
        "reps": args.reps,
        "reuse_counts": reuse_counts,
        "sizes": sizes,
        "boundary_size": args.boundary_size,
        "batch_candidates": BATCH_CANDIDATES,
        "backends": requested_backends,
        "available_backends": available,
        "unavailable_backends": unavailable,
        "vmap": vmap_status,
        "threading": threading,
        "loadavg_start": loadavg_start,
        "loadavg_end": run["loadavg_end"],
        "env": {"EVOXC_DISABLE_LOCATIONS":
                os.environ["EVOXC_DISABLE_LOCATIONS"]},
        "last_label": label,
        "labels": None,  # filled in from the merged document below
        "shared_host_caveat": SHARED_HOST_CAVEAT,
        "limits": LIMITS,
        "protocol": run["protocol"],
        "accounting": run["accounting"],
        "note": (
            "the top-level _meta describes the most recent run; every "
            "runs.<label> entry carries its own date_utc/command/versions/"
            "loadavg snapshot so older labels stay self-describing"
        ),
    }

    document = _write_json(args.out, label, run, meta)
    if args.out:
        document["_meta"]["labels"] = sorted(document["runs"])
        _write_json(args.out, label, run, document["_meta"])
        print(f"\nwrote {_repo_relative(args.out)} (label {label}, "
              f"labels {sorted(document['runs'])})")

    print(f"\nloadavg(end)={run['loadavg_end']}")
    print(f"STATUS: {run['status']}")
    for failure in summary["failures"]:
        print(f"  FAILURE: {failure}")
    print("SUMMARY_JSON: " + json.dumps(
        {key: value for key, value in run.items() if key != "cells"},
        sort_keys=True))
    if args.json:
        print(json.dumps(document, indent=2, sort_keys=True))
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    sys.exit(main())
