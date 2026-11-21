"""Compile-pipeline performance evidence probe (deterministic, public API only).

Measures the cost of every stage of the explicit evoxc pipeline on synthetic
graphs of a requested size, so a change to tracing, lowering, compilation,
loading, the memoized-build sugar or the native transform seam can be compared
against a recorded baseline on the same host:

* per (size, backend) cell, min and median over ``--reps`` of ``evoxc.trace``,
  ``evoxc.lower``, ``evoxc.compile``, ``evoxc.load``, the derived
  ``staging_total = lower + compile + load`` and the warm ``evoxc.run``
  (after one untimed warmup run) — plus the ACTUAL op count of the traced
  module (``--sizes`` names the REQUESTED traced-op count; the chain shape
  decides the exact number and both are recorded);
* end-to-end ``evoxc.build(fn, spec, count, backend=...)`` (cold, once per
  repetition) and a ``evoxc.build_cached`` pair on a fresh ``CompileCache``:
  the first call (a miss — the full pipeline) and the second (a hit, expected
  to be ~0);
* ``evoxc.vmap(graph, 0)`` and ``evoxc.grad(graph, 0)`` on a ~750-op traced
  graph — the native-transform seam, measured once per run because it does not
  depend on the chosen backend.

The graph is an ``@evoxc.defn`` that applies ``x = tanh(x) * x`` a STATIC
Python ``int`` number of times (two IR ops per iteration) and then a full
``sum`` — so ``count`` is a static argument that specializes the graph, and the
run call is ``evoxc.run(exe, x, count)``.

``EVOXC_DISABLE_LOCATIONS=1`` is forced in this process when it is not already
set (bench policy: location capture is per-node trace overhead that is not part
of what is being compared); the effective value is recorded in ``_meta``.

Run it explicitly (this module is evidence tooling — ``evoxc`` never imports it
and pytest never collects it)::

    source scripts/dev-env.sh
    .venv/bin/python -m evoxc.bench.probe_compile_perf --seed 0 --label before \
        --out evoxc/bench/compile_perf_results.json --json

Recorded artefacts: ``evoxc/bench/compile_perf_results.json`` (``_meta`` plus
one entry per ``--label`` under ``runs``) and the narrated
``evoxc/bench/compile_perf_findings.md``.
A re-run with a new ``--label`` PRESERVES every other label in the file, so an
``after`` measurement lands beside the ``before`` baseline it is compared with;
the top-level ``_meta`` always describes the most recent run while each
``runs.<label>`` entry carries its own ``date_utc``/``command``/``loadavg``/
``versions`` snapshot, so an older label stays self-describing.

The host is shared: every number is min/median over ``--reps`` with the load
average recorded, and the cells are indicative rather than isolated-hardware
measurements. A genuine correctness failure (empty graph, a run whose values
diverge from the numpy reference, a non-cpu output device on the numpy
backend) exits nonzero; an unavailable backend requested via ``--backends``
(e.g. ``native``, which has no registered backend in this tree) is recorded as
unavailable and does not fail the run.

Backends (``--backends``, comma- or space-separated, default ``numpy``): any
name registered with ``evoxc.backends.get``; ``native`` is accepted for
symmetry with a future native backend and prints
``native backend unavailable`` when it is not registered.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time

REPS_DEFAULT = 7  # min + median over this many repetitions
SIZES_DEFAULT = "200,1700,5000"  # REQUESTED traced-op counts
TRANSFORM_OPS = 750  # ~750-op graph for the vmap/grad seam
INPUT_SHAPE = (8, 16)  # small enough that run cost is interpreter overhead
PARITY_TOL = 1e-6  # numpy backend: same kernels, so this is generous

SHARED_HOST_CAVEAT = (
    "the host is shared and load-dependent: every cell is min/median over "
    "--reps with the load average recorded, one timing process at a time, so "
    "the numbers are indicative rather than isolated-hardware measurements"
)


# ---------------------------------------------------------------------------
# small timing / formatting helpers
# ---------------------------------------------------------------------------
def _timed(fn, reps):
    """Run ``fn`` ``reps`` times; return (last_result, min_ms, median_ms)."""
    samples = []
    result = None
    for _ in range(reps):
        start = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - start) * 1e3)
    return result, min(samples), statistics.median(samples)


def _timed_once(fn):
    """Run ``fn`` once; return (result, elapsed_ms)."""
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1e3


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

    return {
        "python": sys.version.split()[0],
        "numpy": _dist("numpy"),
        "evoxc": _dist("evoxc"),
        "native_extension": _native.version() if _native.available() else None,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
    }


# ---------------------------------------------------------------------------
# deterministic graph builders (public API only)
# ---------------------------------------------------------------------------
def _chain_iterations(requested_ops):
    """Iterations of ``x = tanh(x) * x`` giving ~``requested_ops`` IR ops."""
    return max(1, requested_ops // 2)


def _make_chain(iterations):
    """Two IR ops per iteration (constant ``tanh`` is hoisted by the tracer)."""
    import evoxc

    @evoxc.defn
    def chain(x, count):
        y = x
        for _ in range(count):
            y = evoxc.tanh(y) * y
        return evoxc.sum(y)

    return chain


def _chain_input():
    import numpy as np

    return np.full(INPUT_SHAPE, 0.5, dtype=np.float32)


def _chain_reference(iterations):
    """float32 numpy reference for the same graph (same kernels, float32)."""
    import numpy as np

    y = _chain_input()
    for _ in range(iterations):
        y = np.tanh(y) * y
    return np.asarray(y.sum(), dtype=np.float32)


def _op_count(graph):
    return len(graph.module.main.entry_block.ops)


def _max_abs_error(got, want):
    import numpy as np

    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    if got.shape != want.shape:
        raise AssertionError(f"shape mismatch: {got.shape} != {want.shape}")
    return float(np.max(np.abs(got - want)))


# ---------------------------------------------------------------------------
# backend resolution
# ---------------------------------------------------------------------------
def _resolve_backends(evoxc, requested, summary):
    """Registered backend names; record (do not fail on) an unavailable one."""
    import evoxc.backends as backends

    available = []
    unavailable = {}
    for name in requested:
        try:
            backends.get(name)
        except Exception as exc:  # noqa: BLE001 - a missing backend is data
            reason = f"{type(exc).__name__}: {exc}"
            unavailable[name] = reason
            if name == "native":
                print(f"{name} backend unavailable ({reason})")
            else:
                print(f"BACKEND_UNAVAILABLE: {name} ({reason})")
                summary["failures"].append(f"backend {name!r} unavailable: {reason}")
            continue
        available.append(name)
    return available, unavailable


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def measure_cell(evoxc, backend, requested_ops, reps, summary):
    """One (size, backend) cell: the four stages, warm run, build, cached build."""
    iterations = _chain_iterations(requested_ops)
    fn = _make_chain(iterations)
    spec = evoxc.TensorSpec(INPUT_SHAPE, evoxc.float32)
    x = evoxc.from_numpy(_chain_input())
    reference = _chain_reference(iterations)

    trace_ms, lower_ms, compile_ms, load_ms, staging_ms = [], [], [], [], []
    graph = artifact = executable = None
    for _ in range(reps):
        # One repetition runs the whole explicit pipeline, timing each stage on
        # the previous stage's output (so trace/lower/compile/load all see the
        # real input of a fresh graph, never a cached intermediate).
        graph, t = _timed_once(lambda: evoxc.trace(fn, spec, iterations))
        trace_ms.append(t)
        lowered, t = _timed_once(lambda: evoxc.lower(graph, backend=backend))
        lower_ms.append(t)
        artifact, t = _timed_once(lambda: evoxc.compile(lowered, backend=backend))
        compile_ms.append(t)
        executable, t = _timed_once(lambda: evoxc.load(artifact, backend=backend))
        load_ms.append(t)
        staging_ms.append(lower_ms[-1] + compile_ms[-1] + load_ms[-1])

    op_count = _op_count(graph)
    if not op_count:
        summary["failures"].append(f"{backend}/{requested_ops}: empty traced graph")

    # Warm run: one untimed warmup, then --reps timed runs on the same input.
    output = evoxc.run(executable, x, iterations)
    evoxc.run(executable, x, iterations)
    _, run_min, run_median = _timed(
        lambda: evoxc.run(executable, x, iterations), reps
    )
    parity = _max_abs_error(output.numpy(), reference)
    if parity > PARITY_TOL:
        summary["failures"].append(
            f"{backend}/{requested_ops}: run parity {parity:g} > {PARITY_TOL:g}"
        )
    device = str(output.device)
    if backend == "numpy" and device != "Device(kind='cpu', index=0)":
        summary["failures"].append(
            f"{backend}/{requested_ops}: numpy output on {device}, expected cpu:0"
        )

    # End-to-end shorthand: cold trace -> lower -> compile -> load.
    _, build_min, build_median = _timed(
        lambda: evoxc.build(fn, spec, iterations, backend=backend), reps
    )
    built = evoxc.build(fn, spec, iterations, backend=backend)
    built_parity = _max_abs_error(
        evoxc.run(built, x, iterations).numpy(), reference
    )
    if built_parity > PARITY_TOL:
        summary["failures"].append(
            f"{backend}/{requested_ops}: build parity {built_parity:g} "
            f"> {PARITY_TOL:g}"
        )

    # Memoized shorthand: a fresh cache isolates this cell; miss then hit.
    cache = evoxc.CompileCache()
    cached, miss_ms = _timed_once(
        lambda: evoxc.build_cached(fn, spec, iterations, backend=backend,
                                   cache=cache)
    )
    hit_result, hit_min, hit_median = _timed(
        lambda: evoxc.build_cached(fn, spec, iterations, backend=backend,
                                   cache=cache),
        reps,
    )
    if hit_result is not cached:
        summary["failures"].append(
            f"{backend}/{requested_ops}: build_cached hit returned a new object"
        )
    if cache.misses != 1 or cache.hits != reps:
        summary["failures"].append(
            f"{backend}/{requested_ops}: build_cached bookkeeping "
            f"hits={cache.hits} misses={cache.misses} (expected {reps}/1)"
        )

    return {
        "backend": backend,
        "requested_ops": requested_ops,
        "iterations": iterations,
        "op_count": op_count,
        "reps": reps,
        "trace": {"min_ms": min(trace_ms), "median_ms": statistics.median(trace_ms)},
        "lower": {"min_ms": min(lower_ms), "median_ms": statistics.median(lower_ms)},
        "compile": {
            "min_ms": min(compile_ms),
            "median_ms": statistics.median(compile_ms),
        },
        "load": {"min_ms": min(load_ms), "median_ms": statistics.median(load_ms)},
        "staging_total": {
            "min_ms": min(staging_ms),
            "median_ms": statistics.median(staging_ms),
        },
        "warm_run": {
            "warmup_runs": 1,
            "min_ms": run_min,
            "median_ms": run_median,
            "output_device": device,
        },
        "build": {"min_ms": build_min, "median_ms": build_median},
        "build_cached": {
            "miss_ms": miss_ms,
            "hit": {"min_ms": hit_min, "median_ms": hit_median},
            "hits": cache.hits,
            "misses": cache.misses,
        },
        "max_abs_err": parity,
        "build_max_abs_err": built_parity,
        "loadavg": _loadavg(),
    }


def measure_transforms(evoxc, reps, summary):
    """``vmap`` / ``grad`` on a ~750-op traced graph (the native seam)."""
    iterations = _chain_iterations(TRANSFORM_OPS)
    graph = evoxc.trace(
        _make_chain(iterations), evoxc.TensorSpec(INPUT_SHAPE, evoxc.float32),
        iterations,
    )
    op_count = _op_count(graph)
    cases = {
        "vmap": lambda: evoxc.vmap(graph, 0),
        "grad": lambda: evoxc.grad(graph, 0),
    }
    print(f"\n[transforms] input graph {op_count} IR ops "
          f"({iterations} iterations, n={reps})")
    print(f"{'transform':<12}{'min_ms':>10}{'median_ms':>12}{'out_ops':>10}")
    out = {}
    for name, fn in cases.items():
        result, low, median = _timed(fn, reps)
        out_ops = _op_count(result)
        print(f"{name:<12}{low:>10.2f}{median:>12.2f}{out_ops:>10}")
        out[name] = {"min_ms": low, "median_ms": median, "out_ops": out_ops}
        if not out_ops:
            summary["failures"].append(f"{name} produced an empty graph")
    out["graph_op_count"] = op_count
    out["loadavg"] = _loadavg()
    return out


# ---------------------------------------------------------------------------
# result document
# ---------------------------------------------------------------------------
def _repo_relative(path):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    absolute = os.path.abspath(path)
    try:
        rel = os.path.relpath(absolute, root)
    except ValueError:  # different drive — leave the path as given
        return path
    return rel if not rel.startswith("..") else path


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


def _print_table(cell):
    print(
        f"\n[{cell['backend']}] requested {cell['requested_ops']} ops -> "
        f"{cell['op_count']} IR ops ({cell['iterations']} iterations, "
        f"n={cell['reps']})"
    )
    print(f"{'stage':<16}{'min_ms':>10}{'median_ms':>12}")
    rows = (
        ("trace", cell["trace"]),
        ("lower", cell["lower"]),
        ("compile", cell["compile"]),
        ("load", cell["load"]),
        ("staging_total", cell["staging_total"]),
        ("build", cell["build"]),
        ("build_cached(miss)", {"min_ms": cell["build_cached"]["miss_ms"],
                                "median_ms": cell["build_cached"]["miss_ms"]}),
        ("build_cached(hit)", cell["build_cached"]["hit"]),
        ("warm_run", cell["warm_run"]),
    )
    for name, row in rows:
        print(f"{name:<16}{row['min_ms']:>10.3f}{row['median_ms']:>12.3f}")
    print(
        f"parity max_abs_err {cell['max_abs_err']:.3e} "
        f"(build {cell['build_max_abs_err']:.3e}), "
        f"output {cell['warm_run']['output_device']}"
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _parse_list(raw):
    """Flatten comma-separated values from a string or a list of strings."""
    chunks = [raw] if isinstance(raw, str) else raw
    return [item.strip() for chunk in chunks for item in chunk.split(",")
            if item.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0,
                        help="seed recorded in _meta (graphs are static, not "
                             "random; default 0)")
    parser.add_argument("--reps", type=int, default=REPS_DEFAULT,
                        help=f"repetitions per measurement, min + median "
                             f"(default {REPS_DEFAULT})")
    parser.add_argument("--sizes", default=SIZES_DEFAULT,
                        help=f"comma-separated REQUESTED traced-op counts "
                             f"(default {SIZES_DEFAULT})")
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
    sizes = [int(value) for value in _parse_list(args.sizes)]
    if not sizes or any(size <= 0 for size in sizes):
        parser.error("--sizes must be a non-empty list of positive integers")
    requested_backends = _parse_list(args.backends)
    if not requested_backends:
        parser.error("--backends must name at least one backend")
    label = args.label or "default"

    # Bench policy: location capture is per-node trace overhead, not part of
    # what this probe compares. Set BEFORE evoxc is imported.
    os.environ.setdefault("EVOXC_DISABLE_LOCATIONS", "1")

    import evoxc

    versions = _versions()
    loadavg_start = _loadavg()
    print("== compile pipeline performance probe ==")
    print(f"seed={args.seed} reps={args.reps} sizes={sizes} "
          f"backends={requested_backends} label={label}")
    print(f"python={versions['python']} numpy={versions['numpy']} "
          f"evoxc={versions['evoxc']} native={versions['native_extension']}")
    print(f"platform={versions['platform']} cpus={versions['cpu_count']} "
          f"cpu={versions['cpu_model']}")
    print(f"EVOXC_DISABLE_LOCATIONS={os.environ['EVOXC_DISABLE_LOCATIONS']}")
    print(f"loadavg(start)={loadavg_start}")

    summary = {"failures": []}
    available, unavailable = _resolve_backends(
        evoxc, requested_backends, summary
    )

    run = {
        "label": label,
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "command": "python -m evoxc.bench.probe_compile_perf "
                   f"--seed {args.seed} --reps {args.reps} "
                   f"--sizes {','.join(str(size) for size in sizes)} "
                   f"--backends {','.join(requested_backends)}"
                   + (f" --out {_repo_relative(args.out)}" if args.out else "")
                   + f" --label {label}",
        "versions": versions,
        "seed": args.seed,
        "reps": args.reps,
        "sizes": sizes,
        "backends": requested_backends,
        "available_backends": available,
        "unavailable_backends": unavailable,
        "graph": (
            "evoxc.tanh(x) * x applied `count` times (a STATIC Python int "
            "argument, two IR ops per iteration) then evoxc.sum(x); input "
            f"{INPUT_SHAPE} float32; --sizes names the REQUESTED traced-op "
            "count, op_count is the actual traced module size"
        ),
        "timing": (
            "min and median over --reps; each stage rep runs trace -> lower -> "
            "compile -> load chained in one repetition; staging_total is the "
            "per-repetition sum lower+compile+load; warm run = one untimed "
            "warmup then --reps timed runs; build_cached miss is one cold call "
            "on a fresh CompileCache and its hit is min/median over --reps"
        ),
        "env": {"EVOXC_DISABLE_LOCATIONS": os.environ["EVOXC_DISABLE_LOCATIONS"]},
        "loadavg_start": loadavg_start,
        "cells": {},
        "transforms": None,
        "failures": summary["failures"],
        "status": "PASS",
    }

    for backend in available:
        for size in sizes:
            cell = measure_cell(evoxc, backend, size, args.reps, summary)
            run["cells"][f"{backend}|{size}"] = cell
            _print_table(cell)

    if available:
        run["transforms"] = measure_transforms(evoxc, args.reps, summary)
    else:
        print("\n[transforms] skipped (no available backend)")

    run["loadavg_end"] = _loadavg()
    run["status"] = "PASS" if not summary["failures"] else "FAIL"

    meta = {
        "probe": "evoxc/bench/probe_compile_perf.py",
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
        "sizes": sizes,
        "backends": requested_backends,
        "available_backends": available,
        "unavailable_backends": unavailable,
        "loadavg_start": loadavg_start,
        "loadavg_end": run["loadavg_end"],
        "env": {"EVOXC_DISABLE_LOCATIONS": os.environ["EVOXC_DISABLE_LOCATIONS"]},
        "last_label": label,
        "labels": None,  # filled in from the merged document below
        "shared_host_caveat": SHARED_HOST_CAVEAT,
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
    print("SUMMARY_JSON: " + json.dumps(run, sort_keys=True))
    if args.json:
        print(json.dumps(document, indent=2, sort_keys=True))
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    sys.exit(main())
