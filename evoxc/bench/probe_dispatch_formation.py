"""IREE dispatch-formation probe on evoxc.batch_functions merged population graphs.

Reproduces the EvoGP GP-engine merged-graph shape -- a deduped population of
random expression programs over ONE shared f32 input, merged with
evoxc.batch_functions (bands, function axis = population), split into N chunk
bodies and compiled concurrently exactly like the EvoGP batch engine -- and
measures, per iree-compile dispatch-formation configuration and per target:

  * the dispatch / executable count (hal.executable.export count from
    `iree-compile --compile-to=executable-sources` on the same text + flags),
  * the per-dispatch maximum binding count (the HIP/CUDA 16-binding wall),
  * cold compile wall time (per chunk and summed over chunk bodies),
  * warm per-call time (chunks invoked back to back, timed run + synchronize,
    median and best of --reps),
  * parity vs the numpy backend (max-abs error).

Configurations (--configs): default, fuse_multi_use, aggressive_fusion, both,
inline_const0, reshape_move, all_three, no_multi_use, o3, o0, o3_both.
Targets (--targets): llvm-cpu, rocm.

Results for the measured matrix are recorded in
`evoxc/bench/dispatch_formation_results.json` and narrated in
`evoxc/bench/dispatch_formation_findings.md`.

Usage (needs iree compiler+runtime, and a GPU for the rocm target):

    EVOXC_DISABLE_LOCATIONS=1 python evoxc/bench/probe_dispatch_formation.py \
        --pop 1000 --rows 1024 --targets llvm-cpu rocm \
        --configs default aggressive_fusion --reps 5 --out timed.json

    # dispatch count only (fast: --compile-to=executable-sources, no load)
    python evoxc/bench/probe_dispatch_formation.py --pop 1000 --count-only --out count.json
"""
import random

import evoxc

UNARY = ("square", "abs", "tanh", "sigmoid", "negate", "exp")
BINARY = ("add", "subtract", "multiply", "maximum", "minimum")


def gen_expr(rng, p_leaf=0.42, p_unary=0.33, depth_cap=5):
    def build(depth):
        r = rng.random()
        if depth >= depth_cap or r < p_leaf:
            return ("x",)
        r = rng.random()
        if r < p_unary:
            return (rng.choice(UNARY), build(depth + 1))
        op = rng.choice(BINARY)
        if rng.random() < 0.30:
            return (op, build(depth + 1), ("c", round(rng.uniform(0.5, 2.0), 3)))
        return (op, build(depth + 1), build(depth + 1))
    return build(0)


def count_nodes(expr):
    if expr[0] in ("x", "c"):
        return 1
    return 1 + sum(count_nodes(a) for a in expr[1:])


def ev(expr, x):
    head = expr[0]
    if head == "x":
        return x
    if head == "c":
        return expr[1]
    return getattr(evoxc, head)(*[ev(a, x) for a in expr[1:]])


def make_fn(expr, idx):
    def program(x):
        return ev(expr, x)
    program.__name__ = f"prog_{idx}"
    return program


def population(n, seed=0, p_leaf=0.42):
    rng = random.Random(seed)
    seen, exprs = {}, []
    for _ in range(n):
        e = gen_expr(rng, p_leaf=p_leaf)
        key = repr(e)
        if key not in seen:
            seen[key] = len(exprs)
            exprs.append(e)
    return exprs


import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

import evoxc
from evoxc.backends.adapters import iree as iree_adapt


def iree_compile_path():
    """Path to the `iree-compile` binary that ships with the installed iree.

    Resolved from the imported `iree.compiler` package so the probe works in any
    venv that has iree installed, with an explicit `EVOXC_IREE_COMPILE`
    override for unusual layouts.
    """
    override = os.environ.get("EVOXC_IREE_COMPILE")
    if override:
        return override
    import iree.compiler as _iree_compiler
    return os.path.join(os.path.dirname(os.path.abspath(_iree_compiler.__file__)),
                        "_mlir_libs", "iree-compile")


#: name -> (iree_compile_args, opt_level or None)
CONFIGS = {
    "default": ([], None),
    "fuse_multi_use": (["--iree-dispatch-creation-fuse-multi-use"], None),
    "aggressive_fusion": (
        ["--iree-dispatch-creation-enable-aggressive-fusion"], None),
    "both": (["--iree-dispatch-creation-fuse-multi-use",
              "--iree-dispatch-creation-enable-aggressive-fusion"], None),
    "inline_const0": (["--iree-flow-inline-constants-max-byte-length=0"], None),
    "reshape_move": (["--iree-dispatch-creation-enable-aggressive-reshape-movement"],
                     None),
    "all_three": (["--iree-dispatch-creation-fuse-multi-use",
                   "--iree-dispatch-creation-enable-aggressive-fusion",
                   "--iree-dispatch-creation-enable-aggressive-reshape-movement"],
                  None),
    "no_multi_use": (["--iree-dispatch-creation-fuse-multi-use=false"], None),
    "o3": ([], "O3"),
    "o0": ([], "O0"),
    "o3_both": (["--iree-dispatch-creation-fuse-multi-use",
                 "--iree-dispatch-creation-enable-aggressive-fusion"], "O3"),
}


def count_ops(module):
    total = 0

    def walk(region):
        nonlocal total
        for block in region.blocks:
            for op in block.ops:
                total += 1
                for reg in getattr(op, "regions", ()) or ():
                    walk(reg)

    for fn in module.functions:
        walk(fn.region)
    return total


def plan_k(n_funcs, band_size=16, cap=15):
    k = max(1, math.ceil(n_funcs / band_size))
    return min(k, cap)


def build_chunks(n_programs, seed, p_leaf, rows, chunks, band_size, cap):
    exprs = population(n_programs, seed, p_leaf)
    fns = [make_fn(e, i) for i, e in enumerate(exprs)]
    spec = evoxc.TensorSpec((rows,), evoxc.float32)
    size = math.ceil(len(fns) / chunks)
    out = []
    for c in range(chunks):
        part = fns[c * size : (c + 1) * size]
        if not part:
            continue
        t0 = time.perf_counter()
        g = evoxc.batch_functions(part, axes=None, out_axes=0,
                                  bands=plan_k(len(part), band_size, cap))(spec)
        dt = time.perf_counter() - t0
        out.append({"graph": g, "n_funcs": len(part), "merge_ms": dt * 1000.0,
                    "ops": count_ops(g.module),
                    "bands": g.batch_report.bands,
                    "band_sizes": list(g.batch_report.band_sizes)})
    return out, len(exprs)


def hal_counts(mlir_text, flags, target, workdir):
    """(n_executable, n_dispatch, hal_text_bytes) via --compile-to=hal."""
    ic = iree_compile_path()
    path = os.path.join(workdir, "in.mlir")
    out = os.path.join(workdir, "hal.mlir")
    with open(path, "w") as fh:
        fh.write(mlir_text)
    cmd = [ic, path, "--iree-input-type=stablehlo",
           f"--iree-hal-target-backends={target}",
           "--compile-to=executable-sources", "-o", out]
    if target == "rocm":
        arch = iree_adapt._detect_rocm_target() or "gfx942"
        cmd.append(f"--iree-rocm-target={arch}")
    cmd += list(flags)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = (time.perf_counter() - t0) * 1000.0
    if proc.returncode != 0:
        return {"hal_ok": False, "hal_ms": dt,
                "hal_err": (proc.stderr or proc.stdout)[-1500:]}
    txt = open(out).read()
    exports = [ln for ln in txt.splitlines() if "hal.executable.export" in ln]
    per_export_bindings = [ln.count("hal.pipeline.binding<") for ln in exports]
    return {"hal_ok": True, "hal_ms": dt,
            "n_executable": txt.count("hal.executable "),
            "n_export": len(exports),
            "max_bindings": max(per_export_bindings) if per_export_bindings else 0,
            "hal_bytes": len(txt)}


def _first(obj):
    """A single-output run returns the Tensor itself; keep both shapes."""
    return obj[0] if isinstance(obj, (tuple, list)) else obj


def numpy_parity_graphs(chunk_graphs, x_host, seeds=(0, 1)):
    """Reference values from the numpy backend for each chunk graph."""
    refs = []
    for c in chunk_graphs:
        lowered = evoxc.lower(c["graph"], backend="numpy")
        art = evoxc.compile(lowered, backend="numpy")
        run_exe = evoxc.load(art, backend="numpy")
        refs.append([np.asarray(_first(evoxc.run(run_exe, x_host)).numpy())])
    return refs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=1000)
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--band-size", type=int, default=16)
    ap.add_argument("--band-cap", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-leaf", type=float, default=0.16)
    ap.add_argument("--targets", default="llvm-cpu,rocm")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--no-run", action="store_true")
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--no-parity", action="store_true")
    ap.add_argument("--device-index", type=int, default=0)
    args = ap.parse_args(argv)

    results = {"config": vars(args), "cells": {}}
    if os.path.exists(args.out):
        try:
            results = json.load(open(args.out))
        except Exception:
            pass

    t0 = time.perf_counter()
    chunks, distinct = build_chunks(args.pop, args.seed, args.p_leaf, args.rows,
                                   args.chunks, args.band_size, args.band_cap)
    results["plan"] = {
        "distinct_programs": distinct,
        "chunks": [{k: v for k, v in c.items() if k != "graph"} for c in chunks],
        "total_ops": sum(c["ops"] for c in chunks),
        "build_ms": (time.perf_counter() - t0) * 1000.0,
    }
    print(json.dumps(results["plan"], indent=1), flush=True)

    x_np = np.random.default_rng(7).random(args.rows, dtype=np.float32)
    x_host = evoxc.tensor(x_np)

    refs = None
    if not args.no_parity:
        refs = numpy_parity_graphs(chunks, x_host)
        results["numpy_ref"] = [[float(v.sum()) for v in rs] for rs in refs]
        print("numpy refs done", flush=True)

    for target in args.targets.split(","):
        lowered = [evoxc.lower(c["graph"], backend="iree", target_backends=[target])
                   for c in chunks]
        dev = None
        if target == "rocm":
            dev = evoxc.Device("rocm", args.device_index)
        else:
            dev = evoxc.Device("cpu", 0)
        x_dev = x_host.to(dev) if target != "llvm-cpu" else x_host
        for name in args.configs.split(","):
            flags, opt_level = CONFIGS[name]
            key = f"{target}|{name}"
            print(f"=== {key} flags={flags}", flush=True)
            cell = {"target": target, "config": name, "flags": flags,
                    "opt_level": opt_level}
            wd = tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "/tmp"))
            try:
                counts = [hal_counts(lo.text(), flags, target, wd) for lo in lowered]
                cell["hal"] = counts
                cell["n_executable"] = sum(c.get("n_executable", 0) for c in counts)
                cell["n_export"] = sum(c.get("n_export", 0) for c in counts)
                cell["hal_ms"] = sum(c.get("hal_ms", 0.0) for c in counts)
                cell["max_bindings"] = max((c.get("max_bindings", 0) for c in counts),
                                           default=0)
                cell["hal_ok"] = all(c.get("hal_ok") for c in counts)
                if not cell["hal_ok"]:
                    cell["hal_err"] = [c.get("hal_err", "") for c in counts
                                       if not c.get("hal_ok")][:1]
                print(f"  hal: exec={cell['n_executable']} "
                      f"dispatch={cell['n_export']} "
                      f"maxbind={cell['max_bindings']} ok={cell['hal_ok']} "
                      f"{cell['hal_ms']:.0f}ms", flush=True)

                if args.count_only:
                    results["cells"][key] = cell
                    json.dump(results, open(args.out, "w"), indent=1)
                    continue
                t0 = time.perf_counter()
                compile_opts = {"iree_compile_args": list(flags)}
                if opt_level is not None:
                    compile_opts["opt_level"] = opt_level
                arts = [evoxc.compile(lo, backend="iree", target_backends=[target],
                                      **compile_opts) for lo in lowered]
                cell["compile_ms"] = (time.perf_counter() - t0) * 1000.0
                cell["compile_ok"] = True
                print(f"  compile: {cell['compile_ms']:.0f} ms", flush=True)

                exes = [evoxc.load(a, backend="iree", device=dev) for a in arts]
                if not args.no_run:
                    outs = [evoxc.run(e, x_dev) for e in exes]
                    for e in exes:
                        e.synchronize()
                    samples = []
                    for _ in range(args.warmup + args.reps):
                        t0 = time.perf_counter()
                        outs = [evoxc.run(e, x_dev) for e in exes]
                        for e in exes:
                            e.synchronize()
                        samples.append((time.perf_counter() - t0) * 1000.0)
                    warm = samples[args.warmup:]
                    cell["warm_ms_median"] = statistics.median(warm)
                    cell["warm_ms_best"] = min(warm)
                    cell["warm_samples"] = warm
                    print(f"  warm: {cell['warm_ms_median']:.3f} ms "
                          f"(best {cell['warm_ms_best']:.3f})", flush=True)
                    if refs is not None:
                        dev_err = []
                        for out, ref in zip(outs, refs):
                            got = np.asarray(
                                _first(out).to(evoxc.core.Device("cpu", 0)).numpy())
                            dev_err.append(float(np.max(np.abs(got - ref[0]))))
                        cell["max_abs_vs_numpy"] = max(dev_err)
                        print(f"  parity max_abs={cell['max_abs_vs_numpy']:.3e}",
                              flush=True)
            except Exception as exc:  # noqa: BLE001 - record and continue
                cell["error"] = f"{type(exc).__name__}: {exc}"
                cell["traceback"] = traceback.format_exc()[-2000:]
                print(f"  ERROR {cell['error']}", flush=True)
            results["cells"][key] = cell
            json.dump(results, open(args.out, "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
