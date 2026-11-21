# Compile-pipeline performance: `before` baseline

## Provenance and scope

Source of record: `evoxc/bench/probe_compile_perf.py`, run with `--seed 0 --reps 7 --label before`.
All numbers below are transcribed from that probe's stdout; `evoxc/bench/compile_perf_results.json` carries the same values under `runs.before` (plus the `_meta` provenance).
The probe is evidence tooling: `evoxc` never imports it and pytest never collects it.

Host: Intel(R) Core(TM) Ultra 7 356H, Linux 7.3.4 x86_64 (glibc 2.44), 16 CPUs, Python 3.14.10, numpy 2.5.2.
Versions: evoxc 0.1.0, `evoxc._evoxc_native` 0.1.0.
Backend: `numpy` (the only registered backend in this tree — requesting `native` prints `native backend unavailable`, records the reason under `unavailable_backends` and leaves the run green).
Environment: `EVOXC_DISABLE_LOCATIONS=1` (set by the probe; location capture is per-node trace overhead and is excluded from the comparison).
Load average: `[3.84, 3.67, 1.92]` at the start and `[3.61, 3.63, 1.91]` at the end; the 5000-op cell ran at `[3.61, 3.63, 1.91]`.
Utc date of the run: 2026-11-21T18:28:34Z.

**The host is shared and load-dependent.** Every measurement is min and median over 7 repetitions with one timing process at a time, so these are indicative numbers, not isolated-hardware ones — the 5000-op `trace` median (84.7 ms) sitting well above its min (65.6 ms) is exactly that load, not a property of tracing.

## Exact command

```
source scripts/dev-env.sh
uv sync --locked --all-packages --extra dev --extra iree
.venv/bin/python -m evoxc.bench.probe_compile_perf --seed 0 --reps 7 \
    --label before --out evoxc/bench/compile_perf_results.json
```

The measurement matrix is `--sizes 200,1700,5000` (the default) on `--backends numpy` (the default).

## Graph

An `@evoxc.defn` of the form

```python
@evoxc.defn
def chain(x, count):
    y = x
    for _ in range(count):
        y = evoxc.tanh(y) * y
    return evoxc.sum(y)
```

`count` is a STATIC Python `int` (it specializes the graph, so `evoxc.run(exe, x, count)` passes it on every call), each iteration is two IR ops (`tanh`, `multiply`) and the graph ends in one full `sum`.
`--sizes` names the REQUESTED op count; the chain shape decides the actual one, which is recorded per cell (`200 -> 202`, `1700 -> 1702`, `5000 -> 5002` IR ops).
Input: `(8, 16)` float32; every run output matched the float32 numpy reference for the same loop exactly (`max_abs_err = 0.0`, also for the `evoxc.build` executable), and the numpy backend's output stayed on `Device(kind='cpu', index=0)`.

## Baseline table (numpy backend, n = 7)

Each repetition runs `trace -> lower -> compile -> load` chained on the previous stage's output; `staging total` is that repetition's `lower + compile + load`.
`warm run` is one untimed warmup followed by 7 timed `evoxc.run` calls on the same input.
`build` is the cold `evoxc.build(fn, spec, count)` shorthand; `cached` is `evoxc.build_cached` on a fresh `CompileCache` — the miss is one cold call, the hit is min/median over 7 further calls (all returning the same `Executable` object).

| Ops (requested → traced) | Stage | min (ms) | median (ms) |
|---|---:|---:|---:|
| 200 → 202 | trace | 1.361 | 1.533 |
| | lower | 2.234 | 2.284 |
| | compile | 1.948 | 2.064 |
| | load | 2.017 | 2.063 |
| | **staging total** | **6.285** | **6.387** |
| | warm run | 0.200 | 0.201 |
| | build | 7.798 | 8.651 |
| | build_cached miss / hit | 8.131 / 0.037 | — / 0.039 |
| 1700 → 1702 | trace | 11.983 | 12.474 |
| | lower | 17.582 | 18.527 |
| | compile | 17.239 | 17.430 |
| | load | 16.948 | 17.487 |
| | **staging total** | **52.036** | **53.246** |
| | warm run | 1.587 | 1.592 |
| | build | 64.940 | 66.207 |
| | build_cached miss / hit | 65.737 / 0.036 | — / 0.038 |
| 5000 → 5002 | trace | 65.636 | 84.718 |
| | lower | 53.894 | 54.352 |
| | compile | 50.804 | 51.320 |
| | load | 50.679 | 51.432 |
| | **staging total** | **156.371** | **157.048** |
| | warm run | 4.634 | 4.659 |
| | build | 236.420 | 265.186 |
| | build_cached miss / hit | 271.127 / 0.037 | — / 0.041 |

`build` tracks `trace + staging total` (7.7 vs 7.7 ms at 200 ops, 64.9 vs 64.0 ms at 1700, 236.4 vs 222.0 ms at 5000); the shorthand adds no hidden work.
`build_cached` hits cost ~0.04 ms regardless of graph size — the key computation itself, with zero pipeline work.

## Native-transform seam (numpy backend, 752-op input graph, n = 7)

| Transform | min (ms) | median (ms) | output IR ops |
|---|---:|---:|---:|
| `evoxc.vmap(graph, 0)` | 9.27 | 9.90 | 752 |
| `evoxc.grad(graph, 0)` | 24.72 | 25.61 | 3380 |

`vmap` maps an input that the chain immediately multiplies, so the batch dims fold back into the same op count; `grad` expands the chain's derivatives.

## What will be re-measured

A later run adds a labelled section to the same JSON — `runs.after` — by re-running the identical command with `--label after` (other labels are preserved, so `before` and `after` sit side by side), and documents the delta next to this table.
The comparison covers: the four stage times and `staging total` per size, the cold `evoxc.build`, the `build_cached` miss and hit, the warm `evoxc.run`, and the `vmap`/`grad` seam — i.e. everything the pipeline itself owns between a static program and a loaded executable.
Because the host is shared, a delta smaller than the run-to-run spread between the `min` and `median` columns above should be read as noise, and any re-run must record its own load average.
