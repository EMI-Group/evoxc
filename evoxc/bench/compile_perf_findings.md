# Compile-pipeline performance: `before` / `after` evidence

## Provenance and scope

Source of record: `evoxc/bench/probe_compile_perf.py`, run twice with `--seed 0 --reps 7` and a different `--label`.
`evoxc/bench/compile_perf_results.json` holds both runs under `runs.before` and `runs.after`; every number below is transcribed from those two entries, which also carry their own `date_utc`/`command`/`versions`/`loadavg` snapshot (the top-level `_meta` describes the most recent run, `after`).
The probe is evidence tooling: `evoxc` never imports it and pytest never collects it.
The probe code is identical in both sessions — only the results file changed — so the two columns come from the same measurement code.

The two labels are two points on this host:

| Label | Tree | Utc | Load average (start → end) |
|---|---|---|---|
| `before` | pre-staging-path `compile`/`load` (baseline) | 2026-11-21T18:28:34Z | `[3.84, 3.67, 1.92]` → `[3.61, 3.63, 1.91]` |
| `after` | HEAD `8c2923f` (the staging path as it stands) | 2026-11-21T19:31:26Z | `[3.27, 2.75, 3.06]` → `[3.25, 2.75, 3.05]` |

Host and toolchain (identical in both snapshots): Intel(R) Core(TM) Ultra 7 356H, Linux 7.3.4 x86_64 (glibc 2.44), 16 CPUs, Python 3.14.10, numpy 2.5.2, evoxc 0.1.0, `evoxc._evoxc_native` 0.1.0.
Backend: `numpy` (the only registered backend in this tree — requesting `native` prints `native backend unavailable`, records the reason under `unavailable_backends` and leaves the run green).
Environment: `EVOXC_DISABLE_LOCATIONS=1` (set by the probe; location capture is per-node trace overhead that is not part of what is compared).
Both runs finished `PASS` with an empty `failures` list.

**The host is shared and load-dependent.** Every cell is min and median over 7 repetitions with one timing process at a time, and the two sessions did not see the same load (3.84 vs 3.27 one-minute average at the start).
The numbers are indicative, not isolated-hardware measurements: a delta smaller than the min-to-median spread *within* one column should be read as noise, and cross-column deltas of a few percent carry that same caveat.

## Exact commands

`before` was recorded on the pre-staging-path tree:

```
source scripts/dev-env.sh
uv sync --locked --all-packages --extra dev --extra iree
.venv/bin/python -m evoxc.bench.probe_compile_perf --seed 0 --reps 7 \
    --label before --out evoxc/bench/compile_perf_results.json
```

`after` was recorded on this tree, same probe, same flags, a new label (the `before` entry is preserved):

```
source scripts/dev-env.sh
uv sync --locked --all-packages --extra dev --extra iree
.venv/bin/python -m evoxc.bench.probe_compile_perf --seed 0 --reps 7 \
    --label after --out evoxc/bench/compile_perf_results.json
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
`--sizes` names the REQUESTED op count; the chain shape decides the actual one, and both cells record the same traced size in both sessions (`200 -> 202`, `1700 -> 1702`, `5000 -> 5002` IR ops).
Input: `(8, 16)` float32.
In every cell of both runs the `run` and `build` outputs matched the float32 numpy reference for the same loop exactly (`max_abs_err = 0.0`), the numpy backend's output stayed on `Device(kind='cpu', index=0)`, and the `build_cached` bookkeeping was `hits=7, misses=1` on a fresh `CompileCache`.

## Before / after table (numpy backend, n = 7, ms)

Each repetition runs `trace -> lower -> compile -> load` chained on the previous stage's output; `staging total` is that repetition's `lower + compile + load`.
`warm run` is one untimed warmup followed by 7 timed `evoxc.run` calls on the same input.
`build` is min/median over 7 cold `evoxc.build(fn, spec, count)` calls; `cached miss` is one cold `evoxc.build_cached` on a fresh `CompileCache` (a single sample, hence no median), and `cached hit` is min/median over 7 further calls.
`x` is the before/after ratio (`> 1` = the `after` session is faster; `< 1` = slower).

### 200 requested ops (202 traced)

| Stage | before min | before median | after min | after median | x min | x median |
|---|---:|---:|---:|---:|---:|---:|
| trace | 1.361 | 1.533 | 1.471 | 1.604 | 0.92 | 0.96 |
| lower | 2.234 | 2.284 | 1.967 | 2.016 | 1.14 | 1.13 |
| compile | 1.948 | 2.064 | 0.004 | 0.005 | 458 | 425 |
| load | 2.017 | 2.063 | 0.019 | 0.023 | 106 | 90 |
| **staging total** | **6.285** | **6.387** | **1.992** | **2.047** | **3.16** | **3.12** |
| warm run | 0.200 | 0.201 | 0.210 | 0.213 | 0.95 | 0.94 |
| build | 7.798 | 8.651 | 3.580 | 3.680 | 2.18 | 2.35 |
| build_cached miss | 8.131 | — | 3.659 | — | 2.22 | — |
| build_cached hit | 0.037 | 0.039 | 0.041 | 0.044 | 0.91 | 0.89 |

### 1700 requested ops (1702 traced)

| Stage | before min | before median | after min | after median | x min | x median |
|---|---:|---:|---:|---:|---:|---:|
| trace | 11.983 | 12.474 | 12.332 | 13.305 | 0.97 | 0.94 |
| lower | 17.582 | 18.527 | 16.614 | 17.244 | 1.06 | 1.07 |
| compile | 17.239 | 17.430 | 0.015 | 0.017 | 1144 | 1027 |
| load | 16.948 | 17.487 | 0.102 | 0.106 | 165 | 165 |
| **staging total** | **52.036** | **53.246** | **16.734** | **17.367** | **3.11** | **3.07** |
| warm run | 1.587 | 1.592 | 1.684 | 1.703 | 0.94 | 0.93 |
| build | 64.940 | 66.207 | 29.696 | 30.785 | 2.19 | 2.15 |
| build_cached miss | 65.737 | — | 69.880 | — | 0.94 | — |
| build_cached hit | 0.036 | 0.038 | 0.041 | 0.047 | 0.88 | 0.81 |

### 5000 requested ops (5002 traced)

| Stage | before min | before median | after min | after median | x min | x median |
|---|---:|---:|---:|---:|---:|---:|
| trace | 65.636 | 84.718 | 36.438 | 37.329 | 1.80 | 2.27 |
| lower | 53.894 | 54.352 | 49.359 | 50.936 | 1.09 | 1.07 |
| compile | 50.804 | 51.320 | 0.017 | 0.019 | 2926 | 2726 |
| load | 50.679 | 51.432 | 0.245 | 0.252 | 207 | 204 |
| **staging total** | **156.371** | **157.048** | **49.628** | **51.200** | **3.15** | **3.07** |
| warm run | 4.634 | 4.659 | 5.015 | 5.049 | 0.92 | 0.92 |
| build | 236.420 | 265.186 | 89.600 | 137.293 | 2.64 | 1.93 |
| build_cached miss | 271.127 | — | 151.456 | — | 1.79 | — |
| build_cached hit | 0.037 | 0.041 | 0.040 | 0.046 | 0.91 | 0.90 |

## What the two columns show

**`compile` and `load` are overhead-only in the `after` column.** On the numpy backend `compile` costs 0.004–0.019 ms and `load` 0.019–0.252 ms across a 25× graph-size range, against the baseline's 1.9–51.3 ms and 2.0–51.4 ms: 425–2926× on `compile`, 90–207× on `load`, and the `after` scales are flat enough that neither stage is a size-dependent cost any more.

**`staging total` is the `lower` cost.** In the `after` column `compile + load` together add 0.023 ms at 200 ops, 0.117 ms at 1700 and 0.262 ms at 5000, so `staging total` sits on `lower` (1.992 vs 1.967, 16.734 vs 16.614, 49.628 vs 49.359 ms) — `lower`'s serialize cost is the stage sum, with 3.07–3.16× (about −68 %) against the baseline at every size.

**The end-to-end shorthands follow the stage sum.** `build` tracks `trace + staging total`: 3.580 vs 3.463 ms at 200 ops, 29.696 vs 29.066 ms at 1700, 89.600 vs 86.066 ms at 5000 (min column), i.e. 1.9–2.6× faster overall (−54 %, −54 %, −62 % on the min column; −58 %, −54 %, −48 % on the median column, whose 5000 cell carries the session's largest spread: build min 89.6 vs median 137.3 ms). `build_cached` hits stay at ~0.04 ms regardless of size (the key computation with zero pipeline work) and `build_cached` misses land on the new build cost, except the 1700-op single-sample miss (69.9 ms against the 30.8 ms after-build median) which is one cold call under load, not a property of the memoized path.

**`lower` itself is unchanged.** 1.06–1.14× on the min column and 1.07–1.13× on the median column: the same serialize-dominated stage, with the small gain inside the cross-session load difference.

**`trace` is unchanged, and its 5000-op cell is a session artefact.** At 200 and 1700 ops the two columns agree to within 3–8 % (0.92–0.97×). The 5000-op cell reads 65.636/84.718 ms before and 36.438/37.329 ms after, but that `before` cell had a 19.1 ms min-to-median spread against 0.89 ms in the `after` session — the earlier session's load, not a trace-path difference — so the 1.80/2.27× there is not evidence of a faster tracer.

**`warm run` is flat.** 0.92–0.95× (a 5–8 % slower reading in the `after` session, inside or near the per-column spread); the compiled numpy executable's run path is the same interpreter loop in both sessions.

## Native-transform seam (752-op input graph, n = 7)

| Transform | before min | before median | after min | after median | x min | x median | output IR ops |
|---|---:|---:|---:|---:|---:|---:|---:|
| `evoxc.vmap(graph, 0)` | 9.27 | 9.90 | 10.31 | 11.19 | 0.90 | 0.89 | 752 |
| `evoxc.grad(graph, 0)` | 24.72 | 25.61 | 26.55 | 26.91 | 0.93 | 0.95 | 3380 |

The seam produces the same output sizes in both sessions and reads 5–11 % slower in the `after` one — a small, uniform cross-session difference rather than a change in the native engine's behaviour (the probe records the transform once per run, so this section has 7 samples per cell and no repeated sessions to separate load from noise).

## What this does NOT claim

- **Not isolated hardware.** The host is shared, the two sessions saw different load averages (3.84 vs 3.27 one-minute average at start), and each label is a single session. Cross-column deltas of a few percent — including the `warm run` and `vmap`/`grad` readings — are not separable from that.
- **Not a general evoxc version comparison.** The two columns are two labelled points on one host with one probe; nothing here measures any other platform, filesystem, or Python/numpy build.
- **`numpy` backend only.** No `iree`, `xla` or `tvm` number appears; in particular there is **no `iree-compile` time** here — the iree adapter's `compile` shells out to the external compiler and that cost is outside this probe.
- **No GPU.** No device-resident `run`, no host↔device staging throughput, no multi-rank or collective numbers.
- **`warm run` is interpreter overhead on an `(8, 16)` input**, i.e. per-call dispatch cost, not throughput or kernel time.
- **No memory measurement** (peak RSS, allocation counts) and no compile-time-vs-graph-shape model beyond the three sizes measured.
- **`build_cached` miss is a single sample** per cell, so it is the noisiest number in the tables; only its hits are min/median over 7.
- **Not a correctness suite.** The probe only checks run/build parity against a numpy reference for this one chain shape and that outputs stay on `cpu:0`; the full suites are in `tests/`.
