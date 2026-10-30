# IREE dispatch formation on `batch_functions` merged-population graphs

Probe: `evoxc/bench/probe_dispatch_formation.py` (self-describing; `--help`).
Raw records: `evoxc/bench/dispatch_formation_results.json` (cells keyed `<target>|<config>`).
Reproduction inputs: a throwaway venv's Python 3.12 (`rocvenv`, a machine-local uv venv created for the run — never a repo path), iree-base 3.11.0 compiler+runtime, numpy 2.5.4, a ROCm device = AMD MI300X VF (gfx942) at `Device("rocm", 0)` → iree HAL device id 1, and the probe's fixed `--seed 0` (identical program population on every run).
`EVOXC_DISABLE_LOCATIONS=1` throughout; exactly one timing process at a time.

## Question

A `evoxc.batch_functions` merged population graph fragments into ~one IREE executable per
program (recorded medians 15 / 15 / 21.7 ops per executable on three real EvoGP graphs), and the
warm call is dispatch-bound (`run` = 84–90 % of the per-call time). Which `iree-compile`
dispatch-formation knob reduces the dispatch count, and what does it cost cold and on the row axis?

## Graph under test

The recorded GP shape, reproduced with `evoxc.batch_functions`:
~1000 deduped random expression programs (`p_leaf=0.16`, depth cap 5, 782 distinct at pop 1000),
each over ONE shared `(rows,)` f32 input (`axes=None`), a single f32 output, `out_axes=0`,
`bands` capped at 15, split into 4 chunk bodies and compiled/run like the EvoGP engine does.
pop 1000 / rows 1024 = 7892 emitted ops; pop 4000 = 3121 distinct programs / 28 546 ops.

IREE forms dispatches after lowering, so **the dispatch count is identical on `llvm-cpu` and
`rocm`** for the same MLIR — confirmed on every configuration below (596/548 and 2191/2021 on both),
which is why the tables list one dispatch column for both targets.

## pop 1000 / rows 1024 (782 distinct programs, 4 chunks × ~196 functions, 7892 ops)

| config | dispatches | max bindings | compile ms (rocm) | warm rocm ms (median) | warm llvm-cpu ms (median) |
|---|---|---|---|---|---|
| adapter default | 596 | 6 | 11691 | 2.439 | 117.24 |
| `+enable-aggressive-fusion` | 548 | 9 | 11404 | 2.220 | 118.02 |
| `+aggressive +fuse-multi-use` | 548 | 9 | 11478 | 2.299 | 119.00 |
| `--iree-flow-inline-constants-max-byte-length=0` | 591 | 7 | 12616 | 2.510 | 119.66 |
| `opt_level="O3"` | 596 | 6 | 11456 | 2.298 | 119.02 |
| `opt_level="O0"` | 596 | 6 | 11803 | 2.489 | 118.14 |
| `opt_level="O3"` + aggressive | 548 | 9 | 11405 | 2.322 | 117.06 |

Independent replicate of the top two rows (rocm): 2.454 → 2.226 ms (−9.3 %), 596 → 548 dispatches.
Per-chunk dispatches 137/151/159/149 → 126/137/145/140.
Results are unchanged by the flag in every cell (the merge's parity column is identical), i.e. this
is purely a scheduling knob.

Dispatch-count-only sweep (no load, `--compile-to=executable-sources`): `fuse_multi_use` 596,
`aggressive_reshape_movement` 596, all three flags 548 — **only aggressive fusion moves the count**.

`--iree-flow-inline-constants-max-byte-length=0` is worse on all three axes: constants are ~86 % of
the exported text, and inlining them into every dispatch blocks elementwise fusion (596 → 591 only)
while inflating compile time (+8 %) and warm time (+3 %).

## Rows 16384 (same pop-1000 graph, the batch route's win axis)

| config | dispatches | warm rocm ms | warm llvm-cpu ms |
|---|---|---|---|
| default | 596 | 2.547 | 1881.70 |
| aggressive | 548 | 2.367 (−7.1 %) | 1886.21 (flat) |

The rocm route is dispatch-bound at BOTH row counts, so the row axis is untouched — the batch
route's 3.13× advantage over the dense route at 16384 rows survives the flag. llvm-cpu is flat
within the ~10 % drift band of a shared MI300X VF.

## pop 4000 / rows 1024 (3121 distinct programs, 4 chunks × ~781 functions, band sizes 52/53, 28 546 ops)

| config | dispatches | max bindings | compile ms (4 sequential chunks) | warm rocm ms (median / best) |
|---|---|---|---|---|
| default | 2191 | 6 | 51376 | 8.356 / 8.244 |
| aggressive | 2021 | 10 | 48628 | 7.749 / 7.582 |

−7.8 % dispatches, −5.3 % compile, −7.3 % warm. Per-chunk `hal` phase 6587/6878/6653/6824 ms →
6223/6495/6385/6327 ms; per-chunk dispatches 552/554/538/547 → 507/509/496/509.

**pop-4096 feasibility:** this shape (3121 programs / 28.5 k ops / 2191 dispatches) compiles in
51.4 s as four SEQUENTIAL chunk compiles of ~12.8 s each (12.2 s aggressive) — ~13 s wall with the
EvoGP engine's 4-way concurrent chunk build. So the pop-4096 shape IS compile-feasible; aggressive
fusion trims ~5 % and does not change the compile-bound verdict at scale.

## Why the graph fragments (the multi-use question)

**Not multi-use splitting.** `--iree-dispatch-creation-fuse-multi-use` is ON by DEFAULT in iree
3.11: forcing it off (`=false`) on one 137-dispatch chunk worsens the count to 222, and passing it
explicitly on changes nothing (596). `--iree-dispatch-creation-enable-aggressive-reshape-movement`
alone also changes nothing (596).

The fragmentation is **elementwise-fusion granularity**: each dispatch is one `linalg.generic`
carrying 37–69 arith/math ops, and the dispatcher breaks exactly where aggressive fusion will not
(and the default policy cannot) merge two consumers. So the lever is "fuse more of each program's
op chain", not "stop splitting shared values".

## Verdict

* **`dispatch_formation="aggressive"` is the only knob worth exposing** and it is OPT-IN
  (adapter default stays flag-free): the win is rocm-dispatch-shaped (−9 % warm at pop 1000,
  −7 % at pop 4000, −5 % compile) while llvm-cpu warm is flat-to-marginally-worse
  (117.24 → 118.02 ms), so it does not meet the "win on both targets" bar for a default change.
* **Enable when** the warmed merged-population call is dispatch-bound on rocm (pop ≳ 1000).
  Never enable it on llvm-cpu, and there is no cold cost either way (compile is equal-to-better).
* **Binding headroom:** max bindings rise 6 → 9 (pop 1000) / 6 → 10 (pop 4000) — still under the
  16-binding HIP/CUDA wall, but the margin shrinks as the band size grows; keep the band cap at 15.
* **Not the fix for the compile-bound loop:** at pop 4000 the merged build is still ~49–51 s
  (compile dominates), so aggressive fusion does not change the "compile-bound per generation"
  conclusion for a full-turnover GP loop.

## Reproduce

```
PYTHONPATH=<repo> EVOXC_DISABLE_LOCATIONS=1 python evoxc/bench/probe_dispatch_formation.py \
    --pop 1000 --rows 1024 --targets llvm-cpu,rocm --seed 0 \
    --configs default,aggressive_fusion,inline_const0,o3,o0 --reps 5 \
    --out results.json

# dispatch counts only (fast, no load)
python evoxc/bench/probe_dispatch_formation.py --pop 1000 --seed 0 --count-only --out count.json
```
