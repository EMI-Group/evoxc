# IREE 3.13.0 on ROCm — re-validation of the 3.11 loop / scan / while-in-cond / MO-class defects

Probe: `evoxc/bench/probe_iree313_rocm.py` (self-describing; `--help`).
Raw records: `evoxc/bench/iree313_rocm_results.json` (cells keyed `<target>|<cell>`, `_meta` = provenance).
Reproduce (exactly what produced the records below):

```bash
uv sync --locked --all-packages --extra dev --extra iree     # iree-base 3.13.0 compiler+runtime
EVOXC_DISABLE_LOCATIONS=1 python evoxc/bench/probe_iree313_rocm.py \
    --targets llvm-cpu,rocm --seed 0 --reps 10 --warmup 3 \
    --out evoxc/bench/iree313_rocm_results.json
```

`--seed 0` fixes every graph's data (per-cell RNG offsets), so a re-run reproduces the parity
columns bit for bit; the timing columns are host/GPU-load dependent. One process per graph (the
parent only aggregates child exit codes — a compiler SIGSEGV surfaces as `returncode -11` and is
recorded verbatim, never retried away), one timing process at a time, nothing else timing the GPU.

## Environment of the recorded run

`rocm-7-1-software-gpu-mi300x1-192gb-devcloud-atl1`, Linux 6.8 (QEMU VM, 20 vCPUs), AMD Instinct
MI300X VF (**gfx942**, ROCm **7.1.0**, `Device("rocm", 0)` → iree `hip` `device_id 1`),
**iree-base-compiler 3.13.0 + iree-base-runtime 3.13.0**, numpy 2.5.2, evoxc 0.1.0, Python 3.14.8.
The GPU is **shared with sibling work** — timings are sequential but indicative, not
isolated-hardware numbers; correctness and compile-crash results are unaffected.
All 3.11 rows below come from the committed record: `../../backends/adapters/CONTEXT.md`
Known Issues 8f/10 + the ROCm note under "Known Issues", and
`../../backends/adapters/PER-ADAPTER-NOTES.md` (`## Known Issue 10`). No 3.11 measurement was
re-run for this note — there is no 3.11 environment on this box any more.

## The four 3.11 defects, re-measured on 3.13.0

| 3.11 defect | 3.11 symptom (documented) | 3.13.0 measurement (this run) | verdict |
|---|---|---|---|
| `while_loop` carrying an **INT32 counter**, carries initialized from a graph constant | run-time **ABORT** in the HIP HAL: `RuntimeError: … hip/event_semaphore.c:786: ABORTED; the semaphore was aborted; while invoking native function hal.fence.await` (adapters CONTEXT, ROCm note; `repro_minimal.py --variant abort`) | **rocm**: compiles in 720 ms, runs, warm 1.177 ms (median), parity max_abs 1.2e-07 (counter exact, vector leaf 1.2e-07); **llvm-cpu**: 0.269 ms, 6.1e-05 | **FIXED** — arms and runs on the HIP HAL |
| `evoxc.scan` (constant carry init — the shape every `scan` desugars to) | run-time **ABORT**: `stream_command_buffer.c:452 INVALID_ARGUMENT … hal.device.queue.alloca`, or the same `hal.fence.await` ABORT (two signatures, one graph) | **rocm**: compiles in 457 ms, runs, warm 3.47 ms median / 2.37 best, parity 1.2e-07; **llvm-cpu**: 0.682 ms, 7.6e-06 | **FIXED** |
| `while_loop` **nested in `evoxc.cond`** | **`iree-compile` SIGSEGV**, `exit code -11` (`Error invoking IREE compiler tool iree-compile`) — a compile-time crash, no runtime number existed | **5/5 clean compiles on llvm-cpu** (592–699 ms) **and 5/5 on rocm** (583–685 ms); the artifact runs and both branches match the numpy oracle (`pred=True` 1.2e-07, `pred=False` bit-exact) | **FIXED** — 0 crashes in 10 compiles |
| MO/NSGA2-class `while` (input-derived `(n, n)` i1 dominance matrix, in-body reductions, bool→i32 cast chain, **3 carries + `select`**) | **`iree-compile` SIGSEGV in `IREE::Stream::AffinityAnalysis::run()`**, NONDETERMINISTIC: ~85–100 % of runs crashed (observed ok-rates 3/20, 1/6, 1/8, 0/20 — on **llvm-cpu AND rocm**), so a single green compile proved nothing (Known Issues 8f/10) | **10/10 clean compiles on llvm-cpu** (364–471 ms) **and 10/10 on rocm** (399–542 ms); 0 SIGSEGVs in 20 — plus, unlike 3.11, the artifact **runs on rocm**: `rank` is bit-exact vs the front-peeling oracle, warm 1.66 ms | **FIXED** on both targets |

Every cell ran with **default adapter options** — no retry loop, no unrolling, no `while_init_rewrite`
override (the exporter rewrite is ON by default, which is the setting that matters for the MO cell's
constant-init carries), no `opt_level`, no `dispatch_formation`.

## Measured cells

Per-cell geometry (all f32 unless stated, `--seed 0`): `elementwise` = `sum(relu(x @ w + b))` on
`x (256,128)`, `w (128,64)`, `b (64,)`; `while_i32_counter` = 5 iterations carrying
`(i32 counter, f32 scalar acc, f32 (256,))`, condition `less(counter, <python int literal>)`;
`scan` = 32 steps over a `(128,)` carry from a constant init; `so_step` = 4 loop iterations of a
`(10000, 100)` population step (row scores → rank-0 `argmin` → `gather` of the best row → masked
mean shift, carried scalar accumulator); `cond_while` = `cond(pred, while×3, negate)` on `(64,)`;
`mo_class` = `non_dominate_rank` at n=32, 2 objectives.

| cell | target | compile ms | load ms | warm ms (median / best) | parity max_abs | parity rel (to leaf scale) |
|---|---|---|---|---|---|---|
| `elementwise` | llvm-cpu | 276 | 41.0 | 0.164 / 0.152 | 9.8e-04 | 1.4e-07 |
| `elementwise` | rocm | 490 | 256.1 | **0.098 / 0.090** | **0** (bit-exact) | 0 |
| `while_i32_counter` | llvm-cpu | 599 | 43.0 | 0.269 / 0.263 | 6.1e-05 | 2.9e-07 |
| `while_i32_counter` | rocm | 720 | 267.8 | 1.177 / 1.134 | 1.2e-07 | 1.8e-07 |
| `scan` | llvm-cpu | 383 | 48.3 | 0.682 / 0.660 | 7.6e-06 | 1.7e-07 |
| `scan` | rocm | 457 | 269.0 | 3.472 / 2.371 | 1.2e-07 | 1.2e-07 |
| `so_step` | llvm-cpu | 760 | 40.4 | 3.502 / 3.335 | 6.1e-05 | 4.1e-07 |
| `so_step` | rocm | 813 | 265.9 | **1.387 / 1.271** | 6.1e-05 | 4.1e-07 |
| `cond_while` | llvm-cpu | 569 | 43.9 | 0.161 / 0.154 | 1.2e-07 | 1.5e-07 |
| `cond_while` | rocm | 720 | 250.8 | 0.864 / 0.849 | 8.9e-08 | 1.1e-07 |
| `mo_class` | llvm-cpu | 403 | 47.1 | 0.402 / 0.367 | **0** (bit-exact) | 0 |
| `mo_class` | rocm | 436 | 251.6 | 1.660 / 1.562 | **0** (bit-exact) | 0 |

Timing = `evoxc.run` + `Executable.synchronize()`, median and best of 10 samples after 3 warmups.
Parity is against a **hand-written numpy oracle** per cell (independent of the evoxc numpy backend),
`rel` = max |got−ref| / max|ref| of the leaf. Integer outputs (`i32` counter, `rank`) are exact in
every cell; the residual fp32 deltas are reduction-order differences (`so_step`'s population leaf is
bit-exact while its two reductions differ by 6.1e-05).

Reading the numbers honestly: `elementwise` (256×128 @ 128×64) and the small loop cells sit at or
near the per-invoke launch floor on this GPU, so the rocm-vs-llvm-cpu ratios there are
launch/fence artifacts, not throughput claims — rocm is faster only where the work per invoke is
large enough to leave the floor (`elementwise` 0.098 vs 0.164 ms; `so_step` **1.39 vs 3.50 ms**,
~6 data-parallel 1M-element ops × 4 iterations) and slower where it is not (`while_i32_counter`
1.18 vs 0.27 ms, `scan` 3.47 vs 0.68 ms over 32 sequential 128-element steps, `cond_while` 0.86 vs
0.16 ms, `mo_class` 1.66 vs 0.40 ms): on rocm each of those is a chain of tiny invokes paying the
HAL's per-invoke cost, on llvm-cpu the whole graph runs in one local-task dispatch. rocm
`load` is ~250 ms per process (first HAL device creation, once per child) against ~40 ms on
llvm-cpu — a per-process cost, not per-call. Single samples are noisy on this shared box
(`rocm|elementwise` has one 2.675 ms outlier among 0.09–0.13 ms samples, `rocm|scan` ranges
2.4–5.0 ms), which is why both median and best are recorded.

### Compile-repeat cells (the nondeterminism question)

| cell | target | repeats | ok | signals | compile ms min / median / max |
|---|---|---|---|---|---|
| `cond_while` | llvm-cpu | 5 | **5** | — | 592 / 632 / 699 |
| `cond_while` | rocm | 5 | **5** | — | 583 / 599 / 685 |
| `mo_class` | llvm-cpu | 10 | **10** | — | 364 / 404 / 471 |
| `mo_class` | rocm | 10 | **10** | — | 399 / 485 / 542 |

0 crashes in 30 compiles across both targets (each in its own process, so a SIGSEGV could not be
masked by in-process state). 3.11's ~85–100 % MO-class crash rate means even a 10/10 on a single
target would have been decisive; 10/10 on both, plus a running artifact, is unambiguous.

### Device residency (unchanged, and explicit)

Every rocm output leaf reports `Device(kind='rocm', index=0)` and `.numpy()` refuses with
`DeviceError: Tensor.numpy() needs host memory, but this tensor lives on Device(kind='rocm',
index=0): there is no implicit device-to-host transfer. Transfer explicitly first, e.g.
t.to(core.Device('cpu', 0)).numpy().` — i.e. `run` stays device-resident and the readback in the
probe is an explicit `.to(cpu)`. llvm-cpu outputs are `Device(kind='cpu', index=0)` with host data
and `.numpy()` allowed, as documented.

## What this run did NOT validate

- **No CUDA.** This host has no NVIDIA GPU. Every CUDA-scoped 3.11 claim therefore stays a 3.11
  record, **not** re-validated: Known Issue 8f's CUDA-HAL while shapes (`while_fib` SIGSEGV,
  `while_cond_combo` ABORT), 8g (2-operand-sort bufferization), 9 (cuda `eigh`/NSGA2 O2/O3 codegen).
  Nothing here should be read as "fixed on CUDA".
- **No Vulkan.** There is no usable Vulkan device on this box (Mesa RADV policy-rejects CDNA3,
  AMDVLK/PAL has no `FAMILY_AI` IP level, software lavapipe is denylisted by IREE 3.11/3.13 without
  an ICD rename); no vulkan cell was attempted.
- **No multi-rank / collectives** (single rank, no runtime channels) and **no cross-version
  interop** (a 3.11-compiled artifact loaded on the 3.13 runtime was not tried — every cell here is a
  fresh compile on 3.13.0).
- **No xla / tvm leg** (xla needs a user-provided PJRT plugin `.so`; TVM is compile-only on ROCm).
- **Not a performance campaign.** 10 samples per cell, one sequential process at a time, GPU shared
  with sibling work, no dispatch counts, no `dispatch_formation="aggressive"`, no comparison against
  the archived `evoxc-bench` material. Timings are indicative; the correctness and crash counts are
  the load-bearing results.
- **Not the full evox graphs.** The MO cell is `non_dominate_rank` at n=32 / 2 objectives (the shape
  `tests/backends/test_while_init_rewrite.py` pins), not a whole NSGA2 tell (crossover, speciation,
  crowding distance, 4 carries); the SO cell is a representative SO-shaped step at one geometry
  (10000×100, 4 iterations), not the evox DE pipeline.
- **The other 3.11 Known Issues were not re-measured** (2 `rng_bit_generator`/PHILOX legalization,
  13 rank-0-result reduction cost + `split_reduction`, 19 the 16-binding HIP/CUDA wall, …) — they
  remain as recorded in `../../backends/adapters/CONTEXT.md`.

## Verdict

On iree-base 3.13.0 / ROCm 7.1 / gfx942 **all four 3.11 defects are gone** on both `llvm-cpu` and
`rocm`: the int32-counter `while_loop` and `evoxc.scan` run on the HIP HAL with fp32-level parity,
and the two `iree-compile` SIGSEGV classes (`while` in `cond`, MO/NSGA2-class while) compile 30/30
in separate processes and produce correct artifacts — the MO one now even *executes* on ROCm, which
3.11 never reached. Nothing in this run contradicts the adopted 3.13 pin.
