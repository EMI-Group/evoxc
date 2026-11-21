# evoxc.bench — committed benchmark records and the probe scripts behind them

## Intent

The in-repo home for measured benchmark and hardware-validation records and the reproducible evidence probes that produced them.
It is evidence, not library code: nothing here is imported by `evoxc` or collected by pytest, and every probe is run explicitly.
A committed artefact is history — a new measurement is a new file or a re-run with the seed recorded in the artefact itself.

## Contents

| Path | What it is |
|---|---|
| `probe_dispatch_formation.py` | Deterministic (`--seed 0`) iree dispatch-formation probe on a representative `evoxc.batch_functions` merged GP-population graph (pop 1000 / pop 4000, rows 1024 / 16384), one chunk body per compile as the EvoGP batch engine does, measuring dispatches and max bindings (`iree-compile --compile-to=executable-sources`), cold compile and warm per-call time (`run` + `synchronize`) per `--configs` entry on `--targets llvm-cpu,rocm`, plus numpy parity. |
| `dispatch_formation_results.json` | The raw records behind the findings note — `_meta` provenance plus `runs.<plan>` (`config`, `plan`, `cells` keyed `<target>|<config>`, `numpy_ref`) for the five measured plans. |
| `dispatch_formation_findings.md` | The narrated tables, the multi-use-vs-fusion-granularity analysis and the enable rule. |
| `probe_iree313_rocm.py` | Deterministic (`--seed 0`) iree 3.13.0 ROCm re-validation probe on the shapes iree 3.11 broke: an elementwise+reduce graph, an INT32-counter `while_loop`, an `evoxc.scan`, an SO-shaped `(10000, 100)` population step in a loop, a `while` nested in a `cond`, and the MO/NSGA2 `non_dominate_rank` shape — per cell on `--targets llvm-cpu,rocm` it records compile/load ms, warm ms (`run` + `synchronize`, median and best of `--reps`), device residency + the `.numpy()` refusal, parity vs a hand-written numpy oracle, and the per-repeat compile-crash counts for the two 3.11 SIGSEGV classes (5× `cond_while` / 10× `mo_class` per target, one process per compile so a SIGSEGV is recorded as exit `-11`, never retried away). |
| `iree313_rocm_results.json` | The raw records behind the 3.13 findings note — `_meta` provenance (host, GPU gfx942 + ROCm 7.1, iree-base compiler/runtime versions, numpy, seed, shared-host caveat) plus `runs.<target>|<cell>` (compile/load/warm, residency, parity, `compile_repeats`). |
| `iree313_rocm_findings.md` | The narrated 3.11-vs-3.13 defect table (while / scan / while-in-cond / MO-class), the measured cells and parity, the Vulkan device-visibility check (`VK_ICD_FILENAMES` unset on 3.13 enumerates lavapipe), and the explicit list of what the run did NOT validate (no CUDA, no Vulkan EXECUTION cell, no multi-rank, no cross-version interop, not a performance campaign). |
| `probe_vulkan_device_visibility.py` | Deterministic (no RNG, no timing) Vulkan enumeration probe: calls `iree.runtime.get_driver("vulkan").query_available_devices()` in a CHILD process under `--timeout` — once with `VK_ICD_FILENAMES` unset and once with `--icd <renamed lavapipe json>` — and prints the device-info dicts as JSON (a timeout/hang is recorded as data, never retried). Reproduces the findings note's visibility table. |

| `probe_intel_vulkan.py` | Deterministic (no RNG or timing) real-Intel ANV correctness/residency probe, run as `python -m evoxc.bench.probe_intel_vulkan` with the adapter context's scoped ICD/loader recipe; five same-device elementwise/reduction/matmul iterations, explicit readback and host-transfer guards. |
| `intel_vulkan_findings.md` | Panther Lake `8086:b0a0`, Mesa 26.3.0 ANV and pinned IREE 3.13.0 evidence, measured errors and scope boundaries; links to the single maintained reproduction recipe in the adapter context. |
| `probe_native_architecture.py` | Deterministic (`--seed 0`) public-API native-engine evidence probe: ~1500-op chain construction plus native `vectorize`/`vmap`/`grad` min/median of 7, `batch_functions` on 64 identical and 64 branchy (`switch_ops=True`) programs with the 12-key `graph.batch_report` and row-correctness, the explicit `trace → lower → compile → load` CPU pipeline on a vmap-batched per-sample graph, the same graph device-resident on Intel Xe Vulkan through IREE (residency, `.numpy()`/`.to(cpu)` guards, host-input refusal, parity, warm `run`), and the Rust-produced `batch_functions` graph executing on Vulkan; `--skip-vulkan` runs the CPU half and any correctness failure exits nonzero. |
| `native_architecture_findings.md` | Host/CPU/Intel Xe GPU/driver and version provenance, exact commands, the transform/`batch_functions`/CPU/Vulkan numeric tables and parity errors, the 101/101 Vulkan suite count, the shared-host/indicative-timing caveats, and a labelled `--seed 0` re-validation of the same probe at the final native HEAD (not a speed comparison). |
## Measured conclusion

The iree compile option `dispatch_formation="aggressive"` (→ `--iree-dispatch-creation-enable-aggressive-fusion`) is the only dispatch-formation knob that moves the count — 596 → 548 dispatches and −7…−9 % warm on rocm merged-population graphs (−5 % compile at pop 4000) with no llvm-cpu gain — so it ships OPT-IN and is enabled per-call on rocm.
On iree-base 3.13.0 / ROCm 7.1 / gfx942 all four 3.11 loop defects are gone on `llvm-cpu` and `rocm`: the INT32-counter `while_loop` and `evoxc.scan` run on the HIP HAL with fp32-level parity, and `while`-in-`cond` plus the MO/NSGA2 `non_dominate_rank` shape compile 30/30 in separate processes (3.11: HIP `event_semaphore.c:786` ABORT at run, and `AffinityAnalysis` SIGSEGV in ~85–100 % of compiles) — the MO artifact now also executes on rocm, bit-exact.
Those 3.13 timings are indicative only (the GPU is shared, one sequential process per cell, 10 samples) and cover no CUDA, Vulkan, multi-rank or cross-version-interop claim.
In the ROCm 3.13 campaign Vulkan device visibility was checked but no vulkan CELL was run: on the adopted 3.13.0 the host's Mesa lavapipe enumerates with `VK_ICD_FILENAMES` unset (same `device_id`/`path` as the renamed-ICD leg, only the name prefix differs), so the rename is a 3.11-era fallback and a loader-pinning recipe, not a requirement.

The separate Intel Panther Lake ANV validation passed five small-graph same-device iterations with zero `asdevicearray` calls, 17 guards and maximum matmul absolute error `1.91e-6`; the existing Vulkan suites passed 101/101 without skips.
This is correctness/residency coverage only, not general GPU support, performance or model readiness, and does not replace the historical lavapipe evidence.

The native (Rust) transform-engine probe passed end-to-end with exit 0: on a 1495-op chain `vectorize`/`vmap`/`grad` took 27.08/27.79, 26.53/28.14 and 36.07/37.30 ms (min/median of 7), and `batch_functions` merged 64 identical and 64 branchy programs in 8.69/8.81 and 8.86/8.94 ms with row-exact results.
The CPU explicit pipeline ran the vmap-batched per-sample graph at 0.036/0.037 ms warm with bit-exact numpy parity, and the same graph ran device-resident on the Intel Xe Vulkan device at 0.269/0.291 ms warm with maximum absolute error `1.19e-6`.
The Rust-produced 64-program `batch_functions` graph also executed device-resident on Vulkan with row-level maximum absolute error `3.82e-6`, and the three Vulkan suites passed 101/101 with no skips.
Those timings are indicative (one session, shared CPU/GPU), and the GPU leg is correctness+residency evidence rather than a performance claim.
A labelled `--seed 0` re-validation of the same probe at the final native HEAD (`76c633c`) returned the same shapes and parity — 1495 ops, transforms 26.91/27.93, 26.75/28.09 and 37.06/37.92 ms, merges 8.73/8.95 and 8.89/8.99 ms, CPU warm run 0.036/0.037 ms, Vulkan warm run 0.282/0.441 ms, parity 0 / 1.192e-06 / 3.815e-06 — so the recorded status stands at HEAD.
## Constraints

- Never run a probe while another benchmark is timing the GPU: one timing process at a time, `EVOXC_DISABLE_LOCATIONS=1` as recorded.
- Determinism: `--seed 0` fixes the program population, so a re-run must pass it to reproduce a recorded cell.
- Every path inside an artefact is repo-root-relative (`evoxc/bench/...`); no host-absolute scratch paths, no removed directories.
- `iree-compile` is resolved from the installed iree package (`EVOXC_IREE_COMPILE` overrides), so a run needs iree compiler+runtime and, for the `rocm` target, a ROCm device.

Cross-references: the adapter semantics behind these numbers are in [`../backends/adapters/CONTEXT.md`](../backends/adapters/CONTEXT.md) ("Dispatch formation (merged-population graphs)"); the merge transform itself is `../transforms/batch_functions.py`.
