# evoxc.bench — committed benchmark records and the probe scripts behind them

## Intent

The in-repo home for measured bench records and the one-off probe scripts that produced them.
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

## Measured conclusion

The iree compile option `dispatch_formation="aggressive"` (→ `--iree-dispatch-creation-enable-aggressive-fusion`) is the only dispatch-formation knob that moves the count — 596 → 548 dispatches and −7…−9 % warm on rocm merged-population graphs (−5 % compile at pop 4000) with no llvm-cpu gain — so it ships OPT-IN and is enabled per-call on rocm.
On iree-base 3.13.0 / ROCm 7.1 / gfx942 all four 3.11 loop defects are gone on `llvm-cpu` and `rocm`: the INT32-counter `while_loop` and `evoxc.scan` run on the HIP HAL with fp32-level parity, and `while`-in-`cond` plus the MO/NSGA2 `non_dominate_rank` shape compile 30/30 in separate processes (3.11: HIP `event_semaphore.c:786` ABORT at run, and `AffinityAnalysis` SIGSEGV in ~85–100 % of compiles) — the MO artifact now also executes on rocm, bit-exact.
Those 3.13 timings are indicative only (the GPU is shared, one sequential process per cell, 10 samples) and cover no CUDA, Vulkan, multi-rank or cross-version-interop claim.
Vulkan device visibility is checked but no vulkan CELL was run: on the adopted 3.13.0 the host's Mesa lavapipe enumerates with `VK_ICD_FILENAMES` unset (same `device_id`/`path` as the renamed-ICD leg, only the name prefix differs), so the rename is a 3.11-era fallback and a loader-pinning recipe, not a requirement.

## Constraints

- Never run a probe while another benchmark is timing the GPU: one timing process at a time, `EVOXC_DISABLE_LOCATIONS=1` as recorded.
- Determinism: `--seed 0` fixes the program population, so a re-run must pass it to reproduce a recorded cell.
- Every path inside an artefact is repo-root-relative (`evoxc/bench/...`); no host-absolute scratch paths, no removed directories.
- `iree-compile` is resolved from the installed iree package (`EVOXC_IREE_COMPILE` overrides), so a run needs iree compiler+runtime and, for the `rocm` target, a ROCm device.

Cross-references: the adapter semantics behind these numbers are in [`../backends/adapters/CONTEXT.md`](../backends/adapters/CONTEXT.md) ("Dispatch formation (merged-population graphs)"); the merge transform itself is `../transforms/batch_functions.py`.
