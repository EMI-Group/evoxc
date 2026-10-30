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

## Measured conclusion

The iree compile option `dispatch_formation="aggressive"` (→ `--iree-dispatch-creation-enable-aggressive-fusion`) is the only dispatch-formation knob that moves the count — 596 → 548 dispatches and −7…−9 % warm on rocm merged-population graphs (−5 % compile at pop 4000) with no llvm-cpu gain — so it ships OPT-IN and is enabled per-call on rocm.

## Constraints

- Never run a probe while another benchmark is timing the GPU: one timing process at a time, `EVOXC_DISABLE_LOCATIONS=1` as recorded.
- Determinism: `--seed 0` fixes the program population, so a re-run must pass it to reproduce a recorded cell.
- Every path inside an artefact is repo-root-relative (`evoxc/bench/...`); no host-absolute scratch paths, no removed directories.
- `iree-compile` is resolved from the installed iree package (`EVOXC_IREE_COMPILE` overrides), so a run needs iree compiler+runtime and, for the `rocm` target, a ROCm device.

Cross-references: the adapter semantics behind these numbers are in [`../backends/adapters/CONTEXT.md`](../backends/adapters/CONTEXT.md) ("Dispatch formation (merged-population graphs)"); the merge transform itself is `../transforms/batch_functions.py`.
