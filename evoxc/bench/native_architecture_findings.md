# Native (Rust) transform engine: CPU and Intel Xe end-to-end evidence

## Provenance and scope

Source of record: `evoxc/bench/probe_native_architecture.py`, run with `--seed 0`.
All numbers below are transcribed from that probe's stdout; the machine-readable `SUMMARY_JSON` line carries the same values.
The probe is evidence tooling: `evoxc` never imports it and pytest never collects it.
Host: Intel(R) Core(TM) Ultra 7 356H, Linux 7.3.4 x86_64 (glibc 2.44), Python 3.14.10, numpy 2.5.2.
Versions: evoxc 0.1.0, `evoxc._evoxc_native` 0.1.0 (`available=True`), iree-base-compiler 3.13.0, iree-base-runtime 3.13.0.
GPU: `Intel(R) Graphics (PTL) (Intel open-source Mesa driver Mesa 26.3.0)`, enumerated as the only Vulkan device (`Device(kind='vulkan', index=0)`, selected by name from `iree.runtime.get_driver("vulkan").query_available_devices()` under the Intel ANV ICD).
Load average: `(0.42, 1.06, 1.54)` at the start and `(0.46, 1.06, 1.53)` at the end of the recorded run.

## Exact commands

```
source scripts/dev-env.sh
.venv/bin/python evoxc/bench/probe_native_architecture.py --seed 0
```

The Vulkan suites were run read-only with the summary line re-enabled (the project pins `addopts = "-q"`, so an extra `-q` becomes `-qq` and suppresses the count):

```
.venv/bin/python -m pytest tests/backends/test_iree_vulkan_device_resident.py tests/backends/test_vulkan_target.py tests/backends/test_iree_vulkan_acquisition.py -o addopts="" -q
```

Result: `101 passed in 1.10s`, with zero skips, zero failures and exit code 0 (the Intel device was visible, so all six device-guarded tests ran).

## Graph construction and native transform cost (part A)

The traced chain is 560 loop iterations of `tanh`, `multiply+add`, or `add+multiply`, ending in a full reduction; the graph holds 1495 IR ops.
Trace cost alone: min 15.24 ms, median 15.99 ms (7 repetitions).
Each transform below starts from that pre-traced graph, so its time excludes tracing.

| Transform | min (ms) | median (ms) | output IR ops |
|---|---:|---:|---:|
| `vectorize(graph, (0,))` | 27.08 | 27.79 | 2055 |
| `vmap(graph, in_axes=0, out_axes=0)` | 26.53 | 28.14 | 2055 |
| `grad(graph, argnums=0)` | 36.07 | 37.30 | 3738 |

## `batch_functions` (part B)

Population programs share signature `(4, 8) -> (4,)`: `y = 1.5*x + 0.25`, one variant unary op chosen by `k % 4` from `relu`/`tanh`/`negate`/`abs`, `y = y*y + 0.5`, then `sum(y, axis=1)`.
Input tensor for the merged graph is `(64, 4, 8)` f32; row i must equal program i applied to `x[i]`.

| Case | N | merge min (ms) | merge median (ms) | emitted ops | selects | row max abs err |
|---|---:|---:|---:|---:|---:|---:|
| 64 identical programs | 64 | 8.69 | 8.81 | 12 | 0 | 0.0 |
| branchy, `switch_ops=True` | 64 | 8.86 | 8.94 | 21 | 1 group (cascade) | 0.0 |

The 12-key `graph.batch_report` for the identical case: `n_functions=64`, `value_classes=9`, `shared_classes=9`, `switched=[]`, `stacked_constants=0`, `output_muxes=0`, `emitted_ops=12`, `bands=1`, `band_sizes=[64]`, `band_strategy="flat"`, `cost_estimate=null`, `flat_cost_estimate=null`.
The 12-key report for the branchy case: `n_functions=64`, `value_classes=9`, `shared_classes=9`, `switched=[SwitchRecord(group="elementwise", variants=("relu","tanh","negate","abs"), rows_per_variant=rows 0,4,…,60 / 1,5,…,61 / 2,6,…,62 / 3,7,…,63)]`, `stacked_constants=0`, `output_muxes=0`, `emitted_ops=21`, `bands=1`, `band_sizes=[64]`, `band_strategy="flat"`, `cost_estimate=null`, `flat_cost_estimate=null`.
Both merges are row-exact on the numpy interpreter (max abs error `0.0`), and identical programs emit no `select` at all.

## CPU end-to-end (part C)

Graph: the vmap-batched per-sample function `h = tanh(dot(x, w) + b); return sum(h, axis=1)`, vmapped over a leading batch.
Shapes: `x (4, 8, 16)`, `w (16, 8)`, `b (8,)`, all f32; reference is the hand-written numpy expression `tanh(x @ w + b).sum(axis=2)`.
Staging is the explicit pipeline `trace -> lower -> compile -> load` on `backend="numpy"`, then 20 warm `run` calls (each followed by `Executable.synchronize()`).

| Stage | Time |
|---|---:|
| trace + vmap (min / median) | 0.40 / 0.41 ms |
| lower | 0.269 ms |
| compile | 0.191 ms |
| load | 0.184 ms |
| warm run min | 0.036 ms |
| warm run median | 0.037 ms |

numpy parity: max abs error `0.0` (the interpreter runs the same elementwise/reduction kernels on the same arrays).
Output device: `Device(kind='cpu', index=0)`.

## Intel Xe GPU end-to-end (part D)

The SAME vmap graph compiled for `vulkan-spirv` through IREE 3.13.0; the target is derived from the device kind (`vulkan -> ["vulkan-spirv"]`), with no hand-written compiler flags.
Artifact target: `vulkan`; inputs placed explicitly with `from_numpy(...).to(Device("vulkan", 0))`.
A host-input run was rejected with `DeviceError` (`host-input guard: rejected=True`).

| Stage | Time |
|---|---:|
| lower | 16.01 ms |
| compile | 167.93 ms |
| load | 25.62 ms |
| warm run min | 0.269 ms |
| warm run median | 0.291 ms |

Residency: `device_resident=True` (the output is a payload-backed `Tensor` on `Device(kind='vulkan', index=0)`, not an ndarray).
Guards: `.numpy()` on the Vulkan output raised `DeviceError` (`.numpy()_raised_DeviceError=True`), and `.to(Device("cpu", 0))` read back successfully (`.to(cpu)_ok=True`).
Vulkan parity vs the numpy reference: max abs error `1.192e-06` (fp32 tolerance).

## Native transform + `batch_functions` graph on Vulkan (part E)

The graph the GPU runs here is the Rust merge from part B, not a plain trace: the branchy 64-program `batch_functions(..., switch_ops=True)` output.
Input `(64, 4, 8)` f32 placed on the Vulkan device; rows compared against each program's own numpy result.

| Stage | Time |
|---|---:|
| lower | 0.68 ms |
| compile | 204.48 ms |
| load | 0.27 ms |
| warm run min | 0.250 ms |
| warm run median | 0.607 ms |

Residency and guards: `device_resident=True`, `.numpy()` raised `DeviceError`.
Row parity vs the per-program numpy results: max abs error `3.815e-06` (fp32 tolerance), confirming the Rust-produced switch/mux graph is what the GPU executes and that its rows are correct.

## Caveats

These are indicative single-process measurements, not a benchmark campaign: 7 repetitions for transforms/merges and 20 warm `run` calls per leg, on one host in one session.
The CPU was lightly but not exclusively loaded (`loadavg` 0.42–1.54, recorded above); the timings are sensitive to concurrent load and should be treated as orders of magnitude, not precise constants.
Vulkan warm-run medians vary noticeably between otherwise identical runs (e.g. 0.291 ms vs 0.607 ms median across two runs on this shared physical GPU), so the device numbers are correctness and residency evidence, not a performance claim.
The Vulkan graphs are deliberately tiny because IREE's default SPIR-V target profile (`vp_android_baseline_2022`) caps workgroup shared memory; nothing here says anything about large-graph or transformer-class Vulkan codegen.
`--seed 0` fixes the program population and data, so a re-run reproduces the graphs and the parity checks, but not the wall-clock values.
The native extension was already built (`evoxc/_evoxc_native.abi3.so`); the probe asserts `_native.available()` and records its version.

## Re-validation at the final native HEAD (seed 0)

The same committed probe (`evoxc/bench/probe_native_architecture.py`, `--seed 0`) was re-run at the HEAD this note is committed on top of (`76c633c`, "Make the batch_functions batchability gate engine-authoritative") with the final release extension, as a re-run of the recorded probe rather than a replacement for the sections above: the original measured sections and their numbers stand unchanged, and the two runs are NOT a speed comparison.
The re-run finished with `STATUS: PASS` and reproduced the same graph shapes, the same residency/guard results and numeric parity on both legs.

Environment of this re-run: Intel(R) Core(TM) Ultra 7 356H; `Intel(R) Graphics (PTL)` through the Intel open-source Mesa driver 26.3.0 (Vulkan via IREE `vulkan-spirv`, iree-base 3.13.0); Python 3.14.10; numpy 2.5.2; load average `(1.08, 1.34, 1.39)` at the start of the run.
The host is SHARED with other work, so these timings are indicative only and not a controlled benchmark.

Graph construction: the same traced chain, 1495 IR ops, trace min 15.1 ms / median 15.8 ms (7 repetitions).
Each transform starts from that pre-traced graph, so its time excludes tracing.

| Transform | min (ms) | median (ms) | output IR ops |
|---|---:|---:|---:|
| `vectorize(graph, (0,))` | 26.91 | 27.93 | 2055 |
| `vmap(graph, in_axes=0, out_axes=0)` | 26.75 | 28.09 | 2055 |
| `grad(graph, argnums=0)` | 37.06 | 37.92 | 3738 |

`batch_functions`, same population and same `(64, 4, 8)` f32 input:

| Case | N | merge min (ms) | merge median (ms) | emitted ops | selects | row max abs err |
|---|---:|---:|---:|---:|---:|---:|
| 64 identical programs | 64 | 8.73 | 8.95 | 12 | 0 | 0.0 |
| branchy, `switch_ops=True` | 64 | 8.89 | 8.99 | 21 | 1 group (cascade) | 0.0 |

The identical case reports `emitted_ops=12`, `value_classes=9` and zero `select`s; the branchy case reports `emitted_ops=21` with one elementwise switch record of 4 variants and 16 rows each.
Row parity is exact (`0.0`) in both cases.

CPU end-to-end (part C graph, `backend="numpy"`, 20 warm `run` calls each followed by `Executable.synchronize()`):

| Stage | Time |
|---|---:|
| trace + vmap (median) | 0.41 ms |
| lower | 0.271 ms |
| compile | 0.194 ms |
| load | 0.181 ms |
| warm run min | 0.036 ms |
| warm run median | 0.037 ms |

numpy parity: max abs error `0.0`; output device `Device(kind='cpu', index=0)`.

Intel Xe GPU end-to-end (part D graph, `vulkan-spirv` through IREE 3.13.0, artifact target `vulkan`, inputs placed explicitly with `.to(Device("vulkan", 0))`):

| Stage | Time |
|---|---:|
| lower | 16.49 ms |
| compile | 217.81 ms |
| load | 30.52 ms |
| warm run min | 0.282 ms |
| warm run median | 0.441 ms |

Residency: `device_resident=True`; `.numpy()` on the Vulkan output raised `DeviceError`; `.to(Device("cpu", 0))` read back successfully.
Vulkan parity vs the numpy reference: max abs error `1.192e-06`.

Native-produced `batch_functions` graph on Vulkan (part E, the Rust merge of part B's branchy 64-program population):

| Stage | Time |
|---|---:|
| lower | 0.73 ms |
| compile | 200.66 ms |
| load | 0.26 ms |
| warm run min | 0.242 ms |
| warm run median | 0.269 ms |

Residency: `device_resident=True`; row parity vs the per-program numpy results: max abs error `3.815e-06`.

Conclusion: at this HEAD the native engine's `vectorize`/`vmap`/`grad` transforms, the merged-population `batch_functions` merge and both stages of the explicit pipeline (`trace -> lower -> compile -> load -> run`) still pass numeric parity — on the CPU numpy interpreter (`0.0` error) and device-resident on the Intel Xe Vulkan device (`1.192e-06` for the vmap graph, `3.815e-06` for the Rust-merged graph).
Nothing here is a speed claim beyond the recorded millisecond figures; the wall-clock values are single-session, shared-host observations.
