# tools — dev-only generators, the CPU parity gate and the bench harnesses

## Intent

`native/tools/` holds the Python developer aids for the Rust workspace: the two generated-table emitters, the CPU-engine parity gate and the native-vs-numpy benchmark harness.
Nothing here is imported by a crate, compiled into the extension, or collected by pytest — every script is run by hand against a built `.venv` (`source scripts/dev-env.sh` + `scripts/build-native.sh` first).
The recorded measurement table itself lives in `../evoxc-cpu/CONTEXT.md`; this directory holds the harness that reproduces it, never a second copy of the numbers.

## API Surface

| Script | Contract |
|---|---|
| `./gen_opdefs.py` | Regenerates `../evoxc-core/src/schema_table.rs` from `evoxc.ir.op_defs.all_opdefs()` (name, category, arity, results, effect, regions, terminator, shape-fn, attribute schema). |
| `./gen_dtype_promotion.py` | Regenerates `../evoxc-core/src/dtype_table.rs`, the 16384-entry `np.result_type` table over every dtype subset. |
| `./parity_cpu.py` | The CPU-engine gate: `native_compile` + `NativeProgram.run` against the numpy backend, 73 cases (elementwise, reductions, indexing, linalg, control flow, random, dynamic shapes), exact for bool/integer and `allclose` for floats with case-declared tolerances. |
| `./bench_cpu.py` | Native-engine vs numpy benchmark: per workload one `@evoxc.defn` graph, timed through `NativeProgram.run` and a numpy `Executable.run`, plus `native_compile` and engine plan-compile times. |

Both generators are dev aids only: the committed generated files are the authority and a fresh run must reproduce them byte-for-byte.

`bench_cpu.py` columns are `native_compile_ms` (one `native_compile(envelope)` call), `native_run_ms` (`NativeProgram.run`, median warm), `numpy_run_ms` (`Executable.run`, median warm) and the speedup, with `--iterations`/`--warmup`/`--compile-repeats`/`--filter` as the only flags.
Its exit status is 0 when every workload ran, every native output matched the numpy leg and the 2000-op compile target (`<= 2 ms`) held; 1 on a miss; 2 when the extension is not built or a `--filter` matches nothing.

## Bench workloads (`bench_cpu.py`, 17 rows)

`ew_chain_1e6` — an 8-op arithmetic elementwise f32 chain over 1e6 elements.
`ew_transcendental_1e6` — tanh/sigmoid/gelu/erf over 1e6 f32 elements (scalar libm, not the memory path).
`dot_sum_256` — `dot((256,256), (256,256))` plus a `reduce_sum`.
`gp_gather_4096` — gather/select-heavy GP-shaped chain over (4096, 64) f32, built from static indices.
`gp_sort_4096` — row argsort/sort, a top-16 slice, two per-row reductions, a two-axis gather and a broadcast select over (4096, 64) f32.
`tf_softmax_8x128x384` — a transformer-shaped per-row softmax plus a tanh/gelu mix over (8, 128, 384) f32.
`chain_2000` — a ~2000-op elementwise chain over a (64,) f32 array; the compile-time target workload.
`tn_step_pop500` — the shape-faithful tensorneat 0.2.0 `Pipeline.step` approximation at population 500 (2249 static ops); the one row where numpy wins.
`large_ew2_16m` — `a * b + a` over 16M f32 elements (2 ops).
`large_tanh_16m` — `tanh(a) * b + a` over 16M f32 elements (3 ops).
`large_chain4_16m` — `(a * b + a) * c - d` over 16M f32 elements (4 ops).
`large_sum_all_16m` — `sum(x)` over a flat 16M f32 operand (every axis, rank-0 result).
`large_sum_axis_4096` — `sum(x, axes=1)` over a (4096, 4096) f32 grid (`large_sum_axis0_4096` is the `axes=0` twin on the same operand).
`large_dot_1024` — `dot((1024,1024), (1024,1024))` f32.
`large_dot_2048` — `dot((2048,2048), (2048,2048))` f32.
`large_dot_4096x1024` — `dot((4096,1024), (1024,4096))` f32, a 16.8M-element product.

The `large_*` rows are the native engine's large-tensor evidence and come from their own seeded `numpy.random.default_rng`, so the eight original rows keep byte-identical inputs.
The large operands are generated (never literal data) and total ~0.45 GB, so `--filter large_` is the way to run just them.
The reduction operands are drawn uniform in `[0.5, 1.5)` so a 16M-term or 4096-wide sum does not land near cancellation against the comparison's absolute tolerance.
The three large `dot` rows alone compare at `atol 1e-2` (`Workload.atol`, relative tolerance unchanged), the documented f32 gemm reassociation residual that `parity_cpu.py`'s `dot_f32_large` pins; every other row keeps `rtol 1e-3` / `atol 1e-4`.

## Constraints

- Every workload is one `@evoxc.defn` graph plus a `Workload(name, fn, specs, inputs, note=...)` entry in `build_workloads()`; the specs and the input arrays must agree with the defn's arity and shapes.
- Rows run through BOTH legs with the same warmup/repetition policy and print the same columns — a row is never native-only or numpy-only.
- A row's constants and index arrays are built with `evoxc.constant(evoxc.from_numpy(...))` inside the defn; nothing is captured from a closure.
- Keep the file under ~1000 lines and keep the new workload builders in it unless it would exceed that (then a helper module in this directory).
- A committed number is history: a re-measurement is a re-run with its seed and load average recorded, never an edit of an existing record.
