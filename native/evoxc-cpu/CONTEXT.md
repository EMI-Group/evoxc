# evoxc-cpu — the native CPU execution engine of EvoX Compiler

## Intent

`evoxc-cpu` is the native (Rust) CPU backend: it compiles an [`evoxc_core::Program`] into an immutable execution `Plan` and runs it on typed host buffers.
It is the numerical authority for concrete CPU execution — the Rust counterpart of the Python numpy interpreter in `../../evoxc/backends/numpy/` — and it depends on `evoxc-core` only: no Python, no PyO3, no device APIs, no compiler backends.
The `evoxc-python` crate wraps it as `evoxc._evoxc_native.native_compile` / `NativeProgram`; that FFI is documented in `../evoxc-python/CONTEXT.md`.

Shape/type inference is NOT re-derived here: result types come from the graph (`OpSpec.result_types`, filled by `evoxc-core` inference at trace/build time) and every produced buffer is re-validated against them at run time.
The engine never falls back to Python and never guesses: an op without a kernel is an explicit `BackendError` at plan-build time naming the op (and its source location when the graph carries one).

`ndarray` is the crate's N-dimensional array library: every axis/layout/broadcast/reduction indexing rule the kernels need is an ndarray view, `Zip`, `indices` or `axis_iter` expression, and no hand-written per-element stride/offset odometer steps through a layout.
The few run maps beside the views (the broadcast run drivers, the `strided_copy` copier, the sorting segment walk) address whole runs with no per-element stride arithmetic.
The owned value container (`Buffer`/`Storage`) keeps a dtype, a concrete shape and exactly that dtype's typed `Vec`, so a kernel sees a flat typed slice and an ndarray view is a transient zero-copy borrow over it.

```text
Program --compile--> Plan --RunCtx::new--> RunCtx --run--> Vec<Buffer>
```

## API surface

- `plan::compile(&Program) -> Result<Plan>` — one immutable `Plan` per program: per-function entry `FunctionPlan`, per-block `BlockPlan` (held as `Arc`), per-op `Step` (`OpSpec` + resolved kernel + operand/result value ids + liveness free list + per-result static-`(dtype, shape)` marker + a decoded constant payload + an in-place slot). Compiles every reachable region and every `call` callee (BFS), collects the host-callback requirements (`required_blocks` / `required_external` / `required_runtime`) and rejects an unsupported op up front. `Plan::entry_function()`, `Plan::needs_callbacks()`, `Plan::text()` (the human-readable dispatch plan behind `NativeProgram.plan_text()`).
- `exec::RunCtx` — the interpreter: `RunCtx::new(program, Arc<Plan>, RunOptions, Option<&dyn Callbacks>)`, `run(flat_inputs) -> Result<Vec<Buffer>>`, `run_block(block_index, args)`, `run_region(region, args)`, `eval_shape`/`eval_shape_strict` (symbolic-dim binding + `DimExpr` evaluation), `result_shape`/`result_shape_strict`/`result_type`, `validate_result`, and the `try_in_place` accumulate path. Values live in ONE flat `value_id`-indexed store per block execution; a value is dropped at its last use (nested-region uses are attributed to the enclosing op, so a control-flow parent's operand survives its region).
- `exec::Callbacks` — the host-callback trait the PyO3 layer implements: `call_block(name, args, static_args)`, `call_external(name, args)`, `call_runtime(callback, args)`, each returning `Result<Vec<Buffer>>`.
- `exec::RunOptions { rank, world_size }` — the rank context of the run (defaults `0` / `1`).
- `buffer::{Buffer, Storage}` — the typed value: `Storage` covers `bool`, `i8/i16/i32/i64`, `u8/u16/u32/u64`, `f16`, `f32`, `f64`, `c64`, `c128` (all 14 evoxc dtypes); `Buffer { dtype, shape, storage }` with `new` (validates dtype + element count), `from_payload` (graph constants), `zeros`, `cast`, `with_shape`, `numel`, `rank`, `is_flat_for`, the numpy shape algebra (`broadcast_shape`/`broadcast_strides`), and the ndarray seam (`as_array`/`as_array_mut`/`as_array2`/`from_array`/`into_array` plus the free `array_view`/`array_view_mut`).
- The ndarray seam (`buffer/ndarray_views.rs`): `trait NdScalar` (`DTYPE`, `into_storage`/`take_storage`/`slice_of`/`slice_of_mut`) is the per-dtype bridge; `as_array*` borrow the buffer as an `ArrayViewD`/`ArrayViewMutD`/`ArrayView2`, `from_array` takes an owned `ArrayD` (standardizing a non-standard layout), `into_array` moves the `Vec` into `ArrayD` with no copy for a C-contiguous buffer. A dtype mismatch is `Error::dtype` naming both dtypes and a shape/length mismatch is `Error::shape`; nothing here panics.
- The performance substrate, all `crate::buffer::…` paths: the thread-local recycled-storage pool (`take_pooled_vec`, `POOL_MIN_BYTES`/`POOL_MAX_BYTES`/`POOL_MAX_ENTRIES`/`POOL_MAX_PER_SIZE`), the flat builders (`zip1..3`, `zip2_accumulate`, `map1..3`, plus the pooled broadcast builders `map2_runs`/`map3_runs`), the layout drivers (`walk`, `apply1/apply2/apply3` — one `ndarray::Zip` pass over broadcast views — and `fill_broadcast`), the repeating-run and periodic drivers (below), the scoped split (`split_parallel`/`PARALLEL_MIN_ELEMS`/`PARALLEL_MIN_PER_THREAD`/`PARALLEL_MAX_THREADS`) and the row split (`rowsplit`'s `RowSplit`/`Rider`/`split_rows`), the block/repeat builders (`copy_blocks`, `block_storage_filled`, `repeat_storage`) and the dispatch macros `with_type!`/`with_same_type!`/`with_same_type_mut!`.
- `scalar::{Scalar, Num, ScalarValue}` — the numeric vocabulary kernels are generic over (`add/sub/mul/div/rem/pow/neg/abs/sqrt/exp/ln/log1p/sin/cos/tan/tanh/sign/floor/ceil/round_even/max/min/bit_*/shl/shr`), with numpy semantics baked into the impls (integer `rem` truncates toward zero, unsigned wraparound, `round` half-to-even, NaN-propagating `max`/`min`).
- `dims::{Bindings, eval_dim, eval_extent, eval_shape, eval_shape_strict, bind_dim, bind_shape}` — the `shapes.py` contract: bind named dims against concrete extents (conflict → `ShapeError`), bind `DimExpr` leaves then evaluate, a free dim → `ShapeError`, division/modulo by zero → `ShapeError`, negative extent → `ShapeError`.
- `run_program(program, &Arc<Plan>, inputs, options, callbacks) -> Result<Vec<Buffer>>` — one call; a plan that needs callbacks and gets `callbacks: None` is an explicit `BackendError` naming the missing entry points, before any op runs. `evaluate(program, inputs, options, callbacks)` — compile + run (tests, harnesses).
- `kernels::{lookup, is_supported, unsupported_ops, DENSE_OPS}` — the machine-readable support inventory: `unsupported_ops()` is the sorted list of dense ops in `DENSE_OPS` without a native kernel (currently EMPTY — the two `custom`-category callback ops are deliberately outside `DENSE_OPS`, see the support matrix), and three tests pin it (`every_dense_op_has_a_kernel`, `table_has_no_foreign_entries`, `the_custom_callback_ops_have_no_kernel`).

## The array substrate: which path is the reference, which is the fast path

Every family has ONE ndarray expression that defines its behaviour and, where measurement justifies it, a documented specialization that must stay bit-identical to it.

- **elementwise/comparison/select/cast.** The flat typed-slice `zip1..3`/`map1..3` builders are the fast path for same-shape operands (`Layout::Flat`), with the kind dispatched once per call.
  A broadcast or offset layout goes through `map2_runs`/`map3_runs`, which offer it to the repeating-run driver and the periodic-pattern driver (`buffer/broadcast/patterns.rs`) before falling back to `apply2`/`apply3` — one `ndarray::Zip` pass over the operands' broadcast views, the reference.
  `pattern_view` proves the periodic family's `result[i] reads values[(i / stretch) % span]` mapping from the shapes and the broadcast strides (never by sampling), and the run bodies are monomorphized on const-generic run widths 2/4/8 (`fixed2`/`fixed3`) with a runtime-bounded body for any other width.
  Every family's tests compare the production path against an INDEPENDENT ndarray `Zip` expression written out in the test.
- **reductions.** `flat::fold_block` is the specialization for a contiguous reduced block (`outer × red × inner`), folding straight off the operand slice with an 8-lane accumulator split; `layout.rs`'s ndarray lane walk (kept axes permuted first, reduced axes last, `axis_iter` recursion) is the reference for every other layout, and the flat fold is pinned against it.
  The fold's thread split is a SPEED-ONLY layer over that expression: outputs are disjoint and each one folds in the single-threaded order, so `the_block_fold_is_bit_identical_at_every_thread_count` gates it against the 1-thread fold exactly as the fold itself is gated against the lane walk.
- **indexing/structure.** `reshape`/`transpose`/`slice`/`pad`/`tile`/`flip`/`roll`/`diag`/`diagonal`/`tril`/`triu` are ndarray expressions over views (`permuted_axes`, `slice_axis_inplace`/`slice_each_axis`, `broadcast`, `diag`/`diag_mut`).
  The materializing copies go through the two-sided run copier (`kernels/indexing/strided_copy.rs`: `CopyPlan`/`AxisCopy`/`RunCopy`), which is a run map, not an odometer, and every layout it declines receives ndarray's `assign`.
- **sorting.** The hand-written per-segment walk stays: an ndarray `lanes(Axis(axis))` expression cost ~37 ns/segment against a ~7 ns core (measured 2.7–6.3× slower), so `permutation.rs` keeps flat base/stride arithmetic and the Batcher odd-even networks for short extents.
- **linalg `dot`.** The plain matrix-matrix form in f32/f64 runs `faer::linalg::matmul::matmul` on row-major slices with `Accum::Replace` — the fast path; the general triple loop is the reference for integer, complex, f16 and the rank-1 forms, and it stays bit-exact. Batched traversal uses ndarray `broadcast` views plus `ndarray::indices`, one typed `ArrayView2` block per batch position.
- **linalg_aux / solve / conv.** ndarray plumbing only (`Array3` operand, per-batch `ArrayView2`, `Buffer::from_array` results); the factorizations and `solve` keep their hand-written kernels, and `faer` appears only in `#[cfg(test)]` A/B oracles with the recorded reasons (see Known Issues).

## Support matrix

The schema registry holds 121 op defs; the engine implements its dense kernel set (102 of them), and the remaining defs — the `return` terminator, the 16 `sparse_*` ops and the two `custom`-category callback ops (`custom_call`, `inline_stablehlo`) — have no kernel and are rejected at plan compile with a `BackendError` naming the op.
That rejection is where these ops surface as unsupported; the plan-level `unsupported` inventory stays EMPTY because such a graph never reaches a compiled plan (the Python native backend's capability gate reads `stats()["unsupported"]`).

The kernel set by category:
| Category | Ops |
|---|---|
| control | `constant`, `if`, `while`, `call`, `block_call`, `external_call`, `runtime_call`, `stop_gradient` |
| elementwise | `add`, `subtract`, `multiply`, `divide`, `remainder`, `power`, `negate`, `abs`, `square`, `sqrt`, `exp`, `log`, `log1p`, `sin`, `cos`, `tan`, `tanh`, `acos`, `sign`, `floor`, `ceil`, `round`, `relu`, `sigmoid`, `erf`, `gelu`, `nan_to_num`, `maximum`, `minimum`, `bitwise_and/or/xor`, `bitwise_left_shift`, `bitwise_right_shift`, `logical_and/or/not`, `cast` |
| comparison | `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal` |
| structure | `broadcast`, `concatenate`, `diag`, `flip`, `gather`, `pad`, `reshape`, `roll`, `scatter`, `select`, `slice`, `tile`, `transpose` |
| reduction | `reduce_sum`, `reduce_prod`, `reduce_max`, `reduce_min`, `reduce_mean`, `argmax`, `argmin`, `cumsum`, `cumprod` |
| sorting | `sort`, `argsort` |
| linalg | `dot` (incl. batched), `conv`, `solve`, `cholesky`, `eigh`, `qr`, `svd`, `matrix_rank`, `matrix_exp`, `diagonal`, `tril`, `triu` |
| collective | `rank`, `world_size`, `all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`, `broadcast_collective`, `collective_permute` |
| random | `random_key_mix`, `random_uniform`, `random_normal`, `random_randint`, `random_permutation`, `random_multinomial` |

`dot` is numpy's `matmul` contraction: a rank-1 operand is promoted to a 1-row (resp. 1-column) matrix and the batch dims broadcast.
Its plain matrix-matrix form in f32/f64 — both operands rank >= 2, C-contiguous batch blocks — runs `faer`'s BLAS-3 gemm (`Accum::Replace`, alpha 1), one call per `m × k @ k × n` block; integer, complex and f16 results keep the general triple loop, so they stay bit-exact and only the two IEEE float dtypes see a different (association-order) summation.
Semantics are the numpy backend's, op by op: `remainder` is floormod (sign of the divisor), `round` is half-to-even, `maximum`/`minimum` propagate NaN and reject complex, comparisons produce bool and complex supports only `equal`/`not_equal`, `abs` on complex is the real magnitude, integer `divide`/`remainder` by zero and a negative integer exponent are explicit errors, integer arithmetic wraps like numpy.
`if` takes a 0-d bool predicate in operand 0 and passes ALL operands as region arguments; `while` has regions `[cond, body]` with no iteration limit; `call` runs the callee's entry block.
Result validation matches the numpy interpreter's messages ("kernel for op '{op}' produced result {i} with …"), a dtype mismatch is a `BackendError` and a shape mismatch a `ShapeError`.
Collectives are single-rank identities (`rank`/`world_size` read `RunOptions`); `world_size != 1` is an explicit `BackendError` — there is no multi-rank executor in this engine.
Random is BIT-EXACT with `../../evoxc/backends/numpy/kernels/random.py` for splitmix64, threefry2x32 and philox4x32_10 (key handling, salt folding, counter layout, uniform/randint/permutation/multinomial/normal, including the f32 Box–Muller normal fast path).

## Performance (measured on this host, shared CPU)

Intel Core Ultra 7 356H, 16 cpus, Python 3.14.10 / numpy 2.5.2, loadavg 0.2–1.5 at the recorded runs (2026-11-23, shared host).
Warm `run` = median of 15–25 iterations after 3–5 warmups; the numpy leg is `evoxc.build(fn, *specs, backend="numpy", device="cpu")` + `evoxc.run` (`evoxc.evaluate` re-traces per call).
Reproduce with `.venv/bin/python native/tools/bench_cpu.py` (records `model name` and loadavg itself).
Every row of every recorded run reports `match yes`: the native and numpy results agree numerically on that workload.
Eight workloads, `--iterations 15 --warmup 3` in one process:

| workload | ops | native compile (ms) | native run (ms) | numpy run (ms) | speedup |
|---|---|---|---|---|---|
| `ew_chain_1e6` (8 elementwise ops, 1e6 f32) | 14 | 0.064 | 1.708 | 2.773 | 1.62× |
| `ew_transcendental_1e6` (`erf`/`gelu`/`tanh` chain) | 7 | 0.041 | 17.891 | 163.272 | 9.13× |
| `dot_sum_256` (256² f32 `dot` + row sum) | 3 | 0.025 | 0.305 | 0.117 | 0.38× (isolated 4.8×) |
| `gp_gather_4096` (GP-shaped gather/select chain, 4096×64) | 80 | 0.293 | 0.470 | 0.849 | 1.81× |
| `gp_sort_4096` (row argsort + sort + two-axis gather + broadcast select + reductions) | 14 | 0.072 | 3.563 | 2.056 | 0.58× |
| `tf_softmax_8x128x384` (per-row softmax + tanh/gelu mix) | 15 | 0.074 | 3.418 | 30.520 | 8.93× |
| `chain_2000` (2000-op elementwise chain) | 2000 | 6.343 | 0.117 | 1.272 | 10.89× |
| `tn_step_pop500` (2249 static ops, shape-faithful TensorNEAT `Pipeline.step`) | 2249 | 10.058 | 53.846 | 56.546 | 1.05× |

`native_compile_ms` is one full `native_compile(envelope)` call (envelope decode + FFI + engine plan compile); the engine's own plan compile is far cheaper and printed per workload by the harness, and a `constant` payload is decoded once at compile time.
`tn_step_pop500` under-weights exactly what the engine wins on (the many-small-op tail and the `while`/`if` region machinery) and over-weights large per-call shapes, so it is an approximation, not the acceptance measurement — the next section is the authoritative number.
`dot_sum_256`'s numpy leg is the unstable one (threaded OpenBLAS), so only its isolated ratio means anything: there the native 256² `dot` is 4.8× faster.
`gp_gather_4096` is process-state-sensitive on BOTH legs (see Known Issues): the quoted 1.81× is the low-load reading and inside a busier session it reads 1.39× (0.805 / 1.120 ms).

Reading of the numbers, which is load- and shape-dependent:
- The contiguous elementwise fast paths beat numpy on a 1e6-element mixed chain because per-op overhead is lower than numpy's ufunc dispatch, even though numpy fuses a whole expression into fewer passes.
- The transcendental chain wins by ~9× because `erf`/`gelu`/`tanh` go through scalar `libm` calls while numpy composes them from several ufuncs, and `exp` runs libm `expf` too (no polynomial arm), which keeps the softmax-shaped row at ~9×.
- `dot` at 256² runs `faer`'s gemm and loses to numpy's threaded OpenBLAS on a matrix this small, because the engine's own threading only engages above its `GEMM_*` work thresholds.
- numpy still wins where it has SIMD kernels safe Rust cannot express (`sort`/`argsort`, `astype`, `expf`) and on threaded BLAS; native wins where per-op dispatch dominates (`chain_2000` at ~0.06 µs/op), on transcendental compositions, and on every real many-small-op graph whose op mix includes control flow.
- Compiled plans are reusable across runs (`Arc<Plan>`), so a caller that keeps the plan pays only `run`.

### Real downstream workload: TensorNEAT 0.2.0 `Pipeline.step`

The authoritative workload is the external TensorNEAT 0.2.0 harness: one `evoxc.build` + one warm `evoxc.run` per generation over a 500-genome population.
Reproduce with `PYTHONPATH=<tensorneat>/src .venv/bin/python <tensorneat>/benchmarks/run_benchmark.py --impl new --backend numpy,native --cases xor_p500,cartpole_p100 --generations 8 --warmup 2 --repeats 1`.

| case | numpy ms/gen | native ms/gen |
|---|---|---|
| `xor_p500` (2249 static ops, ~7900 executed) | 49.0–49.6 | **40.4–41.0** |
| `cartpole_p100` (2238 static ops, ~31.6 k executed) | 68.5–69.6 | **24.1** |

Each leg is `--seed 42 --generations 8 --warmup 2 --repeats 3`; the quoted figures are the spread over seeds 42/43 and two interleaved passes.
Seed 44 (the third repeat) inflates both legs of `cartpole_p100` — native ~32–35 ms/gen and numpy 90.2 vs 69.1 ms/gen — a population-depth outlier, not a code effect, so only seeds 42/43 are scored.
The engine is held to the pre-migration reference (`e07bac8`, same host and session, A/B interleaved): `cartpole_p100` 23.9 → **24.1**, `xor_p500` 40.9–41.4 → **40.4–41.0**; the numpy legs are identical across builds (49.0–49.6 and 68.5–69.6), which is the control that the harness itself did not move.

### Per-element measurements (release, the `bench_*` instruments)

`gather` (`bench_gather`, ns/element): `f32 (500,100,2)` 1.08–1.10, `f32 (2000,50,4)` 0.56, `f32 (100,50,4)` 0.55, `bool (50000)` ~0.98, `i8 (500,200)` 0.71, the interleaved last axis 0.35.
`gather` one-element-per-index (the cartpole weight gather, `(5000,)[(100,50,4)]` indices, 300 calls per leg): 24.9 µs/call = 1.25 ns/element through the `store_singles` direct store, against 27.3 µs/call (1.37 ns/element) when the same plan is routed through `append_runs`; numpy's interpreter takes 23.3 µs/call for the equivalent graph.
`slice` (`bench_slice`, ns/element): `f32 (500,50,4)` 1.10, `(100,50,1)` 0.33.
Broadcast elementwise (`bench_broadcast_zip`, ns/element, out `(2000,50,4)` f32): `(4,)` bias 0.127, `(50,1)` per-row scalar 0.184, `()` scalar 0.086, `(2000,1,1)` per-block constant 0.106 — at the memory-bandwidth floor.
Reductions (`bench_reduce_flat_block_vs_lane_walk`, re-measured single-threaded after the pooled result): `(4096,64)` 17.3 µs, `(500,1,50)` 5.1 µs (was 9.4 — a 25 k-element result whose zero-fill the pool removed), `(500,50,4)` 39.7 µs.
The `faer` gemm A/B (`bench_dot_gemm`) records 256² at 59.6 GFLOP/s against the triple loop's 46.1.

### Large reductions: where the time went (`bench_reduce_large`)
`kernels/reductions/bench_reduce_large.rs` decomposes a whole-graph `reduce_sum` into its stages — input `Buffer::clone` (what `run_program` charges its caller, since it takes owned inputs), a `Vec::with_capacity`+`push` result, `fold_block` forced to 1 thread, `fold_block` at its own gated thread count, the kernel (`reductions::mod` dispatch) and the full `run_program` — over the 16M f32/f64 families, plus a `bench_reduce_large_threads` 1/2/4/8 sweep. Medians of 11 timed runs after 3 warmups, loadavg 1.6–2.8 on the 16-cpu host (2026-11-24):

| stage (ms) | full 16M→rank-0 | 4096² axis=1 | 4096² axis=0 |
|---|---|---|---|
| input `Buffer::clone` | 26.499 → **2.782** | 25.444 → **2.782** | 27.639 → **2.755** |
| `Vec::with_capacity`+`push` result | 30.878 | 30.607 | 30.749 |
| `fold_block` @1 thread | 2.419 | 2.666 | 3.113 |
| `fold_block` @policy threads | 2.517 | **0.889** | **1.592** |
| kernel dispatch (`reductions::mod`) | 2.426 → **2.525** | 2.857 → **0.915** | 3.123 → **1.630** |
| full `run_program` | 27.774 → **6.193** | 30.662 → **3.762** | 28.614 → **4.380** |

f64 (doubled bytes): full 16M `run` 57.105 → **9.374**, axis=1 56.738 → **6.277**, axis=0 57.351 → **6.736**.
The root cause was NOT the kernel: the fold was already at 0.15 ns/element (memory bandwidth) for every row. A fresh 64 MiB allocation costs ~1.8 ns/element, so the input clone and a push-built result — both plain allocations — were 25–31 ms of the 27.7 ms, and the kernel was 2.4 ms.
Thread sweep (`bench_reduce_large_threads`, entries are what `fold_threads` returns and the best measured count):

| shape | policy threads | t1 | t2 | t4 | t8 | best@ |
|---|---|---|---|---|---|---|
| full 16M | 1 | 2.568 | 2.527 | 2.537 | 2.533 | 2 (noise; one output) |
| 4096² axis=1 | 8 | 2.630 | 2.565 | 1.627 | **1.083** | 8 |
| 4096² axis=0 | 8 | 3.624 | 3.887 | 2.027 | **1.571** | 8 |
| `(2048,2048)` axis=1 (2^22) | 8 | 0.509 | 0.233 | 0.143 | **0.117** | 8 |
| `(2048,1024)` axis=1 (2^21) | 4 | 0.193 | 0.098 | **0.069** | 0.094 | 4 |
| `(1024,1024)` axis=1 (2^20) | 2 | 0.076 | 0.056 | **0.046** | 0.075 | 4 |
| `(4096,64)` axis=1 (2^18) | 1 | **0.017** | 0.028 | 0.037 | 0.069 | 1 |
| `(500,50,4)` axis=2 | 1 | **0.040** | 0.048 | 0.043 | 0.067 | 1 |
| `(500,1,50)` axis=1 | 1 | **0.005** | 0.019 | 0.034 | 0.061 | 1 |

Threading wins 2.3–4.4× where the output count is large and loses below 2^19 elements, which is exactly what `REDUCE_MIN_PER_THREAD`/`parallel_threads` gate on (the 2^20 row is the one place the gate leaves ~1.6× on the table — a 1/2/4/8 policy tuned per element count is the remaining optimization).

### The honest losses, with their reason — these are floors, not tuning gaps

- **`dot` vs threaded OpenBLAS.** 0.32× at 256² inside the sequence (4.8× faster isolated): numpy's BLAS threads a matrix this small while the engine's `gemm_threads` runs `Par::Seq` below its work thresholds.
- **`sort`/`argsort` vs AVX2/AVX-512.** 0.53× (`gp_sort_4096`): the Batcher networks are scalar, and explicit SIMD would need `unsafe`.
- **`cast`/`expf` vs numpy SIMD.** numpy's `astype` ~0.21 ns/element against the scalar `convert_generic`, and its vectorized `expf` ~0.58 against libm's ~1.0.
- **`random_normal`'s word-stream tail.** The Box–Muller path materializes the salted word stream, the broadcast parameter chain and a `Vec<ScalarValue>` pass per call (~14.5 ns/element against numpy's 7.8); `random_uniform`/`random_randint` take the typed contiguous path and are 2–3× faster than numpy.
- **`gather`'s multi-axis residual.** The middle-axis `f32 (500,4,2)` family (a strided destination) measures 1.48 ns/element, and the one-element-per-index family's direct store is a scalar store per index (1.25 ns/element) where numpy's `take` is vectorized.
- **The copy-vs-view floor.** `Storage` owns plain `Vec`s, so `reshape`/identity-`transpose`/`slice`/`broadcast` materialize a full array where numpy returns a view (`np.reshape`/`arr.transpose`/`np.broadcast_to`); a view/Tensor-owning-a-layout refactor is the documented deferral.

## Optimizations

- **Recycled storage pool (`buffer/pool.rs`):** a thread-local free list of large result storages — `impl Drop for Storage` hands an allocation of at least `POOL_MIN_BYTES` (4 KiB) back to the pool instead of to `malloc`, and `take_pooled_vec::<R>(count)` hands a stock of exactly the requested `(dtype, element count)` out again, which removes the per-op mmap/page-fault cost on many-small-op graphs.
- Retention is bounded per thread: at most `POOL_MAX_ENTRIES` (64) values totalling `POOL_MAX_BYTES` (64 MiB), and at most `POOL_MAX_PER_SIZE` (8) stocks of any one exact size class.
  A stock of at least `POOL_LARGE_MIN_BYTES` (8 MiB) lives in its own LARGE class — a separate `POOL_MAX_BYTES_LARGE` (256 MiB) budget and `POOL_LARGE_PER_SIZE` (4) stocks per size — so a 64 MiB operand or result can be recycled without evicting the small-op working set (and vice versa); that is what makes a repeated 16M f32 `Buffer::clone` cost 0.166 ns/element instead of a fresh-allocation 1.8.
- A full size class evicts its own least-recently-used stock rather than refusing the incoming buffer, and admission evicts across size classes by least recent use, so a preceding unrelated workload can never starve the one that is running.
- A pooled vector is only handed to a caller that overwrites every element (the flat maps gate on an exact length match), so no stale value can reach a result; the pool is thread-local and never shared or synchronized.
- **Plan-time specialization (`plan.rs`):** a `constant` op's payload is decoded into a `Buffer` once at compile time (`Step.constant`), so a run clones a buffer instead of re-decoding the same bytes, and a plan-time constant is permanently excluded from the in-place candidates.
- Each result's static `(dtype, shape)` is precomputed at compile time (`Step.fast`, `None` for a symbolic or dynamic dim), so a run does not re-evaluate `Dim`s or `DimExpr`s for a static result.
- `Step.in_place` records a dying, uniquely-ownable operand that a kernel may overwrite; the plan only marks a binary elementwise op whose two operands are distinct and whose declared types equal the result type, and never a block argument, a `return` operand or a plan-time constant.
- **In-place accumulation (`exec.rs::try_in_place`, `kernels/elementwise/binary.rs::accumulate_as` / `buffer/zipmap.rs::zip2_accumulate`):** the whitelisted kinds `add`, `subtract`, `multiply`, `divide`, `maximum` and `minimum` accumulate into a freed operand when liveness proves nobody else reads it, and the run falls back to the allocating kernel when the buffer is still shared.
- The kind is dispatched ONCE per call into monomorphic arms (`arm!(Add, Sub, Mul, Div, Max, Min)`), so LLVM still auto-vectorizes the loop; a per-element `match` on the kind defeats vectorization, which is why the dispatch is arm-for-arm.
- **Contiguous dense fast paths (`kernels/elementwise/{binary,compare,unary}.rs`):** a `Layout` probe picks `Flat`, `Scalar` or `Broadcast` per operand, and the flat arms — `dense_unary`, `dense_binary` / `dense_binary_left_scalar` / `dense_binary_right_scalar` (the 13 arithmetic, bitwise and shift kinds) and `dense_compare` / `dense_compare_right_scalar` / `dense_compare_left_scalar` — run one auto-vectorized typed-slice loop through `map1`/`map2`/`map3`, with the kind dispatched once per call.
- `select` (`select_apply`) gates the same broadcast-free case on `Buffer::is_flat_for` and maps with `map3`, and `cast` converts whole storages elementwise (`convert_generic` in `buffer/storage.rs`), so the whole comparison/cast/select family stays on typed slices.
- Each fast path builds its result storage from the mapped iterator instead of zero-filling it first, and takes a pooled vector when one of the exact length is available.
- **The broadcast drivers (`buffer/broadcast.rs`, `buffer/broadcast/patterns.rs`):** the `Layout::Broadcast` arms of `binary`/`compare`/`select` enter `map2_runs`/`map3_runs`, which first try the repeating-run driver (`result[i] reads operand element i % period`, proved exactly from the shapes and broadcast strides by `repeat_view`) and then the periodic-pattern driver (`values[(i / stretch) % span]`, proved by `pattern_view` and driven by const-generic run widths 2/4/8); what they decline runs as one `ndarray::Zip` pass over the broadcast views, the reference.
- The run drivers write whole contiguous runs out of one tile — one fixed-size array map or one `fill`/`copy_from_slice` per run, with no per-element stride or offset arithmetic — and only above `RUN_MIN_ELEMS` (64); a layout that is neither an exact `i % period` nor a periodic pattern, or a result under `RUN_MIN_ELEMS`, keeps the `Zip`.
- **Wide maps split over scoped threads (`buffer/parallel.rs::split_parallel`):** one `std::thread::scope` per wide kernel call over disjoint contiguous output ranges, gated at `PARALLEL_MIN_ELEMS` (2^18 elements) with one thread per `PARALLEL_MIN_PER_THREAD` (2^17) elements and capped at `PARALLEL_MAX_THREADS` (4) — raised to `PARALLEL_MAX_THREADS_LARGE` (8) at or above `PARALLEL_LARGE_ELEMS` (2^22, a 16 MB f32 result), because a DRAM-bound 16M-element map keeps scaling past four threads (measured on 64 MiB results: 1t 9.34, 2t 5.90, 4t 4.85, 6t 4.71, 8t 4.57, 12t 4.67, 16t 4.77 ms) while the per-op thread spawn stays ~0.11–0.20 ms.
- The pooled broadcast builders split differently (`buffer/rowsplit.rs::split_rows`): row-aligned outermost-axis chunks, each operand riding as a contiguous row block (`Rider::Rows`) or a whole shared block (`Rider::Whole`), leading chunk on the calling thread; a result that is not row-splittable keeps the single `apply2`/`apply3` call.
- Each thread reads its own operand window at its own running offset, so results are identical to the single-threaded loop; the engine's own threads are scoped inside one call, never a pool.
- **Reductions (`kernels/reductions/{flat,layout}.rs`):** a contiguous reduced block is folded flat by `fold_block` in one pass over the slice (no per-lane view construction), with an 8-lane (`fold_lanes`) unrolled `fold_run` whose independent accumulators break the sequential dependency chain and one accumulator per output lane for `inner > 1`; every other layout takes the ndarray `axis_iter` lane walk.
- The result is taken from the pool (`take_pooled_vec::<T>(out_len)`, else `Vec::with_capacity`) and every slot is written by an index store, so no reduction pays a zero-fill of its own result: the 16M f32 full sum writes one element, an `outer × inner` block writes each output exactly once.
- A flat fold is **memory-bound at ≥ 2^19 elements** (0.150 ns/element on 16M f32 through `fold_run`, the same as the whole kernel), so the only thing left to remove was allocation: the `run` path measured 27.7 ms while the kernel core measured 2.4 ms, and the missing 25 ms was the **input `Buffer::clone`** (a fresh 64 MiB `Vec` → mmap page faults, 26.5 ms) plus a `Vec::with_capacity`+`push` result (31 ms).
- `Storage::clone` (`buffer/storage.rs`) and the reduction result therefore both take pooled storage and copy through `split_parallel` above `PARALLEL_MIN_ELEMS`: the same 16M f32 clone measures **2.78 ms** (0.166 ns/element) instead of 26.5 ms, and a `Vec::with_capacity`+`push` result of that size costs ~30 ms, which is why no hot path may build a large result by pushing.
- The block fold splits over **independent outputs** only, gated by `fold_threads`/`REDUCE_MIN_PER_THREAD` (2^19 elements per thread) and never inside one run — `inner == 1` hands each thread a whole output's contiguous run and `inner > 1` hands it a whole output-lane window, so every output's summation order is unchanged: `the_block_fold_is_bit_identical_at_every_thread_count` pins 1/2/4/8 threads against each other, and `fold_block` itself is pinned against the ndarray lane walk.
- Measured policy (`bench_reduce_large_threads`, next section): the gate picks the best thread count at 2^18, 2^21 and 2^22 elements and within ~25 % of it at 2^20, and it keeps a full 16M→rank-0 sum single-threaded, which has one output and can only gain from removing overhead.
- Max/min folds go through the NaN-propagating `FoldSelect` (seeded from each run's first element, no identity), and the arg reductions pick their scan by run length: `arg_run_branchless` below `ARG_BRANCHY_MIN_RUN` (1024) elements and the plain `arg_run_branchy` scan at or above it.
- **Sorting (`kernels/sorting/`):** a packed monotone row key `(rank_key << 32) | slot` with `u32` positions, so one `u64` comparison orders both the value and its index for `sort` and `argsort`; 64-bit keys sort an index vector by `(key, index)`.
- Extents of 2..=`NETWORK_MAX` (16) run branchless Batcher odd-even networks of their own pad width (2/4/8/16 keys, `PAD_KEY = u64::MAX` sentinel) with every comparator index a literal; every extent outside that range keeps `sort_unstable` over the packed keys.
- The per-axis layout is a flat C-contiguous base/stride segment walk (kept hand-written: an ndarray `lanes()` expression measured 2.7–6.3× slower), a non-contiguous axis gathers each segment into one reused scratch buffer, and the scratch buffers are stack-resident up to `STACK_KEYS`.
- **`dot`'s gemm (`kernels/linalg/matmul.rs`):** `faer::linalg::matmul::matmul` does the packing, register blocking and SIMD for one `m × k @ k × n` block, `gemm_threads` chooses `Par::Seq` / `Par::rayon(n)` from the `GEMM_PARALLEL_MIN_BLOCK_WORK`/`GEMM_PARALLEL_MIN_WORK`/`GEMM_PARALLEL_WORK_PER_THREAD` work thresholds (capped at `GEMM_PARALLEL_MAX_THREADS` (8) below `GEMM_WIDE_WORK`), and `faer` splits the OUTPUT — every output element still reduces over K in the same order, so the threaded result is bit-identical to the one-thread result (`dot_parallel_matches_the_single_threaded_path_bit_for_bit`).
- `dot` casts only a differing operand (`dot_impl` compares each operand's dtype with the compute dtype before calling `Buffer::cast`), so the common same-dtype matmul is copy-free on the input side.
- A large result block's storage comes from the pool too (`pooled_destination`, the `gemm_run_with` destination hook): a 2048² or 4096×1024 f32 output is 16–32 MiB, so a fresh `vec![T::zero(); n]` per `dot` paid the same ~0.4 ms/MiB first touch the reduction path did; measured medians, `4096×1024 @ 1024×4096` 63.7–76.1 ms → **30.7–31.9 ms** and `2048³` 28.98 → **~24 ms**, while `1024³` stays a wash (its output is only 4 MiB).
- **The two-sided run copier (`kernels/indexing/strided_copy.rs`):** `slice`/`reshape`-transpose/`gather` build an innermost-first run plan (`CopyPlan`/`AxisCopy`/`RunCopy`, collapsed where strides line up) with no odometer; a contiguous innermost run is one `copy_from_slice` per run, a strided one is `fill_strided`'s `LANES = 4` lane-grouped window, and the whole plan is validated up front and applied to a pooled destination of exactly the result size.
- `slice`/`transpose` take a pooled destination and copy a C-contiguous view with one `copy_from_slice` (`kernels/indexing/{slice,reshape}.rs`); a strided view goes to the run copier and a layout it declines (a backwards step) to ndarray's `assign`.
- `gather` resolves its index operand inline inside the run copy (`gather_runs`: one contiguous or strided `A`-element run per index, no intermediate `Vec<usize>`) and takes an append destination from `pooled_append` when the positions continue the output in order (no zero fill) or `pooled_total` otherwise.
- **`gather`'s one-element-per-index direct store (`RunCopy::store_singles`):** when the run plan is exactly one element per index and the result holds exactly that many elements, the run copier stores straight into a pre-sized `pooled_total` destination (`*slot = *value`, bounds-checked per element) instead of growing the destination with a `Vec::push` per element (`append_runs`).
  That is the shape of cartpole's weight gather `(5000,)[(100,50,4)]` — the largest single data movement in that workload, ~8 M elements per generation — and it measures 1.25 ns/element against 1.37 ns/element through the push path.
  Every other appending family still goes through `append_runs`, whose positions continue the output in order and therefore pay no zero fill.
- **`pooled_append` (`kernels/indexing/mod.rs`):** the append-only counterpart of `pooled_total` — a recycled stock with its length reset or a fresh reservation, both fully written by the run copy, so the run-copier paths pay no memset.
- **`pad` is a row-wise copy, not an index map (`kernels/structure/pad.rs`):** each innermost-axis row is either pad-only (one `fill`) or `before` pad cells, one `copy_from_slice` of its interior run out of the contiguous operand and `after` pad cells; an all-zero config is one bulk copy and a rank-0 operand one element, and the config is validated against the declared result shape before any allocation.
- **`tile` is one ndarray broadcast assignment (`kernels/structure/tile.rs`):** the operand is viewed with a size-1 axis interleaved before every axis and broadcast to `[rep, extent, ...]`, whose C order is `np.tile`'s `index % extent` mapping; a whole-source repeat builds its storage with `repeat_storage` (one `copy_from_slice` per run-length chunk).
- **Outputs freed before extraction:** a value is released from the flat store at its last use and the run clears what is left of the store before pulling the output buffers out, so an output is moved out of its `Rc` instead of copied (a value returned twice still copies), and the intermediates released there hand their storage back to the pool.
- No SIMD intrinsics appear anywhere in the crate (`unsafe` is absent), so elementwise throughput rests on LLVM auto-vectorizing safe slice loops.
- **Per-run setup is small and allocation-bounded:** a run never recompiles or re-decodes the plan (`run_program` takes `&Arc<Plan>`), and only allocates one flat `store: vec![None; plan.max_value]`, one `Bindings` map and one reusable `operands` scratch `Vec` (`exec.rs` `RunCtx::new`); `run` borrows the entry `FunctionPlan` signature for its input/output validation, and the `call` kernel resolves a callee by linear name search per call op (`kernels/control.rs` `call_kernel`).
- **`exp` on f32 uses libm, with no polynomial arm:** glibc's vectorized `expf` is faster than a hand-rolled precheck + Taylor arm at every size and the more accurate of the two, so `exp` runs the ordinary `Num::exp` path for every dtype.

## Constraints

- Depends on `evoxc-core` only — never `evoxc-compiler`, `evoxc-runtime`, `evoxc-python` or PyO3. Third-party deps: `ndarray` (the array library: every N-d axis/layout/broadcast rule), `faer` (the one numerical-performance dependency: `dot`'s f32/f64 gemm and the test-only factorization oracles), `half` (f16), `num-complex` (complex), `libm` (`erf`, `ln_1p`) and `serde_json` (`block_call.static_args`).
- `ndarray` is built with its default features only (no `blas`, no `rayon` feature); `faer` is built with `default-features = false, features = ["std", "rayon"]`, and its `rayon` feature is the crate's ONLY threading library and is used solely by `dot`'s gemm above the work thresholds.
- **No `unsafe` anywhere in the crate.** There are no raw-pointer or FFI blocks and no SIMD intrinsics, so elementwise throughput and the gemm both ride on safe Rust plus LLVM auto-vectorization.
- **No foreign-kernel hook.** The kernel table is a closed compile-time map of Rust `fn` pointers (`KernelMap = HashMap<&'static str, KernelFn>`) built once into a process-wide `OnceLock` behind `kernels::table()`, and the only host-supplied code the engine ever calls is a per-run `Option<&dyn Callbacks>` for `block_call`/`external_call`/`runtime_call`; there is no dlopen/libloading/dlsym, no `extern "C"` pointer, no plugin registration and no Triton/custom-call path in this crate.
- Threading policy: the engine's own threads are scoped inside one kernel call only — never a persistent pool, never a background worker — gated by the real work thresholds (`PARALLEL_MIN_ELEMS`/`PARALLEL_MIN_PER_THREAD` for flat maps, `REDUCE_MIN_PER_THREAD` and the output count for the reduction block fold, the row count for the pooled broadcast builders, `GEMM_*` for `dot`), and results are bit-identical because each thread owns a disjoint output range and reads only its own operand window; `faer`'s `rayon` gemm applies only above those `GEMM_*` thresholds.
- A single reduction run is NEVER split across threads: the lane reassociation inside `fold_run` is part of the documented numerics, so only independent OUTPUTS are distributed.
- The storage pool is thread-local and never shared or synchronized, so no lock and no cross-thread allocation exists.
- Never a second layout engine: new N-dimensional addressing is an ndarray view/`Zip`/`indices`/`axis_iter` expression, never a hand-written increment/decrement odometer. The two run drivers (`repeat_view`, `pattern_view`) address whole contiguous runs from a layout the shapes and broadcast strides prove exactly, and the sorting segment walk and the `strided_copy` run map are run maps with no per-element stride arithmetic.
- `Buffer`/`Storage` keep their owned representation and public fields; result buffers are always fresh C-contiguous `Vec`s (a shape-only op copies — the documented cost of having no views in `Storage`).
- User errors are `Result` values — no `unwrap`/`expect` on user input, no panics on malformed graphs; a shape or dtype the fast path does not cover falls back to the general kernel, never to an error.
- Result types and shapes come from the graph; never re-infer, and always validate produced buffers against `OpSpec.result_types`.
- Symmetry rule: every fast path is a documented SPECIALIZATION whose ndarray expression is the reference in tests — elementwise→`Zip`, reductions→`axis_iter`, the run drivers→`Zip`/`assign`, the run copier→`assign`, `dot`'s gemm→the triple loop.
- One concern per file, target <~1000 lines including inline tests; a category that outgrows that becomes a module directory (`kernels/{elementwise,indexing,structure,sorting,reductions,linalg,linalg_aux,random}/`, `buffer/`) with a thin `mod.rs` holding `pub fn register` and the re-exports the rest of the crate reaches, one submodule per op family or shared mechanism, and a `#[cfg(test)] mod test_support`/`test_util` for fixtures sibling test modules share.
- `kernels/linalg/matmul.rs` is the ONE justified exception (~1364 lines): it owns both `dot`'s semantics and the BLAS-3 specialization that must agree with them, and splitting the acceptance gate away from the two implementations it gates would make the bit-exactness contract impossible to review in one place.
- `cargo test -p evoxc-cpu` and `cargo clippy --workspace --all-targets --no-deps -- -D warnings` must stay clean.

## Test strategy

- `nan_to_num` reads its three `any` replacements from the attrs: `posinf`/`neginf` default to JSON `null` (the dtype's max/min finite value — numpy's own default), `nan` is `0.0` when absent and keeps NaN when explicitly `null` (the IR op's "None keeps the value"; numpy itself raises for `nan=None`), and a `null` replacement is never a type error. Pinned end-to-end (envelope decode → compile → run) in `tests/nullable_attrs.rs`, which also covers `flip.axes`, `roll.axis`, `argmax.axis`, `transpose.permutation`, `slice.strides`, `matrix_rank.tol`, `conv.strides`/dilations and the collective `group_size`.
- In-crate `#[cfg(test)]` per module: **396 lib tests, 24 ignored** (`cargo test -p evoxc-cpu --lib`; the crate's full target set — those 396, the 13 below, the ignored `tests/bench_large.rs` rows and `tests/perf_probe.rs` — is green in release) plus 13 in the `tests/nullable_attrs.rs` integration target — kernel numerics (promotions, NaN/complex/rounding edge cases, error cases), symbolic-dim binding and the `DimExpr` failure modes, control flow (nested regions, multi-carry loops, recursion depth), the bit-exact random vectors for all three algorithms, collectives, the registry-inventory tests that pin `unsupported_ops()` to empty and the two `custom`-category callback ops to "no kernel, never in `DENSE_OPS`", and the `plan` compile-rejection tests that assert `custom_call`/`inline_stablehlo` raise a named `BackendError` (with the `[at file:line:col]` suffix when a location is known) while an ordinary graph still compiles with an empty plan-level inventory.
- The layout substrate is pinned against INDEPENDENT references: `broadcast/tests.rs` runs every `Zip`/`walk` driver, the repeating-run driver and the periodic-pattern driver against a plain index-arithmetic `reference_slot` and against `apply2`/`apply3` with the specialization forced off, over i32/f32 and a rank/shape matrix (`[2,3]`, `[2,2,2]`, `[4,3,5]`, `[4096,64]`, `[1000,1000]`, `[3]`, rank 0, zero-length axes), with NaN/`-0.0` bit pins and a pooled garbage stock that must be fully overwritten; the whole file serializes on one mutex because `RUNS_DISABLED`/`SPLIT_DISABLED` are process-wide.
- The row split is pinned in `zipmap.rs`: `the_row_split_agrees_with_the_per_element_reference`, `the_row_split_agrees_with_the_per_element_reference_for_select`, `the_row_split_covers_whole_rows_or_stays_whole`, `an_operand_that_cannot_ride_the_split_keeps_the_single_threaded_behaviour` and `a_split_result_is_fully_overwritten_from_the_pool`.
- The elementwise/comparison/select families are pinned against an independent ndarray `Zip` expression written out in each test (a plain `Zip` for binary/compare/select), so a fast path can only change speed, never a value.
- The reductions are pinned flat-fold-vs-ndarray-lane-walk (`the_flat_block_fold_matches_the_ndarray_lane_walk`), the arg scans (`the_two_arg_scans_agree_on_short_and_long_runs`, `arg_reductions_keep_the_first_nan_and_the_first_of_a_tie`) and the sorting networks against `sort_unstable` (`the_network_covers_only_the_extents_it_wins_at`, `matches_the_reference_across_the_network_extents`, `matches_the_reference_past_the_network_ceiling`, `nan_sorts_last_through_the_padded_network`).
- The run copier (`strided_copy_tests.rs`, a `#[path]` child module) pins `copy_view` against ndarray `assign` on every stride and its decline of a negative stride, the `AxisCopy` interleaved families, `append_runs`/`place_runs` on the flat-table and axis-last gathers, `plan_axis_runs`'s decline of a middle axis, float-bit pass-through, and the index-error propagation; `gather`'s direct-store arm is pinned by `gather_one_element_per_index_fills_every_slot_in_c_order` and `gather_one_element_per_index_reports_the_first_bad_index_in_order`.
- The contiguous-copy paths pin their own behaviour: `slice`'s `the_last_axis_single_element_family_matches_the_reference` and `a_whole_array_slice_is_one_contiguous_run`, `transpose`'s view copy, and `pad`'s `pad_is_asymmetric_on_two_axes`, `pad_copies_a_long_row_and_fills_only_the_pad_cells`, `pad_places_every_element_of_a_padded_result`, `pad_matches_the_per_element_reference`, `pad_fills_every_dtype`, `pad_rejects_a_result_shape_that_contradicts_the_padding`, `pad_rejects_a_mismatched_config_and_negative_padding`, `a_pooled_pad_result_leaves_no_stale_element` and `pad_handles_empty_extents`.
- `tile` is pinned against the equivalent ndarray broadcast (`the_whole_source_repeat_agrees_with_the_ndarray_broadcast`, `a_leading_axis_rep_repeats_the_whole_operand`), and the pool/parallel/`pooled_append` paths keep their own pins (`stocks_are_retained_per_size_up_to_the_per_size_cap`, `the_pool_evicts_the_least_recently_used_other_size_class`, `stock_reuse_is_last_in_first_out`, `the_thread_split_hands_every_chunk_its_running_offset`, `the_thread_split_scales_with_the_work_per_thread`, `pooled_append_drops_a_recycled_stock_and_leaves_no_stale_element`, `pooled_append_reserves_without_writing_on_a_pool_miss`).
- `dot`'s gemm is pinned for bit-identity across thread counts (`dot_parallel_matches_the_single_threaded_path_bit_for_bit`, `dot_faer_gemm_agrees_with_the_triple_loop`) and `ndarray_views.rs` round-trips all 14 dtypes through `from_array`/`into_array`/`as_array`, including a rank-0 buffer, a zero-element shape, a dtype mismatch naming both dtypes, a non-standard-layout `from_array`, and `into_array` moving the allocation.
- The `#[ignore]`d measurement aids never run in the normal suite: `bench_thread_crossover` (flat-map crossover), `buffer::bench_broadcast` (`bench_broadcast_zip`, `bench_broadcast_zip_threads`, `bench_walk`), `kernels::sorting`'s `bench_segment_walk`, `kernels::reductions::flat`'s `bench_reduce_flat_block_vs_lane_walk`/`bench_reduce_large_fold`, `kernels::reductions::bench_reduce_large`'s `bench_reduce_large` (the whole-graph stage decomposition) and `bench_reduce_large_threads` (the 1/2/4/8 sweep), `kernels::indexing`'s `bench_gather`/`bench_scatter`/`bench_slice`/`bench_concatenate`, `kernels::linalg::matmul`'s `bench_dot_gemm`, and the `linalg_aux` faer A/B evidence benches.
- The reduction block fold's thread split is pinned by `kernels::reductions::bench_reduce_large::the_block_fold_is_bit_identical_at_every_thread_count` (1/2/4/8 threads against the single-threaded fold over `inner == 1`, `inner > 1` and rank-0 shapes), and `the_probe_fixtures_agree` pins the probe's hand-built graph against the real kernel path.
- `tests/perf_probe.rs` is the `#[ignore]`d per-op cost probe: one op per graph (`slice`, `reshape`, `pad`) through `envelope::decode` + `plan::compile` + `run_program`, minus the same-input one-element-output floor, run with `cargo test --release -p evoxc-cpu --test perf_probe -- --ignored --nocapture`.
- Python parity harness `../tools/parity_cpu.py` (needs `scripts/build-native.sh` first and a `.venv`) compares `NativeProgram` against the numpy backend over elementwise/reductions/indexing/linalg/control-flow/random/dynamic-shape graphs plus the error and extension-resolver cases (the recorded run is 73/73, 4 flagged `PASS~`); it asserts exact dtype/shape/count/order, `array_equal` for bool/int, `allclose` for floats (`rtol 1e-5 / atol 1e-6` for f16/f32/c64 and `1e-12/1e-13` for f64/c128) unless a case opts into bit-identity, bit-exact RNG leaves except the three f32 `normal` rows (the documented 1-ulp Box–Muller window), and `dot_f32_large` (64×96@96×48) at `atol 1e-5` because `faer`'s blocking differs from OpenBLAS `sgemm`; error parity is asserted when both sides raise.
- `../tools/bench_cpu.py` reproduces the eight-workload table above.
- Tests never require a GPU, the network or Python (the Python harnesses are separate scripts, not cargo tests).

## Known Issues / Notes for Agents

- **`random_split_n` salts above `u64::MAX`:** `evoxc/ops/random.py` stamps `random_split_n`'s salts as `i * 0x9E3779B97F4A7C15`, which for `i >= 2` exceeds `u64` (`2 * GOLDEN` = 22801429638646396970) and so has no 64-bit envelope encoding. `../tools/parity_cpu.py` reduces such values modulo `2**64` (the same mask the numpy kernel applies through `np.uint64` arithmetic), and the real fix belongs to the frontend, out of this node's write scope.
- `evoxc-core`'s `int` attribute decode accepts any value that fits `u64` (it tries `as_u64` first and falls through to `as_i64` for negatives), so a single 64-bit salt such as the SplitMix64 golden gamma travels verbatim; `evoxc-python`'s inbound integer conversion likewise accepts `i64` then `u64`, so no 64-bit unsigned value needs narrowing.
- **`sort`/`argsort` are always stable:** with `stable=false` and duplicate values, numpy's quicksort may return a different (equally valid) permutation. One stable code path; ties are order-stable rather than numpy-quicksort-order.
- **`sort`/`argsort` lose to numpy's AVX2/AVX-512 sorting networks:** the Batcher odd-even networks are scalar (0.53× on `gp_sort_4096`), and explicit SIMD intrinsics would need `unsafe`, which the crate forbids.
- **`sort`/`argsort` each compute their own permutation:** `sorted_positions` allocates a full-size `Vec<u32>` (`kernels/sorting/permutation.rs`), gathers every non-innermost segment through a per-segment scratch copy, sorts long extents with scalar `sort_unstable` (only extents ≤ 16 use the Batcher network) and resolves wide `f64`/`i64` keys through an index comparator over packed integer keys, so a frontend `topk` = sort + argsort + gather composition pays the whole ordering twice per call.
- **Elementwise throughput is per-op, not fused:** contiguous same-shape binary/unary ops and the comparison/cast/select family take dedicated slice fast paths and lead numpy on a 1e6-element chain, but there is still no cross-op fusion, so an expression numpy can fuse into a single pass can win on very large single arrays.
- **`dot` loses to numpy's threaded OpenBLAS on small matrices:** 0.32× at 256² in sequence (4.8× faster isolated), because `faer`'s gemm runs `Par::Seq` below the `GEMM_*` work thresholds while numpy's BLAS threads a matrix this small at hundreds of GFLOP/s.
- **`gp_gather_4096` is process-state-sensitive, not graph-limited:** inside the eight-workload sequence it reads 0.94× (0.789 ms vs 0.740 ms) and alone in a fresh process 1.37× (0.55 ms vs 0.74 ms), because the per-op scoped-thread splits pay thread-wake latency that varies with host load and core state; the remedy would be a persistent worker pool, which stays out of the design.
- **`power` is specialised apart from the rest of the elementwise family:** `kernels/pow.rs` owns numpy's `npy_pow` arm table (`0`/`1`/`2`/`3`/`0.5`/`-1`/small-integer exponents, f32/f64 only, scalar exponent only) plus the negative-integer-exponent guard; `elementwise::binary` calls into it for the `power` kind and keeps the generic driver for everything else.
- **Single-rank only:** the collectives are identity kernels guarded by `world_size == 1`; a rank/world-size context exists (`RunOptions`) but no multi-rank executor.
- **Host callbacks are the caller's job:** `block_call`, `external_call` and `runtime_call` dispatch through the `Callbacks` trait; `run_program` refuses a callback-needing plan without one, and the FFI's `resolver` is `None` by default. Results returned by a callback are validated like any kernel result.
- **f32 `normal` is not bit-exact by design:** it shares the numpy backend's Box–Muller fast path (within 1 ulp); every other random path (integer words, `uniform`, f64 `normal`) is bit-exact.
- **`sparse_*` and the `custom` callback ops are out of scope:** the dense engine rejects the 16 sparse ops with an explicit `BackendError` naming the op, mirroring the numpy backend's `SparseError`/compiler-backend deferral; densify with the graph's `sparse_to_dense` first.
  The `custom`-category `custom_call`/`inline_stablehlo` ops carry opaque payloads (an external target name, an inlined StableHLO body) the engine cannot execute, so they are absent from `DENSE_OPS` and have no kernel (not even an erroring one): `plan::compile` rejects any graph that uses one with the same named `BackendError`, and because such a graph never produces a plan the plan-level `unsupported` inventory stays empty.
- **Shape-only ops copy the whole array:** `reshape` and an identity `transpose` take a pooled destination and copy the whole storage, a same-shape `broadcast` clones, and a `constant` step deep-clones its plan payload once per use; `Buffer`/`Storage` own plain `Vec`s, so each is a full array memcpy where the numpy backend returns a view (`np.reshape`/`np.broadcast_to`/`arr.transpose`). A view/Tensor-owning-a-layout refactor is the documented deferral; until then `reshape`/`transpose`/`slice` stay at memcpy speed. `pooled_append` gives the run-copier paths a destination with no memset (a recycled stock with its length reset, or a fresh reservation).
- **`gather`'s residual is the multi-axis families:** the flat-table run families are at 0.55–1.10 ns/element (`gather_runs` resolves the index inline), the one-element-per-index family stores directly at 1.25 ns/element (`store_singles`), and the middle-axis `f32 (500,4,2)` strided-destination family measures 1.48 ns/element; every `run == 1` family still issues one scalar store per index where numpy's `take` is vectorized, and `pooled_total` pays a full `T::zeros` memset on a pool miss, where `pooled_append` does not.
- **`faer`'s global parallelism keeps it out of the factorizations:** `faer`'s `Llt::new`/`Qr::new` read faer's global parallelism (all cores by default) and the `Par`-taking low-level entry is unreachable (`MemStack` is crate-private, no `dyn-stack` dependency), so `cholesky`, `qr`, `svd`, `eigh`, `matrix_rank` and `matrix_exp` keep their hand-written deterministic kernels; the per-op A/B verdicts and reasons live in `kernels/linalg_aux/mod.rs`.
- **The factorizations have honest perf gaps against faer:** `qr` at 1024² is 5.16 s against faer's 0.089 s (0.31× at 64², 57.8× at 1024²), and `eigh` is 145× slower at 256² and 312× at 512² — kept because the `faer` results differ on rank-deficient or degenerate inputs (a null-space basis or a tied eigenspace has no sign/order convention to reproduce).
- **A flat map over ≥ 2^18 elements spawns scoped threads per op:** `PARALLEL_MIN_ELEMS = 1 << 18` with one thread per 2^17 elements capped at 4, and each split is one `std::thread::scope` at a documented ~10 µs setup per extra thread; the pooled broadcast builders split over row-aligned chunks instead and pay ~4 µs on a memory-bound result.
- **Result allocation is pooled on the hot kernels only:** `map1..3`, the run copier (`block_storage_filled`/`fill_strided`), `pooled_append`, `reshape`/`transpose`/`pad`, the comparison/logical `bool_results` family and `cast`'s `convert_generic`, the reduction result and `Storage::clone` take pooled storage, while the general broadcast/strided drivers still zero-fill (`vec![T::zero(); n]`), `gather`/`concatenate` take a `pooled_total` destination (a full `T::zeros` memset on a pool miss) and `sort` allocates with `Vec::with_capacity`; every kernel also returns a fresh `Vec<Value>` plus one `Rc` per result (`exec.rs`).
- **A large allocation is the floor, not the kernel:** a fresh 64 MiB `Vec` costs ~1.8 ns/element (mmap + page faults) against 0.15 ns/element for the fold itself, and a `Vec::with_capacity` + `push` of a 16M-element result costs ~30 ms, so no hot path may build a large result by pushing — the measured decomposition (27.7 → 6.19 ms for a full 16M f32 `reduce_sum`) is in "Large reductions" above.
- **A caller's input copy is inherent to `run_program`:** it takes owned `Buffer`s, so `tests/bench_large.rs` rows and any caller that keeps its host array charges one 16M f32 clone — 2.78 ms pooled, 26.5 ms with a cold allocator — on top of the kernel, which is why the post-change `run` row (6.19 ms) is 2.78 + 2.52 + ~0.9 ms of run overhead and not 2.5 ms.
- **Broadcast layouts the specializations decline keep the `Zip`:** a middle-axis broadcast, two chunky stretches that cannot share one run length, a result under `RUN_MIN_ELEMS` (64) and any operand whose slice length disagrees with its shape run the ndarray `Zip` (or the `walk` fallback); a run length other than 2, 4 or 8 keeps the runtime-bounded body, ~2× the fixed-size one per element.
- **`random_normal` is the random family's slow path:** it materializes the salted word stream, each parameter operand through `broadcast_to` + `Vec<ScalarValue>` and a final `Vec<ScalarValue>` cast, and its Box–Muller tail is a scalar `ln`/`sqrt`/`cos` loop (~14.5 ns/element against numpy's 7.8). `random_uniform` and `random_randint` take the typed contiguous path in `kernels/random/sampled.rs` and are 2–3× faster than numpy.
- **`dot` still copies an operand when the compute dtype differs:** `dot_impl` casts only a differing operand, so a same-dtype matmul is copy-free on the input side; `conv.rs` and `solve.rs` keep the older unguarded `cast` call and still memcpy both inputs.
- **Coverage of the slow paths:** `../tools/bench_cpu.py`'s eight workloads cover flat elementwise/transcendental chains, `dot_sum_256`, GP gather/sort, softmax and the `tn_step_pop500` approximation; the `bench_*` instruments cover the gather/scatter/slice/concatenate/broadcast/reduce/factorization families in isolation. What remains unbenched in isolation is `random_normal`'s word-stream tail and the broadcast `Zip`'s fully-general case.

## Routing Table

| Path | Area |
|---|---|
| `./Cargo.toml` | Crate manifest: `evoxc-core` + `ndarray` (default features only) / `faer` (`std` + `rayon`) / `half` / `num-complex` / `libm` / `serde_json`. |
| `./src/lib.rs` | Module wiring, `run_program` / `evaluate`, the callback-requirement pre-flight. |
| `./src/plan.rs` | `compile(&Program) -> Plan`: `OpSpec`, `Step` (kernel, operand/result ids, liveness free list, per-result static-`(dtype, shape)` marker, plan-time decoded `constant` payload, in-place slot), `InPlace`, `BlockPlan` (held as `Arc`), `FunctionPlan`, `Plan::text()`, unsupported-op rejection. |
| `./src/exec.rs` | `RunCtx` (flat store, last-use release, region execution, dim binding, result validation, the `try_in_place` accumulate path), `Callbacks`, `RunOptions`, `Value`. |
| `./src/buffer/` | The typed value representation and its ndarray layout substrate (submodule map below). |
| `./src/scalar.rs` | `Scalar` / `Num` / `ScalarValue` and the per-dtype numpy-semantics impls. |
| `./src/dims.rs` | Symbolic-dim binding and `DimExpr` evaluation (the `shapes.py` contract). |
| `./src/kernels/mod.rs` | `DENSE_OPS`, the kernel registry, `unsupported_ops()`, the registry-inventory tests. |
| `./src/kernels/{elementwise,indexing,structure,sorting,reductions,linalg,linalg_aux,random}/` | Per-category kernels as module directories: each `mod.rs` holds `pub fn register`, the kernel-level dispatch, the re-exports other files reach and its inline tests, with one submodule per op family or shared mechanism (submodule map below). |
| `./src/kernels/pow.rs` | `power`'s scalar-exponent path (`PowScalar`/`pow_scalar`: numpy's exact 0/1/2/3/0.5/−1 special cases, one `powf`/`pow` call for every other exponent), the negative-integer-exponent guard and its unit tests; private to `kernels`. |
| `./src/kernels/util.rs` | Axis normalization, dtype/shape helpers, shared result-shape lookup (genuinely shared vocabulary only). |
| `./tests/nullable_attrs.rs` | JSON-`null` attribute end-to-end coverage (envelope decode → compile → run), 13 tests. |
| `./tests/perf_probe.rs` | The `#[ignore]`d per-op cost probe for `slice`/`reshape`/`pad`. |
| `../tools/parity_cpu.py` | Python parity harness against the numpy backend. |
| `../tools/bench_cpu.py` | Native-vs-numpy benchmark, the source of the eight-workload table. |

### Module layout (one line per submodule)

Each `mod.rs` keeps `pub fn register` plus the re-exports the rest of the crate reaches, so no `use` path outside a directory changed when it was split.

- `buffer/mod.rs` — `Buffer` + inherent impl, `shape_text`, the three dispatch macros (`with_type!`/`with_same_type!`/`with_same_type_mut!`), the module wiring and every `crate::buffer::…` re-export.
- `buffer/storage.rs` — `Storage` over the 14 dtypes, little-endian payload decode, `cast_storage`/`convert_generic`, and the hand-written pooled `Clone` (a copied storage takes `take_pooled_vec` and splits the copy over `split_parallel` above `PARALLEL_MIN_ELEMS`, which is what makes a large input `Buffer::clone` cost 0.17 ns/element instead of a fresh-allocation 1.8).
- `buffer/ndarray_views.rs` — `NdScalar` and the zero-copy `Buffer`↔ndarray seam (`as_array`/`as_array_mut`/`as_array2`/`from_array`/`into_array`, `array_view`/`array_view_mut`).
- `buffer/broadcast.rs` — `broadcast_shape`/`broadcast_strides`, the ndarray `Zip` reference drivers (`walk`, `apply1..3`, `broadcast_to`, `broadcast_vec`, `fill_broadcast`) and the repeating-run driver (`runs2`/`runs3`/`copy_runs` with `repeat_view`, `run_block`, `run_source`, `walk_blocks`, `run_chunks`, `RUN_TILE_MAX`/`RUN_BLOCK_TARGET`/`RUN_MIN_ELEMS`/`RUN_SPLIT_MIN_ELEMS`, the `#[cfg(test)]` `RUNS_DISABLED` A/B switch), plus the `patterns` re-export (`periodic2`/`periodic3`).
- `buffer/broadcast/patterns.rs` — the periodic-pattern driver: `Pattern`/`pattern_view` (the `values[(i / stretch) % span]` proof), `block_run`, `Feed`/`Source`, `drive2`/`drive3`, the const-generic run bodies `fixed2`/`fixed3` (widths 2/4/8) and the runtime-bounded `runs2_body`/`runs3_body`, and `copy_pattern` behind `fill_broadcast`.
- `buffer/broadcast/tests.rs` — the pins for every driver in `broadcast.rs` and `patterns.rs` against the independent naive reference and the `Zip`, serialized on one `switch_guard()` mutex because the switches are process-wide.
- `buffer/zipmap.rs` — the flat `zip1..3`/`zip2_accumulate`/`map1..3` fast paths and the pooled broadcast builders `map2_runs`/`map3_runs` (repeating-run driver, then periodic driver, then row split).
- `buffer/rowsplit.rs` — `RowSplit`/`Rider`/`split_rows`/`chunk_shape`: the row-aligned outermost-axis split the pooled broadcast builders use, plus the `#[cfg(test)]` `SPLIT_DISABLED` switch.
- `buffer/parallel.rs` — `PARALLEL_*`, `parallel_threads`, `split_parallel` (both `pub(crate)`, so the reduction fold and `Storage::clone` reuse the same policy), `window` and their crossover pins.
- `buffer/pool.rs` — `POOL_*`, `StoragePool`, `take_pooled_vec`, the `Drop for Storage` admission hook, and the retention/eviction pins.
- `buffer/blocks.rs` — `copy_blocks`, `block_storage`, `block_storage_filled`, `repeat_storage`.
- `buffer/bench_broadcast.rs` — the `#[ignore]`d `bench_broadcast_zip`/`bench_broadcast_zip_threads`/`bench_walk` instruments, serialized on one `bench_guard()` mutex.
- `buffer/test_support.rs` — the shared `#[cfg(test)]` garbage-stock fixture.
- `kernels/elementwise/mod.rs` — registry, the `stop_gradient`/`cast`/`broadcast` kernels, `bool_results`, the `in_place_kernel` re-export.
- `kernels/elementwise/binary.rs` — `BinKind`, `Layout`, the dense binary drivers (arithmetic/bitwise/shift), the `map2_runs` broadcast arm, the in-place `accumulate_as` path.
- `kernels/elementwise/compare.rs` — the comparisons and `select` (the `map3_runs` broadcast arm).
- `kernels/elementwise/unary.rs` — `UnKind` and the unary drivers.
- `kernels/elementwise/transcendental.rs` — `erf`, `gelu`, `nan_to_num` and their fixups.
- `kernels/indexing/mod.rs` — registry, the shared index vocabulary (`resolve_index_fast`, `reconcile`, block builders) and the two pooled destinations `pooled_total`/`pooled_append`.
- `kernels/indexing/reshape.rs` — `reshape`, `transpose` (ndarray view + run copier).
- `kernels/indexing/slice.rs` — `slice` (ndarray `slice_axis_inplace`, the contiguous `copy_from_slice` arms and the run copier).
- `kernels/indexing/strided_copy.rs` — `CopyPlan`/`AxisCopy`/`RunCopy`, `plan_axis_copy`, `plan_axis_runs`, `plan_steps`, `append_runs`, `place_runs`, `copy_view`, `fill_run`, `fill_strided` (`LANES`/`RUN_MEMCPY_MIN`/`MAX_STACK_AXES`), `copy_elements`; the two-sided run copier shared by `slice`/`reshape`/`gather`.
- `kernels/indexing/strided_copy_tests.rs` — the run copier's `#[path]` child test module (against ndarray `assign`, the append/place families, declines, float bits, index errors).
- `kernels/indexing/gather.rs` — `gather` (`gather_runs` inline-index fast arm, `gather_typed` plan replay, `resolve_indices`, `pooled_append`/`pooled_total` destinations).
- `kernels/indexing/scatter.rs` — `scatter` (typed indices, `scatter_selection`, the run and general drivers).
- `kernels/indexing/concatenate.rs` — `concatenate`.
- `kernels/indexing/bench_gather.rs` — the `#[ignore]`d `bench_gather`/`bench_scatter` instruments.
- `kernels/indexing/bench_slice.rs` — the `#[ignore]`d `bench_slice` instrument.
- `kernels/indexing/bench_concatenate.rs` — the `#[ignore]`d `bench_concatenate` instrument.
- `kernels/indexing/test_support.rs` — the shared `#[cfg(test)]` builders.
- `kernels/structure/mod.rs` — registry, dispatch, `identity_shape`/`same_shape`, attribute helpers.
- `kernels/structure/tile.rs` — `tile` (interleaved broadcast view, whole-source `repeat_storage`, block and general paths).
- `kernels/structure/flip_roll.rs` — `flip` (`slice_each_axis`) and `roll`.
- `kernels/structure/pad.rs` — `pad` (config validation against the declared shape, the pad value, the row-wise pooled copy).
- `kernels/structure/diag.rs` — `diag` (`diag`/`diag_mut` views).
- `kernels/structure/test_support.rs` — the shared `#[cfg(test)]` harness.
- `kernels/sorting/mod.rs` — registry, sort/argsort dispatch and axis handling, the `bench_segment_walk` instrument.
- `kernels/sorting/keys.rs` — the dtype → order-preserving `u64` row-key table and `STACK_KEYS`.
- `kernels/sorting/network.rs` — the Batcher odd-even networks, `order_keys`, `NETWORK_MAX`/`PAD_KEY`.
- `kernels/sorting/permutation.rs` — the hand-written per-segment walk, `sorted_positions`/`segment_positions`, the permuted-storage builders.
- `kernels/sorting/test_support.rs` — the shared `#[cfg(test)]` reference oracle.
- `kernels/reductions/mod.rs` — registry, `Kind`, the reduce/mean driver, `divide_by`, the flat-fold-vs-lane-walk dispatch.
- `kernels/reductions/flat.rs` — `fold_block`/`fold_block_with`, the contiguous-block specialization (pooled result, `REDUCE_MIN_PER_THREAD`/`fold_threads` output-only thread split, `fold_outputs` for the `inner == 1` run and the `inner > 1` lane-window walk), and the `bench_reduce_flat_block_vs_lane_walk`/`bench_reduce_large_fold` instruments.
- `kernels/reductions/bench_reduce_large.rs` — the `#[ignore]`d whole-graph stage decomposition (`bench_reduce_large`: clone / `with_capacity`+push result / `fold_block` at 1 and policy threads / kernel / `run_program` over the 16M f32/f64 families) and `bench_reduce_large_threads` (the 1/2/4/8 sweep), plus the bit-identity and fixture pins.
- `kernels/reductions/layout.rs` — the ndarray lane walk (`for_each_lane`/`fold_all`/`fold_first`, `axis_permutation`, `Block`, `FoldSelect`) and the 8-lane accumulator split (`fold_run`/`fold_lanes`).
- `kernels/reductions/argmax.rs` — `argmax`/`argmin` and their scan selection.
- `kernels/reductions/cumulative.rs` — `cumsum`/`cumprod`.
- `kernels/linalg/mod.rs` — registry, `mod` wiring, the shared linalg test fixtures.
- `kernels/linalg/common.rs` — arity/attribute/result-shape helpers.
- `kernels/linalg/matmul.rs` — `dot`, the ndarray broadcast/`indices` batch traversal, the `FaerGemm` trait and the `faer` BLAS-3 fast path with its `gemm_threads`/`Par` selection and the `bench_dot_gemm` instrument (the one file over 1000 lines, by design).
- `kernels/linalg/conv.rs` — `conv` (padding, dilations, group counts).
- `kernels/linalg/solve.rs` — `solve` (LU in f64 work space).
- `kernels/linalg/structural.rs` — `diagonal`, `tril`/`triu` (ndarray `permuted_axes`/`slice_axis` views).
- `kernels/linalg_aux/mod.rs` — registry, the six factorization kernels' dispatch and the `faer`-evaluation verdict table.
- `kernels/linalg_aux/common.rs` — `prepare`, result materialization, rank/tolerance helpers.
- `kernels/linalg_aux/eigh.rs` — cyclic-Jacobi `eigh` and its faer A/B evidence/timing benches.
- `kernels/linalg_aux/cholesky.rs` — `cholesky` and its faer A/B evidence/bench.
- `kernels/linalg_aux/qr.rs` — Householder `qr` and its faer A/B evidence/bench.
- `kernels/linalg_aux/svd.rs` — one-sided-Jacobi `svd` and its faer A/B evidence/bench.
- `kernels/linalg_aux/matrix_exp.rs` — `matrix_exp` (Taylor terms) and its helpers.
- `kernels/linalg_aux/test_support.rs` — the shared `#[cfg(test)]` numeric assertions.
- `kernels/random/mod.rs` — registry plus the op-level dispatch/attribute/result helpers.
- `kernels/random/bits.rs` — the splitmix64/threefry2x32/philox4x32_10 word streams and their constants.
- `kernels/random/key.rs` — `Algorithm`/`Key`/`Words`, key extraction and child derivation.
- `kernels/random/sampling.rs` — `random_key_mix`, `random_uniform`, `random_normal`, `random_randint`.
- `kernels/random/sampled.rs` — the typed contiguous fast path for `random_uniform`/`random_normal`/`random_randint` (bit-exact word visitors).
- `kernels/random/permutation.rs` — `random_permutation`, `random_multinomial`.
- `kernels/random/test_util.rs` — the shared `#[cfg(test)]` harness.
- `kernels/{control,collective,util,pow}.rs` — single-file modules.
