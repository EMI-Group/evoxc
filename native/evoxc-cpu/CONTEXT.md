# evoxc-cpu — the native CPU execution engine of EvoX Compiler

## Intent

`evoxc-cpu` is the native (Rust) CPU backend: it compiles an [`evoxc_core::Program`] into an immutable execution `Plan` and runs it on typed host buffers.
It is the numerical authority for concrete CPU execution — the Rust counterpart of the Python numpy interpreter in `../../evoxc/backends/numpy/` — and it depends on `evoxc-core` only: no Python, no PyO3, no device APIs, no compiler backends.
The `evoxc-python` crate wraps it as `evoxc._evoxc_native.native_compile` / `NativeProgram`; that FFI is documented in `../evoxc-python/CONTEXT.md`.

Shape/type inference is NOT re-derived here: result types come from the graph (`OpSpec.result_types`, filled by `evoxc-core` inference at trace/build time) and every produced buffer is re-validated against them at run time.
The engine never falls back to Python and never guesses: an op without a kernel is an explicit `BackendError` at plan-build time naming the op (and its source location when the graph carries one).

```text
Program --compile--> Plan --RunCtx::new--> RunCtx --run--> Vec<Buffer>
```

## API surface

- `plan::compile(&Program) -> Result<Plan>` — one immutable `Plan` per program: per-function entry `FunctionPlan`, per-block `BlockPlan`, per-op `Step` (`OpSpec` + resolved kernel + operand/result value ids + liveness free list + per-result static-`(dtype, shape)` marker + a decoded constant payload + an in-place slot). Compiles every reachable region and every `call` callee (BFS), collects the host-callback requirements (`required_blocks` / `required_external` / `required_runtime`) and rejects an unsupported op up front. `Plan::entry_function()`, `Plan::needs_callbacks()`, `Plan::text()` (the human-readable dispatch plan behind `NativeProgram.plan_text()`).
- `exec::RunCtx` — the interpreter: `RunCtx::new(program, Arc<Plan>, RunOptions, Option<&dyn Callbacks>)`, `run(flat_inputs) -> Result<Vec<Buffer>>`, `run_block(block_index, args)`, `run_region(region, args)`, `eval_shape`/`eval_shape_strict` (symbolic-dim binding + `DimExpr` evaluation), `result_shape`/`result_shape_strict`/`result_type`, `validate_result`. Values live in ONE flat `value_id`-indexed store per block execution; a value is dropped at its last use (nested-region uses are attributed to the enclosing op, so a control-flow parent's operand survives its region).
- `exec::Callbacks` — the host-callback trait the PyO3 layer implements: `call_block(name, args, static_args)`, `call_external(name, args)`, `call_runtime(callback, args)`, each returning `Result<Vec<Buffer>>`.
- `exec::RunOptions { rank, world_size }` — the rank context of the run (defaults `0` / `1`).
- `buffer::{Buffer, Storage}` — the typed value: `Storage` covers `bool`, `i8/i16/i32/i64`, `u8/u16/u32/u64`, `f16`, `f32`, `f64`, `c64`, `c128` (all 14 evoxc dtypes); `Buffer { dtype, shape, storage }` with `new` (validates dtype + element count), `from_payload` (graph constants), `zeros`, `cast`, `with_shape`, broadcasting helpers (`broadcast_shape`/`broadcast_strides`/`broadcast_to`), and the generic `walk`/`apply1`/`apply2`/`apply3` drivers the kernels are built on.
- The same module owns the performance substrate: the thread-local recycled-storage pool (`take_pooled_vec`, `POOL_MIN_BYTES`/`POOL_MAX_BYTES`/`POOL_MAX_ENTRIES`/`POOL_MAX_PER_SIZE`), the flat builders (`zip1..3`, `zip2_accumulate`, `map1..3`), the scoped split (`split_parallel`, `PARALLEL_MIN_ELEMS`/`PARALLEL_MIN_PER_THREAD`/`PARALLEL_MAX_THREADS`) and the block/repeat storage builders (`copy_blocks`, `block_storage_filled`, `repeat_storage`).
- `scalar::{Scalar, Num, ScalarValue}` — the numeric vocabulary kernels are generic over (`add/sub/mul/div/rem/pow/neg/abs/sqrt/exp/ln/log1p/sin/cos/tan/tanh/sign/floor/ceil/round_even/max/min/bit_*/shl/shr`), with numpy semantics baked into the impls (integer `rem` truncates toward zero, unsigned wraparound, `round` half-to-even, NaN-propagating `max`/`min`).
- `dims::{Bindings, eval_dim, eval_extent, eval_shape, eval_shape_strict, bind_dim, bind_shape}` — the `shapes.py` contract: bind named dims against concrete extents (conflict → `ShapeError`), bind `DimExpr` leaves then evaluate, a free dim → `ShapeError`, division/modulo by zero → `ShapeError`, negative extent → `ShapeError`.
- `run_program(program, &Arc<Plan>, inputs, options, callbacks) -> Result<Vec<Buffer>>` — one call; a plan that needs callbacks and gets `callbacks: None` is an explicit `BackendError` naming the missing entry points, before any op runs. `evaluate(program, inputs, options, callbacks)` — compile + run (tests, harnesses).
- `kernels::{lookup, is_supported, unsupported_ops, DENSE_OPS}` — the machine-readable support inventory: `unsupported_ops()` is the sorted list of dense ops without a native kernel (currently EMPTY), and two tests pin it (`every_dense_op_has_a_kernel`, `table_has_no_foreign_entries`).

## Support matrix

Every dense IR op has a native kernel (94 of the 119 op defs; the rest are `return`/`sparse_*`):

| Category | Ops |
|---|---|
| control | `constant`, `if`, `while`, `call`, `block_call`, `external_call`, `runtime_call`, `stop_gradient` |
| elementwise | `add`, `subtract`, `multiply`, `divide`, `remainder`, `power`, `negate`, `abs`, `square`, `sqrt`, `exp`, `log`, `log1p`, `sin`, `cos`, `tan`, `tanh`, `acos`, `sign`, `floor`, `ceil`, `round`, `relu`, `sigmoid`, `erf`, `gelu`, `nan_to_num`, `maximum`, `minimum`, `bitwise_and/or/xor`, `bitwise_left_shift`, `bitwise_right_shift`, `logical_and/or/not`, `cast`, `select`, `broadcast` |
| comparison | `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal` |
| reduction | `reduce_sum`, `reduce_prod`, `reduce_max`, `reduce_min`, `reduce_mean`, `argmax`, `argmin`, `cumsum`, `cumprod` |
| sorting | `sort`, `argsort` |
| indexing/structure | `reshape`, `transpose`, `slice`, `gather`, `scatter`, `concatenate`, `pad`, `tile`, `flip`, `roll`, `diag`, `diagonal`, `tril`, `triu` |
| linalg | `dot` (incl. batched), `conv`, `solve`, `cholesky`, `eigh`, `qr`, `svd`, `matrix_rank`, `matrix_exp` |
| collective | `rank`, `world_size`, `all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`, `broadcast_collective`, `collective_permute` |
| random | `random_key_mix`, `random_uniform`, `random_normal`, `random_randint`, `random_permutation`, `random_multinomial` |

`dot` is numpy's `matmul` contraction: a rank-1 operand is promoted to a 1-row (resp. 1-column) matrix and the batch dims broadcast.
Its plain matrix-matrix form in f32/f64 — both operands rank >= 2, C-contiguous batch blocks — runs through the `matrixmultiply` BLAS-3 kernel, one `sgemm`/`dgemm` call (alpha 1, beta 0) per row strip of each batch element; every other dtype or shape keeps the general triple loop, so integer and complex results stay bit-exact and only the two IEEE float dtypes see a different (association-order) summation.
Semantics are the numpy backend's, op by op: `remainder` is floormod (sign of the divisor), `round` is half-to-even, `maximum`/`minimum` propagate NaN and reject complex, comparisons produce bool and complex supports only `equal`/`not_equal`, `abs` on complex is the real magnitude, integer `divide`/`remainder` by zero and a negative integer exponent are explicit errors, integer arithmetic wraps like numpy.
`if` takes a 0-d bool predicate in operand 0 and passes ALL operands as region arguments; `while` has regions `[cond, body]` with no iteration limit; `call` runs the callee's entry block.
Result validation matches the numpy interpreter's messages ("kernel for op '{op}' produced result {i} with …"), a dtype mismatch is a `BackendError` and a shape mismatch a `ShapeError`.
Collectives are single-rank identities (`rank`/`world_size` read `RunOptions`); `world_size != 1` is an explicit `BackendError` — there is no multi-rank executor in this engine.
Random is BIT-EXACT with `../../evoxc/backends/numpy/kernels/random.py` for splitmix64, threefry2x32 and philox4x32_10 (key handling, salt folding, counter layout, uniform/randint/permutation/multinomial/normal, including the f32 Box–Muller normal fast path).

## Performance (measured on this host, shared CPU)

Intel Core Ultra 7 356H, 16 cpus, Python 3.14.10 / numpy 2.5.2–2.5.5, loadavg 0.7–1.2 at the recorded runs (2026-11-22).
Warm `run` = median of 15–25 iterations after 3–5 warmups; the numpy leg is `evoxc.build(fn, *specs, backend="numpy", device="cpu")` + `evoxc.run` (`evoxc.evaluate` re-traces per call).
Reproduce with `.venv/bin/python native/tools/bench_cpu.py` (records `model name` and loadavg itself).
Every row of every recorded run reports `match yes`: the native and numpy results agree numerically on that workload.
Eight workloads, `--iterations 15 --warmup 3` in one process:

| workload | ops | native compile (ms) | native run (ms) | numpy run (ms) | speedup |
|---|---|---|---|---|---|
| `ew_chain_1e6` (8 elementwise ops, 1e6 f32) | 14 | 0.064 | 1.584 | 2.763 | 1.74× |
| `ew_transcendental_1e6` (`erf`/`gelu`/`tanh` chain) | 7 | 0.036 | 17.675 | 160.351 | 9.07× |
| `dot_sum_256` (256² f32 `dot` + row sum) | 3 | 0.023 | 0.380 | 0.117 | 0.31× (isolated 4.8×) |
| `gp_gather_4096` (GP-shaped gather/select chain, 4096×64) | 80 | 0.290 | 1.512 | 0.719 | 0.48× (isolated 1.39–1.59×) |
| `gp_sort_4096` (row argsort + sort + two-axis gather + broadcast select + reductions) | 14 | 0.072 | 3.827 | 2.049 | 0.54× |
| `tf_softmax_8x128x384` (per-row softmax + tanh/gelu mix) | 15 | 0.073 | 3.258 | 31.232 | 9.59× |
| `chain_2000` (2000-op elementwise chain) | 2000 | 6.281 | 0.114 | 1.280 | 11.25× |
| `tn_step_pop500` (2249 static ops, shape-faithful TensorNEAT `Pipeline.step`) | 2249 | 10.4 | 67.190 | 56.325 | 0.84× |

`native_compile_ms` is one full `native_compile(envelope)` call (envelope decode + FFI + engine plan compile); the engine's own plan compile is printed per workload by the harness (0.012–0.044 ms on every synthetic row, 0.86 ms on `chain_2000` and ~8 ms on `tn_step_pop500`, whose 2249 ops are mostly region-bearing).
`tn_step_pop500` is a synthetic approximation of the real downstream graph, not the acceptance measurement: it under-weights exactly what the engine wins on (the many-small-op tail and the `while`/`if` region machinery) and over-weights large per-call shapes, so its native/numpy ratio (0.84×) is worse than the real graph's — the next section is the authoritative number.
Two rows are process-state artifacts rather than graph costs:
`gp_gather_4096` alone in a fresh process (`--filter gp_gather --iterations 25`) is 0.51–0.54 ms against numpy's 0.71–0.86 ms (1.39–1.59×), and only inside the eight-row sequence does the preceding 1e6-element work leave the thread and core state in a shape that costs it ~1.5 ms; it is bimodal on BOTH legs there (the pre-round-2 build measured 0.80–2.77 ms in consecutive rounds).
`dot_sum_256`'s numpy leg is the unstable one (threaded OpenBLAS: 0.117 ms in sequence, 1.8 ms isolated in the same session), so only its isolated ratio (native 0.377 ms, 4.8× faster) means anything.
`gp_sort_4096` is the honest loss: 0.54–0.60×, because `sort`/`argsort` are scalar Batcher networks against numpy's AVX2/AVX-512 sorting networks.

Reading of the numbers, which is load- and shape-dependent:
- The contiguous elementwise fast paths beat numpy on a 1e6-element mixed chain (1.6–1.9×) because per-op overhead is lower than numpy's ufunc dispatch, even though numpy fuses a whole expression into fewer passes.
- The transcendental chain wins by 9× because `erf`/`gelu`/`tanh` go through scalar `libm` calls while numpy composes them from several ufuncs, and `exp` runs libm `expf` too (no polynomial arm), which is what keeps the softmax-shaped row at 9.6×.
- `dot` at 256² runs the `matrixmultiply` BLAS-3 kernel at ~58 GFLOP/s single-threaded but loses to numpy's threaded OpenBLAS on a matrix this small, because the engine's scoped threading only engages above its work thresholds.
- numpy still wins where it has SIMD kernels safe Rust cannot express (`sort`/`argsort`, `astype`, `expf`) and on threaded BLAS; native wins where per-op dispatch dominates (`chain_2000` at ~0.06 µs/op), on transcendental compositions, and on every real many-small-op graph whose op mix includes control flow.
- Compiled plans are reusable across runs (`Arc<Plan>`), so a caller that keeps the plan pays only `run`.

### Real downstream workload: TensorNEAT 0.2.0 `Pipeline.step`

The authoritative workload is the external TensorNEAT 0.2.0 harness: one `evoxc.build` + one warm `evoxc.run` per generation over a 500-genome population.
Reproduce with `PYTHONPATH=<tensorneat>/src .venv/bin/python <tensorneat>/benchmarks/run_benchmark.py --impl new --backend numpy,native --cases xor_p500,cartpole_p100 --generations 8 --warmup 2 --repeats 1`.

| case | numpy ms/gen | native ms/gen at `e07bac8` (pre-migration) | native ms/gen at HEAD | verdict |
|---|---|---|---|---|
| `xor_p500` (2249 static ops, ~7900 executed) | 48.8 | 43.1 | **61.3** | +17 ms (+40 %) over the pre-migration engine — the residual `gather` copier replay (see Known Issues) |
| `cartpole_p100` (2238 static ops, ~31.6 k executed) | 69.2 | 27.4 | **46.6** | +19 ms (+70 %) over the pre-migration engine — the residual `gather` copier replay (see Known Issues) |

The HEAD/base columns are the same worker (seed 42, 8 generations, 2 warmup, one repeat) loading two scratch instrumented builds, so they are directly comparable; the pre-migration `e07bac8` engine is the accepted native baseline.
The per-op breakdown below is that pre-migration reference engine measured against numpy — its gather figures (0.4–1.0 ns/element) are the baseline HEAD is now compared against, not the current engine.

Per-op attribution on the run-copy gather path (nested-aware `Instant` brackets around the `exec.rs` step-kernel call in a scratch build loaded through `EVOXC_NATIVE_LIB`, ~0.2 % overhead, median of 3 runs; the numpy leg is the numpy interpreter's own kernels wrapped the same way), kernel totals: `xor_p500` native **42.65** vs numpy **50.54 ms/run**, `cartpole_p100` native **32.80** vs numpy **82.86 ms/run**.
The xor split is **+15.04 ms over the 146 ops where native loses** and **−22.90 ms over the 411 ops where native wins** (net −7.86); the largest single win is the many-small-op tail — 505 ops at ~2.3 µs/call against numpy's ~7 µs — i.e. interpreter and plan overhead, which is what the block-plan work targets.

| xor op | calls/run | native ms | numpy ms | delta |
|---|---:|---:|---:|---:|
| `gather f32 (500,100,2)` | 40 | 3.748 | 1.206 | +2.542 |
| `slice f32 (500,50,1)` | 364 | 2.471 | 0.394 | +2.077 |
| `while` (5 region ops) | 5 | 2.435 | 5.869 | −3.434 |
| `gather f32 (500,50,5)` | 40 | 1.891 | 1.664 | +0.227 |
| `cast int32 (500,100)` | 86 | 1.650 | 0.923 | +0.727 |
| `slice f32 (500,50,4)` | 40 | 1.532 | 0.047 | +1.485 |
| `scatter f32 (50500)` | 40 | 1.488 | 2.968 | −1.480 |
| `multiply f32 (2000,50,4)` (broadcast) | 4.2 | 0.927 | 0.491 | +0.436 |
| `slice f32 (500,100,1)` | 80 | 0.848 | 0.095 | +0.753 |
| `random_normal f32 (500,100,1)` | 1 | 0.725 | 0.390 | +0.335 |
| `exp f32 (2000,50)` | 4.2 | 0.687 | 0.245 | +0.442 |
| `argsort int64 (500,200)` | 1 | 0.660 | 1.260 | −0.600 |
| `reshape f32 (500,100)` | 82 | 0.386 | 0.060 | +0.326 |
| `select f32 (500,50,4)` | 10 | 0.348 | 1.123 | −0.775 |
| `not_equal bool (500,50,1)` | 80 | 0.254 | 1.192 | −0.938 |
| 505 smaller ops (< 0.05 ms each) | | 4.386 | 9.999 | −5.613 |

`cartpole_p100` splits **+3.00 ms over 51 losing ops** and **−52.82 ms over 463 winning ops**; its single largest loss is `exp f32 (100,50)` (+2.05 ms, 68 % of all its losses) and its largest wins are the `while` regions (−19.2 ms), `select` (−6.8 ms), the elementwise family (−12.6 ms) and the small-op tail (−6.1 ms).

The residual, with its reason — these are floors, not tuning gaps:
- **Views vs copies: `slice` +4.83 ms and `reshape`/identity-`transpose`/`concatenate`/`broadcast` ≈ +1.3 ms on xor.** `Storage` owns plain `Vec`s, so the engine cannot return a strided view the way numpy does (`np.take`'s consumer pays ~0.3 ns/element for a strided read; the engine pays a copy at ~0.2–0.38 ns/element). Every strided slice runs the `CopyPlan`/`stride_copy` path, but that path is no longer uniformly at parity: `slice f32 (500,50,4)` (40 calls/run, xor) is 38.3 → 66.6 µs (1.74×, +1.1 ms/run) and the `bool (100,50,1)`/`f32 (100,50,1)` families lose ~15 % at HEAD vs `e07bac8`, so `slice`'s `dst == 1`/`src == 1` collapsed arms share the copier's stride-0 slow path; the remaining `slice` gap is the copy-vs-view floor, so a view/Tensor-owning-a-layout refactor is the only way past it.
- **numpy SIMD: `cast` +1.06 ms, `exp` +0.44 ms, `sort`/`argsort` +0.29/−0.65 ms.** `astype` runs at 0.21 ns/element against the engine's scalar `convert_generic` 0.38, numpy's AVX-512 `expf` is 0.58 ns/element against libm's 1.0, and the sorting networks are vectorized where safe Rust cannot reach.
- **`gather` +2.54 ms on the interleaved `(500,100,2)` family:** both sides copy (no view exists), the engine already resolves the index operand once per call and copies innermost runs, but numpy's C `take` loop is 0.30 ns/element against the engine's 0.94.
- **Broadcast elementwise +0.44 ms:** the `buffer/stride` run driver keeps a broadcast `multiply (2000,50,4)` at 0.55 ns/element against numpy's 0.29.
- **`random_normal` +0.64 ms/run across its three call shapes:** the Box–Muller tail still materializes the salted word stream, the broadcast parameter chain and a `Vec<ScalarValue>` pass per call (~14.5 ns/element against numpy's 7.8); `random_uniform`/`random_randint` take the typed contiguous path and are already 2–3× faster than numpy.

## Optimizations

- **Recycled storage pool (`buffer/pool.rs`):** a thread-local free list of large result storages — `impl Drop for Storage` hands an allocation of at least `POOL_MIN_BYTES` (4 KiB) back to the pool instead of to `malloc`, and `take_pooled_vec::<R>(count)` hands a stock of exactly the requested `(dtype, element count)` out again, which removes the per-op mmap/page-fault cost on many-small-op graphs.
- Retention is bounded per thread: at most `POOL_MAX_ENTRIES` (64) values totalling `POOL_MAX_BYTES` (64 MiB), and at most `POOL_MAX_PER_SIZE` (8) stocks of any one exact size class.
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
- Broadcasting, non-contiguous layouts and the remaining dtypes keep the general `walk`/`apply1..3` driver with identical semantics (NaN propagation, integer wraparound and complex arithmetic unchanged).
- **Wide maps split over scoped threads (`buffer/parallel.rs::split_parallel`):** one `std::thread::scope` per wide kernel call over disjoint contiguous output ranges, gated at `PARALLEL_MIN_ELEMS` (2^18 elements) with one thread per `PARALLEL_MIN_PER_THREAD` (2^17) elements and capped at `PARALLEL_MAX_THREADS` (4).
- Each thread reads its own operand window at its own running offset, so results are identical to the single-threaded loop; there is no rayon and no global pool.
- **Reductions:** a contiguous-block fast path (`Block`) where the reduced axes form one block, folding `outer × red × inner` groups with no per-element offset table, and an 8-lane (`LANE_COUNT`) unrolled `fold_run` whose independent accumulators break the sequential dependency chain (f16 and complex keep the single-lane sequential order their tolerance requires).
- Max/min folds go through the NaN-propagating `Scalar::max`/`min`, seeded from each run's first element (`fold_groups` with no identity), and the arg reductions pick their scan by run length: `arg_run_branchless` below `ARG_BRANCHY_MIN_RUN` (1024) elements and the plain `arg_run_branchy` scan at or above it.
- **Sorting:** a packed monotone row key `(rank_key << 32) | slot` with `u32` positions, so one `u64` comparison orders both the value and its index for `sort` and `argsort`.
- Extents of 2..=`NETWORK_MAX` (16) run branchless Batcher odd-even networks of their own pad width (2/4/8/16 keys, `PAD_KEY = u64::MAX` sentinel) with every comparator index a literal; every extent outside that range keeps `sort_unstable`.
- **`dot` threading (`kernels/linalg/matmul.rs`):** the BLAS-3 call splits row strips of each batch block round-robin across scoped threads — so a batch of many blocks spreads over the batch axis too — above `GEMM_PARALLEL_MIN_BLOCK_WORK` / `GEMM_PARALLEL_MIN_WORK`, one thread per `GEMM_PARALLEL_WORK_PER_THREAD` multiply-adds, capped at `GEMM_PARALLEL_MAX_THREADS` (8) below `GEMM_WIDE_WORK` and unbounded above it, and never more threads than row strips.
- Every output element is still produced by exactly one `gemm_into` call with an untouched K axis, so the threaded result is bit-identical to the single-threaded one.
- **Structure/indexing:** a `tile` that repeats the whole source builds its storage with `repeat_storage` (one `copy_from_slice` per run-length chunk rather than one per index row, split over the same scoped threads), and contiguous block copies (`gather`/`slice`/`tile`) go through `copy_blocks` with the pre-filled `block_storage_filled` builder instead of a zero fill plus growth.
- `copy_blocks` walks its destination with `chunks_exact_mut(inner)`, so every block is one `copy_from_slice` at a running destination offset (no per-block multiply, no re-derived range) and a whole-array block — the identity `slice`/`reshape`/`transpose` and whole-source `tile` case — is ONE memcpy.
- **The stride copy (`kernels/indexing/stride_copy.rs::CopyPlan`):** `slice`'s non-negative-step path and `gather`'s block path build an innermost-first axis list (collapsed where strides line up) and materialize it into a POOLED destination of exactly the result size, which is fully overwritten; there is no `Vec` growth, no zero fill and no per-element odometer. A contiguous innermost run is one `copy_from_slice` per run ([`block_storage_filled`]); a strided one is [`fill_strided`], which emits `LANES = 4` elements per iteration out of one `stride * 4`-element source window (~0.2 ns/element measured against ~0.55 ns for the `step_by` walk it replaced), with an exact single-element tail and one up-front bounds validation of the whole run. An empty outer axis skips the offset table entirely — the identity/full-slice and whole-array `reshape`/`transpose` case is one bulk memcpy.
- **`pad` is a row-wise copy, not an index map (`kernels/structure/pad.rs`):** the pooled destination is fully written — pad rows get one `fill`, an interior row gets the leading pad cells `fill`ed, ONE `copy_from_slice` of its innermost run out of the contiguous operand (at a running C-order source offset) and its trailing pad cells `fill`ed. An all-zero padding config is one bulk `copy_from_slice` of the operand, a rank-0 operand copies its single element, and the config is validated against the declared result shape before any allocation (same messages as before). Measured: a 100 000-element i32 rank-1 pad with 2+2 zeros costs 7.25 µs of kernel against a 6.20 µs bare memcpy of the same size (was a 1.6 MB `Vec<Option<usize>>` index map plus a zero fill plus a per-element gather).
- **Outputs freed before extraction:** a value is released from the flat store at its last use and the run clears what is left of the store before pulling the output buffers out, so an output is moved out of its `Rc` instead of copied (a value returned twice still copies), and the intermediates released there hand their storage back to the pool.
- No SIMD intrinsics are used (they would need `unsafe`, which this crate forbids outside the two `matrixmultiply` call sites), so elementwise throughput rests on LLVM auto-vectorizing safe slice loops.
- **Per-run setup is small and allocation-bounded:** a run never recompiles or re-decodes the plan (`run_program` takes `&Arc<Plan>`), and only allocates one flat `store: vec![None; plan.max_value]`, one `Bindings` map and one reusable `operands` scratch `Vec` (`exec.rs` `RunCtx::new`); `RunCtx::run` additionally clones the entry `FunctionPlan` (name + type vectors) per call for its input/output validation, and the `call` kernel resolves a callee by linear name search per call op (`kernels/control.rs` `call_kernel`).

- **Typed index resolution, once per call (`kernels/indexing/{gather,scatter,mod}.rs`):** `gather`/`scatter` resolve their index operand into a typed `Vec<usize>` in ONE pass (`resolve_index_fast`) instead of widening every element to `i128` per use, and `scatter_selection` chooses between the single-block and the per-index path from the resolved indices, so the copy runs are slices of a pre-offset destination.
- **The broadcast run driver (`buffer/stride/`):** a run-oriented walk — the innermost axis is a contiguous run, outer axes carry strides — used by the `Layout::Broadcast` arms of `binary`, `compare` and `select`, so a bias `(n,)` against `(pop,m,n)`, a `(pop,1)` per-row scalar and a broadcast `select` predicate no longer fall to the rank-deep `for_each_index` odometer; `broadcast_to` materializes through `fill_broadcast` (one copy per contiguous run) instead of a per-element gather.
- **Pooled comparison/logical results and pooled casts:** the comparison family builds its bool result through `bool_results` (a pooled vector; `BOOL_RESULT_THREAD_MIN = 1 << 19` above which the result is built per scoped chunk) and `cast` converts whole storages through a pooled `convert_generic`, so neither pays a zero fill.
- **`reshape`/identity-`transpose`/`pad` take a pooled destination and copy it with `copy_from_slice`** (`kernels/indexing/reshape.rs`, `kernels/structure/pad.rs`), so the shape-only ops are one bulk memcpy with no per-element work; `constant` still clones its plan payload per use (see Known Issues).
- **`random_uniform`/`random_normal`/`random_randint` typed contiguous path (`kernels/random/sampled.rs`):** a contiguous same-shape draw runs a word visitor straight into the destination storage per dtype — no `Vec<ScalarValue>` chain, no `broadcast_to` of the bounds, no per-element `%`/`/` axis arithmetic — and stays bit-exact with the numpy backend for all three algorithms (the multi-pass reference path remains for broadcast draw shapes).
- **`dot` casts only a differing operand:** `dot_impl` compares each operand's dtype with the compute dtype before calling `Buffer::cast` (`kernels/linalg/matmul.rs`), so the common same-dtype matmul no longer memcpys both inputs (the older unguarded pattern remains in `conv.rs`/`solve.rs`).
- **Plan entries are reference counted, not cloned:** `Plan.blocks` holds `Arc<BlockPlan>` and `RunCtx::run` borrows the entry `FunctionPlan` signature, so a region execution and a run no longer deep-copy `BlockPlan`/`Step` (the per-region-iteration clone the `while` kernel used to pay). `Plan` must stay `Send + Sync` (the PyO3 handle holds `Arc<Plan>`), which is why `Step.constant` remains an owned `Buffer` and not a shared `Rc<Buffer>`.
- **`exp` on f32 uses libm, with no polynomial arm:** glibc's vectorized `expf` is 1.0 ns/element against the hand-rolled precheck + Taylor arm's 1.64 at every size (100 k elements +15 %, 1 M +49 %) and is the more accurate of the two, so `exp` runs the ordinary `Num::exp` path for every dtype.

## Constraints

- Depends on `evoxc-core` only — never `evoxc-compiler`, `evoxc-runtime`, `evoxc-python` or PyO3. Third-party deps: `half` (f16), `num-complex` (complex), `libm` (`erf`, `ln_1p`), `nalgebra` (linalg), `serde_json` (`block_call.static_args`) and `matrixmultiply` — the ONE performance dependency, a pure-Rust BLAS-3 microkernel used only by `dot`'s f32/f64 fast path (its threading feature is off, so one call is single-threaded and any parallelism comes from the engine's own scoped split).
- No `unsafe` logic of our own: the crate's only `unsafe` is the two `matrixmultiply` kernel calls, one macro-generated call site per float dtype in `kernels/linalg/matmul.rs`, each carrying a `SAFETY` argument that the three slices are exactly `m·k`/`k·n`/`m·n` long and the output is a fresh allocation that aliases neither input.
- Threading policy: scoped threads live inside one kernel call only — never a persistent pool, never a background worker, never `rayon` — gated by the real work thresholds (`PARALLEL_MIN_ELEMS` / `PARALLEL_MIN_PER_THREAD` for flat maps, `GEMM_PARALLEL_MIN_WORK` / `GEMM_PARALLEL_MIN_BLOCK_WORK` / `GEMM_PARALLEL_WORK_PER_THREAD` for `dot`), and results are unchanged because each thread owns a disjoint output range and reads only its own operand window.
- The storage pool is thread-local and never shared or synchronized, so no lock and no cross-thread allocation exists.
- User errors are `Result` values — no `unwrap`/`expect` on user input, no panics on malformed graphs; a shape or dtype the fast path does not cover falls back to the general kernel, never to an error.
- Result types and shapes come from the graph; never re-infer, and always validate produced buffers against `OpSpec.result_types`.
- One concern per file, target <~1000 lines including inline tests; a category that outgrows that becomes a module directory (`kernels/{elementwise,indexing,structure,sorting,reductions,linalg,linalg_aux,random}/`, `buffer/`) with a thin `mod.rs` holding `pub fn register` and the re-exports the rest of the crate reaches, one submodule per op family or shared mechanism, and a `#[cfg(test)] mod test_support`/`test_util` for fixtures sibling test modules share. `power`'s scalar-exponent specialisation is the one split-out category module, `kernels/pow.rs`.
- `cargo test -p evoxc-cpu` and `cargo clippy --workspace --all-targets --no-deps -- -D warnings` must stay clean.

## Test strategy

- `nan_to_num` reads its three `any` replacements from the attrs: `posinf`/`neginf` default to JSON `null` (the dtype's max/min finite value — numpy's own default), `nan` is `0.0` when absent and keeps NaN when explicitly `null` (the IR op's "None keeps the value"; numpy itself raises for `nan=None`), and a `null` replacement is never a type error. Pinned end-to-end (envelope decode → compile → run) in `tests/nullable_attrs.rs`, which also covers `flip.axes`, `roll.axis`, `argmax.axis`, `transpose.permutation`, `slice.strides`, `matrix_rank.tol`, `conv.strides`/dilations and the collective `group_size`.
- In-crate `#[cfg(test)]` per module: **309 lib tests (1 ignored)** plus 13 in the `tests/nullable_attrs.rs` integration target (nullable JSON-`null` attributes end-to-end) — kernel numerics (promotions, NaN/complex/rounding edge cases, error cases), symbolic-dim binding and the `DimExpr` failure modes, control flow (nested regions, multi-carry loops, recursion depth), the bit-exact random vectors for all three algorithms, collectives, plus the two registry-inventory tests that pin `unsupported_ops()` to empty.
The contiguous-copy paths pin their own behaviour: [`stride_copy`](src/kernels/indexing/stride_copy.rs)'s `lane_grouped_runs_match_the_elementwise_reference` (every stride/extent combination against a per-element reference, i32 and bool) and `a_pooled_strided_result_leaves_no_stale_element` (a parked garbage stock of the result's exact size is fully overwritten, for both a strided and a contiguous run), `slice`'s `the_last_axis_single_element_family_is_one_strided_run` (the `(500,50,4) → (500,50,1)` family is ONE axis of extent 25 000 stride 4) and `a_whole_array_slice_is_one_contiguous_run`, and `pad`'s `pad_copies_a_long_row_and_fills_only_the_pad_cells`, `pad_places_every_element_of_a_padded_result` (asymmetric pads on both axes, i32 and f32/NaN), `pad_rejects_a_result_shape_that_contradicts_the_padding`, `pad_rejects_a_mismatched_config_and_negative_padding`, `a_pooled_pad_result_leaves_no_stale_element` and `pad_handles_empty_extents`.
- The performance paths have their own pins: the storage pool's retention and eviction policy (`stocks_are_retained_per_size_up_to_the_per_size_cap`, `a_full_size_class_keeps_its_newest_stocks`, `the_pool_evicts_the_least_recently_used_other_size_class`, `a_differently_sized_predecessor_does_not_starve_later_workloads`, `a_stock_larger_than_the_byte_cap_is_refused_and_evicts_nothing`, `a_recycled_result_is_fully_overwritten`, `stock_reuse_is_last_in_first_out`), the scoped split (`the_thread_split_hands_every_chunk_its_running_offset`, `threaded_chunks_read_their_operands_at_their_own_offset`, `the_thread_split_scales_with_the_work_per_thread`), in-place accumulation against the allocating path (`in_place_accumulation_matches_the_allocating_map`, `in_place_accumulation_is_bit_identical_to_the_allocating_path`, plus the plan's `a_dying_broadcast_free_operand_is_the_in_place_target` / `a_constant_payload_is_never_an_in_place_target`), the reduction lane and arg-scan paths (`long_integer_sums_are_bit_exact_under_the_lane_split`, `multi_accumulator_max_and_min_propagate_nan_in_a_long_run`, `the_two_arg_scans_agree_on_short_and_long_runs`) and the sorting networks against `sort_unstable` (`the_network_covers_only_the_extents_it_wins_at`, `matches_the_reference_across_the_network_extents`, `matches_the_reference_past_the_network_ceiling`).
- The lib-level `#[ignore]`d test is `bench_thread_crossover`, a measurement aid that prints the per-element cost of a flat `add` at several sizes and thread counts (it is what `PARALLEL_MIN_ELEMS` is chosen from), run with `cargo test --release -p evoxc-cpu bench_thread_crossover -- --ignored --nocapture`.
- `tests/perf_probe.rs` is the second measurement aid (also `#[ignore]`d): one op per graph through `envelope::decode` + `plan::compile` + `run_program` (min of 200 runs, minus the same-input one-element-output floor) for the `slice`/`reshape`/`pad` families and a bare memcpy baseline, run with `cargo test --release -p evoxc-cpu --test perf_probe -- --ignored --nocapture`.
- Python parity harness `../tools/parity_cpu.py` (needs `scripts/build-native.sh` first and a `.venv`): **71 cases, 71 passed, 0 failed, 0 skipped** over elementwise/reductions/indexing/linalg/control-flow/random/dynamic-shape graphs plus the error and extension-resolver cases (elementwise 14, reductions 5, indexing 17, linalg 13, control-flow 5, random 9, dynamic 8), comparing `NativeProgram` against the numpy backend; exact for bool/int, `rtol 1e-5 / atol 1e-6` for f16/f32/c64 and `1e-12/1e-13` for f64/c128, with bit-exact RNG leaves (`PASS~` only for a f32 `normal` leaf, the documented 1-ulp Box–Muller window) and error parity when both sides raise.
- `../tools/bench_cpu.py` reproduces the table above.
- Tests never require a GPU, the network or Python (the Python harnesses are separate scripts, not cargo tests).

## Known Issues / Notes for Agents

- **`random_split_n` salts above `u64::MAX`:** `evoxc/ops/random.py` stamps `random_split_n`'s salts as `i * 0x9E3779B97F4A7C15`, which for `i >= 2` exceeds `u64` (`2 * GOLDEN` = 22801429638646396970) and so has no 64-bit envelope encoding. `../tools/parity_cpu.py` reduces such values modulo `2**64` (the same mask the numpy kernel applies through `np.uint64` arithmetic), and the real fix belongs to the frontend, out of this node's write scope.
- `evoxc-core`'s `int` attribute decode accepts any value that fits `u64` (it tries `as_u64` first and falls through to `as_i64` for negatives), so a single 64-bit salt such as the SplitMix64 golden gamma travels verbatim; `evoxc-python`'s inbound integer conversion likewise accepts `i64` then `u64`, so no 64-bit unsigned value needs narrowing.
- **`sort`/`argsort` are always stable:** with `stable=false` and duplicate values, numpy's quicksort may return a different (equally valid) permutation. One stable code path; ties are order-stable rather than numpy-quicksort-order.
- **`sort`/`argsort` lose to numpy's AVX2/AVX-512 sorting networks:** the engine's Batcher odd-even networks are scalar, so a row-wise 4096×64 sort/argsort measures 0.39× numpy — safe Rust rules out explicit SIMD intrinsics (they would need `unsafe`).
- **Elementwise throughput is per-op, not fused:** contiguous same-shape binary/unary ops and the comparison/cast/select family take dedicated slice fast paths and lead numpy on a 1e6-element chain (1.5–1.9×), but there is still no cross-op fusion, so an expression numpy can fuse into a single pass can win on very large single arrays; broadcasting and non-contiguous operands take the general strided driver.
- **`dot` loses to numpy's threaded OpenBLAS on small matrices:** 0.19× at 256², because the engine's scoped threading only engages above `GEMM_PARALLEL_MIN_WORK` / `GEMM_PARALLEL_MIN_BLOCK_WORK` and each `matrixmultiply` call is single-thread-limited by design, while numpy's BLAS threads a matrix this small at ~300 GFLOP/s.
- **`gp_gather_4096` is process-state-sensitive, not graph-limited:** alone in a fresh process it is 1.46–1.56× numpy (0.55–0.59 ms vs 0.86 ms), and inside the full eight-workload sequence the same code measures 0.80–1.5× because the per-op two-thread split pays thread-wake latency that varies with host load and core state.
  The remedy would be a persistent worker pool; rayon's scope would make that safe Rust, but it is gated above 2^18 elements by design, so the medium-op splits stay scoped-thread based.
- **`power` is specialised apart from the rest of the elementwise family:** `kernels/pow.rs` owns numpy's `npy_pow` arm table (`0`/`1`/`2`/`3`/`0.5`/`-1`/small-integer exponents, f32/f64 only, scalar exponent only) plus the negative-integer-exponent guard; `elementwise::binary` calls into it for the `power` kind and keeps the generic driver for everything else.
- **Single-rank only:** the collectives are identity kernels guarded by `world_size == 1`; a rank/world-size context exists (`RunOptions`) but no multi-rank executor.
- **Host callbacks are the caller's job:** `block_call`, `external_call` and `runtime_call` dispatch through the `Callbacks` trait; `run_program` refuses a callback-needing plan without one, and the FFI's `resolver` is `None` by default. Results returned by a callback are validated like any kernel result.
- **f32 `normal` is not bit-exact by design:** it shares the numpy backend's Box–Muller fast path (within 1 ulp); every other random path (integer words, `uniform`, f64 `normal`) is bit-exact.
- **`sparse_*` is out of scope:** the dense engine rejects the 16 sparse ops with an explicit `BackendError` naming the op, mirroring the numpy backend's `SparseError`/compiler-backend deferral. Densify with the graph's `sparse_to_dense` first.
- **`nalgebra` is declared but unused by the linalg kernels:** the factorizations are hand-written (`eigh` via cyclic Jacobi, `qr` via Householder, `svd` via one-sided Jacobi) so the numerics stay deterministic and dependency-light; remove the dependency rather than grow the surface.
- **Shape-only ops COPY the whole array:** `reshape` and an identity `transpose` take a pooled destination and `copy_from_slice` the whole storage (`kernels/indexing/reshape.rs:65`, `:127` — one bulk memcpy, but still a copy), a same-shape `broadcast` returns `buffer.clone()` (`buffer/broadcast.rs:147`) and otherwise materializes through `broadcast_vec` (`:157-173`), and a `constant` step deep-clones its plan payload once per use (`exec.rs:245-247`); `Buffer`/`Storage` own plain `Vec`s (`buffer/storage.rs:14-30`), so each is a full array memcpy where the numpy backend returns a view (`np.reshape`/`np.broadcast_to`/`arr.transpose`).
This is the floor for `reshape`/identity-`slice`/`transpose`: without views in `Storage` the engine cannot match numpy's ~1 µs view return, so those ops stay at memcpy speed (`reshape f32 (500,100)`, 200 KB, 3.14 µs of kernel against 6.20 µs for a 400 KB i32 memcpy baseline; the 800 KB identity `reshape` 19.8 µs) rather than view speed.
- **`gather`'s run copier leaves a 2.5–6× residual on the run-per-index families and is almost the whole remaining native-vs-pre-migration gap:** `kernels/indexing/gather.rs::gather_typed` resolves every index into a `Vec<usize>` and, per index, replays a pre-built `strided_copy::AxisCopy` plan (`AxisCopy::copy` → `copy_at` → `write_run` → `fill_run`).
`fill_strided`'s lane path (LANES = 4) is only reached for a stride>1 run and is at parity; the loss is the per-index call chain plus the separate resolve pass, over shapes the pre-migration `gather_into` fused into one loop.
The interleaved one-element-per-index families are the worst because their innermost axis collapses to stride 0, so `copy_elements` walks ONE PAIR AT A TIME instead of the pre-migration scalar `values[selected]` / `chunks_exact_mut` block loops.
`pooled_total` (`kernels/indexing/mod.rs`) also falls back to a full `T::zeros` memset whenever the pool misses under streaming load.
Measured (scratch `Instant` brackets, HEAD vs `e07bac8`, `xor_p500`/`cartpole_p100` seed 42, 8 gens, medians of 8): gather contributes **+17.3 ms** of `xor_p500`'s +18.0 ms leaf gap and **+18.7 ms** of `cartpole_p100`'s +19.1 ms.
Per family (HEAD ns/element vs pre-migration): `xor gather f32 (500,100,2)` (40 calls) 38.1 → 94.7 (2.5×), `xor gather f32 (500,50,5)` (40) 2.6×, `xor gather f32 (2000,50,4)` (5) 2.2 → 13.1 (5.9×), `xor gather bool (50000)` (4) 0.51 → 2.95 (5.8×), `cartpole gather f32 (100,50,4)` (404) 0.52 → 2.74 (5.2×).
Fix direction: a monomorphic `gather` fast loop for the two collapsed plans (one contiguous `A`-run per index → inline `copy_from_slice`; one strided run → inline `fill_strided`), fusing `resolve_indices` into it and dropping the `Vec<usize>`; a `copy_elements` stride-0 arm plus an `#[inline]` of the replay chain is a weaker version of the same.
- **A flat map over ≥ 2^18 elements spawns scoped threads per op:** `PARALLEL_MIN_ELEMS = 1 << 18` (`buffer/parallel.rs:41`) with one thread per 2^17 elements capped at 4 (`:47-49`), and each split is one `std::thread::scope` with `scope.spawn` (`:70-98`) at a documented ~10 µs setup per extra thread (`:26-27`).
- **Result allocation is pooled on the hot kernels only:** `map1..3`, the stride copies (`block_storage_filled`/`fill_strided`), `reshape`/`transpose`/`pad`, the comparison/logical `bool_results` family and `cast`'s `convert_generic` take pooled storage, while the general broadcast/strided drivers still zero-fill (`vec![T::zero(); n]`), `gather` takes a `pooled_total` destination (a full `T::zeros` memset on a pool miss) and `concatenate`/`sort` allocate with `Vec::with_capacity`; every kernel also returns a fresh `Vec<Value>` plus one `Rc` per result (`exec.rs`).
- **`random_normal` is the random family's slow path:** it still materializes the salted word stream, each parameter operand through `broadcast_to` + `Vec<ScalarValue>` and a final `Vec<ScalarValue>` cast, and its Box–Muller tail is a scalar `ln`/`sqrt`/`cos` loop (~14.5 ns/element against numpy's 7.8; +0.34/+0.30 ms per run on `xor_p500`'s two call shapes). `random_uniform` and `random_randint` take the typed contiguous path in `kernels/random/sampled.rs` and are 2–3× faster than numpy.
- **Broadcast-shaped elementwise ops run the stride driver, not the flat maps:** a broadcast operand takes `buffer/stride`'s run walk (innermost contiguous run, outer axes strided) instead of `map2`/`map3`, which measures ~0.55 ns/element against numpy's 0.29 for a broadcast `multiply (2000,50,4)` (+0.44 ms/run on `xor_p500`) — the residual is per-element dispatch, not layout math.
- **`dot` still copies an operand when the compute dtype differs:** `dot_impl` casts only a differing operand now (`kernels/linalg/matmul.rs`), so a same-dtype matmul is copy-free on the input side; `conv.rs` and `solve.rs` keep the older unguarded `cast` call and still memcpy both inputs.
- **`sort`/`argsort` each compute their own permutation:** `sorted_positions` allocates a full-size `Vec<u32>` (`kernels/sorting/permutation.rs:45`), gathers every non-innermost segment through a per-segment scratch copy (`:78-97`), sorts long extents with scalar `sort_unstable` (only extents ≤ 16 use the Batcher network, `network.rs:56`) and resolves wide `f64`/`i64` keys through an index-comparator closure with two indirect loads per comparison (`permutation.rs:128-144`), so the frontend's `topk` = sort + argsort + gather composition pays the whole ordering twice per call.
- **Coverage of the slow paths:** `../tools/bench_cpu.py`'s eight workloads cover flat elementwise/transcendental chains, `dot_sum_256`, GP gather/sort, softmax and the `tn_step_pop500` real-workload approximation, which exercises `random_*` and broadcast-dim elementwise ops; what remains unbenched in isolation is `random_normal`'s word-stream tail and the stride driver's fully-general case.
## Routing Table

| Path | Area |
|---|---|
| `./Cargo.toml` | Crate manifest: `evoxc-core` + `half` / `num-complex` / `libm` / `nalgebra` / `serde_json` / `matrixmultiply`. |
| `./src/lib.rs` | Module wiring, `run_program` / `evaluate`, the callback-requirement pre-flight. |
| `./src/plan.rs` | `compile(&Program) -> Plan`: `OpSpec`, `Step` (kernel, operand/result ids, liveness free list, per-result static-`(dtype, shape)` marker, plan-time decoded `constant` payload, in-place slot), `InPlace`, `BlockPlan` (held as `Arc`), `FunctionPlan`, `Plan::text()`, unsupported-op rejection. |
| `./src/exec.rs` | `RunCtx` (flat store, last-use release, region execution, dim binding, result validation, the `try_in_place` accumulate path), `Callbacks`, `RunOptions`, `Value`. |
| `./src/buffer/` | The typed value representation, one submodule per mechanism: `mod.rs` (`Buffer` + impl, `shape_text`, the crate-internal re-exports), `storage.rs` (`Storage` over all 14 dtypes, little-endian decode, `cast_storage`/`convert_generic`), `broadcast.rs` (`broadcast_shape`/`broadcast_strides`/`walk`/`apply1..3`/`broadcast_to`), `zipmap.rs` (the flat `zip1..3`/`zip2_accumulate`/`map1..3` builders), `parallel.rs` (`PARALLEL_*`, `split_parallel`, `window`), `pool.rs` (`POOL_*` caps, `StoragePool`, `take_pooled_vec`, the `Drop` impls), `blocks.rs` (`copy_blocks`, `block_storage*`, `repeat_storage`), `test_support.rs` (the one shared `#[cfg(test)]` fixture). |
| `./src/scalar.rs` | `Scalar` / `Num` / `ScalarValue` and the per-dtype numpy-semantics impls. |
| `./src/dims.rs` | Symbolic-dim binding and `DimExpr` evaluation (the `shapes.py` contract). |
| `./src/kernels/mod.rs` | `DENSE_OPS`, the kernel registry, `unsupported_ops()`, the registry-inventory tests. |
| `./src/kernels/{elementwise,indexing,structure,sorting,reductions,linalg,linalg_aux,random}/` | Per-category kernels as module directories: each `mod.rs` holds `pub fn register`, the kernel-level dispatch, the re-exports other files reach and its inline tests, with one submodule per op family or shared mechanism (submodule map below). `linalg/matmul.rs` holds `dot`'s threaded `matrixmultiply` BLAS-3 fast path and its `SAFETY` guards. |
| `./src/kernels/pow.rs` | `power`'s scalar-exponent specialisation (`PowBase`/`powi`/`PowScalar`/`pow_scalar`), the negative-integer-exponent guard and its unit tests; private to `kernels`. |
| `./src/kernels/util.rs` | Axis normalization, dtype/shape helpers, shared result-shape lookup (genuinely shared vocabulary only). |
| `../tools/parity_cpu.py` | Python parity harness against the numpy backend (71 cases). |
| `../tools/bench_cpu.py` | Native-vs-numpy benchmark, the source of the table above. |

### Module layout (one line per submodule)

Each `mod.rs` keeps `pub fn register` plus the re-exports the rest of the crate reaches, so no `use` path outside a directory changed when it was split.

- `buffer/mod.rs` — `Buffer` + impl, `shape_text`, the re-exports.
- `buffer/storage.rs` — `Storage` over the 14 dtypes, LE decode, `cast_storage`/`convert_generic`.
- `buffer/broadcast.rs` — `broadcast_shape`/`broadcast_strides`/`walk`/`apply1..3`/`broadcast_to`/`broadcast_vec`.
- `buffer/zipmap.rs` — `zip1..3`, `zip2_accumulate`, `map1..3`.
- `buffer/parallel.rs` — `PARALLEL_*`, `parallel_threads`, `split_parallel`, `window`.
- `buffer/pool.rs` — `POOL_*`, `StoragePool`, `take_pooled_vec`, `Drop for Storage`.
- `buffer/blocks.rs` — `copy_blocks`, `block_storage`, `block_storage_filled`, `repeat_storage`.
- `buffer/stride/` — the broadcast run driver (`stride/mod.rs`: the run walk and `fill_broadcast`) with its pins (`stride/tests.rs`).
- `buffer/test_support.rs` — the shared `#[cfg(test)]` fixture.
- `kernels/elementwise/mod.rs` — registry, the `stop_gradient`/`cast`/`broadcast` kernels, the `in_place_kernel` re-export.
- `kernels/elementwise/binary.rs` — `BinKind`, the dense binary drivers (arithmetic/bitwise/shift), the in-place accumulate path.
- `kernels/elementwise/compare.rs` — the comparisons and `select`.
- `kernels/elementwise/unary.rs` — `UnKind` and the unary drivers.
- `kernels/elementwise/transcendental.rs` — `erf`, `gelu`, `nan_to_num` and their fixups.
- `kernels/indexing/mod.rs` — registry plus the shared index vocabulary (bounds, integer indices, block builders).
- `kernels/indexing/reshape.rs` — `reshape`, `transpose`.
- `kernels/indexing/slice.rs` — `slice` (bounds, strides, the row-block and `CopyPlan` selection).
- `kernels/indexing/stride_copy.rs` — `CopyPlan`/`CopyAxis` (innermost-first collapsed axes), `block_offsets`, `fill_strided` (the lane-grouped strided run); shared with `gather`'s block path.
- `kernels/indexing/gather.rs` — `gather` (typed index resolution, block and stride-copy paths).
- `kernels/indexing/scatter.rs` — `scatter` (typed indices, `scatter_selection`).
- `kernels/indexing/concatenate.rs` — `concatenate`.
- `kernels/indexing/test_support.rs` — the shared `#[cfg(test)]` builders.
- `kernels/structure/mod.rs` — registry, dispatch, `identity_shape`/`same_shape`, attribute helpers.
- `kernels/structure/tile.rs` — `tile` (promoted, whole-repeat, block and general paths).
- `kernels/structure/flip_roll.rs` — `flip`, `roll`.
- `kernels/structure/pad.rs` — `pad` (config validation against the declared shape, the pad value, the row-wise pooled copy).
- `kernels/structure/diag.rs` — `diag`.
- `kernels/structure/test_support.rs` — the shared `#[cfg(test)]` harness.
- `kernels/sorting/mod.rs` — registry, sort/argsort dispatch and axis handling.
- `kernels/sorting/keys.rs` — the dtype → order-preserving `u64` row-key table.
- `kernels/sorting/network.rs` — the Batcher odd-even networks and `order_keys`.
- `kernels/sorting/permutation.rs` — sorted/segment positions and the permuted-storage builders.
- `kernels/sorting/test_support.rs` — the shared `#[cfg(test)]` reference oracle.
- `kernels/reductions/mod.rs` — registry, `Kind`, the reduce/mean driver, `divide_by`.
- `kernels/reductions/layout.rs` — `Layout`/`Block` and the shared fold/accumulator machinery (lane split).
- `kernels/reductions/argmax.rs` — `argmax`/`argmin` and their scan selection.
- `kernels/reductions/cumulative.rs` — `cumsum`/`cumprod`.
- `kernels/linalg/mod.rs` — registry, `mod` wiring, the shared linalg test fixtures.
- `kernels/linalg/common.rs` — arity/attribute/result-shape helpers.
- `kernels/linalg/matmul.rs` — `dot`, the `Gemm` trait, the BLAS-3 fast path and its threading (the one file over 1000 lines, by design).
- `kernels/linalg/conv.rs` — `conv` (padding, dilations, group counts).
- `kernels/linalg/solve.rs` — `solve` (LU in f64 work space).
- `kernels/linalg/structural.rs` — `diagonal`, `tril`/`triu`.
- `kernels/linalg_aux/mod.rs` — registry and the six factorization kernels.
- `kernels/linalg_aux/common.rs` — `Mat`/`prepare`, result materialization, rank/tolerance helpers.
- `kernels/linalg_aux/eigh.rs` — cyclic-Jacobi `eigh`.
- `kernels/linalg_aux/cholesky.rs` — `cholesky`.
- `kernels/linalg_aux/qr.rs` — Householder `qr`.
- `kernels/linalg_aux/svd.rs` — one-sided-Jacobi `svd`.
- `kernels/linalg_aux/matrix_exp.rs` — `matrix_exp` (Taylor terms) and its helpers.
- `kernels/linalg_aux/test_support.rs` — the shared `#[cfg(test)]` numeric assertions.
- `kernels/random/mod.rs` — registry plus the op-level dispatch/attribute/result helpers.
- `kernels/random/bits.rs` — the splitmix64/threefry2x32/philox4x32_10 word streams and their constants.
- `kernels/random/key.rs` — `Algorithm`/`Key`/`Words`, key extraction and child derivation.
- `kernels/random/sampling.rs` — `random_key_mix`, `random_uniform`, `random_normal`, `random_randint`.
- `kernels/random/sampled.rs` — the typed contiguous fast path for `random_uniform`/`random_normal`/`random_randint` (bit-exact word visitors).
- `kernels/random/permutation.rs` — `random_permutation`, `random_multinomial`.
- `kernels/random/test_util.rs` — the shared `#[cfg(test)]` harness.
- `kernels/{control,collective,util,pow}.rs` — single-file modules, unchanged by the split.
