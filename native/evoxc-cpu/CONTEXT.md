# evoxc-cpu — the native CPU execution engine of EvoX Compiler

## Intent

`evoxc-cpu` is the native (Rust) CPU backend: it compiles an [`evoxc_core::Program`] into an immutable execution `Plan` and runs it on typed host buffers.
It is the numerical authority for concrete CPU execution — the Rust counterpart of the Python numpy interpreter in `../evoxc/backends/numpy/` — and it depends on `evoxc-core` only: no Python, no PyO3, no device APIs, no compiler backends.
The `evoxc-python` crate wraps it as `evoxc._evoxc_native.native_compile` / `NativeProgram`; that FFI is documented in `../evoxc-python/CONTEXT.md`.

Shape/type inference is NOT re-derived here: result types come from the graph (`OpSpec.result_types`, filled by `evoxc-core` inference at trace/build time) and every produced buffer is re-validated against them at run time.
The engine never falls back to Python and never guesses: an op without a kernel is an explicit `BackendError` at plan-build time naming the op (and its source location when the graph carries one).

```text
Program --compile--> Plan --RunCtx::new--> RunCtx --run--> Vec<Buffer>
```

## API surface

- `plan::compile(&Program) -> Result<Plan>` — one immutable `Plan` per program: per-function entry `FunctionPlan`, per-block `BlockPlan`, per-op `Step` (`OpSpec` + resolved kernel + operand/result value ids + liveness free list + static-`(dtype, shape)` fast-path flag). Compiles every reachable region and every `call` callee (BFS), collects the host-callback requirements (`required_blocks` / `required_external` / `required_runtime`) and rejects an unsupported op up front. `Plan::entry_function()`, `Plan::needs_callbacks()`, `Plan::text()` (the human-readable dispatch plan behind `NativeProgram.plan_text()`).
- `exec::RunCtx` — the interpreter: `RunCtx::new(program, Arc<Plan>, RunOptions, Option<&dyn Callbacks>)`, `run(flat_inputs) -> Result<Vec<Buffer>>`, `run_block(block_index, args)`, `run_region(region, args)`, `eval_shape`/`eval_shape_strict` (symbolic-dim binding + `DimExpr` evaluation), `result_shape`/`result_shape_strict`/`result_type`, `validate_result`. Values live in ONE flat `value_id`-indexed store per block execution; a value is dropped at its last use (nested-region uses are attributed to the enclosing op, so a control-flow parent's operand survives its region).
- `exec::Callbacks` — the host-callback trait the PyO3 layer implements: `call_block(name, args, static_args)`, `call_external(name, args)`, `call_runtime(callback, args)`, each returning `Result<Vec<Buffer>>`.
- `exec::RunOptions { rank, world_size }` — the rank context of the run (defaults `0` / `1`).
- `buffer::{Buffer, Storage}` — the typed value: `Storage` covers `bool`, `i8/i16/i32/i64`, `u8/u16/u32/u64`, `f16`, `f32`, `f64`, `c64`, `c128` (all 14 evoxc dtypes); `Buffer { dtype, shape, storage }` with `new` (validates dtype + element count), `from_payload` (graph constants), `zeros`, `cast`, `with_shape`, broadcasting helpers (`broadcast_shape`/`broadcast_strides`/`broadcast_to`), and the generic `walk`/`apply1`/`apply2`/`apply3` drivers the kernels are built on.
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

Semantics are the numpy backend's, op by op: `remainder` is floormod (sign of the divisor), `round` is half-to-even, `maximum`/`minimum` propagate NaN and reject complex, comparisons produce bool and complex supports only `equal`/`not_equal`, `abs` on complex is the real magnitude, integer `divide`/`remainder` by zero and a negative integer exponent are explicit errors, integer arithmetic wraps like numpy.
`if` takes a 0-d bool predicate in operand 0 and passes ALL operands as region arguments; `while` has regions `[cond, body]` with no iteration limit; `call` runs the callee's entry block.
Result validation matches the numpy interpreter's messages ("kernel for op '{op}' produced result {i} with …"), a dtype mismatch is a `BackendError` and a shape mismatch a `ShapeError`.
Collectives are single-rank identities (`rank`/`world_size` read `RunOptions`); `world_size != 1` is an explicit `BackendError` — there is no multi-rank executor in this engine.
Random is BIT-EXACT with `../evoxc/backends/numpy/kernels/random.py` for splitmix64, threefry2x32 and philox4x32_10 (key handling, salt folding, counter layout, uniform/randint/permutation/multinomial/normal, including the f32 Box–Muller normal fast path).

## Performance (measured on this host, shared CPU)

Intel Core Ultra 7 356H, 16 cpus, loadavg 4.3–6.4 during the runs, Python 3.14.10 / numpy 2.5.2.
Warm `run` = median of 25 iterations after 5 warmups; the numpy leg is `evoxc.build(fn, *specs, backend="numpy")` + `evoxc.run` (`evoxc.evaluate` re-traces per call). Reproduce with `.venv/bin/python native/tools/bench_cpu.py` (records `model name` and loadavg itself).

| workload | ops | native compile (ms) | native run (ms) | numpy run (ms) | speedup |
|---|---|---|---|---|---|
| `ew_chain_1e6` (8 elementwise ops, 1e6 f32) | 14 | 0.062 | 41.83 | 3.00 | 0.07× |
| `ew_transcendental_1e6` (`erf`/`gelu`/`tanh` chain) | 7 | 0.035 | 37.81 | 181.62 | 4.80× |
| `dot_sum_256` (256² f32 `dot` + row sum) | 3 | 0.029 | 10.74 | 15.86 | 1.48× |
| `gp_gather_4096` (GP-shaped gather/select chain) | 80 | 0.534 | 19.52 | 0.86 | 0.04× |
| `chain_2000` (2000-op elementwise chain) | 2000 | 8.962 | 0.34 | 1.53 | 4.51× |

Compile target: the ENGINE's plan compile for a 2000-op graph is **0.811 ms** (target ≤ ~2 ms — pass); the full `native_compile` is 8.962 ms, of which 8.15 ms is envelope decode + FFI crossings, and the Python trace + `encode_module` on the caller's side is ~31 ms.

Reading of the numbers, which is load- and shape-dependent:
- Native elementwise kernels cost ~3 ns/element and the call has a sizeable fixed cost, so on large flat elementwise arrays numpy's vectorized ufuncs win (0.07×); native wins where numpy's per-op dispatch dominates (`chain_2000` at 0.17 µs/op, 4.5×) and where numpy's composition is expensive (`erf`/`gelu`, 4.8×).
- `dot` at 256² runs ~3 GFLOP/s single-threaded here; numpy's BLAS leg is unstable on this loaded host (0.11–15.9 ms across runs for the same cell — thread contention), so that row is indicative only.
- Per-op overhead, not per-element throughput, is the current strength; the gather-heavy row is numpy's best case (a single fancy-index ufunc) and native's worst.
- Compiled plans are reusable across runs (`Arc<Plan>`), so a caller that keeps the plan pays only `run`.

## Constraints

- Depends on `evoxc-core` only — never `evoxc-compiler`, `evoxc-runtime`, `evoxc-python` or PyO3. Third-party deps: `half` (f16), `num-complex` (complex), `libm` (`erf`, `ln_1p`), `nalgebra` (linalg), `serde_json` (`block_call.static_args`).
- No `unsafe` anywhere in this crate; user errors are `Result` values — no `unwrap`/`expect` on user input, no panics on malformed graphs.
- Result types and shapes come from the graph; never re-infer, and always validate produced buffers against `OpSpec.result_types`.
- One file per op category, each under ~1000 lines (test modules included; `kernels/indexing.rs`, `kernels/linalg.rs` and `kernels/random.rs` run over it because their per-op unit tests live inline).
- `cargo test -p evoxc-cpu` and `cargo clippy --workspace --all-targets --no-deps -- -D warnings` must stay clean.

## Test strategy

- In-crate `#[cfg(test)]` per module: **158 lib tests** — kernel numerics (promotions, NaN/complex/rounding edge cases, error cases), symbolic-dim binding and the `DimExpr` failure modes, control flow (nested regions, multi-carry loops, recursion depth), the bit-exact random vectors for all three algorithms, collectives, plus the two registry-inventory tests that pin `unsupported_ops()` to empty.
- Python parity harness `../tools/parity_cpu.py` (needs `scripts/build-native.sh` first and a `.venv`): 63 cases over elementwise/reductions/indexing/linalg/control-flow/random/dynamic-shape graphs, comparing `NativeProgram` against the numpy backend — **63 passed, 0 failed**; exact for bool/int, `rtol 1e-5 / atol 1e-6` for f16/f32/c64 and `1e-12/1e-13` for f64/c128, with bit-exact RNG leaves (`PASS~` only for f32 `normal`, the documented 1-ulp Box–Muller window) and error parity when both sides raise.
- `../tools/bench_cpu.py` reproduces the table above.
- Tests never require a GPU, the network or Python (the Python harnesses are separate scripts, not cargo tests).

## Known Issues / Notes for Agents

- **`random_key_mix` salt above `i64::MAX`:** the frontend stamps the splitmix64 golden-ratio salt (`0x9E3779B97F4A7C15`) as a plain Python int, and `evoxc-core/src/attr.rs` decodes the `int` attribute tag through `serde_json::Value::as_i64`, so an unsigned salt that does not fit `i64` is rejected at envelope decode (`expected an int, got 11400714819323198485`). The engine casts the attribute with `as u64`, so a two's-complement value is bit-identical; the parity harness narrows the attribute. Fixing it belongs to `evoxc-core`'s attribute decoding, not here (out of this node's contract — report upstream).
- **`sort`/`argsort` are always stable:** with `stable=false` and duplicate values, numpy's quicksort may return a different (equally valid) permutation. One stable code path; ties are order-stable rather than numpy-quicksort-order.
- **Elementwise throughput is scalar:** one element at a time through the `Scalar`/`Num` generics (~3 ns/element on f32). Batch-wide fusion, SIMD or `rayon` are not wired; the win today is per-op overhead, not element throughput.
- **Single-rank only:** the collectives are identity kernels guarded by `world_size == 1`; a rank/world-size context exists (`RunOptions`) but no multi-rank executor.
- **Host callbacks are the caller's job:** `block_call`, `external_call` and `runtime_call` dispatch through the `Callbacks` trait; `run_program` refuses a callback-needing plan without one, and the FFI's `resolver` is `None` by default. Results returned by a callback are validated like any kernel result.
- **f32 `normal` is not bit-exact by design:** it shares the numpy backend's Box–Muller fast path (within 1 ulp); every other random path (integer words, `uniform`, f64 `normal`) is bit-exact.
- **`sparse_*` is out of scope:** the dense engine rejects the 16 sparse ops with an explicit `BackendError` naming the op, mirroring the numpy backend's `SparseError`/compiler-backend deferral. Densify with the graph's `sparse_to_dense` first.
- **`nalgebra` is declared but unused by the linalg kernels:** the factorizations are hand-written (`eigh` via cyclic Jacobi, `qr` via Householder, `svd` via one-sided Jacobi) so the numerics stay deterministic and dependency-light; remove the dependency rather than grow the surface.

## Routing Table

| Path | Area |
|---|---|
| `./Cargo.toml` | Crate manifest: `evoxc-core` + `half` / `num-complex` / `libm` / `nalgebra` / `serde_json`. |
| `./src/lib.rs` | Module wiring, `run_program` / `evaluate`, the callback-requirement pre-flight. |
| `./src/plan.rs` | `compile(&Program) -> Plan`: `OpSpec`, `Step` (kernel, operand/result ids, liveness, static fast path), `BlockPlan`, `FunctionPlan`, `Plan::text()`, unsupported-op rejection. |
| `./src/exec.rs` | `RunCtx` (flat store, region execution, dim binding, result validation), `Callbacks`, `RunOptions`, `Value`. |
| `./src/buffer.rs` | `Storage` (14 dtypes), `Buffer`, broadcasting + the `walk`/`apply1..3` drivers. |
| `./src/scalar.rs` | `Scalar` / `Num` / `ScalarValue` and the per-dtype numpy-semantics impls. |
| `./src/dims.rs` | Symbolic-dim binding and `DimExpr` evaluation (the `shapes.py` contract). |
| `./src/kernels/mod.rs` | `DENSE_OPS`, the kernel registry, `unsupported_ops()`, the registry-inventory tests. |
| `./src/kernels/{elementwise,reductions,sorting,indexing,structure,linalg,linalg_aux,random,control,collective}.rs` | Per-category kernels (each file owns its op set and its inline unit tests). |
| `./src/kernels/util.rs` | Axis normalization, dtype/shape helpers, shared result-shape lookup. |
| `../tools/parity_cpu.py` | Python parity harness against the numpy backend (63 cases). |
| `../tools/bench_cpu.py` | Native-vs-numpy benchmark, the source of the table above. |
