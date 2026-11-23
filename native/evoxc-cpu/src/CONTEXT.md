# src — the `evoxc-cpu` engine: plan, interpreter and the buffer substrate

## Intent

`src/` turns an `evoxc_core::Program` into an immutable `Plan` and executes it on typed host buffers: `plan.rs` compiles, `exec.rs` interprets, `buffer/` owns the value representation and the ndarray-backed layout substrate, `scalar.rs`/`dims.rs` supply numpy semantics and symbolic-dim binding, and `kernels/` holds one kernel per dense op.
The crate-level contract (support matrix, substrate rules, measured numbers, known deferrals) is `../CONTEXT.md`; per-family detail lives in the child nodes' own CONTEXT.md.

## API Surface

- `lib.rs` — module wiring, `run_program`/`evaluate`, the callback pre-flight.
- `plan.rs` — `compile(&Program) -> Plan`; `Step` (kernel, operand/result ids, liveness free list, per-result static-`(dtype, shape)` marker in `Step.fast`, plan-time decoded `constant` payload, in-place slot), `BlockPlan` behind an `Arc`, `FunctionPlan`, `Plan::text()`.
- `exec.rs` — `RunCtx` (ONE flat `value_id`-indexed store per run, last-use release, nested regions, dim binding, `validate_result`), `Callbacks`, `RunOptions`, `single(buffer)`.
- `buffer/` — `Buffer`/`Storage`, the thread-local storage pool, the flat `zip*`/`map*` builders, the repeat/periodic/row-split run drivers, the ndarray seam.
- `scalar.rs` — `Scalar`/`Num`/`ScalarValue` with numpy semantics baked in.
- `dims.rs` — `bind_dim`/`eval_shape`/`eval_shape_strict` (the `evoxc.trace.shapes` contract).
- `kernels/` — `DENSE_OPS` registry, `unsupported_ops()` (empty), the per-category module directories.

## Per-op execution contract

Each executed op pays, in this order: `ctx.result_shape` (the declared shape evaluated into a `Vec<Option<usize>>`), the kernel's own attribute decoding and shape reconciliation, one result construction, `validate_result` (a dtype compare plus a shape slice compare whenever `Step.fast` holds a static marker, so no re-evaluation on the hot path), and the store writes.
`single(buffer)` costs a `Vec` plus an `Rc` per op; the flat store, the `Bindings` map and the `operands` scratch are allocated once per run, never per block, so a `while` body executed many times pays only the step loop.
A block entry clones an `Arc<BlockPlan>`, never the block's steps.

## Cross-cutting per-call costs (many-small-op graphs)

These are the fixed costs that dominate a graph with ~28k executed ops over small arrays; they are engine-wide, not per-family, so a fix belongs here or in `buffer/`.

- Per op: `single` (`Vec` + `Rc`), the evaluated declared shape (`Vec<Option<usize>>`), the strict `util::result_shape` copy, each builder's `out_shape.to_vec()`, and `broadcast_shape` where applicable — roughly 4 to 6 small allocations per op.
- Per op: `take_pooled_vec` and the `Drop for Storage` admission both scan the thread-local pool's entry table (up to `POOL_MAX_ENTRIES` = 64) by `(dtype, element count)`, and a result under `POOL_MIN_BYTES` (4 KiB) can never be retained at all, so every small result pays a scanning miss plus a malloc.
- `exp` on f32 runs one scalar libm `expf` per element (~1 ns/element); the polynomial arm was measured slower and removed, so this is a floor rather than a tuning gap.
- Elementwise ops against a rank-0 operand take `Layout::Scalar` and the flat `map1` arm; a one-element operand of any other shape (a `(1,)` constant) does not, and falls into the broadcast classification.
- `select` requires ALL THREE operands to match the result shape exactly before it takes the flat `map3` arm, so a scalar or one-element branch pays the full broadcast-driver classification (the stride proofs plus a possible zero fill).
- `Storage` owns a plain `Vec`, so `reshape`, an identity `transpose` and every shape-only op materialize a full copy where a view would do.

## Constraints

- Depends on `evoxc-core` only; no Python, no PyO3, no device APIs, no compiler backends.
- No `unsafe` anywhere; user errors are `Result` values (`ShapeError`/`BackendError`/`DTypeError`), never panics.
- Never re-infer: result types and shapes come from the graph, and every produced buffer is validated against `OpSpec.result_types`.
- One concern per file, target ~1000 lines; a family that outgrows it becomes a module directory with a thin `mod.rs`.
- Every fast path in `buffer/` and `kernels/indexing/` is a documented specialization whose ndarray expression is the reference in the sibling tests.

## Routing Table

| Path | Area |
|---|---|
| `./plan.rs` | `compile` → `Plan`, `Step`/`Step.fast`/`InPlace`, `BlockPlan`, `FunctionPlan`, `Plan::text()`, unsupported-op rejection. |
| `./exec.rs` | `RunCtx` (flat store, last-use release, `run_block`/`run_region`, `try_in_place`, `validate_result`), `Callbacks`, `RunOptions`, `single`. |
| `./buffer/` | `Buffer`/`Storage`, ndarray seam, pool, flat builders, repeat/periodic/row-split drivers. Owns its own CONTEXT.md. |
| `./scalar.rs` | `Scalar`/`Num`/`ScalarValue` and the per-dtype numpy-semantics impls. |
| `./dims.rs` | Symbolic-dim binding and `DimExpr` evaluation. |
| `./kernels/elementwise/` | Unary/binary/compare/select drivers, `Layout` probe, transcendental fixups. Owns its own CONTEXT.md. |
| `./kernels/indexing/` | `slice`/`gather`/`scatter`/`reshape`/`transpose`/`concatenate` over the shared run copier. Owns its own CONTEXT.md. |
| `./kernels/reductions/` | `reduce_*`, `argmax`/`argmin`, `cumsum`/`cumprod`. Owns its own CONTEXT.md. |
| `./kernels/sorting/` | `sort`/`argsort` packed-key segment walk. Owns its own CONTEXT.md. |
| `./kernels/structure/` | `tile`/`pad`/`flip`/`roll`/`diag`. Owns its own CONTEXT.md. |
| `./kernels/linalg/`, `./kernels/linalg_aux/` | `dot`/`conv`/`solve`/`diagonal`/`tril`/`triu` and the six deterministic factorizations; contract in `../../CONTEXT.md`. |
| `./kernels/random/` | The three bit-exact algorithms and the typed contiguous sampling path. |
| `./kernels/{control,collective,util,pow}.rs` | Control flow with host callbacks, single-rank collectives, shared axis/shape helpers, `power`'s scalar-exponent table. |
