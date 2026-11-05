# evoxc/backends/numpy — reference numpy CPU interpreter backend (default backend)

## Intent

The **reference CPU interpreter backend** (name `"numpy"`, the DEFAULT backend): a pure-Python numpy interpreter proving EvoXIR semantics. It is one of only two concrete-computation paths in evoxc (the other being `core`'s concrete creators). Compiler-neutral by design: it does NOT optimize, schedule, or generate code — external compilers (IREE/XLA/TVM via the StableHLO export path) own compilation.

Two roles:

1. **Staging** (`NumpyBackend`): verified `Graph` → `LoweredProgram` (payload = versioned self-describing `ir.serialize_module`) → `CompiledArtifact` (target `"cpu"`; the artifact IS serialized IR — there is no machine code) → `NumpyExecutable` at `load`.
2. **Interpretation** (`NumpyExecutable.run`): executes the IR on concrete `core.Tensor`s (numpy arrays in v1) in block op order.

The parent contract (`../CONTEXT.md`, "Numpy backend design" section) is **BINDING** — the design below restates it, never modifies it. Root principles (`../../CONTEXT.md`) and the package contract (`../CONTEXT.md`) are also binding.

## API Surface

Exports (re-exported via `evoxc/backends/__init__.py` and surfaced at `evoxc.backends`):

| Name | Where | Notes |
|---|---|---|
| `NumpyBackend` | `__init__.py` | `Backend` ABC impl; `name="numpy"`; `capabilities`: `dynamic_shapes=True`, `dtypes=` all numpy dtypes (numpy.sctypes flattening + `bool_`), `collectives=True` (single-process simulation), `runtime_calls=True`, `custom_blocks=True`, `async_collectives=False` |
| `NumpyExecutable` | `__init__.py` | satisfies the `Executable` protocol: attrs `functions` (module function names), `device`, `signature`, `artifact`; `run(flat_input_tensors) -> flat_outputs`, `save(path)`, classmethod `load(path, device=None)` |
| `numpy_backend` | `__init__.py` | `NumpyBackend()` instance, registered via `..registry.register` at import time |
| `CollectiveExecutor` | `collectives.py` | runtime-checkable Protocol — CANONICAL home is `evoxc/dist/context.py` (`dist.context.CollectiveExecutor`); re-exported here as an alias: `all_reduce(tensor, group, op)`, `all_gather(tensor, axis, group)`, `reduce_scatter(tensor, axis, group)`, `all_to_all(tensor, axis, group)`, `broadcast(tensor, src_rank, group)`, `collective_permute(tensor, mapping, group)` — all `-> Tensor` (group/mapping are graph-constant objects) |
| `SingleRankCollectiveExecutor` | `collectives.py` | DEFAULT executor: every collective returns the tensor unchanged (identity semantics on one rank); installed into the `dist.context` slot at `numpy/__init__` import time |
| `set_collective_executor` / `get_collective_executor` | `evoxc/dist/context.py` (CANONICAL) | process-wide hook; numpy installs `SingleRankCollectiveExecutor()` at import (tests simulate multi-rank in-process) |
| `evaluate_dim_expr` / `evaluate_shape` | `shapes.py` | runtime `Dim`/`DimExpr` evaluation against name→int bindings (free dims / div-by-zero / negative results ⇒ `ShapeError`) |
| `KernelContext` / `Interpreter` / `entry_function` | `interpreter.py` | the execution engine (see "Implemented execution engine" below) |
| `set_rank_context` / `get_rank_context` | `exec_context.py` | thread-local `RankContext` hook (default `RankContext(rank=0, world_size=1)`; per-run override on `NumpyExecutable.run(..., rank_context=...)`) |
| `dispatch(op_name)` / `register_all()` / `KERNEL_TABLE` | `kernels/__init__.py` | op-name → kernel dispatch table; `register_all()` idempotent, duplicate keys across category modules ⇒ `BackendError` |

Not public: `_register_block_impls` (documented no-op — block dispatch resolves via `evoxc.block.registry` at lower/load/run time), `_module_function_names` (single touch point for the `ir.Module` accessor). The bottom-up op walk (`iter_ops`/`iter_block_ops`), the portable-splicing helpers (`clone_ops_into` / `drop_op_uses`), and the fixpoint driver (`inline_portables`) MOVED to the shared `../inline.py` (also used by compiler backends); `./inline.py` is a thin re-export.

## Implemented execution engine

`interpreter.py` holds the engine the staging flow and every kernel build on:

- **`KernelContext`** — per-execution context handed to kernels. Attributes: `.bindings` (dim name → int), `.rank_context` (from `exec_context.get_rank_context()`), `.module` (ir.Module). Methods: `.run_region(region, arg_tensors)` (bind entry-block arguments to tensors, extend `.bindings` positionally from region-arg types vs concrete shapes — Dim binds by name with conflict checking, DimExpr binds its Dim leaves, int must equal, None unchecked — then run the block ops and return the `return` terminator's operand tensors), `.compute_output_shapes(op, input_shapes, input_dtypes)` (evaluates `op.results[i].type.shape` against `.bindings`; None dims stay None — the IR result types were already inferred by ops-level inference at trace time, so this is the mandated shape-rule reuse, never a second copy of shape rules), `.resolve_callback(callback_id)` (lazy `evoxc.ops.constant._get_callback`; missing ⇒ `BackendError` naming the id), `.evaluate_shape(shape)` (convenience over `shapes.evaluate_shape`).
- **`Interpreter`** — holds module + signature. `run(flat_input_tensors, rank_context=None)`: resolves the entry function (`module.get_function("main")`, fallback `module.main` when exactly one function); validates input count, per-input dtype (⇒ `DTypeError`) and per-input shape against spec shapes (Dim binds by name with conflict/known-size checks, DimExpr binds leaves then evaluates and compares, int must equal, None unchecked — ⇒ `ShapeError`); wraps the run in `exec_context.set_rank_context(rank_context)` with try/finally restore; executes the entry region; validates output count + dtype vs `signature.output_specs`.
- **Op loop** (`_run_block`, used recursively for nested regions): executes a block from its cached per-block EXECUTION PLAN (`_build_plan`, built on the block's first execution, cached on the `Interpreter` by `id(block)`; the plan pins the block, so a cached key's id can never be reused). A plan step carries `(op, kernel, operand_ids, result_ids, free_ids, fast, one)` — the kernel already RESOLVED through `kernels.dispatch` (an unknown op still raises the historical `no numpy kernel for op '...'` `BackendError`, at plan build), the flat-store value ids of operands/results, the value ids whose last use is this op, and the static-result fast path (`one = (dtype, shape)` when the op has exactly one result whose declared shape is all plain `int`s). Per op: read operands from the flat store (a missing value ⇒ the historical unresolved-value `BackendError`), call `kernel(ctx, op, operands)`, validate + store the results, then release the values whose last use was this op.
  Validation: the fully static single-result case is ONE `(dtype, shape)` tuple compare on the tensor's ndarray payload; any mismatch (and every multi-result / symbolic / `None`-dim / non-ndarray-payload / tuple-returning case) is handed to `_store_results` → `_validate_result`, i.e. the unchanged full path — count, exact dtype (⇒ `BackendError` — kernels never silently coerce), rank and per-dim shape against evaluated `op.results[i].type.shape` with None dims unchecked (⇒ `ShapeError`). Both paths build shape-mismatch wording through the single `_mismatch_error` helper, so messages are byte-identical whichever path catches the mismatch.
  The `return` terminator is not a step: the plan stops at the FIRST `return` op in block order (the op the pre-plan loop stopped at) and keeps its operand ids.

## Interpreter design (binding from parent)

**Staging flow** (`NumpyBackend`; implemented):
- `lower(graph)`: (1) `graph.verify()` (surfaces `core.VerificationError`); (2) capability pre-check (v1 numpy supports everything — the check pattern stays); (3) inline `block_call` portable decompositions as graph→graph expansion at LOWER time (block with neither portable decomposition nor registered numpy impl ⇒ `core.BackendError` naming the block); (4) record `Signature` from the Graph (input/output TreeSpec + per-leaf specs + static values — passed down, never re-derived); (5) payload = `ir.serialize_module(graph.module)`.
- `compile(lowered)`: validate `lowered.backend == "numpy"` (else `BackendError`); record `required_custom_ops` + `runtime_dependencies` (self-describing); wrap serialized module as `CompiledArtifact(target="cpu")`. No machine code.
- `load(artifact, device)`: validate backend/device (None or CPU; else `BackendError`/`DeviceError`), required custom ops availability; `ir.deserialize_module`; build `NumpyExecutable`. **Never re-compiles.**

**Execution model** (`NumpyExecutable.run`):
- **Execution order = block op order.** This IS the effect ordering (write/read/collective/callback ops anchor order); pure ops keep program order for determinism.
- **Shape inference: reuse, don't duplicate.** Runtime shapes = ops-level inference rules with symbolic dims bound to concrete values (`shapes.py` evaluates `DimExpr` against name→int bindings). The backend carries NO second copy of shape rules. Free symbolic dims ⇒ `ShapeError`.
- **dtypes**: numpy dtype ↔ evoxc dtype is 1:1; kernels validate support; no promotion beyond what ops defines.
- **Control flow**: `cond`/`while_loop`/`scan` region ops are interpreted by recursively running region blocks — genuinely dynamic runtime control flow (the graph is NOT specialized per iteration).
- **`runtime_call`**: executes the Python callback synchronously at the op's position — a documented sync point (no async execution in v1).
- **`block_call` dispatch**: portable decompositions are inlined at `lower()` time, so the interpreter only ever dispatches blocks that have a registered numpy impl (`kernels/custom.py`); otherwise `BackendError` (safety net, never a fallback).
- **Collectives**: ALL collective ops dispatch through the `CollectiveExecutor` hook; default `SingleRankCollectiveExecutor` = identity on one rank; tests install multi-rank simulators via `set_collective_executor`. Group semantics are simulation-only in v1.
- **Persistence**: `NumpyExecutable.save` saves the underlying `CompiledArtifact`; the executable is reconstructed explicitly at `load` — device handles are never serialized.

## Run cost model (what one `run` pays, structurally)

- **The interpreter is an op-by-op Python loop, so per-call cost is LINEAR in op count** — one plan-step iteration per op with no fusion, no batching and no IR-level precompilation (the only precompilation is the per-block plan below).
- Per op the loop pays: a list-comprehension operand fetch from the flat store (one dict subscript per operand), the kernel call, ONE `(dtype, shape)` tuple compare against the plan's precomputed static result (the static fast path — on failure, or whenever the result type is symbolic/`None`/multi-result, the full symbolic validation runs and builds the identical message), one `store[value.id] = tensor` write, and one `store.pop` per value whose last use was this op.
- **The plan is computed once per block** and reused across every execution (including loop iterations of nested regions): kernel resolution (`kernels.dispatch`), operand/result id tuples, the static result `(dtype, shape)` tuples, and the liveness map all come from it — nothing is re-derived per op per call.
- **Value liveness is per op**: uses inside nested regions are attributed to the ENCLOSING op's index (the op is still running while the region executes) and the terminator's operands are never released; only values DEFINED in the executing block are release candidates, so enclosing-block captures and loop-carried values stay resolvable. A wrongly released value surfaces as the unresolved-value `BackendError`, never as a silently wrong result.
- Kernels add their own per-operand/result Python cost on top of the numpy work: every non-`constant` kernel unwraps operands with `Tensor.numpy()` and re-wraps each result in `core.Tensor`; `constant` re-wraps the snapped payload WITHOUT copying it.
- `op.name`/`op.operands`/`op.results` are plain `ir.Op` dataclass fields (cheap); `op.opdef`/`op.effect` consult the registry and are therefore NOT on the run path.
- **What is cached across calls**: the `Interpreter` object, the kernel table and every block's plan are built once per executable (`NumpyExecutable.__init__`; `register_all()` idempotent via a module flag), so `run` rebuilds only per-execution state — the dim `bindings` dict, one `KernelContext`, and the (cleared-at-run-start) flat value store.
- **Control flow multiplies the loop cost per iteration**: `if`/`while`/`call` kernels re-enter `run_region` → `_run_block`, so a loop body pays the same per-op Python cost on EVERY iteration (the graph is never specialized per iteration) — but its plan and its static result shapes are already computed.
- The pipeline wrapper's plan caching (input/output plans, `_prepare_flat_inputs`/`_unflatten_outputs`) belongs to `evoxc/pipeline.py`, not to this backend: the numpy backend's own per-call work is the per-op loop above, so a merged `batch_functions` graph's numpy runtime is `emitted_ops` × (Python dispatch overhead + numpy work on row-stacked arrays).
- Recorded measurements live in sibling nodes, not here: numpy merged-graph per-program-row / per-call runtimes (including the `bands` matrix) are in `../../transforms/CONTEXT.md`, and the leaf-driven `evoxc.run` wrapper cost is in `../../CONTEXT.md` "Notes for agents".

## Measured cost (TensorNEAT generation graphs)

Protocol: external TensorNEAT harness (`run_benchmark.py --impl new --backend numpy --generations 30 --warmup 3 --repeats 2`), one fresh process per cell, alternating pristine and current tree in one session so drift cancels (`ms/gen` = the run's mean; the two means per side are the two interleaved rounds).

| cell | executed ops/gen | ms/gen pristine | ms/gen current | peak RSS pristine → current | compile pristine → current |
|---|---|---|---|---|---|
| `xor_p500` | 8548 | 179.6 / 175.8 | **127.7 / 122.6** | 235.5 → **120.0 MB** | 0.551 → 0.463 s |
| `custom_pagie_p500` | 12521 | 1045.2 / 1032.0 | **959.8 / 952.6** | 1247.5 → **877.3 MB** | 1.693 → 1.534 s |

Both cells' per-generation fitness curves are bit-identical to the pristine tree (all four runs; the harness mixes no timing into the search), so the win is pure overhead removal.
Per-change deltas (`xor_p500` / `custom_pagie_p500`, one repeat each): base `167.8 / 978.4`; plan + flat store + static result validation but NO dead-value release `152.6 / 1129.0` — the release is what keeps `pagie`'s live set (2.6 GB without it) and therefore its RSS bounded, so it is a performance component, not hygiene; plan + flat store + release but the static fast path forced off `154.4 / 1026.5`; all three plus the kernel hot-path pass `122–125 / 953–958`.
Where the remaining time goes (`xor_p500`, cProfile per generation; the profiler inflates the unprofiled 123 ms to ~169 ms): numpy compute dominates — elementwise kernels ≈26 ms, `argsort` 22, `take` 15, `select` 14, `ufunc.reduce` 7, `dot` 6, `astype` 6 — then the interpreter loop `_run_block` ≈14 ms, and `core.Tensor.__init__` + `core.Device.__post_init__` ≈8 ms for ≈8.7 k tensor constructions.
`custom_pagie_p500` is numpy-compute-bound at ≈78 µs/op (populations are large; the kernel Python share is ≈1 %), so its wall time is dominated by array work, not by this package's Python.
Per-op attribution must wrap `KERNEL_TABLE` entries (plans resolve kernels at build time, so wrapping `kernels.dispatch` no longer observes anything) and a `perf_counter` wrapper itself costs ≈1 µs/op — subtract it before believing a per-op delta.
Kernel-side hot-path rules and their own numbers live in `./kernels/CONTEXT.md`; re-measure with the external harness after unrelated changes rather than quoting these as a standing property.

## Error behavior (binding)

- Unsupported ops/features encountered at run time (capability drift) ⇒ `core.BackendError` **naming the op** — never a silent skip.
- Free symbolic dims at run time ⇒ `core.ShapeError`.
- `lower()` surfaces `core.VerificationError` from `graph.verify()` as-is.
- Load-time mismatches (backend/device/ABI/custom-op availability) ⇒ `core.PersistenceError` — **never** a silent re-lower/re-compile.
- No silent fallbacks or partial semantics anywhere in this package.

## Constraints

- **Import acyclicity (binding)**: top-level imports restricted to `evoxc.core` and `evoxc.ir`; `evoxc.ops` may be imported ONLY inside function bodies (`_register_block_impls` is the sole allowed site — lazy import); NEVER import `evoxc.pipeline` or `evoxc.persist` at top level (persist is lazy inside `save`/`load` bodies per `../program.py`).
- **Files < ~1000 lines**: kernels are pre-split by category (`kernels/`); split along the declared category boundaries instead of growing files.
- **CPU only in v1**: never requires or touches GPU.

## Routing table

| Path | Area |
|---|---|
| `./__init__.py` | `NumpyBackend` (lower/compile/load — incl. lower-time block_call portable inlining via the shared `../inline.py::inline_portables(keep_backend_impls="numpy")`: `get_impl("numpy")` → keep, `get_portable` → splice, neither → `BackendError`), `NumpyExecutable`, `numpy_backend` + registration, `_register_block_impls` (documented no-op), re-exports, import-time `dist.context.set_collective_executor(SingleRankCollectiveExecutor())` |
| `./interpreter.py` | `KernelContext` (bindings, rank_context, module, run_region, compute_output_shapes, resolve_callback, evaluate_shape), `Interpreter` (run + entry-function resolution + `_run_block` plan-driven op loop + the flat `value.id -> Tensor` store shared by nested regions + the per-block plan cache `_plans`/`_build_plan`/`_BlockPlan` with liveness-based dead-value release), `entry_function` |
| `./exec_context.py` | thread-local `set_rank_context`/`get_rank_context` hook over `dist.context.RankContext` (default rank=0, world_size=1) |
| `./inline.py` | thin re-export of the shared `../inline.py` (`clone_ops_into` — portable block_call splicing with fresh ids + Use bookkeeping + output-type guards; `drop_op_uses` — required before `Block.erase` under strict `ir.verify`; plus `inline_portables`/`iter_ops`/`iter_block_ops`) |
| `./collectives.py` | `CollectiveExecutor` alias of `dist.context.CollectiveExecutor` (CANONICAL home), `SingleRankCollectiveExecutor` (identity default) |
| `./shapes.py` | `evaluate_dim_expr` / `evaluate_shape` — runtime `Dim`/`DimExpr` evaluation (shape-rule reuse) |
| `./kernels/__init__.py` | `KERNEL_TABLE`, `dispatch(op_name)`, `register_all()` (idempotent) — the kernel contract is documented in this module's docstring and is binding for all category modules |
| `./kernels/elementwise.py` | add/subtract/multiply/divide/power/remainder/maximum/minimum/abs/negate/square/sqrt/exp/log/log1p/sin/cos/tan/tanh/sigmoid/relu/gelu/erf/sign/bitwise_*/logical_*/cast/equal/not_equal/less/less_equal/greater/greater_equal/select/broadcast/stop_gradient |
| `./kernels/reductions.py` | reduce_* family (reduce_sum/reduce_max/reduce_min/reduce_mean/reduce_prod), argmax/argmin, cumprod (cumsum moved out to indexing.py) |
| `./kernels/indexing.py` | reshape/transpose/slice/gather/scatter/concatenate/pad/tril/triu/cumsum |
| `./kernels/linalg.py` | dot/conv/solve |
| `./kernels/control_flow.py` | if/while/call — recursive region execution (the `return` terminator is special-cased by the interpreter loop, never dispatched) |
| `./kernels/collective.py` | dist collective ops dispatched through `dist.context.get_collective_executor()` (rank/world_size resolved from the per-run `RankContext`) |
| `./kernels/random.py` | the 6 `random_*` kernels (evoxc.random family): three algorithms — splitmix64 (v1 default, rank-0 int64 key), threefry2x32 (2×u32 key, 20 rounds), philox4x32_10 (4×u32 key, 10 rounds) — pure/deterministic (same key ⇒ bit-identical values across runs); the module docstring pins the WORD-STREAM CONTRACT (salt folded into the key, counter (p, 0), C-order element mapping, width-parametrized uniform scaling) that the StableHLO export must mirror exactly. **f32 `random_normal` fast path**: when the op's output dtype is float32 the Box–Muller tail runs entirely in float32, mirroring the stablehlo exporter's f32 semantics EXACTLY (same words 2i/2i+1 drawn directly as two parallel chains via `_normal_word_pairs` — no interleaved `(2*count,)` stream; word→f32 conversion first, exact `2^-53`/`2^-32` power-of-two scaling, word-width u1 guard, `sqrt(-2 log u1) * cos(2π u2)` and the mean/std combine in f32 with mean/std rounded to f32 first); f64/f16 outputs and every other op keep the exact f64 math (uniform/randint/permutation/multinomial stay bit-exact vs the StableHLO export). Deterministic; values deviate from the f64 chain at the ~1e-6 relative scale (f32 mantissa rounding + f32 transcendental ulps) |
| `./kernels/sorting.py` | sort/argsort — np-exact |
| `./kernels/structural.py` | tile/flip/roll/diag + nan_to_num — np-exact |
| `./kernels/custom.py` | `constant`, `runtime_call` (sync callback via `ctx.resolve_callback` — artifacts with `runtime_call` require the same callback registrations at load time), `block_call` (registered numpy impl dispatch) |
| `./kernels/sparse.py` | the 16 sparse kernels: `sparse_from_dense`/`sparse_to_dense`, coo↔csr / coo↔csc conversions, `sparse_negate`/`add`/`multiply`/`multiply_dense`, `sparse_reduce_sum`, `sparse_transpose`, `sparse_reshape`, `sparse_concatenate`, `sparse_dot_dense`/`dense_dot_sparse` — batched via per-batch loops with lex-max stored-zero padding to a common nnz; shared values-aware canonical validation (see Notes for agents) |

Sibling: `../../../tests/` → test suite (read-only from here; escalate test-related writes to root). Parent: `../` → Backend ABC, `LoweredProgram`/`CompiledArtifact`/`Signature` (owned there), registry (optional-adapter auto-activation), StableHLO exporter, shared `inline.py` block-inlining machinery, `compiler.py` (`CompilerBackend`/`CompilerExecutable`), `adapters/` (separate parallel effort).

## Test strategy

Planned tests live in `../../../tests/backends/` (sibling — read-only from here; test-related writes escalate to root), CPU only, pytest, numpy-only deps:
- **Per-op coverage per kernel category** (elementwise/reductions/indexing/linalg) against the ops-level semantics.
- **Symbolic dims**: runtime binding of `Dim`/`DimExpr` shapes; free symbolic dim ⇒ `ShapeError`.
- **Dynamic control flow**: `cond`/`while_loop`/`scan` genuinely dynamic (per-iteration shapes, early exit).
- **`runtime_call`**: synchronous execution at the op position; output-spec mismatch ⇒ `BackendError`.
- **Block dispatch**: registered impl vs portable decomposition (inlined at lower) vs missing ⇒ `BackendError` naming the block.
- **Collectives**: single-rank identity AND multi-rank in-process simulation via `set_collective_executor` (shared module state — reset in teardown).
- **Persistence round-trips**: `LoweredProgram`/`CompiledArtifact`/`NumpyExecutable` save/load; backend mismatch ⇒ `PersistenceError`; never recompiles.

## Notes for agents

- **Kernel dispatch coverage (complete)**: all 104 non-`return` IR op names are registered in `KERNEL_TABLE` (88 non-sparse ops + the 16 sparse ops — elementwise 40, reductions 8, indexing 10, linalg 3, sorting 2, structural 5, control_flow 3, collective 8, custom 3, random 6; the registry holds 105 op defs incl. `return`); the `return` terminator is special-cased by the interpreter loop and is never dispatched. `register_all()` is idempotent; duplicate keys across category modules ⇒ `BackendError`.
- **Sparse kernels (16 ops, `kernels/sparse.py`)** — key contracts for frontends and future agents:
  - **Values-aware canonical validation** (shared `_validate_canonical` + `_check_segment_sorted`): sorted/unique/in-range is enforced, BUT adjacent duplicate rows/segment entries are TOLERATED when at least one of the two values is 0 (stored zeros — this is what makes batched `sparse_from_dense`'s lex-max stored-zero padding legal). Duplicate NONZERO rows always raise `ShapeError`. All consumers handle stored-zero duplicates correctly (add.at / merge collapse / intersect1d first-occurrence).
  - **Batched `sparse_from_dense`** pads each batch element to the common max nnz with lex-max stored-zero rows; consumers see `(B..., nnz_max, ndim)`.
  - **`sparse_reduce_sum`**: an EMPTY `axes` attr = NO reduction (identity dense, per `infer_sparse_reduce_sum`); the frontend must emit the full axis tuple for an all-axes reduction.
  - **`sparse_concatenate`** REQUIRES the `operand_extents` attr (per-operand dense_shape[axis]); missing ⇒ explicit `BackendError`. The opdef amendment declaring it is pending upstream (evoxc/ir) — e2e concat graphs are blocked until then; the kernel is validated via direct invocation.
  - **`sparse_coo_to_csc`** emits the STANDARD CSC indptr of length `cols+1` (matching `infer_sparse_coo_to_csc`'s declaration `batch + (dense_shape[1] + 1,)`): `indptr[0] == 0`, monotone non-decreasing, `indptr[-1] == nnz`, batched per element. `_csc_to_coo` accepts the standard `cols+1` convention (primary; also emitted by the concrete `evoxc.sparse.csc` path) and, defensively for backward compat, the legacy padded `rows+1` convention (true column pointers in `ip[0:cols+1]` + monotone `nnz` tail).
  - **`sparse_reshape`** is static-shape only (None/dynamic dims ⇒ explicit `ShapeError` with a "dynamic" hint).
- `_register_block_impls()` is a documented no-op — block dispatch resolves via `evoxc.block.registry` at lower/load/run time (`get_impl(name, "numpy")` keeps the op; `get_portable(name)` is traced and spliced at lower; neither ⇒ `BackendError` naming the block). User blocks register via `BlockOp.impl("numpy")`/`.portable(...)`.
- **Block-impl call convention (finalized with `evoxc.block`)**: a registered numpy impl is called as `impl(*numpy_arrays, **static_args) -> ndarray | tuple[ndarray, ...]` — operand tensors arrive as raw numpy arrays (`.numpy()`), the op's JSON-able `static_args` attribute arrives as kwargs, and the return is normalized and validated exactly like `runtime_call` (count/dtype/shape against the declared result specs).
- **Known v1 collective-protocol limitations** (documented, never silently worked around): the `CollectiveExecutor` protocol's `reduce_scatter` has NO `reduce_op` parameter, so the op's `reduce_op` attr is not forwarded to the executor; `all_to_all` takes a single `axis`, so only `split_axis` is forwarded (`concat_axis` is not — v1 protocol limitation).
- **Output validation with runtime-dynamic dims**: op result shapes with `None` (runtime-dynamic) dims are UNCHECKED per-dim by the interpreter (rank is still validated exactly); every non-`None` dim must evaluate to the concrete runtime dim (⇒ `ShapeError` otherwise).
- **Nested-region outer-value resolution**: region ops may legally reference SSA values defined in ENCLOSING blocks (e.g. `evoxc.scan`'s desugared regions capture the length constant and the `xs` leaves — they are not while-op operands). One FLAT store (`Interpreter._values`, `value.id -> Tensor`) is shared by every executing block — value ids are module-unique — and block arguments are re-bound at each block entry; the plan's liveness release keeps it bounded and never touches enclosing-block values. A miss ⇒ the `BackendError` naming the value (invalid module, never a silent skip).
- **0-d scalar normalization**: numpy returns bare scalars (`np.bool_`/`np.float64`/…) for 0-d ufunc/matmul/take/slice results; kernels normalize with `np.asarray` to the 0-d ndarray `core.Tensor` requires (no-op for real arrays; dtype never changes — this is shape normalization, not coercion).
- `_module_function_names` is the single touch point for the `ir.Module` accessor API (now finalized).
- `Capabilities.dtypes` includes all `numpy.sctypes` entries (+`bool_`) per the binding contract, including non-numeric "others" dtypes — per-op kernels validate concrete dtype support at run time (no silent coercion).
- Kernel call convention (finalized, binding — see the docstring of `kernels/__init__.py`): `kernel(ctx, op, operands) -> Tensor | tuple[Tensor, ...]`; `ctx` is `KernelContext` (`.bindings`, `.rank_context`, `.module`, `.run_region`, `.compute_output_shapes`, `.resolve_callback`, `.evaluate_shape`). The interpreter validates outputs against `op.results` types (dtype exact; symbolic dims via `ctx.bindings`; None dims unchecked).
- **Random kernels (`kernels/random.py`) implementation notes**: `random_normal` with an f32 output dtype takes the documented f32 Box–Muller fast path (`_normal_kernel` branches on `out_dtype == np.float32`); f64/f16 outputs fall through to the exact f64 chain — never touch the f64/f16 branch (bit-exactness reference for the StableHLO export). `_normal_word_pairs` returns the two (u1, u2) word chains directly (splitmix64: two parallel `_mix3` chains over `iota * 2G`; threefry: one cipher call per pair; philox: interleaved block words truncated to count) — bit-identical to `_words_for(..., 2*count)[0::2]`/`[1::2]` without materializing the interleaved stream. `_mix3` computes IN PLACE (callers pass freshly derived arrays) and no `_MASK64`-style masks are used anywhere: numpy uint32/uint64 arithmetic wraps mod 2^32/2^64 by definition — the threefry2x32 cipher runs end-to-end in uint32 lanes (adds and the 32-bit rotate wrap mod 2^32 by definition; value-identical to the masked uint64-lane form it replaced; 0-d counters run as 1-element array views so the numpy scalar wrap warning never fires). The splitmix64 per-chain scalar `(seed + k*GOLDEN) mod 2^64` bases are folded to Python-int/`np.uint64` scalars before the broadcast adds. `_words` is used ONLY by the value paths (uniform/randint/permutation/multinomial via `_words_for`) whose f64 semantics are pinned bit-exact vs the StableHLO export — mask removal is value-identical, but any further pass-count reduction there must stay provably value-identical.
- `runtime_call` callback resolution goes through `evoxc.ops.constant._get_callback` (via `ctx.resolve_callback`) — artifacts containing `runtime_call` require the same callback registrations at load time.
- **Porting-relevant semantics pins** (empirically verified on numpy 2.4.x): `sort`/`argsort` place NaNs at the END of the ascending result (descending = flip, so NaNs come FIRST), kind = numpy default quicksort unless `stable=True` → `"stable"`; `scatter` = `np.put_along_axis` on a copy — duplicate indices are LAST-WRITE-WINS in assignment order (no accumulate mode), out-of-bounds ⇒ `ShapeError`; `gather` = `np.take` — negative indices wrap, out-of-bounds ⇒ `ShapeError`; `argmax`/`argmin` tie-break to the FIRST occurrence (numpy semantics); `np.maximum`/`np.minimum` propagate NaN (so `clamp` NaN bounds propagate, `min > max` ⇒ all-`max`).
- Collectives dispatch through `dist.context.get_collective_executor()`; the identity `SingleRankCollectiveExecutor` is installed at `numpy/__init__` import time. rank/world_size resolve from the per-execution `RankContext` (`exec_context.py`; override via `NumpyExecutable.run(..., rank_context=...)`).
- `evoxc.dist` never imports backends, so `exec_context.py` and `__init__.py` may import `dist.context` at top level (acyclic). `evoxc.ops`, `evoxc.block`, `evoxc.trace`, `evoxc.persist` stay function-body-lazy.
- `ir.verify` enforces strict use-bookkeeping: erase a spliced `block_call` only after `drop_op_uses` removes its `Use` records (shared `../inline.py`).
