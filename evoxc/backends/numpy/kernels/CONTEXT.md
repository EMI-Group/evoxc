# evoxc/backends/numpy/kernels — the numpy interpreter's per-op kernels

## Intent

One module per op category, each exposing `register_kernels(table)`; `__init__.py` assembles `KERNEL_TABLE` and `register_all()` is idempotent.
Kernel contract (binding): `kernel(ctx, op, operands) -> core.Tensor | tuple[core.Tensor, ...]`, operands are already-computed `core.Tensor`s in op order, `ctx` is `KernelContext` (`.bindings`, `.rank_context`, `.module`, `.run_region`, `.compute_output_shapes`, `.resolve_callback`, `.evaluate_shape`).
Kernels never validate result dtype/shape — the interpreter does that against `op.results` types.
Kernels must be np-exact: no dtype promotion beyond what the op defines, no accumulation-order change, and the pinned semantics (NaN sorts last ascending, `stable` → `kind="stable"`, scatter last-write-wins, gather negative-index wrap, argmax/argmin first-tie) are load-bearing.

## Hot-path rules (why the code looks like this)

Every kernel pays Python overhead proportional to op count, and the op count of an interpreted generation graph is thousands — so the happy path of a hot kernel does as little Python as possible while the error path keeps its exact wording.
- **Dtype-kind tests are inlined** as `arr.dtype.kind in "<chars>"` against a module-level plain STRING (a single-character C scan, ~25 ns) — never a frozenset (a hash lookup, ~100 ns) and never a `_check_*` call; the `_check_*` helpers are ERROR-path message builders only.
- **No broadcast pre-check on the happy path.** `_check_broadcast` (which called `np.broadcast_shapes`, ~1.2 µs/op) is invoked from the ufunc's `except ValueError` handler, reproducing the identical `core.ShapeError` message; any other `ValueError` is re-raised untouched.
- **Unbound ndarray methods replace module-level numpy wrappers** (`np.ndarray.sum`/`max`/`min`/`mean`/`prod`, `arr.reshape`, `arr.transpose`, `arr.take`, `arr.argsort`, `arr.argmax`/`argmin`, `arr.cumprod`) — identical computation, minus numpy's Python-level dispatch layer (~0.25–1 µs/call). `np.sort` stays a module function because it is the non-mutating form (`arr.sort` sorts in place).
- `Tensor.numpy()` / `.dtype` / `.shape` are grabbed ONCE per operand per kernel; `core` property overhead is not re-paid inside a kernel.
- Static per-rank / per-op data is cached at module level (e.g. `indexing._slice_defaults(rank)`), never rebuilt per call.
- 0-d normalization stays `np.asarray(...)` on the numpy result (numpy returns bare scalars for 0-d ufunc/take/slice results; `np.asarray` is a no-op for real arrays and never changes the dtype).

## Constraints

- Files stay under ~1000 lines; `sparse.py` is the pre-existing exception (1209) and is left alone.
- `KERNEL_TABLE` op names and the `register_kernels(table)` structure are fixed (the interpreter's capability drift net and `register_all()`'s duplicate check depend on them).
- `__init__.py` imports each category module at module scope; no kernel may import `evoxc.ops`/`evoxc.pipeline`/`evoxc.persist` at module level.

## Measured (TensorNEAT generation graphs, `run_benchmark.py --impl new --backend numpy`)

`xor_p500` 140.1 → 125.2 ms/gen (−10.7 %, same-moment control), `custom_pagie_p500` unchanged within noise (~979 ms/gen — that cell is numpy-compute-bound at ~78 µs/op, so its kernel Python share is ~1 %).
Per-change on `xor_p500`: elementwise hot-path alone −8 %, plus indexing/reductions/sorting −3 % more.
These are the numbers behind the code shape above; re-measure with the external TensorNEAT harness rather than trusting them after unrelated changes.

## Routing table

| Path | Area |
|---|---|
| `./__init__.py` | `KERNEL_TABLE`, `dispatch(op_name)`, `register_all()` (idempotent; duplicate keys ⇒ `BackendError`) |
| `./elementwise.py` | arith/activations/comparisons/logical/bitwise/cast/select/broadcast/stop_gradient — `_binary`/`_unary` factories are the hottest kernels in interpreted graphs |
| `./reductions.py` | `reduce_*` family (via `_REDUCE_METHODS`), `argmax`/`argmin`, `cumprod` |
| `./indexing.py` | reshape/transpose/slice/gather/scatter/concatenate/pad/tril/triu/cumsum |
| `./linalg.py` | dot/conv/solve (genuine BLAS work; `_dot` is the largest single tottime on matmul-heavy graphs) |
| `./sorting.py` | sort/argsort (real sorting work — dominates `xor_p500`'s per-op cost) |
| `./structural.py` | tile/flip/roll/diag/nan_to_num |
| `./control_flow.py` | if/while/call recursive region execution |
| `./collective.py` | dist collectives through `dist.context.get_collective_executor()` |
| `./custom.py` | constant/runtime_call/block_call/external_call dispatch |
| `./random.py` | the 6 `random_*` kernels over all 3 algorithms (bit-exactness contract in `../CONTEXT.md`) |
| `./sparse.py` | the 16 sparse kernels (numpy-backend-only; batched, canonical-form validated) |
