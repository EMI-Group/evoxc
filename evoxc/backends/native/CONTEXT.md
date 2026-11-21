# evoxc/backends/native — the Rust CPU-engine backend

## Intent

Execute an evoxc `Graph` on the **in-process Rust CPU execution engine** (`evoxc-cpu` under `native/`), reached through the `evoxc._evoxc_native` PyO3 extension.
The backend is a peer of `numpy` (the reference interpreter): the SAME four-stage contract (`lower` → `compile` → `load` → `run`), the SAME `Signature`/`LoweredProgram`/`CompiledArtifact` objects, the SAME structured-I/O wrapping in `pipeline` and the SAME host-data device story.
It is OPTIONAL and lazily activated like the compiler adapters — `import evoxc` never imports the extension, and a missing/broken engine is an actionable `core.BackendError`, never a silent fallback to numpy.

The payload is the serialized EvoXIR module (exactly like numpy), so lowered programs and artifacts stay savable, portable and backend-name-validated; there is NO serialized machine code. A saved `native` executable re-runs `native_compile` from its payload at `load()`.

## FFI contract (binding for both sides — keep in sync with `native/`)

The Python side calls the extension through this surface and nothing else (`engine.py` is the only seam):

```
native_compile(envelope: dict) -> NativeProgram
    # envelope = the existing `evoxc-native-ir` module envelope that
    # `transform()`/`batch_functions()` consume
    # (`evoxc._native_codec.encode_module(graph.module)`).
    # errors raise NativeError with kind
    # VerificationError / ShapeError / DTypeError / BackendError / General
NativeProgram.run(flat_inputs: list[numpy.ndarray], resolver=None) -> list[numpy.ndarray]
NativeProgram.entry() -> str
NativeProgram.plan_text() -> str
NativeProgram.stats() -> dict
    # ops, dispatched_ops, blocks, plan_entries, compile_ms, unsupported ops
NativeProgram.output_types() -> list[dict]     # encoded ValueType dicts
NativeProgram.required_ops() -> dict
    # {"blocks": [...], "external": [...], "runtime_calls": [...]}
```

- `run` / `entry` are REQUIRED (a returned object lacking them is a contract violation surfaced as `core.BackendError`). `plan_text` / `stats` are required by the diagnostics accessors; `required_ops` / `output_types` are optional accessors that are cross-checked when present and skipped when an engine build predates them (a diagnostics gap never blocks staging).
- `NativeError.kind` is translated onto the public `evoxc.core` family by `evoxc._native._translate` — the ONE mapping implementation (message verbatim, chained, unknown kinds re-raised unchanged). The backend reuses it rather than duplicating the table.
- The `stats()` unsupported-op report is read tolerantly (`"unsupported_ops"` / `"unsupported ops"` / `"unsupported"`; entries as strings or `{"op"/"name": ...}` records) so spelling drift cannot silently report "no unsupported ops" — `lower()` fails loudly instead.

**Callback resolver protocol** (the engine calls these on the object passed as `resolver`; one stateless instance per executable):

```
call_block(name, args, static_args=None) -> list[numpy.ndarray]
call_external(name, args)                -> list[numpy.ndarray]
call_runtime(name, args)                 -> list[numpy.ndarray]
```

`name` is the op's static identifier (`block_call`'s `block_name`, `external_call`'s `name`, `runtime_call`'s `callback` id); `args` is the operand list in IR order as host numpy arrays; the return is one array per declared result, in declaration order, which the ENGINE validates against the op's declared result types.
`static_args` carries the `block_call` op's `static_args` attribute (the impl is called `impl(*arrays, **static_args)`; the VALUES are the JSON-safe `{"kind": ..., "value": ...}` payloads recorded on the op — NOT decoded Python values, exactly like the numpy impl contract in `../../../tests/block/CONTEXT.md`, so one impl serves both backends); it may be passed directly or as an attributes mapping containing `static_args`. **An engine build that calls `call_block(name, args)` without `static_args` breaks blocks that declare static kwargs** — the resolver then raises an explicit `core.BackendError` naming the block rather than calling the impl with missing kwargs.
`args` may also be a single array/`core.Tensor`, and 0-d NumPy scalars are coerced (`np.asarray`); anything else is an explicit error.

## API Surface

| Name | Notes |
|---|---|
| `NativeBackend` | `name = "native"`; `Capabilities` below; `KNOWN_OPTIONS` = all four stages empty (v1 takes NO pipeline options — an option meant for a compiler adapter fails loudly here instead of being silently ignored). |
| `NativeExecutable` | Backend executable (satisfies the `Executable` protocol): `backend_name = "native"`, `functions` = `(program.entry(),)`, `device`, `signature`, `artifact`; `run(flat_input_tensors, options=None)`, `synchronize()` (documented no-op — `run` blocks), `save(path)`, classmethod `load(path, device=None)`; diagnostics `plan_text()`, `stats()`, `required_ops()`. |
| `NativeCallbackResolver` | The resolver above; stateless, one per executable. |
| `native_backend` | The registered singleton instance. |
| `register()` | Adapter-convention activation: probes the extension (`engine.engine_module()`), raising `core.BackendError` with build instructions when the extension or its `native_compile` entry point is missing; then registers the singleton (idempotent). Called by `registry.get("native")` via `OPTIONAL_ADAPTERS`. |
| `engine` (module) | `engine_available()`, `engine_version()`, `compile_module(module)`, `program_run/plan_text/stats/required_ops/output_types`, `unsupported_ops(stats)`, `SUPPORTED_DTYPES`, `BACKEND_NAME`, `ENGINE_VERSION_KEY` (`"evoxc-native"`). |

`pipeline` usage is unchanged: `evoxc.build(fn, *specs, backend="native")`, `evoxc.evaluate(..., backend="native")`, `evoxc.lower/compile/load(..., backend="native")`, `EVOXC_BACKEND=native`.
The user-facing `Executable.summary()` inspection API lives in `evoxc/pipeline.py` and reaches the engine through this executable's `plan_text()` / `stats()` (never import that module here).

## Staging contract

`lower(graph, options)` — the binding order (error precedence follows it):

1. `options.validate_options(options, KNOWN_OPTIONS, "native", "lower")`.
2. `graph.verify()` (`core.VerificationError` surfaces as-is).
3. Capability pre-check over the live module: `runtime_call` / `block_call` / collective effects vs `Capabilities`; `sparse_*` ops (category `"sparse"`); every op result and function-signature dtype against `Capabilities.dtypes`.
4. `inline_portables(graph.module, keep_backend_impls="native")` — the SHARED `../inline.py` fixpoint: blocks with a registered `native` impl stay, blocks with only a portable decomposition are spliced in, a block with neither is an explicit `BackendError` naming it (never a cross-backend impl fallback). External-kernel portables are NOT inlined (`external_calls=True` — the engine calls back through `call_external`).
5. Only when step 4 spliced something: `graph.verify()` + repeat step 3 (the module changed).
6. **Engine capability gate**: `native_compile` the module and read `stats()`; every op the engine reports as unsupported becomes one `core.BackendError` naming the ops and pointing at `backend="numpy"`. This is the authoritative gate — the Rust engine, not this package, knows what it implements.
7. `Signature` recorded from the Graph's LIVE attributes, EXACTLY like `NumpyBackend.lower` (`input_tree`, `output_tree`, `input_specs` = `graph.tensor_specs`, `output_specs` from the entry function's result types, input static values, output static values).
8. Payload = `ir.serialize_module(graph.module)` (`LoweredProgram.text()`/`save()`/`load()` therefore work through the same code paths as numpy).

`compile(lowered, options)` — validates `lowered.backend == "native"`, deserializes the payload, scans it for `block_call`/`external_call`/`runtime_call` names, records `target="cpu"`, `required_custom_ops` = sorted(blocks ∪ external kernels) and `runtime_dependencies = {"numpy": <version>, "evoxc-native": <extension version>}`. No lowering/engine work.

`load(artifact, device=None, options=None)` — backend-name mismatch ⇒ `PersistenceError`; `device` must be `None` or a CPU `Device` (non-`Device` ⇒ `DeviceError`, non-CPU kind ⇒ `BackendError` naming the kind); every remaining `block_call` needs a registered block AND a registered `native` impl, else `PersistenceError` naming the block (portables were already inlined); then the engine program is built from the payload and a `NativeExecutable` returned. Never re-lowers.

**External kernels and `runtime_call` callbacks are deliberately NOT required at load** — the binding persistence contract (`../../CONTEXT.md`, "External kernels"; `../program.py::_validate_required_custom_ops`) keeps both registries lazy: they are never serialized, must be re-established before RUN, and a missing registration raises `core.BackendError` naming it from the resolver. This is the ONE place the backend is more permissive than the objective's "assert external kernels at load" wording; the binding contract wins, and the failure is still explicit (just later).

**In-process compile reuse**: `lower()` keeps the compiled program as `lowered._native_program = (payload, program)` and `compile()` forwards it to the artifact as `(artifact.payload, program)`. `load()` reuses it ONLY when that tuple's payload is the identical object (`is`) — a reloaded/replaced payload always recompiles from the module. The hint is never serialized and never trusted across a process boundary; it exists because `lower()` must compile once as the capability gate and the common `build` path would otherwise compile the same graph twice.

## Capability matrix (v1)

| Capability | Value | Notes |
|---|---|---|
| `dynamic_shapes` | `True` | the engine plans over symbolic dims bound at run time (the exporter-style per-op restrictions do not apply — there is no MLIR text hop). |
| `dtypes` | the 14 evoxc dtypes (`bool`, int8/16/32/64, uint8/16/32/64, float16/32/64, complex64/128) | an op result outside the set is a `BackendError` naming op + dtype. |
| `collectives` | `True` | single-rank identity semantics (the process-wide `dist.context` executor contract installed by `evoxc.backends.numpy`; nothing re-installed here). |
| `runtime_calls` | `True` | Python callbacks execute synchronously at the op's position through `call_runtime` — a documented sync point, exactly like numpy. |
| `external_calls` | `True` | host-mode kernels through `call_external` (`evoxc.external` `native` slot → default slot). |
| `custom_blocks` | `True` | registered `native` impls through `call_block`; portable decompositions inlined at `lower()`. |
| `async_collectives` | `False` | execution is synchronous. |
| `sparse_ops` | `False` | v1 deferral, explicit. |
| devices | `cpu` only | host numpy data; `Device("cpu", 0)`. |

## Explicit deferrals (always an explicit error, never a silent fallback)

- **Sparse ops** — `Capabilities.sparse_ops=False`; the pre-check names the op and suggests `backend="numpy"` (sparse is numpy-backend-only by contract).
- **Non-CPU devices** — `load()` rejects a non-CPU `Device`; `run()` rejects a non-CPU input tensor with `DeviceError` + the explicit `.to(Device("cpu", 0))` remedy (the pipeline already enforces this at the run boundary — defence in depth, no implicit transfer).
- **Any op the Rust engine does not implement** — reported by the engine at `lower()` as a `BackendError` listing the ops.
- **`device_resident=True` external-kernel slots** — the resolver rejects them explicitly (host arrays cannot honor device-resident semantics).
- **Blocks whose impl exists only for another backend** — `inline_portables(..., keep_backend_impls="native")` requires a `native` impl or a portable, and `load()` re-checks it.
- **Pipeline options** — none in v1 (`KNOWN_OPTIONS` all-empty): a compiler option passed here fails loudly rather than being ignored.

## Import acyclicity

Top-level imports: `evoxc.core`, `evoxc.ir`, this backend's siblings (`..backend`, `..program`, `..registry`, `..options`, `..inline`, `..external_validate`), numpy.
`evoxc.block` / `evoxc.external` / `evoxc.ops` only inside function bodies; `evoxc.persist` only inside `save`/`load`; `evoxc._native` / `evoxc._native_codec` only inside `engine.py` functions (so importing this package never loads the extension); never `evoxc.pipeline`.
`registry.OPTIONAL_ADAPTERS["native"]` is the only import path — nothing imports this package at `evoxc`/`evoxc.backends` import time.

## Notes for agents

- `NativeExecutable.run` is synchronous and returns host `core.Tensor`s (`Device("cpu", 0)`), so `synchronize()` is a documented no-op and a timing measurement is a plain `run` wall-clock.
- One executable is not required to be thread-safe for concurrent `run` calls; the resolver is stateless, but the engine's internal buffers are its own contract.
- The engine program is compiled at `lower()` (capability gate) and again at `load()` only when the in-process hint is absent (a reloaded artifact) — report `stats()["compile_ms"]` when measuring staging cost.
- `Graph` mutation warning: like numpy/compiler `lower()`, portable inlining mutates the caller's module, so a `lower()` that fails at the engine gate (step 6) leaves blocks already inlined.

## Test strategy

`../../../tests/backends/` is a sibling (read-only from here; test writes escalate to root) and has no `native`-backend file yet — that is the parent's call.
Validation in this node is done with a stub extension (`evoxc._native._MODULE` replaced by an object exposing `native_compile`) plus the real extension once `native_compile` lands: `get("native")`/`lower`/`load` error shapes without the engine, capability/deferral errors, `run` output validation, and native-vs-numpy parity on the corpus (elementwise+reduction+dot chain, gather/sort, `cond`/`while_loop`/`scan`, a `grad`-transformed graph, an `external_call`, a custom `block_call`, and a dynamic-shape graph).

## Routing table

No child nodes.

| Path | Area |
|---|---|
| `./engine.py` | The ONLY FFI seam: extension lookup/`engine_available`, `compile_module`, `program_run`/`plan_text`/`stats`/`required_ops`/`output_types`, unsupported-op extraction, `NativeError` translation, `SUPPORTED_DTYPES`. |
| `./resolver.py` | `NativeCallbackResolver` — `call_block` (static_args-aware) / `call_external` / `call_runtime`, host-array normalization and the shared output normalization. |
| `./__init__.py` | `NativeBackend` (staging + capability checks + engine gate), `NativeExecutable` (run/output validation, diagnostics, persistence), `native_backend`, `register()`. |
