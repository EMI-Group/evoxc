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
| `engine` (module) | `engine_available()`, `engine_version()`, `encode_module(module)` (the `evoxc-native-ir` envelope, one `_native_codec.encode_module` call), `compile_module(module, envelope=None)` (an already-encoded envelope is used verbatim; the caller owns the "envelope belongs to this module" invariant), `program_run(program, flat_inputs, resolver=None, native=None)` (a pre-resolved extension module, else resolved with the same explicit error), `program_plan_text/stats/required_ops/output_types`, `unsupported_ops(stats)`, `SUPPORTED_DTYPES`, `BACKEND_NAME`, `ENGINE_VERSION_KEY` (`"evoxc-native"`). |
| `staging_stats()` / `reset_staging_stats()` | TEST SEAM (module-level, `__all__`, deliberately NOT re-exported from `evoxc`): `{"decode", "encode", "compile"}` counters over this backend's single `_decode_payload` / `_encode_payload` / `_compile_program` helpers — `reset` first, then read. |

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
   The custom-op names, the encoded envelope and the compiled program are computed ONCE here (see "Single-decode staging").

`compile(lowered, options)` — validates `lowered.backend == "native"`, resolves the required `block_call`/`external_call`/`runtime_call` names (reused from `lower()` in-process; ONE decode + ONE scan for a disk-loaded program), records `target="cpu"`, `required_custom_ops` = sorted(blocks ∪ external kernels) and `runtime_dependencies = {"numpy": <version>, "evoxc-native": <extension version>}`, plus the non-persisted `artifact.runtime_call_ids` self-description when the module carries `runtime_call` ops (diagnostics only — callbacks resolve lazily at run time and are never serialized). No lowering/engine work.

`load(artifact, device=None, options=None)` — backend-name mismatch ⇒ `PersistenceError`; `device` must be `None` or a CPU `Device` (non-`Device` ⇒ `DeviceError`, non-CPU kind ⇒ `BackendError` naming the kind); every remaining `block_call` needs a registered block AND a registered `native` impl (`_require_block_impls(blocks)` takes the already-collected name tuple — it never re-walks the module), else `PersistenceError` naming the block (portables were already inlined); then the engine program is reused (in-process) or built from the attached module and a `NativeExecutable` returned. Never re-lowers.

**External kernels and `runtime_call` callbacks are deliberately NOT required at load** — the binding persistence contract (`../../CONTEXT.md`, "External kernels"; `../program.py::_validate_required_custom_ops`) keeps both registries lazy: they are never serialized, must be re-established before RUN, and a missing registration raises `core.BackendError` naming it from the resolver. This is the ONE place the backend is more permissive than the objective's "assert external kernels at load" wording; the binding contract wins, and the failure is still explicit (just later).

## Single-decode staging (numpy parity)

`lower()` attaches five PRIVATE, runtime-only objects to the `LoweredProgram` (never declared dataclass fields, so `LoweredProgram.save`/`CompiledArtifact.save` persist nothing of them and the container bytes stay byte-identical): `_native_module` (the live module it verified), `_native_payload` (the exact payload object), `_native_required_ops` (the `(blocks, external, runtime_calls)` triple it scanned once), `_native_envelope` (the one `engine.encode_module` result, a fresh dict per `lower`) and `_native_program = (payload, program)` (the compiled capability-gate program).
`compile()` attaches the same triple plus the (identity-gated) envelope and forwards the program, and `load()` reads them.
EVERY reuse is gated on the identity of the payload object the attachment was built from (`getattr(x, "_native_payload", None) is x.payload`), so a replaced/mutated payload can never reuse a stale module, envelope or program.
Attached-but-not-yet-collected / attached-but-not-yet-encoded cases fall back defensively (one scan / one encode), never silently to a wrong graph.
A `LoweredProgram`/`CompiledArtifact` loaded from disk carries none of them (they are never persisted) and takes the fallback: exactly ONE decode, ONE scan, ONE encode, ONE `native_compile` — never zero, never two.
`engine.encode_module(module)` is the single encode entry point and `engine.compile_module(module, envelope=...)` accepts the pre-computed envelope, so a full in-process `lower → compile → load` chain performs `decode 0, encode 1, compile 1` (pinned by the `staging_stats()` seam, which counts calls through this backend's one `_decode_payload` / `_encode_payload` / `_compile_program` helper).
`lower()` still compiles once as its capability gate — that is the `encode 1, compile 1`, not a redundancy.

## Staging cost

Measurements on a 16-cpu x86-64 box, loadavg ≈4–5, `100-op` = a chain of 100 `evoxc.tanh` ops on `(64,)` f32, `2000-op` = the same ×2000; min/median ms over 7 repeats.

| Stage | 100-op before | 100-op after | 2000-op before | 2000-op after |
|---|---|---|---|---|
| `lower` | 1.426/1.472 | 1.473/1.499 | 27.5/28.4 | 28.4/28.9 |
| `compile` | 0.923/0.950 | 0.002/0.002 | 18.0/18.8 | 0.002/0.002 |
| `load` | 0.918/0.941 | 0.001/0.001 | 18.5/18.7 | 0.001/0.001 |
| **`lower+compile+load` total** | **3.457/3.488** | **1.475/1.499** | **65.9/67.8** | **28.3/28.9** |
| numpy total (control) | 0.785/0.802 | 0.794/0.810 | 15.4/15.7 | 15.5/15.7 |

The two removed deserializations (`ir.deserialize_module` ≈0.93 ms at 100 ops / ≈18.5 ms at 2000 ops, each followed by a full `iter_ops` scan) were the whole of `compile`/`load`; the remaining `lower` cost is deliberately not removed:
`graph.verify` 3.5 ms, `_scan_required_ops` 0.09 ms, `encode_module` 6.3 ms (it verifies the module before emitting), the Rust `native_compile` 6.0 ms, `ir.serialize_module` 11.4 ms (it verifies too), plus ≈2 ms of capability walks and `Signature` recording — at 2000 ops that sums to the measured ≈29 ms.
Those four heavy pieces live in `evoxc/ir`, `evoxc/_native_codec` and the Rust `native/` tree, i.e. OUTSIDE this node: the persisted payload and every verification stay exactly as they are.
A same-order-of-magnitude cost remains for the same reason at 100 ops (0.18 verify + 0.31 encode + 0.29 compile + 0.56 serialize ≈ 1.34 of the 1.50 ms `lower`).

Warm `run` overhead (`evoxc.run`, best-of-300, host tensors, loadavg ≈4): 1-op graph native 7.5 µs vs numpy 6.1 µs (backend executable 5.04 vs 3.63 µs — the raw Rust `NativeProgram.run` alone is ≈3.55 µs, i.e. the engine call is already numpy's whole backend run, and the residual seam is `_validated_inputs` 0.30 + output wrapping + a 0.2 µs `core.Tensor` construction); 100-op graph native 32.0 µs vs numpy 88.2 µs.
Caching the resolved extension module on the executable (`engine.program_run(..., native=...)`, 0.34 µs/call saved) and the identity-keyed input/output spec layout (0.3–0.4 µs/call) took the 1-op pipeline call from 8.05 → 7.5 µs with every validation and error message unchanged.

## Numerical equivalence with the numpy backend (binding)

The two CPU backends are not a bit-exact pair: the contract is exact for some values and a measured tolerance elsewhere.
Measured on this host (numpy 2.5, glibc) by running one graph on both backends with the same inputs:
- **Exact**: every bool/integer path; every float op built from one-rounding IEEE-754 primitives (`add`/`subtract`/`multiply`/`divide`/`sqrt`/`minimum`/`maximum`/`select`/`cast`, reductions of them in the same order, and `power` for the scalar exponents 0/1/2/3/0.5/−1); `random_*` sampling except the documented f32 `normal` 1-ulp window.
- **Last-bit tolerance — transcendental elementwise ops**: `sin`/`cos`/`tan`/`tanh`/`exp`/`log`/`log1p`/`erf`/`gelu` call the PLATFORM libm in f32 and their results equal glibc `sinf`/`cosf`/`expf`/`tanhf` bit-for-bit (verified through ctypes), while numpy uses its own SIMD kernels whose output depends on the CPU dispatch; a cross-backend bit-identity contract cannot exist for them.
  Measured f32 windows over 4000 samples in [−0.9, 0.9]: `sin` 219 differ (≤1 ulp), `cos` 250 (≤1 ulp), `tanh` 1210 (≤1 ulp), `exp` 1558 (≤2 ulp, max abs 2.4e-7).
- **Relative tolerance — f32 `dot`**: `faer` blocks and threads the gemm differently from numpy's OpenBLAS `sgemm`, so every output element reassociates its K-sum.
  Measured 64×96 @ 96×48 with N(0,1) operands: 580/3072 elements differ, max abs 9.5e-6, up to ~5e-5 relative where cancellation leaves a near-zero result; both sides are equally far (9.5e-6) from the f64 reference.
  The parity harness's default f32 tolerance (rtol 1e-5/atol 1e-6) does NOT cover a realistic f32 `dot`; atol 1e-5 does.
- **`power` with an integer exponent 4..=64 (known deviation)**: the engine routes those exponents through a repeated-squaring arm (`native/evoxc-cpu/src/kernels/pow.rs`) while numpy's `npy_pow` calls `powf` for everything outside its 0/1/2/3/0.5/−1 table.
  Measured f32: exponent 21 → 3588/4000 samples differ, up to 13 ulp (1.8e-7 abs); exponent 64 → 3066/4000, up to 40 ulp (5.6e-8 abs, 2.4e-6 relative); exponents 4/5 up to 2 ulp.
  Dropping that arm makes those exponents one `powf` per element and restores bit-identity with numpy.
- **Consequence for callers — trajectory, not per-op correctness**: any algorithm that branches on float comparisons amplifies a last-bit difference.
  An evolutionary selection operator reads `<=`/`==` on fitness values, so a 1-ulp `fit` change flips domination relations, ranks and crowding orders, and the generation trajectory diverges while every op stays inside the windows above.
  Do not read a numpy-vs-native trajectory difference as a staging or version bug without an op-level bisect.

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
- The engine program is compiled at `lower()` (capability gate) and again at `load()` only when the in-process program attachment is absent (a reloaded artifact) — report `stats()["compile_ms"]` when measuring staging cost, and `staging_stats()` for the decode/encode/compile tallies.
- The `run` hot path caches two pure derivations per executable — the resolved extension module and the flattened spec layout (rebuilt when `signature` is REPLACED) — so a `NativeExecutable` resolves `engine.engine_module()` once, not per call. Validation and every error message are unchanged.
- A `random_key_mix` graph fails at `lower()` on the Rust envelope decoder (`VerificationError: attribute "salt": expected an int, got 11400714819323198485` — a u64 value above `i64::MAX`). That is an engine-side int-attribute limit (outside this node; there is no narrowing workaround here — nothing in this package touches attribute encoding), reproducible at the same boundary in every staging chain, in-process or from disk.
- `Graph` mutation warning: like numpy/compiler `lower()`, portable inlining mutates the caller's module, so a `lower()` that fails at the engine gate (step 6) leaves blocks already inlined.

## Test strategy

`../../../tests/backends/` is a sibling (read-only from here; test writes escalate to root) and has no `native`-backend file yet — that is the parent's call.
Validation in this node is done with a stub extension (`evoxc._native._MODULE` replaced by an object exposing `native_compile`) plus the real extension once `native_compile` lands: `get("native")`/`lower`/`load` error shapes without the engine, capability/deferral errors, `run` output validation, and native-vs-numpy parity on the corpus (elementwise+reduction+dot chain, gather/sort, `cond`/`while_loop`/`scan`, a `grad`-transformed graph, an `external_call`, a custom `block_call`, and a dynamic-shape graph).

## Routing table

No child nodes.

| Path | Area |
|---|---|
| `./engine.py` | The ONLY FFI seam: extension lookup/`engine_available`, `encode_module`, `compile_module` (optional pre-computed envelope), `program_run` (optional pre-resolved extension module)/`plan_text`/`stats`/`required_ops`/`output_types`, unsupported-op extraction, `NativeError` translation, `SUPPORTED_DTYPES`. |
| `./resolver.py` | `NativeCallbackResolver` — `call_block` (static_args-aware) / `call_external` / `call_runtime`, host-array normalization and the shared output normalization. |
| `./__init__.py` | `NativeBackend` (staging + capability checks + engine gate + single-decode attachments), `NativeExecutable` (run/output validation, diagnostics, persistence), `native_backend`, `register()`, and the `staging_stats()`/`reset_staging_stats()` test seam with its single `_decode_payload`/`_encode_payload`/`_compile_program` helpers. |
