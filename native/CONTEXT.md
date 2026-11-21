# native — the Rust numerical authority of EvoX Compiler

## Intent

`native/` is the Rust core of evoxc and the **single numerical authority** for builtin op schemas, dtype promotion, shape inference, graph transforms (`vectorize`, `vmap`, `jvp`, `vjp`, `grad`) and the runtime session/buffer model.
Python remains the frontend: it executes `defn` bodies and adapts Python objects, but it must never hold a second implementation of builtin inference or of a builtin transform rule.
Language bindings (PyO3 now, others later) are not a second implementation of evoxc.
The design rationale and the proposed end state are `../NATIVE_ARCHITECTURE_PROPOSAL.md`; this file is the binding contract for what exists here.
The implemented Python packages stay authoritative for behavior **until** a code path delegates natively: the port must match `evoxc/ir/inference.py`, `evoxc/transforms/rules.py`, `evoxc/transforms/batching.py` and `evoxc/transforms/autodiff.py` exactly.

## Crate layout, ownership and dependency rules

Cargo workspace at `./Cargo.toml`; crates are `evoxc-core`, `evoxc-compiler`, `evoxc-runtime`, `evoxc-python`.

```
evoxc-core      dtypes, dims, ValueType, attrs, op schemas, graph IR, envelope codec, inference
    ↑                       ↑
evoxc-compiler        evoxc-runtime
 builder, vectorize,      sessions, buffers, submissions,
 AD engines, rules        executables (skeleton)
    ↑                       ↑
    └──────── evoxc-python (PyO3) ────────┘
```

Dependency rules (binding):

- `evoxc-core` depends on no other workspace crate and on **no Python**: allowed third-party deps are `serde`, `serde_json`, `base64` only.
- `evoxc-compiler` depends on `evoxc-core` only.
- `evoxc-runtime` depends on `evoxc-core` only and never on `evoxc-compiler` or PyO3.
- `evoxc-python` is the only crate with `pyo3` (`features = ["extension-module", "abi3-py311"]`) and is the only crate that may know about Python objects.
- Builtin inference, builtin batching rules and builtin AD rules live in Rust only; the Python graph path may not keep a parallel implementation for a capability the native path claims.
- An unimplemented capability fails explicitly (`TransformError` / `Unsupported` diagnostic naming the op), never through a silent Python fallback and never by guessing shapes.
- No `unsafe` outside FFI glue; user errors are `Result` values, never panics (no `unwrap`/`expect` on user input).

## Generated files are authoritative

Two generated tables are committed and are the authority; the generators are dev aids re-run only when the Python registry changes:

- `tools/gen_opdefs.py` reads `evoxc.ir.op_defs.all_opdefs()` and emits `evoxc-core/src/schema_table.rs` (every op: name, category, arity, result count, effect, regions, terminator flag, `has_shape_fn`, attribute schema with JSON-encoded defaults).
- `tools/gen_dtype_promotion.py` emits `evoxc-core/src/dtype_table.rs`: `np.result_type` over **every subset** of the 14 evoxc dtypes as a 16384-entry table indexed by a 14-bit dtype mask.

Both generators run from the repo root with the venv: `source scripts/dev-env.sh && .venv/bin/python native/tools/gen_opdefs.py`.
Generated files carry a `@generated` header; edit the generator, never the generated file.
The generator must reproduce them byte-for-byte (a test asserts the committed file matches a fresh run when the venv is present; otherwise the test is skipped).

**Why a set-based promotion table:** `np.result_type` is *not* associative under pairwise folding (`result_type(int8, uint8, float16) == float16`, while pairwise gives `float32`), but it depends only on the *set* of dtypes (verified exhaustively over all 16384 subsets, including order and duplicate permutations).
`promote_all` therefore folds operand dtypes into a mask and looks the result up; `promote(a, b)` is the same table with two bits.
Never replace this with a hand-written kind/width lattice.

## Wire format: the envelope

`transform` and `infer` exchange **one JSON-able Python dict** (the envelope) per call — never per-node callbacks.
The envelope mirrors the `evoxc-ir` payload documented in `../evoxc/ir/serialize.py` with one deliberate change: constant payloads are **raw little-endian bytes**, not `np.save` blobs.

```
{
  "format": "evoxc-native-ir",
  "version": 1,
  "module": {"name": str, "metadata": {json}},
  "functions": [Function, ...],
  "ops": [Op, ...],
}

Function = {
  "name": str,
  "input_types":  [Type, ...],
  "output_types": [Type, ...],       # informational; validated against the "return" operands
  "metadata": {json},
  "block": {"arguments": [{"id": int, "type": Type}, ...],
            "ops": [{"id": int, "ref": str}, ...]},   # ref = decimal op id (string is authoritative)
}

Op = {"id": int,
      "name": str,
      "operands": [value_id, ...],
      "results": [{"id": value_id, "type": Type}, ...],
      "attributes": {json},
      "regions": [Region, ...],
      "location": {"file": str, "line": int, "col": int, "code_snippet": str | null} | null}

Region = Block | {"blocks": [Block, ...]}     # single-block regions encode as the Block

Type = {"dtype": str, "shape": [Dim, ...]}     # dtype = numpy dtype.name

Dim = {"int": n}
    | {"dim": name, "size": n?}                 # symbolic Dim, optional known size
    | {"expr": {"op": "add"|"sub"|"mul"|"floordiv"|"mod"|"min"|"max",
                "args": [Dim, Dim]}}
    | null                                      # runtime-dynamic dimension
```

Value ids are module-unique and **preserved** by a transform where the value survives (rewritten values get fresh ids; branch (block-argument) ids stay block-local).
Op ids are module-unique; a transform renumbers new ops from a fresh counter above the maximum input id.
`attributes` are plain JSON values; array-valued attributes (notably `constant.value`) encode as
`{"__evoxc_ndarray__": {"dtype": str, "shape": [int, ...], "data_b64": str}}` where `data_b64` is base64 of the array's **C-order raw little-endian bytes** for that dtype.
Tuples that must survive the round trip encode as `{"__evoxc_tuple__": [...]}`; `result_specs` on `runtime_call` / `block_call` / `external_call` is a list of `Type` dicts.
A returned envelope has the same shape plus an optional top-level `"transform"` object with the transform's result metadata (`"output_ids"`, `"input_ids"`, `"kind"`).

Serialization must be deterministic (stable key order for maps, stable op order) so a decode→encode round trip is byte-identical for the same input.

## FFI surface (`evoxc._evoxc_native`)

Built as crate `evoxc-python` → `libevoxc_python.so`; `scripts/build-native.sh` copies it to `evoxc/_evoxc_native.abi3.so` and smoke-tests `version()` through the real interpreter.
The module init symbol is `PyInit__evoxc_native` (the Python module name is `_evoxc_native`).

- `version() -> str` — the crate version.
- `infer(op_name: str, operand_types: list[Type], attributes: dict, result_types: list[Type] | None = None, module: dict | None = None) -> list[Type]`
  - validates the op exists (unknown → `VerificationError`), arity and attribute schema (missing required / wrong type → `VerificationError` with the op name and attribute name);
  - dispatches to the op's shape function with exact parity to `evoxc/ir/inference.py`;
  - `constant` derives its result type from the `value` payload; `call` resolves the callee from `module` (`functions[].name` + `input_types`/`output_types`);
  - op-specific ops (`if`, `runtime_call`, `block_call`, `external_call`; `while` result types) use `result_types` when given; if absent and not derivable → `VerificationError` naming the op;
  - `return` infers zero results.
- `transform(kind: str, module: dict, config: dict) -> dict` for `kind in {"vectorize", "vmap", "grad", "jvp", "vjp"}`; returns the transformed envelope (plus `"transform"` metadata).
- `NativeError` — exception base with `.kind` (`TraceError`, `ShapeError`, `TransformError`, `BackendError`, `PersistenceError`, `DeviceError`, `DTypeError`, `VerificationError`, `EVOXCError`) and `.message`.
  The binding maps the Rust `Error` variant to `kind`; Python code raises the corresponding public evoxc error class from `(kind, message)` so error text and family match the pure-Python paths.

## Transform configs

All transforms take the input envelope and a config dict; they never see Python callables except `extension_rule` (below).
`axes` maps a **value id (string key) or an input position** — the binding uses value ids of the entry function's block arguments.

- `vectorize`: `{"axes": {"<value_id>": axis | null, ...}, "batch_dim_name": "batch" | null}`
  - every tensor input must have an entry; `null` = unmapped (shared);
  - one fresh batch `Dim` per pass, named from `batch_dim_name` (`null` → derive `batch`, `batch_1`, … from the module metadata pass depth);
  - mapped inputs share that single batch extent; their leading axis is the mapped axis; region-bearing ops raise `TransformError`.
- `vmap`: same keys plus `{"out_axes": out_axes}` where `out_axes` is an int (all tensor outputs) or a list `[int | null, ...]` aligned with the function's return operands:
  - an int moves an output's batch dim to that position (a mapped output moves its dim; an unmapped output broadcasts to the batch extent);
  - `null` requires an output proved unmapped (never picks a row).
- `grad`: `{"argnums": [int, ...]}` over flattened tensor inputs (absent → all); single function, single rank-0 real floating output, selected inputs must be float/complex; the result graph returns one gradient per selected index, in selector order.
- `jvp`: `{"tangent_inputs": [int, ...]}` — flattened input indices that get an explicit tangent block argument (the rest use the zero tangent); result function inputs are the original inputs followed by the explicit tangents, and its return operands are the primal outputs followed by one tangent per tensor output.
- `vjp`: `{"cotangent_inputs": [int, ...]}` — flattened **output** indices that get an explicit cotangent block argument; the others must be rank-0 (an in-graph scalar-one cotangent); result function inputs are the original inputs followed by the explicit cotangents, and its return operands are the primal outputs followed by one cotangent per tensor input.

Every transform returns a module whose entry function is the first function and whose return op's operands are the documented output order.
Static values, trees and Python reconstruction metadata stay on the Python side.

## Extension-rule bridge (`extension_rule`)

For an op with no builtin rule, the engine may call `config["extension_rule"]` — a Python callable — **only** when the op is a registered host rule: `block:*`, `external:*` or a `sparse_*` op (and only if the caller supplied the callback).
The callback receives one small neutral request dict:

```
{"kind": "batching" | "jvp" | "vjp",
 "op_name": str, "rule_key": str,
 "attributes": {json},
 "operand_types": [Type, ...], "result_types": [Type, ...],
 "axes": [int | null, ...] | null,                     # batching only: axis per operand
 "tangents": [bool, ...] | null,                       # jvp only: operand carries a tangent
 "tangent_values": [value_id | null, ...] | null,      # jvp only: one per operand, operand order
 "cotangent_type": Type | null,                        # vjp only
 "cotangent_values": [value_id | null, ...] | null}    # vjp only: one per result, result order
```

`tangent_values` / `cotangent_values` are the dual SSA values themselves: `Some(id)` is that operand's tangent (resp. that result's cotangent) as a value of the transform's **output** graph, `null` is the zero tangent / zero cotangent.
Each key is present only on its own kind — absent for batching and for the other kind — with exactly one entry per op operand (`tangent_values`) resp. per op result (`cotangent_values`).
At the entry function's block arguments this is exactly the selector order of `tangent_inputs` / `cotangent_inputs`; for an inner op, operand order resp. result order is the only coherent global contract, and the ids index the rewritten graph, so a rule may pass a tangent straight through.

It returns `None` (no rule → `TransformError` naming the op) or a subgraph response:

```
{"module": Envelope, "entry": str,
 "output_ids": [value_id, ...],            # values of the subgraph to splice in, in order
 "output_axes": [int | null, ...] | null}  # batching only: mappedness of each spliced output
```

The subgraph's entry block arguments are, in order: the op's operands, the op's results (proxy values for `jvp`/`vjp` rules that need the primal results), and then — `jvp` / `vjp` only — the non-zero duals (`jvp`: the `Some` entries of `tangent_values` in operand order, `vjp`: the `Some` entries of `cotangent_values` in result order; a `null` entry contributes no block argument).
A `batching` subgraph therefore takes exactly `operands ++ results`.
The engine inlines the subgraph into the active builder with SSA remapping: operand proxies bind to the rewritten operands, result proxies to the rewritten results, the trailing dual arguments to the values those `Some` entries name; `output_ids` select which values become the rewritten results (batching) or which values are the spliced JVP/VJP contributions.
No other Python callback may occur during a transform; the envelope/GIL crossing happens once per transform (plus once per extension-ruled op).

## Performance rules

- Envelope decode happens once per transform into `Vec`-backed arenas; value ids index arrays directly (a `Vec<Option<T>>` or sparse map only when the id space is sparse) — no `HashMap<String, …>` in a hot loop.
- `infer` allocates one result `Vec`; shape helpers avoid cloning shapes where a slice suffices.
- Numeric ids (`ValueId(u32)`, `OpId(u32)`, `FunctionId(u32)`), not strings, inside the IR.
- Builtin rules run GIL-free (`Python::allow_threads` around native work); only extension-rule calls re-acquire the GIL.
- Keep each Rust source file under ~1000 lines; split inference and rules by op category so independent workers never share a file.

## Test strategy

- Rust unit tests per module, in-file (`#[cfg(test)]`), run with `cargo test --workspace`.
- Inference parity: `evoxc-core` tests cover broadcasting, count folding, axis normalization, promotion table (every pair and a sampled n-ary set), and per-category shape functions ported case-by-case from `../tests/ir/test_inference.py`.
- Envelope: round-trip tests (decode→encode byte-identical), value-id preservation, raw-byte constant payloads, symbolic dim identities, tuple/ndarray attribute tags.
- Transforms: structural tests (op sequence, result types, mappedness) plus numeric equivalence from Python later; every v1 deferral asserts an explicit `TransformError` with the op name.
- `scripts/build-native.sh` is the end-to-end gate: it must build and its smoke test must print the version.
- Tests must not require a GPU, the network, or the Python interpreter (except the generated-table freshness test, which skips without the venv).

## Status (phase 1 — vertical slice complete)

Four crates implemented and green: `evoxc-core` (dtypes + generated promotion table, `Dim`/`DimExpr`, `ValueType`, `AttrValue`, generated 119-op schema table, graph IR, envelope codec, full per-category inference, `verify`), `evoxc-compiler` (`Builder`, `vectorize`/`vmap` with the full builtin batching-rule set, `jvp`/`vjp`/`grad` with the builtin derivative-rule set, extension-rule inlining), `evoxc-runtime` (session/device/buffer/submission/executable skeleton), `evoxc-python` (`version`, `infer`, `transform`, `NativeError`).
`scripts/build-native.sh` builds `evoxc-python` → `libevoxc_python.so`, installs `evoxc/_evoxc_native.abi3.so` and its smoke test prints the version.
Rust workspace suite: **415 passed, 0 failed** (`evoxc-core` 224 + 2 parity, `evoxc-compiler` 114 + 1 transform-parity gate, `evoxc-python` 13, `evoxc-runtime` 61); `cargo clippy --workspace --all-targets --no-deps -- -D warnings` is clean.
Inference parity: **518 cases over 113 ops, 0 mismatches** through the FFI (`evoxc._evoxc_native.infer`), with all 461 successful cases re-derived live from `evoxc.ir.op_defs.opdef(name).shape_fn`.
Transform parity: **61 cases × 122 mode-runs, 0 mismatches** (`vectorize`, `vmap` incl. non-leading `out_axes`, `grad`, `jvp`, `vjp`) comparing native-transformed graphs numerically against the Python transforms; 12 mode-runs are documented deferrals where both engines raise `TransformError`.
Coverage: inference for elementwise/comparison/cast/select/broadcast, reshape/transpose/slice/gather/scatter/concatenate/pad/diagonal/tile/diag, reductions/argmax/argmin/sort/argsort, dot/conv/solve, linalg, random (algorithm-aware), collectives, all 16 sparse ops, and the op-specific cases (`constant`, `call`, `if`, `while`, `runtime_call`, `block_call`, `external_call`).
Batching rules cover elementwise, reductions, structural, dot, linalg aux and constants; AD rules cover the pointwise/unary table, reductions, structural, dot/conv, tril/triu/cumsum/solve, zero rules and `stop_gradient`, with `conv` VJP and the other Python-side deferrals preserved as explicit errors.

## Known Issues / Notes for Agents

- `native/CONTEXT.md` is the binding FFI contract; crate-level `CONTEXT.md` files may refine internals but never change the wire format or the FFI names.
- The Python integration (`evoxc/_native.py`, delegating `evoxc.ir.Builder`/`ops`/transforms to this module) is a **separate workstream**: nothing here edits `../evoxc/`, `../tests/` or `../scripts/`.
- `evoxc-runtime` is deliberately a compiling skeleton with real types and invariants; it must not depend on PyO3 or the compiler and must not contain dead stubs inside the compiler path.
- Device kinds and placement semantics are owned by the Python contract (`../evoxc/CONTEXT.md` "Device kinds"); the runtime skeleton models `cpu` plus the generic device-kind string, never a second enumeration.
- Build products (`native/target/`, `evoxc/_evoxc_native.abi3.so`) are gitignored; commit only sources and generated tables.
- `source scripts/dev-env.sh` is required in every shell: it exports a writable `CARGO_HOME`, the Nix library paths, and `.venv/bin` on `PATH`.
- Parity harnesses (rerunnable; both need `scripts/build-native.sh` first): `native/evoxc-python/verify_infer_parity.py` (FFI inference over the committed fixture) and `native/evoxc-compiler/tests/parity_transform/run_parity.py` (numeric transform equivalence), the latter also gated by `cargo test -p evoxc-compiler --test transform_parity`.
- Known gap: `vmap`/`vectorize` normalize an `out_axes` entry against the output rank `r`, so an unmapped output accepts insertion positions `0..r-1`; `out_axes == r` is an explicit `TransformError` (Python still defers non-leading entries entirely).
- Known gap: transform deferrals inherited from the Python rules — `conv` VJP, `reduce_mean` over a symbolic reduced dim, multi-axis `gather` sweeps, multi-axis/symbolic slice/pad/concatenate VJPs, `solve` with `left_side=false`, mapped-kernel `conv`, `batch_group_count != 1` — all explicit `TransformError`s naming the op.
- Known gap: the committed inference fixture records two layered divergences — `sparse_concatenate#2` (fixture records the bare `shape_fn`'s `ShapeError`; the real pipeline and native infer raise `VerificationError` on arity first) and `random_uniform#2` (the Python authority raises a plain `ValueError`, which has no public error family; native raises `VerificationError` with the identical message).
- Known gap: `envelope::encode` is infallible, so a structurally corrupt program would emit `"type": null` instead of an error; transform outputs are builder-produced, but a validation pass belongs here before any foreign producer uses `encode`.
- Known gap: `Dim::min_of`/`max_of`/`fold` use `expect` on a `Dim::Dynamic` operand (a documented shape-arithmetic invariant); making them fallible needs a signature change rippling through both crates.
- Known gap: `evoxc-runtime` is a skeleton — no drivers or allocators, opaque `DeviceHandle` placeholders, and no `MappedView`/`ForeignExport`/DLPack surface yet.
- Out of this node: the Python packages still use their own implementations; delegating `evoxc.ir`/`evoxc.transforms` to this extension is a separate workstream that must not edit `../evoxc/`, `../tests/` or `../scripts/` from here.

## Routing Table

| Path | Area |
|---|---|
| `./Cargo.toml` | Workspace manifest: members, resolver, shared dependency versions, release profile. |
| `./evoxc-core/` | Dtypes + promotion table, `Dim`/`DimExpr`, `ValueType`, `AttrValue`, op schemas, graph IR (`Program`/`Function`/`Block`/`Op`/`Value`), envelope codec, per-category shape inference. Owns `CONTEXT.md` with the inference contract. |
| `./evoxc-compiler/` | Mutable `Builder` (schema validation + native inference), `vectorize`/`vmap` engine, builtin batching rules, AD engines (`jvp`/`vjp`/`grad`) and builtin derivative rules, extension-rule inlining. Owns `CONTEXT.md` with the token/mappedness and sweep contracts. |
| `./evoxc-runtime/` | Session / device-context / buffer / submission / executable skeleton with real ownership invariants. Owns `CONTEXT.md`. |
| `./evoxc-python/` | PyO3 module `evoxc._evoxc_native`: `version`, `infer`, `transform`, `NativeError`, envelope conversion, extension-rule callback plumbing. Owns `CONTEXT.md`. |
| `./tools/` | Dev-only generators (`gen_opdefs.py`, `gen_dtype_promotion.py`) and parity helpers; never imported by the crates. |
