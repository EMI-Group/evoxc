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

The codec exchanges **one JSON-able Python dict** (the envelope) per call — never per-node callbacks: `transform` and every `batch_functions` program take one, and `infer` takes one as its optional callee-signature `module`.
`evoxc-core::envelope` (`decode`/`encode`) is the authority; `evoxc-core/src/wire.rs` owns the dim/shape/type/array conversions and `evoxc-core/src/attr.rs` the typed attribute maps.
The shape mirrors the stored `evoxc-ir` payload of `../evoxc/ir/serialize.py` (`"format": "evoxc-ir"`, a `constants` table of `np.save` bytes, a canonical `sha256`) with three deliberate differences: the discriminator is `"evoxc-native-ir"`, array payloads are inline raw little-endian bytes, and there is neither a `constants` table nor a `sha256` (the envelope is an in-process value, not a stored file).
The top level is a FLAT module: one `functions` array plus one module-wide `ops` table, and blocks/regions reference ops by id.

```text
{
  "format": "evoxc-native-ir",                    # required, exact string
  "version": 1,                                   # required, exact JSON integer 1
  "module": {"name": str, "metadata": {json}},    # name required; metadata optional (missing/null -> {})
  "functions": [Function, ...],                   # required; vector order is function order
  "ops": [Op, ...],                               # required; ONE flat module-wide op table, any order
}

Function = {
  "name": str,                                    # required, unique in the module
  "input_types":  [Type, ...],                    # required
  "output_types": [Type, ...],                    # required; informational, cross-checked against the "return" operands
  "metadata": {json},                             # optional; missing/null -> {}; any JSON value is kept verbatim
  "block": Region,                                # required body region
}

Block = {
  "arguments": [{"id": value_id, "type": Type}, ...],   # required
  "ops": [{"id": op_id, "ref": str(op_id)}, ...],       # required; the decimal STRING ref is authoritative
}

Op = {
  "id": op_id,                                    # required, non-negative u32, module-unique
  "name": str,                                    # required, must exist in the generated op schema registry
  "operands": [value_id, ...],                    # required
  "results": [{"id": value_id, "type": Type}, ...],     # required
  "attributes": {json},                           # optional; missing/null -> {}; schema-validated, defaults NOT filled
  "regions": [Region, ...],                       # optional; missing/null -> no regions
  "location": {"file": str, "line": int, "col": int, "code_snippet": str | null} | null,   # optional
}

Region = Block | {"blocks": [Block, ...]}   # a single-block region IS the block object; {"blocks": [...]} is the non-empty multi-block form

Type = {"dtype": str, "shape": [Dim, ...]}  # dtype = numpy dtype.name (the 14 evoxc dtypes); an absent shape is rank 0

Dim = {"int": n}
    | {"dim": name} | {"dim": name, "size": n}      # named symbolic dim, optional known size
    | {"expr": {"op": "add"|"sub"|"mul"|"floordiv"|"mod"|"min"|"max", "args": [Dim, Dim]}}
    | null                                          # runtime-dynamic dimension
```

**Discriminator and top level.** `decode` requires a JSON object whose `format` is exactly `"evoxc-native-ir"` and whose `version` is exactly the JSON integer `1`.
A missing or different `format`/`version` (a wrong string, a JSON float, any other number) is a `PersistenceError` naming what was seen.
`module` must be an object with a string `name`; its `metadata` is optional (missing or `null` → `{}`) and a non-object value is a `VerificationError`.
`functions` and `ops` must both be present and be lists.
Extra keys are ignored at every level (top level, module, function, op, location, array payload), so a payload carrying a foreign key still decodes.
Nothing requires a function to exist: an empty module (empty `functions` and `ops`) decodes.

**The flat op table.** Every op lives in the single module-wide `ops` array; a block's op entries reference it by the decimal-string `ref` (authoritative), and an optional numeric `id` must equal it.
Decode sorts the table by op id once and binary-searches it, so array order carries no meaning and ids need not be dense.
A duplicate op id, an unparseable `ref`, a `ref` with no table entry, a second block entry referencing the same op, and a table entry no block references are each a `VerificationError`.
Regions are decoded lazily through that same table, so an op is built at most once and an op reachable from inside its own region is a `VerificationError` (region cycle).
An op `name` missing from the schema registry is a `VerificationError`.

**Functions and blocks.** Function names are unique in the module, and `input_types`/`output_types` are required lists of `Type`.
`output_types` is informational only, but at the end of decode it must equal the types of the entry block's `return` operands; a body without a `return` terminator or a mismatch is a `VerificationError` naming the function.
A block's `arguments` (`{"id", "type"}`) and each op's `results` are the ONLY value definitions, and every duplicate value id is a `VerificationError`.
Block ids never appear on the wire: decode allocates a fresh `BlockId` per decoded block in decode order, and a region is the ordered list of its blocks (entry first).
Every operand must resolve to a defined value once the whole module is decoded, otherwise a `VerificationError` naming the op and the value.
Decode never re-infers: `results[].type` is taken verbatim from the wire, so a `while`/`if` result type is whatever the payload says.

**Types and dims.** `Type` is `{"dtype", "shape"}` with `dtype` a required numpy `dtype.name` string for one of the 14 evoxc dtypes (an unknown name is a `DTypeError`) and `shape` optional (absent = rank 0).
A dim is `{"int": n}`, `{"dim": name}` optionally with `"size": n`, `{"expr": {"op", "args": [left, right]}}` (exactly two args; an unknown op name is a `ShapeError`), or `null` for a runtime-dynamic dim.
A non-object/non-null dim, and a dim object with none of `int`/`dim`/`expr`, are `VerificationError`s.
Symbolic identity is structural: `{"dim": "B"}` and `{"dim": "B", "size": 4}` are different dims, and decode cross-checks a named dim against nothing.

**Attributes and payloads.** `attributes` values are typed by the op's schema: an unknown attribute name, a missing required attribute, or a value that does not match the declared tag is a `VerificationError`.
The declared tags are `bool`, `int`, `float`, `str`, `dtype`, `ints`, `floats`, `strs`, `nested_ints`, `shape`, `ndarray` and `any`; the per-tag decode rules live in `./evoxc-core/CONTEXT.md` "Envelope contract".
JSON `null` is legal for every declared tag and decodes to an explicit null value that re-encodes as `null`.
Defaults are NOT filled during decode: an absent optional attribute stays absent and the builder/inference layer fills it.
An `ndarray` payload is `{"__evoxc_ndarray__": {"dtype": str, "shape": [int, ...], "data_b64": str}}` where `data_b64` is base64 of the array's **C-order raw little-endian bytes** for that dtype; the tag is optional on decode (a bare `{"dtype", "shape", "data_b64"}` object decodes identically) and is always emitted on encode.
A payload shape accepts plain JSON ints or `{"int": n}` entries; a symbolic or `null` dim and a negative extent are `ShapeError`s, and a byte count that does not equal element count × dtype itemsize (or invalid base64) is a `VerificationError`.
A tuple is `{"__evoxc_tuple__": [...]}`: it decodes by the DECLARED tag (sequence tags take its members elementwise, `any` keeps the tuple and re-emits the marker, a scalar tag rejects it), and typed sequences re-encode as plain lists so the marker survives only under `any`.
`result_specs` on `runtime_call` / `block_call` / `external_call` is a required `any` attribute carried as a list of `Type` dicts.

**Ids.** Op ids and value ids are module-unique, appear exactly as written, and are never renumbered by the codec; decode reseeds the allocators above the highest op/value/block id it saw so later construction cannot collide.
A transform never mutates its input module: it builds a fresh `Program` (or, for `vmap`, resumes the `vectorize` output) and allocates every value and op id from that program's own allocator, so a transformed or merged envelope shares no ids with its inputs (`"transform".output_ids` / `"input_ids"` name the new ones).
Output ids are therefore allocated from 0 upward for a fresh module (`vectorize`, `grad`, `jvp`, `vjp`, and the `batch_functions` merge) and continue above the `vectorize` output's maximum for `vmap`.
Block ids are internal in both directions.

**Determinism and limits.** Encode is a pure function of the `Program`: ops are emitted ascending by id, functions in vector order, region blocks in block order, and every object is a `serde_json::Map` (a `BTreeMap` — the crate enables no `preserve_order`), so keys serialize in ascending key order and nothing depends on insertion order.
A second `encode(decode(e))` is byte-identical to the first (test-asserted), as is encoding the same `Program` twice.
`encode` is infallible: a value id missing from the arena is emitted as `"type": null` and fails on the next decode, which is the one malformed payload it can produce.
Decode validates structure and ids only — no arity, region-count, SSA-dominance or result-type/schema agreement check, and no call to `verify` — so a decoded-but-nonsensical module is `verify`'s job.
A returned transform envelope has the same shape plus a top-level `"transform"` object (`"kind"`, `"output_ids"`, `"input_ids"`, `"batch_dim"` — `null` for the AD transforms) added by the binding after encoding; `decode` ignores it.

## FFI surface (`evoxc._evoxc_native`)

Built as crate `evoxc-python` → `libevoxc_python.so`; `scripts/build-native.sh` copies it to `evoxc/_evoxc_native.abi3.so` and smoke-tests `version()` through the real interpreter.
The module init symbol is `PyInit__evoxc_native` (the Python module name is `_evoxc_native`).

- `version() -> str` — the crate version.
- `infer(op_name: str, operand_types: list[Type], attributes: dict, result_types: list[Type] | None = None, module: dict | None = None) -> list[Type]`
  - validates the op exists (unknown → `VerificationError`), arity and attribute schema (missing required / wrong type → `VerificationError` with the op name and attribute name);
  - dispatches to the op's shape function with exact parity to `evoxc/ir/inference.py`;
  - `constant` derives its result type from the `value` payload; `call` resolves the callee from `module` (`functions[].name` + `input_types`/`output_types`);
  - `module` is either a full `evoxc-native-ir` envelope or a signature-only context `{"functions": [[name, [output types]], ...]}` (also `[name, [input types], [output types]]` and `{"name", "output_types", "input_types"?}` entries) synthesized into a body-less `Program` named `signature-context`, so `call` infers without a full module (malformed shapes are `TypeError`, never a silent no-module fallback);
  - op-specific ops (`if`, `runtime_call`, `block_call`, `external_call`; `while` result types) use `result_types` when given; if absent and not derivable → `VerificationError` naming the op;
  - `return` infers zero results.
- `transform(kind: str, module: dict, config: dict) -> dict` for `kind in {"vectorize", "vmap", "grad", "jvp", "vjp"}`; returns the transformed envelope (plus `"transform"` metadata).
- `batch_functions(programs: list[dict], config: dict) -> dict` — decodes one `evoxc-native-ir` envelope per program and merges them with `evoxc-compiler`'s `batch_functions` engine; returns `{"module": <merged envelope>, "batch_report": {12 keys}}`.
  Config keys (Python spellings): `axes` (int | null | per-input list, default `0`), `out_axes` (int | null | per-output list, default `0`), `switch_ops` (`null` | bool | category/op group list), `bands` (`null` | int K>=1 | `"auto"`), `alignment` (`null` | `"variant"` | `"occurrence"` | `"auto"`); unknown keys/values are explicit errors.
  The merge runs with the same `extension_rule` bridge `transform` uses, so `block:*` / `external:*` / `sparse_*` populations merge; without a bridge they raise the documented pre-flight message.
- `batching_rule_keys() -> list[str]` — read-only introspection: the sorted rule keys of the compiled-in builtin batching registry (`evoxc-compiler::batching::builtin_rule_keys`), i.e. exactly the ops the native engines can map, so a caller checks batchability against the engine instead of a Python-side mirror; host rules (`block:*` / `external:*` / `sparse_*`) are never listed.
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
 "output_ids": [value_id | null, ...],     # values of the subgraph to splice in, in order
 "output_axes": [int | null, ...] | null}  # batching only: mappedness of each spliced output
```

The subgraph's entry block arguments are, in order: the op's operands, the op's results (proxy values for `jvp`/`vjp` rules that need the primal results), and then — `jvp` / `vjp` only — the non-zero duals (`jvp`: the `Some` entries of `tangent_values` in operand order, `vjp`: the `Some` entries of `cotangent_values` in result order; a `null` entry contributes no block argument).
A `batching` subgraph therefore takes exactly `operands ++ results`.
The engine inlines the subgraph into the active builder with SSA remapping: operand proxies bind to the rewritten operands, result proxies to the rewritten results, the trailing dual arguments to the values those `Some` entries name; `output_ids` select which values become the rewritten results (batching) or which values are the spliced JVP/VJP contributions.
An `output_ids` entry of `null` means that contribution is structurally zero (`Dual::ZeroTangent`), with no op spliced for it; it is legal only for `jvp`/`vjp` responses, and a batching response with a `null` entry is an explicit `TransformError` naming the op and the output index.
The engine materializes such a zero (`multiply` by a zero constant of the primal's own dtype) only where a zero dual reaches a graph output, so a null that feeds only downstream rules stays structurally zero — identical numerics to a rule that returned an explicit zeros value, with fewer ops.
Because `inline_function` ignores the subgraph's terminator, an AD response subgraph may `return` only its non-zero contributions, and its declared `output_types` must match those return operands.
The Python bridge sends `None` for every `autodiff.ZeroTangent`/`None` dual slot and omits the `autodiff._materialize` zero values, so the scratch subgraph returns only the non-zero contributions.
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

Four crates implemented and green: `evoxc-core` (dtypes + generated promotion table, `Dim`/`DimExpr`, `ValueType`, `AttrValue`, generated 119-op schema table, graph IR, envelope codec, full per-category inference, `verify`), `evoxc-compiler` (`Builder`, `vectorize`/`vmap` with the full builtin batching-rule set, `jvp`/`vjp`/`grad` with the builtin derivative-rule set, extension-rule inlining including the dual SSA values, and the native `batch_functions` merge engine), `evoxc-runtime` (session/device/buffer/submission/executable skeleton), `evoxc-python` (`version`, `infer`, `transform`, `batch_functions`, `NativeError`).
`scripts/build-native.sh` builds `evoxc-python` → `libevoxc_python.so`, installs `evoxc/_evoxc_native.abi3.so` and its smoke test prints the version.
Rust workspace suite: **499 passed, 0 failed** (`evoxc-core` 224 + 2 parity, `evoxc-compiler` 134 in-file + 30 `batch_functions.rs` + 13 `batch_functions_bands.rs` + 6 `batch_functions_extension.rs` + the `batch_functions_parity` and `transform_parity` cargo gates, `evoxc-python` 27, `evoxc-runtime` 61); `cargo clippy --workspace --all-targets --no-deps -- -D warnings` is clean.
The `tests/transforms` + `tests/ir` Python suites run against the rebuilt extension: 1333 passed, 3 skipped, 1 xfailed.
Inference parity: **529 cases over 113 ops, 0 mismatches** through the FFI (`evoxc._evoxc_native.infer`), with every successful case re-derived live from `evoxc.ir.op_defs.opdef(name).shape_fn`.
Transform parity: **61 cases × 122 mode-runs, 0 mismatches** (`vectorize`, `vmap` incl. non-leading `out_axes`, `grad`, `jvp`, `vjp`) comparing native-transformed graphs numerically against the Python transforms; 12 mode-runs are documented deferrals where both engines raise `TransformError`.
`batch_functions` parity: **25 populations, 87 checks, 0 mismatches** through `evoxc._evoxc_native.batch_functions` against `evoxc.batch_functions` (structural envelope comparison with id remapping, leaf-by-leaf numpy numerics, the exact 12-key `batch_report`), including all six documented deferrals.
Coverage: inference for elementwise/comparison/cast/select/broadcast, reshape/transpose/slice/gather/scatter/concatenate/pad/diagonal/tile/diag, reductions/argmax/argmin/sort/argsort, dot/conv/solve, linalg, random (algorithm-aware), collectives, all 16 sparse ops, and the op-specific cases (`constant`, `call`, `if`, `while`, `runtime_call`, `block_call`, `external_call`).
Batching rules cover elementwise, reductions, structural, dot, linalg aux and constants; AD rules cover the pointwise/unary table, reductions, structural, dot/conv, tril/triu/cumsum/solve, zero rules and `stop_gradient`, with `conv` VJP and the other Python-side deferrals preserved as explicit errors.

## Known Issues / Notes for Agents

- `native/CONTEXT.md` is the binding FFI contract; crate-level `CONTEXT.md` files may refine internals but never change the wire format or the FFI names.
- The Python integration (`evoxc/_native.py`, delegating `evoxc.ir.Builder`/`ops`/transforms to this module) is a **separate workstream**: nothing here edits `../evoxc/`, `../tests/` or `../scripts/`.
- `evoxc-runtime` is deliberately a compiling skeleton with real types and invariants; it must not depend on PyO3 or the compiler and must not contain dead stubs inside the compiler path.
- Device kinds and placement semantics are owned by the Python contract (`../evoxc/CONTEXT.md` "Device kinds"); the runtime skeleton models `cpu` plus the generic device-kind string, never a second enumeration.
- Build products (`native/target/`, `evoxc/_evoxc_native.abi3.so`) are gitignored; commit only sources and generated tables.
- `source scripts/dev-env.sh` is required in every shell: it exports a writable `CARGO_HOME`, the Nix library paths, and `.venv/bin` on `PATH`.
- Parity harnesses (rerunnable; all need `scripts/build-native.sh` first and a `.venv`): `native/evoxc-python/verify_infer_parity.py` (FFI inference over the committed fixture + the signature-only module forms), `native/evoxc-compiler/tests/parity_transform/run_parity.py` (numeric transform equivalence) and `native/evoxc-compiler/tests/parity_transform/run_batch_parity.py` (the `batch_functions` merge), the latter two also gated by `cargo test -p evoxc-compiler --test transform_parity` / `--test batch_functions_parity`.
- Known gap: `vmap`/`vectorize` normalize an `out_axes` entry against the output rank `r`, so an unmapped output accepts insertion positions `0..r-1`; `out_axes == r` is an explicit `TransformError` (Python still defers non-leading entries entirely).
- Known gap: transform deferrals inherited from the Python rules — `conv` VJP, `reduce_mean` over a symbolic reduced dim, multi-axis `gather` sweeps, multi-axis/symbolic slice/pad/concatenate VJPs, `solve` with `left_side=false`, mapped-kernel `conv`, `batch_group_count != 1` — all explicit `TransformError`s naming the op.
- Known gap: the committed inference fixture records two layered divergences — `sparse_concatenate#2` (fixture records the bare `shape_fn`'s `ShapeError`; the real pipeline and native infer raise `VerificationError` on arity first) and `random_uniform#2` (the Python authority raises a plain `ValueError`, which has no public error family; native raises `VerificationError` with the identical message).
- Known gap: `envelope::encode` is infallible, so a structurally corrupt program would emit `"type": null` instead of an error; transform outputs are builder-produced, but a validation pass belongs here before any foreign producer uses `encode`.
- Known gap: `Dim::min_of`/`max_of`/`fold` use `expect` on a `Dim::Dynamic` operand (a documented shape-arithmetic invariant); making them fallible needs a signature change rippling through both crates.
- Known gap: `evoxc-runtime` is a skeleton — no drivers or allocators, opaque `DeviceHandle` placeholders, and no `MappedView`/`ForeignExport`/DLPack surface yet.
- Cost note: the native seam is cheap and linear (envelope decode ~0.46 ms / encode ~0.91 ms at 752 ops, see `./evoxc-core/CONTEXT.md` "Decode/encode cost model"; the outbound conversion shares immutable leaf objects — strings, ints, per-type and per-location records — within one call, see `./evoxc-python/CONTEXT.md` "Outbound object sharing"), yet the native `grad`/`vjp` path measures **1.22x** (752 ops) and **1.27x** (2252 ops) the old pure-Python sweep end to end. The gap is NOT the Rust work: at 752 ops the sweep is ~6 ms of a 34.6 ms path while the Python side's own `encode_module` + `decode_module` are ~22.5 ms (decode alone 20.3 ms; ~59 % of the path), and with conversion free the floor is exactly 1.10x. Closing it needs the Python side to consume a decoded module instead of re-walking an envelope; no native-side change can reach parity with the old engine alone.
- `evoxc-compiler/src/batch_functions/` is deliberately Python-free: it takes `Program`s and typed config enums and returns a `Program` + `BatchReport`; pytrees, static values, device placement, source locations and the registered-static-output-node check stay in the FFI/frontend layer. The FFI must pass the extension bridge (the engine only merges `block:*`/`external:*`/`sparse_*` populations with one) — see `./evoxc-python/CONTEXT.md`.
- Out of this node: `evoxc.ir`/`evoxc.transforms` delegate their graph rewrite to this extension; changes to those Python modules are a separate workstream that must not edit `../evoxc/`, `../tests/` or `../scripts/` from here.
- Known gap: the Python bridge (`evoxc/transforms/_native_rules.py::_derivative_response`) still materializes zero duals with `autodiff._materialize` and sends real ids; the nullable `output_ids` contract above is a capability this engine exposes for the Python side to adopt (send `None`, return only the non-zero contributions).

## Routing Table

| Path | Area |
|---|---|
| `./Cargo.toml` | Workspace manifest: members, resolver, shared dependency versions, release profile. |
| `./evoxc-core/` | Dtypes + promotion table, `Dim`/`DimExpr`, `ValueType`, `AttrValue`, op schemas, graph IR (`Program`/`Function`/`Block`/`Op`/`Value`), envelope codec, per-category shape inference. Owns `CONTEXT.md` with the inference contract. |
| `./evoxc-compiler/` | Mutable `Builder` (schema validation + native inference), `vectorize`/`vmap` engine, builtin batching rules, AD engines (`jvp`/`vjp`/`grad`) and builtin derivative rules, extension-rule inlining, and the native `batch_functions` merge engine (`src/batch_functions/`: skeleton alignment, variant switching, muxing, bands cost model, `BatchReport`). Owns `CONTEXT.md` with the token/mappedness, sweep and merge contracts. |
| `./evoxc-runtime/` | Session / device-context / buffer / submission / executable skeleton with real ownership invariants. Owns `CONTEXT.md`. |
| `./evoxc-python/` | PyO3 module `evoxc._evoxc_native`: `version`, `infer` (envelope or signature-only context), `transform`, `batch_functions`, `NativeError`, envelope conversion, extension-rule callback plumbing. Owns `CONTEXT.md`. |
| `./tools/` | Dev-only generators (`gen_opdefs.py`, `gen_dtype_promotion.py`) and parity helpers; never imported by the crates. |
