# native — Rust-first evoxc core

## Intent

The native implementation of evoxc's numerical authority.
This directory holds the Rust workspace that owns the semantics the public Python package delegates to: dtypes and dtype promotion, symbolic shape arithmetic and inference, op schemas and attribute contracts, graph construction/verification, the batching/vectorization engine (`vectorize`/`vmap`), the automatic-differentiation engines (`jvp`/`vjp`/`grad`), and (in later phases) lowering/export, canonical persistence and the runtime session/buffer model.

Python remains the frontend host: it executes Python `defn` bodies, owns pytrees and the public `Graph`/`Defn`/`TensorSpec` objects, and hosts the NumPy reference interpreter and the compiler adapters.
The native core is not a second implementation of evoxc; it is the only implementation of the semantics above.
Python modules that used to implement those semantics delegate to this core (see `../evoxc/CONTEXT.md`).

Binding design document: [`../NATIVE_ARCHITECTURE_PROPOSAL.md`](../NATIVE_ARCHITECTURE_PROPOSAL.md).
The envelope and FFI contract below are the implemented binding between that design and this repository.

## Why a bulk envelope, not per-node handles (v1)

The proposal's target is native graph handles held by the bindings.
The v1 binding takes one bulk conversion per stage boundary instead: Python's `ir.Module` is a host-side object model that many tests and adapters inspect directly, so replacing it with opaque handles would be a separate, larger migration.
A bulk conversion is not serialization: nothing is written to bytes or disk; the envelope is a private in-process PyO3 data structure, and the cost is one O(nodes) walk per transform/lower call (the transforms themselves are also O(nodes)).
This keeps the single-authority property that matters: there is one implementation of inference, batching rules, derivative rules and transforms, and it is Rust.

## Workspace layout and dependency DAG

```text
native/
  Cargo.toml                 workspace
  evoxc-core/                dtypes, dims, shapes, attrs, op schema registry,
                             inference, graph IR, verification, envelope types
  evoxc-compiler/            Builder, vectorize engine + batching rules,
                             AD engines + jvp/vjp rules, (later) lowering
  evoxc-runtime/             session/buffer/submission model, (later) DLPack
  evoxc-python/              the PyO3 module `evoxc._evoxc_native`
  tools/                     development codegen/fixture helpers (not shipped)
```

Dependency direction is strict: `evoxc-core` ← `evoxc-compiler` ← `evoxc-python`; `evoxc-runtime` depends on core only; the compiler never depends on the runtime and vice versa.
`evoxc-python` is the only crate that depends on PyO3; nothing in core/compiler/runtime may mention Python types.

## Binding contract: the envelope

`evoxc/_native_codec.py` encodes a Python `ir.Module` into the neutral envelope and decodes one back.
The envelope is a private, versioned in-process format (`"format": 1`); it is **not** the persistence format (that stays `evoxc/persist` for v1).

```python
# ValueType
{"dtype": "float32", "shape": [dim, ...]}

# dim: int | None | ["dim", name, size_or_None] | ["expr", op, left, right]
#      op ∈ add|sub|mul|floordiv|mod|min|max ; left/right are dims recursively
#      None is a runtime-dynamic dimension

# attribute value (tagged only when JSON cannot represent it unambiguously)
#   bool | int | float | str | None            -> the Python value
#   tuple                                      -> {"__evoxc__": "tuple", "items": [...]}
#   list                                       -> {"__evoxc__": "list", "items": [...]}
#   dtype name                                 -> {"__evoxc__": "dtype", "name": "float32"}
#   Dim                                        -> {"__evoxc__": "dim", "name": ..., "size": ...}
#   DimExpr                                    -> {"__evoxc__": "expr", "op": ..., "left": ..., "right": ...}
#   numpy.ndarray                              -> reference into the module's `constants` table
#                                                 {"__evoxc__": "const_ref", "index": i}
#   dict                                       -> {"__evoxc__": "dict", "items": [[key, value], ...]}

# module
{
  "format": 1,
  "name": "main",
  "functions": [
    {
      "name": "main",
      "input_types": [type, ...],
      "blocks": [block, ...],          # entry block first; v1 single-block bodies
    },
  ],
  "constants": [                       # ndarray payload table, referenced by index
    {"dtype": "float32", "shape": [2, 3], "data": b"..."},   # little-endian raw bytes
  ],
  "value_types": {"<value id>": type, ...},   # every SSA value: block args + results
}

# block
{
  "arguments": [value_id, ...],
  "ops": [
    {
      "id": int,
      "name": "add",
      "operands": [value_id, ...],
      "results": [value_id, ...],
      "attributes": {name: attr_value, ...},
      "regions": [ [block, ...], ... ],   # one nested block list per region
      "location": {"file": str, "line": int, "column": int} | None,
    }
  ],
}
```

Value ids are the Python module's id space; they are unique within the envelope and are preserved across a transform.
Location is diagnostic only and participates in no semantic decision.

### FFI surface (`evoxc._evoxc_native`)

- `version() -> str`
- `infer(op_name, operand_types, attributes, result_types=None, module=None) -> list[type]`
  - `operand_types`: list of types (never None).
  - `attributes`: normalized attribute dict (Python `ir.Builder._validate_attributes` already ran).
  - `result_types`: explicit result types for ops with no shape function (`runtime_call`/`block_call`/`external_call`); `None` otherwise.
  - `module`: `None` for pure shape functions; `{"functions": [(name, output_types), ...]}` when an op needs the enclosing signature (`call`).
  - Returns all result types in one call (no per-result getter ladder).
  - Raises `NativeError(kind="ShapeError"|"VerificationError"|"DTypeError", message)`.
- `transform(kind, module, config) -> {"module": envelope, "output_types": [type...], ...}`
  - `kind` ∈ `vectorize`, `vmap`, `grad`, `jvp`, `vjp`.
  - `config` carries the *host-neutral* transform parameters (axes per tensor input, out-axes, argnums, tangent/cotangent specs, batch depth/name); the Python side has already normalized pytrees/selectors against the host signature.
  - `config["extension_rule"]` is an optional Python callable used **only** for ops whose rule is a host-registered extension (`block:*`, `external:*`, `sparse_*`); builtin ops never call it (single native implementation).
  - Returns the transformed module plus the transform-specific metadata the Python wrapper needs to rebuild its `Graph` (output types, kept/added static records are Python-side).
- `export_stablehlo(module, options) -> str` (later phase)
- `verify(module) -> None` (later phase; Python `ir.verify` remains authoritative until ported)

`NativeError` is a Python exception class exported by the extension; `evoxc/_native.py` maps `kind` to `evoxc.core`'s public error family so callers keep catching `ShapeError`, `TransformError`, etc.

## Single-authority rules (binding)

- Builtin op inference lives **only** in `evoxc-core::infer`; Python `ir/inference.py` becomes a thin delegation to `infer`.
- Builtin batching rules live **only** in `evoxc-compiler::transform::batching`; Python `transforms/rules.py` builtin tables are deleted once the native path is the default.
- Builtin JVP/VJP rules live **only** in `evoxc-compiler::transform::autodiff`; same rule.
- Extension rules (`block:<name>`, `external:<name>`, `sparse_*`) stay in Python by design: they are host-local extension code. The native engine invokes them through the `extension_rule` callback and inlines the returned subgraph (SSA remap, no Python objects in the native graph).
- The NumPy interpreter (`evoxc/backends/numpy`) remains an independent Python reference implementation, deliberately not native, so it can detect native semantic defects.
- Compiler adapters and their StableHLO lowering stay Python in v1 (backend/codegen code, not frontend semantics). `evoxc.backends.stablehlo.export` is therefore unchanged by this phase.

## Axis metadata

The vectorize engine tracks `MappedAxes` = a count of leading mapped axes per value (v1 axis 0 only; nesting gives 1, 2, …).
The batch axis identity is a fresh `Dim` named from the pass depth (`batch`, `batch_1`, …); it is a diagnostic name only — the engine keys on the per-pass batch id, never on the name or extent.
Python supplies `batch_name` in the transform config; output-tree static-leaf remapping for registered pytree nodes (sparse) stays in Python (`_remap_output_aux`).

## Routing table

| Path | Area |
|---|---|
| `./evoxc-core/` | Dtypes, dims, shapes, op schemas, inference, graph IR, verification, envelope types. |
| `./evoxc-compiler/` | Builder, vectorize/vmap engine and batching rules, JVP/VJP/grad engines and rules. |
| `./evoxc-runtime/` | Runtime session/buffer/submission model (later phase). |
| `./evoxc-python/` | PyO3 module `evoxc._evoxc_native` and the envelope conversion. |
| `./tools/` | Development-only codegen/fixture helpers. |

## Build and test

```bash
source scripts/dev-env.sh
scripts/build-native.sh          # cargo build --release + copy the extension into evoxc/
cargo test --manifest-path native/Cargo.toml
.venv/bin/python -m pytest tests/ir tests/transforms -q
```

`CARGO_HOME` must be writable (`scripts/dev-env.sh` points it at `~/.cache/cargo` on this NixOS host).
The extension file is `evoxc/_evoxc_native.abi3.so` (gitignored); it is built from `native/evoxc-python` and loaded by `evoxc/_native.py`.
