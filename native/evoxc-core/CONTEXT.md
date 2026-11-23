# evoxc-core — the native numerical authority (IR, envelope, inference, verification)

## Intent

`evoxc-core` is the Rust half of EvoX Compiler's numerical authority: dtypes and `np.result_type` promotion, symbolic `Dim`/`DimExpr`, `ValueType`, typed attributes with raw-byte tensor payloads, the generated op schema registry, the region-based SSA graph IR, the `evoxc-native-ir` envelope codec, native shape inference and the IR verifier.
It has no Python dependency and no knowledge of storage, devices, backends or compilation: those live in `evoxc-compiler`, `evoxc-runtime` and the PyO3 crate.
The Python packages stay authoritative for behavior: `src/infer/` must match `evoxc/ir/inference.py` exactly and `src/verify.rs` must match `evoxc/ir/verify.py`, pinned by the committed differential fixture.
Allowed third-party dependencies are `serde`, `serde_json` and `base64` only (`../../native/CONTEXT.md` is the binding FFI/wire contract; this file refines crate internals and never changes the wire format or FFI names).

## Module map

| Path | Area |
|---|---|
| `./src/lib.rs` | Module declarations and the crate-level re-exports (`Program`/`Function`/`Block`/`Op`/`Value`/`Region`/`Location`/`ValueId`/`OpId`, `decode`/`encode`, `infer_op`, `verify`). |
| `./src/dtype.rs` | The 14 dtypes with numpy-compatible names/sizes and promotion through the generated table. |
| `./src/dtype_table.rs` | **Generated** 16384-entry `np.result_type` table (one per 14-bit dtype subset); regenerate with `../tools/gen_dtype_promotion.py`, never hand-edit. |
| `./src/dim.rs` | `Dim` (`Int`/`Named`/`Expr`/`Dynamic`), `DimExpr`, `DimOp`, structural equality and the `add`/`sub`/`mul`/`floordiv`/`modulo`/`min`/`max` builders with int folding. |
| `./src/value_type.rs` | `ValueType { dtype, shape }`, `Shape = Vec<Dim>`, the `tensor<3xBxf32>` `Display`. |
| `./src/attr.rs` | `AttrValue`/`Attrs`/`ArrayPayload`, the `__evoxc_ndarray__` and `__evoxc_tuple__` markers, schema validation and default filling. |
| `./src/schema.rs` + `./src/schema_table.rs` | `OpSchema`/`AttrSchema` and the **generated** 121-op registry (arity, result count, effect, region count, terminator flag, `has_shape_fn`, attribute schema with JSON defaults), with `OP_INDEX` mapping every name to its `OPS` position; regenerate with `../tools/gen_opdefs.py`. |
| `./src/wire.rs` | Envelope JSON conversions for dims, shapes, value types and array payloads. |
| `./src/graph.rs` | The IR: `Program`, `Function`, `Block`, `Region`, `Op`, `Value`, `ValueOwner`, `Use`, `Location`, numeric ids, `Table<T>` arenas, id allocators, use bookkeeping. |
| `./src/envelope/mod.rs` + `./src/envelope/tests.rs` | `decode`/`encode` of `evoxc-native-ir` v1, the tolerant attribute decoder and its round-trip/validation tests. |
| `./src/infer/mod.rs` | `infer_op`, `Ctx`, attribute validation/default filling, op-specific result rules, the category dispatch chain, the per-category handled-op lists and helper parity tests. |
| `./src/infer/common.rs` | The shared inference helpers (`one`, `normalize_axis`, `broadcast_shapes`/`broadcast_dim`, `Count`/`count_factors`/`check_counts_equal`/`divide_counts`, `sum_dims`/`add_dim`/`mul_dim`/`div_dim`, `agree_dim`, `dim_min`/`dim_max`/`min_dim_linalg`, `reduce_dtype`, the linalg dtype rules, `shape_attr`, the `attr_*` getters, `ConvPad`/`conv_padding`/`conv_out_dim`/`check_conv_channels`, `pad_config`, `fmt_shape_py`). |
| `./src/infer/{elementwise,reduction,structure,linalg,random,collective,sparse}.rs` | One category file each, private to the category, so parallel workers never share a file. |
| `./src/verify.rs` | `verify(&Program)`: module/function/block/op structure, SSA dominance, attribute schemas, result-type agreement, use bookkeeping in both directions. |
| `./tests/parity.rs` + `./tests/parity/` | The committed differential fixture (`cases.json`, `expected.json`, `generate.py`) pinning native inference against the Python authority. |

## Public API

- `Program` owns every value, op and block in `Vec`-backed `Table` arenas; ids are `ValueId(u32)`, `OpId(u32)` and `BlockId(u32)` and are module-unique and never renumbered by the codec.
- `Program::new/new_op_id/new_value_id/new_block_id/max_value_id/max_op_id/reseed_allocators/add_function/function/function_mut/main/value/value_mut/op/op_mut/block/block_mut/add_value/add_op/add_block/push_op/entry_block/terminator/operand_types/computed_output_types/add_use/remove_use/rebuild_uses`.
- `max_value_id`/`max_op_id` expose the highest occupied id (`None` on an empty arena) for dense id-indexed transform tables; the `_mut` accessors destructure the `Program` fields so their "not defined in module …" error borrows `name` instead of cloning it per call.
- `Value` carries type, owner (`ValueOwner::OpResult { op, index }` or `BlockArgument { block, index }`) and the recorded `uses`; the IR stores no data.
- `envelope::decode(&Json) -> Result<Program>` / `envelope::encode(&Program) -> Json`, plus `attrs_from_json` / `validate_attrs` for op attribute maps.
- `infer::infer_op(op_name, operand_types, attrs, result_types, module) -> Result<Vec<ValueType>>` with `InferCtx` carrying the optional explicit result types and the owning module.
- `verify::verify(&Program) -> Result<()>`.

## Graph IR invariants

- Ids are module-unique and preserved end-to-end; `Value.id` always equals its table key, and a block argument's owner index equals its position in `Block.arguments`.
- Functions hold a signature plus one body `Region`; a region is a list of block ids, a block is argument ids plus an ordered op-id list, and the terminator is the last op.
- Ops carry operands, results, typed attributes, nested regions (0, 1 or 2 per the schema) and an optional `Location`; `Op::schema/effect/is_terminator/result` consult the generated registry.
- Use bookkeeping is maintained by `Program::rebuild_uses` and `add_use`/`remove_use`; `verify` checks recorded uses and operands against each other in both directions.
- Construction helpers return `Result`, never panic on user input; allocation is monotonic and `reseed_allocators` raises the counters above every decoded id.

## Envelope contract

- Format tag `evoxc-native-ir`, version `1`; a wrong/missing format or an unsupported version is a `PersistenceError`.
- The module-wide `ops` table is referenced from blocks by decimal-string `ref` (authoritative) with a matching numeric `id`; a second reference to one op id is a `VerificationError`.
- A single-block region encodes as the block object itself, a multi-block region as `{"blocks": [...]}`; nested regions are decoded lazily from the flat table with cycle detection.
- Constant payloads are raw little-endian bytes inside `{"__evoxc_ndarray__": {"dtype", "shape", "data_b64"}}`; tuples use `{"__evoxc_tuple__": [...]}`.
- Decode validates duplicate value/op ids, dangling operands, unreferenced table entries, unknown op names, malformed payloads and declared-vs-computed `output_types`, then rebuilds uses and reseeds the allocators.
- Encode is deterministic (ascending op ids, function/block/region order, ascending object key order — `serde_json::Map` is a `BTreeMap`, no `preserve_order`) and always emits `location`, `metadata`, `input_types`, `output_types` and both `id` and `ref`.
- Attribute decoding turns JSON `null` into `AttrValue::Null` — the documented "absent / `None`" representation — and accepts it exactly where the schema does (`AttrSchema::accepts_null`/`accepts`): every `any` attribute and the optional `int`/`ints` ones declared `default=None` (`conv.strides`/`input_dilation`/`kernel_dilation`, `slice.strides`, `transpose.permutation`, `argmax.axis`/`argmin.axis`, the collective `group_size`s, `random_permutation.n`, `flip.axes`/`roll.axis`, `matrix_rank.tol`, `nan_to_num.posinf`/`neginf`).
- A `null` under any OTHER attribute — a required one included, or an optional one whose default is a value (`sort.axis`, `pad.value`, the random `algorithm` name) — is a `VerificationError`, exactly as the Python builder rejects it; the same predicate (`AttrSchema::accepts`) serves `envelope::validate_attrs`, `Attrs::from_json_map`/`validate`, `infer::validate_attrs` and `verify`. `attr.rs::tests::null_is_accepted_exactly_where_the_schema_declares_it` pins both sides.
- `AttrValue::Null` round-trips back to `null`, and the typed accessors reject it while `opt_i64`/`opt_ints`/`opt_strs`/`opt_shape` read it as `None`.
- A `{"__evoxc_tuple__": [...]}` marker decodes by the DECLARED tag (Python tuples reach typed attributes: `gather.axes = (0,)`, `slice.strides = (1, 2)`, tuple-built `pad.padding_config`).
- `ints`/`floats`/`strs` decode the marker to their sequence variant, elementwise, and a wrong member type is a `VerificationError` naming the item index.
- `nested_ints` decodes each item as a plain list OR a nested tuple marker, elementwise as ints.
- `shape` decodes each item with `wire::dim_from_json` (`{"int": ...}`/`{"dim": ...}`/`{"expr": ...}`/`null`).
- `any` keeps `AttrValue::Tuple` of generic `AttrValue::Any` members and re-emits the marker.
- A scalar tag (`bool`/`int`/`float`/`str`/`dtype`/`ndarray`) rejects a tuple with an explicit `VerificationError` naming the tag.
- `int` accepts a JSON integer that fits `u64` (decoded through `as_u64`) with an `as_i64` fallback, so a 64-bit unsigned value above `i64::MAX` such as the splitmix64 salt `0x9E3779B97F4A7C15` (11400714819323198485) decodes from its true bit pattern — the bit pattern the engine reads back with an `as u64` cast and feeds to the splitmix kernel — while a negative value keeps its sign and re-encodes as a negative JSON integer.
- A JSON float, bool or string under the `int` tag is an explicit `VerificationError` ("expected an int"), and a value outside both 64-bit ranges never reaches this crate because the Python-side conversion rejects it first.
- Typed sequence variants (`Ints`/`Floats`/`Strs`/`NestedInts`/`Shape`) re-encode as plain lists (semantically identical for typed attributes), and `AttrValue::Tuple` matches only the `any` tag, so a malformed envelope that bypasses decoding is still rejected.

## Envelope cost model (decode / encode)
- Decode is two passes over the flat `ops` array: `OpTable` holds `(op id, &Json)` entries sorted once and indexed by binary search, and `in_progress`/`referenced` are dense `Vec<bool>` indexed by an op's position in that table (no `BTreeMap`/`BTreeSet`, no per-op string keys).
- `Program::reserve_tables(ops, result_values)` pre-sizes the dense `Table<Op>`/`Table<Value>` arenas before decoding, so a whole program decodes without a table rehash.
- Error context is lazy: `Ctx` (`&dyn Fn() -> String`) is invoked only on the error path, so the success path formats no `where`/`op '...'` prefixes.
- `encode` walks the finished `Program` in the fixed wire key order; block arguments and op lists are borrowed slices (`&[ValueId]`/`&[OpId]`) and are never cloned to build the payload.
- The FFI glue (`evoxc-python/src/convert.rs`) converts each call exactly once (Python → `Json` inbound, `Json` → Python outbound); the outbound direction interns every repeated `str` (dict keys, dtypes, op names) in one per-call `HashMap<Box<str>, PyString>`, so repeated keys reuse one object and its cached hash.
- Measured on the AD seam (`grad` over a 752-op elementwise chain, release build, CPU time): `decode` ~0.5 ms and `encode` ~0.9 ms of the ~7.4 ms `transform` call, Python → `Json` ~1.1 ms; the outbound `Json` → Python materialization (~5 ms) dominates the seam because every envelope node becomes a fresh dict/list/int, so its floor is Python object allocation, not this crate.
- Cost is linear in envelope nodes: ~0.3 µs/op decode, ~0.5–1.2 µs/op encode, ~2.8–3.0 µs/op `Json` → Python (5255 out ops: 1.9 / 6.5 / 15.6 ms).

## Inference contract

- `infer_op` requires the schema, checks the declared arity, validates caller attributes (unknown key, missing required, wrong tag), fills declared defaults (`null` default ⇒ `AttrValue::Null`), then dispatches.
- Ops without a shape hook resolve their result types op-specifically: `constant` from its payload, `call` from the callee's `output_types` in the module, `if`/`runtime_call`/`block_call`/`external_call` and the two `custom`-category callback ops `custom_call`/`inline_stablehlo` from explicit `result_types` (an absent one is a `VerificationError` naming the op; their required `result_specs` attribute is machine-checked by `verify` but is never used to derive the types).
- `return` yields zero results; `while` and `stop_gradient` are operand-identity; every other op belongs to exactly one of the seven category modules.
- `schema::op_specific_results()` exposes the generated `OP_SPECIFIC_RESULTS` list: exactly the eight no-shape-hook ops, alphabetically sorted (`block_call`, `call`, `constant`, `custom_call`, `external_call`, `if`, `inline_stablehlo`, `runtime_call`), so a caller can tell them apart from ops whose shape is simply unported.
- The two `custom`-category callback ops are schema-only here: `custom_call` (`call_target_name` and `result_specs` required; optional `backend_config` default `""`, `api_version` default `1`, `has_side_effect` default `false`) and `inline_stablehlo` (`body` and `result_specs` required; optional `has_side_effect` default `false`), both with variadic arity and result counts, effect `callback`, zero regions and no shape hook.
- The eight handled-op lists (`ELEMENTWISE_HANDLED`, `REDUCTION_HANDLED`, `STRUCTURE_HANDLED`, `LINALG_HANDLED`, `RANDOM_HANDLED`, `COLLECTIVE_HANDLED`, `SPARSE_HANDLED`, `CONTROL_HANDLED`) cover 113 of the 121 registry ops — exactly the 113 with a shape hook, since the eight no-shape-hook ops (`custom_call`, `inline_stablehlo` and the six op-specific ones) belong to no category list; a test asserts each `has_shape_fn` op appears in exactly one list, and the parity coverage test asserts each appears in a fixture case.
- A final unresolved op is an explicit `VerificationError` ("no native shape function is registered for this op"), never an empty result.
- Symbolic semantics match Python: broadcasting yields the other side for `1`, passes equal dims, errors only on unequal static ints, propagates `Dim::Dynamic`, and otherwise builds `DimExpr(max, result, dim)`; element counts fold multiplication chains (`Count` keeps an insertion-ordered factor list because Python dict order decides the emitted wildcard `DimExpr`); `sum_dims` folds left-associatively with int folding.
- Dtype promotion always goes through the generated set-based table (`DType::promote`/`promote_all`), never a hand-written lattice.

## Verification contract

- `verify` checks module/function structure (at least one function, unique names, one-block v1 bodies, entry-argument signature agreement, a final `return` terminator, `output_types` agreement), then block/op structure (region and block ownership, nested entry-argument types, arity, region count, result count, attribute schema, result ownership/index), then result-type agreement through `infer_op` (or the op-specific rules for hook-less ops), then use bookkeeping.
- Messages are the Python texts and carry `[at file:line:col]` when the op has a known location (the native placeholder location maps to Python's `None`).
- For `runtime_call`/`block_call`/`external_call`/`custom_call`/`inline_stablehlo` the op-specific rule decodes the declared `result_specs` attribute (an `any` list of wire types) and requires it to equal the recorded op result types; a non-sequence `result_specs` is a `VerificationError` naming the op.
- Python's module `version` and `parent`-pointer checks have no native counterpart: the envelope owns the wire version and the `Program` tables express ownership.

## Error families

- `ShapeError` for shape/axis conflicts, `DTypeError` for dtype problems, `VerificationError` for schema/SSA/protocol violations, `PersistenceError` for envelope format/version problems; all via `Error` with `kind_str()` matching the Python class name.
- Documented divergences from Python's exception classes: an unknown `reduce_op` or random `algorithm` is a `VerificationError` (Python raises a bare `ValueError`); `div_dim` with a zero group is a `ShapeError` (Python `ZeroDivisionError`); an empty `gather.axes` is an explicit `ShapeError` (Python `IndexError`).
- Diagnostic rendering uses the compact `Dim` `Display` (`max(B, 4)`) instead of Python's dataclass repr; only symbolic-dim error text differs.
- `count_factors` returns `None` on `i64` overflow (Python has big integers) — the conservative "not statically known" path.

## Test strategy

- `cargo test --manifest-path native/Cargo.toml -p evoxc-core` runs 230 in-crate lib tests plus the 2-test parity integration test (`tests/parity.rs`); no GPU, network or Python interpreter is needed.
- Per-module in-file `#[cfg(test)]` tests cover dims/dtypes/attrs/wire, graph construction and use tracking, envelope round-trips (symbolic dims, nested `if`/`while` regions, tuple/ndarray attributes, every validation error), helper parity (broadcasting, count folding/division, axis normalization, reduce dtypes, default filling) and each inference category ported case-by-case from `tests/ir/test_inference.py`.
- `verify.rs` tests construct small programs and assert one violation each (terminator, dominance, arity, dtype/shape disagreement, constant payload, `if` branches, use records, location annotation), plus two decode-then-check tests for the `custom`-category callback ops: an `evoxc-native-ir` v1 envelope TEXT per op is decoded, its `result_specs`/scalar attributes are asserted, `infer_op` is shown to require explicit result types (and to return them unchanged), `verify` passes, and a variant whose recorded results or `result_specs` disagree fails with a `VerificationError` naming the op.
- `src/attr.rs` pins the `int` tag with three regression tests — `int_attribute_decodes_unsigned_values_above_i64_max` (the splitmix64 salt above `i64::MAX`, also through the `random_key_mix` schema), `negative_int_attribute_still_decodes_and_roundtrips` (a negative value's decode and round-trip, plus `0` and `i64::MAX` exactness) and `int_attribute_rejects_non_integers` (a JSON float, bool and string each a `VerificationError`) — alongside the existing encode-determinism and round-trip coverage.
- `tests/parity.rs` replays 518 committed cases over all 113 shape-fn ops (461 successful, 57 explicit-rejection) and requires exact result types for the successes and error-ness for the rejections; `KNOWN_DIVERGENCES` is empty.
- `tests/parity/generate.py` is dev-only and needs a venv with numpy + evoxc; regenerate the fixture whenever `evoxc/ir/inference.py` or `evoxc/ir/op_defs/**` changes, then re-run `gen_opdefs.py` for the schema table.
- The Python-free gates are `cargo test -p evoxc-core`, `cargo build -p evoxc-python` and `cargo test --workspace`.

## Known gaps / Notes for agents

- This crate implements inference and verification only; the mutable builder, the `vectorize`/`vmap` and AD engines (and their rules) and the extension-rule bridge belong to `evoxc-compiler`, the session/buffer model to `evoxc-runtime`, and the PyO3 surface to `evoxc-python`.
- `envelope::encode` has no `Result`: a value id missing from the arena is emitted as `"type": null` and fails on the next decode; a well-formed `Program` never hits it.
- `verify` deliberately has no per-call inference memo (a pure speed optimization); its verdicts are identical without it.
- The category files own only their own op lists; adding an op means updating the Python registry, regenerating `schema_table.rs`, adding it to exactly one handled list, implementing the hook and regenerating the parity fixture.
- `src/schema_table.rs` (2067 lines) and `src/dtype_table.rs` (712 lines) are generated data and must not be split or hand-edited.
- `src/verify.rs` (1550 lines) and `src/infer/common.rs` (983 lines) are near the size guideline: `verify.rs` holds ~858 implementation lines plus ~692 test lines, and `common.rs` is the single shared helper vocabulary every category imports — split only if a category's helpers become genuinely independent.
- Every shell command in this repository must start with `source scripts/dev-env.sh &&`, and cargo needs `--manifest-path native/Cargo.toml`.
