# batch_functions — the native merge engine

## Intent

`batch_functions` merges N programs with identical input/output signatures but different bodies into ONE `Program` whose leading axis is a STATIC function axis of length N (row i evaluates program i).
It is the native port of `evoxc/transforms/batch_functions.py` + `_merge.py` + `_emit.py` + `_bands.py` and must reproduce their semantics, report and v1 deferral messages exactly.
The Python files are the authoritative behavioral spec; `tests/transforms/test_batch_functions*.py` are the behavioral tests (read-only here).
The merged graph is an ordinary graph of ordinary ops: it needs no new backend support and every op goes through the existing batching registry (`crate::batching::dispatch`).

## API Surface

```rust
pub struct BatchFunctionsConfig {
    /// One entry per entry-function block argument: Some(0) maps the leading axis, None shares it.
    pub axes: Vec<Option<i64>>,
    /// One entry per return operand (tensor output leaf): Some(0) keeps the function axis, None requires unmapped.
    pub out_axes: Vec<Option<i64>>,
    pub switch_ops: SwitchOps,
    pub bands: Bands,
    pub alignment: Alignment,
}

pub enum SwitchOps { None, Default, Groups(Vec<MergeGroupSpec>) }
pub enum MergeGroupSpec { Category(String), Ops(Vec<String>) }
pub enum Bands { Flat, Explicit(usize), Auto }
pub enum Alignment { Variant, Occurrence, Auto }        // Default = Variant
impl Alignment { pub fn parse(s: &str) -> Result<Alignment>; pub fn as_str(self) -> &'static str }

pub struct BatchOutput {
    pub program: Program,
    pub output_ids: Vec<ValueId>,   // the merged return operands, in order
    pub input_ids: Vec<ValueId>,    // the merged function's block arguments, in order
    pub report: BatchReport,
}

pub fn batch_functions(programs: &[Program], config: &BatchFunctionsConfig) -> Result<BatchOutput>;
/// The same engine with an explicit rule registry / host bridge (sparse, block, external rules).
pub fn batch_functions_with_env(programs: &[Program], config: &BatchFunctionsConfig, env: &TransformEnv<'_>) -> Result<BatchOutput>;
```

`batch_functions` uses `TransformEnv::builtin()`.
The config carries already-normalized per-leaf entries: parsing a pytree / JSON / `switch_ops` spellings belongs to the FFI layer.
This crate still validates lengths, entries in `{0, None}`, band counts and group spellings so a direct Rust caller cannot bypass them.

`BatchReport` has exactly the 12 Python keys, same names, nested `SwitchRecord`s:

```rust
pub struct BatchReport {
    pub n_functions: usize,
    pub value_classes: usize,
    pub shared_classes: usize,
    pub switched: Vec<SwitchRecord>,
    pub stacked_constants: usize,
    pub output_muxes: usize,
    pub emitted_ops: usize,
    pub bands: usize,
    pub band_sizes: Vec<usize>,
    pub band_strategy: &'static str,      // "flat" | "explicit" | "auto"
    pub cost_estimate: Option<f64>,
    pub flat_cost_estimate: Option<f64>,
}
pub struct SwitchRecord { pub group: String, pub variants: Vec<String>, pub rows_per_variant: Vec<Vec<usize>> }
impl BatchReport { pub fn to_json(&self) -> serde_json::Value; }
```

`to_json()` mirrors Python `BatchReport.as_dict()`: tuples become arrays, `switched` entries become objects with `group` / `variants` / `rows_per_variant`, `cost_estimate` values are JSON numbers or `null`.
`SwitchRecord.rows_per_variant` always holds GLOBAL program indices, also in a banded merge.

## Module map

| File | Responsibility |
|---|---|
| `./mod.rs` | Public config/report/output types, config + program validation, the flat/banded driver, `batch_functions(_with_env)`. |
| `./canon.rs` | Hashable canonical keys: attribute/dim/JSON canonicalization, `VariantKey`, `Identity`, `Skeleton`, operand `Slot`, `MergeGroup` + `parse_switch_groups`. |
| `./dag.rs` | `MergeDag` / `ClassInfo` / `ProgramInfo`, occurrence-indexed skeleton alignment under `alignment`, reachability from the output slots. |
| `./emit.rs` | Flat emission: new module/function, constants (share/stack), single-variant classes through the batching registry, switch muxing, output muxing, counters. |
| `./bands.rs` | Partition rules (`even_partition`, fingerprints, ladder), the static cost model, band-local input views, the banded driver and output combination. |

## Canonical identity and the skeleton

* `VariantKey` = `(op name, canonical attributes sorted by attribute name)`.
  Constraint: attributes must be hashable and deterministic, so floats hash by `f64::to_bits`, arrays by `(dtype, static shape, payload bytes)` (payload CONTENT is part of the key), JSON is canonically ordered, tuples/lists map to one `seq` form, `AttrValue::Null` maps to `none`.
* `Identity` = `Constant` for `constant`, else `Group(group key)` when a `switch_ops` group (first match in declaration order) covers the op, else the op's `VariantKey`.
* `Skeleton` = `(Identity, Vec<Slot>, Vec<ValueType>)`; `Slot::Input(leaf_index)` for a block argument, `Slot::Class{class_id, result_index}` otherwise.
* Per program, per skeleton, `occurrence` is the 0-based index of the distinct `VariantKey` seen at that skeleton in THAT program; an exact repeat reuses its class.
* `MergeGroup`: `Category(name)` matches `schema::lookup(op).category`; `Ops(names)` matches the op name. `label` is the category name or the comma-joined sorted names. `key` is hashable (`category` or the sorted name set).
* `Alignment`:
  * `Occurrence` — the class at the occurrence bucket (created on demand; byte-identical legacy output).
  * `Variant` (default) — for a switchable (`Group`) skeleton, first a class of that skeleton that already holds this variant and does NOT already hold this program, else the occurrence bucket.
  * `Auto` — variant-first plus a last-resort reuse of any other free class of the skeleton before creating one.
  * Constants are never aligned (they stack by occurrence). Non-switchable skeletons have one variant per skeleton by construction.
* A class holds at most one distinct variant per program: every lookup filters on "does not already hold this program". Contributors arrive in non-decreasing program order, so the last contributor's program index is the test.
* `MergeDag.classes` is a `Vec<ClassInfo>` indexed by class id; `order` is creation order; `reachable` filters order to classes reachable from the return operands. Never iterate a `HashMap` to make emission or fingerprint order.

## Emission contract

* New module: `Builder::with_metadata(programs[0].name, programs[0].metadata)`; the entry function keeps its name/metadata; a mapped input leaf gains a STATIC leading `Dim::Int(n)` (never a symbolic dim), an unmapped leaf keeps its type.
* Input metadata seeds the env with `MappedAxes::new(1)` for mapped leaves and `UNMAPPED` otherwise.
* **Constant class**: all contributing payloads equal -> ONE shared UNMAPPED constant; otherwise ONE stacked `(n, *payload_shape)` constant where row i is program i's payload and non-contributing rows use the lowest contributing program's payload (never NaN) -> MAPPED on the leading axis. Increment `stacked_constants` only for the stacked case.
* **Single-variant class**: rebuild the op ONCE through `crate::batching::dispatch` with a `RuleOp::from_source(...).bind_operands(...)` and the operands' `MappedAxes`. Identical programs therefore emit zero `select`s. Stamp every op emitted by the rule with the contributing op's `Location`.
* **Switch class** (>=2 variants at one `Group` skeleton): emit every variant through its own rule over the SAME rewritten operand stack; require identical emitted result types (else the exact Python `TransformError`); align every candidate to `(n, *(1,), *per_row)` (an unmapped candidate gains the leading function axis; per-row ranks are padded with size-one dims; a non-broadcastable candidate is an explicit `TransformError`); mux with one static bool mask constant per extra variant, shape `(n,) + (1,)*(rank-1)`, TRUE on that variant's rows, selecting variant 0 as the accumulator. A switch class is ALWAYS MAPPED. Record one `SwitchRecord`.
* **Outputs**: one value per return operand. One distinct producing ref -> pass through (broadcast an UNMAPPED value to `(n, *shape)` only when the out entry is `Some(0)`); several refs -> one broadcast/mux per distinct ref, mask-cascade select, increment `output_muxes`, result ALWAYS MAPPED. `out_axes=None` on a mapped or muxed output is the exact Python `TransformError`.
* Terminate with the original return location and `Builder::finish()`; `input_ids` are the new block arguments, `output_ids` the return operands.
* `emitted_ops` = entry-block op count minus the `return` terminator.
* `value_classes` = `dag.reachable.len()` for the flat merge; `shared_classes` = reachable classes with >= 2 distinct contributing program indices (summed over bands).
* Locations: rule-emitted ops inherit the contributing source op's `Location`; helper ops the mux machinery emits itself (masks, selects, broadcasts, band slices/indices) get `None` — Rust has no frame-capture call site. `Program` has no per-value source map, so op locations are the whole contract.

## Banded merge

* `Bands::Explicit(k)` with `1 <= k <= n`: `even_partition` — k contiguous bands, sizes as even as possible, larger bands first (`n=10, k=3` -> `(4,3,3)`).
* `Bands::Flat` and `Bands::Explicit(1)` and an auto-resolved single band delegate to the FLAT emitter; only the report differs (`bands`, `band_sizes`, `band_strategy`, costs).
* `Bands::Auto`: candidates in priority order (earlier wins ties): flat, fingerprint partition (programs whose full structural fingerprint agrees), that partition with every singleton bucket coalesced into one trailing misc band, then the geometric ladder `k in {2,4,8,16,32,64} ∩ 2..n-1` plus `k = n`. Drop duplicates preserving order.
* Cost model (element-equivalents): `DISPATCH_COST = 512.0`, `BAND_OVERHEAD = 8192.0`, `CONSTANT_COST = 128.0`, category weights `linalg 12, reduction 2, sorting 2, sparse 4`, otherwise 1; a symbolic/dynamic dim counts as 1 element.
  A class costs `DISPATCH_COST + op_work * rows` per distinct variant plus `DISPATCH_COST + result_elements * band_size` per extra variant (the select mux), where `rows = band_size` when the variant's operands are row-dependent and 1 otherwise.
  Row dependence propagates through operand slots and is seeded by a mapped input leaf, a switch class, and a STACKED constant class.
  `CONSTANT_COST` is paid per reachable constant class.
  A multi-band partition adds `BAND_OVERHEAD * band_count`. Memoize per-band costs in the auto planner.
* Band emission: every band gets its own `MergeDag` over its programs and emits into the SAME builder with band-local input views. A mapped input view is materialized lazily, once per `(band, leaf)`: a `slice` when the band is contiguous and every trailing dim is a static int, otherwise a `gather` with one int32 row-index constant. Unmapped inputs use the block argument itself. The band index map makes `SwitchRecord.rows_per_variant` global.
* Combine: concatenate the per-band outputs along axis 0 (broadcast an unmapped band output to `(band_size, *shape)` first); when the partition's concatenation order is not `0..n-1`, re-permute once with one shared int32 inverse-index constant via `gather`.
* `Bands != Flat` rejects any `out_axes=None` output leaf with the exact Python message (checked after the op-batchability check, before the partition).

## v1 deferrals and exact error families

Every deferral is a `TransformError` with the Python message text (naming the op / function index / group / leaf), never a silent fallback:

* an op with no rule: `batch_functions: cannot batch op '<op>' from function <i>: no batching rule registered for key '<key>' — v1 deferral of batch_functions (...) ...`; rule availability = registry hit, or `batching::extension_eligible(key)` when an extension bridge is present (`sparse_*` / `dense_dot_sparse` / `block:*` / `external:*`).
* region-bearing control flow: the exact `region-bearing control-flow ops (cond/while_loop/scan) are not supported by batch_functions in v1` message.
* non-leading axes entries (`{0, None}` only) and `out_axes=None` on a mapped/muxed output.
* `Bands != Flat` with an `out_axes=None` output.
* a switch group's candidates with non-leading batch axes, conflicting result types, or a shape one select cannot mux.
* program-shape violations (not exactly one function, not one entry block, no `return`, block-arg/input-type mismatch) and cross-program signature mismatches (input/output count, dtype -> `DTypeError`, shape -> `ShapeError`).
  Tree structure, static leaves, devices and registered output pytree nodes are FFI-layer concerns: `Program` already lost the pytree and has no device.

## Deviations from the Python contract

* Inputs are `Program`s, not `Graph`s: no pytrees, static values, devices, `TensorSpec`s or `source_locations`. Count/type checks remain here; tree/static/device/output-node checks move to the FFI layer.
* `Bands`/`SwitchOps`/`Alignment` are typed enums; spellings, `bool` rejection and JSON parsing belong to the FFI layer.
* Helper ops get no source location (see above).
* `band_sizes` / `switched` are `Vec`s where Python uses tuples.

## Test strategy

* In-crate integration tests in `native/evoxc-compiler/tests/batch_functions.rs` build programs with `Builder` and assert: identical programs -> zero selects and one batched op per source op; divergent topology -> one output mux; `switch_ops` on/off; stacked vs shared constants; `bands` None/1 (identical op sequence)/K/auto; the three alignments; every report field; and one test per deferral asserting `TransformError` + message.
* `native/evoxc-compiler/tests/parity_transform/run_batch_parity.py` is the future-FFI differential harness: it compares `evoxc._evoxc_native.batch_functions(envelopes, config)` against `evoxc.batch_functions` (structural envelope comparison, numpy numerics, `BatchReport.as_dict()`). It exits 0 with an explicit SKIP line while the FFI entry does not exist, and must not disturb `run_parity.py`.

## Routing Table

| Path | Area |
|---|---|
| `./mod.rs` | Public types, validation order, flat/banded driver, entry points. |
| `./canon.rs` | Canonical keys, merge groups, skeleton identity. |
| `./dag.rs` | Class DAG construction, alignment, reachability. |
| `./emit.rs` | Flat emission (constants, variants, switches, outputs) and counters. |
| `./bands.rs` | Partitions, fingerprints, static cost model, banded emission. |
