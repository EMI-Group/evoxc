# evoxc/ir — EvoXIR: the region-based SSA IR

## Intent

EvoXIR is evoxc's compiler-neutral, region-based SSA intermediate representation
(MLIR-inspired, but not MLIR): every `evoxc.trace` produces a `Module` of
`Function`s of `Block`s of `Op`s over typed `Value`s. EvoXIR is *the* frontend
IR — StableHLO is an important export target, never the definition. This
directory owns the IR data model, the op-definition registry (full canonical
v1 op set, 121 ops), the `Builder` (op-construction API), `verify`
(structural/type/attribute validation), serialization (the self-describing IR
payload — core of the `.evoxcgraph` format; the outer container lives in
`../persist`), the trusted fast decode of an in-process `evoxc-native-ir`
envelope, and `pretty_print`.

**Import rule (binding):** `ir` imports ONLY `evoxc.core` (Dim/DimExpr, errors),
stdlib, and numpy (constant payloads). Nothing else from evoxc. `ops`, `trace`,
`backends`, … import from `ir` — never the reverse.

## API surface (must-expose names; re-exported from `__init__.py`)

- SSA structures: `Module`, `Function`, `Region`, `Block`, `Op`, `Value`, `Use`.
- Types/location: `ValueType` (dtype + shape of `int | DimExpr | None`),
  `Location(file, line, col, code_snippet=None)`.
- Effects: `EFFECT_PURE/EFFECT_WRITE/EFFECT_READ/EFFECT_COLLECTIVE/EFFECT_CALLBACK`
  + `EFFECT_KINDS` (see "Effect model").
- `Builder` — op construction; see "Builder" section.
- Op registry: `OpDef`, `AttrSpec`, `opdef(name)` (KeyError if unknown),
  `has_opdef`, `op_names()`, `all_opdefs()`, `register_opdef()`, attr-type
  tags (`ATTR_INT`, …).
- `verify(module) -> None` raising `VerificationError` (owned by `evoxc.core`,
  re-exported).
- `IR_FORMAT_VERSION` (in `version.py`), `serialize_module(module) -> dict`,
  `deserialize_module(payload) -> Module`.
- `pretty_print(module) -> str`.

## Design decisions

### 1. Value types: reuse core.DimExpr
`ValueType(dtype, shape)`; `shape` is a tuple of `int | DimExpr | None`:
int = static dim, `Dim`/`DimExpr` = symbolic (e.g. batch `B`), `None` =
runtime-dynamic (unchecked; backends declare `dynamic_shapes` capability).
Rank is always known. `ValueType` is IR-level; the trace-level `TensorSpec`
(core) additionally carries device/name and is not used inside the IR. Shape
arithmetic over symbolic dims uses core's `DimExpr`.

### 2. Nested regions (if/while)
Control-flow ops own their bodies as `Op.regions`: `if` holds `(true, false)`,
`while` holds `(cond, body)`. Operand values are passed by binding them to the
entry-block arguments of each region (MLIR style): block-arg count/type must
match operand count/type. v1: FUNCTION regions must be single-block
(`verify.py:197-204`); nested regions must be non-empty (`verify.py:369`) but
`verify`/`serialize` tolerate several blocks there (`verify.py:413`,
`serialize.py:575-580`) — only the Builder ever builds one. `return`
is the v1 terminator (category "terminator") — **added to the canonical set**
because verification requires a terminator and the frontend op list contains
none; the frontend never names it directly.

### 3. Effect model
Effect is declared per-op in its `OpDef` (one of the five kinds). There are no
public effect tokens in v1: ordering is positional — op order inside a `Block`
IS program order for effectful ops. Binding rules for transforms/backends:
pure ops may be reordered/CSE'd but never moved across an effectful op nor out
of their region; effectful ops are never reordered relative to each other,
duplicated, or eliminated; `collective` ops additionally impose
device-synchronization ordering; `callback` ops are opaque side effects.
`write` is reserved for future stateful ops — in v1 only `rank`/`world_size`/
`block_call` use `read`.

### 4. call + multi-function modules vs v1 Graphs
The IR supports N functions per module and the `call` op (callee by name) —
needed for custom blocks and future lowering. v1 `trace.Graph`s build exactly
one function (conventionally `main`) and never emit `call`. `call`'s effect is
declared `pure` at the IR level; a backend supporting `call` must account for
callee effects itself (the v1 numpy backend may simply reject `call`).

### 5. Serialization integrity
`serialize_module` returns a self-describing, JSON-able payload dict:
`{format, version, module, functions, ops, values-by-id, constants (base64
npy), sha256}` — see the schema in `serialize.py`. The sha256 covers the
canonical JSON of the whole payload and is recomputed on load: tampering or
structural violation raises `VerificationError`; unknown format/version raises
`PersistenceError` (both owned by core). `../persist` wraps this payload in
its own container (magic header etc.) — the payload already carries everything
it needs, so the wrapper stays dumb. `IR_FORMAT_VERSION` (IR payload) is
distinct from `EVOXC_FORMAT_VERSION` (persist container).

Wire-format decisions (binding, documented in `serialize.py`): block op lists
reference the flat ops table via `"ref"` = the op id as a decimal STRING;
dims encode as `{"int"}` | `{"dim": name[, "size"]}` | `{"expr": {op, args}}`
| `null` (decoded back to real `Dim`/`DimExpr`, preserving names/sizes);
ndarray attr values move to the `constants` table (attr becomes
`{"__evoxc_ndarray__": key}`); tuples are wrapped as `{"__evoxc_tuple__": [...]}`
to survive the JSON round-trip; `runtime_call`/`block_call` `result_specs`
are encoded as a list of type dicts and NORMALIZED back to `ValueType`
instances on decode (so `verify`'s in-memory comparison still passes).
Rebuilt modules keep their ORIGINAL ids and their `_op_ids`/`_value_ids`
counters are fast-forwarded past the payload maximum, so post-load
`Builder.create` calls never collide ids. `serialize_module` runs `verify`
first; `deserialize_module` runs it on the result.

### 6. Result-type resolution
`OpDef.shape_fn: (input_types, attributes) -> result_types` computes result
types when the builder has no explicit `result_types`. `shape_fn=None` means
op-specific resolution by the Builder (`constant`: from payload; `call`: from
the callee signature; `if`: from region terminators; `runtime_call`/
`block_call`: from declared specs) — or explicit `result_types=` at the call
site. `verify` enforces agreement between recorded result types and the op's
contract.

### 7. Identity & ownership
`Value.id`/`Op.id` are module-unique, assigned by the Builder from the owning
Module's counters (stable ids for serialization). Parent pointers
(`op.parent`, `region.parent`, `fn.parent`) are wired by the constructing API;
`verify` checks consistency. `Value.uses` is maintained by the Builder and by
`Value.replace_all_uses_with`.

## Op registry (canonical v1 set — 121 ops, all declared in `op_defs/`)

| File | Category | Ops | Effect |
|---|---|---|---|
| elementwise.py | elementwise | add subtract multiply divide power remainder maximum minimum; abs negate square sqrt exp log log1p sin cos tan acos tanh floor ceil round sigmoid relu gelu erf sign; logical_and logical_or logical_not; bitwise_and bitwise_or bitwise_xor bitwise_left_shift bitwise_right_shift; cast nan_to_num | pure |
| elementwise.py | comparison | equal not_equal less less_equal greater greater_equal (result bool) | pure |
| structure.py | structure | select broadcast reshape transpose slice gather scatter concatenate pad tile flip roll diag | pure |
| reduction.py | reduction | reduce_sum reduce_max reduce_min reduce_mean reduce_prod argmax argmin cumsum cumprod | pure |
| sorting.py | sorting | sort argsort | pure |
| linalg.py | linalg | dot conv tril triu diagonal solve eigh cholesky qr matrix_rank svd matrix_exp | pure |
| control.py | control | constant stop_gradient if while call runtime_call external_call block_call | pure; runtime_call/external_call=c**allback**; block_call=read |
| control.py | terminator | return | pure |
| collective.py | collective | all_reduce all_gather reduce_scatter all_to_all broadcast_collective collective_permute | collective |
| collective.py | collective | rank world_size (scalar int64) | read |
| sparse.py | sparse | sparse_from_dense sparse_to_dense sparse_coo_to_csr sparse_csr_to_coo sparse_coo_to_csc sparse_csc_to_coo sparse_negate sparse_add sparse_multiply sparse_multiply_dense sparse_reduce_sum sparse_transpose sparse_reshape sparse_concatenate sparse_dot_dense dense_dot_sparse | pure |
| random.py | random | random_key_mix random_uniform random_normal random_randint random_permutation random_multinomial | pure |
| custom.py | custom | custom_call inline_stablehlo | callback |

**Opaque custom ops (`op_defs/custom.py`, category `custom` — binding):**
`custom_call` and `inline_stablehlo` mirror `external_call`: both are
`arity=(0, None)`, `result_count=None`, `effect=EFFECT_CALLBACK`, zero
regions, `shape_fn=None`, and their result types resolve from the required
`result_specs` attribute (or explicit `result_types`), with `verify` checking
the two agree.
`custom_call.call_target_name` is a REQUIRED `ATTR_STR` naming the StableHLO
custom-call target symbol (non-empty; `verify` rejects an empty string).
`custom_call.result_specs` is a REQUIRED `ATTR_ANY` sequence of `ValueType`s
exactly equal to the op's result types (same rule as `external_call`).
`custom_call.backend_config` is an `ATTR_STR` defaulting to `""` (opaque
backend-specific configuration passed through with the call).
`custom_call.api_version` is an `ATTR_INT` defaulting to `1` (`verify`
rejects a non-int, a bool, or a value < 1).
`custom_call.has_side_effect` is an `ATTR_BOOL` defaulting to `False`.
`inline_stablehlo.body` is a REQUIRED `ATTR_STR` holding a raw
StableHLO/MLIR snippet (non-empty; `verify` rejects an empty string).
`inline_stablehlo.result_specs` is a REQUIRED `ATTR_ANY` sequence of
`ValueType`s exactly equal to the op's result types.
`inline_stablehlo.has_side_effect` is an `ATTR_BOOL` defaulting to `False`.
Both ops' attribute VALUES are checked by `verify._verify_op_specific_attrs`
(guarded by the `_OP_SPECIFIC_ATTR_OPS` set, so other ops pay nothing beyond
their tag checks); serialization needs no special case because the
`result_specs` encoding is keyed by ATTRIBUTE NAME.
Neither op has a frontend builder, exporter entry, transform rule or backend
implementation yet — declaring them does not make any backend accept them.

**Multi-algorithm random framework (binding):** `op_defs/random.py` is the
SINGLE source of truth for the canonical algorithm names — `ALGORITHMS =
("splitmix64", "threefry2x32", "philox4x32_10")`, `DEFAULT_ALGORITHM =
ALGORITHMS[0]` (splitmix64, the v1 default), `validate_algorithm(name)` and
`algorithm_key_type(name) -> (shape, dtype)` — imported by `ops` and the
backend writers (never duplicate the literals). Key types: splitmix64 →
rank-0 int64; threefry2x32 → shape (2,) int32; philox4x32_10 → shape (4,)
int32. All 6 random OpDefs carry the shared `algorithm` attribute
(`AttrSpec("algorithm", ATTR_STR, default=DEFAULT_ALGORITHM)`); the frontend
stamps it explicitly once multi-algorithm support lands — until then the
default applies and behavior is unchanged. `inference.py` validates key
shape/dtype against the op's `algorithm` in `_check_random_key` (key form of
a different algorithm → `ShapeError` naming both; a form matching no
algorithm → `ShapeError` naming the three accepted forms; unknown algorithm
name → `ValueError` via `validate_algorithm`); `infer_random_key_mix`'s
result type is the algorithm's key type via `algorithm_key_type`. Hooks read
the attribute with `attributes.get("algorithm", DEFAULT_ALGORITHM)` so
graphs serialized before the attribute existed still verify with the default.

Declaring an op here does NOT mean every backend implements it — backends
reject unsupported ops explicitly via capabilities, never silently. IR name
note: the collective is `broadcast_collective` (the shape op `broadcast`
already owns that name).

**Registered arities (binding):** `concatenate` is VARIADIC — `arity=(1, None)`, ONE op with N operands and no cap (a 1-operand concatenate is an identity); `if`/`while` are `(1, None)`; `call`/`runtime_call`/`external_call`/`block_call`/`return` are `(0, None)`; `OpDef.check_arity` is the only arity gate (Builder `_prepare_op`, `verify._verify_op`), so nothing else caps operand count.
`concatenate.axis` is a REQUIRED `ATTR_INT` (no default); `scatter`/`gather`/`slice`/`pad` arities are fixed 3/2/1/1.
There is NO `dynamic_update_slice`/`iota`/`arange` IR op — `evoxc.arange` is a frontend Constant-op composition (`../numpy/creation.py`), so a static index vector for `scatter`/`gather` is a `constant` op with an inlined payload.
`infer_scatter` only returns operand 0's type (no IR-level indices/updates shape validation — `../ops/indexing.py` and the numpy kernel enforce it); `infer_concatenate` promotes dtypes via `np.result_type(*dtypes)` and folds the axis dims into a left-associative `DimExpr("add", ...)` chain.

**15-op batch (design notes):** `topk` is a frontend composition over
sort/argsort + gather (no IR op; static `k` ≤ extent → `ShapeError`, symbolic
extent → runtime error). `stack` = reshape + concatenate composition. `flip`/
`roll` are dedicated IR ops with np-exact kernels. `clamp` = maximum/minimum
composition (scalar bounds pre-cast to x's dtype when same_kind-castable; a
float bound on an int tensor → float64 promotion — numpy 2.x parity, deviation
vs numpy 1.x's TypeError; documented edge: int bound on uint tensor falls back
to weak promotion int64). `eye`/`linspace` = Constant-op compositions
(`linspace` float64 default — deliberate deviation from the evoxc float32
creation convention, explicit dtype param; symbolic bounds → `TraceError`, v2
deferral). `matmul` = frontend sugar over dot with rank-1 promote/squeeze
(dot's rank ≥ 2 contract and `__matmul__` → dot unchanged). No vjp/batching
rules for the 8 new IR ops + eye/linspace → `TransformError` (the random-op
pattern); clamp/matmul/isnan/stack/topk inherit their composition's rules.
Numpy backend: full reference. Compiler backends (stablehlo/iree/xla/tvm)
defer the 8 new IR ops with explicit `BackendError` (no exporter entries
added); compositions work via their components.

**Absent op kinds (verified against `op_defs/`):** there is NO `iota`, NO `dynamic_update_slice`, NO `topk`/`stack` IR op, and NO buffer/alias or in-place op — `EFFECT_WRITE` is declared but unused (`effects.py`: "reserved for future stateful ops (none in v1)"), so every value is produced exactly once (SSA) and `scatter` is the ONLY update-shaped op, marked `pure` and functional (numpy `put_along_axis` on a COPY; `infer_scatter` returns operand 0's exact type).
A generated index grid (`arange`) and zero/one/eye/linspace grids are frontend Constant compositions (`../numpy/creation.py`, `../ops/structural.py` → `ops.constant`), i.e. a payload-bearing `constant` op, never an index op; `broadcast` of a 0-d constant is the payload-free way to materialize a zero-filled buffer of a static or symbolic shape.
`concatenate` is VARIADIC (`arity=(1, None)`) — K operands in one op, no pairwise tree.
`slice`'s `start_indices`/`limit_indices`/`strides` are `ATTR_INTS` (plain static ints only; `limit_indices` is schema-required, so its `None` branch in `infer_slice` is unreachable through the Builder), while `broadcast`/`reshape` `shape` attrs accept `int | Dim | DimExpr | None`.
`return` is variadic (`arity=(0, None)`, `result_count=0`) — a function yields N values; `Function.output_types` is derived from its operands.

## Builder (op-construction API)

`Builder.create(op_name, operands=(), attributes=None, result_types=None,
location=None, regions=()) -> Op` — creates result `Value`s (ids from module
counters; types via `shape_fn` or explicit `result_types`), wires operand
`Use`s and parent pointers, validates arity/attrs/regions eagerly.
`emit(...) -> Value` is the single-result convenience (raises unless exactly
one result). Helpers: `set_insertion_point(block_or_region)`,
`push_region/pop_region`, `build_region(input_types) -> Region` (entry block +
args — the body of an upcoming if/while), `insert_block(region,
position=None)`, `set_terminator(block, op_name, ...)`,
`build_module(name, metadata=None)`, `build_function(name, input_types,
metadata=None)`. The frontend `ops` module obtains the active builder from
`trace.current_builder()`; ALL IR mutation funnels through Builder.

**Implementation notes (binding for `verify` agreement):** result types
resolve in order — explicit `result_types` → `OpDef.shape_fn` → op-specific
rules (`constant`: dtype/shape from the payload; `call`: callee output
signature; `if`: operand types of both branches' `return` terminators, which
must agree; `runtime_call`/`block_call`: `result_specs` entries — `ValueType`
| core `TensorSpec` | `{"dtype", "shape"}` dict — converted to `ValueType`);
anything else with `shape_fn=None` demands explicit types. A `ShapeError`
from a `shape_fn` propagates with the op's source location appended to its
message (when a real location is known — never for `Location.unknown()`).
NOTE: the `result_specs` conversion applies
to result-TYPE RESOLUTION only — the stored attribute is NOT rewritten, and
`verify` requires the stored `result_specs` to be a sequence of `ValueType`
instances. Frontends must pass `ValueType`s (not `TensorSpec`/dict entries)
to pass verification. Attributes are validated eagerly against the
schema (unknown keys / missing required / wrong tag → `VerificationError`),
stored as a copy with defaults applied: `dtype` values normalize to the
dtype-name string, sequence tags normalize to tuples, `int` tags accept `None`
only where the spec's `default is None`. Region count must equal
`OpDef.regions` exactly. `set_terminator` appends (last by construction),
never uses the insertion point, and rejects non-terminator names and blocks
that already have a terminator.

## verify(module)

First-violation-wins, `VerificationError` with source location when available.
Full invariant list is the docstring of `verify` in `verify.py`; summary:
module version + unique function names; function: single-block region (v1),
block args match `input_types`, terminator is `return` and last; block/region:
parent wiring, nested-region counts and entry-arg binding per `OpDef.regions`;
op: registered name, arity/result-count/region-count within OpDef, attributes
match schema (no unknown keys, required present, types tagged correctly),
result types agree with `shape_fn` (when not None); value/SSA: unique ids,
operands defined before use (dominance), `Use` bookkeeping consistent.

### Verification cost model (measured; binding for future edits)

`verify` is O(ops + uses) on a valid module, and every cache it uses is a
per-CALL local — nothing is module-global or retained across calls, so
concurrent `verify` calls (chunked compile pool) cannot interfere and a
mutated module can never be served a stale verdict.
Use bookkeeping is ONE forward pass that validates every `Use` record (type,
owner, index, correct value, in-module op, no duplicate) and records the
`(value id, op id, operand index)` triple in a per-call set; the duplicate-record
check reuses that same set (the key already carries the value id), and the
reverse pass ("every operand of every op has a matching record") is one O(1)
membership test per operand.
The reverse pass must NOT scan `operand.uses`: doing so made verification
QUADRATIC in the use count of a shared value (a graph input or constant feeding
thousands of ops — measured 2.8 M `any(...)` generator calls, exponent ≈1.4).
The original scan survives only as the fallback for a record whose value the
forward pass never visited (impossible for a valid module) — both-directions
semantics and every message stay byte-identical.
Result-type agreement memoizes `shape_fn(input_types, {})` in a per-call dict
keyed on `(op name, operand ValueTypes)` for ops carrying NO attributes (a
`shape_fn` is pure in those two inputs; `ValueType`/`Dim`/`DimExpr` are frozen);
ops with attributes, and shapes that are unhashable (only in a corrupt
module), simply skip the memo — symbolic broadcast/promotion is then derived
once per distinct operand-type tuple instead of once per op.
Message prefixes (`op 'x' (id N)`) are formatted only inside failure branches,
the attribute check is skipped when `attributes` is an empty dict and the
`OpDef` declares none, and type agreement / `constant` results compare against
the recorded type before building an intermediate `ValueType`.
Measured (CPython 3.12, `batch_functions`-merged random-expression graphs and
`add` chains): 8 070 ops 56.5 → 30.6 ms, 31 921 ops 254.3 → 125.4 ms, 125 906
ops 1 642 → 647 ms (whole-`verify` exponent 1.24 → 1.07, `_verify_uses`
exponent 1.42 → 1.06); `evoxc.lower(graph, backend="iree")` on those graphs
81.7 → 57.4 ms and 368.7 → 240.2 ms.
`verify` is on every `lower()`/`Graph.verify()` path, so these constants matter:
re-measure before adding any per-op work (f-strings, dict construction,
re-derivation) to `_verify_op`/`_verify_result_types`.

## Serialization schema (payload dict)

See the docstring in `serialize.py` (the full schema). Summary: `{format:
"evoxc-ir", version, module, functions, ops, constants, sha256}`; values are
defined inline by ops (`results`) and blocks (`arguments`) with module-unique
ids, operands referenced by id; dims encode as `{"int": n}` | `{"dim": name}`
| `{"expr": {"op", "args"}}` | `null`; dtypes as numpy `dtype.name` strings;
constant payloads as base64 `np.save` bytes in the `constants` table; sha256
over canonical JSON (`sort_keys`, compact separators) of everything else.

## Trusted fast decode (`_native_decode.py`)

`./_native_decode.py` is the private TRUSTED decode path for an `evoxc-native-ir` v1 envelope produced IN-PROCESS by the native extension from a module this runtime encoded moments earlier — that module is known valid, so the generic `_require_*` structural validation, the sha256 and the post-decode `ir.verify` are all skipped.
`build_module_fast(body)` is its only export (`__all__`) and the hot-path sibling of `serialize._build_module`; both produce structurally identical modules — original op/value ids, attributes, types, locations, parents, ordered complete `uses`, and id counters advanced past every decoded id.
It imports `serialize`'s shared pieces instead of duplicating them: `_attr_schema`, `_decode_attrs_with_schema`, `_decode_location`, `_decode_generic`, `_fast_forward_counters`, the frozen `_VALUE_TYPE_CACHE` and `_npy._dtype_from_name`.
Construction is four phases over the flat ops table: (1) every `Op`/`Value` built positionally (no generic validation walk) with its attributes and location; (2) function signature, entry-block arguments and op ordering (`op.parent` wiring, `Block.ops.append`); (3) nested op regions (control-flow bodies) through `_build_region_fast`; (4) operands plus `Use(op, index)` records APPENDED directly.
`_build_region_fast` mirrors the block path's `id`-or-`int(ref)` op-entry fallback, so a `cond`/`while_loop`/`scan` graph decodes exactly like a flat one.
`Value.add_use`'s dedup index is skipped on purpose: a trusted envelope has no duplicate use records, so the append is the whole bookkeeping.
Attributes still decode through the schema-aware `_decode_attrs_with_schema`, so an unknown attribute still raises `VerificationError`; only the generic structural checks are gone.
An empty attribute dict takes a fast `{}` path, and a missing `constants` table falls back to a shared empty dict (the native envelope inlines every ndarray payload).
Memoization: a per-decode `id()`-keyed `type_memo` over `_decode_type_fast` (envelope type dicts are shared objects out of the native writer, so identity is the correct key), which in turn keys the frozen `_VALUE_TYPE_CACHE` on `(dtype name, decoded dims)` — the SAME cache the validated decoder fills.
`_DIM_CACHE`/`_DIMEXPR_CACHE` at module level reuse frozen `Dim`/`DimExpr` instances, plus a per-decode location memo and a per-decode attribute-schema memo.
The module-level tables are capped at 4096 entries so they stay bounded across decodes; the per-decode memos die with their decode.
`ir.verify` is deliberately NOT run on a trusted decode: verification is linear in use count but not free (see "Verification cost model"), and the envelope cannot be invalid unless the native writer is broken.
The wire format is unchanged — still `evoxc-native-ir` v1 (`ENVELOPE_FORMAT = 1` in `../_native_codec.py`), same `module`/`functions`/`ops` body, value ids and dim encodings.
`evoxc._native_codec.decode_native_module(envelope, *, verify_module=None)` is the ONE entry point for trusted envelopes: it checks the discriminator/version via `_validated_envelope_body` (which keeps only `module`/`functions`/`ops`) and hands the body to `build_module_fast`.
`evoxc._native.transform` (all five builtin transforms) and `evoxc._native.batch_functions` decode through it.
`decode_module` remains the VALIDATED decoder — discriminator/version check, then `serialize._deserialize_envelope_body` (`_build_module` + `ir.verify`) — for any envelope of uncertain provenance: files, caches, and tests.
`EVOXC_NATIVE_DECODE_CHECK=1` (read ONCE at `_native_codec` import; `""`/`"0"`/`"false"`/`"False"` leave it off) is the dev-only cross-check: the fast-decoded module is run through `ir.verify`, so the trusted path can be audited without paying verification in the hot path.
It does not re-decode through the validated path or diff the two modules — `ir.verify` on the fast result is the whole check, raising the same `VerificationError` a bad module would.
The cross-check therefore costs the fast decode plus one `ir.verify` pass (≈4 µs/op at a few thousand ops, rising with size).
Cost, measured at the native AD seam on this host and recorded with the fast-decoder work (host- and load-dependent, not a fixed per-op budget; reproducible via `git log --grep 'fast decode'`): the trusted `build_module_fast` decode took 3.4 ms at 2 008 ops and 11.8 ms at 6 008 ops.
The validated decoder was measured at ≈20–21 µs/op on this host (15.4 ms at 753 ops, 47.6 ms at 2 253 ops), so the trusted path is roughly an order of magnitude cheaper per op.
That validated cost is dominated by `_build_module`'s generic validation walk plus `ir.verify`: `../../native/CONTEXT.md` records 20.3 ms of a 34.6 ms native-`grad` path at 752 ops for the validated decode alone, which is why the same end-to-end `grad`/`vjp` hop is ≈17 ms over the trusted path versus ≈29 ms when the seam decodes validated.

## Cross-language boundary facts

IR objects are mutable Python dataclasses with direct parent/owner links, operand references and use lists, and verification checks Python class membership plus object identity, not only numeric ids.
`Value.replace_all_uses_with` and `Op._replace_operand` mutate existing objects and use lists, so an alternative storage implementation must preserve observable wrapper identity and mutation semantics or introduce an explicit snapshot boundary.
Builder is the standard construction path, not an encapsulation boundary: dataclass fields remain public, `Block.append`/`insert` only set the parent, `Block.erase` only removes the op and clears its parent without unlinking operand uses, and RAUW performs no same-module or type checks.
Module ids are unique only within a module and have no generation or provenance component, so cross-module cloning/inlining must allocate destination ids and remap operands, results, region arguments and parent/use links rather than splice source objects.
The IR layer exposes no generic clone/inlining or validated-handle API.
Explicit `result_types` bypass Builder inference but not verifier inference, and a custom `shape_fn=None` definition without a recognized op-specific rule is rejected by verification even if explicit result types were supplied.
`ValueType` directly normalizes through `np.dtype`, shape arithmetic uses core `Dim`/`DimExpr`, promotion uses `np.result_type`, and `_reduce_dtype` probes numpy reductions on a one-element ndarray.
Builder and verifier both execute registered Python `shape_fn` callbacks, and verifier memoization assumes those callbacks are pure functions of operand types and attributes.
The IR serialization payload is independent of the outer persistence container but depends on the process registry for schema-directed encoding/decoding and subsequent verification; it does not carry executable op definitions.
Generic serialization stringifies dictionary keys, uses reserved marker dictionaries, and restores generic `ValueType` attributes as plain dictionaries except for schema-handled `result_specs`, so arbitrary `ATTR_ANY` objects do not imply portable lossless serialization.
Constant bytes use base64 NPY with `allow_pickle=False`, while integrity hashing uses Python `json.dumps(sort_keys=True, separators=(",", ":"))`; another language must match its escaping and numeric spelling to preserve hashes.
See `./op_defs/CONTEXT.md` for the registry extension boundary and its Python-specific callback/default behavior.

## pretty_print

Readable SSA text, one op per line: `%0 = evoxc.add(%arg0, %arg1) :
tensor<BxNxf32> loc("model.py":12:8)`. Block args print as `%argN`; results
numbered sequentially per function; nested regions indent inline. Format spec
in `printer.py`.

## Constraints

- Import rule above; nothing from `ops`/`trace`/`backends`/etc.
- Implementation status: data structures, registry, shape-inference hooks
  (`inference.py`, 56 hooks), `pretty_print`, `verify`, the `Builder`,
  serialization (`serialize_module`/`deserialize_module`) and the private
  trusted native decode (`_native_decode.build_module_fast`) are implemented.
- Shape-inference conventions (binding for `verify` agreement): broadcasting
  resolves symbolic conflicts as `DimExpr("max", a, b)` (left dim first);
  `None` dims are runtime-dynamic and yield `None`; element-count checks
  raise `ShapeError` only on definite mismatch; sum folds are
  left-associative `DimExpr("add", ...)` chains. Dtype promotion uses
  `np.result_type`; reduction dtypes follow numpy per `reduce_op` (the
  `reduce_*` OpDefs carry a `reduce_op` attribute: 'sum'|'max'|'min'|'mean'|
  'prod').
- Files < ~1000 lines; op_defs split by category (done).
- `_native_decode.build_module_fast` must stay structurally in lockstep with
  `serialize._build_module`: a change to the envelope layout, the phase order or
  the `Use` wiring lands in BOTH, and an `EVOXC_NATIVE_DECODE_CHECK=1` run is
  the audit that catches divergence.
- Unknown op name in `opdef()` → `KeyError`; duplicate registration →
  `ValueError`; version/format mismatch on deserialize → `PersistenceError`;
  structural violations → `VerificationError`.
- `Op.opdef`/`Op.effect`/`Op.is_terminator` are hot read paths (the
  `batch_functions` merge reads them for every op of every program). `op.py`
  keeps the mandatory lazy `op_defs` import but caches only the registry
  CALLABLE (`_OPDEF_LOOKUP`, populated by `_load_opdef_lookup` on first use) —
  never a `name -> OpDef` result: the registry may grow after import
  (`ir.register_opdef`, lazily-imported `op_defs` category modules), so a name
  looked up before registration MUST resolve afterwards, and an unknown name
  must still raise the registry's `KeyError`. `effect`/`is_terminator`
  delegate to `opdef`, so one cache serves all three.
- No `numpy()`, storage, device, or DLPack on any IR object — the IR is pure
  structure (values are references, not data).

## Routing table

| Path | Area |
|---|---|
| `./op_defs/` | OpDef/AttrSpec, registry, category tables and extension/portability contract (Python shape callbacks, arbitrary attribute payloads, verbatim defaults, additive mutable registry); see its own CONTEXT.md |
| `./value.py`, `./op.py`, `./block.py`, `./region.py`, `./function.py`, `./module.py` | SSA data model |
| `./types.py`, `./location.py`, `./effects.py`, `./version.py` | Small shared definitions |
| `./inference.py` | Shape-inference hooks referenced by OpDefs (56 hooks, implemented; ~1929 lines — legitimately long, one hook module for all categories; split only if it grows much further) |
| `./op_defs/sparse.py` | Sparse op defs: 16 ops (from_dense/to_dense, coo/csr/csc conversions, negate, add, multiply, multiply_dense, reduce_sum, transpose, reshape, concatenate, dot variants), all pure |
| `./builder.py` | Op-construction API (implemented) |
| `./verify.py` | Structural/type/attribute verification (implemented) |
| `./serialize.py` | IR payload serialization: self-describing payload, sha256 integrity, round-trip rebuild with original ids (implemented); private `_serialize_module`/`_deserialize_envelope_body` serve the native envelope with inline arrays and no hash, and its schema/decode helpers (`_attr_schema`, `_decode_attrs_with_schema`, `_decode_location`, `_decode_generic`, `_fast_forward_counters`, `_VALUE_TYPE_CACHE`) are shared with the trusted fast decoder |
| `./_native_decode.py` | Private TRUSTED fast decode of the in-process `evoxc-native-ir` v1 envelope (`build_module_fast`): positional IR construction, direct `Use` appends, memoized types/dims/locations/schemas, no generic validation and no `ir.verify` (see "Trusted fast decode") |
| `./_npy.py` | Private ndarray containers: the `np.save` writer/parser for the constants table, the raw little-endian inline form of the native envelope, and the shared dtype memo tables |
| `./printer.py` | SSA text printing (implemented) |

Sibling: `../../tests/` → test suite (read-only from here; escalate test-related
writes to root).

## Test strategy (Phase 2+)

pytest under `../../tests/ir/`: registry completeness (every public op in the evoxc
contract is declared; arities/attrs/effects correct); shape-inference hooks
(static + symbolic dims per hook, broadcast/reshape/conv formulas, ShapeError
cases); `verify()` invariants
(valid modules + one test per violation class); serialization round-trips
(shapes with symbolic dims, constants as base64 npy, tamper detection, version
rejection); `pretty_print` golden output; Builder wiring (uses, ids, regions,
terminators). CPU only.
The trusted decode has no dedicated unit suite here: it is exercised through the
extension-gated native transform/`batch_functions` legs, and
`EVOXC_NATIVE_DECODE_CHECK=1` turns on its `ir.verify` audit.

## Known Issues
- **Count of record is 121 ops** (`__init__.py`'s module docstring is in sync). Verified split: elementwise 38 + comparison 6, structure 13, reduction 9, sorting 2, linalg 12, control 8 + terminator 1, collective 8, sparse 16, random 6, custom 2 — 12 categories, `{o.category for o in ir.all_opdefs()}`. Use `ir.op_names()`/`ir.all_opdefs()` rather than prose.

- **Generic inference + ops-side compensation (sanctioned division of
  labor):** the registry keeps generic hooks — `divide` →
  `infer_elementwise_binary` (plain `np.result_type`, int/int stays int),
  unary math ops (`sqrt`, `exp`, `log`, `log1p`, `sin`, `cos`, `tan`,
  `tanh`, `sigmoid`, `relu`, `gelu`, `erf`) → `infer_elementwise_unary`
  (dtype-preserving), `cumsum` → `infer_identity` — while `../ops` achieves
  its binding dtype rules (true division int/bool → float64; int/bool →
  float64 for unary math; bool cumsum → int64) by composing explicit
  pre-cast ops — documented in `../ops/CONTEXT.md` as sanctioned transparent
  sugar, not a bug. `abs` has its own dedicated `infer_abs` hook (complex →
  real magnitude dtype), so no compensation is needed there. If dedicated
  hooks (`infer_true_divide`, `infer_unary_math`, `infer_cumsum`) are ever
  added here, the ops compositions may be removed — but only when registered
  in `inference.py` + `op_defs/` (the numpy backend kernel must agree on
  either form).
- Builder attribute schema is stricter than inference in two places:
  `pad.padding_config` (ATTR_NESTED_INTS) requires nonempty integer sequence entries of any length
  (bare int entries fail `VerificationError`, though `infer_pad` accepts
  them); `slice.limit_indices` is schema-required and can be neither `None`
  nor contain `None` entries (the None-limit branch of `infer_slice` is
  unreachable via the Builder).

## Status

Phase 2 complete for this directory: SSA data model, op registry (121 ops),
shape-inference hooks (`inference.py`, 56 hooks), `pretty_print`, `verify`
(the full invariant set — module/function/region/op/value levels, SSA
dominance, use bookkeeping, shape_fn result-type agreement), the `Builder`,
and serialization (`serialize_module`/`deserialize_module` — payload schema,
integrity hash, round-trip rebuild with original ids and fast-forwarded
counters) plus the private trusted native decode
(`_native_decode.build_module_fast` — positional rebuild of a trusted
`evoxc-native-ir` v1 body, no validation and no `ir.verify`) are implemented.
