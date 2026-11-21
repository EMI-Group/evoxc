# evoxc/transforms — graph-to-graph transformations

## Intent

Frontend graph transformations: `vectorize` (the batching primitive), `vmap` (function-side sugar over it), `batch_functions` (merge N same-signature programs onto one static function axis), and automatic differentiation (`grad` / `jvp` / `vjp`).
Every transform is **graph → graph**: the result is an ordinary `Graph` of ordinary `evoxc.ops` ops, so no backend needs transform-specific runtime support (binding: `../../CONTEXT.md` design principle 7).
Nothing here executes: `evoxc.backends` / `evoxc.pipeline` are never imported and no tensor is materialized.
Given a plain function or `Defn` instead of a `Graph`, a transform returns a `TransformCallable` (`TensorSpec`s → `Graph`); concrete `Tensor` arguments raise `TraceError` (see "Spec-callable convention").

The **native Rust engine is the single implementation of all six transforms** — `vectorize`, `vmap`, `grad`, `jvp`, `vjp` and the `batch_functions` merge — reached through `evoxc/_native.py` (`transform`, `batch_functions`, `infer`).
Python in this package is the frontend seam only: validation and normalization with the pinned error wording, the native config, `TensorSpec`/pytree adaptation, `Graph` reconstruction, the extension-rule bridge, and report rehydration.
`evoxc._native.require()` raising `ImportError` with the build instructions (`scripts/build-native.sh`, the expected `evoxc/_evoxc_native.abi3.so`, the `EVOXC_NATIVE_LIB` override) is the only failure mode when the extension is missing — there is no Python-fallback implementation of any transform.
`import evoxc.transforms` does not require the extension: `evoxc/_native.py` loads it lazily on the first native call, so the rule registries and the frontend validators stay importable without it.

## Native seam (binding)

### Façade and wire format (`evoxc/_native.py`, `evoxc/_native_codec.py`)

- `transform(kind, module, config)` encodes the `ir.Module` to the `evoxc-native-ir` v1 envelope, calls the extension, and returns `{"module": ir.Module, "transform": {...}}`, where the metadata carries `kind`, `output_ids`, `input_ids` and `batch_dim` (validated, copied); any other top-level result key is preserved verbatim.
- `batch_functions(programs, config)` accepts `ir.Module`s or already-encoded envelopes and returns `{"module": ir.Module, "batch_report": {12 keys}, "input_ids": [...]?}`.
- The returned envelope is TRUSTED (produced in-process from the module just encoded), so it is rebuilt with the fast `_native_codec.decode_native_module`; `EVOXC_NATIVE_DECODE_CHECK=1` enables the verifying cross-check.
- `NativeError.kind` is mapped onto the public `evoxc.core` error family with the message preserved verbatim.
- `config` passes through untouched — a callable `config["extension_rule"]` is never encoded.
- The envelope itself, the transform configs and the extension-rule request/response protocol are documented in `../../native/CONTEXT.md`.

### The batching bridge (`_native_bridge.py`)

`call_native(kind, module, config)` is the one call path for `vectorize` / `vmap` (and the base for the AD helpers):
- for the batching kinds it supplies `batch_dim_name` from `batching._batch_dim()` (the CURRENT Python pass depth), so a transformed module never carries a stale depth record;
- it always supplies `extension_rule = _native_rules.extension_rule`;
- it resets the rule-failure slot before the call and, if the engine wrapped a Python rule's exception in its own `TransformError`, re-raises the rule's original exception (same message, own class);
- it cross-checks the returned `output_ids` against the decoded `return` operands and the `input_ids` count against the source tensor specs — a divergence is a `TransformError`, never a silently wrong `Graph`.

`rebuild_graph(source, module, transform)` wraps a transformed module back into a `Graph`:
- input specs are derived from the SOURCE specs, never re-decoded: a mapped leaf gains the unified batch `Dim` (dtype/device/name preserved) and an unmapped leaf reuses the spec object, so the caller's symbolic row dims survive by identity;
- the envelope round trip rebuilds every `Dim` from its name, so `_unify_batch_dim` gives every `Dim(batch_name)` in the module ONE canonical object — object identity is the documented mappedness contract;
- `source_locations` is carried old-block-arg → new-id positionally through `transform["input_ids"]`;
- the output tree is the source tree, with a registered pytree output node's static leaves remapped for the pass's batch `Dim` via `batching._remap_output_aux` (each node's `register_batched_aux_remap` callback); callers that change the output structure pass their own tree (the AD helpers build their trees directly rather than going through this function).

`vectorize_graph` / `vmap_graph` are the two batching compositions the public frontends call: normalized per-tensor entries → `input_axes_config` (one entry per block argument) → `call_native` → `rebuild_graph`.
`vmap` also passes the per-tensor-output `out_axes` entries; the output rearrangement (a leading transpose, a size-one axis insertion, the "a batch axis is never silently dropped" error) is native.

### Extension-rule bridge (`_native_rules.py`)

A handful of ops keep their batching/derivative rules in Python: custom blocks (`block:<name>`), external kernels (`external:<name>`) and the sparse ops (`sparse_*`, plus the sparse-category `sparse_dot_dense`/`dense_dot_sparse`).
For those the engine calls `extension_rule(request)` and expects `None` (no rule — the engine raises its explicit `TransformError` naming the op) or a response `{"module": <envelope>, "entry": <function>, "output_ids": [...], "output_axes": [...]?}` describing a straight-line subgraph.
Request keys: `kind` (`"batching"` | `"jvp"` | `"vjp"`), `op_name`, `rule_key`, `attributes`, `operand_types`, `result_types`, `axes`, plus `tangents`/`tangent_values` (jvp) and `cotangent_type`/`cotangent_values` (vjp).
Rule lookup is LATE (every call) against `batching.batching_rules` / `autodiff.jvp_rules` / `autodiff.vjp_rules`, so a rule registered after import is picked up.

Batching: the scratch module's block arguments are the op's REWRITTEN operands (exactly what the engine supplies), and the proxy op carries the original name/attributes plus detached result values of the request's result types, so a rule reads `op.name`, `op.attributes`, `op.results[i].type`.
The rule runs under `trace.with_builder`, its result is validated by `batching._validate_rule_result`, the values become the scratch `return` operands, and `output_axes` holds one `0`/`None` per returned value.
`output_axes` is OMITTED when a rule returns a multi-axis value (nested vmap), letting native derive mappedness from the result types' leading batch dims.

JVP / VJP: the engine passes the dual SSA values themselves — `tangent_values` (one entry per op operand, operand order) and `cotangent_values` (one entry per op result, result order) — so the scratch block arguments are, in order: the operands (n), the result proxies (r), then the NON-ZERO duals in list order.
A dual argument carries its primal's type; a zero slot contributes no argument and is bound as `autodiff.ZeroTangent`.
`jvp` uses the registered JVP rule, else derives the tangent from the registered VJP rule through `autodiff._jvp_from_vjp`; `vjp` requires a registered VJP rule; neither present returns `None`.
The response's `output_ids` holds exactly one entry per op RESULT (jvp, tangents) / op OPERAND (vjp, cotangents), in order, and is NULLABLE: a `None`/`ZeroTangent` rule entry is sent as `None` (native's structurally-zero dual) and the scratch subgraph returns only its real contributions, so no zero-materialization op is emitted — native materializes zeros only where a zero dual reaches a graph output.
A rule's own exception class is propagated verbatim; a rule that cannot answer raises the documented `TransformError` (never a silent fallback).

### AD delegation (`_native_ad.py`)

`grad_graph(graph, selected, single)` makes exactly one `call_native("grad", …, {"argnums": [...]})` call: result inputs are the original inputs, result outputs one gradient per selected index in selector order; the rebuilt output tree is a bare `type=None` leaf for a scalar `argnums` and a tuple of leaves for a sequence.
`sweep_graph(graph, kind, flat_specs)` makes exactly one `call_native(kind, …)`:
- `jvp` passes `tangent_inputs` = the flattened INPUT indices with an explicit tangent spec; result inputs are the primal inputs followed by those tangents, outputs the primal outputs followed by one tangent per tensor output.
- `vjp` passes `cotangent_inputs` = the flattened OUTPUT indices; the unlisted ones must be rank-0 and are seeded in-graph with a scalar one; outputs are the primal outputs followed by one cotangent per tensor input.
- Both rebuild the documented 2-tuple trees: input `(original input tree, flat extra tuple tree)` — a `None` extra spec becomes a recorded `None` static leaf — and output `(original output tree, flat extra outputs tree)`.
- The number of returned extra outputs is checked against the expected count (one per tensor output for `jvp`, one per tensor input for `vjp`), with the transform's own message on a mismatch.
There is no Python AD sweep on the production path and no retry: a native failure (missing rule, shape/dtype inference failure) surfaces verbatim.

### `batch_functions` frontend (`batch_functions.py`, `_native_batch.py`, `_report.py`)

`batch_functions(fns, axes=0, out_axes=0, *, switch_ops=None, bands=None, alignment="variant")` validates and normalizes on the Python side, then hands the programs to the native merge.
Python owns, before any native work:
- `fns`: a non-empty `collections.abc.Sequence` (`str`/`bytes` rejected) of traced `Graph`s OR of callables/`Defn`s — never mixed, with the offending element index named;
- `bands` / `alignment` / `switch_ops` validation (so a bad value raises before any tracing, the callable path included);
- `axes` normalization through `vectorize._normalize_axes` + `batching._tensor_input_axes` (leading axis only, `{0, None}`, static positions `None`), and `out_axes` normalization to one `0`/`None` per TENSOR output leaf with no output rank range-check (`0` on a rank-0 unmapped output is the legal "broadcast to the function axis" case);
- the callable path: `vmap._derive_unvectorized_args_and_entries` strips the leading function-axis dim per mapped spec, every function is traced exactly ONCE, and `axes` is never re-normalized against the traced graphs;
- the cross-function contract against function 0 — identical input/output tree structure, tensor-leaf count, per-leaf dtype/shape/device, and identical static leaves (messages render the first mismatch pytree path);
- the per-op batchability gate `_check_ops` (region-bearing `cond`/`while_loop`/`scan` and any op with no rule under `batching._rule_key`/`batching.batching_rules`), then the registered-output-node check, then the banded `out_axes=None` restriction — so the standard merge errors always win over a band-strategy restriction.

The native call is `_native_batch.batch_functions(modules, config)` → `evoxc._native.batch_functions`: config `{"axes", "out_axes", "switch_ops", "bands", "alignment", "extension_rule"}`.
Python then rebuilds the merged `Graph`: mapped input specs are function 0's with a STATIC leading `int` function axis (never a symbolic `Dim`), unmapped leaves reuse their spec object, the output tree and output static values are function 0's, and input source locations are carried through the reported `input_ids`.
`graph.batch_report` is rehydrated by `_report.report_from_native` into `BatchReport` / `SwitchRecord` and the report's `n_functions` is checked against the population size.
The merge semantics, the class-DAG/alignment rules, the constant stacking and muxing, the band partitions and the `bands="auto"` cost model are the native engine's: `../../native/evoxc-compiler/src/batch_functions/CONTEXT.md`.
User-visible summary: row i evaluates program i; identical programs produce one batched op per source op and ZERO `select`s; `switch_ops` turns same-skeleton different-op positions into one op per variant plus a static per-row `select` cascade (a switch class is ALWAYS mapped on the function axis); differing constant payloads stack into one `(N, …)` constant; any remaining divergence is `select`-muxed at its consumers/outputs; `alignment` is a PURE performance heuristic (every assignment is row-correct and bit-identical); `bands` trades dispatch count for re-computed cross-band work, and `bands=1` / an auto-resolved single band emit the byte-identical flat graph.
Measured guidance: banding wins only on DIVERGENT populations (`bands="auto"` picks the argmin on GP-shaped trees, 2.6–10.7× under flat; `identical` resolves to flat and any `K>1` is a loss), `switch_ops` should not be combined with banding, and the data-supported rule is `bands="auto"` on CPU and `bands=16` (or `bands=None`) on an accelerator.
A banded merge glues the per-band outputs with ONE variadic `concatenate`: IREE fuses the pieces into one dispatch whose binding count grows with K, and the HIP/CUDA HAL caps a dispatch at 16 bindings (llvm-cpu 64), so a caller must cap K (`15` on cuda/rocm, `31` on cpu; the merge cannot see the device).

The native merge owns rank-0 mapped outputs directly: a mapped `out_axes=0` on a rank-0 per-function output becomes rank 1 in the merged graph (the static function axis is prepended), so the façade is a single `_merge` call for every population.

## API surface (public — re-exported from `__init__.py`)

- `vectorize(graph, axes) -> Graph` — the primitive; a callable/`Defn` raises `TypeError` pointing at `evoxc.vmap`. `axes` is an `int`, `None`, or a pytree thereof over the input structure; `0` maps a tensor input's leading axis, `None` leaves it unmapped (mandatory for static inputs). Normalization lives in `vectorize._normalize_axes` (shared with `vmap` and `batch_functions`).
- `vmap(fn_or_graph, in_axes=0, out_axes=0) -> Graph | TransformCallable` — the same machinery plus native output rearrangement. A `TransformCallable` input composes (nested `vmap`, `vmap(grad(f))`, …) by invoking the inner callable at one deeper batch depth.
- `grad(fn_or_graph, argnums=None) -> Graph | TransformCallable` — reverse mode; exactly one scalar tensor output (`ShapeError` otherwise); `argnums` indexes the flattened TENSOR inputs (`None` = all) and each selected input must be floating/complex (`TransformError`).
- `jvp(fn_or_graph, tangents) -> Graph | TransformCallable` — tangents mirror the INPUT structure (`TensorSpec` or `None` = zero); result inputs = primal inputs then tangents, outputs = `(primal outputs, tangent outputs)`. A bare `TensorSpec`/`None` (and a 1-tuple) is accepted with exactly one tensor input.
- `vjp(fn_or_graph, cotangents=None) -> Graph | TransformCallable` — cotangents mirror the OUTPUT structure; `None` (the default) becomes an EXPLICIT scalar-one cotangent input and requires a single scalar output; result outputs = `(primal outputs, input cotangents)`. A bare `TensorSpec`/`None` (and a 1-tuple) is accepted with exactly one tensor output.
- `batch_functions(fns, axes=0, out_axes=0, *, switch_ops=None, bands=None, alignment="variant") -> Graph | TransformCallable` — see above; the `Graph` result carries `graph.batch_report`.
- `BatchReport` / `SwitchRecord` — frozen `SwitchRecord(group, variants, rows_per_variant)`; `BatchReport` with `n_functions`, `value_classes`, `shared_classes`, `switched`, `stacked_constants`, `output_muxes`, `emitted_ops`, `bands`, `band_sizes`, `band_strategy`, `cost_estimate`, `flat_cost_estimate` and `as_dict()` carrying exactly those 12 keys.
- `TransformCallable` — `__call__(*specs) -> Graph` (fresh graph per call), `.kind` (`"vmap" | "batch_functions" | "grad" | "jvp" | "vjp"`).
- Rule registries (public mutable dicts + register functions): `batching_rules` / `register_batching_rule`, `jvp_rules` / `register_jvp_rule`, `vjp_rules` / `register_vjp_rule`, `register_batched_aux_remap`.
- `graph.batch_report` is a plain attribute, never part of the persisted payload — a graph saved and loaded back has none.

## The vmap ⇔ vectorize equivalence contract (binding, tested)

1. **Graph case**: `vmap(graph, in_axes, out_axes=0)` produces the same IR as `vectorize(graph, in_axes)` — same ops, same input/output specs.
2. **Function case**: `vmap(f)(*batched_specs)` produces the same IR as `vectorize(trace(f, *stripped_specs), in_axes)` with `out_axes=0`, where `stripped_specs` drops the leading mapped dim of each mapped spec (dtype/device/name preserved).
3. **Rearrangement is post-processing**: `out_axes` handling happens AFTER the vectorize pass, as ordinary ops (a transpose of a mapped axis, a size-one axis insertion for an unmapped output), never inside batching rules; with `out_axes=0` on every mapped output no extra op is emitted at all.
4. **Once-only tracing**: the wrapped function is traced exactly once per `TransformCallable` invocation.
5. **Numerical check**: running the vmap graph on a batch equals running the unvectorized graph per row and stacking.

`out_axes` semantics (public v1): `0` keeps a mapped output's leading axis, or inserts a size-one axis for an unmapped output; `None` requires the output unmapped (a mapped output raises `TransformError` — a batch axis is never silently dropped). Non-zero entries are deferred at the public frontend, although the native engine implements the full normalization.

## Axis metadata (`MappedAxes`) and the batch dim

`_metadata.MappedAxes` is a tuple of leading mapped axis indices — `(0,)` after one vectorization, `(0, 1)` after nesting two, `()` = `UNMAPPED` — and `_metadata.ValueEnv` maps values to it for the Python rule path (`_native_rules`, `batching`, `rules`).
Rules receive operand metadata as arguments and never touch the env themselves.
One batching pass creates ONE fresh batch `Dim` shared by all its mapped inputs, named from the active pass depth (`batching._batch_dim`, a contextvars counter): `batch` at depth 0, `batch_1` at depth 1, …; `vmap._vmap_fn` increments the depth around a nested `TransformCallable`, and the name reaches the engine as `config["batch_dim_name"]`.
The name must be level-distinct because the numpy interpreter binds symbolic dims by name: equal extents would accidentally pass and unequal extents would raise a ShapeError on a valid graph.
Scope: the depth counter covers `TransformCallable` composition only — `vectorize(vectorize(g))` on the Graph path runs both passes at depth 0 and yields same-named dims, a documented v1 fail-explicit `ShapeError` for unequal extents.

## Rule-call signatures (binding)

Rules are pure graph builders: they emit ops into the active builder (native pushes its own; the Python bridge pushes an `ir.Builder` on the trace builder stack) and never loop over batch elements or swallow errors.
- **Batching**: `rule(op, operands, axes) -> (new_values, new_axes)` — `operands` are the REWRITTEN new-module values (rules must build over THESE, never over `op.operands`), `axes` their metadata; returns values aligned with `op.results` plus real `MappedAxes` (leading-contiguous) which the machinery validates.
- **JVP**: `rule(op, tangents) -> output_tangents` — `tangents` is a tuple of `ir.Value | None | ZeroTangent` aligned with `op.operands`; returns a tuple aligned with `op.results`.
- **VJP**: `rule(op, cotangents, primals) -> input_cotangents` — `cotangents` aligned with `op.results`, `primals` = `op.operands`; returns a tuple of `ir.Value | None | ZeroTangent` aligned with `op.operands`.
**Proxy-op convention**: rules never see the original op's values. They receive a fresh `ir.Op` (never inserted into a block) carrying the original `name`/`id`/`attributes`/`location` (attributes shallow-copied, shared read-only) but whose `operands`/`results` are the transformed graph's values; a bridge proxy's results are negative-id detached values. Rules must not follow `value.owner` off a proxy value.
The `constant` op is handled structurally by the machinery (fixed data ⇒ zero tangent; the batching rule treats it as unmapped) and has NO derivative registry entry.

## AD semantics (binding)

- `grad` requires exactly one scalar output (`ShapeError`) and returns the selected input gradients in selector order; the backward sweep, `jvp`'s forward sweep and every builtin derivative rule are the native engine's, with Python doing validation/normalization and `Graph` reconstruction.
- Multiply-used values accumulate cotangents with `add`; `stop_gradient` yields a zero gradient/tangent in both directions; boolean/int-producing ops (`equal`, `less`, `argmax`, …) yield `ZeroTangent` — not an error.
- Ops with no rule — `runtime_call`, collectives, `external_call`, `block_call` of a block without a registered rule — raise `TransformError` naming the op/block (external keys `external:<name>` also hint the external-kernel handle). Never a silent fallback.
- **JVP derived from VJP**: when only a VJP rule is registered, the tangent is derived by the adjoint trick instead of raising — the VJP rule's linear map `c ↦ Jᵀc` is instantiated with `c` = the op's primal results, then a second reverse sweep over the rule's emitted ops applies its transpose to the tangents, propagating cotangents only along paths reaching a result slot (primal paths are second-order terms and are dropped). All-zero tangents short-circuit to `ZeroTangent` without calling the rule.
- Extension-ruled graphs are NOT special-cased: their Python rules run inside the native sweep through the bridge with the dual SSA values, so `grad`/`jvp`/`vjp` have exactly one implementation.

## Measured cost (whole-graph AD on this host)

Measured with `perf_counter`, min/median of 7 reps, at loadavg ≈ 7, on a scalar rank-0 program (a chain of `sin(y) * 1.0001`, so 3 ops per iteration).
750-op graph (trace 5.88/6.23 ms): `grad` 14.48/15.97 ms, `vjp` 13.93/14.98 ms.
2250-op graph (trace 18.26/18.56 ms): `grad` 52.71/59.22 ms, `vjp` 52.64/53.11 ms.
Envelope cost on the same graphs: `encode_module` 2.75/2.78 ms (750 ops) and 10.39/10.51 ms (2250 ops); validated `decode_module` 5.83/5.99 ms and 20.48/20.81 ms; trusted `decode_native_module` 2.39/2.44 ms and 8.52/10.10 ms (≈2.4× faster than the validated decoder), which is the seam that keeps the native round trip cheap.

## Retained Python modules

- `autodiff.py` — the rule registries and shared helpers: `jvp_rules` / `vjp_rules` with `register_jvp_rule` / `register_vjp_rule` / `get_jvp_rule` / `get_vjp_rule` / `require_vjp_rule`, `ZeroTangent`, `_jvp_from_vjp`, `_rule_name` / `_rule_display` / `_rule_hint`, `_combine`, `_sym` / `_is_real`. Holds no sweep: every `grad`/`jvp`/`vjp` rewrite is native.
- `batching.py` — the shared per-op batching-rule registry and helpers, not a rewrite engine: `batching_rules` / `register_batching_rule` / `_rule_key`, `_tensor_input_axes`, `_validate_rule_result`, `_remap_output_aux`, the batched-aux remap registry (`register_batched_aux_remap`, `_lookup_batched_aux_remap`), the batch-depth context (`_batch_dim`, `with_batch_depth`), `_entry_function` and `block_call_pass_through_rule`. Consumers: `_native_rules`, `_native_bridge`, `vmap`, `batch_functions`, `rules`, `../block/rules.py`, `../external_rules.py`.
- `rules.py` — the builtin Python rule TABLES (batching, JVP, VJP for the standard op set) registered at import: the availability table `batch_functions._check_ops` consults and the Python mirror of the native engine's compiled-in builtin rules. A builtin rule gap must be closed natively AND mirrored here.

## Spec-callable convention (design decision)

Transforms are graph → graph, but `vmap(f)` / `batch_functions(fns)` / `grad(f)` / `jvp(f)` / `vjp(f)` on bare functions cannot know input shapes at transform time, and importing pipeline/backends is forbidden.
Therefore a callable/`Defn` input returns a `TransformCallable`: `tf(*specs) -> Graph`, with each public module's docstring stating the exact expansion (trace → transform → rearrange) so every composition stays transparent.
`evoxc.build(tf(*specs))` is the documented way to run the result.

## v1 scope vs deferred

- **Supported**: any straight-line graph whose ops all have a batching/derivative rule — the builtin op surface, plus `block:*` / `external:*` / `sparse_*` rules served through the bridge; nested vmap through `TransformCallable` composition; `batch_functions` over any program whose ops all have batching rules, with every merge strategy (flat, `bands=K`, `bands="auto"`) and every alignment.
- **Deferred → `TransformError`, never silent**: vectorizing region-bearing control flow (`cond` / `while_loop` / `scan`); non-zero `in_axes` / `out_axes` entries at the public frontend (mapped axes must be leading); `vectorize` of `runtime_call`; `external_call` with no `external:<name>` rule and no registered portable; ops without a rule (collectives, `random_*`, `sort`/`argsort`/`tile`/`flip`/`roll`/`diag`/`cumprod`/`nan_to_num`, the linalg factorizations, …).
- **`batch_functions`-specific deferrals**: an op without a batching rule (named with its function index), region-bearing control flow, a registered pytree OUTPUT node carrying static leaves (no batched-aux remapping is attempted), non-zero `axes`/`out_axes` entries, `out_axes=None` on a mapped or muxed output, and `out_axes=None` on ANY tensor output when `bands` is not `None`.
- **Rule-level v1 gaps** (native and Python tables must agree): `gather` batching with mapped data + unmapped indices; multi-axis gather/scatter transforms; `solve` batching/AD with `left_side=False`; `conv` VJP (the IR has no transposed-convolution op); `mean`/`concatenate`/`pad` VJPs over symbolic dims (static dims work); `vmap(vjp(f))` / `vmap(jvp(f))` composition (their callables produce extended input trees the outer axes normalization cannot address — plain `vmap(vmap(f))` and `vmap(grad(f))` compose fine).
- **Non-goals here**: no transformed-graph caching (design principle 3), no backend awareness, no lowering.

## Constraints

- **Imports (binding)**: only `evoxc.core`, `evoxc.ir`, `evoxc.trace`, `evoxc.ops`, the private native seam (`evoxc._native`, `evoxc._native_codec`) and this package's own submodules. NEVER `evoxc.backends` / `evoxc.pipeline` / `evoxc.dist` / `evoxc.block`. Block rules arrive through the registries (`block:<name>`), keeping `block → transforms` one-directional.
- Files < ~1000 lines: one file per public entry point, the delegation glue in `_native_bridge.py` / `_native_ad.py` / `_native_rules.py` / `_native_batch.py`, the report types in `_report.py`.
- **Errors**: `TransformError` (missing rule, axis/`out_axes` mismatch, non-differentiable argnums, native transform failure), `ShapeError` (`grad` of a non-scalar output, cotangent count mismatch), `TraceError` (concrete tensors to a `TransformCallable`, cross-function structure/static mismatch), `TypeError` (wrong argument kinds), `ImportError` (missing native extension, with build instructions). Messages name the offending op/block and carry source locations where available.

## Routing table (flat module — concerns → files)

| File | Area |
|---|---|
| `vectorize.py` | public `vectorize` + axes validation/normalization (registered-node broadcast, leading-axis rules); delegates the rewrite |
| `vmap.py` | public `vmap`, spec derivation (`_derive_unvectorized_args`, shared `_derive_unvectorized_args_and_entries`), `out_axes` structure/`{0, None}` validation, `TransformCallable` composition with the depth bump; delegates the rewrite and the rearrangement |
| `grad.py` | public `grad` + scalar-output/argnum/dtype validation; hosts the shared AD frontend helpers (`_return_values`, `_single_scalar_output`, `_normalize_argnums`, `_normalize_extra_pytree`) |
| `jvp.py`, `vjp.py` | public `jvp`/`vjp` + tangent/cotangent tree validation and normalization; delegate the sweep |
| `_native_bridge.py` | native call path and `Graph` rebuild: `call_native`, `input_axes_config`, `rebuild_graph`, batch-`Dim` unification, `vectorize_graph`, `vmap_graph` |
| `_native_rules.py` | extension-rule bridge: batching + JVP/VJP responses (scratch module, proxy op, dual binding, nullable output ids), schema-driven attribute decoding, rule-failure capture |
| `_native_ad.py` | AD delegation: `grad_graph`, `sweep_graph`, input/output tree construction |
| `_native_batch.py` | `batch_functions` façade over `evoxc._native.batch_functions`: config type check, report decoding, single native merge call |
| `_report.py` | `BatchReport` / `SwitchRecord` + `REPORT_KEYS` and `report_from_native` |
| `batch_functions.py` | public `batch_functions` + all frontend validation (fns/axes/out_axes/switch_ops/bands/alignment), cross-function contract, batchability gate, callable path, merged `Graph` rebuild |
| `batching.py` | shared batching-rule registry + helpers (`_rule_key`, `_tensor_input_axes`, `_validate_rule_result`, `_remap_output_aux`, aux-remap registry, batch depth, `_entry_function`, block pass-through rule) |
| `autodiff.py` | AD rule registries, `ZeroTangent`, `_jvp_from_vjp`, rule-display/hint helpers |
| `rules.py` | builtin Python rule tables (batching + JVP + VJP) registered at import |
| `_metadata.py` | `MappedAxes`, `UNMAPPED`, `ValueEnv` |
| `_wrappers.py` | `TransformCallable` (spec → Graph convention) |

Sibling: `../../tests/transforms/` → test suite for this module (read-only from here — escalate test-related writes to root).

## Test strategy

`../../tests/transforms/`: one file per public entry point (`test_vectorize.py`, `test_vmap.py`, `test_vmap_control_flow.py`, `test_grad.py` / `test_jvp.py` / `test_vjp.py` against finite differences, `test_equivalence_meta.py` for the vmap⇔vectorize IR contract, `test_autodiff_rules.py`, `test_errors.py`) plus the `batch_functions` layer (`test_batch_functions.py` for the merge contract and validation, `test_batch_functions_errors.py`, `test_batch_functions_integration.py`, `test_batch_functions_bands.py` for band validation/row-exactness/`bands=1`≡`bands=None`/auto decisions, `test_batch_functions_alignment.py` for variant-vs-occurrence op counts, `test_batch_functions_speed_pins.py` for the exactly-one call-site location capture and the exact `emitted_ops` counts).
The native-vs-reference numeric gate lives in the native workspace: `native/evoxc-compiler/tests/parity_transform/run_parity.py` (61 cases, 122 mode-runs, 0 mismatches) and `run_batch_parity.py` (25 populations, 87 checks, 0 mismatches), plus the `cargo test` gates `--test transform_parity` / `--test batch_functions_parity`.
Both need `scripts/build-native.sh` first; run them with `source scripts/dev-env.sh && .venv/bin/python <harness>`.
Spec-compliance additions live in `../../tests/test_spec_compliance.py`; the per-suite inventory is owned by `../../tests/CONTEXT.md`. CPU only, pytest.

## Known Issues

- **`vmap.py`'s `_collect_batch_dims` / `_is_batch_entry` / `_dims_in_batch` / `_leading_batch_count` are off the active path**: mapped-output detection is native; these helpers only reference each other.
- **The Python batching table must track the native builtin set**: `batch_functions._check_ops` accepts an op only when `batching.batching_rules` has its key, so an op with a native batching rule but no `rules.py` entry is wrongly rejected as a v1 deferral. Adding a builtin batching rule means updating `rules.py` (and vice versa) in the same change.
- **AD registration is table-driven — a category-tuple op without a rule-table entry is a latent bug**: every op in `ELEMENTWISE_OPS` and the other category tuples is auto-registered to the generic handlers, which raise `TransformError` at rule-call time unless the op also appears in `_UNARY_DERIV` / `_POINTWISE_JVP` / `_POINTWISE_VJP`. When adding an op to a tuple, add its rule-table entry and mirror the rule natively.
- **`bands="auto"` prices a dispatch at the CPU crossover (`DISPATCH_COST = 512`), so its pick can overshoot on an accelerator**: dispatch is nearly free on a GPU and the model is static program metadata with no backend input, so `k=N` can lose to flat there (measured earlier on iree-ROCm). The data-supported rule stays `bands="auto"` on CPU, `bands=16` (or `bands=None`) on an accelerator; a backend-aware dispatch price is the known shape of the fix and is not implemented.
- **Mapped inputs of one pass share one runtime batch extent (intended)**: the per-pass shared `Dim` makes the numpy interpreter's name-keyed input binding the within-level guard, so a mismatched extent raises a clear `ShapeError` ("dimension 'batch' is already bound to 4, got 1") instead of silent numpy broadcasting.
- **No generic pass machinery exists here (no DCE, no CSE, no region rewriting)**: the rewrites are the fixed-purpose native passes (vectorize/vmap/AD/batch_functions); Python only serves extension rules. Region-bearing ops are hard-REJECTED, not walked. A graph-level transform that must inspect or rewrite `while`-loop carried values (e.g. packing carries to shrink a `stablehlo.while`) would be the FIRST region-walking transform and must build its own region-rewrite utilities.
- **The native AD sweep requires a rule for every non-constant source op before rebuilding**, even when its dual is zero or the result is irrelevant to the selected gradient — there is no source-level dead-code or activity analysis.
- **`vjp(..., cotangents=None)` adds an explicit scalar cotangent input**, whereas `grad` and an explicit per-output `None` cotangent seed an in-graph scalar one; the default form therefore still requires the output to be a single scalar.

## Notes for agents

- **Registered pytree nodes in axes normalization**: a LEAF axes entry (int/`None`) at a registered-node position broadcasts across the node's leaves — tensor leaves get the entry, static leaves get `None`; a bare int/`None` works for any tree containing a registered node. A CONTAINER entry at a registered-node position raises `TransformError` (per-leaf axes for a registered node are v1-deferred). Implemented in `vectorize._normalize_axes` / `_broadcast_registered_axes`, shared by `vmap`.
- **`TransformCallable.__call__` rejects concrete `Tensor` leaves** (`TraceError`) before invoking its stored build closure; it has no active-trace check, symbolic-input adaptation or transform plan.
- Only `vmap._vmap_fn` composes another `TransformCallable`; `grad._grad_fn`, `jvp._jvp_fn` and `vjp._vjp_fn` pass their wrapped object straight to `trace`.
- The callable `vmap` path discards mapped concrete extents when stripping specs; native introduces fresh symbolic batch dims rather than specializing to those extents.
- **The extension bridge is the only place Python rules run inside a native transform**: a new extension op family must register under a key the engine routes to the callback (`block:*`, `external:*`, `sparse_*`) and declare its attributes in `ir`, or the decoded proxy op will be missing them. Attribute decoding is schema-driven from `ir.opdef(...)`; tuple attributes travel as `{"__evoxc_tuple__": [...]}` and ndarray attributes as inline `dtype`/`shape`/`base64` records.
- `batching.block_call_pass_through_rule()` is the reusable batching rule for elementwise / map-over-batch block calls: it rebuilds the `block_call` with `result_specs` == result types and batch dims from the operand with the most mapped axes. `../block/` pre-registers it under `block:<name>`; the native engine reaches it through `_native_rules` like any other extension rule.
- **Portable-decomposition VJP fallback seeding (binding)**: `../block/rules.py::_portable_vjp_rule` seeds its local reverse sweep on the decomposition's OUTPUTS (the post-inline values), never on the `block_call`'s own result ids (which are not inserted into the builder; seeding on them would yield all-zero gradients). There is no repair layer in transforms.
- `evoxc.trace` the ATTRIBUTE is the trace function (it shadows the submodule) — inside scripts use `from evoxc.trace import trace as trace_fn`.
- `evoxc.transforms.vectorize` the NAME is shadowed by the re-exported FUNCTION inside the package: import private helpers explicitly (`from evoxc.transforms.vectorize import _normalize_axes, _normalize_axis_entries, _is_int`).
- `batch_functions` has NO Python merge implementation: the merge, its strategies and its report are native only, so a historical Python emitter/analyzer exists solely in git history.
- **When adding a public name here, update `__init__.py`, this CONTEXT.md, and the parent contract `../CONTEXT.md` together.** `evoxc/__init__.py` (parent-owned) re-exports `vectorize`, `vmap`, `batch_functions`, `BatchReport`, `SwitchRecord`, `grad`, `jvp`, `vjp`, the three registries + register functions, and `TransformCallable`.

## Representation and extension boundaries

The seam is language-neutral: `evoxc/_native_codec.py` plus `evoxc/_native.transform`/`batch_functions` are the whole boundary, and the `evoxc-native-ir` v1 envelope (documented in `../../native/CONTEXT.md`) is the only representation the engine sees.
The Python side of that seam is object-identity-bound: `rebuild_graph` rebuilds trees and input specs from the SOURCE graph, unifies the decoded batch `Dim`s and remaps `source_locations` through the native `input_ids`; nothing downstream may assume a decoded `Dim` is the source's object.
Pytree handling is strictly LEAF-LEVEL: transforms flatten axes/tangent/cotangent pytrees with `core.flatten`, validate structure with `core.first_mismatch_path`, and carry the graph's `TreeSpec`s forward verbatim (plus the registered-node aux remap); only `vmap._derive_unvectorized_args` rebuilds a user pytree with `core.unflatten`.
Per-op rules never see pytrees (flat `ir.Value` tuples plus an attributes dict), so a structured representation must carry shape data in operand value types or attributes and rules must prepend batch dims to result types themselves.
The public mutable rule dictionaries permit direct replacement as well as registration, while the native engine's builtin rules are compiled in and never observe those dicts — a separately cached Python rule table can go stale, and a rule gap must be closed natively to affect a transform.
