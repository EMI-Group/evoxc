# evoxc/trace — tracing machinery (defn, trace, Graph, control flow)

## Intent

The tracer: turns a plain Python function into an EvoXIR graph by executing
it ONCE under an active `ir.Builder`. Owns `@evoxc.defn`, `evoxc.trace`, the
`Graph` type (IR module + structured I/O trees), the active-builder context
(the hook `evoxc.ops` uses), and runtime tensor control flow (`cond` /
`while_loop` / `scan`, traced into `if`/`while` IR regions).

Principles honored here (see root `../../CONTEXT.md`): **no implicit eager**
(`Defn` calls always raise), **explicit staging** (`trace` never
verifies/compiles), **Python value → Python semantics** (static values
specialize the graph; only tensor values need explicit control flow), and
**no magic** (no caching, no hidden re-traces).

Architecture rule for this module: **all trace-time state lives in one
`_TraceSession` object; there is no global state besides the contextvars
builder stack.** See "Data structures & ownership" below — that section is
the map you need before touching any file here.

## API Surface

Exported from `__init__.py` (the public contract — exact names, unchanged
by the 2025-08 cleanup refactor):

- `defn(fn=None, **options) -> Defn` — decorator (bare or with options).
  `Defn` stores `fn` (+ `__evoxc_defn__` marker on both), `options` dict
  (reserved for future defn-compiler config; ignored in v1). Calling a
  `Defn` ALWAYS raises `core.TraceError` directing to
  `evoxc.trace(defn, *specs)` / `evoxc.evaluate(defn, *args)`. Idempotent.
- `trace(fn_or_defn, *specs) -> Graph` — THE tracer (algorithm below).
  Accepts `Defn`, `__evoxc_defn__`-marked objects, or plain callables.
- `Graph` — attrs: `module` (`ir.Module`), `input_specs` (`core.TreeSpec`
  whose leaves are the original `TensorSpec`/static objects),
  `tensor_specs` (tuple of `TensorSpec` in block-arg order),
  `output_tree` (`core.TreeSpec`), `static_values` /
  `output_static_values` (tuples of `StaticValue` records: index, path,
  value, kind), `source_locations` (dict: input value ids →
  `ir.Location`). Methods: `print()`, `text()`, `verify()`, `save(path)` /
  `load(path)` (classmethod; lazy-import `evoxc.persist`),
  `flatten_inputs(args) -> list[core.Tensor]`, `validate_inputs(args)`
  (alias), `unflatten_outputs(flat) -> structured`, `signature_info()`,
  `summary()`, `digest()` (see "Inspection" below).
  Constructor accepts a prebuilt module + trees (transforms build Graphs
  directly).
- `current_builder() -> ir.Builder` — innermost active builder; raises
  `core.TraceError` when no trace is active (this is `evoxc.ops`'s hook).
- `with_builder(builder)` — nestable context manager installing the active
  builder; alias `builder_stack`.
- `cond(pred, true_fn, false_fn, *operands, **static_kwargs) -> outputs`
- `while_loop(cond_fn, body_fn, init) -> outputs`
- `scan(f, init, xs, length=None) -> (carry, stacked_outputs)`
- `StaticValue` — frozen dataclass record for static leaves.

## Inspection (pure, side-effect-free)

`Graph` and `Defn` expose read-only inspection; none of it traces, compiles, mutates the IR, or touches a cache.

- `repr(graph)` — one line: total op count across every nesting level (nested `if`/`while` bodies, hence `cond`/`while_loop`/`scan`, included), function count, tensor-input count, static-value count, tensor-output leaf count, plus a `batch report` marker when `batch_functions` attached `graph.batch_report`. Keeps the pinned substrings `<n> tensor inputs` / `<n> static values`.
- `graph.summary() -> dict` — a JSON-safe structural summary with exact keys: `functions` (list of names), `op_count` (all nesting levels), `op_counts_by_category` (sorted dict; `ir.has_opdef`-guarded `ir.opdef(op.name).category`, unregistered names bucketed under `"unknown"`), `tensor_inputs` / `tensor_outputs` (lists of `{"path": list, "shape": [str(dim)…], "dtype": str}` in `tensor_specs` / `return`-leaf order, paths from the input/output trees), `static_values` (list of `{"path", "kind", "value_repr"}` with `repr(value)` truncated to 80 chars), `source_location_count`, `has_batch_report`, `digest`.
- `graph.digest() -> str` — SHA-256 hex over the canonical JSON (`sort_keys=True`, compact separators, length-prefixed) of TWO parts: the module's STRUCTURE and `signature_info()`.
  The structural part is `ir.serialize_module(self.module)` with each op entry's `location` removed and the payload's own `sha256` field dropped, so it covers only module/function names + metadata, block arguments (SSA ids + types), every op's name, attributes, operand ids and result ids/types, the nested regions, and the constant table.
  Process-stable and identifying graph STRUCTURE + I/O signature only: `source_locations` (the `file:line` a graph was DEFINED at), the location-derived payload hash and `batch_report` are excluded, and the module carries no path- or memory-address-dependent data, so the same graph defined in a DIFFERENT FILE, re-traced in a different PROCESS, or deserialized from disk digests identically.
  Structural changes — an op name, an attribute value, a dtype/shape, a static value, the I/O trees — change the digest; `summary()["digest"]` is this same string.
  `digest()` is the ONE inspection helper with a validation side effect: `ir.serialize_module` runs `ir.verify(module)` first, so digesting an invalid/unverified module raises `core.VerificationError` (e.g. a Graph a new transform built but has not verified).
- `graph.text() -> str` — `ir.pretty_print(self.module)`; the same string `print()` writes to stdout (`print()` itself is unchanged).
- `repr(defn)` — `<Defn <qualname> (<__module__>)>` (module dropped when absent), metadata-only; keeps the qualname substring.

## Module layout & routing table

Flat module — no child directories. Responsibility split (single owner per
concern):

| File | Owns |
|---|---|
| `defn.py` | the `Defn` marker decorator (nothing else) |
| `builder.py` | the active-builder context (contextvars tuple stack) + `_return_terminator` (the canonical block-ending op) |
| `_tree.py` | ALL shared pytree/static helpers: `_is_static_value`, `_static_equal` (array-aware static-leaf equality), `_registered_pytree_base`, `_flatten`/`_flatten_into` (parameterized leaf policy), `_iter_leaf_paths`, `_to_symbolic`, `_PYTREE_NODE_REGISTRY` alias. Imports core/ir ONLY — any trace module may import it. |
| `trace.py` | the 7-step tracer: `_TraceSession`, input classification, output classification, `_trace_call_site`. Re-exports `_format_path` / `_iter_leaf_paths` for `evoxc.transforms.grad` (private cross-module import — keep the names and paths stable). |
| `graph.py` | the `Graph` type: I/O validation, `StaticValue` records, `_static_record` factory, persistence delegation. Also exports `_normalize_leaf_types` (imported by `evoxc.transforms.vmap` — keep stable). |
| `control_flow.py` | `cond`/`while_loop`/`scan`: region building via the `ir.Builder` (`_run_in_region`), operand classification (`_classify_operands`), registered-node rules. NEVER imports `evoxc.ops` (ops imports trace — the DAG stays acyclic). |
| `__init__.py` | re-exports only + the shadowing note |

Sibling: `../../tests/` → test suite (read-only from here; escalate test
writes to the package root).

Private cross-module imports that MUST keep working (do not rename/move
without updating consumers): `evoxc.pipeline` imports
`_SymbolicLeaf`/`_TensorSpecLeaf` from `evoxc.trace.trace`;
`evoxc.transforms.grad` imports `_format_path`/`_iter_leaf_paths` from
`evoxc.trace.trace`; `evoxc.transforms.vmap` imports `_normalize_leaf_types`
from `evoxc.trace.graph`.

## Data structures & ownership (the map)

**`_TraceSession`** (`trace.py`) — the single named owner of one `trace()`
call. Fields: `builder`/`module`/`function` (the IR under construction; ONE
builder serves the whole trace — control flow re-positions it via
`push_region`/`pop_region`, never creates a second), `input_tree`/
`tensor_specs`/`static_values` (the classified input contract),
`symbolics` (block-arg SymbolicTensors, one per tensor spec),
`source_locations` (input value id → trace call-site), `args` (the
reconstructed call arguments), `call_site` (`ir.Location`). Lifecycle maps
1:1 onto the trace algorithm: `open(specs)` = steps 2–4, `run(fn)` = step
5, `finish(outputs)` = steps 6–7. One session per call — traces are never
cached or reused.

**Builder context** (`builder.py`) — a `contextvars.ContextVar` holding an
immutable tuple stack (outermost first). `with_builder` copies the tuple on
push and restores the saved parent on exit (even on error) — never mutated
in place, so nested contexts and async tasks cannot corrupt each other.
`current_builder()` returns the innermost builder or raises `TraceError`.

**Leaf markers & static records** — trace trees record tensor positions as
plain non-dataclass markers (`_TensorSpecLeaf` carrying the `TensorSpec`,
`_SymbolicLeaf` carrying the `SymbolicTensor`) so leaf-type equality
distinguishes tensor leaves from static leaves while the TreeSpec stays
`core.unflatten`-compatible. Static leaves are recorded as the object
itself with `TreeSpec.type = type(obj)` (pipeline's `_is_tensor_leaf_spec`
relies on this: any non-marker, non-None leaf type = static position).
`StaticValue` (`graph.py`) is the run-time validation record: frozen
dataclass with `index` (flat leaf index), `path`, `value`, `kind`
(`type(value).__qualname__` — so a recorded `1` never matches run-time
`True`). The single construction site is `graph._static_record(index, path,
value)`.

**The shared walker** (`_tree.py::_flatten`/`_flatten_into`) — one pytree
walk with two knobs so every caller's leaf convention stays explicit
instead of being copy-pasted:
- `leaf_spec(obj) -> (recorded_leaf, TreeSpec) | None` — decides which
  objects are leaves and what gets appended (trace: marker leaves; control
  flow: plain leaves for `_LEAF_TYPES`).
- `plain_leaf_type(obj) -> type` — the `TreeSpec.type` for fallback leaves
  (objects the policy declined that are not containers). trace passes
  `type` (static leaves record their own Python type — pipeline depends on
  it); control flow uses the default constant `None`.
Container descent mirrors `core.tree._flatten_into` (registered nodes via
MRO walk first, namedtuple before tuple, user dataclasses — evoxc-module
dataclasses stay leaves, tuple/list/dict with sorted keys). Deliberate
difference: no `defaultdict`/`Counter` default-factory handling in
`node_data` — that is a `core.flatten`-only feature.

## Trace algorithm (binding contract)

1. Unwrap `Defn`/`__evoxc_defn__` → plain fn (`_unwrap_defn`).
2. Treat `specs` (positional args) as ONE pytree; flatten via the shared
   walker with the trace leaf policy. Leaves must be `TensorSpec` (tensor
   input; shape may hold `Dim`/`DimExpr`, `None` = dynamic) or static
   Python values (`None`/bool/int/float/complex/str/`Enum`/numpy `dtype`/
   `slice`/`ndarray`/`core.Dim`/`core.DimExpr`/`core.Device` — see
   `_is_static_value`). Anything else (concrete `Tensor`, `SymbolicTensor`,
   …) → `TraceError` with the pytree path. Capturing a concrete tensor is
   NEVER silently allowed.
3. Build `ir.Module` + entry `ir.Function` "main" with one block arg per
   tensor leaf (arg type = (shape, dtype)). Wrap each block arg as
   `core.SymbolicTensor`; record trace-call-site `ir.Location`s.
4. Reconstruct args (unflatten: SymbolicTensors at tensor positions,
   original static values at static positions).
5. Call `fn(*args)` ONCE under `with_builder(builder)`. Normal Python
   execution — static control flow specializes the graph; ops build IR via
   `current_builder()`. Closure-captured concrete tensors fail inside ops
   (ops' `TraceError`) — the tracer does not pre-scan closures.
6. Flatten outputs (pytree). `SymbolicTensor` leaves → results; static
   leaves → `output_static_values` (re-inserted by `unflatten_outputs`);
   anything else → `TraceError`. Emit the function `return` terminator with
   the symbolic leaves (zero-result graphs are legal).
7. Return `Graph(...)` — NOT verified automatically (staging explicit;
   `evoxc.build` verifies). Every `trace` call yields a NEW Graph.

## Static-value snapshot semantics

Closure static values are naturally snapshotted because the function body
runs once at trace time — the values are read when ops are built, and only
explicit tensor values ever reach the IR. Run-time graph specialization is
validated by `Graph.flatten_inputs` (static leaves must match recorded
type+value; ndarray leaves compare via the shared `_static_equal` —
`np.array_equal` semantics, since plain `==` on arrays is elementwise →
ambiguous truth value; dtype/shape/device checked against specs with
`DTypeError`/`ShapeError`/`DeviceError`; tree mismatch → `TraceError`).
numpy arrays are accepted at run time and wrapped via `core.from_numpy`
(documented convenience) — that is the TENSOR path (via `TensorSpec` or
`evoxc.evaluate`'s spec derivation), distinct from an ndarray STATIC leaf,
which specializes the graph and never enters the IR.

## Callable protocols and nested tracing

`trace.py:301–316` only unwraps `Defn` or truthy `__evoxc_defn__` objects with a non-None `.fn`; it does not consult `__evoxc_trace_graph__`, validate callability up front, or recognize a returned `Graph` specially.
`Defn.__call__` always raises, including inside an active trace, and control-flow callbacks are invoked directly without Defn unwrapping.
An ordinary Python helper called with symbolic arguments naturally appends ops into the current builder, but this is Python execution rather than an explicit graph-inlining protocol in trace/.
A nested explicit `trace(helper, *specs)` creates an independent session/module/builder and temporarily shadows the parent builder; `with_builder` restores the parent on exception as well as success.
Nested explicit traces reject symbolic arguments as specs, and `_classify_outputs` records symbolic `.value` references without checking that they belong to its session/module, so cross-session symbolic capture must not be mistaken for supported graph inlining.
The builder ContextVar isolates stack values, not the mutable Builder objects referenced by inherited async contexts, so concurrent construction through an inherited builder is not made safe by immutable stack tuples alone.

## Compiler-core boundary and static ownership

`Graph` combines a numerical `ir.Module` with a Python-facing structured-call adapter, rather than being a language-neutral compiler input by itself.
`Graph.signature_info()` exports encoded TreeSpecs and static values alongside flat tensor specs, so its full signature is not just a flat numerical ABI.
`Graph.verify()` verifies only the IR module; the constructor assigns metadata without checking agreement between trees, static records, and module arguments/results.
`_static_record` stores the original static object by reference, and `_reconstruct_args` passes that same object to the traced function; accepted mutable ndarray static leaves are not defensively copied, despite the snapshot terminology elsewhere in this file.
`StaticValue.kind` uses Python `type(value).__qualname__`, and static equality uses Python equality or `np.array_equal`, so these specialization guards belong to the Python frontend rather than a portable compiler schema.
The callables used by `cond`, `while_loop`, and `scan`, including `_run_in_region`'s `after` callbacks, execute during tracing to build regions and are not runtime callback implementations.
A language-neutral compiler seam can consume the module, flat tensor argument/result types, and source locations while leaving pytree reconstruction, Python static guards, callable execution, and Python persistence metadata in a frontend-owned wrapper.

## Graph layout

Entry `ir.Function` "main": block args = tensor inputs in `tensor_specs`
order; terminator `return` yields results in output-tree SymbolicTensor-leaf
order. `input_specs`/`output_tree` carry the structured I/O contract
(tuple/list/dict/namedtuple/dataclass — pytree machinery from `core`).
Static leaves live ONLY in `static_values`/`output_static_values` records,
never in the IR. `unflatten_outputs` interleaves tensor results and static
records in ONE pass over the combined leaf positions (validates indices in
range, no duplicates).

## Control-flow region conventions (binding — implemented against the real `evoxc/ir` registry)

Region ops are built via `ir.opdef(name)` + the `ir.Builder` — this module
NEVER imports `evoxc.ops` (ops imports trace — keep the DAG acyclic). The
conventions below are what the IMPLEMENTED `ir` registry + `verify` enforce
(they differ from earlier architecture assumptions — `ir` is authoritative):

- **Region execution** is unified in `_run_in_region(builder, region, fn,
  *args, after=None)` (`control_flow.py`): it pushes the region's entry
  block onto the builder's insertion-point stack (creating the block first
  if the region is empty), installs the builder as the active builder for
  `current_builder()` (the SAME builder — control flow never creates a
  second), runs `fn`, then calls `after(result)` still inside the region
  (callbacks emit the `return` terminator via `_return_terminator`), and
  pops the region in a `finally` so the stack stays consistent on error.
- `if` op ("if", effects none, 2 regions, arity (1, None)): **operand 0 IS
  the boolean predicate**; operands 1..n are the captured tensor values.
  `ir.verify` binds EVERY operand (predicate included) to each region's
  entry-block args — one arg per operand, identical count and types. Branch
  callables therefore receive the REGION BLOCK ARGS (wrapped as
  SymbolicTensor), not the enclosing captured values. Each block's `return`
  terminator yields the branch outputs (identical trees, unified
  dtypes/shapes across branches — mismatch raises `DTypeError`/`ShapeError`/
  `TraceError`); explicit `result_types` are passed to `create`. Branch
  callables run once each at trace time.
- `while` op ("while", effects none, 2 regions, arity (1, None)):
  operands = n initial carried values (≥1 — all-static init raises
  `TraceError`); regions = condition, body — one block each with n block args
  bound to the operands (same verify convention). `shape_fn=infer_identity`
  → op results = operand types. Condition region returns ONE 0-d bool; body
  region returns n next-carried values. Loop-carried types must stay constant
  across iterations (checked at trace time). cond_fn/body_fn run once each.
- `return` op: terminator-only (operands = yielded values), emitted via
  `_return_terminator(builder, values)` (`builder.py`) which delegates to
  `Builder.set_terminator` — it APPENDS as the block's last op (never via
  `create` at the insertion point).
- Registered pytree nodes (types registered via `core.register_pytree_node`,
  e.g. sparse tensors) are valid carried values / branch outputs /
  scan carries: the shared walker consults the registry FIRST (MRO walk over
  `core.tree._PYTREE_NODE_REGISTRY`), the node's registered `flatten_fn`
  yields its children (SymbolicTensor leaves + static leaves) which recurse
  and classify as usual, and reconstruction goes through `core.unflatten`
  with the registered (polymorphic) `unflatten_fn` — tensor children become
  region block args / op results, static children pass through unchanged.
- `scan` desugaring uses these registered op defs (raw, via `ir.opdef`):
  `constant` (int32 0/1/length scalars), `add` (counter increment), `less`
  (counter < length), `gather` (`axes=(0,)` with a 0-d int32 counter — the
  DYNAMIC index step; `slice` is static-int-only and unusable here),
  `reshape` (leading 1-dim), `broadcast` + `scatter` (fixed-size stacking —
  a grow-by-`concatenate` stack would change the carried type per
  iteration, which a typed `while` op cannot carry). The registry has NO
  `expand_dims`.
  Step 0 runs pre-loop via `gather` with a constant-zero index;
  `length == 0` raises `TraceError` (v1).

`scan` invokes its user function twice during tracing: step zero directly at the enclosing insertion point (`control_flow.py:835–837`) and once inside the generated while body (`control_flow.py:897–909`), including when `length == 1`.
Step-zero indexing uses `gather` with a constant-zero index (`control_flow.py:824–832`), not `slice`.
The initial-to-step-zero carry check compares structure and allowed leaf kinds, while subsequent while-body checking compares against `carry0`, so the original init-to-first-result dtype/shape/static equality is not comprehensively enforced (`control_flow.py:839–854`, `945`).
`_run_in_region` invokes its `after` callback after exiting its own `with_builder`, while the insertion-point region remains pushed until `finally` (`control_flow.py:240–248`); current finish callbacks use their explicit builder rather than relying on that inner active-builder scope.

`scan` v1 scope: STATIC length only (an explicit static `int`, or derived
from xs's static leading dim). An explicit static `length` SMALLER than a
statically-known leading dim is allowed and runs a PREFIX scan over the
first `length` elements (stacked outputs have shape `(length, ...)`); only
an explicit `length` LARGER than a statically-known leading dim raises
`TraceError`. Symbolic length → `TraceError` (dynamic-scan region ops
reserved; no silent fallback).

## Structured weight-DICT inputs (hundreds of named tensors)

The supported pattern for "one dict of N named weight tensors → one graph": pass the dict as ONE positional `specs` argument and declare the defn with one matching parameter.

- `trace(fn, *specs)` treats the whole `specs` TUPLE as one pytree (`_flatten_specs`, `trace.py:333`), so `evoxc.build(model, {"w1": TensorSpec(...), ..., "w400": ...})` gives the traced function its dict back as its single argument — N tensor leaves, N block args, N flat leaves, no cap anywhere (`return` op arity is `(0, None)`, function args unbounded, IR/persist serialization is linear JSON+base64).
- Nesting is arbitrary (dict/list/tuple/namedtuple/dataclass, mixed, and a registered pytree node's children), and dict KEY ORDER DOES NOT MATTER: `_tree.py:244` sorts keys, so insertion order never affects `tensor_specs`/block-arg order and a run-time dict may be built in any order.
- Static leaves may sit beside tensors in the SAME dict: each `_is_static_value` leaf (None/bool/int/float/complex/str/Enum/np.dtype/slice/`core.Dim`/`core.DimExpr`/`core.Device`/np.ndarray) is recorded in `static_values` and re-validated (kind + value) on every run — it specializes the graph, never occupies a block arg. A small config object must be a user dataclass with all-static fields (it is then descended field-wise), not an opaque object (opaque objects → `TraceError`).
- Symbolic dims: `evoxc.dim("batch")` / `Dim(name, size)` / `DimExpr` arithmetic, freely mixed with int and `None` dims inside one shape. `Dim` equality is `(name, size)` structural, so separately constructed same-name `Dim`s unify — reuse of one object is NOT required. Caveat: run-time shape checking is PER-LEAF and stateless (`graph._check_shape`, `pipeline._validate_tensor_meta`) — there is no dim-binding environment in v1, so the same `Dim("B")` across two inputs is NOT cross-checked, and `Dim(name)`-without-size / `None` dims accept ANY size. Tying dims together is only enforced by trace-time shape inference in `ir`.

## Caching & performance

Source-location capture for frontend ops is owned by `../ops/CONTEXT.md`, not by trace/; use that context for its current implementation and measured cost.

Caching that EXISTS in this module (all semantic-neutral dedup):
- `_classify_operands` (`cond`): every container operand is flattened
  EXACTLY once per `cond` call (was ~5×: validation + operand list + once
  per branch); `_leaf_registered_flags` is computed once per `cond` call.
- `Graph.unflatten_outputs`: single pass interleaving tensor results and
  static records (was O(n·s) repeated list inserts).
- `_return_terminator`/`_run_in_region`: one canonical spelling each for
  block-ending and region-running (dedup, not speed).

Caching deliberately NOT added: per-`Graph` input-validation plans
(`flatten_inputs` plan caching). The dominant per-call cost is the input
pytree walk itself, which a plan cannot remove, and `evoxc.pipeline`
already caches input/output plans per `Executable`/`BoundExecutable` — a
second cache here would just duplicate that. Traces themselves are never
cached (every `trace()` call builds a fresh Graph — that is the contract).

## Error cases (all public errors derive from `core.EVOXCError`)

`Defn.__call__` (always), `current_builder()` outside trace, invalid trace
input/output leaves, branch tree/arity mismatch or non-scalar-bool `pred`/
`cond_fn` result, static-kwargs that are not static, symbolic scan length
(v1) — all → `core.TraceError` with pytree path / graph location where
applicable. `flatten_inputs`: `ShapeError` (shape vs DimExpr), `DTypeError`,
`DeviceError`, `TraceError` (tree/static mismatch). `Graph.load` mismatch →
`PersistenceError` (from persist). `verify()` → `VerificationError`.
Error messages are byte-stable (the test suite matches substrings) — when
editing messages, keep the wording of existing paths intact.

## Constraints

- Imports: `evoxc.core` and `evoxc.ir` ONLY (plus stdlib/numpy). `evoxc.persist`
  is imported lazily inside `Graph.save/load` (persist does not import
  trace). NEVER import `evoxc.ops` — ops imports trace for
  `current_builder()`; control flow builds raw ops via `ir.opdef`.
- Files < ~1000 lines; current layout: `defn.py`, `builder.py`, `_tree.py`,
  `graph.py`, `trace.py`, `control_flow.py`, `__init__.py`.
- All `trace` outputs are ordinary ops — backends need no control-flow
  runtime support (numpy interpreter selects regions/loops natively).
- No caching, no eager execution, no silent fallbacks anywhere in this
  module.

## Design decisions

- **`_TraceSession` owns all trace state.** The alternative (module-level
  globals, or threading a state object through every helper) either leaks
  state across nested/reentrant traces or buries it in signatures. The
  dataclass makes the trace lifecycle visible: `open`/`run`/`finish` map
  1:1 onto the documented algorithm steps.
- **The leaf policy is a parameter, not a copy.** `trace.py` and
  `control_flow.py` previously shipped separate pytree walkers that drifted
  apart (they disagreed about which objects are leaves). `_tree.py`
  parameterizes the ONLY thing that differed. The two knobs
  (`leaf_spec` + `plain_leaf_type`) are explicit because the leaf
  conventions are load-bearing for `evoxc.pipeline` (`_is_tensor_leaf_spec`
  reads `TreeSpec.type`).
- **A fully-static dataclass config object passed as a trace argument is
  legitimate static specialization, not an error.** Dataclass instances are
  pytree containers: the tracer descends into them, and a config whose
  fields are all static Python values specializes statically at trace time —
  each static leaf is recorded in `static_values` and validated at run time
  via `Graph.flatten_inputs`. (A dataclass containing non-static,
  non-`TensorSpec` leaves still raises `TraceError` per the trace algorithm
  above.)
- **Markers are plain classes, not dataclasses.** `_TensorSpecLeaf` /
  `_SymbolicLeaf` are deliberately not dataclasses (no `__eq__`): identity
  is enough, and `TreeSpec` equality never has to compare their payloads.

## Known warts

- **Trace-tree dict sorting is NOT wrapped, unlike `core.flatten`.** `_tree.py::_flatten_into` step 7 does a bare `keys = sorted(obj)` (no try/except), whereas `core.tree._flatten_into` and `pipeline._fast_structure_walk` wrap the same call into `flatten: cannot sort dict keys of mixed types (types: [...])`. Consequence: a dict with unorderable mixed-type keys (e.g. `{"a": spec, 1: spec}`) passed as a TRACE input (or as a `cond`/`while_loop`/`scan` operand, since control flow uses the same walker) raises the raw Python `TypeError` from `sorted` with no guidance. Keys of one sortable type are fine — they need NOT be strings (`int` keys work), they just must be mutually orderable.
- **`_is_static_value` numpy-scalar asymmetry** (`_tree.py`): `np.float64`/`np.float32`-family values that are real Python `float`/`complex`/`str` subclasses (e.g. `np.float64`, `np.complex128`, `np.str_`) PASS the `isinstance(obj, (float, complex, str))` checks and count as static, while `np.int32`/`np.int64`/`np.uint*`/`np.bool_` (not Python `int`/`bool` subclasses) and `np.float32` are REJECTED — pinned by `tests/ops/test_getitem.py::test_numpy_scalar_key_unsupported_v1` and `tests/trace/test_static_snapshot.py::test_non_static_specs_are_rejected` ("is neither a core.TensorSpec nor a static"). `np.ndarray` leaves ARE static (accepted class — snapshotted at trace time, validated with `_static_equal`/`np.array_equal` semantics at run time in `Graph.flatten_inputs` and in the cond/while static cross-checks; pinned by `tests/trace/test_static_snapshot.py::test_ndarray_static_arg_accepted_and_preserved` + `tests/ops/test_getitem.py::test_ndarray_static_arg_accepted_and_preserved`); ndarrays still become TENSOR inputs via `TensorSpec` or `evoxc.evaluate` (which derives specs from them) — the two paths are distinct.
- **`Graph.input_specs` is a TreeSpec, not "specs".** The name predates the
  input_tree concept; it is kept for API stability. New code should read
  `tensor_specs` + `static_values` for leaf-level data.
- **`StaticValue` (this module) collides in name with
  `evoxc.block.decl.StaticValue`.** Different types, both public-ish; do not
  import one where the other is expected. (No import conflict — separate
  modules.)
- **Marker-leaf privates are cross-module imports.** `evoxc.pipeline`
  imports `_TensorSpecLeaf`/`_SymbolicLeaf` by name — they must stay at
  `evoxc.trace.trace` with those names (see "Module layout").
- **`evoxc.trace` is the function, not the submodule**, after `import evoxc`
  (the package re-exports the function under that attribute). Use
  `from evoxc.trace import ...` (works — the import system consults
  `sys.modules` first) or `sys.modules["evoxc.trace"]` for the module object.
- **`core.TreeSpec.context` (registered-node aux) is invisible to
  `core.first_mismatch_path`** — aux-bearing registered nodes get no
  automatic run-time aux-equality validation via `Graph.flatten_inputs`;
  check aux explicitly or model it as static leaves.
- **`while_loop` carry count is 1:1 with `init`'s symbolic leaves, and NO
  carry-rewrite machinery exists anywhere in evoxc.** `control_flow.py:545`
  carries exactly the `SymbolicTensor` leaves of `init` (one `while`-op
  operand + one region block arg each — no dedup, reordering, or packing);
  `evoxc/transforms` REJECTS region-bearing ops outright
  (the native `vectorize` engine raises `TransformError`, and
  `batch_functions._check_ops` names the function index), and the stablehlo
  writer emits every operand positionally with no count gate
  (`backends/stablehlo/writer.py::_emit_while`). Known consequence (iree
  3.9.0): a ≥4-carry `stablehlo.while` in a large module (NSGA3 tell
  graph) SIGSEGVs in AffinityAnalysis while 3-carry whiles (NSGA2) compile
  — see `../backends/stablehlo/CONTEXT.md` ("Loop-CARRIED-COUNT crash
  observation"). Reducing carries today means USER-SIDE graph changes (e.g.
  recompute a derived carry such as `dominates = (count == 0)` locally in
  the cond/body instead of carrying it) — there is no CSE/DCE/region
  simplification anywhere in evoxc that would do it for you.
- **`source_locations` has no error-message consumer.** Despite the docstrings ("so error messages can name the user's trace line"), nothing in `evoxc` reads it to build an error; its real consumers are `Graph.save`/`load`, `summary()["source_location_count"]`, and the transforms that remap it positionally through the native `input_ids` (`transforms/_native_bridge.py:182–189`, `transforms/_native_ad.py:182`, `transforms/batch_functions.py:953–958`). Keys are `ir.Value.id` (module-unique ints), one per TENSOR input block arg — static input leaves and op results/locations are NOT in it (per-op `op.location` is a separate, ops-owned mechanism).

## How to extend (hack-ability)

- **Add a new traceable op**: nothing to do in trace/ — `evoxc.ops` builds IR
  via `current_builder()`, and control flow needs no changes. (The op's
  shape/dtype rules live in ops/ir.)
- **Add a new static leaf type** (e.g. a config object): extend
  `_is_static_value` in `_tree.py` (the single copy) and check that the
  type survives `Graph.save/load` (persist codec in `evoxc/persist/`).
- **Add a new control-flow construct**: copy the `_run_in_region` pattern —
  build a detached region via `builder.build_region(...)`, classify operands
  (`_classify_operands` for cond-style, the carry rules for while-style),
  run callables inside the region, emit the `return` via
  `_return_terminator`, `create` the region op with explicit `result_types`
  (or `shape_fn`), pop in `finally`. Follow the `scan` desugar pattern for
  anything loop-shaped.
- **Change how static values are recorded**: the single construction site
  is `graph._static_record`; the classification sits in
  `trace._flatten_specs` / `trace._classify_outputs`. The recording format
  (`StaticValue.index/path/value/kind`) is consumed by `graph.py` and
  encoded by `evoxc.persist` — change all three together.
- **Add an input/output plan cache**: see "Caching deliberately NOT added"
  before doing it.

## Test strategy

pytest; mirror under `../../tests/trace/` (or per-module files):
- `defn`: calling a Defn raises `TraceError` with the staging-direction
  message; bare/options decorator forms; idempotence.
- `trace`: static values specialize (python `if` over an int works at trace
  time); concrete-tensor and unknown-leaf inputs raise with path; nested
  structures (tuple/list/dict/namedtuple/dataclass) round-trip;
  SymbolicTensor inputs have no `.numpy()` (SymbolicTensor purity);
  closure-captured `Tensor` fails via ops' `TraceError`; output static
  values re-inserted; zero-input and zero-result graphs.
- `current_builder` raises outside trace; ops route into nested regions.
- `cond`/`while_loop`/`scan`: region ops appear in the IR (inspect module),
  loop-carried shapes, branch-tree mismatch errors, scan static-length
  round-trip vs equivalent while_loop, symbolic-length `TraceError`.
- `Graph`: flatten/unflatten validation (ShapeError/DTypeError/DeviceError/
  static mismatch), save/load round-trip via persist (integrity,
  `PersistenceError` on mismatch).
- Spec-compliance coverage lives in `../../tests/test_spec_compliance.py`
  (owned by the package root): staging explicitness, bind-as-sugar,
  vmap≡vectorize, etc.

## Status

Implemented and green. The 2025-08 cleanup refactor (commit 507e42a):
extracted `_tree.py` (shared walker + static helpers), introduced
`_TraceSession`, unified region execution (`_run_in_region`) and block
termination (`_return_terminator`), deduped `cond` operand flattening and
output unflattening. Public API unchanged; error messages byte-identical;
full suite 5873 passed / 20 skipped / 0 failed (5893 collected — skips
environmental: torch absent / xla plugin absent / GPU-guarded);
`evoxc.bench` conformance 98/98.
