# Rust-first evoxc: native architecture proposal

**Status: proposed target architecture, not implemented policy.**
This document describes the intended clean architecture if a Rust-first evoxc is approved.
All interfaces, crate names, code fragments, and packaging choices below are proposed, not current APIs.
The repository remains a pure-Python implementation governed by [CONTEXT.md](CONTEXT.md) and [the package contract](evoxc/CONTEXT.md).
This proposal neither changes those contracts nor claims that native code has been benchmarked.
It specifies an end state and explicit compatibility decisions, not a migration plan.

## Contents

1. [Decisions and scope](#1-decisions-and-scope)
2. [Ownership and dependency boundaries](#2-ownership-and-dependency-boundaries)
3. [The program and value model](#3-the-program-and-value-model)
4. [Native construction, inference, and diagnostics](#4-native-construction-inference-and-diagnostics)
5. [Staging and the single materialization dispatcher](#5-staging-and-the-single-materialization-dispatcher)
6. [Deferred function transforms and native inlining](#6-deferred-function-transforms-and-native-inlining)
7. [Complete native ownership of vmap](#7-complete-native-ownership-of-vmap)
8. [Automatic differentiation and transform composition](#8-automatic-differentiation-and-transform-composition)
9. [Operations, composites, effects, and support](#9-operations-composites-effects-and-support)
10. [Extensions and portability manifests](#10-extensions-and-portability-manifests)
11. [Compiler stages and backend adapters](#11-compiler-stages-and-backend-adapters)
12. [Sessions, buffers, submissions, and lifetimes](#12-sessions-buffers-submissions-and-lifetimes)
13. [Placement, host access, and asynchronous IO](#13-placement-host-access-and-asynchronous-io)
14. [Python, Rust, BEAM, and C interfaces](#14-python-rust-beam-and-c-interfaces)
15. [DLPack and foreign buffers](#15-dlpack-and-foreign-buffers)
16. [Portable persistence and cache identity](#16-portable-persistence-and-cache-identity)
17. [End-to-end examples](#17-end-to-end-examples)
18. [Performance model and measurement](#18-performance-model-and-measurement)
19. [Validation and acceptance criteria](#19-validation-and-acceptance-criteria)
20. [Compatibility and packaging decisions](#20-compatibility-and-packaging-decisions)
21. [Unresolved engineering details](#21-unresolved-engineering-details)
22. [Evidence and references](#22-evidence-and-references)

## 1. Decisions and scope

The architecture has one numerical authority: the Rust compiler core.
It owns builtin op schemas, dtype promotion, shape inference, numerical compositions, graph verification, graph transformations, and lowering.
Python executes Python `defn` bodies and adapts Python objects; it does not implement builtin graph algorithms.
Rust, Python, Elixir, and C clients observe the same numerical program and rules.
A language binding is not a second implementation of evoxc.

The staging model remains `defn → trace → Graph → transform → lower → compile → load → run`.
Applying a transform to a Python callable may implicitly trace that callable once specs are available.
It must never implicitly compile, load, execute numerical work, transfer a tensor, or specialize from concrete runtime tensors.
There is no hidden JIT, eager numerical mode, or concrete-versus-symbolic mode switch.
`build`, `evaluate`, and `build_cached` remain documented compositions of explicit stages.

**Settled directions:**
- All builtin transformations, including the entire function-side `vmap` planner, live in Rust.
- Deferred transforms are native `GraphFunction`/`TransformPlan` objects, not Python rewrite closures.
- One native materialization dispatcher serves tracing, transforms, and build sugar.
- Sealed numerical graphs contain no Python objects or host callback references.
- Runtime placement includes runtime/client identity, not just a physical device label.
- Native runtime-only loading does not require compiler packages or Python.
- NumPy remains an independent reference backend/oracle; this proposal does not add a native reference interpreter.
- Extension dependencies and nonportable callbacks are explicit and stage-specific.
- Hardware optimization, kernel scheduling, memory planning, and code generation remain external compiler responsibilities.

Rust is chosen for shared semantics, ownership, reusable native interfaces, and reduced host overhead.
It is not a claim that rewriting host code fixes compiler dispatch formation or GPU code generation.
Existing explicit collectives, local physical tensors, large explicit inputs, and graph persistence remain central.
No automatic sharding, inserted communication, host fallback, or implicit tensor movement is introduced.

## 2. Ownership and dependency boundaries

A modest Cargo workspace separates numerical description, compilation, persistence, execution, and bindings.
Modules should become crates only when dependency or distribution isolation warrants it.
The following is a dependency DAG, not a mandate to create a crate for every feature:

```text
evoxc-core       shapes, dtypes, IDs, schemas, sealed NumericalProgram, diagnostics
    ↑
evoxc-format     portable program/artifact envelopes and canonical encoding
    ↑                         ↑
evoxc-compiler              evoxc-runtime
 builder, inference,         sessions, buffers, executables,
 transforms, lowering        submissions, adapter interfaces
    ↑                         ↑
    └──────── bindings ───────┘
       python (PyO3), beam (Rustler), c-api

optional adapter integrations depend on core/format and the relevant
compiler or runtime interface; runtime never depends on compiler.
```

`evoxc-core` contains immutable definitions and validation primitives, not language runtimes or device drivers.
`evoxc-compiler` owns the mutable builder, materialization, builtin rules, and export machinery.
`evoxc-format` depends on core, and both compiler and runtime may use it without a cycle.
`evoxc-runtime` understands executable metadata and the flat numerical ABI, not Python call signatures.
Compile-side and runtime-side implementations of an adapter can ship independently.
Neither PyO3 nor Rustler is a transitive dependency of the shared core or runtime.

| Concern | Authority | Binding responsibility |
|---|---|---|
| Python body execution | Python frontend | Invoke callable with symbolic proxies |
| Op inference and promotion | Native compiler | Adapt arguments; deliver one emit request |
| Builtin numerical composition | Native compiler | Delegate by operation identifier |
| All builtin graph transforms | Native compiler | Adapt selectors/spec trees; retain host callable roots |
| Graph formatting/inspection | Native compiler/format | Render optional host-friendly objects |
| Tree classes and reconstruction | Host adapter | Describe neutral structure; rebuild host containers |
| Device execution and readiness | Native runtime/adapter | Wrap handles and surface errors |
| NumPy reference execution | Optional Python backend | Independent kernel implementation, explicit dependency |

Python operator overloads and `evoxc.numpy` exports are thin native delegations.
For example, `softmax` expands once in Rust into ordinary operations; Python must not also own its expansion.
Backend kernels and backend-specific lowerings necessarily implement execution/codegen paths, not competing frontend rules.
The NumPy oracle is intentionally independent so it can detect native semantic defects.
It consumes verified programs and cannot decide alternate promotion, shape, batching, or differentiation semantics.

## 3. The program and value model

Separate a language-neutral numerical program from its frontend call convention:

```rust
struct NumericalProgram {
    functions: FrozenFunctions,
    constants: ConstantTable,
    registry: RegistryFingerprint,
    abi: FlatSignature,
    diagnostics: SourceMap,
} // sealed, immutable; no host object pointers
struct Graph {
    program: Arc<NumericalProgram>,
    signature: CallSignature,
} // frontend wrapper; host reconstruction metadata lives outside the program
```

`FlatSignature` describes tensor slots, explicit static specialization data, composite schemas, and shape constraints.
`CallSignature` adds argument names, neutral structural trees, tensor-slot paths, and return reconstruction information.
Python class objects and custom pytree reconstruction functions belong in a Python sidecar, never in `NumericalProgram`.
A portable signature uses only supported structural tags and versioned schemas.
A host-local signature may refer to a sidecar token and is marked nonportable when that token cannot be encoded.

The four public concepts remain distinct:
- Static values are immutable specialization snapshots, not tensor storage.
- `TensorSpec` describes a future tensor, including shape, dtype, and optional placement requirements.
- `SymbolicTensor` is a builder-scoped SSA reference, never a buffer or DLPack exporter.
- `Tensor` owns a runtime `Buffer` handle and reports the physical placement of that buffer.

Shapes use checked native integers and a typed symbolic expression language.
The encoding declares fixed bounds for dimensions, indices, and byte lengths; every narrowing conversion is checked.
A Python integer outside the supported range raises a structured error, not truncation or platform-dependent overflow.
Dimension expressions are canonicalized with checked arithmetic, explicit constraints, and deterministic serialization.
Symbolic dimension variables, tensor dimensions, and logical batching axes are different types.

Numerical composite values, including sparse tensors, have portable component schemas.
A CSR value describes data, indices, indptr, dense shape, and structural invariants independently of Python pytree classes.
A sparse value may have several physical tensor slots but remains one logical numerical argument.
Its `nnz` dimension is not automatically a batch dimension.
Hosts may offer richer containers by adapting them into these component schemas and neutral structures.

## 4. Native construction, inference, and diagnostics

Construction uses a mutable arena builder; publication produces an immutable sealed program.
Typed generational IDs distinguish `ValueId`, `OpId`, `BlockId`, and `RegionId` and reject stale references.
Every symbolic handle also identifies its owning builder and scope.
A foreign-builder value is an error, even if an integer ID happens to match.
Use-def ownership, dominance, region arguments, terminators, and effects are verified before sealing.

```rust
fn emit(&mut self, request: EmitRequest) -> Result<EmitResult, Diagnostic>;
// request: op/schema ID, operand IDs, normalized attrs, optional source span
// result: result IDs AND inferred dtype/shape metadata in one return
fn with_region<T>(&mut self, f: impl FnOnce(RegionScope<'_>) -> Result<T>) -> Result<T>;
fn seal(self, outputs: &[ValueId]) -> Result<Arc<NumericalProgram>>;
```

Scoped region construction prevents IDs from escaping illegally and validates explicit region captures.
Inlining remaps SSA IDs and region references rather than sharing mutable builder objects.
Rollback or a poisoned scope ensures a failed nested construction cannot publish a partly valid graph.
Sealing establishes the immutable invariants needed by transformation, serialization, and concurrent inspection.

Native inference is authoritative for every builtin primitive and composition.
The same promotion tables, symbolic constraints, and validation helpers serve all frontends.
A binding may validate object types for a better error, but must not maintain an alternate shape calculator.
One emit call returns sufficient metadata to avoid a per-result getter ladder across the language boundary.
Bulk graph construction and batched inspection are available to non-Python frontends and tooling.

Explicit constant embedding accepts host-resident data only, validates representation, and emits size warnings.
The builder owns or immutably pins constant bytes according to the public copy contract.
It never captures a concrete closure tensor, downloads device data, or preserves a mutable NumPy view as a supposedly immutable constant.
Source spans and optional provenance travel with operations and survive native rewrites where meaningful.
Diagnostics carry category, message, span, op identity, structural path, and transform context.
Bindings map these to `TraceError`, `ShapeError`, `TransformError`, `BackendError`, and the other public error families.
Expected user failures are `Result` values, not panics; richer host traceback text is an optional presentation layer.

## 5. Staging and the single materialization dispatcher

The canonical materialization API is **`trace(function_like, *specs) -> Graph`**.
A function-like object is a frontend `Defn`, an unresolved callable leaf, or a native `GraphFunction` plan.
`Graph` is already materialized; `trace(graph, ...)` is rejected with guidance rather than silently retracing it.
`GraphFunction` is a function description, not an executable.
Calling `plan(specs)` as an alternate materializer is deliberately not the recommended API.
Outside active tracing, a `GraphFunction` call raises and points to `trace`; this avoids ambiguity with symbolic invocation.

```rust
enum PlanNode {
    Program(Arc<NumericalProgram>),
    FrontendLeaf(FrontendToken),
    Transform { kind: TransformKind, input: PlanId, config: TransformConfig },
}
fn materialize(plan: &TransformPlan, specs: &NormalizedCall,
               host: &dyn FrontendBridge) -> Result<MaterializedGraph>;
```

The native evaluator derives required source specs, requests unresolved frontend leaves through the bridge, then performs native rewrites.
Python only executes the requested body and adapts the returned objects into neutral output structure.
Opaque frontend tokens index a host-owned table; they are not Python references embedded in an `Arc` core graph.
A host-side plan wrapper keeps the token roots alive until materialization no longer needs them.
A portable graph-backed plan needs no frontend bridge; a callable-backed unresolved plan is host-local.
Unknown required specs leave a plan deferred; they are never guessed from runtime `Tensor` values.

| Operation | Accepted object | Allowed work | Forbidden hidden work |
|---|---|---|---|
| `defn(f)` | Host callable | Mark/retain frontend leaf | Trace, compile, run |
| `vmap(f)`, `grad(f)` | Callable/plan | Construct native deferred plan | Compile, run, transfer |
| `grad(g)` | Sealed `Graph` | Immediate native graph transform | Python body execution |
| `trace(p, specs)` | Callable/plan | Materialize; execute needed Python bodies | Compile, run, transfer |
| `lower(g, config)` | `Graph` | Validate backend support; lower | Trace a callable, run |
| `compile(l, config)` | `LoweredProgram` | Backend compilation | Run numerical program |
| `load(a, session)` | `CompiledArtifact` | Adapter loading/allocation | Trace, compile, implicit placement repair |
| `run(e, tensors)` | `Executable` | Validate flat ABI; enqueue execution | Trace, compile, move inputs |
| `build(p, specs)` | Callable/plan | Documented trace/lower/compile/load | Run or transfer tensors |
| `evaluate(p, tensors)` | Explicit shorthand | Explicitly documented spec derivation/build/run | Transfers or hidden eager fallback |
| `Tensor.to(destination)` | Concrete tensor | Explicit transfer/import operation | Compile a graph |

`evaluate` is a named stage-composition API, not permission for transforms or ordinary calls to derive specs from concrete tensors.
`build_cached` uses the same materializer as `trace` and `build`; there is no special transform-wrapper detection.
Static signature adaptation, tree normalization, and plan evaluation therefore cannot disagree between entry points.
Explicit `lower`, `compile`, and `load` continue to reject earlier-stage objects.

## 6. Deferred function transforms and native inlining

All builtin function transforms return the same native `GraphFunction` abstraction when materialization is deferred.
The native catalog includes `vectorize`, `vmap`, `grad`, `jvp`, `vjp`, and `batch_functions`.
For `batch_functions`, graph alignment, variant switching, band selection, cost estimation, and merge reports are native too.
A host may adapt a list of callables, but cannot own a second population-merge planner or cost model.
A plan records transform ordering, selectors, axes, and immutable configuration as typed native data.
No Python closure unbatches specs, chooses transform order, rewrites signatures, or handles nested-transform special cases.
Python constructor functions adapt values and invoke the native plan constructor once.

```python
@evoxc.defn
def loss(x):
    return evoxc.sum(x * x)

per_example = evoxc.vmap(evoxc.grad(loss), axes=0, out_axes=0)
graph = evoxc.trace(per_example, evoxc.TensorSpec((B, D), "float32"))
# No compilation, execution, or concrete tensor inspection occurred.
```

Within an active trace, a `GraphFunction` may be invoked with symbolic arguments.
The native dispatcher derives symbolic argument specs, materializes any unresolved leaf, and inlines the resulting program into the active builder.
A graph-backed function uses the same native inlining path without a Python body callback.
Inlining handles SSA remapping, symbol/dimension substitution, region ownership, effects, and output reconstruction.
A symbolic call never executes the function's numerical program.
A `Defn` called from an active trace follows this explicit graph-call protocol too; outside tracing it remains non-executable.

The host bridge must support nested frontend requests and reentrant tracing deliberately.
It uses a context stack, not a single global current builder, and rejects illegal cross-context captures.
Native work must release builder/registry locks before calling the host; no native mutex is held while arbitrary Python executes.
Python callbacks attach to the correct interpreter, propagate exceptions, and restore the prior tracing context on every exit path.
Native-only transform work detaches from Python; frontend execution reattaches only for the required callback interval.
A plan rooted in one Python interpreter cannot be invoked from another without an explicit supported reconstruction.
Memoization of materialization requires explicit identity/spec keys; plan construction does not authorize an implicit closure cache.

## 7. Complete native ownership of vmap

`vmap` is function-side sugar over native graph vectorization, but **all of that sugar's numerical planning is native**.
Python does not own unbatching specs, axis matching, output-axis movement, function signature rewrites, or composition.
It supplies neutral structural descriptions and executes an unresolved callable when the native evaluator requests it.
`vectorize(graph, axes=..., out_axes=..., batch_size=...)` is the immediate graph-level primitive when the extent is supplied.
`vmap(graph)` can remain deferred until batched input specs supply the extent; neither API invents one.

Native materialization of `trace(vmap(f), batched_specs)` proceeds as follows:
1. Normalize `axes` against the neutral input structure and resolve each selected tensor slot.
2. Allocate one fresh `BatchAxisId` for this pass and validate mapped extents/constraints.
3. Remove the selected physical dimension from each mapped source spec; leave shared specs unchanged.
4. Materialize the source function, asking Python to execute its body only if its leaf is unresolved.
5. Apply native batching rules using logical axis provenance, not shape-name heuristics.
6. Normalize output axes, move or broadcast dimensions as requested, verify, and seal an ordinary graph.

`BatchAxisId` is an internal identity distinct from a public `Dim`, display name, or integer extent.
All mapped inputs in one pass share that ID and must satisfy the same extent constraint.
Nested passes always receive fresh IDs, even when their extents are equal.
Readable names such as `batch` and `batch_1` are diagnostics only and have no semantic role.
A batching value tracks logical mappedness separately from the positions of its physical tensor dimensions.
Rules may move dimensions, but cannot infer mappedness merely because a dimension has size `B`.

Axes are normalized natively from a host-neutral tree with tensor slots and structural paths.
An integer axis applies to compatible tensor leaves in the selected subtree; explicit axis trees must match its structure.
Negative axes normalize against the relevant tensor rank; static leaves must be shared (`None`).
Mapped input extents must agree; all-shared inputs require explicit `batch_size`.
Scalar source tensors can be mapped via a rank-one batched input; an axis on a rank-zero input spec is invalid.
Zero-length batches are valid where the selected ops/backend support their shapes; no sample is executed to infer outputs.
Dynamic extents are native constraints, with per-op transform and lowering support checked separately.
A Rust implementation alone does not make dynamic reshape or control-flow batching supported.

`out_axes=k` requests a physical output batch dimension at normalized position `k`.
A mapped output moves its corresponding dimension; an unmapped output broadcasts to the full batch extent.
This **intentionally changes** the current size-one insertion behavior for unmapped `out_axes=0`.
`out_axes=None` requires an output proved unmapped; it never picks one row or silently reduces a mapped value.
Output shape equality does not prove unmappedness.
Non-tensor static outputs must be batch-invariant and reconstructible; incompatible static outputs are errors.
Effects cannot be duplicated or reordered by a batching rule without a declared effect-aware contract.
Sparse batching operates on composite schemas and registered sparse rules, not on every component's leading dimension.
An unsupported sparse layout or rule produces a diagnostic naming the component/operation, not a guessed `nnz` axis.

For `vmap(vmap(grad(loss)))` on `(B, C, D)`, the outer pass owns axis `a` and requests `(C, D)` source specs.
The inner pass owns fresh axis `b`, requests `(D,)`, and materializes the scalar-loss gradient.
Inner vectorization produces `(C, D)` gradients; outer vectorization produces `(B, C, D)` gradients.
The same sequence holds for `B == C`: IDs, not equal extents, distinguish the axes.
Higher-order operations see ordinary transformed graphs and native provenance, not nested Python wrapper classes.

## 8. Automatic differentiation and transform composition

Rust owns derivative selection, signature rewriting, rule dispatch, accumulation, residuals, and output reconstruction.
`grad(graph)` is immediate when its selectors and result contract are resolvable from the graph signature.
`grad(callable_or_plan)` creates a deferred native plan and traces only when specs are supplied.
The primary `argnums` convention is **top-level positional argument indexing**, matching function signatures.
This intentionally replaces the current flattened tensor-leaf indexing convention.
`argpaths` is the explicit alternative for individual tensor leaves, using neutral paths such as `(0, "weights")`.
Supplying both `argnums` and `argpaths` is an error; selectors resolve once against the normalized signature.

A selected argument's gradient preserves its structure, with `NoTangent` markers for static/nondifferentiable leaves.
At least one selected differentiable leaf is required; selecting an integer tensor leaf directly is an error.
A single integer `argnums` returns that argument's gradient tree; a sequence returns a tuple in selector order.
`grad` initially requires one rank-zero real floating output; complex/non-scalar cases require an explicitly supported contract or error.
`NoTangent` is an explicit structural marker, not `None`, an implicit zero array, or permission to omit a runtime input.

JVP and VJP transform whole functions; they do not overload missing arguments with different execution meanings:

```text
jvp(g: Graph) -> Graph
  inputs:  (primals, tangents)
  outputs: (primal_outputs, output_tangents)
vjp(g: Graph) -> Graph
  inputs:  (primals, output_cotangents)
  outputs: (primal_outputs, input_cotangents)
jvp(f: FunctionLike) / vjp(f: FunctionLike) -> deferred GraphFunction
  materialize with specs for the complete derivative-domain input signature above
```

Primal trees carry static specializations; tangent/cotangent trees contain matching differentiable tensor slots and explicit `NoTangent` markers elsewhere.
The flat runtime ABI contains only the actual tensor slots, including explicit tangent or cotangent tensor inputs.
There is no hidden runtime pullback closure or implicit all-ones cotangent input.
`grad` derives from the reverse linearization/pullback engine with an explicit scalar seed graph constant.
JVP uses the shared linearization/rule infrastructure in forward direction; it does not copy a second differentiation framework.
Primitive rules may expose forward and transpose facets where needed, with common shape/selection machinery and conformance tests.
Higher-order AD differentiates ordinary rule-generated graphs; unsupported derivative rules fail explicitly.

Order remains meaningful: `vmap(grad(loss))` yields per-example gradients.
`grad(batch_loss)` differentiates the scalar reduction chosen by `batch_loss`; it is not automatically a per-example gradient.
If `batch_loss` invokes `vmap(loss)(symbolic_x)` during tracing, native materialization and inlining insert its batched graph before AD.
Nested `grad`/`vmap`/JVP/VJP use one native evaluator, not pairwise Python composition exceptions.
No support for vectorizing `cond`, `while_loop`, or `scan` is implied by these interfaces.
Each requires sound native rules for its control-flow, shape, and effect semantics before being advertised.

## 9. Operations, composites, effects, and support

An op schema and the implementations of its capabilities are separate facets:

```text
OpSchema: stable ID/version, operand/result structure, attrs, effect signature
Inference: dtype promotion, shape constraints, static validation
Batching: logical-axis-aware graph construction rule
Differentiation: linearization/transpose facets and declared restrictions
Lowering: backend/export-specific implementation and support predicate
ReferenceKernel: independent optional backend execution implementation
```

Presence of inference does not imply batching, differentiation, or compiler-backend support.
A native capability query reports restrictions as well as booleans, including dynamic-shape and dtype constraints.
Builtin composites such as activation functions, statistical helpers, and RNG expansions are registered native graph builders.
They produce ordinary EvoXIR operations rather than opaque Python callbacks.
Algorithm attributes, precision modes, overflow behavior, and RNG key representation participate in semantics and hashing.

The registry is constructed and frozen into an immutable snapshot before materialization.
Existing graphs and caches cannot change meaning because another library later registers a rule.
Adding extensions creates a new snapshot/fingerprint; handles retain the snapshot they need.
Load validates required schema and implementation versions rather than consulting an unconstrained mutable global table.
Extension resources remain alive as long as any snapshot or executable may call them.

Effects are explicit in IR, including callbacks, communication, and any sequencing constraints.
Transform verification checks effect preservation in addition to types and dominance.
`cond` is lazy: only the selected branch executes.
A branch being marked pure does not permit speculative execution when it may fail or not terminate.
Such rewrites require a proof of safe speculation, not an assumption that absence of side effects is sufficient.
No automatic communication, callback fallback, or backend substitution is inserted to make an unsupported graph run.

## 10. Extensions and portability manifests

There are three extension classes, with declared capabilities rather than a blanket “custom op works everywhere” promise.
A portable extension provides a versioned schema plus a native-readable decomposition into known operations.
A native plugin supplies inference/rules/runtime or compile implementations with explicit binary and semantic compatibility requirements.
A frontend-local extension may call Python/another host, but advertises that dependency at each stage where it is needed.
A frontend-local rule is an explicit exception for user extensions, never the implementation of builtin transforms.

```yaml
extension: org.example.segment_sum
schema_version: 2
semantic_version: 1.0.0
schema_hash: sha256:<canonical-schema-digest>
effects: []
portability:
  decomposition: evoxir-portable-v1
  inference: native-schema
  batching: decomposition
  differentiation: decomposition
  compile: [stablehlo]
  runtime: no-host-callback
requirements:
  materialize: [schema:org.example.segment_sum@2]
  lower: [decomposition:org.example.segment_sum@1.0.0]
  load: []
  run: []
```

The placeholder hash is illustrative, not an actual registered extension.
A Python-only runtime kernel instead declares `run: [python-interpreter, kernel:qualified-name@version]`.
A Python inference callback declares a materialization dependency even when its eventual runtime is fully native.
Portable save/load rejects or explicitly labels unresolved dependencies; it never serializes a callable and executes it on import.
A decomposition can remove a runtime dependency only when lowering actually substitutes and verifies it.
Extension registration is resolved against a frozen snapshot, including per-backend implementations.

External kernels use a versioned call contract covering tensor specs, symbolic shape relations, effects, and execution mode.
Static or symbolically related result shapes are distinguished from genuinely runtime-unknown shapes.
Unknown shapes require a backend-supported dynamic result/allocation protocol; they are not filled in by guessing metadata.
Host mode and device-resident mode are explicit, as are accepted memory kinds, session identities, and layout constraints.
Ownership specifies borrowed inputs, retained buffers, output deleters, and a completion/dependency token.
Returning a foreign buffer does not imply its producing stream has completed.
Runtime Python callbacks attach only when the manifest explicitly requires Python; pure native paths never acquire it per node.
An unavailable device kernel raises an error rather than silently downloading inputs to a host callback.

## 11. Compiler stages and backend adapters

`lower` consumes a sealed graph and a resolved stage configuration, checking every required facet.
`LoweredProgram` identifies its representation, target, ABI, effects, dependency manifest, and compatibility versions.
`compile` produces an artifact envelope with executable bytes and backend/runtime requirements.
`load` validates those requirements against an explicit session before creating an executable.
Artifact inspection and deserialization do not activate adapters, create clients, or probe devices.

```rust
trait CompilerAdapter {
    fn lower(&self, graph: &NumericalProgram, cfg: &LowerConfig) -> Result<LoweredProgram>;
    fn compile(&self, lower: &LoweredProgram, cfg: &CompileConfig) -> Result<Artifact>;
}
trait RuntimeAdapter {
    fn open(&self, cfg: &SessionConfig) -> Result<Session>;
    fn load(&self, artifact: &Artifact, session: &Session,
            cfg: &LoadConfig) -> Result<Executable>;
}
```

These interfaces do not promise that every backend supports offline compilation.
PJRT compilation may require a live client, topology, allocator, and plugin resources; a compile context owns them explicitly.
There is no numerical `run` hidden in that dependency, but resource creation must be observable and configurable.
Runtime-only load must never import a compiler package to repair an incomplete artifact.
A non-loadable intermediate is a `LoweredProgram`, not a misleading compiled artifact.

| Adapter | Compile route | Runtime route and limits |
|---|---|---|
| IREE | StableHLO/MLIR assembly → isolated compiler subprocess or supported compiler C API → VMFB | Native HAL/VM; target/runtime compatibility checked |
| PJRT/XLA | StableHLO → plugin PJRT C API, possibly using a compile client | PJRT client/executable/buffer APIs; plugin-bound serialization compatibility |
| TVM | Optional explicit Python compilation service until native compilation support is justified | Supported TVM runtime/VM hosting the exported artifact, not a bare tensor-function `.so` ABI |
| NumPy reference | Explicit Python backend preparation of verified graph | Python/NumPy required only for this backend; independent kernels |

StableHLO textual MLIR is not a long-term portable StableHLO artifact.
When interchange compatibility is required, use StableHLO portable-artifact APIs with a declared target version and their documented guarantees.
An IREE VMFB or PJRT serialized executable has its own narrower backend/device/plugin compatibility contract.
TVM export must record whether its runtime module, VM, additional libraries, and device support are required.
No adapter is advertised on a device simply because it can lower an intermediate representation for that device.

Configuration resolves once into typed lower/compile/load/run sections with provenance.
Environment defaults are read at the public boundary; explicit values win, and backend raw-flag conflicts follow a documented single-winner policy.
Every pipeline shorthand forwards the same resolved configuration to the appropriate stage.
Placement never comes from “the last backend used” or a mutable registry lookup.
Crash-prone compiler/plugin invocation supports explicit process isolation with a declared transport/copy contract.
A subprocess isolates faults; a Python thread, Rust panic boundary, or BEAM dirty scheduler does not isolate segmentation faults.
The core does not add optimizers, scheduling, memory planning, or device codegen to compensate for backend limitations.

## 12. Sessions, buffers, submissions, and lifetimes

A physical device and the runtime context capable of using its buffers are distinct identities.
Two clients on the same GPU may own mutually incompatible buffers.
The runtime model makes that distinction public enough to diagnose and resolve placement errors:

```text
Session → DeviceContext → allocator/queues/plugin lease
Executable → loaded program + Session lease + ABI
Buffer → storage + DeviceContext lease + readiness dependency
Submission → retained inputs + completion + produced-buffer handles
Completion → status/error + bounded runtime tracking
MappedView → host mapping lease + Buffer lease
ForeignExport → foreign consumer lease + exactly-once release state
```

An output buffer owns what it needs independently of the executable and survives executable destruction.
Dropping one output must not retain unrelated completed sibling outputs through a whole-result ownership cycle.
Inputs remain alive until submitted work no longer reads them, even if user tensor handles are dropped.
Plugin libraries remain loaded through every buffer deleter, event, foreign export, and asynchronous cleanup that may call them.
A session dropping its public handle cannot unload resources still leased by those objects.

`run` may enqueue and return output tensors carrying readiness dependencies.
Consumers on compatible queues submit dependencies instead of forcing unnecessary global synchronization.
Adapters without dependency chaining may wait explicitly as part of their documented run behavior.
Public tensor metadata is available without reading tensor contents.
Host reads, exports, and explicit waits surface asynchronous failures rather than returning invalid data.

`Executable.synchronize()` waits for a snapshot of submissions issued through that executable before the call.
`Session.synchronize()` provides the analogous session scope; a timeout never cancels work.
An explicit `Completion.wait()` reports submission-specific status, including asynchronous errors.
An adapter may expose `submit` returning `(outputs, Completion)` while ordinary `run` returns the same output structure as today.
Errors are recorded for explicit observation, including operations whose output handles were dropped.
Completed events and bookkeeping are reclaimed with bounded retention; synchronization cannot require an ever-growing historical list.

`Drop` never panics and is not the API for discovering execution errors.
Normal cleanup may defer frees until completion; explicit wait/close methods report failures.
Backend teardown that must wait is documented; the API does not promise universally nonblocking destruction.
Thread safety is backend-qualified: queue ordering, client calls, maps, and callbacks have explicit synchronization contracts.
Do not universally assert `Send + Sync` over raw backend handles without the adapter's guarantees and tests.

## 13. Placement, host access, and asynchronous IO

A `Tensor` always reports where its data actually resides, including its runtime/session identity where relevant.
NumPy host memory is CPU memory, never a label-only CUDA, ROCm, or Vulkan tensor.
`run` validates placement and compatibility; it cannot reconcile clients by downloading and reuploading.
A same-device but incompatible-session input raises an actionable `DeviceError`.
An explicit destination can include both physical device and session:

```python
session = evoxc.open_session(backend="iree", device="rocm:0")
x_device = x_host.to(session.device_context())
y_device = evoxc.run(executable, x_device)
y_host = y_device.to(evoxc.host_context())
array = y_host.numpy()
```

`Tensor.to(destination, copy=...)` is the sole public placement/transfer primitive.
An import into another client without copying is still explicit and succeeds only when compatibility is proven.
If copying is forbidden, an unavailable zero-copy route errors; it never falls back to staging.
Same-context no-op, cross-client import, host upload, and device readback are distinguishable transfer outcomes.
Device outputs remain on their executing device and retain readiness; bindings do not automatically make host arrays.

`.numpy()` is available only on CPU-resident data and must make alias/copy behavior explicit.
A CPU-kind opaque payload may require a readiness wait and a mapping lease; mapping is not the same as copying.
A non-CPU tensor requires explicit `.to(host_context)` first, even if the driver could temporarily map its memory.
`MappedView` pins both the mapping and storage for the view's entire lifetime.
Mutable host views require exclusive/coherent access rules; immutable tensor APIs cannot expose a writable alias silently.

`prefetch`, `sink`, and async-copy handles are sugar over the same native transfer submission primitive used by `Tensor.to`.
There is no second Python implementation of placement or ownership for asynchronous IO.
Schedule-time validation is synchronous; transfer failures surface at explicit completion/result access.
FIFO guarantees, per-context ordering, timeout behavior, and wait-all snapshot scope are documented independently of a particular worker implementation.
Callbacks and driver context attachment belong in the relevant adapter, not duplicated frontend workers.

## 14. Python, Rust, BEAM, and C interfaces

Direct Rust clients use idiomatic `Result`, owned handles, borrowed views, typed configuration, and explicit lifetimes.
PyO3 and Rustler call these Rust APIs directly; routing them through a C ABI merely for uniformity adds no semantic value.
Python classes wrap native handles while retaining only host-specific signatures, callable roots, and adapters.
Dynamic registered pytrees are flattened/rebuilt in Python; the core receives neutral node tags, paths, static snapshots, and tensor slots.
Neither batching nor AD imports Python class definitions to interpret those descriptions.

Long native-only materialization, transformation, compilation, and waits detach from the Python interpreter.
This matters on both GIL-enabled and free-threaded Python; native shared state still needs its own synchronization.
A frontend callback reattaches deliberately and cannot be called while native graph locks are held.
Free-threaded builds require tests for races in handle caches and host sidecars, not an assumption that removal of the GIL makes them safe.

Rustler bindings expose the same graph and runtime contracts with BEAM resource handles.
Lengthy CPU/IO work uses appropriately scheduled dirty NIFs or an explicit job/process interface.
Cancellation is cooperative where supported and never means silently freeing in-flight device inputs.
Dirty schedulers protect scheduler responsiveness, not VM fault isolation; unsafe plugins may require an external worker process.
An optional Elixir frontend can execute its own graph-definition bodies without making Python a core dependency.

The C interface is a versioned interoperability boundary, not an exposure of Rust layouts:

```c
evoxc_status_t evoxc_graph_transform_v1(
    evoxc_graph_t *graph, const evoxc_transform_config_v1 *config,
    evoxc_graph_t **out_graph, evoxc_error_t **out_error);
void evoxc_graph_retain(evoxc_graph_t *graph);
void evoxc_graph_release(evoxc_graph_t *graph);
void evoxc_error_release(evoxc_error_t *error);
/* Buffers/arrays returned by the API have an explicit owner and release API. */
```

Opaque handles have retain/release semantics, declared thread affinity, and defined invalid-argument behavior.
Versioned structs carry size/version fields; byte spans carry lengths and explicit borrowing/ownership rules.
Errors own stable message/span data until released; no thread-local borrowed string is the only diagnostic channel.
Rust panics never unwind across C, and foreign exceptions never unwind through Rust.
Catching a panic prevents unwinding only; it cannot recover memory corruption or a plugin segmentation fault.
C callback interfaces declare context lifetime, reentrancy, thread attachment, completion ownership, and destruction ordering.

## 15. DLPack and foreign buffers

DLPack is an interop protocol, not the internal buffer/session/readiness model.
Native buffers can export/import it through explicit adapter capabilities without forcing every internal tensor into DLPack structs.
The API follows the [Python DLPack specification](https://dmlc.github.io/dlpack/latest/python_spec.html), including stream and capsule ownership.
A tensor capsule is consumed once; producer/consumer transitions ensure its deleter is called exactly once.
Version negotiation uses `max_version` and versioned managed tensors where supported; unsupported versions fail clearly.

Copy permission is explicit and defaults to no copy for interop operations promising zero-copy import.
Layout validation covers dtype, shape, strides, byte offsets, alignment, address space, and supported negative/overlapping strides.
A requested compatible layout must not be produced by a secret contiguous copy.
Device ordinals may differ across frameworks, so imports validate physical identity and runtime mapping rather than comparing integers alone.
Unsupported remapping is an error with a placement remedy.

A consumer passes its stream; the producer coordinates readiness according to the protocol.
An adapter that cannot honor a stream request must report unsupported behavior or offer an explicit synchronized route, not ignore the stream.
Zero-copy does not mean zero synchronization, and a driver-wide synchronization is observable in measurements.
Exports keep storage, plugin code, and any required completion resources alive until the consumer releases them.
Imports retain the foreign owner/deleter and record readiness before accepting execution dependencies.
Cross-session import requires explicit compatibility negotiation even when the capsule names the same GPU.
Newer optional C exchange mechanisms are not a prerequisite for this design or a substitute for ownership tests.

## 16. Portable persistence and cache identity

A portable numerical encoding is separate from any Python reconstruction sidecar.
The existing Python codec must not be treated as a cross-language safe format by declaration.
The new envelope carries format/schema versions, manifests, integrity checks, limits, and a deterministic numerical payload.
Safe loading validates structure and dependencies without module imports, callable execution, adapter activation, or device access.
Loading a host sidecar capable of executing code requires a separately named trusted operation, never ordinary graph load.

Canonical encoding fixes integer widths, byte order, dtype identifiers, attribute ordering, and dimension-expression normalization.
Constants encode dtype/shape and canonical-endian bytes; bit patterns, signed zero, and NaN payloads are preserved rather than normalized away.
Static scalar floats use an explicit bit encoding too; strings use a specified Unicode encoding without locale dependence.
Unordered static maps are sorted by canonical encoded keys; ordered structures retain their order.
Schema versions and semantic versions distinguish changes in interpretation even when bytes otherwise match.
Hashing excludes incidental arena IDs, host addresses, timestamps, and nonsemantic display names.
Source diagnostics may be separately hashed/stored without changing numerical identity.

Static arguments are immutable snapshots, including nested configurations, not live references observed again at compile time.
Supported static data has a portable canonical form; opaque host-static objects are explicitly host-local.
A callable identity or closure environment cannot promise a portable content hash.
There is no automatic cache over arbitrary closures, captured mutable state, or Python object `repr` values.
A host-only pretrace cache may use pinned callable identity plus explicit snapshot/spec keys and must label that limitation.

```text
materialization key = function/plan identity or portable plan digest
                    + normalized specs/static snapshots + registry/rule snapshot
artifact key        = numerical digest + semantic/schema versions
                    + resolved lower/compile options + backend/compiler manifest
load compatibility  = artifact requirements + runtime/plugin ABI + device/context capabilities
```

A runtime pointer is not a persistent artifact key; a hardware/plugin compatibility descriptor is.
Build caches retain required host roots explicitly and never confuse object-ID reuse with function identity.
Compile failures insert no usable artifact; manifest mismatches fail instead of silently recompiling at load.
Portable graph save need not make unresolved frontend plans portable.
Integrity validation does not make an executable artifact safe to run: native artifact loading is an explicit trust boundary.

## 17. End-to-end examples

The following proposed examples emphasize stage boundaries rather than a hidden convenience path.
`B` and `D` below are known static extents; symbolic versions require the selected ops/backend to support them.

```python
@evoxc.defn
def loss(params, x):
    residual = evoxc.dot(x, params["w"]) - params["target"]
    return evoxc.sum(residual * residual)

per_example_grad = evoxc.vmap(
    evoxc.grad(loss, argnums=0), axes=(None, 0), out_axes=0)
g = evoxc.trace(per_example_grad, param_specs, evoxc.TensorSpec((B, D), "float32"))
# Rust normalizes trees/selectors, unbatches x, asks Python to emit loss once,
# differentiates the source graph, batches the derivative graph, and seals g.
```

`params` is shared; its per-example gradients are mapped outputs despite shared primal inputs.
Weights are explicit tensor inputs, never captured constants.
The gradient tree for argument zero uses the same native slot/structure description as the call signature.

```python
batched_loss = evoxc.vmap(loss, axes=(None, 0), out_axes=0)

@evoxc.defn
def total_loss(params, xs):
    return evoxc.sum(batched_loss(params, xs))  # symbolic call, native inlining

total_grad = evoxc.trace(evoxc.grad(total_loss, argnums=0), param_specs, batch_spec)
grad_of_existing_graph = evoxc.grad(evoxc.trace(total_loss, param_specs, batch_spec), argnums=0)
# Both routes use the same native AD implementation and normalized signature.
```

The frontend executes `total_loss`, but it neither performs batching nor splices Python graph objects.
The active native builder receives an inlined ordinary graph before differentiation.
A concrete call `batched_loss(params_tensor, xs_tensor)` outside a trace raises a staging error.

```python
cfg = evoxc.resolve_config(backend="iree", device="rocm:0", opt_level="O2")
lowered = evoxc.lower(g, config=cfg)
artifact = evoxc.compile(lowered, config=cfg)
session = evoxc.open_session(config=cfg)
exe = evoxc.load(artifact, session=session, config=cfg)
placed = evoxc.tree_map(lambda t: t.to(session.device_context()), host_inputs)
outputs = evoxc.run(exe, *placed)
exe.synchronize()                 # include this in completed-work timing
host_outputs = evoxc.tree_map(lambda t: t.to(evoxc.host_context()), outputs)
```

Placement is deliberately separate from `build` and `run`.
A prepared invocation can pin the flat slot map and native tensor handles for repeated calls without flattening containers each time.
It cannot stage tensors or bypass dtype/shape/context validation; rebinding inputs is explicit argument substitution.
A runtime-only Rust client can load the compatible artifact and pass flat buffers without importing Python or the compiler.

## 18. Performance model and measurement

Boundary cost is a sum, not one universal “FFI overhead” constant:

```text
host overhead = boundary calls + conversion + allocation + copies + synchronization
end-to-end    = frontend execution + native transforms/lowering + external compilation
              + load/setup + host overhead + device work
```

A Python `defn` with many primitive calls still crosses the boundary per emit.
Return metadata with IDs, intern immutable attrs, expose bulk emit/inspection, and avoid repeated container conversion.
A graph transform is one coarse native call over native graph storage, not graph serialization to Rust and back.
Prepared flat invocations retain tensor handles and validate slots without new placement or staging.
Benchmark Python-originated tracing separately from direct Rust/bulk graph construction; they exercise different bottlenecks.

| Axis | Required measurements |
|---|---|
| Construction | Per-node emit, scalar/static attr conversion, small/large graphs, region depth |
| Transforms | vmap/AD/nested plans, graph sizes, shape complexity, sparse composites |
| Binding | Python versus direct Rust, getter counts, allocations, bulk inspection |
| Invocation | Flat/prepared versus tree calls, varying tensor-leaf counts and tensor sizes |
| Execution | Enqueue latency AND completed latency, same-session resident inputs |
| Lifecycle | Cold/warm compile/load, session creation, live/peak memory, cleanup |
| Interop/IO | Actual copied bytes, mapping, stream waits, cross-session imports |
| Concurrency | Backend-qualified parallel calls, free-threaded hosts, callback contention |

Instrumentation counts callbacks, boundary calls, temporary allocations, copied bytes, and waits.
Native-only builtin transforms must show zero Python callbacks after their source graph is materialized.
Reports include environment, backend/plugin/compiler versions, graph/op sizes, shapes, warmup, and synchronization method.
Compare identical programs and support levels; an unsupported leg is recorded, not silently replaced.
No universal nanosecond promise or invented speedup threshold is part of this proposal.
Acceptance targets should be set from measured baselines and reported with distributions, not chosen to justify a rewrite.
Existing [dispatch-formation evidence](evoxc/bench/dispatch_formation_findings.md) demonstrates a backend bottleneck, not expected Rust speedups.
External compile/codegen and dispatch-bound device workloads must remain separate from host-overhead claims.

## 19. Validation and acceptance criteria

The architecture is acceptable only if its ownership claims are demonstrable in tests and instrumentation.
Native semantic tests run without Python, NumPy, BEAM, or optional compiler installations.
Python and Rust binding conformance tests then exercise the same fixtures and native results.
NumPy reference kernels and external compiler adapters supply independent execution comparisons with explicit numerical tolerances.

- **Single authority:** builtin promotion, inference, compositions, batching, AD, and signature rewriting have one native implementation.
- **Staging:** transform construction/materialization cannot compile, execute, transfer, or inspect concrete tensor contents.
- **Materialization:** trace/build/build_cached use the same dispatcher for callable and nested native plans.
- **vmap:** equal/unequal nested extents, mapped/shared inputs, negative axes, scalars, empty batches, and output broadcasts are covered.
- **Axis identity:** equal dimension names or extents never collapse distinct nested `BatchAxisId`s.
- **AD:** selector paths, structured arguments, static leaves, explicit JVP/VJP input trees, and higher-order composition are tested.
- **Inlining:** native graph calls preserve regions, dominance, effects, provenance, and builder isolation.
- **Reentrancy:** nested callbacks, exceptions, interpreter attachment, and lock ordering cannot corrupt active builders.
- **Effects:** lazy branches remain lazy; an unselected failing/nonterminating branch is never speculated without proof.
- **Deferrals:** unsupported control-flow batching, dynamic lowering, and sparse/kernel facets report precise errors.
- **Runtime identity:** same physical device with incompatible clients errors until an explicit supported transfer/import.
- **Lifetime:** outputs survive executable drop; inputs survive submission; independent completed outputs can be reclaimed independently.
- **Plugin ownership:** foreign exports and asynchronous deleters outlive public plugin/session handles safely.
- **Async behavior:** dependency chaining, snapshot waits, timeouts, dropped outputs, and asynchronous errors are observable and bounded.
- **Host access:** no implicit non-CPU mapping/readback through `.numpy()` or output conversion.
- **DLPack:** once-only capsule consumption/deletion, version negotiation, streams, strides, ordinal mapping, and copy refusal are tested.
- **Persistence:** deterministic cross-language round trips, limits, corrupted bytes, unknown schemas, and nonportable sidecars are covered.
- **Cache validity:** rule snapshots, static mutations, callable keepalive, options, and backend compatibility cannot yield stale hits.
- **Runtime-only distribution:** loading/using an artifact cannot import or require a compiler package.
- **Performance evidence:** cold/warm and enqueue/completed measurements include copies and synchronization, without unsupported claims.

Fuzz parsers, shape arithmetic, graph verification, transform inputs, and safe FFI argument decoding.
Use sanitizers and ownership stress tests around adapter boundaries, callbacks, cancellation, and plugin unload.
Test native plugins in fault-isolated subprocesses where a crash would otherwise kill the test runner.
C/BEAM/Python exception boundaries must never leak Rust unwinding or assume recoverability after memory corruption.
A capability matrix records exactly which ops, dtypes, dynamic shapes, effects, and extensions each backend supports.
A passing NumPy oracle does not prove the native transform correct; structural invariants and independent differential tests are both required.

## 20. Compatibility and packaging decisions

The following decisions describe the target, not changes authorized by this document:

| Area | Target decision |
|---|---|
| Explicit pipeline | Preserve distinct stage objects and named composition APIs |
| Python bodies | Preserve execution by Python during tracing; prohibit concrete eager fallback |
| Callable transforms | Permit spec-driven implicit tracing; replace wrappers with native deferred plans |
| Plan materialization | Recommend `trace(plan, *specs)`; reserve call syntax for active symbolic tracing |
| `vmap` shared outputs | Broadcast unmapped outputs to batch extent for integer `out_axes`, intentionally changing size-one behavior |
| AD selectors | Top-level `argnums`; explicit `argpaths` for leaves, replacing flattened-leaf indexing |
| JVP/VJP | Explicit derivative-domain tensor inputs and structural `NoTangent`; no missing-argument semantic overload |
| Portable persistence | New versioned neutral encoding; no claim that existing Python files are cross-language portable |
| Runtime placement | Add explicit session/context compatibility to physical-device checks |
| Reference backend | Retain independent Python/NumPy backend; do not add native interpreter |
| `evoxc-hf` | Keep separate distribution built only on the public evoxc API |

Recommended Python packaging is native wheels using maturin or equivalent Rust-aware tooling.
Users installing supported binary wheels need no Rust toolchain; source builds require a documented compatible Rust toolchain.
The shared runtime and compiler features remain separable; runtime-only artifacts do not pull compiler dependencies.
The Python distribution deliberately retains NumPy as a dependency for the reference backend and existing host-array ergonomics.
The Rust core/runtime do not depend on NumPy or Python; using the NumPy backend explicitly requires the Python host.
Compiler adapters and their toolchains stay optional and independently selectable.
Supported platform wheels, Python ABI strategy, and backend library bundling require explicit release qualification.

`evoxc-hf` must not acquire a direct dependency on internal Rust crates, native IDs, or private binding types.
It may use public session/placement APIs where needed without taking ownership of compilation or transfer semantics.
No Cargo manifests, Python build backend, dependency extras, CI policy, release process, or README claims are changed here.
There is no implementation schedule or migration sequence in this proposal.

## 21. Unresolved engineering details

These details require implementation experiments or compatibility specifications; they do not reopen the ownership decisions above.
Each must be resolved explicitly before the affected feature is considered supported.

| Detail | Required resolution / boundary already fixed |
|---|---|
| Shape integer limits | Choose exact portable bounds and overflow diagnostics; unchecked conversion is forbidden |
| Schema evolution | Specify version negotiation and upgrade tooling; safe load cannot execute host code |
| Native rule representation | Choose compiled rules versus a portable rule DSL per extension need; builtin authority stays native |
| Compile isolation transport | Choose file/shared-memory/RPC routes with declared copy/ownership costs and explicit isolation configuration |
| PJRT executable portability | Qualify each plugin's serialization and client/topology requirements; never promise universal offline artifacts |
| TVM runtime variants | Pin tested module/VM/runtime combinations before claiming native-only load support |
| Backend concurrency | Qualify queue/client thread safety and completion cleanup per adapter, not globally |
| Sparse batching | Specify supported batch layouts and constraints per schema/rule; never equate `nnz` with a batch axis |
| Runtime-unknown kernel shapes | Choose supported dynamic allocation/result handshake per backend or defer explicitly |
| Host mutable aliases | Specify exclusivity/coherence and read-only view behavior before exposing writable mappings |
| Wheel/ABI support | Select stable/limited Python ABI where viable and test free-threaded builds and platform SDK requirements |
| Performance budgets | Establish measured regression budgets per benchmark class; no predicted native speedups substitute for evidence |

A feature can be intentionally deferred while the architecture is complete.
An unresolved implementation choice is not permission for bindings to implement their own semantics temporarily as a second authority.

## 22. Evidence and references

Current-code observations motivate this proposal but are not claims of newly demonstrated test failures.
The present callable-wrapper path and target native plan model are different designs, not interchangeable descriptions of existing APIs.
Use the following repository contracts and evidence to distinguish current behavior from proposed ownership:

- [Root contract](CONTEXT.md) and [public package contract](evoxc/CONTEXT.md): current explicitness, value model, and pure-Python policy.
- [Core](evoxc/core/CONTEXT.md), [IR](evoxc/ir/CONTEXT.md), [ops](evoxc/ops/CONTEXT.md), and [trace](evoxc/trace/CONTEXT.md): current value, construction, and frontend boundaries.
- [Transforms](evoxc/transforms/CONTEXT.md): current rules, callable adaptation, batching semantics, and explicit deferrals.
- [Persistence](evoxc/persist/CONTEXT.md): current encoding/cache behavior, not a promised neutral native format.
- [Backend contract](evoxc/backends/CONTEXT.md), [adapter contract](evoxc/backends/adapters/CONTEXT.md), and [per-adapter notes](evoxc/backends/adapters/PER-ADAPTER-NOTES.md): current adapter capabilities and limitations.
- [StableHLO export](evoxc/backends/stablehlo/CONTEXT.md): existing lowering semantics and dynamic-shape support boundaries.
- [Dispatch-formation findings](evoxc/bench/dispatch_formation_findings.md): measured external-compiler/device dispatch costs, not Rust performance evidence.
- [HuggingFace contract](evoxc_hf/CONTEXT.md): independent public-API-only integration boundary.

External primary references informing the boundary contracts:
- [PyO3 0.30.1 performance](https://pyo3.rs/v0.30.1/performance): boundary/conversion costs and detaching native-only work on GIL and free-threaded builds.
- [DLPack Python specification](https://dmlc.github.io/dlpack/latest/python_spec.html): stream coordination, capsule/deleter ownership, version negotiation, copies, and device conventions.
- [Rustler 0.39 NIF scheduling](https://docs.rs/rustler/0.39.0/rustler/attr.nif.html): dirty CPU/IO scheduling for lengthy NIF work; scheduling is not fault isolation.
- [StableHLO compatibility](https://openxla.org/stablehlo/compatibility): portable-artifact compatibility APIs and target-version guarantees, not arbitrary MLIR assembly compatibility.

The resulting architecture is Rust-first in numerical authority and runtime ownership, not merely a Python implementation with a native hot loop.
Python remains an ergonomic frontend that genuinely executes user Python graph definitions.
Every other frontend shares the same compiler and runtime contracts without inheriting Python's object model.
