<div align="center">
<img src="https://raw.githubusercontent.com/EMI-Group/evoxc/main/assets/evoxc-mark.svg" width="96" alt="evoxc">

<h1 align="center">evoxc</h1>

<p><strong>EvoX Compiler</strong> — a minimal, explicit, compiler-neutral tensor graph compiler and runtime.</p>

<p>
  <a href="#install"><img src="https://img.shields.io/badge/python-3.11%2B-7c5cff?style=flat-square" alt="python 3.11+"></a>
  <a href="#install"><img src="https://img.shields.io/badge/deps-numpy%20only-7c5cff?style=flat-square" alt="deps: numpy only"></a>
  <a href="#backends"><img src="https://img.shields.io/badge/backends-numpy%20%C2%B7%20IREE%20%C2%B7%20XLA%20%C2%B7%20TVM-4c9a72?style=flat-square" alt="backends: numpy, IREE, XLA, TVM"></a>
  <a href="#status--deferrals"><img src="https://img.shields.io/badge/tests-6792%20passing-4c9a72?style=flat-square" alt="tests: 6792 passing"></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-Apache--2.0-555566?style=flat-square" alt="license: Apache-2.0"></a>
</p>

<p><a href="#why-evoxc-exists">Why</a> · <a href="#the-stack-from-a-python-graph-to-an-ml-compiler">The stack</a> · <a href="#quickstart--sixty-seconds">Quickstart</a> · <a href="#features">Features</a> · <a href="#batch-transformation--one-graph-for-a-whole-population">Batch</a> · <a href="#backends">Backends</a> · <a href="#hardware-support">Hardware</a> · <a href="#how-evoxc-compares">Compared</a> · <a href="#install">Install</a></p>
</div>

**evoxc** is the **EvoX Compiler**: a Python-embedded tensor graph compiler and runtime whose input language is an explicit SSA tensor graph. You build the graph, and evoxc owns the frontend (trace → transform → EvoXIR → StableHLO export) and the explicit staging pipeline around it — `lower`, `compile`, `load`, `run`. External ML compilers own the optimization, scheduling, memory planning and code generation. With no compiler installed it still runs — on a reference numpy interpreter.

---

## Why evoxc exists

A modern numerical framework fuses five different things into one opaque stack: a tensor language, a tracer, a compiler, a scheduler and an allocator. The result is convenient, and the program stops being an object you can hold — it is re-derived on every call, re-specialized when a Python value changes, re-captured when a guard fails, and the compiler quietly owns semantics the author never wrote down: implicit tracing, hidden caches, automatic resharding, eager/graph mode switches.

evoxc splits that stack along one hard line. Above it is the **frontend**, which evoxc owns: the tensor-graph language in which the program is an explicit, inspectable, persistable, transformable SSA graph, and the staging contract that carries it to a compiler — every step a public function with a public object for its result. Below it are **external compilers** — IREE, XLA via PJRT, TVM — which receive StableHLO and own how the computation is optimized, scheduled, allocated and generated. evoxc owns the frontend, the IR and the staging contract, never kernel scheduling or code generation; nothing below the line may change what the program means.

## Four words

**Minimal** — `numpy` is the only hard dependency; pure Python, no native code, no device runtime, no code generator, no kernel library. The default backend interprets the IR with numpy kernels. *So what: no compiler toolchain stands between you and a running graph.*

**Explicit** — every stage is a public function (`trace`, `lower`, `compile`, `load`, `run`) and every object a distinct public type (`Defn`, `Graph`, `LoweredProgram`, `CompiledArtifact`, `Executable`, `Tensor`); convenience APIs are documented shorthand whose exact expansion is in their docstrings. *So what: you can stop at any stage, inspect it, save it, or ship it.*

**Compiler-neutral** — the contract with the backend world is EvoXIR in the frontend and StableHLO MLIR at the boundary. No vendor IR is the definition of the IR; no adapter is privileged. *So what: swapping IREE for XLA or TVM is a `backend=` argument, not a rewrite.*

**Simple** — one obvious way per thing: `vmap` is sugar over `vectorize`, `bind` is argument-passing sugar, `build` is `trace → lower → compile → load`. No eager mode, no second semantics behind a flag. *So what: when the code surprises you, the surprise is in your graph.*

## The stack: from a Python graph to an ML compiler

<img src="https://raw.githubusercontent.com/EMI-Group/evoxc/main/assets/evoxc-stack.svg" alt="The explicit evoxc (EvoX Compiler) stack: defn → Graph (EvoXIR) → transforms → lower → LoweredProgram → compile (numpy interpreter or IREE / XLA-PJRT / TVM) → CompiledArtifact → load → Executable → run → Tensor" width="100%">

| stage | public function | resulting object | what it produces | what it never does silently |
|---|---|---|---|---|
| graph function | `@evoxc.defn` | `Defn` | a Python callable marked as a graph *definition* | run eagerly — calling it on concrete tensors raises `TraceError` |
| trace | `evoxc.trace(fn, *specs)` | `Graph` | EvoXIR: a region-based SSA module + input/output trees + snapshotted static values | compile, optimize, or turn a symbolic `Dim` into a constant |
| transform | `evoxc.vectorize` / `vmap` / `batch_functions` / `grad` / `jvp` / `vjp` | `Graph` | an ordinary graph of ordinary ops | execute, or leave a `vmap`/`grad` op for a backend to interpret |
| lower | `evoxc.lower(graph, backend=…)` | `LoweredProgram` | StableHLO MLIR text, or the backend's own payload | optimize or compile; an unsupported op is an explicit `BackendError` |
| compile | `evoxc.compile(lowered, **opts)` | `CompiledArtifact` | the compiler's artifact (vmfb, PJRT executable, serialized program) | get replaced at load — a version/ABI mismatch is a `PersistenceError` |
| load | `evoxc.load(artifact, device=…)` | `Executable` | a device-bound executable that can be called | move your data onto the device, or recompile for another target |
| run | `evoxc.run(exe, *args)` | `Tensor` (tree) | outputs in the same pytree shape you passed in | copy results to the host, cast, reshard, or insert communication |

**Two backends below the line.** The default path stops at `lower`: the **numpy interpreter** runs the EvoXIR program directly with numpy kernels — no compiler, no codegen, no artifact. The compiler paths take the *same* StableHLO text: **IREE** (llvm-cpu, cuda, rocm, vulkan-spirv), **XLA** (PJRT C API over a user-provided plugin `.so`) and **TVM** (StableHLO frontend + Relax VM).

**Layer discipline** — a strict one-way import DAG, so every problem has one owner:

```
core  ←  ir  ←  ops  ←  { trace, numpy }  ←  { block, transforms, backends, dist, sparse, pipeline, persist }
```

<sub>`core` imports nothing from evoxc (numpy only) · `ir` imports `core` · `ops` imports `core`, `ir`, `trace`'s builder hook · `trace` imports `core`, `ir` (never `ops`) · each module to the right imports only what is to its left.</sub>

## Quickstart — sixty seconds

The explicit pipeline; every intermediate is a real object you can keep:

```python
import evoxc
from evoxc import TensorSpec

@evoxc.defn
def f(x, w):
    return evoxc.sum(evoxc.tanh(evoxc.dot(x, w)), axes=1)

spec_x, spec_w = TensorSpec((4, 8), evoxc.float32), TensorSpec((8, 4), evoxc.float32)
graph    = evoxc.trace(f, spec_x, spec_w)                       # Graph          (EvoXIR)
lowered  = evoxc.lower(graph)                                   # LoweredProgram
artifact = evoxc.compile(lowered)                               # CompiledArtifact
exe      = evoxc.load(artifact, device="cpu")                   # Executable
y        = evoxc.run(exe, evoxc.ones((4, 8)), evoxc.ones((8, 4)))   # Tensor
print(y.numpy())     # [3.999999 3.999999 3.999999 3.999999]
print(graph)         # <Graph 2 tensor inputs, 0 static values>
```

The same thing in one call — *documented shorthand, no extra semantics*:

```python
exe = evoxc.build(f, spec_x, spec_w, device="cpu")          # trace → lower → compile → load
y   = evoxc.evaluate(f, evoxc.ones((4, 8)), evoxc.ones((8, 4))) # derive specs → build → run
```

Same graph, real compiler (`pip install evoxc[iree]`, or `evoxc[tvm]`, or a PJRT plugin `.so` for `xla`):

```python
exe = evoxc.build(f, spec_x, spec_w, backend="iree", device=evoxc.Device("cuda", 0),
                target_backends=["cuda"], opt_level="O3")
y   = evoxc.run(exe, evoxc.ones((4, 8)).to(exe.device), evoxc.ones((8, 4)).to(exe.device))
```

**The payoff: transforms are graph → graph, so they compose.** A batched per-sample gradient is a `vmap` of a `grad` — no backend ever learns what either word means:

```python
@evoxc.defn
def loss(x, y, w):
    d = evoxc.subtract(evoxc.tanh(evoxc.dot(x, w)), y)
    return evoxc.mean(evoxc.multiply(d, d))          # one scalar per sample

per_sample_grad = evoxc.vmap(evoxc.grad(loss, argnums=(2,)), in_axes=(0, 0, None), out_axes=(0,))
specs = (TensorSpec((8, 4, 8), evoxc.float32), TensorSpec((8, 4, 4), evoxc.float32),
         TensorSpec((8, 4), evoxc.float32))        # 8 samples of [4, 8]; the weights are shared
graph = per_sample_grad(*specs)                  # Graph — nothing has run yet
exe   = evoxc.load(evoxc.compile(evoxc.lower(graph)), device="cpu")
dw,   = evoxc.run(exe, evoxc.ones((8, 4, 8)), evoxc.zeros((8, 4, 4)), evoxc.ones((8, 4)))
print(dw.shape, dw.dtype, dw.device)             # (8, 8, 4) float32 Device(kind='cpu', index=0)
```

**Device-resident iteration.** A non-cpu executable rejects host inputs at the run boundary, so the loop *is* the pattern: place state once, then feed each run's device outputs back as the next run's inputs — zero host round-trips, zero implicit staging. At DE scale (4096×50, ~8 IREE dispatches per step) this costs ~0.15–0.5 ms/step, against ~1.05–2.8 ms when re-staging the ~800 KB state each step.

```python
device = evoxc.Device("cuda", 0)
exe = evoxc.build(step, *specs, backend="iree", device=device,   # step: @evoxc.defn
                target_backends=["cuda"], opt_level="O3")       # (key, state) -> (state, key)
state, key = evoxc.tensor(state0).to(device), evoxc.tensor(seed).to(device)  # upload once
for _ in range(steps):
    state, key = evoxc.run(exe, key, state)     # inputs and outputs stay on `device`
host = state.to(evoxc.Device("cpu", 0)).numpy() # the only way back
```

## The value model

Four concepts, deliberately not three. Python values *specialize* a graph, symbolic values are *program* values, and only a `Tensor` has storage.

| concept | meaning | has storage? | guarantees |
|---|---|---|---|
| Python/static value | `None`, bool, int, float, str, `Enum`, dtype, `slice`, config objects | n/a | evaluated at trace time and snapshotted into the graph; changing it means a *new* graph — no hidden guards, no recompile-on-change |
| `TensorSpec` | describes a future runtime tensor: `shape` (symbolic dims allowed), `dtype`, optional `device` | no | the only thing you can pass as a trace input besides static values |
| `SymbolicTensor` | an SSA value inside a graph | **no** | `dtype`, symbolic shape, source location, SSA identity — and deliberately **no** `numpy()`, `data_ptr()` or `__dlpack__` |
| `Tensor` | a materialized runtime tensor | yes | concrete shape/dtype and a **physical** `device`: numpy host data is `Device('cpu', 0)`, a device payload reports its real device |

Because storage exists only on the last row, misuse fails where the mistake is: `evoxc.constant` is the single, warned way to embed host data into a graph; capturing a `Tensor` in a closure is a `TraceError`; `.numpy()` on a non-cpu tensor raises `DeviceError` (`.to(cpu)` first), so no API can copy your data behind your back.

## Features

### Explicit graph construction — and no eager mode

`@evoxc.defn` marks a graph definition; it does not make the function callable on real data — calling it with a `Tensor` raises `TraceError` pointing at `evoxc.trace`/`evoxc.evaluate`. `evoxc.trace` turns the definition into a `Graph` you can print, verify, save, transform and lower:

```python
graph = evoxc.trace(f, spec_x, spec_w)
graph.print()                  # the EvoXIR module, one op per line
graph.verify()                 # explicit IR verification
graph.save("f.evoxcgraph")       # portable, versioned JSON container
```

```
module @"main" version 1 {
  func @main(%arg0: tensor<4x8xf32>, %arg1: tensor<8x4xf32>) -> tensor<4xf32> {
    %0 = evoxc.dot(%arg0, %arg1) : tensor<4x4xf32> loc("f.py":6:0)
    %1 = evoxc.tanh(%0) : tensor<4x4xf32> loc("f.py":6:0)
    %2 = evoxc.reduce_sum(%1) attributes {axes = [1], keepdims = False, reduce_op = "sum"} : tensor<4xf32>
    evoxc.return(%2)
  }
}
```

### Graph → graph transforms

`evoxc.vectorize`, `vmap`, `grad`, `jvp`, `vjp` and `batch_functions` are functions on graphs: a `Graph` in, a `Graph` out; a function/`Defn` in, a callable that builds one from specs out. The result is ordinary ops, so **a backend never sees `vmap`** — it sees the vectorized arithmetic a hand-written batched graph would have. An op with no batching or derivative rule is a `TransformError` naming the op, never a silent Python-loop fallback.

```python
@evoxc.defn
def scalar_loss(x, y, w):
    d = evoxc.subtract(evoxc.dot(x, w), y)
    return evoxc.mean(evoxc.multiply(d, d))

g  = evoxc.trace(scalar_loss, *specs)
gb = evoxc.vmap(g, in_axes=(0, 0, None))    # batch the samples, share the weights
gw = evoxc.grad(g, argnums=(2,))            # d/dw — still just a Graph
```

### Batch transformation — one graph for a whole population

An outer loop that writes a *different static program* every iteration — neural-architecture search, genetic programming, algorithm synthesis — normally pays a trace, a compile and a dispatch per candidate. `evoxc.batch_functions` evaluates the whole population as **one** ordinary graph:

```
evoxc.batch_functions(fns, axes=0, out_axes=0, *, switch_ops=None, bands=None, alignment="variant")
```

`fns` is a non-empty list of programs that share their input AND output specs: traced `Graph`s, or callables/`Defn`s — never a mix. The result is a single `Graph` whose leading axis is a **static**, length-N *function axis*: row `i` evaluates program `i`, a mapped input (`axes=0`) supplies its own row, and an unmapped one (`axes=None`) is shared by every program.

Programs are aligned by an occurrence-indexed structural skeleton signature. A position where every program has the same op collapses into **one** batched op, rebuilt through the same per-op batching rules `vmap` uses — identical programs emit zero `select`s. A same-skeleton position whose op differs merges into one switch class once `switch_ops` opts in (`True` = the `elementwise` + `comparison` categories; an explicit list names categories or op-name sets), emitting one op per variant plus a static per-row `select` cascade. All remaining divergence is `select`-muxed at the consumers and the outputs, so **any** same-signature program list merges correctly, and per-program constants stack into one `(N, …)` constant.

```python
progs  = [evoxc.trace(program(i), spec_x, spec_c) for i in range(32)]   # same signature, different bodies
specs  = (TensorSpec((32, 64), evoxc.float32), TensorSpec((1,), evoxc.float32))

merged = evoxc.batch_functions(progs, axes=(0, None), switch_ops=True)  # ONE Graph, a 32-wide function axis
exe    = evoxc.load(evoxc.compile(evoxc.lower(merged)), device="cpu")   # an ordinary executable
y      = evoxc.run(exe, stacked_params, c)      # y[i] is program(i)'s own output, bit for bit
print(merged.batch_report.value_classes)        # the merge itself, as data
```

Two keyword-only knobs tune the merge, and neither can change a row's value:

- **`bands`** — `None` (the default) merges all N programs on one flat function axis; an `int K` merges K contiguous bands independently and concatenates their outputs, so a class that only one band needs is computed over that band's rows; `"auto"` groups programs by structural fingerprint and lets a static work-vs-dispatch cost model pick a partition (never worse than flat by its own estimate).
- **`alignment`** — `"variant"` (the default) sends a program's switched op into an existing class that already holds that variant, which collapses sibling branches such as `(x+c)*(x-c)` written in a different order per program; `"occurrence"` is the byte-stable legacy merge; `"auto"` is variant-first with an occurrence fallback.

The recipe follows the hardware: **CPU → `bands="auto"`; GPU → `bands=16` or `bands=None`**, because on a GPU dispatch is cheap and the flat (or a fixed-band) merge wins where `"auto"`'s CPU-calibrated dispatch constant misjudges.

The payoff is measured, not argued. 32 programs of a 10-deep elementwise chain over a shared length-64 vector: **544 program ops collapse into 133 merged ops**, bit-identical to running the 32 programs separately, **2.8× faster per evaluation on numpy** (0.78 vs 2.18 ms) and **2.1× on `iree`/llvm-cpu** (0.63 vs 1.31 ms) — and **one merged IREE compile (0.45 s) replaces 32 (6.7 s, 15×)**. At N=100 unrelated expression trees over two `(64,)` inputs, separate → flat → banded is 5.4 → 3.32 → **2.24 ms** on numpy, and 4.7 → 18.73 → **0.386 ms** on `iree`/llvm-cpu: flat is 4× *worse* than not merging there, and banding recovers 12×.

**Inspectable, like everything else.** `graph.batch_report` is a `BatchReport` — `value_classes`, `shared_classes`, `switched` (a `SwitchRecord` per switch class: its variants and the rows that use them), `output_muxes`, `emitted_ops`, `bands`, `band_sizes`, `cost_estimate`, … — so the merge is data you can read rather than magic you have to trust. It is transform metadata, not part of the saved graph. **The rule is the usual one:** an op without a registered batching rule — control flow, a `runtime_call`/`external_call` without its namespaced rule, collectives, `random_*`, a few structural ops — raises an explicit `TransformError` naming the op, never a silent Python-loop fallback.

### In-graph control flow

`evoxc.cond`, `evoxc.while_loop` and `evoxc.scan` trace their branch and body functions into IR regions, so control flow is part of the persisted program. Python control flow over a symbolic value fails loudly instead of tracing something you did not write.

```python
@evoxc.defn
def sqrt_newton(x):
    def cond_fn(s):
        y, prev = s
        return evoxc.greater(evoxc.abs(evoxc.subtract(y, prev)), 1e-6)
    def body_fn(s):
        y, prev = s
        return 0.5 * (y + x / y), y
    y, _prev = evoxc.while_loop(cond_fn, body_fn, (x, evoxc.constant(evoxc.zeros((), evoxc.float32))))
    return y

exe = evoxc.build(sqrt_newton, TensorSpec((), evoxc.float32), device="cpu")
print(evoxc.run(exe, evoxc.tensor(2.0)).numpy())     # 1.4142135
```

A Python `if`/`while` on a symbolic value raises `TraceError: SymbolicTensor cannot be used as a Python boolean … runtime tensor control flow must use evoxc.cond`.

### Symbolic shapes and pytree I/O

A `Dim` is a named symbolic extent (`DimExpr` supports `+ - * // % min max` over dims and ints). Inputs and outputs are arbitrary pytrees, flattened once per call and validated structurally with first-mismatch-path errors; per-call host cost is leaf-driven, not graph-driven (~2–3.5 µs per tensor leaf, ~0 per op).

```python
B = evoxc.dim("B")

@evoxc.defn
def model(params, batch):
    h = evoxc.tanh(evoxc.dot(batch["x"], params["w1"]) + params["b1"])
    return {"logits": evoxc.dot(h, params["w2"]), "hidden": h}

spec = ({"w1": TensorSpec((4, 8), evoxc.float32), "b1": TensorSpec((8,), evoxc.float32),
         "w2": TensorSpec((8, 3), evoxc.float32)},
        {"x": TensorSpec((B, 4), evoxc.float32)})      # symbolic batch
exe = evoxc.build(model, *spec, device="cpu")          # one executable, any batch size
out = evoxc.run(exe, {"w1": evoxc.ones((4, 8)), "b1": evoxc.zeros((8,)), "w2": evoxc.ones((8, 3))},
              {"x": evoxc.ones((5, 4))})               # {'hidden': (5, 8), 'logits': (5, 3)}
```

### Explicit collectives and physical placement

Collectives are ops in the graph (`evoxc.dist.*`); a compiler may optimize them, never invent them. Multi-device *data preparation* (`split_tensor`, `replicate_tensor`) is a separate, explicit host-memory utility. Placement is physical: a `Tensor`'s `device` is where its data actually is, `Tensor.to(device)` is the only transfer primitive, run outputs stay where they ran, and a mismatched input is a `DeviceError` **before** dispatch.

```python
@evoxc.defn
def distributed_step(x):
    grp = evoxc.dist.group("world", [0, 1])
    return evoxc.dist.all_reduce(x, group=grp, op="sum")

evoxc.trace(distributed_step, TensorSpec((4,), evoxc.float32)).print()
#   %0 = evoxc.all_reduce(%arg0) attributes {group = "world", group_size = 2, reduce_op = "sum"}
```

### Sparse tensors

`evoxc.sparse` is a first-class value model — `SparseTensor`/`CSRTensor`/`CSCTensor` for data, `SparseTensorSpec` at trace time, symbolic instances inside graphs — with 16 graph ops (COO/CSR/CSC conversion, union `add`, intersection `multiply`, `multiply_dense`, `reduce_sum`, `transpose`, `reshape`, `concatenate`, sparse@dense and dense@sparse `matmul`) and numpy kernels that validate structure at runtime. Batching and derivative rules travel with the ops, so sparse values flow through `vmap`/`grad` and through `cond`/`while_loop`/`scan`. Compiler backends defer sparse with an explicit `BackendError` suggesting `evoxc.sparse.to_dense`.

```python
s = evoxc.sparse.coo(np.array([[0, 0], [1, 2]]), np.array([1.5, 2.5], np.float32), (3, 4))

@evoxc.defn
def sparse_step(s):
    return evoxc.sparse.to_dense(evoxc.sparse.multiply(s, s))

exe = evoxc.build(sparse_step, evoxc.sparse.SparseTensorSpec.from_concrete(s), device="cpu")
print(evoxc.run(exe, s).numpy())          # [[2.25 0 0 0] [0 0 6.25 0] [0 0 0 0]]
```

### Key-based functional RNG

`evoxc.random` is stateless and deterministic: a key is a value, ops never consume it, and the same key with the same operands yields bit-identical numbers across processes. The algorithm *is* the key's static shape/dtype — `splitmix64` (rank-0 i64, default), `threefry2x32` (`(2,)` i32), `philox4x32_10` (`(4,)` i32). The default was chosen by measurement: splitmix64 is fastest on both CPU paths and ties on GPU, winning every ≥4M-element cuda cell; threefry2x32 wins the small-sample GPU cells and is one `algorithm=` argument away.

```python
@evoxc.defn
def sample(key):
    k1, k2 = evoxc.random.split(key)
    return evoxc.random.normal(k1, (3, 2)), evoxc.random.randint(k2, (3,), 0, 10)

exe = evoxc.build(sample, TensorSpec((), evoxc.int64), device="cpu")
a, b = evoxc.run(exe, evoxc.random.key(42))     # same key, same bits — in any process
```

### Custom kernels

When a kernel must stay opaque — Triton, hand-written CUDA, anything you cannot express as a graph — declare the call inside the graph and register the implementation out of band. Kernels resolve per backend (exact slot → default slot), may be **device-resident** (device tensors in and out, zero host round-trips) and may declare batched variants:

```python
@evoxc.defn
def model(x, w):
    h = evoxc.external_call("triton_add", x, result=evoxc.TensorSpec((4, 8), evoxc.float32))
    return evoxc.dot(h, w)                    # downstream ops consume it like any SSA value

handle = evoxc.register_external_kernel("triton_add", _no_cpu_fallback)  # default slot
handle.batch_variant(fused_add)                                        # batched kernel sugar

@handle.impl("iree")                        # per-backend slot: numpy in, numpy out
def triton_add_iree(x):
    ...
```

evoxc contains no Triton (or any other) kernel code — kernels are written against this interface in another repo, and graphs carry only the name, so a process that loads a saved graph must re-register its kernels. Backends without host dispatch fall back to a registered portable decomposition (`@handle.portable` over an `@evoxc.defn`) with a warning, or raise an explicit `BackendError`. Worked example: [`evoxc/bench/examples/op_external.py`](evoxc/bench/examples/op_external.py).

### Custom ops

Not every custom op needs an external kernel. `evoxc.block` declares an op with specs, a portable decomposition written as an ordinary `@evoxc.defn` graph, optional per-backend implementations and optional batching/vjp rule registrations:

```python
l2norm = evoxc.block("l2norm", inputs=[TensorSpec((4, 8), evoxc.float32)],
                   outputs=[TensorSpec((4, 8), evoxc.float32)])

@l2norm.portable
@evoxc.defn
def _l2norm(x):
    return evoxc.divide(x, evoxc.sqrt(evoxc.sum(evoxc.square(x), axes=-1, keepdims=True)))

@l2norm.impl("numpy")                                     # optional fast path
def _l2norm_numpy(x):
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
```

The numpy backend keeps a registered impl as a `block_call` and dispatches it, and inlines the portable when a block has no numpy impl. Compiler adapters declare `custom_blocks=False` in v1 and reject a block graph with an explicit `BackendError` — never a silent reinterpretation.

### Inspectable and persistable graphs

Every stage object round-trips: a `Graph` as `.evoxcgraph`, a StableHLO `LoweredProgram` as `.mlir` text, a `CompiledArtifact` as `.evoxcartifact`. Loading with a mismatched backend, device or ABI is a `PersistenceError`, never a silent recompile. `evoxc.persist.Cache`/`FileCache` are the explicit caching primitives, and `build_cached` + `CompileCache` memoize the whole `trace → lower → compile → load` composition for meta-level loops (NAS, genetic programming, algorithm synthesis) that emit a different static program per outer iteration:

```python
cache = evoxc.CompileCache(maxsize=64)             # in-memory LRU; keys computed BEFORE tracing
exe_a = evoxc.build_cached(f, spec_x, spec_w, cache=cache, device="cpu")
exe_b = evoxc.build_cached(f, spec_x, spec_w, cache=cache, device="cpu")
print(exe_a is exe_b, cache.hits, cache.misses)  # True 1 1
```

### Async host ↔ device copies

`evoxc.io` schedules the explicit `Tensor.to` transfers on one process-global FIFO worker and hands back handles: validation is synchronous, transfer errors surface verbatim at `wait`/`result`, timeouts never cancel a copy. (Needs a device-transfer provider, e.g. `evoxc[iree]` on cuda/rocm.)

```python
copy = evoxc.io.prefetch(host_tensor, evoxc.Device("cuda", 0))   # -> AsyncCopy
copy.direction                                    # "host_to_device"
t = copy.result()                                 # wait + the device Tensor
out = evoxc.io.sink(device_tensor)                  # -> AsyncCopy, "device_to_host"
evoxc.io.wait_all()
```

### A conformance and benchmark harness

`evoxc.bench` ships 98 curated example programs — operation-level, small-block and end-to-end — stages every one through the explicit pipeline, and compares against pure-numpy references (plus PyTorch when installed) while reporting precision and speed.

```bash
python -m evoxc.bench                              # 98 examples, numpy/cpu conformance in ~10 s
python -m evoxc.bench --examples matmul,attention  # filter by name, category, or tag
```

Committed results, raw records and the probe scripts live in [`evoxc/bench/`](evoxc/bench) and [`bench_logs/`](bench_logs).

## Backends

| backend | how to enable | what it owns | validated on |
|---|---|---|---|
| **numpy interpreter** | nothing — the default | interprets EvoXIR directly with numpy kernels; all symbolic shapes | CPU (always) |
| **StableHLO export** | nothing — `evoxc.backends.stablehlo.export(graph)` | MLIR text emission; also the lowering input for every compiler | n/a (text) |
| **IREE** | `pip install evoxc[iree]`, `lower(..., backend="iree")` | IREE compile + HAL runtime, external-kernel host dispatch, device-resident payloads | llvm-cpu, CUDA, ROCm, Vulkan-SPIRV |
| **XLA** | `EVOXC_PJRT_PLUGIN=/path/to/xla_plugin.so`, `backend="xla"` | PJRT C API via ctypes on a user-provided plugin `.so` — no `jax`/`jaxlib` | CUDA and ROCm (real plugins); CPU plumbing via a compiled test plugin |
| **TVM** | `pip install evoxc[tvm]`, `backend="tvm"` | StableHLO frontend + Relax VM (jaxlib only for its MLIR bindings) | CPU end-to-end; ROCm compile-only |

**`opt_level`** is a first-class compile option (`"O0"`–`"O3"`, `"0"`–`"3"`, ints), one spelling, env `EVOXC_OPT_LEVEL`, per-adapter mapping — and an explicit raw compiler flag with the same name always wins over the derived one, never both. **`Capabilities`** lets an adapter state what it can and cannot do (custom calls, blocks, native RNG algorithms, …), so callers read the limits instead of finding them by crashing. And **a backend that cannot do something raises an explicit `BackendError` naming the op** and suggesting the remedy — never a silent fallback. Device kinds are `cpu`, `cuda`, `rocm` (alias `hip`) and `vulkan`; `evoxc.devices()` enumerates what this host actually has.

## Hardware support

<img src="https://raw.githubusercontent.com/EMI-Group/evoxc/main/assets/evoxc-hardware.svg" alt="Hardware support in evoxc (EvoX Compiler): one traced graph runs on the numpy interpreter (cpu only, the default backend) or through IREE (llvm-cpu, NVIDIA CUDA, AMD ROCm, Vulkan-SPIRV), XLA via a user-provided PJRT plugin (NVIDIA CUDA and AMD ROCm; cpu only through a compiled test plugin) or TVM (cpu end-to-end, AMD ROCm compile-only)." width="100%">

The same graph runs on CPU, NVIDIA CUDA, AMD ROCm and Vulkan-SPIRV — what changes is a `backend=` and a `device=`, never a rewrite. Device kinds are `cpu`, `cuda`, `rocm` (alias `hip`) and `vulkan`, and `evoxc.devices()` enumerates what this host actually has.

- **numpy interpreter** — the default backend and the always-available CPU fallback: no compiler, no codegen, every symbolic shape. `pip install evoxc` (numpy only) is the whole install.
- **IREE** — llvm-cpu, NVIDIA CUDA, AMD ROCm/HIP and Vulkan-SPIRV, all validated end-to-end (the Vulkan path on Mesa lavapipe, a software device). `pip install evoxc[iree]`.
- **XLA** — the PJRT C API driven directly over a **user-provided** plugin `.so` (`EVOXC_PJRT_PLUGIN` or `plugin_path`): CUDA and ROCm validated against real plugins, CPU plumbing validated with a compiled test plugin. No `jax`/`jaxlib`.
- **TVM** — CPU end-to-end; ROCm compile-only, because the wheel is built `USE_ROCM=OFF`. `pip install evoxc[tvm]`.

Placement is physical and explicit: `Tensor.to(device)` is the only transfer primitive, `run` outputs stay on the device they ran on, host inputs to a non-cpu executable are a `DeviceError` at the run boundary, and `.numpy()` on a non-cpu tensor raises `DeviceError` (`.to(cpu)` first). No API moves your data behind your back.

## Built on evoxc: `evoxc_hf`

[`evoxc_hf`](evoxc_hf/CONTEXT.md) loads a real HuggingFace checkpoint and runs the **whole model forward as ONE evoxc graph** — a 124M-parameter GPT-2, or a 14B-parameter Qwen3, goes through the same `trace → lower → compile → load → run` pipeline as everything else. It is a separate top-level package, used from this checkout, **not part of the `evoxc` distribution**; `import evoxc` never imports it, `torch` or `transformers`.

The API follows evoxc's value model in three layers that never blur: checkpoint IO produces **data** (`load_config` → frozen `ModelConfig`, `load_params` → `dict[str, np.ndarray]`, `load_hf` → a plain `LoadedModel` record); `forward_fn` is the pure **compute function** `(params, input_ids) -> logits` that captures no weights — every parameter is an ordinary graph input, never `evoxc.constant`; and `build`/`run` are the **staging** stages, `build` being `trace → lower → compile → load` via `build_cached`.

```python
import numpy as np
import evoxc, evoxc_hf

ckpt = "/path/to/checkpoint"                     # config.json + weights
loaded = evoxc_hf.load_hf(ckpt)                    # LoadedModel: .config, .params, .family, .model_dir
cfg, params = loaded.config, loaded.params

fwd = evoxc_hf.forward_fn(cfg, seq_len=32)         # (params, input_ids) -> logits; captures no weights
ids = np.zeros((1, 32), np.int32)                # static (batch, seq_len)

exe = evoxc_hf.build(fwd, cfg, backend="iree")     # numpy interpreter by default
logits = evoxc_hf.run(exe, params, ids)            # -> evoxc.Tensor on the executable's device
host = evoxc_hf.predict(loaded, ids, backend="iree")    # host-facing shorthand -> np.ndarray
oracle = evoxc_hf.reference_logits(cfg, params, ids)    # independent pure-numpy oracle
```

- **Families:** `llama` (also `mistral`/`qwen2`/`qwen3` — RMSNorm, RoPE, GQA, SwiGLU; Qwen3 adds per-head QK-Norm) and `gpt2` (also `distilgpt2` — LayerNorm, learned positions, `gelu_new`, tied head).
- **Parity** against real `torch`/`transformers` logits on eight checkpoints (`tiny_llama`, `tiny_mistral`, `tiny_gpt2`, `distilgpt2`, `gpt2`, `qwen3_0_6b`, `qwen3_1_7b`, `qwen3_8b`) and against the independent numpy oracle at 14B: ≤2e-4 max-abs / ≈5e-6 relative on the largest fp32 legs — fp32 accumulation order only.
- **Device-resident on ROCm, and measured against PyTorch honestly:** weights and ids are placed once (`place_params`/`place_ids`) and the device outputs feed back in (re-placing per call measured 2.4×–11.7× slower); every timed device call is `run` + the explicit `Executable.synchronize()` (the xla device-resident `run` only enqueues), so the head-to-head against eager **and** compiled `torch-rocm` is apples-to-apples — `xla`/rocm beats eager torch in 13 of 14 GPU cells (up to 8.3×, tightening to parity and then losing at Qwen3-14B) at `seq_len` 32/128, while `torch.compile` — carried as its own baseline — wins the 8–14B short-sequence cells; `iree`/rocm wins 6 of 14 and trails badly at 8B+. Numbers: [`bench_logs/hf_bench_results.md`](bench_logs/hf_bench_results.md).
- **Opt-in weight packing** (`pack_weights=True`, `pack_params`) collapses the weight tree into ~4 graph inputs — bit-identical, and it works around IREE's llvm-cpu SIGSEGV above ~161 distinct inputs (an unpacked 124M model has 196).
- **Deferred by design:** exact-`erf` GELU configs are numpy-only (`erf` is a StableHLO v1 deferral); dynamic shapes, kv-cache/incremental decode and generation loops are out of scope — this is the forward graph, not a serving stack.

## How evoxc compares

**evoxc is a compiler frontend and runtime, not a framework.** PyTorch and JAX are complete numerical ecosystems; Nx is the Elixir-native equivalent whose graph semantics evoxc deliberately mirrors (explicit `defn` graphs, `block`, `vectorize`, pluggable backends). evoxc does not try to replace them and borrows freely: numpy ergonomics, Nx's graph semantics, JAX's key-based RNG and function transforms.

**The one real difference: the program is a first-class object.** No eager mode, no implicit tracing, no mode switch, no compiler-owned semantics. Whatever a framework has to derive internally — by tracing at every call, by capturing a graph on a guard, by compiling behind an annotation — evoxc hands you directly: the graph is what you wrote, an inspectable, diffable, saveable, transformable value, and every stage from `trace` to `run` is a public function returning a public type you can stop at, keep, or ship.

**Choose evoxc when…** you want the program itself as an artifact (inspect, diff, save, transform, ship it to a device, hand it to another process); you need compiler neutrality — the same graph on a numpy interpreter, IREE, XLA or TVM, with no vendor IR in your source; you build meta-level or graph-generating workloads (NAS, genetic programming, algorithm synthesis) where a static program is data, `build_cached` is the compile boundary and `batch_functions` turns a whole population into one graph; or you want numpy-only deployment, a pure-Python dependency tree, and no compiler in the loop.

**Reach for something else when…** you need an ecosystem (layers, optimizers, pretrained models, serving), the most mature vendor GPU kernel coverage, fast eager iteration on dynamic Python control flow over tensors, or tens of thousands of users' worth of battle-testing. Those are real gaps in evoxc, not opinions — but they are also not reasons to accept a compiler that owns your program's meaning. If the library you would reach for does not itself run on evoxc, and **its license permits**, the concrete move is to rewrite or port that library onto evoxc with **EvoX Genesis** ([EMI-Group/genesis](https://github.com/EMI-Group/genesis)): the graph, the staging pipeline and the compiler adapters are already here, so the port adds the ecosystem instead of a second runtime.

## Install

```bash
pip install evoxc               # numpy only — a complete system, running on the default numpy interpreter
pip install "evoxc[iree]"       # + the IREE adapter: llvm-cpu, CUDA, ROCm, Vulkan-SPIRV
pip install "evoxc[compilers]"  # + IREE and TVM
```

From source — the checkout, for development (`requires-python >= 3.11`):

```bash
uv sync --all-extras          # the locked dev environment: numpy + every optional adapter
uv run pytest                 # CPU-only suite

# or with pip:
pip install -e ".[dev]"       # editable install + pytest
pytest
```

| extra | installs | enables |
|---|---|---|
| `dev` | `pytest` | the test suite |
| `interop` | `torch` | DLPack interop tests |
| `bench` | `torch` | `evoxc.bench`'s PyTorch comparison column |
| `iree` | `iree-base-compiler` + `iree-base-runtime` (3.11.x) | the IREE adapter on llvm-cpu / cuda / rocm / vulkan |
| `tvm` | `apache-tvm`, `jaxlib` | the TVM adapter (jaxlib only for its bundled MLIR bindings; `jax` is never imported) |
| `compilers` | IREE + TVM | both adapters at once |
| `hf` | `torch`, `transformers` | `evoxc_hf` reference oracles — never a runtime requirement |
| *(xla)* | **no extra** | the XLA adapter loads a user-provided PJRT C API plugin `.so` (`EVOXC_PJRT_PLUGIN` or `plugin_path`) |

```
evoxc/            the package: core, ir, ops, trace, transforms, backends, dist, sparse,
                random, io, pipeline, persist, build_cache, block, external, bench
evoxc_hf/         HuggingFace integration — separate top-level package, used from the checkout
tests/          pytest suite mirroring evoxc/ + spec-compliance tests
bench_logs/     probe scripts, committed bench records and generated reports
assets/         the diagrams in this README
CONTEXT.md      architecture, contracts and status (root, the source of truth)
```

`python -m evoxc.bench` runs the harness. Docs map: [`./CONTEXT.md`](CONTEXT.md) (architecture, design principles, benchmark status) · [`./evoxc/CONTEXT.md`](evoxc/CONTEXT.md) (the binding public API contract) · [`./evoxc_hf/CONTEXT.md`](evoxc_hf/CONTEXT.md) · [`./tests/CONTEXT.md`](tests/CONTEXT.md) — every package directory carries its own `CONTEXT.md` describing its purpose, API surface and constraints.

## Status & deferrals

Implemented and tested end to end: the explicit pipeline plus `bind`/`build`/`evaluate`/`build_cached`, graph transforms, `evoxc.dist` collectives, `evoxc.block` custom ops, `evoxc.external_call` kernels (numpy + IREE host and device-resident dispatch), `evoxc.sparse`, `evoxc.random` (3 algorithms, bit-exact against numpy), `evoxc.io`, `evoxc.persist`/`Cache`, `evoxc.numpy`, the numpy interpreter, the StableHLO exporter and the IREE / XLA / TVM adapters, plus the `evoxc.bench` harness.

Test suite: **6792 passed, 44 skipped, 4 xfailed, 0 failed** with all extras installed (IREE + TVM + torch); **6367 passed, 95 skipped, 3 xfailed, 0 failed** compiler-free. Every skip is a host or dependency guard (no NVIDIA CUDA device, no Vulkan ICD, no PJRT plugin, a torch-present inverse pin, the io pristine-process guard).

<details>
<summary><strong>v1 deferrals</strong> — each of them an explicit error naming the op</summary>

- Vectorizing control-flow regions; `conv` VJP (there is no transposed-conv op); symbolic-length `scan`.
- `linalg.inv`, `norm` beyond vector norms, `det`; non-constant `pad` modes; symbolic-bound `arange`/`linspace`.
- StableHLO export deferrals: `cumsum`, `cumprod`, `diagonal`, `erf`/`gelu`, `tril`/`triu`, `flip`, `roll`, `nan_to_num`, `random_multinomial`, and the linalg `solve`/`cholesky`/`qr`/`matrix_rank`/`svd`/`matrix_exp` (`eigh` is v1 via a cyclic-Jacobi while-loop composition).
- Sparse on compiler backends (densify with `evoxc.sparse.to_dense`), `sparse@sparse` matmul, sparse constants, whole-sparse `bind`.
- Collectives and `runtime_call` are not compiled by any adapter in v1 (multi-rank runtime channels are not wired).
- Dynamic dims are supported per-op by the exporter, not universally: reshape/slice/pad/conv and `reduce_mean` over a dynamic reduced dim are rejected at `lower()` time with a clear `BackendError` (which is why vmap of softmax/layernorm/attention-style per-sample functions defers on compiler backends, while the numpy backend handles every symbolic shape).
- `vmap`/`grad`/`jvp`/`vjp` have no rules for the random family and for ops without a registered rule → `TransformError`, never a Python-loop fallback.

**The guarantee: a deferral is an explicit `BackendError` (or `TransformError`) naming the op and suggesting the remedy — never a silent fallback.**

</details>

## Acknowledgements

evoxc is written solely with **EvoX Genesis** ([EMI-Group/genesis](https://github.com/EMI-Group/genesis)): every line of code, every test and every document in this repository — this README and the diagrams above included — is produced with it.

## License

Apache-2.0. evoxc is part of the **EvoX** ecosystem: the compiler, frontend and graph runtime underneath evolutionary computation workloads, deliberately small enough to read.
