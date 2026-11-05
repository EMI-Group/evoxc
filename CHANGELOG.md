# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
- The numpy reference interpreter no longer pays a full operand/validation walk per op: each block executes a cached dispatch plan (kernel resolved once, operand/result value ids, a static-result shape+dtype fast path) over one flat SSA-id value store, dead intermediates are released at their last use, and the kernels keep only the error path for broadcast validation. On the TensorNEAT 0.2.0 workload this is −30 % (`xor_p500`), −58 % (`cartpole_p100`) and −8 % (`custom_pagie_p500`, which is numpy-compute-bound) ms/generation, with bit-identical fitness trajectories, lower peak RSS and unchanged executed op counts.
- ndarray-backed `core.Tensor` construction shares one immutable `Device("cpu", 0)` instance (0.55 → 0.18 µs per tensor); no public API change.
- Adopted IREE 3.13: the `iree` and `compilers` extras now pin `iree-base-compiler>=3.13.0,<3.14` and `iree-base-runtime>=3.13.0,<3.14` (compiler and runtime stay matched, artifacts are version-specific). The llvm-cpu and ROCm/HIP legs were re-validated device-resident on an AMD Instinct MI300X (gfx942, ROCm 7.1): `while_loop`, `scan`, `while` nested in `cond` and the MO/NSGA2-class `non_dominate_rank` loop now compile and run with numpy parity, 0 crashes in 30 compiles, where 3.10/3.11 aborted at run on the HIP HAL and SIGSEGV'd `iree-compile` nondeterministically (records: `evoxc/bench/iree313_rocm_findings.md`). The adapter now also owns the device→host readback explicitly, because iree 3.13's `DeviceArray.to_host()` fast path is broken on the Vulkan HAL (`host allocation import pointer does not satisfy Vulkan minImportedHostPointerAlignment`): host-mappable buffers keep a zero-copy mapping and device buffers stage through a thread-local `queue_copy`, which fixed the Vulkan suite (6/6) and costs a real DMA copy-out on device memory (16 MB ROCm readback 0.37 → 1.51 ms) while llvm-cpu readback stays zero-copy. CUDA (no NVIDIA GPU on the validating host), GPU Vulkan, multi-rank/collectives, cross-version artifact interop, the xla/tvm legs and the HF/benchmark campaigns were NOT re-validated and stay scoped to the 3.11 release they were measured on.
### Fixed
- `evoxc.remainder` now gives the same values on every backend: its contract is numpy floormod (result takes the sign of the divisor), which the numpy kernel already implemented, and the StableHLO export now emits `stablehlo.remainder` (fmod) plus the inline sign correction instead of bare fmod, so negative operands agree between the numpy interpreter and iree. The op docs state floormod semantics correctly.
### Added
- Regression tests for the interpreter's dispatch plan and dead-value release, for the floormod `remainder` export (goldens, `BackendError` for dtypes numpy has no remainder for, iree llvm-cpu parity) and for the shared host `Device` instance.
## [0.1.0] - 2026-10-29

First release: the explicit `trace → Graph → lower → LoweredProgram → compile →
CompiledArtifact → load → Executable → run → Tensor` pipeline, the frontend
language, EvoXIR and the backend-adapter contract. Optimization, kernel
scheduling and code generation are delegated to external compilers.

### Added

- The explicit staging pipeline (`trace`/`lower`/`compile`/`load`/`run`) with its
  documented shorthand compositions `bind`, `build`, `evaluate` and
  `build_cached` (memoized build over a `CompileCache`).
- `@evoxc.defn` graph definitions, `evoxc.trace`, `Graph`, and runtime control
  flow via `cond`/`while_loop`/`scan`.
- EvoXIR: region-based SSA frontend IR with `Value`/`Op`/`Block`/`Region`/
  `Function`/`Module`/`Builder`, a 119-def op registry, verification and
  serialization.
- Graph→graph transforms: `vectorize`, `vmap`, `grad`, `jvp`, `vjp`, plus
  `batch_functions` to merge a list of same-signature programs into one graph.
- The numpy interpreter backend (default, CPU) implementing every frontend op.
- The StableHLO exporter (MLIR text), the shared lowering input for compiler
  backends.
- Optional compiler adapters: IREE (llvm-cpu, cuda, rocm, vulkan), XLA via the
  PJRT C API over a user-provided plugin `.so`, and TVM — each loaded lazily and
  failing with an explicit error when absent.
- `evoxc.numpy` (`enp`), a graph-building NumPy-style namespace.
- `evoxc.dist`: explicit collectives (`all_reduce`, `all_gather`,
  `reduce_scatter`, `all_to_all`, `broadcast`, `collective_permute`) plus
  `group`/`rank`/`world_size` graph scalars.
- `evoxc.block` custom ops (portable impls, per-backend impls, batching and
  derivative rules) and `evoxc.external_call` external kernels with per-backend
  slots, device-resident dispatch and declarative batched-variant sugar.
- `evoxc.sparse`: COO/CSR/CSC value model and 16 graph ops with batching and
  derivative rules (numpy backend).
- `evoxc.random`: stateless, key-based functional RNG with three algorithms
  (`splitmix64`, `threefry2x32`, `philox4x32_10`) and bit-identical results for a
  given key.
- `evoxc.io`: explicit async host↔device copy handles (`prefetch`/`sink`/
  `wait_all`) over the explicit `Tensor.to` transfer.
- `evoxc.persist`: versioned, self-describing, integrity-checked save/load
  container, an explicit `Cache`, and `Graph.save`/`Graph.load`.
- Pytree utilities (`tree_map`, `tree_leaves`, `tree_structure`,
  `tree_flatten`/`tree_unflatten`, `register_pytree_node`) for structured inputs
  and outputs.
- `evoxc.bench`: a conformance and benchmark harness with 98 curated example
  programs (op, block and end-to-end groups) validated against pure-numpy — and,
  when available, PyTorch — references.
- A pure-Python wheel built with `uv_build`, with the version single-sourced from
  `pyproject.toml` and read back through `importlib.metadata`.
- Two independently published distributions from one repository: `evoxc` (the
  workspace-root project) and `evoxc-hf` (the workspace member at
  `evoxc_hf/`, src layout `evoxc_hf/src/evoxc_hf/`), each with its own version,
  its own release tag (`v<version>` / `evoxc-hf-v<version>`) and its own PyPI
  release. `pip install evoxc` is unchanged and numpy-only; `pip install
  evoxc-hf` adds the HuggingFace layer, which builds only on evoxc's public API
  (`evoxc` never imports `evoxc_hf`).
- The HuggingFace reference oracles (`torch` + `transformers`, used only to
  validate and benchmark that layer — never runtime requirements) are the `hf`
  extra of `evoxc-hf`, installed with `pip install "evoxc-hf[hf]"`; this
  replaces `pip install evoxc[hf]`, which no longer exists.
- Development and CI install both projects from the single root `uv.lock` with
  `uv sync --all-packages` (e.g. `uv sync --all-packages --extra dev`); a plain
  `uv sync` installs the root project only and uninstalls the workspace member.

[0.1.0]: https://github.com/EMI-Group/evoxc/releases/tag/v0.1.0
