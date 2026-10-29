# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
