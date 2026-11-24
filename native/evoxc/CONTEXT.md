# evoxc — the published Rust facade

## Intent

`evoxc` is the crate other Rust programs and language frontends depend on: one crates.io package that re-exports the workspace's numerical core under stable module names.
It holds no logic of its own — every item is a re-export of `evoxc-core`, `evoxc-compiler`, the optional `evoxc-cpu` and the optional `evoxc-runtime` — so a change here never changes behavior.
The PyO3 binding (`../evoxc-python/`) depends on this crate only (`features = ["full"]`), never on the individual crates.

## API Surface

- `core` = `evoxc_core` (always): dtypes, dims, `ValueType`, op schemas, graph IR, envelope codec, inference, `verify`.
- `compiler` = `evoxc_compiler` (always): `Builder`, `vectorize`/`vmap`, `jvp`/`vjp`/`grad`, `batch_functions`.
- `cpu` = `evoxc_cpu` (feature `cpu`, on by default): `plan::compile`, `Plan`, `run_program`, `Buffer`, `Callbacks`, `RunOptions`.
- `runtime` = `evoxc_runtime` (feature `runtime`): the session/device/buffer/executable skeleton.
- `prelude`: `Program`, `DType`, `ValueType`, `Builder`, plus — under `cpu` — `plan`, `compile`, `Plan`, `run_program`, `Buffer`, `Callbacks`, `RunOptions`.

Features: `default = ["cpu"]`, `cpu = ["dep:evoxc-cpu"]`, `runtime = ["dep:evoxc-runtime"]`, `full = ["cpu", "runtime"]`.
`--no-default-features` is the graph + transform layer only (`core` + `compiler`), for a frontend that brings its own execution.

## Constraints

- Re-exports only: nothing may be defined here beyond the `prelude` and the crate doc; a real implementation belongs in the owning crate.
- No `pyo3`, no third-party dependency, no Python knowledge: the facade stays a pure Rust library.
- Publishing order and metadata rules are `../CONTEXT.md` "Publishing (crates.io)"; this crate is last in that order because it depends on the other four.
- The crate-doc example is a doctest and must keep compiling under BOTH `cargo test -p evoxc` and `cargo test -p evoxc --no-default-features` — its body is wrapped in `# #[cfg(feature = "cpu")] { … }` because cargo passes the crate's feature cfgs to doctests, so it executes under the default features and becomes an empty body without them.
- `tests/smoke.rs` is `#![cfg(feature = "cpu")]` and must prove the facade alone is enough: build through `compiler::Builder`, `core::verify`, encode/decode through `core::envelope`, `cpu::plan::compile`, `cpu::run_program`, then check a value and a shape.
- The crates.io surface lives here: `readme = "README.md"`, inherited `documentation`, keywords/categories, and `[package.metadata.docs.rs] all-features = true` so docs.rs renders `cpu` and `runtime`.

## Routing Table

| Path | Area |
|---|---|
| `./Cargo.toml` | Package metadata, the feature graph (`cpu` default / `runtime` / `full`), docs.rs metadata, and the four re-export dependencies. |
| `./src/lib.rs` | The whole implementation: module re-exports, the crate doc with its runnable doctest, and `prelude`. |
| `./tests/smoke.rs` | Facade-only end-to-end test (build → verify → encode/decode → compile → run). |
| `./README.md` | The crates.io front page: what the crate is, the explicit pipeline, the feature flags, links. |
