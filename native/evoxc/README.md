# evoxc (Rust)

The reusable Rust core of [EvoX Compiler](https://github.com/EMI-Group/evoxc) — one
crate over the workspace's numerical core, so a Rust program (or another language
binding) depends on this facade instead of on the individual crates.

It contains:

- **`evoxc::core`** — dtypes and promotion, symbolic dims (`Dim`/`DimExpr`),
  `ValueType`, typed attributes, the generated op schema table, the EvoXIR graph
  (`Program`/`Function`/`Block`/`Region`/`Op`/`Value`), the `evoxc-native-ir`
  envelope codec, native shape inference and `verify`.
- **`evoxc::compiler`** — the mutable `Builder`, `vectorize`/`vmap`,
  `jvp`/`vjp`/`grad`, and the `batch_functions` population merge.
- **`evoxc::cpu`** — the native CPU execution engine: compile a `Program` into an
  immutable plan and run it over typed host buffers.
- **`evoxc::runtime`** — the session / device / buffer / executable skeleton.

The pipeline is explicit:

```text
Program → plan::compile → Plan → run_program → Buffer
```

## Features

- `cpu` (default) — the `evoxc-cpu` execution engine.
- `runtime` — the `evoxc-runtime` session skeleton.
- `full` — both.
- `--no-default-features` — the graph + transform layer only (`core` + `compiler`),
  for a frontend that brings its own execution.

## Usage

```toml
[dependencies]
evoxc = "0.1"
```

```rust
use evoxc::compiler::Builder;
use evoxc::core::{Dim, DType, ValueType};
use evoxc::cpu::{plan, run_program, Buffer, RunOptions, Storage};
```

## Links

- Repository: <https://github.com/EMI-Group/evoxc>
- API documentation: <https://docs.rs/evoxc>
