//! evoxc — the reusable Rust core of [EvoX Compiler](https://github.com/EMI-Group/evoxc).
//!
//! One crate over the workspace's numerical core, so a Rust program (or another
//! language binding) depends on this facade instead of on the individual crates:
//!
//! | module | crate | contents |
//! |---|---|---|
//! | [`core`] | `evoxc-core` | dtypes, dims, `ValueType`, op schemas, graph IR, envelope codec, inference, `verify` |
//! | [`compiler`] | `evoxc-compiler` | `Builder`, `vectorize`/`vmap`, `jvp`/`vjp`/`grad`, `batch_functions` |
//! | [`runtime`] | `evoxc-runtime` | session / device / buffer / executable skeleton (feature `runtime`) |
//! | [`cpu`] | `evoxc-cpu` | `plan::compile` + `run_program`: the native CPU engine (feature `cpu`, default) |
//!
//! The pipeline is explicit: `Program` (built through the `Builder` or decoded
//! from an `evoxc-native-ir` envelope) → [`cpu::plan::compile`] → [`cpu::Plan`]
//! → [`cpu::run_program`] → `Buffer`s.
//!
//! # Features
//!
//! * `cpu` (default) — the `evoxc-cpu` execution engine.
//! * `runtime` — the `evoxc-runtime` session skeleton.
//! * `full` — both.
//!
//! With `--no-default-features` the crate is the graph + transform layer only
//! (`core` + `compiler`), for a frontend that brings its own execution.
//!
//! # Example
//!
//! ```
//! # #[cfg(feature = "cpu")] {
//! use evoxc::compiler::Builder;
//! use evoxc::core::{Dim, DType, ValueType};
//! use evoxc::cpu::{plan, run_program, Buffer, RunOptions, Storage};
//!
//! // f(x) = x * x, elementwise over a float32[2].
//! let ty = ValueType::of(DType::Float32, [Dim::Int(2)]);
//! let mut builder = Builder::new("square");
//! builder.open_function("main", &[ty]).unwrap();
//! let args = builder.block_arguments().unwrap();
//! let out = builder.binary("multiply", args[0], args[0]).unwrap();
//! builder.set_terminator(&[out], None).unwrap();
//! let program = builder.finish();
//!
//! // Compile once, run with typed host buffers.
//! let plan = std::sync::Arc::new(plan::compile(&program).unwrap());
//! let input = Buffer::new(DType::Float32, vec![2], Storage::F32(vec![3.0, 4.0])).unwrap();
//! let outputs =
//!     run_program(&program, &plan, vec![input], RunOptions::default(), None).unwrap();
//! assert_eq!(outputs[0].storage, Storage::F32(vec![9.0, 16.0]));
//! # }
//! ```
//!
//! Crate-by-crate contracts: `native/CONTEXT.md` in the repository.
pub use evoxc_compiler as compiler;
pub use evoxc_core as core;
#[cfg(feature = "cpu")]
pub use evoxc_cpu as cpu;
#[cfg(feature = "runtime")]
pub use evoxc_runtime as runtime;

/// The most-used types of the facade.
pub mod prelude {
    pub use evoxc_compiler::Builder;
    pub use evoxc_core::graph::Program;
    pub use evoxc_core::{DType, ValueType};
    #[cfg(feature = "cpu")]
    pub use evoxc_cpu::plan::{self, compile, Plan};
    #[cfg(feature = "cpu")]
    pub use evoxc_cpu::{run_program, Buffer, Callbacks, RunOptions};
}
