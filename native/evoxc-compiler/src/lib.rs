//! evoxc-compiler: native graph construction and builtin transforms.
//!
//! Owns the mutable `Builder` (schema validation + native inference), the
//! `vectorize`/`vmap` engine with the builtin batching rules, the AD engines
//! (`jvp`/`vjp`/`grad`) with the builtin derivative rules, and extension-rule
//! inlining.  See `native/CONTEXT.md` for the FFI contract and
//! `native/evoxc-compiler/CONTEXT.md` for the internal invariants.

/// Crate version marker; the compiler modules are added by the crate's
/// builder / batching / autodiff modules.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
