//! evoxc-runtime: native runtime skeleton (sessions, buffers, submissions).
//!
//! Depends on `evoxc-core` only — never on the compiler crate and never on
//! PyO3.  The model is the proposal's §12/§13: a session owns device contexts,
//! buffers lease the context that can use them, submissions retain their
//! inputs until the work completes, and executables outlive their outputs.

/// Crate version marker; the runtime model modules are added by the crate's
/// session / buffer / submission modules.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
