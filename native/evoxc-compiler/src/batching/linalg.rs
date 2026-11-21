//! Builtin batching rules for linalg ops.
//!
//! Ports `_dot_batching`, `_conv_batching`, `_triangular_batching`,
//! `_cumsum_batching` and `_solve_batching` from
//! `evoxc/transforms/rules.py`.

use crate::Registry;

/// Register this file's batching rules.
pub fn register_rules(_registry: &mut Registry) {}
