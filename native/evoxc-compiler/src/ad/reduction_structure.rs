//! Reduction and structural derivative rules.
//!
//! Port of `_reduction_jvp` / `_reduction_vjp` and `_structural_jvp` /
//! `_structural_vjp` from `evoxc/transforms/rules.py`.  Filled in by the
//! reduction/structural AD rule workstream.

use crate::Registry;

/// Register the reduction and structural JVP and VJP rules.
pub fn register_rules(_registry: &mut Registry) {}
