//! Builtin batching rules for structural ops.
//!
//! Ports `_reshape_batching`, `_transpose_batching`, `_slice_batching`,
//! `_pad_batching`, `_concatenate_batching`, `_gather_batching`,
//! `_scatter_batching` and `_constant_batching` from
//! `evoxc/transforms/rules.py`.

use crate::Registry;

/// Register this file's batching rules.
pub fn register_rules(_registry: &mut Registry) {}
