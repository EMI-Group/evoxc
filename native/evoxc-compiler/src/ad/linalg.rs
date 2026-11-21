//! Linalg derivative rules: `dot`, `conv`, `tril`/`triu`, `cumsum`, `solve`.
//!
//! Port of the linalg JVP/VJP rules in `evoxc/transforms/rules.py`.  Filled in
//! by the linalg AD rule workstream; `conv` VJP is an explicit v1 deferral
//! (the IR has no transposed-convolution op).

use crate::Registry;

/// Register the linalg JVP and VJP rules.
pub fn register_rules(_registry: &mut Registry) {}
