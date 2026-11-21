//!  kernels (registered into the shared table).
//!
//! Every entry is `pub fn register(map: &mut KernelMap)`; the table is built
//! once in `super::build_table`.
use super::KernelMap;

pub fn register(_map: &mut KernelMap) {}
