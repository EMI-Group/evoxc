//! Kernel registry: op name → native implementation.
//!
//! Each category module owns its file and registers its ops into the shared
//! table (see [`KernelMap`]).  An op with no entry is reported by
//! [`unsupported_ops`] and rejected at plan build with an explicit
//! `BackendError` — the engine never falls back to Python or guesses.
//!
//! Kernel contract:
//!
//! ```text
//! fn(ctx: &mut RunCtx, op: &OpSpec, operands: &[Value]) -> Result<Vec<Value>>
//! ```
//!
//! * `ctx` carries the symbolic-dim bindings, the rank context, host
//!   callbacks and the region executor.
//! * `op.attrs` already has schema defaults filled in; `op.result_types`
//!   are the declared result types the runtime validates afterwards.
//! * operands arrive in op order; control-flow kernels call
//!   `ctx.run_region(region, args)`.
use std::collections::HashMap;
use std::sync::OnceLock;

use evoxc_core::Result;

use crate::exec::{RunCtx, Value};
use crate::plan::OpSpec;

pub mod collective;
pub mod control;
pub mod elementwise;
pub mod indexing;
pub mod linalg;
pub mod linalg_aux;
pub mod random;
pub mod reductions;
pub mod sorting;
pub mod structure;
pub mod util;

/// A resolved kernel implementation.
pub type KernelFn = fn(&mut RunCtx<'_>, &OpSpec, &[Value]) -> Result<Vec<Value>>;

/// The registration table each category module fills.
pub type KernelMap = HashMap<&'static str, KernelFn>;

/// Every dense (non-sparse, non-terminator) IR op.  `sparse_*` ops are out of
/// scope for the dense CPU engine and are rejected explicitly.
pub const DENSE_OPS: &[&str] = &[
    // control
    "block_call",
    "call",
    "constant",
    "external_call",
    "if",
    "runtime_call",
    "stop_gradient",
    "while",
    // elementwise
    "abs",
    "acos",
    "add",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "cast",
    "ceil",
    "cos",
    "divide",
    "erf",
    "exp",
    "floor",
    "gelu",
    "log",
    "log1p",
    "logical_and",
    "logical_not",
    "logical_or",
    "maximum",
    "minimum",
    "multiply",
    "nan_to_num",
    "negate",
    "power",
    "relu",
    "remainder",
    "round",
    "sigmoid",
    "sign",
    "sin",
    "sqrt",
    "square",
    "subtract",
    "tan",
    "tanh",
    // comparison
    "equal",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "not_equal",
    // structure
    "broadcast",
    "concatenate",
    "diag",
    "flip",
    "gather",
    "pad",
    "reshape",
    "roll",
    "scatter",
    "select",
    "slice",
    "tile",
    "transpose",
    // reduction
    "argmax",
    "argmin",
    "cumprod",
    "cumsum",
    "reduce_max",
    "reduce_mean",
    "reduce_min",
    "reduce_prod",
    "reduce_sum",
    // sorting
    "argsort",
    "sort",
    // linalg
    "cholesky",
    "conv",
    "diagonal",
    "dot",
    "eigh",
    "matrix_exp",
    "matrix_rank",
    "qr",
    "solve",
    "svd",
    "tril",
    "triu",
    // collective
    "all_gather",
    "all_reduce",
    "all_to_all",
    "broadcast_collective",
    "collective_permute",
    "rank",
    "reduce_scatter",
    "world_size",
    // random
    "random_key_mix",
    "random_multinomial",
    "random_normal",
    "random_permutation",
    "random_randint",
    "random_uniform",
];

fn build_table() -> KernelMap {
    let mut table = KernelMap::new();
    elementwise::register(&mut table);
    reductions::register(&mut table);
    sorting::register(&mut table);
    indexing::register(&mut table);
    structure::register(&mut table);
    linalg::register(&mut table);
    linalg_aux::register(&mut table);
    random::register(&mut table);
    control::register(&mut table);
    collective::register(&mut table);
    table
}

/// The process-wide kernel table (built once).
pub fn table() -> &'static KernelMap {
    static TABLE: OnceLock<KernelMap> = OnceLock::new();
    TABLE.get_or_init(build_table)
}

/// Resolve an op's kernel, or `None` when the engine does not implement it.
pub fn lookup(name: &str) -> Option<KernelFn> {
    table().get(name).copied()
}

/// True when the engine implements `name`.
pub fn is_supported(name: &str) -> bool {
    table().contains_key(name)
}

/// The dense ops this engine does NOT implement, sorted.
pub fn unsupported_ops() -> Vec<&'static str> {
    let table = table();
    let mut missing: Vec<&'static str> = DENSE_OPS
        .iter()
        .copied()
        .filter(|name| !table.contains_key(name))
        .collect();
    missing.sort_unstable();
    missing
}
