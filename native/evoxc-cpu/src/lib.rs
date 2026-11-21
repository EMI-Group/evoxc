//! Native CPU execution engine for EvoX Compiler.
//!
//! `evoxc-cpu` compiles an [`evoxc_core::Program`] (the same IR the envelope
//! codec and the transforms use) into an immutable [`Plan`] and executes it on
//! typed host buffers.  It depends only on `evoxc-core`: no Python, no device
//! APIs, no compiler backends.
//!
//! Pipeline:
//!
//! ```text
//! Program --compile--> Plan --new--> RunCtx --run--> Vec<Buffer>
//! ```
//!
//! * [`plan::compile`] resolves every op's kernel, records liveness and the
//!   static-result fast path, and fails explicitly on an op it cannot run.
//! * [`exec::RunCtx`] is the flat-store interpreter (nested regions, dynamic
//!   control flow, symbolic-dim binding, host callbacks).
//! * [`buffer`] is the typed value representation, [`scalar`] the numeric
//!   vocabulary kernels are generic over.
pub mod buffer;
pub mod dims;
pub mod exec;
pub mod kernels;
pub mod plan;
pub mod scalar;

pub use buffer::{Buffer, Storage};
pub use exec::{Callbacks, RunCtx, RunOptions, Value};
pub use plan::{compile, BlockPlan, FunctionPlan, OpSpec, Plan, Step};
pub use scalar::{Num, Scalar, ScalarValue};

use evoxc_core::{Error, Program, Result};

/// Compile `program` and execute one call over flat inputs.
///
/// A plan that needs `block_call`/`external_call`/`runtime_call` and runs
/// without callbacks is an explicit `BackendError` naming what is missing,
/// raised before any op executes.
pub fn run_program(
    program: &Program,
    plan: &std::sync::Arc<Plan>,
    inputs: Vec<Buffer>,
    options: RunOptions,
    callbacks: Option<&dyn Callbacks>,
) -> Result<Vec<Buffer>> {
    if plan.needs_callbacks() && callbacks.is_none() {
        let mut needs = Vec::new();
        if !plan.required_blocks.is_empty() {
            needs.push(format!("block_call {:?}", plan.required_blocks));
        }
        if !plan.required_external.is_empty() {
            needs.push(format!("external_call {:?}", plan.required_external));
        }
        if !plan.required_runtime.is_empty() {
            needs.push(format!("runtime_call {:?}", plan.required_runtime));
        }
        return Err(Error::backend(format!(
            "the graph requires host callbacks ({}) but no resolver was supplied",
            needs.join(", ")
        )));
    }
    let mut ctx = RunCtx::new(program, plan.clone(), options, callbacks);
    ctx.run(inputs)
}

/// Compile and run in one call (convenience for tests and the CLI harness).
pub fn evaluate(
    program: &Program,
    inputs: Vec<Buffer>,
    options: RunOptions,
    callbacks: Option<&dyn Callbacks>,
) -> Result<Vec<Buffer>> {
    let plan = std::sync::Arc::new(plan::compile(program)?);
    run_program(program, &plan, inputs, options, callbacks)
}
