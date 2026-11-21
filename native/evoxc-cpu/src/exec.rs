//! The runtime: a flat SSA value store plus one plan-driven block loop.
//!
//! Execution order is block op order (the effect ordering).  Regions are
//! re-entered through [`RunCtx::run_region`], which binds the region's entry
//! block arguments and extends the symbolic-dim bindings; genuinely dynamic
//! control flow is interpreted per iteration (never specialized).
//!
//! Values are `Rc<Buffer>` so that an operand or a region result can be shared
//! without copying array data.  Every kernel result is validated against the
//! op's declared result type (exact dtype, evaluated shape) — a wrong kernel
//! can never silently produce a wrong value.
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use evoxc_core::{Dim, Error, Program, Region, Result, ValueType};

use crate::buffer::{shape_text, Buffer};
use crate::dims::{self, Bindings};
use crate::plan::{OpSpec, Plan};

/// One SSA value in the flat run-time store.
pub type Value = Rc<Buffer>;

/// Host callbacks for `block_call`, `external_call` and `runtime_call`.
///
/// The engine itself is Python-free; the PyO3 layer implements this trait over
/// a user resolver.  A plan that needs a callback and runs without one is an
/// explicit `BackendError`, never a silent skip.
pub trait Callbacks {
    fn call_block(
        &self,
        name: &str,
        args: &[Value],
        static_args: &serde_json::Value,
    ) -> Result<Vec<Buffer>>;
    fn call_external(&self, name: &str, args: &[Value]) -> Result<Vec<Buffer>>;
    fn call_runtime(&self, callback: &str, args: &[Value]) -> Result<Vec<Buffer>>;
}

/// Per-run process configuration (the `rank`/`world_size` graph scalars).
#[derive(Debug, Clone, Copy)]
pub struct RunOptions {
    pub rank: i64,
    pub world_size: i64,
}

impl Default for RunOptions {
    fn default() -> RunOptions {
        RunOptions {
            rank: 0,
            world_size: 1,
        }
    }
}

/// Build the `Vec<Value>` a kernel returns for fresh buffers.
pub fn values(items: Vec<Buffer>) -> Vec<Value> {
    items.into_iter().map(Rc::new).collect()
}

/// Build a one-result kernel return value.
pub fn single(buffer: Buffer) -> Result<Vec<Value>> {
    Ok(vec![Rc::new(buffer)])
}

/// Move a `Buffer` out of a shared value (copying only when still shared).
pub fn unwrap_value(value: Value) -> Buffer {
    Rc::try_unwrap(value).unwrap_or_else(|shared| (*shared).clone())
}

/// One execution's mutable state.
pub struct RunCtx<'p> {
    pub program: &'p Program,
    pub plan: Arc<Plan>,
    pub store: Vec<Option<Value>>,
    pub bindings: Bindings,
    pub rank: i64,
    pub world_size: i64,
    pub callbacks: Option<&'p dyn Callbacks>,
}

impl<'p> RunCtx<'p> {
    pub fn new(
        program: &'p Program,
        plan: Arc<Plan>,
        options: RunOptions,
        callbacks: Option<&'p dyn Callbacks>,
    ) -> RunCtx<'p> {
        let capacity = plan.max_value;
        RunCtx {
            program,
            plan,
            store: vec![None; capacity],
            bindings: Bindings::new(),
            rank: options.rank,
            world_size: options.world_size,
            callbacks,
        }
    }

    /// Execute the entry function over flat inputs.
    pub fn run(&mut self, inputs: Vec<Buffer>) -> Result<Vec<Buffer>> {
        let signature = self.plan.functions[self.plan.entry].clone();
        if inputs.len() != signature.input_types.len() {
            return Err(Error::backend(format!(
                "program expects {} input(s), got {}",
                signature.input_types.len(),
                inputs.len()
            )));
        }
        for (index, (buffer, ty)) in inputs.iter().zip(&signature.input_types).enumerate() {
            if buffer.dtype != ty.dtype {
                return Err(Error::dtype(format!(
                    "input {index}: expected dtype {}, got {}",
                    ty.dtype.name(),
                    buffer.dtype.name()
                )));
            }
            dims::bind_shape(
                &mut self.bindings,
                &ty.shape,
                &buffer.shape,
                &format!("input {index}"),
            )?;
        }
        let wrapped: Vec<Value> = inputs.into_iter().map(Rc::new).collect();
        let outputs = self.run_block(signature.entry, wrapped)?;
        if outputs.len() != signature.output_types.len() {
            return Err(Error::backend(format!(
                "program produced {} output tensor(s), expected {}",
                outputs.len(),
                signature.output_types.len()
            )));
        }
        for (index, (buffer, ty)) in outputs.iter().zip(&signature.output_types).enumerate() {
            if buffer.dtype != ty.dtype {
                return Err(Error::backend(format!(
                    "output {index}: expected dtype {}, got {}",
                    ty.dtype.name(),
                    buffer.dtype.name()
                )));
            }
        }
        Ok(outputs.into_iter().map(unwrap_value).collect())
    }

    /// Execute one block's plan; returns the terminator's values.
    pub fn run_block(&mut self, block_index: usize, args: Vec<Value>) -> Result<Vec<Value>> {
        let plan = self.plan.clone();
        let block = plan
            .blocks
            .get(block_index)
            .and_then(|slot| slot.as_ref())
            .cloned()
            .ok_or_else(|| {
                Error::backend(format!(
                    "block #{block_index} was not compiled (unreachable from the entry function)"
                ))
            })?;
        if block.arg_ids.len() != args.len() {
            return Err(Error::backend(format!(
                "block #{block_index} expects {} argument(s), got {}",
                block.arg_ids.len(),
                args.len()
            )));
        }
        for (value_id, value) in block.arg_ids.iter().zip(args) {
            if let Some(slot) = self.store.get_mut(*value_id) {
                *slot = Some(value);
            }
        }
        for step in &block.steps {
            let mut operands: Vec<Value> = Vec::with_capacity(step.operands.len());
            for value_id in &step.operands {
                match self.store.get(*value_id).and_then(|slot| slot.clone()) {
                    Some(value) => operands.push(value),
                    None => return Err(unresolved(*value_id)),
                }
            }
            let results = (step.kernel)(self, &step.spec, &operands)?;
            if results.len() != step.results.len() {
                return Err(Error::backend(format!(
                    "kernel for op '{}' produced {} result(s), expected {}",
                    step.spec.name,
                    results.len(),
                    step.results.len()
                )));
            }
            for (index, value) in results.into_iter().enumerate() {
                self.validate_result(step, index, &value)?;
                let slot = self.store.get_mut(step.results[index]).ok_or_else(|| {
                    Error::verification(format!(
                        "result value #{} is out of the store's range",
                        step.results[index]
                    ))
                })?;
                *slot = Some(value);
            }
            for value_id in &step.free {
                if let Some(slot) = self.store.get_mut(*value_id) {
                    *slot = None;
                }
            }
        }
        let mut outputs = Vec::with_capacity(block.return_ids.len());
        for value_id in &block.return_ids {
            match self.store.get(*value_id).and_then(|slot| slot.clone()) {
                Some(value) => outputs.push(value),
                None => return Err(unresolved(*value_id)),
            }
        }
        Ok(outputs)
    }

    /// Bind a region entry block's arguments and execute it.
    pub fn run_region(&mut self, region: &Region, args: Vec<Value>) -> Result<Vec<Value>> {
        let block = self.program.block(region.entry()?)?;
        if block.arguments.len() != args.len() {
            return Err(Error::backend(format!(
                "region entry block expects {} argument(s), got {}",
                block.arguments.len(),
                args.len()
            )));
        }
        for (index, (argument, value)) in block.arguments.iter().zip(args.iter()).enumerate() {
            let ty = self.program.value(*argument)?.ty.clone();
            dims::bind_shape(
                &mut self.bindings,
                &ty.shape,
                &value.shape,
                &format!("region argument {index}"),
            )?;
        }
        let entry = block.id.index();
        self.run_block(entry, args)
    }

    /// Evaluate a declared shape, keeping runtime-dynamic dims unchecked.
    pub fn eval_shape(&self, shape: &[Dim]) -> Result<Vec<Option<usize>>> {
        dims::eval_shape(shape, &self.bindings)
    }

    /// Evaluate a declared shape that may not contain dynamic dims.
    pub fn eval_shape_strict(&self, shape: &[Dim]) -> Result<Vec<usize>> {
        dims::eval_shape_strict(shape, &self.bindings)
    }

    /// Evaluate one declared result shape against the current bindings.
    pub fn result_shape(&self, spec: &OpSpec, index: usize) -> Result<Vec<Option<usize>>> {
        self.eval_shape(&spec.result_types[index].shape)
    }

    /// Evaluate one result type to a concrete shape (dynamic dims are errors).
    pub fn result_shape_strict(&self, spec: &OpSpec, index: usize) -> Result<Vec<usize>> {
        let ty = &spec.result_types[index];
        let shape = self.eval_shape_strict(&ty.shape)?;
        // A dynamic dim still needs a concrete extent for the allocation; the
        // kernel must derive it from operands instead of calling this.
        Ok(shape)
    }

    /// The declared result type of a one-result op.
    pub fn result_type<'a>(&self, spec: &'a OpSpec, index: usize) -> &'a ValueType {
        &spec.result_types[index]
    }

    fn validate_result(&self, step: &crate::plan::Step, index: usize, value: &Buffer) -> Result<()> {
        let ty = &step.spec.result_types[index];
        if value.dtype != ty.dtype {
            return Err(Error::backend(format!(
                "kernel for op '{}' produced result {index} with dtype {}, expected {} — kernels must never silently coerce dtypes",
                step.spec.name,
                value.dtype.name(),
                ty.dtype.name()
            )));
        }
        if let Some((dtype, shape)) = &step.fast {
            if value.dtype == *dtype && value.shape == *shape {
                return Ok(());
            }
        }
        let expected = self.eval_shape(&ty.shape)?;
        if value.shape.len() != expected.len() {
            return Err(Error::shape(format!(
                "kernel for op '{}' produced result {index} with shape {}, expected {} — rank mismatch",
                step.spec.name,
                shape_text(&value.shape),
                render_expected(&expected)
            )));
        }
        for (axis, (want, got)) in expected.iter().zip(value.shape.iter()).enumerate() {
            if let Some(want) = want {
                if want != got {
                    return Err(Error::shape(format!(
                        "kernel for op '{}' produced result {index} with shape {}, expected {} — dim {axis}: got {got}, expected {want}",
                        step.spec.name,
                        shape_text(&value.shape),
                        render_expected(&expected)
                    )));
                }
            }
        }
        Ok(())
    }
}

fn unresolved(value_id: usize) -> Error {
    Error::backend(format!(
        "cannot resolve value #{value_id} at run time — it is defined in no enclosing block of the executing block (the module must be invalid)"
    ))
}

fn render_expected(shape: &[Option<usize>]) -> String {
    let parts: Vec<String> = shape
        .iter()
        .map(|dim| match dim {
            Some(value) => value.to_string(),
            None => "None".to_string(),
        })
        .collect();
    format!("({})", parts.join(", "))
}

/// A convenience map from a callback's returned flat buffers to values.
pub fn callback_values(items: Vec<Buffer>) -> Vec<Value> {
    values(items)
}

/// Re-exported for kernels that need the binding map type.
pub type RunBindings = HashMap<String, i64>;
