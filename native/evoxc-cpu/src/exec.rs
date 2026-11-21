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
    /// Reusable step operand list: a `Vec<Value>` the block loop hands to each
    /// kernel and takes back, so a step never allocates its operand vector.
    /// Scratch state of ONE run, never part of the (immutable) plan; a
    /// control-flow kernel that re-enters [`RunCtx::run_block`] takes the
    /// (then empty) scratch and gives it back on the way out.
    operands: Vec<Value>,
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
            operands: Vec::new(),
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
        // The store still holds a second reference to every returned value, and
        // `unwrap_value` can only MOVE a buffer out of an `Rc` when it is the
        // last owner.  Releasing the run's store here therefore turns a whole
        // output array copy per call into a pointer move (values that are
        // genuinely shared — the same value returned twice — still copy).
        for slot in self.store.iter_mut() {
            *slot = None;
        }
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
        // One scratch operand list for the whole block (and, through the take /
        // put-back below, for the whole run): the step loop clears it per step
        // instead of allocating a fresh `Vec<Value>` per op.
        let mut operands = std::mem::take(&mut self.operands);
        for step in &block.steps {
            operands.clear();
            let mut missing = None;
            for value_id in &step.operands {
                match self.store.get(*value_id).and_then(|slot| slot.as_ref()) {
                    Some(value) => operands.push(Rc::clone(value)),
                    None => {
                        missing = Some(*value_id);
                        break;
                    }
                }
            }
            if let Some(value_id) = missing {
                self.operands = operands;
                return Err(unresolved(value_id));
            }
            // A `constant` carries its payload decoded ONCE at plan build: the
            // step is then one clone of that buffer rather than a per-run
            // decode of the same bytes (`Buffer::from_payload`), with identical
            // values and the same result validation below.
            let results = if let Some(constant) = &step.constant {
                vec![Rc::new(constant.clone())]
            } else {
                match (step.kernel)(self, &step.spec, &operands) {
                    Ok(results) => results,
                    Err(err) => {
                        self.operands = operands;
                        return Err(err);
                    }
                }
            };
            if results.len() != step.results.len() {
                self.operands = operands;
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
        self.operands = operands;
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
        if let Some((dtype, shape)) = step.fast.get(index).and_then(|entry| entry.as_ref()) {
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

#[cfg(test)]
mod tests {
    //! Contracts of the plan/exec seam that the per-op fast paths added:
    //! the per-result static-shape marker, the plan-time `constant` decode, the
    //! reusable operand scratch, and the store release that lets a run MOVE its
    //! outputs instead of copying them.  Every fast path keeps the general
    //! check as its fallback and the exact error wording.
    use super::*;
    use crate::buffer::Storage;
    use crate::plan::{self, Step};
    use evoxc_core::{
        ArrayPayload, AttrValue, Attrs, Block, DType, ErrorKind, Function, Op as IrOp,
        Value as IrValue,
    };

    /// The smallest module `plan::compile` accepts: `main` with a body that is
    /// just a `return` (no inputs, no outputs).  Enough to build a `RunCtx`.
    fn return_only_program() -> Program {
        let mut program = Program::new("fixture");
        let block_id = program.new_block_id();
        let return_id = program.new_op_id();
        program.add_op(IrOp::new(return_id, "return")).unwrap();
        let mut block = Block::new(block_id);
        block.ops = vec![return_id];
        program.add_block(block).unwrap();
        let mut function = Function::new("main", Vec::new());
        function.output_types = Vec::new();
        function.body = Region::single(block_id);
        program.add_function(function).unwrap();
        program
    }

    fn context(program: &Program) -> RunCtx<'_> {
        let plan = Arc::new(plan::compile(program).expect("the fixture program compiles"));
        RunCtx::new(program, plan, RunOptions::default(), None)
    }

    fn spec(name: &str, result_types: Vec<ValueType>) -> OpSpec {
        OpSpec {
            name: name.to_string(),
            attrs: Attrs::new(),
            operand_types: Vec::new(),
            result_types,
            regions: Vec::new(),
            location: None,
        }
    }

    fn step(spec: OpSpec, fast: Vec<Option<(DType, Vec<usize>)>>) -> Step {
        let kernel = crate::kernels::lookup(&spec.name).expect("the fixture op has a kernel");
        let results = (0..spec.result_types.len()).collect();
        Step {
            spec,
            kernel,
            operands: Vec::new(),
            results,
            free: Vec::new(),
            fast,
            constant: None,
        }
    }

    fn f32_buffer(shape: &[usize], values: &[f32]) -> Buffer {
        Buffer::new(DType::Float32, shape.to_vec(), Storage::F32(values.to_vec()))
            .unwrap_or_else(|err| panic!("test fixture is malformed: {}", err.message))
    }

    #[test]
    fn static_result_marker_validates_every_result_of_a_multi_result_op() {
        let program = return_only_program();
        let ctx = context(&program);
        let f32_2 = ValueType::new(DType::Float32, vec![Dim::int(2)]);
        let s = step(
            spec("add", vec![f32_2.clone(), f32_2]),
            vec![
                Some((DType::Float32, vec![2])),
                Some((DType::Float32, vec![2])),
            ],
        );
        let good = f32_buffer(&[2], &[1.0, 2.0]);
        ctx.validate_result(&s, 1, &good).expect("a static marker accepts its own shape");
        let wrong_dtype = Buffer::new(DType::Float64, vec![2], Storage::F64(vec![1.0, 2.0]))
            .expect("fixture");
        let err = ctx.validate_result(&s, 0, &wrong_dtype).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Backend);
        assert!(
            err.message
                .contains("produced result 0 with dtype float64, expected float32"),
            "{}",
            err.message
        );
        let wrong_extent = f32_buffer(&[3], &[0.0; 3]);
        let err = ctx.validate_result(&s, 0, &wrong_extent).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Shape);
        assert!(err.message.contains("dim 0: got 3, expected 2"), "{}", err.message);
        let wrong_rank = f32_buffer(&[2, 1], &[0.0; 2]);
        let err = ctx.validate_result(&s, 0, &wrong_rank).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Shape);
        assert!(err.message.contains("rank mismatch"), "{}", err.message);
    }

    #[test]
    fn symbolic_and_dynamic_result_dims_keep_the_general_check() {
        let program = return_only_program();
        let mut ctx = context(&program);
        let symbolic = ValueType::new(DType::Float32, vec![Dim::named("N")]);
        let dynamic = ValueType::new(DType::Float32, vec![Dim::dynamic(), Dim::int(3)]);
        let s = step(spec("add", vec![symbolic, dynamic]), vec![None, None]);
        let value = f32_buffer(&[2], &[0.0; 2]);
        // An unbound symbolic dim fails loudly instead of accepting anything.
        assert_eq!(
            ctx.validate_result(&s, 0, &value).unwrap_err().kind,
            ErrorKind::Shape
        );
        dims::bind_dim(&mut ctx.bindings, &Dim::named("N"), 2, "test").expect("bind N");
        ctx.validate_result(&s, 0, &value).expect("the bound symbolic dim matches");
        let wider = f32_buffer(&[4], &[0.0; 4]);
        let err = ctx.validate_result(&s, 0, &wider).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Shape);
        assert!(err.message.contains("dim 0: got 4, expected 2"), "{}", err.message);
        // A runtime-dynamic dim accepts whatever extent the kernel derived ...
        let wide = f32_buffer(&[7, 3], &[0.0; 21]);
        ctx.validate_result(&s, 1, &wide).expect("a runtime-dynamic dim is unchecked");
        // ... but the rank and every static sibling dim are still checked.
        let flat = f32_buffer(&[7], &[0.0; 7]);
        assert_eq!(
            ctx.validate_result(&s, 1, &flat).unwrap_err().kind,
            ErrorKind::Shape
        );
        let wrong_sibling = f32_buffer(&[7, 4], &[0.0; 28]);
        assert_eq!(
            ctx.validate_result(&s, 1, &wrong_sibling).unwrap_err().kind,
            ErrorKind::Shape
        );
    }

    /// Build `main()` returning one `constant` op carrying `payload`.
    fn constant_program(payload: ArrayPayload) -> Program {
        let mut program = Program::new("constant");
        let block_id = program.new_block_id();
        let out_id = program.new_value_id();
        let constant_id = program.new_op_id();
        let return_id = program.new_op_id();
        let out_ty = ValueType::new(
            payload.dtype,
            payload.shape.iter().map(|dim| Dim::int(*dim)).collect(),
        );
        program
            .add_value(IrValue::op_result(out_id, out_ty.clone(), constant_id, 0))
            .expect("fixture value");
        let mut constant = IrOp::new(constant_id, "constant");
        constant.attributes = Attrs::from_pairs([("value", AttrValue::Array(payload))]);
        constant.results = vec![out_id];
        program.add_op(constant).expect("fixture op");
        let mut ret = IrOp::new(return_id, "return");
        ret.operands = vec![out_id];
        program.add_op(ret).expect("fixture op");
        let mut block = Block::new(block_id);
        block.ops = vec![constant_id, return_id];
        program.add_block(block).expect("fixture block");
        let mut function = Function::new("main", Vec::new());
        function.output_types = vec![out_ty];
        function.body = Region::single(block_id);
        program.add_function(function).expect("fixture function");
        program
    }

    fn f32_payload(values: &[f32], shape: Vec<i64>) -> ArrayPayload {
        let mut data = Vec::with_capacity(values.len() * 4);
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        ArrayPayload {
            dtype: DType::Float32,
            shape,
            data,
        }
    }

    #[test]
    fn a_constant_is_decoded_once_at_plan_time_and_runs_through_the_plan() {
        let payload = f32_payload(&[1.5, -2.5, 0.0, 4.0], vec![2, 2]);
        let program = constant_program(payload.clone());
        let plan = plan::compile(&program).expect("the constant program compiles");
        let block = plan.blocks[0].as_ref().expect("the entry block is compiled");
        let step = &block.steps[0];
        let decoded = step.constant.as_ref().expect("the payload is decoded at compile time");
        // Identical to the decode the `constant` kernel used to do per run.
        let expected = Buffer::from_payload(&payload).expect("the payload decodes");
        assert_eq!(decoded.dtype, expected.dtype);
        assert_eq!(decoded.shape, expected.shape);
        assert_eq!(decoded.storage, expected.storage);
        // The per-result static marker covers `constant` too.
        assert_eq!(step.fast, vec![Some((DType::Float32, vec![2, 2]))]);
        assert!(plan.text().contains("static"), "{}", plan.text());
        // And the run returns those values (twice: the same plan is reusable).
        let mut ctx = context(&program);
        for _ in 0..2 {
            let out = ctx.run(Vec::new()).expect("the constant run succeeds");
            assert_eq!(out.len(), 1);
            assert_eq!(out[0].shape, vec![2, 2]);
            assert_eq!(out[0].storage, Storage::F32(vec![1.5, -2.5, 0.0, 4.0]));
        }
    }

    #[test]
    fn a_malformed_constant_payload_fails_at_plan_time() {
        // Two elements declared, four bytes carried.
        let payload = f32_payload(&[1.0], vec![2]);
        let program = constant_program(payload);
        let err = plan::compile(&program).err().expect("compile rejects the payload");
        assert!(
            err.message
                .contains("constant payload declares 8 byte(s) but carries 4"),
            "{}",
            err.message
        );
    }

    /// `main(x) = add(x, x)` over one `f32[2]` input, plus a `return x, x` form
    /// exercising the shared-output case.
    fn add_program(twice: bool) -> Program {
        let mut program = Program::new("add");
        let block_id = program.new_block_id();
        let x_id = program.new_value_id();
        let sum_id = program.new_value_id();
        let add_id = program.new_op_id();
        let return_id = program.new_op_id();
        let ty = ValueType::new(DType::Float32, vec![Dim::int(2)]);
        program
            .add_value(IrValue::block_argument(x_id, ty.clone(), block_id, 0))
            .expect("fixture value");
        program
            .add_value(IrValue::op_result(sum_id, ty.clone(), add_id, 0))
            .expect("fixture value");
        let mut add = IrOp::new(add_id, "add");
        add.operands = vec![x_id, x_id];
        add.results = vec![sum_id];
        program.add_op(add).expect("fixture op");
        let mut ret = IrOp::new(return_id, "return");
        ret.operands = if twice { vec![sum_id, sum_id] } else { vec![sum_id] };
        program.add_op(ret).expect("fixture op");
        let mut block = Block::new(block_id);
        block.arguments = vec![x_id];
        block.ops = vec![add_id, return_id];
        program.add_block(block).expect("fixture block");
        let mut function = Function::new("main", vec![ty.clone()]);
        function.output_types = if twice { vec![ty.clone(), ty] } else { vec![ty] };
        function.body = Region::single(block_id);
        program.add_function(function).expect("fixture function");
        program
    }

    #[test]
    fn one_run_state_is_reusable_across_runs_and_shares_no_values() {
        let program = add_program(false);
        let plan = Arc::new(plan::compile(&program).expect("the add program compiles"));
        let mut ctx = RunCtx::new(&program, plan, RunOptions::default(), None);
        for input in [[1.0f32, 2.0], [5.0, -1.0]] {
            let out = ctx
                .run(vec![f32_buffer(&[2], &input)])
                .expect("the add run succeeds");
            assert_eq!(out.len(), 1);
            assert_eq!(out[0].storage, Storage::F32(vec![input[0] * 2.0, input[1] * 2.0]));
            // The run releases its store, so a second run re-binds every value
            // and no buffer of the previous run survives into this one.
            assert!(ctx.store.iter().all(|slot| slot.is_none()));
        }
    }

    #[test]
    fn a_value_returned_twice_is_still_correct() {
        // The output move turns into a copy exactly when a value is genuinely
        // shared (here: the same op result returned twice).
        let program = add_program(true);
        let out = crate::evaluate(
            &program,
            vec![f32_buffer(&[2], &[1.0, 2.0])],
            RunOptions::default(),
            None,
        )
        .expect("the shared-output run succeeds");
        assert_eq!(out.len(), 2);
        for buffer in &out {
            assert_eq!(buffer.storage, Storage::F32(vec![2.0, 4.0]));
        }
    }
}
