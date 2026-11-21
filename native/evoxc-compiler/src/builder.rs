//! Mutable graph builder over an [`evoxc_core::graph::Program`].
//!
//! Every op is emitted through the same door the Python `ir.Builder` uses:
//! the op schema must exist, operands must match the declared arity, attributes
//! are validated and defaulted, and result types come from native inference
//! (`infer_op`) unless the caller passes explicit result types (only the
//! op-specific ops — `constant`, `if`, `runtime_call`, `block_call`,
//! `external_call` — and inlining an extension subgraph do that).
//!
//! The builder always appends to one open block; there is no insertion cursor.
//! `take_terminator` / `set_terminator` let a transform resume a program and
//! rewrite its return op (the `vmap` output rearrangement and `grad`'s
//! selective return).

use std::collections::HashMap;

use evoxc_core::attr::{ArrayPayload, Attrs};
use evoxc_core::dtype::DType;
use evoxc_core::envelope::validate_attrs;
use evoxc_core::error::{Error, Result};
use evoxc_core::graph::{
    Block, BlockId, Function, Location, Op, OpId, Program, Region, Value, ValueId,
};
use evoxc_core::infer::infer_op;
use evoxc_core::schema::{self, OpSchema};
use evoxc_core::value_type::ValueType;
use serde_json::Value as Json;

/// A mutable builder that appends to one block of one function of a program.
#[derive(Debug, Clone)]
pub struct Builder {
    program: Program,
    function: String,
    block: BlockId,
}

impl Builder {
    /// Start an empty module named `module_name`.
    pub fn new(module_name: impl Into<String>) -> Builder {
        Builder::with_metadata(module_name, Json::Null)
    }

    /// Start an empty module named `module_name` carrying `metadata`.
    pub fn with_metadata(module_name: impl Into<String>, metadata: Json) -> Builder {
        let mut program = Program::new(module_name);
        program.metadata = metadata;
        Builder {
            program,
            function: String::new(),
            block: BlockId(0),
        }
    }

    /// Resume an existing program at `function`'s single entry block.
    ///
    /// Allocators are reseeded above the maximum live id, so fresh values and
    /// ops never collide with decoded ids.
    pub fn resume(program: Program, function: &str, block: BlockId) -> Result<Builder> {
        let mut program = program;
        program.reseed_allocators();
        if program.function(function).is_err() {
            return Err(Error::transform(format!(
                "builder: cannot resume at '{function}' — the module has no such function"
            )));
        }
        Ok(Builder {
            program,
            function: function.to_string(),
            block,
        })
    }

    /// Open a function with `input_types` as its entry block arguments.
    pub fn open_function(&mut self, name: &str, input_types: &[ValueType]) -> Result<BlockId> {
        let block_id = self.program.new_block_id();
        let mut block = Block::new(block_id);
        for (index, ty) in input_types.iter().enumerate() {
            let value_id = self.program.new_value_id();
            self.program.add_value(Value::block_argument(
                value_id,
                ty.clone(),
                block_id,
                index as u32,
            ))?;
            block.arguments.push(value_id);
        }
        self.program.add_block(block)?;
        let mut function = Function::new(name, input_types.to_vec());
        function.body = Region::single(block_id);
        self.program.add_function(function)?;
        self.function = name.to_string();
        self.block = block_id;
        Ok(block_id)
    }

    /// Replace the open function's metadata (carried forward by a transform).
    pub fn set_function_metadata(&mut self, metadata: Json) -> Result<()> {
        self.program.function_mut(&self.function)?.metadata = metadata;
        Ok(())
    }
    /// The program under construction.
    pub fn program(&self) -> &Program {
        &self.program
    }

    /// The open block.
    pub fn block(&self) -> BlockId {
        self.block
    }

    /// The open function's name.
    pub fn function_name(&self) -> &str {
        &self.function
    }

    /// The open block's arguments, in order.
    pub fn block_arguments(&self) -> Result<Vec<ValueId>> {
        Ok(self.program.block(self.block)?.arguments.clone())
    }

    /// The type of a value of the program under construction.
    pub fn value_type(&self, id: ValueId) -> Result<ValueType> {
        Ok(self.program.value(id)?.ty.clone())
    }

    /// The types of `ids`, in order.
    pub fn value_types(&self, ids: &[ValueId]) -> Result<Vec<ValueType>> {
        ids.iter().map(|id| self.value_type(*id)).collect()
    }

    /// The open block's op ids, in order (the terminator included).
    pub fn block_op_ids(&self) -> Result<Vec<OpId>> {
        Ok(self.program.block(self.block)?.ops.clone())
    }

    /// The open block's terminator operands (`TransformError` when there is none).
    pub fn terminator_operands(&self) -> Result<Vec<ValueId>> {
        let terminator = self.program.terminator(self.block)?.ok_or_else(|| {
            Error::transform("builder: the block has no terminator (expected a 'return')")
        })?;
        Ok(terminator.operands.clone())
    }

    /// Emit `name(operands)` with native inference for its result types.
    pub fn emit_typed(
        &mut self,
        name: &str,
        operands: &[ValueId],
        attrs: &Attrs,
    ) -> Result<Vec<ValueId>> {
        self.emit(name, operands, attrs, None, None)
    }

    /// Emit a one-result op with native inference; returns its value id.
    pub fn emit_typed_one(
        &mut self,
        name: &str,
        operands: &[ValueId],
        attrs: &Attrs,
    ) -> Result<ValueId> {
        let mut results = self.emit_typed(name, operands, attrs)?;
        if results.len() != 1 {
            return Err(Error::verification(format!(
                "op '{name}': expected exactly one result, got {}",
                results.len()
            )));
        }
        Ok(results.remove(0))
    }

    /// Emit an op with full control over validation.
    ///
    /// `result_types = None` asks native inference; `Some(types)` is the
    /// explicit override the Python builder allows for op-specific ops and
    /// extension-subgraph inlining.
    pub fn emit(
        &mut self,
        name: &str,
        operands: &[ValueId],
        attrs: &Attrs,
        result_types: Option<&[ValueType]>,
        location: Option<Location>,
    ) -> Result<Vec<ValueId>> {
        let schema = schema::require(name)?;
        if !schema.check_arity(operands.len()) {
            return Err(Error::verification(format!(
                "op '{name}' expects arity {}, got {} operand(s)",
                schema.arity_text(),
                operands.len()
            )));
        }
        self.require_open_block()?;
        // Unknown attribute names are a hard error (parity with the Python
        // builder's `unknown attribute(s) ...; declared: ...` message).
        for (key, _) in attrs.iter() {
            if schema.attr(key).is_none() {
                return Err(Error::verification(format!(
                    "op '{name}': unknown attribute(s) [\"{key}\"]; declared: {}",
                    declared_attrs(schema)
                )));
            }
        }
        let filled = attrs.with_defaults(schema)?;
        validate_attrs(&filled, schema)?;
        let operand_types = self.value_types(operands)?;
        let types = match result_types {
            Some(types) => types.to_vec(),
            None => infer_op(name, &operand_types, &filled, None, Some(&self.program))?,
        };
        if !schema.check_result_count(types.len()) {
            return Err(Error::verification(format!(
                "op '{name}' declares {} result(s), got {}",
                result_text(schema),
                types.len()
            )));
        }
        let op_id = self.program.new_op_id();
        let mut op = Op::new(op_id, name);
        op.operands = operands.to_vec();
        op.attributes = filled;
        op.location = location;
        let mut result_ids = Vec::with_capacity(types.len());
        for (index, ty) in types.iter().enumerate() {
            let value_id = self.program.new_value_id();
            self.program.add_value(Value::op_result(
                value_id,
                ty.clone(),
                op_id,
                index as u32,
            ))?;
            op.results.push(value_id);
            result_ids.push(value_id);
        }
        self.program.push_op(self.block, op)?;
        Ok(result_ids)
    }

    /// Emit a one-result op with no attributes.
    pub fn unary(&mut self, name: &str, operand: ValueId) -> Result<ValueId> {
        self.emit_typed_one(name, &[operand], &Attrs::new())
    }

    /// Emit a three-operand, one-result op with no attributes (`select`).
    pub fn ternary(
        &mut self,
        name: &str,
        first: ValueId,
        second: ValueId,
        third: ValueId,
    ) -> Result<ValueId> {
        self.emit_typed_one(name, &[first, second, third], &Attrs::new())
    }
    /// Emit a two-operand, one-result op with no attributes.
    pub fn binary(&mut self, name: &str, left: ValueId, right: ValueId) -> Result<ValueId> {
        self.emit_typed_one(name, &[left, right], &Attrs::new())
    }

    /// Emit a `constant` op carrying `payload`.
    pub fn constant(&mut self, payload: ArrayPayload) -> Result<ValueId> {
        let ty = payload.value_type();
        let attrs = Attrs::from_pairs([("value", evoxc_core::attr::AttrValue::Array(payload))]);
        let results = self.emit("constant", &[], &attrs, Some(&[ty]), None)?;
        Ok(results[0])
    }

    /// Emit the `return` terminator; the block must not already have one.
    pub fn set_terminator(
        &mut self,
        operands: &[ValueId],
        location: Option<Location>,
    ) -> Result<()> {
        self.require_open_block()?;
        self.emit("return", operands, &Attrs::new(), Some(&[]), location)?;
        Ok(())
    }

    /// Remove and return the open block's terminator (`operands`, `location`).
    pub fn take_terminator(&mut self) -> Result<Option<(Vec<ValueId>, Option<Location>)>> {
        let terminator = match self.program.terminator(self.block)? {
            Some(op) => (op.id, op.operands.clone(), op.location.clone()),
            None => return Ok(None),
        };
        let (op_id, operands, location) = terminator;
        self.program.block_mut(self.block)?.ops.pop();
        self.program.ops.remove(op_id.0);
        Ok(Some((operands, location)))
    }

    /// Record a source location on an already-emitted op.
    pub fn set_op_location(&mut self, op: OpId, location: Option<Location>) -> Result<()> {
        self.program.op_mut(op)?.location = location;
        Ok(())
    }

    /// Append one op of a fresh `results` list with an explicit location.
    pub fn push_op_with(
        &mut self,
        name: &str,
        operands: &[ValueId],
        attrs: &Attrs,
        result_types: Option<&[ValueType]>,
        location: Option<Location>,
    ) -> Result<Vec<ValueId>> {
        self.emit(name, operands, attrs, result_types, location)
    }

    /// Inline `entry` of `sub` into the open block, binding its block arguments
    /// to `arg_values`; returns the sub-value → new-value remapping.
    ///
    /// Region-bearing sub-ops are rejected (an extension subgraph is a straight
    /// line of ordinary ops terminating with `return`).
    pub fn inline_function(
        &mut self,
        sub: &Program,
        entry: &str,
        arg_values: &[ValueId],
    ) -> Result<HashMap<ValueId, ValueId>> {
        let function = sub.function(entry).map_err(|_| {
            Error::transform(format!(
                "extension rule: the returned module has no function '{entry}'"
            ))
        })?;
        let block = sub.entry_block(function)?;
        if block.arguments.len() != arg_values.len() {
            return Err(Error::transform(format!(
                "extension rule: entry function '{entry}' takes {} argument(s) but the engine supplies {}",
                block.arguments.len(),
                arg_values.len()
            )));
        }
        let mut mapping: HashMap<ValueId, ValueId> = HashMap::new();
        for (argument, value) in block.arguments.iter().zip(arg_values) {
            mapping.insert(*argument, *value);
        }
        for op_id in &block.ops {
            let op = sub.op(*op_id)?;
            if op.is_terminator()? {
                continue;
            }
            if !op.regions.is_empty() {
                return Err(Error::transform(format!(
                    "extension rule: op '{}' has {} nested region(s); an extension subgraph must be a straight-line block",
                    op.name,
                    op.regions.len()
                )));
            }
            let mut operands = Vec::with_capacity(op.operands.len());
            for operand in &op.operands {
                let mapped = mapping.get(operand).copied().ok_or_else(|| {
                    Error::transform(format!(
                        "extension rule: op '{}' reads value %{operand}, which the inlined block does not define",
                        op.name
                    ))
                })?;
                operands.push(mapped);
            }
            let result_types = op
                .results
                .iter()
                .map(|id| Ok(sub.value(*id)?.ty.clone()))
                .collect::<Result<Vec<_>>>()?;
            let new_results = self.emit(
                &op.name,
                &operands,
                &op.attributes,
                Some(&result_types),
                op.location.clone(),
            )?;
            for (old, new) in op.results.iter().zip(new_results) {
                mapping.insert(*old, new);
            }
        }
        Ok(mapping)
    }

    /// Finish: recompute use lists and refresh every function's `output_types`
    /// from its terminator, then return the program.
    pub fn finish(mut self) -> Program {
        self.program.rebuild_uses();
        for index in 0..self.program.functions.len() {
            let types = {
                let function = &self.program.functions[index];
                self.program.computed_output_types(function).ok()
            };
            if let Some(types) = types {
                self.program.functions[index].output_types = types;
            }
        }
        self.program
    }

    fn require_open_block(&self) -> Result<()> {
        if self.program.terminator(self.block)?.is_some() {
            return Err(Error::verification(
                "builder: the block already has a terminator; nothing can be appended after it",
            ));
        }
        Ok(())
    }
}

/// `{"a", "b"}` — the declared attribute names of a schema, for messages.
pub fn declared_attrs(schema: &OpSchema) -> String {    let mut names: Vec<String> = schema.attrs.iter().map(|attr| format!("\"{}\"", attr.name)).collect();
    names.sort();
    format!("[{}]", names.join(", "))
}

/// `[n, m]` — the declared result-count range of a schema, for messages.
pub fn result_text(schema: &OpSchema) -> String {
    match schema.result_max {
        Some(hi) if hi == schema.result_min => format!("{}", schema.result_min),
        Some(hi) => format!("{}..{hi}", schema.result_min),
        None => format!("{}..", schema.result_min),
    }
}

/// Convenience: a `constant` op payload from an `f64` scalar in `dtype`.
pub fn scalar_payload(dtype: DType, value: f64) -> Result<ArrayPayload> {
    crate::emit::scalar_payload(dtype, value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use evoxc_core::attr::AttrValue;
    use evoxc_core::dim::Dim;

    fn f32_matrix(rows: i64, cols: i64) -> ValueType {
        ValueType::of(
            DType::parse("float32").unwrap(),
            [Dim::Int(rows), Dim::Int(cols)],
        )
    }

    #[test]
    fn emits_and_infers_elementwise_and_reduction() {
        let a = f32_matrix(2, 3);
        let mut builder = Builder::new("main");
        let block = builder.open_function("main", &[a.clone(), a.clone()]).unwrap();
        let args = builder.block_arguments().unwrap();
        let sum = builder.binary("add", args[0], args[1]).unwrap();
        assert_eq!(builder.value_type(sum).unwrap(), a);
        let axes = Attrs::from_pairs([
            ("axes", AttrValue::Ints(vec![1])),
            ("reduce_op", AttrValue::Str("sum".to_string())),
        ]);
        let reduced = builder
            .emit_typed_one("reduce_sum", &[sum], &axes)
            .unwrap();
        assert_eq!(
            builder.value_type(reduced).unwrap(),
            ValueType::of(DType::parse("float32").unwrap(), [Dim::Int(2)])
        );
        builder.set_terminator(&[reduced], None).unwrap();
        let program = builder.finish();
        assert_eq!(program.functions.len(), 1);
        assert_eq!(program.entry_block(&program.functions[0]).unwrap().id, block);
        assert_eq!(program.functions[0].output_types.len(), 1);
        assert_eq!(program.functions[0].output_types[0].rank(), 1);
    }

    #[test]
    fn rejects_unknown_op_arity_and_attributes() {
        let a = f32_matrix(2, 3);
        let mut builder = Builder::new("main");
        builder.open_function("main", &[a]).unwrap();
        let args = builder.block_arguments().unwrap();
        let err = builder.emit("no_such_op", &args, &Attrs::new(), None, None).unwrap_err();
        assert_eq!(err.kind_str(), "VerificationError");
        let err = builder.emit("add", &args, &Attrs::new(), None, None).unwrap_err();
        assert!(err.message.contains("expects arity"));
        let bad = Attrs::from_pairs([("axis", AttrValue::Int(0))]);
        let err = builder.emit("transpose", &args, &bad, None, None).unwrap_err();
        assert!(err.message.contains("unknown attribute"), "{}", err.message);
        let err = builder.emit("cast", &args, &Attrs::new(), None, None).unwrap_err();
        assert_eq!(err.kind_str(), "VerificationError");
        assert!(err.message.contains("missing required attribute"), "{}", err.message);
    }

    #[test]
    fn defaults_are_filled_and_terminator_is_unique() {
        let a = f32_matrix(2, 3);
        let mut builder = Builder::new("main");
        builder.open_function("main", &[a]).unwrap();
        let args = builder.block_arguments().unwrap();
        // `reduce_op` is a required IR attribute (the frontend always passes
        // it); `keepdims` is optional and comes from the schema default.
        let attrs = Attrs::from_pairs([
            ("axes", AttrValue::Ints(vec![0])),
            ("reduce_op", AttrValue::Str("sum".to_string())),
        ]);
        let out = builder.emit_typed_one("reduce_sum", &args, &attrs).unwrap();
        let op_id = *builder.block_op_ids().unwrap().last().unwrap();
        let op = builder.program().op(op_id).unwrap();
        assert!(!op.attributes.bool("keepdims").unwrap());
        assert_eq!(op.attributes.str("reduce_op").unwrap(), "sum");
        assert_eq!(out.0, op.results[0].0);
        builder.set_terminator(&[out], None).unwrap();
        // a well-formed op is still refused once the block has a terminator
        let err = builder
            .emit("add", &[args[0], args[0]], &Attrs::new(), None, None)
            .unwrap_err();
        assert!(err.message.contains("already has a terminator"));
        assert!(builder.set_terminator(&[out], None).is_err());
    }

    #[test]
    fn take_and_replace_terminator() {
        let a = f32_matrix(2, 3);
        let mut builder = Builder::new("main");
        builder.open_function("main", &[a]).unwrap();
        let args = builder.block_arguments().unwrap();
        let out = builder.unary("negate", args[0]).unwrap();
        builder.set_terminator(&[out], None).unwrap();
        let mut program = builder.finish();
        let function = program.functions[0].name.clone();
        let block = program.entry_block(&program.functions[0]).unwrap().id;
        let mut builder = Builder::resume(program, &function, block).unwrap();
        let (operands, _) = builder.take_terminator().unwrap().unwrap();
        assert_eq!(operands, vec![out]);
        assert!(builder.program().terminator(block).unwrap().is_none());
        let doubled = builder.binary("add", operands[0], operands[0]).unwrap();
        builder.set_terminator(&[doubled], None).unwrap();
        program = builder.finish();
        assert_eq!(program.computed_output_types(&program.functions[0]).unwrap()[0].rank(), 2);
    }

    #[test]
    fn inlines_a_subgraph_with_ssa_remapping() {
        // sub: (x: f32[2,3]) -> return add(x, constant(1))
        let a = f32_matrix(2, 3);
        let mut sub = Builder::new("rule");
        sub.open_function("rule", std::slice::from_ref(&a)).unwrap();
        let sub_args = sub.block_arguments().unwrap();
        let one = sub
            .constant(scalar_payload(DType::parse("float32").unwrap(), 1.0).unwrap())
            .unwrap();
        let sum = sub.binary("add", sub_args[0], one).unwrap();
        sub.set_terminator(&[sum], None).unwrap();
        let sub_program = sub.finish();

        let mut builder = Builder::new("main");
        // A leading dummy input keeps the two id spaces apart, so the
        // remapping below is observable (both programs number from 0).
        builder
            .open_function("main", &[f32_matrix(8, 8), a.clone()])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let mapping = builder
            .inline_function(&sub_program, "rule", &[args[1]])
            .unwrap();
        // the entry block argument, the inlined `constant` and the inlined `add`
        assert_eq!(mapping.len(), 3);
        // the sub graph's entry argument is bound to the supplied main value
        assert_eq!(mapping[&sub_args[0]], args[1]);
        assert_ne!(mapping[&sub_args[0]], sub_args[0]);
        builder.set_terminator(&[mapping[&sum]], None).unwrap();
        let program = builder.finish();
        // add + constant were appended before the return.
        assert_eq!(program.ops.len(), 3);
        // the inlined `add` was rewritten onto the main program's values
        let add = program
            .ops
            .iter()
            .map(|(_, op)| op)
            .find(|op| op.name == "add")
            .expect("the sub graph's add was inlined");
        assert_eq!(add.operands[0], args[1]);
        assert_ne!(add.operands[1], add.operands[0]);
        assert_ne!(add.results[0], sum);
        assert_eq!(program.computed_output_types(&program.functions[0]).unwrap()[0], a);
    }

    #[test]
    fn inline_rejects_wrong_arity_and_unknown_entry() {
        let a = f32_matrix(2, 3);
        let mut sub = Builder::new("rule");
        sub.open_function("rule", std::slice::from_ref(&a)).unwrap();
        let args = sub.block_arguments().unwrap();
        sub.set_terminator(&args, None).unwrap();
        let sub_program = sub.finish();

        let mut builder = Builder::new("main");
        builder.open_function("main", &[a]).unwrap();
        let err = builder.inline_function(&sub_program, "rule", &[]).unwrap_err();
        assert!(err.message.contains("takes 1 argument(s)"), "{}", err.message);
        let args = builder.block_arguments().unwrap();
        let err = builder
            .inline_function(&sub_program, "missing", &args)
            .unwrap_err();
        assert!(err.message.contains("no function 'missing'"));
    }
}
