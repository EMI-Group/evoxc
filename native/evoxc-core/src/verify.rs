//! Module verification: the Rust port of `evoxc/ir/verify.py`.
//!
//! [`verify`] is the IR contract checker.  It validates structural invariants,
//! result-type agreement, attribute schemas, SSA well-formedness and the v1
//! restrictions, raising `VerificationError` on the FIRST violation (no silent
//! recovery, no partial reporting), with the Python message text and the op's
//! source location appended as `"<message> [at {location}]"` when one is known
//! (Python's `_fail`).
//!
//! Checked invariants (Python's order):
//!
//! *Module* — at least one function; unique function names.
//!
//! *Function* — the body region holds exactly one block (v1); entry-block
//! argument count/types match `input_types`; the last op is the `return`
//! terminator (nothing after a terminator); `output_types` agree with the
//! return operands' types.
//!
//! *Region/block* (function bodies AND nested op regions) — a region has at
//! least one block, each block id exists in the module's block table, each
//! nested-region entry block has one argument per operand with the operand's
//! type, block arguments carry a consistent owner/index, and every op id
//! exists in the op table and is listed in exactly one block.
//!
//! *Op* — the schema exists; operand count within the declared arity; region
//! count equals the schema's; result count within the declared range (when
//! known); attributes match the schema; results are values owned by this op at
//! the matching index; operands are SSA-dominant; result types agree with
//! inference (`has_shape_fn`) or the op-specific rule.
//!
//! *Uses* — `Value.uses` agrees with the ops' operand lists in BOTH directions.
//!
//! Deliberate differences from the Python checker, none behavioral: the Python
//! module's `version` and `parent` pointers are not modelled here (the envelope
//! codec owns the wire version, and ownership is expressed by the [`Program`]
//! tables, where an op's single-block membership is checked explicitly); the
//! Python `isinstance` guards have no Rust counterpart because the types are
//! static; and the Python per-call result-type memo is omitted (it is a pure
//! speed memo — verification reaches the same verdict without it).  Duplicate
//! ids cannot occur either: [`crate::graph::Table::insert`] rejects them at
//! construction time.
//!
//! Result-type agreement delegates to [`crate::infer::infer_op`], so an op whose
//! native shape function is not ported yet fails explicitly ("result-type
//! inference failed: ... no native shape function is registered") instead of
//! passing unchecked.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value as Json;

use crate::attr::AttrValue;
use crate::dim::Dim;
use crate::error::{Error, Result};
use crate::graph::{
    Block, BlockId, Function, Location, Op, OpId, Program, Region, ValueId, ValueOwner,
};
use crate::infer;
use crate::schema::{self, attr_type, AttrSchema, OpSchema};
use crate::value_type::ValueType;
use crate::wire;

/// Validate `program` against all EvoXIR invariants.
///
/// Returns the FIRST violation as a `VerificationError` whose message mirrors
/// `evoxc/ir/verify.py` and names the offending op/value (plus the source
/// location when the op has one).
pub fn verify(program: &Program) -> Result<()> {
    verify_module(program)?;
    for function in &program.functions {
        verify_function(program, function)?;
    }
    verify_op_block_membership(program)?;
    verify_uses(program)
}

// ---------------------------------------------------------------------------
// Message helpers
// ---------------------------------------------------------------------------

/// Python `_fail`: raise, annotating with the source location when known.
fn fail(message: impl Into<String>, location: Option<&Location>) -> Error {
    let message = message.into();
    match location {
        Some(loc) if loc.is_known() => Error::verification(format!("{message} [at {loc}]")),
        _ => Error::verification(message),
    }
}

/// The Python message prefix `op '{name}' (id {id})`.
///
/// Built only on failure branches, so the success path stays allocation-free.
fn where_op(op: &Op) -> String {
    format!("op '{}' (id {})", op.name, op.id)
}

/// Python `_value_desc`: a short description of an attribute value's type.
fn describe_attr(value: &AttrValue) -> String {
    match value {
        AttrValue::Ints(items) => format!("ints with {} entries", items.len()),
        AttrValue::Floats(items) => format!("floats with {} entries", items.len()),
        AttrValue::Strs(items) => format!("strs with {} entries", items.len()),
        AttrValue::NestedInts(items) => format!("nested_ints with {} entries", items.len()),
        AttrValue::Shape(shape) => format!("shape with {} entries", shape.len()),
        AttrValue::Tuple(items) => format!("tuple with {} entries", items.len()),
        other => other.type_name().to_string(),
    }
}

/// Python tuple-of-`ValueType` spelling: `()`, `(tensor<f32>,)`, `(a, b)`.
fn format_types(types: &[ValueType]) -> String {
    join_tuple(types.iter().map(|ty| ty.to_string()).collect())
}

/// Python repr of one dimension inside a shape tuple.
fn format_dim(dim: &Dim) -> String {
    match dim {
        Dim::Int(extent) => extent.to_string(),
        Dim::Named(named) => match named.size {
            None => format!("Dim({:?})", named.name),
            Some(size) => format!("Dim({:?}, size={size})", named.name),
        },
        Dim::Dynamic => "None".to_string(),
        Dim::Expr(_) => dim.to_string(),
    }
}

/// Python `{shape!r}`: `()`, `(2,)`, `(2, 3)`.
fn format_shape(shape: &[Dim]) -> String {
    join_tuple(shape.iter().map(format_dim).collect())
}

/// Python tuple repr of already-rendered items.
fn join_tuple(items: Vec<String>) -> String {
    let mut out = format!("({})", items.join(", "));
    if items.len() == 1 {
        out.insert(out.len() - 1, ',');
    }
    out
}

// ---------------------------------------------------------------------------
// Module level
// ---------------------------------------------------------------------------

fn verify_module(program: &Program) -> Result<()> {
    if program.functions.is_empty() {
        return Err(fail(
            format!("module '{}': must contain at least one function", program.name),
            None,
        ));
    }
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for function in &program.functions {
        if !seen.insert(function.name.as_str()) {
            let message = format!(
                "module '{}': duplicate function name '{}'",
                program.name, function.name
            );
            return Err(fail(message, None));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Function level
// ---------------------------------------------------------------------------

fn verify_function(program: &Program, function: &Function) -> Result<()> {
    let fname = &function.name;
    let region = &function.body;
    if region.blocks.is_empty() {
        return Err(fail(format!("function '{fname}': region has no blocks"), None));
    }
    if region.blocks.len() != 1 {
        let message = format!(
            "function '{fname}': region has {} blocks; v1 requires exactly one",
            region.blocks.len()
        );
        return Err(fail(message, None));
    }
    let block_id = region.blocks[0];
    let block = program.block(block_id).map_err(|_| {
        let message = format!(
            "function '{fname}': region block {block_id} is not defined in module '{}'",
            program.name
        );
        fail(message, None)
    })?;

    if block.arguments.len() != function.input_types.len() {
        let message = format!(
            "function '{fname}': entry block has {} arguments, expected {}",
            block.arguments.len(),
            function.input_types.len()
        );
        return Err(fail(message, None));
    }
    for (index, (arg_id, expected)) in block.arguments.iter().zip(&function.input_types).enumerate()
    {
        let arg = program.value(*arg_id).map_err(|_| {
            let message = format!(
                "function '{fname}': entry argument {index} is not defined in module '{}'",
                program.name
            );
            fail(message, None)
        })?;
        if &arg.ty != expected {
            let message = format!(
                "function '{fname}': entry argument {index} type {} does not match input type \
                 {expected}",
                arg.ty
            );
            return Err(fail(message, None));
        }
    }

    verify_block_terminator(program, block, &format!("function '{fname}'"))?;

    // output_types consistency: recompute from the return terminator and
    // compare with the recorded signature.
    let Some(&terminator_id) = block.ops.last() else {
        let message = format!("function '{fname}': block has no ops (missing 'return' terminator)");
        return Err(fail(message, None));
    };
    let terminator = program.op(terminator_id)?;
    let mut expected_outputs = Vec::with_capacity(terminator.operands.len());
    for (index, operand_id) in terminator.operands.iter().enumerate() {
        let operand = program.value(*operand_id).map_err(|_| {
            let message = format!(
                "function '{fname}': return operand {index} is not defined in module '{}'",
                program.name
            );
            fail(message, None)
        })?;
        expected_outputs.push(operand.ty.clone());
    }
    if expected_outputs != function.output_types {
        let message = format!(
            "function '{fname}': output_types {} disagree with the return operands' types {}",
            format_types(&function.output_types),
            format_types(&expected_outputs)
        );
        return Err(fail(message, None));
    }

    verify_block_ops(program, block, &BTreeSet::new())
}

// ---------------------------------------------------------------------------
// Region/block level (depth-first walk)
// ---------------------------------------------------------------------------

/// Every block must end with the `return` terminator, and nothing may follow a
/// terminator.
fn verify_block_terminator(program: &Program, block: &Block, context: &str) -> Result<()> {
    let Some(&last_id) = block.ops.last() else {
        let message = format!("{context}: block has no ops (missing 'return' terminator)");
        return Err(fail(message, None));
    };
    let mut terminator_indexes: Vec<usize> = Vec::new();
    for (index, op_id) in block.ops.iter().enumerate() {
        let op = program.op(*op_id)?;
        // Unknown names are reported by the op-level check; skip them here.
        if schema::lookup(&op.name).is_some_and(|schema| schema.is_terminator) {
            terminator_indexes.push(index);
        }
    }
    if terminator_indexes.is_empty() {
        let last = program.op(last_id)?;
        if schema::lookup(&last.name).is_none() {
            let message = format!(
                "{context}: no terminator (last op '{}' is not a registered op)",
                last.name
            );
            return Err(fail(message, None));
        }
        return Err(fail(format!("{context}: no 'return' terminator"), None));
    }
    for index in terminator_indexes {
        if index != block.ops.len() - 1 {
            let op = program.op(block.ops[index])?;
            let message = format!("{context}: terminator '{}' is not the last op", op.name);
            return Err(fail(message, None));
        }
    }
    let last = program.op(last_id)?;
    if last.name != "return" {
        let message = format!(
            "{context}: terminator must be the 'return' op, got '{}'",
            last.name
        );
        return Err(fail(message, None));
    }
    Ok(())
}

/// Check every op of `block` in order, then recurse into nested regions.
///
/// `scope` holds the ids of all values defined before this block (enclosing
/// block arguments and results of earlier ops) — the SSA in-scope set.
fn verify_block_ops(program: &Program, block: &Block, scope: &BTreeSet<ValueId>) -> Result<()> {
    let mut local = scope.clone();
    for (index, arg_id) in block.arguments.iter().enumerate() {
        let arg = program.value(*arg_id).map_err(|_| {
            let message = format!(
                "block argument %{arg_id} is not defined in module '{}'",
                program.name
            );
            fail(message, None)
        })?;
        let consistent = match arg.owner {
            ValueOwner::BlockArgument { block: owner, index: owner_index } => {
                owner == block.id && owner_index as usize == index
            }
            ValueOwner::OpResult { .. } => false,
        };
        if !consistent {
            let kind = if arg.is_block_arg() { "Block" } else { "Op" };
            let message = format!(
                "block argument %{arg_id} has inconsistent owner/index ({kind}, {}), expected \
                 (this block, {index})",
                arg.index()
            );
            return Err(fail(message, None));
        }
        local.insert(*arg_id);
    }

    for op_id in &block.ops {
        let op = program.op(*op_id)?;
        verify_op(program, op)?;
        for operand_id in &op.operands {
            if !local.contains(operand_id) {
                let message = format!(
                    "{}: operand %{operand_id} is not defined before this use (SSA dominance)",
                    where_op(op)
                );
                return Err(fail(message, op.location.as_ref()));
            }
        }
        for region in &op.regions {
            verify_nested_region(program, op, region, &local)?;
        }
        local.extend(op.results.iter().copied());
    }
    Ok(())
}

fn verify_nested_region(
    program: &Program,
    op: &Op,
    region: &Region,
    scope: &BTreeSet<ValueId>,
) -> Result<()> {
    let loc = op.location.as_ref();
    let unknown_block = |block_id: BlockId| {
        let message = format!(
            "{}: region block {block_id} is not defined in module '{}'",
            where_op(op),
            program.name
        );
        fail(message, loc)
    };
    if region.blocks.is_empty() {
        return Err(fail(format!("{}: nested region has no blocks", where_op(op)), loc));
    }
    let entry_id = region.blocks[0];
    let entry = program.block(entry_id).map_err(|_| unknown_block(entry_id))?;
    // v1 binding convention: entry-block argument count/type == operand
    // count/type (for `if` ALL operands, including the predicate, bind).
    if entry.arguments.len() != op.operands.len() {
        let message = format!(
            "{}: region entry block has {} arguments, expected {} (one per operand)",
            where_op(op),
            entry.arguments.len(),
            op.operands.len()
        );
        return Err(fail(message, loc));
    }
    let missing = |index: usize, what: &str| {
        let message = format!(
            "{}: region entry {what} {index} is not defined in module '{}'",
            where_op(op),
            program.name
        );
        fail(message, loc)
    };
    for (index, (arg_id, operand_id)) in entry.arguments.iter().zip(&op.operands).enumerate() {
        let arg = program.value(*arg_id).map_err(|_| missing(index, "argument"))?;
        let operand = program.value(*operand_id).map_err(|_| missing(index, "operand"))?;
        if arg.ty != operand.ty {
            let message = format!(
                "{}: region entry argument {index} type {} does not match operand type {}",
                where_op(op),
                arg.ty,
                operand.ty
            );
            return Err(fail(message, loc));
        }
    }

    let mut base = scope.clone();
    base.extend(entry.arguments.iter().copied());
    for block_id in &region.blocks {
        let block = program.block(*block_id).map_err(|_| unknown_block(*block_id))?;
        verify_block_terminator(program, block, &format!("op '{}' region", op.name))?;
        verify_block_ops(program, block, &base)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Op level
// ---------------------------------------------------------------------------

fn verify_op(program: &Program, op: &Op) -> Result<()> {
    let loc = op.location.as_ref();
    let Some(op_schema) = schema::lookup(&op.name) else {
        let message = format!("unknown op '{}' (not in the op registry)", op.name);
        return Err(fail(message, loc));
    };

    if !op_schema.check_arity(op.operands.len()) {
        let message = format!(
            "{}: operand count {} violates declared arity {}",
            where_op(op),
            op.operands.len(),
            op_schema.arity_text()
        );
        return Err(fail(message, loc));
    }
    if op.regions.len() != op_schema.regions {
        let message = format!(
            "{}: has {} regions, declared {}",
            where_op(op),
            op.regions.len(),
            op_schema.regions
        );
        return Err(fail(message, loc));
    }
    if op_schema.result_count_known && !op_schema.check_result_count(op.results.len()) {
        let count = op.results.len();
        let detail = match (op_schema.result_min, op_schema.result_max) {
            (min, Some(max)) if min == max => format!("result count {count} != declared {min}"),
            (min, max) => {
                let hi = max.map_or_else(|| "None".to_string(), |value| value.to_string());
                format!("result count {count} outside declared range ({min}, {hi})")
            }
        };
        return Err(fail(format!("{}: {detail}", where_op(op)), loc));
    }

    verify_attrs(op, op_schema, loc)?;

    for (index, operand_id) in op.operands.iter().enumerate() {
        program.value(*operand_id).map_err(|_| {
            let message = format!(
                "{}: operand {index} (%{operand_id}) is not defined in module '{}'",
                where_op(op),
                program.name
            );
            fail(message, loc)
        })?;
    }
    for (index, result_id) in op.results.iter().enumerate() {
        let value = program.value(*result_id).map_err(|_| {
            let message = format!(
                "{}: result {index} (%{result_id}) is not defined in module '{}'",
                where_op(op),
                program.name
            );
            fail(message, loc)
        })?;
        match value.owner {
            ValueOwner::OpResult { op: owner, index: owner_index } => {
                if owner != op.id {
                    let message = format!("{}: result %{result_id} is not owned by this op", where_op(op));
                    return Err(fail(message, loc));
                }
                if owner_index as usize != index {
                    let message = format!(
                        "{}: result %{result_id} has index {owner_index}, expected {index}",
                        where_op(op)
                    );
                    return Err(fail(message, loc));
                }
            }
            ValueOwner::BlockArgument { .. } => {
                let message = format!("{}: result %{result_id} is not owned by this op", where_op(op));
                return Err(fail(message, loc));
            }
        }
    }

    verify_result_types(program, op, op_schema, loc)
}

// ---------------------------------------------------------------------------
// Attribute schema
// ---------------------------------------------------------------------------

/// Does `value` satisfy the declared attribute type tag?
///
/// Same policy as `envelope::validate_attrs` / `infer::validate_attrs`: a
/// nullable `INTS` attribute may be an explicit `Null` (the Python encoder
/// emits `null` for `conv.strides`, `slice.strides`, `transpose.permutation`,
/// ...).  `AttrValue::matches_type` already accepts `Null` under `INT`.
fn attr_accepts(declared: &AttrSchema, value: &AttrValue) -> bool {
    value.matches_type(declared.ty)
        || (declared.ty == attr_type::INTS && matches!(value, AttrValue::Null))
}

fn verify_attrs(op: &Op, schema: &'static OpSchema, loc: Option<&Location>) -> Result<()> {
    for (name, value) in op.attributes.iter() {
        let Some(declared) = schema.attr(name) else {
            let message = format!("{}: unknown attribute '{name}'", where_op(op));
            return Err(fail(message, loc));
        };
        if !attr_accepts(declared, value) {
            let message = format!(
                "{}: attribute '{name}' must be {}, got {}",
                where_op(op),
                declared.ty,
                describe_attr(value)
            );
            return Err(fail(message, loc));
        }
    }
    for declared in schema.attrs {
        if declared.required && !op.attributes.contains(declared.name) {
            let message = format!(
                "{}: missing required attribute '{}'",
                where_op(op),
                declared.name
            );
            return Err(fail(message, loc));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Result-type agreement
// ---------------------------------------------------------------------------

fn verify_result_types(
    program: &Program,
    op: &Op,
    schema: &'static OpSchema,
    loc: Option<&Location>,
) -> Result<()> {
    if !schema.has_shape_fn {
        return verify_op_specific_results(program, op, loc);
    }
    let mut operand_types = Vec::with_capacity(op.operands.len());
    for operand_id in &op.operands {
        operand_types.push(program.value(*operand_id)?.ty.clone());
    }
    let expected = match infer::infer_op(&op.name, &operand_types, &op.attributes, None, Some(program))
    {
        Ok(types) => types,
        Err(error) => {
            let message = format!("{}: result-type inference failed: {}", where_op(op), error.message);
            return Err(fail(message, loc));
        }
    };
    if expected.len() != op.results.len() {
        let message = format!(
            "{}: inference yields {} results, recorded {}",
            where_op(op),
            expected.len(),
            op.results.len()
        );
        return Err(fail(message, loc));
    }
    for (index, expected_type) in expected.iter().enumerate() {
        check_type_agreement(op, index, expected_type, &program.value(op.results[index])?.ty, loc)?;
    }
    Ok(())
}

fn check_type_agreement(
    op: &Op,
    index: usize,
    expected: &ValueType,
    actual: &ValueType,
    loc: Option<&Location>,
) -> Result<()> {
    if expected.dtype != actual.dtype {
        let message = format!(
            "{}: result {index} dtype mismatch: inferred {}, recorded {}",
            where_op(op),
            expected.dtype,
            actual.dtype
        );
        return Err(fail(message, loc));
    }
    if expected.shape != actual.shape {
        let message = format!(
            "{}: result {index} shape mismatch: inferred {}, recorded {}",
            where_op(op),
            format_shape(&expected.shape),
            format_shape(&actual.shape)
        );
        return Err(fail(message, loc));
    }
    Ok(())
}

/// The block's last op when it is the `return` terminator, else `None`.
fn region_return(program: &Program, block_id: BlockId) -> Option<&Op> {
    let block = program.block(block_id).ok()?;
    let op = program.op(*block.ops.last()?).ok()?;
    let is_return = schema::lookup(&op.name).is_some_and(|schema| schema.is_terminator)
        && op.name == "return";
    if is_return {
        Some(op)
    } else {
        None
    }
}

/// Decode the `result_specs` attribute (a JSON list of type dicts).
fn decode_result_specs(value: &AttrValue) -> Option<Vec<ValueType>> {
    let items: Vec<&Json> = match value {
        AttrValue::Any(Json::Array(items)) => items.iter().collect(),
        AttrValue::Tuple(items) => items
            .iter()
            .map(|item| match item {
                AttrValue::Any(json) => Some(json),
                _ => None,
            })
            .collect::<Option<Vec<&Json>>>()?,
        _ => return None,
    };
    items
        .into_iter()
        .map(wire::value_type_from_json)
        .collect::<Result<Vec<ValueType>>>()
        .ok()
}

/// Result-type agreement for `shape_fn=None` ops (op-specific rules).
fn verify_op_specific_results(program: &Program, op: &Op, loc: Option<&Location>) -> Result<()> {
    let where_ = where_op(op);
    let result_types = || -> Result<Vec<ValueType>> {
        op.results
            .iter()
            .map(|result_id| program.value(*result_id).map(|value| value.ty.clone()))
            .collect()
    };
    match op.name.as_str() {
        "constant" => {
            let payload = op
                .attributes
                .array("value")
                .map_err(|error| fail(format!("{where_}: {}", error.message), loc))?;
            let Some(&result_id) = op.results.first() else {
                return Err(fail(format!("{where_}: constant has no result"), loc));
            };
            let actual = &program.value(result_id)?.ty;
            check_type_agreement(op, 0, &payload.value_type(), actual, loc)
        }
        "if" => {
            let result_types = result_types()?;
            for (region_index, region) in op.regions.iter().enumerate() {
                let Some(&entry_id) = region.blocks.first() else {
                    let message = format!("{where_}: region {region_index} has no blocks");
                    return Err(fail(message, loc));
                };
                let Some(terminator) = region_return(program, entry_id) else {
                    let message = format!("{where_}: region {region_index} has no 'return' terminator");
                    return Err(fail(message, loc));
                };
                let mut branch_types = Vec::with_capacity(terminator.operands.len());
                for (index, operand_id) in terminator.operands.iter().enumerate() {
                    let operand = program.value(*operand_id).map_err(|_| {
                        let message = format!(
                            "{where_}: region {region_index} return operand {index} is not \
                             defined in module '{}'",
                            program.name
                        );
                        fail(message, loc)
                    })?;
                    branch_types.push(operand.ty.clone());
                }
                if branch_types != result_types {
                    let message = format!(
                        "{where_}: region {region_index} returns {}, op results are {}",
                        format_types(&branch_types),
                        format_types(&result_types)
                    );
                    return Err(fail(message, loc));
                }
            }
            Ok(())
        }
        "call" => {
            let callee_name = op
                .attributes
                .str("callee")
                .map_err(|error| fail(format!("{where_}: {}", error.message), loc))?;
            let callee = program.function(callee_name).map_err(|_| {
                let message = format!(
                    "{where_}: no function named '{callee_name}' in module '{}'",
                    program.name
                );
                fail(message, loc)
            })?;
            let result_types = result_types()?;
            if callee.output_types != result_types {
                let message = format!(
                    "{where_}: callee '{callee_name}' returns {}, op results are {}",
                    format_types(&callee.output_types),
                    format_types(&result_types)
                );
                return Err(fail(message, loc));
            }
            Ok(())
        }
        "runtime_call" | "block_call" | "external_call" => {
            let specs_value = op
                .attributes
                .get("result_specs")
                .ok_or_else(|| fail(format!("{where_}: missing attribute 'result_specs'"), loc))?;
            let Some(specs) = decode_result_specs(specs_value) else {
                let message = format!(
                    "{where_}: attribute 'result_specs' must be a sequence of ValueTypes, got {}",
                    describe_attr(specs_value)
                );
                return Err(fail(message, loc));
            };
            let result_types = result_types()?;
            if specs != result_types {
                let message = format!(
                    "{where_}: declared result_specs {} do not match op results {}",
                    format_types(&specs),
                    format_types(&result_types)
                );
                return Err(fail(message, loc));
            }
            Ok(())
        }
        // Zero results — already enforced by the declared result count.
        "return" => Ok(()),
        _ => {
            let message = format!(
                "{where_}: has no shape-inference hook and no op-specific result-type rule"
            );
            Err(fail(message, loc))
        }
    }
}

// ---------------------------------------------------------------------------
// Op <-> block membership
// ---------------------------------------------------------------------------

/// Every op reachable from a function body must be listed in exactly one block.
///
/// The Rust graph has no `op.parent` pointer, so this replaces Python's
/// `op.parent is not block` check: an op reachable from two blocks is rejected.
fn verify_op_block_membership(program: &Program) -> Result<()> {
    let mut counts: BTreeMap<OpId, usize> = BTreeMap::new();
    for function in &program.functions {
        count_region_ops(program, &function.body, &mut counts);
    }
    for (op_id, count) in counts {
        if count > 1 {
            let op = program.op(op_id)?;
            let message = format!(
                "op '{}' (id {op_id}): listed in {count} blocks; every op must belong to exactly \
                 one block",
                op.name
            );
            return Err(fail(message, None));
        }
    }
    Ok(())
}

fn count_region_ops(program: &Program, region: &Region, counts: &mut BTreeMap<OpId, usize>) {
    for block_id in &region.blocks {
        let Ok(block) = program.block(*block_id) else {
            continue; // reported by the structural walk
        };
        for op_id in &block.ops {
            *counts.entry(*op_id).or_insert(0) += 1;
            if let Ok(op) = program.op(*op_id) {
                for nested in &op.regions {
                    count_region_ops(program, nested, counts);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Use bookkeeping
// ---------------------------------------------------------------------------

/// Check use bookkeeping in BOTH directions.
///
/// The forward pass validates every recorded [`crate::graph::Use`] and, in the
/// same pass, builds an index of the recorded `(value, op, operand)` triples;
/// the reverse pass then answers "does this operand have a matching record?"
/// with one set lookup instead of scanning the value's whole use list (the scan
/// is quadratic in the use count of a shared value).
fn verify_uses(program: &Program) -> Result<()> {
    let mut recorded: BTreeSet<(u32, u32, u32)> = BTreeSet::new();
    for (value_id, value) in program.values.iter() {
        for use_ in &value.uses {
            let op = program.op(use_.op).map_err(|_| {
                let message = format!(
                    "value %{value_id}: use record refers to op {} which is not in the module",
                    use_.op
                );
                fail(message, None)
            })?;
            let index = use_.operand_index;
            if index as usize >= op.operands.len() || op.operands[index as usize] != ValueId(value_id)
            {
                let message = format!(
                    "value %{value_id}: use record does not refer to this value at the recorded \
                     operand index"
                );
                return Err(fail(message, None));
            }
            if !recorded.insert((value_id, use_.op.0, index)) {
                let message = format!(
                    "value %{value_id}: duplicate use record for op '{}' operand {index}",
                    op.name
                );
                return Err(fail(message, None));
            }
        }
    }
    for (op_id, op) in program.ops.iter() {
        for (index, operand_id) in op.operands.iter().enumerate() {
            if !recorded.contains(&(operand_id.0, op_id, index as u32)) {
                let message = format!(
                    "op '{}' (id {op_id}): operand {index} (%{operand_id}) has no matching Use \
                     record",
                    op.name
                );
                return Err(fail(message, None));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::{ArrayPayload, Attrs};
    use crate::dtype::DType;
    use crate::graph::{OpId, Value};

    fn f32_scalar() -> ValueType {
        ValueType::scalar(DType::Float32)
    }

    fn matrix() -> ValueType {
        ValueType::of(DType::Float32, [Dim::int(2), Dim::int(3)])
    }

    /// Append `op` to `block` and to the arena.
    fn push_op(program: &mut Program, block: &mut Block, op: Op) -> OpId {
        let id = op.id;
        block.ops.push(id);
        program.add_op(op).unwrap();
        id
    }

    /// Register `op` in the arena (the caller places it in a block).
    fn register_op(program: &mut Program, op: Op) -> OpId {
        let id = op.id;
        program.add_op(op).unwrap();
        id
    }

    /// Register one result value for `op_id` at the next result index.
    fn add_result(program: &mut Program, op_id: OpId, ty: ValueType) -> ValueId {
        let id = program.new_value_id();
        let index = program.op(op_id).unwrap().results.len() as u32;
        program.op_mut(op_id).unwrap().results.push(id);
        program.add_value(Value::op_result(id, ty, op_id, index)).unwrap();
        id
    }

    /// A block argument value registered with its block.
    fn add_block_arg(
        program: &mut Program,
        block: &mut Block,
        ty: ValueType,
    ) -> ValueId {
        let index = block.arguments.len() as u32;
        let id = program.new_value_id();
        program
            .add_value(Value::block_argument(id, ty, block.id, index))
            .unwrap();
        block.arguments.push(id);
        id
    }

    fn add_return(program: &mut Program, block: &mut Block, operands: Vec<ValueId>) -> OpId {
        let id = program.new_op_id();
        let mut op = Op::new(id, "return");
        op.operands = operands;
        push_op(program, block, op)
    }

    fn op_named(program: &Program, name: &str) -> OpId {
        program
            .ops
            .iter()
            .find(|(_, op)| op.name == name)
            .map(|(id, _)| OpId(id))
            .unwrap_or_else(|| panic!("no op named {name}"))
    }

    fn block_of(program: &Program) -> BlockId {
        program.functions[0].body.blocks[0]
    }

    fn error_of(program: &Program) -> Error {
        match verify(program) {
            Ok(()) => panic!("verification unexpectedly passed"),
            Err(error) => error,
        }
    }

    /// `def main(x: f32[2,3]) -> f32[2,3] { return stop_gradient(x) }`.
    ///
    /// `stop_gradient` is used (rather than a reduction) because its shape
    /// function is one of the inline controls `infer_op` already implements:
    /// the category bodies of the native inference port are still stubs, and
    /// verification legitimately reports their absence.
    fn tiny_program() -> Program {
        let mut program = Program::new("main");
        let x_ty = matrix();
        let mut function = Function::new("main", vec![x_ty.clone()]);

        let block_id = program.new_block_id();
        let mut block = Block::new(block_id);
        let arg_id = add_block_arg(&mut program, &mut block, x_ty.clone());

        let grad_id = program.new_op_id();
        let mut stop = Op::new(grad_id, "stop_gradient");
        stop.operands.push(arg_id);
        push_op(&mut program, &mut block, stop);
        let out_id = add_result(&mut program, grad_id, x_ty.clone());
        add_return(&mut program, &mut block, vec![out_id]);

        program.add_block(block).unwrap();
        function.body = Region::single(block_id);
        function.output_types = vec![x_ty];
        program.add_function(function).unwrap();
        program.rebuild_uses();
        program
    }

    /// One `f32` scalar constant op whose payload is `value`.
    fn push_constant(program: &mut Program, block: &mut Block, ty: ValueType) -> (OpId, ValueId) {
        let op_id = program.new_op_id();
        let mut op = Op::new(op_id, "constant");
        op.attributes = Attrs::from_pairs([(
            "value",
            AttrValue::Array(ArrayPayload { dtype: DType::Float32, shape: vec![], data: 1.0f32.to_le_bytes().to_vec() }),
        )]);
        push_op(program, block, op);
        let out_id = add_result(program, op_id, ty);
        (op_id, out_id)
    }

    /// `def main() -> T { return constant(f32[2]) }` with `T` recorded as given.
    fn constant_program(recorded: ValueType) -> Program {
        let mut program = Program::new("main");
        let mut function = Function::new("main", vec![]);
        let block_id = program.new_block_id();
        let mut block = Block::new(block_id);

        let op_id = program.new_op_id();
        let mut op = Op::new(op_id, "constant");
        op.attributes = Attrs::from_pairs([(
            "value",
            AttrValue::Array(ArrayPayload { dtype: DType::Float32, shape: vec![2], data: vec![0u8; 8] }),
        )]);
        push_op(&mut program, &mut block, op);
        let out_id = add_result(&mut program, op_id, recorded.clone());
        add_return(&mut program, &mut block, vec![out_id]);

        program.add_block(block).unwrap();
        function.body = Region::single(block_id);
        function.output_types = vec![recorded];
        program.add_function(function).unwrap();
        program.rebuild_uses();
        program
    }

    /// `def main(pred: bool) -> f32 { if (pred) { return 1.0 } else { ... } }`.
    ///
    /// `mismatched` makes the second branch return its `bool` block argument.
    fn if_program(mismatched: bool) -> Program {
        let mut program = Program::new("main");
        let pred_ty = ValueType::scalar(DType::Bool);
        let out_ty = f32_scalar();
        let mut function = Function::new("main", vec![pred_ty.clone()]);

        let block_id = program.new_block_id();
        let mut block = Block::new(block_id);
        let pred_id = add_block_arg(&mut program, &mut block, pred_ty.clone());

        let mut regions = Vec::new();
        for branch in 0..2u8 {
            let branch_id = program.new_block_id();
            let mut branch_block = Block::new(branch_id);
            let arg_id = add_block_arg(&mut program, &mut branch_block, pred_ty.clone());
            let (_, const_out) = push_constant(&mut program, &mut branch_block, out_ty.clone());
            let returned = if mismatched && branch == 1 { arg_id } else { const_out };
            add_return(&mut program, &mut branch_block, vec![returned]);
            program.add_block(branch_block).unwrap();
            regions.push(Region::single(branch_id));
        }

        let if_id = program.new_op_id();
        let mut if_op = Op::new(if_id, "if");
        if_op.operands.push(pred_id);
        if_op.regions = regions;
        push_op(&mut program, &mut block, if_op);
        let if_out = add_result(&mut program, if_id, out_ty.clone());
        add_return(&mut program, &mut block, vec![if_out]);

        program.add_block(block).unwrap();
        function.body = Region::single(block_id);
        function.output_types = vec![out_ty];
        program.add_function(function).unwrap();
        program.rebuild_uses();
        program
    }

    // --- valid programs ---------------------------------------------------

    #[test]
    fn valid_programs_pass() {
        verify(&tiny_program()).unwrap();
        verify(&constant_program(ValueType::of(DType::Float32, [Dim::int(2)]))).unwrap();
        verify(&if_program(false)).unwrap();
    }

    // --- module / function structure --------------------------------------

    #[test]
    fn empty_module_fails() {
        let error = error_of(&Program::new("empty"));
        assert_eq!(error.kind_str(), "VerificationError");
        assert!(error.message.contains("must contain at least one function"));
    }

    #[test]
    fn duplicate_function_names_fail() {
        let mut program = tiny_program();
        program.functions.push(Function::new("main", vec![]));
        let error = error_of(&program);
        assert!(error.message.contains("duplicate function name 'main'"), "{}", error.message);
    }

    #[test]
    fn function_without_blocks_fails() {
        let mut program = Program::new("main");
        program.add_function(Function::new("main", vec![])).unwrap();
        let error = error_of(&program);
        assert!(error.message.contains("region has no blocks"), "{}", error.message);
    }

    #[test]
    fn multi_block_region_fails() {
        let mut program = tiny_program();
        let extra = program.new_block_id();
        program.add_block(Block::new(extra)).unwrap();
        program.functions[0].body.blocks.push(extra);
        let error = error_of(&program);
        assert!(error.message.contains("v1 requires exactly one"), "{}", error.message);
    }

    #[test]
    fn block_argument_type_mismatch_fails() {
        let mut program = tiny_program();
        let arg_id = program.block(block_of(&program)).unwrap().arguments[0];
        let wrong = ValueType::of(DType::Int32, [Dim::int(2), Dim::int(3)]);
        program.value_mut(arg_id).unwrap().ty = wrong;
        let error = error_of(&program);
        assert!(error.message.contains("entry argument 0 type"), "{}", error.message);
        assert!(error.message.contains("does not match input type"), "{}", error.message);
    }

    #[test]
    fn block_argument_owner_index_mismatch_fails() {
        let mut program = tiny_program();
        let arg_id = program.block(block_of(&program)).unwrap().arguments[0];
        program.value_mut(arg_id).unwrap().owner = ValueOwner::BlockArgument { block: BlockId(99), index: 0 };
        let error = error_of(&program);
        assert!(
            error.message.contains("inconsistent owner/index (Block, 0)"),
            "{}",
            error.message
        );
    }

    // --- terminators ------------------------------------------------------

    #[test]
    fn missing_return_terminator_fails() {
        let mut program = tiny_program();
        program.block_mut(block_of(&program)).unwrap().ops.pop();
        let error = error_of(&program);
        assert!(error.message.contains("no 'return' terminator"), "{}", error.message);
    }

    #[test]
    fn block_without_ops_fails() {
        let mut program = tiny_program();
        program.block_mut(block_of(&program)).unwrap().ops.clear();
        let error = error_of(&program);
        assert!(error.message.contains("missing 'return' terminator"), "{}", error.message);
    }

    #[test]
    fn terminator_not_last_fails() {
        let mut program = tiny_program();
        let grad_id = op_named(&program, "stop_gradient");
        let out_id = program.op(grad_id).unwrap().results[0];
        let block_id = block_of(&program);
        let mut extra = Op::new(program.new_op_id(), "stop_gradient");
        extra.operands.push(out_id);
        let extra_id = register_op(&mut program, extra);
        add_result(&mut program, extra_id, matrix());
        program.block_mut(block_id).unwrap().ops.push(extra_id);
        let error = error_of(&program);
        assert!(error.message.contains("terminator 'return' is not the last op"), "{}", error.message);
    }

    #[test]
    fn output_types_disagreement_fails() {
        let mut program = tiny_program();
        program.functions[0].output_types = vec![ValueType::scalar(DType::Int32)];
        let error = error_of(&program);
        assert!(error.message.contains("disagree with the return operands' types"), "{}", error.message);
    }

    // --- op level ---------------------------------------------------------

    #[test]
    fn unknown_op_fails() {
        let mut program = tiny_program();
        let block_id = block_of(&program);
        let unknown = Op::new(program.new_op_id(), "does_not_exist");
        let unknown_id = register_op(&mut program, unknown);
        program.block_mut(block_id).unwrap().ops.insert(1, unknown_id);
        assert_eq!(program.block(block_id).unwrap().ops[1], unknown_id);
        let error = error_of(&program);
        assert_eq!(error.kind_str(), "VerificationError");
        assert!(
            error.message.contains("unknown op 'does_not_exist' (not in the op registry)"),
            "{}",
            error.message
        );
    }

    #[test]
    fn arity_violation_fails() {
        let mut program = tiny_program();
        let block_id = block_of(&program);
        let arg_id = program.block(block_id).unwrap().arguments[0];
        let mut add = Op::new(program.new_op_id(), "add");
        add.operands.push(arg_id);
        let add_id = register_op(&mut program, add);
        add_result(&mut program, add_id, matrix());
        program.block_mut(block_id).unwrap().ops.insert(1, add_id);
        let error = error_of(&program);
        assert!(
            error.message.contains("operand count 1 violates declared arity 2"),
            "{}",
            error.message
        );
    }

    #[test]
    fn result_dtype_disagreement_with_inference_fails() {
        let mut program = tiny_program();
        let grad_id = op_named(&program, "stop_gradient");
        let out_id = program.op(grad_id).unwrap().results[0];
        let f64_matrix = ValueType::of(DType::Float64, [Dim::int(2), Dim::int(3)]);
        program.value_mut(out_id).unwrap().ty = f64_matrix.clone();
        program.functions[0].output_types = vec![f64_matrix];
        let error = error_of(&program);
        assert!(
            error.message.contains("result 0 dtype mismatch: inferred float32, recorded float64"),
            "{}",
            error.message
        );
    }

    #[test]
    fn result_shape_disagreement_with_inference_fails() {
        let mut program = tiny_program();
        let grad_id = op_named(&program, "stop_gradient");
        let out_id = program.op(grad_id).unwrap().results[0];
        let wrong_shape = ValueType::of(DType::Float32, [Dim::int(1)]);
        program.value_mut(out_id).unwrap().ty = wrong_shape.clone();
        program.functions[0].output_types = vec![wrong_shape];
        let error = error_of(&program);
        assert!(
            error.message.contains("result 0 shape mismatch: inferred (2, 3), recorded (1,)"),
            "{}",
            error.message
        );
    }

    #[test]
    fn location_is_appended_to_the_message() {
        let mut program = tiny_program();
        let grad_id = op_named(&program, "stop_gradient");
        let out_id = program.op(grad_id).unwrap().results[0];
        let f64_matrix = ValueType::of(DType::Float64, [Dim::int(2), Dim::int(3)]);
        program.value_mut(out_id).unwrap().ty = f64_matrix.clone();
        program.functions[0].output_types = vec![f64_matrix];
        program.op_mut(grad_id).unwrap().location = Some(Location::new("model.py", 83, 5, None));
        let error = error_of(&program);
        assert!(error.message.ends_with("[at model.py:83:5]"), "{}", error.message);
    }

    #[test]
    fn result_owner_index_mismatch_fails() {
        let mut program = tiny_program();
        let grad_id = op_named(&program, "stop_gradient");
        let out_id = program.op(grad_id).unwrap().results[0];
        program.value_mut(out_id).unwrap().owner = ValueOwner::OpResult { op: grad_id, index: 1 };
        let error = error_of(&program);
        assert!(error.message.contains("has index 1, expected 0"), "{}", error.message);
    }

    #[test]
    fn dangling_operand_fails() {
        let mut program = tiny_program();
        let block_id = block_of(&program);
        let mut dangling = Op::new(program.new_op_id(), "stop_gradient");
        dangling.operands.push(ValueId(999));
        let dangling_id = register_op(&mut program, dangling);
        add_result(&mut program, dangling_id, matrix());
        program.block_mut(block_id).unwrap().ops.insert(1, dangling_id);
        let error = error_of(&program);
        assert!(error.message.contains("operand 0 (%999) is not defined"), "{}", error.message);
    }

    #[test]
    fn operand_outside_scope_fails() {
        let mut program = tiny_program();
        let grad_id = op_named(&program, "stop_gradient");
        let out_id = program.op(grad_id).unwrap().results[0];
        let block_id = block_of(&program);
        let mut early = Op::new(program.new_op_id(), "stop_gradient");
        early.operands.push(out_id);
        let early_id = register_op(&mut program, early);
        add_result(&mut program, early_id, matrix());
        program.block_mut(block_id).unwrap().ops.insert(0, early_id);
        let error = error_of(&program);
        assert!(error.message.contains("(SSA dominance)"), "{}", error.message);
    }

    // --- op-specific result types ----------------------------------------

    #[test]
    fn constant_payload_disagreement_fails() {
        let program = constant_program(ValueType::of(DType::Float64, [Dim::int(2)]));
        let error = error_of(&program);
        assert!(
            error.message.contains("result 0 dtype mismatch: inferred float32, recorded float64"),
            "{}",
            error.message
        );
    }

    #[test]
    fn if_branch_type_disagreement_fails() {
        let error = error_of(&if_program(true));
        assert!(
            error.message.contains("region 1 returns (tensor<i1>,), op results are (tensor<f32>,)"),
            "{}",
            error.message
        );
    }

    #[test]
    fn nested_region_entry_argument_count_fails() {
        let mut program = if_program(false);
        let if_id = op_named(&program, "if");
        let branch_id = program.op(if_id).unwrap().regions[0].blocks[0];
        program.block_mut(branch_id).unwrap().arguments.clear();
        let error = error_of(&program);
        assert!(
            error.message.contains("region entry block has 0 arguments, expected 1"),
            "{}",
            error.message
        );
    }

    #[test]
    fn op_listed_in_two_blocks_fails() {
        // Listing the second branch block in BOTH `if` regions makes every op
        // of that block reachable from two blocks.
        let mut program = if_program(false);
        let if_id = op_named(&program, "if");
        let second = program.op(if_id).unwrap().regions[1].blocks[0];
        program.op_mut(if_id).unwrap().regions[0].blocks.push(second);
        let error = error_of(&program);
        assert!(error.message.contains("listed in 2 blocks"), "{}", error.message);
    }

    // --- use bookkeeping --------------------------------------------------

    #[test]
    fn missing_use_record_fails() {
        let mut program = tiny_program();
        let arg_id = program.block(block_of(&program)).unwrap().arguments[0];
        program.value_mut(arg_id).unwrap().uses.clear();
        let error = error_of(&program);
        assert!(error.message.contains("has no matching Use record"), "{}", error.message);
    }

    #[test]
    fn duplicate_use_record_fails() {
        let mut program = tiny_program();
        let arg_id = program.block(block_of(&program)).unwrap().arguments[0];
        let recorded = program.value(arg_id).unwrap().uses[0];
        program.value_mut(arg_id).unwrap().uses.push(recorded);
        let error = error_of(&program);
        assert!(error.message.contains("duplicate use record"), "{}", error.message);
    }

    #[test]
    fn misplaced_use_record_fails() {
        let mut program = tiny_program();
        let grad_id = op_named(&program, "stop_gradient");
        let out_id = program.op(grad_id).unwrap().results[0];
        program
            .value_mut(out_id)
            .unwrap()
            .uses
            .push(crate::graph::Use { op: grad_id, operand_index: 3 });
        let error = error_of(&program);
        assert!(error.message.contains("does not refer to this value"), "{}", error.message);
    }
}
