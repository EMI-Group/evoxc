//! The `vectorize` engine: rewrite a graph so every mapped input carries one
//! shared leading batch dimension.
//!
//! Port of `evoxc/transforms/batching.py` (`vectorize_graph`): the input module
//! is never mutated; a fresh module is built, block arguments are rebound, and
//! every op is dispatched through the batching registry in program order.  All
//! mapped inputs share ONE fresh symbolic `Dim` (named `batch` at depth 0,
//! `batch_1`, … when nested), so same-level batch dims always unify.

use evoxc_core::dim::Dim;
use evoxc_core::error::{Error, Result};
use evoxc_core::graph::{Function, Program, ValueId};
use evoxc_core::value_type::ValueType;

use crate::batching::{dispatch, MappedAxes};
use crate::builder::Builder;
use crate::emit::{RuleCtx, RuleOp};
use crate::{TransformConfig, TransformEnv, TransformOutput};

/// Store `value` in the dense id-indexed table, growing it if a freshly built
/// value id outruns the source program's id space.
#[inline]
fn set_dense<T: Copy>(slots: &mut Vec<Option<T>>, id: ValueId, value: T) {
    let index = id.0 as usize;
    if slots.len() <= index {
        slots.resize(index + 1, None);
    }
    slots[index] = Some(value);
}

/// Read the dense id-indexed table (`None` for an absent or out-of-range id).
#[inline]
fn get_dense<T: Copy>(slots: &[Option<T>], id: ValueId) -> Option<T> {
    slots.get(id.0 as usize).and_then(|slot| *slot)
}

/// The transform's entry function (exactly one function with one entry block).
pub fn entry_function(program: &Program) -> Result<&Function> {
    if program.functions.len() != 1 {
        return Err(Error::transform(format!(
            "vectorize: graph modules with {} functions are not supported in v1 — a transform \
             needs exactly one entry function",
            program.functions.len()
        )));
    }
    let function = &program.functions[0];
    if !function.body.is_single_block() {
        return Err(Error::transform(
            "vectorize: multi-block function regions are not supported in v1",
        ));
    }
    Ok(function)
}

/// `"(2, 3)"` — a shape rendered the way the Python messages render tuples.
pub fn format_shape(shape: &[Dim]) -> String {
    let entries: Vec<String> = shape
        .iter()
        .map(|dim| match dim {
            Dim::Int(value) => value.to_string(),
            Dim::Named(named) => named.name.clone(),
            Dim::Dynamic => "None".to_string(),
            other => format!("{other:?}"),
        })
        .collect();
    format!("({})", entries.join(", "))
}

/// The fresh batch `Dim` name of this pass.
///
/// `config.batch_dim_name` wins; otherwise the name comes from the module
/// metadata key `batch_depth` (`batch` at depth 0, `batch_1`, `batch_2`, …).
pub fn batch_dim_name(program: &Program, config: &TransformConfig) -> String {
    if let Some(name) = &config.batch_dim_name {
        if !name.trim().is_empty() {
            return name.clone();
        }
    }
    let depth = program
        .metadata
        .get("batch_depth")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    if depth == 0 {
        "batch".to_string()
    } else {
        format!("batch_{depth}")
    }
}

/// `vectorize`: the graph plus the mappedness of each output.
pub fn vectorize_graph(
    program: &Program,
    config: &TransformConfig,
    env: &TransformEnv<'_>,
) -> Result<TransformOutput> {
    Ok(vectorize_impl(program, config, env)?.0)
}

/// The shared implementation: also returns the mappedness of each return
/// operand (what `vmap` needs to rearrange outputs).
pub fn vectorize_impl(
    program: &Program,
    config: &TransformConfig,
    env: &TransformEnv<'_>,
) -> Result<(TransformOutput, Vec<MappedAxes>)> {
    let function = entry_function(program)?;
    let block = program.entry_block(function)?.clone();
    let batch_dim = batch_dim_name(program, config);
    let input_axes = resolve_input_axes(program, &block.arguments, config)?;
    let input_types: Vec<ValueType> = block
        .arguments
        .iter()
        .zip(&input_axes)
        .map(|(id, axis)| {
            let ty = &program.value(*id)?.ty;
            if axis.is_some() {
                let mut shape = Vec::with_capacity(ty.shape.len() + 1);
                shape.push(Dim::named(&batch_dim));
                shape.extend(ty.shape.iter().cloned());
                Ok(ValueType::new(ty.dtype, shape))
            } else {
                Ok(ty.clone())
            }
        })
        .collect::<Result<_>>()?;

    let mut builder = Builder::with_metadata(program.name.clone(), program.metadata.clone());
    builder.open_function(&function.name, &input_types)?;
    builder.set_function_metadata(function.metadata.clone())?;
    let new_args = builder.block_arguments()?;
    if new_args.len() != block.arguments.len() {
        return Err(Error::transform(
            "vectorize: internal error — the rebuilt function did not preserve the input arity",
        ));
    }

    // Dense id-indexed tables: the source program's ids are bounded by
    // `max_value_id`, and freshly built values are appended sequentially, so a
    // `Vec<Option<_>>` beats a `HashMap` without changing any error path (a
    // missing slot is the same "no rewritten value" case).
    let width = program.max_value_id().map_or(0, |id| id as usize + 1);
    let mut remap: Vec<Option<ValueId>> = vec![None; width];
    let mut axes_env: Vec<Option<MappedAxes>> = vec![None; width];
    for ((old, new), axis) in block.arguments.iter().zip(&new_args).zip(&input_axes) {
        set_dense(&mut remap, *old, *new);
        set_dense(&mut axes_env, *new, MappedAxes::from_axis(*axis));
    }

    let mut output_ids: Vec<ValueId> = Vec::new();
    let mut terminated = false;
    for &op_id in &block.ops {
        let op = program.op(op_id)?;
        if op.name == "return" {
            output_ids = op
                .operands
                .iter()
                .map(|value| {
                    get_dense(&remap, *value).ok_or_else(|| {
                        Error::transform(format!(
                            "vectorize: the graph returns %{value}, which has no rewritten value"
                        ))
                    })
                })
                .collect::<Result<_>>()?;
            builder.set_terminator(&output_ids, op.location.clone())?;
            terminated = true;
            break;
        }
        if !op.regions.is_empty() {
            return Err(Error::transform(format!(
                "vectorize: cannot batch op '{}': region-bearing control-flow ops \
                 (cond/while_loop/scan) are not vectorizable in v1",
                op.name
            )));
        }
        let operands = op
            .operands
            .iter()
            .map(|value| {
                get_dense(&remap, *value).ok_or_else(|| {
                    Error::transform(format!(
                        "vectorize: internal error — operand %{value} of op '{}' has no rewritten \
                         value",
                        op.name
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let operand_axes = operands
            .iter()
            .map(|value| {
                get_dense(&axes_env, *value).ok_or_else(|| {
                    Error::transform(format!(
                        "vectorize: internal error — rewritten operand %{value} of op '{}' has no \
                         recorded batching metadata",
                        op.name
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let rule_op = RuleOp::from_source(program, op)?.bind_operands(&builder, &operands)?;
        let len_before = builder.block_op_len()?;
        let (values, result_axes) = {
            let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some(batch_dim.clone()));
            dispatch(env, &mut ctx, &rule_op, &operand_axes)?
        };
        // Dispatch only appends, so the ops emitted by this rule are exactly
        // `ops[len_before..]` — no op-id clone, no membership scan.
        builder.set_op_location_from(len_before, op.location.clone())?;
        if values.len() != op.results.len() {
            return Err(Error::transform(format!(
                "vectorize: batching rule for op '{}' returned {} value(s) for {} result(s)",
                op.name,
                values.len(),
                op.results.len()
            )));
        }
        for ((old, new), axes) in op.results.iter().zip(&values).zip(&result_axes) {
            set_dense(&mut remap, *old, *new);
            set_dense(&mut axes_env, *new, *axes);
        }
    }
    if !terminated {
        return Err(Error::transform(
            "vectorize: the graph function has no 'return' terminator",
        ));
    }
    let output_axes = output_ids
        .iter()
        .map(|value| {
            get_dense(&axes_env, *value).ok_or_else(|| {
                Error::transform(format!(
                    "vectorize: internal error — output %{value} has no recorded batching metadata"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let program = builder.finish();
    Ok((
        TransformOutput {
            program,
            output_ids,
            input_ids: new_args,
            batch_dim: Some(batch_dim),
        },
        output_axes,
    ))
}

/// Normalize the per-input axes: one `Some(0)` / `None` entry per block
/// argument, from `config.input_axes` (positional) overridden by
/// `config.axes` (value ids).
/// Normalize the per-input axes (`config.axes` over `config.input_axes`).
///
/// `pub(crate)`: `vmap` reuses it to decide whether the pass has a mapped input
/// (and therefore a real runtime batch extent to broadcast an unmapped output
/// to).
pub(crate) fn resolve_input_axes(
    _program: &Program,
    arguments: &[ValueId],
    config: &TransformConfig,
) -> Result<Vec<Option<i64>>> {
    if arguments.is_empty() {
        return Ok(Vec::new());
    }
    if config.input_axes.is_empty() && config.axes.is_empty() {
        return Err(Error::transform(
            "vectorize: config.axes is required — one entry per input value id (0 maps the \
             leading axis, null leaves the input unmapped/shared)",
        ));
    }
    let mut entries = if config.input_axes.is_empty() {
        vec![None; arguments.len()]
    } else {
        if config.input_axes.len() != arguments.len() {
            return Err(Error::transform(format!(
                "vectorize: the axes list has {} entries but the function has {} input(s)",
                config.input_axes.len(),
                arguments.len()
            )));
        }
        config.input_axes.clone()
    };
    for (value, axis) in &config.axes {
        let position = arguments.iter().position(|arg| arg == value).ok_or_else(|| {
            Error::transform(format!(
                "vectorize: the axes entry names %{value}, which is not an input of the function"
            ))
        })?;
        entries[position] = *axis;
    }
    for (value, entry) in arguments.iter().zip(&entries) {
        match entry {
            None | Some(0) => {}
            Some(other) => {
                return Err(Error::transform(format!(
                    "vectorize: mapped axis for input %{value} must be 0 (the leading axis) or \
                     None in v1, got {other}"
                )));
            }
        }
    }
    Ok(entries)
}

impl MappedAxes {
    /// `None` → unmapped, `Some(0)` → one leading batch axis.
    pub(crate) fn from_axis(axis: Option<i64>) -> MappedAxes {
        match axis {
            None => MappedAxes::UNMAPPED,
            Some(_) => MappedAxes::new(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::dims_i64;
    use crate::Registry;
    use evoxc_core::attr::{AttrValue, Attrs};
    use evoxc_core::dtype::DType;
    use evoxc_core::graph::{Block, Location, Program, Region};

    fn f32(shape: &[i64]) -> ValueType {
        ValueType::new(DType::parse("float32").unwrap(), dims_i64(shape))
    }

    pub(crate) fn add_program() -> Program {
        let mut builder = Builder::new("main");
        builder
            .open_function("main", &[f32(&[3]), f32(&[3])])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let out = builder.binary("add", args[0], args[1]).unwrap();
        builder.set_terminator(&[out], None).unwrap();
        builder.finish()
    }

    fn axes(entries: &[(usize, Option<i64>)]) -> TransformConfig {
        TransformConfig {
            input_axes: entries.iter().map(|(_, axis)| *axis).collect(),
            ..TransformConfig::default()
        }
    }

    #[test]
    fn vectorize_prepends_one_shared_batch_dim() {
        let program = add_program();
        let args = program.entry_block(&program.functions[0]).unwrap().arguments.clone();
        let config = TransformConfig {
            axes: vec![(args[0], Some(0)), (args[1], None)],
            ..TransformConfig::default()
        };
        let mut registry = Registry::new();
        registry.batching.register("add", crate::batching::elementwise::pointwise_rule);
        let env = TransformEnv::with_registry(&registry);
        let output = vectorize_graph(&program, &config, &env).unwrap();
        let function = &output.program.functions[0];
        assert_eq!(function.input_types[0].shape[0], Dim::named("batch"));
        assert_eq!(function.input_types[0].shape[1], Dim::Int(3));
        assert_eq!(function.input_types[1], f32(&[3]));
        assert_eq!(output.batch_dim.as_deref(), Some("batch"));
        let ty = output.program.value(output.output_ids[0]).unwrap().ty.clone();
        assert_eq!(ty.shape[0], Dim::named("batch"));
        let names: Vec<&str> = output
            .program
            .ops
            .iter()
            .map(|(_, op)| op.name.as_str())
            .collect();
        // Both operands already agree per row ((3) vs (3)), so the pointwise
        // rule aligns nothing: a `batch x 3` add broadcasts against `3` with no
        // reshape (parity with the Python `_pointwise_batching` rule).
        assert_eq!(names, vec!["add", "return"]);
    }

    #[test]
    fn unmapped_inputs_produce_no_batch_axis() {
        let program = add_program();
        let output = vectorize_graph(&program, &axes(&[(0, None), (1, None)]), &TransformEnv::builtin())
            .unwrap();
        assert_eq!(output.program.functions[0].input_types, vec![f32(&[3]), f32(&[3])]);
        let ty = output.program.value(output.output_ids[0]).unwrap().ty.clone();
        assert_eq!(ty, f32(&[3]));
    }

    #[test]
    fn batch_dim_name_follows_metadata_depth() {
        let mut program = add_program();
        program.metadata = serde_json::json!({"batch_depth": 2});
        assert_eq!(batch_dim_name(&program, &TransformConfig::default()), "batch_2");
        let config = TransformConfig {
            batch_dim_name: Some("outer".into()),
            ..TransformConfig::default()
        };
        assert_eq!(batch_dim_name(&program, &config), "outer");
    }

    #[test]
    fn axis_validation_errors() {
        let program = add_program();
        let env = TransformEnv::builtin();
        let err = vectorize_graph(&program, &TransformConfig::default(), &env).unwrap_err();
        assert!(err.message.contains("config.axes is required"));
        let err = vectorize_graph(&program, &axes(&[(0, Some(1)), (1, None)]), &env).unwrap_err();
        assert!(err.message.contains("must be 0 (the leading axis) or None"));
        let mut config = axes(&[(0, None), (1, None)]);
        config.axes = vec![(ValueId(99), Some(0))];
        let err = vectorize_graph(&program, &config, &env).unwrap_err();
        assert!(err.message.contains("is not an input"));
        let config = TransformConfig {
            input_axes: vec![None],
            ..TransformConfig::default()
        };
        let err = vectorize_graph(&program, &config, &env).unwrap_err();
        assert!(err.message.contains("1 entries but the function has 2"));
    }

    #[test]
    fn region_bearing_and_unknown_ops_defer() {
        let mut program = add_program();
        let block_id = program.entry_block(&program.functions[0]).unwrap().id;
        let add_id = *program.block(block_id).unwrap().ops.first().unwrap();
        let region_block = program.new_block_id();
        program.add_block(Block::new(region_block)).unwrap();
        program.op_mut(add_id).unwrap().regions.push(Region::single(region_block));
        let err = vectorize_graph(&program, &axes(&[(0, None), (1, None)]), &TransformEnv::builtin())
            .unwrap_err();
        assert!(err.message.contains("region-bearing"));
        assert_eq!(err.kind_str(), "TransformError");

        let mut program = add_program();
        let block_id = program.entry_block(&program.functions[0]).unwrap().id;
        let add_id = *program.block(block_id).unwrap().ops.first().unwrap();
        program.op_mut(add_id).unwrap().name = "sort".to_string();
        let err = vectorize_graph(&program, &axes(&[(0, None), (1, None)]), &TransformEnv::builtin())
            .unwrap_err();
        assert!(err.message.contains("no batching rule for op 'sort'"));
        assert_eq!(err.kind_str(), "TransformError");
    }

    #[test]
    fn multi_function_modules_defer() {
        let mut program = add_program();
        let extra = Function::new("helper", vec![f32(&[3])]);
        let region_block = program.new_block_id();
        program.add_block(Block::new(region_block)).unwrap();
        program.add_function(Function {
            body: Region::single(region_block),
            ..extra
        })
        .unwrap();
        let err = entry_function(&program).unwrap_err();
        assert!(err.message.contains("graph modules with 2 functions"));

        let mut program = add_program();
        let extra = program.new_block_id();
        program.add_block(Block::new(extra)).unwrap();
        program.functions[0].body.blocks.push(extra);
        let err = entry_function(&program).unwrap_err();
        assert!(err.message.contains("multi-block"));
    }

    #[test]
    fn format_shape_matches_python_tuples() {
        assert_eq!(format_shape(&[]), "()");
        assert_eq!(format_shape(&dims_i64(&[2, 3])), "(2, 3)");
        assert_eq!(format_shape(&[Dim::named("batch"), Dim::Int(3)]), "(batch, 3)");
    }

    /// A `len`-op pointwise chain followed by one multi-op batching rule
    /// (`concatenate` of a mapped and an unmapped operand emits a `broadcast`
    /// plus the `concatenate`), every source op carrying its own location.
    fn located_chain(len: usize) -> Program {
        let location = |line: usize| {
            Some(Location {
                file: "chain.py".to_string(),
                line: line as i64,
                col: 0,
                code_snippet: None,
            })
        };
        let mut builder = Builder::new("main");
        builder
            .open_function("main", &[f32(&[3]), f32(&[3])])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let mut acc = args[0];
        for index in 0..len {
            acc = builder.binary("add", acc, acc).unwrap();
            let op_id = *builder.block_op_ids().unwrap().last().unwrap();
            builder.set_op_location(op_id, location(index + 1)).unwrap();
        }
        let attrs = Attrs::from_pairs([("axis", AttrValue::Int(0))]);
        let out = builder
            .emit_typed_one("concatenate", &[acc, args[1]], &attrs)
            .unwrap();
        let op_id = *builder.block_op_ids().unwrap().last().unwrap();
        builder.set_op_location(op_id, location(len + 1)).unwrap();
        builder.set_terminator(&[out], None).unwrap();
        builder.finish()
    }

    #[test]
    fn locations_cover_exactly_each_dispatch() {
        let len = 200;
        let program = located_chain(len);
        let config = TransformConfig {
            input_axes: vec![Some(0), None],
            ..TransformConfig::default()
        };
        let output = vectorize_graph(&program, &config, &TransformEnv::builtin()).unwrap();
        let block_id = output.program.entry_block(&output.program.functions[0]).unwrap().id;
        let ops: Vec<&evoxc_core::graph::Op> = output
            .program
            .block(block_id)
            .unwrap()
            .ops
            .iter()
            .map(|id| output.program.op(*id).unwrap())
            .collect();
        // len pointwise adds + (broadcast, concatenate) + return.
        assert_eq!(ops.len(), len + 3);
        let names: Vec<&str> = ops.iter().map(|op| op.name.as_str()).collect();
        assert_eq!(&names[..3], &["add", "add", "add"]);
        assert_eq!(&names[len..], &["broadcast", "concatenate", "return"]);

        // Every emitted op carries the location of the source op that emitted it
        // — never a neighbour's (the Θ(n²) rewrite overwrote earlier ops when
        // the "new ops" range was mis-computed).
        for (index, op) in ops[..len].iter().enumerate() {
            let location = op.location.as_ref().expect("pointwise op has a location");
            assert_eq!(location.file, "chain.py");
            assert_eq!(location.line, index as i64 + 1, "op {index} ({})", op.name);
        }
        for op in &ops[len..len + 2] {
            let location = op.location.as_ref().expect("concatenate-rule op has a location");
            assert_eq!(location.line, len as i64 + 1, "op {}", op.name);
        }
        // The terminator keeps its own (absent) source location.
        assert!(ops[len + 2].location.is_none());
    }
}
