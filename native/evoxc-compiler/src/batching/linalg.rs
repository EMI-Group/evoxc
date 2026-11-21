//! Builtin batching rules for linalg ops.
//!
//! Ports `_dot_batching`, `_conv_batching`, `_triangular_batching`,
//! `_cumsum_batching` and `_solve_batching` from
//! `evoxc/transforms/rules.py`.  Every rule receives the op's *rewritten*
//! operands (their batch dims already prepended by the engine) plus one
//! [`MappedAxes`] per operand, and returns one rewritten value + one
//! [`MappedAxes`] per result.

use evoxc_core::attr::AttrValue;
use evoxc_core::dim::Dim;
use evoxc_core::error::{Error, Result};
use evoxc_core::graph::ValueId;

use super::common::mapped;
use super::MappedAxes;
use crate::emit::{ConvParams, RuleCtx, RuleOp};
use crate::Registry;

/// `_dot_batching`: matmul already broadcasts batch dims, so align every
/// operand that is unmapped for some of the pass's batch axes with leading
/// size-one dims and rebuild the op.
pub fn dot_rule(
    ctx: &mut RuleCtx<'_>,
    op: &RuleOp,
    axes: &[MappedAxes],
) -> Result<(Vec<ValueId>, Vec<MappedAxes>)> {
    let (left, right) = binary_operands(op)?;
    let left_count = operand_axes(axes, 0, op)?.count();
    let right_count = operand_axes(axes, 1, op)?.count();
    let mapped_count = left_count.max(right_count);
    let left = align_leading(ctx, left, left_count, mapped_count)?;
    let right = align_leading(ctx, right, right_count, mapped_count)?;
    let out = ctx.dot(left, right)?;
    Ok((vec![out], vec![mapped(mapped_count)]))
}

/// `_conv_batching`: the IR conv op has exactly ONE batch dim (`x.shape[0]`),
/// so a mapped input's batch dims are folded into it via reshape (per-row
/// convs are independent — numerically exact).  A mapped kernel would need a
/// symbolic `batch_group_count` (a static int attribute) → v1 deferral; so is
/// an already-batch-grouped source conv (validated by
/// [`ConvParams::from_op`]).
pub fn conv_rule(
    ctx: &mut RuleCtx<'_>,
    op: &RuleOp,
    axes: &[MappedAxes],
) -> Result<(Vec<ValueId>, Vec<MappedAxes>)> {
    let (x, kernel) = binary_operands(op)?;
    let kx = operand_axes(axes, 0, op)?.count();
    let kw = operand_axes(axes, 1, op)?.count();
    if kw > 0 {
        return Err(Error::transform(
            "vectorize: cannot batch op 'conv' with a mapped kernel — batch_group_count must be \
             a static int (v1 gap)",
        ));
    }
    let params = ConvParams::from_op(op)?;
    if kx == 0 {
        let out = ctx.conv(x, kernel, &params)?;
        return Ok((vec![out], vec![MappedAxes::UNMAPPED]));
    }
    let old_shape = op.original_operand(0)?.shape.clone();
    let batch = old_shape.first().cloned().ok_or_else(|| {
        Error::transform("vectorize: cannot batch op 'conv' with a rank-0 input (v1 gap)")
    })?;
    if batch.is_dynamic() {
        return Err(Error::transform(
            "vectorize: cannot batch op 'conv' over a dynamic batch dim (v1 gap)",
        ));
    }
    let x_shape = ctx.shape(x)?;
    if kx > x_shape.len() {
        return Err(Error::transform(format!(
            "vectorize: batching metadata for op 'conv' marks {kx} leading axes on a rank-{} \
             operand",
            x_shape.len()
        )));
    }
    let batch_dims: Vec<Dim> = x_shape[..kx].to_vec();
    // merged = prod(batch_dims) * old_batch — folded in the frontend's order.
    let mut merged = batch_dims[0].clone();
    for dim in &batch_dims[1..] {
        merged = merged.mul(dim.clone());
    }
    let merged = merged.mul(batch.clone());
    let mut flat_dims: Vec<Dim> = vec![merged];
    flat_dims.extend(old_shape[1..].iter().cloned());
    let flat_x = ctx.reshape(x, &flat_dims)?;
    let out = ctx.conv(flat_x, kernel, &params)?;
    let out_shape = ctx.shape(out)?;
    let mut back_dims: Vec<Dim> = batch_dims;
    back_dims.push(batch);
    back_dims.extend(out_shape[1..].iter().cloned());
    let back = ctx.reshape(out, &back_dims)?;
    Ok((vec![back], vec![mapped(kx)]))
}

/// `_triangular_batching`: the mask applies to the last two dims of every batch
/// row — rebuild `tril`/`triu` on the (possibly mapped) operand unchanged.
pub fn triangular_rule(
    ctx: &mut RuleCtx<'_>,
    op: &RuleOp,
    axes: &[MappedAxes],
) -> Result<(Vec<ValueId>, Vec<MappedAxes>)> {
    let value = unary_operand(op)?;
    let k = op.opt_int("k")?.unwrap_or(0);
    let out = match op.name.as_str() {
        "tril" => ctx.tril(value, k)?,
        "triu" => ctx.triu(value, k)?,
        other => {
            return Err(Error::transform(format!(
                "vectorize: triangular batching rule cannot rewrite op '{other}'"
            )))
        }
    };
    Ok((vec![out], vec![operand_axes(axes, 0, op)?]))
}

/// `_cumsum_batching`: the scan runs per batch row, so the axis shifts by the
/// mapped count; the shape is preserved and the mapped axes pass through.
pub fn cumsum_rule(
    ctx: &mut RuleCtx<'_>,
    op: &RuleOp,
    axes: &[MappedAxes],
) -> Result<(Vec<ValueId>, Vec<MappedAxes>)> {
    let value = unary_operand(op)?;
    let mapped = operand_axes(axes, 0, op)?;
    let axis = op.int("axis")?;
    let reverse = optional_bool(op, "reverse", false)?;
    let out = ctx.cumsum(value, axis + mapped.count() as i64, reverse)?;
    Ok((vec![out], vec![mapped]))
}

/// `_solve_batching`: `solve` broadcasts batch dims like `dot` — align
/// unmapped operands with leading size-one dims and rebuild.
/// `left_side=False` is a v1 deferral (the IR op is left-side only).
pub fn solve_rule(
    ctx: &mut RuleCtx<'_>,
    op: &RuleOp,
    axes: &[MappedAxes],
) -> Result<(Vec<ValueId>, Vec<MappedAxes>)> {
    if !optional_bool(op, "left_side", true)? {
        return Err(Error::transform(
            "vectorize: cannot batch op 'solve' with left_side=False (v1 gap)",
        ));
    }
    let (a, b) = binary_operands(op)?;
    let a_count = operand_axes(axes, 0, op)?.count();
    let b_count = operand_axes(axes, 1, op)?.count();
    let mapped_count = a_count.max(b_count);
    let a = align_leading(ctx, a, a_count, mapped_count)?;
    let b = align_leading(ctx, b, b_count, mapped_count)?;
    let out = ctx.solve(a, b)?;
    Ok((vec![out], vec![mapped(mapped_count)]))
}

/// Prepend `target - count` leading size-one dims so a lower-batch-rank
/// operand broadcasts against the pass's batch axes (the Python
/// `ops.broadcast(value, (1,) * (mapped_count - count) + shape)` step).
fn align_leading(
    ctx: &mut RuleCtx<'_>,
    value: ValueId,
    count: usize,
    target: usize,
) -> Result<ValueId> {
    if count >= target {
        return Ok(value);
    }
    let mut dims = vec![Dim::Int(1); target - count];
    dims.extend(ctx.shape(value)?);
    ctx.broadcast(value, &dims)
}

/// The two operands of a binary op (rules never index blindly).
fn binary_operands(op: &RuleOp) -> Result<(ValueId, ValueId)> {
    match op.operands.as_slice() {
        [left, right] => Ok((*left, *right)),
        other => Err(Error::transform(format!(
            "vectorize: batching rule for op '{}' expects 2 operand(s), got {}",
            op.name,
            other.len()
        ))),
    }
}

/// The single operand of a unary op.
fn unary_operand(op: &RuleOp) -> Result<ValueId> {
    match op.operands.as_slice() {
        [value] => Ok(*value),
        other => Err(Error::transform(format!(
            "vectorize: batching rule for op '{}' expects 1 operand, got {}",
            op.name,
            other.len()
        ))),
    }
}

/// The mapped axes recorded for operand `index`.
fn operand_axes(axes: &[MappedAxes], index: usize, op: &RuleOp) -> Result<MappedAxes> {
    axes.get(index).copied().ok_or_else(|| {
        Error::transform(format!(
            "vectorize: batching rule for op '{}': no mapped-axes metadata for operand {index}",
            op.name
        ))
    })
}

/// An optional bool attribute with the schema's documented default.
fn optional_bool(op: &RuleOp, name: &str, default: bool) -> Result<bool> {
    match op.attributes.get(name) {
        None | Some(AttrValue::Null) => Ok(default),
        Some(value) => value.as_bool(),
    }
}

/// Register this file's batching rules.
pub fn register_rules(registry: &mut Registry) {
    registry.batching.register("dot", dot_rule);
    registry.batching.register("conv", conv_rule);
    registry.batching.register("tril", triangular_rule);
    registry.batching.register("triu", triangular_rule);
    registry.batching.register("cumsum", cumsum_rule);
    registry.batching.register("solve", solve_rule);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::dims_i64;
    use crate::{Builder, TransformConfig, TransformEnv};
    use evoxc_core::attr::Attrs;
    use evoxc_core::dtype::DType;
    use evoxc_core::graph::Program;
    use evoxc_core::value_type::ValueType;

    fn f32(shape: &[i64]) -> ValueType {
        ValueType::of(DType::parse("float32").unwrap(), dims_i64(shape))
    }

    fn batched(shape: &[Dim]) -> ValueType {
        ValueType::new(DType::parse("float32").unwrap(), shape.to_vec())
    }

    /// The attribute set the frontend's `conv` always writes (one entry per
    /// spatial dim for the three dilation/stride lists).
    fn conv_attrs(spatial: usize, batch_group_count: i64) -> Attrs {
        let ones = vec![1i64; spatial];
        Attrs::from_pairs([
            ("strides", AttrValue::Ints(ones.clone())),
            ("input_dilation", AttrValue::Ints(ones.clone())),
            ("kernel_dilation", AttrValue::Ints(ones)),
            ("feature_group_count", AttrValue::Int(1)),
            ("batch_group_count", AttrValue::Int(batch_group_count)),
        ])
    }

    /// A source op presented to a rule: source (unbatched) types, the rewritten
    /// batched operand values and their batched types.
    fn rule_op(
        name: &str,
        attributes: Attrs,
        original_operand_types: Vec<ValueType>,
        original_result_types: Vec<ValueType>,
        operands: Vec<ValueId>,
        operand_types: Vec<ValueType>,
    ) -> RuleOp {
        RuleOp {
            name: name.into(),
            attributes,
            original_operand_types,
            original_result_types,
            operands,
            operand_types,
            results: Vec::new(),
            result_types: Vec::new(),
            location: None,
        }
    }

    fn op_names(ctx: &RuleCtx<'_>) -> Vec<String> {
        ctx.builder
            .program()
            .ops
            .iter()
            .map(|(_, op)| op.name.clone())
            .collect()
    }

    fn axis_of(ctx: &RuleCtx<'_>, name: &str) -> i64 {
        ctx.builder
            .program()
            .ops
            .iter()
            .find(|(_, op)| op.name == name)
            .map(|(_, op)| op.attributes.i64("axis").unwrap())
            .unwrap_or_else(|| panic!("no '{name}' op emitted"))
    }

    #[test]
    fn register_rules_covers_every_linalg_rule() {
        let mut registry = Registry::new();
        register_rules(&mut registry);
        for name in ["dot", "conv", "tril", "triu", "cumsum", "solve"] {
            assert!(
                registry.batching.contains(name),
                "missing rule for '{name}'"
            );
        }
        assert_eq!(registry.batching.len(), 6);
    }

    #[test]
    fn dot_batching_aligns_the_unmapped_operand() {
        let mut builder = Builder::new("main");
        builder
            .open_function(
                "main",
                &[
                    batched(&[Dim::named("batch"), Dim::Int(2), Dim::Int(3)]),
                    f32(&[3, 4]),
                ],
            )
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let source = rule_op(
            "dot",
            Attrs::new(),
            vec![f32(&[2, 3]), f32(&[3, 4])],
            vec![f32(&[2, 4])],
            args.clone(),
            vec![
                batched(&[Dim::named("batch"), Dim::Int(2), Dim::Int(3)]),
                f32(&[3, 4]),
            ],
        );
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let (values, axes) = dot_rule(
            &mut ctx,
            &source,
            &[MappedAxes::new(1), MappedAxes::UNMAPPED],
        )
        .unwrap();
        assert_eq!(axes, vec![MappedAxes::new(1)]);
        assert_eq!(
            ctx.shape(values[0]).unwrap(),
            vec![Dim::named("batch"), Dim::Int(2), Dim::Int(4)]
        );
        // the unmapped rhs is broadcast to one leading size-one dim, then dotted
        assert_eq!(op_names(&ctx), vec!["broadcast", "dot"]);
    }

    #[test]
    fn dot_batching_emits_no_broadcast_when_both_are_mapped() {
        let mut builder = Builder::new("main");
        let outer = batched(&[
            Dim::named("batch"),
            Dim::named("batch_1"),
            Dim::Int(2),
            Dim::Int(3),
        ]);
        let rhs = batched(&[
            Dim::named("batch"),
            Dim::named("batch_1"),
            Dim::Int(3),
            Dim::Int(4),
        ]);
        builder
            .open_function("main", &[outer.clone(), rhs.clone()])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let source = rule_op(
            "dot",
            Attrs::new(),
            vec![f32(&[2, 3]), f32(&[3, 4])],
            vec![f32(&[2, 4])],
            args.clone(),
            vec![outer, rhs],
        );
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let (values, axes) =
            dot_rule(&mut ctx, &source, &[MappedAxes::new(2), MappedAxes::new(2)]).unwrap();
        assert_eq!(axes, vec![MappedAxes::new(2)]);
        assert_eq!(
            ctx.shape(values[0]).unwrap(),
            vec![
                Dim::named("batch"),
                Dim::named("batch_1"),
                Dim::Int(2),
                Dim::Int(4)
            ]
        );
        assert_eq!(op_names(&ctx), vec!["dot"]);
    }

    #[test]
    fn conv_batching_folds_the_batch_dims_into_the_batch_axis() {
        let mut builder = Builder::new("main");
        let x = batched(&[
            Dim::named("batch"),
            Dim::Int(1),
            Dim::Int(4),
            Dim::Int(8),
            Dim::Int(8),
        ]);
        let w = f32(&[1, 4, 3, 3]);
        builder
            .open_function("main", &[x.clone(), w.clone()])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let source = rule_op(
            "conv",
            conv_attrs(2, 1),
            vec![f32(&[1, 4, 8, 8]), f32(&[1, 4, 3, 3])],
            vec![f32(&[1, 1, 6, 6])],
            args.clone(),
            vec![x, w],
        );
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let (values, axes) = conv_rule(
            &mut ctx,
            &source,
            &[MappedAxes::new(1), MappedAxes::UNMAPPED],
        )
        .unwrap();
        assert_eq!(axes, vec![MappedAxes::new(1)]);
        assert_eq!(
            ctx.shape(values[0]).unwrap(),
            vec![
                Dim::named("batch"),
                Dim::Int(1),
                Dim::Int(1),
                Dim::Int(6),
                Dim::Int(6)
            ]
        );
        assert_eq!(op_names(&ctx), vec!["reshape", "conv", "reshape"]);
        // the folded batch dim lands on the conv input's N axis
        let program = ctx.builder.program();
        let conv_op = program
            .ops
            .iter()
            .find(|(_, op)| op.name == "conv")
            .map(|(_, op)| op)
            .unwrap();
        let conv_input = &program.value(conv_op.operands[0]).unwrap().ty;
        assert_eq!(conv_input.rank(), 4);
        assert_eq!(
            conv_input.shape[1..].to_vec(),
            dims_i64(&[4, 8, 8]),
            "the batch axis replaces N, C stays"
        );
        assert_eq!(
            conv_op.attributes.ints("strides").unwrap(),
            &[1, 1],
            "the conv parameters are re-emitted"
        );
    }

    #[test]
    fn conv_batching_defers_batch_grouped_and_mapped_kernel() {
        let attrs = Attrs::from_pairs([("batch_group_count", AttrValue::Int(2))]);
        let mut builder = Builder::new("main");
        let x = batched(&[Dim::named("batch"), Dim::Int(1), Dim::Int(4), Dim::Int(8)]);
        let w = f32(&[1, 4, 3]);
        builder
            .open_function("main", &[x.clone(), w.clone()])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let source = rule_op(
            "conv",
            attrs,
            vec![f32(&[1, 4, 8]), f32(&[1, 4, 3])],
            vec![f32(&[1, 1, 6])],
            args.clone(),
            vec![x, w],
        );
        let mut ctx = RuleCtx::new(&mut builder);
        let err = conv_rule(
            &mut ctx,
            &source,
            &[MappedAxes::new(1), MappedAxes::UNMAPPED],
        )
        .unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        assert!(
            err.message.contains("batch_group_count 2"),
            "{}",
            err.message
        );
        assert!(
            err.message.contains("batch-grouped conv"),
            "{}",
            err.message
        );

        // a mapped kernel needs a symbolic batch_group_count → explicit deferral
        let mut builder = Builder::new("main");
        let x = f32(&[1, 4, 8]);
        let w = batched(&[Dim::named("batch"), Dim::Int(1), Dim::Int(4), Dim::Int(3)]);
        builder
            .open_function("main", &[x.clone(), w.clone()])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let source = rule_op(
            "conv",
            Attrs::new(),
            vec![f32(&[1, 4, 8]), f32(&[1, 4, 3])],
            vec![f32(&[1, 1, 6])],
            args.clone(),
            vec![x, w],
        );
        let mut ctx = RuleCtx::new(&mut builder);
        let err = conv_rule(
            &mut ctx,
            &source,
            &[MappedAxes::UNMAPPED, MappedAxes::new(1)],
        )
        .unwrap_err();
        assert!(err.message.contains("mapped kernel"), "{}", err.message);
    }

    #[test]
    fn triangular_batching_passes_mapped_axes_through() {
        for (name, k) in [("tril", 1i64), ("triu", -1)] {
            let mut builder = Builder::new("main");
            let value = batched(&[Dim::named("batch"), Dim::Int(3), Dim::Int(3)]);
            builder
                .open_function("main", std::slice::from_ref(&value))
                .unwrap();
            let args = builder.block_arguments().unwrap();
            let source = rule_op(
                name,
                Attrs::from_pairs([("k", AttrValue::Int(k))]),
                vec![f32(&[3, 3])],
                vec![f32(&[3, 3])],
                args.clone(),
                vec![value],
            );
            let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
            let (values, axes) = triangular_rule(&mut ctx, &source, &[MappedAxes::new(1)]).unwrap();
            assert_eq!(axes, vec![MappedAxes::new(1)], "{name}");
            assert_eq!(
                ctx.shape(values[0]).unwrap(),
                vec![Dim::named("batch"), Dim::Int(3), Dim::Int(3)],
                "{name}"
            );
            assert_eq!(op_names(&ctx), vec![name.to_string()]);
            let emitted = ctx
                .builder
                .program()
                .ops
                .iter()
                .find(|(_, op)| op.name == name)
                .map(|(_, op)| op.attributes.i64("k").unwrap())
                .unwrap();
            assert_eq!(emitted, k, "{name} keeps its diagonal offset");
        }
    }

    #[test]
    fn cumsum_batching_shifts_the_scan_axis() {
        let mut builder = Builder::new("main");
        let value = batched(&[Dim::named("batch"), Dim::Int(4)]);
        builder
            .open_function("main", std::slice::from_ref(&value))
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let source = rule_op(
            "cumsum",
            Attrs::from_pairs([
                ("axis", AttrValue::Int(0)),
                ("reverse", AttrValue::Bool(true)),
            ]),
            vec![f32(&[4])],
            vec![f32(&[4])],
            args.clone(),
            vec![value],
        );
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let (values, axes) = cumsum_rule(&mut ctx, &source, &[MappedAxes::new(1)]).unwrap();
        assert_eq!(axes, vec![MappedAxes::new(1)]);
        assert_eq!(
            ctx.shape(values[0]).unwrap(),
            vec![Dim::named("batch"), Dim::Int(4)]
        );
        assert_eq!(op_names(&ctx), vec!["cumsum"]);
        assert_eq!(axis_of(&ctx, "cumsum"), 1);
        let emitted = ctx
            .builder
            .program()
            .ops
            .iter()
            .next()
            .map(|(_, op)| op.attributes.bool("reverse").unwrap())
            .unwrap();
        assert!(emitted, "the scan direction passes through");
    }

    #[test]
    fn solve_batching_broadcasts_the_unmapped_rhs() {
        let mut builder = Builder::new("main");
        let a = batched(&[Dim::named("batch"), Dim::Int(3), Dim::Int(3)]);
        let b = f32(&[3, 2]);
        builder
            .open_function("main", &[a.clone(), b.clone()])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let source = rule_op(
            "solve",
            Attrs::new(),
            vec![f32(&[3, 3]), f32(&[3, 2])],
            vec![f32(&[3, 2])],
            args.clone(),
            vec![a, b],
        );
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let (values, axes) = solve_rule(
            &mut ctx,
            &source,
            &[MappedAxes::new(1), MappedAxes::UNMAPPED],
        )
        .unwrap();
        assert_eq!(axes, vec![MappedAxes::new(1)]);
        assert_eq!(
            ctx.shape(values[0]).unwrap(),
            vec![Dim::named("batch"), Dim::Int(3), Dim::Int(2)]
        );
        assert_eq!(op_names(&ctx), vec!["broadcast", "solve"]);
    }

    #[test]
    fn solve_batching_defers_left_side_false() {
        let mut builder = Builder::new("main");
        let a = batched(&[Dim::named("batch"), Dim::Int(3), Dim::Int(3)]);
        let b = f32(&[3, 2]);
        builder
            .open_function("main", &[a.clone(), b.clone()])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let source = rule_op(
            "solve",
            Attrs::from_pairs([("left_side", AttrValue::Bool(false))]),
            vec![f32(&[3, 3]), f32(&[3, 2])],
            vec![f32(&[3, 2])],
            args.clone(),
            vec![a, b],
        );
        let mut ctx = RuleCtx::new(&mut builder);
        let err = solve_rule(
            &mut ctx,
            &source,
            &[MappedAxes::new(1), MappedAxes::UNMAPPED],
        )
        .unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        assert!(
            err.message
                .contains("cannot batch op 'solve' with left_side=False"),
            "{}",
            err.message
        );
    }

    #[test]
    fn vectorize_wires_the_linalg_rules_into_a_graph() {
        let mut builder = Builder::new("main");
        builder
            .open_function("main", &[f32(&[2, 3]), f32(&[3, 4])])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let out = builder.emit_typed_one("dot", &args, &Attrs::new()).unwrap();
        builder.set_terminator(&[out], None).unwrap();
        let program: Program = builder.finish();

        let config = TransformConfig {
            axes: vec![(args[0], Some(0)), (args[1], None)],
            ..TransformConfig::default()
        };
        let output =
            crate::vectorize::vectorize_graph(&program, &config, &TransformEnv::builtin()).unwrap();
        let function = &output.program.functions[0];
        assert_eq!(
            function.input_types[0],
            batched(&[Dim::named("batch"), Dim::Int(2), Dim::Int(3)])
        );
        assert_eq!(function.input_types[1], f32(&[3, 4]));
        assert_eq!(
            output.program.value(output.output_ids[0]).unwrap().ty,
            batched(&[Dim::named("batch"), Dim::Int(2), Dim::Int(4)])
        );
        let names: Vec<&str> = output
            .program
            .ops
            .iter()
            .map(|(_, op)| op.name.as_str())
            .collect();
        assert_eq!(names, vec!["broadcast", "dot", "return"]);
    }
}
