//! Elementwise batching rules: per-element ops and reductions.
//!
//! Port of `_pointwise_batching` / `_reduction_batching` from
//! `evoxc/transforms/rules.py`.  Every rule receives the op's *rewritten*
//! operands (values of the output module) plus one [`MappedAxes`] per operand,
//! and returns one rewritten value + one [`MappedAxes`] per result.

use evoxc_core::attr::AttrValue;
use evoxc_core::dim::Dim;
use evoxc_core::dtype::DType;
use evoxc_core::error::{Error, Result};
use evoxc_core::graph::ValueId;

use super::common::{fold_product, reshape_to_broadcast, row_rank, shift_axes};
use super::MappedAxes;
use crate::emit::{RuleCtx, RuleOp};
use crate::Registry;

/// `_pointwise_batching`: align every operand to
/// `own batch dims + (1,)*pad + per-row shape` so per-row dims line up on the
/// right (numpy trailing alignment), then rebuild the op.
///
/// Result mappedness = the longest operand batch-dims tuple (numpy broadcast
/// keeps the shared batch prefix).
pub fn pointwise_rule(
    ctx: &mut RuleCtx<'_>,
    op: &RuleOp,
    axes: &[MappedAxes],
) -> Result<(Vec<ValueId>, Vec<MappedAxes>)> {
    let counts: Vec<usize> = axes.iter().map(|axes| axes.count()).collect();
    let mapped_count = counts.iter().copied().max().unwrap_or(0);
    let row_ranks: Vec<usize> = op
        .operands
        .iter()
        .zip(&counts)
        .map(|(value, count)| Ok(row_rank(&ctx.ty(*value)?, *count)))
        .collect::<Result<Vec<_>>>()?;
    let max_row_rank = row_ranks.iter().copied().max().unwrap_or(0);
    let mut values: Vec<ValueId> = Vec::with_capacity(op.operands.len());
    for ((value, count), rank) in op.operands.iter().zip(&counts).zip(&row_ranks) {
        let _ = rank;
        values.push(reshape_to_broadcast(ctx, *value, *count, max_row_rank)?);
    }
    let value = if op.name == "broadcast" {
        // Mapped dims are preserved: the fresh target is the operand's batch
        // dims followed by the ORIGINAL broadcast target.
        let original = op.shape_attr("shape")?.to_vec();
        let mut target: Vec<Dim> = ctx.shape(values[0])?[..mapped_count].to_vec();
        target.extend(original);
        ctx.broadcast(values[0], &target)?
    } else {
        ctx.op1(&op.name, &values, &op.attributes)?
    };
    Ok((vec![value], vec![MappedAxes::new(mapped_count)]))
}

/// `_reduction_batching`: shift the reduced axes by the operand's mapped
/// count; the result keeps the operand's [`MappedAxes`].
///
/// `argmax`/`argmin` live here too because they are axis-based rather than
/// pointwise: with `axis = None` on a mapped operand the per-row dims are
/// flattened through a `reshape` to `prod(dims)`, the LAST axis is arg-reduced
/// and (with `keepdims`) the result is reshaped back to one `1` per original
/// dim.
pub fn reduction_rule(
    ctx: &mut RuleCtx<'_>,
    op: &RuleOp,
    axes: &[MappedAxes],
) -> Result<(Vec<ValueId>, Vec<MappedAxes>)> {
    let mapped = axes.first().copied().unwrap_or(MappedAxes::UNMAPPED);
    let count = mapped.count();
    let value = match op.operands.first() {
        Some(value) => *value,
        None => {
            return Err(Error::transform(format!(
                "vectorize: batching rule for op '{}' received no operand",
                op.name
            )))
        }
    };
    if op.name == "argmax" || op.name == "argmin" {
        let axis = op.opt_int("axis")?;
        let keepdims = keepdims_attr(op)?;
        let out = if axis.is_none() && count > 0 {
            // Flatten-over-all per batch row:
            // (batch..., *dims) → (batch..., prod(dims)), arg-reduce the last axis.
            let old_shape = op.original_operand(0)?.shape.clone();
            let flat = fold_product(&old_shape).ok_or_else(|| {
                Error::transform(format!(
                    "vectorize: cannot batch '{}' with axis=None over dynamic dims (v1 gap)",
                    op.name
                ))
            })?;
            let batched = ctx.shape(value)?;
            let mut target: Vec<Dim> = batched[..count.min(batched.len())].to_vec();
            target.push(flat);
            let flattened = ctx.reshape(value, &target)?;
            let reduced = arg_reduce(ctx, &op.name, flattened, Some(count as i64), false)?;
            if keepdims {
                let mut target: Vec<Dim> = batched[..count.min(batched.len())].to_vec();
                target.extend(std::iter::repeat_n(Dim::Int(1), old_shape.len()));
                ctx.reshape(reduced, &target)?
            } else {
                reduced
            }
        } else {
            let new_axis = axis.map(|axis| axis + count as i64);
            arg_reduce(ctx, &op.name, value, new_axis, keepdims)?
        };
        return Ok((vec![out], vec![mapped]));
    }
    let keepdims = keepdims_attr(op)?;
    let reduced = op.opt_ints("axes")?.unwrap_or(&[]);
    let old_rank = op.original_operand(0)?.rank();
    if !reduced.is_empty() {
        let new_axes = shift_axes(reduced, count);
        let out = reduce(ctx, &op.name, value, &new_axes, keepdims)?;
        return Ok((vec![out], vec![mapped]));
    }
    if old_rank == 0 && count > 0 {
        // Reduce-all over a mapped scalar input: identity per row, with the
        // reduction's dtype rule applied.
        let kind = op.str("reduce_op")?;
        let dtype = ctx.dtype(value)?;
        let out = if kind == "mean" && (dtype.is_bool() || dtype.is_int()) {
            ctx.cast(value, DType::Float64)?
        } else if (kind == "sum" || kind == "prod") && dtype.is_bool() {
            ctx.cast(value, DType::Int64)?
        } else {
            value
        };
        return Ok((vec![out], vec![mapped]));
    }
    // Empty axes = ALL (unvectorized) axes: reduce over the per-row axes only
    // (the mapped dims are excluded).
    let per_row: Vec<i64> = (count as i64..(count + old_rank) as i64).collect();
    let out = reduce(ctx, &op.name, value, &per_row, keepdims)?;
    Ok((vec![out], vec![mapped]))
}

/// `keepdims` with the schema default (`false`) applied to a missing/`null`
/// attribute; a present attribute of the wrong type still errors.
fn keepdims_attr(op: &RuleOp) -> Result<bool> {
    match op.attributes.get("keepdims") {
        None | Some(AttrValue::Null) => Ok(false),
        Some(_) => op.bool("keepdims"),
    }
}

/// Rebuild one `reduce_*` op (the exact IR names; the bare `sum`/`max`/`min`/
/// `mean`/`prod` sugar is not an IR op).
fn reduce(
    ctx: &mut RuleCtx<'_>,
    name: &str,
    value: ValueId,
    axes: &[i64],
    keepdims: bool,
) -> Result<ValueId> {
    match name {
        "reduce_sum" => ctx.reduce_sum(value, axes, keepdims),
        "reduce_prod" => ctx.reduce_prod(value, axes, keepdims),
        "reduce_mean" => ctx.reduce_mean(value, axes, keepdims),
        "reduce_max" => ctx.reduce_max(value, axes, keepdims),
        "reduce_min" => ctx.reduce_min(value, axes, keepdims),
        other => Err(Error::transform(format!(
            "vectorize: no reduction batching rule for op '{other}'"
        ))),
    }
}

/// Rebuild one `argmax`/`argmin` op.
fn arg_reduce(
    ctx: &mut RuleCtx<'_>,
    name: &str,
    value: ValueId,
    axis: Option<i64>,
    keepdims: bool,
) -> Result<ValueId> {
    match name {
        "argmax" => ctx.argmax(value, axis, keepdims),
        "argmin" => ctx.argmin(value, axis, keepdims),
        other => Err(Error::transform(format!(
            "vectorize: no arg-reduction batching rule for op '{other}'"
        ))),
    }
}

/// Register the elementwise batching rules (pointwise, the non-differentiable
/// pointwise ops and the reduction family).
pub fn register_rules(registry: &mut Registry) {
    for name in POINTWISE_OPS.iter().chain(NONDIFF_POINTWISE_OPS) {
        registry.batching.register(name, pointwise_rule);
    }
    for name in REDUCTION_OPS.iter().chain(ARG_REDUCTION_OPS) {
        registry.batching.register(name, reduction_rule);
    }
}

/// Every op `_pointwise_batching` covers (the `ELEMENTWISE_OPS` categories of
/// the Python registry plus the aliases the frontend emits).
pub const POINTWISE_OPS: &[&str] = &[
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
    "remainder",
    "maximum",
    "minimum",
    "negate",
    "abs",
    "sign",
    "exp",
    "log",
    "log1p",
    "sqrt",
    "square",
    "sin",
    "cos",
    "tan",
    "tanh",
    "sigmoid",
    "relu",
    "erf",
    "gelu",
    "cast",
    "select",
    "broadcast",
    "stop_gradient",
];

/// `NONDIFFERENTIABLE_OUTPUT_OPS` minus `argmax`/`argmin`: these batch exactly
/// like the pointwise ops (their results cannot backpropagate, which is the AD
/// rules' concern, not batching's).
pub const NONDIFF_POINTWISE_OPS: &[&str] = &[
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "logical_and",
    "logical_or",
    "logical_not",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
];

/// The reduction family, by exact IR op name (the bare `sum`/`max`/`min`/
/// `mean`/`prod` frontend aliases are compositions, not IR ops).
pub const REDUCTION_OPS: &[&str] = &[
    "reduce_sum",
    "reduce_prod",
    "reduce_mean",
    "reduce_max",
    "reduce_min",
];

/// `argmax`/`argmin`: axis-based, batched by [`reduction_rule`].
pub const ARG_REDUCTION_OPS: &[&str] = &["argmax", "argmin"];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::dims_i64;
    use evoxc_core::attr::Attrs;
    use crate::Builder;
    use evoxc_core::dtype::DType;

    fn f32(shape: &[i64]) -> evoxc_core::value_type::ValueType {
        evoxc_core::value_type::ValueType::new(DType::parse("float32").unwrap(), dims_i64(shape))
    }

    fn bool_ty(shape: &[i64]) -> evoxc_core::value_type::ValueType {
        evoxc_core::value_type::ValueType::new(DType::Bool, dims_i64(shape))
    }

    fn batched_ty(dtype: DType, shape: &[i64]) -> evoxc_core::value_type::ValueType {
        let mut dims = vec![Dim::named("batch")];
        dims.extend(dims_i64(shape));
        evoxc_core::value_type::ValueType::new(dtype, dims)
    }

    /// One single-operand reduction-category source op: the original
    /// (unbatched) type plus the rewritten batched operand.
    fn unary_source(
        name: &str,
        attributes: Attrs,
        original: evoxc_core::value_type::ValueType,
        operand: ValueId,
        operand_type: evoxc_core::value_type::ValueType,
    ) -> RuleOp {
        RuleOp {
            name: name.into(),
            attributes,
            original_operand_types: vec![original.clone()],
            original_result_types: vec![original],
            operands: vec![operand],
            operand_types: vec![operand_type],
            results: Vec::new(),
            result_types: Vec::new(),
            location: None,
        }
    }

    fn reduce_attrs(axes: &[i64], keepdims: bool, kind: &str) -> Attrs {
        Attrs::from_pairs([
            ("axes", AttrValue::Ints(axes.to_vec())),
            ("keepdims", AttrValue::Bool(keepdims)),
            ("reduce_op", AttrValue::Str(kind.to_string())),
        ])
    }

    fn arg_attrs(axis: Option<i64>, keepdims: bool) -> Attrs {
        let mut attrs = Attrs::from_pairs([("keepdims", AttrValue::Bool(keepdims))]);
        if let Some(axis) = axis {
            attrs.insert("axis", AttrValue::Int(axis));
        }
        attrs
    }

    /// The names/attributes of every op the rule emitted, in order.
    fn emitted(ctx: &RuleCtx<'_>) -> Vec<(String, Attrs)> {
        ctx.builder
            .program()
            .ops
            .iter()
            .map(|(_, op)| (op.name.clone(), op.attributes.clone()))
            .collect()
    }

    #[test]
    fn pointwise_aligns_unmapped_operands_and_prepends_the_batch_axis() {
        let mut builder = Builder::new("main");
        builder
            .open_function(
                "main",
                &[
                    evoxc_core::value_type::ValueType::new(
                        DType::parse("float32").unwrap(),
                        vec![Dim::named("batch"), Dim::Int(3)],
                    ),
                    f32(&[3]),
                ],
            )
            .unwrap();
        let args = builder.block_arguments().unwrap();
        // the op's source form (unbatched types) — the rule reads attributes
        let source = RuleOp {
            name: "add".into(),
            attributes: Attrs::new(),
            original_operand_types: vec![f32(&[3]), f32(&[3])],
            original_result_types: vec![f32(&[3])],
            operands: args.clone(),
            operand_types: vec![f32(&[3]), f32(&[3])],
            results: Vec::new(),
            result_types: Vec::new(),
            location: None,
        };
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let (values, axes) = pointwise_rule(
            &mut ctx,
            &source,
            &[MappedAxes::new(1), MappedAxes::UNMAPPED],
        )
        .unwrap();
        assert_eq!(axes, vec![MappedAxes::new(1)]);
        assert_eq!(
            ctx.shape(values[0]).unwrap(),
            vec![Dim::named("batch"), Dim::Int(3)]
        );
        let names: Vec<String> = ctx
            .builder
            .program()
            .ops
            .iter()
            .map(|(_, op)| op.name.clone())
            .collect();
        // (batch, 3) against (3): the row ranks already match, so the pointwise
        // rule aligns nothing and emits only the rebuilt op.
        assert_eq!(names, vec!["add"]);
    }

    #[test]
    fn reduction_shifts_axes_with_and_without_keepdims() {
        for keepdims in [true, false] {
            let mut builder = Builder::new("main");
            let batched = batched_ty(DType::Float32, &[3, 4]);
            builder
                .open_function("main", std::slice::from_ref(&batched))
                .unwrap();
            let args = builder.block_arguments().unwrap();
            // reduce_sum(x (3, 4), axes=[0], keepdims=?) — batched operand
            // (batch, 3, 4): axis 0 shifts to axis 1 and the batch axis is kept.
            let source = unary_source(
                "reduce_sum",
                reduce_attrs(&[0], keepdims, "sum"),
                f32(&[3, 4]),
                args[0],
                batched,
            );
            let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
            let (values, axes) =
                reduction_rule(&mut ctx, &source, &[MappedAxes::new(1)]).unwrap();
            // the result keeps the operand's mapped axes
            assert_eq!(axes, vec![MappedAxes::new(1)]);
            let expected = if keepdims {
                vec![Dim::named("batch"), Dim::Int(1), Dim::Int(4)]
            } else {
                vec![Dim::named("batch"), Dim::Int(4)]
            };
            assert_eq!(ctx.shape(values[0]).unwrap(), expected);
            let ops = emitted(&ctx);
            assert_eq!(ops.len(), 1);
            assert_eq!(ops[0].0, "reduce_sum");
            assert_eq!(ops[0].1.ints("axes").unwrap().to_vec(), vec![1]);
            assert_eq!(ops[0].1.bool("keepdims").unwrap(), keepdims);
        }
    }

    #[test]
    fn reduction_with_axis_none_reduces_every_per_row_axis() {
        // The frontend normalizes axis=None to ALL axes; on the batched operand
        // that means "everything except the mapped batch axes".
        for keepdims in [true, false] {
            let mut builder = Builder::new("main");
            let batched = batched_ty(DType::Float32, &[3, 4]);
            builder
                .open_function("main", std::slice::from_ref(&batched))
                .unwrap();
            let args = builder.block_arguments().unwrap();
            let source = unary_source(
                "reduce_mean",
                reduce_attrs(&[0, 1], keepdims, "mean"),
                f32(&[3, 4]),
                args[0],
                batched,
            );
            let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
            let (values, axes) =
                reduction_rule(&mut ctx, &source, &[MappedAxes::new(1)]).unwrap();
            assert_eq!(axes, vec![MappedAxes::new(1)]);
            let expected = if keepdims {
                vec![Dim::named("batch"), Dim::Int(1), Dim::Int(1)]
            } else {
                vec![Dim::named("batch")]
            };
            assert_eq!(ctx.shape(values[0]).unwrap(), expected);
            let ops = emitted(&ctx);
            assert_eq!(ops[0].0, "reduce_mean");
            assert_eq!(ops[0].1.ints("axes").unwrap().to_vec(), vec![1, 2]);
            assert_eq!(ops[0].1.bool("keepdims").unwrap(), keepdims);
        }
    }

    #[test]
    fn argmax_with_axis_none_flattens_per_row_and_restores_keepdims() {
        let mut builder = Builder::new("main");
        let batched = batched_ty(DType::Float32, &[3, 4]);
        builder
            .open_function("main", std::slice::from_ref(&batched))
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let source = unary_source(
            "argmax",
            arg_attrs(None, true),
            f32(&[3, 4]),
            args[0],
            batched,
        );
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let (values, axes) = reduction_rule(&mut ctx, &source, &[MappedAxes::new(1)]).unwrap();
        assert_eq!(axes, vec![MappedAxes::new(1)]);
        // (batch, 3, 4) -> (batch, 12) -> argmax axis 1 -> (batch,) ->
        // (batch, 1, 1) with keepdims.
        assert_eq!(
            ctx.shape(values[0]).unwrap(),
            vec![Dim::named("batch"), Dim::Int(1), Dim::Int(1)]
        );
        assert_eq!(
            ctx.dtype(values[0]).unwrap(),
            DType::parse("int64").unwrap()
        );
        let ops = emitted(&ctx);
        let names: Vec<&str> = ops.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, vec!["reshape", "argmax", "reshape"]);
        match ops[0].1.get("shape") {
            Some(AttrValue::Shape(shape)) => assert_eq!(
                shape,
                &vec![Dim::named("batch"), Dim::Int(12)]
            ),
            other => panic!("reshape target must be the flattened per-row shape, got {other:?}"),
        }
        assert_eq!(ops[1].1.opt_i64("axis").unwrap(), Some(1));
        assert!(!ops[1].1.bool("keepdims").unwrap());
        match ops[2].1.get("shape") {
            Some(AttrValue::Shape(shape)) => assert_eq!(
                shape,
                &vec![Dim::named("batch"), Dim::Int(1), Dim::Int(1)]
            ),
            other => panic!("keepdims reshape target missing, got {other:?}"),
        }
    }

    #[test]
    fn argmax_with_an_axis_shifts_it_and_keeps_the_operand_axes() {
        let mut builder = Builder::new("main");
        let batched = batched_ty(DType::Float32, &[3, 4]);
        builder
            .open_function("main", std::slice::from_ref(&batched))
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let source = unary_source(
            "argmin",
            arg_attrs(Some(1), false),
            f32(&[3, 4]),
            args[0],
            batched,
        );
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let (values, axes) = reduction_rule(&mut ctx, &source, &[MappedAxes::new(1)]).unwrap();
        assert_eq!(axes, vec![MappedAxes::new(1)]);
        assert_eq!(
            ctx.shape(values[0]).unwrap(),
            vec![Dim::named("batch"), Dim::Int(3)]
        );
        let ops = emitted(&ctx);
        // axis 1 shifts to axis 2: one rebuilt op, no reshape.
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].0, "argmin");
        assert_eq!(ops[0].1.opt_i64("axis").unwrap(), Some(2));
    }

    #[test]
    fn nondifferentiable_output_ops_register_the_pointwise_rule() {
        let mut registry = Registry::default();
        crate::batching::register_builtin(&mut registry);
        for name in NONDIFF_POINTWISE_OPS {
            assert!(
                registry.batching.get(name).is_some(),
                "'{name}' must batch pointwise"
            );
            // the documented ELEMENTWISE_OPS set stays the pointwise constant
            assert!(!POINTWISE_OPS.contains(name));
        }
        assert_eq!(POINTWISE_OPS.len(), 28);
        assert_eq!(NONDIFF_POINTWISE_OPS.len(), 12);

        // the rule handles a comparison unchanged (bool output, no new magic)
        let mut builder = Builder::new("main");
        let batched = batched_ty(DType::Float32, &[3]);
        builder
            .open_function("main", &[batched.clone(), f32(&[3])])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let source = RuleOp {
            name: "equal".into(),
            attributes: Attrs::new(),
            original_operand_types: vec![f32(&[3]), f32(&[3])],
            original_result_types: vec![bool_ty(&[3])],
            operands: args.clone(),
            operand_types: vec![batched.clone(), f32(&[3])],
            results: Vec::new(),
            result_types: Vec::new(),
            location: None,
        };
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let (values, axes) = pointwise_rule(
            &mut ctx,
            &source,
            &[MappedAxes::new(1), MappedAxes::UNMAPPED],
        )
        .unwrap();
        assert_eq!(axes, vec![MappedAxes::new(1)]);
        assert_eq!(
            ctx.shape(values[0]).unwrap(),
            vec![Dim::named("batch"), Dim::Int(3)]
        );
        assert_eq!(
            ctx.dtype(values[0]).unwrap(),
            DType::parse("bool").unwrap()
        );
        let ops = emitted(&ctx);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].0, "equal");
    }

    #[test]
    fn argmax_axis_none_over_dynamic_dims_defers_explicitly() {
        // fold_product is the flatten gate: a runtime-dynamic dim cannot fold.
        assert_eq!(fold_product(&[Dim::Int(2), Dim::Dynamic]), None);
        let mut builder = Builder::new("main");
        let batched = evoxc_core::value_type::ValueType::new(
            DType::Float32,
            vec![Dim::named("batch"), Dim::Dynamic, Dim::Int(4)],
        );
        builder
            .open_function("main", std::slice::from_ref(&batched))
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let original = evoxc_core::value_type::ValueType::new(
            DType::Float32,
            vec![Dim::Dynamic, Dim::Int(4)],
        );
        let source = unary_source("argmax", arg_attrs(None, false), original, args[0], batched);
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let err = reduction_rule(&mut ctx, &source, &[MappedAxes::new(1)]).unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        assert!(err
            .message
            .contains("cannot batch 'argmax' with axis=None over dynamic dims"));
        // nothing was emitted before the deferral
        assert!(emitted(&ctx).is_empty());
    }

    #[test]
    fn the_builtin_registry_registers_the_full_batching_key_set() {
        let mut registry = Registry::default();
        crate::batching::register_builtin(&mut registry);
        for name in POINTWISE_OPS
            .iter()
            .chain(NONDIFF_POINTWISE_OPS)
            .chain(REDUCTION_OPS)
            .chain(ARG_REDUCTION_OPS)
        {
            assert!(
                registry.batching.contains(name),
                "missing batching rule for '{name}'"
            );
        }
        // 28 pointwise + 12 nondiff pointwise + 5 reduce + 2 arg-reduce
        // + 8 structural + 6 linalg
        assert_eq!(registry.batching.len(), 61);
        // the bare reduction aliases are frontend compositions, not IR ops
        assert!(!registry.batching.contains("sum"));
        assert!(!registry.batching.contains("max"));
    }

    #[test]
    fn every_sparse_op_and_the_namespaced_keys_reach_the_extension_bridge() {
        // `dense_dot_sparse` is one of the 16 sparse ops but carries no
        // `sparse_` prefix, so the gate must name it explicitly.
        assert!(crate::batching::extension_eligible("dense_dot_sparse"));
        assert!(crate::batching::extension_eligible("sparse_add"));
        assert!(crate::batching::extension_eligible("sparse_matmul"));
        assert!(crate::batching::extension_eligible("block:mine"));
        assert!(crate::batching::extension_eligible("external:kernel"));
        // a builtin op and an unknown name never reach the host
        assert!(!crate::batching::extension_eligible("reduce_sum"));
        assert!(!crate::batching::extension_eligible("not_a_real_op"));
    }
}
