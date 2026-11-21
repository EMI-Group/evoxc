//! Elementwise batching rules: per-element ops and reductions.
//!
//! Port of `_pointwise_batching` / `_reduction_batching` from
//! `evoxc/transforms/rules.py`.  Every rule receives the op's *rewritten*
//! operands (values of the output module) plus one [`MappedAxes`] per operand,
//! and returns one rewritten value + one [`MappedAxes`] per result.

use evoxc_core::dim::Dim;
use evoxc_core::error::Result;
use evoxc_core::graph::ValueId;

use super::common::{reshape_to_broadcast, row_rank};
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

/// Register the elementwise batching rules.
pub fn register_rules(registry: &mut Registry) {
    for name in POINTWISE_OPS {
        registry.batching.register(name, pointwise_rule);
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
        assert_eq!(names, vec!["reshape", "add"]);
    }
}
