//! Linalg derivative rules: `dot`, `conv`, `tril`/`triu`, `cumsum`, `solve`.
//!
//! Port of `_dot_conv_jvp`, `_dot_conv_vjp`, `_triangular_jvp`,
//! `_triangular_vjp`, `_cumsum_jvp`, `_cumsum_vjp`, `_solve_jvp` and
//! `_solve_vjp` from `evoxc/transforms/rules.py`.  `conv` VJP is an explicit v1
//! deferral (the IR has no transposed-convolution op).
//!
//! Rules read the *rewritten* values: `op.operands` are the primals of the
//! output module and `op.results` its rebuilt results.

use evoxc_core::attr::AttrValue;
use evoxc_core::dim::Dim;
use evoxc_core::error::{Error, Result};
use evoxc_core::graph::ValueId;

use super::common::{reduce_leading, sum_terms, zeros_of};
use super::Dual;
use crate::emit::{ConvParams, RuleCtx, RuleOp};
use crate::Registry;

// ---- small guarded accessors (explicit errors, never a panic) --------------

/// Operand `index` of `op` (the rewritten value).
fn operand(op: &RuleOp, index: usize) -> Result<ValueId> {
    op.operands.get(index).copied().ok_or_else(|| {
        Error::transform(format!(
            "derivative rule for op '{}': no operand {index} (the op has {})",
            op.name,
            op.operands.len()
        ))
    })
}

/// Result `index` of `op` (the rebuilt value).
fn result(op: &RuleOp, index: usize) -> Result<ValueId> {
    op.results.get(index).copied().ok_or_else(|| {
        Error::transform(format!(
            "derivative rule for op '{}': no result {index} (the op has {})",
            op.name,
            op.results.len()
        ))
    })
}

/// Tangent/cotangent slot `index`.
fn dual_at(duals: &[Dual], index: usize, op: &RuleOp, kind: &str) -> Result<Dual> {
    duals.get(index).copied().ok_or_else(|| {
        Error::transform(format!(
            "{kind} rule for op '{}': no slot {index} (received {})",
            op.name,
            duals.len()
        ))
    })
}

/// An optional bool attribute with its schema default.
fn opt_bool(op: &RuleOp, name: &str, default: bool) -> Result<bool> {
    match op.attributes.get(name) {
        None | Some(AttrValue::Null) => Ok(default),
        Some(_) => op.bool(name),
    }
}

/// The permutation that swaps the last two axes and keeps the leading ones
/// (`tuple(range(rank - 2)) + (rank - 1, rank - 2)` in the Python rules).
fn swap_last_two(rank: usize) -> Vec<i64> {
    let rank = rank as i64;
    let mut permutation: Vec<i64> = (0..rank - 2).collect();
    permutation.push(rank - 1);
    permutation.push(rank - 2);
    permutation
}

// ---- dot / conv ------------------------------------------------------------

/// Dot/conv JVP: the two product terms (`dot(a, tb) + dot(ta, b)` /
/// `conv(a, tw) + conv(ta, w)`), zero tangents skipped.
pub fn dot_conv_jvp(ctx: &mut RuleCtx<'_>, op: &RuleOp, tangents: &[Dual]) -> Result<Vec<Dual>> {
    let ta = dual_at(tangents, 0, op, "jvp")?.value();
    let tb = dual_at(tangents, 1, op, "jvp")?.value();
    let a = operand(op, 0)?;
    let b = operand(op, 1)?;
    let mut terms = Vec::new();
    if op.name == "dot" {
        if let Some(tb) = tb {
            terms.push(ctx.dot(a, tb)?);
        }
        if let Some(ta) = ta {
            terms.push(ctx.dot(ta, b)?);
        }
    } else {
        let params = ConvParams::from_op(op)?;
        if let Some(tb) = tb {
            terms.push(ctx.conv(a, tb, &params)?);
        }
        if let Some(ta) = ta {
            terms.push(ctx.conv(ta, b, &params)?);
        }
    }
    if terms.is_empty() {
        return Ok(vec![Dual::ZERO]);
    }
    Ok(vec![Dual::Value(sum_terms(ctx, &terms)?)])
}

/// Dot VJP: `g_a = dot(ct, bᵀ)`, `g_b = dot(aᵀ, ct)` (the last two axes are
/// swapped), with broadcast-only leading batch dims summed away.
fn dot_vjp(ctx: &mut RuleCtx<'_>, op: &RuleOp, cotangents: &[Dual]) -> Result<Vec<Dual>> {
    let Some(ct) = dual_at(cotangents, 0, op, "vjp")?.value() else {
        return Ok(vec![Dual::ZERO; op.operands.len()]);
    };
    let a = operand(op, 0)?;
    let b = operand(op, 1)?;
    let b_t = ctx.transpose(b, Some(&swap_last_two(ctx.rank(b)?)))?;
    let grad_a = ctx.dot(ct, b_t)?;
    let a_t = ctx.transpose(a, Some(&swap_last_two(ctx.rank(a)?)))?;
    let grad_b = ctx.dot(a_t, ct)?;
    Ok(vec![
        Dual::Value(reduce_leading(ctx, grad_a, a)?),
        Dual::Value(reduce_leading(ctx, grad_b, b)?),
    ])
}

/// Dot/conv VJP: a transposed dot per primal operand; `conv` has NO transposed
/// form in the IR → explicit v1 deferral.
pub fn dot_conv_vjp(ctx: &mut RuleCtx<'_>, op: &RuleOp, cotangents: &[Dual]) -> Result<Vec<Dual>> {
    if op.name == "conv" {
        return Err(Error::transform(
            "grad/vjp: no VJP rule for op 'conv' — the IR has no transposed-convolution op \
             (v1 gap)",
        ));
    }
    dot_vjp(ctx, op, cotangents)
}

// ---- tril / triu -----------------------------------------------------------

/// The mask op named by `op.name` applied to `value` with the same `k`.
fn triangular(ctx: &mut RuleCtx<'_>, op: &RuleOp, value: ValueId) -> Result<ValueId> {
    let k = op.opt_int("k")?.unwrap_or(0);
    match op.name.as_str() {
        "tril" => ctx.tril(value, k),
        "triu" => ctx.triu(value, k),
        other => Err(Error::transform(format!(
            "derivative rule: no triangular mask for op '{other}'"
        ))),
    }
}

/// tril/triu JVP: the mask is linear — apply it to the tangent.
pub fn triangular_jvp(ctx: &mut RuleCtx<'_>, op: &RuleOp, tangents: &[Dual]) -> Result<Vec<Dual>> {
    let Some(tangent) = dual_at(tangents, 0, op, "jvp")?.value() else {
        return Ok(vec![Dual::ZERO]);
    };
    Ok(vec![Dual::Value(triangular(ctx, op, tangent)?)])
}

/// tril/triu VJP: the mask is self-adjoint — apply it to the cotangent.
pub fn triangular_vjp(
    ctx: &mut RuleCtx<'_>,
    op: &RuleOp,
    cotangents: &[Dual],
) -> Result<Vec<Dual>> {
    let Some(cotangent) = dual_at(cotangents, 0, op, "vjp")?.value() else {
        return Ok(vec![Dual::ZERO]);
    };
    Ok(vec![Dual::Value(triangular(ctx, op, cotangent)?)])
}

// ---- cumsum ----------------------------------------------------------------

/// cumsum JVP: linear — scan the tangent in the same direction.
pub fn cumsum_jvp(ctx: &mut RuleCtx<'_>, op: &RuleOp, tangents: &[Dual]) -> Result<Vec<Dual>> {
    let Some(tangent) = dual_at(tangents, 0, op, "jvp")?.value() else {
        return Ok(vec![Dual::ZERO]);
    };
    let axis = op.int("axis")?;
    let reverse = opt_bool(op, "reverse", false)?;
    Ok(vec![Dual::Value(ctx.cumsum(tangent, axis, reverse)?)])
}

/// cumsum VJP: the adjoint scans in the opposite direction.
pub fn cumsum_vjp(ctx: &mut RuleCtx<'_>, op: &RuleOp, cotangents: &[Dual]) -> Result<Vec<Dual>> {
    let Some(cotangent) = dual_at(cotangents, 0, op, "vjp")?.value() else {
        return Ok(vec![Dual::ZERO]);
    };
    let axis = op.int("axis")?;
    let reverse = !opt_bool(op, "reverse", false)?;
    Ok(vec![Dual::Value(ctx.cumsum(cotangent, axis, reverse)?)])
}

// ---- solve -----------------------------------------------------------------

/// Reject `solve` with `left_side = false` (a v1 gap in both sweeps).
fn require_left_side(op: &RuleOp, kind: &str) -> Result<()> {
    if !opt_bool(op, "left_side", true)? {
        return Err(Error::transform(format!(
            "{kind}: cannot differentiate op 'solve' with left_side=False (v1 gap)"
        )));
    }
    Ok(())
}

/// The first dim of `value`'s shape (the `n` of a vector right-hand side).
fn first_dim(ctx: &RuleCtx<'_>, value: ValueId) -> Result<Dim> {
    ctx.shape(value)?.into_iter().next().ok_or_else(|| {
        Error::transform("derivative rule for op 'solve': the right-hand side must have rank >= 1")
    })
}

/// Solve JVP: `a·dx + da·x = db ⇒ dx = solve(a, db − da·x)`; a vector `b`
/// (rank 1) goes through an `(n, 1)` reshape (matvec via `dot`).
pub fn solve_jvp(ctx: &mut RuleCtx<'_>, op: &RuleOp, tangents: &[Dual]) -> Result<Vec<Dual>> {
    require_left_side(op, "jvp")?;
    let ta = dual_at(tangents, 0, op, "jvp")?.value();
    let tb = dual_at(tangents, 1, op, "jvp")?.value();
    if ta.is_none() && tb.is_none() {
        return Ok(vec![Dual::ZERO]);
    }
    let a = operand(op, 0)?;
    let b = operand(op, 1)?;
    let x = result(op, 0)?;
    if ctx.rank(x)? == 1 {
        let n = first_dim(ctx, b)?;
        let matrix = [n.clone(), Dim::Int(1)];
        let x_m = ctx.reshape(x, &matrix)?;
        let diff = match tb {
            Some(tb) => ctx.reshape(tb, &matrix)?,
            None => {
                let zeros = zeros_of(ctx, b)?;
                ctx.reshape(zeros, &matrix)?
            }
        };
        let diff = match ta {
            Some(ta) => {
                let product = ctx.dot(ta, x_m)?;
                ctx.binary("subtract", diff, product)?
            }
            None => diff,
        };
        let dx = ctx.solve(a, diff)?;
        return Ok(vec![Dual::Value(ctx.reshape(dx, &[n])?)]);
    }
    let diff = match tb {
        Some(tb) => tb,
        None => zeros_of(ctx, b)?,
    };
    let diff = match ta {
        Some(ta) => {
            let product = ctx.dot(ta, x)?;
            ctx.binary("subtract", diff, product)?
        }
        None => diff,
    };
    Ok(vec![Dual::Value(ctx.solve(a, diff)?)])
}

/// Solve VJP: `g_b = solve(aᵀ, ct)`, `g_a = −g_b·xᵀ` (an outer product via a
/// reshaped multiply for a vector `x`, `dot` for a matrix `x`); broadcast-only
/// leading batch dims are summed away.
pub fn solve_vjp(ctx: &mut RuleCtx<'_>, op: &RuleOp, cotangents: &[Dual]) -> Result<Vec<Dual>> {
    require_left_side(op, "vjp")?;
    let Some(ct) = dual_at(cotangents, 0, op, "vjp")?.value() else {
        return Ok(vec![Dual::ZERO; op.operands.len()]);
    };
    let a = operand(op, 0)?;
    let b = operand(op, 1)?;
    let x = result(op, 0)?;
    let a_t = ctx.transpose(a, Some(&swap_last_two(ctx.rank(a)?)))?;
    let grad_b = ctx.solve(a_t, ct)?;
    let grad_a = if ctx.rank(x)? == 1 {
        let n = first_dim(ctx, x)?;
        let column = ctx.reshape(grad_b, &[n.clone(), Dim::Int(1)])?;
        let row = ctx.reshape(x, &[Dim::Int(1), n])?;
        let product = ctx.binary("multiply", column, row)?;
        ctx.unary("negate", product)?
    } else {
        let x_t = ctx.transpose(x, Some(&swap_last_two(ctx.rank(x)?)))?;
        let product = ctx.dot(grad_b, x_t)?;
        ctx.unary("negate", product)?
    };
    Ok(vec![
        Dual::Value(reduce_leading(ctx, grad_a, a)?),
        Dual::Value(reduce_leading(ctx, grad_b, b)?),
    ])
}

/// Every op this file rules (both sweeps).
pub const RULED_OPS: &[&str] = &["dot", "conv", "tril", "triu", "cumsum", "solve"];

/// Register the linalg JVP and VJP rules.
pub fn register_rules(registry: &mut Registry) {
    for name in ["dot", "conv"] {
        registry.jvp.register(name, dot_conv_jvp);
        registry.vjp.register(name, dot_conv_vjp);
    }
    for name in ["tril", "triu"] {
        registry.jvp.register(name, triangular_jvp);
        registry.vjp.register(name, triangular_vjp);
    }
    registry.jvp.register("cumsum", cumsum_jvp);
    registry.vjp.register("cumsum", cumsum_vjp);
    registry.jvp.register("solve", solve_jvp);
    registry.vjp.register("solve", solve_vjp);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::dims_i64;
    use crate::Builder;
    use evoxc_core::attr::Attrs;
    use evoxc_core::dtype::DType;
    use evoxc_core::value_type::ValueType;

    fn f32(shape: &[i64]) -> ValueType {
        ValueType::new(DType::parse("float32").unwrap(), dims_i64(shape))
    }

    /// A [`RuleOp`] over already-emitted values with explicit attributes.
    fn rule_op(
        builder: &Builder,
        name: &str,
        attrs: Attrs,
        operands: &[ValueId],
        results: &[ValueId],
    ) -> RuleOp {
        let program = builder.program();
        let types = |ids: &[ValueId]| -> Vec<ValueType> {
            ids.iter()
                .map(|id| program.value(*id).unwrap().ty.clone())
                .collect()
        };
        RuleOp {
            name: name.into(),
            attributes: attrs,
            original_operand_types: types(operands),
            original_result_types: types(results),
            operands: operands.to_vec(),
            operand_types: Vec::new(),
            results: results.to_vec(),
            result_types: Vec::new(),
            location: None,
        }
    }

    /// The op names the module emitted, in order.
    fn emitted(builder: &Builder) -> Vec<String> {
        builder
            .program()
            .ops
            .iter()
            .map(|(_, op)| op.name.clone())
            .collect()
    }

    fn one(dual: Dual) -> ValueId {
        dual.value().expect("rule returned a real value")
    }

    #[test]
    fn dot_jvp_sums_only_the_non_zero_terms() {
        // a (2, 3) · b (3, 4) -> (2, 4); only b carries a tangent.
        let mut builder = Builder::new("main");
        builder
            .open_function(
                "main",
                &[f32(&[2, 3]), f32(&[3, 4]), f32(&[2, 3]), f32(&[3, 4])],
            )
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let product = builder.binary("dot", args[0], args[1]).unwrap();
        let op = rule_op(&builder, "dot", Attrs::new(), &args[..2], &[product]);

        let mut ctx = RuleCtx::new(&mut builder);
        let out = dot_conv_jvp(&mut ctx, &op, &[Dual::ZERO, Dual::Value(args[3])]).unwrap();
        assert_eq!(out.len(), 1);
        let value = one(out[0]);
        assert_eq!(ctx.shape(value).unwrap(), dims_i64(&[2, 4]));
        // one product term: dot(a, tb) on top of the primal dot
        assert_eq!(emitted(ctx.builder), vec!["dot", "dot"]);

        let mut ctx = RuleCtx::new(&mut builder);
        let out = dot_conv_jvp(&mut ctx, &op, &[Dual::ZERO, Dual::ZERO]).unwrap();
        assert_eq!(out, vec![Dual::ZERO]);
    }

    #[test]
    fn dot_jvp_with_a_mapped_first_factor() {
        let mut builder = Builder::new("main");
        builder
            .open_function(
                "main",
                &[f32(&[2, 3]), f32(&[3, 4]), f32(&[2, 3]), f32(&[3, 4])],
            )
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let product = builder.binary("dot", args[0], args[1]).unwrap();
        let op = rule_op(&builder, "dot", Attrs::new(), &args[..2], &[product]);
        let mut ctx = RuleCtx::new(&mut builder);
        let out = dot_conv_jvp(&mut ctx, &op, &[Dual::Value(args[2]), Dual::ZERO]).unwrap();
        let value = one(out[0]);
        assert_eq!(ctx.shape(value).unwrap(), dims_i64(&[2, 4]));
        let names = emitted(ctx.builder);
        assert_eq!(names.iter().filter(|name| *name == "dot").count(), 2);
    }

    #[test]
    fn dot_vjp_shapes_for_a_two_dimensional_dot() {
        let mut builder = Builder::new("main");
        builder
            .open_function("main", &[f32(&[2, 3]), f32(&[3, 4]), f32(&[2, 4])])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let product = builder.binary("dot", args[0], args[1]).unwrap();
        let op = rule_op(&builder, "dot", Attrs::new(), &args[..2], &[product]);
        let mut ctx = RuleCtx::new(&mut builder);
        let out = dot_conv_vjp(&mut ctx, &op, &[Dual::Value(args[2])]).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(ctx.shape(one(out[0])).unwrap(), dims_i64(&[2, 3]));
        assert_eq!(ctx.shape(one(out[1])).unwrap(), dims_i64(&[3, 4]));

        let mut ctx = RuleCtx::new(&mut builder);
        let out = dot_conv_vjp(&mut ctx, &op, &[Dual::ZERO]).unwrap();
        assert_eq!(out, vec![Dual::ZERO, Dual::ZERO]);
    }

    #[test]
    fn dot_vjp_shapes_for_a_batched_dot_and_a_broadcast_lhs() {
        // batched: a (2, 3, 4) · b (2, 4, 5) -> ct (2, 3, 5)
        let mut builder = Builder::new("main");
        builder
            .open_function("main", &[f32(&[2, 3, 4]), f32(&[2, 4, 5]), f32(&[2, 3, 5])])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let product = builder.binary("dot", args[0], args[1]).unwrap();
        let op = rule_op(&builder, "dot", Attrs::new(), &args[..2], &[product]);
        let mut ctx = RuleCtx::new(&mut builder);
        let out = dot_conv_vjp(&mut ctx, &op, &[Dual::Value(args[2])]).unwrap();
        assert_eq!(ctx.shape(one(out[0])).unwrap(), dims_i64(&[2, 3, 4]));
        assert_eq!(ctx.shape(one(out[1])).unwrap(), dims_i64(&[2, 4, 5]));

        // broadcast: a (3, 4) · b (2, 4, 5) -> ct (2, 3, 5); g_a loses the
        // broadcast leading dim.
        let mut builder = Builder::new("main");
        builder
            .open_function("main", &[f32(&[3, 4]), f32(&[2, 4, 5]), f32(&[2, 3, 5])])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let product = builder.binary("dot", args[0], args[1]).unwrap();
        let op = rule_op(&builder, "dot", Attrs::new(), &args[..2], &[product]);
        let mut ctx = RuleCtx::new(&mut builder);
        let out = dot_conv_vjp(&mut ctx, &op, &[Dual::Value(args[2])]).unwrap();
        assert_eq!(ctx.shape(one(out[0])).unwrap(), dims_i64(&[3, 4]));
        assert_eq!(ctx.shape(one(out[1])).unwrap(), dims_i64(&[2, 4, 5]));
    }

    #[test]
    fn conv_vjp_is_an_explicit_v1_deferral() {
        let mut builder = Builder::new("main");
        builder
            .open_function(
                "main",
                &[
                    f32(&[1, 4, 8, 8]),
                    f32(&[4, 4, 3, 3]),
                    f32(&[4, 4, 3, 3]),
                    f32(&[1, 4, 6, 6]),
                ],
            )
            .unwrap();
        let args = builder.block_arguments().unwrap();
        // the optional `ints` attributes must be explicit (their schema
        // default is `null`, which default-filling does not accept)
        let attrs = Attrs::from_pairs([
            ("strides", AttrValue::Ints(vec![1, 1])),
            ("input_dilation", AttrValue::Ints(vec![1, 1])),
            ("kernel_dilation", AttrValue::Ints(vec![1, 1])),
        ]);
        let convolved = builder.emit_typed_one("conv", &args[..2], &attrs).unwrap();
        let op = rule_op(&builder, "conv", attrs, &args[..2], &[convolved]);
        let mut ctx = RuleCtx::new(&mut builder);
        let err = dot_conv_vjp(&mut ctx, &op, &[Dual::Value(args[3])]).unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        assert!(
            err.message.contains("no transposed-convolution op"),
            "{}",
            err.message
        );

        // the conv JVP stays implemented (a single tangent -> one conv term)
        let mut ctx = RuleCtx::new(&mut builder);
        let out = dot_conv_jvp(&mut ctx, &op, &[Dual::ZERO, Dual::Value(args[2])]).unwrap();
        assert_eq!(ctx.shape(one(out[0])).unwrap(), dims_i64(&[1, 4, 6, 6]));
    }

    #[test]
    fn tril_and_triu_keep_the_mask_parameter() {
        let mut builder = Builder::new("main");
        builder
            .open_function("main", &[f32(&[3, 4]), f32(&[3, 4]), f32(&[3, 4])])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let masked = builder
            .emit_typed_one(
                "tril",
                &[args[0]],
                &Attrs::from_pairs([("k", AttrValue::Int(-1))]),
            )
            .unwrap();
        let attrs = builder
            .program()
            .op(*builder.block_op_ids().unwrap().last().unwrap())
            .unwrap()
            .attributes
            .clone();
        let op = rule_op(&builder, "tril", attrs, &args[..1], &[masked]);
        let mut ctx = RuleCtx::new(&mut builder);
        let out = triangular_vjp(&mut ctx, &op, &[Dual::Value(args[1])]).unwrap();
        assert_eq!(ctx.shape(one(out[0])).unwrap(), dims_i64(&[3, 4]));
        let (_, last) = ctx.builder.program().ops.iter().last().unwrap();
        assert_eq!(last.name, "tril");
        assert_eq!(last.attributes.i64("k").unwrap(), -1);
    }

    #[test]
    fn cumsum_vjp_flips_the_scan_direction() {
        let attrs = Attrs::from_pairs([
            ("axis", AttrValue::Int(1)),
            ("reverse", AttrValue::Bool(true)),
        ]);
        let mut builder = Builder::new("main");
        builder
            .open_function("main", &[f32(&[2, 4]), f32(&[2, 4])])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let scanned = builder
            .emit_typed_one("cumsum", &[args[0]], &attrs)
            .unwrap();
        let op = rule_op(&builder, "cumsum", attrs, &args[..1], &[scanned]);

        let mut ctx = RuleCtx::new(&mut builder);
        let out = cumsum_vjp(&mut ctx, &op, &[Dual::Value(args[1])]).unwrap();
        assert_eq!(ctx.shape(one(out[0])).unwrap(), dims_i64(&[2, 4]));
        let (_, last) = ctx.builder.program().ops.iter().last().unwrap();
        assert_eq!(last.name, "cumsum");
        assert_eq!(last.attributes.i64("axis").unwrap(), 1);
        assert!(
            !last.attributes.bool("reverse").unwrap(),
            "reverse is flipped"
        );

        // the JVP scans in the primal's direction
        let mut ctx = RuleCtx::new(&mut builder);
        let out = cumsum_jvp(&mut ctx, &op, &[Dual::Value(args[1])]).unwrap();
        assert_eq!(ctx.shape(one(out[0])).unwrap(), dims_i64(&[2, 4]));
        let (_, last) = ctx.builder.program().ops.iter().last().unwrap();
        assert!(last.attributes.bool("reverse").unwrap());

        // a zero tangent/ cotangent short-circuits
        let mut ctx = RuleCtx::new(&mut builder);
        assert_eq!(
            cumsum_jvp(&mut ctx, &op, &[Dual::ZERO]).unwrap(),
            vec![Dual::ZERO]
        );
        assert_eq!(
            cumsum_vjp(&mut ctx, &op, &[Dual::ZERO]).unwrap(),
            vec![Dual::ZERO]
        );
    }

    #[test]
    fn solve_jvp_and_vjp_shapes_for_a_matrix_rhs() {
        let mut builder = Builder::new("main");
        builder
            .open_function(
                "main",
                &[
                    f32(&[3, 3]),
                    f32(&[3, 2]),
                    f32(&[3, 3]),
                    f32(&[3, 2]),
                    f32(&[3, 2]),
                ],
            )
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let solved = builder.binary("solve", args[0], args[1]).unwrap();
        let op = rule_op(&builder, "solve", Attrs::new(), &args[..2], &[solved]);

        let mut ctx = RuleCtx::new(&mut builder);
        let out = solve_jvp(&mut ctx, &op, &[Dual::Value(args[2]), Dual::Value(args[3])]).unwrap();
        assert_eq!(ctx.shape(one(out[0])).unwrap(), dims_i64(&[3, 2]));

        let mut ctx = RuleCtx::new(&mut builder);
        let out = solve_vjp(&mut ctx, &op, &[Dual::Value(args[4])]).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(ctx.shape(one(out[0])).unwrap(), dims_i64(&[3, 3]));
        assert_eq!(ctx.shape(one(out[1])).unwrap(), dims_i64(&[3, 2]));

        let mut ctx = RuleCtx::new(&mut builder);
        assert_eq!(
            solve_jvp(&mut ctx, &op, &[Dual::ZERO, Dual::ZERO]).unwrap(),
            vec![Dual::ZERO]
        );
        assert_eq!(
            solve_vjp(&mut ctx, &op, &[Dual::ZERO]).unwrap(),
            vec![Dual::ZERO, Dual::ZERO]
        );
    }

    #[test]
    fn solve_jvp_and_vjp_shapes_for_a_vector_rhs() {
        let mut builder = Builder::new("main");
        builder
            .open_function(
                "main",
                &[f32(&[3, 3]), f32(&[3]), f32(&[3, 3]), f32(&[3]), f32(&[3])],
            )
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let solved = builder.binary("solve", args[0], args[1]).unwrap();
        assert_eq!(builder.program().value(solved).unwrap().ty.rank(), 1);
        let op = rule_op(&builder, "solve", Attrs::new(), &args[..2], &[solved]);

        let mut ctx = RuleCtx::new(&mut builder);
        let out = solve_jvp(&mut ctx, &op, &[Dual::Value(args[2]), Dual::Value(args[3])]).unwrap();
        assert_eq!(ctx.shape(one(out[0])).unwrap(), dims_i64(&[3]));

        let mut ctx = RuleCtx::new(&mut builder);
        let out = solve_vjp(&mut ctx, &op, &[Dual::Value(args[4])]).unwrap();
        assert_eq!(ctx.shape(one(out[0])).unwrap(), dims_i64(&[3, 3]));
        assert_eq!(ctx.shape(one(out[1])).unwrap(), dims_i64(&[3]));

        // a tangent on `a` alone still produces the (3,) solution tangent
        let mut ctx = RuleCtx::new(&mut builder);
        let out = solve_jvp(&mut ctx, &op, &[Dual::Value(args[2]), Dual::ZERO]).unwrap();
        assert_eq!(ctx.shape(one(out[0])).unwrap(), dims_i64(&[3]));
    }

    #[test]
    fn solve_left_side_false_is_an_explicit_v1_deferral() {
        let attrs = Attrs::from_pairs([("left_side", AttrValue::Bool(false))]);
        let mut builder = Builder::new("main");
        builder
            .open_function("main", &[f32(&[3, 3]), f32(&[3, 2]), f32(&[3, 2])])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let solved = builder.binary("solve", args[0], args[1]).unwrap();
        let op = rule_op(&builder, "solve", attrs, &args[..2], &[solved]);
        let mut ctx = RuleCtx::new(&mut builder);
        let err = solve_jvp(&mut ctx, &op, &[Dual::ZERO, Dual::Value(args[2])]).unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        assert!(err.message.contains("left_side=False"), "{}", err.message);
        let mut ctx = RuleCtx::new(&mut builder);
        let err = solve_vjp(&mut ctx, &op, &[Dual::Value(args[2])]).unwrap_err();
        assert!(err.message.contains("left_side=False"), "{}", err.message);
    }

    #[test]
    fn registration_covers_every_ruled_op() {
        let mut registry = Registry::default();
        register_rules(&mut registry);
        for name in RULED_OPS {
            assert!(registry.jvp.contains(name), "jvp rule for {name}");
            assert!(registry.vjp.contains(name), "vjp rule for {name}");
        }
        assert_eq!(registry.jvp.len(), RULED_OPS.len());
        assert_eq!(registry.vjp.len(), RULED_OPS.len());
    }
}
