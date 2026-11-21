//! Pointwise derivative rules (JVP + VJP) for the elementwise op family.
//!
//! Port of `_UNARY_DERIV`, `_POINTWISE_JVP` and `_POINTWISE_VJP` from
//! `evoxc/transforms/rules.py`.  Unary ops share one self-adjoint coefficient
//! table between the forward and reverse sweeps; the special cases (`add`,
//! `subtract`, `multiply`, `divide`, `power`, `remainder`, `maximum`,
//! `minimum`, `relu`, `cast`, `select`, `broadcast`) carry their own rules.
//!
//! Rules read the *rewritten* values: `op.operands` are the primals of the
//! output module and `op.results` its rebuilt results.

use evoxc_core::dtype::DType;
use evoxc_core::error::{Error, Result};
use evoxc_core::graph::ValueId;

use super::common::{reduce_to_operand, sum_terms, zeros_of};
use super::Dual;
use crate::emit::{RuleCtx, RuleOp};
use crate::Registry;

/// A scalar constant of `dtype`.
fn constant(ctx: &mut RuleCtx<'_>, dtype: DType, value: f64) -> Result<ValueId> {
    ctx.scalar(dtype, value)
}

/// `_gelu_coeff`: `0.5·(1 + erf(x/√2)) + (x/√(2π))·exp(−x²/2)`.
fn gelu_coeff(ctx: &mut RuleCtx<'_>, a: ValueId) -> Result<ValueId> {
    let dtype = ctx.dtype(a)?;
    let inv_sqrt2 = constant(ctx, dtype, 0.707_106_781_186_547_6)?;
    let u = ctx.binary("multiply", a, inv_sqrt2)?;
    let erf_u = ctx.unary("erf", u)?;
    let one = constant(ctx, dtype, 1.0)?;
    let inner = ctx.binary("add", one, erf_u)?;
    let half = constant(ctx, dtype, 0.5)?;
    let term1 = ctx.binary("multiply", half, inner)?;
    let inv_sqrt_2pi = constant(ctx, dtype, 0.398_942_280_401_432_7)?;
    let scaled = ctx.binary("multiply", a, inv_sqrt_2pi)?;
    let u_squared = ctx.unary("square", u)?;
    let negated = ctx.unary("negate", u_squared)?;
    let exponential = ctx.unary("exp", negated)?;
    let term2 = ctx.binary("multiply", scaled, exponential)?;
    ctx.binary("add", term1, term2)
}

/// `d y / d x` for one unary pointwise op, given the primal `a` and result `y`.
fn unary_coefficient(
    ctx: &mut RuleCtx<'_>,
    name: &str,
    a: ValueId,
    y: ValueId,
) -> Result<ValueId> {
    let dtype = ctx.dtype(a)?;
    match name {
        "abs" => ctx.unary("sign", a),
        "negate" => {
            let one = constant(ctx, dtype, 1.0)?;
            ctx.unary("negate", one)
        }
        "square" => {
            let two = constant(ctx, dtype, 2.0)?;
            ctx.binary("multiply", two, a)
        }
        "sqrt" => {
            let half = constant(ctx, dtype, 0.5)?;
            ctx.binary("divide", half, y)
        }
        "exp" => Ok(y),
        "log" => {
            let one = constant(ctx, dtype, 1.0)?;
            ctx.binary("divide", one, a)
        }
        "log1p" => {
            let one = constant(ctx, dtype, 1.0)?;
            let denominator = ctx.binary("add", one, a)?;
            ctx.binary("divide", one, denominator)
        }
        "sin" => ctx.unary("cos", a),
        "cos" => {
            let sine = ctx.unary("sin", a)?;
            ctx.unary("negate", sine)
        }
        "tan" => {
            let one = constant(ctx, dtype, 1.0)?;
            let squared = ctx.unary("square", y)?;
            ctx.binary("add", one, squared)
        }
        "tanh" => {
            let one = constant(ctx, dtype, 1.0)?;
            let squared = ctx.unary("square", y)?;
            ctx.binary("subtract", one, squared)
        }
        "sigmoid" => {
            let one = constant(ctx, dtype, 1.0)?;
            let complement = ctx.binary("subtract", one, y)?;
            ctx.binary("multiply", y, complement)
        }
        "gelu" => gelu_coeff(ctx, a),
        "erf" => {
            let coefficient = constant(ctx, dtype, 1.128_379_167_095_512_6)?;
            let squared = ctx.unary("square", a)?;
            let negated = ctx.unary("negate", squared)?;
            let exponential = ctx.unary("exp", negated)?;
            ctx.binary("multiply", coefficient, exponential)
        }
        "sign" => {
            // the derivative is zero almost everywhere (a ⊙ 0 = 0)
            let zero = constant(ctx, dtype, 0.0)?;
            ctx.binary("multiply", a, zero)
        }
        other => Err(Error::transform(format!(
            "pointwise derivative: no coefficient for unary op '{other}'"
        ))),
    }
}

fn unary_jvp(ctx: &mut RuleCtx<'_>, op: &RuleOp, tangent: Dual) -> Result<Vec<Dual>> {
    let Some(tangent) = tangent.value() else {
        return Ok(vec![Dual::ZERO]);
    };
    let coefficient = unary_coefficient(ctx, &op.name, op.operands[0], op.results[0])?;
    Ok(vec![Dual::Value(ctx.binary("multiply", tangent, coefficient)?)])
}

fn unary_vjp(ctx: &mut RuleCtx<'_>, op: &RuleOp, cotangent: Dual) -> Result<Vec<Dual>> {
    let Some(cotangent) = cotangent.value() else {
        return Ok(vec![Dual::ZERO]);
    };
    let coefficient = unary_coefficient(ctx, &op.name, op.operands[0], op.results[0])?;
    Ok(vec![Dual::Value(ctx.binary("multiply", cotangent, coefficient)?)])
}

/// The second operand of a binary pointwise op.
///
/// `relu`, `cast` and `broadcast` carry a rule here but are arity-1 and are
/// not in the shared unary derivative table, so their arms never read operand
/// 1 and a missing second operand is an explicit `TransformError`, never an
/// index panic.
fn second_operand(op: &RuleOp) -> Result<ValueId> {
    op.operands.get(1).copied().ok_or_else(|| {
        Error::transform(format!(
            "pointwise derivative: op '{}' has no second operand ({} operand(s))",
            op.name,
            op.operands.len()
        ))
    })
}

/// The per-op JVP (one `Dual` out per result).
pub fn jvp_rule(
    ctx: &mut RuleCtx<'_>,
    op: &RuleOp,
    tangents: &[Dual],
) -> Result<Vec<Dual>> {
    if is_unary(&op.name) {
        return unary_jvp(ctx, op, tangents[0]);
    }
    let a = op.operands[0];
    let (ta, tb) = (
        tangents.first().copied().unwrap_or(Dual::ZERO).value(),
        tangents.get(1).copied().unwrap_or(Dual::ZERO).value(),
    );
    match op.name.as_str() {
        "add" => match (ta, tb) {
            (None, None) => Ok(vec![Dual::ZERO]),
            (None, Some(tb)) => Ok(vec![Dual::Value(tb)]),
            (Some(ta), None) => Ok(vec![Dual::Value(ta)]),
            (Some(ta), Some(tb)) => Ok(vec![Dual::Value(ctx.binary("add", ta, tb)?)]),
        },
        "subtract" => match (ta, tb) {
            (None, None) => Ok(vec![Dual::ZERO]),
            (None, Some(tb)) => {
                let negated = ctx.unary("negate", tb)?;
                Ok(vec![Dual::Value(negated)])
            }
            (Some(ta), None) => Ok(vec![Dual::Value(ta)]),
            (Some(ta), Some(tb)) => {
                Ok(vec![Dual::Value(ctx.binary("subtract", ta, tb)?)])
            }
        },
        "multiply" => {
            let b = second_operand(op)?;
            let mut terms = Vec::new();
            if let Some(tb) = tb {
                terms.push(ctx.binary("multiply", a, tb)?);
            }
            if let Some(ta) = ta {
                terms.push(ctx.binary("multiply", ta, b)?);
            }
            if terms.is_empty() {
                return Ok(vec![Dual::ZERO]);
            }
            Ok(vec![Dual::Value(sum_terms(ctx, &terms)?)])
        }
        "divide" => {
            let b = second_operand(op)?;
            let y = op.results[0];
            let mut terms = Vec::new();
            if let Some(ta) = ta {
                terms.push(ctx.binary("divide", ta, b)?);
            }
            if let Some(tb) = tb {
                let scaled = ctx.binary("multiply", y, tb)?;
                let quotient = ctx.binary("divide", scaled, b)?;
                terms.push(ctx.unary("negate", quotient)?);
            }
            if terms.is_empty() {
                return Ok(vec![Dual::ZERO]);
            }
            Ok(vec![Dual::Value(sum_terms(ctx, &terms)?)])
        }
        "power" => {
            let b = second_operand(op)?;
            let y = op.results[0];
            let mut terms = Vec::new();
            if let Some(tb) = tb {
                let log_a = ctx.unary("log", a)?;
                let scaled = ctx.binary("multiply", tb, log_a)?;
                terms.push(ctx.binary("multiply", y, scaled)?);
            }
            if let Some(ta) = ta {
                let numerator = ctx.binary("multiply", b, ta)?;
                let quotient = ctx.binary("divide", numerator, a)?;
                terms.push(ctx.binary("multiply", y, quotient)?);
            }
            if terms.is_empty() {
                return Ok(vec![Dual::ZERO]);
            }
            Ok(vec![Dual::Value(sum_terms(ctx, &terms)?)])
        }
        "remainder" => Ok(vec![Dual::from_option(ta)]),
        "maximum" | "minimum" => {
            let b = second_operand(op)?;
            if ta.is_none() && tb.is_none() {
                return Ok(vec![Dual::ZERO]);
            }
            let ta = ta.unwrap_or(zeros_of(ctx, a)?);
            let tb = tb.unwrap_or(zeros_of(ctx, b)?);
            let comparison = if op.name == "maximum" { "greater" } else { "less" };
            let mask = ctx.binary(comparison, a, b)?;
            Ok(vec![Dual::Value(ctx.ternary("select", mask, ta, tb)?)])
        }
        "relu" => {
            let Some(ta) = ta else {
                return Ok(vec![Dual::ZERO]);
            };
            let zero = constant(ctx, ctx.dtype(a)?, 0.0)?;
            let mask = ctx.binary("greater", a, zero)?;
            let zeros = zeros_of(ctx, a)?;
            Ok(vec![Dual::Value(ctx.ternary("select", mask, ta, zeros)?)])
        }
        "cast" => {
            let Some(ta) = ta else {
                return Ok(vec![Dual::ZERO]);
            };
            let dtype = ctx.dtype(op.results[0])?;
            Ok(vec![Dual::Value(ctx.cast(ta, dtype)?)])
        }
        "select" => {
            let truthy = tangents[1].value();
            let falsy = tangents[2].value();
            if truthy.is_none() && falsy.is_none() {
                return Ok(vec![Dual::ZERO]);
            }
            let truthy = truthy.unwrap_or(zeros_of(ctx, op.operands[1])?);
            let falsy = falsy.unwrap_or(zeros_of(ctx, op.operands[2])?);
            Ok(vec![Dual::Value(ctx.ternary("select", a, truthy, falsy)?)])
        }
        "broadcast" => {
            let Some(ta) = ta else {
                return Ok(vec![Dual::ZERO]);
            };
            let shape = op.shape_attr("shape")?.to_vec();
            Ok(vec![Dual::Value(ctx.broadcast(ta, &shape)?)])
        }
        other => Err(Error::transform(format!(
            "jvp: no JVP rule for op '{other}'"
        ))),
    }
}

/// The per-op VJP (one `Dual` out per operand).
pub fn vjp_rule(
    ctx: &mut RuleCtx<'_>,
    op: &RuleOp,
    cotangents: &[Dual],
) -> Result<Vec<Dual>> {
    if is_unary(&op.name) {
        return unary_vjp(ctx, op, cotangents[0]);
    }
    let a = op.operands[0];
    let Some(ct) = cotangents[0].value() else {
        let zeros = vec![Dual::ZERO; op.operands.len()];
        return Ok(zeros);
    };
    match op.name.as_str() {
        "add" => {
            let b = second_operand(op)?;
            Ok(vec![
                Dual::Value(reduce_to_operand(ctx, ct, a)?),
                Dual::Value(reduce_to_operand(ctx, ct, b)?),
            ])
        }
        "subtract" => {
            let b = second_operand(op)?;
            let negated = ctx.unary("negate", ct)?;
            Ok(vec![
                Dual::Value(reduce_to_operand(ctx, ct, a)?),
                Dual::Value(reduce_to_operand(ctx, negated, b)?),
            ])
        }
        "multiply" => {
            let b = second_operand(op)?;
            let grad_a = ctx.binary("multiply", ct, b)?;
            let grad_b = ctx.binary("multiply", ct, a)?;
            Ok(vec![
                Dual::Value(reduce_to_operand(ctx, grad_a, a)?),
                Dual::Value(reduce_to_operand(ctx, grad_b, b)?),
            ])
        }
        "divide" => {
            let b = second_operand(op)?;
            let y = op.results[0];
            let grad_a = ctx.binary("divide", ct, b)?;
            let scaled = ctx.binary("multiply", ct, y)?;
            let quotient = ctx.binary("divide", scaled, b)?;
            let grad_b = ctx.unary("negate", quotient)?;
            Ok(vec![
                Dual::Value(reduce_to_operand(ctx, grad_a, a)?),
                Dual::Value(reduce_to_operand(ctx, grad_b, b)?),
            ])
        }
        "power" => {
            let b = second_operand(op)?;
            let y = op.results[0];
            let ratio = ctx.binary("divide", b, a)?;
            let scaled = ctx.binary("multiply", ct, y)?;
            let grad_a = ctx.binary("multiply", scaled, ratio)?;
            let log_a = ctx.unary("log", a)?;
            let grad_b = ctx.binary("multiply", scaled, log_a)?;
            Ok(vec![
                Dual::Value(reduce_to_operand(ctx, grad_a, a)?),
                Dual::Value(reduce_to_operand(ctx, grad_b, b)?),
            ])
        }
        "remainder" => Ok(vec![Dual::Value(ct), Dual::ZERO]),
        "maximum" | "minimum" => {
            let b = second_operand(op)?;
            let comparison = if op.name == "maximum" { "greater" } else { "less" };
            let bool_mask = ctx.binary(comparison, a, b)?;
            let dtype = ctx.dtype(ct)?;
            let mask = ctx.cast(bool_mask, dtype)?;
            let one = constant(ctx, dtype, 1.0)?;
            let complement = ctx.binary("subtract", one, mask)?;
            Ok(vec![
                Dual::Value(ctx.binary("multiply", ct, mask)?),
                Dual::Value(ctx.binary("multiply", ct, complement)?),
            ])
        }
        "relu" => {
            let dtype = ctx.dtype(ct)?;
            let zero = constant(ctx, dtype, 0.0)?;
            let bool_mask = ctx.binary("greater", a, zero)?;
            let mask = ctx.cast(bool_mask, dtype)?;
            Ok(vec![Dual::Value(ctx.binary("multiply", ct, mask)?)])
        }
        "cast" => {
            let dtype = ctx.dtype(a)?;
            Ok(vec![Dual::Value(ctx.cast(ct, dtype)?)])
        }
        "select" => {
            let dtype = ctx.dtype(ct)?;
            let mask = ctx.cast(a, dtype)?;
            let one = constant(ctx, dtype, 1.0)?;
            let complement = ctx.binary("subtract", one, mask)?;
            Ok(vec![
                Dual::ZERO,
                Dual::Value(ctx.binary("multiply", ct, mask)?),
                Dual::Value(ctx.binary("multiply", ct, complement)?),
            ])
        }
        "broadcast" => Ok(vec![Dual::Value(reduce_to_operand(ctx, ct, a)?)]),
        other => Err(Error::transform(format!(
            "grad/vjp: no VJP rule for op '{other}'"
        ))),
    }
}

/// True when `name` is one of the `_UNARY_DERIV` ops.
pub fn is_unary(name: &str) -> bool {
    matches!(
        name,
        "abs"
            | "negate"
            | "square"
            | "sqrt"
            | "exp"
            | "log"
            | "log1p"
            | "sin"
            | "cos"
            | "tan"
            | "tanh"
            | "sigmoid"
            | "gelu"
            | "erf"
            | "sign"
    )
}

/// Every op this file rules (both sweeps).
pub const RULED_OPS: &[&str] = &[
    "abs",
    "negate",
    "square",
    "sqrt",
    "exp",
    "log",
    "log1p",
    "sin",
    "cos",
    "tan",
    "tanh",
    "sigmoid",
    "gelu",
    "erf",
    "sign",
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
    "remainder",
    "maximum",
    "minimum",
    "relu",
    "cast",
    "select",
    "broadcast",
];

/// Register the pointwise JVP and VJP rules.
pub fn register_rules(registry: &mut Registry) {
    for name in RULED_OPS {
        registry.jvp.register(name, jvp_rule);
        registry.vjp.register(name, vjp_rule);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::dims_i64;
    use crate::Builder;
    use evoxc_core::value_type::ValueType;

    fn f32(shape: &[i64]) -> ValueType {
        ValueType::new(DType::parse("float32").unwrap(), dims_i64(shape))
    }

    fn graph(name: &str, operand_shapes: &[&[i64]]) -> (Builder, Vec<ValueId>, ValueId) {
        let types: Vec<ValueType> = operand_shapes.iter().map(|shape| f32(shape)).collect();
        let mut builder = Builder::new("main");
        builder.open_function("main", &types).unwrap();
        let args = builder.block_arguments().unwrap();
        let op = match types.len() {
            // `cast` is arity-1 but carries a required `dtype` attribute
            1 if name == "cast" => {
                let attrs = evoxc_core::attr::Attrs::from_pairs([(
                    "dtype",
                    evoxc_core::attr::AttrValue::DType(DType::parse("float32").unwrap()),
                )]);
                builder.emit_typed_one(name, &[args[0]], &attrs).unwrap()
            }
            1 => builder.unary(name, args[0]).unwrap(),
            2 => builder.binary(name, args[0], args[1]).unwrap(),
            other => panic!("graph helper: {name} takes {other} operands; build it explicitly"),
        };
        (builder, args, op)
    }

    fn source(builder: &Builder, name: &str, operands: &[ValueId], results: &[ValueId]) -> RuleOp {
        let program = builder.program();
        let original_operand_types = operands
            .iter()
            .map(|id| program.value(*id).unwrap().ty.clone())
            .collect();
        let original_result_types = results
            .iter()
            .map(|id| program.value(*id).unwrap().ty.clone())
            .collect();
        RuleOp {
            name: name.into(),
            attributes: evoxc_core::attr::Attrs::new(),
            original_operand_types,
            original_result_types,
            operands: operands.to_vec(),
            operand_types: Vec::new(),
            results: results.to_vec(),
            result_types: Vec::new(),
            location: None,
        }
    }

    #[test]
    fn multiply_jvp_sums_the_product_rule_terms() {
        let (mut builder, args, product) = graph("multiply", &[&[2, 3], &[2, 3]]);
        let op = source(&builder, "multiply", &args, &[product]);
        let tangent_a = {
            let mut ctx = RuleCtx::new(&mut builder);
            ctx.scalar(DType::parse("float32").unwrap(), 1.0).unwrap()
        };
        let mut ctx = RuleCtx::new(&mut builder);
        let out = jvp_rule(&mut ctx, &op, &[Dual::Value(tangent_a), Dual::ZERO]).unwrap();
        assert_eq!(out.len(), 1);
        let value = out[0].value().unwrap();
        assert_eq!(ctx.shape(value).unwrap(), dims_i64(&[2, 3]));
        let names: Vec<String> = ctx
            .builder
            .program()
            .ops
            .iter()
            .map(|(_, op)| op.name.clone())
            .collect();
        // the source `multiply`, the scalar-one tangent constant, then the
        // single product-rule term `tangent_a * b` (zero tangents are skipped)
        assert_eq!(names, vec!["multiply", "constant", "multiply"]);
    }

    #[test]
    fn multiply_vjp_reduces_to_each_operand_shape() {
        let (mut builder, args, product) = graph("multiply", &[&[2, 3], &[2, 3]]);
        let op = source(&builder, "multiply", &args, &[product]);
        let cotangent = {
            let mut ctx = RuleCtx::new(&mut builder);
            let one = ctx.scalar(DType::parse("float32").unwrap(), 1.0).unwrap();
            ctx.broadcast(one, &dims_i64(&[2, 3])).unwrap()
        };
        let mut ctx = RuleCtx::new(&mut builder);
        let out = vjp_rule(&mut ctx, &op, &[Dual::Value(cotangent)]).unwrap();
        assert_eq!(out.len(), 2);
        for dual in out {
            let value = dual.value().unwrap();
            assert_eq!(ctx.shape(value).unwrap(), dims_i64(&[2, 3]));
        }
    }

    #[test]
    fn unary_table_covers_every_declared_op_and_zero_shorts() {
        for name in RULED_OPS {
            if !is_unary(name) {
                continue;
            }
            let (mut builder, args, result) = graph(name, &[&[2, 3]]);
            let op = source(&builder, name, &args, &[result]);
            let mut ctx = RuleCtx::new(&mut builder);
            let out = jvp_rule(&mut ctx, &op, &[Dual::ZERO]).unwrap();
            assert_eq!(out, vec![Dual::ZERO], "{name}");
            let out = vjp_rule(&mut ctx, &op, &[Dual::ZERO]).unwrap();
            assert_eq!(out, vec![Dual::ZERO], "{name}");
        }
    }

    #[test]
    fn relu_select_and_cast_rules_work() {
        let (mut builder, args, result) = graph("relu", &[&[4]]);
        let op = source(&builder, "relu", &args, &[result]);
        let tangent = {
            let mut ctx = RuleCtx::new(&mut builder);
            let one = ctx.scalar(DType::parse("float32").unwrap(), 1.0).unwrap();
            ctx.broadcast(one, &dims_i64(&[4])).unwrap()
        };
        let mut ctx = RuleCtx::new(&mut builder);
        let out = vjp_rule(&mut ctx, &op, &[Dual::Value(tangent)]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(ctx.shape(out[0].value().unwrap()).unwrap(), dims_i64(&[4]));

        let (mut builder, args, result) = graph("cast", &[&[4]]);
        let cast = source(&builder, "cast", &args, &[result]);
        // the cotangent is an ordinary value of THIS builder
        let cotangent = {
            let mut ctx = RuleCtx::new(&mut builder);
            let one = ctx.scalar(DType::parse("float32").unwrap(), 1.0).unwrap();
            ctx.broadcast(one, &dims_i64(&[4])).unwrap()
        };
        let mut ctx = RuleCtx::new(&mut builder);
        let out = vjp_rule(&mut ctx, &cast, &[Dual::Value(cotangent)]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(ctx.shape(out[0].value().unwrap()).unwrap(), dims_i64(&[4]));

        let mut builder = Builder::new("main");
        builder
            .open_function(
                "main",
                &[
                    ValueType::new(DType::parse("bool").unwrap(), dims_i64(&[4])),
                    f32(&[4]),
                    f32(&[4]),
                ],
            )
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let selected = builder.ternary("select", args[0], args[1], args[2]).unwrap();
        let op = source(&builder, "select", &args, &[selected]);
        // the truthy tangent is an ordinary value of THIS builder
        let tangent = {
            let mut ctx = RuleCtx::new(&mut builder);
            let one = ctx.scalar(DType::parse("float32").unwrap(), 1.0).unwrap();
            ctx.broadcast(one, &dims_i64(&[4])).unwrap()
        };
        let mut ctx = RuleCtx::new(&mut builder);
        let out = jvp_rule(
            &mut ctx,
            &op,
            &[Dual::ZERO, Dual::Value(tangent), Dual::ZERO],
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(ctx.shape(out[0].value().unwrap()).unwrap(), dims_i64(&[4]));
    }
}
