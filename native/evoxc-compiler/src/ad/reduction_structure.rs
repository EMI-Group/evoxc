//! Reduction and structural derivative rules.
//!
//! Port of `_reduction_jvp` / `_reduction_vjp` / `_expand_reduced` /
//! `_slice_vjp` and `_structural_jvp` / `_structural_vjp` from
//! `evoxc/transforms/rules.py`, plus the trivial zero rules (`_zero_jvp`,
//! `_zero_vjp`, `_stop_gradient_jvp`, `_stop_gradient_vjp`).
//!
//! Reductions are linear in their operand, so the forward sweep re-reduces the
//! tangent over the same axes; the reverse sweep broadcasts the cotangent back
//! to the operand shape with the per-kind weighting (`mean` divides by the
//! static reduced count, `prod` multiplies by y/x, `max`/`min` select on the
//! arg mask).  Structural ops are linear in their tensor operands: the forward
//! sweep reapplies the op to the tangents and the reverse sweep applies its
//! inverse (reshape back, inverse permutation, pad-back / scatter for a slice,
//! inverse slice for pad, per-segment slices for concatenate, scatter for
//! gather, gather for scatter).  Every v1 gap is an explicit `TransformError`
//! naming the op — never a silent fallback.
//!
//! Rules read the *rewritten* values: `op.operands` are the primals of the
//! output module and `op.results` its rebuilt results.

use evoxc_core::attr::AttrValue;
use evoxc_core::dim::Dim;
use evoxc_core::error::{Error, Result};
use evoxc_core::graph::ValueId;
use serde_json::Value as Json;

use super::common::{expand_reduced, zeros_of};
use super::Dual;
use crate::emit::{RuleCtx, RuleOp};
use crate::vectorize::format_shape;
use crate::Registry;

/// The reduction ops this file rules (the `sum`/`max`/`min`/`mean`/`prod`
/// frontend sugar expands onto `reduce_*` and never reaches the IR).
pub const REDUCTION_OPS: &[&str] = &[
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_mean",
    "reduce_prod",
];

/// The structural ops this file rules (all linear in their tensor operands).
pub const STRUCTURAL_OPS: &[&str] = &[
    "reshape",
    "transpose",
    "slice",
    "pad",
    "concatenate",
    "gather",
    "scatter",
];

/// Ops whose outputs are not differentiable: zero tangents/cotangents, not an
/// error (bool/integer outputs cannot carry a derivative).
pub const NONDIFFERENTIABLE_OPS: &[&str] = &[
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
    "argmax",
    "argmin",
];

// ---------------------------------------------------------------------------
// attribute readers
//
// The rules see attributes as the output module carries them: the typed forms
// the builder emits (`ints`/`bool`/`float`) plus the envelope's bare-int and
// `__evoxc_tuple__` spellings, exactly like the inference layer's
// `attr_i64_or_ints` / `pad_config` readers.
// ---------------------------------------------------------------------------

/// An attribute as a flat int sequence; `None` when absent or explicitly null.
fn int_seq(op: &RuleOp, name: &str) -> Result<Option<Vec<i64>>> {
    match op.attributes.get(name) {
        None | Some(AttrValue::Null) => Ok(None),
        Some(AttrValue::Int(value)) => Ok(Some(vec![*value])),
        Some(AttrValue::Ints(values)) => Ok(Some(values.clone())),
        Some(AttrValue::Tuple(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                let value = match item {
                    AttrValue::Int(value) => Some(*value),
                    other => other.to_json().as_i64(),
                };
                match value {
                    Some(value) => out.push(value),
                    None => return Err(Error::transform(format!(
                        "op '{}': attribute {name:?} must be a list of ints, got a non-int entry",
                        op.name
                    ))),
                }
            }
            Ok(Some(out))
        }
        Some(other) => Err(Error::transform(format!(
            "op '{}': attribute {name:?} must be a list of ints, got {}",
            op.name,
            other.type_name()
        ))),
    }
}

/// An optional bool attribute (`absent`/`null` → `false`).
fn bool_attr(op: &RuleOp, name: &str) -> Result<bool> {
    match op.attributes.get(name) {
        None | Some(AttrValue::Null) => Ok(false),
        Some(AttrValue::Bool(value)) => Ok(*value),
        Some(other) => Err(Error::transform(format!(
            "op '{}': attribute {name:?} must be a bool, got {}",
            op.name,
            other.type_name()
        ))),
    }
}

/// A float attribute with a default.
fn float_attr(op: &RuleOp, name: &str, default: f64) -> Result<f64> {
    match op.attributes.get(name) {
        None | Some(AttrValue::Null) => Ok(default),
        Some(AttrValue::Float(value)) => Ok(*value),
        Some(AttrValue::Int(value)) => Ok(*value as f64),
        Some(other) => Err(Error::transform(format!(
            "op '{}': attribute {name:?} must be a float, got {}",
            op.name,
            other.type_name()
        ))),
    }
}

/// The `pad` op's `padding_config` as per-axis `(low, high)` pairs.
///
/// Port of the entry normalization of `infer_pad`: a `[lo, hi]` pair per axis,
/// or a bare int replicated on both sides.
fn pad_pairs(op: &RuleOp) -> Result<Vec<(i64, i64)>> {
    let Some(value) = op.attributes.get("padding_config") else {
        return Err(Error::transform(format!(
            "op '{}' is missing the 'padding_config' attribute",
            op.name
        )));
    };
    let raw = match value {
        AttrValue::Tuple(items) => Json::Array(items.iter().map(AttrValue::to_json).collect()),
        other => other.to_json(),
    };
    let Some(entries) = raw.as_array() else {
        return Err(Error::transform(format!(
            "op '{}': padding_config must be a list of entries, got {raw}",
            op.name
        )));
    };
    entries
        .iter()
        .map(|entry| {
            if let Some(both) = entry.as_i64() {
                return Ok((both, both));
            }
            let pair = entry
                .as_array()
                .and_then(|items| Some((items.first()?.as_i64()?, items.get(1)?.as_i64()?)));
            match pair {
                Some(pair) => Ok(pair),
                None => Err(Error::transform(format!(
                    "op '{}': invalid padding entry {entry} (expected int or (lo, hi) pair)",
                    op.name
                ))),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// reductions
// ---------------------------------------------------------------------------

/// `reduce_<kind>` → `<kind>`.
fn reduce_kind(name: &str) -> &str {
    name.strip_prefix("reduce_").unwrap_or(name)
}

/// The reduction axes normalized to non-negative positions; empty means every
/// axis (the IR convention).
fn reduced_axes(raw: Option<Vec<i64>>, rank: usize) -> Vec<i64> {
    let rank = rank as i64;
    match raw {
        None => (0..rank).collect(),
        Some(axes) if axes.is_empty() => (0..rank).collect(),
        Some(axes) => axes
            .iter()
            .map(|axis| if *axis < 0 { *axis + rank } else { *axis })
            .collect(),
    }
}

/// The static element count reduced by `axes` (the `mean` weight).
///
/// A symbolic reduced dim is an explicit v1 deferral: the count is unknown at
/// graph-construction time.
fn static_count(ctx: &RuleCtx<'_>, op: &RuleOp, x: ValueId, axes: &[i64]) -> Result<i64> {
    let shape = ctx.shape(x)?;
    let mut count: i64 = 1;
    for axis in axes {
        let dim = shape.get(*axis as usize).ok_or_else(|| {
            Error::transform(format!(
                "grad/vjp: reduction axis {axis} of op '{}' is out of range for rank {}",
                op.name,
                shape.len()
            ))
        })?;
        match dim {
            Dim::Int(value) => count *= *value,
            other => {
                return Err(Error::transform(format!(
                    "grad/vjp: cannot differentiate 'mean' over symbolic dim {other} — \
                     the reduction count is unknown (v1 gap)"
                )))
            }
        }
    }
    Ok(count)
}

/// Reduction JVP: reduce the tangent over the same axes (the op is linear).
fn reduction_jvp(ctx: &mut RuleCtx<'_>, op: &RuleOp, tangents: &[Dual]) -> Result<Vec<Dual>> {
    let Some(tangent) = tangents.first().copied().and_then(Dual::value) else {
        return Ok(vec![Dual::ZERO]);
    };
    let kind = reduce_kind(&op.name);
    let rank = ctx.rank(op.operands[0])?;
    let axes = reduced_axes(int_seq(op, "axes")?, rank);
    let keepdims = bool_attr(op, "keepdims")?;
    Ok(vec![Dual::Value(
        ctx.reduce(kind, tangent, &axes, keepdims)?,
    )])
}

/// Reduction VJP: broadcast the cotangent back to the operand shape, with the
/// per-kind weighting.
fn reduction_vjp(ctx: &mut RuleCtx<'_>, op: &RuleOp, cotangents: &[Dual]) -> Result<Vec<Dual>> {
    let Some(ct) = cotangents.first().copied().and_then(Dual::value) else {
        return Ok(vec![Dual::ZERO]);
    };
    let x = op.operands[0];
    let kind = op.str("reduce_op")?.to_string();
    let rank = ctx.rank(x)?;
    let axes = reduced_axes(int_seq(op, "axes")?, rank);
    let keepdims = bool_attr(op, "keepdims")?;
    if axes.is_empty() {
        // rank-0 identity: nothing was reduced, so the cotangent is the gradient.
        return Ok(vec![Dual::Value(ct)]);
    }
    match kind.as_str() {
        "sum" => Ok(vec![Dual::Value(expand_reduced(
            ctx, ct, x, &axes, keepdims,
        )?)]),
        "mean" => {
            let count = static_count(ctx, op, x, &axes)?;
            let dtype = ctx.dtype(ct)?;
            let divisor = ctx.scalar(dtype, count as f64)?;
            let weighted = ctx.binary("divide", ct, divisor)?;
            Ok(vec![Dual::Value(expand_reduced(
                ctx, weighted, x, &axes, keepdims,
            )?)])
        }
        "prod" => {
            // d(prod)/dx_i = prod / x_i — broadcast the cotangent and the result
            // back to the operand shape, then divide by x.
            let cb = expand_reduced(ctx, ct, x, &axes, keepdims)?;
            let yb = expand_reduced(ctx, op.results[0], x, &axes, keepdims)?;
            let scaled = ctx.binary("multiply", cb, yb)?;
            Ok(vec![Dual::Value(ctx.binary("divide", scaled, x)?)])
        }
        "max" | "min" => {
            // The gradient flows only through the arg positions.
            let cb = expand_reduced(ctx, ct, x, &axes, keepdims)?;
            let yb = expand_reduced(ctx, op.results[0], x, &axes, keepdims)?;
            let matches = ctx.binary("equal", x, yb)?;
            let dtype = ctx.dtype(ct)?;
            let mask = ctx.cast(matches, dtype)?;
            Ok(vec![Dual::Value(ctx.binary("multiply", cb, mask)?)])
        }
        other => Err(Error::transform(format!(
            "grad/vjp: unknown reduce_op '{other}' on op '{}'",
            op.name
        ))),
    }
}

// ---------------------------------------------------------------------------
// structural ops
// ---------------------------------------------------------------------------

/// Structural JVP: apply the same structural op to the tangents (all of them
/// are linear in their tensor operands).
fn structural_jvp(ctx: &mut RuleCtx<'_>, op: &RuleOp, tangents: &[Dual]) -> Result<Vec<Dual>> {
    match op.name.as_str() {
        "concatenate" => {
            if tangents.iter().all(|dual| dual.is_zero()) {
                return Ok(vec![Dual::ZERO]);
            }
            // An unmapped operand contributes an explicit zero segment.
            let mut values = Vec::with_capacity(op.operands.len());
            for (dual, operand) in tangents.iter().zip(&op.operands) {
                values.push(match dual.value() {
                    Some(value) => value,
                    None => zeros_of(ctx, *operand)?,
                });
            }
            let axis = op.opt_int("axis")?.unwrap_or(0);
            Ok(vec![Dual::Value(ctx.concatenate(&values, axis)?)])
        }
        "scatter" => {
            // Linear in x (identity at untouched positions) and in updates.
            let tx = tangents.first().copied().and_then(Dual::value);
            let tupd = tangents.get(2).copied().and_then(Dual::value);
            if tx.is_none() && tupd.is_none() {
                return Ok(vec![Dual::ZERO]);
            }
            let x = match tx {
                Some(value) => value,
                None => zeros_of(ctx, op.operands[0])?,
            };
            let updates = match tupd {
                Some(value) => value,
                None => zeros_of(ctx, op.operands[2])?,
            };
            let axis = op.opt_int("axis")?.unwrap_or(0);
            Ok(vec![Dual::Value(ctx.scatter(
                x,
                op.operands[1],
                updates,
                axis,
            )?)])
        }
        _ => {
            let Some(tangent) = tangents.first().copied().and_then(Dual::value) else {
                return Ok(vec![Dual::ZERO]);
            };
            let value = match op.name.as_str() {
                "reshape" => {
                    let shape = op.shape_attr("shape")?.to_vec();
                    ctx.reshape(tangent, &shape)?
                }
                "transpose" => {
                    let permutation = int_seq(op, "permutation")?;
                    ctx.transpose(tangent, permutation.as_deref())?
                }
                "slice" => {
                    let starts = int_seq(op, "start_indices")?.unwrap_or_default();
                    let limits = int_seq(op, "limit_indices")?.unwrap_or_default();
                    let strides = int_seq(op, "strides")?;
                    ctx.slice(tangent, &starts, &limits, strides.as_deref())?
                }
                "pad" => {
                    let config = pad_pairs(op)?;
                    let fill = float_attr(op, "value", 0.0)?;
                    ctx.pad(tangent, &config, fill)?
                }
                "gather" => {
                    let axes = int_seq(op, "axes")?.unwrap_or_else(|| vec![0]);
                    if axes.len() != 1 {
                        return Err(Error::transform(
                            "jvp: cannot differentiate 'gather' with multiple axes (v1 gap)",
                        ));
                    }
                    ctx.gather(tangent, op.operands[1], &axes)?
                }
                other => {
                    return Err(Error::transform(format!(
                        "jvp: no JVP rule for op '{other}'"
                    )))
                }
            };
            Ok(vec![Dual::Value(value)])
        }
    }
}

/// Slice VJP: a stride-1 slice restores via `pad` (one `(start, dim - limit)`
/// per sliced axis); a single strided axis restores via `scatter` with a static
/// arange index; anything else is a v1 `TransformError`.
fn slice_vjp(ctx: &mut RuleCtx<'_>, op: &RuleOp, ct: ValueId, x: ValueId) -> Result<ValueId> {
    let rank = ctx.rank(x)?;
    let starts = int_seq(op, "start_indices")?.unwrap_or_default();
    let limits = int_seq(op, "limit_indices")?.unwrap_or_default();
    let strides = match int_seq(op, "strides")? {
        Some(strides) => strides,
        None => vec![1; starts.len()],
    };
    if starts.len() != rank || limits.len() != rank || strides.len() != rank {
        return Err(Error::transform(format!(
            "grad/vjp: slice attribute arity ({} start(s), {} limit(s), {} stride(s)) \
             does not match rank {rank}",
            starts.len(),
            limits.len(),
            strides.len()
        )));
    }
    let shape = ctx.shape(x)?;
    let mut partial: Vec<usize> = Vec::new();
    for index in 0..rank {
        let full = starts[index] == 0
            && strides[index] == 1
            && matches!(shape[index], Dim::Int(value) if value == limits[index]);
        if !full {
            partial.push(index);
        }
    }
    if partial.is_empty() {
        // the full slice: the gradient is the identity
        return Ok(ct);
    }
    let strided: Vec<usize> = partial
        .iter()
        .copied()
        .filter(|index| strides[*index] > 1)
        .collect();
    if strided.is_empty() {
        let mut config: Vec<(i64, i64)> = Vec::with_capacity(rank);
        for index in 0..rank {
            if !partial.contains(&index) {
                config.push((0, 0));
                continue;
            }
            let dim = &shape[index];
            let high = match dim {
                Dim::Int(value) => value - limits[index],
                other => {
                    return Err(Error::transform(format!(
                        "grad/vjp: cannot invert a slice over symbolic dim {other} (v1 gap)"
                    )))
                }
            };
            config.push((starts[index], high));
        }
        return ctx.pad(ct, &config, 0.0);
    }
    if partial.len() != 1 || strided.len() != 1 {
        return Err(Error::transform(
            "grad/vjp: cannot invert a slice with strided entries over multiple axes (v1 gap)",
        ));
    }
    let axis = partial[0];
    let zeros = zeros_of(ctx, x)?;
    let indices = ctx.arange_i64(starts[axis], limits[axis], strides[axis])?;
    ctx.scatter(zeros, indices, ct, axis as i64)
}

/// Structural VJP: the inverse structural op applied to the cotangent.
fn structural_vjp(ctx: &mut RuleCtx<'_>, op: &RuleOp, cotangents: &[Dual]) -> Result<Vec<Dual>> {
    let Some(ct) = cotangents.first().copied().and_then(Dual::value) else {
        return Ok(vec![Dual::ZERO; op.operands.len()]);
    };
    let x = op.operands[0];
    match op.name.as_str() {
        "reshape" => {
            let shape = ctx.shape(x)?;
            Ok(vec![Dual::Value(ctx.reshape(ct, &shape)?)])
        }
        "transpose" => {
            let value = match int_seq(op, "permutation")? {
                None => ctx.transpose(ct, None)?,
                Some(permutation) => {
                    let mut inverse = vec![0i64; permutation.len()];
                    for (index, axis) in permutation.iter().enumerate() {
                        let slot = inverse.get_mut(*axis as usize).ok_or_else(|| {
                            Error::transform(format!(
                                "grad/vjp: transpose permutation entry {axis} is out of range \
                                 for rank {}",
                                permutation.len()
                            ))
                        })?;
                        *slot = index as i64;
                    }
                    ctx.transpose(ct, Some(&inverse))?
                }
            };
            Ok(vec![Dual::Value(value)])
        }
        "slice" => Ok(vec![Dual::Value(slice_vjp(ctx, op, ct, x)?)]),
        "pad" => {
            let config = pad_pairs(op)?;
            let starts: Vec<i64> = config.iter().map(|(low, _)| *low).collect();
            let shape = ctx.shape(x)?;
            if starts.len() != shape.len() {
                return Err(Error::transform(format!(
                    "grad/vjp: pad has {} padding entr(ies) for rank {}",
                    starts.len(),
                    shape.len()
                )));
            }
            let mut limits = Vec::with_capacity(shape.len());
            for dim in &shape {
                match dim {
                    Dim::Int(value) => limits.push(*value),
                    _ => {
                        return Err(Error::transform(format!(
                            "grad/vjp: cannot invert a pad over symbolic dim {dim} \
                             (the inverse slice needs static lengths, v1 gap)"
                        )))
                    }
                }
            }
            // The Python rule passes SIZES; the IR slice wants absolute limits.
            for (limit, start) in limits.iter_mut().zip(&starts) {
                *limit += start;
            }
            let ones = vec![1i64; shape.len()];
            Ok(vec![Dual::Value(ctx.slice(
                ct,
                &starts,
                &limits,
                Some(&ones),
            )?)])
        }
        "concatenate" => {
            let rank = ctx.rank(ct)?;
            let axis = op.opt_int("axis")?.unwrap_or(0);
            let axis = if axis < 0 { axis + rank as i64 } else { axis };
            if axis < 0 || axis as usize >= rank {
                return Err(Error::transform(format!(
                    "grad/vjp: concatenate axis {axis} is out of range for rank {rank}"
                )));
            }
            let axis = axis as usize;
            let ct_shape = ctx.shape(ct)?;
            let mut limits: Vec<i64> = Vec::with_capacity(ct_shape.len());
            for (index, dim) in ct_shape.iter().enumerate() {
                if index == axis {
                    limits.push(0);
                    continue;
                }
                match dim {
                    Dim::Int(value) => limits.push(*value),
                    _ => {
                        return Err(Error::transform(format!(
                            "grad/vjp: cannot split a concatenate with symbolic dims {} (v1 gap)",
                            format_shape(&ct_shape)
                        )))
                    }
                }
            }
            let ones = vec![1i64; ct_shape.len()];
            let mut offset = 0i64;
            let mut grads = Vec::with_capacity(op.operands.len());
            for operand in &op.operands {
                let length = match ctx.shape(*operand)?.get(axis) {
                    Some(Dim::Int(value)) => *value,
                    Some(other) => {
                        return Err(Error::transform(format!(
                            "grad/vjp: cannot split a concatenate over symbolic dim {other} \
                             (v1 gap)"
                        )))
                    }
                    None => {
                        return Err(Error::transform(format!(
                            "grad/vjp: concatenate operand has no axis {axis}"
                        )))
                    }
                };
                // The Python rule passes SIZES; the IR slice wants absolute limits.
                let mut starts = vec![0i64; ct_shape.len()];
                starts[axis] = offset;
                limits[axis] = offset + length;
                grads.push(Dual::Value(ctx.slice(ct, &starts, &limits, Some(&ones))?));
                offset += length;
            }
            Ok(grads)
        }
        "gather" => {
            let axes = int_seq(op, "axes")?.unwrap_or_else(|| vec![0]);
            if axes.len() != 1 {
                return Err(Error::transform(
                    "grad/vjp: cannot differentiate 'gather' with multiple axes (v1 gap)",
                ));
            }
            let zeros = zeros_of(ctx, x)?;
            let grad = ctx.scatter(zeros, op.operands[1], ct, axes[0])?;
            Ok(vec![Dual::Value(grad), Dual::ZERO])
        }
        "scatter" => {
            // g_x = scatter(ct, idx, zeros-of-updates); g_updates = gather(ct, idx);
            // the indices are not differentiable.
            let axis = op.opt_int("axis")?.unwrap_or(0);
            let zeros = zeros_of(ctx, op.operands[2])?;
            let g_x = ctx.scatter(ct, op.operands[1], zeros, axis)?;
            let g_updates = ctx.gather(ct, op.operands[1], &[axis])?;
            Ok(vec![Dual::Value(g_x), Dual::ZERO, Dual::Value(g_updates)])
        }
        other => Err(Error::transform(format!(
            "grad/vjp: no VJP rule for op '{other}'"
        ))),
    }
}

// ---------------------------------------------------------------------------
// trivial zero rules
// ---------------------------------------------------------------------------

/// JVP rule for ops with non-differentiable outputs: zero tangents.
fn zero_jvp(_ctx: &mut RuleCtx<'_>, op: &RuleOp, _tangents: &[Dual]) -> Result<Vec<Dual>> {
    Ok(vec![Dual::ZERO; op.results.len()])
}

/// VJP rule for ops with non-differentiable outputs: zero cotangents.
fn zero_vjp(_ctx: &mut RuleCtx<'_>, op: &RuleOp, _cotangents: &[Dual]) -> Result<Vec<Dual>> {
    Ok(vec![Dual::ZERO; op.operands.len()])
}

/// Register the reduction and structural JVP and VJP rules.
///
/// `stop_gradient` shares the zero rules: its gradient is zero by definition
/// (the Python `_stop_gradient_jvp` / `_stop_gradient_vjp` have the same body).
pub fn register_rules(registry: &mut Registry) {
    for name in REDUCTION_OPS {
        registry.jvp.register(name, reduction_jvp);
        registry.vjp.register(name, reduction_vjp);
    }
    for name in STRUCTURAL_OPS {
        registry.jvp.register(name, structural_jvp);
        registry.vjp.register(name, structural_vjp);
    }
    for name in NONDIFFERENTIABLE_OPS {
        registry.jvp.register(name, zero_jvp);
        registry.vjp.register(name, zero_vjp);
    }
    registry.jvp.register("stop_gradient", zero_jvp);
    registry.vjp.register("stop_gradient", zero_vjp);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::dims_i64;
    use crate::Builder;
    use evoxc_core::attr::{AttrValue, Attrs};
    use evoxc_core::dtype::DType;
    use evoxc_core::value_type::ValueType;

    fn f32(shape: &[i64]) -> ValueType {
        ValueType::new(DType::parse("float32").unwrap(), dims_i64(shape))
    }

    fn i64_vector(len: i64) -> ValueType {
        ValueType::new(DType::parse("int64").unwrap(), dims_i64(&[len]))
    }

    fn graph(types: &[ValueType]) -> (Builder, Vec<ValueId>) {
        let mut builder = Builder::new("main");
        builder.open_function("main", types).unwrap();
        let args = builder.block_arguments().unwrap();
        (builder, args)
    }

    fn source(
        builder: &Builder,
        name: &str,
        attributes: Attrs,
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
            name: name.to_string(),
            attributes,
            original_operand_types: types(operands),
            original_result_types: types(results),
            operands: operands.to_vec(),
            operand_types: Vec::new(),
            results: results.to_vec(),
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

    fn slice_attrs(starts: &[i64], limits: &[i64], strides: &[i64]) -> Attrs {
        Attrs::from_pairs([
            ("start_indices", AttrValue::Ints(starts.to_vec())),
            ("limit_indices", AttrValue::Ints(limits.to_vec())),
            ("strides", AttrValue::Ints(strides.to_vec())),
        ])
    }

    fn emitted_names(ctx: &RuleCtx<'_>, from: usize) -> Vec<String> {
        ctx.builder.block_op_ids().unwrap()[from..]
            .iter()
            .map(|id| ctx.builder.program().op(*id).unwrap().name.clone())
            .collect()
    }

    #[test]
    fn reduce_sum_jvp_and_vjp_round_trip() {
        let (mut builder, args) = graph(&[f32(&[2, 3])]);
        let x = args[0];
        let attrs = reduce_attrs(&[1], false, "sum");
        let reduced = builder.emit_typed_one("reduce_sum", &[x], &attrs).unwrap();
        assert_eq!(builder.value_type(reduced).unwrap().shape, dims_i64(&[2]));
        let op = source(&builder, "reduce_sum", attrs, &[x], &[reduced]);

        // JVP: the tangent is reduced over the same axis.
        let mut ctx = RuleCtx::new(&mut builder);
        let out = reduction_jvp(&mut ctx, &op, &[Dual::Value(x)]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(ctx.shape(out[0].value().unwrap()).unwrap(), dims_i64(&[2]));

        // A zero tangent short-circuits.
        let mut ctx = RuleCtx::new(&mut builder);
        assert_eq!(
            reduction_jvp(&mut ctx, &op, &[Dual::ZERO]).unwrap(),
            vec![Dual::ZERO]
        );

        // VJP: the (2,) cotangent broadcasts back to the (2, 3) operand.
        let mut ctx = RuleCtx::new(&mut builder);
        let out = reduction_vjp(&mut ctx, &op, &[Dual::Value(reduced)]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(
            ctx.shape(out[0].value().unwrap()).unwrap(),
            dims_i64(&[2, 3])
        );
    }

    #[test]
    fn reduce_mean_vjp_weights_by_the_static_count() {
        let (mut builder, args) = graph(&[f32(&[2, 3])]);
        let x = args[0];
        let attrs = reduce_attrs(&[0, 1], true, "mean");
        let mean = builder.emit_typed_one("reduce_mean", &[x], &attrs).unwrap();
        let op = source(&builder, "reduce_mean", attrs, &[x], &[mean]);

        let (shape, names) = {
            let mut ctx = RuleCtx::new(&mut builder);
            let before = ctx.builder.block_op_ids().unwrap().len();
            let out = reduction_vjp(&mut ctx, &op, &[Dual::Value(mean)]).unwrap();
            let grad = out[0].value().unwrap();
            (ctx.shape(grad).unwrap(), emitted_names(&ctx, before))
        };
        assert_eq!(shape, dims_i64(&[2, 3]));
        assert!(names.iter().any(|name| name == "divide"), "{names:?}");
    }

    #[test]
    fn mean_vjp_defers_on_a_symbolic_reduced_dim() {
        let symbolic = ValueType::new(
            DType::parse("float32").unwrap(),
            vec![Dim::named("B"), Dim::Int(3)],
        );
        let (mut builder, args) = graph(&[symbolic]);
        let x = args[0];
        let attrs = reduce_attrs(&[0], true, "mean");
        let mean = builder.emit_typed_one("reduce_mean", &[x], &attrs).unwrap();
        let op = source(&builder, "reduce_mean", attrs, &[x], &[mean]);

        let mut ctx = RuleCtx::new(&mut builder);
        let err = reduction_vjp(&mut ctx, &op, &[Dual::Value(mean)]).unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        assert!(
            err.message
                .contains("cannot differentiate 'mean' over symbolic dim"),
            "{}",
            err.message
        );
    }

    #[test]
    fn reduce_max_vjp_selects_on_the_arg_mask() {
        let (mut builder, args) = graph(&[f32(&[2, 3])]);
        let x = args[0];
        let attrs = reduce_attrs(&[1], false, "max");
        let max = builder.emit_typed_one("reduce_max", &[x], &attrs).unwrap();
        let op = source(&builder, "reduce_max", attrs, &[x], &[max]);

        let mut ctx = RuleCtx::new(&mut builder);
        let out = reduction_vjp(&mut ctx, &op, &[Dual::Value(max)]).unwrap();
        assert_eq!(
            ctx.shape(out[0].value().unwrap()).unwrap(),
            dims_i64(&[2, 3])
        );
    }

    #[test]
    fn reshape_and_transpose_vjps_invert_themselves() {
        let (mut builder, args) = graph(&[f32(&[2, 3])]);
        let x = args[0];

        let attrs = Attrs::from_pairs([("shape", AttrValue::Shape(dims_i64(&[3, 2])))]);
        let reshaped = builder.emit_typed_one("reshape", &[x], &attrs).unwrap();
        let op = source(&builder, "reshape", attrs, &[x], &[reshaped]);
        let mut ctx = RuleCtx::new(&mut builder);
        let out = structural_vjp(&mut ctx, &op, &[Dual::Value(reshaped)]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(
            ctx.shape(out[0].value().unwrap()).unwrap(),
            dims_i64(&[2, 3])
        );

        let attrs = Attrs::from_pairs([("permutation", AttrValue::Ints(vec![1, 0]))]);
        let transposed = builder.emit_typed_one("transpose", &[x], &attrs).unwrap();
        assert_eq!(
            builder.value_type(transposed).unwrap().shape,
            dims_i64(&[3, 2])
        );
        let op = source(&builder, "transpose", attrs, &[x], &[transposed]);
        let mut ctx = RuleCtx::new(&mut builder);
        let out = structural_vjp(&mut ctx, &op, &[Dual::Value(transposed)]).unwrap();
        assert_eq!(
            ctx.shape(out[0].value().unwrap()).unwrap(),
            dims_i64(&[2, 3])
        );

        // transpose JVP: the tangent follows the same permutation.
        let mut ctx = RuleCtx::new(&mut builder);
        let out = structural_jvp(&mut ctx, &op, &[Dual::Value(x)]).unwrap();
        assert_eq!(
            ctx.shape(out[0].value().unwrap()).unwrap(),
            dims_i64(&[3, 2])
        );
    }

    #[test]
    fn slice_vjp_pads_back_to_the_operand() {
        let (mut builder, args) = graph(&[f32(&[4, 5])]);
        let x = args[0];
        let attrs = slice_attrs(&[1, 2], &[3, 4], &[1, 1]);
        let sliced = builder.emit_typed_one("slice", &[x], &attrs).unwrap();
        assert_eq!(builder.value_type(sliced).unwrap().shape, dims_i64(&[2, 2]));
        let op = source(&builder, "slice", attrs, &[x], &[sliced]);

        let (shape, names) = {
            let mut ctx = RuleCtx::new(&mut builder);
            let before = ctx.builder.block_op_ids().unwrap().len();
            let out = structural_vjp(&mut ctx, &op, &[Dual::Value(sliced)]).unwrap();
            let grad = out[0].value().unwrap();
            (ctx.shape(grad).unwrap(), emitted_names(&ctx, before))
        };
        assert_eq!(shape, dims_i64(&[4, 5]));
        assert_eq!(
            names.iter().filter(|name| *name == "pad").count(),
            1,
            "{names:?}"
        );
    }

    #[test]
    fn strided_slice_vjp_scatters_into_the_axis() {
        let (mut builder, args) = graph(&[f32(&[6])]);
        let x = args[0];
        let attrs = slice_attrs(&[0], &[6], &[2]);
        let sliced = builder.emit_typed_one("slice", &[x], &attrs).unwrap();
        assert_eq!(builder.value_type(sliced).unwrap().shape, dims_i64(&[3]));
        let op = source(&builder, "slice", attrs, &[x], &[sliced]);

        let (shape, names) = {
            let mut ctx = RuleCtx::new(&mut builder);
            let before = ctx.builder.block_op_ids().unwrap().len();
            let out = structural_vjp(&mut ctx, &op, &[Dual::Value(sliced)]).unwrap();
            let grad = out[0].value().unwrap();
            (ctx.shape(grad).unwrap(), emitted_names(&ctx, before))
        };
        assert_eq!(shape, dims_i64(&[6]));
        assert!(names.iter().any(|name| name == "scatter"), "{names:?}");
    }

    #[test]
    fn pad_vjp_slices_the_padding_away_and_jvp_pads_the_tangent() {
        let (mut builder, args) = graph(&[f32(&[2, 3])]);
        let x = args[0];
        let attrs = Attrs::from_pairs([
            (
                "padding_config",
                AttrValue::NestedInts(vec![vec![1, 0], vec![0, 2]]),
            ),
            ("value", AttrValue::Float(0.5)),
        ]);
        let padded = builder.emit_typed_one("pad", &[x], &attrs).unwrap();
        assert_eq!(builder.value_type(padded).unwrap().shape, dims_i64(&[3, 5]));
        let op = source(&builder, "pad", attrs, &[x], &[padded]);

        let mut ctx = RuleCtx::new(&mut builder);
        let out = structural_vjp(&mut ctx, &op, &[Dual::Value(padded)]).unwrap();
        assert_eq!(
            ctx.shape(out[0].value().unwrap()).unwrap(),
            dims_i64(&[2, 3])
        );

        let mut ctx = RuleCtx::new(&mut builder);
        let out = structural_jvp(&mut ctx, &op, &[Dual::Value(x)]).unwrap();
        assert_eq!(
            ctx.shape(out[0].value().unwrap()).unwrap(),
            dims_i64(&[3, 5])
        );
    }

    #[test]
    fn concatenate_vjp_splits_per_operand() {
        let (mut builder, args) = graph(&[f32(&[2, 3]), f32(&[2, 4])]);
        let attrs = Attrs::from_pairs([("axis", AttrValue::Int(1))]);
        let joined = builder
            .emit_typed_one("concatenate", &args, &attrs)
            .unwrap();
        assert_eq!(builder.value_type(joined).unwrap().shape, dims_i64(&[2, 7]));
        let op = source(&builder, "concatenate", attrs, &args, &[joined]);

        let mut ctx = RuleCtx::new(&mut builder);
        let out = structural_vjp(&mut ctx, &op, &[Dual::Value(joined)]).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(
            ctx.shape(out[0].value().unwrap()).unwrap(),
            dims_i64(&[2, 3])
        );
        assert_eq!(
            ctx.shape(out[1].value().unwrap()).unwrap(),
            dims_i64(&[2, 4])
        );
    }

    #[test]
    fn gather_jvp_gathers_the_tangent_and_vjp_scatters_it() {
        let (mut builder, args) = graph(&[f32(&[2, 3]), i64_vector(2)]);
        let (x, indices) = (args[0], args[1]);
        let attrs = Attrs::from_pairs([("axes", AttrValue::Ints(vec![1]))]);
        let gathered = builder
            .emit_typed_one("gather", &[x, indices], &attrs)
            .unwrap();
        assert_eq!(
            builder.value_type(gathered).unwrap().shape,
            dims_i64(&[2, 2])
        );
        let op = source(&builder, "gather", attrs, &[x, indices], &[gathered]);

        let mut ctx = RuleCtx::new(&mut builder);
        let out = structural_jvp(&mut ctx, &op, &[Dual::Value(x), Dual::ZERO]).unwrap();
        assert_eq!(
            ctx.shape(out[0].value().unwrap()).unwrap(),
            dims_i64(&[2, 2])
        );

        // The indices are not differentiable: the mapping stays false.
        let mut ctx = RuleCtx::new(&mut builder);
        let out = structural_jvp(&mut ctx, &op, &[Dual::ZERO, Dual::ZERO]).unwrap();
        assert_eq!(out, vec![Dual::ZERO]);

        let mut ctx = RuleCtx::new(&mut builder);
        let out = structural_vjp(&mut ctx, &op, &[Dual::Value(gathered)]).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(
            ctx.shape(out[0].value().unwrap()).unwrap(),
            dims_i64(&[2, 3])
        );
        assert_eq!(out[1], Dual::ZERO);
    }

    #[test]
    fn scatter_vjp_gathers_the_updates_gradient() {
        let (mut builder, args) = graph(&[f32(&[6]), i64_vector(2), f32(&[2])]);
        let attrs = Attrs::from_pairs([("axis", AttrValue::Int(0))]);
        let scattered = builder.emit_typed_one("scatter", &args, &attrs).unwrap();
        let op = source(&builder, "scatter", attrs, &args, &[scattered]);

        let mut ctx = RuleCtx::new(&mut builder);
        let out = structural_vjp(&mut ctx, &op, &[Dual::Value(scattered)]).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(ctx.shape(out[0].value().unwrap()).unwrap(), dims_i64(&[6]));
        assert_eq!(out[1], Dual::ZERO);
        assert_eq!(ctx.shape(out[2].value().unwrap()).unwrap(), dims_i64(&[2]));
    }

    #[test]
    fn non_differentiable_ops_yield_zero_duals() {
        let (mut builder, args) = graph(&[f32(&[2, 3]), f32(&[2, 3])]);
        let compared = builder.binary("greater", args[0], args[1]).unwrap();
        let op = source(&builder, "greater", Attrs::new(), &args, &[compared]);

        let mut ctx = RuleCtx::new(&mut builder);
        assert_eq!(
            zero_jvp(&mut ctx, &op, &[Dual::Value(args[0]), Dual::Value(args[1])]).unwrap(),
            vec![Dual::ZERO]
        );
        assert_eq!(
            zero_vjp(&mut ctx, &op, &[Dual::ZERO]).unwrap(),
            vec![Dual::ZERO, Dual::ZERO]
        );

        // Everything is registered under the names the dispatcher looks up.
        let mut registry = Registry::default();
        register_rules(&mut registry);
        for name in REDUCTION_OPS
            .iter()
            .chain(STRUCTURAL_OPS)
            .chain(NONDIFFERENTIABLE_OPS)
            .copied()
            .chain(["stop_gradient"])
        {
            assert!(registry.jvp.contains(name), "jvp {name}");
            assert!(registry.vjp.contains(name), "vjp {name}");
        }
    }

    #[test]
    fn deferrals_are_explicit_transform_errors() {
        // multi-axis gather JVP
        let (mut builder, args) = graph(&[
            ValueType::new(DType::parse("float32").unwrap(), dims_i64(&[5, 4, 3])),
            i64_vector(2),
        ]);
        let attrs = Attrs::from_pairs([("axes", AttrValue::Ints(vec![0, 1]))]);
        let gathered = builder.emit_typed_one("gather", &args, &attrs).unwrap();
        let op = source(&builder, "gather", attrs, &args, &[gathered]);
        let mut ctx = RuleCtx::new(&mut builder);
        let err = structural_jvp(&mut ctx, &op, &[Dual::Value(args[0]), Dual::ZERO]).unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        assert!(
            err.message
                .contains("jvp: cannot differentiate 'gather' with multiple axes"),
            "{}",
            err.message
        );

        // a slice with a stride on two axes cannot be inverted in v1
        let (mut builder, args) = graph(&[f32(&[6, 6])]);
        let x = args[0];
        let attrs = slice_attrs(&[0, 0], &[6, 6], &[2, 2]);
        let sliced = builder.emit_typed_one("slice", &[x], &attrs).unwrap();
        let op = source(&builder, "slice", attrs, &[x], &[sliced]);
        let mut ctx = RuleCtx::new(&mut builder);
        let err = structural_vjp(&mut ctx, &op, &[Dual::Value(sliced)]).unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        assert!(
            err.message
                .contains("cannot invert a slice with strided entries over multiple axes"),
            "{}",
            err.message
        );
    }
}
