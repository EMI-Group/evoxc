//! Builtin batching rules for structural ops.
//!
//! Ports `_reshape_batching`, `_transpose_batching`, `_slice_batching`,
//! `_pad_batching`, `_concatenate_batching`, `_gather_batching`,
//! `_scatter_batching` and `_constant_batching` from
//! `evoxc/transforms/rules.py`.
//!
//! Every rule receives the op's *rewritten* operands (values of the output
//! module) plus one [`MappedAxes`] per operand, and returns one rewritten
//! value + one [`MappedAxes`] per source result.  Mapped (batch) axes are
//! always the leading ones: a structural op rewrites the PER-ROW dims only and
//! lets the leading batch prefix pass through untouched.
//!
//! The frontend/IR API differences this file bridges (deliberately, never a
//! silent translation):
//!
//! * Python `ops.slice(x, starts, lengths, strides)` passes SIZES while the IR
//!   `slice` op carries `start_indices` / `limit_indices`;
//! * Python `ops.gather(x, indices, axis=a)` is the IR `gather` with the
//!   `axes` list `[a]`;
//! * `padding_config` is one `(low, high)` pair per axis;
//! * `transpose`'s `permutation: null` means full reversal.

use evoxc_core::attr::AttrValue;
use evoxc_core::dim::Dim;
use evoxc_core::error::{Error, Result};
use evoxc_core::graph::ValueId;

use super::common::fold_product;
use super::MappedAxes;
use crate::emit::{RuleCtx, RuleOp};
use crate::vectorize::format_shape;
use crate::Registry;

/// Register this file's batching rules.
pub fn register_rules(registry: &mut Registry) {
    registry.batching.register("reshape", reshape_rule);
    registry.batching.register("transpose", transpose_rule);
    registry.batching.register("slice", slice_rule);
    registry.batching.register("pad", pad_rule);
    registry.batching.register("concatenate", concatenate_rule);
    registry.batching.register("gather", gather_rule);
    registry.batching.register("scatter", scatter_rule);
    registry.batching.register("constant", constant_rule);
}

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

/// The single operand value of a one-operand rule (`TransformError` otherwise).
fn sole_operand(op: &RuleOp) -> Result<ValueId> {
    op.operands.first().copied().ok_or_else(|| {
        Error::transform(format!(
            "vectorize: op '{}' has no operand to batch",
            op.name
        ))
    })
}

/// One `MappedAxes` per operand (`TransformError` when the metadata does not
/// align with the operands).
fn operand_axes(op: &RuleOp, axes: &[MappedAxes], index: usize) -> Result<MappedAxes> {
    axes.get(index).copied().ok_or_else(|| {
        Error::transform(format!(
            "vectorize: rule metadata mismatch for op '{}': {} operand axes for {} operand(s)",
            op.name,
            axes.len(),
            op.operands.len()
        ))
    })
}

/// Per-axis int attribute: an `int` broadcasts to every axis, a list must have
/// one entry per (per-row) axis, absent/`null` uses `default`.
fn per_axis_ints(op: &RuleOp, name: &str, rank: usize, default: i64) -> Result<Vec<i64>> {
    match op.attributes.get(name) {
        None | Some(AttrValue::Null) => Ok(vec![default; rank]),
        Some(AttrValue::Int(value)) => Ok(vec![*value; rank]),
        Some(AttrValue::Ints(values)) => {
            if values.len() != rank {
                return Err(Error::transform(format!(
                    "vectorize: cannot batch op '{}': {} has {} entr(ies) for a per-row rank of \
                     {rank} (v1 gap)",
                    op.name,
                    name,
                    values.len()
                )));
            }
            Ok(values.clone())
        }
        Some(other) => Err(Error::transform(format!(
            "vectorize: cannot batch op '{}': {} must be an int or a tuple of {rank} ints, got {}",
            op.name,
            name,
            other.type_name()
        ))),
    }
}

/// Per-axis optional int attribute (`None` entries mean "the whole dim", the
/// IR's `limit_indices: null`).
fn per_axis_optional_ints(op: &RuleOp, name: &str, rank: usize) -> Result<Vec<Option<i64>>> {
    match op.attributes.get(name) {
        None | Some(AttrValue::Null) => Ok(vec![None; rank]),
        Some(AttrValue::Int(value)) => Ok(vec![Some(*value); rank]),
        Some(AttrValue::Ints(values)) => {
            if values.len() != rank {
                return Err(Error::transform(format!(
                    "vectorize: cannot batch op '{}': {} has {} entr(ies) for a per-row rank of \
                     {rank} (v1 gap)",
                    op.name,
                    name,
                    values.len()
                )));
            }
            Ok(values.iter().map(|value| Some(*value)).collect())
        }
        Some(other) => Err(Error::transform(format!(
            "vectorize: cannot batch op '{}': {} must be an int or a tuple of {rank} ints, got {}",
            op.name,
            name,
            other.type_name()
        ))),
    }
}

/// Normalize `axis` against `rank` (`TransformError` when out of range).
fn normalized_axis(op: &RuleOp, axis: i64, rank: usize) -> Result<usize> {
    let rank = rank as i64;
    let normalized = if axis < 0 { axis + rank } else { axis };
    if normalized < 0 || normalized >= rank {
        return Err(Error::transform(format!(
            "vectorize: cannot batch op '{}' with axis {axis} on a rank-{rank} operand",
            op.name
        )));
    }
    Ok(normalized as usize)
}

/// The `arange(start, limit, stride)` element count (`stride > 0`).
fn arange_len(start: i64, limit: i64, stride: i64) -> i64 {
    if stride <= 0 || limit <= start {
        0
    } else {
        (limit - start + stride - 1) / stride
    }
}

// ---------------------------------------------------------------------------
// reshape / transpose
// ---------------------------------------------------------------------------

/// `_reshape_batching`: mapped dims must be preserved — the new target is
/// `operand batch dims + the original target shape` (the `-1` wildcard still
/// resolves against the batched element count, so it never swallows the batch
/// extent).
fn reshape_rule(
    ctx: &mut RuleCtx<'_>,
    op: &RuleOp,
    axes: &[MappedAxes],
) -> Result<(Vec<ValueId>, Vec<MappedAxes>)> {
    let value = sole_operand(op)?;
    let mapped = operand_axes(op, axes, 0)?;
    let count = mapped.count();
    let original_target = op.shape_attr("shape")?.to_vec();
    let mut target: Vec<Dim> = Vec::with_capacity(count + original_target.len());
    if count > 0 {
        let shape = ctx.shape(value)?;
        if count > shape.len() {
            return Err(Error::transform(format!(
                "vectorize: reshape operand of rank {} cannot carry {count} mapped axes",
                shape.len()
            )));
        }
        target.extend(shape[..count].iter().cloned());
    }
    target.extend(original_target);
    let out = ctx.reshape(value, &target)?;
    Ok((vec![out], vec![mapped]))
}

/// `_transpose_batching`: the mapped leading axes stay leading; the source
/// permutation (`None` = full reversal) applies to the unvectorized axes.
///
/// The permutation is always materialized explicitly: `null` means "reverse"
/// in the IR, and the reversal is computed from the source operand rank (the
/// batch prefix of a mapped operand is the identity either way).
fn transpose_rule(
    ctx: &mut RuleCtx<'_>,
    op: &RuleOp,
    axes: &[MappedAxes],
) -> Result<(Vec<ValueId>, Vec<MappedAxes>)> {
    let value = sole_operand(op)?;
    let mapped = operand_axes(op, axes, 0)?;
    let count = mapped.count();
    let old_rank = op.original_operand(0)?.rank();
    let base: Vec<i64> = match op.opt_ints("permutation")? {
        None => (0..old_rank as i64).rev().collect(),
        Some(permutation) => permutation.to_vec(),
    };
    let mut permutation: Vec<i64> = (0..count as i64).collect();
    permutation.extend(base.iter().map(|axis| axis + count as i64));
    let out = ctx.transpose(value, Some(&permutation))?;
    Ok((vec![out], vec![mapped]))
}

// ---------------------------------------------------------------------------
// slice / pad / concatenate
// ---------------------------------------------------------------------------

/// `_slice_batching`: static int `limit_indices` cannot pass through the
/// symbolic batch dims, so a mapped operand is lowered to per-axis `gather`s
/// over static int64 index constants (batch dims untouched).
///
/// The operand is first flattened to ONE batch axis, each sliced axis is
/// gathered with its index constant (shifted by that one batch axis), and the
/// result is reshaped back to `batch dims + sliced per-row dims`.  A
/// full-range stride-1 slice over a static dim is skipped (the gather would be
/// the identity); an unmapped operand rebuilds the `slice` op unchanged.
fn slice_rule(
    ctx: &mut RuleCtx<'_>,
    op: &RuleOp,
    axes: &[MappedAxes],
) -> Result<(Vec<ValueId>, Vec<MappedAxes>)> {
    let value = sole_operand(op)?;
    let mapped = operand_axes(op, axes, 0)?;
    let count = mapped.count();
    if count == 0 {
        // Unmapped: the IR's start/limit/strides attributes are rebuilt
        // explicitly (Python's `ops.slice` passes sizes, the IR carries limits).
        let rank = ctx.rank(value)?;
        let starts = per_axis_ints(op, "start_indices", rank, 0)?;
        let limits = per_axis_ints(op, "limit_indices", rank, 0)?;
        let strides = per_axis_ints(op, "strides", rank, 1)?;
        let out = ctx.slice(value, &starts, &limits, Some(&strides))?;
        return Ok((vec![out], vec![mapped]));
    }
    let shape = ctx.shape(value)?;
    if count > shape.len() {
        return Err(Error::transform(format!(
            "vectorize: slice operand of rank {} cannot carry {count} mapped axes",
            shape.len()
        )));
    }
    let rank = shape.len() - count;
    let starts = per_axis_ints(op, "start_indices", rank, 0)?;
    let limits = per_axis_optional_ints(op, "limit_indices", rank)?;
    let strides = per_axis_ints(op, "strides", rank, 1)?;
    let batch: Vec<Dim> = shape[..count].to_vec();
    let dims: Vec<Dim> = shape[count..].to_vec();
    let Some(batch_product) = fold_product(&batch) else {
        return Err(Error::transform(format!(
            "vectorize: cannot batch op 'slice' with a dynamic batch dim in {} (v1 gap)",
            format_shape(&batch)
        )));
    };
    let mut flat: Vec<Dim> = Vec::with_capacity(dims.len() + 1);
    flat.push(batch_product);
    flat.extend(dims.iter().cloned());
    let mut lowered = ctx.reshape(value, &flat)?;
    let mut sliced: Vec<Dim> = Vec::with_capacity(rank);
    for index in 0..rank {
        let dim = &dims[index];
        let start = starts[index];
        let stride = strides[index];
        let full_range = stride == 1
            && start == 0
            && match limits[index] {
                Some(limit) => dim.as_int() == Some(limit),
                None => true,
            };
        if full_range {
            sliced.push(dim.clone()); // gather would be the identity
            continue;
        }
        let Some(limit) = limits[index] else {
            return Err(Error::transform(format!(
                "vectorize: cannot batch op 'slice' with a symbolic limit on axis {index} (v1 gap)"
            )));
        };
        let indices = ctx.arange_i64(start, limit, stride)?;
        lowered = ctx.gather(lowered, indices, &[(index + 1) as i64])?;
        sliced.push(Dim::Int(arange_len(start, limit, stride)));
    }
    let mut target = batch;
    target.extend(sliced);
    let out = ctx.reshape(lowered, &target)?;
    Ok((vec![out], vec![mapped]))
}

/// `_pad_batching`: insert leading zero-padding entries for the mapped dims
/// (the batch prefix is never padded).
fn pad_rule(
    ctx: &mut RuleCtx<'_>,
    op: &RuleOp,
    axes: &[MappedAxes],
) -> Result<(Vec<ValueId>, Vec<MappedAxes>)> {
    let value = sole_operand(op)?;
    let mapped = operand_axes(op, axes, 0)?;
    let count = mapped.count();
    let mut config: Vec<(i64, i64)> = vec![(0, 0); count];
    for entry in op.nested_ints("padding_config")? {
        config.push(match entry.as_slice() {
            [both] => (*both, *both),
            [low, high] => (*low, *high),
            _ => {
                return Err(Error::transform(format!(
                    "vectorize: cannot batch op 'pad' with padding entry {entry:?} (expected an \
                     int or a (low, high) pair)"
                )))
            }
        });
    }
    let fill = match op.attributes.get("value") {
        Some(_) => op.float("value")?,
        None => 0.0,
    };
    let out = ctx.pad(value, &config, fill)?;
    Ok((vec![out], vec![mapped]))
}

/// `_concatenate_batching`: broadcast operands missing mapped axes up to the
/// batch dims (numpy concatenation does NOT broadcast), then shift `axis` by
/// the mapped count.
fn concatenate_rule(
    ctx: &mut RuleCtx<'_>,
    op: &RuleOp,
    axes: &[MappedAxes],
) -> Result<(Vec<ValueId>, Vec<MappedAxes>)> {
    if op.operands.len() != axes.len() {
        return Err(Error::transform(format!(
            "vectorize: rule metadata mismatch for op '{}': {} operand axes for {} operand(s)",
            op.name,
            axes.len(),
            op.operands.len()
        )));
    }
    let counts: Vec<usize> = axes.iter().map(|axes| axes.count()).collect();
    let mapped_count = counts.iter().copied().max().unwrap_or(0);
    let axis = op.int("axis")?;
    if mapped_count == 0 {
        let out = ctx.concatenate(&op.operands, axis)?;
        return Ok((vec![out], vec![MappedAxes::UNMAPPED]));
    }
    // The batch dims come from the first operand that carries the full prefix
    // (one has to — `mapped_count` is the maximum).
    let mut batch_dims: Option<Vec<Dim>> = None;
    for (value, count) in op.operands.iter().zip(&counts) {
        if *count == mapped_count {
            batch_dims = Some(ctx.shape(*value)?[..mapped_count].to_vec());
            break;
        }
    }
    let Some(batch_dims) = batch_dims else {
        return Err(Error::transform(format!(
            "vectorize: cannot batch op 'concatenate': no operand carries all {mapped_count} \
             mapped axes"
        )));
    };
    let mut values: Vec<ValueId> = Vec::with_capacity(op.operands.len());
    for (value, count) in op.operands.iter().zip(&counts) {
        if *count < mapped_count {
            let mut target = batch_dims.clone();
            target.extend(ctx.shape(*value)?);
            values.push(ctx.broadcast(*value, &target)?);
        } else {
            values.push(*value);
        }
    }
    let out = ctx.concatenate(&values, axis + mapped_count as i64)?;
    Ok((vec![out], vec![MappedAxes::new(mapped_count)]))
}

// ---------------------------------------------------------------------------
// gather / scatter
// ---------------------------------------------------------------------------

/// `_gather_batching`: remove ALL mapped axes plus the data axis with one
/// multi-axis `gather` (`infer_gather` splices `indices.shape` wholesale at the
/// smallest removed axis — the indices' batch dims double as the removed data
/// batch axes, exactly per-row semantics), then transpose the index dims into
/// the data axis's position and reshape back to the original index shape.
///
/// Mapped data with unmapped indices and a multi-axis `gather` are v1
/// deferrals (explicit `TransformError`, never a silent fallback).
fn gather_rule(
    ctx: &mut RuleCtx<'_>,
    op: &RuleOp,
    axes: &[MappedAxes],
) -> Result<(Vec<ValueId>, Vec<MappedAxes>)> {
    let data = op.operands[0];
    let indices = op.operands[1];
    let data_count = axes[0].count();
    let index_count = axes[1].count();
    // `axes` is an optional IR attribute whose schema default is `[0]`.
    let attr_axes: Vec<i64> = match op.opt_ints("axes")? {
        None => vec![0],
        Some(axes) => axes.to_vec(),
    };
    if attr_axes.len() != 1 {
        return Err(Error::transform(
            "vectorize: cannot batch op 'gather' with multiple axes (v1 gap)",
        ));
    }
    if data_count == 0 && index_count == 0 {
        let out = ctx.gather(data, indices, &attr_axes)?;
        return Ok((vec![out], vec![MappedAxes::UNMAPPED]));
    }
    if data_count > 0 && index_count != data_count {
        return Err(Error::transform(
            "vectorize: cannot batch op 'gather' with mapped data but unmapped indices \
             (v1 deferral: index broadcasting is not supported)",
        ));
    }
    let mapped_count = data_count.max(index_count);
    let original_shape = op.original_operand(0)?.shape.clone();
    let axis = normalized_axis(op, attr_axes[0], original_shape.len())?;
    let mut data = data;
    if data_count < mapped_count {
        // Unmapped data: pad it with size-one batch dims, then broadcast the
        // actual batch dims the indices carry.
        let index_shape = ctx.shape(indices)?;
        let batch_dims: Vec<Dim> = index_shape[..mapped_count].to_vec();
        let mut target: Vec<Dim> = vec![Dim::Int(1); mapped_count - data_count];
        target.extend(original_shape.iter().cloned());
        data = ctx.broadcast(data, &target)?;
        let mut target = batch_dims;
        target.extend(original_shape.iter().cloned());
        data = ctx.broadcast(data, &target)?;
    }
    // Merge the index dims beyond the batch dims into one axis so the indices
    // rank matches the removed-axes count (mapped + 1); a batch-only index
    // shape (B,) becomes (B, 1).
    let index_shape = ctx.shape(indices)?;
    let index_dims: Vec<Dim> = index_shape[mapped_count.min(index_shape.len())..].to_vec();
    let Some(merged_index) = fold_product(&index_dims) else {
        return Err(Error::transform(
            "vectorize: cannot batch op 'gather' with dynamic index dims (v1 gap)",
        ));
    };
    let mut indices = indices;
    if merged_index.as_int() != Some(1) || index_dims.is_empty() {
        let mut target: Vec<Dim> = index_shape[..mapped_count].to_vec();
        target.push(merged_index);
        indices = ctx.reshape(indices, &target)?;
    }
    // Remove the mapped axes plus the (shifted) data axis in one gather.
    let mut removed: Vec<i64> = (0..mapped_count as i64).collect();
    removed.push(axis as i64 + mapped_count as i64);
    let gathered = ctx.gather(data, indices, &removed)?;
    // gathered = (batch..., J, prefix..., suffix...): move J into the data
    // axis's position — (batch..., prefix..., J, suffix...).
    let rank = ctx.rank(gathered)?;
    let mut permutation: Vec<i64> = (0..mapped_count as i64).collect();
    for offset in 0..axis {
        permutation.push(mapped_count as i64 + 1 + offset as i64);
    }
    permutation.push(mapped_count as i64);
    for rest in (mapped_count + 1 + axis)..rank {
        permutation.push(rest as i64);
    }
    let reordered = ctx.transpose(gathered, Some(&permutation))?;
    // Restore the original multi-dim index shape (drop the size-one J axis for
    // batch-only indices).
    let reordered_shape = ctx.shape(reordered)?;
    let mut target: Vec<Dim> = reordered_shape[..mapped_count].to_vec();
    target.extend(original_shape[..axis].iter().cloned());
    target.extend(index_dims.iter().cloned());
    target.extend(original_shape[axis + 1..].iter().cloned());
    let value = if reordered_shape == target {
        reordered
    } else {
        ctx.reshape(reordered, &target)?
    };
    Ok((vec![value], vec![MappedAxes::new(mapped_count)]))
}

/// `_scatter_batching`: shift `axis` by the mapped count; an unmapped data
/// operand is broadcast up to the batch dims the indices carry.
///
/// Disagreeing mappedness (mapped data with unmapped indices/updates, or
/// indices and updates disagreeing) is an explicit v1 `TransformError`.
fn scatter_rule(
    ctx: &mut RuleCtx<'_>,
    op: &RuleOp,
    axes: &[MappedAxes],
) -> Result<(Vec<ValueId>, Vec<MappedAxes>)> {
    let data = op.operands[0];
    let indices = op.operands[1];
    let updates = op.operands[2];
    let data_count = axes[0].count();
    let index_count = axes[1].count();
    let update_count = axes[2].count();
    // `axis` is an optional IR attribute whose schema default is `0`.
    let axis = op.opt_int("axis")?.unwrap_or(0);
    if data_count == 0 && index_count == 0 && update_count == 0 {
        let out = ctx.scatter(data, indices, updates, axis)?;
        return Ok((vec![out], vec![MappedAxes::UNMAPPED]));
    }
    let mapped_count = data_count.max(index_count).max(update_count);
    if data_count > 0 && (index_count != data_count || update_count != data_count) {
        return Err(Error::transform(
            "vectorize: cannot batch op 'scatter' when the data operand is mapped but \
             indices/updates are not (v1 deferral)",
        ));
    }
    if data_count == 0 && index_count != update_count {
        return Err(Error::transform(
            "vectorize: cannot batch op 'scatter' when indices and updates disagree on mapped \
             axes (v1 deferral)",
        ));
    }
    let mut data = data;
    if data_count < mapped_count {
        let original_shape = ctx.shape(data)?;
        let batch_dims: Vec<Dim> = ctx.shape(indices)?[..mapped_count].to_vec();
        let mut target: Vec<Dim> = vec![Dim::Int(1); mapped_count - data_count];
        target.extend(original_shape.iter().cloned());
        data = ctx.broadcast(data, &target)?;
        let mut target = batch_dims;
        target.extend(original_shape.iter().cloned());
        data = ctx.broadcast(data, &target)?;
    }
    let out = ctx.scatter(data, indices, updates, axis + mapped_count as i64)?;
    Ok((vec![out], vec![MappedAxes::new(mapped_count)]))
}

// ---------------------------------------------------------------------------
// constant
// ---------------------------------------------------------------------------

/// `_constant_batching`: rebuild the `constant` op with the SAME payload; the
/// result is UNMAPPED (the data is shared across the batch, so it carries no
/// mapped dims — the per-row broadcast happens at the consumers, e.g. the
/// pointwise alignment).
fn constant_rule(
    ctx: &mut RuleCtx<'_>,
    op: &RuleOp,
    _axes: &[MappedAxes],
) -> Result<(Vec<ValueId>, Vec<MappedAxes>)> {
    let payload = op.attributes.array("value")?.clone();
    let out = ctx.constant(payload)?;
    Ok((vec![out], vec![MappedAxes::UNMAPPED]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::dims_i64;
    use crate::Builder;
    use evoxc_core::attr::Attrs;
    use evoxc_core::dtype::DType;
    use evoxc_core::value_type::ValueType;

    fn f32(shape: &[Dim]) -> ValueType {
        ValueType::new(DType::parse("float32").unwrap(), shape.to_vec())
    }

    fn batch() -> Dim {
        Dim::named("batch")
    }

    fn batched(shape: &[i64]) -> ValueType {
        let mut dims = vec![batch()];
        dims.extend(dims_i64(shape));
        f32(&dims)
    }

    fn gather_axes(axes: &[i64]) -> Attrs {
        Attrs::from_pairs([("axes", AttrValue::Ints(axes.to_vec()))])
    }

    fn builder_with(types: &[ValueType]) -> (Builder, Vec<ValueId>) {
        let mut builder = Builder::new("main");
        builder.open_function("main", types).unwrap();
        let args = builder.block_arguments().unwrap();
        (builder, args)
    }

    fn rule_op(
        builder: &Builder,
        name: &str,
        attrs: Attrs,
        source_operands: &[ValueType],
        source_results: &[ValueType],
        operands: &[ValueId],
    ) -> RuleOp {
        RuleOp {
            name: name.to_string(),
            attributes: attrs,
            original_operand_types: source_operands.to_vec(),
            original_result_types: source_results.to_vec(),
            operand_types: builder.value_types(operands).unwrap(),
            operands: operands.to_vec(),
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

    #[test]
    fn every_structural_op_is_registered() {
        let mut registry = Registry::default();
        register_rules(&mut registry);
        let names = ["reshape", "transpose", "slice", "pad", "concatenate", "gather", "scatter", "constant"];
        for name in names {
            assert!(registry.batching.contains(name), "op '{name}' has no batching rule");
        }
    }

    #[test]
    fn reshape_keeps_the_batch_prefix_and_resolves_the_wildcard() {
        let (mut builder, args) = builder_with(&[batched(&[4, 3])]);
        let source = rule_op(
            &builder,
            "reshape",
            Attrs::from_pairs([("shape", AttrValue::Shape(dims_i64(&[-1, 3])))]),
            &[f32(&dims_i64(&[4, 3]))],
            &[f32(&dims_i64(&[4, 3]))],
            &args,
        );
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let (values, axes) = reshape_rule(&mut ctx, &source, &[MappedAxes::new(1)]).unwrap();
        assert_eq!(axes, vec![MappedAxes::new(1)]);
        assert_eq!(ctx.shape(values[0]).unwrap(), vec![batch(), Dim::Int(4), Dim::Int(3)]);
        assert_eq!(op_names(&ctx), vec!["reshape"]);
    }

    #[test]
    fn transpose_keeps_mapped_axes_leading() {
        // source operand f32[2,3,4] transposed with permutation (2,0,1)
        let (mut builder, args) = builder_with(&[batched(&[2, 3, 4])]);
        let source = rule_op(
            &builder,
            "transpose",
            Attrs::from_pairs([("permutation", AttrValue::Ints(vec![2, 0, 1]))]),
            &[f32(&dims_i64(&[2, 3, 4]))],
            &[f32(&dims_i64(&[4, 2, 3]))],
            &args,
        );
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let (values, axes) = transpose_rule(&mut ctx, &source, &[MappedAxes::new(1)]).unwrap();
        assert_eq!(axes, vec![MappedAxes::new(1)]);
        // batch + (2,0,1) shifted by one → (batch, 4, 2, 3)
        assert_eq!(ctx.shape(values[0]).unwrap(), vec![batch(), Dim::Int(4), Dim::Int(2), Dim::Int(3)]);
        let op = &ctx.builder.program().ops.iter().next().unwrap().1;
        assert_eq!(op.name, "transpose");
        assert_eq!(op.attributes.ints("permutation").unwrap(), &[0, 3, 1, 2]);
        // an unmapped operand keeps the `permutation: null` = full reversal meaning
        let (mut builder, args) = builder_with(&[f32(&dims_i64(&[2, 3]))]);
        let source = rule_op(&builder, "transpose", Attrs::new(), &[f32(&dims_i64(&[2, 3]))], &[f32(&dims_i64(&[3, 2]))], &args);
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let (values, axes) = transpose_rule(&mut ctx, &source, &[MappedAxes::UNMAPPED]).unwrap();
        assert_eq!(axes, vec![MappedAxes::UNMAPPED]);
        assert_eq!(ctx.shape(values[0]).unwrap(), vec![Dim::Int(3), Dim::Int(2)]);
        assert_eq!(op_names(&ctx), vec!["transpose"]);
    }

    #[test]
    fn slice_lowers_a_mapped_operand_to_per_axis_gathers() {
        // source f32[4,5]: start (1,0), limit (3,4), stride (1,2)
        let (mut builder, args) = builder_with(&[batched(&[4, 5])]);
        let source = rule_op(
            &builder,
            "slice",
            Attrs::from_pairs([
                ("start_indices", AttrValue::Ints(vec![1, 0])),
                ("limit_indices", AttrValue::Ints(vec![3, 4])),
                ("strides", AttrValue::Ints(vec![1, 2])),
            ]),
            &[f32(&dims_i64(&[4, 5]))],
            &[f32(&dims_i64(&[2, 2]))],
            &args,
        );
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let (values, axes) = slice_rule(&mut ctx, &source, &[MappedAxes::new(1)]).unwrap();
        assert_eq!(axes, vec![MappedAxes::new(1)]);
        assert_eq!(
            ctx.shape(values[0]).unwrap(),
            vec![batch(), Dim::Int(2), Dim::Int(2)]
        );
        assert_eq!(
            op_names(&ctx),
            vec![
                "reshape",
                "constant",
                "gather",
                "constant",
                "gather",
                "reshape"
            ]
        );
    }

    #[test]
    fn slice_skips_full_range_axes_and_keeps_unmapped_attrs() {
        // mapped, axis 0 full range (stride 1, static dim) → skipped
        let (mut builder, args) = builder_with(&[batched(&[4, 5])]);
        let source = rule_op(
            &builder,
            "slice",
            Attrs::from_pairs([
                ("start_indices", AttrValue::Ints(vec![0, 1])),
                ("limit_indices", AttrValue::Ints(vec![4, 5])),
                ("strides", AttrValue::Ints(vec![1, 1])),
            ]),
            &[f32(&dims_i64(&[4, 5]))],
            &[f32(&dims_i64(&[4, 4]))],
            &args,
        );
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let (values, axes) = slice_rule(&mut ctx, &source, &[MappedAxes::new(1)]).unwrap();
        assert_eq!(axes, vec![MappedAxes::new(1)]);
        assert_eq!(ctx.shape(values[0]).unwrap(), vec![batch(), Dim::Int(4), Dim::Int(4)]);
        assert_eq!(op_names(&ctx), vec!["reshape", "constant", "gather", "reshape"]);
        // unmapped: the slice is rebuilt with the same start/limit/strides ints
        let (mut builder, args) = builder_with(&[f32(&dims_i64(&[4, 5]))]);
        let source = rule_op(
            &builder,
            "slice",
            Attrs::from_pairs([
                ("start_indices", AttrValue::Ints(vec![0, 1])),
                ("limit_indices", AttrValue::Ints(vec![4, 5])),
            ]),
            &[f32(&dims_i64(&[4, 5]))],
            &[f32(&dims_i64(&[4, 4]))],
            &args,
        );
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let (values, axes) = slice_rule(&mut ctx, &source, &[MappedAxes::UNMAPPED]).unwrap();
        assert_eq!(axes, vec![MappedAxes::UNMAPPED]);
        assert_eq!(ctx.shape(values[0]).unwrap(), vec![Dim::Int(4), Dim::Int(4)]);
        assert_eq!(op_names(&ctx), vec!["slice"]);
        // a dynamic batch dim cannot be folded into the gather plan
        let (mut builder, args) = builder_with(&[f32(&[Dim::Dynamic, Dim::Int(4)])]);
        let source = rule_op(&builder, "slice", Attrs::from_pairs([("start_indices", AttrValue::Ints(vec![1])), ("limit_indices", AttrValue::Ints(vec![3]))]), &[f32(&dims_i64(&[4]))], &[f32(&dims_i64(&[2]))], &args);
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let err = slice_rule(&mut ctx, &source, &[MappedAxes::new(1)]).unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        assert!(err.message.contains("dynamic batch dim"));
    }

    #[test]
    fn pad_inserts_leading_zero_padding() {
        let (mut builder, args) = builder_with(&[batched(&[3])]);
        let source = rule_op(
            &builder,
            "pad",
            Attrs::from_pairs([
                ("padding_config", AttrValue::NestedInts(vec![vec![1, 2]])),
                ("value", AttrValue::Float(0.5)),
            ]),
            &[f32(&dims_i64(&[3]))],
            &[f32(&dims_i64(&[6]))],
            &args,
        );
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let (values, axes) = pad_rule(&mut ctx, &source, &[MappedAxes::new(1)]).unwrap();
        assert_eq!(axes, vec![MappedAxes::new(1)]);
        assert_eq!(ctx.shape(values[0]).unwrap(), vec![batch(), Dim::Int(6)]);
        let op = &ctx.builder.program().ops.iter().next().unwrap().1;
        assert_eq!(op.name, "pad");
        assert_eq!(op.attributes.nested_ints("padding_config").unwrap(), &[vec![0, 0], vec![1, 2]]);
        assert_eq!(op.attributes.f64("value").unwrap(), 0.5);
    }

    #[test]
    fn concatenate_shifts_the_axis_by_the_mapped_count() {
        let (mut builder, args) = builder_with(&[batched(&[2]), f32(&dims_i64(&[2]))]);
        let source = rule_op(
            &builder,
            "concatenate",
            Attrs::from_pairs([("axis", AttrValue::Int(0))]),
            &[f32(&dims_i64(&[2])), f32(&dims_i64(&[2]))],
            &[f32(&dims_i64(&[4]))],
            &args,
        );
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let (values, axes) =
            concatenate_rule(&mut ctx, &source, &[MappedAxes::new(1), MappedAxes::UNMAPPED]).unwrap();
        assert_eq!(axes, vec![MappedAxes::new(1)]);
        assert_eq!(ctx.shape(values[0]).unwrap(), vec![batch(), Dim::Int(4)]);
        assert_eq!(op_names(&ctx), vec!["broadcast", "concatenate"]);
        let op = &ctx.builder.program().ops.iter().last().unwrap().1;
        assert_eq!(op.attributes.i64("axis").unwrap(), 1);
        // no operand carries a mapped axis → the concatenate is unmapped
        let (mut builder, args) = builder_with(&[f32(&dims_i64(&[2])), f32(&dims_i64(&[3]))]);
        let source = rule_op(
            &builder,
            "concatenate",
            Attrs::from_pairs([("axis", AttrValue::Int(0))]),
            &[f32(&dims_i64(&[2])), f32(&dims_i64(&[3]))],
            &[f32(&dims_i64(&[5]))],
            &args,
        );
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let (values, axes) =
            concatenate_rule(&mut ctx, &source, &[MappedAxes::UNMAPPED, MappedAxes::UNMAPPED]).unwrap();
        assert_eq!(axes, vec![MappedAxes::UNMAPPED]);
        assert_eq!(ctx.shape(values[0]).unwrap(), vec![Dim::Int(5)]);
        assert_eq!(op_names(&ctx), vec!["concatenate"]);
    }

    #[test]
    fn gather_with_mapped_indices_and_unmapped_data() {
        // source data f32[4,5], indices i64[2], axis 0: data broadcast up to the batch dims
        let (mut builder, args) = builder_with(&[f32(&dims_i64(&[4, 5])), f32(&[batch(), Dim::Int(2)])]);
        let source = rule_op(&builder, "gather", gather_axes(&[0]), &[f32(&dims_i64(&[4, 5])), f32(&dims_i64(&[2]))], &[f32(&dims_i64(&[2, 5]))], &args);
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let (values, axes) =
            gather_rule(&mut ctx, &source, &[MappedAxes::UNMAPPED, MappedAxes::new(1)]).unwrap();
        assert_eq!(axes, vec![MappedAxes::new(1)]);
        assert_eq!(ctx.shape(values[0]).unwrap(), vec![batch(), Dim::Int(2), Dim::Int(5)]);
        assert_eq!(op_names(&ctx), vec!["broadcast", "broadcast", "reshape", "gather", "transpose"]);
    }

    #[test]
    fn gather_with_fully_mapped_operands_is_per_row() {
        // source data f32[4,5], indices i64[2], both mapped, axis 0
        let (mut builder, args) = builder_with(&[batched(&[4, 5]), batched(&[2])]);
        let source = rule_op(&builder, "gather", gather_axes(&[0]), &[f32(&dims_i64(&[4, 5])), f32(&dims_i64(&[2]))], &[f32(&dims_i64(&[2, 5]))], &args);
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let (values, axes) =
            gather_rule(&mut ctx, &source, &[MappedAxes::new(1), MappedAxes::new(1)]).unwrap();
        assert_eq!(axes, vec![MappedAxes::new(1)]);
        assert_eq!(ctx.shape(values[0]).unwrap(), vec![batch(), Dim::Int(2), Dim::Int(5)]);
        assert_eq!(op_names(&ctx), vec!["reshape", "gather", "transpose"]);
    }

    #[test]
    fn gather_deferrals_are_explicit() {
        let data = f32(&dims_i64(&[4, 5]));
        let index = f32(&dims_i64(&[2]));
        // multi-axis gather
        let (mut builder, args) = builder_with(&[batched(&[4, 5]), batched(&[2])]);
        let source = rule_op(&builder, "gather", gather_axes(&[0, 1]), &[data.clone(), index.clone()], &[f32(&dims_i64(&[2]))], &args);
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let err = gather_rule(&mut ctx, &source, &[MappedAxes::new(1), MappedAxes::new(1)]).unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        assert!(err.message.contains("multiple axes"));
        // mapped data with unmapped indices
        let (mut builder, args) = builder_with(&[batched(&[4, 5]), index.clone()]);
        let source = rule_op(&builder, "gather", gather_axes(&[0]), &[data.clone(), index.clone()], &[f32(&dims_i64(&[2, 5]))], &args);
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let err = gather_rule(&mut ctx, &source, &[MappedAxes::new(1), MappedAxes::UNMAPPED]).unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        assert!(err.message.contains("mapped data but unmapped indices"));
        // dynamic index dims
        let (mut builder, args) = builder_with(&[batched(&[4, 5]), f32(&[batch(), Dim::Dynamic])]);
        let source = rule_op(&builder, "gather", gather_axes(&[0]), &[data, f32(&[Dim::Dynamic])], &[f32(&[Dim::Dynamic, Dim::Int(5)])], &args);
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let err = gather_rule(&mut ctx, &source, &[MappedAxes::new(1), MappedAxes::new(1)]).unwrap_err();
        assert!(err.message.contains("dynamic index dims"));
    }

    #[test]
    fn scatter_broadcasts_unmapped_data_and_shifts_the_axis() {
        // source data f32[4,5], indices i64[2], updates f32[2,5], axis 0
        let (mut builder, args) = builder_with(&[
            f32(&dims_i64(&[4, 5])),
            batched(&[2]),
            batched(&[2, 5]),
        ]);
        let source = rule_op(
            &builder,
            "scatter",
            Attrs::from_pairs([("axis", AttrValue::Int(0))]),
            &[
                f32(&dims_i64(&[4, 5])),
                f32(&dims_i64(&[2])),
                f32(&dims_i64(&[2, 5])),
            ],
            &[f32(&dims_i64(&[4, 5]))],
            &args,
        );
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let (values, axes) = scatter_rule(
            &mut ctx,
            &source,
            &[MappedAxes::UNMAPPED, MappedAxes::new(1), MappedAxes::new(1)],
        )
        .unwrap();
        assert_eq!(axes, vec![MappedAxes::new(1)]);
        assert_eq!(
            ctx.shape(values[0]).unwrap(),
            vec![batch(), Dim::Int(4), Dim::Int(5)]
        );
        assert_eq!(op_names(&ctx), vec!["broadcast", "broadcast", "scatter"]);
        let op = &ctx.builder.program().ops.iter().last().unwrap().1;
        assert_eq!(op.attributes.i64("axis").unwrap(), 1);
    }

    #[test]
    fn scatter_disagreement_defers() {
        let (mut builder, args) =
            builder_with(&[batched(&[4, 5]), f32(&dims_i64(&[2])), f32(&dims_i64(&[2, 5]))]);
        let source = rule_op(
            &builder,
            "scatter",
            Attrs::from_pairs([("axis", AttrValue::Int(0))]),
            &[
                f32(&dims_i64(&[4, 5])),
                f32(&dims_i64(&[2])),
                f32(&dims_i64(&[2, 5])),
            ],
            &[f32(&dims_i64(&[4, 5]))],
            &args,
        );
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let err = scatter_rule(
            &mut ctx,
            &source,
            &[MappedAxes::new(1), MappedAxes::UNMAPPED, MappedAxes::UNMAPPED],
        )
        .unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        assert!(err.message.contains("data operand is mapped"));

        let (mut builder, args) =
            builder_with(&[f32(&dims_i64(&[4, 5])), batched(&[2]), f32(&dims_i64(&[2, 5]))]);
        let source = rule_op(
            &builder,
            "scatter",
            Attrs::from_pairs([("axis", AttrValue::Int(0))]),
            &[
                f32(&dims_i64(&[4, 5])),
                f32(&dims_i64(&[2])),
                f32(&dims_i64(&[2, 5])),
            ],
            &[f32(&dims_i64(&[4, 5]))],
            &args,
        );
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let err = scatter_rule(
            &mut ctx,
            &source,
            &[MappedAxes::UNMAPPED, MappedAxes::new(1), MappedAxes::UNMAPPED],
        )
        .unwrap_err();
        assert!(err.message.contains("indices and updates disagree"));
    }

    #[test]
    fn constant_rebuilds_the_payload_and_is_unmapped() {
        let (mut builder, _args) = builder_with(&[]);
        let payload = crate::emit::i64_vector_payload(&[1, 2, 3]);
        let source = rule_op(
            &builder,
            "constant",
            Attrs::from_pairs([("value", AttrValue::Array(payload))]),
            &[],
            &[f32(&dims_i64(&[3]))],
            &[],
        );
        let mut ctx = RuleCtx::with_batch_dim(&mut builder, Some("batch".into()));
        let (values, axes) = constant_rule(&mut ctx, &source, &[]).unwrap();
        assert_eq!(axes, vec![MappedAxes::UNMAPPED]);
        assert_eq!(ctx.shape(values[0]).unwrap(), dims_i64(&[3]));
        assert_eq!(op_names(&ctx), vec!["constant"]);
        // the payload is preserved verbatim
        let payload = ctx
            .builder
            .program()
            .ops
            .iter()
            .next()
            .unwrap()
            .1
            .attributes
            .array("value")
            .unwrap();
        assert_eq!(payload.shape, vec![3]);
        assert_eq!(payload.dtype, DType::parse("int64").unwrap());
    }
}
