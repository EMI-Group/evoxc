//! Indexing / data-movement kernels: `reshape`, `transpose`, `slice`,
//! `gather`, `scatter` and `concatenate`.
//!
//! The behavioural reference is `evoxc/backends/numpy/kernels/indexing.py`;
//! this module reproduces those numpy semantics on flat C-contiguous typed
//! buffers with the same error family and wording. The decisions that matter:
//!
//! * `slice` is Python `slice(start, limit, stride)` per dim — a missing entry
//!   list means numpy's defaults, negative bounds count from the end, bounds
//!   CLAMP (Python slicing never raises; the result is just shorter, possibly
//!   empty) and a negative stride steps backwards. A zero stride and a bound
//!   list whose length is not the operand rank are explicit `ShapeError`s.
//! * `gather` is numpy `take` along ONE axis (`x.shape[:axis] + indices.shape
//!   + x.shape[axis+1:]`, so a rank-0 index drops the axis); negative indices
//!   wrap and an out-of-range index is an explicit `ShapeError`. A multi-axis
//!   `axes` tuple has no single-shared-index-tensor equivalent (the numpy
//!   reference defers it too) → explicit `BackendError`, never a silent
//!     fallback to one axis.
//! * `scatter` is functional numpy `put_along_axis`: a COPY of the input is
//!   updated, low-rank indices pad to full rank (0-d → all-ones) and `updates`
//!   broadcasts against numpy's advanced-indexing result. DUPLICATE INDICES
//!   ARE NOT REJECTED — like numpy, the last write in C order wins.
//! * `transpose` (`permutation = null` reverses every axis), `reshape` (a
//!   runtime-dynamic target dim is an error — the engine never guesses a
//!   dimension; a count mismatch is numpy's `cannot reshape array of size N
//!   into shape ...`) and `concatenate` (casts every operand to the declared
//!   promoted dtype; rank-0 operands, a NULL `axis` and non-axis mismatches are
//!   errors) follow the same rule: never a wrong value, never a panic.
//!
//! Result shapes come from the op's declared result type: a static dim must
//! AGREE with the shape derived from the operands (a disagreement is an error,
//! never a wrongly-shaped buffer) while a runtime-dynamic dim takes the derived
//! extent — which is why the inner kernels take the already-evaluated declared
//! shape as `&[Option<usize>]` (`RunCtx::result_shape`) rather than calling the
//! strict helper.
use evoxc_core::{DType, Error, Result};

use crate::buffer::{block_storage, shape_text, walk, Buffer, Storage};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::kernels::KernelMap;
use crate::plan::OpSpec;
use crate::scalar::{Scalar, ScalarValue};
use crate::{with_same_type, with_type};

pub fn register(map: &mut KernelMap) {
    map.insert("reshape", reshape_kernel);
    map.insert("transpose", transpose_kernel);
    map.insert("slice", slice_kernel);
    map.insert("gather", gather_kernel);
    map.insert("scatter", scatter_kernel);
    map.insert("concatenate", concatenate_kernel);
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// The `position`-th operand, with an explicit error instead of an index panic.
fn operand<'a>(op: &str, operands: &'a [Value], position: usize) -> Result<&'a Buffer> {
    operands
        .get(position)
        .map(|value| &**value)
        .ok_or_else(|| {
            Error::backend(format!(
                "op '{op}': missing operand {position} (got {} operand(s))",
                operands.len()
            ))
        })
}

/// Reconcile the declared result shape with the shape derived from the
/// operands: a STATIC declared dim must agree (a disagreement is an explicit
/// error, never a wrongly-shaped buffer) and a runtime-dynamic dim takes the
/// derived extent, which the caller computed from the operands. This is the
/// dynamic-tolerant form of `util::result_shape` (that helper is the strict
/// one: it rejects any `None` dim instead of taking the operand extent).
fn reconcile(op: &str, declared: &[Option<usize>], derived: Vec<usize>) -> Result<Vec<usize>> {
    if declared.len() != derived.len() {
        return Err(Error::shape(format!(
            "op '{op}': derived result rank {} does not match the declared rank {}",
            derived.len(),
            declared.len()
        )));
    }
    for (axis, (want, got)) in declared.iter().zip(derived.iter()).enumerate() {
        if let Some(want) = want {
            if want != got {
                return Err(Error::shape(format!(
                    "op '{op}': derived result dim {axis} is {got} but the declared type says {want}"
                )));
            }
        }
    }
    Ok(derived)
}

/// An out-of-range flat-offset diagnostic — a shape/data mismatch is an error,
/// never a panic and never a silently wrong value.
fn offset_error(offset: impl std::fmt::Display, len: usize, op: &str) -> Error {
    Error::backend(format!(
        "{op}: computed flat offset {offset} is outside a {len}-element buffer — the declared result type does not match the operands"
    ))
}

/// Read `values[offset]` with a SIGNED offset (slice strides may be negative).
fn read_checked<T: Copy>(values: &[T], offset: i64, op: &str) -> Result<T> {
    usize::try_from(offset)
        .ok()
        .and_then(|slot| values.get(slot).copied())
        .ok_or_else(|| offset_error(offset, values.len(), op))
}

/// Visit every C-order multi-index of `shape`: a rank-0 shape visits the empty
/// index once and an empty shape dimension visits nothing.
fn for_each_index(shape: &[usize], mut body: impl FnMut(&[usize]) -> Result<()>) -> Result<()> {
    if shape.is_empty() {
        return body(&[]);
    }
    let mut index = vec![0usize; shape.len()];
    for _ in 0..util::numel(shape) {
        body(&index)?;
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    Ok(())
}

/// A tuple rendering for diagnostics (numpy's `(0, 1)` spelling).
fn render_ints(values: &[i64]) -> String {
    let parts: Vec<String> = values.iter().map(|value| value.to_string()).collect();
    format!("({})", parts.join(", "))
}

/// numpy's `reshape` message rendering — `(4,2)` (no space after the comma).
fn render_shape(values: &[usize]) -> String {
    let parts: Vec<String> = values.iter().map(|value| value.to_string()).collect();
    format!("({})", parts.join(","))
}

fn require_integer_indices(op: &str, indices: &Buffer) -> Result<()> {
    if !indices.dtype.is_int() {
        return Err(Error::backend(format!(
            "{op}: indices must be an integer dtype (int32/int64), got {}",
            indices.dtype.name()
        )));
    }
    Ok(())
}

/// Read an index tensor as widened integers (any integer dtype, numpy accepts
/// them all; `require_integer_indices` has already rejected the rest).
fn index_values(indices: &Buffer) -> Result<Vec<i128>> {
    with_type!(indices, |values| {
        Ok(values
            .iter()
            .map(|value| match value.to_scalar() {
                ScalarValue::I(value) => value as i128,
                ScalarValue::U(value) => value as i128,
                ScalarValue::F(value) => value as i128,
                ScalarValue::C(value) => value.re as i128,
            })
            .collect())
    })
}

/// numpy index semantics: a negative index counts from the end, anything else
/// outside `[0, extent)` is an explicit `ShapeError` with numpy's wording.
fn resolve_index(raw: i128, extent: usize, axis: usize, op: &str) -> Result<usize> {
    let length = extent as i128;
    let wrapped = if raw < 0 { raw + length } else { raw };
    if wrapped < 0 || wrapped >= length {
        return Err(Error::shape(format!(
            "{op}: index {raw} is out of bounds for axis {axis} with size {extent}"
        )));
    }
    Ok(wrapped as usize)
}

// ---------------------------------------------------------------------------
// reshape
// ---------------------------------------------------------------------------

fn reshape_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    let source = operand("reshape", operands, 0)?;
    let declared = ctx.result_shape(spec, 0)?;
    single(reshape(source, &declared)?)
}

/// Replace the shape of an already C-contiguous buffer (a pure reinterpretation
/// — no data movement). The declared result shape is the target: inference has
/// already resolved a `-1` wildcard, so it must be fully static here.
fn reshape(source: &Buffer, declared: &[Option<usize>]) -> Result<Buffer> {
    let mut target = Vec::with_capacity(declared.len());
    for dim in declared {
        match dim {
            Some(extent) => target.push(*extent),
            None => {
                return Err(Error::shape(
                    "reshape: target shape contains a None (runtime-dynamic) dim — the engine never guesses a dimension",
                ))
            }
        }
    }
    let total = source.numel();
    if util::numel(&target) != total {
        return Err(Error::shape(format!(
            "reshape: cannot reshape array of size {total} into shape {}",
            render_shape(&target)
        )));
    }
    Buffer::new(source.dtype, target, source.storage.clone())
}

// ---------------------------------------------------------------------------
// transpose
// ---------------------------------------------------------------------------

fn transpose_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    let source = operand("transpose", operands, 0)?;
    let permutation = transpose_permutation(spec, source.rank())?;
    let declared = ctx.result_shape(spec, 0)?;
    single(transpose(source, &permutation, &declared)?)
}

/// The op's permutation: `null` (or absent) reverses every axis; otherwise it
/// must be a bijection of `0..rank` (numpy's `axes don't match array` /
/// `repeated axis in transpose`).
fn transpose_permutation(spec: &OpSpec, rank: usize) -> Result<Vec<usize>> {
    let Some(axes) = spec.attrs.opt_ints("permutation")? else {
        return Ok((0..rank).rev().collect());
    };
    if axes.len() != rank {
        return Err(Error::shape(format!(
            "transpose: axes don't match array — permutation {} has {} entries for rank {rank}",
            render_ints(axes),
            axes.len()
        )));
    }
    let mut seen = vec![false; rank];
    let mut permutation = Vec::with_capacity(rank);
    for axis in axes {
        let normalized = util::normalize_axis(*axis, rank, "transpose")?;
        if seen[normalized] {
            return Err(Error::shape(format!(
                "transpose: repeated axis in transpose — permutation {}",
                render_ints(axes)
            )));
        }
        seen[normalized] = true;
        permutation.push(normalized);
    }
    Ok(permutation)
}

fn transpose(source: &Buffer, permutation: &[usize], declared: &[Option<usize>]) -> Result<Buffer> {
    let derived: Vec<usize> = permutation
        .iter()
        .map(|axis| source.shape[*axis])
        .collect();
    let shape = reconcile("transpose", declared, derived)?;
    // An identity permutation (which is what a `null` permutation means below
    // rank 2) moves no data: the result is the operand, re-labelled.
    if permutation.iter().enumerate().all(|(axis, target)| axis == *target) {
        return Buffer::new(source.dtype, shape, source.storage.clone());
    }
    let source_strides = util::c_strides(&source.shape);
    let strides: Vec<usize> = permutation
        .iter()
        .map(|axis| source_strides[*axis])
        .collect();
    with_type!(source, |values| transpose_typed(
        values,
        source.dtype,
        shape,
        strides
    ))
}

fn transpose_typed<T: Scalar>(
    values: &[T],
    dtype: DType,
    shape: Vec<usize>,
    strides: Vec<usize>,
) -> Result<Buffer> {
    let rank = shape.len();
    let mut out = Vec::with_capacity(values.len());
    walk(rank, &shape, &[strides], |offsets| out.push(values[offsets[0]]));
    Buffer::new(dtype, shape, T::into_storage(out))
}

// ---------------------------------------------------------------------------
// slice
// ---------------------------------------------------------------------------

/// One operand axis mapped onto a Python `slice(start, stop, step)`.
#[derive(Debug, Clone, Copy)]
struct DimSlice {
    start: i64,
    step: i64,
    extent: usize,
}

fn slice_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let source = operand("slice", operands, 0)?;
    let ranges = slice_ranges(spec, &source.shape)?;
    let declared = ctx.result_shape(spec, 0)?;
    single(slice(source, &ranges, &declared)?)
}

/// Resolve the op's `start_indices` / `limit_indices` / `strides` attributes —
/// a missing list means numpy's defaults for EVERY axis — and reject a list
/// whose length is not the operand rank.
fn slice_ranges(spec: &OpSpec, shape: &[usize]) -> Result<Vec<DimSlice>> {
    let rank = shape.len();
    let starts = spec.attrs.opt_ints("start_indices")?;
    let limits = spec.attrs.opt_ints("limit_indices")?;
    let strides = spec.attrs.opt_ints("strides")?;
    check_bound_len(starts, rank, "start_indices")?;
    check_bound_len(limits, rank, "limit_indices")?;
    check_bound_len(strides, rank, "strides")?;
    let mut ranges = Vec::with_capacity(rank);
    for axis in 0..rank {
        ranges.push(plan_slice_dim(
            axis,
            shape[axis],
            starts.map(|values| values[axis]),
            limits.map(|values| values[axis]),
            strides.map(|values| values[axis]),
        )?);
    }
    Ok(ranges)
}

fn check_bound_len(values: Option<&[i64]>, rank: usize, name: &str) -> Result<()> {
    if let Some(values) = values {
        if values.len() != rank {
            return Err(Error::shape(format!(
                "slice: {name} has {} entries for rank {rank}",
                values.len()
            )));
        }
    }
    Ok(())
}

/// Python slice bound normalization, including that side's clamping.
fn normalize_bound(value: Option<i64>, length: i64, default: i64) -> i64 {
    match value {
        None => default,
        Some(value) if value < 0 => value + length,
        Some(value) => value,
    }
}

/// Map one axis of extent `size` onto the slice plan, exactly like Python's
/// `slice.indices` pre-clamping: the start of a positive slice clamps to
/// `[0, size]`, a negative one to `[-1, size-1]`, and out-of-range bounds clamp
/// instead of raising (a fully out-of-range slice is simply empty).
fn plan_slice_dim(
    axis: usize,
    size: usize,
    start: Option<i64>,
    limit: Option<i64>,
    stride: Option<i64>,
) -> Result<DimSlice> {
    let step = stride.unwrap_or(1);
    if step == 0 {
        return Err(Error::shape(format!(
            "slice: slice step cannot be zero (axis {axis})"
        )));
    }
    let length = size as i64;
    let (start, stop) = if step > 0 {
        let start = normalize_bound(start, length, 0).clamp(0, length);
        let stop = normalize_bound(limit, length, length).clamp(0, length);
        (start, stop)
    } else {
        let start = normalize_bound(start, length, length - 1).clamp(-1, length - 1);
        let stop = normalize_bound(limit, length, -1).clamp(-1, length - 1);
        (start, stop)
    };
    let extent = if step > 0 {
        if stop > start {
            ((stop - start + step - 1) / step) as usize
        } else {
            0
        }
    } else {
        let magnitude = -step;
        if start > stop {
            ((start - stop + magnitude - 1) / magnitude) as usize
        } else {
            0
        }
    };
    Ok(DimSlice { start, step, extent })
}

fn slice(source: &Buffer, ranges: &[DimSlice], declared: &[Option<usize>]) -> Result<Buffer> {
    let derived: Vec<usize> = ranges.iter().map(|range| range.extent).collect();
    let shape = reconcile("slice", declared, derived)?;
    let source_strides = util::c_strides(&source.shape);
    let mut base = 0i64;
    let mut steps = Vec::with_capacity(ranges.len());
    for (range, stride) in ranges.iter().zip(&source_strides) {
        base += range.start * *stride as i64;
        steps.push(range.step * *stride as i64);
    }
    if let Some(storage) = slice_row_blocks(source, ranges, &shape, base, &steps)? {
        return Buffer::new(source.dtype, shape, storage);
    }
    with_type!(source, |values| slice_typed(
        values,
        source.dtype,
        &shape,
        base,
        &steps
    ))
}

/// `slice`'s contiguous fast path.
///
/// The trailing axes a slice selects WHOLE (`start == 0`, `step == 1`, the whole
/// extent) are one contiguous run per block, so the result is `copy_blocks` over
/// the leading axes' offsets instead of one source index per output element.
/// Only a non-negative step plan qualifies; anything else (negative strides,
/// per-axis steps, an empty result) stays on the general driver.
fn slice_row_blocks(
    source: &Buffer,
    ranges: &[DimSlice],
    out_shape: &[usize],
    base: i64,
    steps: &[i64],
) -> Result<Option<Storage>> {
    let rank = ranges.len();
    if util::numel(out_shape) == 0 || steps.iter().any(|step| *step < 0) {
        return Ok(None);
    }
    let mut trailing = 0usize;
    while trailing < rank {
        let axis = rank - 1 - trailing;
        let range = &ranges[axis];
        if range.start != 0 || range.step != 1 || range.extent != source.shape[axis] {
            break;
        }
        trailing += 1;
    }
    let inner: usize = out_shape[rank - trailing..].iter().product();
    if inner == 0 {
        return Ok(None);
    }
    let start = usize::try_from(base).map_err(|_| {
        Error::verification("slice: a negative base offset reached the row copy (internal engine error)")
    })?;
    let leading_shape = &out_shape[..rank - trailing];
    let mut offsets = Vec::with_capacity(util::numel(leading_shape));
    for_each_index(leading_shape, |index| {
        let mut offset = start as i128;
        for (coordinate, step) in index.iter().zip(steps) {
            offset += *coordinate as i128 * *step as i128;
        }
        match usize::try_from(offset) {
            Ok(offset) => {
                offsets.push(offset);
                Ok(())
            }
            Err(_) => Err(Error::verification(
                "slice: a negative source offset reached the row copy (internal engine error)",
            )),
        }
    })?;
    let storage = with_type!(source, |values| {
        block_storage(values, out_shape, &offsets, inner, "slice")
    })?;
    Ok(Some(storage))
}

fn slice_typed<T: Scalar>(
    values: &[T],
    dtype: DType,
    shape: &[usize],
    base: i64,
    steps: &[i64],
) -> Result<Buffer> {
    let mut out = Vec::with_capacity(util::numel(shape));
    for_each_index(shape, |index| {
        let mut offset = base;
        for (coordinate, step) in index.iter().zip(steps) {
            offset += *coordinate as i64 * *step;
        }
        out.push(read_checked(values, offset, "slice")?);
        Ok(())
    })?;
    Buffer::new(dtype, shape.to_vec(), T::into_storage(out))
}

// ---------------------------------------------------------------------------
// gather
// ---------------------------------------------------------------------------

fn gather_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let source = operand("gather", operands, 0)?;
    let indices = operand("gather", operands, 1)?;
    require_integer_indices("gather", indices)?;
    let axis = gather_axis(spec, source.rank())?;
    let declared = ctx.result_shape(spec, 0)?;
    let index_values = index_values(indices)?;
    single(gather(
        source,
        &index_values,
        &indices.shape,
        axis,
        &declared,
    )?)
}

/// `axes` must name exactly one gathered axis. A multi-axis tuple is the
/// documented numpy-interpreter deferral: several gathered axes have no single
/// shared index tensor, so it is an explicit error rather than a silent
/// "first axis only".
fn gather_axis(spec: &OpSpec, rank: usize) -> Result<usize> {
    let default_axes = [0i64];
    let axes = spec.attrs.opt_ints("axes")?.unwrap_or(&default_axes);
    let mut normalized = Vec::with_capacity(axes.len());
    for axis in axes {
        let axis = util::normalize_axis(*axis, rank, "gather")?;
        if !normalized.contains(&axis) {
            normalized.push(axis);
        }
    }
    normalized.sort_unstable();
    if normalized.len() != 1 {
        let rendered: Vec<i64> = normalized.iter().map(|axis| *axis as i64).collect();
        return Err(Error::backend(format!(
            "gather: multi-axis gather is not supported by the native CPU engine (axes {}) — a single shared index tensor has no numpy equivalent for several gathered axes",
            render_ints(&rendered)
        )));
    }
    Ok(normalized[0])
}

fn gather(
    source: &Buffer,
    indices: &[i128],
    index_shape: &[usize],
    axis: usize,
    declared: &[Option<usize>],
) -> Result<Buffer> {
    let mut derived: Vec<usize> = source.shape[..axis].to_vec();
    derived.extend_from_slice(index_shape);
    derived.extend_from_slice(&source.shape[axis + 1..]);
    let shape = reconcile("gather", declared, derived)?;
    if let Some(storage) = gather_blocks(source, indices, axis, &shape)? {
        return Buffer::new(source.dtype, shape, storage);
    }
    with_type!(source, |values| gather_typed(
        values,
        source.dtype,
        &source.shape,
        axis,
        index_shape,
        indices,
        &shape
    ))
}

/// `gather`'s contiguous fast path.
///
/// Only the gathered axis selects anything, so the result is `outer × n` runs
/// of `inner` whole elements copied verbatim out of the operand (`outer` = the
/// extents before the axis, `n` = the index count, `inner` = the extents after
/// it — exactly the C order the spliced result shape has).  The index tensor is
/// resolved once per row through [`resolve_index`], so numpy's negative-index
/// and bounds rules are unchanged; only the per-ELEMENT index arithmetic and
/// the `i128` re-read per output element are gone.  `None` for an empty result
/// or a shape this decomposition does not cover — the general path owns those.
fn gather_blocks(
    source: &Buffer,
    indices: &[i128],
    axis: usize,
    out_shape: &[usize],
) -> Result<Option<Storage>> {
    let outer: usize = source.shape[..axis].iter().product();
    let inner: usize = source.shape[axis + 1..].iter().product();
    let total = util::numel(out_shape);
    if total == 0 || indices.is_empty() || inner == 0 {
        return Ok(None);
    }
    let extent = source.shape[axis];
    let band = extent * inner;
    let count = outer * indices.len();
    if count.checked_mul(inner) != Some(total) {
        return Ok(None);
    }
    let mut offsets = Vec::with_capacity(count);
    for block in 0..outer {
        let base = block * band;
        for raw in indices {
            offsets.push(base + resolve_index(*raw, extent, axis, "gather")? * inner);
        }
    }
    let storage = with_type!(source, |values| {
        block_storage(values, out_shape, &offsets, inner, "gather")
    })?;
    Ok(Some(storage))
}

#[allow(clippy::too_many_arguments)]
fn gather_typed<T: Scalar>(
    values: &[T],
    dtype: DType,
    source_shape: &[usize],
    axis: usize,
    index_shape: &[usize],
    indices: &[i128],
    shape: &[usize],
) -> Result<Buffer> {
    let source_strides = util::c_strides(source_shape);
    let index_strides = util::c_strides(index_shape);
    let extent = source_shape[axis];
    let index_rank = index_shape.len();
    let mut out = Vec::with_capacity(util::numel(shape));
    for_each_index(shape, |index| {
        // The index tensor's own coordinates occupy `axis .. axis + index_rank`
        // of the result index (they are spliced in where the gathered axes were).
        let index_offset = util::flat_index(&index[axis..axis + index_rank], &index_strides);
        let raw = *indices
            .get(index_offset)
            .ok_or_else(|| offset_error(index_offset, indices.len(), "gather"))?;
        let selected = resolve_index(raw, extent, axis, "gather")?;
        let before = util::flat_index(&index[..axis], &source_strides[..axis]);
        let after = util::flat_index(
            &index[axis + index_rank..],
            &source_strides[axis + 1..],
        );
        let offset = before + selected * source_strides[axis] + after;
        let value = values
            .get(offset)
            .copied()
            .ok_or_else(|| offset_error(offset, values.len(), "gather"))?;
        out.push(value);
        Ok(())
    })?;
    Buffer::new(dtype, shape.to_vec(), T::into_storage(out))
}

// ---------------------------------------------------------------------------
// scatter
// ---------------------------------------------------------------------------

fn scatter_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    let source = operand("scatter", operands, 0)?;
    let indices = operand("scatter", operands, 1)?;
    let updates = operand("scatter", operands, 2)?;
    require_integer_indices("scatter", indices)?;
    let axis = util::normalize_axis(
        spec.attrs.opt_i64("axis")?.unwrap_or(0),
        source.rank(),
        "scatter",
    )?;
    let declared = ctx.result_shape(spec, 0)?;
    let index_values = index_values(indices)?;
    let cast = updates_for(source, updates)?;
    single(scatter(
        source,
        &index_values,
        &indices.shape,
        cast.as_ref().unwrap_or(updates),
        axis,
        &declared,
    )?)
}

/// `put_along_axis` writes into the data's own dtype, so a differing update
/// dtype is converted first (numpy would reject the mismatch outright; the
/// declared result dtype is already the data's).
fn updates_for(source: &Buffer, updates: &Buffer) -> Result<Option<Buffer>> {
    if updates.dtype == source.dtype {
        Ok(None)
    } else {
        Ok(Some(updates.cast(source.dtype)?))
    }
}

/// The interpreter's rank normalization for scatter index tensors: 0-d indices
/// become all-ones (the `scan` stack-update pattern), a rank-`r` tensor is used
/// as-is, and a lower-rank one is padded with `1`s around `axis`.
fn full_rank_indices(index_shape: &[usize], rank: usize, axis: usize) -> Result<Vec<usize>> {
    if index_shape.len() > rank {
        return Err(Error::shape(
            "scatter: `indices` and `arr` must have the same number of dimensions",
        ));
    }
    if index_shape.is_empty() {
        return Ok(vec![1; rank]);
    }
    if index_shape.len() == rank {
        return Ok(index_shape.to_vec());
    }
    let mut shape = vec![1usize; axis];
    shape.extend_from_slice(index_shape);
    shape.resize(rank, 1);
    Ok(shape)
}

/// numpy's advanced-indexing broadcast: along `axis` the extent is the index
/// tensor's, elsewhere the data and index extents broadcast (equal or 1).
fn selection_shape(source: &Buffer, index_shape: &[usize], axis: usize) -> Result<Vec<usize>> {
    let mut selection = vec![0usize; index_shape.len()];
    for (position, (data, index)) in source.shape.iter().zip(index_shape).enumerate() {
        selection[position] = if position == axis || data == index || *data == 1 {
            *index
        } else if *index == 1 {
            *data
        } else {
            return Err(Error::shape(format!(
                "scatter: shape mismatch: indexing arrays could not be broadcast together (data dim {position} is {data}, indices dim {position} is {index})"
            )));
        };
    }
    Ok(selection)
}

fn scatter(
    source: &Buffer,
    indices: &[i128],
    index_shape: &[usize],
    updates: &Buffer,
    axis: usize,
    declared: &[Option<usize>],
) -> Result<Buffer> {
    let rank = source.rank();
    let index_shape = full_rank_indices(index_shape, rank, axis)?;
    let selection = selection_shape(source, &index_shape, axis)?;
    if !util::broadcasts_to(&updates.shape, &selection) {
        return Err(Error::shape(format!(
            "scatter: shape mismatch: value array of shape {} could not be broadcast to indexing result of shape {}",
            shape_text(&updates.shape),
            shape_text(&selection)
        )));
    }
    let shape = reconcile("scatter", declared, source.shape.clone())?;
    with_same_type!(source, updates, |data, values| scatter_typed(
        data,
        values,
        source.dtype,
        &source.shape,
        axis,
        &index_shape,
        indices,
        &selection,
        &updates.shape,
        &shape
    ))
}

#[allow(clippy::too_many_arguments)]
fn scatter_typed<T: Scalar>(
    source: &[T],
    updates: &[T],
    dtype: DType,
    source_shape: &[usize],
    axis: usize,
    index_shape: &[usize],
    indices: &[i128],
    selection: &[usize],
    update_shape: &[usize],
    shape: &[usize],
) -> Result<Buffer> {
    let mut out = source.to_vec();
    let capacity = out.len();
    let update_len = updates.len();
    let source_strides = util::c_strides(source_shape);
    let index_strides = util::c_strides(index_shape);
    let update_strides = util::c_strides(update_shape);
    let extent = source_shape[axis];
    // `updates` broadcasts against the advanced-indexing result, so it aligns
    // with the TRAILING selection dims.
    let offset_rank = selection.len() - update_shape.len();
    for_each_index(selection, |index| {
        let mut index_offset = 0usize;
        for (position, stride) in index_strides.iter().enumerate() {
            let coordinate = if index_shape[position] == 1 {
                0
            } else {
                index[position]
            };
            index_offset += coordinate * stride;
        }
        let raw = *indices
            .get(index_offset)
            .ok_or_else(|| offset_error(index_offset, indices.len(), "scatter"))?;
        let selected = resolve_index(raw, extent, axis, "scatter")?;
        let mut destination = 0usize;
        for (position, stride) in source_strides.iter().enumerate() {
            let coordinate = if position == axis {
                selected
            } else if source_shape[position] == 1 {
                0
            } else {
                index[position]
            };
            destination += coordinate * stride;
        }
        let mut update_offset = 0usize;
        for (part, stride) in update_strides.iter().enumerate() {
            let coordinate = if update_shape[part] == 1 {
                0
            } else {
                index[offset_rank + part]
            };
            update_offset += coordinate * stride;
        }
        let value = updates
            .get(update_offset)
            .copied()
            .ok_or_else(|| offset_error(update_offset, update_len, "scatter"))?;
        let slot = out
            .get_mut(destination)
            .ok_or_else(|| offset_error(destination, capacity, "scatter"))?;
        *slot = value;
        Ok(())
    })?;
    Buffer::new(dtype, shape.to_vec(), T::into_storage(out))
}

// ---------------------------------------------------------------------------
// concatenate
// ---------------------------------------------------------------------------

/// The C-order three-part view of a concatenate along `axis`: `outer` blocks of
/// `total` axis rows, each row `inner` elements wide.
struct Layout {
    outer: usize,
    axis: usize,
    total: usize,
    inner: usize,
}

fn concatenate_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    let sources: Vec<&Buffer> = operands.iter().map(|value| &**value).collect();
    let declared = ctx.result_shape(spec, 0)?;
    single(concatenate(
        &sources,
        spec.attrs.opt_i64("axis")?,
        spec.result_types[0].dtype,
        &declared,
    )?)
}

fn concatenate(
    sources: &[&Buffer],
    axis: Option<i64>,
    dtype: DType,
    declared: &[Option<usize>],
) -> Result<Buffer> {
    let Some(first) = sources.first() else {
        return Err(Error::shape("concatenate: expected at least one operand"));
    };
    let rank = first.rank();
    for (position, value) in sources.iter().enumerate() {
        if value.rank() != rank {
            return Err(Error::shape(format!(
                "concatenate: all the input arrays must have same number of dimensions, but the array at index 0 has {rank} dimension(s) and the array at index {position} has {} dimension(s)",
                value.rank()
            )));
        }
    }
    if rank == 0 {
        return Err(Error::shape(
            "concatenate: zero-dimensional arrays cannot be concatenated",
        ));
    }
    let Some(axis) = axis else {
        return Err(Error::shape(
            "concatenate: axis must be an int, got None — a null axis has no numpy equivalent for zero-dimensional operands, which this op rejects",
        ));
    };
    let axis = util::normalize_axis(axis, rank, "concatenate")?;
    let mut shape = first.shape.clone();
    let mut total = 0usize;
    for (position, value) in sources.iter().enumerate() {
        for dim in 0..rank {
            if dim == axis {
                continue;
            }
            if value.shape[dim] != first.shape[dim] {
                return Err(Error::shape(format!(
                    "concatenate: all the input array dimensions except for the concatenation axis must match exactly, but along dimension {dim}, the array at index 0 has size {} and the array at index {position} has size {}",
                    first.shape[dim], value.shape[dim]
                )));
            }
        }
        total = total
            .checked_add(value.shape[axis])
            .ok_or_else(|| Error::shape("concatenate: the concatenated dimension is too large"))?;
    }
    shape[axis] = total;
    let shape = reconcile("concatenate", declared, shape)?;
    // Every operand is cast to the op's DECLARED (already promoted) dtype.
    let mut casts: Vec<Option<Buffer>> = Vec::with_capacity(sources.len());
    for value in sources {
        casts.push(if value.dtype == dtype {
            None
        } else {
            Some(value.cast(dtype)?)
        });
    }
    let parts: Vec<&Buffer> = sources
        .iter()
        .enumerate()
        .map(|(position, value)| casts[position].as_ref().unwrap_or(value))
        .collect();
    let layout = Layout {
        outer: shape[..axis].iter().product(),
        axis,
        total: shape[axis],
        inner: shape[axis + 1..].iter().product(),
    };
    let mut out = Buffer::zeros(dtype, shape);
    let mut column = 0usize;
    for part in &parts {
        copy_part(&mut out, part, &layout, column)?;
        column += part.shape[axis];
    }
    Ok(out)
}

/// Copy one operand's block of rows into the output at `column` (an element
/// offset along the concatenation axis).
fn copy_part(out: &mut Buffer, part: &Buffer, layout: &Layout, column: usize) -> Result<()> {
    let width = part.shape[layout.axis];
    let dtype = out.dtype;
    macro_rules! copy {
        ($variant:ident) => {{
            let Storage::$variant(out_values) = &mut out.storage else {
                unreachable!()
            };
            let Storage::$variant(part_values) = &part.storage else {
                return Err(Error::dtype(format!(
                    "concatenate: the {} payload does not match the declared result dtype {}",
                    stringify!($variant),
                    dtype.name()
                )));
            };
            copy_strided(out_values, part_values, layout, column, width)?;
        }};
    }
    match dtype {
        DType::Bool => copy!(Bool),
        DType::Int8 => copy!(I8),
        DType::Int16 => copy!(I16),
        DType::Int32 => copy!(I32),
        DType::Int64 => copy!(I64),
        DType::UInt8 => copy!(U8),
        DType::UInt16 => copy!(U16),
        DType::UInt32 => copy!(U32),
        DType::UInt64 => copy!(U64),
        DType::Float16 => copy!(F16),
        DType::Float32 => copy!(F32),
        DType::Float64 => copy!(F64),
        DType::Complex64 => copy!(C64),
        DType::Complex128 => copy!(C128),
    }
    Ok(())
}

fn copy_strided<T: Copy>(
    out: &mut [T],
    part: &[T],
    layout: &Layout,
    column: usize,
    width: usize,
) -> Result<()> {
    let part_span = width * layout.inner;
    let out_span = layout.total * layout.inner;
    let count = part_span;
    for block in 0..layout.outer {
        let source = block * part_span;
        let destination = block * out_span + column * layout.inner;
        let (Some(source_slice), Some(destination_slice)) = (
            part.get(source..source + count),
            out.get_mut(destination..destination + count),
        ) else {
            return Err(Error::shape(format!(
                "concatenate: computed slice {destination}..{} is outside the {}-element result — the declared result type does not match the operands",
                destination + count,
                out.len()
            )));
        };
        destination_slice.copy_from_slice(source_slice);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use evoxc_core::Attrs;

    fn buffer(dtype: DType, shape: &[usize], storage: Storage) -> Buffer {
        Buffer::new(dtype, shape.to_vec(), storage)
            .unwrap_or_else(|err| panic!("test fixture is malformed: {}", err.message))
    }

    fn f32_buffer(shape: &[usize], values: &[f32]) -> Buffer {
        buffer(DType::Float32, shape, Storage::F32(values.to_vec()))
    }

    fn i64_buffer(shape: &[usize], values: &[i64]) -> Buffer {
        buffer(DType::Int64, shape, Storage::I64(values.to_vec()))
    }

    fn declared(dims: &[usize]) -> Vec<Option<usize>> {
        dims.iter().map(|dim| Some(*dim)).collect()
    }

    fn attrs(pairs: &[(&str, evoxc_core::AttrValue)]) -> Attrs {
        Attrs::from_pairs(pairs.iter().cloned())
    }

    fn spec(attrs: Attrs) -> OpSpec {
        OpSpec {
            name: "test".to_string(),
            attrs,
            operand_types: Vec::new(),
            result_types: Vec::new(),
            regions: Vec::new(),
            location: None,
        }
    }

    // -- reshape ----------------------------------------------------------

    #[test]
    fn reshape_reinterprets_without_moving_data() {
        let x = f32_buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let out = reshape(&x, &declared(&[3, 2])).unwrap();
        assert_eq!(out.shape, vec![3, 2]);
        assert_eq!(out.storage, Storage::F32(vec![1., 2., 3., 4., 5., 6.]));
    }

    #[test]
    fn reshape_supports_the_rank_zero_case_both_ways() {
        let scalar = f32_buffer(&[], &[7.]);
        let vector = reshape(&scalar, &declared(&[1])).unwrap();
        assert_eq!(vector.shape, vec![1]);
        let back = reshape(&vector, &[]).unwrap();
        assert_eq!(back.shape, Vec::<usize>::new());
        assert_eq!(back.storage, Storage::F32(vec![7.]));
    }

    #[test]
    fn reshape_element_count_mismatch_is_an_error() {
        let x = f32_buffer(&[2, 3], &[0.; 6]);
        let err = reshape(&x, &declared(&[4, 2])).unwrap_err();
        assert!(err.message.contains("cannot reshape array of size 6"), "{}", err.message);
    }

    #[test]
    fn reshape_rejects_a_runtime_dynamic_target_dim() {
        let x = f32_buffer(&[2, 3], &[0.; 6]);
        let err = reshape(&x, &[Some(3), None]).unwrap_err();
        assert!(err.message.contains("never guesses a dimension"), "{}", err.message);
    }

    // -- transpose --------------------------------------------------------

    #[test]
    fn transpose_defaults_to_reversing_every_axis() {
        let x = f32_buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let permutation = transpose_permutation(&spec(Attrs::new()), 2).unwrap();
        assert_eq!(permutation, vec![1, 0]);
        let out = transpose(&x, &permutation, &declared(&[3, 2])).unwrap();
        assert_eq!(out.shape, vec![3, 2]);
        assert_eq!(out.storage, Storage::F32(vec![1., 4., 2., 5., 3., 6.]));
    }

    #[test]
    fn transpose_applies_an_explicit_permutation() {
        let x = f32_buffer(&[2, 3, 4], &(0..24).map(|value| value as f32).collect::<Vec<_>>());
        let op = spec(attrs(&[(
            "permutation",
            evoxc_core::AttrValue::Ints(vec![2, 0, 1]),
        )]));
        let permutation = transpose_permutation(&op, 3).unwrap();
        assert_eq!(permutation, vec![2, 0, 1]);
        let out = transpose(&x, &permutation, &declared(&[4, 2, 3])).unwrap();
        assert_eq!(out.shape, vec![4, 2, 3]);
        // out[p2][p0][p1] == x[p0][p1][p2]; element (1, 2, 3) of x is index 23.
        assert_eq!(
            out.storage,
            Storage::F32(vec![
                0., 4., 8., 12., 16., 20., 1., 5., 9., 13., 17., 21., 2., 6., 10., 14., 18., 22.,
                3., 7., 11., 15., 19., 23.
            ])
        );
    }

    #[test]
    fn transpose_rejects_a_repeated_or_short_permutation() {
        let repeated = spec(attrs(&[(
            "permutation",
            evoxc_core::AttrValue::Ints(vec![0, 0]),
        )]));
        let err = transpose_permutation(&repeated, 2).unwrap_err();
        assert!(err.message.contains("repeated axis in transpose"), "{}", err.message);

        let short = spec(attrs(&[(
            "permutation",
            evoxc_core::AttrValue::Ints(vec![0, 1]),
        )]));
        let err = transpose_permutation(&short, 3).unwrap_err();
        assert!(err.message.contains("axes don't match array"), "{}", err.message);

        let out_of_range = spec(attrs(&[(
            "permutation",
            evoxc_core::AttrValue::Ints(vec![0, 2]),
        )]));
        let err = transpose_permutation(&out_of_range, 2).unwrap_err();
        assert!(err.message.contains("out of range"), "{}", err.message);
    }

    // -- slice ------------------------------------------------------------

    #[test]
    fn slice_applies_limits_and_strides() {
        let x = f32_buffer(&[4, 5], &(0..20).map(|value| value as f32).collect::<Vec<_>>());
        let op = spec(attrs(&[
            ("start_indices", evoxc_core::AttrValue::Ints(vec![1, 0])),
            ("limit_indices", evoxc_core::AttrValue::Ints(vec![3, 4])),
            ("strides", evoxc_core::AttrValue::Ints(vec![2, 2])),
        ]));
        let ranges = slice_ranges(&op, &[4, 5]).unwrap();
        let out = slice(&x, &ranges, &declared(&[1, 2])).unwrap();
        assert_eq!(out.shape, vec![1, 2]);
        assert_eq!(out.storage, Storage::F32(vec![5., 7.]));
    }

    #[test]
    fn slice_supports_a_negative_stride() {
        let x = f32_buffer(&[4], &[10., 20., 30., 40.]);
        // x[3::-1] — a null limit list means numpy's default for every axis.
        let op = spec(attrs(&[
            ("start_indices", evoxc_core::AttrValue::Ints(vec![3])),
            ("limit_indices", evoxc_core::AttrValue::Null),
            ("strides", evoxc_core::AttrValue::Ints(vec![-1])),
        ]));
        let ranges = slice_ranges(&op, &[4]).unwrap();
        let out = slice(&x, &ranges, &declared(&[4])).unwrap();
        assert_eq!(out.storage, Storage::F32(vec![40., 30., 20., 10.]));

        // x[2:0:-2] keeps one element.
        let op = spec(attrs(&[
            ("start_indices", evoxc_core::AttrValue::Ints(vec![2])),
            ("limit_indices", evoxc_core::AttrValue::Ints(vec![0])),
            ("strides", evoxc_core::AttrValue::Ints(vec![-2])),
        ]));
        let ranges = slice_ranges(&op, &[4]).unwrap();
        let out = slice(&x, &ranges, &declared(&[1])).unwrap();
        assert_eq!(out.storage, Storage::F32(vec![30.]));
    }

    #[test]
    fn slice_clamps_out_of_range_bounds_and_allows_an_empty_result() {
        let x = f32_buffer(&[4], &[10., 20., 30., 40.]);
        // A start beyond the axis is empty, exactly like Python slicing.
        let op = spec(attrs(&[
            ("start_indices", evoxc_core::AttrValue::Ints(vec![10])),
            ("limit_indices", evoxc_core::AttrValue::Ints(vec![99])),
            ("strides", evoxc_core::AttrValue::Null),
        ]));
        let ranges = slice_ranges(&op, &[4]).unwrap();
        let out = slice(&x, &ranges, &declared(&[0])).unwrap();
        assert_eq!(out.shape, vec![0]);
        assert!(out.storage == Storage::F32(Vec::new()));

        // An oversized limit clamps to the axis extent.
        let op = spec(attrs(&[
            ("start_indices", evoxc_core::AttrValue::Ints(vec![2])),
            ("limit_indices", evoxc_core::AttrValue::Ints(vec![99])),
            ("strides", evoxc_core::AttrValue::Null),
        ]));
        let ranges = slice_ranges(&op, &[4]).unwrap();
        let out = slice(&x, &ranges, &declared(&[2])).unwrap();
        assert_eq!(out.storage, Storage::F32(vec![30., 40.]));

        // A negative start counts from the end.
        let op = spec(attrs(&[
            ("start_indices", evoxc_core::AttrValue::Ints(vec![-2])),
            ("limit_indices", evoxc_core::AttrValue::Null),
            ("strides", evoxc_core::AttrValue::Null),
        ]));
        let ranges = slice_ranges(&op, &[4]).unwrap();
        let out = slice(&x, &ranges, &declared(&[2])).unwrap();
        assert_eq!(out.storage, Storage::F32(vec![30., 40.]));
    }

    #[test]
    fn slice_rejects_a_zero_stride_and_a_short_bound_list() {
        let op = spec(attrs(&[("strides", evoxc_core::AttrValue::Ints(vec![0]))]));
        let err = slice_ranges(&op, &[4]).unwrap_err();
        assert!(err.message.contains("slice step cannot be zero"), "{}", err.message);

        let op = spec(attrs(&[(
            "start_indices",
            evoxc_core::AttrValue::Ints(vec![0]),
        )]));
        let err = slice_ranges(&op, &[4, 5]).unwrap_err();
        assert!(
            err.message.contains("start_indices has 1 entries for rank 2"),
            "{}",
            err.message
        );
    }

    // -- gather -----------------------------------------------------------

    #[test]
    fn gather_takes_rows_along_the_axis() {
        let x = f32_buffer(&[3, 2], &[1., 2., 3., 4., 5., 6.]);
        let out = gather(&x, &[2, 0], &[2], 0, &declared(&[2, 2])).unwrap();
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(out.storage, Storage::F32(vec![5., 6., 1., 2.]));
    }

    #[test]
    fn gather_negative_indices_count_from_the_end() {
        let x = f32_buffer(&[3, 2], &[1., 2., 3., 4., 5., 6.]);
        let out = gather(&x, &[-1, 0, -3], &[3], 0, &declared(&[3, 2])).unwrap();
        assert_eq!(out.storage, Storage::F32(vec![5., 6., 1., 2., 1., 2.]));
    }

    #[test]
    fn gather_splices_the_index_shape_at_the_gathered_axis() {
        let x = f32_buffer(&[2, 3, 2], &(0..12).map(|value| value as f32).collect::<Vec<_>>());
        // axis 1: result is (2, 2, 2, 2) — the index tensor's own rank is 2.
        let out = gather(&x, &[0, 2, 1, 1], &[2, 2], 1, &declared(&[2, 2, 2, 2])).unwrap();
        assert_eq!(out.shape, vec![2, 2, 2, 2]);
        assert_eq!(
            out.storage,
            Storage::F32(vec![
                0., 1., 4., 5., 2., 3., 2., 3., 6., 7., 10., 11., 8., 9., 8., 9.
            ])
        );
    }

    #[test]
    fn gather_rank_zero_indices_drop_the_axis() {
        let x = f32_buffer(&[3, 2], &[1., 2., 3., 4., 5., 6.]);
        let out = gather(&x, &[1], &[], 0, &declared(&[2])).unwrap();
        assert_eq!(out.shape, vec![2]);
        assert_eq!(out.storage, Storage::F32(vec![3., 4.]));
    }

    /// `(input shape, indices, index shape, axis)` for the gather agreement test.
    type GatherCase<'a> = (&'a [usize], &'a [i128], &'a [usize], usize);

    #[test]
    fn gather_block_fast_path_agrees_with_the_per_element_path() {
        // Every single-axis gather on a contiguous operand takes the row-copy
        // fast path; it must equal the general element-at-a-time driver for
        // rank-1/2/3 operands, every axis, a lower- and a higher-rank index
        // tensor, negative indices and duplicate indices.
        let cases: [GatherCase; 7] = [
            (&[3, 2], &[2, 0], &[2], 0),
            (&[3, 2], &[-1, 0, -3], &[3], 0),
            (&[2, 3, 2], &[0, 2, 1, 1], &[2, 2], 1),
            (&[3, 2], &[1], &[], 0),
            (&[4, 3], &[0, 2, 2], &[3], 1),
            (&[2, 4, 3], &[3, 0], &[2], 1),
            (&[2, 2, 3], &[1, 0, 1, 1], &[2, 2], 2),
        ];
        for (in_shape, indices, index_shape, axis) in cases {
            let total: usize = in_shape.iter().product();
            let values: Vec<f32> = (0..total).map(|value| value as f32).collect();
            let x = f32_buffer(in_shape, &values);
            let mut derived: Vec<usize> = in_shape[..axis].to_vec();
            derived.extend_from_slice(index_shape);
            derived.extend_from_slice(&in_shape[axis + 1..]);
            let shape = derived.clone();
            let fast = gather_blocks(&x, indices, axis, &shape)
                .expect("the fast path succeeds")
                .expect("the fast path applies");
            let general = with_type!(x, |values| gather_typed(
                values, x.dtype, &x.shape, axis, index_shape, indices, &shape
            ))
            .expect("the general path succeeds");
            assert_eq!(fast, general.storage, "shapes {in_shape:?} axis {axis}");
        }
    }

    #[test]
    fn gather_axis_attribute_accepts_one_axis_and_rejects_several() {
        let one = spec(attrs(&[("axes", evoxc_core::AttrValue::Ints(vec![-1]))]));
        assert_eq!(gather_axis(&one, 2).unwrap(), 1);

        let several = spec(attrs(&[(
            "axes",
            evoxc_core::AttrValue::Ints(vec![0, 1]),
        )]));
        let err = gather_axis(&several, 2).unwrap_err();
        assert!(err.message.contains("multi-axis gather is not supported"), "{}", err.message);
    }

    #[test]
    fn gather_out_of_bounds_index_is_an_error() {
        let x = f32_buffer(&[3], &[1., 2., 3.]);
        let err = gather(&x, &[5], &[1], 0, &declared(&[1])).unwrap_err();
        assert!(
            err.message.contains("index 5 is out of bounds for axis 0 with size 3"),
            "{}",
            err.message
        );
        let err = gather(&x, &[-4], &[1], 0, &declared(&[1])).unwrap_err();
        assert!(err.message.contains("index -4 is out of bounds"), "{}", err.message);
    }

    #[test]
    fn gather_requires_an_integer_index_dtype() {
        let indices = f32_buffer(&[1], &[0.]);
        let err = require_integer_indices("gather", &indices).unwrap_err();
        assert!(
            err.message
                .contains("gather: indices must be an integer dtype (int32/int64), got float32"),
            "{}",
            err.message
        );
        assert!(require_integer_indices("gather", &i64_buffer(&[1], &[0])).is_ok());
    }

    // -- scatter ----------------------------------------------------------

    #[test]
    fn scatter_updates_a_copy_of_the_input() {
        let x = f32_buffer(&[3, 2], &[0.; 6]);
        let updates = f32_buffer(&[2, 2], &[1., 2., 3., 4.]);
        let out = scatter(&x, &[2, 0], &[2], &updates, 0, &declared(&[3, 2])).unwrap();
        assert_eq!(out.storage, Storage::F32(vec![3., 4., 0., 0., 1., 2.]));
        assert_eq!(x.storage, Storage::F32(vec![0.; 6]), "the input must not be mutated");
    }

    #[test]
    fn scatter_duplicate_indices_keep_the_last_write() {
        let x = f32_buffer(&[3], &[0., 0., 0.]);
        let updates = f32_buffer(&[2], &[1., 2.]);
        let out = scatter(&x, &[0, 0], &[2], &updates, 0, &declared(&[3])).unwrap();
        assert_eq!(out.storage, Storage::F32(vec![2., 0., 0.]));
    }

    #[test]
    fn scatter_rank_zero_indices_write_one_row() {
        let x = f32_buffer(&[3, 2], &[0.; 6]);
        let updates = f32_buffer(&[2], &[7., 8.]);
        let out = scatter(&x, &[1], &[], &updates, 0, &declared(&[3, 2])).unwrap();
        assert_eq!(out.storage, Storage::F32(vec![0., 0., 7., 8., 0., 0.]));
    }

    #[test]
    fn scatter_along_axis_one_uses_the_axis_attribute() {
        let x = f32_buffer(&[2, 3], &[0.; 6]);
        let updates = f32_buffer(&[2, 2], &[1., 2., 3., 4.]);
        let op = spec(attrs(&[("axis", evoxc_core::AttrValue::Int(1))]));
        let axis = util::normalize_axis(
            op.attrs.opt_i64("axis").unwrap().unwrap_or(0),
            2,
            "scatter",
        )
        .unwrap();
        assert_eq!(axis, 1);
        let out = scatter(&x, &[2, 0], &[2], &updates, axis, &declared(&[2, 3])).unwrap();
        // The rank-1 index tensor pads to (1, 2) around axis 1, so it
        // broadcasts across rows: every row writes columns 2 then 0.
        assert_eq!(out.storage, Storage::F32(vec![2., 0., 1., 4., 0., 3.]));
    }

    #[test]
    fn scatter_rejects_out_of_range_indices_and_bad_updates() {
        let x = f32_buffer(&[3], &[0.; 3]);
        let updates = f32_buffer(&[1], &[1.]);
        let err = scatter(&x, &[7], &[1], &updates, 0, &declared(&[3])).unwrap_err();
        assert!(err.message.contains("index 7 is out of bounds"), "{}", err.message);

        let updates = f32_buffer(&[5], &[0.; 5]);
        let err = scatter(&x, &[1], &[1], &updates, 0, &declared(&[3])).unwrap_err();
        assert!(err.message.contains("shape mismatch"), "{}", err.message);
    }

    // -- concatenate ------------------------------------------------------

    #[test]
    fn concatenate_joins_along_axis_zero() {
        let a = f32_buffer(&[2, 2], &[1., 2., 3., 4.]);
        let b = f32_buffer(&[1, 2], &[5., 6.]);
        let out = concatenate(&[&a, &b], Some(0), DType::Float32, &declared(&[3, 2])).unwrap();
        assert_eq!(out.shape, vec![3, 2]);
        assert_eq!(out.storage, Storage::F32(vec![1., 2., 3., 4., 5., 6.]));
    }

    #[test]
    fn concatenate_joins_along_axis_one() {
        let a = f32_buffer(&[2, 1], &[1., 3.]);
        let b = f32_buffer(&[2, 2], &[2., 9., 4., 8.]);
        let out = concatenate(&[&a, &b], Some(1), DType::Float32, &declared(&[2, 3])).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(out.storage, Storage::F32(vec![1., 2., 9., 3., 4., 8.]));
    }

    #[test]
    fn concatenate_casts_to_the_declared_result_dtype() {
        let a = buffer(DType::Int32, &[2], Storage::I32(vec![1, 2]));
        let b = buffer(DType::Float32, &[2], Storage::F32(vec![0.5, 1.5]));
        let out = concatenate(&[&a, &b], Some(0), DType::Float64, &declared(&[4])).unwrap();
        assert_eq!(out.dtype, DType::Float64);
        assert_eq!(out.storage, Storage::F64(vec![1., 2., 0.5, 1.5]));
    }

    #[test]
    fn concatenate_rejects_rank_zero_operands_and_a_null_axis() {
        let scalar = f32_buffer(&[], &[1.]);
        let err = concatenate(&[&scalar, &scalar], Some(0), DType::Float32, &[]).unwrap_err();
        assert!(
            err.message
                .contains("zero-dimensional arrays cannot be concatenated"),
            "{}",
            err.message
        );

        let a = f32_buffer(&[2], &[1., 2.]);
        let b = f32_buffer(&[1], &[3.]);
        let err = concatenate(&[&a, &b], None, DType::Float32, &declared(&[3])).unwrap_err();
        assert!(err.message.contains("axis must be an int, got None"), "{}", err.message);
    }

    #[test]
    fn concatenate_rejects_a_non_axis_shape_mismatch() {
        let a = f32_buffer(&[2, 2], &[0.; 4]);
        let b = f32_buffer(&[2, 3], &[0.; 6]);
        let err = concatenate(&[&a, &b], Some(0), DType::Float32, &declared(&[4, 2])).unwrap_err();
        assert!(
            err.message.contains("must match exactly"),
            "{}",
            err.message
        );
    }

    #[test]
    fn reconcile_rejects_a_wrong_static_dim() {
        let err = reconcile("gather", &declared(&[2, 2]), vec![2, 3]).unwrap_err();
        assert!(err.message.contains("declared type says 2"), "{}", err.message);
        let err = reconcile("gather", &declared(&[2]), vec![2, 3]).unwrap_err();
        assert!(err.message.contains("declared rank 1"), "{}", err.message);
        // A runtime-dynamic dim takes the operand-derived extent.
        assert_eq!(reconcile("gather", &[None, Some(3)], vec![2, 3]).unwrap(), vec![2, 3]);
    }
}
