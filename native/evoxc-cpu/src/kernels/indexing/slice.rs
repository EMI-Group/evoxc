//! `slice` kernel: Python `slice(start, limit, stride)` per dim.
//!
//! Bounds CLAMP (Python slicing never raises), a negative bound counts from the
//! end, a negative stride steps backwards, and a zero stride or a bound list
//! whose length is not the operand rank is an explicit `ShapeError`. The module
//! doc in [`super`] carries the shared declared-shape and error-wording rules.
use evoxc_core::{DType, Error, Result};

use crate::buffer::{Buffer, Storage};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::plan::OpSpec;
use crate::scalar::Scalar;
use crate::with_type;

use super::{block_storage_from, for_each_index, operand, read_checked, reconcile};

/// One operand axis mapped onto a Python `slice(start, stop, step)`.
#[derive(Debug, Clone, Copy)]
struct DimSlice {
    start: i64,
    step: i64,
    extent: usize,
}

pub(super) fn slice_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
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
        block_storage_from(values, out_shape, &offsets, inner, "slice")
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::indexing::test_support::{attrs, declared, f32_buffer, spec};
    use crate::Storage;

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

    #[test]
    fn slice_row_copy_accepts_a_runtime_dynamic_result_dim() {
        // The row-copy fast path is driven by the DECLARED shape, so a dynamic
        // leading dim still takes it and a static trailing dim is still checked.
        let x = f32_buffer(&[4, 3], &(0..12).map(|value| value as f32).collect::<Vec<_>>());
        let op = spec(attrs(&[
            ("start_indices", evoxc_core::AttrValue::Ints(vec![1, 0])),
            ("limit_indices", evoxc_core::AttrValue::Ints(vec![3, 3])),
            ("strides", evoxc_core::AttrValue::Null),
        ]));
        let ranges = slice_ranges(&op, &[4, 3]).unwrap();
        let out = slice(&x, &ranges, &[None, Some(3)]).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(out.storage, Storage::F32(vec![3., 4., 5., 6., 7., 8.]));
        let err = slice(&x, &ranges, &[None, Some(2)]).unwrap_err();
        assert_eq!(err.kind, evoxc_core::ErrorKind::Shape);
    }
}
