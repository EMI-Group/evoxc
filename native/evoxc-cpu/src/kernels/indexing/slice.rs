//! `slice` kernel: Python `slice(start, limit, stride)` per dim.
//!
//! Bounds CLAMP (Python slicing never raises), a negative bound counts from the
//! end, a negative stride steps backwards, and a zero stride or a bound list
//! whose length is not the operand rank is an explicit `ShapeError`. The module
//! doc in [`super`] carries the shared declared-shape and error-wording rules.
use evoxc_core::{DType, Error, Result};

use crate::buffer::Buffer;
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::plan::OpSpec;
use crate::scalar::Scalar;
use crate::with_type;

use super::stride_copy::{CopyAxis, CopyPlan};
use super::{for_each_index, operand, read_checked, reconcile};

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
    if let Some(plan) = copy_plan(ranges, &source_strides) {
        let total = util::numel(&shape);
        let storage = with_type!(source, |values| plan.copy(values, total, "slice"))?;
        return Buffer::new(source.dtype, shape, storage);
    }
    // A backwards walk (a negative step over an axis that keeps more than one
    // element) is the only plan the collapsed-stride copy declines: it stays on
    // the signed general driver.
    let mut base = 0i64;
    let mut steps = Vec::with_capacity(ranges.len());
    for (range, stride) in ranges.iter().zip(&source_strides) {
        base += range.start * *stride as i64;
        steps.push(range.step * *stride as i64);
    }
    with_type!(source, |values| slice_typed(
        values,
        source.dtype,
        &shape,
        base,
        &steps
    ))
}

/// The collapsed-stride copy plan for a slice, or `None` when the mapping
/// walks backwards (a negative step along an axis that keeps two or more
/// elements) — [`CopyPlan`] only describes a forward, non-negative walk, so
/// that case stays on the signed general driver.
///
/// Each axis contributes `(extent, step × source stride)`; [`CopyPlan::new`]
/// then collapses the axes whose strides line up, so the whole slice is one
/// contiguous run, one strided run, or a handful of short blocks.
fn copy_plan(ranges: &[DimSlice], source_strides: &[usize]) -> Option<CopyPlan> {
    let mut base = 0usize;
    let mut axes = Vec::with_capacity(ranges.len());
    for (range, stride) in ranges.iter().zip(source_strides) {
        if range.extent == 0 {
            continue; // an empty result reads nothing
        }
        // A one-element axis (which `CopyPlan` drops again) contributes its
        // start offset only, so even a negative step there is a forward walk.
        let start = usize::try_from(range.start).ok()?;
        base = base.checked_add(start.checked_mul(*stride)?)?;
        if range.extent == 1 {
            continue;
        }
        if range.step < 0 {
            return None;
        }
        let step = usize::try_from(range.step).ok()?;
        axes.push(CopyAxis {
            extent: range.extent,
            stride: step.checked_mul(*stride)?,
        });
    }
    // `axes` is outermost-first, the plan wants innermost-first.
    axes.reverse();
    Some(CopyPlan::new(base, axes))
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

    /// The frame-free reference: normalize the ranges, then read every output
    /// element with plain `i64` offset arithmetic and nested loops.
    fn reference(values: &[f32], strides: &[usize], ranges: &[DimSlice]) -> Vec<f32> {
        let mut out = Vec::new();
        let mut index = vec![0usize; ranges.len()];
        let total: usize = ranges.iter().map(|range| range.extent).product();
        for _ in 0..total {
            let mut offset = 0i64;
            for (axis, range) in ranges.iter().enumerate() {
                offset += (range.start + index[axis] as i64 * range.step) * strides[axis] as i64;
            }
            out.push(values[offset as usize]);
            for axis in (0..ranges.len()).rev() {
                index[axis] += 1;
                if index[axis] < ranges[axis].extent {
                    break;
                }
                index[axis] = 0;
            }
        }
        out
    }

    /// Every fast-path shape the collapsed-stride copy takes — one column of a
    /// 3-D operand (ONE strided run), a partially selected trailing axis (short
    /// contiguous runs), a stepped middle axis and a fully contiguous block —
    /// must reproduce the plain offset walk element for element.
    #[test]
    fn slice_fast_paths_match_an_elementwise_reference() {
        /// `(starts, limits, steps, expected result shape)`.
        type Case = (&'static [i64], &'static [i64], &'static [i64], &'static [usize]);
        let shape = [4usize, 3, 5];
        let values: Vec<f32> = (0..60).map(|value| value as f32 * 0.5).collect();
        let x = f32_buffer(&shape, &values);
        let strides = util::c_strides(&shape);
        let cases: [Case; 6] = [
            // one column of a 3-D operand: a single 12-element stride-15 run
            (&[0, 0, 2], &[4, 3, 3], &[1, 1, 1], &[4, 3, 1]),
            // three of five trailing elements: 12 runs of 3
            (&[0, 0, 1], &[4, 3, 5], &[1, 1, 1], &[4, 3, 4]),
            // a stepped middle axis on top of a whole trailing axis
            (&[0, 0, 0], &[4, 3, 5], &[1, 2, 1], &[4, 2, 5]),
            // a stepped trailing axis (a strided gather, not a contiguous run)
            (&[0, 0, 0], &[4, 3, 5], &[1, 1, 2], &[4, 3, 3]),
            // fully contiguous interior block
            (&[1, 0, 0], &[3, 2, 5], &[1, 1, 1], &[2, 2, 5]),
            // a negative step over a one-element axis stays on the fast path
            (&[0, 0, 4], &[4, 3, 3], &[1, 1, -1], &[4, 3, 1]),
        ];
        for (starts, limits, steps, declared_shape) in cases {
            let op = spec(attrs(&[
                ("start_indices", evoxc_core::AttrValue::Ints(starts.to_vec())),
                ("limit_indices", evoxc_core::AttrValue::Ints(limits.to_vec())),
                ("strides", evoxc_core::AttrValue::Ints(steps.to_vec())),
            ]));
            let ranges = slice_ranges(&op, &shape).unwrap();
            let out = slice(&x, &ranges, &declared(declared_shape)).unwrap();
            assert_eq!(out.shape, declared_shape, "cases {starts:?}/{steps:?}");
            match &out.storage {
                Storage::F32(copied) => {
                    assert_eq!(*copied, reference(&values, &strides, &ranges), "cases {starts:?}/{steps:?}");
                }
                other => panic!("unexpected storage {other:?}"),
            }
        }
    }

    /// The `(500, 50, 4) -> (500, 50, 1)` family — one element out of every
    /// 4-element innermost run — is ONE strided run for every selected offset,
    /// which is the lane-grouped copy path (never a per-element walk).
    #[test]
    fn the_last_axis_single_element_family_is_one_strided_run() {
        let shape = [500usize, 50, 4];
        let values: Vec<f32> = (0..util::numel(&shape)).map(|index| index as f32).collect();
        let x = f32_buffer(&shape, &values);
        let source_strides = util::c_strides(&shape);
        for start in [0i64, 1, 2, 3] {
            let op = spec(attrs(&[
                ("start_indices", evoxc_core::AttrValue::Ints(vec![0, 0, start])),
                (
                    "limit_indices",
                    evoxc_core::AttrValue::Ints(vec![500, 50, start + 1]),
                ),
                ("strides", evoxc_core::AttrValue::Null),
            ]));
            let ranges = slice_ranges(&op, &shape).unwrap();
            let plan = copy_plan(&ranges, &source_strides).expect("a forward plan");
            assert_eq!(
                plan.axes,
                vec![CopyAxis {
                    extent: 25_000,
                    stride: 4
                }],
                "start {start}"
            );
            assert_eq!(plan.base, start as usize);
            let out = slice(&x, &ranges, &declared(&[500, 50, 1])).unwrap();
            assert_eq!(out.shape, vec![500, 50, 1]);
            match &out.storage {
                Storage::F32(copied) => {
                    assert_eq!(copied.len(), 25_000);
                    for (index, value) in copied.iter().enumerate() {
                        assert_eq!(
                            *value,
                            values[index * 4 + start as usize],
                            "start {start}, element {index}"
                        );
                    }
                }
                other => panic!("unexpected storage {other:?}"),
            }
        }
    }

    /// An identity (or whole-array) slice is ONE contiguous run: a single bulk
    /// `copy_from_slice` into a pooled result, for every dtype.
    #[test]
    fn a_whole_array_slice_is_one_contiguous_run() {
        let shape = [4usize, 3];
        let strides = util::c_strides(&shape);
        let op = spec(attrs(&[
            ("start_indices", evoxc_core::AttrValue::Ints(vec![0, 0])),
            ("limit_indices", evoxc_core::AttrValue::Ints(vec![4, 3])),
            ("strides", evoxc_core::AttrValue::Null),
        ]));
        let ranges = slice_ranges(&op, &[4, 3]).unwrap();
        let plan = copy_plan(&ranges, &strides).expect("a forward plan");
        assert_eq!(
            plan.axes,
            vec![CopyAxis {
                extent: 12,
                stride: 1
            }]
        );
        assert_eq!(plan.base, 0);
        // i32 and bool keep the same plan and the same element order.
        let ints: Vec<i32> = (0..12).collect();
        let plan = CopyPlan::new(0, vec![CopyAxis { extent: 12, stride: 1 }]);
        assert_eq!(
            plan.copy(&ints, 12, "slice").unwrap(),
            Storage::I32(ints.clone())
        );
        let flags: Vec<bool> = (0..12).map(|index| index % 3 == 0).collect();
        assert_eq!(
            plan.copy(&flags, 12, "slice").unwrap(),
            Storage::Bool(flags.clone())
        );
    }

    /// A nonzero `start` on every axis of the plan still lands on the right
    /// element: the base offset is the sum of `start x source stride`.
    #[test]
    fn slice_offsets_are_relative_to_every_axis_start() {
        let shape = [3usize, 4];
        let values: Vec<f32> = (0..12).map(|value| value as f32).collect();
        let x = f32_buffer(&shape, &values);
        let op = spec(attrs(&[
            ("start_indices", evoxc_core::AttrValue::Ints(vec![1, 2])),
            ("limit_indices", evoxc_core::AttrValue::Ints(vec![3, 4])),
            ("strides", evoxc_core::AttrValue::Null),
        ]));
        let ranges = slice_ranges(&op, &shape).unwrap();
        let out = slice(&x, &ranges, &declared(&[2, 2])).unwrap();
        assert_eq!(out.storage, Storage::F32(vec![6., 7., 10., 11.]));
        assert_eq!(
            match &out.storage {
                Storage::F32(copied) => copied.clone(),
                other => panic!("unexpected storage {other:?}"),
            },
            reference(&values, &util::c_strides(&shape), &ranges)
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
