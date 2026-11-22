//! `slice` kernel: Python `slice(start, limit, stride)` per dim.
//!
//! Bounds CLAMP (Python slicing never raises), a negative bound counts from the
//! end, a negative stride steps backwards, and a zero stride or a bound list
//! whose length is not the operand rank is an explicit `ShapeError`. The module
//! doc in [`super`] carries the shared declared-shape and error-wording rules.
use evoxc_core::{DType, Error, Result};

use crate::buffer::{array_view, array_view_mut, Buffer, NdScalar};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::plan::OpSpec;
use crate::scalar::Scalar;
use crate::with_type;

use super::strided_copy;
use super::{operand, pooled_total, reconcile};

/// One operand axis mapped onto a Python `slice(start, stop, step)`.
#[derive(Debug, Clone, Copy)]
struct DimSlice {
    start: i64,
    /// The normalized, clamped Python STOP bound (exclusive): the low bound of a
    /// negative step, the high bound of a positive one.
    stop: i64,
    step: i64,
    extent: usize,
}

impl DimSlice {
    /// The equivalent [`ndarray::Slice`] for this axis.
    ///
    /// The bounds are already Python-normalized and clamped, so they are
    /// non-negative here except for a negative step's exclusive low bound; that
    /// one is shifted to the inclusive `stop + 1` ndarray reverses from.
    fn to_ndarray(self) -> ndarray::Slice {
        if self.extent == 0 {
            // An empty axis selects nothing regardless of the bounds.
            return ndarray::Slice::new(0, Some(0), 1);
        }
        if self.step > 0 {
            ndarray::Slice::new(self.start as isize, Some(self.stop as isize), self.step as isize)
        } else {
            // A negative step walks `start, start+step, ... down to stop+1`, which
            // ndarray expresses as the range `[stop+1, start+1)` reversed.
            ndarray::Slice::new(
                (self.stop + 1) as isize,
                Some((self.start + 1) as isize),
                self.step as isize,
            )
        }
    }
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
    Ok(DimSlice { start, stop, step, extent })
}

fn slice(source: &Buffer, ranges: &[DimSlice], declared: &[Option<usize>]) -> Result<Buffer> {
    let derived: Vec<usize> = ranges.iter().map(|range| range.extent).collect();
    let shape = reconcile("slice", declared, derived)?;
    with_type!(source, |values| slice_typed(
        values,
        source.dtype,
        &source.shape,
        ranges,
        shape
    ))
}

/// Materialize the sliced view through ndarray.
///
/// One [`ndarray::Slice`] per axis is applied to a borrowed view with
/// `slice_axis_inplace` (Python's bounds/steps, including a negative step), so
/// the addressing is ndarray's — no per-axis offset table and no per-element
/// odometer. A view that lands C-contiguous (a whole-array or interior-block
/// slice) copies with ONE `copy_from_slice` into the pooled destination; a
/// strided one goes to [`super::strided_copy`]'s lane-grouped run copy, and
/// every layout it declines (a backwards step) receives ndarray's `assign`.
fn slice_typed<T: Scalar + NdScalar>(
    values: &[T],
    dtype: DType,
    source_shape: &[usize],
    ranges: &[DimSlice],
    shape: Vec<usize>,
) -> Result<Buffer> {
    let mut view = array_view(values, source_shape)?;
    for (axis, range) in ranges.iter().enumerate() {
        view.slice_axis_inplace(ndarray::Axis(axis), range.to_ndarray());
    }
    let total = util::numel(&shape);
    let mut out = pooled_total::<T>(total);
    if let Some(source) = view.as_slice() {
        out.copy_from_slice(source);
    } else if !strided_copy::copy_view(values, &view, &mut out) {
        let mut dest = array_view_mut(&mut out, &shape)?;
        dest.assign(&view);
    }
    Buffer::new(dtype, shape, <T as NdScalar>::into_storage(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::indexing::test_support::{attrs, buffer, declared, f32_buffer, spec};
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
    /// 4-element innermost run — must read exactly the reference's elements for
    /// every selected offset; the ndarray view is non-contiguous here, so this
    /// pins the `assign` path against the independent per-element oracle.
    #[test]
    fn the_last_axis_single_element_family_matches_the_reference() {
        let shape = [500usize, 50, 4];
        let values: Vec<f32> = (0..util::numel(&shape)).map(|index| index as f32).collect();
        let x = f32_buffer(&shape, &values);
        let strides = util::c_strides(&shape);
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
            let out = slice(&x, &ranges, &declared(&[500, 50, 1])).unwrap();
            assert_eq!(out.shape, vec![500, 50, 1]);
            match &out.storage {
                Storage::F32(copied) => {
                    assert_eq!(copied.len(), 25_000);
                    assert_eq!(*copied, reference(&values, &strides, &ranges), "start {start}");
                }
                other => panic!("unexpected storage {other:?}"),
            }
        }
    }

    /// A whole-array slice lands C-contiguous, so it takes the single
    /// `copy_from_slice` fast path (`as_slice()` is `Some`) and reproduces the
    /// operand's element order for every dtype.
    #[test]
    fn a_whole_array_slice_is_one_contiguous_run() {
        let shape = [4usize, 3];
        let op = spec(attrs(&[
            ("start_indices", evoxc_core::AttrValue::Ints(vec![0, 0])),
            ("limit_indices", evoxc_core::AttrValue::Ints(vec![4, 3])),
            ("strides", evoxc_core::AttrValue::Null),
        ]));
        let ranges = slice_ranges(&op, &shape).unwrap();
        // Pin the bulk-memcpy gate: the sliced ndarray view is standard layout.
        let plane: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mut view = array_view::<f32>(&plane, &shape).expect("a view");
        for (axis, range) in ranges.iter().enumerate() {
            view.slice_axis_inplace(ndarray::Axis(axis), range.to_ndarray());
        }
        assert!(view.as_slice().is_some(), "a whole-array slice is contiguous");
        // f32, i32 and bool keep the same element order through `slice`.
        let floats = f32_buffer(&shape, &(0..12).map(|i| i as f32).collect::<Vec<_>>());
        assert_eq!(slice(&floats, &ranges, &declared(&shape)).unwrap().storage, floats.storage);
        let ints = buffer(evoxc_core::DType::Int32, &shape, Storage::I32((0..12).collect()));
        assert_eq!(slice(&ints, &ranges, &declared(&shape)).unwrap().storage, ints.storage);
        let flags = buffer(
            evoxc_core::DType::Bool,
            &shape,
            Storage::Bool((0..12).map(|index| index % 3 == 0).collect()),
        );
        assert_eq!(slice(&flags, &ranges, &declared(&shape)).unwrap().storage, flags.storage);
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
