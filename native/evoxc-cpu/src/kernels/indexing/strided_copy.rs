//! The one surviving hand-written copy specialization for indexing kernels.
//!
//! Every layout decision in this module family is ndarray's: a `slice` builds
//! its view with `slice_axis_inplace`, `transpose` with `permuted_axes`, and a
//! `gather` selects an axis slice with `index_axis`. Those views are then
//! materialized into a C-contiguous [`Buffer`](crate::buffer::Buffer), because a
//! `Storage` owns one flat `Vec<T>` and never lends a layout out.
//!
//! `ndarray`'s `assign` does that materialization generically, and it is the
//! reference path: every kernel here falls back to it. It walks a strided source
//! element by element, though, which measures ~2.2x the per-element cost of the
//! run-oriented copy below on the strided families the downstream workloads
//! spend their time in (an axis-last `slice (q, m, 4) -> (q, m, 1)`, a
//! non-identity `transpose`, an interleaved last-axis `gather`).
//!
//! This module therefore keeps ONE specialization, and it derives everything
//! from [`ArrayViewD`]s: each side's shape, its element strides and its base
//! offset inside the owning slice (through `as_ptr`, so no layout fact is
//! re-derived by hand). Those are collapsed innermost-first into "outer axis
//! list, innermost contiguous run", and each run is emitted [`LANES`] elements
//! per iteration out of one `stride * LANES`-element source window. Anything the
//! plan cannot describe — a negative stride (a backwards `slice`), a shape or an
//! offset it cannot prove — returns `false` with the destination untouched, and
//! the caller uses ndarray's `assign` instead. So the specialization can only
//! ever change speed, never a value; the kernel tests pin it against `assign`
//! itself.
//!
//! The plan is TWO-SIDED, which is what lets `gather` use it: a range copy names
//! the strides of the source range AND of the destination range, so a
//! destination that is not contiguous (a gathered axis slice of a middle axis)
//! copies just as cheaply as a C-contiguous destination. [`copy_view`] is the
//! one-sided front door (`slice`, `transpose`: a strided source into a
//! C-contiguous destination) and [`AxisCopy`] the `gather` one — one plan whose
//! two bases advance by a fixed step per selected index / result position, so a
//! whole multi-index gather re-uses a single plan.
use ndarray::ArrayViewD;

use crate::scalar::Scalar;

/// Elements written per iteration by the strided run filler (one source window
/// of `LANES * stride` elements).
const LANES: usize = 4;

/// One axis of a two-sided copy plan, innermost first: `extent` elements whose
/// source stride is `src` and destination stride is `dst`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Step {
    extent: usize,
    src: usize,
    dst: usize,
}

/// A copy plan between two STRIDED ranges of two owning slices.
///
/// Built once from the two [`ArrayViewD`]s describing the ranges and then
/// replayed per (source base, destination base) pair; it holds no borrows, so a
/// caller can build it while reading the destination and then write to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CopyPlan {
    steps: Vec<Step>,
    src_base: usize,
    dst_base: usize,
}

impl CopyPlan {
    /// The plan copying `view` (borrowing `source`) into `dest`, whose layout is
    /// `dest_view`'s (the two views must have the same shape).
    ///
    /// `None` when a layout fact cannot be proven: mismatched shapes or lengths,
    /// a zero extent, or a negative/oversized stride.
    fn build<T: Scalar>(
        source: &[T],
        view: &ArrayViewD<'_, T>,
        dest: &[T],
        dest_view: &ArrayViewD<'_, T>,
    ) -> Option<CopyPlan> {
        if view.shape() != dest_view.shape() || view.len() != dest.len() {
            return None;
        }
        Some(CopyPlan {
            steps: plan_steps(view, dest_view)?,
            src_base: base_offset(view, source)?,
            dst_base: base_offset(dest_view, dest)?,
        })
    }

    /// Copy the whole planned range. Returns `true` when `dest` was fully
    /// written, `false` when the plan did not hold (nothing to do then).
    pub(super) fn copy<T: Scalar>(&self, source: &[T], dest: &mut [T]) -> bool {
        copy_at(&self.steps, source, self.src_base, dest, self.dst_base)
    }
}

/// A copy plan for a gather-style axis slice.
///
/// The gathered axis slice has the same shape and strides at every axis
/// position; only its BASE moves — by `src_step` in the source slice per
/// selected index, and by `dst_step` in the destination slice per result
/// position. One plan therefore serves every index of a multi-index gather.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct AxisCopy {
    steps: Vec<Step>,
    src_base: usize,
    dst_base: usize,
    src_step: usize,
    dst_step: usize,
}

impl AxisCopy {
    /// Copy the axis slice at `selected` into the axis slice at `position`.
    ///
    /// Returns `true` when that destination range was fully written.
    pub(super) fn copy<T: Scalar>(
        &self,
        source: &[T],
        selected: usize,
        dest: &mut [T],
        position: usize,
    ) -> bool {
        copy_at(
            &self.steps,
            source,
            self.src_base + selected * self.src_step,
            dest,
            self.dst_base + position * self.dst_step,
        )
    }
}

/// Build the [`AxisCopy`] plan for `gather`'s per-index slice copy.
///
/// `source_view` views all of `source` as `source_shape`; `dest_shape` is the
/// shape whose every `dest_axis` slice receives one selected source axis slice.
/// Both layouts are read out of ndarray views of the two owning slices, so this
/// module re-derives no stride by hand.
pub(super) fn plan_axis_copy<T: Scalar>(
    source: &[T],
    source_view: &ArrayViewD<'_, T>,
    source_axis: usize,
    dest: &[T],
    dest_shape: &[usize],
    dest_axis: usize,
) -> Option<AxisCopy> {
    // A zero-extent axis has no slice to take (and `index_axis` would panic).
    if *source_view.shape().get(source_axis)? == 0 || *dest_shape.get(dest_axis)? == 0 {
        return None;
    }
    let src_slice = source_view.index_axis(ndarray::Axis(source_axis), 0);
    let dst_view = ArrayViewD::<T>::from_shape(ndarray::IxDyn(dest_shape), dest).ok()?;
    let dst_slice = dst_view.index_axis(ndarray::Axis(dest_axis), 0);
    if src_slice.shape() != dst_slice.shape() {
        return None;
    }
    Some(AxisCopy {
        steps: plan_steps(&src_slice, &dst_slice)?,
        src_base: base_offset(&src_slice, source)?,
        dst_base: base_offset(&dst_slice, dest)?,
        // The step from one axis slice's base to the next.
        src_step: usize::try_from(source_view.stride_of(ndarray::Axis(source_axis))).ok()?,
        dst_step: usize::try_from(dst_view.stride_of(ndarray::Axis(dest_axis))).ok()?,
    })
}

/// Copy `view` into `dest`, which must be exactly `view.len()` elements long.
///
/// Returns `true` when the whole destination was written from `source` (the
/// slice the view borrows), `false` when this specialization does not apply — in
/// which case `dest` is untouched and the caller must fall back to
/// [`ndarray::ArrayBase::assign`].
pub(super) fn copy_view<T: Scalar>(source: &[T], view: &ArrayViewD<'_, T>, dest: &mut [T]) -> bool {
    if view.len() != dest.len() {
        return false;
    }
    if dest.is_empty() {
        return true;
    }
    // The destination of a `slice`/`transpose` is C-contiguous in `view`'s
    // shape: the layout comes from a view of it, exactly like the source's.
    let plan = {
        let Ok(dest_view) = ArrayViewD::<T>::from_shape(ndarray::IxDyn(view.shape()), &*dest) else {
            return false;
        };
        CopyPlan::build(source, view, &*dest, &dest_view)
    };
    match plan {
        Some(plan) => plan.copy(source, dest),
        None => false,
    }
}

/// The view's first element as an index into `source` (the slice it borrows).
///
/// `ArrayViewD` exposes its base address, not its offset, so the offset is the
/// address difference — exactly representable because `view` borrows `source`.
fn base_offset<T>(view: &ArrayViewD<'_, T>, source: &[T]) -> Option<usize> {
    let start = view.as_ptr() as usize;
    let origin = source.as_ptr() as usize;
    let width = std::mem::size_of::<T>();
    if start < origin || width == 0 {
        return None;
    }
    let bytes = start - origin;
    if bytes % width != 0 {
        return None;
    }
    let base = bytes / width;
    if base + view.len() > source.len() {
        return None;
    }
    Some(base)
}

/// Collapse the two equally shaped views' axes innermost-first, dropping
/// extent-1 axes and folding every axis whose stride continues the inner one on
/// BOTH sides (`stride_outer == stride_inner * extent_inner`).
///
/// `None` for a zero extent or a negative stride — the collapsed plan only
/// describes a forward walk, and a backwards `slice` keeps ndarray's `assign`.
fn plan_steps<T: Scalar>(
    view: &ArrayViewD<'_, T>,
    dest_view: &ArrayViewD<'_, T>,
) -> Option<Vec<Step>> {
    let mut steps: Vec<Step> = Vec::with_capacity(view.ndim().min(dest_view.ndim()));
    for index in (0..view.ndim()).rev() {
        let extent = view.shape()[index];
        if extent == 0 {
            return None;
        }
        if extent == 1 {
            continue;
        }
        let src = usize::try_from(view.strides()[index]).ok()?;
        let dst = usize::try_from(dest_view.strides()[index]).ok()?;
        match steps.last_mut() {
            Some(inner)
                if inner.src.checked_mul(inner.extent) == Some(src)
                    && inner.dst.checked_mul(inner.extent) == Some(dst) =>
            {
                inner.extent = inner.extent.checked_mul(extent)?;
            }
            _ => steps.push(Step { extent, src, dst }),
        }
    }
    Some(steps)
}

/// How many outer axes the replay odometer keeps on the stack, so a per-index
/// [`AxisCopy::copy`] allocation-free for every rank the engine sees. A deeper
/// operand falls back to a heap odometer (a rank above 9 is exotic, and the
/// fallback is still one allocation per CALL, not per element).
const MAX_STACK_AXES: usize = 8;

/// Replay a plan's runs from `src_base` in `source` to `dst_base` in `dest`.
///
/// Both running offsets are the START of the current run, so a run does not
/// advance them — the odometer below displaces them by each outer axis's own
/// step (the innermost OUTER axis varies fastest, so a carry walks outward one
/// axis at a time).
fn copy_at<T: Scalar>(
    steps: &[Step],
    source: &[T],
    src_base: usize,
    dest: &mut [T],
    dst_base: usize,
) -> bool {
    let Some((run, outer)) = steps.split_first() else {
        // Every extent was 1: the plan is a single element.
        return match (source.get(src_base), dest.get_mut(dst_base)) {
            (Some(value), Some(slot)) => {
                *slot = *value;
                true
            }
            _ => false,
        };
    };
    let run = *run;
    if outer.is_empty() {
        // ONE run and no outer axis — the whole plan is a single run copy
        // (every flat-table and interleaved gather collapse to this), so there
        // is no odometer and no allocation at all.
        return write_run(run, source, src_base, dest, dst_base);
    }
    let mut stack = [0usize; MAX_STACK_AXES];
    let mut heap;
    let odometer: &mut [usize] = if outer.len() <= MAX_STACK_AXES {
        &mut stack[..outer.len()]
    } else {
        heap = vec![0usize; outer.len()];
        heap.as_mut_slice()
    };
    let mut src = src_base;
    let mut dst = dst_base;
    loop {
        if !write_run(run, source, src, dest, dst) {
            return false;
        }
        let mut axis = 0usize;
        loop {
            if axis == odometer.len() {
                return true;
            }
            let outer = outer[axis];
            odometer[axis] += 1;
            src += outer.src;
            dst += outer.dst;
            if odometer[axis] < outer.extent {
                break;
            }
            src -= outer.src * outer.extent;
            dst -= outer.dst * outer.extent;
            odometer[axis] = 0;
            axis += 1;
        }
    }
}

/// Emit one planned run: a contiguous destination run takes one `copy_from_slice`
/// (or a [`LANES`]-grouped strided fill), a strided destination an element walk.
fn write_run<T: Scalar>(
    run: Step,
    source: &[T],
    src: usize,
    dest: &mut [T],
    dst: usize,
) -> bool {
    if run.dst == 1 {
        match dest.get_mut(dst..dst + run.extent) {
            Some(chunk) => fill_run(source, src, chunk, run.src),
            None => false,
        }
    } else {
        copy_elements(source, src, dest, dst, run.extent, run.src, run.dst)
    }
}

/// Fill one C-contiguous destination run from `source[offset..]`, taking every
/// `stride`-th element (a contiguous source is one memcpy).
fn fill_run<T: Scalar>(source: &[T], offset: usize, dest: &mut [T], stride: usize) -> bool {
    if stride == 1 {
        return match source.get(offset..offset + dest.len()) {
            Some(run) => {
                dest.copy_from_slice(run);
                true
            }
            None => false,
        };
    }
    fill_strided(source, offset, dest, stride)
}

/// Fill one contiguous destination run from `source[offset..]` taking every
/// `stride`-th element, [`LANES`] elements per iteration.
fn fill_strided<T: Scalar>(source: &[T], offset: usize, dest: &mut [T], stride: usize) -> bool {
    let mut out = 0usize;
    let mut base = offset;
    while out + LANES <= dest.len() {
        let Some(window) = source.get(base..base + stride * LANES) else {
            break;
        };
        dest[out] = window[0];
        dest[out + 1] = window[stride];
        dest[out + 2] = window[2 * stride];
        dest[out + 3] = window[3 * stride];
        out += LANES;
        base += stride * LANES;
    }
    while out < dest.len() {
        match source.get(base) {
            Some(value) => {
                dest[out] = *value;
                out += 1;
                base += stride;
            }
            None => return false,
        }
    }
    true
}

/// Copy a run element by element when the DESTINATION is strided too (neither
/// side is a contiguous block, so there is no bulk copy to make).
fn copy_elements<T: Scalar>(
    source: &[T],
    src_base: usize,
    dest: &mut [T],
    dst_base: usize,
    extent: usize,
    src_stride: usize,
    dst_stride: usize,
) -> bool {
    let mut src = src_base;
    let mut dst = dst_base;
    for _ in 0..extent {
        match (source.get(src), dest.get_mut(dst)) {
            (Some(value), Some(slot)) => *slot = *value,
            _ => return false,
        }
        src += src_stride;
        dst += dst_stride;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, IxDyn, Slice};

    /// Every strided family the kernels use must equal ndarray's own `assign`
    /// exactly — this is the pin that lets the specialization exist.
    #[test]
    fn copy_view_agrees_with_ndarray_assign_on_every_stride() {
        let source: Vec<f32> = (0..24).map(|value| value as f32).collect();
        let base = Array::from_shape_vec(IxDyn(&[4, 3, 2]), source).unwrap();
        let storage = base.as_slice().unwrap();
        for axis in 0..3usize {
            for start in 0..3usize {
                for step in [1isize, 2] {
                    let mut view = base.view();
                    view.slice_axis_inplace(
                        ndarray::Axis(axis),
                        Slice::new(start as isize, None, step),
                    );
                    let mut reference = ndarray::Array::<f32, _>::zeros(view.raw_dim());
                    reference.assign(&view);
                    let mut actual = vec![0f32; view.len()];
                    assert!(
                        copy_view(storage, &view, &mut actual),
                        "axis {axis} start {start} step {step}"
                    );
                    assert_eq!(
                        actual,
                        reference.into_raw_vec_and_offset().0,
                        "axis {axis} start {start} step {step}"
                    );
                }
            }
        }
    }

    /// A backwards view keeps ndarray's `assign`: the plan declines it.
    #[test]
    fn copy_view_declines_a_negative_stride() {
        let base = Array::from_shape_vec(IxDyn(&[3, 2]), (0..6).collect::<Vec<i32>>()).unwrap();
        let storage = base.as_slice().unwrap();
        let mut view = base.view();
        view.slice_axis_inplace(ndarray::Axis(0), Slice::new(0, None, -1));
        let mut dest = vec![0i32; view.len()];
        assert!(!copy_view(storage, &view, &mut dest));
        assert_eq!(dest, vec![0i32; 6], "a declined copy leaves the destination untouched");
    }

    /// The family the specialization exists for: `slice`'s `(500, 50, 4) ->
    /// (500, 50, 1)` collapses to ONE axis of 25 000 elements 4 apart, so it is
    /// a single lane-grouped run rather than 500 short strided walks.
    #[test]
    fn the_axis_last_single_element_family_is_one_strided_axis() {
        let source: Vec<f32> = (0..100_000).map(|value| value as f32).collect();
        let base = Array::from_shape_vec(IxDyn(&[500, 50, 4]), source).unwrap();
        let storage = base.as_slice().unwrap();
        let mut view = base.view();
        view.slice_axis_inplace(ndarray::Axis(2), Slice::new(0, Some(1), 1));
        assert_eq!(view.shape(), &[500, 50, 1]);
        let dest_view = Array::<f32, _>::zeros(IxDyn(&[500, 50, 1]));
        assert_eq!(
            plan_steps(&view, &dest_view.view()),
            Some(vec![Step { extent: 25_000, src: 4, dst: 1 }])
        );
        let mut dest = vec![0f32; view.len()];
        assert!(copy_view(storage, &view, &mut dest));
        for (position, value) in dest.iter().enumerate() {
            assert_eq!(*value, (position * 4) as f32, "element {position}");
        }
    }

    /// A pooled destination is fully overwritten by the strided filler: park a
    /// sentinel and confirm no element survives.
    #[test]
    fn copy_view_overwrites_every_destination_element() {
        let source: Vec<f32> = (0..16).map(|value| value as f32).collect();
        let base = Array::from_shape_vec(IxDyn(&[4, 4]), source).unwrap();
        let storage = base.as_slice().unwrap();
        let mut view = base.view();
        view.slice_axis_inplace(ndarray::Axis(1), Slice::new(1, Some(2), 1));
        let mut dest = vec![-1.0f32; view.len()];
        assert!(copy_view(storage, &view, &mut dest));
        assert!(!dest.contains(&-1.0), "a stale element survived the copy");
    }

    /// The `gather` front door on the family that regressed: an interleaved
    /// LAST-axis gather of a `(500, 100, 2)` operand into a `(500, 100)` result
    /// is ONE strided run of 50 000 elements, and it must equal ndarray's
    /// `assign` reference element for element.
    #[test]
    fn axis_copy_covers_an_interleaved_last_axis() {
        let source: Vec<f32> = (0..100_000).map(|value| value as f32).collect();
        let flat = Array::from_shape_vec(IxDyn(&[500, 100, 2]), source.clone()).unwrap();
        let storage = flat.as_slice().unwrap();
        let shape = [500usize, 100, 2];
        let view = ArrayViewD::<f32>::from_shape(IxDyn(&shape), storage).unwrap();
        let mut out = vec![0f32; 50_000];
        let plan = plan_axis_copy(storage, &view, 2, &out, &[500, 100, 1], 2).expect("a plan");
        assert!(plan.copy(storage, 1, &mut out, 0));
        let mut reference = Array::<f32, _>::zeros(IxDyn(&[500, 100]));
        reference.assign(&view.index_axis(ndarray::Axis(2), 1));
        assert_eq!(out, reference.as_slice().unwrap());
    }

    /// A strided DESTINATION — an interleaved last axis of a multi-index gather,
    /// where every result position's slice takes every `count`-th element —
    /// copies element by element and still matches ndarray's `assign`.
    #[test]
    fn axis_copy_covers_an_interleaved_destination() {
        let source: Vec<f32> = (0..100_000).map(|value| value as f32).collect();
        let flat = Array::from_shape_vec(IxDyn(&[500, 100, 2]), source.clone()).unwrap();
        let storage = flat.as_slice().unwrap();
        let shape = [500usize, 100, 2];
        let view = ArrayViewD::<f32>::from_shape(IxDyn(&shape), storage).unwrap();
        let mut out = vec![0f32; 100_000];
        let plan = plan_axis_copy(storage, &view, 2, &out, &[500, 100, 2], 2).expect("a plan");
        let mut reference = Array::<f32, _>::zeros(IxDyn(&[500, 100, 2]));
        for (position, selected) in [1usize, 0].into_iter().enumerate() {
            reference
                .index_axis_mut(ndarray::Axis(2), position)
                .assign(&view.index_axis(ndarray::Axis(2), selected));
            assert!(plan.copy(storage, selected, &mut out, position), "position {position}");
            assert_eq!(out, reference.as_slice().unwrap(), "position {position}");
        }
    }

    /// A strided DESTINATION (a gathered MIDDLE axis of a C-contiguous result)
    /// copies run by run and still matches ndarray's `assign`.
    #[test]
    fn axis_copy_covers_a_non_contiguous_destination() {
        let source: Vec<i32> = (0..12i32).collect();
        let shape = [2usize, 3, 2];
        let flat = Array::from_shape_vec(IxDyn(&shape), source.clone()).unwrap();
        let storage = flat.as_slice().unwrap();
        let view = ArrayViewD::<i32>::from_shape(IxDyn(&shape), storage).unwrap();
        let mut out = vec![0i32; 2 * 2 * 2];
        let plan = plan_axis_copy(storage, &view, 1, &out, &[2, 2, 2], 1).expect("a plan");
        let mut reference = Array::<i32, _>::zeros(IxDyn(&[2, 2, 2]));
        for (position, selected) in [2usize, 0].into_iter().enumerate() {
            reference
                .index_axis_mut(ndarray::Axis(1), position)
                .assign(&view.index_axis(ndarray::Axis(1), selected));
            assert!(plan.copy(storage, selected, &mut out, position), "position {position}");
            assert_eq!(out, reference.as_slice().unwrap(), "position {position}");
        }
    }

    /// A zero-extent gathered axis has no slice to take: the plan declines so
    /// the caller takes ndarray's `assign` (which is a no-op there).
    #[test]
    fn axis_copy_declines_a_zero_extent_axis() {
        let storage: Vec<f32> = Vec::new();
        let view = ArrayViewD::<f32>::from_shape(IxDyn(&[0usize, 3]), storage.as_slice()).unwrap();
        let out: Vec<f32> = vec![0.0; 3];
        assert!(plan_axis_copy(&storage, &view, 0, &out, &[1, 3], 0).is_none());
    }
}
