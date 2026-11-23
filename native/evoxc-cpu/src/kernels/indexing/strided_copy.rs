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
use evoxc_core::{Error, Result};
use ndarray::ArrayViewD;

use crate::scalar::Scalar;

/// Elements written per iteration by the strided run filler (one source window
/// of `LANES * stride` elements).
const LANES: usize = 4;

/// Runs at least this long are one `copy_from_slice`; shorter ones are an
/// element loop. A `gather`'s collapsed run is typically 1–8 elements, where the
/// memcpy dispatch costs more than the 4–32 bytes it moves.
const RUN_MEMCPY_MIN: usize = 32;

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

/// A gather's per-index copy COLLAPSED to one run: `run` elements read from
/// `source` at `src_base + selected * src_step` (stride `src_stride` between
/// them), written at `position * dst_step` in a C-contiguous destination.
///
/// This is the shape family the downstream workloads spend their time in — a
/// flat table gathered with a whole block of indices, which is one contiguous
/// run per index (`src_stride == 1`, the axis-0 and rank-1 families) or one
/// strided run (`src_stride > 1`, an axis-last gather). It is a plain value the
/// CALLER's loop drives, which is the point: resolving an index and copying its
/// run stay in one inlined loop instead of a per-index
/// `copy` → `copy_at` → `write_run` chain, and no per-index plan is replayed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RunCopy {
    run: usize,
    src_base: usize,
    src_stride: usize,
    src_step: usize,
    dst_stride: usize,
    dst_step: usize,
}

impl RunCopy {
    /// True when consecutive result positions continue the destination in
    /// order, so the whole destination can be APPENDED run by run — which is
    /// what lets a destination be built with no zero fill at all.
    ///
    /// True exactly when the gathered axis is the outermost one: every position
    /// then takes the next `run` contiguous elements.
    pub(super) fn appends(&self) -> bool {
        self.dst_stride == 1 && self.dst_step == self.run
    }

    /// Copy every selected run, APPENDING into `dest` in destination order:
    /// `dest` starts empty and receives exactly `count * run` elements, so a
    /// caller may hand it a destination that was never zero-filled.
    pub(super) fn append_runs<T: Scalar, I>(
        &self,
        source: &[T],
        selected: I,
        dest: &mut Vec<T>,
    ) -> Result<()>
    where
        I: Iterator<Item = Result<usize>>,
    {
        for index in selected {
            let start = self.src_base + index? * self.src_step;
            if self.run == 1 {
                // One element per index — the flat-table family: a single push
                // and no run slice at all.
                match source.get(start) {
                    Some(value) => dest.push(*value),
                    None => return Err(run_out_of_bounds(start, 1, source.len())),
                }
                continue;
            }
            let Some(run) = source.get(start..start + self.run) else {
                return Err(run_out_of_bounds(start, self.run, source.len()));
            };
            match self.src_stride {
                1 if self.run >= RUN_MEMCPY_MIN => dest.extend_from_slice(run),
                // A short contiguous run is a handful of loads and stores; the
                // elementwise extend beats a memcpy call at these lengths.
                1 => dest.extend(run.iter().copied()),
                // A stride-0 read repeats one source value (never produced by a
                // C-contiguous operand, kept sound for a future caller).
                0 => {
                    if let Some(value) = run.first() {
                        dest.extend(std::iter::repeat_n(*value, self.run));
                    }
                }
                stride => {
                    let mut cursor = start;
                    for _ in 0..self.run {
                        let Some(value) = source.get(cursor) else {
                            return Err(run_out_of_bounds(cursor, 1, source.len()));
                        };
                        dest.push(*value);
                        cursor += stride;
                    }
                }
            }
        }
        Ok(())
    }

    /// Copy every selected run into a pre-sized destination, each run starting
    /// at `position * dst_step` (the interleaved families: a gathered innermost
    /// axis, where consecutive result positions interleave element by element).
    pub(super) fn place_runs<T: Scalar, I>(
        &self,
        source: &[T],
        selected: I,
        dest: &mut [T],
    ) -> Result<()>
    where
        I: Iterator<Item = Result<usize>>,
    {
        for (position, index) in selected.enumerate() {
            let start = self.src_base + index? * self.src_step;
            let at = position * self.dst_step;
            if self.dst_stride == 1 {
                let Some(slot) = dest.get_mut(at..at + self.run) else {
                    return Err(run_out_of_bounds(at, self.run, dest.len()));
                };
                if !fill_run(source, start, slot, self.src_stride) {
                    return Err(run_out_of_bounds(start, self.run, source.len()));
                }
                continue;
            }
            // Both sides strided: the destination run takes every `dst_stride`-th
            // element too, so the run is an element walk (still one inlined loop
            // with no odometer and no per-index plan).
            let mut cursor = start;
            let mut slot = at;
            for _ in 0..self.run {
                match (source.get(cursor), dest.get_mut(slot)) {
                    (Some(value), Some(target)) => *target = *value,
                    (Some(_), None) => return Err(run_out_of_bounds(slot, self.run, dest.len())),
                    (None, _) => return Err(run_out_of_bounds(cursor, self.run, source.len())),
                }
                cursor += self.src_stride;
                slot += self.dst_stride;
            }
        }
        Ok(())
    }
}

/// Classify a gather's per-index axis-slice copy as ONE run (see [`RunCopy`]).
///
/// `source_view` views all of `source` as `source_shape`; `dest_shape` is the
/// shape whose every `dest_axis` slice receives one selected source axis slice
/// and is C-contiguous, like every [`Buffer`](crate::buffer::Buffer) storage, so
/// its strides are the C strides of `dest_shape` with that axis removed.
///
/// `None` when the range is not one run: gathering a MIDDLE axis splits the
/// range at that axis (the leading part's destination stride is scaled by the
/// index count), so those keep the general multi-run replay.
pub(super) fn plan_axis_runs<T: Scalar>(
    source: &[T],
    source_view: &ArrayViewD<'_, T>,
    source_axis: usize,
    dest_shape: &[usize],
    dest_axis: usize,
) -> Option<RunCopy> {
    if *source_view.shape().get(source_axis)? == 0 || *dest_shape.get(dest_axis)? == 0 {
        return None;
    }
    let src_slice = source_view.index_axis(ndarray::Axis(source_axis), 0);
    let range = src_slice.shape();
    // The destination slice's strides: a C-contiguous `dest_shape` with the
    // gathered axis removed, where every axis BEFORE the gathered one also spans
    // the `count` index positions it sits outside of.
    let count = *dest_shape.get(dest_axis)?;
    let mut dst_strides: Vec<isize> = crate::kernels::util::c_strides(range)
        .into_iter()
        .map(|stride| stride as isize)
        .collect();
    for stride in dst_strides.iter_mut().take(dest_axis.min(range.len())) {
        *stride *= count as isize;
    }
    let steps = collapse_steps(range, src_slice.strides(), &dst_strides)?;
    // ONE run per index, for the whole range. An EMPTY collapse is a rank-0 (or
    // all-extent-1) range — one element per index, the flat-table family every
    // attribute/window lookup takes.
    let (extent, src_stride, dst_stride) = match steps.split_first() {
        Some((step, [])) => (step.extent, step.src, step.dst),
        Some(_) => return None,
        None => (1, 1, 1),
    };
    Some(RunCopy {
        run: extent,
        src_base: base_offset(&src_slice, source)?,
        src_stride,
        src_step: usize::try_from(source_view.stride_of(ndarray::Axis(source_axis))).ok()?,
        dst_stride,
        // The destination slice at position 0 starts at element 0 of a
        // C-contiguous destination, and every position advances by the gathered
        // axis' own C stride.
        dst_step: dest_shape[dest_axis + 1..].iter().product(),
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

/// An out-of-range run diagnostic: the run arms WRITE as they go (no fallback
/// pass rewrites the destination afterwards), so a run the layout cannot
/// accommodate is an explicit error rather than a silently short copy. A
/// C-contiguous operand whose shape produced the plan never triggers it.
#[cold]
#[inline(never)]
fn run_out_of_bounds(offset: usize, extent: usize, len: usize) -> Error {
    Error::backend(format!(
        "indexing copy: a {extent}-element run at offset {offset} is outside a {len}-element buffer — the declared result shape does not match the operands"
    ))
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
    collapse_steps(view.shape(), view.strides(), dest_view.strides())
}

/// [`plan_steps`] over explicit shapes/strides, so a caller that knows its
/// destination's layout without materializing a view of it (a C-contiguous
/// result buffer) shares the same collapsing rule.
fn collapse_steps(shape: &[usize], src_strides: &[isize], dst_strides: &[isize]) -> Option<Vec<Step>> {
    let mut steps: Vec<Step> = Vec::with_capacity(shape.len());
    for index in (0..shape.len()).rev() {
        let extent = *shape.get(index)?;
        if extent == 0 {
            return None;
        }
        if extent == 1 {
            continue;
        }
        let src = usize::try_from(*src_strides.get(index)?).ok()?;
        let dst = usize::try_from(*dst_strides.get(index)?).ok()?;
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
#[inline]
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
/// `stride`-th element (a contiguous source is one memcpy when the run is long
/// enough to amortize it, an element loop when it is not).
#[inline]
fn fill_run<T: Scalar>(source: &[T], offset: usize, dest: &mut [T], stride: usize) -> bool {
    if stride == 1 {
        let Some(run) = source.get(offset..offset + dest.len()) else {
            return false;
        };
        if run.len() >= RUN_MEMCPY_MIN {
            dest.copy_from_slice(run);
        } else {
            // The short-run arm every collapsed gather/slice family lands on: a
            // handful of loads and stores, with no memcpy dispatch between them.
            for (slot, value) in dest.iter_mut().zip(run) {
                *slot = *value;
            }
        }
        return true;
    }
    if stride == 0 {
        // A stride-0 read repeats one source value; a C-contiguous operand never
        // produces one, but the run filler stays sound if it ever does.
        return match source.get(offset) {
            Some(value) => {
                dest.fill(*value);
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

    /// Append `selected`'s runs into a fresh destination, the way `gather`'s
    /// non-zero-fill arm does, and compare with ndarray's `assign` reference for
    /// the same selection.
    fn appended<T: Scalar + Copy>(
        storage: &[T],
        view: &ArrayViewD<'_, T>,
        axis: usize,
        dest_shape: &[usize],
        selected: &[usize],
    ) -> (RunCopy, Vec<T>) {
        let runs = plan_axis_runs(storage, view, axis, dest_shape, axis).expect("one run per index");
        let mut out: Vec<T> = Vec::new();
        runs.append_runs(storage, selected.iter().map(|&index| Ok(index)), &mut out)
            .expect("the runs copy");
        (runs, out)
    }

    /// The flat-table family (the downstream workloads' own shape) — a rank-1
    /// operand (one element per index) and a trailing run of 2 — collapses to
    /// ONE run per index with an APPENDABLE destination, and the append is
    /// bit-identical to ndarray's `assign`.
    #[test]
    fn append_runs_covers_the_flat_table_family() {
        let shape = [50_000usize, 2];
        let values: Vec<f32> = (0..100_000).map(|value| value as f32 * 1.5).collect();
        let base = Array::from_shape_vec(IxDyn(&shape), values.clone()).unwrap();
        let storage = base.as_slice().unwrap();
        let view = ArrayViewD::<f32>::from_shape(IxDyn(&shape), storage).unwrap();
        let selected: Vec<usize> = (0..5_000).map(|position| (position * 7919 + 3) % 50_000).collect();

        let (runs, out) = appended(storage, &view, 0, &[5_000, 2], &selected);
        assert_eq!(
            (runs.run, runs.src_stride, runs.src_step, runs.dst_stride, runs.dst_step),
            (2, 1, 2, 1, 2)
        );
        assert!(runs.appends(), "a leading gathered axis appends");
        let mut reference = Array::<f32, _>::zeros(IxDyn(&[5_000, 2]));
        for (position, &index) in selected.iter().enumerate() {
            reference
                .index_axis_mut(ndarray::Axis(0), position)
                .assign(&view.index_axis(ndarray::Axis(0), index));
        }
        assert_eq!(out, reference.into_raw_vec_and_offset().0);

        // The rank-1 flat table: a rank-0 axis slice, i.e. ONE element per index.
        let shape = [50_000usize];
        let view = ArrayViewD::<f32>::from_shape(IxDyn(&shape), storage).unwrap();
        let selected: Vec<usize> = (0..20_000).map(|position| (position * 7919 + 3) % 50_000).collect();
        let (runs, out) = appended(storage, &view, 0, &[20_000], &selected);
        assert_eq!((runs.run, runs.src_stride, runs.dst_step), (1, 1, 1));
        assert!(runs.appends());
        let expected: Vec<f32> = selected.iter().map(|&index| values[index]).collect();
        assert_eq!(out, expected);
    }

    /// The axis-LAST family collapses to one STRIDED run per index whose
    /// destination interleaves; it is not appendable, so it goes through
    /// [`RunCopy::place_runs`] — still bit-identical to ndarray's `assign`.
    #[test]
    fn place_runs_covers_an_axis_last_gather() {
        let shape = [500usize, 50, 5];
        let values: Vec<f32> = (0..125_000).map(|value| value as f32).collect();
        let base = Array::from_shape_vec(IxDyn(&shape), values.clone()).unwrap();
        let storage = base.as_slice().unwrap();
        let view = ArrayViewD::<f32>::from_shape(IxDyn(&shape), storage).unwrap();
        let dest_shape = [500usize, 50, 4];
        let selected = [3usize, 0, 4, 1];

        let runs = plan_axis_runs(storage, &view, 2, &dest_shape, 2).expect("one run per index");
        assert_eq!(
            (runs.run, runs.src_stride, runs.src_step, runs.dst_stride, runs.dst_step),
            (25_000, 5, 1, 4, 1)
        );
        assert!(!runs.appends(), "the destination positions interleave");
        let mut out = vec![f32::NAN; 500 * 50 * 4];
        runs.place_runs(storage, selected.iter().map(|&index| Ok(index)), &mut out)
            .expect("the runs copy");
        let mut reference = Array::<f32, _>::zeros(IxDyn(&dest_shape));
        for (position, &index) in selected.iter().enumerate() {
            reference
                .index_axis_mut(ndarray::Axis(2), position)
                .assign(&view.index_axis(ndarray::Axis(2), index));
        }
        assert_eq!(out, reference.into_raw_vec_and_offset().0);
    }

    /// A gathered MIDDLE axis splits the range at that axis (the leading part's
    /// destination stride spans the index count), so the classifier declines and
    /// the caller keeps the general plan.
    #[test]
    fn plan_axis_runs_declines_a_middle_axis() {
        let shape = [500usize, 100, 2];
        let values: Vec<f32> = (0..100_000).map(|value| value as f32).collect();
        let base = Array::from_shape_vec(IxDyn(&shape), values).unwrap();
        let storage = base.as_slice().unwrap();
        let view = ArrayViewD::<f32>::from_shape(IxDyn(&shape), storage).unwrap();
        assert!(plan_axis_runs(storage, &view, 1, &[500, 4, 2], 1).is_none());

        // A zero-extent gathered axis has no run to copy at all.
        let empty: Vec<f32> = Vec::new();
        let view = ArrayViewD::<f32>::from_shape(IxDyn(&[0usize, 3]), empty.as_slice()).unwrap();
        assert!(plan_axis_runs(&empty, &view, 0, &[1, 3], 0).is_none());
    }

    /// Every arm moves BITS: NaN payloads, `-0.0` and infinities survive both
    /// the append and the place path unchanged.
    #[test]
    fn the_run_arms_copy_float_bits_unchanged() {
        let values = [
            f32::NAN,
            -0.0,
            f32::INFINITY,
            3.5,
            -0.0,
            f32::NEG_INFINITY,
            f32::NAN,
            2.0,
        ];
        // Append arm: a rank-1 table, one element per index.
        let view = ArrayViewD::<f32>::from_shape(IxDyn(&[8]), &values[..]).unwrap();
        let (_, out) = appended(&values, &view, 0, &[4], &[6, 1, 6, 2]);
        let expected: Vec<u32> = [6usize, 1, 6, 2]
            .iter()
            .map(|&index| values[index].to_bits())
            .collect();
        assert_eq!(out.iter().map(|value| value.to_bits()).collect::<Vec<u32>>(), expected);

        // Place arm: an axis-last gather of the same operand.
        let shape = [2usize, 4];
        let base = Array::from_shape_vec(IxDyn(&shape), values.to_vec()).unwrap();
        let storage = base.as_slice().unwrap();
        let view = ArrayViewD::<f32>::from_shape(IxDyn(&shape), storage).unwrap();
        let runs = plan_axis_runs(storage, &view, 1, &[2, 3], 1).expect("one run per index");
        assert!(!runs.appends());
        let mut out = vec![0f32; 6];
        runs.place_runs(storage, [3usize, 1, 3].iter().map(|&index| Ok(index)), &mut out)
            .expect("the runs copy");
        let mut reference = Array::<f32, _>::zeros(IxDyn(&[2, 3]));
        for (position, &index) in [3usize, 1, 3].iter().enumerate() {
            reference
                .index_axis_mut(ndarray::Axis(1), position)
                .assign(&view.index_axis(ndarray::Axis(1), index));
        }
        let expected = reference.into_raw_vec_and_offset().0;
        assert_eq!(
            out.iter().map(|value| value.to_bits()).collect::<Vec<u32>>(),
            expected.iter().map(|value| value.to_bits()).collect::<Vec<u32>>(),
        );
    }

    /// An index the caller's resolver rejects stops the copy through the same
    /// error the resolving walk reports (the arms resolve inline, so the first
    /// bad index is the one that surfaces).
    #[test]
    fn the_run_arms_propagate_an_index_error() {
        let values: Vec<f32> = (0..8).map(|value| value as f32).collect();
        let view = ArrayViewD::<f32>::from_shape(IxDyn(&[8]), &values[..]).unwrap();
        let runs = plan_axis_runs(&values, &view, 0, &[3], 0).expect("one run per index");
        let mut out: Vec<f32> = Vec::new();
        let err = runs
            .append_runs(
                &values,
                [0usize, 99, 2].iter().map(|&index| match index {
                    0..=7 => Ok(index),
                    _ => Err(Error::shape(format!("gather: index {index} is out of bounds"))),
                }),
                &mut out,
            )
            .unwrap_err();
        assert!(err.message.contains("index 99 is out of bounds"), "{}", err.message);
        assert_eq!(out, vec![0.0], "the run before the bad index was appended");
    }
}
