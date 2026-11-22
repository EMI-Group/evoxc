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
//! from an [`ArrayViewD`]: its shape, its element strides and its base offset
//! inside the owning slice (through `as_ptr`, so no layout fact is re-derived by
//! hand). Those are collapsed innermost-first into "outer axis list, innermost
//! contiguous run", and each run is emitted [`LANES`] elements per iteration out
//! of one `stride * LANES`-element source window. Anything the plan cannot
//! describe — a negative stride (a backwards `slice`), a shape/offset it cannot
//! prove — returns `false` with the destination untouched, and the caller uses
//! ndarray's `assign` instead. So the specialization can only ever change speed,
//! never a value; the kernel tests pin it against `assign` itself.
use ndarray::ArrayViewD;

use crate::scalar::Scalar;

/// Elements written per iteration by the strided run filler (one source window
/// of `LANES * stride` elements).
const LANES: usize = 4;

/// One axis of the copy plan, innermost first: `extent` elements `stride` apart.
type Axis = (usize, usize);

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
    let Some(base) = base_offset(view, source) else {
        return false;
    };
    let Some(axes) = plan_axes(view) else {
        return false;
    };
    if axes.is_empty() {
        // Every extent was 1: the view is a single element.
        return match source.get(base) {
            Some(value) if dest.len() == 1 => {
                dest[0] = *value;
                true
            }
            _ => false,
        };
    }
    match axes.as_slice() {
        // A wholly contiguous view: one bulk memcpy.
        [(extent, 1)] if *extent == dest.len() => {
            match source.get(base..base + dest.len()) {
                Some(run) => {
                    dest.copy_from_slice(run);
                    true
                }
                None => false,
            }
        }
        _ => fill_runs(source, base, dest, &axes),
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

/// Collapse the view's axes innermost-first, dropping extent-1 axes and folding
/// every axis whose stride continues the inner one (`stride_outer ==
/// stride_inner * extent_inner`).
///
/// `None` for a negative stride — the collapsed plan only describes a forward
/// walk, and a backwards `slice` keeps ndarray's `assign`.
fn plan_axes<T>(view: &ArrayViewD<'_, T>) -> Option<Vec<Axis>> {
    let mut axes: Vec<Axis> = Vec::with_capacity(view.ndim());
    for index in (0..view.ndim()).rev() {
        let extent = view.shape()[index];
        let stride = view.strides()[index];
        if extent == 0 {
            return None;
        }
        if extent == 1 {
            continue;
        }
        let stride = usize::try_from(stride).ok()?;
        match axes.last_mut() {
            Some(inner) if inner.1 * inner.0 == stride => {
                inner.0 = inner.0.checked_mul(extent)?;
            }
            _ => axes.push((extent, stride)),
        }
    }
    Some(axes)
}

/// Walk the outer axes with one running offset and fill the innermost run of
/// `outer.0` elements `outer.1` apart per iteration.
fn fill_runs<T: Scalar>(source: &[T], base: usize, dest: &mut [T], axes: &[Axis]) -> bool {
    let (run_extent, run_stride) = axes[0];
    let mut out = 0usize;
    let mut offset = base;
    // The outer axes: a plain odometer over `axes[1..]`, outermost last.
    let mut odometer = vec![0usize; axes.len() - 1];
    loop {
        let Some(chunk) = dest.get_mut(out..out + run_extent) else {
            return false;
        };
        if !fill_strided(source, offset, chunk, run_stride) {
            return false;
        }
        out += run_extent;
        // Advance the odometer: the innermost OUTER axis varies fastest, so a
        // carry walks outward one axis at a time.
        let mut axis = 0usize;
        loop {
            if axis == odometer.len() {
                return out == dest.len();
            }
            odometer[axis] += 1;
            offset += axes[axis + 1].1;
            if odometer[axis] < axes[axis + 1].0 {
                break;
            }
            offset -= axes[axis + 1].1 * axes[axis + 1].0;
            odometer[axis] = 0;
            axis += 1;
        }
    }
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
        assert_eq!(plan_axes(&view), Some(vec![(25_000, 4)]));
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
}
