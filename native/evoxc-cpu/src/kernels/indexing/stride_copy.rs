//! Collapsed-stride copies for the indexing kernels.
//!
//! A strided view (a `slice`, or any op whose output is an affine walk over
//! one operand) maps every output multi-index onto a source element through
//! one `(extent, stride)` pair per axis.  Two adjacent axes whose strides line
//! up — `outer.stride == inner.stride * inner.extent` — describe one
//! contiguous run of the source, so they collapse into a single axis.  Applied
//! innermost-first over every axis, that reduces the whole copy to
//!
//! * ONE contiguous run (`(500, 200)` over a `(500, 200)` operand, or a
//!   `(500, 4)` operand sliced to `(500, 1)` through a single axis of stride
//!   4),
//! * ONE strided run (the `(500, 50)` slice of a `(500, 50, 4)` operand: 25000
//!   elements, stride 4), or
//! * a handful of contiguous runs (a `(500, 50, 3)` slice of that operand:
//!   2500 runs of 3),
//!
//! instead of one odometer step per output ELEMENT.  That matters because the
//! per-element form costs ~4 ns/element on the shapes a downstream workload
//! actually slices, while a collapsed run costs 0.2–0.4 ns/element.
//!
//! Every offset here is a non-negative `usize` walking strictly forward, so a
//! plan can only be built for a non-negative step per axis; a backwards walk
//! stays on the caller's general (signed) path.
use evoxc_core::Result;

use crate::buffer::{block_storage_filled, take_pooled_vec, Storage};
use crate::scalar::Scalar;

use super::block_error;

/// One axis of a collapsed copy: `extent` elements spaced `stride` apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CopyAxis {
    /// Number of elements along the axis (never 0 in a built plan).
    pub extent: usize,
    /// Distance between consecutive elements along the axis (never 0).
    pub stride: usize,
}

/// A resolved, non-negative strided copy: `base` plus INNERMOST-FIRST axes.
///
/// An empty axis list means every output element is `values[base]` (which only
/// a one-element result can be).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CopyPlan {
    pub base: usize,
    pub axes: Vec<CopyAxis>,
}

/// How many output elements one iteration of the strided-run loop emits.
///
/// A strided run reads one element out of every `stride`-element window, so the
/// loop's cost is set by how well the compiler can fold a group of those reads
/// into wide loads: four lanes is the measured sweet spot on this host (a
/// 4-element group of a stride-4 f32 run is exactly one 16-byte load, and
/// larger groups only add register pressure).
const LANES: usize = 4;

impl CopyPlan {
    /// Build a plan from innermost-first axes: axes of extent 1 contribute no
    /// offset, so they are dropped, and adjacent axes whose strides line up
    /// are collapsed (which is what turns a per-element walk into a run).
    pub fn new(base: usize, mut axes: Vec<CopyAxis>) -> CopyPlan {
        axes.retain(|axis| axis.extent > 1);
        let mut collapsed: Vec<CopyAxis> = Vec::with_capacity(axes.len());
        for axis in axes {
            match collapsed.last_mut() {
                // `last` is the axis closer to the innermost: the two describe
                // one run when the outer stride steps over the whole inner one.
                Some(inner) if axis.stride == inner.stride * inner.extent => {
                    inner.extent *= axis.extent;
                }
                _ => collapsed.push(axis),
            }
        }
        CopyPlan {
            base,
            axes: collapsed,
        }
    }

    /// Number of elements the plan describes.
    pub fn total(&self) -> usize {
        self.axes.iter().map(|axis| axis.extent).product()
    }

    /// Materialize the plan over `values` into a fresh (C-contiguous) storage
    /// of `total` elements.
    ///
    /// Both shapes of the plan take a POOLED destination of exactly `total`
    /// elements and overwrite every element of it, so a temporary copy pays one
    /// live allocation instead of a page-faulted (and, for the zero-filled
    /// helper, doubly written) `malloc` per call:
    ///
    /// * contiguous innermost runs go through [`block_storage_filled`] — one
    ///   `copy_from_slice` per run, which for a whole-array slice or an
    ///   identity `reshape`/`transpose` is ONE bulk memcpy;
    /// * strided runs go through [`fill_strided`], which reads a group of
    ///   [`LANES`] elements out of one contiguous source window per iteration
    ///   instead of walking the source with a `step_by` iterator (measured
    ///   0.55 ns/element for the latter, ~0.2 ns/element here).
    pub fn copy<T: Scalar>(&self, values: &[T], total: usize, op: &str) -> Result<Storage> {
        if total == 0 {
            return Ok(T::into_storage(Vec::new()));
        }
        if self.total() != total {
            return Err(block_error(op));
        }
        if self.axes.is_empty() {
            if total != 1 {
                return Err(block_error(op));
            }
            let value = *values.get(self.base).ok_or_else(|| block_error(op))?;
            return Ok(T::into_storage(vec![value]));
        }
        let inner = self.axes[0];
        let outer = &self.axes[1..];
        // The plan's extents multiply to `total`, so `inner.extent` divides it
        // and every run of the result is exactly `inner.extent` elements long.
        let blocks = total / inner.extent;
        if outer.is_empty() {
            // ONE run: no offset list at all (which is the identity/full-slice
            // and whole-array `reshape`/`transpose` case).
            if inner.stride == 1 {
                return block_storage_filled(values, &[total], &[self.base], inner.extent, op);
            }
            let mut out = take_pooled_vec::<T>(total).unwrap_or_else(|| vec![T::zero(); total]);
            fill_strided(&mut out, values, self.base, inner.stride, op)?;
            return Ok(T::into_storage(out));
        }
        let offsets = block_offsets(outer, blocks, self.base);
        if inner.stride == 1 {
            return block_storage_filled(values, &[total], &offsets, inner.extent, op);
        }
        let mut out = take_pooled_vec::<T>(total).unwrap_or_else(|| vec![T::zero(); total]);
        for (run, offset) in out.chunks_exact_mut(inner.extent).zip(&offsets) {
            fill_strided(run, values, *offset, inner.stride, op)?;
        }
        Ok(T::into_storage(out))
    }
}

/// The start offset of every contiguous/strided run, in C order: the innermost
/// outer axis varies fastest and each step adds that axis' stride, so the walk
/// is a running sum rather than a per-element multiplication.
///
/// One outer axis is the overwhelmingly common case once the plan is collapsed
/// (a run shape like `(500, 50, 3)` out of `(500, 50, 4)`), and there the whole
/// list is one arithmetic progression — 0.12 ns per entry instead of the
/// odometer's 1.3, which matters when the result is tens of thousands of short
/// runs.
fn block_offsets(outer: &[CopyAxis], blocks: usize, base: usize) -> Vec<usize> {
    if outer.is_empty() {
        return vec![base];
    }
    if outer.len() == 1 {
        let stride = outer[0].stride;
        return (0..blocks).map(|block| base + block * stride).collect();
    }
    let mut offsets = Vec::with_capacity(blocks);
    let mut index = vec![0usize; outer.len()];
    let mut offset = base;
    for _ in 0..blocks {
        offsets.push(offset);
        for (axis, entry) in outer.iter().enumerate() {
            entry_index_step(&mut index[axis], &mut offset, entry);
            if index[axis] != 0 {
                break;
            }
        }
    }
    offsets
}

/// Advance one axis of the odometer, carrying into `offset` as it goes: a
/// single step adds the axis stride, a wrap subtracts the whole span (which
/// undoes the step just added too).
fn entry_index_step(index: &mut usize, offset: &mut usize, axis: &CopyAxis) {
    *index += 1;
    *offset += axis.stride;
    if *index == axis.extent {
        *index = 0;
        *offset -= axis.stride * axis.extent;
    }
}

/// Write one strided run into `run`: element `j` of `run` is
/// `values[base + j * stride]`.
///
/// The source window is clamped to exactly the elements the run touches, and
/// the copy emits [`LANES`] elements per iteration out of one
/// `stride * LANES`-element window, so the compiler sees a contiguous (wider)
/// load per lane group instead of a single strided read: measured ~0.2 ns per
/// output element for the `(500, 50, 4) -> (500, 50, 1)` family against
/// ~0.55 ns for the `step_by` form this replaced.
fn fill_strided<T: Copy>(
    run: &mut [T],
    values: &[T],
    base: usize,
    stride: usize,
    op: &str,
) -> Result<()> {
    if run.is_empty() {
        return Ok(());
    }
    let span = stride.checked_mul(run.len() - 1).ok_or_else(|| block_error(op))?;
    let last = base.checked_add(span).ok_or_else(|| block_error(op))?;
    // Validate the whole run up front (a plan that reads past the operand is an
    // explicit error, never a truncated result).
    values.get(base..=last).ok_or_else(|| block_error(op))?;
    let group = stride * LANES;
    let mut done = 0usize;
    // A full group reads one `stride * LANES`-element window, so the compiler
    // sees a contiguous group of lanes per iteration. The FINAL group of a run
    // usually has no such window left (the window ends exactly on the last
    // element the run touches), so it falls through to the single-element tail.
    while done + LANES <= run.len() {
        let start = base + done * stride;
        let Some(end) = start.checked_add(group) else { break };
        let Some(window) = values.get(start..end) else { break };
        let slots = &mut run[done..done + LANES];
        for (lane, slot) in slots.iter_mut().enumerate() {
            *slot = window[lane * stride];
        }
        done += LANES;
    }
    while done < run.len() {
        let offset = base + done * stride;
        run[done] = *values.get(offset).ok_or_else(|| block_error(op))?;
        done += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axis(extent: usize, stride: usize) -> CopyAxis {
        CopyAxis { extent, stride }
    }

    #[test]
    fn unit_axes_drop_and_aligned_axes_collapse() {
        // A (500, 50) slice of a (500, 50, 4) operand: innermost-first
        // (50, 4) then (500, 200) — 200 == 4 * 50, so it is ONE strided run.
        let plan = CopyPlan::new(3, vec![axis(1, 1), axis(50, 4), axis(500, 200)]);
        assert_eq!(plan.axes, vec![axis(25000, 4)]);
        assert_eq!(plan.total(), 25000);
        // A (500, 16) slice of a (500, 50): the innermost run stays contiguous.
        let plan = CopyPlan::new(0, vec![axis(16, 1), axis(500, 50)]);
        assert_eq!(plan.axes, vec![axis(16, 1), axis(500, 50)]);
        // A fully contiguous copy is one axis of stride 1.
        let plan = CopyPlan::new(7, vec![axis(6, 1), axis(4, 6)]);
        assert_eq!(plan.axes, vec![axis(24, 1)]);
    }

    #[test]
    fn a_strided_run_matches_the_elementwise_reference() {
        let values: Vec<f32> = (0..100_000).map(|index| index as f32).collect();
        let plan = CopyPlan::new(3, vec![axis(1, 1), axis(50, 4), axis(500, 200)]);
        let storage = plan.copy(&values, 25000, "slice").unwrap();
        match &storage {
            Storage::F32(out) => {
                assert_eq!(out.len(), 25000);
                for (index, value) in out.iter().enumerate() {
                    assert_eq!(*value, values[3 + index * 4], "element {index}");
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }

    #[test]
    fn a_three_of_four_copy_matches_the_elementwise_reference() {
        let values: Vec<i32> = (0..100_000).collect();
        // (500, 50, 3) out of (500, 50, 4): 25000 runs of 3, stride 4 apart.
        let plan = CopyPlan::new(1, vec![axis(3, 1), axis(50, 4), axis(500, 200)]);
        assert_eq!(plan.axes, vec![axis(3, 1), axis(25000, 4)]);
        let storage = plan.copy(&values, 75_000, "slice").unwrap();
        match &storage {
            Storage::I32(out) => {
                assert_eq!(out.len(), 75_000);
                for block in 0..25000 {
                    for slot in 0..3 {
                        assert_eq!(out[block * 3 + slot], (1 + block * 4 + slot) as i32);
                    }
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }

    #[test]
    fn a_contiguous_plan_is_one_run_and_the_out_of_bounds_plan_errors() {
        let values: Vec<f32> = (0..12).map(|index| index as f32).collect();
        let plan = CopyPlan::new(4, vec![axis(4, 1), axis(2, 4)]);
        assert_eq!(plan.copy(&values, 8, "slice").unwrap(), Storage::F32(values[4..12].to_vec()));
        // A plan that walks past the operand is an explicit error, never a
        // truncated or wrapped result.
        let plan = CopyPlan::new(4, vec![axis(16, 1)]);
        assert!(plan.copy(&values, 16, "slice").is_err());
        let plan = CopyPlan::new(4, vec![axis(4, 4)]);
        assert!(plan.copy(&values, 4, "slice").is_err());
        // An empty result is empty, and a scalar plan repeats its one element.
        let plan = CopyPlan::new(0, vec![axis(4, 1)]);
        assert_eq!(plan.copy::<f32>(&values, 0, "slice").unwrap(), Storage::F32(Vec::new()));
        let plan = CopyPlan::new(5, Vec::new());
        assert_eq!(plan.copy(&values, 1, "slice").unwrap(), Storage::F32(vec![5.]));
        assert!(plan.copy(&values, 2, "slice").is_err());
    }

    /// The lane-grouped strided run (`LANES` outputs per source window) must
    /// read exactly what a per-element stride walk reads, for every combination
    /// of a run extent that is and is not a multiple of `LANES` and a stride
    /// that does and does not fit whole lane groups into the window.
    #[test]
    fn lane_grouped_runs_match_the_elementwise_reference() {
        let values: Vec<i32> = (0..1000).collect();
        for stride in [2usize, 3, 4, 7, 16] {
            for extent in [1usize, 2, 3, 4, 5, 7, 8, 9, 16, 17, 63, 64, 65] {
                let base = 3usize;
                let span = stride * (extent - 1);
                if base + span >= values.len() {
                    continue;
                }
                let mut expected = Vec::with_capacity(extent);
                for index in 0..extent {
                    expected.push(values[base + index * stride]);
                }
                let plan = CopyPlan::new(base, vec![axis(extent, stride)]);
                assert_eq!(
                    plan.copy(&values, extent, "slice").unwrap(),
                    Storage::I32(expected),
                    "stride {stride}, extent {extent}"
                );
            }
        }
        // bool takes the same path (the storage is one byte per element).
        let flags: Vec<bool> = (0..400).map(|index| index % 3 == 0).collect();
        let plan = CopyPlan::new(1, vec![axis(100, 3)]);
        let storage = plan.copy(&flags, 100, "slice").unwrap();
        match &storage {
            Storage::Bool(out) => {
                assert_eq!(out.len(), 100);
                for (index, value) in out.iter().enumerate() {
                    assert_eq!(*value, flags[1 + index * 3], "element {index}");
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }

    /// A recycled stock of exactly the result's size is fully overwritten, for
    /// both the strided and the contiguous run.
    #[test]
    fn a_pooled_strided_result_leaves_no_stale_element() {
        const COUNT: usize = 100_000;
        let values: Vec<f32> = (0..4 * COUNT).map(|index| index as f32 * 0.5).collect();
        // Park a garbage stock of exactly this result's size in the pool (a
        // dropped large storage is handed to the thread-local pool).
        drop(Storage::F32(vec![f32::NAN; COUNT]));
        // 100_000 elements, stride 4 — one strided run out of a 400k window.
        let plan = CopyPlan::new(2, vec![axis(COUNT, 4)]);
        let storage = plan.copy(&values, COUNT, "slice").unwrap();
        match &storage {
            Storage::F32(out) => {
                assert_eq!(out.len(), COUNT);
                for (index, value) in out.iter().enumerate() {
                    assert_eq!(*value, values[2 + index * 4], "element {index}");
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
        drop(Storage::F32(vec![f32::NAN; COUNT]));
        // ... and one contiguous run, which is the pooled bulk memcpy.
        let plan = CopyPlan::new(7, vec![axis(COUNT, 1)]);
        let storage = plan.copy(&values, COUNT, "slice").unwrap();
        match &storage {
            Storage::F32(out) => {
                assert_eq!(out.len(), COUNT);
                for (index, value) in out.iter().enumerate() {
                    assert_eq!(*value, values[7 + index], "element {index}");
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }
}
