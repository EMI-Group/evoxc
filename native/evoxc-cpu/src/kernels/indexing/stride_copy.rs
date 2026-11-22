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
use evoxc_core::{Error, Result};

use crate::buffer::{block_storage_filled, Storage};
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
    /// A contiguous innermost run goes through [`block_storage_filled`], so the
    /// result takes a pooled allocation and needs no zero fill; a strided one
    /// reads with a hoisted window and an exact-count step, so the inner loop
    /// carries no bounds check and no odometer.
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
        let blocks = total.checked_div(inner.extent).ok_or_else(|| block_error(op))?;
        let offsets = block_offsets(outer, blocks, self.base);
        if inner.stride == 1 {
            return block_storage_filled(values, &[total], &offsets, inner.extent, op);
        }
        let mut out: Vec<T> = Vec::with_capacity(total);
        for offset in offsets {
            push_strided(&mut out, values, offset, inner.extent, inner.stride, op)?;
        }
        if out.len() != total {
            return Err(block_error(op));
        }
        Ok(T::into_storage(out))
    }
}

/// The start offset of every contiguous/strided run, in C order: the innermost
/// outer axis varies fastest and each step adds that axis' stride, so the walk
/// is a running sum rather than a per-element multiplication.
fn block_offsets(outer: &[CopyAxis], blocks: usize, base: usize) -> Vec<usize> {
    if outer.is_empty() {
        return vec![base];
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

/// Append `run` elements of `values` starting at `start`, `stride` apart.
///
/// The window is clamped to exactly the elements the run touches, so the read
/// of an in-bounds run has no slice bound check left in the loop body.
fn push_strided<T: Copy>(
    out: &mut Vec<T>,
    values: &[T],
    start: usize,
    run: usize,
    stride: usize,
    op: &str,
) -> Result<()> {
    let last = start.saturating_add(stride.saturating_mul(run - 1));
    let window = values
        .get(start..=last)
        .ok_or_else(|| block_error(op))?;
    out.extend(window.iter().step_by(stride).take(run));
    Ok(())
}

/// Read one element through a plan (used by the empty-plan case only).
#[allow(dead_code)]
fn one<T: Copy>(values: &[T], offset: usize, op: &str) -> Result<T> {
    values.get(offset).copied().ok_or_else(|| {
        let _ = op;
        Error::verification("strided copy: offset outside the operand")
    })
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
}
