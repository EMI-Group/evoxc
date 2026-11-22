//! `argmax` / `argmin`: one index per output, numpy's first-occurrence tie-break
//! and its first-NaN rule, with the length-dispatched branchy/branchless scans.
use evoxc_core::{DType, Error, Result};
use ndarray::{Array1, ArrayViewD, Axis, Zip};
use num_complex::Complex;

use crate::buffer::{array_view, Buffer, NdScalar, Storage};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::plan::OpSpec;
use crate::scalar::Scalar;
use crate::with_type;

use super::expect_rank;
use super::layout::{for_each_lane, kept_extent, reduced_extent, Block};

// ---------------------------------------------------------------------------
// argmax / argmin
// ---------------------------------------------------------------------------

pub(super) fn arg_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let out_shape = util::result_shape(ctx, spec, 0)?;
    arg_reduce(spec, &operands[0], &out_shape)
}

pub(super) fn arg_reduce(spec: &OpSpec, input: &Buffer, out_shape: &[usize]) -> Result<Vec<Value>> {
    let name = spec.name.as_str();
    let want_max = match name {
        "argmax" => true,
        "argmin" => false,
        other => {
            return Err(Error::verification(format!(
                "op '{other}' is not an arg reduction"
            )))
        }
    };
    util::reject_complex(name, input.dtype)?;
    let keepdims = spec.attrs.bool("keepdims")?;
    let reduced = match spec.attrs.opt_i64("axis")? {
        Some(axis) => {
            let axis = util::normalize_axis(axis, input.rank(), &format!("op '{name}'"))?;
            let mut mask = vec![false; input.rank()];
            mask[axis] = true;
            mask
        }
        // numpy: `axis=None` flattens the input and returns one index.
        None => vec![true; input.rank()],
    };
    let kept = reduced.iter().filter(|flag| !**flag).count();
    expect_rank(name, out_shape, kept, input.rank(), keepdims)?;
    let out_dtype = spec.result_types[0].dtype;
    with_type!(input, |values| {
        compute_arg(want_max, name, values, &input.shape, &reduced, out_shape, out_dtype)
    })
}

fn compute_arg<T: Scalar + ArgOrd + NdScalar>(
    want_max: bool,
    name: &str,
    values: &[T],
    in_shape: &[usize],
    reduced: &[bool],
    out_shape: &[usize],
    out_dtype: DType,
) -> Result<Vec<Value>> {
    if reduced_extent(in_shape, reduced) == 0 {
        return Err(Error::shape(format!(
            "op '{name}': zero-size arg reduction — a reduced axis has extent 0, so there is no \
             index to return (numpy raises the same way)"
        )));
    }
    let indices = match Block::new(in_shape, reduced) {
        // The contiguous-block specialization: a leading / middle-axis reduction
        // folds its `inner` output lanes row by row, the winner's row position
        // being its reported index.
        Some(block) if block.red > 0 && block.inner > 1 => {
            arg_group_block(values, in_shape, &block, want_max)?
        }
        // The general lane walk: each output lane is an arg scan over its reduced
        // positions in C order, so the scanned position IS the reported index.
        _ => {
            let mut indices: Vec<i64> = Vec::with_capacity(kept_extent(in_shape, reduced));
            for_each_lane(values, in_shape, reduced, |view| {
                indices.push(arg_lane(&view, want_max));
                Ok(())
            })?;
            indices
        }
    };
    let buffer = Buffer::new(DType::Int64, out_shape.to_vec(), Storage::I64(indices))?;
    if out_dtype == DType::Int64 {
        single(buffer)
    } else {
        single(buffer.cast(out_dtype)?)
    }
}

/// One output lane: the contiguous fast path is [`arg_run`] over the lane's
/// slice, otherwise the same sequential rule over the strided iterator.
fn arg_lane<T: ArgOrd + NdScalar>(view: &ArrayViewD<'_, T>, want_max: bool) -> i64 {
    match view.as_slice() {
        Some(slice) => arg_run(slice, want_max) as i64,
        None => arg_iter(view.iter().copied(), want_max),
    }
}

/// The branchy rule over an arbitrary iterator — the same left-to-right scan as
/// [`arg_run_branchy`], so the strided lane reports the identical index.
fn arg_iter<T: ArgOrd, I: IntoIterator<Item = T>>(iter: I, want_max: bool) -> i64 {
    let mut iter = iter.into_iter();
    let mut best = match iter.next() {
        Some(value) => value,
        // The caller checks the reduced extent is non-zero, so every lane is
        // non-empty; this branch never runs.
        None => return 0,
    };
    let mut best_index = 0i64;
    for (position, candidate) in iter.enumerate() {
        if arg_better(best, candidate, want_max) {
            best = candidate;
            best_index = (position + 1) as i64;
        }
    }
    best_index
}

/// One contiguous reduced block reducing into `inner` output lanes: each lane
/// walks its positions in operand order, so the tie and NaN rules are exactly
/// the sequential ones.  Output order is C order (outer, then inner).
fn arg_group_block<T: ArgOrd + NdScalar>(
    values: &[T],
    shape: &[usize],
    block: &Block,
    want_max: bool,
) -> Result<Vec<i64>> {
    let view = array_view(values, shape)?;
    let groups = view
        .to_shape((block.outer, block.red, block.inner))
        .map_err(|error| {
            Error::shape(format!(
                "contiguous arg block {}x{}x{} does not reshape the operand: {error}",
                block.outer, block.red, block.inner
            ))
        })?;
    let mut out = Vec::with_capacity(block.out_len());
    for group in groups.axis_iter(Axis(0)) {
        let mut best = group.index_axis(Axis(0), 0).to_owned();
        let mut indices = Array1::from_elem(block.inner, 0i64);
        for (position, row) in group.axis_iter(Axis(0)).enumerate().skip(1) {
            Zip::from(&mut best)
                .and(&mut indices)
                .and(&row)
                .for_each(|slot, index, &value| {
                    if arg_better(*slot, value, want_max) {
                        *slot = value;
                        *index = position as i64;
                    }
                });
        }
        out.extend(indices.iter().copied());
    }
    Ok(out)
}

/// The typed comparison an arg reduction scans with.
///
/// `Num` offers `max`/`min` but no ordering, and the dtype-erased
/// [`ScalarValue`] round trip this used to go through per element was the whole
/// cost of `argmax`/`argmin`; ordering on the concrete scalar keeps the scan one
/// compare per element.
trait ArgOrd: Copy {
    fn arg_nan(self) -> bool;
    fn arg_gt(self, other: Self) -> bool;
}

macro_rules! arg_ord_ordered {
    ($($ty:ty),*) => {$(
        impl ArgOrd for $ty {
            #[inline]
            fn arg_nan(self) -> bool {
                false
            }
            #[inline]
            fn arg_gt(self, other: Self) -> bool {
                self > other
            }
        }
    )*};
}

macro_rules! arg_ord_float {
    ($($ty:ty),*) => {$(
        impl ArgOrd for $ty {
            #[inline]
            fn arg_nan(self) -> bool {
                <$ty>::is_nan(self)
            }
            #[inline]
            fn arg_gt(self, other: Self) -> bool {
                self > other
            }
        }
    )*};
}

arg_ord_ordered!(bool, i8, i16, i32, i64, u8, u16, u32, u64);
arg_ord_float!(half::f16, f32, f64);

/// Complex has no order in the scalar vocabulary (an arg reduction over complex
/// is rejected before a scan runs), so it only needs a total fallback: the real
/// projection, matching what the dtype-erased comparison did.
macro_rules! arg_ord_complex {
    ($($ty:ty),*) => {$(
        impl ArgOrd for $ty {
            #[inline]
            fn arg_nan(self) -> bool {
                self.re.is_nan() || self.im.is_nan()
            }
            #[inline]
            fn arg_gt(self, other: Self) -> bool {
                self.re > other.re
            }
        }
    )*};
}

arg_ord_complex!(Complex<f32>, Complex<f64>);

/// numpy's arg rule over one candidate: a NaN best is never replaced (so the
/// FIRST NaN wins), a NaN candidate always wins, and otherwise only a strict
/// improvement replaces (so a tie keeps the FIRST occurrence).
#[inline]
fn arg_better<T: ArgOrd>(best: T, candidate: T, want_max: bool) -> bool {
    if best.arg_nan() {
        return false;
    }
    if candidate.arg_nan() {
        return true;
    }
    if want_max {
        candidate.arg_gt(best)
    } else {
        best.arg_gt(candidate)
    }
}

/// Run length from which [`arg_run`] uses the branchy scan.
///
/// Which scan wins is length-dependent, measured on this host (262144 f32
/// elements, `argmax`, warm): a short run is better served by the branch-free
/// scan (its bookkeeping is cheaper than the mispredicts a per-row scan pays),
/// while on a long single run the improving branch is almost never taken once
/// the running extreme settles, so it predicts perfectly and the branchy scan
/// skips the bookkeeping instead.  Both scans return identical indices — they
/// are two spellings of the same rule, and a test pins them against each other.
const ARG_BRANCHY_MIN_RUN: usize = 1024;

/// Index of the winning position in one contiguous run (one output).
#[inline]
fn arg_run<T: ArgOrd>(values: &[T], want_max: bool) -> usize {
    if values.len() >= ARG_BRANCHY_MIN_RUN {
        arg_run_branchy(values, want_max)
    } else {
        arg_run_branchless(values, want_max)
    }
}

/// The plain sequential scan: numpy's rule spelled as branches (a NaN best is
/// never replaced — so it keeps the FIRST NaN — a NaN candidate always wins, and
/// otherwise only a strict improvement replaces, so a tie keeps the FIRST
/// occurrence).
#[inline]
fn arg_run_branchy<T: ArgOrd>(values: &[T], want_max: bool) -> usize {
    let mut best = values[0];
    let mut best_index = 0usize;
    for (position, candidate) in values.iter().enumerate().skip(1) {
        if arg_better(best, *candidate, want_max) {
            best = *candidate;
            best_index = position;
        }
    }
    best_index
}

/// The same rule with no data-dependent branch: the value scan ignores NaN (a
/// NaN comparison is false either way, which is exactly "never improves") and the
/// first NaN is tracked beside it, so the answer is the first NaN's position when
/// the run holds one and otherwise the leftmost strict extreme.  For the
/// integer/bool dtypes `arg_nan` is a constant `false`, so that bookkeeping
/// compiles away.
#[inline]
fn arg_run_branchless<T: ArgOrd>(values: &[T], want_max: bool) -> usize {
    let mut best = values[0];
    let mut best_index = 0usize;
    // `usize::MAX` = "no NaN seen yet".
    let mut nan_index = usize::MAX;
    for (position, candidate) in values.iter().enumerate().skip(1) {
        let first_nan = candidate.arg_nan() && nan_index == usize::MAX;
        nan_index = if first_nan { position } else { nan_index };
        let improves = if want_max {
            candidate.arg_gt(best)
        } else {
            best.arg_gt(*candidate)
        };
        best = if improves { *candidate } else { best };
        best_index = if improves { position } else { best_index };
    }
    if values[0].arg_nan() {
        0
    } else if nan_index != usize::MAX {
        nan_index
    } else {
        best_index
    }
}

#[cfg(test)]
mod tests {
    use super::super::layout::axis_permutation;
    use super::super::test_support::{arg_op, buffer};
    use super::*;
    use crate::buffer::Storage;

    /// The independent ndarray reference: permute kept-first, reshape to
    /// `(kept, red)`, zip each value with its C-order reduced index and fold each
    /// lane sequentially with `fold_axis` (the tie / first-NaN rule in the
    /// closure, spelled out as [`arg_better`]).
    fn reference_arg<T: ArgOrd + NdScalar>(
        values: &[T],
        shape: &[usize],
        reduced: &[bool],
        want_max: bool,
    ) -> Vec<i64> {
        let view = array_view::<T>(values, shape).expect("operand view");
        let permuted = view.permuted_axes(axis_permutation(shape.len(), reduced));
        let kept: usize = shape
            .iter()
            .zip(reduced)
            .filter(|(_, flag)| !**flag)
            .map(|(dim, _)| *dim)
            .product();
        let red: usize = shape
            .iter()
            .zip(reduced)
            .filter(|(_, flag)| **flag)
            .map(|(dim, _)| *dim)
            .product();
        let flat = permuted.to_shape((kept, red)).expect("reduced lanes");
        let indices = ndarray::Array2::from_shape_fn((kept, red), |(_, position)| position as i64);
        let pairs = Zip::from(flat.view())
            .and(indices.view())
            .map_collect(|&value, &index| (value, index));
        pairs
            .fold_axis(Axis(1), None::<(T, i64)>, |acc, item| match acc {
                None => Some(*item),
                Some((best, best_index)) => {
                    if arg_better(*best, item.0, want_max) {
                        Some(*item)
                    } else {
                        Some((*best, *best_index))
                    }
                }
            })
            .iter()
            .map(|slot| match slot {
                Some((_, index)) => *index,
                None => 0,
            })
            .collect()
    }


    #[test]
    fn argmax_and_argmin_break_ties_at_the_first_occurrence() {
        let input = buffer(
            DType::Float64,
            &[2, 3],
            Storage::F64(vec![1.0, 3.0, 3.0, 2.0, 0.0, 2.0]),
        );
        // Negative axis: the last axis.
        let op = arg_op("argmax", Some(-1), false, &[2]);
        let out = arg_reduce(&op, &input, &[2]).unwrap();
        assert_eq!(out[0].dtype, DType::Int64);
        assert_eq!(out[0].storage, Storage::I64(vec![1, 0]));
        // axis 0 with keepdims: declared shape (1, 3).
        let op = arg_op("argmin", Some(0), true, &[1, 3]);
        assert_eq!(arg_reduce(&op, &input, &[1, 3]).unwrap()[0].storage, Storage::I64(vec![0, 1, 1]));
    }

    #[test]
    fn argmax_flattens_with_axis_none_and_keepdims() {
        let input = buffer(DType::Int64, &[2, 2], Storage::I64(vec![1, 9, 3, 4]));
        let op = arg_op("argmax", None, false, &[]);
        assert_eq!(arg_reduce(&op, &input, &[]).unwrap()[0].storage, Storage::I64(vec![1]));
        // numpy has no keepdims for a flattened arg reduction; the IR declares
        // the all-1 shape.
        let op = arg_op("argmax", None, true, &[1, 1]);
        let out = arg_reduce(&op, &input, &[1, 1]).unwrap();
        assert_eq!(out[0].shape, vec![1, 1]);
        assert_eq!(out[0].storage, Storage::I64(vec![1]));
    }

    #[test]
    fn arg_reductions_pick_the_first_nan() {
        let input = buffer(DType::Float64, &[3], Storage::F64(vec![1.0, f64::NAN, 2.0]));
        for name in ["argmax", "argmin"] {
            let op = arg_op(name, Some(0), true, &[1]);
            assert_eq!(
                arg_reduce(&op, &input, &[1]).unwrap()[0].storage,
                Storage::I64(vec![1]),
                "op {name}"
            );
        }
    }

    #[test]
    fn arg_reductions_match_the_ndarray_reference() {
        let cases: Vec<(Vec<usize>, Option<i64>)> = vec![
            (vec![7], Some(0)),
            (vec![7], None),
            (vec![2, 3], Some(1)),
            (vec![2, 3], Some(0)),
            (vec![2, 3], None),
            (vec![3, 4, 5], Some(2)),
            (vec![3, 4, 5], Some(1)),
            (vec![3, 4, 5], Some(0)),
            (vec![5, 6, 3, 2], Some(1)),
            (vec![5, 6, 3, 2], Some(2)),
            (vec![5, 6, 3, 2], None),
        ];
        for (shape, axis) in cases {
            let count: usize = shape.iter().product();
            // Ties, a repeated maximum and a NaN at a fixed position.
            let values: Vec<f32> = (0..count)
                .map(|index| ((index * 13 % 7) as f32) + 0.25)
                .collect();
            let mut reduced = vec![false; shape.len()];
            match axis {
                Some(axis) => reduced[axis.rem_euclid(shape.len() as i64) as usize] = true,
                None => reduced = vec![true; shape.len()],
            }
            let kept = reduced.iter().filter(|flag| !**flag).count();
            let declared: Vec<usize> = shape
                .iter()
                .enumerate()
                .filter(|(index, _)| !reduced[*index])
                .map(|(_, dim)| *dim)
                .collect();
            assert_eq!(declared.len(), kept);
            let input = buffer(DType::Float32, &shape, Storage::F32(values.clone()));
            for (name, want_max) in [("argmax", true), ("argmin", false)] {
                let op = arg_op(name, axis, false, &declared);
                let got = arg_reduce(&op, &input, &declared).unwrap().remove(0);
                assert_eq!(
                    got.storage,
                    Storage::I64(reference_arg(&values, &shape, &reduced, want_max)),
                    "op {name} shape {shape:?} axis {axis:?}"
                );
            }
        }
    }

    #[test]
    fn arg_reductions_keep_the_first_nan_and_the_first_of_a_tie() {
        let values = vec![1.0f32, f32::NAN, 5.0, 5.0, f32::NAN, 0.0];
        let input = buffer(DType::Float32, &[2, 3], Storage::F32(values.clone()));
        // Row 0: the first NaN at position 1 wins for both directions.
        // Row 1: NaN at position 1 again, over a 5.0/0.0 tie elsewhere.
        for (name, keepdims) in [("argmax", false), ("argmin", false)] {
            let op = arg_op(name, Some(1), keepdims, &[2]);
            assert_eq!(
                arg_reduce(&op, &input, &[2]).unwrap()[0].storage,
                Storage::I64(vec![1, 1]),
                "op {name}"
            );
        }
        // A NaN later in the run replaces a numeric best; a tie keeps the first.
        let values = vec![3.0f32, 3.0, 1.0, f32::NAN];
        let input = buffer(DType::Float32, &[1, 4], Storage::F32(values));
        let op = arg_op("argmax", Some(1), false, &[1]);
        assert_eq!(arg_reduce(&op, &input, &[1]).unwrap()[0].storage, Storage::I64(vec![3]));
    }

    #[test]
    fn the_two_arg_scans_agree_on_short_and_long_runs() {
        // The length-dispatched scans are two spellings of numpy's rule; both
        // must equal the offset-table reference at every length, with a NaN
        // anywhere and with ties (the data repeats every 5 values).
        for len in [2usize, 5, 8, 64, 1023, 1024, 1500, 4096] {
            let mut cases: Vec<Option<usize>> = vec![None, Some(0), Some(1), Some(len / 2), Some(len - 1)];
            cases.dedup();
            for nan_at in cases {
                let mut values: Vec<f32> = (0..len).map(|index| ((index * 17 % 5) as f32) * 0.5).collect();
                if let Some(position) = nan_at {
                    values[position] = f32::NAN;
                }
                let reference_mask = vec![true];
                for want_max in [true, false] {
                    let want = reference_arg(&values, &[len], &reference_mask, want_max)[0];
                    for (label, got) in [
                        ("branchy", arg_run_branchy(&values, want_max)),
                        ("branchless", arg_run_branchless(&values, want_max)),
                        ("dispatched", arg_run(&values, want_max)),
                    ] {
                        assert_eq!(
                            got as i64, want,
                            "{label} scan: len {len} nan_at {nan_at:?} want_max {want_max}"
                        );
                    }
                }
            }
        }
        // Integer runs have no NaN at all; both scans must still agree.
        let values: Vec<i64> = (0..2000).map(|index| (index * 7 % 13) - 6).collect();
        for want_max in [true, false] {
            assert_eq!(
                arg_run_branchy(&values, want_max),
                arg_run_branchless(&values, want_max)
            );
        }
    }
}
