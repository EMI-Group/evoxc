//! The ndarray-driven fold and accumulator machinery every reduce kind shares.
//!
//! [`for_each_lane`] is the ONE layout walk: it lends the operand to
//! [`ndarray`], permutes the kept axes (ascending, in original order) before the
//! reduced axes (ascending), then recurses `axis_iter` down the leading kept
//! axes so every output lane is handed to the caller as a C-order view of
//! exactly its reduced positions.  [`fold_all`] / [`fold_first`] fold each lane
//! through [`fold_lane`], which keeps the 8-accumulator lane split on a
//! contiguous slice and reproduces the identical rotation element-by-element on
//! a strided lane.
//!
//! [`fold_contiguous_block`] is the one documented SPECIALIZATION: a contiguous
//! reduced block with `inner > 1` (a leading / middle-axis reduction) that the
//! general walk would visit as long strided lanes instead accumulates its
//! `inner` output lanes row by row.  Both paths implement the SAME semantics and
//! must stay bit-for-bit equivalent for integer, bool, float16 and complex folds;
//! only float32/float64 sums may move inside the documented parity window (the
//! per-lane 8-accumulator reassociation).
use evoxc_core::{DType, Error, Result};
use ndarray::{Array1, ArrayViewD, Axis, Zip};
use num_complex::Complex;

use crate::buffer::{array_view, shape_text, NdScalar};
use crate::kernels::util;
use crate::scalar::{Num, Scalar};

// ---------------------------------------------------------------------------
// Axis classification (kept-first permutation, extents)
// ---------------------------------------------------------------------------

/// The permutation that puts the kept axes first and the reduced axes last,
/// each group ascending in the original axis order.  `result_axis[i]` of a
/// `permuted_axes(perm)` view is the operand's axis `perm[i]`.
pub(super) fn axis_permutation(rank: usize, reduced: &[bool]) -> Vec<usize> {
    let kept = reduced
        .iter()
        .enumerate()
        .take(rank)
        .filter(|(_, is_reduced)| !**is_reduced)
        .map(|(axis, _)| axis);
    let red = reduced
        .iter()
        .enumerate()
        .take(rank)
        .filter(|(_, is_reduced)| **is_reduced)
        .map(|(axis, _)| axis);
    let mut perm = Vec::with_capacity(rank);
    perm.extend(kept);
    perm.extend(red);
    perm
}

fn extent(shape: &[usize], reduced: &[bool], flag: bool) -> usize {
    shape
        .iter()
        .zip(reduced)
        .filter(|(_, is_reduced)| **is_reduced == flag)
        .map(|(dim, _)| *dim)
        .product()
}

/// Product of the KEPT axes' lengths — the number of output lanes (1 for a
/// fully reduced operand).
pub(super) fn kept_extent(shape: &[usize], reduced: &[bool]) -> usize {
    extent(shape, reduced, false)
}

/// Product of the REDUCED axes' lengths — the number of positions folded into
/// each output lane (1 when nothing is reduced).
pub(super) fn reduced_extent(shape: &[usize], reduced: &[bool]) -> usize {
    extent(shape, reduced, true)
}

// ---------------------------------------------------------------------------
// Contiguous-block classification
// ---------------------------------------------------------------------------

/// A reduction whose reduced axes form ONE contiguous block in C order.
///
/// Such an operand is `outer` consecutive groups of `red * inner` elements, and
/// each group is `red` interleaved runs of `inner` consecutive lanes: the
/// elements folded into output lane `i` of a group are `group[k * inner + i]`
/// for `k in 0..red`.  Only the `inner > 1` case is worth specialising
/// ([`fold_contiguous_block`], [`fold_lane`]'s caller); `inner == 1` (a trailing
/// axis or full reduction) is a contiguous per-lane run the general walk already
/// folds through [`fold_run`].
pub(super) struct Block {
    /// Groups of `red * inner` elements, one group per `inner` outputs.
    pub(super) outer: usize,
    /// Elements folded into one output.
    pub(super) red: usize,
    /// Output lanes per group.
    pub(super) inner: usize,
}

impl Block {
    /// The block of `reduced`, or `None` when the reduced axes are NOT
    /// contiguous (a gap inside them) or when nothing is reduced at all.
    pub(super) fn new(shape: &[usize], reduced: &[bool]) -> Option<Block> {
        let first = reduced.iter().position(|flag| *flag)?;
        let last = reduced.iter().rposition(|flag| *flag)?;
        if !reduced[first..=last].iter().all(|flag| *flag) {
            return None;
        }
        Some(Block {
            outer: shape[..first].iter().product(),
            red: shape[first..=last].iter().product(),
            inner: shape[last + 1..].iter().product(),
        })
    }

    pub(super) fn out_len(&self) -> usize {
        self.outer * self.inner
    }
}

// ---------------------------------------------------------------------------
// The lane walk
// ---------------------------------------------------------------------------

/// Visit every output lane's reduced sub-view, in C output order.
///
/// The operand is viewed as `shape`, permuted so the kept axes lead (ascending)
/// and the reduced axes trail (ascending), and the leading `kept` axes are
/// walked recursively; each leaf view is one output lane, its reduced positions
/// in C order.  A fully reduced operand (rank 0, or `reduced` over every axis)
/// has a single leaf: the whole operand.
pub(super) fn for_each_lane<T, F>(
    values: &[T],
    shape: &[usize],
    reduced: &[bool],
    mut visit: F,
) -> Result<()>
where
    T: NdScalar,
    F: FnMut(ArrayViewD<'_, T>) -> Result<()>,
{
    let view = array_view(values, shape)?;
    let permuted = view.permuted_axes(axis_permutation(shape.len(), reduced));
    let kept = reduced.iter().filter(|flag| !**flag).count();
    walk_lanes(permuted, 0, kept, &mut visit)
}

fn walk_lanes<'a, T, F>(
    view: ArrayViewD<'a, T>,
    depth: usize,
    kept: usize,
    visit: &mut F,
) -> Result<()>
where
    T: NdScalar,
    F: FnMut(ArrayViewD<'_, T>) -> Result<()>,
{
    if depth == kept {
        return visit(view);
    }
    for sub in view.axis_iter(Axis(0)) {
        walk_lanes(sub, depth + 1, kept, visit)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The per-lane fold
// ---------------------------------------------------------------------------

/// Independent accumulator lanes per contiguous fold run.
///
/// A sequential float fold is ONE dependency chain (4-cycle add latency per
/// element), which is why a full `reduce_sum` used to run ~14x slower than
/// numpy's pairwise sum; [`LANE_COUNT`] accumulators make the chains
/// independent, which is what lets the loop reach the memory bound.
const LANE_COUNT: usize = 8;

/// Lanes one lane fold uses for an accumulator of `dtype`.
///
/// Integer and bool folds are exact under any association (wrapping adds and
/// multiplies are associative), so they take the multi-accumulator path too.
/// Float16 folds IN float16 and complex in complex, where the parity contract's
/// ulp window is tighter than a reassociated sum can honour, so those keep the
/// interpreter's exact sequential order.
fn fold_lanes(dtype: DType) -> usize {
    if matches!(dtype, DType::Float16 | DType::Complex64 | DType::Complex128) {
        1
    } else {
        LANE_COUNT
    }
}

/// Fold one contiguous run with `lanes` independent accumulators.
///
/// A run shorter than `lanes` folds sequentially from `init` in operand order,
/// so every small reduction keeps the interpreter's exact arithmetic.  Integer,
/// bool and complex adds/multiplies are associative (wrapping / exact), and
/// `max`/`min` are NaN-propagating in both directions and idempotent, so
/// reordering only moves float32/float64 sums inside their documented tolerance.
#[inline]
fn fold_run<T: Num, F: Fn(T, T) -> T + Copy>(values: &[T], init: T, op: F, lanes: usize) -> T {
    if lanes == LANE_COUNT {
        fold_run_lanes::<T, F, LANE_COUNT>(values, init, op)
    } else {
        fold_run_lanes::<T, F, 1>(values, init, op)
    }
}

#[inline]
fn fold_run_lanes<T: Num, F: Fn(T, T) -> T, const LANES: usize>(values: &[T], init: T, op: F) -> T {
    let mut acc = [init; LANES];
    let mut chunks = values.chunks_exact(LANES);
    for chunk in &mut chunks {
        for lane in 0..LANES {
            acc[lane] = op(acc[lane], chunk[lane]);
        }
    }
    let mut total = acc[0];
    for value in acc.iter().skip(1) {
        total = op(total, *value);
    }
    for value in chunks.remainder() {
        total = op(total, *value);
    }
    total
}

/// Fold an arbitrary iterator with `lanes` accumulators, arithmetically
/// IDENTICAL to [`fold_run_lanes`]: the first `full = len / lanes * lanes`
/// elements rotate through the accumulators by `index % lanes`, the accumulators
/// are then combined in lane order, and the remainder folds sequentially.
#[inline]
fn fold_iter<T: Num, I: Iterator<Item = T>, F: Fn(T, T) -> T + Copy>(
    iter: I,
    len: usize,
    init: T,
    op: F,
    lanes: usize,
) -> T {
    if lanes == LANE_COUNT {
        fold_iter_lanes::<T, I, F, LANE_COUNT>(iter, len, init, op)
    } else {
        fold_iter_lanes::<T, I, F, 1>(iter, len, init, op)
    }
}

#[inline]
fn fold_iter_lanes<T: Num, I: Iterator<Item = T>, F: Fn(T, T) -> T, const LANES: usize>(
    mut iter: I,
    len: usize,
    init: T,
    op: F,
) -> T {
    let full = (len / LANES) * LANES;
    let mut acc = [init; LANES];
    for (index, value) in iter.by_ref().take(full).enumerate() {
        acc[index % LANES] = op(acc[index % LANES], value);
    }
    let mut total = acc[0];
    for value in acc.iter().skip(1) {
        total = op(total, *value);
    }
    for value in iter {
        total = op(total, value);
    }
    total
}

/// Fold one lane's reduced sub-view from `init` with `lanes` accumulators.
fn fold_lane<T, F>(view: &ArrayViewD<'_, T>, init: T, op: F, lanes: usize) -> T
where
    T: Num + NdScalar,
    F: Fn(T, T) -> T + Copy,
{
    match view.as_slice() {
        Some(slice) => fold_run(slice, init, op, lanes),
        None => fold_iter(view.iter().copied(), view.len(), init, op, lanes),
    }
}

/// Fold one lane starting from its FIRST element (the shape of a max/min
/// reduction, which has no identity element).
fn fold_first_lane<T, F>(view: &ArrayViewD<'_, T>, op: F, lanes: usize) -> T
where
    T: Num + NdScalar,
    F: Fn(T, T) -> T + Copy,
{
    if let Some(slice) = view.as_slice() {
        let first = slice[0];
        return fold_run(&slice[1..], first, op, lanes);
    }
    let mut iter = view.iter().copied();
    let first = match iter.next() {
        Some(value) => value,
        // The caller checks the reduced extent is non-zero, so every lane is
        // non-empty; this branch never runs.
        None => return T::zero(),
    };
    fold_iter(iter, view.len().saturating_sub(1), first, op, lanes)
}

// ---------------------------------------------------------------------------
// The per-op folds (general lane walk + the contiguous-block specialization)
// ---------------------------------------------------------------------------

/// Fold every output lane into its slot, starting from `init` (the reduction's
/// identity: `0` for sum, `1` for prod).
pub(super) fn fold_all<T, F>(
    values: &[T],
    shape: &[usize],
    reduced: &[bool],
    init: T,
    op: F,
) -> Result<Vec<T>>
where
    T: Num + NdScalar,
    F: Fn(T, T) -> T + Copy,
{
    let lanes = fold_lanes(<T as Scalar>::DTYPE);
    let mut out = Vec::with_capacity(kept_extent(shape, reduced));
    for_each_lane(values, shape, reduced, |view| {
        out.push(fold_lane(&view, init, op, lanes));
        Ok(())
    })?;
    Ok(out)
}

/// Fold every output lane starting from its FIRST element — the shape of a
/// max/min reduction, which has no identity element.
pub(super) fn fold_first<T, F>(
    name: &str,
    what: &str,
    values: &[T],
    shape: &[usize],
    reduced: &[bool],
    op: F,
) -> Result<Vec<T>>
where
    T: Num + NdScalar,
    F: Fn(T, T) -> T + Copy,
{
    if reduced_extent(shape, reduced) == 0 {
        return Err(Error::shape(format!(
            "op '{name}': zero-size reduction — a reduced axis has extent 0, and numpy's {what} \
             has no identity element for an empty reduction"
        )));
    }
    let lanes = fold_lanes(<T as Scalar>::DTYPE);
    let mut out = Vec::with_capacity(kept_extent(shape, reduced));
    for_each_lane(values, shape, reduced, |view| {
        out.push(fold_first_lane(&view, op, lanes));
        Ok(())
    })?;
    Ok(out)
}

/// The contiguous-block specialization: fold every output of a `Block` whose
/// `inner > 1`, accumulating the `inner` output lanes of each group row by row.
///
/// `identity` is the reduction's identity element (`0` for sum, `1` for prod);
/// `None` seeds each group from its FIRST row, which is how max/min fold without
/// an identity.  Output order is C order (outer, then inner).
pub(super) fn fold_contiguous_block<T, F>(
    values: &[T],
    shape: &[usize],
    block: &Block,
    identity: Option<T>,
    op: F,
) -> Result<Vec<T>>
where
    T: Num + NdScalar,
    F: Fn(T, T) -> T + Copy,
{
    let view = array_view(values, shape)?;
    let groups = view
        .to_shape((block.outer, block.red, block.inner))
        .map_err(|error| {
            Error::shape(format!(
                "contiguous reduction block {}x{}x{} does not reshape the operand: {error}",
                block.outer, block.red, block.inner
            ))
        })?;
    let seed_from_first_row = identity.is_none();
    let mut out = Vec::with_capacity(block.out_len());
    for group in groups.axis_iter(Axis(0)) {
        let mut acc = match identity {
            Some(init) => Array1::from_elem(block.inner, init),
            None => group.index_axis(Axis(0), 0).to_owned(),
        };
        let mut rows = group.axis_iter(Axis(0));
        if seed_from_first_row {
            // The first row already seeded the accumulators.
            rows.next();
        }
        for row in rows {
            Zip::from(&mut acc)
                .and(&row)
                .for_each(|slot, &value| *slot = op(*slot, value));
        }
        out.extend(acc.iter().copied());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// The per-element comparison a max/min fold uses
// ---------------------------------------------------------------------------

/// The per-element comparison a max/min fold uses.
///
/// `Num::max`/`Num::min` test BOTH operands for NaN first, which is three
/// dependent operations before the select and keeps the fold loop from
/// vectorizing (~2x numpy on a row max); here NaN is decided only when neither
/// `>=` holds, so the common path is one compare-and-select.  The result matches
/// `Num::max`/`Num::min` wherever it is defined — including `max(-0.0, 0.0)`
/// keeping the left operand and any NaN operand propagating — except that a
/// propagated NaN carries the canonical payload rather than the operand's.
pub(super) trait FoldSelect: Copy {
    fn select_max(self, other: Self) -> Self;
    fn select_min(self, other: Self) -> Self;
}

macro_rules! fold_select_total {
    ($($ty:ty),*) => {$(
        impl FoldSelect for $ty {
            #[inline]
            fn select_max(self, other: Self) -> Self {
                if self >= other {
                    self
                } else {
                    other
                }
            }
            #[inline]
            fn select_min(self, other: Self) -> Self {
                if self <= other {
                    self
                } else {
                    other
                }
            }
        }
    )*};
}

macro_rules! fold_select_ieee {
    ($($ty:ty),*) => {$(
        impl FoldSelect for $ty {
            #[inline]
            fn select_max(self, other: Self) -> Self {
                if self >= other {
                    self
                } else if other >= self {
                    other
                } else {
                    // Neither ordering holds: a NaN operand propagates.
                    Self::NAN
                }
            }
            #[inline]
            fn select_min(self, other: Self) -> Self {
                if self <= other {
                    self
                } else if other <= self {
                    other
                } else {
                    Self::NAN
                }
            }
        }
    )*};
}

fold_select_total!(bool, i8, i16, i32, i64, u8, u16, u32, u64);
fold_select_ieee!(half::f16, f32, f64);

/// Complex has no ordering in the scalar vocabulary and an arg/max reduction
/// over it is rejected before a fold runs; `Num::max`/`Num::min` are the
/// identity on it, which this mirrors.
macro_rules! fold_select_identity {
    ($($ty:ty),*) => {$(
        impl FoldSelect for $ty {
            #[inline]
            fn select_max(self, _other: Self) -> Self {
                self
            }
            #[inline]
            fn select_min(self, _other: Self) -> Self {
                self
            }
        }
    )*};
}

fold_select_identity!(Complex<f32>, Complex<f64>);

// ---------------------------------------------------------------------------
// Mask and rank helpers
// ---------------------------------------------------------------------------

/// The reduced-axis mask of a `reduce_*` op; an empty `axes` attribute means
/// ALL axes (numpy's `axis=None`), exactly as `infer_reduction` reads it.
pub(super) fn reduce_mask(axes: &[i64], rank: usize, name: &str) -> Result<Vec<bool>> {
    if axes.is_empty() {
        return Ok(vec![true; rank]);
    }
    let mut mask = vec![false; rank];
    for axis in util::normalize_axes(axes, rank, &format!("op '{name}'"))? {
        mask[axis] = true;
    }
    Ok(mask)
}

/// The declared result rank a reduction must produce: the operand rank with
/// `keepdims`, otherwise one axis fewer per reduced axis.  The declared shape
/// decides the layout, so a mismatch is refused instead of reshaped.
pub(super) fn expect_rank(
    name: &str,
    out_shape: &[usize],
    kept: usize,
    rank: usize,
    keepdims: bool,
) -> Result<()> {
    let expected = if keepdims { rank } else { kept };
    if out_shape.len() != expected {
        return Err(Error::shape(format!(
            "op '{name}': declared result shape {} has rank {}, but the reduction yields rank \
             {expected} (keepdims={keepdims})",
            shape_text(out_shape),
            out_shape.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{
        divide_by, reduce,
        test_support::{buffer, reduce_op},
        Kind,
    };
    use super::*;
    use crate::buffer::Storage;
    use crate::plan::OpSpec;
    use crate::scalar::ScalarValue;
    use half::f16;
    use num_complex::Complex;

    // -----------------------------------------------------------------------
    // The independent ndarray reference: permute kept-first, reshape to
    // (kept, red), and fold each lane sequentially with `fold_axis`.
    // -----------------------------------------------------------------------

    fn lane_matrix<T: Num + NdScalar>(
        values: &[T],
        shape: &[usize],
        reduced: &[bool],
    ) -> ndarray::Array2<T> {
        let view = array_view::<T>(values, shape).expect("operand view");
        let permuted = view.permuted_axes(axis_permutation(shape.len(), reduced));
        let kept = kept_extent(shape, reduced);
        let red = reduced_extent(shape, reduced);
        permuted
            .to_shape((kept, red))
            .expect("reduced lanes")
            .into_owned()
    }

    /// One lane per output, folded sequentially from `init` in operand order.
    fn reference_identity<T, F>(flat: &ndarray::Array2<T>, init: T, op: F) -> Vec<T>
    where
        T: Num + NdScalar,
        F: Fn(T, T) -> T + Copy,
    {
        flat.fold_axis(Axis(1), init, move |acc, &value| op(*acc, value))
            .into_raw_vec_and_offset()
            .0
    }

    /// One lane per output, folded sequentially from its FIRST element.
    fn reference_first<T, F>(flat: &ndarray::Array2<T>, op: F) -> Vec<T>
    where
        T: Num + NdScalar,
        F: Fn(T, T) -> T + Copy,
    {
        flat.fold_axis(Axis(1), None::<T>, move |acc: &Option<T>, &value| {
            Some(match *acc {
                None => value,
                Some(prev) => op(prev, value),
            })
        })
        .iter()
        .map(|slot| slot.unwrap_or_else(T::zero))
        .collect()
    }

    /// The general (ndarray) reference fold — the same per-lane sequential fold
    /// the specialization replaced, kept here as the independent oracle.
    fn general<T: Num + FoldSelect + NdScalar>(
        kind: Kind,
        values: &[T],
        shape: &[usize],
        reduced: &[bool],
        count: usize,
        acc_dtype: DType,
    ) -> Vec<T> {
        let flat = lane_matrix(values, shape, reduced);
        match kind {
            Kind::Sum => reference_identity(&flat, T::zero(), |a, b| a.add(b)),
            Kind::Prod => reference_identity(&flat, T::one(), |a, b| a.mul(b)),
            Kind::Max => reference_first(&flat, |a, b| a.select_max(b)),
            Kind::Min => reference_first(&flat, |a, b| a.select_min(b)),
            Kind::Mean => divide_by(
                acc_dtype,
                reference_identity(&flat, T::zero(), |a, b| a.add(b)),
                count,
            ),
        }
    }

    /// The declared result extent of a reduction, for building the `OpSpec`.
    fn out_shape_of(shape: &[usize], axes: &[i64], keepdims: bool) -> Vec<usize> {
        let mask = reduce_mask(axes, shape.len(), "reference").unwrap();
        let mut out = Vec::new();
        for (axis, dim) in shape.iter().enumerate() {
            if mask[axis] {
                if keepdims {
                    out.push(1);
                }
            } else {
                out.push(*dim);
            }
        }
        out
    }

    /// Shapes that hit every layout class: all-axes, a trailing axis, a leading
    /// axis, a contiguous middle block, a multi-axis block, a gapped axis pair
    /// (the fallback), a rank-1 and a rank-0 operand.
    fn layout_cases() -> Vec<(Vec<usize>, Vec<i64>, bool)> {
        let mut cases = Vec::new();
        for shape in [
            vec![],
            vec![7],
            vec![2, 3],
            vec![3, 4, 5],
            vec![9, 1, 7],
            vec![5, 6, 3, 2],
        ] {
            let rank = shape.len() as i64;
            let mut axes_sets: Vec<Vec<i64>> = vec![vec![]];
            for axis in 0..rank {
                axes_sets.push(vec![axis]);
                axes_sets.push(vec![axis - rank]);
            }
            if rank >= 2 {
                axes_sets.push((0..rank).collect());
                axes_sets.push(vec![0, 1]);
                axes_sets.push(vec![rank - 1, rank - 2]);
            }
            if rank == 3 {
                // A gapped pair: the general path owns this one.
                axes_sets.push(vec![0, 2]);
            }
            if rank == 4 {
                axes_sets.push(vec![1, 2]);
                axes_sets.push(vec![0, 3]);
            }
            for axes in axes_sets {
                for keepdims in [false, true] {
                    cases.push((shape.clone(), axes.clone(), keepdims));
                }
            }
        }
        cases
    }

    fn kind_cases() -> [(&'static str, &'static str, Kind); 5] {
        [
            ("reduce_sum", "sum", Kind::Sum),
            ("reduce_prod", "prod", Kind::Prod),
            ("reduce_max", "max", Kind::Max),
            ("reduce_min", "min", Kind::Min),
            ("reduce_mean", "mean", Kind::Mean),
        ]
    }

    /// Deterministic, sign-varying integer data (no overflow, so the reference
    /// and the kernel agree on the mathematical result whatever the order).
    fn int_data(count: usize) -> Vec<i64> {
        (0..count).map(|index| (index as i64 % 5) - 2).collect()
    }

    fn close(left: f32, right: f32) -> bool {
        left == right || (left - right).abs() <= 1e-6 + 1e-5 * right.abs()
    }

    #[test]
    fn block_classifies_every_layout_class() {
        // All axes: one group, one contiguous run.
        let full = Block::new(&[2, 3], &[true, true]).unwrap();
        assert_eq!((full.outer, full.red, full.inner), (1, 6, 1));
        // Trailing axis: one row per group.
        let rows = Block::new(&[3, 4, 5], &[false, false, true]).unwrap();
        assert_eq!((rows.outer, rows.red, rows.inner), (12, 5, 1));
        // Leading axis: `inner` lanes per group.
        let cols = Block::new(&[3, 4, 5], &[true, false, false]).unwrap();
        assert_eq!((cols.outer, cols.red, cols.inner), (1, 3, 20));
        // A contiguous middle block keeps both an outer and an inner extent.
        let middle = Block::new(&[5, 6, 3, 2], &[false, true, true, false]).unwrap();
        assert_eq!((middle.outer, middle.red, middle.inner), (5, 18, 2));
        // A gap inside the reduced axes, and a reduction over nothing, are the
        // general path's.
        assert!(Block::new(&[2, 3, 4], &[true, false, true]).is_none());
        assert!(Block::new(&[2, 3], &[false, false]).is_none());
        assert_eq!(Block::new(&[], &[]).map(|b| b.out_len()), None);
    }

    #[test]
    fn fast_path_matches_the_offset_table_path_for_integers() {
        for (shape, axes, keepdims) in layout_cases() {
            let count: usize = shape.iter().product();
            let values = int_data(count);
            let input = buffer(DType::Int64, &shape, Storage::I64(values.clone()));
            let reduced = reduce_mask(&axes, shape.len(), "reference").unwrap();
            let declared = out_shape_of(&shape, &axes, keepdims);
            let reduced_count = reduced_extent(&shape, &reduced);
            for (name, attribute, kind) in kind_cases() {
                // Mean of integers accumulates (and stores) float64, so it is
                // pinned by the float test instead.
                if name == "reduce_mean" {
                    continue;
                }
                let op = reduce_op(name, attribute, &axes, keepdims, DType::Int64, &declared);
                let got = reduce(&op, &input, &declared).unwrap().remove(0);
                let want = general(
                    kind,
                    &values,
                    &shape,
                    &reduced,
                    reduced_count,
                    DType::Int64,
                );
                assert_eq!(
                    got.storage,
                    Storage::I64(want),
                    "op {name} shape {shape:?} axes {axes:?} keepdims {keepdims}"
                );
            }
        }
    }

    #[test]
    fn fast_path_matches_the_offset_table_path_for_float32() {
        for (shape, axes, keepdims) in layout_cases() {
            let count: usize = shape.iter().product();
            let values: Vec<f32> = (0..count).map(|index| ((index * 7 % 23) as f32) * 0.5 - 5.5).collect();
            let input = buffer(DType::Float32, &shape, Storage::F32(values.clone()));
            let reduced = reduce_mask(&axes, shape.len(), "reference").unwrap();
            let declared = out_shape_of(&shape, &axes, keepdims);
            let reduced_count = reduced_extent(&shape, &reduced);
            for (name, attribute, kind) in kind_cases() {
                let op = reduce_op(name, attribute, &axes, keepdims, DType::Float32, &declared);
                let got = reduce(&op, &input, &declared).unwrap().remove(0);
                // `reduce_prod` multiplies the raw values, which is why its data
                // is kept near zero here; the comparison is the tolerance one.
                let want = general(
                    kind,
                    &values,
                    &shape,
                    &reduced,
                    reduced_count,
                    DType::Float32,
                );
                match &got.storage {
                    Storage::F32(got) => {
                        assert_eq!(got.len(), want.len(), "op {name} shape {shape:?}");
                        for (index, (left, right)) in got.iter().zip(&want).enumerate() {
                            assert!(
                                close(*left, *right),
                                "op {name} shape {shape:?} axes {axes:?} keepdims {keepdims} \
                                 element {index}: {left} vs {right}"
                            );
                        }
                    }
                    other => panic!("unexpected storage {other:?}"),
                }
            }
        }
    }

    #[test]
    fn float16_and_complex_folds_keep_the_sequential_order() {
        // The lane policy skips float16/complex, where the parity contract's ulp
        // window cannot absorb a reassociated sum — so those stay BIT-identical
        // to the ndarray reference, even on a long run.
        let count = 1000;
        let values: Vec<f16> = (0..count).map(|index| f16::from_f32(index as f32 * 0.01)).collect();
        let input = buffer(DType::Float16, &[count], Storage::F16(values.clone()));
        let op = reduce_op("reduce_sum", "sum", &[], false, DType::Float16, &[]);
        let got = reduce(&op, &input, &[]).unwrap().remove(0);
        assert_eq!(got.storage, Storage::F16(general(Kind::Sum, &values, &[count], &[true], count, DType::Float16)));

        let values: Vec<Complex<f32>> = (0..count)
            .map(|index| Complex::new(index as f32 * 0.25, -(index as f32) * 0.5))
            .collect();
        let input = buffer(DType::Complex64, &[count], Storage::C64(values.clone()));
        let op = reduce_op("reduce_sum", "sum", &[], false, DType::Complex64, &[]);
        let got = reduce(&op, &input, &[]).unwrap().remove(0);
        assert_eq!(got.storage, Storage::C64(general(Kind::Sum, &values, &[count], &[true], count, DType::Complex64)));
    }

    // -----------------------------------------------------------------------
    // Every dtype against the independent ndarray reference.
    // -----------------------------------------------------------------------

    fn sample<T: Num + NdScalar>(index: usize) -> T {
        let dtype = <T as Scalar>::DTYPE;
        let value = if dtype.is_float() || dtype.is_complex() {
            // Fractional, sign-varying: exercises the float reassociation.
            ((index * 37 % 101) as f64) * 0.125 - 6.0
        } else {
            // Small integers (a 0 in the mix keeps products tame).
            (index * 7 % 13) as f64
        };
        T::from_scalar(ScalarValue::F(value))
    }

    fn check_dtype<T>(label: &str)
    where
        T: Num + FoldSelect + NdScalar + std::fmt::Debug,
    {
        let dtype = <T as Scalar>::DTYPE;
        // (shape, axes): full reduction, a long trailing axis, a short trailing
        // axis, a leading contiguous block (inner > 1), a middle block and a
        // gapped pair that forces the general strided lane.
        let cases: [(&[usize], &[i64]); 6] = [
            (&[40], &[]),
            (&[4, 40], &[1]),
            (&[6, 5], &[1]),
            (&[9, 5], &[0]),
            (&[3, 4, 5], &[1]),
            (&[3, 4, 5], &[0, 2]),
        ];
        let kinds: &[(&str, &str, Kind)] = if dtype.is_bool() {
            &[("reduce_max", "max", Kind::Max), ("reduce_min", "min", Kind::Min)]
        } else if dtype.is_complex() {
            &[("reduce_sum", "sum", Kind::Sum)]
        } else {
            &[
                ("reduce_sum", "sum", Kind::Sum),
                ("reduce_max", "max", Kind::Max),
                ("reduce_min", "min", Kind::Min),
            ]
        };
        for (shape, axes) in cases {
            let count: usize = shape.iter().product();
            let values: Vec<T> = (0..count).map(sample::<T>).collect();
            let input = buffer(dtype, shape, <T as Scalar>::into_storage(values.clone()));
            let reduced = reduce_mask(axes, shape.len(), "reference").unwrap();
            let declared = out_shape_of(shape, axes, false);
            let reduced_count = reduced_extent(shape, &reduced);
            // Integer/bool are exact under reassociation; float16/complex stay in
            // one sequential lane. float32/float64 lanes longer than the 8-lane
            // chunk are reassociated, so only those relax to the tolerance.
            let exact = dtype.is_bool()
                || dtype.is_int()
                || dtype == DType::Float16
                || dtype.is_complex();
            let short_lane = reduced_count <= LANE_COUNT;
            for (name, attribute, kind) in kinds {
                let op = reduce_op(name, attribute, axes, false, dtype, &declared);
                let got = reduce(&op, &input, &declared).unwrap().remove(0);
                let got_values = <T as NdScalar>::slice_of(&got.storage)
                    .unwrap_or_else(|| panic!("{label}: unexpected result storage {:?}", got.storage));
                let want = general(*kind, &values, shape, &reduced, reduced_count, dtype);
                assert_eq!(got_values.len(), want.len(), "{label} {name} shape {shape:?}");
                for (index, (left, right)) in got_values.iter().zip(&want).enumerate() {
                    if exact || short_lane {
                        assert_eq!(
                            left, right,
                            "{label} {name} shape {shape:?} axes {axes:?} element {index}: \
                             {left:?} vs {right:?}"
                        );
                    } else {
                        // The per-lane 8-accumulator reassociation is the reason a
                        // long float lane may miss a bit-identical result; the
                        // documented parity window is the acceptance.
                        let l = <T as Scalar>::to_scalar(*left).real();
                        let r = <T as Scalar>::to_scalar(*right).real();
                        assert!(
                            left == right || (l - r).abs() <= 1e-6 + 1e-5 * r.abs(),
                            "{label} {name} shape {shape:?} axes {axes:?} element {index}: \
                             {left:?} vs {right:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn ndarray_reference_matches_the_kernel_for_every_dtype() {
        check_dtype::<bool>("bool");
        check_dtype::<i8>("int8");
        check_dtype::<i16>("int16");
        check_dtype::<i32>("int32");
        check_dtype::<i64>("int64");
        check_dtype::<u8>("uint8");
        check_dtype::<u16>("uint16");
        check_dtype::<u32>("uint32");
        check_dtype::<u64>("uint64");
        check_dtype::<f16>("float16");
        check_dtype::<f32>("float32");
        check_dtype::<f64>("float64");
        check_dtype::<Complex<f32>>("complex64");
        check_dtype::<Complex<f64>>("complex128");
    }

    #[test]
    fn long_integer_sums_are_bit_exact_under_the_lane_split() {
        // 1000 elements is far past the 8-lane chunk, so the multi-accumulator
        // path runs; wrapping integer addition is associative, so the result must
        // still equal the sequential fold exactly.
        let count = 1000;
        let values: Vec<i64> = (0..count as i64).map(|index| index * 1_000_003 - 500).collect();
        let expected: i64 = values.iter().fold(0i64, |acc, value| acc.wrapping_add(*value));
        let input = buffer(DType::Int64, &[count], Storage::I64(values.clone()));
        let op = reduce_op("reduce_sum", "sum", &[], false, DType::Int64, &[]);
        assert_eq!(reduce(&op, &input, &[]).unwrap()[0].storage, Storage::I64(vec![expected]));
        let expected_prod: i64 = values.iter().fold(1i64, |acc, value| acc.wrapping_mul(*value));
        let op = reduce_op("reduce_prod", "prod", &[], false, DType::Int64, &[]);
        assert_eq!(reduce(&op, &input, &[]).unwrap()[0].storage, Storage::I64(vec![expected_prod]));
    }

    #[test]
    fn long_float_sums_stay_within_the_documented_tolerance() {
        let count = 10_000;
        let values: Vec<f32> = (0..count).map(|index| ((index * 37 % 101) as f32) * 0.125 - 3.0).collect();
        let sequential = values.iter().fold(0.0f32, |acc, value| acc + *value);
        let input = buffer(DType::Float32, &[count], Storage::F32(values));
        let op = reduce_op("reduce_sum", "sum", &[], false, DType::Float32, &[]);
        match &reduce(&op, &input, &[]).unwrap()[0].storage {
            Storage::F32(sum) => assert!(
                close(sum[0], sequential),
                "lane-folded sum {} vs sequential {sequential}",
                sum[0]
            ),
            other => panic!("unexpected storage {other:?}"),
        }
    }

    #[test]
    fn multi_accumulator_max_and_min_propagate_nan_in_a_long_run() {
        for position in [0usize, 3, 8, 9, 777] {
            let count = 1000;
            let mut values: Vec<f32> = (0..count).map(|index| index as f32 * 0.5 - 100.0).collect();
            values[position] = f32::NAN;
            let input = buffer(DType::Float32, &[count], Storage::F32(values));
            for (name, attribute) in [("reduce_max", "max"), ("reduce_min", "min")] {
                let op = reduce_op(name, attribute, &[], false, DType::Float32, &[]);
                match &reduce(&op, &input, &[]).unwrap()[0].storage {
                    Storage::F32(out) => assert!(out[0].is_nan(), "op {name} at {position}"),
                    other => panic!("unexpected storage {other:?}"),
                }
                // Row-wise (many short runs) propagates the same way.
                let input = buffer(DType::Float32, &[20, 50], Storage::F32({
                    let mut values: Vec<f32> = (0..1000).map(|index| index as f32).collect();
                    values[position] = f32::NAN;
                    values
                }));
                let out = reduce(&op_for_rows(name, attribute), &input, &[20]).unwrap();
                match &out[0].storage {
                    Storage::F32(out) => {
                        assert!(out[position / 50].is_nan(), "op {name} row at {position}");
                    }
                    other => panic!("unexpected storage {other:?}"),
                }
            }
        }
    }

    fn op_for_rows(name: &str, attribute: &str) -> OpSpec {
        reduce_op(name, attribute, &[1], false, DType::Float32, &[20])
    }
}
