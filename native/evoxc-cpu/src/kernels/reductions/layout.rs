//! The fold and accumulator machinery every reduce kind shares.
//!
//! [`Layout`] is the general C-order iteration plan (kept positions address the
//! output positions, `red_offsets` the reduced ones) and [`Block`] the
//! contiguous fast path; the `fold_*` helpers plus [`FoldSelect`] are the
//! accumulator vocabulary both paths fold through.  They are deliberately one
//! unit: the two paths must stay bit-for-bit equivalent for integer, bool,
//! float16 and complex folds, and only float32/float64 sums may move inside the
//! documented parity window.
use evoxc_core::{DType, Error, Result};
use num_complex::Complex;

use crate::buffer::shape_text;
use crate::kernels::util;
use crate::scalar::Num;

// ---------------------------------------------------------------------------
// Shared layout: C-order iteration over kept and reduced positions
// ---------------------------------------------------------------------------

/// C-order iteration plan for one reduction.
///
/// `kept_strides`/`kept_dims` describe the output positions and map each one
/// to its base input offset; `red_offsets` lists every reduced position's
/// offset relative to that base.  Both together address the reduction with no
/// per-element division, and an empty reduced set (rank 0, or a reduction over
/// no axes) yields exactly one offset — which is why an "all-axes" reduction
/// of a scalar is the identity fold.
pub(super) struct Layout {
    kept_strides: Vec<usize>,
    kept_dims: Vec<usize>,
    pub(super) red_offsets: Vec<usize>,
}

impl Layout {
    pub(super) fn new(shape: &[usize], reduced: &[bool]) -> Layout {
        let strides = util::c_strides(shape);
        let mut kept_strides = Vec::new();
        let mut kept_dims = Vec::new();
        let mut red_strides = Vec::new();
        let mut red_dims = Vec::new();
        for (axis, dim) in shape.iter().enumerate() {
            if reduced[axis] {
                red_strides.push(strides[axis]);
                red_dims.push(*dim);
            } else {
                kept_strides.push(strides[axis]);
                kept_dims.push(*dim);
            }
        }
        let mut red_offsets = Vec::with_capacity(util::numel(&red_dims));
        let mut index = vec![0usize; red_dims.len()];
        for _ in 0..util::numel(&red_dims) {
            red_offsets.push(util::flat_index(&index, &red_strides));
            advance(&mut index, &red_dims);
        }
        Layout {
            kept_strides,
            kept_dims,
            red_offsets,
        }
    }

    pub(super) fn out_len(&self) -> usize {
        util::numel(&self.kept_dims)
    }

    /// Visit every output position in C-order with its base input offset.
    pub(super) fn for_each_output(&self, mut visit: impl FnMut(usize, usize)) {
        let mut index = vec![0usize; self.kept_dims.len()];
        for position in 0..self.out_len() {
            visit(position, util::flat_index(&index, &self.kept_strides));
            advance(&mut index, &self.kept_dims);
        }
    }
}

// ---------------------------------------------------------------------------
// Contiguous-block fast path
// ---------------------------------------------------------------------------

/// A reduction whose reduced axes form ONE contiguous block in C order.
///
/// Such an operand is `outer` consecutive groups of `red * inner` elements, and
/// each group is `red` interleaved runs of `inner` consecutive lanes: the
/// elements folded into output lane `i` of a group are `group[k * inner + i]`
/// for `k in 0..red`.  That covers every shape real graphs reduce — a full
/// reduction, a trailing-axis row reduction, a leading column reduction and any
/// contiguous multi-axis block — with one flat walk and no per-element offset
/// table, which is exactly the cost the generic [`Layout`] pays (it allocates
/// and re-indexes an offset vector, and re-derives the output offsets with a
/// division-free odometer per output).
///
/// `inner == 1` means each output sits on one contiguous run (the case a plain
/// fold can vectorize); `inner > 1` means a group reduces into `inner` lanes at
/// once (the column-sum / outer-product shape).
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
    /// contiguous (a gap inside them forces the general [`Layout`] path) or when
    /// nothing is reduced at all (an identity reduction keeps the operand shape
    /// and values, which the general path handles with one zero offset).
    pub(super) fn new(shape: &[usize], reduced: &[bool]) -> Option<Block> {
        let first = reduced.iter().position(|flag| *flag)?;
        let last = reduced.iter().rposition(|flag| *flag)?;
        if !reduced[first..=last].iter().all(|flag| *flag) {
            return None;
        }
        Some(Block {
            outer: util::numel(&shape[..first]),
            red: util::numel(&shape[first..=last]),
            inner: util::numel(&shape[last + 1..]),
        })
    }

    pub(super) fn out_len(&self) -> usize {
        self.outer * self.inner
    }
}

/// Independent accumulator lanes per contiguous fold run.
///
/// A sequential float fold is ONE dependency chain (4-cycle add latency per
/// element), which is why a full `reduce_sum` used to run ~14x slower than
/// numpy's pairwise sum; [`LANE_COUNT`] accumulators make the chains
/// independent, which is what lets the loop reach the memory bound.
const LANE_COUNT: usize = 8;

/// Lanes one contiguous fold run uses for an accumulator of `dtype`.
///
/// Integer and bool folds are exact under any association (wrapping adds and
/// multiplies are associative), so they take the multi-accumulator path too.
/// Float16 folds IN float16 and complex in complex, where the parity contract's
/// ulp window is tighter than a reassociated sum can honour, so those keep the
/// interpreter's exact sequential order (and neither is on a large workload).
fn fold_lanes(dtype: DType) -> usize {
    if matches!(
        dtype,
        DType::Float16 | DType::Complex64 | DType::Complex128
    ) {
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

/// Fold every output of a contiguous block, in C output order.
///
/// `identity` is the reduction's identity element (`0` for sum, `1` for prod);
/// `None` seeds each group from its FIRST element, which is how max/min fold
/// without an identity (the caller has already rejected an empty reduction).
pub(super) fn fold_groups<T: Num, F: Fn(T, T) -> T + Copy>(
    values: &[T],
    block: &Block,
    identity: Option<T>,
    op: F,
) -> Vec<T> {
    let mut out = Vec::with_capacity(block.out_len());
    if block.red == 0 || block.inner == 0 {
        // An empty reduction has no group to walk; `reduce_*` rejects a
        // zero-size max/min before this point and the identity folds below have
        // nothing to produce when there is no output lane.
        return out;
    }
    let lanes = fold_lanes(T::DTYPE);
    for group in values.chunks_exact(block.red * block.inner) {
        if block.inner == 1 {
            let init = identity.unwrap_or(group[0]);
            out.push(fold_run(group, init, op, lanes));
        } else {
            // One accumulator per lane: the identity when the reduction has one,
            // otherwise the group's first row (max/min have no identity).
            let mut acc: Vec<T> = match identity {
                Some(value) => vec![value; block.inner],
                None => group[..block.inner].to_vec(),
            };
            let mut rows = group.chunks_exact(block.inner);
            if identity.is_none() {
                // The first row already seeded the accumulators.
                rows.next();
            }
            for row in rows {
                for (slot, value) in acc.iter_mut().zip(row) {
                    *slot = op(*slot, *value);
                }
            }
            out.append(&mut acc);
        }
    }
    out
}

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

/// Increment a C-order odometer (the last axis varies fastest).
fn advance(index: &mut [usize], dims: &[usize]) {
    for axis in (0..dims.len()).rev() {
        index[axis] += 1;
        if index[axis] < dims[axis] {
            return;
        }
        index[axis] = 0;
    }
}

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

/// Fold every reduced position into its output slot, starting from `init`
/// (the reduction's identity: `0` for sum, `1` for prod).
pub(super) fn fold_all<T: Copy>(
    layout: &Layout,
    values: &[T],
    init: T,
    combine: impl Fn(T, T) -> T,
) -> Vec<T> {
    let mut out = Vec::with_capacity(layout.out_len());
    layout.for_each_output(|_, base| {
        let mut acc = init;
        for offset in &layout.red_offsets {
            acc = combine(acc, values[base + offset]);
        }
        out.push(acc);
    });
    out
}

/// Fold every reduced position starting from the FIRST element — the shape of
/// a max/min reduction, which has no identity element.
pub(super) fn fold_first<T: Copy>(
    name: &str,
    what: &str,
    layout: &Layout,
    values: &[T],
    combine: impl Fn(T, T) -> T,
) -> Result<Vec<T>> {
    if layout.red_offsets.is_empty() {
        return Err(Error::shape(format!(
            "op '{name}': zero-size reduction — a reduced axis has extent 0, and numpy's {what} \
             has no identity element for an empty reduction"
        )));
    }
    let mut out = Vec::with_capacity(layout.out_len());
    layout.for_each_output(|_, base| {
        let mut acc = values[base + layout.red_offsets[0]];
        for offset in layout.red_offsets.iter().skip(1) {
            acc = combine(acc, values[base + offset]);
        }
        out.push(acc);
    });
    Ok(out)
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
    use half::f16;
    use num_complex::Complex;

    // -----------------------------------------------------------------------
    // The contiguous fast path (see `Block`) against the retained offset-table
    // path: bit-exact for integers/bool/float16, tolerance-equal for f32/f64.
    // -----------------------------------------------------------------------

    /// The general (offset-table) fold — the implementation the fast path
    /// replaced, kept here as the reference.
    fn general<T: Num + FoldSelect>(
        kind: Kind,
        values: &[T],
        shape: &[usize],
        reduced: &[bool],
        count: usize,
        acc_dtype: DType,
    ) -> Vec<T> {
        let layout = Layout::new(shape, reduced);
        match kind {
            Kind::Sum => fold_all(&layout, values, T::zero(), |a, b| a.add(b)),
            Kind::Prod => fold_all(&layout, values, T::one(), |a, b| a.mul(b)),
            Kind::Max => {
                fold_first("reference", "maximum", &layout, values, |a, b| a.select_max(b)).unwrap()
            }
            Kind::Min => {
                fold_first("reference", "minimum", &layout, values, |a, b| a.select_min(b)).unwrap()
            }
            Kind::Mean => divide_by(
                acc_dtype,
                fold_all(&layout, values, T::zero(), |a, b| a.add(b)),
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
            let reduced_count: usize = reduced
                .iter()
                .zip(&shape)
                .filter(|(flag, _)| **flag)
                .map(|(_, dim)| *dim)
                .product();
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
            let reduced_count: usize = reduced
                .iter()
                .zip(&shape)
                .filter(|(flag, _)| **flag)
                .map(|(_, dim)| *dim)
                .product();
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
        // to the offset-table path, even on a long run.
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
