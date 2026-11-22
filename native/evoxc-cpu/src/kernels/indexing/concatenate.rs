//! `concatenate` kernel: join operands along one axis.
//!
//! Every operand is cast to the op's DECLARED (already promoted) dtype; rank-0
//! operands, a NULL `axis` and non-axis shape mismatches are errors. The module
//! doc in [`super`] carries the shared declared-shape and error-wording rules.
use evoxc_core::{DType, Error, Result};

use crate::buffer::{array_view_mut, Buffer, NdScalar};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::plan::OpSpec;
use crate::scalar::Scalar;

use super::{pooled_total, reconcile};

pub(super) fn concatenate_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    let sources: Vec<&Buffer> = operands.iter().map(|value| &**value).collect();
    let declared = ctx.result_shape(spec, 0)?;
    single(concatenate(
        &sources,
        spec.attrs.opt_i64("axis")?,
        spec.result_types[0].dtype,
        &declared,
    )?)
}

pub(super) fn concatenate(
    sources: &[&Buffer],
    axis: Option<i64>,
    dtype: DType,
    declared: &[Option<usize>],
) -> Result<Buffer> {
    let Some(first) = sources.first() else {
        return Err(Error::shape("concatenate: expected at least one operand"));
    };
    let rank = first.rank();
    for (position, value) in sources.iter().enumerate() {
        if value.rank() != rank {
            return Err(Error::shape(format!(
                "concatenate: all the input arrays must have same number of dimensions, but the array at index 0 has {rank} dimension(s) and the array at index {position} has {} dimension(s)",
                value.rank()
            )));
        }
    }
    if rank == 0 {
        return Err(Error::shape(
            "concatenate: zero-dimensional arrays cannot be concatenated",
        ));
    }
    let Some(axis) = axis else {
        return Err(Error::shape(
            "concatenate: axis must be an int, got None — a null axis has no numpy equivalent for zero-dimensional operands, which this op rejects",
        ));
    };
    let axis = util::normalize_axis(axis, rank, "concatenate")?;
    let mut shape = first.shape.clone();
    let mut total = 0usize;
    for (position, value) in sources.iter().enumerate() {
        for dim in 0..rank {
            if dim == axis {
                continue;
            }
            if value.shape[dim] != first.shape[dim] {
                return Err(Error::shape(format!(
                    "concatenate: all the input array dimensions except for the concatenation axis must match exactly, but along dimension {dim}, the array at index 0 has size {} and the array at index {position} has size {}",
                    first.shape[dim], value.shape[dim]
                )));
            }
        }
        total = total
            .checked_add(value.shape[axis])
            .ok_or_else(|| Error::shape("concatenate: the concatenated dimension is too large"))?;
    }
    shape[axis] = total;
    let shape = reconcile("concatenate", declared, shape)?;
    // Every operand is cast to the op's DECLARED (already promoted) dtype.
    let mut casts: Vec<Option<Buffer>> = Vec::with_capacity(sources.len());
    for value in sources {
        casts.push(if value.dtype == dtype {
            None
        } else {
            Some(value.cast(dtype)?)
        });
    }
    let parts: Vec<&Buffer> = sources
        .iter()
        .enumerate()
        .map(|(position, value)| casts[position].as_ref().unwrap_or(value))
        .collect();
    join(&parts, axis, shape)
}

/// Elements per contiguous copy below which one `memcpy` per outer block costs
/// more than walking the block element by element.
///
/// A part's rows inside one outer block are a single contiguous source run of
/// `extent * inner` elements, so a WIDE run is one `copy_from_slice` per outer
/// block while a NARROW one (a last-axis join of 1-wide parts: `inner == 1`)
/// would be one `memcpy` call per ELEMENT — there ndarray's stride-aware `Zip`
/// wins.  8 elements is where the call's fixed cost stops dominating.
const BULK_RUN_ELEMENTS: usize = 8;

/// Assemble the (already same-dtype) operands into ONE pooled, C-contiguous
/// destination of exactly the declared result size.
///
/// The shape checks above have already rejected every rank/axis mismatch, so
/// the layout rule is numpy's: the destination is `outer` blocks of
/// `(total_axis, inner)` where `outer` is the leading-axes product and `inner`
/// the trailing one, and part `p` owns the axis slots its predecessors do not
/// take.  Every operand is C-contiguous, so inside one outer block a part's
/// rows are one contiguous source run and its destination slots are one
/// contiguous destination range — a wide run is a single `copy_from_slice` per
/// outer block (the fast path), and a narrow one takes ndarray's `Zip` over the
/// destination region (`slice_axis_mut`), which is the reference expression the
/// fast path is pinned against.
///
/// Nothing here standardizes a layout: the destination is built in C order from
/// the start, so no transposing copy is ever paid (which is what ndarray's own
/// last-axis `concatenate` hands back — an F-order array — used to cost).
fn join(parts: &[&Buffer], axis: usize, shape: Vec<usize>) -> Result<Buffer> {
    let dtype = parts[0].dtype;
    macro_rules! join_typed {
        ($ty:ty) => {{
            let total = util::numel(&shape);
            let mut out = pooled_total::<$ty>(total);
            assemble::<$ty>(parts, axis, &shape, &mut out)?;
            <$ty as Scalar>::into_storage(out)
        }};
    }
    let storage = match dtype {
        DType::Bool => join_typed!(bool),
        DType::Int8 => join_typed!(i8),
        DType::Int16 => join_typed!(i16),
        DType::Int32 => join_typed!(i32),
        DType::Int64 => join_typed!(i64),
        DType::UInt8 => join_typed!(u8),
        DType::UInt16 => join_typed!(u16),
        DType::UInt32 => join_typed!(u32),
        DType::UInt64 => join_typed!(u64),
        DType::Float16 => join_typed!(half::f16),
        DType::Float32 => join_typed!(f32),
        DType::Float64 => join_typed!(f64),
        DType::Complex64 => join_typed!(num_complex::Complex<f32>),
        DType::Complex128 => join_typed!(num_complex::Complex<f64>),
    };
    Buffer::new(dtype, shape, storage)
}

/// Fill `out` (exactly the concatenated result, in C order) from the parts.
///
/// `out` is fully overwritten: the parts' axis ranges partition `[0,
/// shape[axis])`, so every destination element belongs to exactly one part —
/// which is what lets the caller hand in a recycled (pooled) vector.
fn assemble<T: Scalar + NdScalar>(
    parts: &[&Buffer],
    axis: usize,
    shape: &[usize],
    out: &mut [T],
) -> Result<()> {
    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    let total_axis = shape[axis];
    let block = total_axis * inner;
    // Every offset below is provably in bounds once these two hold; checking
    // them up front keeps the hot loops free of per-element bounds failures
    // (and of any panic on a malformed operand).
    if out.len() != outer * block {
        return Err(Error::shape(format!(
            "concatenate: destination holds {} element(s) for a result of shape {}",
            out.len(),
            crate::buffer::shape_text(shape)
        )));
    }
    let mut offset = 0usize;
    for part in parts {
        let extent = part.shape[axis];
        if extent == 0 {
            continue;
        }
        if offset + extent > total_axis {
            return Err(Error::shape(format!(
                "concatenate: the operand shapes add up to more than the declared concatenated extent {total_axis}"
            )));
        }
        let source = T::slice_of(&part.storage).ok_or_else(|| {
            Error::dtype(format!(
                "concatenate: operand dtype {} does not match the result dtype {}",
                part.dtype.name(),
                <T as NdScalar>::DTYPE.name()
            ))
        })?;
        let run = extent * inner;
        if source.len() != outer * run {
            return Err(Error::shape(format!(
                "concatenate: operand of shape {} holds {} element(s), expected {}",
                crate::buffer::shape_text(&part.shape),
                source.len(),
                outer * run
            )));
        }
        if run >= BULK_RUN_ELEMENTS {
            // One contiguous run per outer block: the block's `extent` rows are
            // adjacent in the operand and adjacent in the destination region.
            for index in 0..outer {
                let src = index * run;
                let dst = index * block + offset * inner;
                out[dst..dst + run].copy_from_slice(&source[src..src + run]);
            }
        } else {
            // Narrow run (a last-axis join of 1-wide parts): a stride-aware
            // element walk over the destination region beats one memcpy per
            // element.  This is the reference expression, pinned against the
            // fast path in the tests below.
            let mut dest = array_view_mut(out, shape)?;
            let region = dest.slice_axis_mut(
                ndarray::Axis(axis),
                ndarray::Slice::new(offset as isize, Some((offset + extent) as isize), 1),
            );
            let view = part.as_array::<T>()?;
            ndarray::Zip::from(region)
                .and(&view)
                .for_each(|destination, value| *destination = *value);
        }
        offset += extent;
    }
    if offset != total_axis {
        return Err(Error::shape(format!(
            "concatenate: the operand shapes add up to {offset}, not the declared concatenated extent {total_axis}"
        )));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::Storage;
    use crate::kernels::indexing::test_support::{buffer, declared, f32_buffer};

    #[test]
    fn concatenate_joins_along_axis_zero() {
        let a = f32_buffer(&[2, 2], &[1., 2., 3., 4.]);
        let b = f32_buffer(&[1, 2], &[5., 6.]);
        let out = concatenate(&[&a, &b], Some(0), DType::Float32, &declared(&[3, 2])).unwrap();
        assert_eq!(out.shape, vec![3, 2]);
        assert_eq!(out.storage, Storage::F32(vec![1., 2., 3., 4., 5., 6.]));
    }

    #[test]
    fn concatenate_joins_along_axis_one() {
        let a = f32_buffer(&[2, 1], &[1., 3.]);
        let b = f32_buffer(&[2, 2], &[2., 9., 4., 8.]);
        let out = concatenate(&[&a, &b], Some(1), DType::Float32, &declared(&[2, 3])).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(out.storage, Storage::F32(vec![1., 2., 9., 3., 4., 8.]));
    }

    #[test]
    fn concatenate_casts_to_the_declared_result_dtype() {
        let a = buffer(DType::Int32, &[2], Storage::I32(vec![1, 2]));
        let b = buffer(DType::Float32, &[2], Storage::F32(vec![0.5, 1.5]));
        let out = concatenate(&[&a, &b], Some(0), DType::Float64, &declared(&[4])).unwrap();
        assert_eq!(out.dtype, DType::Float64);
        assert_eq!(out.storage, Storage::F64(vec![1., 2., 0.5, 1.5]));
    }

    #[test]
    fn concatenate_rejects_rank_zero_operands_and_a_null_axis() {
        let scalar = f32_buffer(&[], &[1.]);
        let err = concatenate(&[&scalar, &scalar], Some(0), DType::Float32, &[]).unwrap_err();
        assert!(
            err.message
                .contains("zero-dimensional arrays cannot be concatenated"),
            "{}",
            err.message
        );

        let a = f32_buffer(&[2], &[1., 2.]);
        let b = f32_buffer(&[1], &[3.]);
        let err = concatenate(&[&a, &b], None, DType::Float32, &declared(&[3])).unwrap_err();
        assert!(err.message.contains("axis must be an int, got None"), "{}", err.message);
    }

    #[test]
    fn concatenate_rejects_a_non_axis_shape_mismatch() {
        let a = f32_buffer(&[2, 2], &[0.; 4]);
        let b = f32_buffer(&[2, 3], &[0.; 6]);
        let err = concatenate(&[&a, &b], Some(0), DType::Float32, &declared(&[4, 2])).unwrap_err();
        assert!(
            err.message.contains("must match exactly"),
            "{}",
            err.message
        );
    }

    // -----------------------------------------------------------------------
    // The reference expression: ndarray's own axis assembly, standardized to C
    // order — exactly what the kernel did before it assembled the destination
    // itself.  Every value the fast path and the `Zip` arm produce is pinned
    // against it, bit for bit.
    // -----------------------------------------------------------------------

    fn ndarray_reference<T: NdScalar>(parts: &[&Buffer], axis: usize) -> Vec<T> {
        let views: Vec<ndarray::ArrayViewD<'_, T>> =
            parts.iter().map(|part| part.as_array::<T>().unwrap()).collect();
        let joined = ndarray::concatenate(ndarray::Axis(axis), &views).unwrap();
        let standard = joined.as_standard_layout().into_owned();
        standard.into_raw_vec_and_offset().0
    }

    /// Bit-exact element equality: a copy must preserve a NaN payload and a
    /// negative zero, so a float comparison by value would be too weak.
    trait SameBits {
        fn same_bits(&self, other: &Self) -> bool;
    }

    macro_rules! same_bits_by_value {
        ($($ty:ty),*) => {$(
            impl SameBits for $ty {
                fn same_bits(&self, other: &Self) -> bool {
                    self == other
                }
            }
        )*};
    }
    same_bits_by_value!(bool, i8, i16, i32, i64, u8, u16, u32, u64);

    macro_rules! same_bits_by_bits {
        ($($ty:ty),*) => {$(
            impl SameBits for $ty {
                fn same_bits(&self, other: &Self) -> bool {
                    self.to_bits() == other.to_bits()
                }
            }
        )*};
    }
    same_bits_by_bits!(half::f16, f32, f64);

    impl SameBits for num_complex::Complex<f32> {
        fn same_bits(&self, other: &Self) -> bool {
            self.re.to_bits() == other.re.to_bits() && self.im.to_bits() == other.im.to_bits()
        }
    }

    impl SameBits for num_complex::Complex<f64> {
        fn same_bits(&self, other: &Self) -> bool {
            self.re.to_bits() == other.re.to_bits() && self.im.to_bits() == other.im.to_bits()
        }
    }

    fn numel(shape: &[usize]) -> usize {
        shape.iter().product()
    }

    /// Deterministic, everywhere-distinct operand values of any dtype (so a
    /// misplaced element can never hide behind a repeat).
    fn ramp<T: Scalar>(count: usize, seed: i64) -> Vec<T> {
        (0..count)
            .map(|index| T::from_scalar(crate::scalar::ScalarValue::I(seed + index as i64 * 3 - 5)))
            .collect()
    }

    /// Run one `concatenate` case and pin every element against
    /// [`ndarray_reference`].
    fn check_against_reference<T: Scalar + NdScalar + SameBits + std::fmt::Debug>(
        label: &str,
        shapes: &[&[usize]],
        axis: usize,
        operand: impl Fn(usize) -> Vec<T>,
    ) -> Buffer {
        let parts: Vec<Buffer> = shapes
            .iter()
            .enumerate()
            .map(|(position, shape)| {
                Buffer::new(
                    <T as NdScalar>::DTYPE,
                    shape.to_vec(),
                    <T as Scalar>::into_storage(operand(position)),
                )
                .expect("the fixture is well formed")
            })
            .collect();
        let borrowed: Vec<&Buffer> = parts.iter().collect();
        let mut shape = shapes[0].to_vec();
        shape[axis] = shapes.iter().map(|part| part[axis]).sum();
        let declared: Vec<Option<usize>> = shape.iter().map(|dim| Some(*dim)).collect();
        let dtype = <T as NdScalar>::DTYPE;
        let out = concatenate(&borrowed, Some(axis as i64), dtype, &declared).expect(label);
        assert_eq!(out.shape, shape, "{label}: result shape");
        assert_eq!(out.dtype, dtype, "{label}: result dtype");
        let reference = ndarray_reference::<T>(&borrowed, axis);
        let produced = T::slice_of(&out.storage).expect("the result has the dispatched dtype");
        assert_eq!(produced.len(), reference.len(), "{label}: element count");
        for (position, (got, want)) in produced.iter().zip(reference.iter()).enumerate() {
            assert!(
                got.same_bits(want),
                "{label}: element {position} is {got:?} but ndarray's assembly gives {want:?}"
            );
        }
        out
    }

    /// Every dtype the dispatch covers, over a leading, a middle and a trailing
    /// axis — with a WIDE run (one `copy_from_slice` per outer block) and a
    /// NARROW one (the `Zip` arm) on each.
    #[test]
    fn concatenate_matches_the_ndarray_reference_for_every_dtype() {
        type Case = (&'static [usize], &'static [usize], usize);
        let cases: [Case; 5] = [
            (&[3, 4, 5], &[2, 4, 5], 0),   // leading axis, run 60/40: bulk
            (&[3, 2, 2], &[3, 3, 2], 1),   // middle axis, run 4/6: narrow
            (&[2, 2, 5], &[2, 3, 5], 1),   // middle axis, run 10/15: bulk
            (&[4, 5, 1], &[4, 5, 3], 2),   // trailing axis, run 1/3: narrow
            (&[4, 5, 8], &[4, 5, 9], 2),   // trailing axis, run 8/9: bulk
        ];
        macro_rules! sweep {
            ($ty:ty) => {
                for (left, right, axis) in cases {
                    let label = format!(
                        "{} axis {axis} ({left:?} + {right:?})",
                        <$ty as NdScalar>::DTYPE.name()
                    );
                    check_against_reference::<$ty>(&label, &[left, right], axis, |position| {
                        ramp::<$ty>(numel([left, right][position]), 1 + position as i64 * 500)
                    });
                }
            };
        }
        sweep!(bool);
        sweep!(i8);
        sweep!(i16);
        sweep!(i32);
        sweep!(i64);
        sweep!(u8);
        sweep!(u16);
        sweep!(u32);
        sweep!(u64);
        sweep!(half::f16);
        sweep!(f32);
        sweep!(f64);
        sweep!(num_complex::Complex<f32>);
        sweep!(num_complex::Complex<f64>);
    }

    /// Ranks 1 and 2, every axis, with the float edge values a copy must carry
    /// verbatim (NaN payloads, `-0.0`, infinities).
    #[test]
    fn concatenate_matches_the_reference_across_ranks_and_float_edge_values() {
        let values = |count: usize, seed: f32| -> Vec<f32> {
            (0..count)
                .map(|index| match (index + seed as usize) % 7 {
                    0 => f32::NAN,
                    1 => -0.0,
                    2 => f32::INFINITY,
                    3 => f32::NEG_INFINITY,
                    other => (other + index) as f32 * -1.5,
                })
                .collect()
        };
        // Rank 1: narrow (run 3) and wide (run 10) single-axis joins.
        check_against_reference::<f32>("rank 1 narrow", &[&[3], &[5]], 0, |position| {
            values([3, 5][position], position as f32)
        });
        check_against_reference::<f32>("rank 1 wide", &[&[10], &[7]], 0, |position| {
            values([10, 7][position], position as f32)
        });
        // Rank 2, leading axis (narrow run 2) and trailing axis (wide run 9).
        check_against_reference::<f32>("rank 2 leading", &[&[2, 2], &[3, 2]], 0, |position| {
            values([4, 6][position], position as f32)
        });
        check_against_reference::<f32>("rank 2 trailing", &[&[2, 9], &[2, 4]], 1, |position| {
            values([18, 8][position], position as f32)
        });
        // Rank 3 with three parts, so the axis offsets accumulate.
        check_against_reference::<f32>(
            "rank 3 three parts",
            &[&[2, 1, 3], &[2, 4, 3], &[2, 2, 3]],
            1,
            |position| values([6, 24, 12][position], position as f32),
        );
    }

    /// numpy semantics for the empty family: a zero-length part contributes
    /// nothing, an all-empty axis yields an empty result, and a zero-length
    /// trailing dim leaves nothing to write at all.
    #[test]
    fn concatenate_handles_zero_length_parts_and_an_empty_axis() {
        let a = f32_buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let empty_rows = f32_buffer(&[0, 3], &[]);
        let empty_cols = f32_buffer(&[2, 0], &[]);

        // A zero-length part on the concatenated axis is a no-op.
        let out = concatenate(
            &[&a, &empty_rows, &a],
            Some(0),
            DType::Float32,
            &declared(&[4, 3]),
        )
        .unwrap();
        assert_eq!(
            out.storage,
            Storage::F32(vec![1., 2., 3., 4., 5., 6., 1., 2., 3., 4., 5., 6.])
        );

        // Every part empty on the axis: an empty result and no write at all.
        let out = concatenate(
            &[&empty_rows, &empty_rows],
            Some(0),
            DType::Float32,
            &declared(&[0, 3]),
        )
        .unwrap();
        assert_eq!(out.shape, vec![0, 3]);
        assert_eq!(out.storage, Storage::F32(Vec::new()));

        // A zero-length trailing dim on a last-axis join: the elements are zero
        // for every part, so the result is empty whatever the axis extents are.
        let out = concatenate(
            &[&empty_cols, &empty_cols],
            Some(0),
            DType::Float32,
            &declared(&[4, 0]),
        )
        .unwrap();
        assert_eq!(out.shape, vec![4, 0]);
        assert_eq!(out.storage, Storage::F32(Vec::new()));

        // The same empty family through the narrow (`Zip`) arm: parts of shape
        // (2,0) joined on axis 1 still leave nothing to write.
        let out = concatenate(
            &[&empty_cols, &empty_cols],
            Some(1),
            DType::Float32,
            &declared(&[2, 0]),
        )
        .unwrap();
        assert_eq!(out.shape, vec![2, 0]);
        assert_eq!(out.storage, Storage::F32(Vec::new()));
    }

    /// The destination is recycled from the thread-local pool, so it must be
    /// written end to end: a stock handed back by an earlier `concatenate` of
    /// the same size must not leak a single element into the next result.
    #[test]
    fn a_pooled_concatenate_result_is_fully_overwritten() {
        let shape = [500usize, 50, 1];
        let result_shape = declared(&[500, 50, 4]);
        // 4 x `shape` f32 = 400 KB, well above the pool's 4 KiB floor.
        let first: Vec<Buffer> = (0..4)
            .map(|part| f32_buffer(&shape, &vec![part as f32 + 1.0; 25_000]))
            .collect();
        let borrowed: Vec<&Buffer> = first.iter().collect();
        let earlier = concatenate(&borrowed, Some(2), DType::Float32, &result_shape).unwrap();
        let mut expected = Vec::with_capacity(100_000);
        for _row in 0..25_000 {
            for part in 1..=4 {
                expected.push(part as f32);
            }
        }
        assert_eq!(earlier.storage, Storage::F32(expected));
        // Dropping the result hands its 100 000-element stock to the pool; the
        // next join of the same size takes that exact stock back.
        drop(earlier);
        let second: Vec<Buffer> = (0..4)
            .map(|_| f32_buffer(&shape, &vec![9.5; 25_000]))
            .collect();
        let borrowed: Vec<&Buffer> = second.iter().collect();
        let reused = concatenate(&borrowed, Some(2), DType::Float32, &result_shape).unwrap();
        assert_eq!(reused.storage, Storage::F32(vec![9.5; 100_000]));
    }
}
