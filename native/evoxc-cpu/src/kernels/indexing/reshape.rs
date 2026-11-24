//! `reshape` and `transpose` kernels.
//!
//! Both are shape-only moves on a C-contiguous buffer: `reshape` re-labels the
//! shape without touching the data and `transpose` walks the operand through
//! the permutation's strides. The module doc in [`super`] carries the shared
//! declared-shape and error-wording rules.
use evoxc_core::{DType, Error, Result};

use crate::buffer::{array_view, array_view_mut, block_storage_filled, Buffer, NdScalar};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::plan::OpSpec;
use crate::scalar::Scalar;
use crate::with_type;

use super::strided_copy;
use super::{operand, pooled_total, reconcile, render_ints, render_shape};

// ---------------------------------------------------------------------------
// reshape
// ---------------------------------------------------------------------------

pub(super) fn reshape_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    let source = operand("reshape", operands, 0)?;
    let declared = ctx.result_shape(spec, 0)?;
    single(reshape(source, &declared)?)
}

/// Replace the shape of an already C-contiguous buffer. The relabel is exact —
/// the element ORDER is untouched — but the result still owns its own storage,
/// so the data is copied. The declared result shape is the target: inference has
/// already resolved a `-1` wildcard, so it must be fully static here.
///
/// The copy is ONE contiguous run through [`block_storage_filled`], which takes
/// a pooled allocation of exactly the result's length and overwrites all of it
/// (the same path `tile`/`slice`/`gather` results take): a fresh
/// `source.storage.clone()` would instead pay a brand-new page-faulted
/// `malloc`/`mmap` per call for every reshape temporary.
fn reshape(source: &Buffer, declared: &[Option<usize>]) -> Result<Buffer> {
    let mut target = Vec::with_capacity(declared.len());
    for dim in declared {
        match dim {
            Some(extent) => target.push(*extent),
            None => {
                return Err(Error::shape(
                    "reshape: target shape contains a None (runtime-dynamic) dim — the engine never guesses a dimension",
                ))
            }
        }
    }
    let total = source.numel();
    if util::numel(&target) != total {
        return Err(Error::shape(format!(
            "reshape: cannot reshape array of size {total} into shape {}",
            render_shape(&target)
        )));
    }
    // A reshape covers the whole result with one contiguous run, so the copy
    // needs no zero fill and reuses a pooled allocation of exactly `total`
    // elements (`inner = total`, one block at offset 0). `total == 0` is an
    // empty buffer of the right shape, never an error.
    let storage = with_type!(source, |values| block_storage_filled(
        values, &target, &[0], total, "reshape"
    ))?;
    Buffer::new(source.dtype, target, storage)
}

// ---------------------------------------------------------------------------
// transpose
// ---------------------------------------------------------------------------

pub(super) fn transpose_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    let source = operand("transpose", operands, 0)?;
    let permutation = transpose_permutation(spec, source.rank())?;
    let declared = ctx.result_shape(spec, 0)?;
    single(transpose(source, &permutation, &declared)?)
}

/// The op's permutation: `null` (or absent) reverses every axis; otherwise it
/// must be a bijection of `0..rank` (numpy's `axes don't match array` /
/// `repeated axis in transpose`).
fn transpose_permutation(spec: &OpSpec, rank: usize) -> Result<Vec<usize>> {
    let Some(axes) = spec.attrs.opt_ints("permutation")? else {
        return Ok((0..rank).rev().collect());
    };
    if axes.len() != rank {
        return Err(Error::shape(format!(
            "transpose: axes don't match array — permutation {} has {} entries for rank {rank}",
            render_ints(axes),
            axes.len()
        )));
    }
    let mut seen = vec![false; rank];
    let mut permutation = Vec::with_capacity(rank);
    for axis in axes {
        let normalized = util::normalize_axis(*axis, rank, "transpose")?;
        if seen[normalized] {
            return Err(Error::shape(format!(
                "transpose: repeated axis in transpose — permutation {}",
                render_ints(axes)
            )));
        }
        seen[normalized] = true;
        permutation.push(normalized);
    }
    Ok(permutation)
}

fn transpose(source: &Buffer, permutation: &[usize], declared: &[Option<usize>]) -> Result<Buffer> {
    let derived: Vec<usize> = permutation
        .iter()
        .map(|axis| source.shape[*axis])
        .collect();
    let shape = reconcile("transpose", declared, derived)?;
    // An identity permutation (which is what a `null` permutation means below
    // rank 2) moves no data: the result is the operand, re-labelled — copied as
    // one contiguous run through the pooled block builder, like `reshape`.
    if permutation.iter().enumerate().all(|(axis, target)| axis == *target) {
        let total = util::numel(&shape);
        let storage = with_type!(source, |values| block_storage_filled(
            values, &shape, &[0], total, "transpose"
        ))?;
        return Buffer::new(source.dtype, shape, storage);
    }
    with_type!(source, |values| transpose_typed(
        values,
        source.dtype,
        &source.shape,
        permutation,
        shape
    ))
}

/// Materialize the transposed view through ndarray.
///
/// `permuted_axes` expresses the axis order and yields a (usually strided) view;
/// the fresh C-contiguous destination then receives it through
/// [`super::strided_copy`]'s lane-grouped run copy (ndarray's `assign` for any
/// layout it declines), so no hand-written permutation walk steps an offset
/// odometer per element. A permutation that happens to leave the view
/// C-contiguous (a size-1 axis, say) copies with a single `copy_from_slice`.
fn transpose_typed<T: Scalar + NdScalar>(
    values: &[T],
    dtype: DType,
    source_shape: &[usize],
    permutation: &[usize],
    shape: Vec<usize>,
) -> Result<Buffer> {
    let view = array_view(values, source_shape)?.permuted_axes(ndarray::IxDyn(permutation));
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
    use crate::kernels::indexing::test_support::{attrs, declared, f32_buffer, spec};
    use crate::Storage;
    use evoxc_core::Attrs;

    // -- reshape ----------------------------------------------------------

    #[test]
    fn reshape_reinterprets_without_moving_data() {
        let x = f32_buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let out = reshape(&x, &declared(&[3, 2])).unwrap();
        assert_eq!(out.shape, vec![3, 2]);
        assert_eq!(out.storage, Storage::F32(vec![1., 2., 3., 4., 5., 6.]));
    }

    #[test]
    fn reshape_supports_the_rank_zero_case_both_ways() {
        let scalar = f32_buffer(&[], &[7.]);
        let vector = reshape(&scalar, &declared(&[1])).unwrap();
        assert_eq!(vector.shape, vec![1]);
        let back = reshape(&vector, &[]).unwrap();
        assert_eq!(back.shape, Vec::<usize>::new());
        assert_eq!(back.storage, Storage::F32(vec![7.]));
    }

    #[test]
    fn reshape_element_count_mismatch_is_an_error() {
        let x = f32_buffer(&[2, 3], &[0.; 6]);
        let err = reshape(&x, &declared(&[4, 2])).unwrap_err();
        // The wording is part of the engine's contract: numpy's message, byte
        // for byte, in the ShapeError family.
        assert_eq!(
            err.message,
            "reshape: cannot reshape array of size 6 into shape (4,2)"
        );
        assert_eq!(err.kind, evoxc_core::ErrorKind::Shape);
    }

    #[test]
    fn reshape_copies_a_large_result_out_of_a_recycled_pool_stock() {
        // 100_000 f32 = 400 KB, well above the pool's retention floor, so the
        // recycled stock parked below really is the allocation the result is
        // built in: every element must be overwritten by the copy.
        const COUNT: usize = 100_000;
        let garbage = f32_buffer(&[COUNT], &vec![f32::NAN; COUNT]);
        drop(garbage);
        let values: Vec<f32> = (0..COUNT).map(|index| index as f32 * 0.5).collect();
        let source = f32_buffer(&[500, 50, 4], &values);
        let out = reshape(&source, &declared(&[200, 500])).unwrap();
        assert_eq!(out.shape, vec![200, 500]);
        match &out.storage {
            Storage::F32(values) => {
                assert_eq!(values.len(), COUNT);
                for (index, value) in values.iter().enumerate() {
                    assert_eq!(*value, index as f32 * 0.5, "element {index}");
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }

    #[test]
    fn reshape_keeps_an_empty_buffer_empty() {
        let empty = f32_buffer(&[0, 3], &[]);
        let flat = reshape(&empty, &declared(&[0])).unwrap();
        assert_eq!(flat.shape, vec![0]);
        assert_eq!(flat.storage, Storage::F32(Vec::new()));
        // An empty source reshaped to a shape whose 0 sits elsewhere: still an
        // empty buffer of the right shape, never an error.
        let out = reshape(&empty, &declared(&[3, 0])).unwrap();
        assert_eq!(out.shape, vec![3, 0]);
        assert_eq!(out.storage, Storage::F32(Vec::new()));
        // Only a 0-element source can reach an empty shape: a non-empty one is
        // the ordinary element-count mismatch.
        let full = f32_buffer(&[4], &[1., 2., 3., 4.]);
        let none = reshape(&full, &declared(&[2, 0])).unwrap_err();
        assert!(none.message.contains("cannot reshape array of size 4"), "{}", none.message);
    }

    #[test]
    fn reshape_rejects_a_runtime_dynamic_target_dim() {
        let x = f32_buffer(&[2, 3], &[0.; 6]);
        let err = reshape(&x, &[Some(3), None]).unwrap_err();
        assert!(err.message.contains("never guesses a dimension"), "{}", err.message);
    }

    // -- transpose --------------------------------------------------------

    #[test]
    fn transpose_identity_permutation_copies_the_operand_in_order() {
        // The no-move path: an explicit identity permutation and (below rank 2)
        // a `null` one both keep the operand's order element for element.
        let x = f32_buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let identity = transpose(&x, &[0, 1], &declared(&[2, 3])).unwrap();
        assert_eq!(identity.shape, vec![2, 3]);
        assert_eq!(identity.storage, Storage::F32(vec![1., 2., 3., 4., 5., 6.]));

        let vector = f32_buffer(&[4], &[1.5, -0.0, 0.25, 3.5]);
        let permutation = transpose_permutation(&spec(Attrs::new()), 1).unwrap();
        assert_eq!(permutation, vec![0]);
        let out = transpose(&vector, &permutation, &declared(&[4])).unwrap();
        assert_eq!(out.shape, vec![4]);
        assert_eq!(out.storage, vector.storage);

        // A neutral-size reshaped identity transpose of a larger operand, so the
        // pooled block builder's contract (`offsets.len() * inner == total`) is
        // pinned on the no-move path too.
        let big = f32_buffer(&[50, 100], &(0..5000).map(|i| i as f32).collect::<Vec<_>>());
        let same = transpose(&big, &[0, 1], &declared(&[50, 100])).unwrap();
        assert_eq!(same.shape, vec![50, 100]);
        assert_eq!(same.storage, big.storage);
    }

    #[test]
    fn transpose_defaults_to_reversing_every_axis() {
        let x = f32_buffer(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let permutation = transpose_permutation(&spec(Attrs::new()), 2).unwrap();
        assert_eq!(permutation, vec![1, 0]);
        let out = transpose(&x, &permutation, &declared(&[3, 2])).unwrap();
        assert_eq!(out.shape, vec![3, 2]);
        assert_eq!(out.storage, Storage::F32(vec![1., 4., 2., 5., 3., 6.]));
    }

    #[test]
    fn transpose_applies_an_explicit_permutation() {
        let x = f32_buffer(&[2, 3, 4], &(0..24).map(|value| value as f32).collect::<Vec<_>>());
        let op = spec(attrs(&[(
            "permutation",
            evoxc_core::AttrValue::Ints(vec![2, 0, 1]),
        )]));
        let permutation = transpose_permutation(&op, 3).unwrap();
        assert_eq!(permutation, vec![2, 0, 1]);
        let out = transpose(&x, &permutation, &declared(&[4, 2, 3])).unwrap();
        assert_eq!(out.shape, vec![4, 2, 3]);
        // out[p2][p0][p1] == x[p0][p1][p2]; element (1, 2, 3) of x is index 23.
        assert_eq!(
            out.storage,
            Storage::F32(vec![
                0., 4., 8., 12., 16., 20., 1., 5., 9., 13., 17., 21., 2., 6., 10., 14., 18., 22.,
                3., 7., 11., 15., 19., 23.
            ])
        );
    }

    #[test]
    fn transpose_rejects_a_repeated_or_short_permutation() {
        let repeated = spec(attrs(&[(
            "permutation",
            evoxc_core::AttrValue::Ints(vec![0, 0]),
        )]));
        let err = transpose_permutation(&repeated, 2).unwrap_err();
        assert!(err.message.contains("repeated axis in transpose"), "{}", err.message);

        let short = spec(attrs(&[(
            "permutation",
            evoxc_core::AttrValue::Ints(vec![0, 1]),
        )]));
        let err = transpose_permutation(&short, 3).unwrap_err();
        assert!(err.message.contains("axes don't match array"), "{}", err.message);

        let out_of_range = spec(attrs(&[(
            "permutation",
            evoxc_core::AttrValue::Ints(vec![0, 2]),
        )]));
        let err = transpose_permutation(&out_of_range, 2).unwrap_err();
        assert!(err.message.contains("out of range"), "{}", err.message);
    }
}
