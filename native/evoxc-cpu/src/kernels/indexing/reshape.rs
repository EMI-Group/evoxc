//! `reshape` and `transpose` kernels.
//!
//! Both are shape-only moves on a C-contiguous buffer: `reshape` re-labels the
//! shape without touching the data and `transpose` walks the operand through
//! the permutation's strides. The module doc in [`super`] carries the shared
//! declared-shape and error-wording rules.
use evoxc_core::{DType, Error, Result};

use crate::buffer::{walk, Buffer};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::plan::OpSpec;
use crate::scalar::Scalar;
use crate::with_type;

use super::{operand, reconcile, render_ints, render_shape};

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

/// Replace the shape of an already C-contiguous buffer (a pure reinterpretation
/// — no data movement). The declared result shape is the target: inference has
/// already resolved a `-1` wildcard, so it must be fully static here.
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
    Buffer::new(source.dtype, target, source.storage.clone())
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
    // rank 2) moves no data: the result is the operand, re-labelled.
    if permutation.iter().enumerate().all(|(axis, target)| axis == *target) {
        return Buffer::new(source.dtype, shape, source.storage.clone());
    }
    let source_strides = util::c_strides(&source.shape);
    let strides: Vec<usize> = permutation
        .iter()
        .map(|axis| source_strides[*axis])
        .collect();
    with_type!(source, |values| transpose_typed(
        values,
        source.dtype,
        shape,
        strides
    ))
}

fn transpose_typed<T: Scalar>(
    values: &[T],
    dtype: DType,
    shape: Vec<usize>,
    strides: Vec<usize>,
) -> Result<Buffer> {
    let rank = shape.len();
    let mut out = Vec::with_capacity(values.len());
    walk(rank, &shape, &[strides], |offsets| out.push(values[offsets[0]]));
    Buffer::new(dtype, shape, T::into_storage(out))
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
        assert!(err.message.contains("cannot reshape array of size 6"), "{}", err.message);
    }

    #[test]
    fn reshape_rejects_a_runtime_dynamic_target_dim() {
        let x = f32_buffer(&[2, 3], &[0.; 6]);
        let err = reshape(&x, &[Some(3), None]).unwrap_err();
        assert!(err.message.contains("never guesses a dimension"), "{}", err.message);
    }

    // -- transpose --------------------------------------------------------

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
