//! `diag` — rank 1 becomes a square matrix with the vector on its main
//! diagonal, rank 2 becomes its main diagonal.

use evoxc_core::{Error, Result};
use ndarray::Ix2;

use crate::buffer::{array_view, array_view_mut, NdScalar, Storage};
use crate::kernels::util;
use crate::scalar::Scalar;

use super::{pooled, same_shape};

/// `diag`'s materialization, from ndarray's own `diag`/`diag_mut` views:
///
/// * rank 1 → the vector is assigned onto the main diagonal of a zero-filled
///   `(n, n)` destination;
/// * rank 2 → the operand's main diagonal (ndarray's diagonal of a rectangle is
///   its shorter main diagonal, `min(rows, cols)`) is assigned into a pooled
///   1-D destination.
pub(super) fn diag_storage<T: Scalar + NdScalar>(
    values: &[T],
    in_shape: &[usize],
    out_shape: &[usize],
    op: &str,
) -> Result<Storage> {
    match in_shape.len() {
        1 => {
            let extent = in_shape[0];
            if !same_shape(out_shape, &[extent, extent]) {
                return Err(util::shape_error(
                    op,
                    "the diagonal matrix",
                    out_shape,
                    &[extent, extent],
                ));
            }
            let total = extent.checked_mul(extent).ok_or_else(|| {
                Error::shape(format!(
                    "op '{op}': a ({extent}, {extent}) diagonal matrix overflows the element count"
                ))
            })?;
            let mut out = pooled::<T>(total);
            {
                let mut dst = array_view_mut::<T>(&mut out, out_shape)?
                    .into_dimensionality::<Ix2>()
                    .map_err(|_| internal(op))?;
                dst.fill(T::zero());
                dst.diag_mut()
                    .assign(&array_view::<T>(values, in_shape)?);
            }
            Ok(<T as NdScalar>::into_storage(out))
        }
        2 => {
            let (rows, columns) = (in_shape[0], in_shape[1]);
            let length = rows.min(columns);
            if !same_shape(out_shape, &[length]) {
                return Err(util::shape_error(
                    op,
                    "the extracted diagonal",
                    out_shape,
                    &[length],
                ));
            }
            let src = array_view::<T>(values, in_shape)?
                .into_dimensionality::<Ix2>()
                .map_err(|_| internal(op))?;
            let mut out = pooled::<T>(length);
            {
                let mut dst = array_view_mut::<T>(&mut out, out_shape)?;
                dst.assign(&src.diag());
            }
            Ok(<T as NdScalar>::into_storage(out))
        }
        other => Err(Error::shape(format!(
            "diag: input must be 1- or 2-d, got rank {other}"
        ))),
    }
}

fn internal(op: &str) -> Error {
    Error::verification(format!(
        "op '{op}': the diagonal view does not compose (internal engine error)"
    ))
}

#[cfg(test)]
mod tests {
    use crate::kernels::structure::test_support::*;
    use crate::Storage;
    use evoxc_core::DType;

    #[test]
    fn diag_of_a_vector_is_a_square_matrix() {
        let input = i32_buffer(&[3], &[1, 2, 3]);
        let spec = op("diag", DType::Int32, &[3, 3], vec![]);
        let out = run(&spec, &input, &[3, 3]);
        assert_eq!(
            out.storage,
            Storage::I32(vec![1, 0, 0, 0, 2, 0, 0, 0, 3])
        );
        // An f32 vector: the off-diagonal cells are exact zeros, the diagonal
        // keeps every bit (including signed zero and NaN).
        let input = buffer(
            DType::Float32,
            &[2],
            Storage::F32(vec![-0.0, f32::NAN]),
        );
        let spec = op("diag", DType::Float32, &[2, 2], vec![]);
        let out = run(&spec, &input, &[2, 2]);
        match &out.storage {
            Storage::F32(got) => {
                assert_eq!(got.len(), 4);
                assert_eq!(got[0].to_bits(), (-0.0f32).to_bits());
                assert_eq!(got[1].to_bits(), 0.0f32.to_bits());
                assert_eq!(got[2].to_bits(), 0.0f32.to_bits());
                assert!(got[3].is_nan());
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }

    #[test]
    fn diag_of_a_matrix_is_its_main_diagonal() {
        let input = i32_buffer(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        let spec = op("diag", DType::Int32, &[2], vec![]);
        let out = run(&spec, &input, &[2]);
        assert_eq!(out.storage, Storage::I32(vec![0, 4]));
        // A tall rectangle still takes the shorter main diagonal.
        let input = i32_buffer(&[3, 2], &[0, 1, 2, 3, 4, 5]);
        let spec = op("diag", DType::Int32, &[2], vec![]);
        let out = run(&spec, &input, &[2]);
        assert_eq!(out.storage, Storage::I32(vec![0, 3]));
        // A 1-column matrix is its own first column.
        let input = i32_buffer(&[3, 1], &[7, 8, 9]);
        let spec = op("diag", DType::Int32, &[1], vec![]);
        assert_eq!(run(&spec, &input, &[1]).storage, Storage::I32(vec![7]));
    }

    #[test]
    fn diag_round_trips_a_vector_through_its_matrix() {
        // diag(diag(v)) is v again, for every element, through the same kernel.
        let values: Vec<i32> = vec![3, -1, 8, 0];
        let spec = op("diag", DType::Int32, &[4, 4], vec![]);
        let matrix = run(&spec, &i32_buffer(&[4], &values), &[4, 4]);
        let spec = op("diag", DType::Int32, &[4], vec![]);
        let back = run(&spec, &matrix, &[4]);
        assert_eq!(back.storage, Storage::I32(values));
    }

    #[test]
    fn diag_rejects_other_ranks() {
        let input = i32_buffer(&[2, 2, 2], &[0; 8]);
        let spec = op("diag", DType::Int32, &[2, 2], vec![]);
        let text = message(&spec, &input, &[2, 2]);
        assert!(text.contains("diag: input must be 1- or 2-d, got rank 3"), "{text}");
        let scalar = i32_buffer(&[], &[7]);
        let text = message(&spec, &scalar, &[]);
        assert!(text.contains("got rank 0"), "{text}");
    }

    #[test]
    fn diag_rejects_a_contradicting_result_shape() {
        let input = i32_buffer(&[3], &[1, 2, 3]);
        let spec = op("diag", DType::Int32, &[2, 2], vec![]);
        let text = message(&spec, &input, &[2, 2]);
        assert!(text.contains("diagonal matrix"), "{text}");
        let input = i32_buffer(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        let spec = op("diag", DType::Int32, &[3], vec![]);
        let text = message(&spec, &input, &[3]);
        assert!(text.contains("extracted diagonal"), "{text}");
    }
}
