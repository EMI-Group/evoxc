//! `diag` — rank 1 becomes a square matrix with the vector on its main
//! diagonal, rank 2 becomes its main diagonal.

use evoxc_core::{Error, Result};

use crate::kernels::util;

use super::same_shape;

/// `diag`'s index map: rank 1 → the vector on the main diagonal of an `(n, n)`
/// matrix (zeros elsewhere), rank 2 → its main diagonal.
pub(super) fn diag_map(op: &str, in_shape: &[usize], out_shape: &[usize]) -> Result<Vec<Option<usize>>> {
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
            let mut map = vec![None; util::numel(out_shape)];
            for row in 0..extent {
                map[row * (extent + 1)] = Some(row);
            }
            Ok(map)
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
            Ok((0..length).map(|row| Some(row * (columns + 1))).collect())
        }
        other => Err(Error::shape(format!(
            "diag: input must be 1- or 2-d, got rank {other}"
        ))),
    }
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
}
