// ---------------------------------------------------------------------------
// Cholesky factorization
// ---------------------------------------------------------------------------

use evoxc_core::{Error, Result};
use ndarray::{Array2, ArrayView2};

/// Lower-triangular `L` with `A = L Lᵀ`, read from the LOWER triangle of `a`
/// (numpy `cholesky`'s default).  A matrix that is not positive definite is an
/// explicit error naming the op and the failing leading minor.
pub(super) fn cholesky_lower(a: ArrayView2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    let mut lower = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        let mut diagonal = a[(col, col)];
        for k in 0..col {
            diagonal -= lower[(col, k)] * lower[(col, k)];
        }
        if diagonal.is_nan() || diagonal <= 0.0 {
            return Err(Error::backend(format!(
                "op 'cholesky': matrix is not positive definite (leading minor of order {} is not positive)",
                col + 1
            )));
        }
        let root = diagonal.sqrt();
        lower[(col, col)] = root;
        for row in (col + 1)..n {
            let mut value = a[(row, col)];
            for k in 0..col {
                value -= lower[(row, k)] * lower[(col, k)];
            }
            lower[(row, col)] = value / root;
        }
    }
    Ok(lower)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::linalg_aux::test_support::{assert_matrix_close, matmul};

    #[test]
    fn cholesky_two_by_two_and_non_positive_definite() {
        let matrix = |values: [f64; 4]| Array2::from_shape_vec((2, 2), values.to_vec()).unwrap();
        // [[4, 2], [2, 3]] = L Lᵀ with L = [[2, 0], [1, sqrt(2)]].
        let a = [4.0, 2.0, 2.0, 3.0];
        let l = cholesky_lower(matrix(a).view()).unwrap();
        assert_matrix_close(
            &l,
            &Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 1.0, 2.0f64.sqrt()]).unwrap(),
            1e-12,
            "L",
        );
        assert_matrix_close(&matmul(&l, &l.t()), &matrix(a), 1e-12, "L Lᵀ");
        // The upper triangle is ignored.
        assert_eq!(
            cholesky_lower(matrix([4.0, -7.5, 2.0, 3.0]).view()).unwrap(),
            l
        );
        // [[1, 2], [2, 1]] has a negative eigenvalue ⇒ explicit error.
        let err = cholesky_lower(matrix([1.0, 2.0, 2.0, 1.0]).view()).unwrap_err();
        assert!(err.message.contains("cholesky"), "message: {}", err.message);
        assert!(
            err.message.contains("not positive definite"),
            "message: {}",
            err.message
        );
    }
}
