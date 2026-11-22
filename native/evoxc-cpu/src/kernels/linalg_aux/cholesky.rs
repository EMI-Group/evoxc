// ---------------------------------------------------------------------------
// Cholesky factorization
// ---------------------------------------------------------------------------

use evoxc_core::{Error, Result};

/// Lower-triangular `L` with `A = L Lᵀ`, read from the LOWER triangle of `a`
/// (numpy `cholesky`'s default).  A matrix that is not positive definite is an
/// explicit error naming the op and the failing leading minor.
pub(super) fn cholesky_lower(a: &[f64], n: usize) -> Result<Vec<f64>> {
    let mut lower = vec![0.0f64; n * n];
    for col in 0..n {
        let mut diagonal = a[col * n + col];
        for k in 0..col {
            diagonal -= lower[col * n + k] * lower[col * n + k];
        }
        if diagonal.is_nan() || diagonal <= 0.0 {
            return Err(Error::backend(format!(
                "op 'cholesky': matrix is not positive definite (leading minor of order {} is not positive)",
                col + 1
            )));
        }
        let root = diagonal.sqrt();
        lower[col * n + col] = root;
        for row in (col + 1)..n {
            let mut value = a[row * n + col];
            for k in 0..col {
                value -= lower[row * n + k] * lower[col * n + k];
            }
            lower[row * n + col] = value / root;
        }
    }
    Ok(lower)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::linalg_aux::test_support::{assert_matrix_close, matmul, transpose};

    #[test]
    fn cholesky_two_by_two_and_non_positive_definite() {
        // [[4, 2], [2, 3]] = L Lᵀ with L = [[2, 0], [1, sqrt(2)]].
        let a = [4.0, 2.0, 2.0, 3.0];
        let l = cholesky_lower(&a, 2).unwrap();
        assert_matrix_close(&l, &[2.0, 0.0, 1.0, 2.0f64.sqrt()], 1e-12, "L");
        assert_matrix_close(
            &matmul(&l, 2, 2, &transpose(&l, 2, 2), 2),
            &a,
            1e-12,
            "L Lᵀ",
        );
        // The upper triangle is ignored.
        assert_eq!(cholesky_lower(&[4.0, -7.5, 2.0, 3.0], 2).unwrap(), l);
        // [[1, 2], [2, 1]] has a negative eigenvalue ⇒ explicit error.
        let err = cholesky_lower(&[1.0, 2.0, 2.0, 1.0], 2).unwrap_err();
        assert!(err.message.contains("cholesky"), "message: {}", err.message);
        assert!(
            err.message.contains("not positive definite"),
            "message: {}",
            err.message
        );
    }
}
