// ---------------------------------------------------------------------------
// Symmetric eigendecomposition (cyclic Jacobi)
// ---------------------------------------------------------------------------

use evoxc_core::{Error, Result};
use ndarray::{Array1, Array2, ArrayView2};

use super::EIGH_MAX_SWEEPS;

/// Eigenvalues ascending + eigenvectors as columns, read from the LOWER
/// triangle of `input` (numpy `eigh`'s default).
pub(super) fn eigh_sym(input: ArrayView2<f64>) -> Result<(Array1<f64>, Array2<f64>)> {
    let n = input.nrows();
    if n == 0 {
        return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
    }
    // Mirror the lower triangle into the working matrix (upper ignored).
    let mut a = Array2::<f64>::zeros((n, n));
    for row in 0..n {
        for col in 0..=row {
            let value = input[(row, col)];
            a[(row, col)] = value;
            a[(col, row)] = value;
        }
    }
    let scale = a.iter().fold(0.0f64, |acc, value| acc.max(value.abs()));
    let threshold = 1e-15 * scale;
    let mut vectors = Array2::<f64>::eye(n);
    let mut converged = false;
    for _ in 0..EIGH_MAX_SWEEPS {
        let mut off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[(p, q)];
                off += apq * apq;
                if apq == 0.0 {
                    continue;
                }
                let app = a[(p, p)];
                let aqq = a[(q, q)];
                let (tan, sin, cos) = jacobi_rotation(app, aqq, apq);
                for k in 0..n {
                    if k == p || k == q {
                        continue;
                    }
                    let akp = a[(k, p)];
                    let akq = a[(k, q)];
                    let new_kp = cos * akp - sin * akq;
                    let new_kq = sin * akp + cos * akq;
                    a[(k, p)] = new_kp;
                    a[(p, k)] = new_kp;
                    a[(k, q)] = new_kq;
                    a[(q, k)] = new_kq;
                }
                a[(p, p)] = app - tan * apq;
                a[(q, q)] = aqq + tan * apq;
                a[(p, q)] = 0.0;
                a[(q, p)] = 0.0;
                for k in 0..n {
                    let vkp = vectors[(k, p)];
                    let vkq = vectors[(k, q)];
                    vectors[(k, p)] = cos * vkp - sin * vkq;
                    vectors[(k, q)] = sin * vkp + cos * vkq;
                }
            }
        }
        if off.sqrt() <= threshold {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(Error::backend(format!(
            "op 'eigh': the symmetric eigenvalue iteration did not converge after {EIGH_MAX_SWEEPS} sweeps (n = {n}); the input may contain non-finite values"
        )));
    }
    // Ascending eigenvalues, with the eigenvector columns permuted alongside.
    let diagonal = a.diag().to_owned();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|left, right| diagonal[*left].total_cmp(&diagonal[*right]));
    let mut eigenvalues = Array1::<f64>::zeros(n);
    let mut eigenvectors = Array2::<f64>::zeros((n, n));
    for (slot, source) in order.iter().enumerate() {
        eigenvalues[slot] = diagonal[*source];
        let mut column = vectors.column(*source).to_owned();
        orient_positive(&mut column);
        eigenvectors.column_mut(slot).assign(&column);
    }
    Ok((eigenvalues, eigenvectors))
}

/// The Jacobi rotation `(tan, sin, cos)` that zeroes `a_pq` of `[[app, apq],
/// [apq, aqq]]` while keeping it similar to the original block.
pub(super) fn jacobi_rotation(app: f64, aqq: f64, apq: f64) -> (f64, f64, f64) {
    let tau = (aqq - app) / (2.0 * apq);
    let t = if tau >= 0.0 {
        1.0 / (tau + (1.0 + tau * tau).sqrt())
    } else {
        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
    };
    let cos = 1.0 / (1.0 + t * t).sqrt();
    (t, t * cos, cos)
}

/// Flip `column` so its largest-magnitude entry is non-negative (ties keep the
/// earliest row's sign).  This is the eigenvector sign convention.
fn orient_positive(column: &mut Array1<f64>) {
    let mut pivot = 0usize;
    let mut magnitude = 0.0f64;
    for (row, value) in column.iter().enumerate() {
        if value.abs() > magnitude {
            magnitude = value.abs();
            pivot = row;
        }
    }
    if column.get(pivot).is_some_and(|value| *value < 0.0) {
        column.mapv_inplace(|value| -value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::linalg_aux::test_support::{
        assert_close, assert_columns_oriented, assert_matrix_close, assert_vector_close, diagonal, matmul,
        matrix, vector,
    };

    #[test]
    fn eigh_two_by_two_known_eigenvalues() {
        // [[2, 1], [1, 2]] has eigenvalues 1 and 3.
        let a = matrix(2, 2, &[2.0, 1.0, 1.0, 2.0]);
        let (w, v) = eigh_sym(a.view()).unwrap();
        assert_close(w[0], 1.0, 1e-12, "w[0]");
        assert_close(w[1], 3.0, 1e-12, "w[1]");
        assert_matrix_close(&matmul(&a, &v), &matmul(&v, &diagonal(&w)), 1e-12, "A V = V diag(w)");
        assert_matrix_close(&matmul(&v.t(), &v), &Array2::<f64>::eye(2), 1e-12, "Vᵀ V");
        assert_columns_oriented(&v, "eigh 2x2");
    }

    #[test]
    fn eigh_three_by_three_known_eigenvalues() {
        // A diagonal 3x3: eigenvalues 2, 3, 5 (ascending) and eigenvector matrix I.
        let a = matrix(3, 3, &[2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0]);
        let (w, v) = eigh_sym(a.view()).unwrap();
        assert_matrix_close(
            &diagonal(&w),
            &matrix(3, 3, &[2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0]),
            1e-12,
            "eigenvalues",
        );
        assert_matrix_close(&v, &Array2::<f64>::eye(3), 1e-12, "eigenvectors");
        // A non-diagonal 3x3 with eigenvalues 0, 2, 2.
        let b = matrix(3, 3, &[1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0]);
        let (w, v) = eigh_sym(b.view()).unwrap();
        assert_vector_close(&w, &vector(&[0.0, 2.0, 2.0]), 1e-12, "eigenvalues");
        assert_matrix_close(&matmul(&b, &v), &matmul(&v, &diagonal(&w)), 1e-12, "B V = V diag(w)");
        assert_columns_oriented(&v, "eigh 3x3");
    }

    #[test]
    fn eigh_reads_the_lower_triangle() {
        // The upper triangle (1e9) must be ignored, like numpy's UPLO='L'.
        let a = matrix(2, 2, &[4.0, 1e9, 2.0, 3.0]);
        let (w, v) = eigh_sym(a.view()).unwrap();
        let symmetric = matrix(2, 2, &[4.0, 2.0, 2.0, 3.0]);
        assert_matrix_close(
            &matmul(&symmetric, &v),
            &matmul(&v, &diagonal(&w)),
            1e-12,
            "A V = V diag(w) from the lower triangle only",
        );
    }
}
