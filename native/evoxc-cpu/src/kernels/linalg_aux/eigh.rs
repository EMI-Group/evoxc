// ---------------------------------------------------------------------------
// Symmetric eigendecomposition (cyclic Jacobi)
// ---------------------------------------------------------------------------

use evoxc_core::{Error, Result};

use super::matrix_exp::identity;
use super::EIGH_MAX_SWEEPS;

/// Eigenvalues ascending + eigenvectors as columns, read from the LOWER
/// triangle of `input` (numpy `eigh`'s default).
pub(super) fn eigh_sym(input: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let mut a = vec![0.0f64; n * n];
    for row in 0..n {
        for col in 0..=row {
            let value = input[row * n + col];
            a[row * n + col] = value;
            a[col * n + row] = value;
        }
    }
    let scale = a.iter().fold(0.0f64, |acc, value| acc.max(value.abs()));
    let threshold = 1e-15 * scale;
    let mut vectors = identity(n);
    let mut converged = false;
    for _ in 0..EIGH_MAX_SWEEPS {
        let mut off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                off += apq * apq;
                if apq == 0.0 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let (tan, sin, cos) = jacobi_rotation(app, aqq, apq);
                for k in 0..n {
                    if k == p || k == q {
                        continue;
                    }
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    let new_kp = cos * akp - sin * akq;
                    let new_kq = sin * akp + cos * akq;
                    a[k * n + p] = new_kp;
                    a[p * n + k] = new_kp;
                    a[k * n + q] = new_kq;
                    a[q * n + k] = new_kq;
                }
                a[p * n + p] = app - tan * apq;
                a[q * n + q] = aqq + tan * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;
                for k in 0..n {
                    let vkp = vectors[k * n + p];
                    let vkq = vectors[k * n + q];
                    vectors[k * n + p] = cos * vkp - sin * vkq;
                    vectors[k * n + q] = sin * vkp + cos * vkq;
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
    let diagonal: Vec<f64> = (0..n).map(|index| a[index * n + index]).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|left, right| diagonal[*left].total_cmp(&diagonal[*right]));
    let mut eigenvalues = Vec::with_capacity(n);
    let mut eigenvectors = vec![0.0f64; n * n];
    for (slot, source) in order.iter().enumerate() {
        eigenvalues.push(diagonal[*source]);
        let mut column: Vec<f64> = (0..n).map(|row| vectors[row * n + source]).collect();
        orient_positive(&mut column);
        for row in 0..n {
            eigenvectors[row * n + slot] = column[row];
        }
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
fn orient_positive(column: &mut [f64]) {
    let mut pivot = 0usize;
    let mut magnitude = 0.0f64;
    for (row, value) in column.iter().enumerate() {
        if value.abs() > magnitude {
            magnitude = value.abs();
            pivot = row;
        }
    }
    if column.get(pivot).is_some_and(|value| *value < 0.0) {
        for value in column.iter_mut() {
            *value = -*value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::linalg_aux::test_support::{
        assert_close, assert_columns_oriented, assert_matrix_close, diagonal, matmul, transpose,
    };

    #[test]
    fn eigh_two_by_two_known_eigenvalues() {
        // [[2, 1], [1, 2]] has eigenvalues 1 and 3.
        let a = [2.0, 1.0, 1.0, 2.0];
        let (w, v) = eigh_sym(&a, 2).unwrap();
        assert_close(w[0], 1.0, 1e-12, "w[0]");
        assert_close(w[1], 3.0, 1e-12, "w[1]");
        assert_matrix_close(
            &matmul(&a, 2, 2, &v, 2),
            &matmul(&v, 2, 2, &diagonal(&w), 2),
            1e-12,
            "A V = V diag(w)",
        );
        assert_matrix_close(
            &matmul(&transpose(&v, 2, 2), 2, 2, &v, 2),
            &identity(2),
            1e-12,
            "Vᵀ V",
        );
        assert_columns_oriented(&v, 2, 2, "eigh 2x2");
    }

    #[test]
    fn eigh_three_by_three_known_eigenvalues() {
        // A diagonal 3x3: eigenvalues 2, 3, 5 (ascending) and eigenvector matrix I.
        let a = [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0];
        let (w, v) = eigh_sym(&a, 3).unwrap();
        assert_matrix_close(&w, &[2.0, 3.0, 5.0], 1e-12, "eigenvalues");
        assert_matrix_close(&v, &identity(3), 1e-12, "eigenvectors");
        // A non-diagonal 3x3 with eigenvalues 0, 2, 2.
        let b = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0];
        let (w, v) = eigh_sym(&b, 3).unwrap();
        assert_matrix_close(&w, &[0.0, 2.0, 2.0], 1e-12, "eigenvalues");
        assert_matrix_close(
            &matmul(&b, 3, 3, &v, 3),
            &matmul(&v, 3, 3, &diagonal(&w), 3),
            1e-12,
            "B V = V diag(w)",
        );
        assert_columns_oriented(&v, 3, 3, "eigh 3x3");
    }

    #[test]
    fn eigh_reads_the_lower_triangle() {
        // The upper triangle (1e9) must be ignored, like numpy's UPLO='L'.
        let a = [4.0, 1e9, 2.0, 3.0];
        let (w, v) = eigh_sym(&a, 2).unwrap();
        let symmetric = [4.0, 2.0, 2.0, 3.0];
        assert_matrix_close(
            &matmul(&symmetric, 2, 2, &v, 2),
            &matmul(&v, 2, 2, &diagonal(&w), 2),
            1e-12,
            "A V = V diag(w) from the lower triangle only",
        );
    }
}
