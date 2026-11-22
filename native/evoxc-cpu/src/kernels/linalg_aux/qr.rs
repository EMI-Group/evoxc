// ---------------------------------------------------------------------------
// QR factorization (Householder, reduced mode)
// ---------------------------------------------------------------------------

use evoxc_core::Result;

/// Reduced QR of an `m × n` matrix: `q (m, k)`, `r (k, n)`, `k = min(m, n)`.
pub(super) fn qr_reduced(a: &[f64], m: usize, n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    let k = m.min(n);
    if k == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    // Work matrix; the reflections are applied in place.
    let mut work = a.to_vec();
    // One (column index, reflector vector) per Householder step; an all-zero
    // column yields an empty vector (no reflection).
    let mut reflectors: Vec<(usize, Vec<f64>)> = Vec::with_capacity(k);
    for j in 0..k {
        let mut norm = 0.0f64;
        for row in j..m {
            let value = work[row * n + j];
            norm += value * value;
        }
        let norm = norm.sqrt();
        if norm == 0.0 || !norm.is_finite() {
            reflectors.push((j, Vec::new()));
            continue;
        }
        let head = work[j * n + j];
        let alpha = if head >= 0.0 { -norm } else { norm };
        let mut reflector = vec![0.0f64; m - j];
        for row in j..m {
            reflector[row - j] = work[row * n + j];
        }
        reflector[0] -= alpha;
        let squared: f64 = reflector.iter().map(|value| value * value).sum();
        if squared == 0.0 {
            reflectors.push((j, Vec::new()));
            continue;
        }
        for col in j..n {
            let mut dot = 0.0f64;
            for row in j..m {
                dot += reflector[row - j] * work[row * n + col];
            }
            let factor = 2.0 * dot / squared;
            for row in j..m {
                work[row * n + col] -= factor * reflector[row - j];
            }
        }
        // The reflection maps the column onto `alpha * e_j`; write the exact
        // result so no rounding residue is left below the diagonal.
        work[j * n + j] = alpha;
        for row in (j + 1)..m {
            work[row * n + j] = 0.0;
        }
        reflectors.push((j, reflector));
    }
    let mut r = Vec::with_capacity(k * n);
    for row in 0..k {
        r.extend_from_slice(&work[row * n..row * n + n]);
    }
    // Q = H_0 H_1 ... H_{k-1} applied to the first k columns of the identity:
    // left-multiplying in REVERSE order accumulates the product in that order.
    let mut q = vec![0.0f64; m * k];
    for index in 0..k {
        q[index * k + index] = 1.0;
    }
    for (j, reflector) in reflectors.iter().rev() {
        if reflector.is_empty() {
            continue;
        }
        let squared: f64 = reflector.iter().map(|value| value * value).sum();
        for col in 0..k {
            let mut dot = 0.0f64;
            for row in *j..m {
                dot += reflector[row - j] * q[row * k + col];
            }
            let factor = 2.0 * dot / squared;
            for row in *j..m {
                q[row * k + col] -= factor * reflector[row - j];
            }
        }
    }
    Ok((q, r))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::linalg_aux::matrix_exp::identity;
    use crate::kernels::linalg_aux::test_support::{
        assert_close, assert_matrix_close, matmul, transpose,
    };

    #[test]
    fn qr_tall_is_orthogonal_and_reconstructs() {
        // 3x2, k = 2.
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (q, r) = qr_reduced(&a, 3, 2).unwrap();
        assert_eq!((q.len(), r.len()), (6, 4));
        assert_matrix_close(&matmul(&q, 3, 2, &r, 2), &a, 1e-12, "Q R = A");
        assert_matrix_close(
            &matmul(&transpose(&q, 3, 2), 2, 3, &q, 2),
            &identity(2),
            1e-12,
            "Qᵀ Q",
        );
        // R is upper triangular (row-major index 2 is R[1,0]).
        assert_close(r[2], 0.0, 0.0, "R[1,0]");
    }

    #[test]
    fn qr_wide_uses_min_dimension() {
        // 2x3, k = 2.
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (q, r) = qr_reduced(&a, 2, 3).unwrap();
        assert_eq!((q.len(), r.len()), (4, 6));
        assert_matrix_close(&matmul(&q, 2, 2, &r, 3), &a, 1e-12, "Q R = A");
        assert_matrix_close(
            &matmul(&transpose(&q, 2, 2), 2, 2, &q, 2),
            &identity(2),
            1e-12,
            "Qᵀ Q",
        );
        // R is 2x3 row-major: index 1*3 + 0 is R[1,0].
        assert_close(r[3], 0.0, 0.0, "R[1,0]");
    }
}
