// ---------------------------------------------------------------------------
// QR factorization (Householder, reduced mode)
// ---------------------------------------------------------------------------

use evoxc_core::Result;
use ndarray::{s, Array1, Array2, ArrayView2, Zip};

/// Reduced QR of an `m × n` matrix: `q (m, k)`, `r (k, n)`, `k = min(m, n)`.
pub(super) fn qr_reduced(a: ArrayView2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
    let (m, n) = a.dim();
    let k = m.min(n);
    if k == 0 {
        return Ok((Array2::zeros((m, k)), Array2::zeros((k, n))));
    }
    // Work matrix; the reflections are applied in place.
    let mut work = a.to_owned();
    // One (column index, reflector vector) per Householder step; an all-zero
    // column yields an empty vector (no reflection).
    let mut reflectors: Vec<(usize, Array1<f64>)> = Vec::with_capacity(k);
    for j in 0..k {
        let norm = work.column(j).slice(s![j..]).dot(&work.column(j).slice(s![j..]));
        let norm = norm.sqrt();
        if norm == 0.0 || !norm.is_finite() {
            reflectors.push((j, Array1::zeros(0)));
            continue;
        }
        let head = work[(j, j)];
        let alpha = if head >= 0.0 { -norm } else { norm };
        let mut reflector = work.column(j).slice(s![j..]).to_owned();
        reflector[0] -= alpha;
        let squared: f64 = reflector.dot(&reflector);
        if squared == 0.0 {
            reflectors.push((j, Array1::zeros(0)));
            continue;
        }
        for col in j..n {
            let dot = work.column(col).slice(s![j..]).dot(&reflector);
            let factor = 2.0 * dot / squared;
            let mut column = work.column_mut(col);
            Zip::from(&mut column.slice_mut(s![j..]))
                .and(&reflector)
                .for_each(|value, &entry| *value -= factor * entry);
        }
        // The reflection maps the column onto `alpha * e_j`; write the exact
        // result so no rounding residue is left below the diagonal.
        work[(j, j)] = alpha;
        for row in (j + 1)..m {
            work[(row, j)] = 0.0;
        }
        reflectors.push((j, reflector));
    }
    let r = work.slice(s![0..k, 0..n]).to_owned();
    // Q = H_0 H_1 ... H_{k-1} applied to the first k columns of the identity:
    // left-multiplying in REVERSE order accumulates the product in that order.
    let mut q = Array2::<f64>::zeros((m, k));
    for index in 0..k {
        q[(index, index)] = 1.0;
    }
    for (j, reflector) in reflectors.iter().rev() {
        if reflector.is_empty() {
            continue;
        }
        let squared: f64 = reflector.dot(reflector);
        for col in 0..k {
            let dot = q.column(col).slice(s![*j..]).dot(reflector);
            let factor = 2.0 * dot / squared;
            let mut column = q.column_mut(col);
            Zip::from(&mut column.slice_mut(s![*j..]))
                .and(reflector)
                .for_each(|value, &entry| *value -= factor * entry);
        }
    }
    Ok((q, r))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::linalg_aux::test_support::{
        assert_close, assert_matrix_close, matmul, matrix,
    };

    #[test]
    fn qr_tall_is_orthogonal_and_reconstructs() {
        // 3x2, k = 2.
        let a = matrix(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (q, r) = qr_reduced(a.view()).unwrap();
        assert_eq!((q.shape(), r.shape()), (&[3, 2][..], &[2, 2][..]));
        assert_matrix_close(&matmul(&q, &r), &a, 1e-12, "Q R = A");
        assert_matrix_close(&matmul(&q.t(), &q), &Array2::<f64>::eye(2), 1e-12, "Qᵀ Q");
        // R is upper triangular.
        assert_close(r[(1, 0)], 0.0, 0.0, "R[1,0]");
    }

    #[test]
    fn qr_wide_uses_min_dimension() {
        // 2x3, k = 2.
        let a = matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (q, r) = qr_reduced(a.view()).unwrap();
        assert_eq!((q.shape(), r.shape()), (&[2, 2][..], &[2, 3][..]));
        assert_matrix_close(&matmul(&q, &r), &a, 1e-12, "Q R = A");
        assert_matrix_close(&matmul(&q.t(), &q), &Array2::<f64>::eye(2), 1e-12, "Qᵀ Q");
        // R is 2x3: the entry below the diagonal must vanish.
        assert_close(r[(1, 0)], 0.0, 0.0, "R[1,0]");
    }
}
