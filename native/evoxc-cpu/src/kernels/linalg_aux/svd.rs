// ---------------------------------------------------------------------------
// Singular value decomposition (one-sided Jacobi)
// ---------------------------------------------------------------------------

use evoxc_core::{Error, Result};
use ndarray::{s, Array1, Array2, ArrayView2, Zip};

use super::common::SvdParts;
use super::eigh::jacobi_rotation;
use super::SVD_MAX_SWEEPS;

/// Reduced SVD of an `m × n` matrix: `u (m, k)`, `s (k)`, `vh (k, n)`,
/// `k = min(m, n)`, singular values descending.
pub(super) fn svd_reduced(input: ArrayView2<f64>) -> Result<SvdParts> {
    let (m, n) = input.dim();
    let k = m.min(n);
    if k == 0 {
        return Ok((
            Array2::zeros((m, k)),
            Array1::zeros(k),
            Array2::zeros((k, n)),
        ));
    }
    if m < n {
        // Factor the transpose (`n × m`, tall) and swap the factors back:
        // `aᵀ = u' diag(s) vh'` ⇒ `a = vh'ᵀ diag(s) u'ᵀ`.
        let (u_tall, s, vh_tall) = svd_reduced(input.t())?;
        // `u = vh'ᵀ` (`m × k`), `vh = u'ᵀ` (`k × n`); both are views' transposes.
        let mut u = vh_tall.t().to_owned();
        let mut vh = u_tall.t().to_owned();
        orient_svd(&mut u, &mut vh);
        return Ok((u, s, vh));
    }
    // Tall (or square) `m >= n`: orthogonalize the columns of `work` pairwise.
    // Every rotation is also accumulated into `vectors`, so `a · vectors` is
    // the orthogonalized matrix (and `vectors` is the right-singular basis).
    let mut work = input.to_owned();
    let mut vectors = Array2::<f64>::eye(n);
    for _ in 0..SVD_MAX_SWEEPS {
        let mut rotated = false;
        for p in 0..n {
            for q in (p + 1)..n {
                let alpha = work.column(p).dot(&work.column(p));
                let beta = work.column(q).dot(&work.column(q));
                let gamma = work.column(p).dot(&work.column(q));
                if gamma == 0.0 || alpha == 0.0 || beta == 0.0 {
                    continue;
                }
                if gamma.abs() <= 1e-15 * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let (_, sin, cos) = jacobi_rotation(alpha, beta, gamma);
                {
                    let (mut left, mut right) = work.multi_slice_mut((s![.., p], s![.., q]));
                    Zip::from(&mut *left).and(&mut *right).for_each(|l, r| {
                        let (old_left, old_right) = (*l, *r);
                        *l = cos * old_left - sin * old_right;
                        *r = sin * old_left + cos * old_right;
                    });
                }
                {
                    let (mut left, mut right) =
                        vectors.multi_slice_mut((s![.., p], s![.., q]));
                    Zip::from(&mut *left).and(&mut *right).for_each(|l, r| {
                        let (old_left, old_right) = (*l, *r);
                        *l = cos * old_left - sin * old_right;
                        *r = sin * old_left + cos * old_right;
                    });
                }
            }
        }
        if !rotated {
            break;
        }
    }
    // Column norms are the singular values; the column directions are `u`.
    let singular: Array1<f64> = work
        .columns()
        .into_iter()
        .map(|column| column.dot(&column).sqrt())
        .collect();
    // A decomposition is only valid while the columns really are orthogonal;
    // this also rejects non-finite input (the comparison is false for NaN).
    let mut worst = 0.0f64;
    for p in 0..n {
        for q in (p + 1)..n {
            let denominator = singular[p] * singular[q];
            if denominator > 0.0 {
                let relative = work.column(p).dot(&work.column(q)).abs() / denominator;
                if relative > worst {
                    worst = relative;
                }
            }
        }
    }
    if worst.is_nan() || worst > 1e-8 {
        return Err(Error::backend(format!(
            "op 'svd': the one-sided Jacobi iteration did not converge (residual column correlation {worst:e}, m = {m}, n = {n}); the input may contain non-finite values"
        )));
    }
    let mut u = Array2::<f64>::zeros((m, n));
    for col in 0..n {
        let sigma = singular[col];
        if sigma > 0.0 {
            u.column_mut(col)
                .assign(&work.column(col).mapv(|value| value / sigma));
        } else if col < m {
            // A vanished column has no direction to recover; the matching unit
            // vector keeps `u` orthonormal for the all-zero matrix.
            u[(col, col)] = 1.0;
        }
    }
    // Descending singular values, with `u` columns and `vh` rows permuted.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|left, right| singular[*right].total_cmp(&singular[*left]));
    let mut u_sorted = Array2::<f64>::zeros((m, n));
    let mut s_sorted = Array1::<f64>::zeros(n);
    let mut vh_sorted = Array2::<f64>::zeros((n, n));
    for (slot, source) in order.iter().enumerate() {
        s_sorted[slot] = singular[*source];
        u_sorted.column_mut(slot).assign(&u.column(*source));
        // `vh = vectorsᵀ`: row `slot` is column `source` of `vectors`.
        vh_sorted.row_mut(slot).assign(&vectors.column(*source));
    }
    orient_svd(&mut u_sorted, &mut vh_sorted);
    Ok((u_sorted, s_sorted, vh_sorted))
}

/// Fix each `u` column's sign so its largest-magnitude entry is non-negative,
/// flipping the matching `vh` row so the product is unchanged.  `u` is `m × k`
/// and `vh` is `k × n`.
fn orient_svd(u: &mut Array2<f64>, vh: &mut Array2<f64>) {
    let (m, k) = u.dim();
    for col in 0..k {
        let mut pivot = 0usize;
        let mut magnitude = 0.0f64;
        for row in 0..m {
            let value = u[(row, col)].abs();
            if value > magnitude {
                magnitude = value;
                pivot = row;
            }
        }
        if u[(pivot, col)] < 0.0 {
            u.column_mut(col).mapv_inplace(|value| -value);
            vh.row_mut(col).mapv_inplace(|value| -value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::linalg_aux::test_support::{
        assert_columns_oriented, assert_matrix_close, assert_vector_close, diagonal, matmul,
        matrix, vector,
    };

    #[test]
    fn svd_two_by_two_reconstructs_with_descending_values() {
        let a = matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let (u, s, vh) = svd_reduced(a.view()).unwrap();
        assert_vector_close(
            &s,
            &vector(&[5.464_985_704_219_043, 0.365_966_190_626_257_5]),
            1e-9,
            "singular values (descending)",
        );
        assert_matrix_close(
            &matmul(&matmul(&u, &diagonal(&s)), &vh),
            &a,
            1e-12,
            "U diag(s) Vh",
        );
        assert_matrix_close(&matmul(&u.t(), &u), &Array2::<f64>::eye(2), 1e-12, "Uᵀ U");
        assert_matrix_close(&matmul(&vh, &vh.t()), &Array2::<f64>::eye(2), 1e-12, "Vh Vhᵀ");
        assert_columns_oriented(&u, "svd 2x2");
    }

    #[test]
    fn svd_wide_and_tall_shapes() {
        // 2x3: k = 2 → u (2,2), s (2), vh (2,3).
        let wide = matrix(2, 3, &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
        let (u, s, vh) = svd_reduced(wide.view()).unwrap();
        assert_eq!(
            (u.shape(), s.shape(), vh.shape()),
            (&[2, 2][..], &[2][..], &[2, 3][..])
        );
        assert_vector_close(&s, &vector(&[2.0, 1.0]), 1e-12, "singular values");
        assert_matrix_close(
            &matmul(&matmul(&u, &diagonal(&s)), &vh),
            &wide,
            1e-12,
            "U diag(s) Vh (wide)",
        );
        // 4x2: k = 2 → u (4,2), s (2), vh (2,2).
        let tall = matrix(4, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let (u, s, vh) = svd_reduced(tall.view()).unwrap();
        assert_eq!(
            (u.shape(), s.shape(), vh.shape()),
            (&[4, 2][..], &[2][..], &[2, 2][..])
        );
        assert_matrix_close(
            &matmul(&matmul(&u, &diagonal(&s)), &vh),
            &tall,
            1e-12,
            "U diag(s) Vh (tall)",
        );
        assert_columns_oriented(&u, "svd 4x2");
    }

    #[test]
    fn svd_of_a_zero_matrix_keeps_u_orthonormal() {
        let (u, s, vh) = svd_reduced(matrix(2, 2, &[0.0; 4]).view()).unwrap();
        assert_vector_close(&s, &vector(&[0.0, 0.0]), 0.0, "zero singular values");
        assert_matrix_close(&u, &Array2::<f64>::eye(2), 0.0, "U of the zero matrix");
        assert_matrix_close(&vh, &Array2::<f64>::eye(2), 0.0, "Vh of the zero matrix");
    }
}
