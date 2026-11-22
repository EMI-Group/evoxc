// ---------------------------------------------------------------------------
// Singular value decomposition (one-sided Jacobi)
// ---------------------------------------------------------------------------

use evoxc_core::{Error, Result};

use super::common::SvdParts;
use super::eigh::jacobi_rotation;
use super::matrix_exp::identity;
use super::SVD_MAX_SWEEPS;

/// Reduced SVD of an `m × n` matrix: `u (m, k)`, `s (k)`, `vh (k, n)`,
/// `k = min(m, n)`, singular values descending.
pub(super) fn svd_reduced(input: &[f64], m: usize, n: usize) -> Result<SvdParts> {
    let k = m.min(n);
    if k == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    if m < n {
        // Factor the transpose (`n × m`, tall) and swap the factors back:
        // `aᵀ = u' diag(s) vh'` ⇒ `a = vh'ᵀ diag(s) u'ᵀ`.
        let mut transposed = vec![0.0f64; n * m];
        for row in 0..m {
            for col in 0..n {
                transposed[col * m + row] = input[row * n + col];
            }
        }
        let (u_tall, s, vh_tall) = svd_reduced(&transposed, n, m)?;
        // `u = vh'ᵀ` (`m × k`, and `k == m` here), `vh = u'ᵀ` (`k × n`).
        let mut u = vec![0.0f64; m * k];
        for row in 0..m {
            for col in 0..k {
                u[row * k + col] = vh_tall[col * m + row];
            }
        }
        let mut vh = vec![0.0f64; k * n];
        for row in 0..k {
            for col in 0..n {
                vh[row * n + col] = u_tall[col * k + row];
            }
        }
        orient_svd(&mut u, &mut vh, m, k, n);
        return Ok((u, s, vh));
    }
    // Tall (or square) `m >= n`: orthogonalize the columns of `work` pairwise.
    // Every rotation is also accumulated into `vectors`, so `a · vectors` is
    // the orthogonalized matrix (and `vectors` is the right-singular basis).
    let mut work = input.to_vec();
    let mut vectors = identity(n);
    for _ in 0..SVD_MAX_SWEEPS {
        let mut rotated = false;
        for p in 0..n {
            for q in (p + 1)..n {
                let mut alpha = 0.0f64;
                let mut beta = 0.0f64;
                let mut gamma = 0.0f64;
                for row in 0..m {
                    let left = work[row * n + p];
                    let right = work[row * n + q];
                    alpha += left * left;
                    beta += right * right;
                    gamma += left * right;
                }
                if gamma == 0.0 || alpha == 0.0 || beta == 0.0 {
                    continue;
                }
                if gamma.abs() <= 1e-15 * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let (_, sin, cos) = jacobi_rotation(alpha, beta, gamma);
                for row in 0..m {
                    let left = work[row * n + p];
                    let right = work[row * n + q];
                    work[row * n + p] = cos * left - sin * right;
                    work[row * n + q] = sin * left + cos * right;
                }
                for row in 0..n {
                    let left = vectors[row * n + p];
                    let right = vectors[row * n + q];
                    vectors[row * n + p] = cos * left - sin * right;
                    vectors[row * n + q] = sin * left + cos * right;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    // Column norms are the singular values; the column directions are `u`.
    let mut singular = vec![0.0f64; n];
    for col in 0..n {
        let mut squared = 0.0f64;
        for row in 0..m {
            squared += work[row * n + col] * work[row * n + col];
        }
        singular[col] = squared.sqrt();
    }
    // A decomposition is only valid while the columns really are orthogonal;
    // this also rejects non-finite input (the comparison is false for NaN).
    let mut worst = 0.0f64;
    for p in 0..n {
        for q in (p + 1)..n {
            let mut gamma = 0.0f64;
            for row in 0..m {
                gamma += work[row * n + p] * work[row * n + q];
            }
            let denominator = singular[p] * singular[q];
            if denominator > 0.0 {
                let relative = gamma.abs() / denominator;
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
    let mut u = vec![0.0f64; m * n];
    for col in 0..n {
        let sigma = singular[col];
        if sigma > 0.0 {
            for row in 0..m {
                u[row * n + col] = work[row * n + col] / sigma;
            }
        } else if col < m {
            // A vanished column has no direction to recover; the matching unit
            // vector keeps `u` orthonormal for the all-zero matrix.
            u[col * n + col] = 1.0;
        }
    }
    // Descending singular values, with `u` columns and `vh` rows permuted.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|left, right| singular[*right].total_cmp(&singular[*left]));
    let mut u_sorted = vec![0.0f64; m * n];
    let mut s_sorted = Vec::with_capacity(n);
    let mut vh_sorted = vec![0.0f64; n * n];
    for (slot, source) in order.iter().enumerate() {
        s_sorted.push(singular[*source]);
        for row in 0..m {
            u_sorted[row * n + slot] = u[row * n + source];
        }
        // `vh = vectorsᵀ`: row `slot` is column `source` of `vectors`.
        for col in 0..n {
            vh_sorted[slot * n + col] = vectors[col * n + source];
        }
    }
    orient_svd(&mut u_sorted, &mut vh_sorted, m, n, n);
    Ok((u_sorted, s_sorted, vh_sorted))
}

/// Fix each `u` column's sign so its largest-magnitude entry is non-negative,
/// flipping the matching `vh` row so the product is unchanged.  `u` is `m × k`
/// and `vh` is `k × n` (C-order).
fn orient_svd(u: &mut [f64], vh: &mut [f64], m: usize, k: usize, n: usize) {
    for col in 0..k {
        let mut pivot = 0usize;
        let mut magnitude = 0.0f64;
        for row in 0..m {
            let value = u[row * k + col].abs();
            if value > magnitude {
                magnitude = value;
                pivot = row;
            }
        }
        if u[pivot * k + col] < 0.0 {
            for row in 0..m {
                u[row * k + col] = -u[row * k + col];
            }
            for other in 0..n {
                vh[col * n + other] = -vh[col * n + other];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::linalg_aux::test_support::{
        assert_columns_oriented, assert_matrix_close, diagonal, matmul, transpose,
    };

    #[test]
    fn svd_two_by_two_reconstructs_with_descending_values() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let (u, s, vh) = svd_reduced(&a, 2, 2).unwrap();
        assert_matrix_close(
            &s,
            &[5.464_985_704_219_043, 0.365_966_190_626_257_5],
            1e-9,
            "singular values (descending)",
        );
        assert_matrix_close(
            &matmul(&matmul(&u, 2, 2, &diagonal(&s), 2), 2, 2, &vh, 2),
            &a,
            1e-12,
            "U diag(s) Vh",
        );
        assert_matrix_close(
            &matmul(&transpose(&u, 2, 2), 2, 2, &u, 2),
            &identity(2),
            1e-12,
            "Uᵀ U",
        );
        assert_matrix_close(
            &matmul(&vh, 2, 2, &transpose(&vh, 2, 2), 2),
            &identity(2),
            1e-12,
            "Vh Vhᵀ",
        );
        assert_columns_oriented(&u, 2, 2, "svd 2x2");
    }

    #[test]
    fn svd_wide_and_tall_shapes() {
        // 2x3: k = 2 → u (2,2), s (2), vh (2,3).
        let wide = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0];
        let (u, s, vh) = svd_reduced(&wide, 2, 3).unwrap();
        assert_eq!((u.len(), s.len(), vh.len()), (4, 2, 6));
        assert_matrix_close(&s, &[2.0, 1.0], 1e-12, "singular values");
        assert_matrix_close(
            &matmul(&matmul(&u, 2, 2, &diagonal(&s), 2), 2, 2, &vh, 3),
            &wide,
            1e-12,
            "U diag(s) Vh (wide)",
        );
        // 4x2: k = 2 → u (4,2), s (2), vh (2,2).
        let tall = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (u, s, vh) = svd_reduced(&tall, 4, 2).unwrap();
        assert_eq!((u.len(), s.len(), vh.len()), (8, 2, 4));
        assert_matrix_close(
            &matmul(&matmul(&u, 4, 2, &diagonal(&s), 2), 4, 2, &vh, 2),
            &tall,
            1e-12,
            "U diag(s) Vh (tall)",
        );
        assert_columns_oriented(&u, 4, 2, "svd 4x2");
    }

    #[test]
    fn svd_of_a_zero_matrix_keeps_u_orthonormal() {
        let (u, s, vh) = svd_reduced(&[0.0; 4], 2, 2).unwrap();
        assert_matrix_close(&s, &[0.0, 0.0], 0.0, "zero singular values");
        assert_matrix_close(&u, &identity(2), 0.0, "U of the zero matrix");
        assert_matrix_close(&vh, &identity(2), 0.0, "Vh of the zero matrix");
    }
}
