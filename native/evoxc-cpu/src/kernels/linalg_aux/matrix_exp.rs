// ---------------------------------------------------------------------------
// Matrix exponential (scaling and squaring + Taylor series)
// ---------------------------------------------------------------------------

use evoxc_core::Result;

use super::MATRIX_EXP_TERMS;

/// Matrix exponential of `count` stacked `n × n` blocks, mirroring the numpy
/// reference: one uniform scaling step for the batch, 25 Taylor terms, then
/// repeated squaring.
pub(super) fn matrix_exp(input: &[f64], n: usize, count: usize) -> Result<Vec<f64>> {
    if n == 0 || count == 0 {
        return Ok(Vec::new());
    }
    // 1-norm (max column sum of magnitudes) over the whole batch.
    let mut norm = 0.0f64;
    for block in input.chunks_exact(n * n) {
        for col in 0..n {
            let mut sum = 0.0f64;
            for row in 0..n {
                sum += block[row * n + col].abs();
            }
            if sum > norm {
                norm = sum;
            }
        }
    }
    // Non-finite input keeps `norm` at its finite maximum, so no scaling happens
    // and the NaNs propagate through the series (numpy's behaviour) instead of
    // driving an unbounded squaring loop.
    let squarings: u32 = if norm.is_finite() && norm > 1.0 {
        norm.log2().ceil().clamp(0.0, 1024.0) as u32
    } else {
        0
    };
    let scale = 0.5f64.powi(squarings as i32);
    let mut out = Vec::with_capacity(count * n * n);
    for block in input.chunks_exact(n * n) {
        let scaled: Vec<f64> = block.iter().map(|value| value * scale).collect();
        let mut term = identity(n);
        let mut result = term.clone();
        for k in 1..MATRIX_EXP_TERMS {
            term = mat_mul(&term, &scaled, n);
            let reciprocal = 1.0 / k as f64;
            for value in term.iter_mut() {
                *value *= reciprocal;
            }
            for (slot, value) in result.iter_mut().zip(term.iter()) {
                *slot += *value;
            }
        }
        for _ in 0..squarings {
            result = mat_mul(&result, &result, n);
        }
        out.extend_from_slice(&result);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Small dense helpers
// ---------------------------------------------------------------------------

pub(super) fn identity(n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for index in 0..n {
        out[index * n + index] = 1.0;
    }
    out
}

/// C-order product of two `n × n` matrices.
fn mat_mul(left: &[f64], right: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for row in 0..n {
        for k in 0..n {
            let factor = left[row * n + k];
            for col in 0..n {
                out[row * n + col] += factor * right[k * n + col];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::linalg_aux::test_support::assert_matrix_close;

    #[test]
    fn matrix_exp_matches_hand_computed_values() {
        // exp(diag(0, ln 2)) = diag(1, 2).
        let out = matrix_exp(&[0.0, 0.0, 0.0, 2.0f64.ln()], 2, 1).unwrap();
        assert_matrix_close(&out, &[1.0, 0.0, 0.0, 2.0], 1e-12, "exp(diag)");
        // exp([[0, 1], [1, 0]]) = [[cosh 1, sinh 1], [sinh 1, cosh 1]].
        let out = matrix_exp(&[0.0, 1.0, 1.0, 0.0], 2, 1).unwrap();
        let (cosh, sinh) = (1.0f64.cosh(), 1.0f64.sinh());
        assert_matrix_close(&out, &[cosh, sinh, sinh, cosh], 1e-12, "exp(swap)");
        // exp(I) = e·I for a batched stack of two identical blocks.
        let blocks = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let out = matrix_exp(&blocks, 2, 2).unwrap();
        let e = 1.0f64.exp();
        assert_eq!(out.len(), 8);
        assert_matrix_close(
            &out[0..4],
            &[e, 0.0, 0.0, e],
            1e-12,
            "batched exp (block 0)",
        );
        assert_matrix_close(
            &out[4..8],
            &[e, 0.0, 0.0, e],
            1e-12,
            "batched exp (block 1)",
        );
    }

    #[test]
    fn matrix_exp_scales_large_norms() {
        // A generator with 1-norm 4 still lands on a finite, accurate result:
        // exp([[0, 4], [-4, 0]]) = [[cos 4, sin 4], [-sin 4, cos 4]].
        let out = matrix_exp(&[0.0, 4.0, -4.0, 0.0], 2, 1).unwrap();
        let (cos, sin) = (4.0f64.cos(), 4.0f64.sin());
        assert_matrix_close(&out, &[cos, sin, -sin, cos], 1e-12, "rotation");
    }
}
