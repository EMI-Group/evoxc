// ---------------------------------------------------------------------------
// Matrix exponential (scaling and squaring + Taylor series)
// ---------------------------------------------------------------------------

use evoxc_core::Result;
use ndarray::{Array2, Array3, Axis, Zip};

use super::MATRIX_EXP_TERMS;

/// Matrix exponential of a `(count, n, n)` batch, mirroring the numpy
/// reference: one uniform scaling step for the batch, 25 Taylor terms, then
/// repeated squaring.
pub(super) fn matrix_exp(input: &Array3<f64>) -> Result<Array3<f64>> {
    let (count, n, _) = input.dim();
    if n == 0 || count == 0 {
        return Ok(Array3::zeros((count, n, n)));
    }
    // 1-norm (max column sum of magnitudes) over the whole batch.
    let mut norm = 0.0f64;
    for block in input.outer_iter() {
        for col in 0..n {
            let sum: f64 = block.column(col).iter().map(|value| value.abs()).sum();
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
    let mut out = Array3::<f64>::zeros((count, n, n));
    for (index, block) in input.outer_iter().enumerate() {
        let scaled = block.mapv(|value| value * scale);
        let mut term = Array2::<f64>::eye(n);
        let mut result = term.clone();
        for k in 1..MATRIX_EXP_TERMS {
            term = mat_mul(&term, &scaled);
            let reciprocal = 1.0 / k as f64;
            term.mapv_inplace(|value| value * reciprocal);
            Zip::from(&mut result)
                .and(&term)
                .for_each(|slot, &value| *slot += value);
        }
        for _ in 0..squarings {
            result = mat_mul(&result, &result);
        }
        out.index_axis_mut(Axis(0), index).assign(&result);
    }
    Ok(out)
}

/// C-order product of two `n × n` matrices, accumulating `k` outermost just
/// like the hand-written triple loop it replaces (same summation order).
fn mat_mul(left: &Array2<f64>, right: &Array2<f64>) -> Array2<f64> {
    let n = left.nrows();
    let mut out = Array2::<f64>::zeros((n, n));
    for row in 0..n {
        for k in 0..n {
            let factor = left[(row, k)];
            Zip::from(out.row_mut(row))
                .and(right.row(k))
                .for_each(|slot, &value| *slot += factor * value);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::linalg_aux::test_support::{assert_matrix_close, matrix};

    /// Stack `blocks` of `(n, n)` values into the `(count, n, n)` operand.
    fn stacked(n: usize, blocks: &[f64]) -> Array3<f64> {
        let count = blocks.len() / (n * n);
        Array3::from_shape_vec((count, n, n), blocks.to_vec()).expect("a consistent stack")
    }

    #[test]
    fn matrix_exp_matches_hand_computed_values() {
        // exp(diag(0, ln 2)) = diag(1, 2).
        let out = matrix_exp(&stacked(2, &[0.0, 0.0, 0.0, 2.0f64.ln()])).unwrap();
        assert_matrix_close(
            &out.index_axis(Axis(0), 0).to_owned(),
            &matrix(2, 2, &[1.0, 0.0, 0.0, 2.0]),
            1e-12,
            "exp(diag)",
        );
        // exp([[0, 1], [1, 0]]) = [[cosh 1, sinh 1], [sinh 1, cosh 1]].
        let out = matrix_exp(&stacked(2, &[0.0, 1.0, 1.0, 0.0])).unwrap();
        let (cosh, sinh) = (1.0f64.cosh(), 1.0f64.sinh());
        assert_matrix_close(
            &out.index_axis(Axis(0), 0).to_owned(),
            &matrix(2, 2, &[cosh, sinh, sinh, cosh]),
            1e-12,
            "exp(swap)",
        );
        // exp(I) = e·I for a batched stack of two identical blocks.
        let out = matrix_exp(&stacked(2, &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])).unwrap();
        let e = 1.0f64.exp();
        assert_eq!(out.dim(), (2, 2, 2));
        assert_matrix_close(
            &out.index_axis(Axis(0), 0).to_owned(),
            &matrix(2, 2, &[e, 0.0, 0.0, e]),
            1e-12,
            "batched exp (block 0)",
        );
        assert_matrix_close(
            &out.index_axis(Axis(0), 1).to_owned(),
            &matrix(2, 2, &[e, 0.0, 0.0, e]),
            1e-12,
            "batched exp (block 1)",
        );
    }

    #[test]
    fn matrix_exp_scales_large_norms() {
        // A generator with 1-norm 4 still lands on a finite, accurate result:
        // exp([[0, 4], [-4, 0]]) = [[cos 4, sin 4], [-sin 4, cos 4]].
        let out = matrix_exp(&stacked(2, &[0.0, 4.0, -4.0, 0.0])).unwrap();
        let (cos, sin) = (4.0f64.cos(), 4.0f64.sin());
        assert_matrix_close(
            &out.index_axis(Axis(0), 0).to_owned(),
            &matrix(2, 2, &[cos, sin, -sin, cos]),
            1e-12,
            "rotation",
        );
    }
}
