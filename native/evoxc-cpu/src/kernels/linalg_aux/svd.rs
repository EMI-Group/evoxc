// ---------------------------------------------------------------------------
// Singular value decomposition (one-sided Jacobi)
// ---------------------------------------------------------------------------
//
// Conventions (fixed, so a repeated run is bit-identical and nothing depends on
// a library's internal sign choice):
//
// * reduced factors `u (m, k)`, `s (k)`, `vh (k, n)` with `k = min(m, n)`;
// * singular values DESCENDING;
// * each `u` column's sign fixed so its largest-magnitude entry is
//   non-negative, the matching `vh` row flipping with it;
// * a wide matrix (`m < n`) is factored through its transpose and the factors
//   are swapped back;
// * a column whose singular value vanished exactly gets the unit vector
//   `u[(col, col)] = 1` (the zero matrix yields `u = I`, `vh = I`, `s = 0`);
// * the factorization is validated by the residual column correlation (above
//   `1e-8`, or NaN, it raises the frozen non-convergence diagnostic); a
//   non-finite input is NOT caught by that check and propagates as NaN factors.
//
// A `faer` 0.24 (`linalg::solvers::Svd::new_thin`) replacement was evaluated and
// KEPT OUT.  faer's singular values already arrive in nonincreasing order and
// it accepts `m < n` by transposing internally, so only the sign/zero-column
// conventions above had to be re-applied; the measured `du`/`ds`/`dvh` are:
//
// | corpus                                   | du     | ds     | dvh    |
// |------------------------------------------|--------|--------|--------|
// | 1x1                                      | 0      | 0      | 0      |
// | 4x4 well-conditioned                     | 3.3e-16| 7.1e-15| 3.9e-16|
// | tall 6x4                                 | 1.4e-15| 6.7e-16| 1.8e-15|
// | wide 4x6                                 | 1.4e-15| 3.1e-15| 1.7e-15|
// | Hilbert 5x5 (ill-conditioned)            | 1.2e-13| 8.9e-16| 5.9e-14|
// | 4x4, singular-value pair gap 5e-4        | 2.6e-14| 1.6e-15| 2.7e-14|
// | 4x4 rank deficient (two equal columns)   | 8.5e-1 | 3.6e-15| 1.4e+0 |
//
// Every full-rank case agrees elementwise, signs and order included, well inside
// 1e-10, and `matrix_rank` (default tolerance) agrees.  The rank-deficient case
// does NOT: for an exactly-vanished singular value this kernel emits the unit
// vector `u[(col, col)] = 1` (leaving `|UᵀU − I|` at 8.6e-1 — not orthonormal),
// while faer returns the genuine left-null-space vector (`|UᵀU − I|` 5.6e-16);
// the spectrum and the rank (3 on both) still agree.  NaN input diverges too:
// this kernel returns NaN factors (the residual-correlation check never fires
// for them) where faer returns `SvdError::NoConvergence`, which would turn a NaN
// `svd`/`matrix_rank` input into an `Err` — a contract change.  So production
// stays as it is; `tests::faer_svd_matches_the_reference_including_signs`
// (ignored) reproduces these numbers, and `tests::bench_faer_vs_reference_svd`
// (ignored) measures the cost forgone: 64x64 0.99 vs 5.97 ms (6.1x), 256x256
// 31.1 vs 972.5 ms (31x), 512x512 103.6 vs 11499 ms (111x); 1024 was skipped
// for time (the reference alone extrapolates to ~90 s per factorization).

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
        return Err(non_convergence(worst, m, n));
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

/// The one fixed non-convergence diagnostic of `svd` (also raised for a
/// non-finite input, whose factors cannot be validated).
fn non_convergence(worst: f64, m: usize, n: usize) -> Error {
    Error::backend(format!(
        "op 'svd': the one-sided Jacobi iteration did not converge (residual column correlation {worst:e}, m = {m}, n = {n}); the input may contain non-finite values"
    ))
}

// ---------------------------------------------------------------------------
// The also-evaluated faer path (evidence only, never production)
// ---------------------------------------------------------------------------

/// faer 0.24 thin bidiagonal SVD behind the same conventions, so the evidence
/// test can compare the two factorizations directly.
#[cfg(test)]
fn svd_via_faer(input: ArrayView2<f64>) -> Result<SvdParts> {
    let (m, n) = input.dim();
    let k = m.min(n);
    if k == 0 {
        return Ok((
            Array2::zeros((m, k)),
            Array1::zeros(k),
            Array2::zeros((k, n)),
        ));
    }
    let source = faer::Mat::from_fn(m, n, |row, col| input[(row, col)]);
    // faer's singular values are already nonincreasing; the reorder below is
    // kept so the contract does not rest on that (and it is a no-op then).
    let factors = faer::linalg::solvers::Svd::new_thin(source.as_ref())
        .map_err(|_| non_convergence(f64::NAN, m, n))?;
    let u_faer = factors.U();
    let v_faer = factors.V();
    let s_faer = factors.S();
    let mut u = Array2::<f64>::zeros((m, k));
    let mut s = Array1::<f64>::zeros(k);
    let mut vh = Array2::<f64>::zeros((k, n));
    for col in 0..k {
        s[col] = s_faer[col];
        for row in 0..m {
            u[(row, col)] = u_faer[(row, col)];
        }
        // faer returns `V` (`n × k`), the columns of which are the rows of `vh`.
        for index in 0..n {
            vh[(col, index)] = v_faer[(index, col)];
        }
    }
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|left, right| s[*right].total_cmp(&s[*left]));
    let mut u_sorted = Array2::<f64>::zeros((m, k));
    let mut s_sorted = Array1::<f64>::zeros(k);
    let mut vh_sorted = Array2::<f64>::zeros((k, n));
    for (slot, source_col) in order.iter().enumerate() {
        s_sorted[slot] = s[*source_col];
        u_sorted.column_mut(slot).assign(&u.column(*source_col));
        vh_sorted.row_mut(slot).assign(&vh.row(*source_col));
    }
    orient_svd(&mut u_sorted, &mut vh_sorted);
    Ok((u_sorted, s_sorted, vh_sorted))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::linalg_aux::test_support::{
        assert_columns_oriented, assert_matrix_close, assert_vector_close, diagonal, matmul,
        matrix, vector,
    };

    /// Deterministic 64-bit LCG (`u64` state, `Numerical Recipes` constants).
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        /// Uniform in `[-1, 1)`.
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64 / (1u64 << 52) as f64) - 1.0
        }
    }

    fn filled(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
        let mut rng = Lcg::new(seed);
        Array2::from_shape_fn((rows, cols), |_| rng.next())
    }

    fn hilbert(n: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, n), |(row, col)| {
            1.0 / (row as f64 + col as f64 + 1.0)
        })
    }

    /// The Givens rotation of the `(i, j)` plane through `theta`.
    fn givens(n: usize, i: usize, j: usize, theta: f64) -> Array2<f64> {
        let mut rotation = Array2::<f64>::eye(n);
        rotation[(i, i)] = theta.cos();
        rotation[(j, j)] = theta.cos();
        rotation[(i, j)] = -theta.sin();
        rotation[(j, i)] = theta.sin();
        rotation
    }

    /// `U diag(s) Vᵀ` with `U`/`V` products of Givens rotations (exactly
    /// orthogonal up to rounding), so the singular values ARE `s`.
    fn prescribed(s: &[f64], seed: u64) -> Array2<f64> {
        let n = s.len();
        let mut rng = Lcg::new(seed);
        let mut left = Array2::<f64>::eye(n);
        let mut right = Array2::<f64>::eye(n);
        for i in 0..n {
            for j in (i + 1)..n {
                left = matmul(&givens(n, i, j, rng.next()), &left);
                right = matmul(&givens(n, i, j, rng.next()), &right);
            }
        }
        let spectrum = Array1::from_vec(s.to_vec());
        matmul(&matmul(&left, &diagonal(&spectrum)), &right.t())
    }

    fn max_diff(left: &Array2<f64>, right: &Array2<f64>) -> f64 {
        left.iter()
            .zip(right.iter())
            .fold(0.0f64, |acc, (a, b)| acc.max((a - b).abs()))
    }

    fn max_diff_vector(left: &Array1<f64>, right: &Array1<f64>) -> f64 {
        left.iter()
            .zip(right.iter())
            .fold(0.0f64, |acc, (a, b)| acc.max((a - b).abs()))
    }

    /// `max |M Mᵀ − I|`: how far a factor's columns/rows are from orthonormal.
    fn orthonormality_error(factor: &Array2<f64>, transposed: bool) -> f64 {
        let k = if transposed {
            factor.nrows()
        } else {
            factor.ncols()
        };
        let product = if transposed {
            matmul(factor, &factor.t())
        } else {
            matmul(&factor.t(), factor)
        };
        if k == 0 {
            return 0.0;
        }
        let mut worst = 0.0f64;
        for row in 0..k {
            for col in 0..k {
                let expected = if row == col { 1.0 } else { 0.0 };
                worst = worst.max((product[(row, col)] - expected).abs());
            }
        }
        worst
    }

    /// Deterministic corpus of the shapes/conditionings this kernel must serve.
    fn corpus() -> Vec<(&'static str, Array2<f64>)> {
        vec![
            ("1x1", matrix(1, 1, &[3.5])),
            (
                "square4",
                matrix(
                    4,
                    4,
                    &[
                        4.0, 1.0, 2.0, 0.5, 1.0, 3.0, -1.0, 2.0, 2.0, -1.0, 5.0, 1.0, 0.5,
                        2.0, 1.0, 6.0,
                    ],
                ),
            ),
            ("tall6x4", filled(6, 4, 11)),
            ("wide4x6", filled(4, 6, 23)),
            // Two identical columns: the fourth singular value vanishes exactly.
            (
                "rank_deficient4",
                matrix(
                    4,
                    4,
                    &[
                        1.0, 1.0, 6.0, 0.0, 2.0, 2.0, 1.0, 1.0, 3.0, 3.0, 0.0, 1.0, 4.0, 4.0,
                        1.0, 0.0,
                    ],
                ),
            ),
            ("hilbert5", hilbert(5)),
            // A closely spaced pair (gap 5e-4, not a tie).
            ("close_pair4", prescribed(&[2.0, 1.0005, 1.0, 0.5], 97)),
        ]
    }

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

    /// Evidence for the `faer` evaluation recorded in this module's doc: the two
    /// factorizations agree (signs and order included) on every full-rank shape
    /// and on the ill-conditioned and closely-spaced spectra, and differ on the
    /// rank-deficient input, where this kernel's vanished-column unit vector is
    /// not faer's left-null-space vector.
    ///
    /// Run with
    /// `cargo test --manifest-path native/Cargo.toml -p evoxc-cpu --lib faer_svd_matches -- --ignored --nocapture`.
    #[ignore = "evidence for the faer-vs-reference decision; prints the measured differences"]
    #[test]
    fn faer_svd_matches_the_reference_including_signs() {
        let tolerance = 1e-10;
        let mut differences = Vec::new();
        for (name, a) in corpus() {
            let (u_ref, s_ref, vh_ref) =
                svd_reduced(a.view()).expect("the reference converges on the corpus");
            let (u_faer, s_faer, vh_faer) =
                svd_via_faer(a.view()).expect("faer converges on the corpus");
            let du = max_diff(&u_faer, &u_ref);
            let ds = max_diff_vector(&s_faer, &s_ref);
            let dvh = max_diff(&vh_faer, &vh_ref);
            // The result is a valid reduced SVD of `a` on both paths.
            let (m, n) = a.dim();
            let k = m.min(n);
            assert_matrix_close(
                &matmul(&matmul(&u_faer, &diagonal(&s_faer)), &vh_faer),
                &a,
                tolerance,
                &format!("{name}: faer U diag(s) Vh"),
            );
            assert_matrix_close(
                &matmul(&matmul(&u_ref, &diagonal(&s_ref)), &vh_ref),
                &a,
                tolerance,
                &format!("{name}: reference U diag(s) Vh"),
            );
            for col in 1..k {
                assert!(
                    s_faer[col - 1] >= s_faer[col],
                    "{name}: faer singular values are not descending"
                );
                assert!(
                    s_ref[col - 1] >= s_ref[col],
                    "{name}: reference singular values are not descending"
                );
            }
            let spectrum = ds <= tolerance;
            assert!(
                spectrum,
                "{name}: the singular-value spectra differ by {ds:e}"
            );
            let reference_rank = crate::kernels::linalg_aux::common::numerical_rank(
                s_ref.as_slice().expect("a standard-layout vector"),
                m,
                n,
                None,
                f64::EPSILON,
            );
            let faer_rank = crate::kernels::linalg_aux::common::numerical_rank(
                s_faer.as_slice().expect("a standard-layout vector"),
                m,
                n,
                None,
                f64::EPSILON,
            );
            assert_eq!(
                reference_rank, faer_rank,
                "{name}: matrix_rank disagrees ({reference_rank} vs {faer_rank})"
            );
            println!(
                "{name:>16}: du={du:e} ds={ds:e} dvh={dvh:e} rank={reference_rank} \
                 |UᵀU−I| ref={:.2e} faer={:.2e} |VhVhᵀ−I| ref={:.2e} faer={:.2e}",
                orthonormality_error(&u_ref, false),
                orthonormality_error(&u_faer, false),
                orthonormality_error(&vh_ref, true),
                orthonormality_error(&vh_faer, true),
            );
            if du > tolerance || dvh > tolerance {
                differences.push(name);
            }
        }
        // The documented outcome: only the exactly-vanished singular value's
        // basis differs, and matrix_rank is unaffected.
        assert_eq!(
            differences,
            vec!["rank_deficient4"],
            "the faer/reference difference is not the documented one"
        );
        // NaN input: the reference returns NaN factors, faer reports
        // `SvdError::NoConvergence` (mapped onto the frozen diagnostic).
        let nan = matrix(2, 2, &[1.0, f64::NAN, 3.0, 4.0]);
        let (_, s_reference, _) = svd_reduced(nan.view()).expect("the reference does not raise");
        assert!(s_reference.iter().all(|value| value.is_nan()));
        let error = svd_via_faer(nan.view()).expect_err("faer reports non-convergence");
        assert_eq!(error.kind, super::non_convergence(f64::NAN, 2, 2).kind);
        assert_eq!(error.message, super::non_convergence(f64::NAN, 2, 2).message);
        println!(
            "NaN input: reference -> {:?}, faer -> {}",
            s_reference.to_vec(),
            error.message
        );
    }

    /// Timing aid for the same decision (run in release):
    /// `cargo test --release --manifest-path native/Cargo.toml -p evoxc-cpu \
    ///  bench_faer_vs_reference_svd -- --ignored --nocapture --test-threads=1`.
    #[ignore = "measurement aid: prints faer vs the hand-written Jacobi timing"]
    #[test]
    fn bench_faer_vs_reference_svd() {
        use std::hint::black_box;
        use std::time::Instant;

        let cases: [(usize, usize); 3] = [(64, 5), (256, 5), (512, 3)];
        println!("shape        faer (ms)  reference (ms)   ratio");
        for (size, runs) in cases {
            let a = filled(size, size, 7 + size as u64);
            // One warmup of each path, then the minimum of `runs` timed calls.
            black_box(svd_via_faer(a.view()).expect("faer converges"));
            black_box(svd_reduced(a.view()).expect("the reference converges"));
            let mut best_faer = f64::INFINITY;
            let mut best_reference = f64::INFINITY;
            for _ in 0..runs {
                let start = Instant::now();
                black_box(svd_via_faer(a.view()).expect("faer converges"));
                best_faer = best_faer.min(start.elapsed().as_secs_f64());
                let start = Instant::now();
                black_box(svd_reduced(a.view()).expect("the reference converges"));
                best_reference = best_reference.min(start.elapsed().as_secs_f64());
            }
            println!(
                "{size:>4}x{size:<4} {:>10.3} {:>14.3} {:>7.2}x",
                best_faer * 1e3,
                best_reference * 1e3,
                best_reference / best_faer
            );
        }
    }
}
