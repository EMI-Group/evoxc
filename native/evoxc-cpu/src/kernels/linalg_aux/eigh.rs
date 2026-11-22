// ---------------------------------------------------------------------------
// Symmetric eigendecomposition (cyclic Jacobi)
// ---------------------------------------------------------------------------
//
// PRODUCTION: the hand-written cyclic Jacobi below.  This module's `eigh` op is
// a contract, not a library call — lower triangle only (numpy's `UPLO='L'`),
// ascending eigenvalues, eigenvector COLUMNS sign-fixed so each column's
// largest-magnitude entry is non-negative (a tie keeps the earliest row's
// sign), empty arrays for `n == 0`, and the frozen `BackendError` below when
// the sweeps do not converge.
//
// `faer` 0.24 (`SelfAdjointEigen`) was evaluated as a replacement and is KEPT
// OUT of production: it reproduces this implementation exactly on every
// non-degenerate case measured (a deterministic corpus — 1x1, 2x2, diagonal
// 3x3, a `Q diag Qᵀ` 5x5 with a NEAR-degenerate pair `{1, 2, 2+1e-3, 5, 9}`,
// the ill-conditioned 5x5 Hilbert matrix, a fixed-seed 8x8, and garbage/NaN in
// the ignored upper triangle — agrees to better than `1e-10` elementwise on
// the eigenvalues AND the eigenvector columns including their signs, with
// faer's documented nondecreasing eigenvalue order), but an EXACT repeated
// eigenvalue leaves the basis of the degenerate subspace free, and there it
// does not: on `Q diag(2, 2, 5) Qᵀ` (a rotated twofold degeneracy) the two
// bases differ by `max |V_faer − V_reference| = 9.226e-2` after the shared
// orientation — first entry `(0, 0)`: 8.16496580927726145e-1 (faer) vs
// 8.11267256036937390e-1 (production) — while the unrotated `diag(2, 2, 5)`
// (subspace = the coordinate plane) agrees exactly (0.0) and the eigenvalues
// agree to about `1e-15` in both.  Both answers are valid eigenvector
// matrices, but a basis rotation is an output change outside tolerance, so the
// rule is KEEP — do not relax the tolerance to adopt it.  `eigh_sym_faer`
// (test-only) is the measured candidate;
// `faer_eigh_matches_the_reference_including_signs` pins the non-degenerate
// agreement, `exact_degenerate_eigenvectors_are_recorded` records the
// degenerate difference, and `faer_eigh_divergence_evidence` (`#[ignore]`d)
// prints the first mismatching entry.
//
// For the record, the speed the KEEP decision forgoes (release, fixed-seed
// symmetric operands, `min` of 5 runs at 64/256 and of 3 at 512 after one
// warmup, `bench_faer_vs_reference_eigh`): 64x64 `0.458 ms` vs `3.774 ms`
// (8.2x), 256x256 `10.100 ms` vs `1466.839 ms` (145x), 512x512 `46.069 ms` vs
// `14389.911 ms` (312x).  1024x1024 was skipped for time: the Jacobi leg
// extrapolates to roughly two minutes per run, so warmup plus three runs would
// not fit a couple of minutes.
//
// Non-finite behaviour, measured: the production sweeps zero an off-diagonal
// pair before re-summing, so a NaN in the lower triangle can "converge" in the
// second sweep and return `Ok` with NaN values (a 2x2 does exactly that);
// faer's QR path instead reports `NoConvergence` there, which `eigh_sym_faer`
// maps onto the same frozen message.  Nothing pins the NaN-`Ok` path (the
// Python suite never feeds `eigh` a non-finite operand).
//
// `jacobi_rotation` is shared with `svd.rs` (its one-sided Jacobi sweep) and is
// not test-only.

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
        return Err(non_convergent(n));
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

/// The frozen non-convergence diagnostic of the `eigh` op: its message and
/// family are part of the backend contract.
fn non_convergent(n: usize) -> Error {
    Error::backend(format!(
        "op 'eigh': the symmetric eigenvalue iteration did not converge after {EIGH_MAX_SWEEPS} sweeps (n = {n}); the input may contain non-finite values"
    ))
}

/// The evaluated `faer` candidate (NOT production — see the module header): the
/// same contract implemented over `SelfAdjointEigen::new(A, Side::Lower)`, with
/// faer's eigenvalues re-sorted ascending, the columns orientation-fixed, and
/// faer's `NoConvergence` (or any non-finite output) mapped onto the frozen
/// diagnostic.
#[cfg(test)]
fn eigh_sym_faer(input: ArrayView2<f64>) -> Result<(Array1<f64>, Array2<f64>)> {
    use faer::linalg::solvers::SelfAdjointEigen;
    use faer::{MatRef, Side};

    let n = input.nrows();
    if n == 0 {
        return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
    }
    // Mirror the lower triangle into a full symmetric matrix (upper ignored).
    let mut symmetric = Vec::with_capacity(n * n);
    for row in 0..n {
        for col in 0..n {
            symmetric.push(if row >= col {
                input[(row, col)]
            } else {
                input[(col, row)]
            });
        }
    }
    let matrix = MatRef::from_row_major_slice(&symmetric, n, n);
    let eigen = SelfAdjointEigen::new(matrix, Side::Lower).map_err(|_| non_convergent(n))?;
    let (u, s) = (eigen.U(), eigen.S());
    // faer documents nondecreasing `S`; the explicit sort keeps the ascending
    // promise independent of that.
    let diagonal: Vec<f64> = s.column_vector().iter().copied().collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|left, right| diagonal[*left].total_cmp(&diagonal[*right]));
    let mut eigenvalues = Array1::<f64>::zeros(n);
    let mut eigenvectors = Array2::<f64>::zeros((n, n));
    for (slot, source) in order.iter().enumerate() {
        eigenvalues[slot] = diagonal[*source];
        let mut column: Array1<f64> = u.col(*source).iter().copied().collect();
        orient_positive(&mut column);
        eigenvectors.column_mut(slot).assign(&column);
    }
    if eigenvalues
        .iter()
        .chain(eigenvectors.iter())
        .any(|value| !value.is_finite())
    {
        return Err(non_convergent(n));
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
        assert_close, assert_columns_oriented, assert_matrix_close, assert_vector_close, diagonal,
        matmul, matrix, vector,
    };

    /// The corpus tolerance: `f64` and both algorithms converge quadratically,
    /// so a finite, non-degenerate case agrees far below this.
    const TOLERANCE: f64 = 1e-10;

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

    // -----------------------------------------------------------------------
    // The differential corpus: the `faer` candidate vs production
    // -----------------------------------------------------------------------

    /// Deterministic modified Gram–Schmidt over fixed seed columns: the corpus'
    /// orthogonal `Q` carries no external crate and no RNG state.
    fn orthonormal_columns(seeds: &[&[f64]]) -> Array2<f64> {
        let n = seeds.len();
        let mut basis: Vec<Vec<f64>> = Vec::with_capacity(n);
        for seed in seeds {
            let mut column = seed.to_vec();
            for previous in &basis {
                let projection: f64 =
                    column.iter().zip(previous.iter()).map(|(a, b)| a * b).sum();
                for (value, base) in column.iter_mut().zip(previous.iter()) {
                    *value -= projection * base;
                }
            }
            let norm = column.iter().map(|value| value * value).sum::<f64>().sqrt();
            assert!(norm > 1e-6, "the fixed seed columns must be independent");
            for value in column.iter_mut() {
                *value /= norm;
            }
            basis.push(column);
        }
        let mut q = Array2::<f64>::zeros((n, n));
        for (col, column) in basis.iter().enumerate() {
            for (row, value) in column.iter().enumerate() {
                q[(row, col)] = *value;
            }
        }
        q
    }

    /// `Q diag(values) Qᵀ`: a symmetric matrix with an exactly known spectrum.
    fn from_spectrum(q: &Array2<f64>, values: &[f64]) -> Array2<f64> {
        matmul(&matmul(q, &diagonal(&vector(values))), &q.t())
    }

    /// The 5x5 Hilbert matrix (symmetric by construction), eigenvalues
    /// `~1.6e-5 … 1.57`: the ill-conditioned case.
    fn hilbert(n: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, n), |(row, col)| 1.0 / (row + col + 1) as f64)
    }

    /// The operand as the op reads it: the lower triangle mirrored (the upper
    /// triangle never reaches the algorithm).
    fn symmetrized_from_lower(a: &Array2<f64>) -> Array2<f64> {
        let n = a.nrows();
        Array2::from_shape_fn((n, n), |(row, col)| a[(row.max(col), row.min(col))])
    }

    /// A fixed-seed (64-bit LCG) symmetric matrix, the bench and the wide
    /// corpus case: no external crate, no clock, no entropy.
    fn pseudo_random_symmetric(n: usize, seed: u64) -> Array2<f64> {
        let mut state = seed | 1;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // `[0, 1)` from the top 53 bits, shifted to `[-0.5, 0.5)`.
            ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        let mut a = Array2::<f64>::zeros((n, n));
        for row in 0..n {
            for col in 0..=row {
                let value = next();
                a[(row, col)] = value;
                a[(col, row)] = value;
            }
        }
        for row in 0..n {
            a[(row, row)] += n as f64;
        }
        a
    }

    /// One decomposition: ascending eigenvalues plus the eigenvector columns.
    type Decomposition = (Array1<f64>, Array2<f64>);

    /// Both decompositions of one operand: the candidate first, production
    /// second.
    fn both(a: &Array2<f64>) -> (Decomposition, Decomposition) {
        (
            eigh_sym_faer(a.view()).expect("the faer candidate"),
            eigh_sym(a.view()).expect("production"),
        )
    }

    /// The elementwise comparison the contract needs: ascending eigenvalues,
    /// eigenvector columns **including their signs**, plus the two invariants
    /// against the matrix the op actually reads (lower triangle mirrored).
    fn assert_same(a: &Array2<f64>, what: &str) {
        let effective = symmetrized_from_lower(a);
        let ((w_candidate, v_candidate), (w_production, v_production)) = both(a);
        assert_vector_close(&w_candidate, &w_production, TOLERANCE, &format!("{what}: eigenvalues"));
        assert_matrix_close(
            &v_candidate,
            &v_production,
            TOLERANCE,
            &format!("{what}: eigenvectors including signs"),
        );
        for slot in 1..w_candidate.len() {
            assert!(
                w_candidate[slot] >= w_candidate[slot - 1],
                "{what}: eigenvalues are not ascending"
            );
        }
        assert_matrix_close(
            &matmul(&effective, &v_candidate),
            &matmul(&v_candidate, &diagonal(&w_candidate)),
            TOLERANCE,
            &format!("{what}: A V = V diag(w)"),
        );
        assert_matrix_close(
            &matmul(&v_candidate.t(), &v_candidate),
            &Array2::<f64>::eye(a.nrows()),
            TOLERANCE,
            &format!("{what}: VᵀV = I"),
        );
        assert_columns_oriented(&v_candidate, what);
    }

    #[test]
    fn faer_eigh_matches_the_reference_including_signs() {
        // n == 0: both return empty arrays.
        let empty = Array2::<f64>::zeros((0, 0));
        let (w_candidate, v_candidate) = eigh_sym_faer(empty.view()).unwrap();
        let (w_production, v_production) = eigh_sym(empty.view()).unwrap();
        assert_eq!(w_candidate.len(), 0);
        assert_eq!(v_candidate.shape(), &[0, 0]);
        assert_eq!(w_production.len(), 0);
        assert_eq!(v_production.shape(), &[0, 0]);

        assert_same(&matrix(1, 1, &[3.5]), "1x1");
        assert_same(&matrix(2, 2, &[2.0, 1.0, 1.0, 2.0]), "2x2 [[2,1],[1,2]]");
        assert_same(
            &matrix(3, 3, &[2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0]),
            "diagonal 3x3 with distinct eigenvalues",
        );

        // A NEAR-degenerate pair (2 and 2 + 1e-3), not an exact tie: the
        // spectrum is fixed and each eigenvector is still unique.
        let q5 = orthonormal_columns(&[
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[2.0, -1.0, 0.0, 3.0, -2.0],
            &[0.0, 3.0, 1.0, -1.0, 2.0],
            &[4.0, 1.0, -2.0, 0.0, 1.0],
            &[-1.0, 2.0, 5.0, 1.0, -3.0],
        ]);
        assert_same(
            &from_spectrum(&q5, &[1.0, 2.0, 2.0 + 1e-3, 5.0, 9.0]),
            "symmetric 5x5 with a near-degenerate pair",
        );

        assert_same(&hilbert(5), "ill-conditioned symmetrized Hilbert 5x5");

        // Garbage in the upper triangle (and a NaN there) must be ignored.
        assert_same(&matrix(2, 2, &[4.0, 1e9, 2.0, 3.0]), "2x2 with an upper 1e9");
        assert_same(
            &matrix(3, 3, &[5.0, -1e9, 1e9, 2.0, 4.0, 1e9, -3.0, 1.0, 6.0]),
            "3x3 with an upper 1e9",
        );
        assert_same(
            &matrix(2, 2, &[1.0, f64::NAN, 0.5, 2.0]),
            "2x2 with a NaN in the ignored upper triangle",
        );

        // A generic (generically non-degenerate) wider matrix.
        assert_same(&pseudo_random_symmetric(8, 0x5EED_0008), "fixed-seed symmetric 8x8");
    }

    /// Exact repeated eigenvalues: the basis of the degenerate subspace is not
    /// unique, so this RECORDS the measured difference between `faer`'s basis
    /// and production's (after the shared orientation) instead of asserting
    /// agreement — the spectrum, the invariants and the orientation are
    /// asserted for both.  This is the measurement that keeps faer out of
    /// production.
    #[test]
    fn exact_degenerate_eigenvectors_are_recorded() {
        // diag(2, 2, 5): the degenerate subspace is the coordinate plane, which
        // both algorithms resolve identically (measured difference 0.0).
        let diagonal_case = matrix(3, 3, &[2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 5.0]);
        // A rotated 2-fold degeneracy: any orthonormal basis of the plane is a
        // valid answer, so a differing basis is a basis difference, not an
        // error — but it IS an output change, so faer is not adopted.
        let q3 = orthonormal_columns(&[
            &[1.0, 2.0, 3.0],
            &[3.0, -1.0, 2.0],
            &[2.0, 3.0, -1.0],
        ]);
        let rotated_case = from_spectrum(&q3, &[2.0, 2.0, 5.0]);
        for (a, what, must_agree) in [
            (diagonal_case, "diag(2, 2, 5)", true),
            (rotated_case, "rotated diag(2, 2, 5)", false),
        ] {
            let ((w_candidate, v_candidate), (w_production, v_production)) = both(&a);
            assert_vector_close(&w_candidate, &w_production, TOLERANCE, &format!("{what}: eigenvalues"));
            assert_vector_close(&w_candidate, &vector(&[2.0, 2.0, 5.0]), TOLERANCE, &format!("{what}: spectrum"));
            for v in [&v_candidate, &v_production] {
                assert_matrix_close(
                    &matmul(&a, v),
                    &matmul(v, &diagonal(&w_candidate)),
                    TOLERANCE,
                    &format!("{what}: A V = V diag(w)"),
                );
                assert_matrix_close(
                    &matmul(&v.t(), v),
                    &Array2::<f64>::eye(3),
                    TOLERANCE,
                    &format!("{what}: VᵀV = I"),
                );
                assert_columns_oriented(v, what);
            }
            let difference = max_abs_difference(&v_candidate, &v_production);
            println!("{what}: max |V_faer - V_production| after orientation = {difference:.3e}");
            if must_agree {
                assert!(
                    difference <= TOLERANCE,
                    "{what}: the degenerate basis must agree, measured {difference:.3e}"
                );
            } else {
                // The recorded basis difference of the evaluation (a real
                // basis rotation, both answers valid): it must stay clearly
                // above the tolerance, which is why faer is not adopted.
                assert!(
                    difference > 1e-3,
                    "{what}: expected a degenerate-basis difference, measured {difference:.3e}"
                );
            }
        }
    }

    fn max_abs_difference(left: &Array2<f64>, right: &Array2<f64>) -> f64 {
        left.iter()
            .zip(right.iter())
            .fold(0.0f64, |acc, (a, b)| acc.max((a - b).abs()))
    }

    /// The evidence test behind the KEEP decision: the first mismatching
    /// eigenvector entry of `Q diag(2, 2, 5) Qᵀ` plus the maximum difference.
    #[test]
    #[ignore = "evidence: prints the degenerate-basis divergence that keeps faer out of production"]
    fn faer_eigh_divergence_evidence() {
        let q3 = orthonormal_columns(&[
            &[1.0, 2.0, 3.0],
            &[3.0, -1.0, 2.0],
            &[2.0, 3.0, -1.0],
        ]);
        let a = from_spectrum(&q3, &[2.0, 2.0, 5.0]);
        let ((w_candidate, v_candidate), (w_production, v_production)) = both(&a);
        println!("eigenvalues (faer candidate): {w_candidate:?}");
        println!("eigenvalues (production)    : {w_production:?}");
        let mut first: Option<(usize, usize, f64, f64)> = None;
        let mut worst = 0.0f64;
        for col in 0..v_candidate.ncols() {
            for row in 0..v_candidate.nrows() {
                let difference = (v_candidate[(row, col)] - v_production[(row, col)]).abs();
                worst = worst.max(difference);
                if difference > TOLERANCE && first.is_none() {
                    first = Some((
                        row,
                        col,
                        v_candidate[(row, col)],
                        v_production[(row, col)],
                    ));
                }
            }
        }
        match first {
            Some((row, col, candidate, production)) => println!(
                "first mismatch at ({row}, {col}): faer {candidate:.17e}, production {production:.17e}; \
                 max |difference| = {worst:.3e} (tolerance {TOLERANCE:.0e})"
            ),
            None => println!("no mismatch: the degenerate bases agree (max {worst:.3e})"),
        }
    }

    /// Timings of the candidate against production on fixed-seed symmetric
    /// matrices: `min` of 5 runs for 64x64/256x256 and of 3 for 512x512, after
    /// one warmup each.
    #[test]
    #[ignore = "measurement: faer vs the hand-written cyclic Jacobi"]
    fn bench_faer_vs_reference_eigh() {
        use std::time::Instant;

        println!("{:>5} | {:>12} | {:>12} | {:>9}", "n", "faer (ms)", "jacobi (ms)", "speedup");
        for &n in &[64usize, 256, 512] {
            let a = pseudo_random_symmetric(n, 0xE16E_0000 ^ n as u64);
            let reps = if n >= 512 { 3 } else { 5 };
            // Warmup: one run of each path.
            let _ = eigh_sym_faer(a.view()).expect("faer warmup");
            let _ = eigh_sym(a.view()).expect("jacobi warmup");
            let mut faer_best = f64::INFINITY;
            let mut jacobi_best = f64::INFINITY;
            for _ in 0..reps {
                let start = Instant::now();
                let _ = eigh_sym_faer(a.view()).expect("faer run");
                faer_best = faer_best.min(start.elapsed().as_secs_f64());
                let start = Instant::now();
                let _ = eigh_sym(a.view()).expect("jacobi run");
                jacobi_best = jacobi_best.min(start.elapsed().as_secs_f64());
            }
            println!(
                "{:>5} | {:>12.3} | {:>12.3} | {:>8.2}x",
                n,
                faer_best * 1e3,
                jacobi_best * 1e3,
                jacobi_best / faer_best
            );
        }
    }
}
