// ---------------------------------------------------------------------------
// Cholesky factorization
// ---------------------------------------------------------------------------

//! `cholesky` in numpy's shape: the lower-triangular factor of a positive
//! definite matrix.
//!
//! [`cholesky_lower`] returns `L` with `A = L Lᵀ`, reading only the LOWER
//! triangle of `a` (numpy's `UPLO='L'` default; the upper triangle is ignored),
//! and a matrix that is not positive definite is an explicit `BackendError`
//! naming the failing leading minor.
//! A NON-finite operand keeps numpy's (LAPACK `potrf`) behaviour: an `inf`
//! pivot passes an infinite factor through and a NaN pivot is the frozen
//! non-positive-definite error.
//!
//! KEPT hand-written, measured against `faer`'s `LLᵀ` solver
//! (`faer::linalg::solvers::Llt` with `Side::Lower`, kept as the test-only A/B
//! oracle `cholesky_via_faer`):
//! - the factor agrees to 1e-10 absolute over a deterministic corpus (1x1, a
//!   well-conditioned 6x6, Hilbert 6x6, a clustered-eigenvalue 5x5, and an
//!   f32-rounded 4x4 at 1e-6), the upper triangle of the input changes nothing,
//!   `L` is lower triangular with exact zeros above a positive diagonal, and
//!   the non-positive-definite inputs yield the identical message
//!   (`faer_cholesky_matches_the_production_kernel`);
//! - but `Llt::new` is the only reachable per-call entry and always reads
//!   faer's PROCESS-GLOBAL parallelism (default: all cores). With that default
//!   the factor costs 0.016 / 7.748 / 57.642 ms at 64² / 256² / 1024² against
//!   this kernel's 0.016 / 1.171 / 76.173 ms — a 6.6x REGRESSION at 256², the
//!   size a CPU graph engine meets most. Sequential faer is 3.0x (256²) and
//!   3.7x (1024²) faster, but the low-level
//!   `faer::linalg::cholesky::llt::factor::cholesky_in_place`, which takes an
//!   explicit `Par`, is unreachable from outside faer: its `&mut MemStack`
//!   argument's type is re-exported only from faer's crate-private
//!   `internal_prelude`, and `dyn-stack` is not a dependency of this crate.
//!   Pinning the global to `Par::Seq` once would be a process-wide faer policy
//!   decision (it governs any future faer kernel here too), so this kernel does
//!   not mutate it.
//!
//! `bench_cholesky_variants` records the full table (both global settings and
//! this kernel); run it with `--release ... -- --ignored --nocapture
//! --test-threads=1`.
use evoxc_core::{Error, Result};
use ndarray::{Array2, ArrayView2};

/// Lower-triangular `L` with `A = L Lᵀ`, read from the LOWER triangle of `a`
/// (numpy `cholesky`'s default).  A matrix that is not positive definite is an
/// explicit error naming the op and the failing leading minor.
pub(super) fn cholesky_lower(a: ArrayView2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(Error::shape(format!(
            "op 'cholesky': matrix must be square, got {n} vs {}",
            a.ncols()
        )));
    }
    let mut lower = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        let mut diagonal = a[(col, col)];
        for k in 0..col {
            diagonal -= lower[(col, k)] * lower[(col, k)];
        }
        if diagonal.is_nan() || diagonal <= 0.0 {
            return Err(Error::backend(format!(
                "op 'cholesky': matrix is not positive definite (leading minor of order {} is not positive)",
                col + 1
            )));
        }
        let root = diagonal.sqrt();
        lower[(col, col)] = root;
        for row in (col + 1)..n {
            let mut value = a[(row, col)];
            for k in 0..col {
                value -= lower[(row, k)] * lower[(col, k)];
            }
            lower[(row, col)] = value / root;
        }
    }
    Ok(lower)
}
#[cfg(test)]
/// The `faer` counterpart, kept as the A/B oracle (see the module doc for why
/// it is not the production path). `Llt` copies only the requested triangle,
/// zeroes the strict upper triangle of its result, and reports
/// `LltError::NonPositivePivot { index }` for the first column whose pivot is
/// not strictly positive (a NaN pivot included).
fn cholesky_via_faer(a: ArrayView2<f64>) -> Result<Array2<f64>> {
    use faer::linalg::solvers::{Llt, LltError};
    use faer::Side;

    let n = a.nrows();
    if n == 0 {
        return Ok(Array2::<f64>::zeros((0, 0)));
    }
    // `faer` fills its own column-major matrix from the closure, so the layout
    // conversion stays on faer's side; the strict upper triangle stays zero,
    // exactly as `Llt`'s `Side::Lower` reads it.
    let matrix = faer::Mat::from_fn(n, n, |row, col| if col <= row { a[(row, col)] } else { 0.0 });
    let factor = Llt::new(matrix.as_ref(), Side::Lower).map_err(|error| match error {
        LltError::NonPositivePivot { index } => Error::backend(format!(
            "op 'cholesky': matrix is not positive definite (leading minor of order {} is not positive)",
            index + 1
        )),
    })?;
    // `L()` is a full `n × n` view whose strict upper triangle is exactly zero.
    let l = factor.L();
    Ok(Array2::from_shape_fn((n, n), |(row, col)| {
        if col <= row {
            l[(row, col)]
        } else {
            0.0
        }
    }))
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;
    use std::time::Instant;

    use evoxc_core::ErrorKind;

    use super::*;
    use crate::kernels::linalg_aux::test_support::{assert_matrix_close, matmul, matrix};

    // -----------------------------------------------------------------------
    // Deterministic corpus builders (no external crates, no RNG crate)
    // -----------------------------------------------------------------------

    /// A fixed-seed 64-bit LCG, so every corpus matrix is reproducible.
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        /// Uniform in `[0, 1)` (the top 53 bits, so every value is exact).
        fn unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }

        /// Uniform in `[-1, 1)`.
        fn centered(&mut self) -> f64 {
            self.unit() * 2.0 - 1.0
        }

        /// An `n × n` matrix of centered values, in row-major finite order.
        fn matrix(&mut self, n: usize) -> Array2<f64> {
            Array2::from_shape_fn((n, n), |_| self.centered())
        }
    }

    /// `B Bᵀ + n·I` from a fixed seed: symmetric positive definite, well
    /// conditioned (the identity term dominates the spectrum).
    fn well_conditioned(n: usize, seed: u64) -> Array2<f64> {
        let mut lcg = Lcg::new(seed);
        let b = lcg.matrix(n);
        let mut a = matmul(&b, &b.t());
        for i in 0..n {
            a[(i, i)] += n as f64;
        }
        a
    }

    /// The Hilbert matrix `H_ij = 1 / (i + j + 1)`: SPD and severely
    /// ill-conditioned (6x6 has condition number ~1.5e7).
    fn hilbert(n: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, n), |(i, j)| 1.0 / (i + j + 1) as f64)
    }

    /// Modified Gram–Schmidt over a fixed-seed matrix, so `Q` is orthonormal.
    fn gram_schmidt(mut m: Array2<f64>) -> Array2<f64> {
        let n = m.nrows();
        for col in 0..n {
            for previous in 0..col {
                let mut projection = 0.0f64;
                for row in 0..n {
                    projection += m[(row, previous)] * m[(row, col)];
                }
                for row in 0..n {
                    let correction = projection * m[(row, previous)];
                    m[(row, col)] -= correction;
                }
            }
            let mut norm = 0.0f64;
            for row in 0..n {
                norm += m[(row, col)] * m[(row, col)];
            }
            let norm = norm.sqrt();
            for row in 0..n {
                m[(row, col)] /= norm;
            }
        }
        m
    }

    /// `Q D Qᵀ` with orthonormal `Q` and `D = diag(1, 1+1e-3, 2, 3, 5)`: SPD
    /// with two nearly repeated eigenvalues.
    fn clustered_eigenvalues() -> Array2<f64> {
        let n = 5;
        let mut lcg = Lcg::new(0x5EED_5EED);
        let q = gram_schmidt(lcg.matrix(n));
        let d = [1.0, 1.0 + 1e-3, 2.0, 3.0, 5.0];
        let mut a = Array2::<f64>::zeros((n, n));
        for row in 0..n {
            for col in 0..n {
                let mut sum = 0.0f64;
                for k in 0..n {
                    sum += q[(row, k)] * d[k] * q[(col, k)];
                }
                a[(row, col)] = sum;
            }
        }
        a
    }

    /// `C Cᵀ / n + I` from a fixed seed: dense SPD with the identity term
    /// keeping the spectrum in `[1, 1 + n]`, so it stays numerically
    /// positive definite at every bench size.
    fn spd(n: usize, seed: u64) -> Array2<f64> {
        let mut lcg = Lcg::new(seed);
        let c = lcg.matrix(n);
        let mut a = matmul(&c, &c.t());
        let scale = n as f64;
        for row in 0..n {
            for col in 0..n {
                a[(row, col)] = a[(row, col)] / scale + if row == col { 1.0 } else { 0.0 };
            }
        }
        a
    }

    /// The same matrix rounded to `f32` precision (what an f32 operand turns
    /// into once the kernel upcasts it).
    fn f32_rounded(a: Array2<f64>) -> Array2<f64> {
        a.mapv(|value| value as f32 as f64)
    }

    /// `(name, matrix, tolerance)` in a fixed order.
    fn corpus() -> Vec<(&'static str, Array2<f64>, f64)> {
        vec![
            ("1x1", matrix(1, 1, &[4.0]), 1e-10),
            ("spd_6", well_conditioned(6, 0x00C0_FFEE), 1e-10),
            ("hilbert_6", hilbert(6), 1e-10),
            ("clustered_5", clustered_eigenvalues(), 1e-10),
            (
                "f32_rounded_4",
                f32_rounded(well_conditioned(4, 0x0BAD_F00D)),
                1e-6,
            ),
        ]
    }

    /// `A` rebuilt from the lower triangle of `a` (the matrix the factor is
    /// actually of).
    fn symmetric_from_lower(a: &Array2<f64>) -> Array2<f64> {
        let n = a.nrows();
        let mut symmetric = Array2::<f64>::zeros((n, n));
        for row in 0..n {
            for col in 0..=row {
                let value = a[(row, col)];
                symmetric[(row, col)] = value;
                symmetric[(col, row)] = value;
            }
        }
        symmetric
    }

    /// Lower triangular, exactly zero above the diagonal, positive diagonal.
    fn assert_lower_triangular(l: &Array2<f64>, what: &str) {
        for col in 0..l.ncols() {
            assert!(
                l[(col, col)] > 0.0,
                "{what}: diagonal entry {col} is {}",
                l[(col, col)]
            );
            for row in 0..col {
                assert!(
                    l[(row, col)] == 0.0,
                    "{what}: entry ({row}, {col}) above the diagonal is {}",
                    l[(row, col)]
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Parity with the hand-written reference
    // -----------------------------------------------------------------------

    #[test]
    fn faer_cholesky_matches_the_production_kernel() {
        for (name, a, tol) in corpus() {
            let n = a.nrows();
            let got = cholesky_via_faer(a.view())
                .unwrap_or_else(|error| panic!("{name}: {}", error.message));
            let want = cholesky_lower(a.view())
                .unwrap_or_else(|error| panic!("{name}: {}", error.message));
            assert_matrix_close(&got, &want, tol, name);
            assert_lower_triangular(&got, name);
            assert_matrix_close(&matmul(&got, &got.t()), &symmetric_from_lower(&a), tol, name);
            // The upper triangle of the input is ignored: a garbage one gives
            // the identical factor.
            let mut garbage = a.clone();
            for row in 0..n {
                for col in (row + 1)..n {
                    garbage[(row, col)] = 1e6 * (1.0 + row as f64 + col as f64);
                }
            }
            let ignored = cholesky_lower(garbage.view())
                .unwrap_or_else(|error| panic!("{name}: {}", error.message));
            assert_eq!(ignored, want, "{name}: the upper triangle changed the factor");
        }
    }

    #[test]
    fn cholesky_reports_the_failing_leading_minor() {
        let cases: Vec<(&str, Array2<f64>, usize)> = vec![
            ("negative 1x1", matrix(1, 1, &[-1.0]), 1),
            ("zero 1x1", matrix(1, 1, &[0.0]), 1),
            ("indefinite 2x2", matrix(2, 2, &[1.0, 2.0, 2.0, 1.0]), 2),
            (
                "late failure 4x4",
                matrix(
                    4,
                    4,
                    &[
                        4.0, 2.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0,
                        2.0, 1.0,
                    ],
                ),
                4,
            ),
            ("nan diagonal", matrix(2, 2, &[f64::NAN, 0.0, 0.0, 1.0]), 1),
            (
                "singular 3x3",
                matrix(3, 3, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                2,
            ),
        ];
        for (name, a, order) in cases {
            let err = cholesky_lower(a.view()).expect_err(name);
            assert_eq!(err.kind, ErrorKind::Backend, "{name}: not a BackendError");
            assert_eq!(
                err.message,
                format!(
                    "op 'cholesky': matrix is not positive definite (leading minor of order {order} is not positive)"
                ),
                "{name}: message"
            );
        }
    }
    /// numpy (LAPACK `potrf`) passes a non-finite pivot through: an `inf`
    /// diagonal yields an `inf` factor, a NaN pivot is the frozen error.
    #[test]
    fn cholesky_non_finite_operands_keep_numpy_behaviour() {
        let got = cholesky_lower(matrix(2, 2, &[f64::INFINITY, 0.0, 0.0, 1.0]).view())
            .expect("an infinite factor passes through");
        assert_eq!(got[(0, 0)], f64::INFINITY);
        assert_eq!(got[(1, 0)], 0.0);
        assert_eq!(got[(1, 1)], 1.0);
        assert!(!got.iter().any(|value| value.is_nan()));
        let err = cholesky_lower(matrix(2, 2, &[f64::NAN, 0.0, 0.0, 1.0]).view())
            .expect_err("a NaN pivot");
        assert_eq!(err.kind, ErrorKind::Backend);
        assert_eq!(
            err.message,
            "op 'cholesky': matrix is not positive definite (leading minor of order 1 is not positive)"
        );
        // A non-finite value in the IGNORED upper triangle changes nothing.
        let garbage = matrix(2, 2, &[4.0, f64::INFINITY, 2.0, 3.0]);
        assert_eq!(
            cholesky_lower(garbage.view()).unwrap(),
            cholesky_lower(matrix(2, 2, &[4.0, 0.0, 2.0, 3.0]).view()).unwrap()
        );
    }

    #[test]
    fn cholesky_two_by_two_and_non_positive_definite() {
        let matrix = |values: [f64; 4]| Array2::from_shape_vec((2, 2), values.to_vec()).unwrap();
        // [[4, 2], [2, 3]] = L Lᵀ with L = [[2, 0], [1, sqrt(2)]].
        let a = [4.0, 2.0, 2.0, 3.0];
        let l = cholesky_lower(matrix(a).view()).unwrap();
        assert_matrix_close(
            &l,
            &Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 1.0, 2.0f64.sqrt()]).unwrap(),
            1e-12,
            "L",
        );
        assert_matrix_close(&matmul(&l, &l.t()), &matrix(a), 1e-12, "L Lᵀ");
        // The upper triangle is ignored.
        assert_eq!(
            cholesky_lower(matrix([4.0, -7.5, 2.0, 3.0]).view()).unwrap(),
            l
        );
        // [[1, 2], [2, 1]] has a negative eigenvalue ⇒ explicit error.
        let err = cholesky_lower(matrix([1.0, 2.0, 2.0, 1.0]).view()).unwrap_err();
        assert!(err.message.contains("cholesky"), "message: {}", err.message);
        assert!(
            err.message.contains("not positive definite"),
            "message: {}",
            err.message
        );
    }

    // -----------------------------------------------------------------------
    // Measurement aid
    // -----------------------------------------------------------------------

    /// Times the faer factor (with faer's default, globally parallel, setting
    /// and with `Par::Seq`) against this kernel on 64/256/1024 SPD matrices;
    /// run with `--release ... -- --ignored --nocapture --test-threads=1`.
    /// It mutates faer's process-global parallelism, so it restores the default
    /// (`Par::rayon(0)`) before returning.
    #[test]
    #[ignore]
    fn bench_cholesky_variants() {
        fn best(mut run: impl FnMut(), warmups: usize, reps: usize) -> f64 {
            for _ in 0..warmups {
                run();
            }
            let mut best = f64::INFINITY;
            for _ in 0..reps {
                let start = Instant::now();
                run();
                best = best.min(start.elapsed().as_secs_f64());
            }
            best
        }
        println!("     n | faer (default par) | faer (seq) |     kernel | par/ref | seq/ref");
        for (n, reps) in [(64usize, 5usize), (256, 5), (1024, 3)] {
            let a = spd(n, 0xA11CE ^ n as u64);
            let view = a.view();
            // faer's default global setting is `Par::rayon(0)` (all cores).
            faer::set_global_parallelism(faer::Par::rayon(0));
            let parallel = best(
                || {
                    black_box(cholesky_via_faer(view).unwrap());
                },
                1,
                reps,
            );
            faer::set_global_parallelism(faer::Par::Seq);
            let sequential = best(
                || {
                    black_box(cholesky_via_faer(view).unwrap());
                },
                1,
                reps,
            );
            let reference = best(
                || {
                    black_box(cholesky_lower(view).unwrap());
                },
                1,
                reps,
            );
            println!(
                "{n:>5} | {:>14.3}ms | {:>8.3}ms | {:>8.3}ms | {:>6.2}x | {:>6.2}x",
                parallel * 1e3,
                sequential * 1e3,
                reference * 1e3,
                reference / parallel,
                reference / sequential
            );
        }
        faer::set_global_parallelism(faer::Par::rayon(0));
    }
}
