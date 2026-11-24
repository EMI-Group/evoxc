//! Shared numeric helpers for the `linalg_aux` submodule tests.
//!
//! The helpers speak `ndarray`: a test builds its operand with
//! `Array2::from_shape_vec`, multiplies with `ndarray`'s own `dot` and
//! transposes with `.t()`, so no test carries a hand-written offset walk.

use ndarray::linalg::Dot;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};

pub(super) fn assert_close(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tol,
        "{what}: got {actual}, expected {expected} (tol {tol})"
    );
}

/// Element-wise agreement of two matrices of the same shape, in logical order.
pub(super) fn assert_matrix_close<S1, S2>(
    actual: &ArrayBase<S1, Ix2>,
    expected: &ArrayBase<S2, Ix2>,
    tol: f64,
    what: &str,
) where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    assert_eq!(actual.shape(), expected.shape(), "{what}: shape mismatch");
    for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= tol,
            "{what}: element {index} is {got}, expected {want} (tol {tol})"
        );
    }
}

/// Element-wise agreement of two vectors of the same length.
pub(super) fn assert_vector_close<S1, S2>(
    actual: &ArrayBase<S1, Ix1>,
    expected: &ArrayBase<S2, Ix1>,
    tol: f64,
    what: &str,
) where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= tol,
            "{what}: element {index} is {got}, expected {want} (tol {tol})"
        );
    }
}

/// Matrix product `a · b` (`ndarray`'s own contraction).
pub(super) fn matmul<S1, S2>(a: &ArrayBase<S1, Ix2>, b: &ArrayBase<S2, Ix2>) -> Array2<f64>
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    a.dot(b)
}

/// `diag(values)` as an `n × n` matrix.
pub(super) fn diagonal<S>(values: &ArrayBase<S, Ix1>) -> Array2<f64>
where
    S: Data<Elem = f64>,
{
    Array2::from_diag(values)
}

/// Every column's largest-magnitude entry must be non-negative (the sign
/// convention this module fixes).
pub(super) fn assert_columns_oriented<S>(matrix: &ArrayBase<S, Ix2>, what: &str)
where
    S: Data<Elem = f64>,
{
    for col in 0..matrix.ncols() {
        let mut pivot = 0usize;
        let mut magnitude = 0.0f64;
        for (row, value) in matrix.column(col).iter().enumerate() {
            if value.abs() > magnitude {
                magnitude = value.abs();
                pivot = row;
            }
        }
        assert!(
            matrix[(pivot, col)] >= 0.0,
            "{what}: column {col} is not sign-oriented"
        );
    }
}

/// `Array2::from_shape_vec` for a row-major literal, the tests' operand builder.
pub(super) fn matrix(rows: usize, cols: usize, values: &[f64]) -> Array2<f64> {
    Array2::from_shape_vec((rows, cols), values.to_vec()).expect("a consistent test matrix")
}

/// `Array1::from_vec` from a literal.
pub(super) fn vector(values: &[f64]) -> Array1<f64> {
    Array1::from_vec(values.to_vec())
}
