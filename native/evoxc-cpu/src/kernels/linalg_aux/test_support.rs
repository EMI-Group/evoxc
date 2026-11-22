//! Shared numeric helpers for the `linalg_aux` submodule tests.

pub(super) fn assert_close(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tol,
        "{what}: got {actual}, expected {expected} (tol {tol})"
    );
}

pub(super) fn assert_matrix_close(actual: &[f64], expected: &[f64], tol: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= tol,
            "{what}: element {index} is {got}, expected {want} (tol {tol})"
        );
    }
}

/// `a (m×k) · b (k×n)`.
pub(super) fn matmul(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for row in 0..m {
        for inner in 0..k {
            for col in 0..n {
                out[row * n + col] += a[row * k + inner] * b[inner * n + col];
            }
        }
    }
    out
}

pub(super) fn transpose(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            out[col * rows + row] = a[row * cols + col];
        }
    }
    out
}

/// `diag(values)` as an `n × n` matrix.
pub(super) fn diagonal(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![0.0f64; n * n];
    for (index, value) in values.iter().enumerate() {
        out[index * n + index] = *value;
    }
    out
}

/// Every column's largest-magnitude entry must be non-negative (the sign
/// convention this module fixes).
pub(super) fn assert_columns_oriented(matrix: &[f64], rows: usize, cols: usize, what: &str) {
    for col in 0..cols {
        let mut pivot = 0usize;
        for row in 0..rows {
            if matrix[row * cols + col].abs() > matrix[pivot * cols + col].abs() {
                pivot = row;
            }
        }
        assert!(
            matrix[pivot * cols + col] >= 0.0,
            "{what}: column {col} is not sign-oriented"
        );
    }
}
