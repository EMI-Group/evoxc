// ---------------------------------------------------------------------------
// Operand / result plumbing
// ---------------------------------------------------------------------------

use evoxc_core::{Attrs, DType, Error, Result};
use half::f16;
use serde_json::Value as Json;

use crate::buffer::{shape_text, Buffer, Storage};
use crate::kernels::util;
use crate::scalar::Scalar;
use crate::with_type;

/// `(u, s, vh)` of a reduced SVD.
pub(super) type SvdParts = (Vec<f64>, Vec<f64>, Vec<f64>);

/// One batched stack of square-or-rectangular matrices, flattened to `f64`.
#[derive(Debug)]
pub(super) struct Mat {
    /// Leading batch dims (the matrix dims are the last two).
    pub(super) batch: Vec<usize>,
    pub(super) rows: usize,
    pub(super) cols: usize,
    /// C-order values, one `rows * cols` block per batch element.
    pub(super) data: Vec<f64>,
}

impl Mat {
    pub(super) fn batch_len(&self) -> usize {
        util::numel(&self.batch)
    }
}

/// Validate an operand's rank/dtype and flatten it to `f64` (batch-major).
pub(super) fn prepare(op: &str, input: &Buffer, square: bool) -> Result<Mat> {
    let rank = input.shape.len();
    if rank < 2 {
        return Err(Error::shape(format!(
            "op '{op}': input must have rank >= 2, got rank {rank} (shape {})",
            shape_text(&input.shape)
        )));
    }
    if input.dtype.is_complex() {
        return Err(Error::backend(format!(
            "op '{op}': complex input is not supported by the native CPU engine (dtype {})",
            input.dtype.name()
        )));
    }
    let rows = input.shape[rank - 2];
    let cols = input.shape[rank - 1];
    if square && rows != cols {
        return Err(Error::shape(format!(
            "op '{op}': matrix must be square, got {rows} vs {cols} (shape {})",
            shape_text(&input.shape)
        )));
    }
    let data: Vec<f64> = with_type!(input, |items| items
        .iter()
        .map(|item| item.to_scalar().real())
        .collect::<Vec<f64>>());
    Ok(Mat {
        batch: input.shape[..rank - 2].to_vec(),
        rows,
        cols,
        data,
    })
}

/// Wrap computed real values in the DECLARED result dtype and shape.
pub(super) fn real_result(
    op: &str,
    shape: Vec<usize>,
    dtype: DType,
    data: Vec<f64>,
) -> Result<Buffer> {
    let expected = util::numel(&shape);
    if data.len() != expected {
        return Err(Error::shape(format!(
            "op '{op}': produced {} value(s) for a declared result of shape {} ({expected})",
            data.len(),
            shape_text(&shape)
        )));
    }
    let buffer = Buffer::new(DType::Float64, shape, Storage::F64(data))?;
    if dtype == DType::Float64 {
        return Ok(buffer);
    }
    if !dtype.is_float() {
        return Err(Error::dtype(format!(
            "op '{op}': a {} result is not supported (expected a real floating dtype)",
            dtype.name()
        )));
    }
    buffer.cast(dtype)
}

/// Read an optional numeric attribute (`null`/absent → `None`).
pub(super) fn attr_float(attrs: &Attrs, name: &str, op: &str) -> Result<Option<f64>> {
    let value = match attrs.get(name) {
        None | Some(evoxc_core::AttrValue::Null) => return Ok(None),
        Some(evoxc_core::AttrValue::Any(Json::Null)) => return Ok(None),
        Some(evoxc_core::AttrValue::Float(value)) => return Ok(Some(*value)),
        Some(evoxc_core::AttrValue::Int(value)) => return Ok(Some(*value as f64)),
        Some(evoxc_core::AttrValue::Any(Json::Number(number))) => number.as_f64(),
        Some(other) => {
            return Err(Error::verification(format!(
                "op '{op}': attribute {name:?} must be a number or null, got {other:?}"
            )))
        }
    };
    match value {
        Some(value) => Ok(Some(value)),
        None => Err(Error::verification(format!(
            "op '{op}': attribute {name:?} is not representable as a float"
        ))),
    }
}

/// The epsilon numpy's default `matrix_rank` tolerance uses: the real
/// precision of the input (`int`/`bool` compute in `float64`).
pub(super) fn rank_epsilon(dtype: DType) -> f64 {
    match dtype {
        DType::Float16 => f64::from(f16::EPSILON),
        DType::Float32 | DType::Complex64 => f64::from(f32::EPSILON),
        _ => f64::EPSILON,
    }
}

/// numpy's `matrix_rank` count: `#{s : s > tol}` with the auto tolerance
/// `s_max · max(m, n) · eps` when `tol` is absent.
pub(super) fn numerical_rank(
    singular: &[f64],
    rows: usize,
    cols: usize,
    tol: Option<f64>,
    eps: f64,
) -> i64 {
    if singular.is_empty() {
        return 0;
    }
    let limit = match tol {
        Some(tol) => tol,
        None => singular[0] * rows.max(cols) as f64 * eps,
    };
    singular.iter().filter(|value| **value > limit).count() as i64
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::linalg_aux::svd::svd_reduced;

    #[test]
    fn matrix_rank_counts_above_tolerance() {
        // Full rank.
        let full = svd_reduced(&[1.0, 0.0, 0.0, 1.0], 2, 2).unwrap().1;
        assert_eq!(numerical_rank(&full, 2, 2, None, f64::EPSILON), 2);
        // Rank deficient (the second row is twice the first).
        let deficient = svd_reduced(&[1.0, 2.0, 2.0, 4.0], 2, 2).unwrap().1;
        assert_eq!(numerical_rank(&deficient, 2, 2, None, f64::EPSILON), 1);
        // A tiny-but-nonzero singular value counts at the default tolerance
        // and not at an explicit larger one.
        let tiny = svd_reduced(&[1.0, 0.0, 0.0, 1e-8], 2, 2).unwrap().1;
        assert_eq!(numerical_rank(&tiny, 2, 2, None, f64::EPSILON), 2);
        assert_eq!(numerical_rank(&tiny, 2, 2, Some(1e-6), f64::EPSILON), 1);
        // An empty spectrum is rank zero.
        assert_eq!(numerical_rank(&[], 0, 3, None, f64::EPSILON), 0);
    }

    #[test]
    fn prepare_rejects_bad_operands() {
        let vector =
            Buffer::new(DType::Float64, vec![3], Storage::F64(vec![1.0, 2.0, 3.0])).unwrap();
        let err = prepare("eigh", &vector, true).unwrap_err();
        assert!(
            err.message.contains("rank >= 2"),
            "message: {}",
            err.message
        );
        let rectangular =
            Buffer::new(DType::Float64, vec![2, 3], Storage::F64(vec![0.0; 6])).unwrap();
        let err = prepare("eigh", &rectangular, true).unwrap_err();
        assert!(err.message.contains("square"), "message: {}", err.message);
        // Non-square is fine for the rectangular factorizations.
        assert!(prepare("qr", &rectangular, false).is_ok());
        let complex = Buffer::new(
            DType::Complex128,
            vec![2, 2],
            Storage::C128(vec![num_complex::Complex::new(1.0, 0.0); 4]),
        )
        .unwrap();
        let err = prepare("svd", &complex, false).unwrap_err();
        assert!(err.message.contains("complex"), "message: {}", err.message);
        // Integers are flattened to f64 for the computation.
        let ints = Buffer::new(DType::Int32, vec![2, 2], Storage::I32(vec![1, 2, 3, 4])).unwrap();
        let matrix = prepare("matrix_rank", &ints, false).unwrap();
        assert_eq!(matrix.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(matrix.batch_len(), 1);
    }

    #[test]
    fn real_result_respects_declared_dtype_and_shape() {
        let out = real_result("eigh", vec![2], DType::Float32, vec![1.0, 2.0]).unwrap();
        assert_eq!(out.dtype, DType::Float32);
        assert_eq!(out.storage, Storage::F32(vec![1.0, 2.0]));
        let out =
            real_result("eigh", vec![2, 2], DType::Float64, vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        assert_eq!(out.dtype, DType::Float64);
        let err = real_result("eigh", vec![3], DType::Float64, vec![1.0, 2.0]).unwrap_err();
        assert!(
            err.message.contains("declared result"),
            "message: {}",
            err.message
        );
        let err = real_result("eigh", vec![1], DType::Int64, vec![1.0]).unwrap_err();
        assert!(
            err.message.contains("real floating"),
            "message: {}",
            err.message
        );
    }
}
