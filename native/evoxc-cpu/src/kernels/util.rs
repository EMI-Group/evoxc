//! Shared kernel helpers (axis normalization, shape arithmetic, C-order
//! indexing).  Category modules may add private helpers in their own file;
//! this module only holds vocabulary shared by three or more categories.
use evoxc_core::{DType, Error, Result};

use crate::buffer::shape_text;
use crate::exec::RunCtx;
use crate::plan::OpSpec;

/// Normalize one axis (`-1` → `rank - 1`); out of range is a `ShapeError`.
pub fn normalize_axis(axis: i64, rank: usize, what: &str) -> Result<usize> {
    let rank_i = rank as i64;
    let normalized = if axis < 0 { axis + rank_i } else { axis };
    if normalized < 0 || normalized >= rank_i {
        return Err(Error::shape(format!(
            "{what}: axis {axis} is out of range for rank {rank}"
        )));
    }
    Ok(normalized as usize)
}

/// Normalize a set of axes, rejecting duplicates.
pub fn normalize_axes(axes: &[i64], rank: usize, what: &str) -> Result<Vec<usize>> {
    let mut out = Vec::with_capacity(axes.len());
    for axis in axes {
        let normalized = normalize_axis(*axis, rank, what)?;
        if out.contains(&normalized) {
            return Err(Error::shape(format!(
                "{what}: duplicate axis {normalized}"
            )));
        }
        out.push(normalized);
    }
    Ok(out)
}

/// Reject a complex dtype with an explicit message naming the op.
pub fn reject_complex(op: &str, dtype: DType) -> Result<()> {
    if dtype.is_complex() {
        return Err(Error::backend(format!(
            "op '{op}': complex input is not supported"
        )));
    }
    Ok(())
}

/// Reject a non-floating dtype.
pub fn require_float(op: &str, dtype: DType) -> Result<()> {
    if !dtype.is_float() {
        return Err(Error::backend(format!(
            "op '{op}': expected a floating dtype, got {}",
            dtype.name()
        )));
    }
    Ok(())
}

/// Number of elements described by a concrete shape.
pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// C-order (row-major) strides for a concrete shape.
pub fn c_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    let mut stride = 1usize;
    for axis in (0..shape.len()).rev() {
        strides[axis] = stride;
        stride *= shape[axis];
    }
    strides
}

/// Flat offset of a C-order multi-index.
pub fn flat_index(index: &[usize], strides: &[usize]) -> usize {
    index.iter().zip(strides).map(|(i, s)| i * s).sum()
}

/// Evaluate a result shape to concrete extents.  Runtime-dynamic (`None`)
/// dims cannot be resolved here; the caller must derive them from operands.
pub fn result_shape(ctx: &RunCtx<'_>, spec: &OpSpec, index: usize) -> Result<Vec<usize>> {
    let evaluated = ctx.result_shape(spec, index)?;
    let mut out = Vec::with_capacity(evaluated.len());
    for (axis, dim) in evaluated.iter().enumerate() {
        match dim {
            Some(extent) => out.push(*extent),
            None => {
                return Err(Error::shape(format!(
                    "op '{}': result {index} dim {axis} is runtime-dynamic; derive it from the operands",
                    spec.name
                )))
            }
        }
    }
    Ok(out)
}

/// A shape-mismatch error in the interpreter's wording.
pub fn shape_error(op: &str, what: &str, actual: &[usize], expected: &[usize]) -> Error {
    Error::shape(format!(
        "op '{op}': {what} has shape {}, expected {}",
        shape_text(actual),
        shape_text(expected)
    ))
}

/// True when `small` broadcasts to exactly `big` (right-aligned).
pub fn broadcasts_to(small: &[usize], big: &[usize]) -> bool {
    if small.len() > big.len() {
        return false;
    }
    let offset = big.len() - small.len();
    small
        .iter()
        .enumerate()
        .all(|(axis, dim)| *dim == 1 || *dim == big[offset + axis])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axes_normalize_and_reject() {
        assert_eq!(normalize_axis(-1, 3, "x").unwrap(), 2);
        assert_eq!(normalize_axis(0, 3, "x").unwrap(), 0);
        assert!(normalize_axis(3, 3, "x").is_err());
        assert!(normalize_axes(&[0, 0], 2, "x").is_err());
    }

    #[test]
    fn strides_are_c_order() {
        assert_eq!(c_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(flat_index(&[1, 2, 3], &[12, 4, 1]), 23);
    }
}
