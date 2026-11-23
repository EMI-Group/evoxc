//! Indexing / data-movement kernels: `reshape`, `transpose`, `slice`,
//! `gather`, `scatter` and `concatenate`.
//!
//! The behavioural reference is `evoxc/backends/numpy/kernels/indexing.py`;
//! this module reproduces those numpy semantics on flat C-contiguous typed
//! buffers with the same error family and wording. The decisions that matter:
//!
//! * `slice` is Python `slice(start, limit, stride)` per dim — a missing entry
//!   list means numpy's defaults, negative bounds count from the end, bounds
//!   CLAMP (Python slicing never raises; the result is just shorter, possibly
//!   empty) and a negative stride steps backwards. A zero stride and a bound
//!   list whose length is not the operand rank are explicit `ShapeError`s.
//! * `gather` is numpy `take` along ONE axis (`x.shape[:axis] + indices.shape
//!   + x.shape[axis+1:]`, so a rank-0 index drops the axis); negative indices
//!   wrap and an out-of-range index is an explicit `ShapeError`. A multi-axis
//!   `axes` tuple has no single-shared-index-tensor equivalent (the numpy
//!   reference defers it too) → explicit `BackendError`, never a silent
//!     fallback to one axis.
//! * `scatter` is functional numpy `put_along_axis`: a COPY of the input is
//!   updated, low-rank indices pad to full rank (0-d → all-ones) and `updates`
//!   broadcasts against numpy's advanced-indexing result. DUPLICATE INDICES
//!   ARE NOT REJECTED — like numpy, the last write in C order wins.
//! * `transpose` (`permutation = null` reverses every axis), `reshape` (a
//!   runtime-dynamic target dim is an error — the engine never guesses a
//!   dimension; a count mismatch is numpy's `cannot reshape array of size N
//!   into shape ...`) and `concatenate` (casts every operand to the declared
//!   promoted dtype; rank-0 operands, a NULL `axis` and non-axis mismatches are
//!   errors) follow the same rule: never a wrong value, never a panic.
//!
//! Result shapes come from the op's declared result type: a static dim must
//! AGREE with the shape derived from the operands (a disagreement is an error,
//! never a wrongly-shaped buffer) while a runtime-dynamic dim takes the derived
//! extent — which is why the inner kernels take the already-evaluated declared
//! shape as `&[Option<usize>]` (`RunCtx::result_shape`) rather than calling the
//! strict helper.
use evoxc_core::{Error, Result};

use crate::buffer::{take_pooled_vec, Buffer};
use crate::exec::Value;
use crate::kernels::KernelMap;
use crate::scalar::{Scalar, ScalarValue};

mod concatenate;
mod gather;
mod reshape;
mod scatter;
mod slice;
mod strided_copy;

#[cfg(test)]
mod bench_concatenate;
#[cfg(test)]
mod bench_gather;
#[cfg(test)]
mod test_support;

use concatenate::concatenate_kernel;
use gather::gather_kernel;
use reshape::{reshape_kernel, transpose_kernel};
use scatter::scatter_kernel;
use slice::slice_kernel;

pub fn register(map: &mut KernelMap) {
    map.insert("reshape", reshape_kernel);
    map.insert("transpose", transpose_kernel);
    map.insert("slice", slice_kernel);
    map.insert("gather", gather_kernel);
    map.insert("scatter", scatter_kernel);
    map.insert("concatenate", concatenate_kernel);
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// The `position`-th operand, with an explicit error instead of an index panic.
fn operand<'a>(op: &str, operands: &'a [Value], position: usize) -> Result<&'a Buffer> {
    operands
        .get(position)
        .map(|value| &**value)
        .ok_or_else(|| {
            Error::backend(format!(
                "op '{op}': missing operand {position} (got {} operand(s))",
                operands.len()
            ))
        })
}

/// Reconcile the declared result shape with the shape derived from the
/// operands: a STATIC declared dim must agree (a disagreement is an explicit
/// error, never a wrongly-shaped buffer) and a runtime-dynamic dim takes the
/// derived extent, which the caller computed from the operands. This is the
/// dynamic-tolerant form of `util::result_shape` (that helper is the strict
/// one: it rejects any `None` dim instead of taking the operand extent).
fn reconcile(op: &str, declared: &[Option<usize>], derived: Vec<usize>) -> Result<Vec<usize>> {
    if declared.len() != derived.len() {
        return Err(Error::shape(format!(
            "op '{op}': derived result rank {} does not match the declared rank {}",
            derived.len(),
            declared.len()
        )));
    }
    for (axis, (want, got)) in declared.iter().zip(derived.iter()).enumerate() {
        if let Some(want) = want {
            if want != got {
                return Err(Error::shape(format!(
                    "op '{op}': derived result dim {axis} is {got} but the declared type says {want}"
                )));
            }
        }
    }
    Ok(derived)
}

/// An out-of-range flat-offset diagnostic — a shape/data mismatch is an error,
/// never a panic and never a silently wrong value.
fn offset_error(offset: impl std::fmt::Display, len: usize, op: &str) -> Error {
    Error::backend(format!(
        "{op}: computed flat offset {offset} is outside a {len}-element buffer — the declared result type does not match the operands"
    ))
}

/// A recycled destination vector of EXACTLY `total` elements, or a fresh zeroed
/// one when the thread-local storage pool has no stock of that size.
///
/// Every caller overwrites all `total` elements (a contiguous `copy_from_slice`
/// or an ndarray `assign` over a same-sized destination view), so a recycled
/// stock can never leak a stale value. The exact-length filter keeps a
/// mismatched recycled stock from ever reaching a result.
fn pooled_total<T: Scalar>(total: usize) -> Vec<T> {
    take_pooled_vec::<T>(total)
        .filter(|values| values.len() == total)
        .unwrap_or_else(|| T::zeros(total))
}

/// A tuple rendering for diagnostics (numpy's `(0, 1)` spelling).
fn render_ints(values: &[i64]) -> String {
    let parts: Vec<String> = values.iter().map(|value| value.to_string()).collect();
    format!("({})", parts.join(", "))
}

/// numpy's `reshape` message rendering — `(4,2)` (no space after the comma).
fn render_shape(values: &[usize]) -> String {
    let parts: Vec<String> = values.iter().map(|value| value.to_string()).collect();
    format!("({})", parts.join(","))
}

fn require_integer_indices(op: &str, indices: &Buffer) -> Result<()> {
    if !indices.dtype.is_int() {
        return Err(Error::backend(format!(
            "{op}: indices must be an integer dtype (int32/int64), got {}",
            indices.dtype.name()
        )));
    }
    Ok(())
}

/// The `i128` view of one typed index scalar: every integer dtype numpy
/// accepts, widened to one comparison domain.
fn index_as_i128<I: Scalar>(value: I) -> i128 {
    match value.to_scalar() {
        ScalarValue::I(value) => value as i128,
        ScalarValue::U(value) => value as i128,
        ScalarValue::F(value) => value as i128,
        ScalarValue::C(value) => value.re as i128,
    }
}

/// The hot half of [`resolve_index`]: a non-negative in-range index, decided
/// with 64-bit arithmetic (every extent a contiguous operand can have fits
/// `i64`).  Everything else — a negative index that wraps, and every
/// out-of-range diagnostic — falls through to [`resolve_index`], so the error
/// wording stays exactly that function's.  [`resolve_index`] on `i128` per
/// element is measurably slower in the flat gather/scatter loops (128-bit
/// compares on the hot path), which is why the fast decision is 64-bit.
#[inline]
fn resolve_index_fast(raw: i128, extent: usize, axis: usize, op: &str) -> Result<usize> {
    if let Ok(value) = i64::try_from(raw) {
        if value >= 0 && (value as usize) < extent {
            return Ok(value as usize);
        }
    }
    resolve_index(raw, extent, axis, op)
}

/// numpy index semantics: a negative index counts from the end, anything else
/// outside `[0, extent)` is an explicit `ShapeError` with numpy's wording.
fn resolve_index(raw: i128, extent: usize, axis: usize, op: &str) -> Result<usize> {
    let length = extent as i128;
    let wrapped = if raw < 0 { raw + length } else { raw };
    if wrapped < 0 || wrapped >= length {
        return Err(out_of_bounds(raw, extent, axis, op));
    }
    Ok(wrapped as usize)
}

/// The out-of-bounds diagnostic, outlined so the hot resolve loop above carries
/// no formatting machinery (the message text is unchanged).
#[cold]
#[inline(never)]
fn out_of_bounds(raw: i128, extent: usize, axis: usize, op: &str) -> Error {
    Error::shape(format!(
        "{op}: index {raw} is out of bounds for axis {axis} with size {extent}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::indexing::test_support::{declared, f32_buffer, i64_buffer};

    #[test]
    fn reconcile_rejects_a_wrong_static_dim() {
        let err = reconcile("gather", &declared(&[2, 2]), vec![2, 3]).unwrap_err();
        assert!(err.message.contains("declared type says 2"), "{}", err.message);
        let err = reconcile("gather", &declared(&[2]), vec![2, 3]).unwrap_err();
        assert!(err.message.contains("declared rank 1"), "{}", err.message);
        // A runtime-dynamic dim takes the operand-derived extent.
        assert_eq!(reconcile("gather", &[None, Some(3)], vec![2, 3]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn gather_requires_an_integer_index_dtype() {
        let indices = f32_buffer(&[1], &[0.]);
        let err = require_integer_indices("gather", &indices).unwrap_err();
        assert!(
            err.message
                .contains("gather: indices must be an integer dtype (int32/int64), got float32"),
            "{}",
            err.message
        );
        assert!(require_integer_indices("gather", &i64_buffer(&[1], &[0])).is_ok());
    }
}
