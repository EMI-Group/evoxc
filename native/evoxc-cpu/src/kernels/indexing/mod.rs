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

use crate::buffer::Buffer;
#[cfg(test)]
use crate::buffer::Storage;
use crate::exec::Value;
use crate::kernels::util;
use crate::kernels::KernelMap;
use crate::scalar::{Scalar, ScalarValue};
use crate::with_type;

mod concatenate;
mod gather;
mod reshape;
mod scatter;
mod slice;
mod stride_copy;

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

/// The block-copy bounds diagnostic [`copy_blocks`](crate::buffer::copy_blocks)
/// raises, with its wording (both entries say the same thing).
pub(super) fn block_error(op: &str) -> Error {
    Error::verification(format!(
        "op '{op}': a block copy reads outside the operand (internal engine error)"
    ))
}

/// A [`Storage`] built by a block copy WITHOUT the zero fill
/// [`block_storage`](crate::buffer::block_storage) performs.
///
/// The result is exactly that helper's — output run `b` is the `inner`-element
/// run starting at `values[offsets[b]]` — but the elements are pushed straight
/// into the result, so a block copy never first writes a full zero buffer it is
/// about to overwrite. That matters for the large-result fast paths (`tile`-,
/// `gather`- and `slice`-shaped results), where the fill is a second pass over
/// the whole output. The element-count and bounds contract is `copy_blocks`'
/// (same checks, same error wording): a short or long offset list is an
/// explicit error, never a silently truncated result.
#[cfg(test)]
fn block_storage_from<T: Scalar>(
    values: &[T],
    out_shape: &[usize],
    offsets: &[usize],
    inner: usize,
    op: &str,
) -> Result<Storage> {
    let total = util::numel(out_shape);
    if offsets.len().saturating_mul(inner) != total {
        return Err(block_error(op));
    }
    if inner == 0 {
        return Ok(T::into_storage(Vec::new()));
    }
    let mut out: Vec<T> = Vec::with_capacity(total);
    for offset in offsets {
        let start = *offset;
        let run = values
            .get(start..start + inner)
            .ok_or_else(|| block_error(op))?;
        out.extend_from_slice(run);
    }
    Ok(T::into_storage(out))
}

/// Read `values[offset]` with a SIGNED offset (slice strides may be negative).
fn read_checked<T: Copy>(values: &[T], offset: i64, op: &str) -> Result<T> {
    usize::try_from(offset)
        .ok()
        .and_then(|slot| values.get(slot).copied())
        .ok_or_else(|| offset_error(offset, values.len(), op))
}

/// Visit every C-order multi-index of `shape`: a rank-0 shape visits the empty
/// index once and an empty shape dimension visits nothing.
fn for_each_index(shape: &[usize], mut body: impl FnMut(&[usize]) -> Result<()>) -> Result<()> {
    if shape.is_empty() {
        return body(&[]);
    }
    let mut index = vec![0usize; shape.len()];
    for _ in 0..util::numel(shape) {
        body(&index)?;
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    Ok(())
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

/// Read an index tensor as widened integers (any integer dtype, numpy accepts
/// them all; `require_integer_indices` has already rejected the rest).
fn index_values(indices: &Buffer) -> Result<Vec<i128>> {
    with_type!(indices, |values| {
        Ok(values
            .iter()
            .map(|value| match value.to_scalar() {
                ScalarValue::I(value) => value as i128,
                ScalarValue::U(value) => value as i128,
                ScalarValue::F(value) => value as i128,
                ScalarValue::C(value) => value.re as i128,
            })
            .collect())
    })
}

/// The `i128` view of one typed index scalar — exactly [`index_values`]'s
/// per-element conversion, without materializing the whole widened vector.
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

    /// The zero-fill-free block copy must produce exactly the same storage as
    /// `buffer::block_storage` (which zero-fills and then copies), including the
    /// empty case and single-element runs.
    #[test]
    fn block_rows_without_zero_fill_match_the_general_helper() {
        let values: Vec<f32> = (0..12).map(|value| value as f32).collect();
        let cases: [(&[usize], &[usize], usize); 3] = [
            (&[2, 3], &[1, 3], 3),
            (&[0, 3], &[], 3),
            (&[4], &[0, 1, 2, 3], 1),
        ];
        for (out_shape, offsets, inner) in cases {
            let fast = block_storage_from(&values, out_shape, offsets, inner, "test").unwrap();
            let general =
                crate::buffer::block_storage(&values, out_shape, offsets, inner, "test").unwrap();
            assert_eq!(fast, general, "shape {out_shape:?} offsets {offsets:?}");
        }
        // A run of two starts at offset 3: [3, 4].
        let storage = block_storage_from(&values, &[2], &[3, 7], 1, "test").unwrap();
        assert_eq!(storage, Storage::F32(vec![3., 7.]));
    }
    #[test]
    fn block_rows_reject_an_offset_outside_the_operand() {
        let values: Vec<f32> = (0..4).map(|value| value as f32).collect();
        // A count mismatch between the offset list and the result element count.
        let err = block_storage_from(&values, &[2], &[0], 1, "gather").unwrap_err();
        assert!(
            err.message.contains("a block copy reads outside the operand"),
            "{}",
            err.message
        );
        // A run past the end of the operand.
        let err = block_storage_from(&values, &[2, 2], &[0, 3], 2, "gather").unwrap_err();
        assert!(
            err.message.contains("op 'gather': a block copy reads outside the operand"),
            "{}",
            err.message
        );
        // Zero-width runs need no source data at all.
        let empty = block_storage_from::<f32>(&[], &[3, 0], &[0, 0, 0], 0, "gather").unwrap();
        assert_eq!(empty, Storage::F32(Vec::new()));
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
