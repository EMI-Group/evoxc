//! Structure kernels: `tile`, `flip`, `roll`, `pad` and `diag`.
//!
//! Every one of them is an `ndarray` expression over a view of the operand —
//! there is no hand-written N-dimensional offset walk (and no per-element index
//! map) in this module, with ONE measured exception: `pad`'s row copy (see the
//! `pad` bullet), which beats the equivalent view assignment and is pinned
//! against it in `pad`'s tests.
//!
//! * `tile` — the promoted operand is viewed with a size-1 axis interleaved
//!   before every axis and broadcast to `[rep, extent, rep, extent, ...]`; that
//!   broadcast's C order *is* `np.tile`'s `index % extent` mapping, so one
//!   assignment into a pooled destination produces the tiling ([`tile`]).
//! * `flip` — `slice_each_axis` with a `-1` step on every requested axis.
//! * `roll` — the two wrapped halves of a `concatenate` per axis (or of the
//!   flattened operand for a null `axis`).
//! * `diag` — the `diag`/`diag_mut` views of a rank-2 array (the rank-1 case
//!   fills zeros and then assigns the vector onto the main diagonal).
//! * `pad` — one innermost-axis row at a time: a row is either pad-only (one
//!   `fill`) or `before` pad cells, one `copy_from_slice` of its interior run
//!   out of the contiguous operand and `after` pad cells ([`pad`]).  An
//!   all-zero config is one bulk copy and a rank-0 operand one element.
//!
//! The results live in C-contiguous [`Storage`] vectors (the engine has no
//! views), so each op materializes into a pooled destination.
//!
//! numpy semantics reproduced exactly (the reference is
//! `evoxc/backends/numpy/kernels/structural.py` + `_pad` in `indexing.py`):
//!
//! * `tile` — `np.tile`'s leading-dimension promotion: the result rank is
//!   `max(operand rank, reps len)`, `reps` is padded with leading 1s and the
//!   operand with leading size-1 axes, then axis `a` repeats as
//!   `index % operand extent`.  A shorter `reps` therefore repeats the last
//!   axes and a longer `reps` adds new leading axes.
//! * `flip` — `np.flip`; `axes` absent/`null` means every axis, otherwise an
//!   int or a list of ints (negative axes allowed, duplicates rejected the way
//!   numpy's `normalize_axis_tuple` rejects them).
//! * `roll` — `np.roll` per axis; `axis` `null` flattens the operand, rolls
//!   the flat sequence (multi-entry shifts fold to their sum, exactly numpy's
//!   flattened-roll folding) and reshapes back.  A shift wider than the extent
//!   wraps modulo the extent and a negative shift rolls the other way.
//! * `pad` — constant padding only (the IR `pad` op IS constant padding), with
//!   `padding_config[i] = (before, after)` per axis.  The `float` `value`
//!   attribute is converted to the operand dtype the numpy way: truncation
//!   toward zero for integers, truthiness for bool.
//! * `diag` — rank 1 → an `(n, n)` matrix with the vector on the main diagonal
//!   and zeros elsewhere; rank 2 → the main diagonal of length
//!   `min(rows, cols)`.  dtype is preserved in both directions.
//!
//! Everything that cannot be executed faithfully is an explicit
//! `evoxc_core::Error` naming the op (a negative `reps` entry, a
//! `padding_config` of the wrong length or with a negative entry, an
//! out-of-range `roll`/`flip` axis, a `diag` rank other than 1 or 2, and a
//! declared result shape that contradicts the operands) — never a guessed
//! value and never a panic.

use evoxc_core::{AttrValue, Error, Result};
use serde_json::Value as Json;

use crate::buffer::{take_pooled_vec, Buffer, NdScalar, Storage};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::kernels::KernelMap;
use crate::plan::OpSpec;
use crate::scalar::Scalar;
use crate::with_type;

mod diag;
mod flip_roll;
mod pad;
mod tile;

#[cfg(test)]
mod test_support;

use diag::diag_storage;
use flip_roll::{flip_axes, flip_storage, roll_storage};
use pad::{pad_result, pad_value, padding};
use tile::{reps, tile_storage, tile_whole_repeat};

pub fn register(map: &mut KernelMap) {
    for name in ["tile", "flip", "roll", "pad", "diag"] {
        map.insert(name, structure_kernel);
    }
}

/// The shared kernel entry point: resolve the declared result shape, then run
/// the `spec.name`-selected ndarray materialization.
fn structure_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let Some(input) = operands.first() else {
        return Err(Error::verification(format!(
            "op '{}': expected one operand, got none",
            spec.name
        )));
    };
    let out_shape = util::result_shape(ctx, spec, 0)?;
    structural(spec, input, &out_shape)
}

/// Materialize `spec.name`'s result from `input` into a fresh buffer of
/// `out_shape`.
fn structural(spec: &OpSpec, input: &Buffer, out_shape: &[usize]) -> Result<Vec<Value>> {
    let name = spec.name.as_str();
    let storage = match name {
        "tile" => {
            let reps = reps(spec)?;
            // Whole-source repeats first: `reps` repeats only the leading axis,
            // so every output element is `source[i % source.len()]` and the
            // result is the operand copied whole, `reps[0]` times — one
            // `repeat_storage` instead of a broadcast pass.
            match tile_whole_repeat(name, input, out_shape, &reps)? {
                Some(storage) => storage,
                None => with_type!(input, |values| tile_storage(
                    values,
                    &input.shape,
                    out_shape,
                    &reps,
                    name
                ))?,
            }
        }
        "flip" => with_type!(input, |values| flip_storage(
            values,
            &input.shape,
            out_shape,
            &flip_axes(spec, input.rank())?,
            name
        ))?,
        "roll" => with_type!(input, |values| roll_storage(
            values,
            &input.shape,
            out_shape,
            spec,
            name
        ))?,
        "pad" => pad_result(name, input, out_shape, &padding(spec)?, pad_value(spec)?)?,
        "diag" => with_type!(input, |values| diag_storage(
            values,
            &input.shape,
            out_shape,
            name
        ))?,
        other => {
            return Err(Error::verification(format!(
                "unexpected structure op '{other}'"
            )))
        }
    };
    single(Buffer::new(input.dtype, out_shape.to_vec(), storage)?)
}

// ---------------------------------------------------------------------------
// ndarray materialization helpers
// ---------------------------------------------------------------------------

/// A pooled result vector of exactly `total` elements (a recycled stock is only
/// reused when the caller overwrites every element of it).
pub(super) fn pooled<T: Scalar + NdScalar>(total: usize) -> Vec<T> {
    take_pooled_vec::<T>(total).unwrap_or_else(|| vec![T::zero(); total])
}

/// Take a finished standard-layout (C-order) array's own allocation as this
/// dtype's [`Storage`] — a move, never a copy.
pub(super) fn storage_of<T: NdScalar>(array: ndarray::ArrayD<T>) -> Storage {
    // A `Storage` is always C-contiguous, and ndarray producers do not all
    // return a standard layout (`concatenate` of a dynamic-rank array returns a
    // non-standard one), so a non-C-order result is standardized — one copy.
    let array = if array.is_standard_layout() {
        array
    } else {
        array.as_standard_layout().into_owned()
    };
    let (values, offset) = array.into_raw_vec_and_offset();
    debug_assert!(
        offset.unwrap_or(0) == 0,
        "a materialized result starts at its allocation"
    );
    <T as NdScalar>::into_storage(values)
}

/// Reject a declared result shape that is not the operand's own shape (the
/// shape-preserving ops `flip` and `roll`).
fn identity_shape(op: &str, in_shape: &[usize], out_shape: &[usize]) -> Result<()> {
    if in_shape == out_shape {
        return Ok(());
    }
    Err(util::shape_error(op, "the operand", in_shape, out_shape))
}

fn same_shape(actual: &[usize], want: &[usize]) -> bool {
    actual == want
}

// ---------------------------------------------------------------------------
// Attribute reading (`any`/`ints` attributes arrive in several concrete forms)
// ---------------------------------------------------------------------------

/// A `Tuple`-marked attribute as its JSON list form; everything else is its
/// own JSON form (`null` for [`AttrValue::Null`]).
fn as_json(value: &AttrValue) -> Json {
    match value {
        AttrValue::Tuple(items) => Json::Array(items.iter().map(as_json).collect()),
        other => other.to_json(),
    }
}

/// An int-list attribute: an int, a list of ints, or the JSON/`ints`/tuple
/// spellings of either.  `None` when absent or `null`.
fn int_list(spec: &OpSpec, name: &str) -> Result<Option<Vec<i64>>> {
    let Some(value) = spec.attrs.get(name) else {
        return Ok(None);
    };
    let json = as_json(value);
    if json.is_null() {
        return Ok(None);
    }
    let items = match &json {
        Json::Number(number) => vec![number.as_i64().ok_or_else(|| {
            Error::verification(format!(
                "op '{}': attribute {name:?} must hold ints, got {json}",
                spec.name
            ))
        })?],
        Json::Array(entries) => entries
            .iter()
            .map(|entry| int_of(spec, name, entry))
            .collect::<Result<Vec<i64>>>()?,
        other => {
            return Err(Error::verification(format!(
                "op '{}': attribute {name:?} must be an int or a list of ints, got {other}",
                spec.name
            )))
        }
    };
    Ok(Some(items))
}

fn int_of(spec: &OpSpec, name: &str, value: &Json) -> Result<i64> {
    value.as_i64().ok_or_else(|| {
        Error::verification(format!(
            "op '{}': attribute {name:?} must hold ints, got {value}",
            spec.name
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::structure::test_support::*;
    use evoxc_core::DType;

    #[test]
    fn registers_exactly_the_structure_ops() {
        let mut map = KernelMap::new();
        register(&mut map);
        let mut names: Vec<&str> = map.keys().copied().collect();
        names.sort_unstable();
        assert_eq!(names, ["diag", "flip", "pad", "roll", "tile"]);
    }

    #[test]
    fn identity_ops_reject_a_contradicting_result_shape() {
        let input = i32_buffer(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        let spec = op("flip", DType::Int32, &[2, 4], vec![]);
        let text = message(&spec, &input, &[2, 4]);
        assert!(text.contains("flip"), "{text}");
        let spec = op("roll", DType::Int32, &[3, 2], vec![("shift", ints(&[1]))]);
        let text = message(&spec, &input, &[3, 2]);
        assert!(text.contains("roll"), "{text}");
    }
}
