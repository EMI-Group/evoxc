//! Structure kernels: `tile`, `flip`, `roll`, `pad` and `diag`.
//!
//! `tile`, `flip`, `roll` and `diag` are pure relabellings of the operand's
//! elements (a rank-1 `diag` adds zeros), so one shape-driven index map plus
//! one typed copy serves them all:
//!
//! * the map is built from the DECLARED result shape ([`util::result_shape`]),
//!   never re-derived, and holds exactly one entry per result element;
//! * `Some(offset)` copies the operand element at C-order `offset`, `None`
//!   leaves the destination's zero.
//!
//! `pad` is the exception: constant padding is a row-wise copy (fill the pad
//! cells, then one `copy_from_slice` per operand row — see [`pad`]), because a
//! per-element index map costs a full odometer per pad cell and a second pass
//! over the whole result.
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

use crate::buffer::{Buffer, Storage};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::kernels::KernelMap;
use crate::plan::OpSpec;
use crate::scalar::{Scalar, ScalarValue};
use crate::with_type;

mod diag;
mod flip_roll;
mod pad;
mod tile;

#[cfg(test)]
mod test_support;

use diag::diag_map;
use flip_roll::{flip_axes, flip_map, roll_map};
use pad::{pad_result, pad_value, padding};
use tile::{reps, tile_blocks, tile_map, tile_whole_repeat};

pub fn register(map: &mut KernelMap) {
    for name in ["tile", "flip", "roll", "pad", "diag"] {
        map.insert(name, structure_kernel);
    }
}

/// The shared kernel entry point: resolve the declared result shape, then run
/// the `spec.name`-selected mapping.
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

/// Build the index map for `spec.name` and materialize the result.
fn structural(spec: &OpSpec, input: &Buffer, out_shape: &[usize]) -> Result<Vec<Value>> {
    let name = spec.name.as_str();
    let (map, fill) = match name {
        "tile" => {
            let reps = reps(spec)?;
            // Whole-source repeats first: `reps` repeats only the leading axis,
            // so every output element is `source[i % source.len()]` and the
            // result is the operand copied whole, `reps[0]` times.
            if let Some(storage) = tile_whole_repeat(name, input, out_shape, &reps)? {
                return single(Buffer::new(input.dtype, out_shape.to_vec(), storage)?);
            }
            // Contiguous operands: whole-run copies, no per-element index map.
            if let Some(storage) = tile_blocks(name, input, out_shape, &reps)? {
                return single(Buffer::new(input.dtype, out_shape.to_vec(), storage)?);
            }
            (tile_map(name, &input.shape, out_shape, &reps)?, None)
        }
        "flip" => (
            flip_map(name, &input.shape, out_shape, &flip_axes(spec, input.rank())?)?,
            None,
        ),
        "roll" => (roll_map(name, &input.shape, out_shape, spec)?, None),
        // Constant padding is a row-wise memcpy, not an index map: it fills the
        // pad cells and copies each operand row whole.
        "pad" => {
            return pad_result(name, input, out_shape, &padding(spec)?, pad_value(spec)?)
        }
        "diag" => (diag_map(name, &input.shape, out_shape)?, None),
        other => {
            return Err(Error::verification(format!(
                "unexpected structure op '{other}'"
            )))
        }
    };
    copy_mapped(spec, input, out_shape, &map, fill)
}

// ---------------------------------------------------------------------------
// Materializing one index map
// ---------------------------------------------------------------------------

/// Copy `input`'s elements through `map` into a fresh buffer of `out_shape`.
fn copy_mapped(
    spec: &OpSpec,
    input: &Buffer,
    out_shape: &[usize],
    map: &[Option<usize>],
    fill: Option<f64>,
) -> Result<Vec<Value>> {
    let total = util::numel(out_shape);
    if map.len() != total {
        return Err(Error::verification(format!(
            "op '{}': the index map covers {} element(s) but the result has {total}",
            spec.name,
            map.len()
        )));
    }
    with_type!(input, |values| {
        let storage = gather(values.as_slice(), total, map, fill)?;
        single(Buffer::new(input.dtype, out_shape.to_vec(), storage)?)
    })
}

/// The dtype-generic copy: pre-fill `total` elements, then overwrite the mapped
/// ones.  `None` in the map keeps the fill value.
fn gather<T: Scalar>(
    source: &[T],
    total: usize,
    map: &[Option<usize>],
    fill: Option<f64>,
) -> Result<Storage> {
    let mut out = match fill {
        // numpy casts `constant_values` to the operand dtype: `as` truncates
        // toward zero for integers and `truthy` is bool's conversion.
        Some(value) => vec![T::from_scalar(ScalarValue::F(value)); total],
        None => vec![T::zero(); total],
    };
    for (slot, entry) in out.iter_mut().zip(map) {
        if let Some(offset) = *entry {
            match source.get(offset) {
                Some(value) => *slot = *value,
                None => {
                    return Err(Error::verification(format!(
                        "structural index map reads element {offset} of a {} element(s) operand (internal engine error)",
                        source.len()
                    )))
                }
            }
        }
    }
    Ok(T::into_storage(out))
}

/// Iterate every C-order multi-index of `shape` (rank 0 yields one empty index;
/// an empty shape yields none).
fn each_index(shape: &[usize], mut body: impl FnMut(&[usize])) {
    let total = util::numel(shape);
    if total == 0 {
        return;
    }
    let mut index = vec![0usize; shape.len()];
    for _ in 0..total {
        body(&index);
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
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
