//! `pad` — constant padding, with `padding_config[i] = (before, after)` per
//! axis.
//!
//! The result is built row by row (one innermost-axis row at a time): the
//! leading/trailing pad cells of a row are filled with the pad value and the
//! row's interior is ONE `copy_from_slice` out of the (contiguous) operand, so
//! a rank-1 pad of a 100 000-element operand costs two memcpys' worth of
//! traffic instead of a per-element index map.  A row whose outer index lands
//! in a pad region is filled whole.  The destination comes from the storage
//! pool and every element of it is written, so nothing is zero-filled first.

use evoxc_core::{AttrValue, Error, Result};
use serde_json::Value as Json;

use crate::buffer::{take_pooled_vec, Buffer, Storage};
use crate::exec::{single, Value};
use crate::kernels::util;
use crate::plan::OpSpec;
use crate::scalar::{Scalar, ScalarValue};
use crate::with_type;

use super::as_json;

/// `padding_config` as `(before, after)` extents per axis.
pub(super) fn padding(spec: &OpSpec) -> Result<Vec<(usize, usize)>> {
    let Some(value) = spec.attrs.get("padding_config") else {
        return Err(Error::shape(
            "pad: padding_config must be supplied (got None)",
        ));
    };
    let json = as_json(value);
    let entries = json.as_array().ok_or_else(|| {
        Error::shape(format!(
            "pad: padding_config must be a list of entries, got {json}"
        ))
    })?;
    let mut config = Vec::with_capacity(entries.len());
    for entry in entries {
        let (lo, hi) = if let Some(symmetric) = entry.as_i64() {
            (symmetric, symmetric)
        } else if let Some(pair) = entry.as_array() {
            match (
                pair.len(),
                pair.first().and_then(Json::as_i64),
                pair.get(1).and_then(Json::as_i64),
            ) {
                (2, Some(lo), Some(hi)) => (lo, hi),
                _ => return Err(invalid_pad_entry(entry)),
            }
        } else {
            return Err(invalid_pad_entry(entry));
        };
        if lo < 0 || hi < 0 {
            return Err(Error::shape(format!("pad: negative padding ({lo}, {hi})")));
        }
        let lo = usize::try_from(lo).map_err(|_| {
            Error::shape(format!("pad: padding {lo} does not fit the address space"))
        })?;
        let hi = usize::try_from(hi).map_err(|_| {
            Error::shape(format!("pad: padding {hi} does not fit the address space"))
        })?;
        config.push((lo, hi));
    }
    Ok(config)
}

fn invalid_pad_entry(entry: &Json) -> Error {
    Error::shape(format!(
        "pad: invalid padding entry {entry} (expected an int or a (lo, hi) pair)"
    ))
}

/// The constant pad value (the `float` attribute; absent/`null` is `0`).
pub(super) fn pad_value(spec: &OpSpec) -> Result<f64> {
    match spec.attrs.get("value") {
        None | Some(AttrValue::Null) | Some(AttrValue::Any(Json::Null)) => Ok(0.0),
        Some(AttrValue::Float(value)) => Ok(*value),
        Some(AttrValue::Any(Json::Number(number))) => number.as_f64().ok_or_else(|| {
            Error::verification(format!(
                "op 'pad': attribute \"value\" is not a finite number: {number}"
            ))
        }),
        Some(other) => Err(Error::verification(format!(
            "op 'pad': attribute \"value\" must be a number, got {other}"
        ))),
    }
}

/// The padded result of `input`: validate the config against the declared
/// result shape, then copy the interior rows and fill the pad cells.
pub(super) fn pad_result(
    op: &str,
    input: &Buffer,
    out_shape: &[usize],
    config: &[(usize, usize)],
    fill: f64,
) -> Result<Vec<Value>> {
    validate(op, &input.shape, out_shape, config)?;
    let storage = with_type!(input, |values| pad_storage(
        values,
        &input.shape,
        out_shape,
        config,
        fill,
        op
    ))?;
    single(Buffer::new(input.dtype, out_shape.to_vec(), storage)?)
}

/// Reject a config / declared shape that does not describe constant padding of
/// the operand (the messages are the engine's contract).
fn validate(
    op: &str,
    in_shape: &[usize],
    out_shape: &[usize],
    config: &[(usize, usize)],
) -> Result<()> {
    let rank = in_shape.len();
    if config.len() != rank {
        return Err(Error::shape(format!(
            "pad: padding_config must have {rank} entries, got {}",
            pairs_text(config)
        )));
    }
    if out_shape.len() != rank {
        return Err(util::shape_error(op, "the padded result", out_shape, in_shape));
    }
    for (axis, &(before, after)) in config.iter().enumerate() {
        let extent = in_shape[axis];
        let want = extent
            .checked_add(before)
            .and_then(|value| value.checked_add(after))
            .ok_or_else(|| {
                Error::shape(format!("pad: axis {axis} padding overflows the element count"))
            })?;
        if out_shape[axis] != want {
            return Err(Error::shape(format!(
                "pad: axis {axis} of the result is {} but the operand extent {extent} with padding ({before}, {after}) needs {want}",
                out_shape[axis]
            )));
        }
    }
    Ok(())
}

/// Fill `run`-long output rows: `element j of a row = operand[j - before]`
/// inside the interior, the pad value outside it.
fn pad_storage<T: Scalar>(
    values: &[T],
    in_shape: &[usize],
    out_shape: &[usize],
    config: &[(usize, usize)],
    fill: f64,
    op: &str,
) -> Result<Storage> {
    let total = util::numel(out_shape);
    let mut out = take_pooled_vec::<T>(total).unwrap_or_else(|| vec![T::zero(); total]);
    if total == 0 {
        return Ok(T::into_storage(out));
    }
    // numpy casts the pad constant to the operand dtype: `as` truncates toward
    // zero for integers and `from_scalar` is bool's truthiness conversion.
    let fill_value = T::from_scalar(ScalarValue::F(fill));
    let rank = in_shape.len();
    if rank == 0 {
        // A rank-0 operand has an empty config; the result is that one element.
        let value = values.first().copied().ok_or_else(|| internal(op))?;
        out[0] = value;
        return Ok(T::into_storage(out));
    }
    if config.iter().all(|&(before, after)| before == 0 && after == 0) {
        // No padding at all: the result IS the operand, so it is one bulk copy.
        if values.len() != total {
            return Err(internal(op));
        }
        out.copy_from_slice(values);
        return Ok(T::into_storage(out));
    }
    let in_strides = util::c_strides(in_shape);
    let (in_row, out_row) = (in_shape[rank - 1], out_shape[rank - 1]);
    let before = config[rank - 1].0;
    let outer = &out_shape[..rank - 1];
    let rows = util::numel(outer);
    let mut index = vec![0usize; outer.len()];
    for row in 0..rows {
        // Is this row's outer index inside the operand? If so, the row's
        // interior is a contiguous run of `in_row` elements at `source`.
        let mut source = Some(0usize);
        for (axis, &position) in index.iter().enumerate() {
            let offset = config[axis].0;
            let extent = in_shape[axis];
            if position < offset || position - offset >= extent {
                source = None;
                break;
            }
            if let Some(offset) = source.as_mut() {
                *offset += (position - config[axis].0) * in_strides[axis];
            }
        }
        let start = row * out_row;
        let dst = out.get_mut(start..start + out_row).ok_or_else(|| internal(op))?;
        match source {
            Some(source) => {
                let interior = values
                    .get(source..source + in_row)
                    .ok_or_else(|| internal(op))?;
                for slot in &mut dst[..before] {
                    *slot = fill_value;
                }
                dst[before..before + in_row].copy_from_slice(interior);
                for slot in &mut dst[before + in_row..] {
                    *slot = fill_value;
                }
            }
            None => dst.fill(fill_value),
        }
        // Advance the outer multi-index in C order (innermost axis last).
        for axis in (0..outer.len()).rev() {
            index[axis] += 1;
            if index[axis] < outer[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    Ok(T::into_storage(out))
}

fn internal(op: &str) -> Error {
    Error::verification(format!(
        "op '{op}': a padded row reads outside the operand (internal engine error)"
    ))
}

fn pairs_text(config: &[(usize, usize)]) -> String {
    let parts: Vec<String> = config
        .iter()
        .map(|(lo, hi)| format!("({lo}, {hi})"))
        .collect();
    format!("({})", parts.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::structure::test_support::*;
    use crate::Storage;
    use evoxc_core::{AttrValue, DType};

    #[test]
    fn pad_is_asymmetric_on_two_axes() {
        let input = i32_buffer(&[2, 2], &[1, 2, 3, 4]);
        let spec = op(
            "pad",
            DType::Int32,
            &[3, 4],
            vec![
                ("padding_config", AttrValue::NestedInts(vec![vec![1, 0], vec![0, 2]])),
                ("value", AttrValue::Float(9.0)),
            ],
        );
        let out = run(&spec, &input, &[3, 4]);
        assert_eq!(
            out.storage,
            Storage::I32(vec![9, 9, 9, 9, 1, 2, 9, 9, 3, 4, 9, 9])
        );
    }

    #[test]
    fn pad_converts_the_constant_value_the_numpy_way() {
        let input = i32_buffer(&[1], &[5]);
        let spec = op(
            "pad",
            DType::Int32,
            &[3],
            vec![
                ("padding_config", AttrValue::NestedInts(vec![vec![1, 1]])),
                ("value", AttrValue::Float(1.7)),
            ],
        );
        let out = run(&spec, &input, &[3]);
        assert_eq!(out.storage, Storage::I32(vec![1, 5, 1]));
        // bool: numpy casts through truthiness.
        let input = buffer(DType::Bool, &[1], Storage::Bool(vec![true]));
        let spec = op(
            "pad",
            DType::Bool,
            &[3],
            vec![
                ("padding_config", AttrValue::NestedInts(vec![vec![1, 1]])),
                ("value", AttrValue::Float(0.5)),
            ],
        );
        let out = run(&spec, &input, &[3]);
        assert_eq!(out.storage, Storage::Bool(vec![true, true, true]));
    }

    #[test]
    fn pad_rejects_a_mismatched_config_and_negative_padding() {
        let input = i32_buffer(&[2], &[1, 2]);
        let spec = op(
            "pad",
            DType::Int32,
            &[4],
            vec![("padding_config", AttrValue::NestedInts(vec![vec![1, 1], vec![0, 0]]))],
        );
        let text = message(&spec, &input, &[4]);
        assert!(text.contains("must have 1 entries"), "{text}");
        let spec = op(
            "pad",
            DType::Int32,
            &[4],
            vec![("padding_config", AttrValue::NestedInts(vec![vec![-1, 3]]))],
        );
        let text = message(&spec, &input, &[4]);
        assert!(text.contains("negative padding"), "{text}");
    }

    #[test]
    fn pad_rejects_a_result_shape_that_contradicts_the_padding() {
        let input = i32_buffer(&[2], &[1, 2]);
        let spec = op(
            "pad",
            DType::Int32,
            &[5],
            vec![("padding_config", AttrValue::NestedInts(vec![vec![1, 1]]))],
        );
        let text = message(&spec, &input, &[5]);
        assert!(
            text.contains("axis 0 of the result is 5 but the operand extent 2"),
            "{text}"
        );
        // A result rank that differs from the operand rank is a shape error too.
        let spec = op(
            "pad",
            DType::Int32,
            &[2, 3],
            vec![("padding_config", AttrValue::NestedInts(vec![vec![0, 0]]))],
        );
        let text = message(&spec, &input, &[2, 3]);
        assert!(text.contains("pad"), "{text}");
    }

    /// A rank-1 pad of a large operand: the interior is ONE copy out of the
    /// operand and only the two pad cells are written separately.
    #[test]
    fn pad_copies_a_long_row_and_fills_only_the_pad_cells() {
        const COUNT: usize = 100_000;
        let values: Vec<i32> = (0..COUNT).map(|index| index as i32 * 3 + 7).collect();
        let storage = pad_storage(&values, &[COUNT], &[COUNT + 4], &[(2, 2)], -5.0, "pad").unwrap();
        match &storage {
            Storage::I32(out) => {
                assert_eq!(out.len(), COUNT + 4);
                assert_eq!(&out[..2], &[-5, -5]);
                assert_eq!(&out[2..2 + COUNT], values.as_slice());
                assert_eq!(&out[2 + COUNT..], &[-5, -5]);
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }

    /// Every pad region of a rank-2 result lands on the right element, with the
    /// operand read through its own strides (asymmetric pads, both f32 and i32).
    #[test]
    fn pad_places_every_element_of_a_padded_result() {
        let input = i32_buffer(&[2, 3], &[1, 2, 3, 4, 5, 6]);
        // (2,3) padded to (3,6): one row before, none after; one column
        // before and two after.
        let out = pad_storage(&[1, 2, 3, 4, 5, 6], &[2, 3], &[3, 6], &[(1, 0), (1, 2)], 0.0, "pad")
            .unwrap();
        assert_eq!(
            out,
            Storage::I32(vec![
                0, 0, 0, 0, 0, 0, //
                0, 1, 2, 3, 0, 0, //
                0, 4, 5, 6, 0, 0, //
            ])
        );
        // The full kernel path produces the same buffer.
        let spec = op(
            "pad",
            DType::Int32,
            &[3, 6],
            vec![("padding_config", AttrValue::NestedInts(vec![vec![1, 0], vec![1, 2]]))],
        );
        assert_eq!(run(&spec, &input, &[3, 6]).storage, out);
        // The same operand with padding on both sides of both axes.
        let out = pad_storage(
            &[1.5f32, -2.5, 3.25, 0.0, -0.0, 8.0],
            &[2, 3],
            &[4, 5],
            &[(1, 1), (1, 1)],
            f64::NAN,
            "pad",
        )
        .unwrap();
        match &out {
            Storage::F32(values) => {
                assert_eq!(values.len(), 20);
                // Row 0 and row 3 are pure pad rows.
                assert!(values[..5].iter().all(|value| value.is_nan()));
                assert!(values[15..].iter().all(|value| value.is_nan()));
                assert!(values[5].is_nan() && values[9].is_nan() && values[10].is_nan());
                assert_eq!(&values[6..9], &[1.5, -2.5, 3.25]);
                assert_eq!(&values[11..14], &[0.0, -0.0, 8.0]);
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }

    /// The pooled destination of a pad result is fully overwritten: a recycled
    /// stock of the result's exact size leaves no stale element behind.
    #[test]
    fn a_pooled_pad_result_leaves_no_stale_element() {
        const COUNT: usize = 100_000;
        // Park a garbage stock of exactly the result's size in the pool.
        let garbage = i32_buffer(&[COUNT + 4], &vec![i32::MIN; COUNT + 4]);
        drop(garbage);
        let values: Vec<i32> = (0..COUNT).map(|index| index as i32).collect();
        let storage = pad_storage(&values, &[COUNT], &[COUNT + 4], &[(2, 2)], 11.0, "pad").unwrap();
        match &storage {
            Storage::I32(out) => {
                assert_eq!(out.len(), COUNT + 4);
                assert_eq!(&out[..2], &[11, 11]);
                for (index, value) in out[2..2 + COUNT].iter().enumerate() {
                    assert_eq!(*value, index as i32, "element {index}");
                }
                assert_eq!(&out[2 + COUNT..], &[11, 11]);
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }

    /// An empty operand padds to pure pad cells, and an empty result is empty —
    /// neither reads operand data nor errors.
    #[test]
    fn pad_handles_empty_extents() {
        let empty = pad_storage::<i32>(&[], &[0], &[2], &[(1, 1)], 3.0, "pad").unwrap();
        assert_eq!(empty, Storage::I32(vec![3, 3]));
        let none = pad_storage::<i32>(&[], &[0], &[0], &[(0, 0)], 0.0, "pad").unwrap();
        assert_eq!(none, Storage::I32(Vec::new()));
    }
}
