//! `pad` — constant padding, with `padding_config[i] = (before, after)` per
//! axis.

use evoxc_core::{AttrValue, Error, Result};
use serde_json::Value as Json;

use crate::kernels::util;
use crate::plan::OpSpec;

use super::{as_json, each_index};

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

/// `pad`'s index map: shift every axis by `before`, leaving everything that
/// falls outside the operand as the fill value.
pub(super) fn pad_map(
    op: &str,
    in_shape: &[usize],
    out_shape: &[usize],
    config: &[(usize, usize)],
) -> Result<Vec<Option<usize>>> {
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
    let in_strides = util::c_strides(in_shape);
    let mut map = Vec::with_capacity(util::numel(out_shape));
    each_index(out_shape, |index| {
        let mut offset = 0usize;
        let mut inside = true;
        for (axis, &extent) in in_shape.iter().enumerate() {
            let before = config[axis].0;
            let position = index[axis];
            if position < before || position - before >= extent {
                inside = false;
                break;
            }
            offset += (position - before) * in_strides[axis];
        }
        map.push(if inside { Some(offset) } else { None });
    });
    Ok(map)
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
}
