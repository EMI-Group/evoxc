//! `pad` — constant padding, with `padding_config[i] = (before, after)` per
//! axis.
//!
//! The result is built row by row (one innermost-axis row at a time): a row's
//! leading/trailing pad cells are filled with the pad value and its interior is
//! ONE `copy_from_slice` out of the contiguous operand, so a rank-1 pad of a
//! 100 000-element operand costs two memcpys' worth of traffic instead of a
//! per-element index map.  A row whose outer index lands in a pad region is
//! filled whole.  The destination comes from the storage pool and every element
//! of it is written, so nothing is zero-filled first, and the two degenerate
//! configurations are flat copies: a rank-0 operand moves its single element
//! and an all-zero `padding_config` is one `copy_from_slice` of the whole
//! operand.
//!
//! This is the node's one surviving specialization, kept because it beats the
//! equivalent ndarray expression (`2 · rank` slab fills plus one
//! `ArrayViewD::assign` of the interior) by 1.2–1.6× of kernel time on the
//! measured rows — see `perf_probe`: 6.7 µs vs 11.1 µs for a 100 000-element
//! i32 rank-1 pad with `(2, 2)` and 31.7 µs vs 39.1 µs for
//! `(100, 50, 20) → (100, 52, 22)`.  `ndarray_pad_storage` below is that
//! expression, kept as the test-only reference the row copy is pinned against.

use evoxc_core::{AttrValue, Error, Result};
use serde_json::Value as Json;

use crate::buffer::{Buffer, NdScalar, Storage};
use crate::kernels::util;
use crate::plan::OpSpec;
use crate::scalar::{Scalar, ScalarValue};

use super::{as_json, pooled};

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
/// result shape, then fill the pad slabs and assign the interior.
pub(super) fn pad_result(
    op: &str,
    input: &Buffer,
    out_shape: &[usize],
    config: &[(usize, usize)],
    fill: f64,
) -> Result<Storage> {
    validate(op, &input.shape, out_shape, config)?;
    let storage = crate::with_type!(input, |values| pad_storage(
        values,
        &input.shape,
        out_shape,
        config,
        fill,
        op
    ))?;
    Ok(storage)
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

/// Fill the pad slabs of a pooled destination and assign the operand into the
/// destination's interior sub-view (`out_shape[axis] = before + extent + after`
/// on every axis, already validated).
fn pad_storage<T: Scalar + NdScalar>(
    values: &[T],
    in_shape: &[usize],
    out_shape: &[usize],
    config: &[(usize, usize)],
    fill: f64,
    op: &str,
) -> Result<Storage> {
    let total = util::numel(out_shape);
    let mut out = pooled::<T>(total);
    if total == 0 {
        return Ok(<T as NdScalar>::into_storage(out));
    }
    // numpy casts the pad constant to the operand dtype: `as` truncates toward
    // zero for integers and `from_scalar` is bool's truthiness conversion.
    let fill_value = T::from_scalar(ScalarValue::F(fill));
    let rank = in_shape.len();
    if rank == 0 {
        // A rank-0 operand has an empty config; the result is that one element.
        let value = values.first().copied().ok_or_else(|| internal(op))?;
        out[0] = value;
        return Ok(<T as NdScalar>::into_storage(out));
    }
    if values.len() != util::numel(in_shape) {
        return Err(internal(op));
    }
    if config
        .iter()
        .all(|&(before, after)| before == 0 && after == 0)
    {
        // No padding at all: the result IS the operand, so it is one bulk copy.
        out.copy_from_slice(values);
        return Ok(<T as NdScalar>::into_storage(out));
    }
    // The pad cells of an output row are its leading `before` and trailing
    // `after` cells, so one row of `out_row` cells is either entirely pad (its
    // outer index falls outside the operand) or `before` pad cells, ONE
    // contiguous run of `in_row` operand cells and `after` pad cells.
    let in_strides = util::c_strides(in_shape);
    let (in_row, out_row) = (in_shape[rank - 1], out_shape[rank - 1]);
    let before = config[rank - 1].0;
    let outer = &out_shape[..rank - 1];
    let rows = util::numel(outer);
    let mut index = vec![0usize; outer.len()];
    for row in 0..rows {
        // Resolve this row's outer index against the operand: `None` means the
        // row is pad-only, `Some(offset)` the flat offset of its first interior
        // cell (the innermost axis is contiguous, so one run follows).
        let mut source = Some(0usize);
        for (axis, &position) in index.iter().enumerate() {
            let offset = config[axis].0;
            let extent = in_shape[axis];
            if position < offset || position - offset >= extent {
                source = None;
                break;
            }
            if let Some(source) = source.as_mut() {
                *source += (position - offset) * in_strides[axis];
            }
        }
        let start = row * out_row;
        let dst = out
            .get_mut(start..start + out_row)
            .ok_or_else(|| internal(op))?;
        match source {
            Some(source) => {
                let interior = values
                    .get(source..source + in_row)
                    .ok_or_else(|| internal(op))?;
                dst[..before].fill(fill_value);
                dst[before..before + in_row].copy_from_slice(interior);
                dst[before + in_row..].fill(fill_value);
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
    Ok(<T as NdScalar>::into_storage(out))
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
    use crate::buffer::{array_view, array_view_mut};
    use ndarray::{Axis, Slice};
    use crate::Storage;
    use evoxc_core::{AttrValue, DType};

    /// An INDEPENDENT per-element reference for constant padding: output
    /// multi-index `i` reads the operand at `i - before` when every axis is
    /// inside the operand, the pad value otherwise (plain index arithmetic in
    /// the test, not another driver).
    fn reference_pad(
        values: &[f64],
        in_shape: &[usize],
        out_shape: &[usize],
        config: &[(usize, usize)],
        fill: f64,
    ) -> Vec<f64> {
        let rank = in_shape.len();
        let total: usize = out_shape.iter().product();
        let mut out = Vec::with_capacity(total);
        for flat in 0..total {
            let mut rest = flat;
            let mut index = vec![0usize; rank];
            for axis in (0..rank).rev() {
                index[axis] = rest % out_shape[axis].max(1);
                rest /= out_shape[axis].max(1);
            }
            let mut offset = 0usize;
            let mut stride = 1usize;
            let mut inside = true;
            for axis in (0..rank).rev() {
                let before = config[axis].0;
                let extent = in_shape[axis];
                let position = index[axis] as i64 - before as i64;
                if position < 0 || position as usize >= extent {
                    inside = false;
                    break;
                }
                offset += position as usize * stride;
                stride *= extent;
            }
            out.push(if inside { values[offset] } else { fill });
        }
        out
    }

    /// The ndarray expression for constant padding the row copy is pinned
    /// against: fill the `2 · rank` slabs the padding adds around the operand,
    /// then assign the operand into the destination's interior sub-view.
    fn ndarray_pad_storage<T: Scalar + NdScalar>(
        values: &[T],
        in_shape: &[usize],
        out_shape: &[usize],
        config: &[(usize, usize)],
        fill: f64,
    ) -> Storage {
        let total: usize = out_shape.iter().product();
        let mut out = vec![T::zero(); total];
        if total > 0 {
            let fill_value = T::from_scalar(ScalarValue::F(fill));
            let src = array_view::<T>(values, in_shape).expect("reference: operand view");
            if in_shape.is_empty() {
                out[0] = *src.first().expect("reference: rank-0 element");
            } else {
                let mut dst =
                    array_view_mut::<T>(&mut out, out_shape).expect("reference: result view");
                for (axis, &(before, after)) in config.iter().enumerate() {
                    if before > 0 {
                        dst.slice_axis_mut(Axis(axis), Slice::from(..before))
                            .fill(fill_value);
                    }
                    if after > 0 {
                        dst.slice_axis_mut(Axis(axis), Slice::from(out_shape[axis] - after..))
                            .fill(fill_value);
                    }
                }
                for (axis, &(before, _)) in config.iter().enumerate() {
                    dst.slice_axis_inplace(Axis(axis), Slice::from(before..before + in_shape[axis]));
                }
                dst.assign(&src);
            }
        }
        <T as NdScalar>::into_storage(out)
    }

    /// The row-wise copy must place exactly what the ndarray expression places,
    /// over every rank / padding / empty-extent shape family it specialises.
    #[test]
    fn the_row_copy_agrees_with_the_ndarray_reference() {
        type Case = (&'static [usize], &'static [usize], &'static [(usize, usize)]);
        let cases: &[Case] = &[
            (&[5], &[9], &[(2, 2)]),
            (&[1], &[1], &[(0, 0)]),
            (&[3], &[3], &[(0, 0)]),
            (&[3, 4], &[4, 6], &[(1, 0), (0, 2)]),
            (&[2, 3], &[2, 3], &[(0, 0), (0, 0)]),
            (&[4, 5], &[6, 7], &[(1, 1), (1, 1)]),
            (&[2, 3, 2], &[3, 5, 4], &[(1, 0), (1, 1), (1, 1)]),
            (&[2, 1, 3], &[3, 2, 3], &[(1, 0), (0, 1), (0, 0)]),
            (&[2, 2, 2, 2], &[3, 2, 4, 3], &[(0, 1), (0, 0), (1, 1), (0, 1)]),
            (&[0, 3], &[2, 3], &[(1, 1), (0, 0)]),
            (&[3, 0], &[4, 2], &[(0, 1), (1, 1)]),
            (&[], &[], &[]),
        ];
        for &(in_shape, out_shape, config) in cases {
            let count: usize = in_shape.iter().product();
            let values: Vec<f64> = (0..count).map(|slot| slot as f64 - 3.0).collect();
            let got = pad_storage::<f64>(&values, in_shape, out_shape, config, 0.25, "pad")
                .expect("the row copy accepts a validated config");
            let want = ndarray_pad_storage::<f64>(&values, in_shape, out_shape, config, 0.25);
            assert_eq!(
                got, want,
                "pad {in_shape:?} -> {out_shape:?} with {config:?} disagrees with ndarray"
            );
            // And the same shape family in a bounded integer dtype, where the
            // pad value truncates instead of being representable as-is.
            let ints: Vec<i32> = (0..count).map(|slot| slot as i32 - 3).collect();
            let got = pad_storage::<i32>(&ints, in_shape, out_shape, config, 1.7, "pad")
                .expect("the row copy accepts a validated config");
            let want = ndarray_pad_storage::<i32>(&ints, in_shape, out_shape, config, 1.7);
            assert_eq!(got, want, "pad i32 {in_shape:?} -> {out_shape:?}");
        }
    }

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

    /// A rank-1 pad of a large operand: only the two pad cells are written
    /// separately and the interior is one assignment out of the operand.
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

    /// The ndarray expression is the reference: a rank-3 pad with asymmetric
    /// pads, and the two degenerate configurations (no padding at all, and a
    /// rank-0 operand) against an independent per-element reference.
    #[test]
    fn pad_matches_the_per_element_reference() {
        let in_shape = [2usize, 3, 2];
        let values: Vec<f64> = (0..12).map(|index| index as f64 * 0.5 - 1.0).collect();
        let config = [(1usize, 0usize), (0, 2), (2, 1)];
        let out_shape = [3usize, 5, 5];
        let expected = reference_pad(&values, &in_shape, &out_shape, &config, 7.5);
        let storage = pad_storage(&values, &in_shape, &out_shape, &config, 7.5, "pad").unwrap();
        match &storage {
            Storage::F64(got) => {
                assert_eq!(got.len(), expected.len());
                for (index, (value, want)) in got.iter().zip(&expected).enumerate() {
                    assert_eq!(value.to_bits(), want.to_bits(), "element {index}");
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
        // The no-padding degenerate configuration is the operand itself.
        let identity = pad_storage(&values, &in_shape, &in_shape, &[(0, 0); 3], 9.0, "pad").unwrap();
        assert_eq!(identity, Storage::F64(values.clone()));
        // A rank-0 operand pads to its own element.
        let scalar = pad_storage(&[4.5f32], &[], &[], &[], 9.0, "pad").unwrap();
        assert_eq!(scalar, Storage::F32(vec![4.5]));
    }

    /// Every dtype pads with its own dtype's conversion of the constant: bool
    /// truthiness, integer truncation, f16 rounding and a complex zero-imaginary
    /// fill.
    #[test]
    fn pad_fills_every_dtype() {
        let bools = buffer(DType::Bool, &[2], Storage::Bool(vec![true, false]));
        let spec = op(
            "pad",
            DType::Bool,
            &[4],
            vec![
                ("padding_config", AttrValue::NestedInts(vec![vec![1, 1]])),
                ("value", AttrValue::Float(0.0)),
            ],
        );
        assert_eq!(
            run(&spec, &bools, &[4]).storage,
            Storage::Bool(vec![false, true, false, false])
        );
        let halves = buffer(
            DType::Float16,
            &[1],
            Storage::F16(vec![half::f16::from_f32(2.5)]),
        );
        let spec = op(
            "pad",
            DType::Float16,
            &[3],
            vec![
                ("padding_config", AttrValue::NestedInts(vec![vec![1, 1]])),
                ("value", AttrValue::Float(-1.25)),
            ],
        );
        assert_eq!(
            run(&spec, &halves, &[3]).storage,
            Storage::F16(vec![
                half::f16::from_f32(-1.25),
                half::f16::from_f32(2.5),
                half::f16::from_f32(-1.25),
            ])
        );
        let complex = buffer(
            DType::Complex64,
            &[1],
            Storage::C64(vec![num_complex::Complex::new(1.0, -2.0)]),
        );
        let spec = op(
            "pad",
            DType::Complex64,
            &[2],
            vec![
                ("padding_config", AttrValue::NestedInts(vec![vec![1, 0]])),
                ("value", AttrValue::Float(3.5)),
            ],
        );
        assert_eq!(
            run(&spec, &complex, &[2]).storage,
            Storage::C64(vec![
                num_complex::Complex::new(3.5, 0.0),
                num_complex::Complex::new(1.0, -2.0),
            ])
        );
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
