//! `flip` and `roll` — ndarray slice views: a `-1` step on every flipped axis,
//! and a `concatenate` of the two wrapped halves per rolled axis.

use evoxc_core::{Error, Result};
use ndarray::{concatenate, s, ArrayD, Axis, Slice};

use crate::buffer::{array_view, array_view_mut, NdScalar, Storage};
use crate::kernels::util;
use crate::plan::OpSpec;
use crate::scalar::Scalar;

use super::{identity_shape, int_list, pooled, storage_of};

/// `flip`'s normalized axes: absent/`null` = every axis.
pub(super) fn flip_axes(spec: &OpSpec, rank: usize) -> Result<Vec<usize>> {
    match int_list(spec, "axes")? {
        None => Ok((0..rank).collect()),
        Some(axes) => util::normalize_axes(&axes, rank, "flip.axes"),
    }
}

/// `flip`'s materialization: one `slice_each_axis` view with a `-1` step on
/// every requested axis, assigned into a pooled destination.
///
/// A zero-length axis keeps the full (empty) slice — there is nothing to
/// reverse, and a negative step from a non-existent last element is invalid.
pub(super) fn flip_storage<T: Scalar + NdScalar>(
    values: &[T],
    in_shape: &[usize],
    out_shape: &[usize],
    axes: &[usize],
    op: &str,
) -> Result<Storage> {
    identity_shape(op, in_shape, out_shape)?;
    let view = array_view::<T>(values, in_shape)?;
    let flipped = view.slice_each_axis(|axis| {
        if axes.contains(&axis.axis.index()) && axis.len > 0 {
            // numpy `flip`: read the axis back to front.  With a negative step
            // ndarray's slice is `[start, end)` walked from `end - 1` down, so
            // the full reversed axis is `0 .. end` with step -1.
            Slice::new(0, None, -1)
        } else {
            Slice::from(..)
        }
    });
    let total = util::numel(out_shape);
    let mut out = pooled::<T>(total);
    {
        let mut dst = array_view_mut::<T>(&mut out, out_shape)?;
        dst.assign(&flipped);
    }
    Ok(<T as NdScalar>::into_storage(out))
}

/// `roll`'s materialization.
///
/// `axis` `null` is the flattened roll (`np.roll`'s ravel → roll → reshape):
/// the operand is read as one 1-D view and the two halves of that sequence are
/// concatenated.  With axes, every axis whose shift is non-zero wraps its own
/// two halves; a single rolled axis is one pooled two-slice assignment into the
/// destination, and several rolled axes go through one `concatenate` per axis
/// (the rolls on distinct axes commute).
pub(super) fn roll_storage<T: Scalar + NdScalar>(
    values: &[T],
    in_shape: &[usize],
    out_shape: &[usize],
    spec: &OpSpec,
    op: &str,
) -> Result<Storage> {
    identity_shape(op, in_shape, out_shape)?;
    let shift = int_list(spec, "shift")?.ok_or_else(|| {
        Error::shape("roll: the required attribute \"shift\" is missing")
    })?;
    // Validate the axis/shift pair before the (possibly empty) mapping, so a
    // bad axis is reported even for a zero-element operand.
    let stepped = match int_list(spec, "axis")? {
        None => None,
        Some(axes) => Some(roll_amounts(&axes, &shift, in_shape)?),
    };
    let total = util::numel(in_shape);
    if total == 0 {
        return Ok(<T as NdScalar>::into_storage(Vec::new()));
    }
    let Some(amounts) = stepped else {
        // numpy folds a multi-entry shift to its sum for a flattened roll.
        let folded: i128 = shift.iter().map(|amount| i128::from(*amount)).sum();
        let amount = folded.rem_euclid(total as i128) as usize;
        let view = array_view::<T>(values, &[total])?;
        let mut out = pooled::<T>(total);
        {
            let mut dst = array_view_mut::<T>(&mut out, &[total])?;
            dst.slice_mut(s![..amount])
                .assign(&view.slice(s![total - amount..]));
            dst.slice_mut(s![amount..])
                .assign(&view.slice(s![..total - amount]));
        }
        return Ok(<T as NdScalar>::into_storage(out));
    };
    let rolled: Vec<(usize, usize)> = amounts
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, amount)| *amount != 0)
        .collect();
    if rolled.len() <= 1 {
        // Nothing moves (one bulk assignment) or exactly one axis wraps (two
        // slice assignments): both are one pooled destination.
        let view = array_view::<T>(values, in_shape)?;
        let mut out = pooled::<T>(total);
        {
            let mut dst = array_view_mut::<T>(&mut out, out_shape)?;
            match rolled.first() {
                None => dst.assign(&view),
                Some(&(axis, amount)) => {
                    let extent = in_shape[axis];
                    dst.slice_axis_mut(Axis(axis), Slice::from(0..amount)).assign(
                        &view.slice_axis(Axis(axis), Slice::from(extent - amount..extent)),
                    );
                    dst.slice_axis_mut(Axis(axis), Slice::from(amount..extent)).assign(
                        &view.slice_axis(Axis(axis), Slice::from(0..extent - amount)),
                    );
                }
            }
        }
        return Ok(<T as NdScalar>::into_storage(out));
    }
    // Several rolled axes: roll them one at a time through ndarray
    // `concatenate`, which is the same element mapping applied in sequence.
    let mut rolled_array: ArrayD<T> = array_view::<T>(values, in_shape)?.to_owned();
    for (axis, amount) in rolled {
        let extent = in_shape[axis];
        let next = {
            let head = rolled_array.slice_axis(Axis(axis), Slice::from(extent - amount..extent));
            let tail = rolled_array.slice_axis(Axis(axis), Slice::from(0..extent - amount));
            concatenate(Axis(axis), &[head, tail]).map_err(|_| internal(op))?
        };
        rolled_array = next;
    }
    Ok(storage_of(rolled_array))
}

/// The per-axis roll amount (already wrapped modulo each axis extent).
///
/// numpy broadcasts `shift` against `axis` — a length-1 `shift` (or a
/// length-1 `axis`) expands and a repeated axis accumulates — and an
/// out-of-range axis is an explicit `ShapeError`.
fn roll_amounts(axes: &[i64], shift: &[i64], in_shape: &[usize]) -> Result<Vec<usize>> {
    let rank = in_shape.len();
    let mut amounts = vec![0i128; rank];
    if axes.len() == shift.len() {
        for (axis, amount) in axes.iter().zip(shift) {
            let slot = util::normalize_axis(*axis, rank, "roll.axis")?;
            amounts[slot] += i128::from(*amount);
        }
    } else if shift.len() == 1 {
        for axis in axes {
            let slot = util::normalize_axis(*axis, rank, "roll.axis")?;
            amounts[slot] += i128::from(shift[0]);
        }
    } else if axes.len() == 1 {
        let slot = util::normalize_axis(axes[0], rank, "roll.axis")?;
        for amount in shift {
            amounts[slot] += i128::from(*amount);
        }
    } else {
        return Err(Error::shape(format!(
            "roll: shift has {} entries but axis has {} — numpy broadcasts only a length-1 shift or a length-1 axis",
            shift.len(),
            axes.len()
        )));
    }
    Ok(in_shape
        .iter()
        .zip(&amounts)
        .map(|(extent, amount)| {
            if *extent == 0 {
                0
            } else {
                amount.rem_euclid(*extent as i128) as usize
            }
        })
        .collect())
}

fn internal(op: &str) -> Error {
    Error::verification(format!(
        "op '{op}': the rolled views do not compose (internal engine error)"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::structure::test_support::*;
    use evoxc_core::{AttrValue, DType};

    /// An INDEPENDENT per-element reference for `np.roll`: every output
    /// multi-index reads the operand at `(i + extent - amount) % extent` per
    /// rolled axis (plain index arithmetic in the test, not another driver).
    fn reference_roll(
        values: &[i32],
        in_shape: &[usize],
        amounts: &[usize],
    ) -> Vec<i32> {
        let rank = in_shape.len();
        let total: usize = in_shape.iter().product();
        let mut out = Vec::with_capacity(total);
        for flat in 0..total {
            let mut rest = flat;
            let mut index = vec![0usize; rank];
            for axis in (0..rank).rev() {
                index[axis] = rest % in_shape[axis].max(1);
                rest /= in_shape[axis].max(1);
            }
            let mut offset = 0usize;
            let mut stride = 1usize;
            for axis in (0..rank).rev() {
                let extent = in_shape[axis];
                let source = (index[axis] + extent - amounts[axis]) % extent;
                offset += source * stride;
                stride *= extent;
            }
            out.push(values[offset]);
        }
        out
    }

    #[test]
    fn flip_all_axes_reverses_every_element() {
        let input = i32_buffer(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        let spec = op("flip", DType::Int32, &[2, 3], vec![]);
        let out = run(&spec, &input, &[2, 3]);
        assert_eq!(out.storage, Storage::I32(vec![5, 4, 3, 2, 1, 0]));
    }

    #[test]
    fn flip_null_axes_equals_all_axes() {
        let input = i32_buffer(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        let spec = op("flip", DType::Int32, &[2, 3], vec![("axes", AttrValue::Null)]);
        let out = run(&spec, &input, &[2, 3]);
        assert_eq!(out.storage, Storage::I32(vec![5, 4, 3, 2, 1, 0]));
    }

    #[test]
    fn flip_subset_of_axes_reverses_only_that_axis() {
        let input = i32_buffer(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        let spec = op("flip", DType::Int32, &[2, 3], vec![("axes", ints(&[-1]))]);
        let out = run(&spec, &input, &[2, 3]);
        assert_eq!(out.storage, Storage::I32(vec![2, 1, 0, 5, 4, 3]));
        // A rank-3 operand with two of three axes flipped: element (i, j, k)
        // reads the operand at (i, J-1-j, k).
        let values: Vec<i32> = (0..24).collect();
        let input = i32_buffer(&[2, 3, 4], &values);
        let spec = op("flip", DType::Int32, &[2, 3, 4], vec![("axes", ints(&[1]))]);
        let out = run(&spec, &input, &[2, 3, 4]);
        let mut expected = Vec::with_capacity(24);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    expected.push(values[(i * 3 + (2 - j)) * 4 + k]);
                }
            }
        }
        assert_eq!(out.storage, Storage::I32(expected));
    }

    #[test]
    fn flip_of_a_zero_length_axis_is_empty() {
        // numpy flips an empty axis without complaint, and the result is empty.
        let empty = i32_buffer(&[0, 3], &[]);
        let spec = op("flip", DType::Int32, &[0, 3], vec![("axes", ints(&[0]))]);
        let out = run(&spec, &empty, &[0, 3]);
        assert_eq!(out.storage, Storage::I32(vec![]));
        // ... even when every axis is flipped.
        let spec = op("flip", DType::Int32, &[0, 3], vec![]);
        let out = run(&spec, &empty, &[0, 3]);
        assert_eq!(out.storage, Storage::I32(vec![]));
    }

    #[test]
    fn flip_rejects_an_out_of_range_axis() {
        let input = i32_buffer(&[2], &[0, 1]);
        let spec = op("flip", DType::Int32, &[2], vec![("axes", ints(&[2]))]);
        let text = message(&spec, &input, &[2]);
        assert!(text.contains("flip.axes"), "{text}");
    }

    #[test]
    fn roll_with_an_axis_wraps_larger_and_negative_shifts() {
        let input = i32_buffer(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        // A shift wider than the extent wraps: 5 % 2 == 1.
        let spec = op("roll", DType::Int32, &[2, 3], vec![("shift", ints(&[5])), ("axis", ints(&[0]))]);
        let out = run(&spec, &input, &[2, 3]);
        assert_eq!(out.storage, Storage::I32(vec![3, 4, 5, 0, 1, 2]));
        // A negative shift rolls the other way: -1 on a 3-long axis moves the
        // elements left (numpy `np.roll([0, 1, 2], -1) == [1, 2, 0]`).
        let spec = op(
            "roll",
            DType::Int32,
            &[2, 3],
            vec![("shift", ints(&[-1])), ("axis", ints(&[1]))],
        );
        let out = run(&spec, &input, &[2, 3]);
        assert_eq!(out.storage, Storage::I32(vec![1, 2, 0, 4, 5, 3]));
    }

    #[test]
    fn roll_with_several_axes_matches_the_per_element_reference() {
        let values: Vec<i32> = (0..24).collect();
        let input = i32_buffer(&[2, 3, 4], &values);
        let spec = op(
            "roll",
            DType::Int32,
            &[2, 3, 4],
            vec![("shift", ints(&[1, 2])), ("axis", ints(&[0, 2]))],
        );
        let out = run(&spec, &input, &[2, 3, 4]);
        let expected = reference_roll(&values, &[2, 3, 4], &[1, 0, 2]);
        assert_eq!(out.storage, Storage::I32(expected));
        // ... and the single-axis case goes through the pooled path with the
        // same numbers.
        let spec = op(
            "roll",
            DType::Int32,
            &[2, 3, 4],
            vec![("shift", ints(&[1])), ("axis", ints(&[2]))],
        );
        let out = run(&spec, &input, &[2, 3, 4]);
        let expected = reference_roll(&values, &[2, 3, 4], &[0, 0, 1]);
        assert_eq!(out.storage, Storage::I32(expected));
    }

    #[test]
    fn roll_without_an_axis_flattens_then_reshapes() {
        let input = i32_buffer(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        // null axis: ravel, roll by 2, reshape back.
        let spec = op("roll", DType::Int32, &[2, 3], vec![("shift", ints(&[2]))]);
        let out = run(&spec, &input, &[2, 3]);
        assert_eq!(out.storage, Storage::I32(vec![4, 5, 0, 1, 2, 3]));
        // A negative flattened shift moves the other way (elements left).
        let spec = op("roll", DType::Int32, &[2, 3], vec![("shift", ints(&[-1]))]);
        let out = run(&spec, &input, &[2, 3]);
        assert_eq!(out.storage, Storage::I32(vec![1, 2, 3, 4, 5, 0]));
        // Multi-entry shifts fold to their sum for a flattened roll (numpy).
        let spec = op("roll", DType::Int32, &[2, 3], vec![("shift", ints(&[1, 1]))]);
        let out = run(&spec, &input, &[2, 3]);
        assert_eq!(out.storage, Storage::I32(vec![4, 5, 0, 1, 2, 3]));
        // A shift that is a whole number of wraps leaves the operand alone.
        let spec = op("roll", DType::Int32, &[2, 3], vec![("shift", ints(&[6]))]);
        let out = run(&spec, &input, &[2, 3]);
        assert_eq!(out.storage, Storage::I32(vec![0, 1, 2, 3, 4, 5]));
    }

    #[test]
    fn roll_of_axis_lengths_zero_and_one_is_identity() {
        // An extent-1 axis never moves, whatever the shift.
        let input = i32_buffer(&[1, 3], &[0, 1, 2]);
        let spec = op("roll", DType::Int32, &[1, 3], vec![("shift", ints(&[7])), ("axis", ints(&[0]))]);
        let out = run(&spec, &input, &[1, 3]);
        assert_eq!(out.storage, Storage::I32(vec![0, 1, 2]));
        // A zero-length operand rolls to itself (empty), axis or not.
        let empty = i32_buffer(&[0], &[]);
        let spec = op("roll", DType::Int32, &[0], vec![("shift", ints(&[3])), ("axis", ints(&[0]))]);
        assert_eq!(run(&spec, &empty, &[0]).storage, Storage::I32(vec![]));
        let spec = op("roll", DType::Int32, &[0], vec![("shift", ints(&[3]))]);
        assert_eq!(run(&spec, &empty, &[0]).storage, Storage::I32(vec![]));
    }

    #[test]
    fn roll_rejects_an_out_of_range_axis() {
        let input = i32_buffer(&[2], &[0, 1]);
        let spec = op("roll", DType::Int32, &[2], vec![("shift", ints(&[1])), ("axis", ints(&[3]))]);
        let text = message(&spec, &input, &[2]);
        assert!(text.contains("roll.axis"), "{text}");
        // The axis is validated even when the operand has no elements.
        let empty = i32_buffer(&[0], &[]);
        assert_eq!(message(&spec, &empty, &[0]), text);
    }
}

