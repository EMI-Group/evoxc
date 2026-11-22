//! `flip` and `roll` — the shape-preserving index maps.

use evoxc_core::{Error, Result};

use crate::kernels::util;
use crate::plan::OpSpec;

use super::{each_index, identity_shape, int_list};

/// `flip`'s normalized axes: absent/`null` = every axis.
pub(super) fn flip_axes(spec: &OpSpec, rank: usize) -> Result<Vec<usize>> {
    match int_list(spec, "axes")? {
        None => Ok((0..rank).collect()),
        Some(axes) => util::normalize_axes(&axes, rank, "flip.axes"),
    }
}

/// `flip`'s index map: reverse the position along each requested axis.
pub(super) fn flip_map(
    op: &str,
    in_shape: &[usize],
    out_shape: &[usize],
    axes: &[usize],
) -> Result<Vec<Option<usize>>> {
    identity_shape(op, in_shape, out_shape)?;
    let in_strides = util::c_strides(in_shape);
    let mut map = Vec::with_capacity(util::numel(out_shape));
    each_index(out_shape, |index| {
        let mut offset = 0usize;
        for (axis, &extent) in in_shape.iter().enumerate() {
            let position = if axes.contains(&axis) {
                extent - 1 - index[axis]
            } else {
                index[axis]
            };
            offset += position * in_strides[axis];
        }
        map.push(Some(offset));
    });
    Ok(map)
}

/// `roll`'s index map.  `axis` `null` is the flattened roll (`np.roll`'s
/// ravel → roll → reshape path); otherwise each axis wraps its own shift.
pub(super) fn roll_map(
    op: &str,
    in_shape: &[usize],
    out_shape: &[usize],
    spec: &OpSpec,
) -> Result<Vec<Option<usize>>> {
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
        return Ok(Vec::new());
    }
    let in_strides = util::c_strides(in_shape);
    let mut map = Vec::with_capacity(total);
    match stepped {
        None => {
            // numpy folds a multi-entry shift to its sum for a flattened roll.
            let folded: i128 = shift.iter().map(|amount| i128::from(*amount)).sum();
            let amount = folded.rem_euclid(total as i128) as usize;
            each_index(out_shape, |index| {
                let flat = util::flat_index(index, &in_strides);
                map.push(Some((flat + total - amount) % total));
            });
        }
        Some(stepped) => {
            each_index(out_shape, |index| {
                let mut offset = 0usize;
                for (axis, &extent) in in_shape.iter().enumerate() {
                    let position = index[axis];
                    let source = if stepped[axis] == 0 {
                        position
                    } else {
                        (position + extent - stepped[axis]) % extent
                    };
                    offset += source * in_strides[axis];
                }
                map.push(Some(offset));
            });
        }
    }
    Ok(map)
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

#[cfg(test)]
mod tests {
    use crate::kernels::structure::test_support::*;
    use crate::Storage;
    use evoxc_core::{AttrValue, DType};

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
