//! `scatter` kernel: functional numpy `put_along_axis`.
//!
//! A COPY of the input is updated, low-rank indices pad to full rank (0-d →
//! all-ones) and `updates` broadcasts against numpy's advanced-indexing result.
//! DUPLICATE INDICES ARE NOT REJECTED — like numpy, the last write in C order
//! wins. The module doc in [`super`] carries the shared declared-shape and
//! error-wording rules.
use evoxc_core::{DType, Error, Result};

use crate::buffer::{shape_text, Buffer};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::plan::OpSpec;
use crate::scalar::Scalar;
use crate::with_same_type;

use super::{
    for_each_index, index_values, offset_error, operand, reconcile, require_integer_indices,
    resolve_index,
};

pub(super) fn scatter_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    let source = operand("scatter", operands, 0)?;
    let indices = operand("scatter", operands, 1)?;
    let updates = operand("scatter", operands, 2)?;
    require_integer_indices("scatter", indices)?;
    let axis = util::normalize_axis(
        spec.attrs.opt_i64("axis")?.unwrap_or(0),
        source.rank(),
        "scatter",
    )?;
    let declared = ctx.result_shape(spec, 0)?;
    let index_values = index_values(indices)?;
    let cast = updates_for(source, updates)?;
    single(scatter(
        source,
        &index_values,
        &indices.shape,
        cast.as_ref().unwrap_or(updates),
        axis,
        &declared,
    )?)
}

/// `put_along_axis` writes into the data's own dtype, so a differing update
/// dtype is converted first (numpy would reject the mismatch outright; the
/// declared result dtype is already the data's).
fn updates_for(source: &Buffer, updates: &Buffer) -> Result<Option<Buffer>> {
    if updates.dtype == source.dtype {
        Ok(None)
    } else {
        Ok(Some(updates.cast(source.dtype)?))
    }
}

/// The interpreter's rank normalization for scatter index tensors: 0-d indices
/// become all-ones (the `scan` stack-update pattern), a rank-`r` tensor is used
/// as-is, and a lower-rank one is padded with `1`s around `axis`.
fn full_rank_indices(index_shape: &[usize], rank: usize, axis: usize) -> Result<Vec<usize>> {
    if index_shape.len() > rank {
        return Err(Error::shape(
            "scatter: `indices` and `arr` must have the same number of dimensions",
        ));
    }
    if index_shape.is_empty() {
        return Ok(vec![1; rank]);
    }
    if index_shape.len() == rank {
        return Ok(index_shape.to_vec());
    }
    let mut shape = vec![1usize; axis];
    shape.extend_from_slice(index_shape);
    shape.resize(rank, 1);
    Ok(shape)
}

/// numpy's advanced-indexing broadcast: along `axis` the extent is the index
/// tensor's, elsewhere the data and index extents broadcast (equal or 1).
fn selection_shape(source: &Buffer, index_shape: &[usize], axis: usize) -> Result<Vec<usize>> {
    let mut selection = vec![0usize; index_shape.len()];
    for (position, (data, index)) in source.shape.iter().zip(index_shape).enumerate() {
        selection[position] = if position == axis || data == index || *data == 1 {
            *index
        } else if *index == 1 {
            *data
        } else {
            return Err(Error::shape(format!(
                "scatter: shape mismatch: indexing arrays could not be broadcast together (data dim {position} is {data}, indices dim {position} is {index})"
            )));
        };
    }
    Ok(selection)
}

fn scatter(
    source: &Buffer,
    indices: &[i128],
    index_shape: &[usize],
    updates: &Buffer,
    axis: usize,
    declared: &[Option<usize>],
) -> Result<Buffer> {
    let rank = source.rank();
    let index_shape = full_rank_indices(index_shape, rank, axis)?;
    let selection = selection_shape(source, &index_shape, axis)?;
    if !util::broadcasts_to(&updates.shape, &selection) {
        return Err(Error::shape(format!(
            "scatter: shape mismatch: value array of shape {} could not be broadcast to indexing result of shape {}",
            shape_text(&updates.shape),
            shape_text(&selection)
        )));
    }
    let shape = reconcile("scatter", declared, source.shape.clone())?;
    with_same_type!(source, updates, |data, values| scatter_typed(
        data,
        values,
        source.dtype,
        &source.shape,
        axis,
        &index_shape,
        indices,
        &selection,
        &updates.shape,
        &shape
    ))
}

#[allow(clippy::too_many_arguments)]
fn scatter_typed<T: Scalar>(
    source: &[T],
    updates: &[T],
    dtype: DType,
    source_shape: &[usize],
    axis: usize,
    index_shape: &[usize],
    indices: &[i128],
    selection: &[usize],
    update_shape: &[usize],
    shape: &[usize],
) -> Result<Buffer> {
    let mut out = source.to_vec();
    let capacity = out.len();
    let update_len = updates.len();
    let source_strides = util::c_strides(source_shape);
    let index_strides = util::c_strides(index_shape);
    let update_strides = util::c_strides(update_shape);
    let extent = source_shape[axis];
    // `updates` broadcasts against the advanced-indexing result, so it aligns
    // with the TRAILING selection dims.
    let offset_rank = selection.len() - update_shape.len();
    for_each_index(selection, |index| {
        let mut index_offset = 0usize;
        for (position, stride) in index_strides.iter().enumerate() {
            let coordinate = if index_shape[position] == 1 {
                0
            } else {
                index[position]
            };
            index_offset += coordinate * stride;
        }
        let raw = *indices
            .get(index_offset)
            .ok_or_else(|| offset_error(index_offset, indices.len(), "scatter"))?;
        let selected = resolve_index(raw, extent, axis, "scatter")?;
        let mut destination = 0usize;
        for (position, stride) in source_strides.iter().enumerate() {
            let coordinate = if position == axis {
                selected
            } else if source_shape[position] == 1 {
                0
            } else {
                index[position]
            };
            destination += coordinate * stride;
        }
        let mut update_offset = 0usize;
        for (part, stride) in update_strides.iter().enumerate() {
            let coordinate = if update_shape[part] == 1 {
                0
            } else {
                index[offset_rank + part]
            };
            update_offset += coordinate * stride;
        }
        let value = updates
            .get(update_offset)
            .copied()
            .ok_or_else(|| offset_error(update_offset, update_len, "scatter"))?;
        let slot = out
            .get_mut(destination)
            .ok_or_else(|| offset_error(destination, capacity, "scatter"))?;
        *slot = value;
        Ok(())
    })?;
    Buffer::new(dtype, shape.to_vec(), T::into_storage(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::indexing::test_support::{attrs, declared, f32_buffer, spec};
    use crate::Storage;

    #[test]
    fn scatter_updates_a_copy_of_the_input() {
        let x = f32_buffer(&[3, 2], &[0.; 6]);
        let updates = f32_buffer(&[2, 2], &[1., 2., 3., 4.]);
        let out = scatter(&x, &[2, 0], &[2], &updates, 0, &declared(&[3, 2])).unwrap();
        assert_eq!(out.storage, Storage::F32(vec![3., 4., 0., 0., 1., 2.]));
        assert_eq!(x.storage, Storage::F32(vec![0.; 6]), "the input must not be mutated");
    }

    #[test]
    fn scatter_duplicate_indices_keep_the_last_write() {
        let x = f32_buffer(&[3], &[0., 0., 0.]);
        let updates = f32_buffer(&[2], &[1., 2.]);
        let out = scatter(&x, &[0, 0], &[2], &updates, 0, &declared(&[3])).unwrap();
        assert_eq!(out.storage, Storage::F32(vec![2., 0., 0.]));
    }

    #[test]
    fn scatter_rank_zero_indices_write_one_row() {
        let x = f32_buffer(&[3, 2], &[0.; 6]);
        let updates = f32_buffer(&[2], &[7., 8.]);
        let out = scatter(&x, &[1], &[], &updates, 0, &declared(&[3, 2])).unwrap();
        assert_eq!(out.storage, Storage::F32(vec![0., 0., 7., 8., 0., 0.]));
    }

    #[test]
    fn scatter_along_axis_one_uses_the_axis_attribute() {
        let x = f32_buffer(&[2, 3], &[0.; 6]);
        let updates = f32_buffer(&[2, 2], &[1., 2., 3., 4.]);
        let op = spec(attrs(&[("axis", evoxc_core::AttrValue::Int(1))]));
        let axis = util::normalize_axis(
            op.attrs.opt_i64("axis").unwrap().unwrap_or(0),
            2,
            "scatter",
        )
        .unwrap();
        assert_eq!(axis, 1);
        let out = scatter(&x, &[2, 0], &[2], &updates, axis, &declared(&[2, 3])).unwrap();
        // The rank-1 index tensor pads to (1, 2) around axis 1, so it
        // broadcasts across rows: every row writes columns 2 then 0.
        assert_eq!(out.storage, Storage::F32(vec![2., 0., 1., 4., 0., 3.]));
    }

    #[test]
    fn scatter_rejects_out_of_range_indices_and_bad_updates() {
        let x = f32_buffer(&[3], &[0.; 3]);
        let updates = f32_buffer(&[1], &[1.]);
        let err = scatter(&x, &[7], &[1], &updates, 0, &declared(&[3])).unwrap_err();
        assert!(err.message.contains("index 7 is out of bounds"), "{}", err.message);

        let updates = f32_buffer(&[5], &[0.; 5]);
        let err = scatter(&x, &[1], &[1], &updates, 0, &declared(&[3])).unwrap_err();
        assert!(err.message.contains("shape mismatch"), "{}", err.message);
    }
}
