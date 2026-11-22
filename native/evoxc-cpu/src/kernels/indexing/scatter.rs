//! `scatter` kernel: functional numpy `put_along_axis`.
//!
//! A COPY of the input is updated, low-rank indices pad to full rank (0-d →
//! all-ones) and `updates` broadcasts against numpy's advanced-indexing result.
//! DUPLICATE INDICES ARE NOT REJECTED — like numpy, the last write in C order
//! wins. The module doc in [`super`] carries the shared declared-shape and
//! error-wording rules.
use evoxc_core::{DType, Error, Result};

use crate::buffer::{array_view, shape_text, take_pooled_vec, Buffer, NdScalar, Storage};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::plan::OpSpec;
use crate::scalar::Scalar;
use crate::with_same_type;
use crate::with_type;

use super::{
    index_as_i128, offset_error, operand, pooled_total, reconcile, require_integer_indices,
    resolve_index_fast,
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
    let cast = updates_for(source, updates)?;
    single(scatter(
        source,
        indices,
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
    indices: &Buffer,
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
    // The overwhelmingly common shape of a `put_along_axis` — an index tensor
    // that selects whole runs (its non-axis extents equal the data's) with a
    // matching `updates` — takes a typed path that reads the index buffer
    // directly (no `i128` widening) and has no per-element stride odometers;
    // every other (broadcast) selection keeps the general driver below, which
    // is the only place the widened indices are built.
    if let Some(storage) = scatter_selection(source, indices, &index_shape, updates, axis)? {
        return Buffer::new(source.dtype, shape, storage);
    }
    with_same_type!(source, updates, |data, values| scatter_typed(
        data,
        values,
        source.dtype,
        &source.shape,
        axis,
        indices,
        &index_shape,
        &selection,
        &updates.shape,
        shape.clone()
    ))
}

/// The no-broadcast `scatter` fast path: a full-rank index tensor whose
/// non-axis extents equal the data's, with `updates` exactly the
/// advanced-indexing selection.
///
/// The index tensor need NOT have full-rank-equal-to-data shape: it may select
/// an arbitrary number of runs along `axis` (the `scatter(vec[50500],
/// idx[25000], values[25000])` shape, where `axis` is the only extent that
/// differs) — that is exactly numpy's `put_along_axis` on a 1-D destination.
/// The general path walks every selected element through three rank-deep
/// stride loops (index, source, update) with a bounds-checked read and write
/// per element, on top of a whole-tensor `i128` widening of the index operand.
/// Here the typed index buffer is resolved ONCE into a compact `Vec<usize>`
/// (numpy's negative-index and bounds rules, unchanged error wording) and the
/// walk is one flat loop in the result's C order — which is what keeps numpy's
/// duplicate-index rule (the LAST write in C order wins).  `Ok(None)` when this
/// decomposition does not apply (a padded or broadcast index/update shape, an
/// empty operand) — the general driver owns those.
fn scatter_selection(
    source: &Buffer,
    indices: &Buffer,
    index_shape: &[usize],
    updates: &Buffer,
    axis: usize,
) -> Result<Option<Storage>> {
    if index_shape.len() != source.rank() || updates.shape.as_slice() != index_shape {
        return Ok(None);
    }
    // No broadcast anywhere but the gathered axis: an index extent of 1 against
    // a longer data extent (or the reverse) writes whole runs from one element
    // and keeps the general driver.
    for (position, (data, index)) in source.shape.iter().zip(index_shape).enumerate() {
        if position != axis && data != index {
            return Ok(None);
        }
    }
    let Some(&extent) = source.shape.get(axis) else {
        return Ok(None);
    };
    let Some(&selected) = index_shape.get(axis) else {
        return Ok(None);
    };
    let inner: usize = source.shape[axis + 1..].iter().product();
    let outer: usize = source.shape[..axis].iter().product();
    let total = source.numel();
    if total == 0 || extent == 0 || inner == 0 || selected == 0 {
        return Ok(None);
    }
    let Some(band) = extent.checked_mul(inner) else {
        return Ok(None);
    };
    let Some(count) = outer
        .checked_mul(selected)
        .and_then(|count| count.checked_mul(inner))
    else {
        return Ok(None);
    };
    if count != indices.numel() || count != updates.numel() {
        return Ok(None);
    }
    let built: Result<Storage> = with_type!(indices, |raw| {
        with_same_type!(source, updates, |data, values| scatter_into(
            data, values, raw, outer, extent, selected, inner, band, axis
        ))
    });
    built.map(Some)
}

/// Apply the resolved selection to a copy of `data` — one flat typed loop per
/// `inner` regime, in C order so the LAST write wins.
///
/// `resolved[block × selected × inner + axis_position × inner + offset]` is the
/// selected axis coordinate for that element, so every destination below lies
/// inside `out` and every source read inside `values` by construction.
#[allow(clippy::too_many_arguments)]
fn scatter_into<I: Scalar, T: Scalar>(
    data: &[T],
    values: &[T],
    indices: &[I],
    outer: usize,
    extent: usize,
    selected: usize,
    inner: usize,
    band: usize,
    axis: usize,
) -> Result<Storage> {
    let total = data.len();
    // Functional update: the destination is copied once (one bulk memcpy) into
    // a preallocated vector — a recycled allocation from the storage pool when
    // one of that exact size is available, fully written by the copy — and the
    // selection then overwrites the selected slots in place.
    let mut out = take_pooled_vec::<T>(total)
        .filter(|out| out.len() == total)
        .unwrap_or_else(|| T::zeros(total));
    out.copy_from_slice(data);
    if outer == 1 && inner == 1 {
        // The whole-axis full-selection form (`put_along_axis` on a rank-1
        // operand, the hot scatter shape): resolve each index where it is used,
        // so there is no intermediate `Vec<usize>` at all.
        for (position, raw) in indices.iter().enumerate() {
            let slot = resolve_index_fast(index_as_i128(*raw), extent, axis, "scatter")?;
            out[slot] = values[position];
        }
        return Ok(T::into_storage(out));
    }
    // Resolve every index exactly once, in the C order the general driver
    // visits it, so the first out-of-range index reported is the same one.
    let mut resolved = Vec::with_capacity(indices.len());
    for raw in indices {
        resolved.push(resolve_index_fast(index_as_i128(*raw), extent, axis, "scatter")?);
    }
    if inner == 1 {
        for block in 0..outer {
            let destination = block * band;
            let read = block * selected;
            for position in read..read + selected {
                out[destination + resolved[position]] = values[position];
            }
        }
    } else {
        for block in 0..outer {
            let destination = block * band;
            let read = block * selected * inner;
            // `inner` trailing elements follow each axis position, each with its
            // OWN index, so the write stays per element — only the odometer and
            // the per-element bounds checks are gone.
            for axis_position in 0..selected {
                let run = read + axis_position * inner;
                for offset in 0..inner {
                    let position = run + offset;
                    out[destination + resolved[position] * inner + offset] = values[position];
                }
            }
        }
    }
    Ok(T::into_storage(out))
}

/// The general (broadcast-capable) scatter, driven by ndarray views.
///
/// A pooled copy of the data is written in place; the destination strides come
/// from an ndarray view of that copy, and the index and `updates` operands are
/// lent as views BROADCAST onto the selection shape (`updates` against the
/// trailing dims), so a 1-extent index/data/update dim is ndarray's zero-stride
/// axis rather than a hand-written coordinate conditional. Iterating the
/// selection in ndarray's own C order keeps numpy's duplicate-index rule (the
/// LAST write wins) and the first-out-of-range-index diagnostic order.
#[allow(clippy::too_many_arguments)]
fn scatter_typed<T: Scalar + NdScalar>(
    source: &[T],
    updates: &[T],
    dtype: DType,
    source_shape: &[usize],
    axis: usize,
    indices: &Buffer,
    index_shape: &[usize],
    selection: &[usize],
    update_shape: &[usize],
    shape: Vec<usize>,
) -> Result<Buffer> {
    let mut out = pooled_total::<T>(source.len());
    out.copy_from_slice(source);
    let extent = source_shape[axis];
    let rank = source_shape.len();
    let source_strides: Vec<isize> = {
        let view = array_view(&out, source_shape)?;
        (0..rank)
            .map(|position| view.stride_of(ndarray::Axis(position)))
            .collect()
    };
    // `updates` aligns with the TRAILING selection dims.
    let mut padded = vec![1usize; selection.len() - update_shape.len()];
    padded.extend_from_slice(update_shape);
    let dimensions = ndarray::IxDyn(selection);
    let update_flat = array_view(updates, &padded)?;
    let update_view = update_flat
        .broadcast(dimensions.clone())
        .ok_or_else(|| broadcast_mismatch(update_shape, selection))?;
    with_type!(indices, |raw| {
        let index_flat = array_view(raw, index_shape)?;
        let index_view = index_flat
            .broadcast(dimensions.clone())
            .ok_or_else(|| broadcast_mismatch(index_shape, selection))?;
        for coordinate in ndarray::indices(dimensions.clone()) {
            let raw = index_view[&coordinate];
            let selected = resolve_index_fast(index_as_i128(raw), extent, axis, "scatter")?;
            let mut offset = 0isize;
            for position in 0..rank {
                let taken = if position == axis {
                    selected
                } else if source_shape[position] == 1 {
                    0
                } else {
                    coordinate[position]
                };
                offset += taken as isize * source_strides[position];
            }
            let capacity = out.len();
            let slot = out
                .get_mut(offset as usize)
                .ok_or_else(|| offset_error(offset, capacity, "scatter"))?;
            *slot = update_view[&coordinate];
        }
        Ok(())
    })?;
    Buffer::new(dtype, shape, <T as NdScalar>::into_storage(out))
}

/// numpy's shape-mismatch diagnostic for an operand ndarray cannot broadcast
/// onto the selection (the pre-check above already rejects the invalid ones, so
/// this is the defensive path that never panics).
fn broadcast_mismatch(operand_shape: &[usize], selection: &[usize]) -> Error {
    Error::shape(format!(
        "scatter: shape mismatch: value array of shape {} could not be broadcast to indexing result of shape {}",
        shape_text(operand_shape),
        shape_text(selection)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::indexing::test_support::{
        attrs, declared, f32_buffer, i32_buffer, i64_buffer, spec,
    };
    use crate::Storage;

    #[test]
    fn scatter_updates_a_copy_of_the_input() {
        let x = f32_buffer(&[3, 2], &[0.; 6]);
        let updates = f32_buffer(&[2, 2], &[1., 2., 3., 4.]);
        let indices = i64_buffer(&[2], &[2, 0]);
        let out = scatter(&x, &indices, &[2], &updates, 0, &declared(&[3, 2])).unwrap();
        assert_eq!(out.storage, Storage::F32(vec![3., 4., 0., 0., 1., 2.]));
        assert_eq!(x.storage, Storage::F32(vec![0.; 6]), "the input must not be mutated");
    }

    #[test]
    fn scatter_duplicate_indices_keep_the_last_write() {
        let x = f32_buffer(&[3], &[0., 0., 0.]);
        let updates = f32_buffer(&[2], &[1., 2.]);
        let indices = i64_buffer(&[2], &[0, 0]);
        let out = scatter(&x, &indices, &[2], &updates, 0, &declared(&[3])).unwrap();
        assert_eq!(out.storage, Storage::F32(vec![2., 0., 0.]));
    }

    #[test]
    fn scatter_rank_zero_indices_write_one_row() {
        let x = f32_buffer(&[3, 2], &[0.; 6]);
        let updates = f32_buffer(&[2], &[7., 8.]);
        let indices = i64_buffer(&[], &[1]);
        let out = scatter(&x, &indices, &[], &updates, 0, &declared(&[3, 2])).unwrap();
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
        let indices = i64_buffer(&[2], &[2, 0]);
        let out = scatter(&x, &indices, &[2], &updates, axis, &declared(&[2, 3])).unwrap();
        // The rank-1 index tensor pads to (1, 2) around axis 1, so it
        // broadcasts across rows: every row writes columns 2 then 0.
        assert_eq!(out.storage, Storage::F32(vec![2., 0., 1., 4., 0., 3.]));
    }

    #[test]
    fn scatter_rejects_out_of_range_indices_and_bad_updates() {
        let x = f32_buffer(&[3], &[0.; 3]);
        let updates = f32_buffer(&[1], &[1.]);
        let indices = i64_buffer(&[1], &[7]);
        let err = scatter(&x, &indices, &[1], &updates, 0, &declared(&[3])).unwrap_err();
        assert!(err.message.contains("index 7 is out of bounds"), "{}", err.message);

        let updates = f32_buffer(&[5], &[0.; 5]);
        let indices = i64_buffer(&[1], &[1]);
        let err = scatter(&x, &indices, &[1], &updates, 0, &declared(&[3])).unwrap_err();
        assert!(err.message.contains("shape mismatch"), "{}", err.message);
    }

    /// The flat index of `position`'s multi-index with the axis coordinate
    /// replaced by `selected` — the independent oracle for the duplicate-index
    /// assertion below (deliberately NOT the kernel's own arithmetic).
    fn flat_destination(shape: &[usize], axis: usize, mut position: usize, selected: usize) -> usize {
        let mut coordinates = vec![0usize; shape.len()];
        for dim in (0..shape.len()).rev() {
            coordinates[dim] = position % shape[dim];
            position /= shape[dim];
        }
        coordinates[axis] = selected;
        coordinates
            .iter()
            .zip(shape)
            .fold(0usize, |flat, (coordinate, extent)| flat * extent + coordinate)
    }

    fn f32_slot(storage: &Storage, slot: usize) -> f32 {
        match storage {
            Storage::F32(values) => values[slot],
            other => panic!("the fixture is f32, got {:?}", other.dtype()),
        }
    }

    /// The general (broadcast-capable) driver, for the same-dtype test inputs.
    fn general_scatter(
        source: &Buffer,
        updates: &Buffer,
        axis: usize,
        shape: &[usize],
        indices: &Buffer,
    ) -> Result<Buffer> {
        with_same_type!(source, updates, |data, update_data| scatter_typed(
            data,
            update_data,
            source.dtype,
            &source.shape,
            axis,
            indices,
            shape,
            shape,
            shape,
            shape.to_vec(),
        ))
    }

    /// [`general_scatter`] with explicit index/selection/update shapes — what a
    /// sub-shaped (not full-rank-equal-to-data) index tensor needs.
    #[allow(clippy::too_many_arguments)]
    fn general_scatter_shaped(
        source: &Buffer,
        updates: &Buffer,
        axis: usize,
        index_shape: &[usize],
        selection: &[usize],
        indices: &Buffer,
    ) -> Result<Buffer> {
        with_same_type!(source, updates, |data, update_data| scatter_typed(
            data,
            update_data,
            source.dtype,
            &source.shape,
            axis,
            indices,
            index_shape,
            selection,
            &updates.shape,
            source.shape.clone(),
        ))
    }

    #[test]
    fn scatter_full_rank_fast_path_agrees_with_the_general_path() {
        // A full-rank (== data-shaped) index tensor with duplicates, in the
        // three decomposition regimes: inner == 1 (rank-1 full selection),
        // inner == 4 (axis 0 of a matrix) and inner == 2 (a middle axis of a
        // rank-3 operand).  (`shape`, index multiplier, axis extent, `axis`).
        let cases: [(Vec<usize>, u64, u64, usize); 3] = [
            (vec![37], 7, 5, 0),
            (vec![3, 4], 5, 3, 0),
            (vec![2, 3, 2], 2, 3, 1),
        ];
        for (in_shape, step, extent, axis) in cases {
            let total: usize = in_shape.iter().product();
            // A pattern with guaranteed duplicates along the scattered axis.
            let indices: Vec<i128> = (0..total)
                .map(|position| ((position as u64 * step + 1) % extent) as i128)
                .collect();
            let values: Vec<f32> = (0..total).map(|value| value as f32).collect();
            let update_values: Vec<f32> =
                (0..total).map(|value| 100.0 + value as f32).collect();
            let x = f32_buffer(&in_shape, &values);
            let updates = f32_buffer(&in_shape, &update_values);
            let declared_dims = declared(&in_shape);

            let narrowed: Vec<i64> = indices.iter().map(|&raw| raw as i64).collect();
            let index_buffer = i64_buffer(&in_shape, &narrowed);
            let fast = scatter(&x, &index_buffer, &in_shape, &updates, axis, &declared_dims)
                .expect("the fast path succeeds");
            let general = general_scatter(&x, &updates, axis, &in_shape, &index_buffer)
                .expect("the general path succeeds");
            assert_eq!(fast, general, "shape {in_shape:?} axis {axis}");
            assert_eq!(x.storage, Storage::F32(values), "the input must not be mutated");

            // Duplicate indices: the LAST write in C order wins.
            let scattered = (0..extent as usize).find(|&selected| {
                indices.iter().filter(|&&raw| raw as usize == selected).count() > 1
            });
            let selected = scattered.expect("the pattern has duplicate indices");
            let last = indices
                .iter()
                .rposition(|&raw| raw as usize == selected)
                .expect("at least one occurrence");
            let destination = flat_destination(&in_shape, axis, last, selected);
            assert_eq!(
                f32_slot(&fast.storage, destination),
                update_values[last],
                "shape {in_shape:?} axis {axis}: the last write must win at index {selected}"
            );
        }
    }

    #[test]
    fn scatter_sub_shaped_selection_fast_path_agrees_with_the_general_path() {
        // The downstream `scatter(dst[n], idx[m], values[m])` shape: the index
        // tensor selects only `m` of the axis positions, so its shape is NOT
        // the data's.  Every such no-broadcast selection must take the flat
        // fast path and agree with the general rank-deep driver, for a rank-1
        // destination and for a 2-D/3-D `axis`-0 or `axis`-1 selection.
        type ScatterCase = (Vec<usize>, Vec<usize>, usize, u64, u64);
        let cases: [ScatterCase; 4] = [
            (vec![41], vec![17], 0, 7919, 41),
            (vec![4, 5], vec![2, 5], 0, 7, 4),
            (vec![3, 6], vec![3, 2], 1, 5, 6),
            (vec![2, 4, 3], vec![2, 4, 2], 2, 3, 3),
        ];
        for (data_shape, index_shape, axis, step, extent) in cases {
            let count: usize = index_shape.iter().product();
            let indices: Vec<i128> = (0..count)
                .map(|position| ((position as u64 * step + 1) % extent) as i128)
                .collect();
            let narrowed: Vec<i64> = indices.iter().map(|&raw| raw as i64).collect();
            let index_buffer = i64_buffer(&index_shape, &narrowed);
            let data_len: usize = data_shape.iter().product();
            let values: Vec<f32> = (0..data_len).map(|value| value as f32).collect();
            let update_values: Vec<f32> =
                (0..count).map(|value| 100.0 + value as f32).collect();
            let x = f32_buffer(&data_shape, &values);
            let updates = f32_buffer(&index_shape, &update_values);

            let fast = scatter_selection(&x, &index_buffer, &index_shape, &updates, axis)
                .expect("the fast path succeeds")
                .expect("the fast path applies");
            let general = general_scatter_shaped(
                &x,
                &updates,
                axis,
                &index_shape,
                &index_shape,
                &index_buffer,
            )
            .expect("the general path succeeds");
            assert_eq!(
                fast,
                general.storage,
                "data {data_shape:?} indices {index_shape:?} axis {axis}"
            );
            assert_eq!(x.storage, Storage::F32(values), "the input must not be mutated");
        }
    }

    #[test]
    fn scatter_fast_path_covers_every_data_dtype() {
        // The selection rewrite is dtype-generic, so integer/bool destinations
        // must scatter exactly like the general driver.
        use crate::kernels::indexing::test_support::buffer;
        let cases: [(Storage, Storage); 3] = [
            (Storage::I32(vec![0; 4]), Storage::I32(vec![7, 9])),
            (Storage::Bool(vec![false; 4]), Storage::Bool(vec![true, true])),
            (Storage::I8(vec![0; 4]), Storage::I8(vec![7, 9])),
        ];
        for (data_storage, update_storage) in cases {
            let dtype = data_storage.dtype();
            let source = buffer(dtype, &[4], data_storage);
            let updates = buffer(dtype, &[2], update_storage);
            let indices = i32_buffer(&[2], &[3, 1]);
            let fast = scatter_selection(&source, &indices, &[2], &updates, 0)
                .expect("the fast path succeeds")
                .expect("the fast path applies");
            let general = general_scatter_shaped(&source, &updates, 0, &[2], &[2], &indices)
                .expect("the general path succeeds");
            assert_eq!(fast, general.storage, "dtype {}", dtype.name());
        }
    }

    #[test]
    fn scatter_fully_overwrites_a_recycled_destination_allocation() {
        // The destination copy MUST write every element of a recycled stock:
        // leave a sentinel-filled buffer of the exact size in this thread's
        // pool, scatter into it, and check no sentinel survived.
        let total = 1 << 16; // 256 KiB f32 — above the pool's minimum size
        let sentinel = 9.0e30f32;
        drop(Storage::F32(vec![sentinel; total]));
        let data: Vec<f32> = (0..total).map(|value| value as f32).collect();
        let x = f32_buffer(&[total], &data);
        let indices: Vec<i64> = (0..total / 2).map(|value| (value * 2) as i64).collect();
        let updates = f32_buffer(&[total / 2], &vec![-1.0; total / 2]);
        let out = scatter(
            &x,
            &i64_buffer(&[total / 2], &indices),
            &[total / 2],
            &updates,
            0,
            &declared(&[total]),
        )
        .unwrap();
        let Storage::F32(result) = &out.storage else {
            panic!("the fixture is f32");
        };
        assert!(
            !result.contains(&sentinel),
            "a recycled destination vector leaked a stale element"
        );
        assert_eq!(result[0], -1.0, "the selected slots are the updates");
        assert_eq!(result[1], 1.0, "the unselected slots are the copied data");
    }
}
