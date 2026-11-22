//! The axis-local permutation (`argsort` positions) and both ops' outputs.
//!
//! [`permutation`] is the dtype dispatch: one monomorphized core per concrete
//! scalar type produces the axis-local position vector, and
//! [`permuted_storage`] / [`index_storage`] turn it into the `sort` /
//! `argsort` result storage.
//!
//! The axis/layout walk is ndarray's: the operand and the permutation are
//! viewed as [`ndarray::ArrayViewD`] (a zero-copy borrow of the C-contiguous
//! storage) and the work runs lane by lane over `lanes(Axis(axis))` — one 1-D
//! view per axis segment, yielded in C order — so no hand-written segment
//! base/stride arithmetic lives here.  A contiguous lane (the last-axis case)
//! is read straight off its `as_slice`; any other lane is gathered once per
//! call into one reused scratch, exactly as before.  The order-preserving `u64`
//! key table ([`RowKey`]) and the Batcher network ([`order_keys`]) are the ordering
//! algorithms ndarray does not provide and stay as their own specializations.

use evoxc_core::{DType, Error, Result};
use half::f16;
use ndarray::{ArrayView1, ArrayViewMut1, Axis};
use num_complex::Complex;

use crate::buffer::{array_view, array_view_mut, Buffer, NdScalar, Storage};

#[cfg(doc)]
use super::network::{NETWORK_MAX, PAD_KEY};
use super::keys::{RowKey, STACK_KEYS};
use super::network::{network_width, order_keys};
use super::complex_unordered;

// ---------------------------------------------------------------------------
// The axis-local permutation
// ---------------------------------------------------------------------------

/// The axis-local permutation that sorts `values` along `axis`.
///
/// Each entry of the returned vector is, at the output's flat position, the
/// operand's index WITHIN its sorted axis segment — numpy's `argsort` result;
/// [`sorted_values`] turns it into the operand's flat offset for `sort`.
/// Ties keep first-occurrence order (`sort_unstable` over keys that carry the
/// original index, so the order is total and reproducible), and `descending`
/// reverses every segment, exactly the reference's ascending result flipped
/// along the axis.
fn sorted_positions<T: RowKey + NdScalar>(
    all: &[T],
    shape: &[usize],
    axis: usize,
    descending: bool,
) -> Result<Vec<u32>> {
    let extent = shape[axis];
    // Positions are axis-local, so `u32` suffices for any axis the caller
    // accepted ([`permutation`] rejects an extent above `u32::MAX`); the narrow
    // width halves this buffer's traffic, which measurably beats `usize` here.
    let mut perm = vec![0u32; all.len()];
    // A zero- or one-element axis segment is already sorted and reversing it is
    // a no-op, so the zero-initialized permutation (identity for `extent <= 1`)
    // is the answer.
    if extent < 2 || all.is_empty() {
        return Ok(perm);
    }
    // Scratch, sized once per call and reused for every segment: on the stack
    // for the extents this engine sees most, on the heap beyond that.  The key
    // scratch is padded to the extent's network width so [`order_keys`] can fill
    // its tail with [`PAD_KEY`]; extents past [`NETWORK_MAX`] keep
    // `sort_unstable` and need the extent only.
    let key_len = match network_width(extent) {
        0 => extent,
        width => width,
    };
    let mut stack_keys = [0u64; STACK_KEYS];
    let mut stack_order = [0u32; STACK_KEYS];
    let mut stack_positions = [0u32; STACK_KEYS];
    let mut heap_keys: Vec<u64> = Vec::new();
    let mut heap_order: Vec<u32> = Vec::new();
    let mut heap_positions: Vec<u32> = Vec::new();
    let (keys, order, positions): (&mut [u64], &mut [u32], &mut [u32]) = if extent <= STACK_KEYS {
        (
            &mut stack_keys[..key_len],
            &mut stack_order[..extent],
            &mut stack_positions[..extent],
        )
    } else {
        heap_keys.resize(key_len, 0);
        heap_order.resize(extent, 0);
        heap_positions.resize(extent, 0);
        (&mut heap_keys[..], &mut heap_order[..], &mut heap_positions[..])
    };
    // A gathered (non-contiguous) lane lands in this one scratch, reused across
    // every segment; a contiguous lane is read straight off its `as_slice`.
    let mut scratch: Vec<T> = Vec::new();
    let view = array_view::<T>(all, shape)?;
    let mut perm_view = array_view_mut::<u32>(&mut perm, shape)?;
    for (out_lane, in_lane) in perm_view
        .lanes_mut(Axis(axis))
        .into_iter()
        .zip(view.lanes(Axis(axis)))
    {
        let segment: &[T] = match in_lane.as_slice() {
            Some(slice) => slice,
            None => {
                scratch.clear();
                scratch.extend(in_lane.iter().copied());
                &scratch
            }
        };
        segment_positions(segment, positions, descending, keys, order);
        assign_positions(out_lane, positions);
    }
    Ok(perm)
}

/// Write one lane's sorted positions into the output lane (a view assignment,
/// so a strided middle axis and a contiguous last axis take the same path).
fn assign_positions(mut out_lane: ArrayViewMut1<'_, u32>, positions: &[u32]) {
    out_lane.assign(&ArrayView1::from(positions));
}

/// Order one gathered segment into `positions`: `positions[p]` is the segment's
/// index that belongs at sorted position `p` (flipped for `descending`).
///
/// `keys`/`order` are the caller's scratch, sized once per call and reused for
/// every segment (`keys` >= `segment.len().next_power_of_two()` for the narrow
/// arms so the padded network has room).
fn segment_positions<T: RowKey>(
    segment: &[T],
    positions: &mut [u32],
    descending: bool,
    keys: &mut [u64],
    order: &mut [u32],
) {
    let extent = segment.len();
    if T::NARROW {
        // Narrow keys: one ascending order over packed `(key, index)` pairs —
        // the low 32 bits are the original index, so ties come out in index
        // order, i.e. the stable order.  [`order_keys`] picks the padded
        // sorting network for the short extents and `sort_unstable` beyond.
        for (slot, value) in segment.iter().enumerate() {
            keys[slot] = (value.rank_key() << 32) | slot as u64;
        }
        order_keys(keys, extent);
        for (position, key) in keys[..extent].iter().enumerate() {
            let index = (*key & 0xFFFF_FFFF) as u32;
            positions[if descending { extent - 1 - position } else { position }] = index;
        }
    } else {
        // 64-bit keys cannot carry the index: sort the index vector by
        // `(key, index)`, which is the same total order.
        for (slot, value) in segment.iter().enumerate() {
            keys[slot] = value.rank_key();
            order[slot] = slot as u32;
        }
        let keys: &[u64] = keys;
        order.sort_unstable_by(|left, right| {
            keys[*left as usize]
                .cmp(&keys[*right as usize])
                .then(left.cmp(right))
        });
        for (position, index) in order.iter().enumerate() {
            positions[if descending { extent - 1 - position } else { position }] = *index;
        }
    }
}

/// Compute the op's permutation from the operand's typed values.
pub(crate) fn permutation(
    input: &Buffer,
    shape: &[usize],
    axis: usize,
    descending: bool,
    op: &str,
) -> Result<Vec<u32>> {
    // Positions are stored as `u32`: an axis extent beyond that cannot be
    // represented and is an explicit error, never a truncated index.
    if shape[axis] > u32::MAX as usize {
        return Err(Error::shape(format!(
            "op '{op}': axis {axis} has extent {} — the native CPU engine indexes sort \
             positions with u32 and cannot represent an axis longer than {}",
            shape[axis],
            u32::MAX
        )));
    }
    let perm = match &input.storage {
        Storage::Bool(values) => sorted_positions(values, shape, axis, descending)?,
        Storage::I8(values) => sorted_positions(values, shape, axis, descending)?,
        Storage::I16(values) => sorted_positions(values, shape, axis, descending)?,
        Storage::I32(values) => sorted_positions(values, shape, axis, descending)?,
        Storage::I64(values) => sorted_positions(values, shape, axis, descending)?,
        Storage::U8(values) => sorted_positions(values, shape, axis, descending)?,
        Storage::U16(values) => sorted_positions(values, shape, axis, descending)?,
        Storage::U32(values) => sorted_positions(values, shape, axis, descending)?,
        Storage::U64(values) => sorted_positions(values, shape, axis, descending)?,
        Storage::F16(values) => sorted_positions(values, shape, axis, descending)?,
        Storage::F32(values) => sorted_positions(values, shape, axis, descending)?,
        Storage::F64(values) => sorted_positions(values, shape, axis, descending)?,
        Storage::C64(_) | Storage::C128(_) => return Err(complex_unordered(op, input.dtype)),
    };
    Ok(perm)
}

// ---------------------------------------------------------------------------
// The outputs
// ---------------------------------------------------------------------------

/// Gather the operand's typed values through `perm` (the `sort` result).
pub(crate) fn permuted_storage(
    input: &Buffer,
    perm: &[u32],
    shape: &[usize],
    axis: usize,
    op: &str,
) -> Result<Storage> {
    let storage = match &input.storage {
        Storage::Bool(values) => Storage::Bool(sorted_values(values, perm, shape, axis)?),
        Storage::I8(values) => Storage::I8(sorted_values(values, perm, shape, axis)?),
        Storage::I16(values) => Storage::I16(sorted_values(values, perm, shape, axis)?),
        Storage::I32(values) => Storage::I32(sorted_values(values, perm, shape, axis)?),
        Storage::I64(values) => Storage::I64(sorted_values(values, perm, shape, axis)?),
        Storage::U8(values) => Storage::U8(sorted_values(values, perm, shape, axis)?),
        Storage::U16(values) => Storage::U16(sorted_values(values, perm, shape, axis)?),
        Storage::U32(values) => Storage::U32(sorted_values(values, perm, shape, axis)?),
        Storage::U64(values) => Storage::U64(sorted_values(values, perm, shape, axis)?),
        Storage::F16(values) => Storage::F16(sorted_values(values, perm, shape, axis)?),
        Storage::F32(values) => Storage::F32(sorted_values(values, perm, shape, axis)?),
        Storage::F64(values) => Storage::F64(sorted_values(values, perm, shape, axis)?),
        Storage::C64(_) | Storage::C128(_) => return Err(complex_unordered(op, input.dtype)),
    };
    Ok(storage)
}

/// The `sort` result: every lane of the output is the operand's lane gathered
/// through that lane's positions.
///
/// The last axis is the common hot path — its lanes are contiguous and appear
/// in C order, so gathering them lane by lane appends the result straight into
/// a fresh allocation with no fill pass.  Any other axis writes each gathered
/// lane into the output's own (strided) lane through an ndarray view
/// assignment, starting from a copy of the values so every slot is defined.
fn sorted_values<T: NdScalar>(
    values: &[T],
    perm: &[u32],
    shape: &[usize],
    axis: usize,
) -> Result<Vec<T>> {
    if values.is_empty() {
        return Ok(Vec::new());
    }
    let view = array_view::<T>(values, shape)?;
    let perm_view = array_view::<u32>(perm, shape)?;
    let mut scratch: Vec<T> = Vec::new();
    if axis + 1 == shape.len() {
        let mut out: Vec<T> = Vec::with_capacity(values.len());
        for (in_lane, perm_lane) in view
            .lanes(Axis(axis))
            .into_iter()
            .zip(perm_view.lanes(Axis(axis)))
        {
            let segment: &[T] = match in_lane.as_slice() {
                Some(slice) => slice,
                None => {
                    scratch.clear();
                    scratch.extend(in_lane.iter().copied());
                    &scratch
                }
            };
            out.extend(perm_lane.iter().map(|&index| segment[index as usize]));
        }
        return Ok(out);
    }
    let mut out = values.to_vec();
    let mut out_view = array_view_mut::<T>(&mut out, shape)?;
    for ((mut out_lane, in_lane), perm_lane) in out_view
        .lanes_mut(Axis(axis))
        .into_iter()
        .zip(view.lanes(Axis(axis)))
        .zip(perm_view.lanes(Axis(axis)))
    {
        scratch.clear();
        scratch.extend(perm_lane.iter().map(|&index| in_lane[index as usize]));
        out_lane.assign(&ArrayView1::from(&scratch[..]));
    }
    Ok(out)
}

/// The `argsort` result storage: the axis-local positions cast into `dtype`
/// (the same `as` casts [`crate::scalar::Scalar::from_scalar`] applies, without
/// the per-element `ScalarValue` round trip).
pub(crate) fn index_storage(dtype: DType, perm: &[u32]) -> Storage {
    match dtype {
        DType::Bool => Storage::Bool(perm.iter().map(|&index| index != 0).collect()),
        DType::Int8 => Storage::I8(perm.iter().map(|&index| index as i8).collect()),
        DType::Int16 => Storage::I16(perm.iter().map(|&index| index as i16).collect()),
        DType::Int32 => Storage::I32(perm.iter().map(|&index| index as i32).collect()),
        DType::Int64 => Storage::I64(perm.iter().map(|&index| index as i64).collect()),
        DType::UInt8 => Storage::U8(perm.iter().map(|&index| index as u8).collect()),
        DType::UInt16 => Storage::U16(perm.iter().map(|&index| index as u16).collect()),
        DType::UInt32 => Storage::U32(perm.to_vec()),
        DType::UInt64 => Storage::U64(perm.iter().map(|&index| index as u64).collect()),
        DType::Float16 => Storage::F16(
            perm.iter()
                .map(|&index| f16::from_f64(index as f64))
                .collect(),
        ),
        DType::Float32 => Storage::F32(perm.iter().map(|&index| index as f32).collect()),
        DType::Float64 => Storage::F64(perm.iter().map(|&index| index as f64).collect()),
        DType::Complex64 => Storage::C64(
            perm.iter()
                .map(|&index| Complex::new(index as f32, 0.0))
                .collect(),
        ),
        DType::Complex128 => Storage::C128(
            perm.iter()
                .map(|&index| Complex::new(index as f64, 0.0))
                .collect(),
        ),
    }
}
