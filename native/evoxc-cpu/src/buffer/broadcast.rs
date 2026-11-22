//! Broadcasting: the numpy shape/stride rules, the strided `walk`/`apply`
//! drivers and `broadcast_to`.

use evoxc_core::{Error, Result};

use super::{shape_text, stride::fill_broadcast, Buffer, Storage};

// ---------------------------------------------------------------------------
// Broadcasting
// ---------------------------------------------------------------------------

/// numpy broadcasting rule: align from the right, equal or 1 per axis.
pub fn broadcast_shape(shapes: &[&[usize]]) -> Result<Vec<usize>> {
    let rank = shapes.iter().map(|shape| shape.len()).max().unwrap_or(0);
    let mut out = vec![1usize; rank];
    for shape in shapes {
        let offset = rank - shape.len();
        for (axis, dim) in shape.iter().enumerate() {
            let slot = offset + axis;
            if out[slot] == 1 {
                out[slot] = *dim;
            } else if *dim != 1 && *dim != out[slot] {
                let rendered: Vec<String> = shapes
                    .iter()
                    .map(|candidate| shape_text(candidate))
                    .collect();
                return Err(Error::shape(format!(
                    "operand shapes cannot broadcast: {}",
                    rendered.join(" vs ")
                )));
            }
        }
    }
    Ok(out)
}

/// Strides of `shape` laid out over `out_shape` (right-aligned), with a `0`
/// stride on broadcast axes.
pub fn broadcast_strides(shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
    let rank = out_shape.len();
    let mut strides = vec![0usize; rank];
    let offset = rank.saturating_sub(shape.len());
    let mut stride = 1usize;
    for axis in (0..shape.len()).rev() {
        if shape[axis] != 1 {
            strides[offset + axis] = stride;
        }
        stride *= shape[axis];
    }
    strides
}

/// C-order flat iteration over `out`, computing one operand offset per entry of
/// `strides` (a broadcast axis contributes a `0` stride, so the same loop serves
/// every strided kernel).  `body` receives those offsets in C-order.
pub fn walk(rank: usize, out_shape: &[usize], strides: &[Vec<usize>], mut body: impl FnMut(&[usize])) {
    if rank == 0 {
        body(&vec![0usize; strides.len()]);
        return;
    }
    let total: usize = out_shape.iter().product();
    let mut index = vec![0usize; rank];
    let mut offsets = vec![0usize; strides.len()];
    for _ in 0..total {
        body(&offsets);
        for axis in (0..rank).rev() {
            index[axis] += 1;
            for (slot, stride) in strides.iter().enumerate() {
                offsets[slot] += stride[axis];
            }
            if index[axis] < out_shape[axis] {
                break;
            }
            index[axis] = 0;
            for (slot, stride) in strides.iter().enumerate() {
                offsets[slot] -= stride[axis] * out_shape[axis];
            }
        }
    }
}

/// Apply a unary function over `source` broadcast to `out_shape`.
pub fn apply1<T: Copy, R>(
    out: &mut [R],
    out_shape: &[usize],
    source: &[T],
    source_shape: &[usize],
    f: impl Fn(T) -> R,
) {
    let strides = [broadcast_strides(source_shape, out_shape)];
    let mut slot = 0usize;
    walk(out_shape.len(), out_shape, &strides, |offsets| {
        out[slot] = f(source[offsets[0]]);
        slot += 1;
    });
}

/// Apply a binary function over two operands broadcast to `out_shape`.
pub fn apply2<T: Copy, U: Copy, R>(
    out: &mut [R],
    out_shape: &[usize],
    left: &[T],
    left_shape: &[usize],
    right: &[U],
    right_shape: &[usize],
    f: impl Fn(T, U) -> R,
) {
    let strides = [
        broadcast_strides(left_shape, out_shape),
        broadcast_strides(right_shape, out_shape),
    ];
    let mut slot = 0usize;
    walk(out_shape.len(), out_shape, &strides, |offsets| {
        out[slot] = f(left[offsets[0]], right[offsets[1]]);
        slot += 1;
    });
}

/// Apply a ternary function over three operands broadcast to `out_shape`.
#[allow(clippy::too_many_arguments)]
pub fn apply3<T: Copy, U: Copy, V: Copy, R>(
    out: &mut [R],
    out_shape: &[usize],
    a: &[T],
    a_shape: &[usize],
    b: &[U],
    b_shape: &[usize],
    c: &[V],
    c_shape: &[usize],
    f: impl Fn(T, U, V) -> R,
) {
    let strides = [
        broadcast_strides(a_shape, out_shape),
        broadcast_strides(b_shape, out_shape),
        broadcast_strides(c_shape, out_shape),
    ];
    let mut slot = 0usize;
    walk(out_shape.len(), out_shape, &strides, |offsets| {
        out[slot] = f(a[offsets[0]], b[offsets[1]], c[offsets[2]]);
        slot += 1;
    });
}

/// Materialize `buffer` broadcast to `shape`.
pub fn broadcast_to(buffer: &Buffer, shape: &[usize]) -> Result<Buffer> {
    if buffer.shape == shape {
        return Ok(buffer.clone());
    }
    let out_shape = broadcast_shape(&[&buffer.shape, shape])?;
    if out_shape != shape {
        return Err(Error::shape(format!(
            "operand shape {} cannot be broadcast to target shape {}",
            shape_text(&buffer.shape),
            shape_text(shape)
        )));
    }
    // The copy itself is the strided-run driver's block copy
    // (`super::stride::fill_broadcast`), which hands back a pooled vector of the
    // exact length and divides the work into contiguous per-repeat blocks; it
    // accepts every broadcastable pair and falls back to [`broadcast_vec`] for
    // anything its run decomposition cannot express, so the result is the same
    // bytes either way.
    let storage = match &buffer.storage {
        Storage::Bool(v) => Storage::Bool(fill_broadcast(v, &buffer.shape, shape)),
        Storage::I8(v) => Storage::I8(fill_broadcast(v, &buffer.shape, shape)),
        Storage::I16(v) => Storage::I16(fill_broadcast(v, &buffer.shape, shape)),
        Storage::I32(v) => Storage::I32(fill_broadcast(v, &buffer.shape, shape)),
        Storage::I64(v) => Storage::I64(fill_broadcast(v, &buffer.shape, shape)),
        Storage::U8(v) => Storage::U8(fill_broadcast(v, &buffer.shape, shape)),
        Storage::U16(v) => Storage::U16(fill_broadcast(v, &buffer.shape, shape)),
        Storage::U32(v) => Storage::U32(fill_broadcast(v, &buffer.shape, shape)),
        Storage::U64(v) => Storage::U64(fill_broadcast(v, &buffer.shape, shape)),
        Storage::F16(v) => Storage::F16(fill_broadcast(v, &buffer.shape, shape)),
        Storage::F32(v) => Storage::F32(fill_broadcast(v, &buffer.shape, shape)),
        Storage::F64(v) => Storage::F64(fill_broadcast(v, &buffer.shape, shape)),
        Storage::C64(v) => Storage::C64(fill_broadcast(v, &buffer.shape, shape)),
        Storage::C128(v) => Storage::C128(fill_broadcast(v, &buffer.shape, shape)),
    };
    Buffer::new(buffer.dtype, shape.to_vec(), storage)
}

/// The general broadcast copy: one odometer offset per element. Kept as the
/// fallback of the strided-run driver (`super::stride::fill_broadcast`), which
/// serves every shape a run decomposition covers.
pub(super) fn broadcast_vec<T: Copy>(values: &[T], shape: &[usize], out_shape: &[usize]) -> Vec<T> {
    let mut out = Vec::with_capacity(out_shape.iter().product());
    let strides = [broadcast_strides(shape, out_shape)];
    walk(out_shape.len(), out_shape, &strides, |offsets| {
        out.push(values[offsets[0]]);
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{Buffer, Storage};
    use evoxc_core::DType;

    #[test]
    fn broadcast_strides_align_right() {
        assert_eq!(broadcast_strides(&[3], &[2, 3]), vec![0, 1]);
        assert_eq!(broadcast_strides(&[1, 3], &[2, 3]), vec![0, 1]);
        assert_eq!(broadcast_strides(&[2, 1], &[2, 3]), vec![1, 0]);
    }

    #[test]
    fn apply2_broadcasts_both_sides() {
        let mut out = vec![0i32; 6];
        apply2(&mut out, &[2, 3], &[1, 2, 3], &[1, 3], &[1, 2], &[2, 1], |a, b| a + b);
        assert_eq!(out, vec![2, 3, 4, 3, 4, 5]);
    }

    #[test]
    fn broadcast_to_rejects_what_it_always_rejected() {
        let buffer = Buffer::new(DType::Int32, vec![3], Storage::I32(vec![1, 2, 3])).unwrap();
        // Right-aligned only: a length-3 axis cannot serve a length-2 target.
        let error = broadcast_to(&buffer, &[2]).unwrap_err();
        assert_eq!(
            error.to_string(),
            "ShapeError: operand shapes cannot broadcast: (3) vs (2)"
        );
        // A target that the source would widen into is refused by the strict
        // equality check (the broadcast of `(1, 3)` is `(1, 3)`, not `(3,)`).
        let row = Buffer::new(DType::Int32, vec![1, 3], Storage::I32(vec![1, 2, 3])).unwrap();
        let error = broadcast_to(&row, &[3]).unwrap_err();
        assert_eq!(
            error.to_string(),
            "ShapeError: operand shape (1, 3) cannot be broadcast to target shape (3)"
        );
        // An equal shape short-circuits to a clone, values untouched.
        let same = broadcast_to(&buffer, &[3]).unwrap();
        assert_eq!(same.storage, Storage::I32(vec![1, 2, 3]));
    }

    /// Element `i` of a broadcast result is element `slots[i]` of the source.
    fn same_sequence<A: Copy, B: Copy>(
        out: &[A],
        source: &[B],
        slots: &[usize],
        eq: impl Fn(A, B) -> bool,
    ) {
        assert_eq!(out.len(), slots.len());
        for (index, slot) in slots.iter().enumerate() {
            assert!(eq(out[index], source[*slot]), "element {index} (slot {slot})");
        }
    }

    #[test]
    fn every_dtype_materializes_through_the_block_copy() {
        // The pooled block copy serves all 14 storages; bool and the complex
        // and float pairs are the ones a per-dtype arm could silently drop.
        let cases: [(Storage, Vec<usize>, Vec<usize>); 3] = [
            (Storage::Bool(vec![true, false]), vec![2], vec![3, 2]),
            (
                Storage::C64(vec![
                    num_complex::Complex::new(1.0, 2.0),
                    num_complex::Complex::new(-0.0, 0.5),
                ]),
                vec![1, 2],
                vec![2, 3, 2],
            ),
            (Storage::F64(vec![1.5, -0.0, f64::NAN]), vec![3, 1], vec![3, 4]),
        ];
        for (storage, shape, out_shape) in cases {
            let dtype = storage.dtype();
            let buffer = Buffer::new(dtype, shape.clone(), storage).unwrap();
            let expanded = broadcast_to(&buffer, &out_shape).unwrap();
            assert_eq!(expanded.shape, out_shape);
            // The general odometer is the reference: same slot per element.
            let mut slots = Vec::new();
            let strides = broadcast_strides(&shape, &out_shape);
            walk(out_shape.len(), &out_shape, &[strides], |offsets| slots.push(offsets[0]));
            match (&expanded.storage, &buffer.storage) {
                (Storage::Bool(out), Storage::Bool(source)) => {
                    same_sequence(out, source, &slots, |a, b| a == b)
                }
                (Storage::C64(out), Storage::C64(source)) => {
                    same_sequence(out, source, &slots, |a, b| a == b)
                }
                (Storage::F64(out), Storage::F64(source)) => {
                    same_sequence(out, source, &slots, |a, b| a.to_bits() == b.to_bits())
                }
                other => panic!("unexpected storage {other:?}"),
            }
        }
    }

    #[test]
    fn scalar_buffer_broadcasts() {
        let scalar = Buffer::new(DType::Int32, vec![], Storage::I32(vec![7])).unwrap();
        let out = broadcast_to(&scalar, &[2, 2]).unwrap();
        assert_eq!(out.storage, Storage::I32(vec![7, 7, 7, 7]));
    }
}
