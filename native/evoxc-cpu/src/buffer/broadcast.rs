//! Broadcasting: the numpy shape/stride rules, the strided `walk`/`apply`
//! drivers and `broadcast_to`.

use evoxc_core::{Error, Result};

use super::{shape_text, Buffer, Storage};

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
    let storage = match &buffer.storage {
        Storage::Bool(v) => Storage::Bool(broadcast_vec(v, &buffer.shape, shape)),
        Storage::I8(v) => Storage::I8(broadcast_vec(v, &buffer.shape, shape)),
        Storage::I16(v) => Storage::I16(broadcast_vec(v, &buffer.shape, shape)),
        Storage::I32(v) => Storage::I32(broadcast_vec(v, &buffer.shape, shape)),
        Storage::I64(v) => Storage::I64(broadcast_vec(v, &buffer.shape, shape)),
        Storage::U8(v) => Storage::U8(broadcast_vec(v, &buffer.shape, shape)),
        Storage::U16(v) => Storage::U16(broadcast_vec(v, &buffer.shape, shape)),
        Storage::U32(v) => Storage::U32(broadcast_vec(v, &buffer.shape, shape)),
        Storage::U64(v) => Storage::U64(broadcast_vec(v, &buffer.shape, shape)),
        Storage::F16(v) => Storage::F16(broadcast_vec(v, &buffer.shape, shape)),
        Storage::F32(v) => Storage::F32(broadcast_vec(v, &buffer.shape, shape)),
        Storage::F64(v) => Storage::F64(broadcast_vec(v, &buffer.shape, shape)),
        Storage::C64(v) => Storage::C64(broadcast_vec(v, &buffer.shape, shape)),
        Storage::C128(v) => Storage::C128(broadcast_vec(v, &buffer.shape, shape)),
    };
    Buffer::new(buffer.dtype, shape.to_vec(), storage)
}

fn broadcast_vec<T: Copy>(values: &[T], shape: &[usize], out_shape: &[usize]) -> Vec<T> {
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
    fn scalar_buffer_broadcasts() {
        let scalar = Buffer::new(DType::Int32, vec![], Storage::I32(vec![7])).unwrap();
        let out = broadcast_to(&scalar, &[2, 2]).unwrap();
        assert_eq!(out.storage, Storage::I32(vec![7, 7, 7, 7]));
    }
}
