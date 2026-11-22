//! Broadcasting: the numpy shape/stride rules, the ndarray-backed elementwise
//! layout drivers (`walk` / `apply1..3`) and `broadcast_to`.
//!
//! [`Storage`] is one C-contiguous `Vec` per dtype, so an operand read through a
//! broadcast (or any other non-contiguous) layout has to be addressed through
//! its shape.  ndarray's broadcast views are the ONLY N-dimensional layout
//! engine here: [`walk`] is a thin index iteration, `apply1..3` are `Zip` passes
//! and [`fill_broadcast`] is one `assign` of a broadcast view, so no hand-rolled
//! stride odometer remains.  Every driver falls back to the [`walk`] loop for a
//! layout a view cannot express (an operand slice whose length disagrees with
//! its shape, an operand of higher rank than the result, a result slice whose
//! length disagrees with its shape), which keeps the pre-existing
//! accept/reject/panic behaviour byte-identical.
//!
//! The dimensions are STATIC (`Ix0..Ix6`), dispatched once per call from the
//! result's rank: `IxDyn` is heap-backed, so an `IxDyn` `indices` iteration
//! clones an owned index on every outer step and an `IxDyn` [`walk`] allocates
//! once per element.  `IxDyn` survives in exactly two places — the single
//! per-call view construction of an operand's own (possibly lower) rank, and the
//! rank > 6 fallback arm, which is correct but slow and which no real graph
//! reaches.

use evoxc_core::{Error, Result};
use ndarray::{ArrayViewD, ArrayViewMut, Dimension, IntoDimension, IxDyn};

use crate::scalar::Scalar;

use super::pool::take_pooled_vec;
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

// ---------------------------------------------------------------------------
// ndarray views
// ---------------------------------------------------------------------------

/// Borrow `values` as a `shape`-shaped view of its OWN rank.
///
/// `None` when a slice's length is not `shape`'s element count — the one layout
/// the [`walk`] fallback has to answer for.  This `IxDyn` is the ONE dynamic dim
/// a call still builds: an operand's rank is not the result's rank (a `(4,)`
/// bias beside a `(2000, 50, 4)` result), so the base view is created at the
/// operand's own rank and then broadcast onto the result's static dim.  ndarray's
/// `broadcast` borrows the view it is called on, so each driver keeps its base
/// views in named locals and broadcasts those.
fn base_view<'a, T: Copy>(values: &'a [T], shape: &[usize]) -> Option<ArrayViewD<'a, T>> {
    ArrayViewD::from_shape(IxDyn(shape), values).ok()
}

/// Dispatch one generic static-dimension helper on the result's rank.
///
/// The helper takes the `Dimension` as its first argument (and may take any
/// further arguments after it), so every driver stays ONE generic
/// implementation over `D: Dimension` and the only per-rank difference is the
/// `Ix0..Ix6` value the dispatch hands it.  Above six axes the dynamic `IxDyn`
/// arm keeps the code correct; no real graph reaches it.
macro_rules! dispatch_rank {
    ($shape:expr, $helper:ident $(, $arg:expr)*) => {
        match $shape.len() {
            0 => $helper(ndarray::Ix0() $(, $arg)*),
            1 => $helper(ndarray::Ix1($shape[0]) $(, $arg)*),
            2 => $helper(ndarray::Ix2($shape[0], $shape[1]) $(, $arg)*),
            3 => $helper(ndarray::Ix3($shape[0], $shape[1], $shape[2]) $(, $arg)*),
            4 => $helper(ndarray::Ix4($shape[0], $shape[1], $shape[2], $shape[3]) $(, $arg)*),
            5 => $helper(
                ndarray::Ix5($shape[0], $shape[1], $shape[2], $shape[3], $shape[4])
                $(, $arg)*
            ),
            6 => $helper(
                ndarray::Ix6($shape[0], $shape[1], $shape[2], $shape[3], $shape[4], $shape[5])
                $(, $arg)*
            ),
            _ => $helper(ndarray::IxDyn($shape) $(, $arg)*),
        }
    };
}

/// C-order iteration over one static `shape`, computing one operand offset per
/// entry of `strides` (a broadcast axis contributes a `0` stride, so the same
/// loop serves every strided kernel).
///
/// `ndarray::indices` supplies the counter and `Dimension::slice` its position
/// — allocation-free for every static dim (an inline array, and `Ix1`'s single
/// `usize`) — so a step costs one increment plus one multiply-add per operand.
fn walk_static<D: Dimension>(shape: D, strides: &[Vec<usize>], body: &mut impl FnMut(&[usize])) {
    let mut offsets = vec![0usize; strides.len()];
    for index in ndarray::indices(shape) {
        let position = index.into_dimension();
        let position = position.slice();
        for (slot, stride) in offsets.iter_mut().zip(strides) {
            *slot = position.iter().zip(stride.iter()).map(|(i, s)| i * s).sum();
        }
        body(&offsets);
    }
}

/// C-order flat iteration over `out_shape[..rank]`, computing one operand offset
/// per entry of `strides` (a broadcast axis contributes a `0` stride, so the
/// same loop serves every strided kernel).  `body` receives those offsets in
/// C-order; `rank == 0` is ONE call with a zero-filled offsets slice and an
/// empty result iterates not at all.
pub fn walk(rank: usize, out_shape: &[usize], strides: &[Vec<usize>], mut body: impl FnMut(&[usize])) {
    if rank == 0 {
        body(&vec![0usize; strides.len()]);
        return;
    }
    let shape = &out_shape[..rank];
    dispatch_rank!(shape, walk_static, strides, &mut body);
}

// ---------------------------------------------------------------------------
// The elementwise drivers
// ---------------------------------------------------------------------------

/// `out[i] = f(a[i])` via one ndarray `Zip` pass over static dims, or `false`
/// when the layout cannot be expressed as views (the caller then takes the
/// [`walk`] loop).
fn apply1_static<T: Copy, R, D: Dimension>(
    out_shape: D,
    out: &mut [R],
    a: &[T],
    a_shape: &[usize],
    f: &impl Fn(T) -> R,
) -> bool {
    if out.len() != out_shape.size() {
        return false;
    }
    let Some(a_base) = base_view(a, a_shape) else {
        return false;
    };
    let Some(av) = a_base.broadcast(out_shape.clone()) else {
        return false;
    };
    let Ok(dest) = ArrayViewMut::<R, D>::from_shape(out_shape, out) else {
        return false;
    };
    ndarray::Zip::from(dest).and(av).for_each(|o, x| *o = f(*x));
    true
}

fn apply2_static<T: Copy, U: Copy, R, D: Dimension>(
    out_shape: D,
    out: &mut [R],
    a: &[T],
    a_shape: &[usize],
    b: &[U],
    b_shape: &[usize],
    f: &impl Fn(T, U) -> R,
) -> bool {
    if out.len() != out_shape.size() {
        return false;
    }
    let Some(a_base) = base_view(a, a_shape) else {
        return false;
    };
    let Some(av) = a_base.broadcast(out_shape.clone()) else {
        return false;
    };
    let Some(b_base) = base_view(b, b_shape) else {
        return false;
    };
    let Some(bv) = b_base.broadcast(out_shape.clone()) else {
        return false;
    };
    let Ok(dest) = ArrayViewMut::<R, D>::from_shape(out_shape, out) else {
        return false;
    };
    ndarray::Zip::from(dest)
        .and(av)
        .and(bv)
        .for_each(|o, x, y| *o = f(*x, *y));
    true
}

#[allow(clippy::too_many_arguments)]
fn apply3_static<T: Copy, U: Copy, V: Copy, R, D: Dimension>(
    out_shape: D,
    out: &mut [R],
    a: &[T],
    a_shape: &[usize],
    b: &[U],
    b_shape: &[usize],
    c: &[V],
    c_shape: &[usize],
    f: &impl Fn(T, U, V) -> R,
) -> bool {
    if out.len() != out_shape.size() {
        return false;
    }
    let Some(a_base) = base_view(a, a_shape) else {
        return false;
    };
    let Some(av) = a_base.broadcast(out_shape.clone()) else {
        return false;
    };
    let Some(b_base) = base_view(b, b_shape) else {
        return false;
    };
    let Some(bv) = b_base.broadcast(out_shape.clone()) else {
        return false;
    };
    let Some(c_base) = base_view(c, c_shape) else {
        return false;
    };
    let Some(cv) = c_base.broadcast(out_shape.clone()) else {
        return false;
    };
    let Ok(dest) = ArrayViewMut::<R, D>::from_shape(out_shape, out) else {
        return false;
    };
    ndarray::Zip::from(dest)
        .and(av)
        .and(bv)
        .and(cv)
        .for_each(|o, x, y, z| *o = f(*x, *y, *z));
    true
}

/// The [`walk`] loop of [`apply1`]: the view-free fallback, identical to the
/// driver this module replaced.
fn walk1<T: Copy, R>(
    out: &mut [R],
    out_shape: &[usize],
    source: &[T],
    source_shape: &[usize],
    f: &impl Fn(T) -> R,
) {
    let strides = [broadcast_strides(source_shape, out_shape)];
    let mut slot = 0usize;
    walk(out_shape.len(), out_shape, &strides, |offsets| {
        out[slot] = f(source[offsets[0]]);
        slot += 1;
    });
}

fn walk2<T: Copy, U: Copy, R>(
    out: &mut [R],
    out_shape: &[usize],
    left: &[T],
    left_shape: &[usize],
    right: &[U],
    right_shape: &[usize],
    f: &impl Fn(T, U) -> R,
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

#[allow(clippy::too_many_arguments)]
fn walk3<T: Copy, U: Copy, V: Copy, R>(
    out: &mut [R],
    out_shape: &[usize],
    a: &[T],
    a_shape: &[usize],
    b: &[U],
    b_shape: &[usize],
    c: &[V],
    c_shape: &[usize],
    f: &impl Fn(T, U, V) -> R,
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

/// Apply a unary function over `source` broadcast to `out_shape`.
///
/// Element `i` of the result reads element `i` of the broadcast operand, so the
/// closure owns every semantic (NaN propagation, signed zeros, wrapping) and the
/// driver only supplies addresses.
pub fn apply1<T: Copy, R>(
    out: &mut [R],
    out_shape: &[usize],
    source: &[T],
    source_shape: &[usize],
    f: impl Fn(T) -> R,
) {
    let accepted = dispatch_rank!(out_shape, apply1_static, out, source, source_shape, &f);
    if !accepted {
        walk1(out, out_shape, source, source_shape, &f);
    }
}

/// Apply a binary function over two operands broadcast to `out_shape` (see
/// [`apply1`]).
pub fn apply2<T: Copy, U: Copy, R>(
    out: &mut [R],
    out_shape: &[usize],
    left: &[T],
    left_shape: &[usize],
    right: &[U],
    right_shape: &[usize],
    f: impl Fn(T, U) -> R,
) {
    if !dispatch_rank!(
        out_shape,
        apply2_static,
        out,
        left,
        left_shape,
        right,
        right_shape,
        &f
    ) {
        walk2(out, out_shape, left, left_shape, right, right_shape, &f);
    }
}

/// Apply a ternary function over three operands broadcast to `out_shape` (see
/// [`apply1`]).
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
    if !dispatch_rank!(
        out_shape,
        apply3_static,
        out,
        a,
        a_shape,
        b,
        b_shape,
        c,
        c_shape,
        &f
    ) {
        walk3(out, out_shape, a, a_shape, b, b_shape, c, c_shape, &f);
    }
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
    // The copy is [`fill_broadcast`]: one ndarray `assign` of a broadcast view
    // out of pooled storage, falling back to [`broadcast_vec`]'s [`walk`] loop
    // for a shape a view cannot express. Either way the bytes are the same.
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

/// The general broadcast copy: one [`walk`] offset per element.  It is the
/// fallback of [`fill_broadcast`] for a layout an ndarray broadcast view cannot
/// express; every shape a real graph produces takes the view.
pub(super) fn broadcast_vec<T: Copy>(values: &[T], shape: &[usize], out_shape: &[usize]) -> Vec<T> {
    if let Some(collected) = dispatch_rank!(out_shape, broadcast_collect, values, shape) {
        return collected;
    }
    let mut out = Vec::with_capacity(out_shape.iter().product());
    let strides = [broadcast_strides(shape, out_shape)];
    walk(out_shape.len(), out_shape, &strides, |offsets| {
        out.push(values[offsets[0]]);
    });
    out
}

/// `broadcast_vec`'s view path: the broadcast view of `values` on the result's
/// static dim, collected in C order (`None` when the layout is not expressible,
/// which is the caller's cue to run the [`walk`] loop).
fn broadcast_collect<T: Copy, D: Dimension>(
    out_shape: D,
    values: &[T],
    shape: &[usize],
) -> Option<Vec<T>> {
    let base = base_view(values, shape)?;
    let view = base.broadcast(out_shape)?;
    let mut out = Vec::with_capacity(view.len());
    out.extend(view.iter().copied());
    Some(out)
}

/// Materialize `values` broadcast from `shape` to `out_shape` — the `broadcast`
/// op's copy, with [`walk`]'s semantics for every shape it accepts.
///
/// The result comes from the storage pool when a stock of the exact size is
/// available (never a zero fill on that path) and is written by ONE ndarray
/// `assign` of a broadcast view of `values` on the result's static dim, so the
/// copy never runs the per-element odometer.  An empty target is empty; a layout
/// no view can express falls back to [`broadcast_vec`].
pub fn fill_broadcast<T: Scalar>(values: &[T], shape: &[usize], out_shape: &[usize]) -> Vec<T> {
    let count: usize = out_shape.iter().product();
    if count == 0 {
        return Vec::new();
    }
    let mut out = take_pooled_vec::<T>(count).unwrap_or_else(|| T::zeros(count));
    let filled = dispatch_rank!(
        out_shape,
        fill_broadcast_static,
        values,
        shape,
        out.as_mut_slice()
    );
    if !filled {
        return broadcast_vec(values, shape, out_shape);
    }
    out
}

/// `fill_broadcast`'s view path: ONE `assign` of the broadcast view into `out`
/// (`false` when the layout is not expressible, which is the caller's cue to run
/// [`broadcast_vec`]).
fn fill_broadcast_static<T: Copy, D: Dimension>(
    out_shape: D,
    values: &[T],
    shape: &[usize],
    out: &mut [T],
) -> bool {
    if out.len() != out_shape.size() {
        return false;
    }
    let Some(base) = base_view(values, shape) else {
        return false;
    };
    let Some(view) = base.broadcast(out_shape.clone()) else {
        return false;
    };
    let Ok(mut dest) = ArrayViewMut::<T, D>::from_shape(out_shape, out) else {
        return false;
    };
    dest.assign(&view);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{Buffer, Storage};
    use evoxc_core::DType;

    /// A deterministic value generator (no `rand` dependency).
    fn values(count: usize, seed: i32) -> Vec<i32> {
        let mut state = seed as u64 | 1;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as i32) % 200 - 100
            })
            .collect()
    }

    fn product(shape: &[usize]) -> usize {
        shape.iter().product()
    }

    /// Independent right-aligned C-order broadcast reference: for flat result
    /// index `flat`, the operand element it reads.
    ///
    /// Deliberately NOT the implementation under test: it walks the result
    /// coordinates one axis at a time (no ndarray views, no strides).
    fn reference_slot(out_shape: &[usize], operand_shape: &[usize], flat: usize) -> usize {
        let rank = out_shape.len();
        let offset = rank - operand_shape.len();
        let mut pos = vec![0usize; rank];
        let mut rest = flat;
        for axis in (0..rank).rev() {
            pos[axis] = rest % out_shape[axis];
            rest /= out_shape[axis];
        }
        let (mut slot, mut stride) = (0usize, 1usize);
        for axis in (0..operand_shape.len()).rev() {
            let p = if operand_shape[axis] == 1 {
                0
            } else {
                pos[offset + axis]
            };
            slot += p * stride;
            stride *= operand_shape[axis];
        }
        slot
    }

    /// `apply2` against the independent reference, for every element.
    fn check_apply2_i32(out_shape: &[usize], a_shape: &[usize], b_shape: &[usize]) {
        let count = product(out_shape);
        let a = values(product(a_shape), 7);
        let b = values(product(b_shape), 11);
        let f = |x: i32, y: i32| x.wrapping_mul(3).wrapping_add(y.wrapping_mul(7));
        let mut out = vec![0i32; count];
        apply2(&mut out, out_shape, &a, a_shape, &b, b_shape, f);
        for (flat, produced) in out.iter().enumerate() {
            let expected = f(
                a[reference_slot(out_shape, a_shape, flat)],
                b[reference_slot(out_shape, b_shape, flat)],
            );
            assert_eq!(
                *produced, expected,
                "i32 out {out_shape:?}, a {a_shape:?}, b {b_shape:?}: element {flat}"
            );
        }
    }

    /// The f32 counterpart (bit comparison, so signed zeros stay visible).
    fn check_apply2_f32(out_shape: &[usize], a_shape: &[usize], b_shape: &[usize]) {
        let count = product(out_shape);
        let a: Vec<f32> = values(product(a_shape), 5).iter().map(|v| *v as f32 * 0.5).collect();
        let b: Vec<f32> = values(product(b_shape), 9).iter().map(|v| *v as f32 * 0.25).collect();
        let f = |x: f32, y: f32| x * 1.5 - y;
        let mut out = vec![f32::NAN; count];
        apply2(&mut out, out_shape, &a, a_shape, &b, b_shape, f);
        for (flat, produced) in out.iter().enumerate() {
            let expected = f(
                a[reference_slot(out_shape, a_shape, flat)],
                b[reference_slot(out_shape, b_shape, flat)],
            );
            assert_eq!(
                produced.to_bits(),
                expected.to_bits(),
                "f32 out {out_shape:?}, a {a_shape:?}, b {b_shape:?}: element {flat}"
            );
        }
    }

    /// `apply3` against the independent reference (a row-predicate `select`).
    fn check_apply3_select(
        out_shape: &[usize],
        p_shape: &[usize],
        a_shape: &[usize],
        b_shape: &[usize],
    ) {
        let count = product(out_shape);
        let p: Vec<bool> = values(product(p_shape), 3).iter().map(|v| *v > 0).collect();
        let a = values(product(a_shape), 5);
        let b = values(product(b_shape), 9);
        let f = |pick: bool, x: i32, y: i32| if pick { x } else { y };
        let mut out = vec![0i32; count];
        apply3(
            &mut out, out_shape, &p, p_shape, &a, a_shape, &b, b_shape, f,
        );
        for (flat, produced) in out.iter().enumerate() {
            let expected = f(
                p[reference_slot(out_shape, p_shape, flat)],
                a[reference_slot(out_shape, a_shape, flat)],
                b[reference_slot(out_shape, b_shape, flat)],
            );
            assert_eq!(
                *produced, expected,
                "select out {out_shape:?}, pred {p_shape:?}, a {a_shape:?}, b {b_shape:?}: element {flat}"
            );
        }
    }

    /// `fill_broadcast` against the independent reference.
    fn check_fill(shape: &[usize], out_shape: &[usize]) {
        let source = values(product(shape), 13);
        let filled = fill_broadcast(&source, shape, out_shape);
        assert_eq!(filled.len(), product(out_shape), "{shape:?} -> {out_shape:?}");
        for (flat, produced) in filled.iter().enumerate() {
            let expected = source[reference_slot(out_shape, shape, flat)];
            assert_eq!(*produced, expected, "{shape:?} -> {out_shape:?}: element {flat}");
        }
    }

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
        // The pooled copy serves all 14 storages; bool and the complex and
        // float pairs are the ones a per-dtype arm could silently drop.
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

    #[test]
    fn apply2_agrees_with_the_reference_on_every_layout() {
        // (out, left, right): the broadcast shapes a real graph produces and
        // every alignment family the right-aligned rule has.
        let cases: &[(&[usize], &[usize], &[usize])] = &[
            (&[2, 3], &[1, 3], &[2, 1]),
            (&[2, 2, 2], &[2, 1], &[2]),
            (&[4, 3, 5], &[4, 3, 5], &[5]),
            (&[4, 3, 5], &[1, 1, 5], &[1, 1, 1]),
            (&[4096, 64], &[4096, 64], &[4096, 1]),
            (&[1000, 1000], &[1000, 1000], &[1000]),
            (&[3], &[1], &[3]),
            (&[7, 6], &[7, 6], &[6]),
            (&[5], &[5], &[5]),
            // A per-row scalar, a short trailing tile and a rank-0 operand.
            (&[500, 100], &[500, 100], &[500, 1]),
            (&[500, 50, 4], &[500, 50, 4], &[4]),
            (&[2, 3, 4], &[3, 4], &[2, 1, 1]),
            (&[3, 4], &[3, 1], &[1, 4]),
            (&[6], &[6], &[]),
            (&[], &[], &[]),
        ];
        for (out_shape, a_shape, b_shape) in cases {
            check_apply2_i32(out_shape, a_shape, b_shape);
            check_apply2_f32(out_shape, a_shape, b_shape);
        }
    }

    #[test]
    fn element_i_reads_element_i_of_each_broadcast_operand() {
        // Distinct ranges make any reordering visible: `a` counts rows, `b`
        // counts columns, so a transposed or shifted read cannot pass.
        let out_shape = [2usize, 3];
        let a = [1i32, 2];
        let b = [10i32, 20, 30];
        let mut out = [0i32; 6];
        apply2(&mut out, &out_shape, &a, &[2, 1], &b, &[3], |x, y| x * 100 + y);
        assert_eq!(out, [110, 120, 130, 210, 220, 230]);

        // The same for a per-axis vector against a 3-D result: a `(2, 1)`
        // operand is right-aligned, so it rides out's MIDDLE axis and the
        // result repeats once the middle index rolls over.
        let out_shape = [2usize, 2, 2];
        let a = [1i32, 2];
        let mut out = [0i32; 8];
        apply2(&mut out, &out_shape, &a, &[2, 1], &b[..2], &[2], |x, y| x * 100 + y);
        assert_eq!(out, [110, 120, 210, 220, 110, 120, 210, 220]);
    }

    #[test]
    fn a_rank_zero_operand_and_a_rank_zero_result_are_one_element() {
        // Rank-0 views broadcast to each other and visit exactly once; the
        // rank-1 result with a rank-0 operand agrees element for element.
        check_apply2_i32(&[3, 2], &[1, 1], &[1, 1]);
        check_apply2_i32(&[], &[], &[]);
        let mut out = [0i32; 1];
        apply2(&mut out, &[], &[4], &[], &[5], &[], |x, y| x * y);
        assert_eq!(out, [20]);
    }

    #[test]
    fn an_empty_result_writes_nothing() {
        // A zero-length axis needs no work at all (and indexes no operand).
        let mut out: Vec<i32> = Vec::new();
        apply2(&mut out, &[2, 0, 3], &[2, 0, 3], &[2, 0, 3], &[3], &[3], |x, y| x + y);
        assert!(out.is_empty());
        let mut out: Vec<i32> = Vec::new();
        let pred = [true, false, true];
        apply3(
            &mut out,
            &[2, 0, 3],
            &pred,
            &[3],
            &[2, 0, 3],
            &[2, 0, 3],
            &[3],
            &[3],
            |p, x, y| if p { x } else { y },
        );
        assert!(out.is_empty());
        let filled: Vec<i32> = fill_broadcast(&[1, 2], &[2], &[0, 3]);
        assert!(filled.is_empty());
        let filled: Vec<i32> = fill_broadcast(&[1, 2], &[2], &[0, 2]);
        assert!(filled.is_empty());
    }

    #[test]
    fn nan_signed_zeros_and_wrapping_ride_through_untouched() {
        // The closure owns the semantics; the driver may not reorder, drop or
        // canonicalize a single element. `np.maximum` propagates NaN and keeps
        // the sign of a zero, and i32 adds wrap.
        let maximum = |x: f32, y: f32| {
            if x.is_nan() || y.is_nan() {
                f32::NAN
            } else if x >= y {
                x
            } else {
                y
            }
        };
        // A rank-0 operand holding a NaN: every element is NaN.
        let rows = [f32::NAN, -0.0, 1.0, 0.0, -1.0, 2.0];
        let mut out = [0.0f32; 6];
        apply2(&mut out, &[2, 3], &rows, &[2, 3], &[f32::NAN], &[], maximum);
        assert!(out.iter().all(|value| value.is_nan()));
        // A signed zero against rows: NaN stays NaN, and -0.0 / 0.0 keep
        // their signs where they win.
        let mut out = [0.0f32; 6];
        apply2(&mut out, &[2, 3], &rows, &[2, 3], &[-0.0f32], &[], maximum);
        assert!(out[0].is_nan());
        assert_eq!(out[1].to_bits(), (-0.0f32).to_bits());
        assert_eq!(out[3].to_bits(), 0.0f32.to_bits());
        // i32 wraparound on both the row-scalar and the trailing-tile layout.
        let mut out = [0i32; 4];
        apply2(&mut out, &[2, 2], &[i32::MAX, 1], &[2, 1], &[1, 2], &[2], |x, y| {
            x.wrapping_add(y)
        });
        assert_eq!(out, [i32::MIN, i32::MIN + 1, 2, 3]);
    }

    #[test]
    fn apply3_agrees_with_the_reference_on_every_select_layout() {
        // (out, predicate, on-true, on-false)
        type TernaryCase<'a> = (&'a [usize], &'a [usize], &'a [usize], &'a [usize]);
        let cases: &[TernaryCase<'_>] = &[
            (&[2, 3], &[3], &[2, 3], &[1]),
            (&[2, 3], &[2, 1], &[2, 3], &[2, 3]),
            (&[2, 3], &[2, 1], &[2, 1], &[2, 3]),
            (&[2, 3], &[2, 1], &[2, 3], &[1, 3]),
            (&[2, 3], &[1, 3], &[2, 1], &[2, 3]),
            (&[2, 3], &[2, 3], &[2, 1], &[1, 3]),
            (&[2, 3, 4], &[2, 1, 4], &[2, 3, 4], &[2, 3, 4]),
            (&[2, 3, 4], &[3, 1], &[2, 3, 4], &[2, 1, 4]),
            (&[2, 3], &[1, 1], &[2, 1], &[2, 3]),
            (&[4], &[], &[4], &[4]),
            (&[4], &[4], &[1], &[4]),
            (&[500, 50, 4], &[500, 1, 1], &[500, 50, 4], &[500, 50, 4]),
            (&[1024, 256], &[1024, 1], &[1024, 256], &[256]),
        ];
        for (out_shape, p_shape, a_shape, b_shape) in cases {
            check_apply3_select(out_shape, p_shape, a_shape, b_shape);
        }
    }

    #[test]
    fn fill_broadcast_agrees_with_the_reference() {
        // Not compared against `broadcast_to`: that is `fill_broadcast` now, so
        // the reference has to be independent.
        let cases: &[(&[usize], &[usize])] = &[
            (&[5], &[7, 5]),
            (&[3], &[499, 3]),
            (&[1], &[9, 4, 1]),
            (&[6], &[2, 3, 6]),
            (&[4], &[2000, 50, 4]),
            (&[], &[2, 3]),
            (&[2, 3], &[2, 3]),
            (&[500, 1, 50], &[500, 4, 50]),
            (&[500, 1], &[500, 100]),
            (&[1, 1, 1], &[5, 1, 4]),
            (&[3, 4], &[2, 3, 4]),
        ];
        for (shape, out_shape) in cases {
            check_fill(shape, out_shape);
        }
    }

    #[test]
    fn a_pooled_broadcast_result_is_fully_overwritten() {
        // A parked garbage stock of the result's exact size must not leak a
        // single NaN into the result: the broadcast map writes every element.
        let out_shape = [2000usize, 50, 4];
        let count = product(&out_shape);
        let left: Vec<f32> = (0..count).map(|index| (index % 97) as f32 * 0.5).collect();
        let right = [0.5f32, -1.5, 2.0, 3.25];
        crate::buffer::test_support::garbage_f32(count);
        let result = crate::buffer::map2_runs(
            DType::Float32,
            &out_shape,
            &left,
            &out_shape,
            &right,
            &[4],
            |a, b| a + b,
        )
        .unwrap();
        match &result.storage {
            Storage::F32(produced) => {
                assert_eq!(produced.len(), count);
                for (flat, value) in produced.iter().enumerate() {
                    let expected = left[flat] + right[flat % 4];
                    assert_eq!(
                        value.to_bits(),
                        expected.to_bits(),
                        "element {flat} was not rewritten"
                    );
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }

    #[test]
    fn the_view_path_agrees_with_the_flat_fast_path() {
        // `map2`/`map3` are the flat typed-slice builders and `apply2`/`apply3`
        // are the ndarray layout drivers: over full-shape operands both must
        // produce the identical storage.
        let out_shape = [3usize, 4];
        let count = product(&out_shape);
        let left: Vec<f32> = (0..count).map(|index| index as f32 * 0.5 - 3.0).collect();
        let right: Vec<f32> = (0..count).map(|index| ((index * 7) % 11) as f32).collect();
        let f2 = |a: f32, b: f32| a * 2.0 - b;
        let flat = crate::buffer::map2(DType::Float32, &out_shape, &left, &right, f2).unwrap();
        let mut driven = vec![0.0f32; count];
        apply2(&mut driven, &out_shape, &left, &out_shape, &right, &out_shape, f2);
        assert_eq!(flat.storage, Storage::F32(driven));

        let pred: Vec<bool> = (0..count).map(|index| index % 3 == 0).collect();
        let f3 = |p: bool, x: f32, y: f32| if p { x } else { y };
        let flat3 = crate::buffer::map3(
            DType::Float32, &out_shape, &pred, &left, &right, f3,
        )
        .unwrap();
        let mut driven3 = vec![0.0f32; count];
        apply3(
            &mut driven3,
            &out_shape,
            &pred,
            &out_shape,
            &left,
            &out_shape,
            &right,
            &out_shape,
            f3,
        );
        assert_eq!(flat3.storage, Storage::F32(driven3));
    }
}
