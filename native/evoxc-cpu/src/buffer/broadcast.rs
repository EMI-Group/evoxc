//! Broadcasting: the numpy shape/stride rules, the ndarray-backed elementwise
//! layout drivers (`walk` / `apply1..3`) and `broadcast_to`.
//!
//! [`Storage`] is one C-contiguous `Vec` per dtype, so an operand read through a
//! broadcast (or any other non-contiguous) layout has to be addressed through
//! its shape.  ndarray's broadcast views are the ONLY N-dimensional layout
//! engine here: [`walk`] is a thin index iteration, `apply1..3` are `Zip` passes
//! and [`fill_broadcast`] is one `assign` of a broadcast view, so no hand-rolled
//! stride odometer remains.  A size-1 operand is captured as a scalar and dropped
//! from that `Zip` ([`fill_broadcast`] becomes one `fill`), so it costs no
//! per-element zero-stride producer work.  Every driver falls back to the
//! [`walk`] loop for a layout a view cannot express (an operand slice whose
//! length disagrees with its shape, an operand of higher rank than the result, a
//! result slice whose length disagrees with its shape), which keeps the
//! pre-existing accept/reject/panic behaviour byte-identical.
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

/// The value of a SIZE-1 operand: exactly one element whose own shape is also a
/// single element (rank 0, or any `[1, 1, ...]`).
///
/// Such an operand broadcasts to a constant, so the elementwise drivers capture
/// its value and drop that zero-stride `Zip` producer instead of paying its
/// offset arithmetic once per result element.  The SHAPE is consulted as well as
/// the slice length on purpose: a one-element slice whose shape claims more than
/// one element is a malformed operand, not a scalar, so it keeps the [`walk`]
/// fallback's behaviour (never a silent fill).
#[inline]
fn size_one_value<T: Copy>(values: &[T], shape: &[usize]) -> Option<T> {
    if values.len() == 1 && shape.iter().product::<usize>() == 1 {
        Some(values[0])
    } else {
        None
    }
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
///
/// A size-1 source ([`size_one_value`]) is captured as a constant and dropped
/// from the `Zip`, so the pass is a plain write of `f(value)` over `out`.
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
    let Ok(dest) = ArrayViewMut::<R, D>::from_shape(out_shape.clone(), out) else {
        return false;
    };
    if let Some(value) = size_one_value(a, a_shape) {
        ndarray::Zip::from(dest).for_each(|o| *o = f(value));
        return true;
    }
    let Some(a_base) = base_view(a, a_shape) else {
        return false;
    };
    let Some(av) = a_base.broadcast(out_shape) else {
        return false;
    };
    ndarray::Zip::from(dest).and(av).for_each(|o, x| *o = f(*x));
    true
}

/// `out[i] = f(a[i], b[i])` via one ndarray `Zip` pass over static dims (`false`
/// when the layout cannot be expressed as views).
///
/// A size-1 operand ([`size_one_value`]) is captured as a constant and dropped
/// from the `Zip`, so a rank-0 or `[1, 1]` operand costs no per-element offset
/// work; `a` stays the left (true-branch) argument and `b` the right in every
/// arm.
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
    // A size-1 operand always views and broadcasts successfully, so these two
    // base views can be built unconditionally: the reachable outcomes (view,
    // or the `walk` fallback) are exactly the pre-existing ones.
    let Some(a_base) = base_view(a, a_shape) else {
        return false;
    };
    let Some(b_base) = base_view(b, b_shape) else {
        return false;
    };
    let Ok(dest) = ArrayViewMut::<R, D>::from_shape(out_shape.clone(), out) else {
        return false;
    };
    match (size_one_value(a, a_shape), size_one_value(b, b_shape)) {
        (Some(x), Some(y)) => {
            ndarray::Zip::from(dest).for_each(|o| *o = f(x, y));
        }
        (Some(x), None) => {
            let Some(bv) = b_base.broadcast(out_shape) else {
                return false;
            };
            ndarray::Zip::from(dest).and(bv).for_each(|o, y| *o = f(x, *y));
        }
        (None, Some(y)) => {
            let Some(av) = a_base.broadcast(out_shape) else {
                return false;
            };
            ndarray::Zip::from(dest).and(av).for_each(|o, x| *o = f(*x, y));
        }
        (None, None) => {
            let (Some(av), Some(bv)) = (a_base.broadcast(out_shape.clone()), b_base.broadcast(out_shape))
            else {
                return false;
            };
            ndarray::Zip::from(dest)
                .and(av)
                .and(bv)
                .for_each(|o, x, y| *o = f(*x, *y));
        }
    }
    true
}

/// `out[i] = f(a[i], b[i], c[i])` via one ndarray `Zip` pass over static dims
/// (`false` when the layout cannot be expressed as views).
///
/// Each size-1 operand ([`size_one_value`]) is captured as a constant and
/// dropped from the `Zip` — the seven arms are the seven non-empty
/// combinations — and the argument order `f(pred, on_true, on_false)` holds in
/// every one of them.
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
    // See `apply2_static`: a size-1 operand can never fail to view/broadcast.
    let Some(a_base) = base_view(a, a_shape) else {
        return false;
    };
    let Some(b_base) = base_view(b, b_shape) else {
        return false;
    };
    let Some(c_base) = base_view(c, c_shape) else {
        return false;
    };
    let Ok(dest) = ArrayViewMut::<R, D>::from_shape(out_shape.clone(), out) else {
        return false;
    };
    // `(pred, on_true, on_false)` — `Some` is a captured scalar, `None` keeps
    // that operand's broadcast view in the `Zip`.
    match (
        size_one_value(a, a_shape),
        size_one_value(b, b_shape),
        size_one_value(c, c_shape),
    ) {
        (Some(x), Some(y), Some(z)) => {
            ndarray::Zip::from(dest).for_each(|o| *o = f(x, y, z));
        }
        (Some(x), Some(y), None) => {
            let Some(cv) = c_base.broadcast(out_shape) else {
                return false;
            };
            ndarray::Zip::from(dest)
                .and(cv)
                .for_each(|o, z| *o = f(x, y, *z));
        }
        (Some(x), None, Some(z)) => {
            let Some(bv) = b_base.broadcast(out_shape) else {
                return false;
            };
            ndarray::Zip::from(dest)
                .and(bv)
                .for_each(|o, y| *o = f(x, *y, z));
        }
        (None, Some(y), Some(z)) => {
            let Some(av) = a_base.broadcast(out_shape) else {
                return false;
            };
            ndarray::Zip::from(dest)
                .and(av)
                .for_each(|o, x| *o = f(*x, y, z));
        }
        (Some(x), None, None) => {
            let (Some(bv), Some(cv)) = (b_base.broadcast(out_shape.clone()), c_base.broadcast(out_shape))
            else {
                return false;
            };
            ndarray::Zip::from(dest)
                .and(bv)
                .and(cv)
                .for_each(|o, y, z| *o = f(x, *y, *z));
        }
        (None, Some(y), None) => {
            let (Some(av), Some(cv)) = (a_base.broadcast(out_shape.clone()), c_base.broadcast(out_shape))
            else {
                return false;
            };
            ndarray::Zip::from(dest)
                .and(av)
                .and(cv)
                .for_each(|o, x, z| *o = f(*x, y, *z));
        }
        (None, None, Some(z)) => {
            let (Some(av), Some(bv)) = (a_base.broadcast(out_shape.clone()), b_base.broadcast(out_shape))
            else {
                return false;
            };
            ndarray::Zip::from(dest)
                .and(av)
                .and(bv)
                .for_each(|o, x, y| *o = f(*x, *y, z));
        }
        (None, None, None) => {
            let (Some(av), Some(bv), Some(cv)) = (
                a_base.broadcast(out_shape.clone()),
                b_base.broadcast(out_shape.clone()),
                c_base.broadcast(out_shape),
            ) else {
                return false;
            };
            ndarray::Zip::from(dest)
                .and(av)
                .and(bv)
                .and(cv)
                .for_each(|o, x, y, z| *o = f(*x, *y, *z));
        }
    }
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
///
/// A size-1 source ([`size_one_value`]) is one `fill` of its single value
/// instead of a broadcast `assign`.
fn fill_broadcast_static<T: Copy, D: Dimension>(
    out_shape: D,
    values: &[T],
    shape: &[usize],
    out: &mut [T],
) -> bool {
    if out.len() != out_shape.size() {
        return false;
    }
    if let Some(value) = size_one_value(values, shape) {
        let Ok(mut dest) = ArrayViewMut::<T, D>::from_shape(out_shape, out) else {
            return false;
        };
        dest.fill(value);
        return true;
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
mod tests;
