//! Typed host buffers: the engine's value representation.
//!
//! One [`Buffer`] is a dtype, a concrete shape and a `Vec` of that dtype's
//! Rust scalar.  Data is always C-contiguous (numpy order); view ops such as
//! `transpose` materialize a copied result, which keeps every kernel a flat
//! loop over plain slices.
//!
//! The two exported macros [`with_type!`] and [`with_same_type!`] let a kernel
//! body stay generic (`T: Num`) while the storage enum is matched once per
//! call site, so a new op costs a generic function, not 14 copies.
use evoxc_core::{ArrayPayload, DType, Error, Result};
use half::f16;
use num_complex::Complex;

use crate::scalar::Scalar;

/// One typed, C-contiguous value buffer.
#[derive(Debug, Clone, PartialEq)]
pub enum Storage {
    Bool(Vec<bool>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    F16(Vec<f16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    C64(Vec<Complex<f32>>),
    C128(Vec<Complex<f64>>),
}

impl Storage {
    pub fn dtype(&self) -> DType {
        match self {
            Storage::Bool(_) => DType::Bool,
            Storage::I8(_) => DType::Int8,
            Storage::I16(_) => DType::Int16,
            Storage::I32(_) => DType::Int32,
            Storage::I64(_) => DType::Int64,
            Storage::U8(_) => DType::UInt8,
            Storage::U16(_) => DType::UInt16,
            Storage::U32(_) => DType::UInt32,
            Storage::U64(_) => DType::UInt64,
            Storage::F16(_) => DType::Float16,
            Storage::F32(_) => DType::Float32,
            Storage::F64(_) => DType::Float64,
            Storage::C64(_) => DType::Complex64,
            Storage::C128(_) => DType::Complex128,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Storage::Bool(v) => v.len(),
            Storage::I8(v) => v.len(),
            Storage::I16(v) => v.len(),
            Storage::I32(v) => v.len(),
            Storage::I64(v) => v.len(),
            Storage::U8(v) => v.len(),
            Storage::U16(v) => v.len(),
            Storage::U32(v) => v.len(),
            Storage::U64(v) => v.len(),
            Storage::F16(v) => v.len(),
            Storage::F32(v) => v.len(),
            Storage::F64(v) => v.len(),
            Storage::C64(v) => v.len(),
            Storage::C128(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One materialized value: dtype + concrete shape + typed storage.
#[derive(Debug, Clone, PartialEq)]
pub struct Buffer {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub storage: Storage,
}

impl Buffer {
    /// Build a buffer, checking that the storage variant matches the dtype.
    pub fn new(dtype: DType, shape: Vec<usize>, storage: Storage) -> Result<Buffer> {
        if storage.dtype() != dtype {
            return Err(Error::dtype(format!(
                "buffer storage is {} but the declared dtype is {}",
                storage.dtype().name(),
                dtype.name()
            )));
        }
        let expected: usize = shape.iter().product();
        if storage.len() != expected {
            return Err(Error::shape(format!(
                "buffer of shape {shape:?} needs {expected} element(s), got {}",
                storage.len()
            )));
        }
        Ok(Buffer { dtype, shape, storage })
    }

    /// True when this buffer already has `shape` and exactly that many
    /// elements -- i.e. element `i` of it is element `i` of an output of that
    /// shape, so a kernel may take a flat, broadcast-free fast path.
    pub fn is_flat_for(&self, shape: &[usize]) -> bool {
        self.shape == shape && self.storage.len() == shape.iter().product::<usize>()
    }
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// A zero-filled buffer of the given dtype and shape.
    pub fn zeros(dtype: DType, shape: Vec<usize>) -> Buffer {
        let count: usize = shape.iter().product();
        let storage = match dtype {
            DType::Bool => Storage::Bool(vec![false; count]),
            DType::Int8 => Storage::I8(vec![0; count]),
            DType::Int16 => Storage::I16(vec![0; count]),
            DType::Int32 => Storage::I32(vec![0; count]),
            DType::Int64 => Storage::I64(vec![0; count]),
            DType::UInt8 => Storage::U8(vec![0; count]),
            DType::UInt16 => Storage::U16(vec![0; count]),
            DType::UInt32 => Storage::U32(vec![0; count]),
            DType::UInt64 => Storage::U64(vec![0; count]),
            DType::Float16 => Storage::F16(vec![f16::from_f32(0.0); count]),
            DType::Float32 => Storage::F32(vec![0.0; count]),
            DType::Float64 => Storage::F64(vec![0.0; count]),
            DType::Complex64 => Storage::C64(vec![Complex::new(0.0, 0.0); count]),
            DType::Complex128 => Storage::C128(vec![Complex::new(0.0, 0.0); count]),
        };
        Buffer { dtype, shape, storage }
    }

    /// Decode a raw little-endian constant payload from the envelope.
    pub fn from_payload(payload: &ArrayPayload) -> Result<Buffer> {
        let shape = payload
            .shape
            .iter()
            .map(|dim| {
                usize::try_from(*dim).map_err(|_| {
                    Error::shape(format!("constant payload has a negative dimension ({dim})"))
                })
            })
            .collect::<Result<Vec<usize>>>()?;
        let expected = payload.byte_len()?;
        if payload.data.len() != expected {
            return Err(Error::verification(format!(
                "constant payload declares {expected} byte(s) but carries {}",
                payload.data.len()
            )));
        }
        let storage = decode_storage(payload.dtype, &payload.data)?;
        Buffer::new(payload.dtype, shape, storage)
    }

    /// Convert to another dtype (numpy `astype` semantics; never silent — the
    /// caller decides which dtype an op computes in).
    pub fn cast(&self, dtype: DType) -> Result<Buffer> {
        if self.dtype == dtype {
            return Ok(self.clone());
        }
        let storage = cast_storage(&self.storage, dtype);
        Buffer::new(dtype, self.shape.clone(), storage)
    }

    /// Replace the shape without moving data (same element count).
    pub fn with_shape(mut self, shape: Vec<usize>) -> Result<Buffer> {
        if shape.iter().product::<usize>() != self.numel() {
            return Err(Error::shape(format!(
                "cannot reshape {:?} to {shape:?}: element count differs",
                self.shape
            )));
        }
        self.shape = shape;
        Ok(self)
    }
}

/// Render a shape in numpy style (`(3, 4)`), used in diagnostic messages.
pub fn shape_text(shape: &[usize]) -> String {
    let parts: Vec<String> = shape.iter().map(|dim| dim.to_string()).collect();
    format!("({})", parts.join(", "))
}

fn decode_storage(dtype: DType, bytes: &[u8]) -> Result<Storage> {
    let storage = match dtype {
        DType::Bool => Storage::Bool(bytes.iter().map(|byte| *byte != 0).collect()),
        DType::Int8 => Storage::I8(bytes.iter().map(|byte| *byte as i8).collect()),
        DType::UInt8 => Storage::U8(bytes.to_vec()),
        DType::Int16 => Storage::I16(decode_le::<i16>(bytes)),
        DType::UInt16 => Storage::U16(decode_le::<u16>(bytes)),
        DType::Float16 => Storage::F16(
            bytes
                .chunks_exact(2)
                .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
                .collect(),
        ),
        DType::Int32 => Storage::I32(decode_le::<i32>(bytes)),
        DType::UInt32 => Storage::U32(decode_le::<u32>(bytes)),
        DType::Float32 => Storage::F32(decode_le::<f32>(bytes)),
        DType::Int64 => Storage::I64(decode_le::<i64>(bytes)),
        DType::UInt64 => Storage::U64(decode_le::<u64>(bytes)),
        DType::Float64 => Storage::F64(decode_le::<f64>(bytes)),
        DType::Complex64 => Storage::C64(decode_complex::<f32>(bytes)),
        DType::Complex128 => Storage::C128(decode_complex::<f64>(bytes)),
    };
    Ok(storage)
}

fn decode_le<T>(bytes: &[u8]) -> Vec<T>
where
    T: Copy + LeScalar,
{
    bytes
        .chunks_exact(T::SIZE)
        .map(|chunk| T::from_le(chunk))
        .collect()
}

fn decode_complex<T: ComplexPart>(bytes: &[u8]) -> Vec<Complex<T>> {
    bytes
        .chunks_exact(T::SIZE * 2)
        .map(|chunk| {
            let re = T::from_le(&chunk[..T::SIZE]);
            let im = T::from_le(&chunk[T::SIZE..]);
            Complex::new(re, im)
        })
        .collect()
}

trait LeScalar: Copy {
    const SIZE: usize;
    fn from_le(bytes: &[u8]) -> Self;
}

trait ComplexPart: Copy {
    const SIZE: usize;
    fn from_le(bytes: &[u8]) -> Self;
}

macro_rules! impl_le {
    ($ty:ty) => {
        impl LeScalar for $ty {
            const SIZE: usize = std::mem::size_of::<$ty>();
            fn from_le(bytes: &[u8]) -> Self {
                let mut buf = [0u8; std::mem::size_of::<$ty>()];
                buf.copy_from_slice(bytes);
                <$ty>::from_le_bytes(buf)
            }
        }
    };
}

impl_le!(i16);
impl_le!(u16);
impl_le!(i32);
impl_le!(u32);
impl_le!(i64);
impl_le!(u64);
impl_le!(f32);
impl_le!(f64);

macro_rules! impl_complex_part {
    ($ty:ty) => {
        impl ComplexPart for $ty {
            const SIZE: usize = std::mem::size_of::<$ty>();
            fn from_le(bytes: &[u8]) -> Self {
                let mut buf = [0u8; std::mem::size_of::<$ty>()];
                buf.copy_from_slice(bytes);
                <$ty>::from_le_bytes(buf)
            }
        }
    };
}

impl_complex_part!(f32);
impl_complex_part!(f64);

fn cast_storage(source: &Storage, target: DType) -> Storage {
    macro_rules! convert {
        ($values:expr) => {
            convert_generic($values, target)
        };
    }
    match source {
        Storage::Bool(v) => convert!(v),
        Storage::I8(v) => convert!(v),
        Storage::I16(v) => convert!(v),
        Storage::I32(v) => convert!(v),
        Storage::I64(v) => convert!(v),
        Storage::U8(v) => convert!(v),
        Storage::U16(v) => convert!(v),
        Storage::U32(v) => convert!(v),
        Storage::U64(v) => convert!(v),
        Storage::F16(v) => convert!(v),
        Storage::F32(v) => convert!(v),
        Storage::F64(v) => convert!(v),
        Storage::C64(v) => convert!(v),
        Storage::C128(v) => convert!(v),
    }
}

fn convert_generic<S: Scalar>(values: &[S], target: DType) -> Storage {
    macro_rules! to {
        ($ty:ty) => {
            <$ty as Scalar>::into_storage(
                values
                    .iter()
                    .map(|value| <$ty as Scalar>::from_scalar(value.to_scalar()))
                    .collect(),
            )
        };
    }
    match target {
        DType::Bool => to!(bool),
        DType::Int8 => to!(i8),
        DType::Int16 => to!(i16),
        DType::Int32 => to!(i32),
        DType::Int64 => to!(i64),
        DType::UInt8 => to!(u8),
        DType::UInt16 => to!(u16),
        DType::UInt32 => to!(u32),
        DType::UInt64 => to!(u64),
        DType::Float16 => to!(f16),
        DType::Float32 => to!(f32),
        DType::Float64 => to!(f64),
        DType::Complex64 => to!(Complex<f32>),
        DType::Complex128 => to!(Complex<f64>),
    }
}

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

// ---------------------------------------------------------------------------
// Broadcast-free (flat) drivers
// ---------------------------------------------------------------------------
//
// The `walk`/`applyN` machines above serve every strided or broadcast kernel:
// they recompute one offset per operand per element.  A kernel whose operands
// all already have the output shape needs none of that -- element `i` of every
// operand is element `i` of the output, so the kernel is one flat `zip` over
// plain slices.  These drivers are monomorphized over the element types AND the
// closure, which is what lets LLVM turn the loop into vector moves; keep them
// that way (no `&dyn Fn`, no per-element `match` on an op code -- dispatch the
// op code once in the caller and let each arm's closure fold to one operation).
//
// Callers gate on [`Buffer::is_flat_for`], so every slice has the output's
// length; the `zip` therefore visits every element.  A slice that is somehow
// shorter would be a broken internal buffer, not user input.
/// Flat unary map: `out[i] = f(source[i])`.
pub fn zip1<T: Copy, R: Copy>(out: &mut [R], source: &[T], f: impl Fn(T) -> R) {
    for (slot, value) in out.iter_mut().zip(source.iter()) {
        *slot = f(*value);
    }
}
/// Flat binary map: `out[i] = f(left[i], right[i])`.
pub fn zip2<T: Copy, R: Copy>(out: &mut [R], left: &[T], right: &[T], f: impl Fn(T, T) -> R) {
    for ((slot, x), y) in out.iter_mut().zip(left.iter()).zip(right.iter()) {
        *slot = f(*x, *y);
    }
}
/// Flat ternary map: `out[i] = f(first[i], second[i], third[i])` (the `select`
/// shape: predicate, on-true, on-false).
pub fn zip3<P: Copy, T: Copy, R: Copy>(
    out: &mut [R],
    first: &[P],
    second: &[T],
    third: &[T],
    f: impl Fn(P, T, T) -> R,
) {
    for (((slot, p), x), y) in out
        .iter_mut()
        .zip(first.iter())
        .zip(second.iter())
        .zip(third.iter())
    {
        *slot = f(*p, *x, *y);
    }
}
/// Flat unary builder: `out[i] = f(source[i])` over `out_shape`'s elements,
/// with the result storage built straight from the mapped iterator.
///
/// Unlike [`zip1`], this never zero-fills a buffer first: `Vec` collects a
/// `Map` over a slice with an exact size hint, so the result is allocated once
/// and written once.  Elementwise kernels that produce a fresh result use
/// these; the in-place `zip*` helpers stay for kernels that already own the
/// memory they write into.  The `Buffer::new` validation is defensive (a
/// caller gates on [`Buffer::is_flat_for`], so the element count and the dtype
/// already agree) and keeps a short or long operand an explicit error rather
/// than a silently truncated result.
pub fn map1<T: Copy, R: Scalar>(
    dtype: DType,
    out_shape: &[usize],
    source: &[T],
    f: impl Fn(T) -> R,
) -> Result<Buffer> {
    let values: Vec<R> = source.iter().map(|value| f(*value)).collect();
    Buffer::new(dtype, out_shape.to_vec(), R::into_storage(values))
}

/// Flat binary builder: `out[i] = f(left[i], right[i])` (see [`map1`]).
pub fn map2<T: Copy, R: Scalar>(
    dtype: DType,
    out_shape: &[usize],
    left: &[T],
    right: &[T],
    f: impl Fn(T, T) -> R,
) -> Result<Buffer> {
    let values: Vec<R> = left.iter().zip(right.iter()).map(|(x, y)| f(*x, *y)).collect();
    Buffer::new(dtype, out_shape.to_vec(), R::into_storage(values))
}

/// Flat ternary builder: `out[i] = f(first[i], second[i], third[i])` -- the
/// `select` shape (see [`map1`]).
pub fn map3<P: Copy, T: Copy, R: Scalar>(
    dtype: DType,
    out_shape: &[usize],
    first: &[P],
    second: &[T],
    third: &[T],
    f: impl Fn(P, T, T) -> R,
) -> Result<Buffer> {
    let values: Vec<R> = first
        .iter()
        .zip(second.iter())
        .zip(third.iter())
        .map(|((p, x), y)| f(*p, *x, *y))
        .collect();
    Buffer::new(dtype, out_shape.to_vec(), R::into_storage(values))
}

fn block_error(op: &str) -> Error {
    Error::verification(format!(
        "op '{op}': a block copy reads outside the operand (internal engine error)"
    ))
}
/// Block copy: output block `b` is the `inner`-element run starting at
/// `source[offsets[b]]` (the `gather`/`slice`/`tile` fast path: whole index-axis
/// rows instead of one offset per element).
pub fn copy_blocks<T: Copy>(
    out: &mut [T],
    source: &[T],
    offsets: &[usize],
    inner: usize,
    op: &str,
) -> Result<()> {
    if offsets.len().saturating_mul(inner) != out.len() {
        return Err(block_error(op));
    }
    if inner == 0 {
        return Ok(());
    }
    for (block, offset) in offsets.iter().enumerate() {
        let start = *offset;
        let run = start..start + inner;
        let src = source.get(run).ok_or_else(|| block_error(op))?;
        let slot = block * inner;
        let dst = out.get_mut(slot..slot + inner).ok_or_else(|| block_error(op))?;
        dst.copy_from_slice(src);
    }
    Ok(())
}
/// A [`Storage`] built by [`copy_blocks`] over `values`.
pub fn block_storage<T: Scalar>(
    values: &[T],
    out_shape: &[usize],
    offsets: &[usize],
    inner: usize,
    op: &str,
) -> Result<Storage> {
    let total: usize = out_shape.iter().product();
    let mut out = vec![T::zero(); total];
    copy_blocks(&mut out, values, offsets, inner, op)?;
    Ok(T::into_storage(out))
}
fn broadcast_vec<T: Copy>(values: &[T], shape: &[usize], out_shape: &[usize]) -> Vec<T> {
    let mut out = Vec::with_capacity(out_shape.iter().product());
    let strides = [broadcast_strides(shape, out_shape)];
    walk(out_shape.len(), out_shape, &strides, |offsets| {
        out.push(values[offsets[0]]);
    });
    out
}

// ---------------------------------------------------------------------------
// Typed-storage dispatch macros
// ---------------------------------------------------------------------------

/// Match a single buffer's storage, binding the typed `Vec` to one identifier.
#[macro_export]
macro_rules! with_type {
    ($buffer:expr, |$values:ident| $body:expr) => {
        match &$buffer.storage {
            $crate::buffer::Storage::Bool($values) => $body,
            $crate::buffer::Storage::I8($values) => $body,
            $crate::buffer::Storage::I16($values) => $body,
            $crate::buffer::Storage::I32($values) => $body,
            $crate::buffer::Storage::I64($values) => $body,
            $crate::buffer::Storage::U8($values) => $body,
            $crate::buffer::Storage::U16($values) => $body,
            $crate::buffer::Storage::U32($values) => $body,
            $crate::buffer::Storage::U64($values) => $body,
            $crate::buffer::Storage::F16($values) => $body,
            $crate::buffer::Storage::F32($values) => $body,
            $crate::buffer::Storage::F64($values) => $body,
            $crate::buffer::Storage::C64($values) => $body,
            $crate::buffer::Storage::C128($values) => $body,
        }
    };
}

/// Match two buffers of the same dtype, binding both typed vectors.
#[macro_export]
macro_rules! with_same_type {
    ($left:expr, $right:expr, |$a:ident, $b:ident| $body:expr) => {
        match (&$left.storage, &$right.storage) {
            ($crate::buffer::Storage::Bool($a), $crate::buffer::Storage::Bool($b)) => $body,
            ($crate::buffer::Storage::I8($a), $crate::buffer::Storage::I8($b)) => $body,
            ($crate::buffer::Storage::I16($a), $crate::buffer::Storage::I16($b)) => $body,
            ($crate::buffer::Storage::I32($a), $crate::buffer::Storage::I32($b)) => $body,
            ($crate::buffer::Storage::I64($a), $crate::buffer::Storage::I64($b)) => $body,
            ($crate::buffer::Storage::U8($a), $crate::buffer::Storage::U8($b)) => $body,
            ($crate::buffer::Storage::U16($a), $crate::buffer::Storage::U16($b)) => $body,
            ($crate::buffer::Storage::U32($a), $crate::buffer::Storage::U32($b)) => $body,
            ($crate::buffer::Storage::U64($a), $crate::buffer::Storage::U64($b)) => $body,
            ($crate::buffer::Storage::F16($a), $crate::buffer::Storage::F16($b)) => $body,
            ($crate::buffer::Storage::F32($a), $crate::buffer::Storage::F32($b)) => $body,
            ($crate::buffer::Storage::F64($a), $crate::buffer::Storage::F64($b)) => $body,
            ($crate::buffer::Storage::C64($a), $crate::buffer::Storage::C64($b)) => $body,
            ($crate::buffer::Storage::C128($a), $crate::buffer::Storage::C128($b)) => $body,
            _ => {
                return Err(Error::dtype(format!(
                    "dtype mismatch: {} vs {}",
                    $left.dtype.name(),
                    $right.dtype.name()
                )))
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn map_builders_skip_the_zero_fill_and_validate() {
        let left = [1.0f32, -2.0, 3.0];
        let right = [2.0f32, 2.0, 0.5];
        let unary = map1(DType::Float32, &[3], &left, |value| value * 2.0).unwrap();
        assert_eq!(unary.storage, Storage::F32(vec![2.0, -4.0, 6.0]));
        assert_eq!(unary.shape, vec![3]);
        let binary = map2(DType::Float32, &[3], &left, &right, |a, b| a + b).unwrap();
        assert_eq!(binary.storage, Storage::F32(vec![3.0, 0.0, 3.5]));
        let mask = [true, false, true];
        let ternary =
            map3(DType::Float32, &[3], &mask, &left, &right, |pick, a, b| if pick { a } else { b })
                .unwrap();
        assert_eq!(ternary.storage, Storage::F32(vec![1.0, 2.0, 3.0]));

        // The shape is authoritative: a short or long operand is an explicit
        // error instead of a silently truncated result.
        assert!(map1(DType::Float32, &[4], &left, |value| value).is_err());
        assert!(map2(DType::Float32, &[2], &left, &right, |a, b| a + b).is_err());
        // And a dtype that disagrees with the value type is rejected there too.
        assert!(map1(DType::Float64, &[3], &left, |value| value).is_err());
    }
}
