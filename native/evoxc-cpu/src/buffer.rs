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

use std::cell::RefCell;

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
    /// Bytes of element storage this value holds (not its capacity).
    fn bytes(&self) -> usize {
        fn size<T>(values: &[T]) -> usize {
            std::mem::size_of_val(values)
        }
        match self {
            Storage::Bool(v) => size(v),
            Storage::I8(v) => size(v),
            Storage::I16(v) => size(v),
            Storage::I32(v) => size(v),
            Storage::I64(v) => size(v),
            Storage::U8(v) => size(v),
            Storage::U16(v) => size(v),
            Storage::U32(v) => size(v),
            Storage::U64(v) => size(v),
            Storage::F16(v) => size(v),
            Storage::F32(v) => size(v),
            Storage::F64(v) => size(v),
            Storage::C64(v) => size(v),
            Storage::C128(v) => size(v),
        }
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

/// Decode `bytes`' little-endian words into a vector, reusing a pooled
/// allocation of exactly that many elements when one is available (a caller
/// decodes fresh multi-megabyte payloads per run, so this is the same
/// page-fault argument as for a result — see the pool note below).
fn decode_le<T>(bytes: &[u8]) -> Vec<T>
where
    T: Copy + LeScalar + Scalar,
{
    let count = bytes.len() / T::SIZE;
    if let Some(mut values) = take_pooled_vec::<T>(count) {
        let threads = parallel_threads(count);
        if threads > 1 {
            split_parallel(&mut values, threads, |offset, chunk| {
                let start = offset * T::SIZE;
                let len = chunk.len() * T::SIZE;
                let window = bytes.get(start..start + len).unwrap_or(&[]);
                for (slot, word) in chunk.iter_mut().zip(window.chunks_exact(T::SIZE)) {
                    *slot = T::from_le(word);
                }
            });
        } else {
            for (slot, chunk) in values.iter_mut().zip(bytes.chunks_exact(T::SIZE)) {
                *slot = T::from_le(chunk);
            }
        }
        return values;
    }
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
//
// These maps are DRAM-bandwidth bound at ~0.21 ms per 1e6 f32 elements on one
// thread (numpy's ufuncs are single-threaded, so that alone only reaches
// parity), which is why a big enough result is split across a few scoped
// threads: four threads measured ~2x the one-thread bandwidth on a 16-cpu host
// (eight and sixteen were slower -- they contend for the same memory ports).
// Scoped-thread setup costs ~10 us per extra thread, so the split only pays for
// a genuinely large result: measured on this host (`bench_thread_crossover`) one
// flat `add`, 1 thread / 4 threads, best of 40 --
//
//   2^16 elems 0.007 / 0.036 ms   2^18 elems 0.030 / 0.043 ms
//   2^17 elems 0.014 / 0.038 ms   2^19 elems 0.100 / 0.049 ms
//   1e6 elems  0.249 / 0.097 ms
//
// -- so the crossover sits between 2^18 and 2^19 elements: at 2^18 four threads
// are 43 % SLOWER than the calling thread, and the 2x win only starts at 2^19
// (which is why the threshold is 2^19 and not 2^18; the `(4096, 64)` f32 tensors
// of a merged-GP-shaped graph are exactly 2^18 elements and belong on one
// thread).  Small maps stay on the calling thread, where spawning would
// dominate.
/// Element count from which a flat map is worth splitting across threads.
const PARALLEL_MIN_ELEMS: usize = 1 << 18;
/// Elements one thread should own before it earns its setup cost: a split adds
/// `count / PARALLEL_MIN_PER_THREAD` threads (capped by the core count), so a
/// 2^18-element map (a `(4096, 64)` f32 tensor) runs on two threads -- measured
/// faster than the calling thread alone, unlike four -- while a >= 2^19-element
/// one still gets the full four.
const PARALLEL_MIN_PER_THREAD: usize = 1 << 17;
/// Upper bound on the threads one flat map uses (the measured optimum).
const PARALLEL_MAX_THREADS: usize = 4;

/// How many threads a flat map over `count` elements should split across.
fn parallel_threads(count: usize) -> usize {
    if count < PARALLEL_MIN_ELEMS {
        return 1;
    }
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let cores = *THREADS.get_or_init(|| {
        std::thread::available_parallelism()
            .map(|threads| threads.get())
            .unwrap_or(1)
            .clamp(1, PARALLEL_MAX_THREADS)
    });
    cores.min((count / PARALLEL_MIN_PER_THREAD).max(1))
}

/// Run `body(offset, chunk)` over disjoint chunks of `out` on up to `threads`
/// scoped threads.  Every chunk is visited exactly once, so a body that writes
/// all of its chunk writes all of `out`; `offset` is the chunk's start index,
/// which is how a caller slices its operands to match.
fn split_parallel<R: Copy + Send, F: Fn(usize, &mut [R]) + Sync>(
    out: &mut [R],
    threads: usize,
    body: F,
) {
    if threads <= 1 || out.len() < threads {
        body(0, out);
        return;
    }
    let chunk = out.len().div_ceil(threads);
    let mut chunks = out.chunks_mut(chunk);
    std::thread::scope(|scope| {
        // The leading chunk stays on the calling thread.  Every spawned chunk
        // starts at the number of elements the previous ones cover, which is
        // accumulated here rather than derived from the chunk length: the last
        // chunk of a size that does not divide evenly is shorter, so deriving
        // the offset from `chunk.len()` shifted every operand window of the last
        // chunk by the difference (a wrong result, not just a slow one).
        let Some(first) = chunks.next() else { return };
        let mut offset = first.len();
        for chunk in chunks {
            let start = offset;
            offset += chunk.len();
            let body = &body;
            scope.spawn(move || body(start, chunk));
        }
        body(0, first);
    });
}

/// The operand window of chunk `[offset, offset + len)` (empty past the end,
/// preserving `zip`'s stop-at-the-shortest behaviour).
#[inline]
fn window<T>(values: &[T], offset: usize, len: usize) -> &[T] {
    values.get(offset..offset + len).unwrap_or(&[])
}

/// Flat unary map: `out[i] = f(source[i])`.
pub fn zip1<T: Copy + Sync, R: Copy + Send>(
    out: &mut [R],
    source: &[T],
    f: impl Fn(T) -> R + Sync,
) {
    let threads = parallel_threads(out.len());
    if threads > 1 {
        split_parallel(out, threads, |offset, chunk| {
            let len = chunk.len();
            for (slot, value) in chunk.iter_mut().zip(window(source, offset, len)) {
                *slot = f(*value);
            }
        });
        return;
    }
    for (slot, value) in out.iter_mut().zip(source.iter()) {
        *slot = f(*value);
    }
}
/// Flat binary map: `out[i] = f(left[i], right[i])`.
pub fn zip2<T: Copy + Sync, R: Copy + Send>(
    out: &mut [R],
    left: &[T],
    right: &[T],
    f: impl Fn(T, T) -> R + Sync,
) {
    let threads = parallel_threads(out.len());
    if threads > 1 {
        split_parallel(out, threads, |offset, chunk| {
            let len = chunk.len();
            for ((slot, x), y) in chunk
                .iter_mut()
                .zip(window(left, offset, len))
                .zip(window(right, offset, len))
            {
                *slot = f(*x, *y);
            }
        });
        return;
    }
    for ((slot, x), y) in out.iter_mut().zip(left.iter()).zip(right.iter()) {
        *slot = f(*x, *y);
    }
}
/// Flat in-place binary accumulate: `slot[i] = f(slot[i], other[i])`, writing
/// the result into memory the caller already owns (an operand whose value dies
/// at the step — see [`crate::plan::InPlace`]).
///
/// Element `i` of the result depends on element `i` of both operands and on
/// nothing else, which is what makes overwriting the first operand in place
/// bit-identical to the allocating path; the chunking policy is [`zip2`]'s.
pub fn zip2_accumulate<T: Copy + Sync + Send>(
    slot: &mut [T],
    other: &[T],
    f: impl Fn(T, T) -> T + Sync,
) {
    let threads = parallel_threads(slot.len());
    if threads > 1 {
        split_parallel(slot, threads, |offset, chunk| {
            let len = chunk.len();
            // The other operand is only READ: every chunk owns a disjoint part
            // of `slot`, so the views never overlap a write.
            let other = window(other, offset, len);
            for (value, rhs) in chunk.iter_mut().zip(other) {
                *value = f(*value, *rhs);
            }
        });
        return;
    }
    for (value, rhs) in slot.iter_mut().zip(other.iter()) {
        *value = f(*value, *rhs);
    }
}
/// Flat ternary map: `out[i] = f(first[i], second[i], third[i])` (the `select`
/// shape: predicate, on-true, on-false).
pub fn zip3<P: Copy + Sync, T: Copy + Sync, R: Copy + Send>(
    out: &mut [R],
    first: &[P],
    second: &[T],
    third: &[T],
    f: impl Fn(P, T, T) -> R + Sync,
) {
    let threads = parallel_threads(out.len());
    if threads > 1 {
        split_parallel(out, threads, |offset, chunk| {
            let len = chunk.len();
            for (((slot, p), x), y) in chunk
                .iter_mut()
                .zip(window(first, offset, len))
                .zip(window(second, offset, len))
                .zip(window(third, offset, len))
            {
                *slot = f(*p, *x, *y);
            }
        });
        return;
    }
    for (((slot, p), x), y) in out
        .iter_mut()
        .zip(first.iter())
        .zip(second.iter())
        .zip(third.iter())
    {
        *slot = f(*p, *x, *y);
    }
}
// ---------------------------------------------------------------------------
// Large-result allocation
// ---------------------------------------------------------------------------
//
// Every elementwise op over a multi-megabyte array produces a fresh `Vec` for
// its result, and those results die at their last use a step or two later.
// Left to the system allocator that is a page-fault treadmill: once a chain's
// live set grows past one temporary, glibc grows the heap, coalesces the freed
// temporaries back into the heap top, trims it again, and the *same* pages are
// re-faulted (and re-zeroed by the kernel) on the next call.  Measured on the
// `ew_chain_1e6` benchmark graph: 3874 minor faults and 9.4 ms per `run`, versus
// 0 faults and 4.2 ms when the churn is removed -- while the kernels themselves
// are already at numpy-parity speed (0.21 ms per 1e6-element flat map, i.e.
// DRAM-bandwidth bound, so there is nothing left to win in the loops).
//
// The engine therefore keeps its own small free list of large result storages:
// dropping a large enough [`Storage`] hands its allocation to a thread-local
// pool instead of back to `malloc`, and [`take_pooled_vec`] hands it out again
// to the next flat map that needs exactly that dtype and element count.  Those
// bytes then never leave the process, so they are never re-faulted.  This is
// plain safe Rust -- no `unsafe`, no `mallopt`, no allocator hook -- and it
// keeps `Buffer` an ordinary owning value: no kernel signature changes.
//
// Contract:
//  * A pooled vector is only handed to a caller that overwrites EVERY element
//    (the flat maps gate on an exact length match), so no stale value can reach
//    a result -- pinned by `recycled_results_are_fully_overwritten` below.
//  * The pool is thread-local: a value is only recycled on the thread that
//    dropped it, and nothing is shared or synchronized.
//  * Retention is bounded and never grows across unrelated calls: at most
//    [`POOL_MAX_ENTRIES`] values totalling [`POOL_MAX_BYTES`], and only values
//    of at least [`POOL_MIN_BYTES`] (small results keep the system allocator,
//    whose bin free lists already make them cheap, and would only crowd out the
//    entries that matter).
//  * The caps are enforced by evicting the LEAST RECENTLY USED entry, never by
//    refusing the incoming one: a pool that refuses new stocks once it is full
//    is poisoned for the rest of the thread's life by whichever workload filled
//    it.  Measured on this host, `gp_gather_4096` (a `(4096, 64)` f32 graph, so
//    1 MiB temporaries) runs a warm `run` in 0.55-0.60 ms with its own stocks
//    pooled and 1.5-2.1 ms when the pool instead holds sixteen 4 MiB stocks
//    from a preceding `ew_chain_1e6` -- the same engine, the same graph, only
//    the order of two unrelated benchmarks in one process; running
//    `gp_gather_4096` on a fresh thread (empty pool, same process heap) is fast
//    again, which is what pinned the poison on the pool rather than on malloc.
//    LRU eviction keeps the stocks of whatever is RUNNING and discards the ones
//    that stopped being used, so the numbers are order-independent.
//    `a_full_pool_evicts_instead_of_poisoning_later_workloads` pins it.
const POOL_MIN_BYTES: usize = 1 << 12;
/// Total bytes the pool may retain per thread (the documented retention cap).
const POOL_MAX_BYTES: usize = 64 << 20;
/// Number of retained storages per thread, across every size class.
const POOL_MAX_ENTRIES: usize = 64;
/// Stocks retained per exact `(dtype, element count)` size class.
///
/// A graph reuses a couple of temporaries per size at most (one live result plus
/// the one it replaced), and a size-blind cap makes a graph with several
/// concurrent sizes thrash: measured on `gp_gather_4096` (whose 80 ops touch
/// four sizes: 1 MiB f32, 16 KiB f32, 4 KiB bool, 256 B f32), a flat
/// 16-entry cap produced 17 evictions and 8 allocation misses per run -- twice
/// the stocks the run actually needs -- and the churn is what made the row
/// order-dependent.  Keeping a few spares per size costs a bounded amount of
/// memory (<= `POOL_MAX_PER_SIZE` x the size class) and makes the steady state
/// allocation-free.
///
/// The cap must never be enforced by FREEING the incoming stock.  A graph that
/// keeps N + 1 buffers of one size live at once would then drop one per run and
/// re-allocate it on the next run, and whether `malloc` serves that megabyte
/// from its own free lists or with a fresh `mmap` + zero-fill depends on the
/// unrelated allocations earlier in the process -- exactly the order-dependent
/// behaviour this pool exists to remove.  A full size class therefore evicts its
/// OWN oldest stock and retains the newest, so a running workload always has
/// `POOL_MAX_PER_SIZE` of its own buffers and never a malloc round trip.
const POOL_MAX_PER_SIZE: usize = 8;

thread_local! {
    static STORAGE_POOL: RefCell<StoragePool> = const { RefCell::new(StoragePool::new()) };
    /// Cleared once the pool's own destructor starts, so a retained value that
    /// is dropped during thread teardown frees itself instead of re-entering the
    /// pool it is a part of (a `Cell<bool>` has no destructor, so it stays
    /// reachable for as long as the thread lives).
    static POOL_OPEN: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// One retained allocation plus the tick of its last use, which is what the
/// eviction policy orders by (smaller = older = evicted first).
struct PoolEntry {
    storage: Storage,
    last_use: u64,
}

/// The thread-local free list of large result storages.
struct StoragePool {
    entries: Vec<PoolEntry>,
    bytes: usize,
    /// Monotonic use counter stamped into [`PoolEntry::last_use`]; `u64` never
    /// wraps in a process's lifetime (one tick per retained or served value).
    tick: u64,
}

impl StoragePool {
    const fn new() -> StoragePool {
        StoragePool { entries: Vec::new(), bytes: 0, tick: 0 }
    }

    fn next_tick(&mut self) -> u64 {
        self.tick = self.tick.wrapping_add(1);
        self.tick
    }

    /// Retain `storage`'s allocation, returning `false` (and leaving `storage`
    /// intact, so the caller frees it as before) when it cannot usefully be
    /// retained: it is outside `POOL_MIN_BYTES .. POOL_MAX_BYTES`, or
    /// [`POOL_MAX_PER_SIZE`] stocks of its exact size are already retained.
    ///
    /// Room within the global caps is made by evicting the least recently used
    /// entry of ANOTHER size class, so a workload's own stocks are never the
    /// victim of its own churn, and a saturated pool of a different size can
    /// never starve the workload that is running (see the contract block above).
    fn push(&mut self, storage: &mut Storage) -> bool {
        let bytes = storage.bytes();
        if !(POOL_MIN_BYTES..=POOL_MAX_BYTES).contains(&bytes) {
            return false;
        }
        let dtype = storage.dtype();
        let len = storage.len();
        if self.count_of(dtype, len) >= POOL_MAX_PER_SIZE {
            // A full size class keeps its newest stocks: evict this class's own
            // oldest spare rather than freeing the incoming one, so a workload
            // that keeps one more buffer of this size in flight than it retains
            // never pays a malloc round trip per run.
            if !self.evict_same_size(dtype, len) {
                return false;
            }
        }
        while self.entries.len() >= POOL_MAX_ENTRIES || self.bytes + bytes > POOL_MAX_BYTES {
            if !self.evict_for(dtype, len) {
                // Nothing of another size can be evicted (only possible when the
                // value alone exceeds the byte cap, excluded above), so the pool
                // keeps what it has and this stock is freed.
                return false;
            }
        }
        let last_use = self.next_tick();
        let retained = std::mem::replace(storage, Storage::Bool(Vec::new()));
        self.bytes += bytes;
        self.entries.push(PoolEntry { storage: retained, last_use });
        true
    }

    /// How many stocks of exactly `dtype` and `len` elements are retained.
    fn count_of(&self, dtype: DType, len: usize) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.storage.dtype() == dtype && entry.storage.len() == len)
            .count()
    }

    /// Evict the least recently used entry of exactly `dtype` and `len`, so the
    /// incoming stock can take its place.  False when the class is empty.
    fn evict_same_size(&mut self, dtype: DType, len: usize) -> bool {
        let victim = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.storage.dtype() == dtype && entry.storage.len() == len)
            .min_by_key(|(_, entry)| entry.last_use)
            .map(|(index, _)| index);
        let Some(index) = victim else { return false };
        let entry = self.entries.swap_remove(index);
        self.bytes -= entry.storage.bytes();
        true
    }
    /// Evict the least recently used entry that is NOT of the incoming size
    /// class (falling back to the plain LRU entry when every entry is).
    fn evict_for(&mut self, dtype: DType, len: usize) -> bool {
        let victim = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| !(entry.storage.dtype() == dtype && entry.storage.len() == len))
            .min_by_key(|(_, entry)| entry.last_use)
            .or_else(|| {
                self.entries
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, entry)| entry.last_use)
            })
            .map(|(index, _)| index);
        let Some(index) = victim else { return false };
        let entry = self.entries.swap_remove(index);
        self.bytes -= entry.storage.bytes();
        true
    }

    /// Take a retained storage of exactly `dtype` and `len` elements.
    fn take(&mut self, dtype: DType, len: usize) -> Option<Storage> {
        let matching = |entry: &PoolEntry| entry.storage.dtype() == dtype && entry.storage.len() == len;
        // LIFO: hand back the MOST recently used matching stock, so a graph
        // cycles through a couple of hot buffers instead of walking its whole
        // retained size class and touching memory that has gone cold.
        let index = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| matching(entry))
            .max_by_key(|(_, entry)| entry.last_use)
            .map(|(index, _)| index)?;
        let entry = self.entries.swap_remove(index);
        self.bytes -= entry.storage.bytes();
        Some(entry.storage)
    }

}

impl Drop for StoragePool {
    fn drop(&mut self) {
        let _ = POOL_OPEN.try_with(|open| open.set(false));
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        if self.bytes() < POOL_MIN_BYTES {
            return;
        }
        if !POOL_OPEN.try_with(|open| open.get()).unwrap_or(false) {
            return;
        }
        // On refusal (or during thread teardown) the allocation simply stays
        // here and is freed by the normal field drop that follows this call.
        let _ = STORAGE_POOL.try_with(|pool| match pool.try_borrow_mut() {
            Ok(mut pool) => pool.push(self),
            Err(_) => false,
        });
    }
}

/// A retained result vector of exactly `count` elements of `R`, or `None`.
///
/// The caller must overwrite every element before the vector becomes visible.
fn take_pooled_vec<R: Scalar>(count: usize) -> Option<Vec<R>> {
    let storage = STORAGE_POOL
        .try_with(|pool| pool.try_borrow_mut().ok()?.take(R::DTYPE, count))
        .ok()
        .flatten();
    match storage {
        Some(mut storage) => R::take_vec(&mut storage),
        None => None,
    }
}

/// Flat unary builder: `out[i] = f(source[i])` over `out_shape`'s elements,
/// with the result storage built straight from the mapped iterator.
///
/// Unlike [`zip1`], this never zero-fills a buffer first: a `Vec` built from a
/// slice with an exact size hint is allocated once and written once (and a
/// recycled vector from the storage pool is already allocated).  Elementwise
/// kernels that produce a fresh result use these; the in-place `zip*` helpers
/// stay for kernels that already own the memory they write into.  The
/// `Buffer::new` validation is defensive (a caller gates on
/// [`Buffer::is_flat_for`], so the element count and the dtype already agree)
/// and keeps a short or long operand an explicit error rather than a silently
/// truncated result.
pub fn map1<T: Copy + Sync, R: Scalar>(
    dtype: DType,
    out_shape: &[usize],
    source: &[T],
    f: impl Fn(T) -> R + Sync,
) -> Result<Buffer> {
    let count = util_numel(out_shape);
    if source.len() == count {
        if let Some(mut values) = take_pooled_vec::<R>(count) {
            zip1(&mut values, source, f);
            return Buffer::new(dtype, out_shape.to_vec(), R::into_storage(values));
        }
    }
    let values: Vec<R> = source.iter().map(|value| f(*value)).collect();
    Buffer::new(dtype, out_shape.to_vec(), R::into_storage(values))
}

/// Flat binary builder: `out[i] = f(left[i], right[i])` (see [`map1`]).
pub fn map2<T: Copy + Sync, R: Scalar>(
    dtype: DType,
    out_shape: &[usize],
    left: &[T],
    right: &[T],
    f: impl Fn(T, T) -> R + Sync,
) -> Result<Buffer> {
    let count = util_numel(out_shape);
    if left.len() == count && right.len() == count {
        if let Some(mut values) = take_pooled_vec::<R>(count) {
            zip2(&mut values, left, right, f);
            return Buffer::new(dtype, out_shape.to_vec(), R::into_storage(values));
        }
    }
    let values: Vec<R> = left.iter().zip(right.iter()).map(|(x, y)| f(*x, *y)).collect();
    Buffer::new(dtype, out_shape.to_vec(), R::into_storage(values))
}

/// Flat ternary builder: `out[i] = f(first[i], second[i], third[i])` -- the
/// `select` shape (see [`map1`]).
pub fn map3<P: Copy + Sync, T: Copy + Sync, R: Scalar>(
    dtype: DType,
    out_shape: &[usize],
    first: &[P],
    second: &[T],
    third: &[T],
    f: impl Fn(P, T, T) -> R + Sync,
) -> Result<Buffer> {
    let count = util_numel(out_shape);
    if first.len() == count && second.len() == count && third.len() == count {
        if let Some(mut values) = take_pooled_vec::<R>(count) {
            zip3(&mut values, first, second, third, f);
            return Buffer::new(dtype, out_shape.to_vec(), R::into_storage(values));
        }
    }
    let values: Vec<R> = first
        .iter()
        .zip(second.iter())
        .zip(third.iter())
        .map(|((p, x), y)| f(*p, *x, *y))
        .collect();
    Buffer::new(dtype, out_shape.to_vec(), R::into_storage(values))
}

/// Element count of a concrete shape (`buffer.rs` cannot reach
/// `kernels::util`, which is a higher layer than the value type).
#[inline]
fn util_numel(shape: &[usize]) -> usize {
    shape.iter().product()
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

/// A [`Storage`] built by [`copy_blocks`] for a caller that knows the offset
/// list covers EVERY output element (a repeated `tile`/`gather`/`slice` block
/// decomposition), so the result needs neither a zero fill nor a
/// `Vec::with_capacity` + `extend_from_slice` growth: it takes a pooled
/// allocation of exactly the result's length and `copy_blocks` overwrites all
/// of it.
///
/// That matters for `tile`-shaped results, which are megabyte-sized and die at
/// their next consumer: without the pool each one is a fresh page-faulted and
/// zeroed allocation, i.e. a full extra pass over the output the copy is about
/// to overwrite.  The contract is [`copy_blocks`]': `offsets.len() * inner` must
/// equal the result's element count (an explicit error, never a truncated
/// result), and `values` must hold every offset's run.
pub fn block_storage_filled<T: Scalar>(
    values: &[T],
    out_shape: &[usize],
    offsets: &[usize],
    inner: usize,
    op: &str,
) -> Result<Storage> {
    let total: usize = out_shape.iter().product();
    if offsets.len().saturating_mul(inner) != total {
        return Err(block_error(op));
    }
    let mut out = take_pooled_vec::<T>(total).unwrap_or_else(|| vec![T::zero(); total]);
    copy_blocks(&mut out, values, offsets, inner, op)?;
    Ok(T::into_storage(out))
}

/// A [`Storage`] whose every element is a copy of a `run`-element slice of
/// `values` repeated to fill `total` elements (whole-source `tile`: the result
/// is the operand repeated `total / run` times).
///
/// One `copy_from_slice` per `run`-element chunk instead of one per index row,
/// so a `(64, 64)` operand tiled to `(4096, 64)` is 64 copies of 16 KiB rather
/// than 4096 copies of 256 B, and (like [`block_storage_filled`]) the
/// allocation is pooled instead of fresh and zeroed.  `run` must be non-zero
/// and no larger than `values`; a short `values` is the [`block_error`] the
/// block-copy paths raise, never a truncated result.
pub fn repeat_storage<T: Scalar>(
    values: &[T],
    total: usize,
    run: usize,
    op: &str,
) -> Result<Storage> {
    if run == 0 || run > values.len() {
        return Err(block_error(op));
    }
    let source = &values[..run];
    let mut out = take_pooled_vec::<T>(total).unwrap_or_else(|| vec![T::zero(); total]);
    let threads = parallel_threads(total);
    if threads > 1 {
        // Each chunk is an independently filled slice of the repetition, so the
        // split needs no shared mutable state: the operands are read-only and
        // the chunks are disjoint.  A chunk that starts mid-run takes the tail
        // of that run first.
        split_parallel(&mut out, threads, |offset, chunk| {
            let mut written = 0usize;
            while written < chunk.len() {
                let at = (offset + written) % run;
                let len = (run - at).min(chunk.len() - written);
                chunk[written..written + len].copy_from_slice(&source[at..at + len]);
                written += len;
            }
        });
        return Ok(T::into_storage(out));
    }
    let mut offset = 0usize;
    while offset < total {
        let end = (offset + run).min(total);
        out[offset..end].copy_from_slice(&source[..end - offset]);
        offset = end;
    }
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

/// Match two buffers of the same dtype, binding the LEFT one's typed vector
/// mutably and the right one's immutably — the in-place counterpart of
/// [`with_same_type!`], for a kernel that owns its first operand (see
/// [`zip2_accumulate`]).
#[macro_export]
macro_rules! with_same_type_mut {
    ($left:expr, $right:expr, |$a:ident, $b:ident| $body:expr) => {
        match (&mut $left.storage, &$right.storage) {
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
    fn the_thread_split_scales_with_the_work_per_thread() {
        // The measured policy (`bench_thread_crossover`): below 2^18 elements one
        // thread, two at 2^18 (a `(4096, 64)` f32 tensor), the full four from
        // 2^19.  This is a performance rule, not a correctness one, but a silent
        // change to it would undo the measurement.
        assert_eq!(parallel_threads(0), 1);
        assert_eq!(parallel_threads(PARALLEL_MIN_ELEMS - 1), 1);
        let cores = parallel_threads(1 << 24);
        assert!(cores <= PARALLEL_MAX_THREADS);
        assert_eq!(parallel_threads(PARALLEL_MIN_ELEMS), cores.min(2));
        assert_eq!(parallel_threads(1_000_000), cores.min(8));
        assert!(parallel_threads(1 << 24) <= PARALLEL_MAX_THREADS);
        assert_eq!(parallel_threads(1_000_000), parallel_threads(1 << 20));
    }

    #[test]
    fn the_thread_split_hands_every_chunk_its_running_offset() {
        // A count the split does not divide evenly makes the LAST chunk shorter,
        // so a chunk's start is the running total of the chunks before it and not
        // a multiple of the chunk length.  Deriving it from the chunk length made
        // every operand window of the trailing chunk read shifted elements — a
        // wrong result, not a slow one — which is why this is pinned: the chunks
        // must partition `out` in order, and each element must be written exactly
        // once by the chunk that owns it.
        for &len in &[(1 << 19) + 5, (1 << 18) + 3, 4099, 1000, 7, 3] {
            let mut out: Vec<u64> = vec![0; len];
            let chunks = std::sync::Mutex::new(Vec::new());
            split_parallel(&mut out, PARALLEL_MAX_THREADS, |offset, chunk| {
                chunks.lock().expect("chunk list").push((offset, chunk.len()));
                for (index, slot) in chunk.iter_mut().enumerate() {
                    *slot = (offset + index) as u64 + 1;
                }
            });
            let mut chunks = chunks.into_inner().expect("chunk list");
            chunks.sort_unstable();
            // The partition `chunks_mut` would walk: a full `chunk` per thread
            // except for the short tail.
            let chunk = if len >= PARALLEL_MAX_THREADS {
                len.div_ceil(PARALLEL_MAX_THREADS)
            } else {
                len
            };
            let mut expected = Vec::new();
            let mut start = 0;
            while start < len {
                let end = (start + chunk).min(len);
                expected.push((start, end - start));
                start = end;
            }
            assert_eq!(chunks, expected, "len {len}: the chunks partition `out` in order");
            let want: Vec<u64> = (1..=len as u64).collect();
            assert_eq!(out, want, "len {len}: every element exactly once, own offset");
        }
        // Four chunks with a short last one: the shape the old arithmetic broke
        // (an offset derived from the chunk length lands 3 elements late).
        let len: usize = (1 << 19) + 5;
        let chunk = len.div_ceil(PARALLEL_MAX_THREADS);
        assert_eq!(
            len - (PARALLEL_MAX_THREADS - 1) * chunk,
            chunk - 3,
            "the last chunk is short, so the offsets are not multiples of `chunk`"
        );
    }

    #[test]
    fn block_storage_filled_matches_the_zero_filled_helper() {
        let values: Vec<f32> = (0..12).map(|index| index as f32 * 0.5).collect();
        let offsets = [0usize, 4, 8, 1];
        let filled = block_storage_filled(&values, &[4, 2], &offsets, 2, "tile").unwrap();
        let zero_filled = block_storage(&values, &[4, 2], &offsets, 2, "tile").unwrap();
        assert_eq!(filled, zero_filled);
        assert_eq!(filled, Storage::F32(vec![0.0, 0.5, 2.0, 2.5, 4.0, 4.5, 0.5, 1.0]));
        // The block-copy contract is unchanged: a short or long offset list is
        // the same explicit error, never a truncated result.
        let short = block_storage_filled(&values, &[3, 2], &offsets, 2, "tile").unwrap_err();
        assert_eq!(
            short.message,
            block_storage(&values, &[3, 2], &offsets, 2, "tile").unwrap_err().message
        );
        assert!(short.message.contains("a block copy reads outside the operand"));
        // A block that runs past the operand is an error too (not a partial copy).
        assert!(block_storage_filled(&values, &[2, 2], &[0, 11], 2, "tile").is_err());
    }

    #[test]
    fn repeat_storage_copies_the_source_over_the_whole_result() {
        let source = [1.0f32, -0.0, f32::NAN, 3.5];
        // An exact multiple of the run...
        let exact = repeat_storage(&source, 12, 4, "tile").unwrap();
        match &exact {
            Storage::F32(values) => {
                assert_eq!(values.len(), 12);
                for (index, value) in values.iter().enumerate() {
                    assert_eq!(value.to_bits(), source[index % 4].to_bits(), "element {index}");
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
        // ... and a result that ends mid-run (the last run is a partial copy).
        let ragged = repeat_storage(&source, 6, 3, "tile").unwrap();
        match &ragged {
            Storage::F32(values) => {
                let expected = [1.0f32, -0.0, f32::NAN, 1.0, -0.0, f32::NAN];
                assert_eq!(values.len(), expected.len());
                for (index, value) in values.iter().enumerate() {
                    assert_eq!(value.to_bits(), expected[index].to_bits(), "element {index}");
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
        // A run longer than the source, or an empty one, is the block-copy error.
        assert!(repeat_storage(&source, 8, 5, "tile").is_err());
        assert!(repeat_storage(&source, 8, 0, "tile").is_err());
        // An empty result is empty, never an error.
        assert_eq!(
            repeat_storage(&source, 0, 4, "tile").unwrap(),
            Storage::F32(Vec::new())
        );
    }

    #[test]
    fn a_pooled_repeat_result_is_fully_overwritten() {
        // More elements than `PARALLEL_MIN_ELEMS` so the copy takes the threaded
        // path, and more than `POOL_MIN_BYTES` so the stock is poolable: the
        // recycled pages must all be overwritten, exactly like `map`'s results.
        let count = PARALLEL_MIN_ELEMS + 512;
        garbage_f32(count);
        let source: Vec<f32> = (0..64).map(|index| index as f32).collect();
        let storage = repeat_storage(&source, count, 64, "tile").unwrap();
        match &storage {
            Storage::F32(values) => {
                assert_eq!(values.len(), count);
                for (index, value) in values.iter().enumerate() {
                    assert_eq!(*value, (index % 64) as f32, "element {index}");
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
        // A run that does not divide the per-thread chunk (100 into a quarter of
        // the result) still lands every element: the boundary case of the
        // threaded copy, which takes the tail of a run first.
        let source: Vec<f32> = (0..100).map(|index| index as f32 * 0.25).collect();
        let storage = repeat_storage(&source, count, 100, "tile").unwrap();
        match &storage {
            Storage::F32(values) => {
                assert_eq!(values.len(), count);
                for (index, value) in values.iter().enumerate() {
                    assert_eq!(*value, source[index % 100], "element {index}");
                }
            }
            other => panic!("unexpected storage {other:?}"),
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

    /// A stock for the pool: `count` f32 elements' worth, filled with garbage
    /// that no correct result can contain.
    fn garbage_f32(count: usize) -> Storage {
        let mut storage = Storage::F32(vec![f32::NAN; count]);
        STORAGE_POOL.with(|pool| {
            pool.borrow_mut().push(&mut storage);
        });
        storage
    }

    /// Empty this thread's pool so a policy test starts from a known state
    /// (tests on the same thread share the thread-local).
    fn reset_pool() {
        STORAGE_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            pool.entries.clear();
            pool.bytes = 0;
        });
    }

    /// `count` f32 elements of `value` as a raw stock the pool can retain.
    fn pool_stock_f32(count: usize, value: f32) -> Storage {
        Storage::F32(vec![value; count])
    }

    #[test]
    fn stocks_are_retained_per_size_up_to_the_per_size_cap() {
        reset_pool();
        let count = (1 << 20) / std::mem::size_of::<f32>();
        let mut stocks: Vec<Storage> = (0..POOL_MAX_PER_SIZE * 3)
            .map(|_| pool_stock_f32(count, 1.0))
            .collect();
        let mut retained = 0;
        for stock in stocks.iter_mut() {
            if STORAGE_POOL.with(|pool| pool.borrow_mut().push(stock)) {
                retained += 1;
            }
        }
        assert_eq!(retained, POOL_MAX_PER_SIZE * 3, "a full class evicts its own oldest stock");
        let kept = STORAGE_POOL.with(|pool| pool.borrow().count_of(DType::Float32, count));
        assert_eq!(kept, POOL_MAX_PER_SIZE, "one size class keeps at most its cap");
        reset_pool();
    }

    #[test]
    fn a_full_size_class_keeps_its_newest_stocks() {
        // The class is at its cap; pushing a fourth stock evicts the OLDEST of
        // that class (never frees the incoming one, never touches another
        // size), so a workload that keeps one buffer more in flight than it
        // retains never pays a malloc round trip per run.
        reset_pool();
        let count = (1 << 16) / std::mem::size_of::<f32>();
        let mut pool = StoragePool::new();
        let mut stock = pool_stock_f32(count / 2, 9.0);
        assert!(pool.push(&mut stock), "another size class is retained");
        for value in 0..POOL_MAX_PER_SIZE {
            let mut stock = pool_stock_f32(count, value as f32);
            assert!(pool.push(&mut stock), "stock {value} fits its own class");
        }
        let mut extra = pool_stock_f32(count, 42.0);
        assert!(pool.push(&mut extra), "a full class makes room instead of freeing");
        assert_eq!(pool.count_of(DType::Float32, count), POOL_MAX_PER_SIZE);
        assert_eq!(pool.count_of(DType::Float32, count / 2), 1, "another size class is untouched");
        // The newest stock is the one the next take serves (LIFO), so the
        // evicted entry is the oldest: the values prove which survived.
        let taken = pool.take(DType::Float32, count).unwrap();
        match &taken {
            Storage::F32(values) => assert_eq!(values[0], 42.0, "the newest stock is served first"),
            _ => panic!("expected f32 storage"),
        }
    }

    #[test]
    fn the_pool_evicts_the_least_recently_used_other_size_class() {
        // Fill the pool to its entry cap with DISTINCT sizes (so no per-size cap
        // triggers) and then push one more size: the victim must be the least
        // recently used entry of another size class, and never a stock of the
        // incoming size.
        let base = POOL_MIN_BYTES / std::mem::size_of::<f32>();
        let mut pool = StoragePool::new();
        for index in 0..POOL_MAX_ENTRIES {
            let mut stock = pool_stock_f32(base + index, index as f32);
            assert!(pool.push(&mut stock), "distinct size {index} fits");
        }
        assert_eq!(pool.entries.len(), POOL_MAX_ENTRIES);
        // Two stocks of one new size are retained -- the second only after
        // evicting a different size class (there is no room otherwise).
        let mut first = pool_stock_f32(base + POOL_MAX_ENTRIES + 1, -1.0);
        assert!(pool.push(&mut first));
        let mut second = pool_stock_f32(base + POOL_MAX_ENTRIES + 1, -2.0);
        assert!(pool.push(&mut second));
        assert_eq!(pool.entries.len(), POOL_MAX_ENTRIES);
        assert_eq!(pool.count_of(DType::Float32, base + POOL_MAX_ENTRIES + 1), 2);
        // The oldest sizes went first, and no entry ever exceeds the caps.
        assert_eq!(pool.count_of(DType::Float32, base), 0, "the oldest stock was evicted");
        assert_eq!(pool.count_of(DType::Float32, base + 1), 0);
        assert!(pool.bytes <= POOL_MAX_BYTES);
        // ... while a size that is still represented survives.
        assert_eq!(pool.count_of(DType::Float32, base + POOL_MAX_ENTRIES - 1), 1);
    }

    #[test]
    fn a_stock_larger_than_the_byte_cap_is_refused_and_evicts_nothing() {
        let count = (1 << 20) / std::mem::size_of::<f32>();
        let mut pool = StoragePool::new();
        let mut stock = pool_stock_f32(count, 1.0);
        assert!(pool.push(&mut stock));
        let mut huge = pool_stock_f32(POOL_MAX_BYTES / std::mem::size_of::<f32>() + 1, 2.0);
        assert!(!pool.push(&mut huge), "a value above the byte cap is never retained");
        assert_eq!(pool.entries.len(), 1);
        assert!(pool.take(DType::Float32, count).is_some(), "the small stock was not evicted");
    }

    #[test]
    fn a_differently_sized_predecessor_does_not_starve_later_workloads() {
        // The order-dependent regression this pins: `ew_chain_1e6`-shaped work
        // (4 MiB temporaries) fills the pool, and a following `gp_gather_4096`
        // (1 MiB temporaries) then never got a single stock back, which cost it
        // 2.5x on a warm run.  Filling the pool with the wrong size class must
        // not stop the next map from recycling ITS result.
        reset_pool();
        let big = (4 << 20) / std::mem::size_of::<f32>();
        let mut stocks: Vec<Storage> = (0..POOL_MAX_PER_SIZE)
            .map(|_| pool_stock_f32(big, f32::NAN))
            .collect();
        STORAGE_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            for stock in stocks.iter_mut() {
                assert!(pool.push(stock), "the pool has room for these stocks");
            }
        });
        let count = (1 << 20) / std::mem::size_of::<f32>();
        assert!(take_pooled_vec::<f32>(count).is_none(), "nothing of this size is pooled yet");
        let source = vec![0.5f32; count];
        let result = map1(DType::Float32, &[count], &source, |value| value + 1.0)
            .expect("the map succeeds without a pooled stock");
        drop(result);
        assert!(
            take_pooled_vec::<f32>(count).is_some(),
            "a 4 MiB-filled pool must still recycle a 1 MiB result"
        );
        STORAGE_POOL.with(|pool| {
            let pool = pool.borrow();
            assert_eq!(pool.count_of(DType::Float32, big), POOL_MAX_PER_SIZE);
        });
        reset_pool();
    }

    #[test]
    fn stock_reuse_is_last_in_first_out() {
        // LIFO reuse keeps a graph cycling through the same few hot buffers
        // instead of walking its whole retained size class and touching memory
        // that has gone cold.
        reset_pool();
        let count = (1 << 20) / std::mem::size_of::<f32>();
        let mut first = pool_stock_f32(count, 1.0);
        let mut second = pool_stock_f32(count, 2.0);
        STORAGE_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            assert!(pool.push(&mut first));
            assert!(pool.push(&mut second));
        });
        let taken = take_pooled_vec::<f32>(count).expect("a stock of this size is pooled");
        assert_eq!(taken[0], 2.0, "the most recently freed stock is handed back first");
        reset_pool();
    }

    #[test]
    fn a_recycled_result_is_fully_overwritten() {
        // More elements than `POOL_MIN_BYTES` so the pool accepts the stock, and
        // more than `PARALLEL_MIN_ELEMS` so the map takes the wide path.
        let count = PARALLEL_MIN_ELEMS + 64;
        assert!(count * std::mem::size_of::<f32>() > POOL_MIN_BYTES);
        garbage_f32(count);
        let left: Vec<f32> = (0..count).map(|index| index as f32).collect();
        let right = vec![2.0f32; count];
        let binary = map2(DType::Float32, &[count], &left, &right, |x, y| x * y).unwrap();
        match &binary.storage {
            Storage::F32(values) => {
                assert_eq!(values.len(), count);
                for (index, value) in values.iter().enumerate() {
                    assert_eq!(*value, left[index] * 2.0, "element {index}");
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
        // Same for the unary path: every element overwritten, none left stale.
        let unary = map1(DType::Float32, &[count], &left, |value| value + 1.0).unwrap();
        match &unary.storage {
            Storage::F32(values) => {
                for (index, value) in values.iter().enumerate() {
                    assert_eq!(*value, left[index] + 1.0, "element {index}");
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
        // ... and the ternary one.
        let mask = vec![true; count];
        let ternary =
            map3(DType::Float32, &[count], &mask, &left, &right, |pick, a, b| if pick { a } else { b })
                .unwrap();
        assert_eq!(ternary.storage, map1(DType::Float32, &[count], &left, |value| value).unwrap().storage);
    }

    #[test]
    fn threaded_chunks_read_their_operands_at_their_own_offset() {
        // A count no thread split divides evenly leaves a SHORT last chunk.  Its
        // operand window must start where the previous chunks end, which is the
        // running total of their lengths — deriving it from a chunk length
        // shifted that window by the rounding difference and silently paired
        // each element with a neighbour's operand.
        for extra in [1usize, 3, 5, 1234] {
            let count = PARALLEL_MIN_ELEMS + extra;
            let left: Vec<f32> = (0..count).map(|index| index as f32 * 0.5 - 3.0).collect();
            let right: Vec<f32> = (0..count).map(|index| 1.25 - index as f32 * 0.125).collect();
            let mut slot = left.clone();
            zip2_accumulate(&mut slot, &right, |value, other| value * other);
            for (index, value) in slot.iter().enumerate() {
                assert_eq!(
                    *value,
                    left[index] * right[index],
                    "count {count} element {index}"
                );
            }
            let expected = map2(DType::Float32, &[count], &left, &right, |x, y| x * y)
                .expect("the allocating map accepts full-length operands");
            match &expected.storage {
                Storage::F32(values) => assert_eq!(&slot, values, "count {count}"),
                other => panic!("unexpected storage {other:?}"),
            }
        }
    }

    #[test]
    fn in_place_accumulation_matches_the_allocating_map() {
        // The in-place driver must be bit-identical to a fresh `map2` result,
        // element for element, on both the one-thread path and the chunked one
        // (with the other operand read from a windowed view per chunk).
        for count in [0usize, 1, 7, PARALLEL_MIN_ELEMS + 3] {
            let left: Vec<f32> = (0..count).map(|index| index as f32 * 0.5 - 3.0).collect();
            let right: Vec<f32> = (0..count).map(|index| 1.25 - index as f32 * 0.125).collect();
            let mut slot = left.clone();
            zip2_accumulate(&mut slot, &right, |value, other| value * other);
            let expected = map2(DType::Float32, &[count], &left, &right, |x, y| x * y)
                .expect("the allocating map accepts full-length operands");
            match &expected.storage {
                Storage::F32(values) => assert_eq!(&slot, values, "count {count}"),
                other => panic!("unexpected storage {other:?}"),
            }
        }
        // A short `other` stops the zip exactly where the allocating driver
        // stops, so a mismatched pair can never read past the operand.
        let mut slot = vec![1.0f32, 2.0, 3.0];
        zip2_accumulate(&mut slot, &[10.0f32, 20.0], |value, other| value + other);
        assert_eq!(slot, vec![11.0, 22.0, 3.0]);
        // NaN/inf semantics are the closure's, so a NaN operand poisons only
        // the element it lands on (no reordering, no accumulation).
        let mut slot = vec![1.0f32, f32::NAN, 3.0];
        zip2_accumulate(&mut slot, &[f32::INFINITY, 2.0, 4.0], |value, other| value + other);
        assert_eq!(slot[0], f32::INFINITY);
        assert!(slot[1].is_nan());
        assert_eq!(slot[2], 7.0);
    }

    #[test]
    fn a_large_result_is_recycled_and_stays_correct() {
        // The result of the first map is dropped, so the second map reuses it.
        let count = POOL_MIN_BYTES / std::mem::size_of::<f32>() + 8;
        let source: Vec<f32> = (0..count).map(|index| index as f32).collect();
        let first = map1(DType::Float32, &[count], &source, |value| value * 3.0).unwrap();
        let expected = map1(DType::Float32, &[count], &source, |value| value * 3.0).unwrap();
        assert_eq!(first.storage, expected.storage);
        // A recycled buffer must not leak an integer result into a float one or
        // vice versa: the pool is keyed by dtype as well as length.
        let ints = map1(DType::Int32, &[count], &source, |value| value as i32).unwrap();
        match &ints.storage {
            Storage::I32(values) => {
                for (index, value) in values.iter().enumerate() {
                    assert_eq!(*value, index as i32);
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }

    #[test]
    fn short_operands_still_error_with_a_pooled_result_available() {
        // A pooled vector of `count` elements must not turn a short operand into
        // a silently truncated result: the length gate falls back to `collect`,
        // where `Buffer::new` reports the mismatch.
        let count = POOL_MIN_BYTES / std::mem::size_of::<f32>() + 4;
        garbage_f32(count);
        let short: Vec<f32> = vec![1.0; count - 1];
        assert!(map1(DType::Float32, &[count], &short, |value| value).is_err());
        assert!(map2(DType::Float32, &[count], &short, &short, |a, b| a + b).is_err());
    }

    #[test]
    fn the_pool_keeps_only_large_values_and_respects_its_cap() {
        let small = Storage::F32(vec![0.0; POOL_MIN_BYTES / std::mem::size_of::<f32>() - 1]);
        let mut small = small;
        STORAGE_POOL.with(|pool| {
            pool.borrow_mut().push(&mut small);
        });
        let (entries, bytes) = STORAGE_POOL.with(|pool| {
            let pool = pool.borrow();
            (pool.entries.len(), pool.bytes)
        });
        assert_eq!((entries, bytes), (0, 0), "a small value must not be retained");
        // Then overfill: the caps hold.
        let count = POOL_MIN_BYTES / std::mem::size_of::<f32>() + 2;
        let mut values: Vec<Storage> = (0..POOL_MAX_ENTRIES * 2)
            .map(|_| Storage::F32(vec![1.0; count]))
            .collect();
        STORAGE_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            for value in values.iter_mut() {
                pool.push(value);
            }
            assert!(pool.entries.len() <= POOL_MAX_ENTRIES);
            assert!(pool.bytes <= POOL_MAX_BYTES);
        });
    }

    #[test]
    fn wide_and_narrow_maps_agree() {
        // The wide (threaded) path and the narrow one must produce identical
        // bits, NaN and signed zeros included.
        let count = PARALLEL_MIN_ELEMS + 1000;
        let left: Vec<f32> = (0..count)
            .map(|index| match index % 5 {
                0 => f32::NAN,
                1 => -0.0,
                2 => f32::INFINITY,
                _ => index as f32,
            })
            .collect();
        let right: Vec<f32> = (0..count).map(|index| (index % 7) as f32 - 3.0).collect();
        let wide = map2(DType::Float32, &[count], &left, &right, |x, y| {
            if x.is_nan() {
                x
            } else if y.is_nan() {
                y
            } else if x >= y {
                x
            } else {
                y
            }
        })
        .unwrap();
        let mut narrow = vec![0.0f32; count];
        for index in 0..count {
            let (x, y) = (left[index], right[index]);
            narrow[index] = if x.is_nan() {
                x
            } else if y.is_nan() {
                y
            } else if x >= y {
                x
            } else {
                y
            };
        }
        match &wide.storage {
            Storage::F32(values) => {
                assert_eq!(values.len(), count);
                for (index, value) in values.iter().enumerate() {
                    assert_eq!(
                        value.to_bits(),
                        narrow[index].to_bits(),
                        "element {index} differs between the wide and narrow paths"
                    );
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }

    /// Measurement aid, not a correctness test: `cargo test --release -p
    /// evoxc-cpu bench_thread_crossover -- --ignored --nocapture` prints the
    /// per-element cost of one flat `add` at several sizes and thread counts,
    /// which is what the `PARALLEL_MIN_ELEMS` threshold is chosen from.
    #[test]
    #[ignore = "measurement aid: prints timings"]
    fn bench_thread_crossover() {
        use std::time::Instant;
        let left: Vec<f32> = (0..(1 << 20)).map(|index| index as f32 * 0.5).collect();
        let right: Vec<f32> = (0..(1 << 20)).map(|index| index as f32 * 0.25).collect();
        for count in [1usize << 16, 1 << 17, 262144, 1 << 19, 1_000_000] {
            let mut line = format!("count {count:>8} ({:>4} KiB):", count * 4 / 1024);
            for threads in [1usize, 2, 3, 4] {
                let mut out = vec![0.0f32; count];
                let mut best = f64::INFINITY;
                for _ in 0..40 {
                    let started = Instant::now();
                    split_parallel(&mut out, threads, |offset, chunk| {
                        let len = chunk.len();
                        for ((slot, x), y) in chunk
                            .iter_mut()
                            .zip(window(&left, offset, len))
                            .zip(window(&right, offset, len))
                        {
                            *slot = *x + *y;
                        }
                    });
                    best = best.min(started.elapsed().as_secs_f64() * 1e3);
                }
                line.push_str(&format!("  {threads}t {best:7.3} ms"));
            }
            let single = {
                let mut out = vec![0.0f32; count];
                let mut best = f64::INFINITY;
                for _ in 0..40 {
                    let started = Instant::now();
                    for ((slot, x), y) in out.iter_mut().zip(&left).zip(&right) {
                        *slot = *x + *y;
                    }
                    best = best.min(started.elapsed().as_secs_f64() * 1e3);
                }
                best
            };
            line.push_str(&format!("   plain {single:7.3} ms"));
            println!("{line}");
        }
    }

    #[test]
    fn decode_recycles_and_still_reads_every_element() {
        let count = POOL_MIN_BYTES / std::mem::size_of::<f32>() + 32;
        garbage_f32(count);
        let words: Vec<f32> = (0..count).map(|index| index as f32 * 0.5).collect();
        let bytes: Vec<u8> = words.iter().flat_map(|value| value.to_le_bytes()).collect();
        let storage = decode_storage(DType::Float32, &bytes).unwrap();
        assert_eq!(storage, Storage::F32(words));
    }
}
