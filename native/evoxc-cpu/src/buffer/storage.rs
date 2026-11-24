//! The typed storage enum ([`Storage`]), constant-payload decoding and dtype
//! casts.

use evoxc_core::{DType, Result};
use half::f16;
use num_complex::Complex;

use crate::scalar::Scalar;

use super::parallel::{parallel_threads, split_parallel};
use super::pool::take_pooled_vec;

/// One typed, C-contiguous value buffer.
#[derive(Debug, PartialEq)]
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

    /// Copy a typed host slice into storage, writing into a RECYCLED
    /// allocation when this thread's pool holds a stock of exactly this dtype
    /// and element count and into fresh storage otherwise.
    ///
    /// That is the same per-size-class pool results, results' clones, reduction
    /// results and `dot` destinations draw from (see `pool.rs`), so a
    /// multi-megabyte host array handed to the engine is not re-faulted on every
    /// call; this constructor is the hand-off the PyO3 binding uses for a `run`'s
    /// inputs.
    ///
    /// The copy is one `copy_from_slice` over the whole storage (split across
    /// scoped threads above the flat map's crossover, exactly like
    /// [`Storage::clone`]), so a pooled destination is FULLY overwritten — the
    /// pool's contract — and the result never shares memory with `values`.
    pub fn from_slice_pooled<T: Scalar>(values: &[T]) -> Storage {
        T::into_storage(copy_slice_pooled(values))
    }

    /// Bytes of element storage this value holds (not its capacity).
    pub(super) fn bytes(&self) -> usize {
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

/// Copy `source` into a pooled stock of exactly that size, falling back to an
/// ordinary `Vec` copy when the pool has no stock (a small value, or a cold
/// thread).  The shared body of [`Storage::from_slice_pooled`] and
/// [`Storage::clone`]: one `copy_from_slice` (threaded above the flat map's
/// crossover), so every element of the destination is written.
pub(crate) fn copy_slice_pooled<T: Scalar>(source: &[T]) -> Vec<T> {
    let Some(mut stock) = take_pooled_vec::<T>(source.len()) else {
        return source.to_vec();
    };
    let threads = parallel_threads(source.len());
    if threads > 1 {
        split_parallel(&mut stock, threads, |offset, chunk| {
            let len = chunk.len();
            chunk.copy_from_slice(&source[offset..offset + len]);
        });
    } else {
        stock.copy_from_slice(source);
    }
    stock
}

/// Cloning a value is the same allocation problem as producing one: a
/// `Storage`'s clone is a full-size `Vec` plus a memcpy, and for anything past
/// the allocator's mmap threshold the destination is a FRESH mapping whose pages
/// the copy then faults in.  Measured on this host (medians of 11 runs after 3
/// warmups), `Buffer::clone` of a 64 MiB f32 value costs **26.5 ms** that way,
/// against **~7 ms** when the destination comes from the pool and its pages stay
/// mapped — the same argument the pool exists for on result and constant paths
/// (see `pool.rs`), now applied where a caller re-hands a host input to
/// `run_program` (which takes owned buffers) or a kernel clones a value.
///
/// The copy is split across `parallel_threads` scoped threads above the flat
/// map's own crossover, exactly like [`decode_le`], so a clone keeps that floor
/// without changing a single value: every byte is copied exactly once.
impl Clone for Storage {
    fn clone(&self) -> Storage {
        macro_rules! clone_arms {
            ($($variant:ident($ty:ty) => $values:ident),* $(,)?) => {
                match self {
                    $(Storage::$variant($values) => Storage::$variant(copy_slice_pooled::<$ty>($values)),)*
                }
            };
        }
        clone_arms! {
            Bool(bool) => values,
            I8(i8) => values,
            I16(i16) => values,
            I32(i32) => values,
            I64(i64) => values,
            U8(u8) => values,
            U16(u16) => values,
            U32(u32) => values,
            U64(u64) => values,
            F16(f16) => values,
            F32(f32) => values,
            F64(f64) => values,
            C64(Complex<f32>) => values,
            C128(Complex<f64>) => values,
        }
    }
}

pub(super) fn decode_storage(dtype: DType, bytes: &[u8]) -> Result<Storage> {
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

pub(super) fn cast_storage(source: &Storage, target: DType) -> Storage {
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
    /// One typed conversion pass into a POOLED destination: a cast overwrites
    /// every element, which is exactly the contract `take_pooled_vec` requires,
    /// and skipping the per-call allocation (and its page faults) is most of a
    /// 50k-element `cast`'s remaining cost.  No stock of the exact size leaves
    /// the plain `collect` arm.
    macro_rules! to {
        ($ty:ty) => {{
            let out: Vec<$ty> = match crate::buffer::take_pooled_vec::<$ty>(values.len()) {
                Some(mut pooled) => {
                    for (slot, value) in pooled.iter_mut().zip(values) {
                        *slot = <$ty as Scalar>::from_scalar(value.to_scalar());
                    }
                    pooled
                }
                None => values
                    .iter()
                    .map(|value| <$ty as Scalar>::from_scalar(value.to_scalar()))
                    .collect(),
            };
            <$ty as Scalar>::into_storage(out)
        }};
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::pool::POOL_MIN_BYTES;
    use crate::buffer::test_support::garbage_f32;
    use evoxc_core::DType;

    #[test]
    fn decode_recycles_and_still_reads_every_element() {
        let count = POOL_MIN_BYTES / std::mem::size_of::<f32>() + 32;
        garbage_f32(count);
        let words: Vec<f32> = (0..count).map(|index| index as f32 * 0.5).collect();
        let bytes: Vec<u8> = words.iter().flat_map(|value| value.to_le_bytes()).collect();
        let storage = decode_storage(DType::Float32, &bytes).unwrap();
        assert_eq!(storage, Storage::F32(words));
    }

    #[test]
    fn a_pooled_host_slice_copy_is_fully_overwritten() {
        // `from_slice_pooled` is the binding's input hand-off, so the stock it
        // takes from the pool must be overwritten end to end: a garbage stock of
        // exactly this size class proves no stale element survives.
        let count = POOL_MIN_BYTES / std::mem::size_of::<f32>() + 8;
        garbage_f32(count);
        let source: Vec<f32> = (0..count).map(|index| index as f32 * 0.25).collect();
        let original = source.clone();
        let storage = Storage::from_slice_pooled(&source);
        assert_eq!(
            storage,
            Storage::F32(original.clone()),
            "a pooled destination must carry the whole source"
        );
        // The copied storage owns its allocation: writing the source afterwards
        // leaves the result untouched.
        let mut source = source;
        source[0] = -1.0;
        assert_eq!(storage, Storage::F32(original));
    }

    #[test]
    fn a_small_host_slice_copy_falls_back_to_fresh_storage() {
        // Below `POOL_MIN_BYTES` the pool never retains anything, so this takes
        // the fresh arm and must still copy every element (and every dtype).
        let values: Vec<i32> = vec![-3, 0, 7, i32::MAX];
        assert_eq!(Storage::from_slice_pooled(&values), Storage::I32(values));
        let empty: Vec<f64> = Vec::new();
        assert_eq!(Storage::from_slice_pooled(&empty), Storage::F64(Vec::new()));
        let complex: Vec<Complex<f32>> = vec![Complex::new(1.0, -2.0)];
        assert_eq!(
            Storage::from_slice_pooled(&complex),
            Storage::C64(complex.clone())
        );
    }

    #[test]
    fn a_pooled_cast_destination_is_fully_overwritten() {
        // The cast recycles a pooled destination, so a stock handed back by a
        // previous conversion must be written end to end — no stale element may
        // survive into the next result.
        let count = POOL_MIN_BYTES / std::mem::size_of::<f32>() + 8;
        let source: Vec<i64> = (0..count as i64).map(|index| index * 3).collect();
        let expected: Vec<i32> = source.iter().map(|value| *value as i32).collect();
        let first = cast_storage(&Storage::I64(source.clone()), DType::Int32);
        assert_eq!(first, Storage::I32(expected.clone()));
        // Dropping the first result hands its storage to the thread-local pool,
        // so the second conversion takes the pooled arm.
        drop(first);
        let second = cast_storage(&Storage::I64(source), DType::Int32);
        assert_eq!(second, Storage::I32(expected));
    }
}
