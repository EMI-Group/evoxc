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
//!
//! The pieces live in submodules (`storage`, `broadcast`, `ndarray_views`,
//! `pool`, `parallel`, `zipmap`, `blocks`) and are re-exported here, so every
//! item keeps its `crate::buffer::…` path.

use evoxc_core::{ArrayPayload, DType, Error, Result};
use half::f16;
use num_complex::Complex;

mod blocks;
mod broadcast;
#[cfg(test)]
mod bench_broadcast;
mod ndarray_views;
mod parallel;
mod pool;
mod storage;
#[cfg(test)]
mod test_support;
mod zipmap;

pub use blocks::{block_storage, block_storage_filled, copy_blocks, repeat_storage};
pub use broadcast::{
    apply1, apply2, apply3, broadcast_shape, broadcast_strides, broadcast_to, fill_broadcast, walk,
};
pub use broadcast::{apply2 as apply2_runs, apply3 as apply3_runs};
pub use ndarray_views::{array_view, array_view_mut, NdScalar};
pub use storage::Storage;
pub use zipmap::{map1, map2, map2_runs, map3, map3_runs, zip1, zip2, zip2_accumulate, zip3};
#[allow(unused_imports)] // kernels adopt it per hot path (see `take_pooled_vec`)
pub(crate) use pool::take_pooled_vec;

use self::storage::{cast_storage, decode_storage};

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
