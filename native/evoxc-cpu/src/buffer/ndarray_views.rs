//! ndarray views over a [`Buffer`]: the transient borrow seam.
//!
//! [`Storage`] stays the engine's owning representation — one `Vec<T>` per
//! dtype, always C-contiguous — because that is what makes every kernel a flat
//! typed-slice loop and what the recycled-storage pool can hand back.  This
//! module adds a *view* seam beside it: [`Buffer::as_array`] lends the very same
//! elements to [`ndarray`] as an [`ndarray::ArrayViewD`] (no copy, no
//! allocation), [`Buffer::as_array_mut`] lends them mutably, and
//! [`Buffer::into_array`] hands the buffer's own allocation to an
//! [`ndarray::ArrayD`] (a move, not a copy).  Nothing here changes what a
//! `Buffer` is: a view borrows it, and a buffer that comes back from
//! [`Buffer::from_array`] is an ordinary `Buffer` again.
//!
//! [`NdScalar`] is the dtype-to-storage bridge the seam needs: the dtype tag
//! plus the four conversions (owning vector in, owning vector out by move, and
//! the two shared borrows).  It is deliberately independent of [`crate::scalar`]
//! so that neither trait can constrain the other.

use evoxc_core::{DType, Error, Result};
use half::f16;
use num_complex::Complex;

use super::{Buffer, Storage};

/// One Rust scalar type backed by an evoxc dtype, as the ndarray seam needs it.
///
/// The four storage conversions are the whole contract: [`NdScalar::into_storage`]
/// wraps a finished `Vec` in its [`Storage`] variant, [`NdScalar::take_storage`]
/// moves that `Vec` back out (no copy — the allocation travels), and
/// [`NdScalar::slice_of`] / [`NdScalar::slice_of_mut`] borrow it, which is what
/// makes [`Buffer::as_array`] and [`Buffer::as_array_mut`] zero-copy views
/// rather than conversions.
pub trait NdScalar: Copy + 'static + Send + Sync {
    /// The evoxc dtype this scalar type stores.
    const DTYPE: DType;
    /// Wrap an owning vector in the matching [`Storage`] variant.
    fn into_storage(values: Vec<Self>) -> Storage;
    /// Move this dtype's vector out of `storage` by value, leaving an empty one
    /// behind (`None` when the variant does not match).
    fn take_storage(storage: Storage) -> Option<Vec<Self>>;
    /// Borrow this dtype's elements out of `storage` (`None` on a mismatch).
    fn slice_of(storage: &Storage) -> Option<&[Self]>;
    /// Borrow this dtype's elements out of `storage` mutably (`None` on a
    /// mismatch).
    fn slice_of_mut(storage: &mut Storage) -> Option<&mut [Self]>;
}

/// Implement [`NdScalar`] for one `(rust type, dtype, storage variant)` triple.
///
/// `take_storage` goes through [`std::mem::take`] because [`Storage`] has a
/// `Drop` impl (the recycled-storage pool's admission hook): the typed vector is
/// moved out and the emptied storage is dropped behind it, which the pool
/// ignores (a zero-element storage is below its size floor).  The allocation
/// itself is untouched — `mem::take` swaps in `Vec::new()`.
macro_rules! impl_nd_scalar {
    ($ty:ty, $dtype:expr, $variant:ident) => {
        impl NdScalar for $ty {
            const DTYPE: DType = $dtype;

            fn into_storage(values: Vec<Self>) -> Storage {
                Storage::$variant(values)
            }

            fn take_storage(mut storage: Storage) -> Option<Vec<Self>> {
                match &mut storage {
                    Storage::$variant(values) => Some(std::mem::take(values)),
                    _ => None,
                }
            }

            fn slice_of(storage: &Storage) -> Option<&[Self]> {
                match storage {
                    Storage::$variant(values) => Some(values.as_slice()),
                    _ => None,
                }
            }

            fn slice_of_mut(storage: &mut Storage) -> Option<&mut [Self]> {
                match storage {
                    Storage::$variant(values) => Some(values.as_mut_slice()),
                    _ => None,
                }
            }
        }
    };
}

impl_nd_scalar!(bool, DType::Bool, Bool);
impl_nd_scalar!(i8, DType::Int8, I8);
impl_nd_scalar!(i16, DType::Int16, I16);
impl_nd_scalar!(i32, DType::Int32, I32);
impl_nd_scalar!(i64, DType::Int64, I64);
impl_nd_scalar!(u8, DType::UInt8, U8);
impl_nd_scalar!(u16, DType::UInt16, U16);
impl_nd_scalar!(u32, DType::UInt32, U32);
impl_nd_scalar!(u64, DType::UInt64, U64);
impl_nd_scalar!(f16, DType::Float16, F16);
impl_nd_scalar!(f32, DType::Float32, F32);
impl_nd_scalar!(f64, DType::Float64, F64);
impl_nd_scalar!(Complex<f32>, DType::Complex64, C64);
impl_nd_scalar!(Complex<f64>, DType::Complex128, C128);

/// The one diagnostic for "this buffer is not of the requested dtype".
fn dtype_mismatch(actual: DType, requested: DType) -> Error {
    Error::dtype(format!(
        "buffer dtype {} does not match requested {}",
        actual.name(),
        requested.name()
    ))
}

/// Borrow `values` as a dynamic-rank [`ndarray::ArrayViewD`] of `shape`.
///
/// The view does not own anything: it borrows `values`, which must hold exactly
/// `shape`'s element count, and it is the caller's job to keep the slices'
/// lifetimes straight.  A shape that does not match the element count is an
/// explicit `ShapeError` (never a truncated or ragged view).
pub fn array_view<'a, T: NdScalar>(
    values: &'a [T],
    shape: &[usize],
) -> Result<ndarray::ArrayViewD<'a, T>> {
    let count = values.len();
    ndarray::ArrayViewD::<T>::from_shape(ndarray::IxDyn(shape), values).map_err(|error| {
        Error::shape(format!(
            "cannot view {count} element(s) as shape {shape:?}: {error}"
        ))
    })
}

/// Borrow `values` mutably as a dynamic-rank [`ndarray::ArrayViewMutD`] of
/// `shape` — the writable counterpart of [`array_view`].
///
/// A write through the view lands in `values` itself; a shape that does not
/// match the element count is an explicit `ShapeError`.
pub fn array_view_mut<'a, T: NdScalar>(
    values: &'a mut [T],
    shape: &[usize],
) -> Result<ndarray::ArrayViewMutD<'a, T>> {
    let count = values.len();
    ndarray::ArrayViewMutD::<T>::from_shape(ndarray::IxDyn(shape), values).map_err(|error| {
        Error::shape(format!(
            "cannot view {count} element(s) mutably as shape {shape:?}: {error}"
        ))
    })
}

impl Buffer {
    /// Borrow this buffer's elements as an [`ndarray::ArrayViewD`].
    ///
    /// Zero-copy and allocation-free: the view borrows the buffer's own storage
    /// (`[`Storage`] stays the owning `Vec`-backed representation), so it is a
    /// transient seam — it cannot outlive `self`, and the buffer cannot be
    /// mutated while it is alive.  Asking for a dtype other than
    /// [`Buffer::dtype`] is an explicit `DTypeError` naming both dtypes.
    pub fn as_array<T: NdScalar>(&self) -> Result<ndarray::ArrayViewD<'_, T>> {
        if self.dtype != T::DTYPE {
            return Err(dtype_mismatch(self.dtype, T::DTYPE));
        }
        let values = T::slice_of(&self.storage)
            .ok_or_else(|| dtype_mismatch(self.dtype, T::DTYPE))?;
        array_view(values, &self.shape)
    }

    /// Borrow this buffer's elements as a mutable [`ndarray::ArrayViewMutD`].
    ///
    /// Zero-copy: a write through the view lands in the buffer's own storage.
    /// A dtype other than [`Buffer::dtype`] is an explicit `DTypeError` naming
    /// both dtypes.
    pub fn as_array_mut<T: NdScalar>(&mut self) -> Result<ndarray::ArrayViewMutD<'_, T>> {
        if self.dtype != T::DTYPE {
            return Err(dtype_mismatch(self.dtype, T::DTYPE));
        }
        let dtype = self.dtype;
        let shape = &self.shape;
        let values = T::slice_of_mut(&mut self.storage)
            .ok_or_else(|| dtype_mismatch(dtype, T::DTYPE))?;
        array_view_mut(values, shape)
    }

    /// Borrow this buffer's elements as a rank-2 [`ndarray::ArrayView2`].
    ///
    /// Sugar over [`Buffer::as_array`] for the common matrix case: the same
    /// zero-copy view, narrowed to two dimensions.  A buffer of any other rank
    /// is an explicit `ShapeError` naming the rank it actually has.
    pub fn as_array2<T: NdScalar>(&self) -> Result<ndarray::ArrayView2<'_, T>> {
        self.as_array::<T>()?
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| {
                Error::shape(format!(
                    "expected a rank-2 buffer, got rank {}",
                    self.rank()
                ))
            })
    }

    /// Take ownership of an [`ndarray::ArrayD`] as a [`Buffer`].
    ///
    /// The array's elements move into the buffer's storage: a standard-layout
    /// (C-order) array moves its own allocation, and any other layout is first
    /// standardized into C order (one copy), because a `Buffer`'s storage is
    /// always C-contiguous.  The resulting buffer is an ordinary owning value.
    pub fn from_array<T: NdScalar>(array: ndarray::ArrayD<T>) -> Result<Buffer> {
        let owned = if array.is_standard_layout() {
            array
        } else {
            array.as_standard_layout().into_owned()
        };
        let shape = owned.shape().to_vec();
        let (values, _offset) = owned.into_raw_vec_and_offset();
        Buffer::new(T::DTYPE, shape, T::into_storage(values))
    }

    /// Move this buffer's elements into an [`ndarray::ArrayD`].
    ///
    /// A move, not a copy: the buffer's own `Vec` becomes the array's data (a
    /// `Buffer`'s storage is always C-contiguous, so no restriding is needed).
    /// A dtype other than [`Buffer::dtype`] is an explicit `DTypeError` naming
    /// both dtypes.
    pub fn into_array<T: NdScalar>(self) -> Result<ndarray::ArrayD<T>> {
        let Buffer { dtype, shape, storage } = self;
        if dtype != T::DTYPE {
            return Err(dtype_mismatch(dtype, T::DTYPE));
        }
        let values = T::take_storage(storage).ok_or_else(|| dtype_mismatch(dtype, T::DTYPE))?;
        let count = values.len();
        ndarray::ArrayD::<T>::from_shape_vec(ndarray::IxDyn(&shape), values).map_err(|error| {
            Error::shape(format!(
                "cannot build an ndarray of shape {shape:?} from {count} element(s): {error}"
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evoxc_core::{DType, ErrorKind};

    /// One dtype's round trip: buffer -> view -> buffer, and buffer -> array.
    fn round_trip<T>(values: Vec<T>)
    where
        T: NdScalar + PartialEq + std::fmt::Debug,
    {
        let shape = vec![values.len()];
        let buffer =
            Buffer::new(T::DTYPE, shape.clone(), T::into_storage(values.clone())).expect("buffer");
        assert_eq!(buffer.dtype, T::DTYPE);

        // as_array: a borrowed view of the very same elements.
        let view = buffer.as_array::<T>().expect("a matching view");
        assert_eq!(view.shape(), shape.as_slice());
        assert_eq!(view.iter().copied().collect::<Vec<T>>(), values);

        // as_array_mut: the writable seam reads the same elements.
        let mut buffer = buffer;
        let view_mut = buffer.as_array_mut::<T>().expect("a matching mut view");
        assert_eq!(view_mut.iter().copied().collect::<Vec<T>>(), values);
        drop(view_mut);

        // from_array: an owned ndarray becomes an owning buffer again.
        let array = ndarray::ArrayD::<T>::from_shape_vec(ndarray::IxDyn(&shape), values.clone())
            .expect("a consistent array");
        let restored = Buffer::from_array(array).expect("a buffer");
        assert_eq!(restored.dtype, T::DTYPE);
        assert_eq!(restored.shape, shape);
        assert_eq!(T::slice_of(&restored.storage).expect("same dtype"), values.as_slice());

        // into_array: the buffer's elements move into an owned array.
        let array = buffer.into_array::<T>().expect("a matching array");
        assert_eq!(array.shape(), shape.as_slice());
        let (raw, _offset) = array.into_raw_vec_and_offset();
        assert_eq!(raw, values);

        // take_storage moves rather than copies: the emptied variant is left
        // behind, and a mismatched variant is `None`.
        assert!(T::take_storage(T::into_storage(values.clone())).is_some());
        let other = Storage::Bool(Vec::new());
        if T::DTYPE == DType::Bool {
            assert!(T::take_storage(other).is_some());
        } else {
            assert!(T::take_storage(other).is_none());
        }
    }

    #[test]
    fn every_dtype_round_trips() {
        round_trip::<bool>(vec![true, false, true, true, false]);
        round_trip::<i8>(vec![-128, -1, 0, 7, 127]);
        round_trip::<i16>(vec![-32768, -3, 0, 5, 32767]);
        round_trip::<i32>(vec![-7, -1, 0, 1, 7]);
        round_trip::<i64>(vec![i64::MIN, -1, 0, 1, i64::MAX]);
        round_trip::<u8>(vec![0, 1, 2, 128, 255]);
        round_trip::<u16>(vec![0, 1, 2, 128, 65535]);
        round_trip::<u32>(vec![0, 1, 2, 3, u32::MAX]);
        round_trip::<u64>(vec![0, 1, 2, 3, u64::MAX]);
        round_trip::<f16>(vec![
            f16::from_f32(0.0),
            f16::from_f32(-1.5),
            f16::from_f32(2.25),
        ]);
        round_trip::<f32>(vec![0.0, -1.5, 2.25, f32::MAX]);
        round_trip::<f64>(vec![0.0, -1.5, 2.25, f64::MIN]);
        round_trip::<Complex<f32>>(vec![
            Complex::new(1.0, -2.0),
            Complex::new(0.0, 0.5),
            Complex::new(-3.5, 4.0),
        ]);
        round_trip::<Complex<f64>>(vec![
            Complex::new(1.0, -2.0),
            Complex::new(0.0, 0.5),
            Complex::new(-3.5, 4.0),
        ]);
    }

    #[test]
    fn into_array_moves_the_allocation() {
        // The seam claims zero copy in the owning direction too: the buffer's
        // Vec (not a fresh one) must become the array's data.
        let values: Vec<f32> = (0..64).map(|index| index as f32 * 0.5).collect();
        let ptr = values.as_ptr();
        let buffer =
            Buffer::new(DType::Float32, vec![8, 8], Storage::F32(values)).expect("buffer");
        let array = buffer.into_array::<f32>().expect("array");
        let (raw, _offset) = array.into_raw_vec_and_offset();
        assert_eq!(raw.as_ptr(), ptr, "into_array must move the Vec, not copy it");
        assert_eq!(raw.len(), 64);
        assert_eq!(raw[9], 4.5);
    }

    #[test]
    fn a_rank_zero_buffer_views_and_moves() {
        let buffer = Buffer::new(DType::Float64, Vec::new(), Storage::F64(vec![2.5]))
            .expect("rank-0 buffer");
        let view = buffer.as_array::<f64>().expect("rank-0 view");
        assert_eq!(view.shape(), &[] as &[usize]);
        assert_eq!(view.iter().copied().collect::<Vec<f64>>(), vec![2.5]);
        let array = buffer.into_array::<f64>().expect("rank-0 array");
        assert_eq!(array.shape(), &[] as &[usize]);
        assert_eq!(array.into_raw_vec_and_offset().0, vec![2.5]);
    }

    #[test]
    fn a_zero_element_shape_round_trips() {
        let buffer = Buffer::new(DType::Int32, vec![0, 3], Storage::I32(Vec::new())).expect("empty");
        let view = buffer.as_array::<i32>().expect("empty view");
        assert_eq!(view.shape(), &[0, 3]);
        assert_eq!(view.len(), 0);
        let array = buffer.into_array::<i32>().expect("empty array");
        assert_eq!(array.shape(), &[0, 3]);
        assert_eq!(array.into_raw_vec_and_offset().0, Vec::<i32>::new());
        // And back in from an empty owned array.
        let empty = ndarray::ArrayD::<i32>::from_shape_vec(ndarray::IxDyn(&[0, 3]), Vec::new())
            .expect("empty array");
        let restored = Buffer::from_array(empty).expect("empty buffer");
        assert_eq!(restored.shape, vec![0, 3]);
        assert_eq!(restored.numel(), 0);
    }

    #[test]
    fn as_array2_requires_rank_two() {
        let matrix = Buffer::new(DType::Float32, vec![2, 3], Storage::F32(vec![0.0; 6]))
            .expect("matrix");
        let view = matrix.as_array2::<f32>().expect("rank-2 view");
        assert_eq!(view.shape(), &[2, 3]);
        assert_eq!(view[[1, 2]], 0.0);

        let cube = Buffer::new(DType::Float32, vec![1, 2, 3], Storage::F32(vec![0.0; 6]))
            .expect("rank-3 buffer");
        let error = cube.as_array2::<f32>().expect_err("rank 3 is not rank 2");
        assert_eq!(error.kind, ErrorKind::Shape);
        assert!(
            error.to_string().contains("rank-2") && error.to_string().contains('3'),
            "unhelpful message: {error}"
        );
    }

    #[test]
    fn as_array_mut_writes_through_to_the_buffer() {
        let mut buffer = Buffer::new(DType::Int64, vec![2, 2], Storage::I64(vec![1, 2, 3, 4]))
            .expect("buffer");
        {
            let mut view = buffer.as_array_mut::<i64>().expect("mut view");
            view[[0, 1]] = 20;
            view[[1, 0]] = 30;
        }
        // C order: [0,1] is flat index 1, [1,0] is flat index 2.
        assert_eq!(buffer.storage, Storage::I64(vec![1, 20, 30, 4]));
    }

    #[test]
    fn a_dtype_mismatch_names_both_dtypes() {
        let buffer = Buffer::new(DType::Float32, vec![2], Storage::F32(vec![1.0, 2.0]))
            .expect("buffer");
        let error = buffer.as_array::<i32>().expect_err("f32 is not i32");
        assert_eq!(error.kind, ErrorKind::DType);
        let text = error.to_string();
        assert!(text.starts_with("DTypeError:"), "unexpected text: {text}");
        assert!(text.contains("float32") && text.contains("int32"), "unexpected text: {text}");

        let mut buffer = buffer;
        let error = buffer.as_array_mut::<f64>().expect_err("f32 is not f64");
        assert_eq!(error.kind, ErrorKind::DType);
        assert!(error.to_string().contains("float32") && error.to_string().contains("float64"));

        let error = buffer.clone().into_array::<u8>().expect_err("f32 is not u8");
        assert_eq!(error.kind, ErrorKind::DType);
        assert!(error.to_string().contains("float32") && error.to_string().contains("uint8"));

        let empty = Buffer::new(DType::Float32, vec![0], Storage::F32(Vec::new()))
            .expect("empty buffer");
        let error = empty.as_array2::<f64>().expect_err("f32 is not f64");
        assert_eq!(error.kind, ErrorKind::DType);
    }

    #[test]
    fn from_array_standardizes_a_non_standard_layout() {
        // A transposed (column-major) 2x3 array must come back C-contiguous,
        // holding the same logical elements in C order.
        let base = ndarray::Array2::<f32>::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("array");
        let transposed = base.clone().reversed_axes();
        assert!(!transposed.is_standard_layout());
        let buffer = Buffer::from_array(transposed.into_dyn()).expect("buffer");
        assert_eq!(buffer.shape, vec![3, 2]);
        assert_eq!(buffer.storage, Storage::F32(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]));

        // The same logical array in standard layout keeps its elements as-is.
        let standard = Buffer::from_array(base.clone().into_dyn()).expect("buffer");
        assert_eq!(standard.shape, vec![2, 3]);
        assert_eq!(standard.storage, Storage::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    }

    #[test]
    fn a_shape_that_does_not_match_the_slice_is_rejected() {
        let error = array_view::<f32>(&[1.0, 2.0, 3.0], &[2, 2]).expect_err("3 is not 4");
        assert_eq!(error.kind, ErrorKind::Shape);
        let mut values = [1.0f32, 2.0, 3.0];
        let error = array_view_mut::<f32>(&mut values, &[4]).expect_err("3 is not 4");
        assert_eq!(error.kind, ErrorKind::Shape);
        // The matching case succeeds (and is a borrow of the caller's slice).
        let view = array_view::<f32>(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("view");
        assert_eq!(view[[1, 0]], 3.0);
    }

    #[test]
    fn an_empty_slice_views_as_an_empty_shape() {
        // An empty shape family must not allocate or panic on either side.
        let view = array_view::<i8>(&[], &[0, 5]).expect("empty view");
        assert_eq!(view.shape(), &[0, 5]);
        let mut empty: Vec<u8> = Vec::new();
        let view = array_view_mut::<u8>(&mut empty, &[0]).expect("empty mut view");
        assert_eq!(view.len(), 0);
    }
}
