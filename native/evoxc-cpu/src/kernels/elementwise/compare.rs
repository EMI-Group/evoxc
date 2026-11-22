//! The comparison family (producing `bool`) and `select`.
use evoxc_core::{DType, Error, Result};
use num_complex::Complex;

use crate::buffer::{apply2, apply3, broadcast_shape, map1, map2, map3, Buffer, Storage};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::plan::OpSpec;
use crate::scalar::Scalar;
use crate::with_same_type;

use super::binary::{is_ordering, layout, BinKind, Layout};

pub(super) fn compare(
    name: &str,
    kind: BinKind,
    a: &Buffer,
    b: &Buffer,
    out_shape: &[usize],
) -> Result<Vec<Value>> {
    let compute = if a.dtype == b.dtype {
        a.dtype
    } else {
        DType::promote(a.dtype, b.dtype)
    };
    if compute.is_complex() && is_ordering(kind) {
        return Err(Error::backend(format!(
            "op '{name}': ordering comparisons are not defined for complex numbers"
        )));
    }
    let left = if a.dtype == compute {
        None
    } else {
        Some(a.cast(compute)?)
    };
    let right = if b.dtype == compute {
        None
    } else {
        Some(b.cast(compute)?)
    };
    let left = left.as_ref().unwrap_or(a);
    let right = right.as_ref().unwrap_or(b);
    // Complex values only support `equal`/`not_equal` (an ordering kind is
    // rejected above) and every dtype implements [`Orderable`], so one driver
    // serves both.
    with_same_type!(left, right, |x, y| {
        compare_typed(x, &left.shape, y, &right.shape, kind, out_shape)
    })
}

/// Ordering comparisons, without requiring `PartialOrd` (complex has none).
trait Orderable: PartialEq + Send + Sync {
    fn lt(&self, other: &Self) -> bool;
    fn le(&self, other: &Self) -> bool;
    fn gt(&self, other: &Self) -> bool;
    fn ge(&self, other: &Self) -> bool;
}

macro_rules! impl_orderable {
    ($($ty:ty),* $(,)?) => {
        $(impl Orderable for $ty {
            fn lt(&self, other: &Self) -> bool { *self < *other }
            fn le(&self, other: &Self) -> bool { *self <= *other }
            fn gt(&self, other: &Self) -> bool { *self > *other }
            fn ge(&self, other: &Self) -> bool { *self >= *other }
        })*
    };
}

impl_orderable!(bool, i8, i16, i32, i64, u8, u16, u32, u64, half::f16, f32, f64);

impl Orderable for Complex<f32> {
    fn lt(&self, _other: &Self) -> bool { false }
    fn le(&self, _other: &Self) -> bool { false }
    fn gt(&self, _other: &Self) -> bool { false }
    fn ge(&self, _other: &Self) -> bool { false }
}

impl Orderable for Complex<f64> {
    fn lt(&self, _other: &Self) -> bool { false }
    fn le(&self, _other: &Self) -> bool { false }
    fn gt(&self, _other: &Self) -> bool { false }
    fn ge(&self, _other: &Self) -> bool { false }
}

fn compare_typed<T: Orderable + Copy>(
    left: &[T],
    left_shape: &[usize],
    right: &[T],
    right_shape: &[usize],
    kind: BinKind,
    out_shape: &[usize],
) -> Result<Vec<Value>> {
    let out = match (
        layout(out_shape, left, left_shape),
        layout(out_shape, right, right_shape),
    ) {
        (Layout::Flat, Layout::Flat) => dense_compare(kind, out_shape, left, right),
        (Layout::Flat, Layout::Scalar) if !right.is_empty() => {
            dense_compare_right_scalar(kind, out_shape, left, right[0])
        }
        (Layout::Scalar, Layout::Flat) if !left.is_empty() => {
            dense_compare_left_scalar(kind, out_shape, left[0], right)
        }
        _ => {
            let mut out = vec![false; util::numel(out_shape)];
            apply2(&mut out, out_shape, left, left_shape, right, right_shape, |p, q| {
                compare_op(kind, p, q)
            });
            Buffer::new(DType::Bool, out_shape.to_vec(), Storage::Bool(out))
        }
    }?;
    single(out)
}

/// One comparison, numpy semantics (a NaN compares false against everything).
fn compare_op<T: Orderable>(kind: BinKind, left: T, right: T) -> bool {
    match kind {
        BinKind::Eq => left == right,
        BinKind::Ne => left != right,
        BinKind::Lt => left.lt(&right),
        BinKind::Le => left.le(&right),
        BinKind::Gt => left.gt(&right),
        BinKind::Ge => left.ge(&right),
        _ => false,
    }
}

/// The broadcast-free comparison fast paths (one arm list, three layouts, as
/// for the arithmetic drivers): `kind` is dispatched once and each arm is a
/// flat map into a freshly built bool result.
macro_rules! dense_compare_drivers {
    ($($kind:ident),* $(,)?) => {
        fn dense_compare<T: Orderable + Copy>(
            kind: BinKind,
            out_shape: &[usize],
            left: &[T],
            right: &[T],
        ) -> Result<Buffer> {
            match kind {
                $(BinKind::$kind => map2(DType::Bool, out_shape, left, right, |p, q| compare_op(BinKind::$kind, p, q)),)*
                _ => map2(DType::Bool, out_shape, left, right, |p, q| compare_op(kind, p, q)),
            }
        }

        fn dense_compare_left_scalar<T: Orderable + Copy>(
            kind: BinKind,
            out_shape: &[usize],
            left: T,
            right: &[T],
        ) -> Result<Buffer> {
            match kind {
                $(BinKind::$kind => map1(DType::Bool, out_shape, right, |q| compare_op(BinKind::$kind, left, q)),)*
                _ => map1(DType::Bool, out_shape, right, |q| compare_op(kind, left, q)),
            }
        }

        fn dense_compare_right_scalar<T: Orderable + Copy>(
            kind: BinKind,
            out_shape: &[usize],
            left: &[T],
            right: T,
        ) -> Result<Buffer> {
            match kind {
                $(BinKind::$kind => map1(DType::Bool, out_shape, left, |p| compare_op(BinKind::$kind, p, right)),)*
                _ => map1(DType::Bool, out_shape, left, |p| compare_op(kind, p, right)),
            }
        }
    };
}

dense_compare_drivers!(Eq, Ne, Lt, Le, Gt, Ge);

pub(super) fn select_kernel(_ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let out_dtype = spec.result_types[0].dtype;
    let predicate = &operands[0];
    let on_true = if operands[1].dtype == out_dtype {
        None
    } else {
        Some(operands[1].cast(out_dtype)?)
    };
    let on_false = if operands[2].dtype == out_dtype {
        None
    } else {
        Some(operands[2].cast(out_dtype)?)
    };
    let on_true = on_true.as_ref().unwrap_or(&operands[1]);
    let on_false = on_false.as_ref().unwrap_or(&operands[2]);
    let pred_owned = if predicate.dtype == DType::Bool {
        None
    } else {
        Some(predicate.cast(DType::Bool)?)
    };
    let predicate = pred_owned.as_ref().unwrap_or(predicate);
    let out_shape = broadcast_shape(&[&predicate.shape, &on_true.shape, &on_false.shape])?;
    // Dispatch over the branch dtype; the predicate is `bool`.
    let out = match (&on_true.storage, &on_false.storage, &predicate.storage) {
        (Storage::Bool(a), Storage::Bool(b), Storage::Bool(p)) => {
            select_apply(DType::Bool, p, predicate, a, on_true, b, on_false, &out_shape)?
        }
        _ => select_numeric(out_dtype, predicate, on_true, on_false, &out_shape)?,
    };
    single(out)
}

fn select_numeric(
    out_dtype: DType,
    predicate: &Buffer,
    on_true: &Buffer,
    on_false: &Buffer,
    out_shape: &[usize],
) -> Result<Buffer> {
    let pred = match &predicate.storage {
        Storage::Bool(values) => values,
        _ => return Err(Error::dtype("op 'select': predicate must be bool")),
    };
    // One `select_apply` instantiation per branch dtype (the macro keeps the
    // arm list flat: a closure cannot be generic over the element type).
    macro_rules! select {
        ($a:expr, $b:expr) => {
            select_apply(out_dtype, pred, predicate, $a, on_true, $b, on_false, out_shape)
        };
    }
    match (&on_true.storage, &on_false.storage) {
        (Storage::I8(a), Storage::I8(b)) => select!(a, b),
        (Storage::I16(a), Storage::I16(b)) => select!(a, b),
        (Storage::I32(a), Storage::I32(b)) => select!(a, b),
        (Storage::I64(a), Storage::I64(b)) => select!(a, b),
        (Storage::U8(a), Storage::U8(b)) => select!(a, b),
        (Storage::U16(a), Storage::U16(b)) => select!(a, b),
        (Storage::U32(a), Storage::U32(b)) => select!(a, b),
        (Storage::U64(a), Storage::U64(b)) => select!(a, b),
        (Storage::F16(a), Storage::F16(b)) => select!(a, b),
        (Storage::F32(a), Storage::F32(b)) => select!(a, b),
        (Storage::F64(a), Storage::F64(b)) => select!(a, b),
        (Storage::C64(a), Storage::C64(b)) => select!(a, b),
        (Storage::C128(a), Storage::C128(b)) => select!(a, b),
        _ => Err(Error::dtype(format!(
            "op 'select': branches have incompatible dtypes {} and {}",
            on_true.dtype.name(),
            on_false.dtype.name()
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn select_apply<T>(
    dtype: DType,
    pred: &[bool],
    predicate: &Buffer,
    a: &[T],
    a_buf: &Buffer,
    b: &[T],
    b_buf: &Buffer,
    out_shape: &[usize],
) -> Result<Buffer>
where
    T: Copy + Scalar,
{
    if predicate.is_flat_for(out_shape) && a_buf.is_flat_for(out_shape) && b_buf.is_flat_for(out_shape)
    {
        // Broadcast-free: one flat three-way map into a freshly built result.
        return map3(dtype, out_shape, pred, a, b, |p, x, y| if p { x } else { y });
    }
    let mut out = vec![T::zero(); util::numel(out_shape)];
    apply3(
        &mut out,
        out_shape,
        pred,
        &predicate.shape,
        a,
        &a_buf.shape,
        b,
        &b_buf.shape,
        |p, x, y| if p { x } else { y },
    );
    Buffer::new(dtype, out_shape.to_vec(), T::into_storage(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use evoxc_core::{Dim, ValueType};

    use crate::kernels::elementwise::test_support::{binary, buffer, spec};

    #[test]
    fn comparisons_produce_bool() {
        let a = buffer(DType::Float32, &[3], Storage::F32(vec![1.0, 2.0, 3.0]));
        let b = buffer(DType::Float32, &[3], Storage::F32(vec![2.0, 2.0, 2.0]));
        let op = spec("less", ValueType::of(DType::Bool, [Dim::int(3)]));
        let out = binary(&op, &a, &b).unwrap();
        assert_eq!(out[0].storage, Storage::Bool(vec![true, false, false]));
    }

    #[test]
    fn flat_select_agrees_with_the_broadcast_path() {
        let predicate = buffer(DType::Bool, &[2, 2], Storage::Bool(vec![true, false, true, false]));
        let on_true = buffer(DType::Float32, &[2, 2], Storage::F32(vec![1., 2., 3., 4.]));
        let on_false = buffer(DType::Float32, &[2, 2], Storage::F32(vec![-1., -2., -1., -2.]));
        let row = buffer(DType::Float32, &[1, 2], Storage::F32(vec![-1., -2.]));
        let out_shape = [2usize, 2];
        let flat = select_numeric(DType::Float32, &predicate, &on_true, &on_false, &out_shape)
            .unwrap()
            .storage;
        let broadcast = select_numeric(DType::Float32, &predicate, &on_true, &row, &out_shape)
            .unwrap()
            .storage;
        assert_eq!(flat, Storage::F32(vec![1., -2., 3., -2.]));
        assert_eq!(flat, broadcast);
        // A scalar branch selects against the general driver too.
        let scalar = buffer(DType::Float32, &[], Storage::F32(vec![-1.0]));
        assert_eq!(
            select_numeric(DType::Float32, &predicate, &on_true, &scalar, &out_shape)
                .unwrap()
                .storage,
            Storage::F32(vec![1., -1., 3., -1.])
        );
    }

}
