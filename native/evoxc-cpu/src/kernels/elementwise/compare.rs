//! The comparison family (producing `bool`) and `select`.
use evoxc_core::{DType, Error, Result};
use num_complex::Complex;

use crate::buffer::{broadcast_shape, map1, map2, map2_runs, map3, map3_runs, Buffer, Storage};
use crate::exec::{single, RunCtx, Value};
use crate::plan::OpSpec;
use crate::scalar::Scalar;
use crate::with_same_type;

use super::binary::{is_ordering, layout, BinKind, Layout};
use super::{bool_results, BOOL_RESULT_THREAD_MIN};

/// Test-only count of [`broadcast_compare`] entries: pins that a rank-0 (and
/// any broadcast-free) comparison never reaches the general broadcast driver.
#[cfg(test)]
mod driver_calls {
    use std::cell::Cell;

    thread_local! {
        static CALLS: Cell<usize> = const { Cell::new(0) };
    }

    pub(super) fn bump() {
        CALLS.with(|calls| calls.set(calls.get() + 1));
    }

    /// The calls since the previous take (which resets the counter).
    pub(super) fn take() -> usize {
        CALLS.with(|calls| {
            let seen = calls.get();
            calls.set(0);
            seen
        })
    }
}

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
        _ => broadcast_compare(kind, out_shape, left, left_shape, right, right_shape),
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
        /// The broadcast-free comparison: a pooled bool destination filled by the
        /// iterator form (see [`bool_results`]) below the thread crossover, and
        /// the threaded `map2` builder at or above it.  `kind` is dispatched once
        /// per arm either way, so no `match` sits inside the element loop.
        fn dense_compare<T: Orderable + Copy>(
            kind: BinKind,
            out_shape: &[usize],
            left: &[T],
            right: &[T],
        ) -> Result<Buffer> {
            if out_shape.iter().product::<usize>() >= BOOL_RESULT_THREAD_MIN {
                return match kind {
                    $(BinKind::$kind => map2(DType::Bool, out_shape, left, right, |p, q| compare_op(BinKind::$kind, p, q)),)*
                    _ => map2(DType::Bool, out_shape, left, right, |p, q| compare_op(kind, p, q)),
                };
            }
            match kind {
                $(BinKind::$kind => bool_results(out_shape, left, right, |p, q| compare_op(BinKind::$kind, p, q)),)*
                _ => bool_results(out_shape, left, right, |p, q| compare_op(kind, p, q)),
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

        /// The broadcast-operand comparison: `buffer::zipmap`'s pooled broadcast
        /// builder `map2_runs` — one [`crate::buffer::broadcast`] `apply2` (an
        /// ndarray `Zip` pass over the operands' broadcast views), a result taken
        /// from the storage pool, and `buffer::rowsplit` cutting a wide result
        /// along its outermost axis — with the kind dispatched once per arm,
        /// exactly as the broadcast-free drivers above, so there is no per-element
        /// offset table and no zero-filled result.
        fn broadcast_compare<T: Orderable + Copy>(
            kind: BinKind,
            out_shape: &[usize],
            left: &[T],
            left_shape: &[usize],
            right: &[T],
            right_shape: &[usize],
        ) -> Result<Buffer> {
            #[cfg(test)]
            driver_calls::bump();
            match kind {
                $(BinKind::$kind => map2_runs(DType::Bool, out_shape, left, left_shape, right, right_shape, |p, q| compare_op(BinKind::$kind, p, q)),)*
                _ => map2_runs(DType::Bool, out_shape, left, left_shape, right, right_shape, |p, q| compare_op(kind, p, q)),
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
    // A broadcast predicate and/or branch: `buffer::zipmap`'s pooled broadcast
    // builder `map3_runs` — one `buffer::broadcast` `apply3` (an ndarray `Zip`
    // pass over the three broadcast views), a pooled result and the same
    // outermost-axis row split — which allocates no zero-filled result, with the
    // branch selection (a constant, not a per-element `match`) dispatched once
    // per pass.
    map3_runs(
        dtype,
        out_shape,
        pred,
        &predicate.shape,
        a,
        &a_buf.shape,
        b,
        &b_buf.shape,
        |p, x, y| if p { x } else { y },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use evoxc_core::{Dim, ValueType};

    use crate::kernels::elementwise::test_support::{
        binary, buffer, spec, zip2_reference, zip3_reference,
    };

    #[test]
    fn comparisons_produce_bool() {
        let a = buffer(DType::Float32, &[3], Storage::F32(vec![1.0, 2.0, 3.0]));
        let b = buffer(DType::Float32, &[3], Storage::F32(vec![2.0, 2.0, 2.0]));
        let op = spec("less", ValueType::of(DType::Bool, [Dim::int(3)]));
        let out = binary(&op, &a, &b).unwrap();
        assert_eq!(out[0].storage, Storage::Bool(vec![true, false, false]));
    }

    /// The broadcast contract: the broadcast builder (`map2_runs`) places the
    /// same bools as an INDEPENDENT ndarray `Zip` expression written out here —
    /// not a second call into a `buffer` driver, which is the code under test.
    fn assert_broadcast_compare_agrees(
        kind_name: &str,
        kind: BinKind,
        left: &[f32],
        left_shape: &[usize],
        right: &[f32],
        right_shape: &[usize],
        out_shape: &[usize],
    ) {
        let a = buffer(DType::Float32, left_shape, Storage::F32(left.to_vec()));
        let b = buffer(DType::Float32, right_shape, Storage::F32(right.to_vec()));
        let dims: Vec<Dim> = out_shape.iter().map(|dim| Dim::int(*dim as i64)).collect();
        let op = spec(kind_name, ValueType::of(DType::Bool, dims));
        let got = binary(&op, &a, &b).unwrap();
        let expected = zip2_reference(left, left_shape, right, right_shape, out_shape, |p, q| {
            compare_op(kind, p, q)
        });
        assert_eq!(
            got[0].storage,
            Storage::Bool(expected),
            "{kind_name}: {out_shape:?} {left_shape:?} vs {right_shape:?}"
        );
    }

    #[test]
    fn broadcast_comparisons_agree_with_the_general_driver() {
        // A row scalar against full rows (the `(n, m) < (n, 1)` shape a GP graph
        // produces), a trailing bias, a per-axis vector and a rank-0 operand —
        // with a NaN and signed zeros, so a reordering could not pass.
        let rows = [f32::NAN, -0.0, 1.0, 0.0, -1.0, 2.0];
        let column = [1.0f32, f32::NAN];
        let bias = [1.0f32, -0.0, 2.0];
        for (kind_name, kind) in [
            ("less", BinKind::Lt),
            ("less_equal", BinKind::Le),
            ("greater", BinKind::Gt),
            ("greater_equal", BinKind::Ge),
            ("equal", BinKind::Eq),
            ("not_equal", BinKind::Ne),
        ] {
            assert_broadcast_compare_agrees(kind_name, kind, &rows, &[2, 3], &column, &[2, 1], &[2, 3]);
            assert_broadcast_compare_agrees(kind_name, kind, &rows, &[2, 3], &bias, &[3], &[2, 3]);
            assert_broadcast_compare_agrees(kind_name, kind, &rows, &[2, 3], &[0.0], &[], &[2, 3]);
            assert_broadcast_compare_agrees(kind_name, kind, &rows, &[1, 2, 3], &column, &[2, 1], &[1, 2, 3]);
            assert_broadcast_compare_agrees(kind_name, kind, &bias, &[3], &rows, &[2, 3], &[2, 3]);
        }
    }

    #[test]
    fn complex_comparisons_keep_their_contract() {
        // `equal`/`not_equal` work on complex (a broadcast operand too), while an
        // ordering kind stays rejected with the documented message.
        let left = [
            Complex::new(1.0f32, 2.0),
            Complex::new(2.0, 2.0),
            Complex::new(3.0, 2.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 2.0),
            Complex::new(-0.0, 0.0),
        ];
        let right = [Complex::new(1.0f32, 2.0), Complex::new(0.0, 0.0)];
        let a = buffer(DType::Complex64, &[2, 3], Storage::C64(left.to_vec()));
        let b = buffer(DType::Complex64, &[2, 1], Storage::C64(right.to_vec()));
        let out_shape = [2usize, 3];
        let dims: Vec<Dim> = out_shape.iter().map(|dim| Dim::int(*dim as i64)).collect();
        let op = spec("equal", ValueType::of(DType::Bool, dims.clone()));
        let out = binary(&op, &a, &b).unwrap();
        // Row 0 against `(1, 2)`, row 1 against `(0, 0)` — where `-0.0` equals
        // `0.0` componentwise, as complex `PartialEq` says.
        assert_eq!(out[0].storage, Storage::Bool(vec![true, false, false, true, false, true]));
        let op = spec("less", ValueType::of(DType::Bool, dims));
        let error = binary(&op, &a, &b).unwrap_err();
        assert_eq!(
            error.to_string(),
            "BackendError: op 'less': ordering comparisons are not defined for complex numbers"
        );
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

    #[test]
    fn a_rank_zero_comparison_visits_one_element_and_no_broadcast_driver() {
        // The 0-d case (a `while_loop` condition in a real graph): both operands
        // are rank 0, so the flat arm produces its single element directly — one
        // element, one storage, no `walk` odometer and no broadcast driver.  That
        // is what keeps a per-iteration graph comparison sub-microsecond, and it
        // is the reason this test counts driver entries instead of timing.
        let a = buffer(DType::Int32, &[], Storage::I32(vec![1]));
        let b = buffer(DType::Int64, &[], Storage::I64(vec![2]));
        driver_calls::take();
        for (name, expected) in [
            ("less", true),
            ("less_equal", true),
            ("greater", false),
            ("greater_equal", false),
            ("equal", false),
            ("not_equal", true),
        ] {
            let op = spec(name, ValueType::of(DType::Bool, []));
            let out = binary(&op, &a, &b).unwrap();
            assert_eq!(out.len(), 1, "{name}");
            assert!(out[0].shape.is_empty(), "{name}");
            assert_eq!(out[0].storage, Storage::Bool(vec![expected]), "{name}");
        }
        assert_eq!(
            driver_calls::take(),
            0,
            "a rank-0 comparison reached the broadcast driver"
        );
        // The counter is live: a broadcast operand does enter that driver once.
        let matrix = buffer(DType::Float32, &[2, 3], Storage::F32(vec![0.0; 6]));
        let bias = buffer(DType::Float32, &[3], Storage::F32(vec![0.0; 3]));
        let op = spec(
            "less",
            ValueType::of(DType::Bool, [Dim::int(2), Dim::int(3)]),
        );
        binary(&op, &matrix, &bias).unwrap();
        assert_eq!(driver_calls::take(), 1);
    }

    #[test]
    fn the_flat_comparison_agrees_across_the_thread_crossover() {
        // Both sides of the builder switch: the pooled iterator form below the
        // crossover and the threaded `map2` arm at it.
        let count = BOOL_RESULT_THREAD_MIN;
        let left: Vec<f32> = (0..count).map(|index| index as f32).collect();
        let right: Vec<f32> = (0..count).map(|index| index as f32 * 0.5).collect();
        for size in [count - 1, count] {
            let out = dense_compare(BinKind::Lt, &[size], &left[..size], &right[..size]).unwrap();
            let expected: Vec<bool> = (0..size).map(|index| left[index] < right[index]).collect();
            assert_eq!(out.storage, Storage::Bool(expected), "{size}");
        }
    }

    /// The flat comparison arm pinned against the INDEPENDENT ndarray reference
    /// ([`zip2_reference`]): `reference` is the comparison written out by the
    /// test, so the oracle is not another `buffer` driver.
    fn assert_flat_compare_matches_reference<T, F>(
        kind: BinKind,
        shape: &[usize],
        left: &[T],
        right: &[T],
        reference: F,
    ) where
        T: Orderable + Copy + crate::buffer::NdScalar,
        F: Fn(T, T) -> bool,
    {
        let got = dense_compare::<T>(kind, shape, left, right)
            .unwrap_or_else(|err| panic!("flat comparison: {}", err.message));
        assert_eq!(got.dtype, DType::Bool);
        assert_eq!(got.shape, shape);
        assert_eq!(
            got.storage,
            Storage::Bool(zip2_reference(left, shape, right, shape, shape, reference)),
            "{kind:?} {shape:?}"
        );
    }

    #[test]
    fn the_flat_comparison_path_matches_an_ndarray_zip_expression() {
        // Every kind over values that include NaN (false against everything for
        // an ordering kind and for `equal`, true for `not_equal`), a signed zero
        // and an infinity — compared element for element with a plain `Zip`.
        let left = [f32::NAN, -0.0, 1.0, 0.0, f32::INFINITY, -1.0];
        let right = [1.0f32, 0.0, 1.0, 1.0, f32::INFINITY, f32::NAN];
        for kind in [
            BinKind::Eq,
            BinKind::Ne,
            BinKind::Lt,
            BinKind::Le,
            BinKind::Gt,
            BinKind::Ge,
        ] {
            let reference: fn(f32, f32) -> bool = match kind {
                BinKind::Eq => |a, b| a == b,
                BinKind::Ne => |a, b| a != b,
                BinKind::Lt => |a, b| a < b,
                BinKind::Le => |a, b| a <= b,
                BinKind::Gt => |a, b| a > b,
                _ => |a, b| a >= b,
            };
            for shape in [vec![6], vec![2, 3], vec![1, 2, 3]] {
                assert_flat_compare_matches_reference(kind, &shape, &left, &right, reference);
            }
        }
        // f64, i32 and a rank-0 pair (a `while_loop` condition's shape).
        let doubles_left = [1.0f64, f64::NAN, -2.5];
        let doubles_right = [1.0f64, 1.0, 0.0];
        assert_flat_compare_matches_reference(
            BinKind::Lt,
            &[3, 1],
            &doubles_left,
            &doubles_right,
            |a: f64, b: f64| a < b,
        );
        let ints_left = [3i32, -1, 7, 7];
        let ints_right = [3i32, 0, -8, 6];
        assert_flat_compare_matches_reference(
            BinKind::Le,
            &[2, 2],
            &ints_left,
            &ints_right,
            |a: i32, b: i32| a <= b,
        );
        assert_flat_compare_matches_reference(
            BinKind::Ge,
            &[],
            &[2i32],
            &[2i32],
            |a: i32, b: i32| a >= b,
        );
    }

    #[test]
    fn select_matches_an_ndarray_zip_expression() {
        // The broadcast-free `select` (one flat three-way map) and the non-flat
        // one (`map3_runs`, with a (2, 1) predicate and a (3,) branch), each
        // against a plain three-view `Zip` expression.
        let shape = [2usize, 3];
        let pred = [true, false, true, false, true, false];
        let on_true = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let on_false = [-1.0f32, -2.0, -3.0, -4.0, -5.0, -6.0];
        let select = |p: bool, x: f32, y: f32| if p { x } else { y };

        let predicate = buffer(DType::Bool, &shape, Storage::Bool(pred.to_vec()));
        let a = buffer(DType::Float32, &shape, Storage::F32(on_true.to_vec()));
        let b = buffer(DType::Float32, &shape, Storage::F32(on_false.to_vec()));
        let got = select_apply(
            DType::Float32,
            &pred,
            &predicate,
            &on_true,
            &a,
            &on_false,
            &b,
            &shape,
        )
        .unwrap();
        assert_eq!(
            got.storage,
            Storage::F32(zip3_reference(
                &pred,
                &shape,
                &on_true,
                &shape,
                &on_false,
                &shape,
                &shape,
                select
            ))
        );

        let column = [true, false];
        let row = [-1.0f32, -2.0, -3.0];
        let predicate = buffer(DType::Bool, &[2, 1], Storage::Bool(column.to_vec()));
        let b = buffer(DType::Float32, &[3], Storage::F32(row.to_vec()));
        let got = select_apply(
            DType::Float32,
            &column,
            &predicate,
            &on_true,
            &a,
            &row,
            &b,
            &shape,
        )
        .unwrap();
        assert_eq!(
            got.storage,
            Storage::F32(zip3_reference(
                &column,
                &[2, 1],
                &on_true,
                &shape,
                &row,
                &[3],
                &shape,
                select
            ))
        );

        // Integer branches go through `select_numeric`'s per-dtype dispatch.
        let ints_true = [10i32, 20, 30, 40, 50, 60];
        let ints_false = [-10i32, -20, -30, -40, -50, -60];
        let predicate = buffer(DType::Bool, &shape, Storage::Bool(pred.to_vec()));
        let a = buffer(DType::Int32, &shape, Storage::I32(ints_true.to_vec()));
        let b = buffer(DType::Int32, &shape, Storage::I32(ints_false.to_vec()));
        let got = select_numeric(DType::Int32, &predicate, &a, &b, &shape).unwrap();
        assert_eq!(
            got.storage,
            Storage::I32(zip3_reference(
                &pred,
                &shape,
                &ints_true,
                &shape,
                &ints_false,
                &shape,
                &shape,
                |p: bool, x: i32, y: i32| if p { x } else { y }
            ))
        );

        // Bool branches take the one-byte arm of `select_apply` (the `bool`
        // storage dispatch in `select_kernel`).
        let bool_true = [true, true, false, false, true, true];
        let bool_false = [false, true, true, false, false, true];
        let a = buffer(DType::Bool, &shape, Storage::Bool(bool_true.to_vec()));
        let b = buffer(DType::Bool, &shape, Storage::Bool(bool_false.to_vec()));
        let got = select_apply(
            DType::Bool,
            &pred,
            &predicate,
            &bool_true,
            &a,
            &bool_false,
            &b,
            &shape,
        )
        .unwrap();
        assert_eq!(
            got.storage,
            Storage::Bool(zip3_reference(
                &pred,
                &shape,
                &bool_true,
                &shape,
                &bool_false,
                &shape,
                &shape,
                |p: bool, x: bool, y: bool| if p { x } else { y }
            ))
        );
    }
}

