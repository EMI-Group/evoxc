//! The arithmetic, bitwise and logical binary family: the [`BinKind`] table,
//! the allocating drivers and the in-place accumulation path.
//!
//! The fast-path/vectorization contract for these drivers is documented in the
//! parent module ([`super`]).
use evoxc_core::{DType, Error, Result};
use std::rc::Rc;

use crate::buffer::{
    apply2, broadcast_shape, map1, map2, shape_text, zip2_accumulate, Buffer, Storage,
};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::pow;
use crate::kernels::util;
use crate::plan::{InPlaceFn, OpSpec};
use crate::scalar::Num;
use crate::{with_same_type, with_same_type_mut};

use super::compare::compare;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BinKind {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Rem,
    Max,
    Min,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

fn bin_kind(name: &str) -> Option<BinKind> {
    Some(match name {
        "add" => BinKind::Add,
        "subtract" => BinKind::Sub,
        "multiply" => BinKind::Mul,
        "divide" => BinKind::Div,
        "power" => BinKind::Pow,
        "remainder" => BinKind::Rem,
        "maximum" => BinKind::Max,
        "minimum" => BinKind::Min,
        "bitwise_and" => BinKind::BitAnd,
        "bitwise_or" => BinKind::BitOr,
        "bitwise_xor" => BinKind::BitXor,
        "bitwise_left_shift" => BinKind::Shl,
        "bitwise_right_shift" => BinKind::Shr,
        "logical_and" => BinKind::Mul,
        "logical_or" => BinKind::Add,
        "equal" => BinKind::Eq,
        "not_equal" => BinKind::Ne,
        "less" => BinKind::Lt,
        "less_equal" => BinKind::Le,
        "greater" => BinKind::Gt,
        "greater_equal" => BinKind::Ge,
        _ => return None,
    })
}

fn is_comparison(kind: BinKind) -> bool {
    matches!(
        kind,
        BinKind::Eq | BinKind::Ne | BinKind::Lt | BinKind::Le | BinKind::Gt | BinKind::Ge
    )
}

pub(super) fn is_ordering(kind: BinKind) -> bool {
    matches!(kind, BinKind::Lt | BinKind::Le | BinKind::Gt | BinKind::Ge)
}

fn is_bitwise(kind: BinKind) -> bool {
    matches!(
        kind,
        BinKind::BitAnd | BinKind::BitOr | BinKind::BitXor | BinKind::Shl | BinKind::Shr
    )
}

pub(super) fn binary_kernel(
    _ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    binary(spec, &operands[0], &operands[1])
}

/// The in-place accumulate implementation for an op, if it has one: `left`
/// selects which side of the operator the owned operand sits on.
///
/// Only ops whose scalar function is a genuine elementwise binary map over two
/// same-dtype operands qualify — writing the result into one of them is then
/// bit-identical to allocating (element `i` reads element `i` and nothing else):
///
///  * `add`, `subtract`, `multiply`, `divide` — plain `apply_bin` arms, with the
///    integer div-by-zero check run BEFORE any element is written;
///  * `maximum`, `minimum` — NaN-propagating elementwise comparisons.
///
/// Deliberately excluded: `power` (its scalar-exponent path may return the
/// operand buffer itself as its identity result), `remainder` (the floormod arm
/// plus its zero-divisor pre-scan), the bitwise/logical ops (their compute dtype
/// is `bool`, not the result dtype) and every comparison (result dtype `bool`).
pub fn in_place_kernel(name: &str, left: bool) -> Option<InPlaceFn> {
    match name {
        "add" | "subtract" | "multiply" | "divide" | "maximum" | "minimum" => Some(if left {
            in_place_left
        } else {
            in_place_right
        }),
        _ => None,
    }
}

/// `out[i] = f(out[i], other[i])`: the dying operand is the LEFT operand.
fn in_place_left(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    target: Buffer,
    other: &Value,
) -> Result<Vec<Value>> {
    let _ = ctx;
    in_place_binary(spec, target, other, true)
}

/// `out[i] = f(other[i], out[i])`: the dying operand is the RIGHT operand.
fn in_place_right(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    target: Buffer,
    other: &Value,
) -> Result<Vec<Value>> {
    let _ = ctx;
    in_place_binary(spec, target, other, false)
}

/// In-place `out[i] = f(left[i], right[i])`, with `out` the dying operand's own
/// buffer and `left_side` saying which operand it was.
///
/// The plan proved the declared types; this re-checks the RUNTIME buffers (a
/// symbolic dim only resolves now) and falls back to the allocating path when
/// anything differs, so a mis-marked step costs a copy at worst, never a wrong
/// result.  A binary elementwise op needs no run context at all.
fn in_place_binary(
    spec: &OpSpec,
    mut target: Buffer,
    other: &Value,
    left_side: bool,
) -> Result<Vec<Value>> {
    let name = spec.name.as_str();
    let kind =
        bin_kind(name).ok_or_else(|| Error::verification(format!("unexpected binary op '{name}'")))?;
    let out_dtype = spec.result_types[0].dtype;
    let flat = target.dtype == out_dtype
        && other.dtype == out_dtype
        && target.shape == other.shape
        && target.storage.len() == other.storage.len();
    if !flat {
        // Exactly what the allocating kernel computes from the same operands.
        let taken = Rc::new(target);
        let (lhs, rhs) = if left_side {
            (taken, Rc::clone(other))
        } else {
            (Rc::clone(other), taken)
        };
        return binary(spec, &lhs, &rhs);
    }
    // The same rejections as [`binary`], before a single element is written.
    validate_binary(name, kind, out_dtype)?;
    if out_dtype.is_int() && matches!(kind, BinKind::Div | BinKind::Rem) {
        // The divisor is the RIGHT operand, which is the owned one when the
        // dying operand sat on the right — so the scan has to follow the side.
        reject_zero_divisor(name, if left_side { other } else { &target })?;
    }
    let storage = with_same_type_mut!(target, other, |values, others| {
        accumulate_typed(values, others, kind, left_side)
    });
    let out = Buffer::new(out_dtype, target.shape.clone(), storage)?;
    single(out)
}

/// `values[i] = f(values[i], others[i])` (left side) or `f(others[i], values[i])`
/// (right side), in place, returning the (then empty) vector as storage.
///
/// `kind` is dispatched ONCE, exactly like the allocating drivers
/// ([`dense_binary`]): a `match` inside the loop keeps the vectorizer from
/// issuing a single-instruction body, and an in-place map that cannot vectorize
/// is slower than the allocating one it replaces (measured: the per-element
/// `apply_bin` dispatch cost `ew_chain_1e6` 1.99 → 1.50 ms and `gp_gather_4096`
/// 0.90 → 0.55 ms once each arm folded to one machine operation).
fn accumulate_typed<T: Num>(values: &mut Vec<T>, others: &[T], kind: BinKind, left: bool) -> Storage {
    macro_rules! arm {
        ($($variant:ident),* $(,)?) => {
            match kind {
                $(BinKind::$variant => accumulate_as::<T, _>(values, others, left, |a, b| {
                    apply_bin(BinKind::$variant, a, b)
                }),)*
                _ => accumulate_as::<T, _>(values, others, left, |a, b| apply_bin(kind, a, b)),
            }
        };
    }
    arm!(Add, Sub, Mul, Div, Max, Min);
    T::into_storage(std::mem::take(values))
}

/// The shared body of [`accumulate_typed`]'s arms: `f` is a monomorphized
/// per-element closure, so it must be `Copy` (every arm is) and is passed by
/// value into the map — the same shape as [`map2`]'s closure.
fn accumulate_as<T: Num, F: Fn(T, T) -> T + Copy + Sync>(
    values: &mut [T],
    others: &[T],
    left: bool,
    f: F,
) {
    if left {
        zip2_accumulate(values, others, f);
    } else {
        zip2_accumulate(values, others, move |value, other| f(other, value));
    }
}

/// The `power` specialisation's tests live in [`crate::kernels::pow`] and reach this
/// entry point, so it is crate-visible.
pub(crate) fn binary(spec: &OpSpec, lhs: &Value, rhs: &Value) -> Result<Vec<Value>> {
    // Shared, immutable values: `power(x, 1.0)` hands `lhs` back as its result
    // rather than copying it.
    let (a, b) = (&**lhs, &**rhs);
    let name = spec.name.as_str();
    let kind = bin_kind(name)
        .ok_or_else(|| Error::verification(format!("unexpected binary op '{name}'")))?;
    let out_dtype = spec.result_types[0].dtype;
    let out_shape = broadcast_shape(&[&a.shape, &b.shape]).map_err(|err| {
        Error::shape(format!(
            "op '{name}': {} (operand shapes {}, {})",
            err.message,
            shape_text(&a.shape),
            shape_text(&b.shape)
        ))
    })?;
    if is_comparison(kind) {
        return compare(name, kind, a, b, &out_shape);
    }
    // Logical ops compute on bool; every other arithmetic op computes in the
    // declared result dtype.
    let compute = if name == "logical_and" || name == "logical_or" {
        DType::Bool
    } else {
        out_dtype
    };
    let left_owned = if a.dtype == compute {
        None
    } else {
        Some(a.cast(compute)?)
    };
    let right_owned = if b.dtype == compute {
        None
    } else {
        Some(b.cast(compute)?)
    };
    let left = left_owned.as_ref().unwrap_or(a);
    let right = right_owned.as_ref().unwrap_or(b);
    validate_binary(name, kind, compute)?;
    if compute.is_int() && matches!(kind, BinKind::Div | BinKind::Rem) {
        reject_zero_divisor(name, right)?;
    }
    if compute.is_int() && kind == BinKind::Pow {
        pow::reject_negative_exponent(name, right)?;
    }
    if kind == BinKind::Pow {
        // A size-one exponent is known once per call, so `power` dispatches on
        // its value instead of paying libm `powf` per element; the whole
        // specialisation (the `left_owned.is_none()` identity arm included) is
        // in `kernels::pow`.
        if let Some(out) = pow::scalar_power(&out_shape, left, right, left_owned.is_none(), lhs)? {
            return Ok(out);
        }
    }
    with_same_type!(left, right, |x, y| {
        typed_binary(x, y, kind, &out_shape, &left.shape, &right.shape, compute)
            .and_then(crate::exec::single)
    })
}

fn reject_zero_divisor(name: &str, divisor: &Buffer) -> Result<()> {
    macro_rules! scan {
        ($values:expr) => {
            $values.iter().any(|value| *value == 0)
        };
    }
    let has_zero = match &divisor.storage {
        Storage::I8(v) => scan!(v),
        Storage::I16(v) => scan!(v),
        Storage::I32(v) => scan!(v),
        Storage::I64(v) => scan!(v),
        Storage::U8(v) => scan!(v),
        Storage::U16(v) => scan!(v),
        Storage::U32(v) => scan!(v),
        Storage::U64(v) => scan!(v),
        _ => false,
    };
    if has_zero {
        return Err(Error::backend(format!(
            "op '{name}': integer division by zero"
        )));
    }
    Ok(())
}


fn validate_binary(name: &str, kind: BinKind, dtype: DType) -> Result<()> {
    if dtype.is_complex() && (matches!(kind, BinKind::Max | BinKind::Min | BinKind::Rem) || is_bitwise(kind)) {
        return Err(Error::backend(format!(
            "op '{name}': this operation is not defined for complex numbers"
        )));
    }
    if dtype.is_float() && is_bitwise(kind) {
        return Err(Error::backend(format!(
            "op '{name}': bitwise operations require an integer or bool dtype"
        )));
    }
    if dtype == DType::Bool
        && matches!(
            kind,
            BinKind::Sub | BinKind::Div | BinKind::Rem | BinKind::Pow
        )
    {
        return Err(Error::backend(format!(
            "op '{name}': this operation is not defined for bool"
        )));
    }
    Ok(())
}

fn apply_bin<T: Num>(kind: BinKind, left: T, right: T) -> T {
    match kind {
        BinKind::Add => left.add(right),
        BinKind::Sub => left.sub(right),
        BinKind::Mul => left.mul(right),
        BinKind::Div => left.div(right),
        BinKind::Pow => left.pow(right),
        BinKind::Rem => left.rem(right),
        BinKind::Max => left.max(right),
        BinKind::Min => left.min(right),
        BinKind::BitAnd => left.bit_and(right),
        BinKind::BitOr => left.bit_or(right),
        BinKind::BitXor => left.bit_xor(right),
        BinKind::Shl => left.shl(right),
        BinKind::Shr => left.shr(right),
        BinKind::Eq
        | BinKind::Ne
        | BinKind::Lt
        | BinKind::Le
        | BinKind::Gt
        | BinKind::Ge => left,
    }
}

fn typed_binary<T: Num>(
    left: &[T],
    right: &[T],
    kind: BinKind,
    out_shape: &[usize],
    left_shape: &[usize],
    right_shape: &[usize],
    out_dtype: DType,
) -> Result<Buffer> {
    match (
        layout(out_shape, left, left_shape),
        layout(out_shape, right, right_shape),
    ) {
        (Layout::Flat, Layout::Flat) => dense_binary(kind, out_dtype, out_shape, left, right),
        (Layout::Flat, Layout::Scalar) if !right.is_empty() => {
            dense_binary_right_scalar(kind, out_dtype, out_shape, left, right[0])
        }
        (Layout::Scalar, Layout::Flat) if !left.is_empty() => {
            dense_binary_left_scalar(kind, out_dtype, out_shape, left[0], right)
        }
        _ => {
            let mut out = T::zeros(util::numel(out_shape));
            apply2(&mut out, out_shape, left, left_shape, right, right_shape, |a, b| {
                apply_bin(kind, a, b)
            });
            Buffer::new(out_dtype, out_shape.to_vec(), T::into_storage(out))
        }
    }
}

/// A broadcast-free operand layout, cheapest first: exactly the result shape, a
/// single element (the `x * 2.0` / `x + bias` form, which is most elementwise
/// ops in a real graph), or anything the general broadcast driver owns.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Layout {
    Flat,
    Scalar,
    Broadcast,
}

pub(super) fn layout<T>(out_shape: &[usize], values: &[T], values_shape: &[usize]) -> Layout {
    if values_shape == out_shape && values.len() == util::numel(out_shape) {
        // `out` always holds exactly `numel(out_shape)` elements, so the length
        // compare also rules out a short slice reaching a driver that would
        // silently stop early.
        Layout::Flat
    } else if values_shape.is_empty() && values.len() == 1 {
        Layout::Scalar
    } else {
        Layout::Broadcast
    }
}

/// The broadcast-free binary fast paths, generated from one op list so the
/// scalar forms cannot drift from the elementwise one.  `kind` is dispatched
/// ONCE in each: every arm is its own flat map whose closure folds to a single
/// machine operation, which is what the vectorizer needs (a `match` inside the
/// loop blocks it), and the result storage is built from the mapped iterator
/// rather than zero-filled first.  Semantics are `apply_bin`'s, arm for arm.
macro_rules! dense_binary_drivers {
    ($($kind:ident),* $(,)?) => {
        fn dense_binary<T: Num>(
            kind: BinKind,
            dtype: DType,
            out_shape: &[usize],
            left: &[T],
            right: &[T],
        ) -> Result<Buffer> {
            match kind {
                $(BinKind::$kind => map2(dtype, out_shape, left, right, |a, b| apply_bin(BinKind::$kind, a, b)),)*
                // Comparisons never reach the arithmetic path (`compare` owns
                // them); the arm keeps the match exhaustive and answers exactly
                // what the general driver would.
                _ => map2(dtype, out_shape, left, right, |a, b| apply_bin(kind, a, b)),
            }
        }

        fn dense_binary_left_scalar<T: Num>(
            kind: BinKind,
            dtype: DType,
            out_shape: &[usize],
            left: T,
            right: &[T],
        ) -> Result<Buffer> {
            match kind {
                $(BinKind::$kind => map1(dtype, out_shape, right, |b| apply_bin(BinKind::$kind, left, b)),)*
                _ => map1(dtype, out_shape, right, |b| apply_bin(kind, left, b)),
            }
        }

        fn dense_binary_right_scalar<T: Num>(
            kind: BinKind,
            dtype: DType,
            out_shape: &[usize],
            left: &[T],
            right: T,
        ) -> Result<Buffer> {
            match kind {
                $(BinKind::$kind => map1(dtype, out_shape, left, |a| apply_bin(BinKind::$kind, a, right)),)*
                _ => map1(dtype, out_shape, left, |a| apply_bin(kind, a, right)),
            }
        }
    };
}

dense_binary_drivers!(
    Add, Sub, Mul, Div, Pow, Rem, Max, Min, BitAnd, BitOr, BitXor, Shl, Shr
);

#[cfg(test)]
mod tests {
    use super::*;
    use evoxc_core::{Dim, ValueType};

    use crate::kernels::elementwise::test_support::{binary, buffer, spec};

    #[test]
    fn only_the_safe_binary_ops_have_an_in_place_implementation() {
        for name in ["add", "subtract", "multiply", "divide", "maximum", "minimum"] {
            assert!(in_place_kernel(name, true).is_some(), "{name} (left)");
            assert!(in_place_kernel(name, false).is_some(), "{name} (right)");
        }
        // `power` may hand its base buffer back as its identity result, so it is
        // never an in-place target; `remainder` carries its own zero-divisor
        // pre-scan; the bitwise/logical ops and every comparison compute in a
        // dtype that is not the operand's.
        for name in [
            "power",
            "remainder",
            "bitwise_and",
            "logical_and",
            "equal",
            "less",
            "abs",
            "select",
            "reduce_sum",
        ] {
            assert!(in_place_kernel(name, true).is_none(), "{name} must allocate");
        }
    }

    #[test]
    fn in_place_accumulation_is_bit_identical_to_the_allocating_path() {
        // Every whitelisted op, both sides, at a size that takes the chunked
        // driver as well as the one-thread one — compared against the very same
        // scalar function through the allocating `binary` kernel.  `(1 << 19) + 5`
        // reaches four chunks with a SHORT last one, so this also pins that the
        // in-place driver's operand windows line up with its slot windows (the
        // arm dispatch is what makes the loop vectorizable in the first place).
        for &count in &[1usize, 9, (1 << 18) + 5, (1 << 19) + 5] {
            let left: Vec<f32> = (0..count).map(|index| 0.25 + index as f32 * 0.5).collect();
            let right: Vec<f32> = (0..count).map(|index| 1.5 - index as f32 * 0.125).collect();
            for name in ["add", "subtract", "multiply", "divide", "maximum", "minimum"] {
                let op = spec(name, ValueType::of(DType::Float32, [Dim::int(count as i64)]));
                let a = buffer(DType::Float32, &[count], Storage::F32(left.clone()));
                let b = buffer(DType::Float32, &[count], Storage::F32(right.clone()));
                let expected = binary(&op, &a, &b).expect(name);
                for left_side in [true, false] {
                    let target = if left_side { a.clone() } else { b.clone() };
                    let other = std::rc::Rc::new(if left_side { b.clone() } else { a.clone() });
                    let got = in_place_binary(&op, target, &other, left_side)
                        .unwrap_or_else(|err| panic!("{name}: {}", err.message));
                    assert_eq!(
                        got[0].storage, expected[0].storage,
                        "{name}: in-place ({left_side}) must be bit-identical"
                    );
                }
            }
        }
        // NaN propagation is per element, exactly as `apply_bin` defines it.
        let op = spec("maximum", ValueType::of(DType::Float32, [Dim::int(3)]));
        let target = buffer(
            DType::Float32,
            &[3],
            Storage::F32(vec![f32::NAN, 2.0, -1.0]),
        );
        let other = std::rc::Rc::new(buffer(
            DType::Float32,
            &[3],
            Storage::F32(vec![1.0, f32::NAN, -2.0]),
        ));
        let got = in_place_binary(&op, target, &other, true).expect("maximum accumulates");
        match &got[0].storage {
            Storage::F32(values) => {
                assert!(values[0].is_nan());
                assert!(values[1].is_nan());
                assert_eq!(values[2], -1.0);
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }

    #[test]
    fn an_integer_operand_pair_accumulates_in_place() {
        // Integers keep numpy's wrapping semantics through the same `apply_bin`
        // arm, and the div-by-zero guard runs before a single element is written.
        let op = spec("add", ValueType::of(DType::Int32, [Dim::int(2)]));
        let target = buffer(DType::Int32, &[2], Storage::I32(vec![i32::MAX, -3]));
        let other = std::rc::Rc::new(buffer(DType::Int32, &[2], Storage::I32(vec![1, 10])));
        let got = in_place_binary(&op, target, &other, true).expect("integer add accumulates");
        assert_eq!(got[0].storage, Storage::I32(vec![i32::MIN, 7]));

        let divide = spec("divide", ValueType::of(DType::Int32, [Dim::int(2)]));
        let target = buffer(DType::Int32, &[2], Storage::I32(vec![7, 9]));
        let other = std::rc::Rc::new(buffer(DType::Int32, &[2], Storage::I32(vec![2, 3])));
        let got = in_place_binary(&divide, target, &other, true).expect("integer divide");
        assert_eq!(got[0].storage, Storage::I32(vec![3, 3]));
        // A zero divisor is the allocating path's explicit error, unchanged.
        let target = buffer(DType::Int32, &[2], Storage::I32(vec![7, 9]));
        let other = std::rc::Rc::new(buffer(DType::Int32, &[2], Storage::I32(vec![0, 3])));
        let err = in_place_binary(&divide, target, &other, true).expect_err("div by zero fails");
        assert!(err.message.contains("integer division by zero"), "{}", err.message);
        // The divisor is the RIGHT operand, so when the OWNED operand is the
        // right one the target itself has to be scanned before it is
        // overwritten — otherwise a zero in it would slip past the guard.
        let target = buffer(DType::Int32, &[2], Storage::I32(vec![0, 3]));
        let other = std::rc::Rc::new(buffer(DType::Int32, &[2], Storage::I32(vec![7, 9])));
        let err = in_place_binary(&divide, target, &other, false)
            .expect_err("a zero divisor on the right fails");
        assert!(err.message.contains("integer division by zero"), "{}", err.message);
        let target = buffer(DType::Int32, &[2], Storage::I32(vec![2, 3]));
        let other = std::rc::Rc::new(buffer(DType::Int32, &[2], Storage::I32(vec![7, 10])));
        let got = in_place_binary(&divide, target, &other, false).expect("right-side integer divide");
        assert_eq!(got[0].storage, Storage::I32(vec![3, 3]));
    }

    #[test]
    fn a_non_flat_or_heterogeneous_pair_falls_back_to_the_allocating_path() {
        // The plan only marks broadcast-free same-dtype pairs, but a symbolic
        // shape resolves at run time: a mismatched buffer must produce the
        // broadcast result, never a truncated in-place write.
        let op = spec("add", ValueType::of(DType::Float32, [Dim::int(2), Dim::int(2)]));
        let target = buffer(DType::Float32, &[2, 2], Storage::F32(vec![1.0, 2.0, 3.0, 4.0]));
        let other = std::rc::Rc::new(buffer(
            DType::Float32,
            &[1, 2],
            Storage::F32(vec![10.0, 20.0]),
        ));
        let got = in_place_binary(&op, target, &other, true).expect("the broadcast fallback runs");
        assert_eq!(
            got[0].storage,
            Storage::F32(vec![11.0, 22.0, 13.0, 24.0])
        );
        // Same for a dtype the plan cannot have proven: the allocating path
        // casts exactly as it always did.
        let mixed = spec("add", ValueType::of(DType::Int32, [Dim::int(2)]));
        let target = buffer(DType::Float32, &[2], Storage::F32(vec![1.5, 2.5]));
        let other = std::rc::Rc::new(buffer(DType::Int32, &[2], Storage::I32(vec![1, 1])));
        let got = in_place_binary(&mixed, target, &other, true).expect("the cast fallback runs");
        assert_eq!(got[0].storage, Storage::I32(vec![2, 3]));
    }

    #[test]
    fn add_broadcasts_and_computes_in_result_dtype() {
        let a = buffer(DType::Int32, &[2, 1], Storage::I32(vec![1, 2]));
        let b = buffer(DType::Int32, &[1, 2], Storage::I32(vec![10, 20]));
        let op = spec(
            "add",
            ValueType::of(DType::Int32, [Dim::int(2), Dim::int(2)]),
        );
        let out = binary(&op, &a, &b).unwrap();
        assert_eq!(out[0].storage, Storage::I32(vec![11, 21, 12, 22]));
    }

    #[test]
    fn divide_promotes_integers_to_the_declared_float_dtype() {
        let a = buffer(DType::Int32, &[2], Storage::I32(vec![1, 3]));
        let b = buffer(DType::Int32, &[2], Storage::I32(vec![2, 2]));
        let op = spec("divide", ValueType::of(DType::Float64, [Dim::int(2)]));
        let out = binary(&op, &a, &b).unwrap();
        assert_eq!(out[0].storage, Storage::F64(vec![0.5, 1.5]));
    }

    #[test]
    fn remainder_is_floormod() {
        let a = buffer(DType::Int64, &[2], Storage::I64(vec![-7, 7]));
        let b = buffer(DType::Int64, &[2], Storage::I64(vec![3, -3]));
        let op = spec("remainder", ValueType::of(DType::Int64, [Dim::int(2)]));
        let out = binary(&op, &a, &b).unwrap();
        assert_eq!(out[0].storage, Storage::I64(vec![2, -2]));
    }

    #[test]
    fn layout_classifies_flat_scalar_and_broadcast_operands() {
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(layout(&[2, 3], &values, &[2, 3]), Layout::Flat);
        assert_eq!(layout(&[2, 3], &values[..1], &[]), Layout::Scalar);
        assert_eq!(layout(&[2, 3], &values, &[2, 1]), Layout::Broadcast);
        assert_eq!(layout(&[2, 3], &values, &[3, 2]), Layout::Broadcast);
        // A rank-0 result is flat against a rank-0 operand, not a broadcast.
        assert_eq!(layout(&[], &values[..1], &[]), Layout::Flat);
    }

    #[test]
    fn flat_binary_and_comparison_agree_with_the_broadcast_path() {
        // Same values, once as a flat (2, 3) pair (fast path) and once as a
        // (1, 3) row broadcast over the same rows (general path).
        let flat_left = buffer(
            DType::Float32,
            &[2, 3],
            Storage::F32(vec![1., -2., 3., -4., 5., 0.]),
        );
        let flat_right = buffer(
            DType::Float32,
            &[2, 3],
            Storage::F32(vec![2., 2., -1., 2., 2., -1.]),
        );
        let row_right = buffer(DType::Float32, &[1, 3], Storage::F32(vec![2., 2., -1.]));
        let column_left = buffer(DType::Float32, &[2, 1], Storage::F32(vec![1., -4.]));
        let cases: [(&str, DType); 13] = [
            ("add", DType::Float32),
            ("subtract", DType::Float32),
            ("multiply", DType::Float32),
            ("divide", DType::Float32),
            ("remainder", DType::Float32),
            ("maximum", DType::Float32),
            ("minimum", DType::Float32),
            ("equal", DType::Bool),
            ("not_equal", DType::Bool),
            ("less", DType::Bool),
            ("less_equal", DType::Bool),
            ("greater", DType::Bool),
            ("greater_equal", DType::Bool),
        ];
        let out = [Dim::int(2), Dim::int(3)];
        for (name, dtype) in cases {
            let op = spec(name, ValueType::of(dtype, out.clone()));
            let flat = binary(&op, &flat_left, &flat_right).unwrap();
            let broadcast = binary(&op, &flat_left, &row_right).unwrap();
            assert_eq!(flat[0].storage, broadcast[0].storage, "op {name}");
            // A rank-1 (2, 1) column broadcasts in the general driver.
            let op = spec(name, ValueType::of(dtype, out.clone()));
            let column = binary(&op, &column_left, &flat_right).unwrap();
            let repeated = buffer(
                DType::Float32,
                &[2, 3],
                Storage::F32(vec![1., 1., 1., -4., -4., -4.]),
            );
            let general = binary(&op, &repeated, &flat_right).unwrap();
            assert_eq!(column[0].storage, general[0].storage, "op {name}: column");
        }
        // Spot-check one value of the flat path so the comparison is not vacuous.
        let op = spec("add", ValueType::of(DType::Float32, out));
        assert_eq!(
            binary(&op, &flat_left, &flat_right).unwrap()[0].storage,
            Storage::F32(vec![3., 0., 2., -2., 7., -1.])
        );
    }

    #[test]
    fn scalar_operand_agrees_with_a_size_one_operand() {
        // `x * 2.0` takes the hoisted-scalar loop; `x * [2.0]` (shape (1,)) is
        // the general broadcast driver.  Same numbers.
        let x = buffer(DType::Float32, &[2, 2], Storage::F32(vec![1., -2., 3., 0.5]));
        let scalar = buffer(DType::Float32, &[], Storage::F32(vec![2.0]));
        let vector = buffer(DType::Float32, &[1], Storage::F32(vec![2.0]));
        let op = spec("multiply", ValueType::of(DType::Float32, [Dim::int(2), Dim::int(2)]));
        let hoisted = binary(&op, &x, &scalar).unwrap();
        let broadcast = binary(&op, &x, &vector).unwrap();
        assert_eq!(hoisted[0].storage, broadcast[0].storage);
        assert_eq!(hoisted[0].storage, Storage::F32(vec![2., -4., 6., 1.]));

        // The scalar-fast-path is implemented for the left operand too.
        let op = spec("subtract", ValueType::of(DType::Float32, [Dim::int(2), Dim::int(2)]));
        let hoisted = binary(&op, &scalar, &x).unwrap();
        let broadcast = binary(&op, &vector, &x).unwrap();
        assert_eq!(hoisted[0].storage, broadcast[0].storage);
        assert_eq!(hoisted[0].storage, Storage::F32(vec![1., 4., -1., 1.5]));

        // Comparisons have the same two shapes.
        let zero = buffer(DType::Float32, &[], Storage::F32(vec![0.0]));
        let zeros = buffer(DType::Float32, &[1], Storage::F32(vec![0.0]));
        let op = spec("greater", ValueType::of(DType::Bool, [Dim::int(2), Dim::int(2)]));
        assert_eq!(
            binary(&op, &x, &zero).unwrap()[0].storage,
            binary(&op, &x, &zeros).unwrap()[0].storage
        );
        assert_eq!(binary(&op, &x, &zero).unwrap()[0].storage, Storage::Bool(vec![true, false, true, true]));
    }

}
