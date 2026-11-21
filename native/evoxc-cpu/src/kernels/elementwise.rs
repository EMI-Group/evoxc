//! Elementwise, comparison, cast/select/broadcast and rounding kernels.
//!
//! All arithmetic is executed in the op's DECLARED result dtype (the frontend
//! already inserted compensation casts); operands are converted only when
//! their dtype differs from it, so the common same-dtype case is a flat
//! broadcast loop with no allocation beyond the result.  The two exception
//! shapes are comparisons (computed in the promoted INPUT dtype, producing
//! `bool`) and `abs` of a complex value (magnitude, a real result).
use evoxc_core::{AttrValue, DType, Error, Result};
use half::f16;
use num_complex::Complex;
use serde_json::Value as Json;

use crate::buffer::{
    apply2, apply3, broadcast_shape, broadcast_to, shape_text, Buffer, Storage,
};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::kernels::KernelMap;
use crate::plan::OpSpec;
use crate::scalar::{Num, Scalar};
use crate::{with_same_type, with_type};

pub fn register(map: &mut KernelMap) {
    for name in [
        "add",
        "subtract",
        "multiply",
        "divide",
        "power",
        "remainder",
        "maximum",
        "minimum",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "bitwise_left_shift",
        "bitwise_right_shift",
        "logical_and",
        "logical_or",
    ] {
        map.insert(name, binary_kernel);
    }
    for name in ["equal", "not_equal", "less", "less_equal", "greater", "greater_equal"] {
        map.insert(name, binary_kernel);
    }
    for name in [
        "abs",
        "acos",
        "negate",
        "square",
        "sqrt",
        "exp",
        "log",
        "log1p",
        "sin",
        "cos",
        "tan",
        "tanh",
        "sigmoid",
        "relu",
        "sign",
        "logical_not",
        "floor",
        "ceil",
        "round",
    ] {
        map.insert(name, unary_kernel);
    }
    map.insert("erf", erf_kernel);
    map.insert("gelu", gelu_kernel);
    map.insert("cast", cast_kernel);
    map.insert("select", select_kernel);
    map.insert("broadcast", broadcast_kernel);
    map.insert("stop_gradient", stop_gradient_kernel);
    map.insert("nan_to_num", nan_to_num_kernel);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinKind {
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

fn is_ordering(kind: BinKind) -> bool {
    matches!(kind, BinKind::Lt | BinKind::Le | BinKind::Gt | BinKind::Ge)
}

fn is_bitwise(kind: BinKind) -> bool {
    matches!(
        kind,
        BinKind::BitAnd | BinKind::BitOr | BinKind::BitXor | BinKind::Shl | BinKind::Shr
    )
}

fn binary_kernel(
    _ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    binary(spec, &operands[0], &operands[1])
}

fn binary(spec: &OpSpec, a: &Buffer, b: &Buffer) -> Result<Vec<Value>> {
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
    validate_binary(name, kind, compute)?;
    if compute.is_int() && matches!(kind, BinKind::Div | BinKind::Rem) {
        reject_zero_divisor(name, right)?;
    }
    if compute.is_int() && kind == BinKind::Pow {
        reject_negative_exponent(name, right)?;
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

fn reject_negative_exponent(name: &str, exponent: &Buffer) -> Result<()> {
    let negative = match &exponent.storage {
        Storage::I8(v) => v.iter().any(|value| *value < 0),
        Storage::I16(v) => v.iter().any(|value| *value < 0),
        Storage::I32(v) => v.iter().any(|value| *value < 0),
        Storage::I64(v) => v.iter().any(|value| *value < 0),
        _ => false,
    };
    if negative {
        return Err(Error::backend(format!(
            "op '{name}': negative integer exponent is not supported"
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
    let mut out = T::zeros(util::numel(out_shape));
    apply2(&mut out, out_shape, left, left_shape, right, right_shape, |a, b| {
        apply_bin(kind, a, b)
    });
    Buffer::new(out_dtype, out_shape.to_vec(), T::into_storage(out))
}

fn compare(
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
    if compute.is_complex() {
        with_same_type!(left, right, |x, y| {
            compare_eq(x, &left.shape, y, &right.shape, kind, out_shape)
        })
    } else {
        with_same_type!(left, right, |x, y| {
            compare_ord(x, &left.shape, y, &right.shape, kind, out_shape)
        })
    }
}

/// Ordering comparisons, without requiring `PartialOrd` (complex has none).
trait Orderable: PartialEq {
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

fn compare_eq<T: PartialEq + Copy>(
    left: &[T],
    left_shape: &[usize],
    right: &[T],
    right_shape: &[usize],
    kind: BinKind,
    out_shape: &[usize],
) -> Result<Vec<Value>> {
    let mut out = vec![false; util::numel(out_shape)];
    apply2(&mut out, out_shape, left, left_shape, right, right_shape, |p, q| {
        match kind {
            BinKind::Eq => p == q,
            BinKind::Ne => p != q,
            _ => false,
        }
    });
    single(Buffer::new(DType::Bool, out_shape.to_vec(), Storage::Bool(out))?)
}

fn compare_ord<T: Orderable + Copy>(
    left: &[T],
    left_shape: &[usize],
    right: &[T],
    right_shape: &[usize],
    kind: BinKind,
    out_shape: &[usize],
) -> Result<Vec<Value>> {
    let mut out = vec![false; util::numel(out_shape)];
    apply2(&mut out, out_shape, left, left_shape, right, right_shape, |p, q| {
        match kind {
            BinKind::Eq => p == q,
            BinKind::Ne => p != q,
            BinKind::Lt => p.lt(&q),
            BinKind::Le => p.le(&q),
            BinKind::Gt => p.gt(&q),
            BinKind::Ge => p.ge(&q),
            _ => false,
        }
    });
    single(Buffer::new(DType::Bool, out_shape.to_vec(), Storage::Bool(out))?)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnKind {
    Abs,
    Negate,
    Square,
    Sqrt,
    Exp,
    Log,
    Log1p,
    Sin,
    Cos,
    Tan,
    Acos,
    Tanh,
    Sigmoid,
    Relu,
    Sign,
    LogicalNot,
    Floor,
    Ceil,
    Round,
}

fn un_kind(name: &str) -> Option<UnKind> {
    Some(match name {
        "abs" => UnKind::Abs,
        "negate" => UnKind::Negate,
        "square" => UnKind::Square,
        "sqrt" => UnKind::Sqrt,
        "exp" => UnKind::Exp,
        "log" => UnKind::Log,
        "log1p" => UnKind::Log1p,
        "sin" => UnKind::Sin,
        "cos" => UnKind::Cos,
        "tan" => UnKind::Tan,
        "acos" => UnKind::Acos,
        "tanh" => UnKind::Tanh,
        "sigmoid" => UnKind::Sigmoid,
        "relu" => UnKind::Relu,
        "sign" => UnKind::Sign,
        "logical_not" => UnKind::LogicalNot,
        "floor" => UnKind::Floor,
        "ceil" => UnKind::Ceil,
        "round" => UnKind::Round,
        _ => return None,
    })
}

fn unary_kernel(
    _ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    unary(spec, &operands[0])
}

fn unary(spec: &OpSpec, input: &Buffer) -> Result<Vec<Value>> {
    let name = spec.name.as_str();
    let kind =
        un_kind(name).ok_or_else(|| Error::verification(format!("unexpected unary op '{name}'")))?;
    let out_dtype = spec.result_types[0].dtype;
    if kind == UnKind::Abs && input.dtype.is_complex() {
        return abs_complex(input, out_dtype);
    }
    let compute = if kind == UnKind::LogicalNot {
        DType::Bool
    } else {
        out_dtype
    };
    validate_unary(name, kind, compute)?;
    let owned = if input.dtype == compute {
        None
    } else {
        Some(input.cast(compute)?)
    };
    let source = owned.as_ref().unwrap_or(input);
    with_type!(source, |values| {
        typed_unary(values, kind, compute, &source.shape)
    })
}

fn validate_unary(name: &str, kind: UnKind, dtype: DType) -> Result<()> {
    if dtype.is_complex() && matches!(kind, UnKind::Floor | UnKind::Ceil | UnKind::Round) {
        return Err(Error::backend(format!(
            "op '{name}': complex input is not supported (numpy defines no rounding for complex numbers)"
        )));
    }
    if dtype == DType::Bool && matches!(kind, UnKind::Negate) {
        return Err(Error::backend(format!(
            "op '{name}': this operation is not defined for bool"
        )));
    }
    Ok(())
}

fn apply_un<T: Num>(kind: UnKind, value: T) -> T {
    match kind {
        UnKind::Abs => value.abs(),
        UnKind::Negate => value.neg(),
        UnKind::Square => value.mul(value),
        UnKind::Sqrt => value.sqrt(),
        UnKind::Exp => value.exp(),
        UnKind::Log => value.ln(),
        UnKind::Log1p => value.log1p(),
        UnKind::Sin => value.sin(),
        UnKind::Cos => value.cos(),
        UnKind::Tan => value.tan(),
        UnKind::Acos => {
            // acos(x) = pi/2 - asin(x); via the complex-safe identity used by
            // the numpy kernel's real-only path.
            let one = T::one();
            let value = value.div(one.sub(value.mul(value)).sqrt());
            one.sub(value.mul(value)).sqrt().sub(value).ln().neg()
        }
        UnKind::Tanh => value.tanh(),
        UnKind::Sigmoid => {
            let one = T::one();
            one.div(one.add(value.neg().exp()))
        }
        UnKind::Relu => value.max(T::zero()),
        UnKind::Sign => value.sign(),
        UnKind::LogicalNot => value, // unreachable: handled on bool below
        UnKind::Floor => value.floor(),
        UnKind::Ceil => value.ceil(),
        UnKind::Round => value.round_even(),
    }
}

fn typed_unary<T: Num>(
    values: &[T],
    kind: UnKind,
    out_dtype: DType,
    shape: &[usize],
) -> Result<Vec<Value>> {
    if kind == UnKind::LogicalNot {
        let out: Vec<bool> = values.iter().map(|value| !is_truthy(*value)).collect();
        return single(Buffer::new(DType::Bool, shape.to_vec(), Storage::Bool(out))?);
    }
    let mut out = T::zeros(values.len());
    for (slot, value) in out.iter_mut().zip(values.iter()) {
        *slot = apply_un(kind, *value);
    }
    single(Buffer::new(out_dtype, shape.to_vec(), T::into_storage(out))?)
}

fn is_truthy<T: Scalar>(value: T) -> bool {
    value.to_scalar().truthy()
}

fn abs_complex(input: &Buffer, out_dtype: DType) -> Result<Vec<Value>> {
    match (&input.storage, out_dtype) {
        (Storage::C64(values), DType::Float32) => {
            let out: Vec<f32> = values.iter().map(|value| value.norm()).collect();
            single(Buffer::new(DType::Float32, input.shape.clone(), Storage::F32(out))?)
        }
        (Storage::C128(values), DType::Float64) => {
            let out: Vec<f64> = values.iter().map(|value| value.norm()).collect();
            single(Buffer::new(DType::Float64, input.shape.clone(), Storage::F64(out))?)
        }
        _ => Err(Error::dtype(format!(
            "op 'abs': complex input cannot produce a {} result",
            out_dtype.name()
        ))),
    }
}

fn stop_gradient_kernel(
    _ctx: &mut RunCtx<'_>,
    _spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    Ok(vec![operands[0].clone()])
}

fn cast_kernel(_ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let target = spec.attrs.dtype("dtype")?;
    let input = &operands[0];
    if input.dtype == target {
        return Ok(vec![operands[0].clone()]);
    }
    single(input.cast(target)?)
}

fn broadcast_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    let target = spec.attrs.shape("shape")?;
    let shape = ctx.eval_shape_strict(target).map_err(|err| {
        err.context("op 'broadcast': cannot evaluate the target shape")
    })?;
    let input = &operands[0];
    let out = broadcast_to(input, &shape).map_err(|err| {
        Error::shape(format!(
            "op 'broadcast': operand shape {} cannot be broadcast to target shape {}: {}",
            shape_text(&input.shape),
            shape_text(&shape),
            err.message
        ))
    })?;
    single(out)
}

fn select_kernel(_ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
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
    let total = util::numel(&out_shape);
    // Dispatch over the branch dtype; the predicate is `bool`.
    let storage = match (&on_true.storage, &on_false.storage, &predicate.storage) {
        (Storage::Bool(a), Storage::Bool(b), Storage::Bool(p)) => {
            let mut out = vec![false; total];
            apply3(&mut out, &out_shape, p, &predicate.shape, a, &on_true.shape, b, &on_false.shape, |q, x, y| if q { x } else { y });
            Storage::Bool(out)
        }
        _ => select_numeric(predicate, on_true, on_false, &out_shape)?,
    };
    single(Buffer::new(out_dtype, out_shape, storage)?)
}

fn select_numeric(
    predicate: &Buffer,
    on_true: &Buffer,
    on_false: &Buffer,
    out_shape: &[usize],
) -> Result<Storage> {
    let pred = match &predicate.storage {
        Storage::Bool(values) => values,
        _ => return Err(Error::dtype("op 'select': predicate must be bool")),
    };
    Ok(match (&on_true.storage, &on_false.storage) {
        (Storage::I8(a), Storage::I8(b)) => select_apply(pred, predicate, a, on_true, b, on_false, out_shape),
        (Storage::I16(a), Storage::I16(b)) => select_apply(pred, predicate, a, on_true, b, on_false, out_shape),
        (Storage::I32(a), Storage::I32(b)) => select_apply(pred, predicate, a, on_true, b, on_false, out_shape),
        (Storage::I64(a), Storage::I64(b)) => select_apply(pred, predicate, a, on_true, b, on_false, out_shape),
        (Storage::U8(a), Storage::U8(b)) => select_apply(pred, predicate, a, on_true, b, on_false, out_shape),
        (Storage::U16(a), Storage::U16(b)) => select_apply(pred, predicate, a, on_true, b, on_false, out_shape),
        (Storage::U32(a), Storage::U32(b)) => select_apply(pred, predicate, a, on_true, b, on_false, out_shape),
        (Storage::U64(a), Storage::U64(b)) => select_apply(pred, predicate, a, on_true, b, on_false, out_shape),
        (Storage::F16(a), Storage::F16(b)) => select_apply(pred, predicate, a, on_true, b, on_false, out_shape),
        (Storage::F32(a), Storage::F32(b)) => select_apply(pred, predicate, a, on_true, b, on_false, out_shape),
        (Storage::F64(a), Storage::F64(b)) => select_apply(pred, predicate, a, on_true, b, on_false, out_shape),
        (Storage::C64(a), Storage::C64(b)) => select_apply(pred, predicate, a, on_true, b, on_false, out_shape),
        (Storage::C128(a), Storage::C128(b)) => select_apply(pred, predicate, a, on_true, b, on_false, out_shape),
        _ => {
            return Err(Error::dtype(format!(
                "op 'select': branches have incompatible dtypes {} and {}",
                on_true.dtype.name(),
                on_false.dtype.name()
            )))
        }
    })
}

fn select_apply<T>(
    pred: &[bool],
    predicate: &Buffer,
    a: &[T],
    a_buf: &Buffer,
    b: &[T],
    b_buf: &Buffer,
    out_shape: &[usize],
) -> Storage
where
    T: Copy + Scalar,
{
    let mut out = vec![a[0]; util::numel(out_shape)];
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
    T::into_storage(out)
}

fn erf_kernel(_ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let input = &operands[0];
    util::reject_complex("erf", input.dtype)?;
    let out_dtype = spec.result_types[0].dtype;
    let storage = match (&input.storage, out_dtype) {
        (Storage::F64(values), DType::Float64) => Storage::F64(
            values.iter().map(|value| libm::erf(*value)).collect(),
        ),
        (Storage::F32(values), DType::Float32) => Storage::F32(
            values.iter().map(|value| libm::erff(*value)).collect(),
        ),
        (Storage::F16(values), DType::Float16) => Storage::F16(
            values
                .iter()
                .map(|value| f16::from_f32(libm::erff(value.to_f32())))
                .collect(),
        ),
        _ => {
            let cast = input.cast(out_dtype)?;
            return erf_kernel(_ctx, spec, &[std::rc::Rc::new(cast)]);
        }
    };
    single(Buffer::new(out_dtype, input.shape.clone(), storage)?)
}

fn gelu_kernel(_ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let input = &operands[0];
    util::reject_complex("gelu", input.dtype)?;
    let out_dtype = spec.result_types[0].dtype;
    if !out_dtype.is_float() {
        return Err(Error::backend(format!(
            "op 'gelu': expected a floating dtype, got {}",
            out_dtype.name()
        )));
    }
    let source = if input.dtype == out_dtype {
        None
    } else {
        Some(input.cast(out_dtype)?)
    };
    let source = source.as_ref().unwrap_or(input);
    let sqrt_two = std::f64::consts::SQRT_2;
    let storage = match &source.storage {
        Storage::F64(values) => Storage::F64(
            values
                .iter()
                .map(|value| {
                    0.5 * value * (1.0 + libm::erf(value / sqrt_two))
                })
                .collect(),
        ),
        Storage::F32(values) => Storage::F32(
            values
                .iter()
                .map(|value| {
                    let x = *value as f64;
                    (0.5 * x * (1.0 + libm::erf(x / sqrt_two))) as f32
                })
                .collect(),
        ),
        Storage::F16(values) => Storage::F16(
            values
                .iter()
                .map(|value| {
                    let x = value.to_f64();
                    f16::from_f64(0.5 * x * (1.0 + libm::erf(x / sqrt_two)))
                })
                .collect(),
        ),
        _ => {
            return Err(Error::backend(format!(
                "op 'gelu': unsupported dtype {}",
                source.dtype.name()
            )))
        }
    };
    single(Buffer::new(out_dtype, input.shape.clone(), storage)?)
}

fn nan_to_num_kernel(
    _ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    let input = &operands[0];
    let nan = any_float(&spec.attrs, "nan")?.unwrap_or(0.0);
    let posinf = any_float(&spec.attrs, "posinf")?;
    let neginf = any_float(&spec.attrs, "neginf")?;
    let storage = match &input.storage {
        Storage::F32(values) => Storage::F32(
            values
                .iter()
                .map(|value| fix_f32(*value, nan as f32, posinf, neginf, f32::MAX, f32::MIN))
                .collect(),
        ),
        Storage::F64(values) => Storage::F64(
            values
                .iter()
                .map(|value| fix_f64(*value, nan, posinf, neginf, f64::MAX, f64::MIN))
                .collect(),
        ),
        Storage::F16(values) => Storage::F16(
            values
                .iter()
                .map(|value| {
                    let fixed = fix_f32(
                        value.to_f32(),
                        nan as f32,
                        posinf,
                        neginf,
                        f16::MAX.to_f32(),
                        f16::MIN.to_f32(),
                    );
                    f16::from_f32(fixed)
                })
                .collect(),
        ),
        Storage::C64(values) => Storage::C64(
            values
                .iter()
                .map(|value| {
                    Complex::new(
                        fix_f32(value.re, nan as f32, posinf, neginf, f32::MAX, f32::MIN),
                        fix_f32(value.im, nan as f32, posinf, neginf, f32::MAX, f32::MIN),
                    )
                })
                .collect(),
        ),
        Storage::C128(values) => Storage::C128(
            values
                .iter()
                .map(|value| {
                    Complex::new(
                        fix_f64(value.re, nan, posinf, neginf, f64::MAX, f64::MIN),
                        fix_f64(value.im, nan, posinf, neginf, f64::MAX, f64::MIN),
                    )
                })
                .collect(),
        ),
        _ => return Ok(vec![operands[0].clone()]),
    };
    single(Buffer::new(input.dtype, input.shape.clone(), storage)?)
}

fn fix_f32(value: f32, nan: f32, posinf: Option<f64>, neginf: Option<f64>, max: f32, min: f32) -> f32 {
    if value.is_nan() {
        nan
    } else if value == f32::INFINITY {
        posinf.map(|v| v as f32).unwrap_or(max)
    } else if value == f32::NEG_INFINITY {
        neginf.map(|v| v as f32).unwrap_or(min)
    } else {
        value
    }
}

fn fix_f64(value: f64, nan: f64, posinf: Option<f64>, neginf: Option<f64>, max: f64, min: f64) -> f64 {
    if value.is_nan() {
        nan
    } else if value == f64::INFINITY {
        posinf.unwrap_or(max)
    } else if value == f64::NEG_INFINITY {
        neginf.unwrap_or(min)
    } else {
        value
    }
}

fn any_float(attrs: &evoxc_core::Attrs, name: &str) -> Result<Option<f64>> {
    match attrs.get(name) {
        None => Ok(None),
        Some(AttrValue::Any(Json::Null)) => Ok(None),
        Some(AttrValue::Any(Json::Number(number))) => Ok(number.as_f64()),
        Some(AttrValue::Float(value)) => Ok(Some(*value)),
        Some(other) => Err(Error::verification(format!(
            "op 'nan_to_num': attribute {name:?} must be a number or null, got {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::OpSpec;
    use evoxc_core::{Attrs, Dim, ValueType};

    fn spec(name: &str, out: ValueType) -> OpSpec {
        OpSpec {
            name: name.to_string(),
            attrs: Attrs::new(),
            operand_types: Vec::new(),
            result_types: vec![out],
            regions: Vec::new(),
            location: None,
        }
    }

    fn buffer(dtype: DType, shape: &[usize], storage: Storage) -> Buffer {
        Buffer::new(dtype, shape.to_vec(), storage).unwrap()
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
    fn round_is_half_to_even() {
        let x = buffer(DType::Float64, &[3], Storage::F64(vec![2.5, 3.5, -2.5]));
        let op = spec("round", ValueType::of(DType::Float64, [Dim::int(3)]));
        let out = unary(&op, &x).unwrap();
        assert_eq!(out[0].storage, Storage::F64(vec![2.0, 4.0, -2.0]));
    }

    #[test]
    fn comparisons_produce_bool() {
        let a = buffer(DType::Float32, &[3], Storage::F32(vec![1.0, 2.0, 3.0]));
        let b = buffer(DType::Float32, &[3], Storage::F32(vec![2.0, 2.0, 2.0]));
        let op = spec("less", ValueType::of(DType::Bool, [Dim::int(3)]));
        let out = binary(&op, &a, &b).unwrap();
        assert_eq!(out[0].storage, Storage::Bool(vec![true, false, false]));
    }

    #[test]
    fn complex_abs_returns_a_real_result() {
        let x = buffer(
            DType::Complex128,
            &[2],
            Storage::C128(vec![Complex::new(3.0, 4.0), Complex::new(0.0, 2.0)]),
        );
        let op = spec("abs", ValueType::of(DType::Float64, [Dim::int(2)]));
        let out = unary(&op, &x).unwrap();
        assert_eq!(out[0].storage, Storage::F64(vec![5.0, 2.0]));
    }

    #[test]
    fn sigmoid_matches_the_naive_formula() {
        let x = buffer(DType::Float64, &[2], Storage::F64(vec![0.0, 100.0]));
        let op = spec("sigmoid", ValueType::of(DType::Float64, [Dim::int(2)]));
        let out = unary(&op, &x).unwrap();
        match &out[0].storage {
            Storage::F64(values) => {
                assert!((values[0] - 0.5).abs() < 1e-15);
                assert!((values[1] - 1.0).abs() < 1e-15);
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }
}
