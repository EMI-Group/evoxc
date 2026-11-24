//! The pointwise unary family, the complex `abs` magnitude and the bool
//! `logical_not`.
use evoxc_core::{DType, Error, Result};

use crate::buffer::{map1, Buffer, Storage};
use crate::exec::{single, RunCtx, Value};
use crate::plan::OpSpec;
use crate::scalar::{Num, Scalar};
use crate::with_type;

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

pub(super) fn unary_kernel(
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
    single(dense_unary(kind, out_dtype, shape, values)?)
}

/// The unary fast path (the operand always has the result's shape).  `kind` is
/// dispatched ONCE so each arm's closure folds to that single operation — the
/// cheap ops become vector moves and the transcendental ones a tight call loop.
/// Every arm delegates to [`apply_un`], so there is one definition per op, and
/// the result storage is built from the mapped iterator instead of zero-filled.
fn dense_unary<T: Num>(
    kind: UnKind,
    dtype: DType,
    shape: &[usize],
    values: &[T],
) -> Result<Buffer> {
    macro_rules! arm {
        ($kind:ident) => {
            map1(dtype, shape, values, |value| apply_un(UnKind::$kind, value))
        };
    }
    match kind {
        UnKind::Abs => arm!(Abs),
        UnKind::Negate => arm!(Negate),
        UnKind::Square => arm!(Square),
        UnKind::Sqrt => arm!(Sqrt),
        UnKind::Exp => arm!(Exp),
        UnKind::Log => arm!(Log),
        UnKind::Log1p => arm!(Log1p),
        UnKind::Sin => arm!(Sin),
        UnKind::Cos => arm!(Cos),
        UnKind::Tan => arm!(Tan),
        UnKind::Acos => arm!(Acos),
        UnKind::Tanh => arm!(Tanh),
        UnKind::Sigmoid => arm!(Sigmoid),
        UnKind::Relu => arm!(Relu),
        UnKind::Sign => arm!(Sign),
        UnKind::Floor => arm!(Floor),
        UnKind::Ceil => arm!(Ceil),
        UnKind::Round => arm!(Round),
        // `logical_not` leaves through `typed_unary` (it produces bool).
        UnKind::LogicalNot => map1(dtype, shape, values, |value| value),
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use evoxc_core::{Dim, ValueType};
    use num_complex::Complex;

    use crate::kernels::elementwise::test_support::{buffer, spec};

    #[test]
    fn round_is_half_to_even() {
        let x = buffer(DType::Float64, &[3], Storage::F64(vec![2.5, 3.5, -2.5]));
        let op = spec("round", ValueType::of(DType::Float64, [Dim::int(3)]));
        let out = unary(&op, &x).unwrap();
        assert_eq!(out[0].storage, Storage::F64(vec![2.0, 4.0, -2.0]));
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

    #[test]
    fn f32_exp_is_libm_exact_including_the_tails() {
        // `exp` on f32 runs the crate's ordinary scalar path (`T::exp`, i.e. the
        // platform's `expf`), so the result is bit-identical to `f32::exp` for
        // normal, overflowing, underflowing and NaN inputs alike. glibc's
        // vectorized `expf` is also faster than a hand-rolled polynomial here
        // (see the perf note in `CONTEXT.md`), so there is no separate arm.
        let mut values: Vec<f32> = (0..512)
            .map(|index| -20.0 + 40.0 * index as f32 / 511.0)
            .collect();
        values.extend_from_slice(&[
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            200.0,
            -200.0,
            -88.0,
            87.5,
            87.0,
            -87.0,
            0.0,
        ]);
        let count = values.len();
        let x = buffer(DType::Float32, &[count], Storage::F32(values.clone()));
        let op = spec("exp", ValueType::of(DType::Float32, [Dim::int(count as i64)]));
        let out = unary(&op, &x).unwrap();
        match &out[0].storage {
            Storage::F32(got) => {
                for (index, (a, b)) in got.iter().zip(values.iter()).enumerate() {
                    let expected = b.exp();
                    assert!(
                        a.to_bits() == expected.to_bits() || (a.is_nan() && expected.is_nan()),
                        "element {index} ({b}): {a} vs {expected}"
                    );
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }
}
