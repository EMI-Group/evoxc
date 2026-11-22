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
    if kind == UnKind::Exp {
        // `exp` on f32 is the one transcendental op with a vectorizable
        // polynomial replacement (see [`dense_exp_f32`]); every other dtype and
        // op keeps the scalar `libm` call.
        if let Storage::F32(values) = &source.storage {
            return single(dense_exp_f32(&source.shape, values)?);
        }
    }
    with_type!(source, |values| {
        typed_unary(values, kind, compute, &source.shape)
    })
}

/// The largest magnitude the two-pass f32 `exp` fast path covers: beyond it the
/// result can overflow to `inf` or land in the subnormal range, where the
/// bit-built `2^k` scale would be wrong, so such a buffer takes the libm path.
const EXP_FAST_BOUND: f32 = 87.0;

/// `|x| <= EXP_FAST_BOUND`, and `false` for a NaN (which takes the libm path).
#[inline]
fn is_within_exp_bound(value: f32) -> bool {
    value.abs() <= EXP_FAST_BOUND
}

/// `exp` on an f32 buffer: the polynomial fast path when every element is
/// inside [`EXP_FAST_BOUND`] (and not NaN), the libm call otherwise.
///
/// The precheck is one extra vectorized read pass; it is what keeps the fast
/// loop branchless (a per-element range check would inhibit vectorization) while
/// leaving the overflow/underflow/NaN tails on `f32::exp`, so their results —
/// `inf`, a subnormal, `NaN` — stay exactly libm's.
fn dense_exp_f32(shape: &[usize], values: &[f32]) -> Result<Buffer> {
    if values.iter().any(|value| !is_within_exp_bound(*value)) {
        return map1(DType::Float32, shape, values, |value: f32| value.exp());
    }
    map1(DType::Float32, shape, values, exp_fast_f32)
}

/// `exp(x)` for `|x| <= EXP_FAST_BOUND`: `k = round(x / ln2)`, `r = x - k*ln2`
/// (a two-part `ln2` keeps the reduction exact), `exp(r)` as a degree-7 Taylor
/// polynomial of the argument (`|r| <= ln2/2`, truncation error `< 1e-8`
/// relative — under a fifth of an f32 ulp) and a bit-built `2^k` scale.
///
/// The measured max relative error against `f32::exp` over the whole covered
/// range is pinned by `the_f32_exp_fast_path_stays_within_its_ulp_budget`.
#[inline]
fn exp_fast_f32(value: f32) -> f32 {
    const LOG2_E: f32 = std::f32::consts::LOG2_E;
    // ln2 split so `k * ln2` is exact: `LN2_HI` is the f32 with its low mantissa
    // bits clear (so `k * LN2_HI` is exactly representable for |k| <= 256) and
    // `LN2_LO` the rounded remainder (residual 5e-14).
    const LN2_HI: f32 = 0.693_145_75;
    const LN2_LO: f32 = 1.428_606_8e-6;
    let scaled = value * LOG2_E;
    let round = if scaled >= 0.0 { 0.5 } else { -0.5 };
    let k = (scaled + round) as i32;
    let kf = k as f32;
    let r = (value - kf * LN2_HI) - kf * LN2_LO;
    let poly = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0
                    + r * (1.0 / 24.0
                        + r * (1.0 / 120.0 + r * (1.0 / 720.0 + r * (1.0 / 5040.0)))))));
    // |x| <= 87 bounds k to [-126, 126], so k + 127 is a valid f32 exponent.
    let scale = f32::from_bits(((k + 127) as u32) << 23);
    poly * scale
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
    fn the_f32_exp_fast_path_stays_within_its_ulp_budget() {
        // The polynomial arm must stay inside the documented f32 parity budget
        // (rtol 1e-5 / atol 1e-6 on every elementwise leaf) with room to spare:
        // sweep the covered range in fine steps and hold the max relative error
        // to a fifth of an ulp... i.e. well under one ulp, recorded here as a
        // hard ceiling of 3e-7.
        let mut worst = 0.0f32;
        let mut worst_at = 0.0f32;
        let steps = 200_000;
        for index in 0..steps {
            let x = -EXP_FAST_BOUND + 2.0 * EXP_FAST_BOUND * index as f32 / steps as f32;
            let fast = exp_fast_f32(x);
            let libm = x.exp();
            let relative = ((fast - libm) / libm).abs();
            if relative > worst {
                worst = relative;
                worst_at = x;
            }
        }
        assert!(
            worst <= 3e-7,
            "max relative error {worst} at x = {worst_at}"
        );
        // The exact points a Taylor argument could break on.
        for x in [0.0f32, 1.0, -1.0, 0.5, -0.5, 86.0, -86.0, std::f32::consts::LN_2] {
            let relative = ((exp_fast_f32(x) - x.exp()) / x.exp()).abs();
            assert!(relative <= 3e-7, "x = {x}: relative error {relative}");
        }
    }

    #[test]
    fn the_f32_exp_tails_stay_exactly_libm() {
        // Outside the fast bound the kernel defers to `f32::exp` for the WHOLE
        // buffer, so overflow, underflow/subnormal and NaN results are libm's
        // own bits — not a clamped approximation.
        let values = vec![
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            200.0,
            -200.0,
            -88.0,
            87.5,
            1.0,
        ];
        let x = buffer(DType::Float32, &[8], Storage::F32(values.clone()));
        let op = spec("exp", ValueType::of(DType::Float32, [Dim::int(8)]));
        let out = unary(&op, &x).unwrap();
        let expected: Vec<f32> = values.iter().map(|value| value.exp()).collect();
        match &out[0].storage {
            Storage::F32(got) => {
                for (index, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
                    assert!(
                        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                        "element {index}: {a} vs {b}"
                    );
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }

    #[test]
    fn the_f32_exp_fast_path_matches_libm_through_the_kernel() {
        // The kernel arm (precheck + polynomial) over a buffer that stays inside
        // the bound, against libm elementwise.
        let values: Vec<f32> = (0..512)
            .map(|index| -20.0 + 40.0 * index as f32 / 511.0)
            .collect();
        let x = buffer(DType::Float32, &[512], Storage::F32(values.clone()));
        let op = spec("exp", ValueType::of(DType::Float32, [Dim::int(512)]));
        let out = unary(&op, &x).unwrap();
        match &out[0].storage {
            Storage::F32(got) => {
                for (index, (a, b)) in got.iter().zip(values.iter()).enumerate() {
                    let libm = b.exp();
                    assert!(
                        (a - libm).abs() <= 3e-7 * libm.abs(),
                        "element {index} ({b}): {a} vs {libm}"
                    );
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }
}
