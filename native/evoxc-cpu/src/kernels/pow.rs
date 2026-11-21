//! `power` with a scalar exponent: numpy's `npy_pow` special cases.
//!
//! numpy's `npy_pow` special-cases a handful of exponents so `x ** 2` does not
//! pay libm `powf` per element, and only for an ARRAY base (there is no shortcut
//! for `2 ** x`).  Here the exponent of a size-one operand is fixed for the
//! whole call, so the dispatch happens ONCE, outside the loop, and each arm is
//! its own monomorphized map -- a `match` inside the element loop would stop the
//! vectorizer.  Everything the table does not name falls through to the same
//! `powf` the general driver uses, so NaN propagation and the
//! negative-base-fractional NaN are unchanged.
//!
//! `elementwise::binary` reaches this module through [`scalar_power`] (the fast
//! path) and [`reject_negative_exponent`] (the integer guard); `util` keeps only
//! genuinely shared helpers.
use evoxc_core::{Error, Result};

use crate::buffer::{map1, Buffer, Storage};
use crate::exec::{single, Value};
use crate::scalar::Num;

/// The float dtypes a scalar-exponent `power` specialisation may cover: not
/// `f16` (numpy computes a float16 `power` in float32 and rounds back, which
/// `x * x` / `sqrt` would not reproduce bit for bit) and not the complex dtypes
/// (their `pow` is `exp(y * ln x)`, a different formula).
trait PowBase: Num {
    /// The nearest constant of this dtype (an `f64` literal folds to a literal).
    fn from_f64(value: f64) -> Self;
    /// The exponent as a `u32` when it is an exact non-negative integer small
    /// enough that repeated squaring beats one `powf`.
    fn small_exponent(self) -> Option<u32>;
}

macro_rules! impl_pow_base {
    ($($ty:ty),*) => {$(
        impl PowBase for $ty {
            fn from_f64(value: f64) -> Self {
                value as $ty
            }
            fn small_exponent(self) -> Option<u32> {
                if (0.0..=64.0).contains(&self) && self.fract() == 0.0 {
                    Some(self as u32)
                } else {
                    None
                }
            }
        }
    )*};
}

impl_pow_base!(f32, f64);

/// `base ** exponent` for a non-negative integer exponent by repeated squaring
/// (`base ** 0 == 1`, NaN included, matching `powf`).
fn powi<T: PowBase>(base: T, exponent: u32) -> T {
    let mut result = T::one();
    let mut square = base;
    let mut remaining = exponent;
    while remaining > 0 {
        if remaining & 1 == 1 {
            result = result.mul(square);
        }
        remaining >>= 1;
        if remaining > 0 {
            square = square.mul(square);
        }
    }
    result
}

/// A `power` result the scalar-exponent fast path produced.
enum PowScalar {
    /// The base buffer IS the answer (`x ** 1.0`); buffers are immutable, so a
    /// kernel may hand an operand back as its result instead of copying it.
    Reuse,
    /// A freshly built result of the op's declared dtype and shape.
    Built(Buffer),
}

/// `power(base, exponent)` when the exponent is a size-one buffer, dispatched on
/// the exponent's value.  `None` leaves the general driver in charge: a
/// broadcast exponent, a base that is not already the result shape, or a dtype
/// without the specialisation keeps its existing behavior exactly.
fn pow_scalar(
    out_shape: &[usize],
    base: &Buffer,
    exponent: &Buffer,
    base_is_original: bool,
) -> Result<Option<PowScalar>> {
    // A one-element exponent broadcasts to every element of the result, so only
    // the base has to be flat for the element-wise map to be exact.
    if exponent.numel() != 1 || !base.is_flat_for(out_shape) {
        return Ok(None);
    }
    let fast = match (&base.storage, &exponent.storage) {
        (Storage::F32(values), Storage::F32(exponent)) => {
            exponent.first().map(|exponent| {
                pow_scalar_typed(values, *exponent, out_shape, base_is_original)
            })
        }
        (Storage::F64(values), Storage::F64(exponent)) => {
            exponent.first().map(|exponent| {
                pow_scalar_typed(values, *exponent, out_shape, base_is_original)
            })
        }
        _ => return Ok(None),
    };
    fast.transpose()
}

fn pow_scalar_typed<T: PowBase>(
    base: &[T],
    exponent: T,
    out_shape: &[usize],
    base_is_original: bool,
) -> Result<PowScalar> {
    let dtype = T::DTYPE;
    if exponent == T::from_f64(0.0) {
        // `x ** 0 == 1` for every base, NaN and the infinities included.
        return map1(dtype, out_shape, base, |_| T::one()).map(PowScalar::Built);
    }
    if exponent == T::from_f64(1.0) {
        return if base_is_original {
            Ok(PowScalar::Reuse)
        } else {
            map1(dtype, out_shape, base, |value| value).map(PowScalar::Built)
        };
    }
    if exponent == T::from_f64(2.0) {
        return map1(dtype, out_shape, base, |value| value.mul(value)).map(PowScalar::Built);
    }
    if exponent == T::from_f64(0.5) {
        // numpy uses `sqrt` here too, so `-0.0 ** 0.5` keeps its sign.
        return map1(dtype, out_shape, base, |value| value.sqrt()).map(PowScalar::Built);
    }
    if exponent == T::from_f64(3.0) {
        return map1(dtype, out_shape, base, |value| value.mul(value).mul(value))
            .map(PowScalar::Built);
    }
    if exponent == T::from_f64(-1.0) {
        return map1(dtype, out_shape, base, |value| T::one().div(value)).map(PowScalar::Built);
    }
    if let Some(exponent) = exponent.small_exponent() {
        return map1(dtype, out_shape, base, move |value| powi(value, exponent))
            .map(PowScalar::Built);
    }
    map1(dtype, out_shape, base, move |value| value.pow(exponent)).map(PowScalar::Built)
}

/// `power(base, exponent)` for one whole kernel call: `Some(results)` when the
/// scalar-exponent fast path produced the answer, `None` to leave the general
/// broadcast driver in charge.  `original` is the caller's own base value, which
/// the exponent-1 identity hands back unchanged (`base_is_original` is false for
/// a base that only reached the kernel through a cast, so that case copies).
pub(crate) fn scalar_power(
    out_shape: &[usize],
    base: &Buffer,
    exponent: &Buffer,
    base_is_original: bool,
    original: &Value,
) -> Result<Option<Vec<Value>>> {
    let fast = match pow_scalar(out_shape, base, exponent, base_is_original)? {
        Some(fast) => fast,
        None => return Ok(None),
    };
    match fast {
        PowScalar::Reuse => Ok(Some(vec![original.clone()])),
        PowScalar::Built(buffer) => single(buffer).map(Some),
    }
}

/// Reject a negative integer exponent (numpy raises there; the engine has no
/// eager path to fall back to, so this is an explicit error).
pub(crate) fn reject_negative_exponent(name: &str, exponent: &Buffer) -> Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::elementwise;
    use crate::scalar::Scalar;
    use crate::plan::OpSpec;
    use evoxc_core::{Attrs, DType, Dim, ValueType};
    use half::f16;

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

    /// `power(x, exponent)` through the elementwise kernel: `binary` takes
    /// shared values, so these tests wrap their buffers.
    fn binary(spec: &OpSpec, a: &Buffer, b: &Buffer) -> Result<Vec<Value>> {
        elementwise::binary(
            spec,
            &std::rc::Rc::new(a.clone()),
            &std::rc::Rc::new(b.clone()),
        )
    }

    // -- the scalar-exponent `power` fast path ----------------------------

    /// `power(x, exponent)` through the fast path (panics if it declines).
    fn pow_arm_f32(x: &[f32], exponent: f32) -> Vec<f32> {
        let shape = [x.len()];
        let base = Buffer::new(DType::Float32, shape.to_vec(), Storage::F32(x.to_vec())).unwrap();
        let exponent = Buffer::new(DType::Float32, vec![], Storage::F32(vec![exponent])).unwrap();
        match pow_scalar(&shape, &base, &exponent, true).unwrap().expect("fast path") {
            PowScalar::Built(mut buffer) => {
                Scalar::take_vec(&mut buffer.storage).expect("f32 storage")
            }
            PowScalar::Reuse => x.to_vec(),
        }
    }

    fn pow_arm_f64(x: &[f64], exponent: f64) -> Vec<f64> {
        let shape = [x.len()];
        let base = Buffer::new(DType::Float64, shape.to_vec(), Storage::F64(x.to_vec())).unwrap();
        let exponent = Buffer::new(DType::Float64, vec![], Storage::F64(vec![exponent])).unwrap();
        match pow_scalar(&shape, &base, &exponent, true).unwrap().expect("fast path") {
            PowScalar::Built(mut buffer) => {
                Scalar::take_vec(&mut buffer.storage).expect("f64 storage")
            }
            PowScalar::Reuse => x.to_vec(),
        }
    }

    /// Bit-exact comparison (NaN payloads and signed zeros included).
    fn assert_bits_f32(got: &[f32], expected: &[f32], what: &str) {
        assert_eq!(got.len(), expected.len(), "{what}: length");
        for (index, (have, want)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(want.to_bits(), have.to_bits(), "{what}: index {index}");
        }
    }

    /// The bases every arm is pinned on: a near-one value (where `x * x` and
    /// `powf` differ in the last bit), negatives, a signed zero, both
    /// infinities and a NaN.
    const POW_BASES: [f32; 8] = [
        1.0000001,
        3.0,
        -2.0,
        -0.0,
        0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];

    #[test]
    fn power_arms_are_the_numpy_special_cases() {
        // 2.0 -> x*x, 3.0 -> x*x*x, 0.5 -> sqrt, -1.0 -> 1/x, 0.0 -> 1.
        assert_bits_f32(&pow_arm_f32(&POW_BASES, 2.0), &POW_BASES.map(|v| v * v), "x ** 2");
        assert_bits_f32(&pow_arm_f32(&POW_BASES, 3.0), &POW_BASES.map(|v| v * v * v), "x ** 3");
        assert_bits_f32(&pow_arm_f32(&POW_BASES, 0.5), &POW_BASES.map(f32::sqrt), "x ** 0.5");
        assert_bits_f32(&pow_arm_f32(&POW_BASES, -1.0), &POW_BASES.map(|v| 1.0 / v), "x ** -1");
        assert_bits_f32(&pow_arm_f32(&POW_BASES, 0.0), &[1.0; 8], "x ** 0");
        // numpy's `sqrt` keeps the sign of -0.0 and `1 / -0.0` is `-inf`.
        assert_eq!(pow_arm_f32(&[-0.0], 0.5)[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(pow_arm_f32(&[-0.0], -1.0), vec![f32::NEG_INFINITY]);
        // Exponent 5 is repeated squaring (`x^4 * x`), not five multiplies.
        let x = [1.0000001f32, -2.0, 3.0, 0.0];
        assert_bits_f32(
            &pow_arm_f32(&x, 5.0),
            &x.map(|v| (v * v) * (v * v) * v),
            "x ** 5",
        );
        // The identity arm hands the base back instead of copying it, unless the
        // base reached the kernel through a cast (`base_is_original == false`).
        let shape = [2usize];
        let base = Buffer::new(DType::Float32, shape.to_vec(), Storage::F32(vec![1.5, 2.5])).unwrap();
        let one = Buffer::new(DType::Float32, vec![], Storage::F32(vec![1.0])).unwrap();
        assert!(matches!(pow_scalar(&shape, &base, &one, true).unwrap(), Some(PowScalar::Reuse)));
        assert!(matches!(pow_scalar(&shape, &base, &one, false).unwrap(), Some(PowScalar::Built(_))));
        // An exponent the table does not name still takes this path (`powi`
        // declines it, so it is one `powf` per element)...
        assert_bits_f32(&pow_arm_f32(&[2.0, 3.0], 2.5), &[2.0f32.powf(2.5), 3.0f32.powf(2.5)], "x ** 2.5");
        // ...but a broadcast (non-size-one) exponent is declined outright.
        let exponent = Buffer::new(DType::Float32, vec![2, 2], Storage::F32(vec![2.0; 4])).unwrap();
        assert!(pow_scalar(&shape, &base, &exponent, true).unwrap().is_none());
        // The same five arms on f64.
        let x64 = [1.0000000000000002f64, -2.0, -0.0, f64::INFINITY, f64::NAN];
        for (exponent, expected) in [
            (2.0f64, x64.map(|v| v * v)),
            (3.0f64, x64.map(|v| v * v * v)),
            (0.5f64, x64.map(f64::sqrt)),
            (-1.0f64, x64.map(|v| 1.0 / v)),
            (0.0f64, [1.0; 5]),
        ] {
            let got = pow_arm_f64(&x64, exponent);
            for (index, (have, want)) in got.iter().zip(expected.iter()).enumerate() {
                assert_eq!(want.to_bits(), have.to_bits(), "f64 x ** {exponent}: index {index}");
            }
        }
    }

    #[test]
    fn powi_is_repeated_squaring() {
        assert_eq!(powi(3.0f32, 0), 1.0);
        assert_eq!(powi(3.0f32, 1), 3.0);
        assert_eq!(powi(3.0f32, 5), 243.0);
        assert_eq!(powi(-2.0f64, 3), -8.0);
        assert_eq!(powi(f64::NAN, 0), 1.0);
        assert!((powi(1.5f32, 7) - 1.5f32.powi(7)).abs() < 1e-5);
        assert_eq!(3.0f32.small_exponent(), Some(3));
        assert_eq!((-1.0f32).small_exponent(), None);
        assert_eq!(0.5f32.small_exponent(), None);
        assert_eq!(65.0f32.small_exponent(), None);
    }

    // -- `power` through the elementwise kernel ----------------------------

    fn pow_spec(dtype: DType, shape: &[usize]) -> OpSpec {
        spec("power", ValueType::of(dtype, shape.iter().map(|dim| Dim::int(*dim as i64))))
    }

    /// `power(x, exponent)` through the kernel; the exponent is a rank-0 buffer.
    fn pow_f32(x: &[f32], exponent: f32) -> Vec<f32> {
        let base = buffer(DType::Float32, &[x.len()], Storage::F32(x.to_vec()));
        let exponent = buffer(DType::Float32, &[], Storage::F32(vec![exponent]));
        let out = binary(&pow_spec(DType::Float32, &[x.len()]), &base, &exponent).unwrap();
        match &out[0].storage {
            Storage::F32(values) => values.clone(),
            other => panic!("unexpected storage {other:?}"),
        }
    }


    #[test]
    fn power_integer_exponents_use_repeated_squaring() {
        // An INTEGER exponent is cast to the float compute dtype first, so it
        // reaches the same fast path: 0 -> ones, 1 -> the base itself (reused,
        // not copied), 5 -> `x * x^2 * x^2` by repeated squaring.
        let x = [1.0000001f32, -2.0, 3.0, 0.0];
        let base = buffer(DType::Float32, &[x.len()], Storage::F32(x.to_vec()));
        let op = pow_spec(DType::Float32, &[x.len()]);
        let integer = |value: i32| buffer(DType::Int32, &[], Storage::I32(vec![value]));
        let f32_values = |out: Vec<Value>| match &out[0].storage {
            Storage::F32(values) => values.clone(),
            other => panic!("unexpected storage {other:?}"),
        };
        assert_eq!(f32_values(binary(&op, &base, &integer(0)).unwrap()), vec![1.0; x.len()]);
        let shared = std::rc::Rc::new(base.clone());
        let out = elementwise::binary(&op, &shared, &std::rc::Rc::new(integer(1))).unwrap();
        assert!(std::rc::Rc::ptr_eq(&out[0], &shared));
        assert_eq!(f32_values(out), x.to_vec());
        let five = f32_values(binary(&op, &base, &integer(5)).unwrap());
        for (index, value) in x.iter().enumerate() {
            let exact = (value * value) * (value * value) * value;
            assert_eq!(exact.to_bits(), five[index].to_bits(), "x ** 5: index {index}");
            let reference = value.powf(5.0);
            assert!(
                (reference - five[index]).abs() <= reference.abs() * 2e-7,
                "x ** 5 at index {index}: {reference} vs {}",
                five[index]
            );
        }
    }

    #[test]
    fn power_keeps_the_general_path_for_everything_else() {
        // A fractional exponent: one `powf` per element.
        let x = [2.0f32, 3.0, -2.0, 0.5];
        assert_bits_f32(
            &pow_f32(&x, 2.5),
            &x.map(|value| value.powf(2.5)),
            "x ** 2.5",
        );
        // A negative base with a fractional exponent stays NaN.
        assert!(pow_f32(&[-2.0, -0.5], 0.5).iter().all(|value| value.is_nan()));
        assert!(pow_f32(&[-2.0, -0.5], 2.5).iter().all(|value| value.is_nan()));

        // A (2, 2) exponent against a (2,) base: no size-one exponent, so the
        // general broadcast driver computes one `powf` per element.
        let base = buffer(DType::Float32, &[2], Storage::F32(vec![2.0, 3.0]));
        let exponent = buffer(
            DType::Float32,
            &[2, 2],
            Storage::F32(vec![2.0, 3.0, 0.5, -1.0]),
        );
        let out = binary(&pow_spec(DType::Float32, &[2, 2]), &base, &exponent).unwrap();
        match &out[0].storage {
            Storage::F32(values) => assert_bits_f32(
                values,
                &[4.0, 27.0, std::f32::consts::SQRT_2, 1.0 / 3.0],
                "broadcast exponent",
            ),
            other => panic!("unexpected storage {other:?}"),
        }

        // The scalar on the LEFT has no shortcut (`2 ** x` is not `x ** 2`).
        let two = buffer(DType::Float32, &[], Storage::F32(vec![2.0]));
        let base = buffer(DType::Float32, &[3], Storage::F32(vec![0.5, 3.0, -1.0]));
        let out = binary(&pow_spec(DType::Float32, &[3]), &two, &base).unwrap();
        match &out[0].storage {
            Storage::F32(values) => assert_bits_f32(
                values,
                &[0.5f32, 3.0, -1.0].map(|value| 2.0f32.powf(value)),
                "2 ** x",
            ),
            other => panic!("unexpected storage {other:?}"),
        }

        // A size-one exponent buffer that is not rank 0 takes the fast path.
        let base = buffer(DType::Float32, &[2], Storage::F32(vec![1.0000001, -2.0]));
        let one = buffer(DType::Float32, &[1], Storage::F32(vec![2.0]));
        let out = binary(&pow_spec(DType::Float32, &[2]), &base, &one).unwrap();
        match &out[0].storage {
            Storage::F32(values) => assert_bits_f32(values, &[1.0000002, 4.0], "x ** [2.0]"),
            other => panic!("unexpected storage {other:?}"),
        }

        // float16 is not specialised (numpy computes it in float32 and rounds
        // back), so this stays the `powf` result; a cast base still produces a
        // float result for the identity arm.
        let near_one = f16::from_f32(1.0005);
        let half = buffer(DType::Float16, &[2], Storage::F16(vec![f16::from_f32(3.0), near_one]));
        let exponent = buffer(DType::Float16, &[], Storage::F16(vec![f16::from_f32(3.0)]));
        let out = binary(&pow_spec(DType::Float16, &[2]), &half, &exponent).unwrap();
        assert_eq!(
            out[0].storage,
            Storage::F16(vec![
                f16::from_f32(3.0f32.powf(3.0)),
                f16::from_f32(near_one.to_f32().powf(3.0)),
            ])
        );
        let ints = buffer(DType::Int32, &[3], Storage::I32(vec![1, -2, 7]));
        let exponent = buffer(DType::Float32, &[], Storage::F32(vec![1.0]));
        let out = binary(&pow_spec(DType::Float32, &[3]), &ints, &exponent).unwrap();
        assert_eq!(out[0].storage, Storage::F32(vec![1.0, -2.0, 7.0]));
    }
}
