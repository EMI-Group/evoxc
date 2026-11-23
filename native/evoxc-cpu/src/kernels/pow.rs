//! `power` with a scalar exponent: numpy's `npy_pow` special cases.
//!
//! numpy's `npy_pow` special-cases a handful of exponents so `x ** 2` does not
//! pay libm `powf` per element, and only for an ARRAY base (there is no shortcut
//! for `2 ** x`).  Here the exponent of a size-one operand is fixed for the
//! whole call, so the dispatch happens ONCE, outside the loop, and each arm is
//! its own monomorphized map -- a `match` inside the element loop would stop the
//! vectorizer.  The table is reproduced EXACTLY -- `0`, `1`, `2`, `3`, `0.5` and
//! `-1` -- and every other exponent, integer or not, is ONE libm `powf`/`pow`
//! call, exactly as `npy_pow` computes outside its table.  That is what makes
//! this backend bit-identical to numpy's `npy_pow` on f32 and f64: a
//! repeated-squaring arm for small integers would round differently from `powf`
//! (measured on f32: exponent 21 up to 13 ulp, exponent 64 up to 40 ulp), so no
//! such arm exists and `powi` has no reason to exist either.  NaN propagation and
//! the negative-base-fractional NaN are therefore also `npy_pow`'s.
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
}

macro_rules! impl_pow_base {
    ($($ty:ty),*) => {$(
        impl PowBase for $ty {
            fn from_f64(value: f64) -> Self {
                value as $ty
            }
        }
    )*};
}

impl_pow_base!(f32, f64);

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
    // Every exponent outside numpy's table is ONE libm `powf`/`pow` call, which
    // is what `npy_pow` does outside its own table (and why the two agree bit
    // for bit -- no repeated squaring, no `powi`).
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
        // The identity arm hands the base back instead of copying it, unless the
        // base reached the kernel through a cast (`base_is_original == false`).
        let shape = [2usize];
        let base = Buffer::new(DType::Float32, shape.to_vec(), Storage::F32(vec![1.5, 2.5])).unwrap();
        let one = Buffer::new(DType::Float32, vec![], Storage::F32(vec![1.0])).unwrap();
        assert!(matches!(pow_scalar(&shape, &base, &one, true).unwrap(), Some(PowScalar::Reuse)));
        assert!(matches!(pow_scalar(&shape, &base, &one, false).unwrap(), Some(PowScalar::Built(_))));
        // An exponent the table does not name still takes this path, as one
        // `powf` per element (the separate `power_off_table_exponents_are_one_powf`
        // test pins that against libm)...
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


    /// The off-table sample.  Half of it is chosen so a repeated-squaring arm
    /// would FAIL this test: `powi` and `powf` disagree in the last bits on
    /// these bases at every exponent below (measured), so the array is pinned by
    /// `from_bits` and nothing floats.  The rest are the edge values every arm
    /// must keep: `1.0`, both signed zeros, a tiny and a huge magnitude, both
    /// infinities and a NaN; the negative base makes the sign rules depend on
    /// the exponent's parity.
    const OFF_TABLE_F32_BITS: [u32; 16] = [
        0x4066_5835, // 3.5991337  -- differs from repeated squaring at 4, 21, 64
        0x4048_02DB, // 3.1251743  -- differs at 4, 21, 64
        0x3F7A_7265, // 0.9783080  -- differs at every exponent here
        0x4027_7959, // 2.6167815  -- differs at 4, 21, 64
        0x406C_95C5, // 3.6966412  -- differs at 5, 64
        0x3F83_E6A6, // 1.0304763  -- differs at 5, 64
        0xBFAF_DDA2, // -1.3739512 -- the negative-base sign rules
        0x3F80_0001, // 1.0000001
        0x0DA2_4260, // 1.0e-30
        0x7149_F2CA, // 3.0e30
        0x0000_0000, // 0.0
        0x8000_0000, // -0.0
        0x3F80_0000, // 1.0
        0x7F80_0000, // +inf
        0xFF80_0000, // -inf
        0x7FC0_0000, // NaN
    ];

    const OFF_TABLE_F64_BITS: [u64; 15] = [
        0x4003_9116_7254_024F, // 2.4430...  -- differs from repeated squaring at 4, 5, 21, 64
        0x3FE5_A942_C9BF_CA71, // 0.6773...  -- differs at every exponent here
        0x4004_33E8_9C21_C23A, // 2.5256...  -- differs at every exponent here
        0x3FD7_FA1B_1628_B818, // 0.3749...  -- differs at 5, 64
        0xC003_9116_7254_024F, // -2.4430... -- the negative-base sign rules
        0xBFE5_A942_C9BF_CA71, // -0.6773...
        0x3FF0_0000_0000_0001, // 1.0000000000000002
        0x01A5_6E1F_C2F8_F359, // 1.0e-300
        0x7E51_EB2D_6600_5835, // 3.0e300
        0x0000_0000_0000_0000, // 0.0
        0x8000_0000_0000_0000, // -0.0
        0x3FF0_0000_0000_0000, // 1.0
        0x7FF0_0000_0000_0000, // +inf
        0xFFF0_0000_0000_0000, // -inf
        0x7FF8_0000_0000_0000, // NaN
    ];

    /// **The binding equivalence**: every exponent outside numpy's special table
    /// -- integer ones (4, 5, 21, 64) included -- is ONE libm `powf`/`pow` call, so
    /// the kernel's result is bit-for-bit `x.powf(e)` on f32 and f64.
    #[test]
    fn power_off_table_exponents_are_one_powf() {
        let bases32 = OFF_TABLE_F32_BITS.map(f32::from_bits);
        for exponent in [4.0f32, 5.0, 21.0, 64.0] {
            let got = pow_arm_f32(&bases32, exponent);
            let want = bases32.map(|value| value.powf(exponent));
            assert_bits_f32(&got, &want, &format!("f32 x ** {exponent}"));
        }
        let bases64 = OFF_TABLE_F64_BITS.map(f64::from_bits);
        for exponent in [4.0f64, 5.0, 21.0, 64.0] {
            let got = pow_arm_f64(&bases64, exponent);
            let want = bases64.map(|value| value.powf(exponent));
            for (index, (have, expected)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    expected.to_bits(),
                    have.to_bits(),
                    "f64 x ** {exponent}: index {index}"
                );
            }
        }
        // An even exponent keeps the sign of the zero, an odd one keeps `-0.0`.
        assert_eq!(pow_arm_f32(&[-0.0], 4.0)[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(pow_arm_f32(&[-0.0], 5.0)[0].to_bits(), (-0.0f32).to_bits());

        // An INTEGER exponent is cast to the float compute dtype first, so it
        // reaches the very same arm: `x ** 21` through an int32 exponent buffer is
        // bit-identical to `powf(21.0)` (and NOT to a repeated-squaring product).
        let x = [1.0000001f32, -2.0, 3.5991337, 0.0];
        let base = buffer(DType::Float32, &[x.len()], Storage::F32(x.to_vec()));
        let op = pow_spec(DType::Float32, &[x.len()]);
        for exponent in [4i32, 5, 21, 64] {
            let integer = buffer(DType::Int32, &[], Storage::I32(vec![exponent]));
            let out = binary(&op, &base, &integer).unwrap();
            match &out[0].storage {
                Storage::F32(values) => assert_bits_f32(
                    values,
                    &x.map(|value| value.powf(exponent as f32)),
                    &format!("f32 x ** {exponent} (int32 exponent)"),
                ),
                other => panic!("unexpected storage {other:?}"),
            }
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
