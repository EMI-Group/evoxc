//! Shared kernel helpers (axis normalization, shape arithmetic, C-order
//! indexing).  Category modules may add private helpers in their own file;
//! this module only holds vocabulary shared by three or more categories.
use evoxc_core::{DType, Error, Result};

use crate::buffer::{map1, shape_text, Buffer, Storage};
use crate::exec::RunCtx;
use crate::plan::OpSpec;
use crate::scalar::Num;

/// Normalize one axis (`-1` → `rank - 1`); out of range is a `ShapeError`.
pub fn normalize_axis(axis: i64, rank: usize, what: &str) -> Result<usize> {
    let rank_i = rank as i64;
    let normalized = if axis < 0 { axis + rank_i } else { axis };
    if normalized < 0 || normalized >= rank_i {
        return Err(Error::shape(format!(
            "{what}: axis {axis} is out of range for rank {rank}"
        )));
    }
    Ok(normalized as usize)
}

/// Normalize a set of axes, rejecting duplicates.
pub fn normalize_axes(axes: &[i64], rank: usize, what: &str) -> Result<Vec<usize>> {
    let mut out = Vec::with_capacity(axes.len());
    for axis in axes {
        let normalized = normalize_axis(*axis, rank, what)?;
        if out.contains(&normalized) {
            return Err(Error::shape(format!(
                "{what}: duplicate axis {normalized}"
            )));
        }
        out.push(normalized);
    }
    Ok(out)
}

/// Reject a complex dtype with an explicit message naming the op.
pub fn reject_complex(op: &str, dtype: DType) -> Result<()> {
    if dtype.is_complex() {
        return Err(Error::backend(format!(
            "op '{op}': complex input is not supported"
        )));
    }
    Ok(())
}

/// Reject a non-floating dtype.
pub fn require_float(op: &str, dtype: DType) -> Result<()> {
    if !dtype.is_float() {
        return Err(Error::backend(format!(
            "op '{op}': expected a floating dtype, got {}",
            dtype.name()
        )));
    }
    Ok(())
}

/// Number of elements described by a concrete shape.
pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// C-order (row-major) strides for a concrete shape.
pub fn c_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    let mut stride = 1usize;
    for axis in (0..shape.len()).rev() {
        strides[axis] = stride;
        stride *= shape[axis];
    }
    strides
}

/// Flat offset of a C-order multi-index.
pub fn flat_index(index: &[usize], strides: &[usize]) -> usize {
    index.iter().zip(strides).map(|(i, s)| i * s).sum()
}

/// Evaluate a result shape to concrete extents.  Runtime-dynamic (`None`)
/// dims cannot be resolved here; the caller must derive them from operands.
pub fn result_shape(ctx: &RunCtx<'_>, spec: &OpSpec, index: usize) -> Result<Vec<usize>> {
    let evaluated = ctx.result_shape(spec, index)?;
    let mut out = Vec::with_capacity(evaluated.len());
    for (axis, dim) in evaluated.iter().enumerate() {
        match dim {
            Some(extent) => out.push(*extent),
            None => {
                return Err(Error::shape(format!(
                    "op '{}': result {index} dim {axis} is runtime-dynamic; derive it from the operands",
                    spec.name
                )))
            }
        }
    }
    Ok(out)
}

/// A shape-mismatch error in the interpreter's wording.
pub fn shape_error(op: &str, what: &str, actual: &[usize], expected: &[usize]) -> Error {
    Error::shape(format!(
        "op '{op}': {what} has shape {}, expected {}",
        shape_text(actual),
        shape_text(expected)
    ))
}

/// True when `small` broadcasts to exactly `big` (right-aligned).
pub fn broadcasts_to(small: &[usize], big: &[usize]) -> bool {
    if small.len() > big.len() {
        return false;
    }
    let offset = big.len() - small.len();
    small
        .iter()
        .enumerate()
        .all(|(axis, dim)| *dim == 1 || *dim == big[offset + axis])
}

/// The float dtypes a scalar-exponent `power` specialisation may cover: not
/// `f16` (numpy computes a float16 `power` in float32 and rounds back, which
/// `x * x` / `sqrt` would not reproduce bit for bit) and not the complex dtypes
/// (their `pow` is `exp(y * ln x)`, a different formula).
pub trait PowBase: Num {
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
pub fn powi<T: PowBase>(base: T, exponent: u32) -> T {
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

// `power` with a scalar exponent: numpy's `npy_pow` special-cases a handful of
// exponents so `x ** 2` does not pay libm `powf` per element, and only for an
// ARRAY base (there is no shortcut for `2 ** x`).  Here the exponent of a
// size-one operand is fixed for the whole call, so the dispatch happens ONCE,
// outside the loop, and each arm is its own monomorphized map -- a `match`
// inside the element loop would stop the vectorizer.  Everything the table does
// not name falls through to the same `powf` the general driver uses, so NaN
// propagation and the negative-base-fractional NaN are unchanged.

/// A `power` result the scalar-exponent fast path produced.
pub enum PowScalar {
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
pub fn pow_scalar(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axes_normalize_and_reject() {
        assert_eq!(normalize_axis(-1, 3, "x").unwrap(), 2);
        assert_eq!(normalize_axis(0, 3, "x").unwrap(), 0);
        assert!(normalize_axis(3, 3, "x").is_err());
        assert!(normalize_axes(&[0, 0], 2, "x").is_err());
    }

    // -- the scalar-exponent `power` fast path ----------------------------

    /// `power(x, exponent)` through the fast path (panics if it declines).
    fn pow_arm_f32(x: &[f32], exponent: f32) -> Vec<f32> {
        let shape = [x.len()];
        let base = Buffer::new(DType::Float32, shape.to_vec(), Storage::F32(x.to_vec())).unwrap();
        let exponent = Buffer::new(DType::Float32, vec![], Storage::F32(vec![exponent])).unwrap();
        match pow_scalar(&shape, &base, &exponent, true).unwrap().expect("fast path") {
            PowScalar::Built(buffer) => match buffer.storage {
                Storage::F32(values) => values,
                other => panic!("unexpected storage {other:?}"),
            },
            PowScalar::Reuse => x.to_vec(),
        }
    }

    fn pow_arm_f64(x: &[f64], exponent: f64) -> Vec<f64> {
        let shape = [x.len()];
        let base = Buffer::new(DType::Float64, shape.to_vec(), Storage::F64(x.to_vec())).unwrap();
        let exponent = Buffer::new(DType::Float64, vec![], Storage::F64(vec![exponent])).unwrap();
        match pow_scalar(&shape, &base, &exponent, true).unwrap().expect("fast path") {
            PowScalar::Built(buffer) => match buffer.storage {
                Storage::F64(values) => values,
                other => panic!("unexpected storage {other:?}"),
            },
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
    #[test]
    fn strides_are_c_order() {
        assert_eq!(c_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(flat_index(&[1, 2, 3], &[12, 4, 1]), 23);
    }
}
