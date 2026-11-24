//! Scalar vocabulary: the typed numeric layer every kernel is generic over.
//!
//! Two traits describe the 14 evoxc dtypes once:
//!
//! * [`Scalar`] — conversion between a concrete Rust scalar and the neutral
//!   [`ScalarValue`] (used by `cast`, comparisons across promoted dtypes and
//!   numpy handoff), plus construction of a typed [`Storage`] vector.
//! * [`Num`] — the arithmetic/transcendental operation set elementwise,
//!   reduction and linalg kernels call.  Integer arithmetic wraps (numpy
//!   semantics on overflow), float `rem` is numpy floormod and `round` rounds
//!   half-to-even.
//!
//! Operations a dtype cannot meaningfully perform (complex ordering, float
//! bitwise) return `self`; the calling kernel validates the dtype combination
//! first and raises an explicit error — the trait never decides semantics.
use evoxc_core::DType;
use half::f16;
use num_complex::Complex;

use crate::buffer::Storage;

/// A dtype-erased scalar used only on conversion paths.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarValue {
    I(i64),
    U(u64),
    F(f64),
    C(Complex<f64>),
}

impl ScalarValue {
    /// numpy truthiness (NaN is true; a complex value is true when either part
    /// is non-zero).
    pub fn truthy(self) -> bool {
        match self {
            ScalarValue::I(value) => value != 0,
            ScalarValue::U(value) => value != 0,
            ScalarValue::F(value) => value != 0.0,
            ScalarValue::C(value) => value.re != 0.0 || value.im != 0.0,
        }
    }

    /// The real part as `f64` (the imaginary part is dropped).
    pub fn real(self) -> f64 {
        match self {
            ScalarValue::I(value) => value as f64,
            ScalarValue::U(value) => value as f64,
            ScalarValue::F(value) => value,
            ScalarValue::C(value) => value.re,
        }
    }
}

/// One concrete Rust scalar type backed by an evoxc dtype.
pub trait Scalar: Copy + PartialEq + 'static + Send + Sync {
    const DTYPE: DType;
    fn to_scalar(self) -> ScalarValue;
    fn from_scalar(value: ScalarValue) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
    fn zeros(count: usize) -> Vec<Self> {
        vec![Self::zero(); count]
    }
    /// Wrap a finished output vector in the matching [`Storage`] variant.
    fn into_storage(values: Vec<Self>) -> Storage;
    /// Move this dtype's vector out of `storage`, leaving an empty one behind
    /// (`None` when the variant does not match) — how the result-storage pool
    /// reclaims a finished result's allocation, see `buffer`'s pool note.
    fn take_vec(storage: &mut Storage) -> Option<Vec<Self>>;
}

/// The arithmetic and transcendental operation set kernels use.
pub trait Num: Scalar {
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;
    /// numpy `remainder` (floor-mod, sign of the divisor).
    fn rem(self, other: Self) -> Self;
    fn pow(self, other: Self) -> Self;
    fn neg(self) -> Self;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    /// `ln(1 + x)` (numpy `log1p`); the default is the naive composition, the
    /// float impls override with the accurate libm form.
    fn log1p(self) -> Self {
        let one = Self::one();
        one.add(self).ln()
    }
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn tanh(self) -> Self;
    fn sign(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round_even(self) -> Self;
    /// numpy `maximum` (NaN propagates; no complex ordering).
    fn max(self, other: Self) -> Self;
    /// numpy `minimum` (NaN propagates; no complex ordering).
    fn min(self, other: Self) -> Self;
    fn bit_and(self, other: Self) -> Self;
    fn bit_or(self, other: Self) -> Self;
    fn bit_xor(self, other: Self) -> Self;
    fn shl(self, other: Self) -> Self;
    fn shr(self, other: Self) -> Self;
}

macro_rules! impl_int_scalar {
    ($ty:ty, $dtype:ident, $variant:ident) => {
        impl Scalar for $ty {
            const DTYPE: DType = DType::$dtype;
            fn to_scalar(self) -> ScalarValue {
                ScalarValue::I(self as i64)
            }
            fn from_scalar(value: ScalarValue) -> Self {
                match value {
                    ScalarValue::I(v) => v as $ty,
                    ScalarValue::U(v) => v as $ty,
                    ScalarValue::F(v) => v as $ty,
                    ScalarValue::C(v) => v.re as $ty,
                }
            }
            fn zero() -> Self {
                0
            }
            fn one() -> Self {
                1
            }
            fn into_storage(values: Vec<Self>) -> Storage {
                Storage::$variant(values)
            }
            fn take_vec(storage: &mut Storage) -> Option<Vec<Self>> {
                match storage {
                    Storage::$variant(values) => Some(std::mem::take(values)),
                    _ => None,
                }
            }
        }
        impl Num for $ty {
            fn add(self, other: Self) -> Self {
                self.wrapping_add(other)
            }
            fn sub(self, other: Self) -> Self {
                self.wrapping_sub(other)
            }
            fn mul(self, other: Self) -> Self {
                self.wrapping_mul(other)
            }
            fn div(self, other: Self) -> Self {
                self.wrapping_div(other)
            }
            fn rem(self, other: Self) -> Self {
                if other == 0 {
                    return 0;
                }
                let r = self.wrapping_rem(other);
                if r != 0 && ((r < 0) != (other < 0)) {
                    r.wrapping_add(other)
                } else {
                    r
                }
            }
            fn pow(self, other: Self) -> Self {
                if other < 0 {
                    return 0;
                }
                self.wrapping_pow(other as u32)
            }
            fn neg(self) -> Self {
                self.wrapping_neg()
            }
            fn abs(self) -> Self {
                self.wrapping_abs()
            }
            fn sqrt(self) -> Self {
                (self as f64).sqrt() as $ty
            }
            fn exp(self) -> Self {
                (self as f64).exp() as $ty
            }
            fn ln(self) -> Self {
                (self as f64).ln() as $ty
            }
            fn sin(self) -> Self {
                (self as f64).sin() as $ty
            }
            fn cos(self) -> Self {
                (self as f64).cos() as $ty
            }
            fn tan(self) -> Self {
                (self as f64).tan() as $ty
            }
            fn tanh(self) -> Self {
                (self as f64).tanh() as $ty
            }
            fn sign(self) -> Self {
                if self == 0 {
                    0
                } else if self > 0 {
                    1
                } else {
                    -1
                }
            }
            fn floor(self) -> Self {
                self
            }
            fn ceil(self) -> Self {
                self
            }
            fn round_even(self) -> Self {
                self
            }
            fn max(self, other: Self) -> Self {
                if self >= other {
                    self
                } else {
                    other
                }
            }
            fn min(self, other: Self) -> Self {
                if self <= other {
                    self
                } else {
                    other
                }
            }
            fn bit_and(self, other: Self) -> Self {
                self & other
            }
            fn bit_or(self, other: Self) -> Self {
                self | other
            }
            fn bit_xor(self, other: Self) -> Self {
                self ^ other
            }
            fn shl(self, other: Self) -> Self {
                self.wrapping_shl(other as u32)
            }
            fn shr(self, other: Self) -> Self {
                self.wrapping_shr(other as u32)
            }
        }
    };
}

impl_int_scalar!(i8, Int8, I8);
impl_int_scalar!(i16, Int16, I16);
impl_int_scalar!(i32, Int32, I32);
impl_int_scalar!(i64, Int64, I64);

/// Unsigned integers: no `wrapping_abs`, no negative literal.
macro_rules! impl_uint_scalar {
    ($ty:ty, $dtype:ident, $variant:ident) => {
        impl Scalar for $ty {
            const DTYPE: DType = DType::$dtype;
            fn to_scalar(self) -> ScalarValue {
                ScalarValue::U(self as u64)
            }
            fn from_scalar(value: ScalarValue) -> Self {
                match value {
                    ScalarValue::I(v) => v as $ty,
                    ScalarValue::U(v) => v as $ty,
                    ScalarValue::F(v) => v as $ty,
                    ScalarValue::C(v) => v.re as $ty,
                }
            }
            fn zero() -> Self {
                0
            }
            fn one() -> Self {
                1
            }
            fn into_storage(values: Vec<Self>) -> Storage {
                Storage::$variant(values)
            }
            fn take_vec(storage: &mut Storage) -> Option<Vec<Self>> {
                match storage {
                    Storage::$variant(values) => Some(std::mem::take(values)),
                    _ => None,
                }
            }
        }
        impl Num for $ty {
            fn add(self, other: Self) -> Self {
                self.wrapping_add(other)
            }
            fn sub(self, other: Self) -> Self {
                self.wrapping_sub(other)
            }
            fn mul(self, other: Self) -> Self {
                self.wrapping_mul(other)
            }
            fn div(self, other: Self) -> Self {
                self.wrapping_div(other)
            }
            fn rem(self, other: Self) -> Self {
                // Unsigned remainder already takes the (positive) divisor's sign.
                self.wrapping_rem(other)
            }
            fn pow(self, other: Self) -> Self {
                self.wrapping_pow(other as u32)
            }
            fn neg(self) -> Self {
                self.wrapping_neg()
            }
            fn abs(self) -> Self {
                self
            }
            fn sqrt(self) -> Self {
                (self as f64).sqrt() as $ty
            }
            fn exp(self) -> Self {
                (self as f64).exp() as $ty
            }
            fn ln(self) -> Self {
                (self as f64).ln() as $ty
            }
            fn sin(self) -> Self {
                (self as f64).sin() as $ty
            }
            fn cos(self) -> Self {
                (self as f64).cos() as $ty
            }
            fn tan(self) -> Self {
                (self as f64).tan() as $ty
            }
            fn tanh(self) -> Self {
                (self as f64).tanh() as $ty
            }
            fn sign(self) -> Self {
                if self == 0 { 0 } else { 1 }
            }
            fn floor(self) -> Self {
                self
            }
            fn ceil(self) -> Self {
                self
            }
            fn round_even(self) -> Self {
                self
            }
            fn max(self, other: Self) -> Self {
                if self >= other { self } else { other }
            }
            fn min(self, other: Self) -> Self {
                if self <= other { self } else { other }
            }
            fn bit_and(self, other: Self) -> Self {
                self & other
            }
            fn bit_or(self, other: Self) -> Self {
                self | other
            }
            fn bit_xor(self, other: Self) -> Self {
                self ^ other
            }
            fn shl(self, other: Self) -> Self {
                self.wrapping_shl(other as u32)
            }
            fn shr(self, other: Self) -> Self {
                self.wrapping_shr(other as u32)
            }
        }
    };
}

impl_uint_scalar!(u8, UInt8, U8);
impl_uint_scalar!(u16, UInt16, U16);
impl_uint_scalar!(u32, UInt32, U32);
impl_uint_scalar!(u64, UInt64, U64);

macro_rules! impl_float_scalar {
    ($ty:ty, $dtype:ident, $variant:ident) => {
        impl Scalar for $ty {
            const DTYPE: DType = DType::$dtype;
            fn to_scalar(self) -> ScalarValue {
                ScalarValue::F(self as f64)
            }
            fn from_scalar(value: ScalarValue) -> Self {
                match value {
                    ScalarValue::I(v) => v as $ty,
                    ScalarValue::U(v) => v as $ty,
                    ScalarValue::F(v) => v as $ty,
                    ScalarValue::C(v) => v.re as $ty,
                }
            }
            fn zero() -> Self {
                0.0
            }
            fn one() -> Self {
                1.0
            }
            fn into_storage(values: Vec<Self>) -> Storage {
                Storage::$variant(values)
            }
            fn take_vec(storage: &mut Storage) -> Option<Vec<Self>> {
                match storage {
                    Storage::$variant(values) => Some(std::mem::take(values)),
                    _ => None,
                }
            }
        }
        impl Num for $ty {
            fn add(self, other: Self) -> Self {
                self + other
            }
            fn sub(self, other: Self) -> Self {
                self - other
            }
            fn mul(self, other: Self) -> Self {
                self * other
            }
            fn div(self, other: Self) -> Self {
                self / other
            }
            fn rem(self, other: Self) -> Self {
                let r = self % other;
                if r != 0.0 && ((r < 0.0) != (other < 0.0)) {
                    r + other
                } else {
                    r
                }
            }
            fn pow(self, other: Self) -> Self {
                self.powf(other)
            }
            fn neg(self) -> Self {
                -self
            }
            fn abs(self) -> Self {
                <$ty>::abs(self)
            }
            fn sqrt(self) -> Self {
                <$ty>::sqrt(self)
            }
            fn exp(self) -> Self {
                <$ty>::exp(self)
            }
            fn ln(self) -> Self {
                <$ty>::ln(self)
            }
            fn log1p(self) -> Self {
                <$ty>::ln_1p(self)
            }
            fn sin(self) -> Self {
                <$ty>::sin(self)
            }
            fn cos(self) -> Self {
                <$ty>::cos(self)
            }
            fn tan(self) -> Self {
                <$ty>::tan(self)
            }
            fn tanh(self) -> Self {
                <$ty>::tanh(self)
            }
            fn sign(self) -> Self {
                if self.is_nan() || self == 0.0 {
                    self
                } else if self > 0.0 {
                    1.0
                } else {
                    -1.0
                }
            }
            fn floor(self) -> Self {
                <$ty>::floor(self)
            }
            fn ceil(self) -> Self {
                <$ty>::ceil(self)
            }
            fn round_even(self) -> Self {
                <$ty>::round_ties_even(self)
            }
            fn max(self, other: Self) -> Self {
                if self.is_nan() {
                    self
                } else if other.is_nan() {
                    other
                } else if self >= other {
                    self
                } else {
                    other
                }
            }
            fn min(self, other: Self) -> Self {
                if self.is_nan() {
                    self
                } else if other.is_nan() {
                    other
                } else if self <= other {
                    self
                } else {
                    other
                }
            }
            fn bit_and(self, other: Self) -> Self {
                let _ = other;
                self
            }
            fn bit_or(self, other: Self) -> Self {
                let _ = other;
                self
            }
            fn bit_xor(self, other: Self) -> Self {
                let _ = other;
                self
            }
            fn shl(self, other: Self) -> Self {
                let _ = other;
                self
            }
            fn shr(self, other: Self) -> Self {
                let _ = other;
                self
            }
        }
    };
}

impl_float_scalar!(f32, Float32, F32);
impl_float_scalar!(f64, Float64, F64);

impl Scalar for f16 {
    const DTYPE: DType = DType::Float16;
    fn to_scalar(self) -> ScalarValue {
        ScalarValue::F(self.to_f64())
    }
    fn from_scalar(value: ScalarValue) -> Self {
        match value {
            ScalarValue::I(v) => f16::from_f64(v as f64),
            ScalarValue::U(v) => f16::from_f64(v as f64),
            ScalarValue::F(v) => f16::from_f64(v),
            ScalarValue::C(v) => f16::from_f64(v.re),
        }
    }
    fn zero() -> Self {
        f16::from_f32(0.0)
    }
    fn one() -> Self {
        f16::from_f32(1.0)
    }
    fn into_storage(values: Vec<Self>) -> Storage {
        Storage::F16(values)
    }
    fn take_vec(storage: &mut Storage) -> Option<Vec<Self>> {
        match storage {
            Storage::F16(values) => Some(std::mem::take(values)),
            _ => None,
        }
    }
}

impl Num for f16 {
    fn add(self, other: Self) -> Self {
        f16::from_f32(self.to_f32() + other.to_f32())
    }
    fn sub(self, other: Self) -> Self {
        f16::from_f32(self.to_f32() - other.to_f32())
    }
    fn mul(self, other: Self) -> Self {
        f16::from_f32(self.to_f32() * other.to_f32())
    }
    fn div(self, other: Self) -> Self {
        f16::from_f32(self.to_f32() / other.to_f32())
    }
    fn rem(self, other: Self) -> Self {
        f16::from_f32(<f32 as Num>::rem(self.to_f32(), other.to_f32()))
    }
    fn pow(self, other: Self) -> Self {
        f16::from_f32(self.to_f32().powf(other.to_f32()))
    }
    fn neg(self) -> Self {
        -self
    }
    fn abs(self) -> Self {
        f16::from_f32(self.to_f32().abs())
    }
    fn sqrt(self) -> Self {
        f16::from_f32(self.to_f32().sqrt())
    }
    fn exp(self) -> Self {
        f16::from_f32(self.to_f32().exp())
    }
    fn ln(self) -> Self {
        f16::from_f32(self.to_f32().ln())
    }
    fn log1p(self) -> Self {
        f16::from_f32(self.to_f32().ln_1p())
    }
    fn sin(self) -> Self {
        f16::from_f32(self.to_f32().sin())
    }
    fn cos(self) -> Self {
        f16::from_f32(self.to_f32().cos())
    }
    fn tan(self) -> Self {
        f16::from_f32(self.to_f32().tan())
    }
    fn tanh(self) -> Self {
        f16::from_f32(self.to_f32().tanh())
    }
    fn sign(self) -> Self {
        f16::from_f32(<f32 as Num>::sign(self.to_f32()))
    }
    fn floor(self) -> Self {
        f16::from_f32(self.to_f32().floor())
    }
    fn ceil(self) -> Self {
        f16::from_f32(self.to_f32().ceil())
    }
    fn round_even(self) -> Self {
        f16::from_f32(self.to_f32().round_ties_even())
    }
    fn max(self, other: Self) -> Self {
        f16::from_f32(<f32 as Num>::max(self.to_f32(), other.to_f32()))
    }
    fn min(self, other: Self) -> Self {
        f16::from_f32(<f32 as Num>::min(self.to_f32(), other.to_f32()))
    }
    fn bit_and(self, other: Self) -> Self {
        let _ = other;
        self
    }
    fn bit_or(self, other: Self) -> Self {
        let _ = other;
        self
    }
    fn bit_xor(self, other: Self) -> Self {
        let _ = other;
        self
    }
    fn shl(self, other: Self) -> Self {
        let _ = other;
        self
    }
    fn shr(self, other: Self) -> Self {
        let _ = other;
        self
    }
}

macro_rules! impl_complex_scalar {
    ($dtype:ident, $variant:ident, $real:ty) => {
        impl Scalar for Complex<$real> {
            const DTYPE: DType = DType::$dtype;
            fn to_scalar(self) -> ScalarValue {
                ScalarValue::C(Complex::new(self.re as f64, self.im as f64))
            }
            fn from_scalar(value: ScalarValue) -> Self {
                match value {
                    ScalarValue::I(v) => Complex::new(v as $real, 0.0),
                    ScalarValue::U(v) => Complex::new(v as $real, 0.0),
                    ScalarValue::F(v) => Complex::new(v as $real, 0.0),
                    ScalarValue::C(v) => Complex::new(v.re as $real, v.im as $real),
                }
            }
            fn zero() -> Self {
                Complex::new(0.0, 0.0)
            }
            fn one() -> Self {
                Complex::new(1.0, 0.0)
            }
            fn into_storage(values: Vec<Self>) -> Storage {
                Storage::$variant(values)
            }
            fn take_vec(storage: &mut Storage) -> Option<Vec<Self>> {
                match storage {
                    Storage::$variant(values) => Some(std::mem::take(values)),
                    _ => None,
                }
            }
        }
        impl Num for Complex<$real> {
            fn add(self, other: Self) -> Self {
                self + other
            }
            fn sub(self, other: Self) -> Self {
                self - other
            }
            fn mul(self, other: Self) -> Self {
                self * other
            }
            fn div(self, other: Self) -> Self {
                self / other
            }
            fn rem(self, other: Self) -> Self {
                let _ = other;
                self
            }
            fn pow(self, other: Self) -> Self {
                (other * self.ln()).exp()
            }
            fn neg(self) -> Self {
                -self
            }
            /// Magnitude (the kernel maps the result dtype to a real one).
            fn abs(self) -> Self {
                Complex::new(self.norm(), 0.0)
            }
            fn sqrt(self) -> Self {
                self.sqrt()
            }
            fn exp(self) -> Self {
                self.exp()
            }
            fn ln(self) -> Self {
                self.ln()
            }
            fn sin(self) -> Self {
                self.sin()
            }
            fn cos(self) -> Self {
                self.cos()
            }
            fn tan(self) -> Self {
                self.tan()
            }
            fn tanh(self) -> Self {
                self.tanh()
            }
            fn sign(self) -> Self {
                let norm = self.norm();
                if norm == 0.0 {
                    self
                } else {
                    self / norm
                }
            }
            fn floor(self) -> Self {
                self
            }
            fn ceil(self) -> Self {
                self
            }
            fn round_even(self) -> Self {
                self
            }
            fn max(self, other: Self) -> Self {
                let _ = other;
                self
            }
            fn min(self, other: Self) -> Self {
                let _ = other;
                self
            }
            fn bit_and(self, other: Self) -> Self {
                let _ = other;
                self
            }
            fn bit_or(self, other: Self) -> Self {
                let _ = other;
                self
            }
            fn bit_xor(self, other: Self) -> Self {
                let _ = other;
                self
            }
            fn shl(self, other: Self) -> Self {
                let _ = other;
                self
            }
            fn shr(self, other: Self) -> Self {
                let _ = other;
                self
            }
        }
    };
}

impl_complex_scalar!(Complex64, C64, f32);
impl_complex_scalar!(Complex128, C128, f64);

impl Scalar for bool {
    const DTYPE: DType = DType::Bool;
    fn to_scalar(self) -> ScalarValue {
        ScalarValue::I(self as i64)
    }
    fn from_scalar(value: ScalarValue) -> Self {
        value.truthy()
    }
    fn zero() -> Self {
        false
    }
    fn one() -> Self {
        true
    }
    fn into_storage(values: Vec<Self>) -> Storage {
        Storage::Bool(values)
    }
    fn take_vec(storage: &mut Storage) -> Option<Vec<Self>> {
        match storage {
            Storage::Bool(values) => Some(std::mem::take(values)),
            _ => None,
        }
    }
}

impl Num for bool {
    fn add(self, other: Self) -> Self {
        self || other
    }
    fn sub(self, other: Self) -> Self {
        let _ = other;
        self
    }
    fn mul(self, other: Self) -> Self {
        self && other
    }
    fn div(self, other: Self) -> Self {
        let _ = other;
        self
    }
    fn rem(self, other: Self) -> Self {
        let _ = other;
        self
    }
    fn pow(self, other: Self) -> Self {
        let _ = other;
        self
    }
    fn neg(self) -> Self {
        self
    }
    fn abs(self) -> Self {
        self
    }
    fn sqrt(self) -> Self {
        self
    }
    fn exp(self) -> Self {
        self
    }
    fn ln(self) -> Self {
        self
    }
    fn sin(self) -> Self {
        self
    }
    fn cos(self) -> Self {
        self
    }
    fn tan(self) -> Self {
        self
    }
    fn tanh(self) -> Self {
        self
    }
    fn sign(self) -> Self {
        self
    }
    fn floor(self) -> Self {
        self
    }
    fn ceil(self) -> Self {
        self
    }
    fn round_even(self) -> Self {
        self
    }
    fn max(self, other: Self) -> Self {
        self || other
    }
    fn min(self, other: Self) -> Self {
        self && other
    }
    fn bit_and(self, other: Self) -> Self {
        self && other
    }
    fn bit_or(self, other: Self) -> Self {
        self || other
    }
    fn bit_xor(self, other: Self) -> Self {
        self ^ other
    }
    fn shl(self, other: Self) -> Self {
        let _ = other;
        self
    }
    fn shr(self, other: Self) -> Self {
        let _ = other;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_remainder_takes_divisor_sign() {
        assert_eq!(<i64 as Num>::rem(-7, 3), 2);
        assert_eq!(<i64 as Num>::rem(7, -3), -2);
        assert_eq!(<i64 as Num>::rem(-7, -3), -1);
        assert_eq!(<f64 as Num>::rem(-7.0, 3.0), 2.0);
    }

    #[test]
    fn float_round_is_half_to_even() {
        assert_eq!(<f64 as Num>::round_even(2.5), 2.0);
        assert_eq!(<f64 as Num>::round_even(3.5), 4.0);
        assert_eq!(<f64 as Num>::round_even(-2.5), -2.0);
    }

    #[test]
    fn conversions_round_trip_through_scalar_value() {
        let value = <i32 as Scalar>::to_scalar(-3);
        assert_eq!(<u64 as Scalar>::from_scalar(value), u64::MAX - 2);
        let float = <f64 as Scalar>::to_scalar(1.5);
        assert_eq!(<f16 as Scalar>::from_scalar(float), f16::from_f32(1.5));
        assert!(<bool as Scalar>::from_scalar(<f64 as Scalar>::to_scalar(
            f64::NAN
        )));
    }

    #[test]
    fn complex_magnitude_and_sign() {
        let z = Complex::new(3.0f64, 4.0);
        assert_eq!(z.abs(), Complex::new(5.0, 0.0));
        assert_eq!(z.sign(), Complex::new(0.6, 0.8));
    }
}
