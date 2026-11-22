//! Key-based functional RNG kernels — bit-exact with the numpy reference.
//!
//! The authority for this file is `evoxc/backends/numpy/kernels/random.py`'s
//! WORD-STREAM CONTRACT: every random op is a pure, deterministic function of
//! its key operand and the op's `algorithm` attribute
//! (`splitmix64` | `threefry2x32` | `philox4x32_10`).
//!
//! * **Per-op salt** — folded into the KEY, never the counter
//!   (`salt_lo`/`salt_hi` XOR into the key words; threefry: `(k0^lo, k1^hi)`,
//!   philox: `(k0^lo, k1^hi, k2^lo, k3^hi)`).
//! * **Counter** — cipher call/block `p` uses counter `(p, 0)`; threefry
//!   emits 2 words per call, philox 4 per block (interleaved C-order).
//! * **Element → word** — `uniform`/`randint`/`permutation` use flat word `i`;
//!   `normal` uses flat words `2i` (u1) and `2i+1` (u2), generated directly as
//!   two parallel chains (identical to the interleaved stream's even/odd
//!   words, without materializing it).
//! * **Uniform conversion** — 64-bit words: `(w >> 11) * 2^-53`; 32-bit
//!   words: `w * 2^-32`.
//! * **normal** — Box–Muller; the documented f32 fast path runs the whole
//!   tail in float32 (word fractions convert to f32 FIRST), every other
//!   output dtype uses the exact f64 path and casts at the end.
//!
//! All cipher arithmetic is exact wrapping integer math; no global state, no
//! time, no OS randomness.
use evoxc_core::{AttrValue, DType, Error, Result};
use half::f16;

use crate::buffer::{Buffer, Storage};
use crate::exec::RunCtx;
use crate::kernels::KernelMap;
use crate::plan::OpSpec;
use crate::scalar::{Scalar, ScalarValue};
use crate::with_type;

use key::Algorithm;

mod bits;
mod key;
mod permutation;
mod sampling;

#[cfg(test)]
mod test_util;

/// Fixed per-op salts (hex digits of pi) — see `_SALTS` in the reference.
const SALT_UNIFORM: u64 = 0x243F_6A88_85A3_08D3;
const SALT_NORMAL: u64 = 0x1319_8A2E_0370_7344;
const SALT_RANDINT: u64 = 0xA409_3822_299F_31D0;
const SALT_PERMUTATION: u64 = 0x082E_FA98_EC4E_6C89;
const SALT_MULTINOMIAL: u64 = 0x4528_21E6_38D0_1377;

// --- small shared helpers --------------------------------------------------

/// An op's `algorithm` attribute (default `splitmix64`).
fn algorithm_of(spec: &OpSpec) -> Result<Algorithm> {
    match spec.attrs.get("algorithm") {
        None | Some(AttrValue::Null) => Ok(Algorithm::SplitMix64),
        Some(AttrValue::Str(name)) => Algorithm::parse(&spec.name, name),
        Some(other) => Err(Error::verification(format!(
            "op '{}': attribute \"algorithm\" must be a str, got {}",
            spec.name,
            other.type_name()
        ))),
    }
}

/// A required `int` attribute, with an explicit error naming the op.
fn int_attr(spec: &OpSpec, name: &str) -> Result<i64> {
    match spec.attrs.get(name) {
        Some(AttrValue::Int(value)) => Ok(*value),
        Some(other) => Err(Error::verification(format!(
            "op '{}': attribute {name:?} must be an int, got {}",
            spec.name,
            other.type_name()
        ))),
        None => Err(Error::verification(format!(
            "op '{}': missing required attribute {name:?}",
            spec.name
        ))),
    }
}

/// The declared result type's concrete shape (symbolic dims resolved).
fn result_shape(ctx: &RunCtx<'_>, spec: &OpSpec) -> Result<Vec<usize>> {
    ctx.eval_shape_strict(&spec.result_types[0].shape)
}

/// The declared result dtype.
fn result_dtype(spec: &OpSpec) -> DType {
    spec.result_types[0].dtype
}

/// The op's DECLARED draw shape (its `shape` attribute, symbolic dims
/// resolved).  The result type is that shape broadcast against the parameter
/// operands, while the word stream is drawn for the attribute shape and then
/// broadcast — exactly the reference kernel's arithmetic.
fn draw_shape(ctx: &RunCtx<'_>, spec: &OpSpec) -> Result<Vec<usize>> {
    let shape = spec.attrs.shape("shape").map_err(|_| {
        Error::verification(format!(
            "op '{}': a random draw needs a \"shape\" attribute",
            spec.name
        ))
    })?;
    ctx.eval_shape_strict(shape)
}

/// Reject an output dtype outside the allowed kind set.
fn require_kind(op: &str, dtype: DType, floats: bool) -> Result<()> {
    let ok = if floats {
        dtype.is_float()
    } else {
        dtype.is_int()
    };
    if ok {
        return Ok(());
    }
    Err(Error::dtype(format!(
        "op '{op}': output dtype must be {} dtype, got {}",
        if floats { "a floating" } else { "an integer" },
        dtype.name()
    )))
}

/// Build a typed buffer from dtype-erased values (numpy casting semantics).
fn build(dtype: DType, shape: Vec<usize>, values: &[ScalarValue]) -> Result<Buffer> {
    macro_rules! collect {
        ($ty:ty, $variant:ident) => {
            Storage::$variant(
                values
                    .iter()
                    .map(|value| <$ty as Scalar>::from_scalar(*value))
                    .collect(),
            )
        };
    }
    let storage = match dtype {
        DType::Bool => collect!(bool, Bool),
        DType::Int8 => collect!(i8, I8),
        DType::Int16 => collect!(i16, I16),
        DType::Int32 => collect!(i32, I32),
        DType::Int64 => collect!(i64, I64),
        DType::UInt8 => collect!(u8, U8),
        DType::UInt16 => collect!(u16, U16),
        DType::UInt32 => collect!(u32, U32),
        DType::UInt64 => collect!(u64, U64),
        DType::Float16 => collect!(f16, F16),
        DType::Float32 => collect!(f32, F32),
        DType::Float64 => collect!(f64, F64),
        other => {
            return Err(Error::dtype(format!(
                "cannot materialize a random draw as {} — expected a real numeric dtype",
                other.name()
            )))
        }
    };
    Buffer::new(dtype, shape, storage)
}

/// Every element of a buffer as a dtype-erased scalar.
fn scalar_values(buffer: &Buffer) -> Vec<ScalarValue> {
    with_type!(buffer, |values| {
        values.iter().map(|value| value.to_scalar()).collect()
    })
}

fn as_i64(value: ScalarValue) -> i64 {
    match value {
        ScalarValue::I(v) => v,
        ScalarValue::U(v) => v as i64,
        ScalarValue::F(v) => v as i64,
        ScalarValue::C(v) => v.re as i64,
    }
}

fn as_u64(value: ScalarValue) -> u64 {
    match value {
        ScalarValue::I(v) => v as u64,
        ScalarValue::U(v) => v,
        ScalarValue::F(v) => v as u64,
        ScalarValue::C(v) => v.re as u64,
    }
}

fn as_f64(value: ScalarValue) -> f64 {
    value.real()
}

// --- registration ----------------------------------------------------------

pub fn register(map: &mut KernelMap) {
    map.insert("random_key_mix", sampling::key_mix_kernel);
    map.insert("random_uniform", sampling::uniform_kernel);
    map.insert("random_normal", sampling::normal_kernel);
    map.insert("random_randint", sampling::randint_kernel);
    map.insert("random_permutation", permutation::permutation_kernel);
    map.insert("random_multinomial", permutation::multinomial_kernel);
}
