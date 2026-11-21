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

use crate::buffer::{broadcast_to, Buffer, Storage};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util::numel;
use crate::kernels::KernelMap;
use crate::plan::OpSpec;
use crate::scalar::{Scalar, ScalarValue};
use crate::with_type;

// --- constants -------------------------------------------------------------

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX_B: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX_C: u64 = 0x94D0_49BB_1331_11EB;
const MASK64: u64 = 0xFFFF_FFFF_FFFF_FFFF;

const INV_2P53: f64 = 1.0 / 9_007_199_254_740_992.0; // 2^-53, exact
const INV_2P32: f64 = 1.0 / 4_294_967_296.0; // 2^-32, exact
const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// f32 fast-path constants (the exact f32 values the numpy kernel uses).
const INV_2P53_F32: f32 = 1.0 / 9_007_199_254_740_992.0; // 2^-53 as f32
const INV_2P32_F32: f32 = 1.0 / 4_294_967_296.0; // 2^-32 as f32
const NEG_TWO_F32: f32 = -2.0;

/// Fixed per-op salts (hex digits of pi) — see `_SALTS` in the reference.
const SALT_UNIFORM: u64 = 0x243F_6A88_85A3_08D3;
const SALT_NORMAL: u64 = 0x1319_8A2E_0370_7344;
const SALT_RANDINT: u64 = 0xA409_3822_299F_31D0;
const SALT_PERMUTATION: u64 = 0x082E_FA98_EC4E_6C89;
const SALT_MULTINOMIAL: u64 = 0x4528_21E6_38D0_1377;

const PARITY32: u32 = 0x1BD1_1BDA;
const ROT_A: [u32; 4] = [13, 15, 26, 6];
const ROT_B: [u32; 4] = [17, 29, 16, 24];

const PHILOX_M0: u64 = 0xD251_1F53;
const PHILOX_M1: u64 = 0xCD9E_8D57;
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;

// --- algorithms ------------------------------------------------------------

/// The canonical RNG algorithm names (`evoxc/ir/op_defs/random.py`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Algorithm {
    SplitMix64,
    ThreeFry2x32,
    Philox4x32Ten,
}

impl Algorithm {
    fn parse(op: &str, name: &str) -> Result<Algorithm> {
        match name {
            "splitmix64" => Ok(Algorithm::SplitMix64),
            "threefry2x32" => Ok(Algorithm::ThreeFry2x32),
            "philox4x32_10" => Ok(Algorithm::Philox4x32Ten),
            other => Err(Error::backend(format!(
                "op '{op}': unknown random algorithm {other:?}; expected one of \
                 splitmix64, threefry2x32, philox4x32_10"
            ))),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Algorithm::SplitMix64 => "splitmix64",
            Algorithm::ThreeFry2x32 => "threefry2x32",
            Algorithm::Philox4x32Ten => "philox4x32_10",
        }
    }

    fn key_words(self) -> usize {
        match self {
            Algorithm::SplitMix64 => 0,
            Algorithm::ThreeFry2x32 => 2,
            Algorithm::Philox4x32Ten => 4,
        }
    }
}

/// The key operand, normalized for its algorithm.
enum Key {
    /// splitmix64: the 64-bit state (rank-0 integer operand).
    State(u64),
    /// threefry2x32 / philox4x32_10: 2 / 4 key words (int32 operand).
    Words(Vec<u32>),
}

/// A flat word stream: uint64 words for splitmix64, uint32 for the
/// counter-based algorithms.
enum Words {
    W64(Vec<u64>),
    W32(Vec<u32>),
}

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

// --- splitmix64 ------------------------------------------------------------

/// The three SplitMix64 mixing steps (wrapping mod 2^64 by definition).
fn mix3(mut z: u64) -> u64 {
    z ^= z >> 30;
    z = z.wrapping_mul(MIX_B);
    z ^= z >> 27;
    z = z.wrapping_mul(MIX_C);
    z ^= z >> 31;
    z
}

/// The canonical SplitMix64 mix of one 64-bit value (adds GOLDEN first).
fn mix_scalar(x: u64) -> u64 {
    mix3(x.wrapping_add(GOLDEN))
}

/// `count` SplitMix64 words: `word_i = mix3(seed + (i + 1) * GOLDEN)`.
fn splitmix_words(seed: u64, count: usize) -> Vec<u64> {
    let base = seed.wrapping_add(GOLDEN);
    (0..count)
        .map(|i| mix3((i as u64).wrapping_mul(GOLDEN).wrapping_add(base)))
        .collect()
}

/// The `(u1, u2)` word chains behind one Box–Muller pair: chain A mixes
/// `seed + (2i+1) * GOLDEN`, chain B `seed + (2i+2) * GOLDEN`.
fn splitmix_pairs(seed: u64, count: usize) -> (Vec<u64>, Vec<u64>) {
    let step = (2u64.wrapping_mul(GOLDEN)) & MASK64;
    let base_a = seed.wrapping_add(GOLDEN);
    let base_b = seed.wrapping_add(2u64.wrapping_mul(GOLDEN));
    let mut u1 = Vec::with_capacity(count);
    let mut u2 = Vec::with_capacity(count);
    for i in 0..count {
        let prod = (i as u64).wrapping_mul(step);
        u1.push(mix3(prod.wrapping_add(base_a)));
        u2.push(mix3(prod.wrapping_add(base_b)));
    }
    (u1, u2)
}

// --- threefry2x32 ----------------------------------------------------------

/// Threefry2x32-20 over cipher calls with counter `(p, 0)`: returns the two
/// output word chains `(x0, x1)` (flat words `2p`, `2p+1`).
fn threefry2x32(counters: &[u32], k0: u32, k1: u32) -> (Vec<u32>, Vec<u32>) {
    let key_schedule = [k0, k1, k0 ^ k1 ^ PARITY32];
    let mut out0 = Vec::with_capacity(counters.len());
    let mut out1 = Vec::with_capacity(counters.len());
    for counter in counters {
        let mut x0 = counter.wrapping_add(k0);
        let mut x1 = k1; // counter word 1 is always 0
        for iteration in 0..5u32 {
            let rotations = if iteration % 2 == 0 { ROT_A } else { ROT_B };
            for rotation in rotations {
                x0 = x0.wrapping_add(x1);
                x1 = x1.rotate_left(rotation) ^ x0;
            }
            let inject = (iteration + 1) % 3;
            x0 = x0.wrapping_add(key_schedule[inject as usize]);
            x1 = x1
                .wrapping_add(key_schedule[((iteration + 2) % 3) as usize])
                .wrapping_add(iteration + 1);
        }
        out0.push(x0);
        out1.push(x1);
    }
    (out0, out1)
}

/// Salt-folded threefry key words.
fn threefry_key(key_words: &[u32], salt: u64) -> (u32, u32) {
    let salt_lo = salt as u32;
    let salt_hi = (salt >> 32) as u32;
    (key_words[0] ^ salt_lo, key_words[1] ^ salt_hi)
}

// --- philox4x32_10 ---------------------------------------------------------

/// Round-key words for round `r` (0-based): the cyclically permuted pair
/// `(k_{2r mod 4} + r*W0, k_{(2r+1) mod 4} + r*W1)` mod 2^32.
fn philox_round_keys(key: &[u32; 4], r: u32) -> (u32, u32) {
    let pair = (2 * (r % 2)) as usize;
    let rk0 = key[pair].wrapping_add(r.wrapping_mul(PHILOX_W0));
    let rk1 = key[pair + 1].wrapping_add(r.wrapping_mul(PHILOX_W1));
    (rk0, rk1)
}

/// Philox4x32-10 over cipher blocks with counter `(p, 0, 0, 0)`: returns the
/// four output word chains (flat words `4p..4p+3`).
fn philox4x32(counters: &[u32], key: [u32; 4]) -> [Vec<u32>; 4] {
    let mut out: [Vec<u32>; 4] = [
        Vec::with_capacity(counters.len()),
        Vec::with_capacity(counters.len()),
        Vec::with_capacity(counters.len()),
        Vec::with_capacity(counters.len()),
    ];
    for counter in counters {
        // 64-bit lanes keep the 32x32 -> 64 multiply exact.
        let (mut x0, mut x1, mut x2, mut x3) = (*counter as u64, 0u64, 0u64, 0u64);
        for r in 0..10u32 {
            let p0 = x0.wrapping_mul(PHILOX_M0);
            let p2 = x2.wrapping_mul(PHILOX_M1);
            let (rk0, rk1) = philox_round_keys(&key, r);
            x0 = (p2 >> 32) ^ x1 ^ rk0 as u64;
            x1 = p2 & 0xFFFF_FFFF;
            x2 = (p0 >> 32) ^ x3 ^ rk1 as u64;
            x3 = p0 & 0xFFFF_FFFF;
        }
        out[0].push(x0 as u32);
        out[1].push(x1 as u32);
        out[2].push(x2 as u32);
        out[3].push(x3 as u32);
    }
    out
}

/// Salt-folded philox key words.
fn philox_key(key_words: &[u32], salt: u64) -> [u32; 4] {
    let salt_lo = salt as u32;
    let salt_hi = (salt >> 32) as u32;
    [
        key_words[0] ^ salt_lo,
        key_words[1] ^ salt_hi,
        key_words[2] ^ salt_lo,
        key_words[3] ^ salt_hi,
    ]
}

// --- word streams ----------------------------------------------------------

fn counters(count: usize) -> Vec<u32> {
    (0..count as u32).collect()
}

/// `count` flat stream words for one op's salted key.
fn stream_words(algorithm: Algorithm, key: &Key, salt: u64, count: usize) -> Result<Words> {
    match (algorithm, key) {
        (Algorithm::SplitMix64, Key::State(state)) => {
            Ok(Words::W64(splitmix_words(state ^ salt, count)))
        }
        (Algorithm::ThreeFry2x32, Key::Words(words)) => {
            let (k0, k1) = threefry_key(words, salt);
            let calls = count.div_ceil(2);
            let (x0, x1) = threefry2x32(&counters(calls), k0, k1);
            let mut flat = Vec::with_capacity(2 * calls);
            for index in 0..calls {
                flat.push(x0[index]);
                flat.push(x1[index]);
            }
            flat.truncate(count);
            Ok(Words::W32(flat))
        }
        (Algorithm::Philox4x32Ten, Key::Words(words)) => {
            let key4 = philox_key(words, salt);
            let blocks = count.div_ceil(4);
            let [w0, w1, w2, w3] = philox4x32(&counters(blocks), key4);
            let mut flat = Vec::with_capacity(4 * blocks);
            for index in 0..blocks {
                flat.push(w0[index]);
                flat.push(w1[index]);
                flat.push(w2[index]);
                flat.push(w3[index]);
            }
            flat.truncate(count);
            Ok(Words::W32(flat))
        }
        // `extract_key` pairs the key representation with its algorithm.
        _ => Err(Error::backend(format!(
            "random algorithm {} received a key of the wrong representation",
            algorithm.name()
        ))),
    }
}

/// The two `(count,)` word chains behind one Box–Muller pair.
fn normal_word_pairs(algorithm: Algorithm, key: &Key, salt: u64, count: usize) -> Result<Words> {
    match (algorithm, key) {
        (Algorithm::SplitMix64, Key::State(state)) => {
            let (u1, u2) = splitmix_pairs(state ^ salt, count);
            // Both chains are stored as one interleaved u64 stream so the
            // downstream conversion is uniform: element i = words 2i, 2i+1.
            let mut flat = Vec::with_capacity(2 * count);
            for index in 0..count {
                flat.push(u1[index]);
                flat.push(u2[index]);
            }
            Ok(Words::W64(flat))
        }
        (Algorithm::ThreeFry2x32, Key::Words(words)) => {
            let (k0, k1) = threefry_key(words, salt);
            let (x0, x1) = threefry2x32(&counters(count), k0, k1);
            let mut flat = Vec::with_capacity(2 * count);
            for index in 0..count {
                flat.push(x0[index]);
                flat.push(x1[index]);
            }
            Ok(Words::W32(flat))
        }
        (Algorithm::Philox4x32Ten, Key::Words(words)) => {
            let key4 = philox_key(words, salt);
            let blocks = count.div_ceil(2);
            let [w0, w1, w2, w3] = philox4x32(&counters(blocks), key4);
            // u1 = interleave(w0, w2), u2 = interleave(w1, w3).
            let mut u1 = Vec::with_capacity(2 * blocks);
            let mut u2 = Vec::with_capacity(2 * blocks);
            for index in 0..blocks {
                u1.push(w0[index]);
                u1.push(w2[index]);
                u2.push(w1[index]);
                u2.push(w3[index]);
            }
            let mut flat = Vec::with_capacity(2 * count);
            for index in 0..count {
                flat.push(u1[index]);
                flat.push(u2[index]);
            }
            Ok(Words::W32(flat))
        }
        // `extract_key` pairs the key representation with its algorithm.
        _ => Err(Error::backend(format!(
            "random algorithm {} received a key of the wrong representation",
            algorithm.name()
        ))),
    }
}

/// float64 uniforms in [0, 1) from stream words, by word width.
fn word_uniforms(words: &Words) -> Vec<f64> {
    match words {
        Words::W64(values) => values
            .iter()
            .map(|word| ((word >> 11) as f64) * INV_2P53)
            .collect(),
        Words::W32(values) => values.iter().map(|word| *word as f64 * INV_2P32).collect(),
    }
}

// --- key extraction --------------------------------------------------------

/// Normalize the key operand for its algorithm (shape/dtype validated).
fn extract_key(op: &str, algorithm: Algorithm, key: &Buffer) -> Result<Key> {
    match algorithm {
        Algorithm::SplitMix64 => {
            if !key.shape.is_empty() {
                return Err(Error::shape(format!(
                    "op '{op}': {op} with algorithm \"splitmix64\" needs a rank-0 int64 key, \
                     got shape {:?}",
                    key.shape
                )));
            }
            Ok(Key::State(as_u64(scalar_values(key)[0]) & MASK64))
        }
        _ => {
            let expected = algorithm.key_words();
            if key.shape != vec![expected] {
                return Err(Error::shape(format!(
                    "op '{op}': random op with algorithm {:?} needs a shape ({expected},) key, \
                     got shape {:?}",
                    algorithm.name(),
                    key.shape
                )));
            }
            if key.dtype != DType::Int32 {
                return Err(Error::dtype(format!(
                    "op '{op}': random op with algorithm {:?} needs an int32 key, got {}",
                    algorithm.name(),
                    key.dtype.name()
                )));
            }
            let words = scalar_values(key)
                .into_iter()
                .map(|value| as_i64(value) as u32)
                .collect();
            Ok(Key::Words(words))
        }
    }
}

/// One child key (the algorithm's own key words) from a 64-bit salt: cipher
/// call 0 with the salt folded into the key words.
fn child_words(key: &Key, salt: u64) -> Result<Vec<u32>> {
    match key {
        Key::Words(words) if words.len() == 2 => {
            let (k0, k1) = threefry_key(words, salt);
            let (x0, x1) = threefry2x32(&[0], k0, k1);
            Ok(vec![x0[0], x1[0]])
        }
        Key::Words(words) if words.len() == 4 => {
            let key4 = philox_key(words, salt);
            let [w0, w1, w2, w3] = philox4x32(&[0], key4);
            Ok(vec![w0[0], w1[0], w2[0], w3[0]])
        }
        _ => Ok(Vec::new()),
    }
}

// --- operand plumbing ------------------------------------------------------

/// Convert one broadcast operand to f64 values (numpy promotion to float64).
fn broadcast_f64(op: &str, operand: &Buffer, shape: &[usize]) -> Result<Vec<f64>> {
    let expanded = broadcast_to(operand, shape).map_err(|err| {
        Error::shape(format!(
            "op '{op}': operand cannot broadcast to the draw shape: {err}"
        ))
    })?;
    Ok(scalar_values(&expanded).into_iter().map(as_f64).collect())
}

/// `(high - low)` per element, computed in the operands' promoted dtype
/// exactly as numpy does before the float64 scaling.
fn promoted_span(op: &str, low: &Buffer, high: &Buffer, shape: &[usize]) -> Result<Vec<f64>> {
    let left = broadcast_to(low, shape)
        .map_err(|err| Error::shape(format!("op '{op}': low cannot broadcast: {err}")))?;
    let right = broadcast_to(high, shape)
        .map_err(|err| Error::shape(format!("op '{op}': high cannot broadcast: {err}")))?;
    let low_values = scalar_values(&left);
    let high_values = scalar_values(&right);
    let dtype = DType::promote(low.dtype, high.dtype);
    let mut span = Vec::with_capacity(shape.iter().product());
    for (low_value, high_value) in low_values.iter().zip(high_values.iter()) {
        // numpy evaluates `high - low` in the operands' promoted dtype.
        span.push(span_scalar(*high_value, *low_value, dtype, op)?);
    }
    Ok(span)
}

/// `a - b` in `dtype` (wrapping ints, rounded floats), returned as f64.
fn span_scalar(a: ScalarValue, b: ScalarValue, dtype: DType, op: &str) -> Result<f64> {
    Ok(match dtype {
        DType::Int8 => (as_i64(a) as i8).wrapping_sub(as_i64(b) as i8) as f64,
        DType::Int16 => (as_i64(a) as i16).wrapping_sub(as_i64(b) as i16) as f64,
        DType::Int32 => (as_i64(a) as i32).wrapping_sub(as_i64(b) as i32) as f64,
        DType::Int64 => as_i64(a).wrapping_sub(as_i64(b)) as f64,
        DType::UInt8 => (as_u64(a) as u8).wrapping_sub(as_u64(b) as u8) as f64,
        DType::UInt16 => (as_u64(a) as u16).wrapping_sub(as_u64(b) as u16) as f64,
        DType::UInt32 => (as_u64(a) as u32).wrapping_sub(as_u64(b) as u32) as f64,
        DType::UInt64 => as_u64(a).wrapping_sub(as_u64(b)) as f64,
        DType::Float16 => f16::from_f32((as_f64(a) as f32) - (as_f64(b) as f32)).to_f64(),
        DType::Float32 => ((as_f64(a) as f32) - (as_f64(b) as f32)) as f64,
        DType::Float64 => as_f64(a) - as_f64(b),
        other => {
            return Err(Error::dtype(format!(
                "op '{op}': cannot scale a draw by operands of dtype {} — expected a real numeric dtype",
                other.name()
            )))
        }
    })
}

// --- kernels ---------------------------------------------------------------

/// `random_key_mix`: derive one child key from the key operand and `salt`.
fn key_mix_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let algorithm = algorithm_of(spec)?;
    let salt = int_attr(spec, "salt")? as u64;
    let key = extract_key(&spec.name, algorithm, &operands[0])?;
    let shape = result_shape(ctx, spec)?;
    let dtype = result_dtype(spec);
    let values: Vec<ScalarValue> = match algorithm {
        Algorithm::SplitMix64 => {
            if dtype != DType::Int64 {
                return Err(Error::dtype(format!(
                    "op '{}': random_key_mix with algorithm \"splitmix64\" produces int64, \
                     the declared result dtype is {}",
                    spec.name,
                    dtype.name()
                )));
            }
            let Key::State(state) = key else {
                return Err(Error::backend(format!(
                    "op '{}': splitmix64 key normalization failed",
                    spec.name
                )));
            };
            vec![ScalarValue::I(mix_scalar(state ^ salt) as i64)]
        }
        _ => {
            if dtype != DType::Int32 {
                return Err(Error::dtype(format!(
                    "op '{}': random_key_mix with algorithm {:?} produces int32, \
                     the declared result dtype is {}",
                    spec.name,
                    algorithm.name(),
                    dtype.name()
                )));
            }
            child_words(&key, salt)?
                .into_iter()
                .map(|word| ScalarValue::I(word as i32 as i64))
                .collect()
        }
    };
    single(build(dtype, shape, &values)?)
}

/// `random_uniform`: `low + u * (high - low)` from the salted word stream.
fn uniform_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let algorithm = algorithm_of(spec)?;
    let draw = draw_shape(ctx, spec)?;
    let shape = result_shape(ctx, spec)?;
    let dtype = result_dtype(spec);
    require_kind(&spec.name, dtype, true)?;
    let key = extract_key(&spec.name, algorithm, &operands[0])?;
    // The stream is drawn for the attribute shape, then broadcast.
    let draw_words = word_uniforms(&stream_words(algorithm, &key, SALT_UNIFORM, numel(&draw))?);
    let u = gather(&draw_words, &broadcast_offsets(&draw, &shape));
    let low = broadcast_f64(&spec.name, &operands[1], &shape)?;
    let span = promoted_span(&spec.name, &operands[1], &operands[2], &shape)?;
    let count = numel(&shape);
    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        values.push(ScalarValue::F(low[index] + u[index] * span[index]));
    }
    single(build(dtype, shape, &values)?)
}

/// `random_normal`: Box–Muller over the two word chains; the documented f32
/// fast path runs the whole tail in float32 when the result dtype is float32.
fn normal_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let algorithm = algorithm_of(spec)?;
    let shape = result_shape(ctx, spec)?;
    let dtype = result_dtype(spec);
    require_kind(&spec.name, dtype, true)?;
    let key = extract_key(&spec.name, algorithm, &operands[0])?;
    let draw = draw_shape(ctx, spec)?;
    // One Box–Muller pair per draw element; the z values broadcast to the
    // result shape afterwards.
    let draw_count = numel(&draw);
    let words = normal_word_pairs(algorithm, &key, SALT_NORMAL, draw_count)?;
    let mean = broadcast_f64(&spec.name, &operands[1], &shape)?;
    let std = broadcast_f64(&spec.name, &operands[2], &shape)?;
    let offsets = broadcast_offsets(&draw, &shape);
    let count = numel(&shape);
    let mut values = Vec::with_capacity(count);
    if dtype == DType::Float32 {
        // The word fractions convert to f32 FIRST, then the 2^-53/2^-32 u1
        // guard, the sqrt/log/cos tail and the mean/std combine all run in
        // float32 (the f64 mean/std round to f32 first).
        let (scale, eps) = match &words {
            Words::W64(_) => (INV_2P53_F32, INV_2P53_F32),
            Words::W32(_) => (INV_2P32_F32, INV_2P32_F32),
        };
        let two_pi = TWO_PI as f32;
        let mut z = Vec::with_capacity(draw_count);
        for index in 0..draw_count {
            let (u1, u2) = word_pair_f32(&words, index, scale);
            let u1 = u1.max(eps);
            let rt = (NEG_TWO_F32 * u1.ln()).sqrt();
            z.push(rt * (two_pi * u2).cos());
        }
        let z = gather(&z, &offsets);
        for index in 0..count {
            let value = (mean[index] as f32) + z[index] * (std[index] as f32);
            values.push(ScalarValue::F(value as f64));
        }
    } else {
        // Exact f64 math (f64/f16 results — f16 casts the f64 result).
        let (scale, step) = match &words {
            Words::W64(_) => (INV_2P53, INV_2P53),
            Words::W32(_) => (INV_2P32, INV_2P32),
        };
        let mut z = Vec::with_capacity(draw_count);
        for index in 0..draw_count {
            let (u1, u2) = word_pair_f64(&words, index, scale);
            let u1 = u1.max(step);
            z.push((-2.0 * u1.ln()).sqrt() * (TWO_PI * u2).cos());
        }
        let z = gather(&z, &offsets);
        for index in 0..count {
            values.push(ScalarValue::F(mean[index] + z[index] * std[index]));
        }
    }
    single(build(dtype, shape, &values)?)
}

/// The `(u1, u2)` f32 word fractions of element `index`.
fn word_pair_f32(words: &Words, index: usize, scale: f32) -> (f32, f32) {
    match words {
        Words::W64(values) => (
            ((values[2 * index] >> 11) as f32) * scale,
            ((values[2 * index + 1] >> 11) as f32) * scale,
        ),
        Words::W32(values) => (
            (values[2 * index] as f32) * scale,
            (values[2 * index + 1] as f32) * scale,
        ),
    }
}

/// The `(u1, u2)` f64 word fractions of element `index`.
fn word_pair_f64(words: &Words, index: usize, scale: f64) -> (f64, f64) {
    match words {
        Words::W64(values) => (
            ((values[2 * index] >> 11) as f64) * scale,
            ((values[2 * index + 1] >> 11) as f64) * scale,
        ),
        Words::W32(values) => (
            (values[2 * index] as f64) * scale,
            (values[2 * index + 1] as f64) * scale,
        ),
    }
}

/// `random_randint`: `low + floor(u * (high - low))` (float scaling — exact
/// for spans below 2^53, no modulo bias).
fn randint_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let algorithm = algorithm_of(spec)?;
    let draw = draw_shape(ctx, spec)?;
    let shape = result_shape(ctx, spec)?;
    let dtype = result_dtype(spec);
    require_kind(&spec.name, dtype, false)?;
    let key = extract_key(&spec.name, algorithm, &operands[0])?;
    // Python casts both bounds to int64 before subtracting them; the bounds
    // and the draws broadcast against the result shape in the arithmetic.
    let low_raw: Vec<i64> = scalar_values(&operands[1])
        .into_iter()
        .map(as_i64)
        .collect();
    let high_raw: Vec<i64> = scalar_values(&operands[2])
        .into_iter()
        .map(as_i64)
        .collect();
    let low = gather(&low_raw, &broadcast_offsets(&operands[1].shape, &shape));
    let high = gather(&high_raw, &broadcast_offsets(&operands[2].shape, &shape));
    let draw_words = word_uniforms(&stream_words(algorithm, &key, SALT_RANDINT, numel(&draw))?);
    let u = gather(&draw_words, &broadcast_offsets(&draw, &shape));
    let count = numel(&shape);
    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        let span = high[index].wrapping_sub(low[index]);
        if span <= 0 {
            return Err(Error::backend(format!(
                "op '{}': random.randint high must be greater than low \
                 (got span <= 0 for low={} high={})",
                spec.name, low[index], high[index]
            )));
        }
        let offset = (u[index] * span as f64).floor();
        values.push(ScalarValue::F(low[index] as f64 + offset));
    }
    single(build(dtype, shape, &values)?)
}

/// The source index of every element of a C-order broadcast to `shape`.
fn broadcast_offsets(source: &[usize], shape: &[usize]) -> Vec<usize> {
    let strides = broadcast_stride_map(source, shape);
    let count = numel(shape);
    let mut offsets = Vec::with_capacity(count);
    for flat in 0..count {
        let mut offset = 0usize;
        let mut rest = flat;
        for axis in (0..shape.len()).rev() {
            let extent = shape[axis];
            let index = if extent == 0 { 0 } else { rest % extent };
            rest /= extent.max(1);
            offset += index * strides[axis];
        }
        offsets.push(offset);
    }
    offsets
}

/// Gather one flat vector through broadcast offsets.
fn gather<T: Copy>(values: &[T], offsets: &[usize]) -> Vec<T> {
    offsets.iter().map(|offset| values[*offset]).collect()
}

/// Right-aligned broadcast strides (`0` for a broadcast axis).
fn broadcast_stride_map(source: &[usize], shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    let mut stride = 1usize;
    let offset = shape.len().saturating_sub(source.len());
    for axis in (0..source.len()).rev() {
        let target = offset + axis;
        strides[target] = if source[axis] == 1 { 0 } else { stride };
        stride *= source[axis];
    }
    strides
}

/// `random_permutation`: stable argsort of the salted word stream.
fn permutation_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    let algorithm = algorithm_of(spec)?;
    let dtype = result_dtype(spec);
    require_kind(&spec.name, dtype, false)?;
    let key = extract_key(&spec.name, algorithm, &operands[0])?;
    let n_values = scalar_values(&operands[1]);
    let Some(n_value) = n_values.first() else {
        return Err(Error::shape(format!(
            "op '{}': random.permutation needs a rank-0 `n` operand, got an empty tensor",
            spec.name
        )));
    };
    let n = as_i64(*n_value);
    if n < 0 {
        return Err(Error::backend(format!(
            "op '{}': random.permutation n must be >= 0, got {n}",
            spec.name
        )));
    }
    let n = n as usize;
    // A runtime-dynamic (unchecked) declared dim takes its extent from the
    // `n` operand; a static one must agree.
    let declared = &spec.result_types[0].shape;
    if declared.len() != 1 {
        return Err(Error::shape(format!(
            "op '{}': random.permutation result must be rank 1, the declared shape \
             is {declared:?}",
            spec.name
        )));
    }
    let shape = if declared[0].is_dynamic() {
        vec![n]
    } else {
        let shape = result_shape(ctx, spec)?;
        if shape != vec![n] {
            return Err(Error::shape(format!(
                "op '{}': random.permutation produced {n} draws but the declared \
                 result shape is {shape:?}",
                spec.name
            )));
        }
        shape
    };
    let mut order: Vec<usize> = (0..n).collect();
    // `sort_by_key` is stable, matching numpy's `argsort(kind="stable")`:
    // colliding words resolve by index order.
    match stream_words(algorithm, &key, SALT_PERMUTATION, n)? {
        Words::W64(words) => order.sort_by_key(|index| words[*index]),
        Words::W32(words) => order.sort_by_key(|index| words[*index]),
    }
    let values: Vec<ScalarValue> = order
        .iter()
        .map(|index| ScalarValue::I(*index as i64))
        .collect();
    single(build(dtype, shape, &values)?)
}

/// `random_multinomial`: inverse-CDF sampling from a 1-D probability vector.
fn multinomial_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    let algorithm = algorithm_of(spec)?;
    let dtype = result_dtype(spec);
    require_kind(&spec.name, dtype, false)?;
    let num_samples = int_attr(spec, "num_samples")?;
    if num_samples < 0 {
        return Err(Error::backend(format!(
            "op '{}': random.multinomial num_samples must be >= 0, got {num_samples}",
            spec.name
        )));
    }
    let shape = result_shape(ctx, spec)?;
    if shape != vec![num_samples as usize] {
        return Err(Error::shape(format!(
            "op '{}': random.multinomial draws {num_samples} sample(s) but the \
             declared result shape is {shape:?}",
            spec.name
        )));
    }
    let key = extract_key(&spec.name, algorithm, &operands[0])?;
    let probabilities = &operands[1];
    if probabilities.rank() != 1 {
        return Err(Error::shape(format!(
            "op '{}': random.multinomial input must be 1-D, got shape {:?}",
            spec.name, probabilities.shape
        )));
    }
    let m = probabilities.shape[0];
    if m == 0 {
        return Err(Error::backend(format!(
            "op '{}': random.multinomial input must be non-empty",
            spec.name
        )));
    }
    let p: Vec<f64> = scalar_values(probabilities)
        .into_iter()
        .map(as_f64)
        .collect();
    let mut total = 0.0f64;
    for probability in &p {
        if *probability < 0.0 {
            return Err(Error::backend(format!(
                "op '{}': random.multinomial input probabilities must be non-negative",
                spec.name
            )));
        }
        total += *probability;
    }
    if (total - 1.0).abs() > 1e-6 {
        return Err(Error::backend(format!(
            "op '{}': random.multinomial input probabilities must sum to 1 (got {total})",
            spec.name
        )));
    }
    let count = numel(&shape);
    let u = word_uniforms(&stream_words(algorithm, &key, SALT_MULTINOMIAL, count)?);
    // Sequential cumulative sum (the `np.cumsum` order).
    let mut cumulative = Vec::with_capacity(m);
    let mut running = 0.0f64;
    for probability in &p {
        running += *probability;
        cumulative.push(running);
    }
    let mut values = Vec::with_capacity(count);
    for draw in &u {
        // `np.searchsorted(cum, u)` (side="left"): first index with cum >= u,
        // clipped defensively to the last bin.
        let mut index = 0usize;
        while index < m && cumulative[index] < *draw {
            index += 1;
        }
        values.push(ScalarValue::I(index.min(m - 1) as i64));
    }
    single(build(dtype, shape, &values)?)
}

pub fn register(map: &mut KernelMap) {
    map.insert("random_key_mix", key_mix_kernel);
    map.insert("random_uniform", uniform_kernel);
    map.insert("random_normal", normal_kernel);
    map.insert("random_randint", randint_kernel);
    map.insert("random_permutation", permutation_kernel);
    map.insert("random_multinomial", multinomial_kernel);
}

#[cfg(test)]
mod tests {
    //! Bit-exactness pins.
    //!
    //! Two independent checks per algorithm:
    //! 1. the kernel cores are compared against a *second*, straightforward
    //!    transcription of the published ciphers written here from the
    //!    Random123 / SplitMix64 definitions (array state, explicit rotate
    //!    helper, repeated-add key bump) — not a copy of the kernel code;
    //! 2. the op kernels are compared against HARD-CODED vectors produced
    //!    once by the authoritative numpy reference
    //!    (`evoxc/backends/numpy/kernels/random.py`), bit-for-bit (f32/f64
    //!    compared by raw bits).
    use super::*;
    use std::rc::Rc;
    use std::sync::Arc;

    use evoxc_core::{Attrs, Dim, Program, ValueType};

    use crate::exec::RunOptions;
    use crate::plan::Plan;

    // --- independent reference ciphers ------------------------------------

    #[allow(clippy::manual_rotate)] // deliberately hand-written in the reference
    fn rotl32(x: u32, r: u32) -> u32 {
        (x << r) | (x >> (32 - r))
    }

    /// Threefry2x32-20 in the published 20-round form (rotation ladder
    /// 13/15/26/6, 17/29/16/24; key injection after every 4th round).
    fn ref_threefry(c: [u32; 2], k: [u32; 2]) -> [u32; 2] {
        let ks = [k[0], k[1], k[0] ^ k[1] ^ 0x1BD1_1BDA];
        let ladder = [13u32, 15, 26, 6, 17, 29, 16, 24];
        let mut x = [c[0].wrapping_add(ks[0]), c[1].wrapping_add(ks[1])];
        for round in 0..20u32 {
            x[0] = x[0].wrapping_add(x[1]);
            x[1] = rotl32(x[1], ladder[(round % 8) as usize]) ^ x[0];
            if round % 4 == 3 {
                let i = round / 4;
                x[0] = x[0].wrapping_add(ks[((i + 1) % 3) as usize]);
                x[1] = x[1]
                    .wrapping_add(ks[((i + 2) % 3) as usize])
                    .wrapping_add(i + 1);
            }
        }
        x
    }

    fn mulhilo(a: u32, b: u32) -> (u32, u32) {
        let product = (a as u64) * (b as u64);
        ((product >> 32) as u32, product as u32)
    }

    /// Philox4x32-10 with the evoxc 4-word key schedule, written as the
    /// two-multiplier round function with an explicit repeated-add key bump.
    fn ref_philox(c: [u32; 4], k: [u32; 4]) -> [u32; 4] {
        let mut x = c;
        for r in 0..10u32 {
            let (hi0, lo0) = mulhilo(0xD251_1F53, x[0]);
            let (hi1, lo1) = mulhilo(0xCD9E_8D57, x[2]);
            let pair = if r % 2 == 0 {
                [k[0], k[1]]
            } else {
                [k[2], k[3]]
            };
            let mut rk = pair;
            for _ in 0..r {
                rk[0] = rk[0].wrapping_add(0x9E37_79B9);
                rk[1] = rk[1].wrapping_add(0xBB67_AE85);
            }
            x = [hi1 ^ x[1] ^ rk[0], lo1, hi0 ^ x[3] ^ rk[1], lo0];
        }
        x
    }

    /// SplitMix64 as a stateful generator: advance by GOLDEN, then the
    /// three-step mix (no extra increment inside the mixer).
    fn ref_splitmix(seed: u64, count: usize) -> Vec<u64> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^ (z >> 31)
            })
            .collect()
    }

    // --- kernel harness ---------------------------------------------------

    fn test_ctx(program: &Program) -> RunCtx<'_> {
        let plan = Plan {
            blocks: Vec::new(),
            functions: Vec::new(),
            entry: 0,
            max_value: 0,
            op_count: 0,
            block_count: 0,
            required_blocks: Vec::new(),
            required_external: Vec::new(),
            required_runtime: Vec::new(),
            unsupported: Vec::new(),
        };
        RunCtx::new(program, Arc::new(plan), RunOptions::default(), None)
    }

    fn value(dtype: DType, shape: Vec<usize>, values: &[ScalarValue]) -> Value {
        Rc::new(build(dtype, shape, values).expect("test buffer"))
    }

    fn key_splitmix(state: i64) -> Value {
        value(DType::Int64, vec![], &[ScalarValue::I(state)])
    }

    fn key_words(words: &[i32]) -> Value {
        let values: Vec<ScalarValue> = words.iter().map(|w| ScalarValue::I(*w as i64)).collect();
        value(DType::Int32, vec![words.len()], &values)
    }

    fn scalar_f64(number: f64) -> Value {
        value(DType::Float64, vec![], &[ScalarValue::F(number)])
    }

    fn scalar_i64(number: i64) -> Value {
        value(DType::Int64, vec![], &[ScalarValue::I(number)])
    }

    fn spec(name: &str, attrs: &[(&str, AttrValue)], result: (DType, Vec<usize>)) -> OpSpec {
        // The draw shape is the declared result shape in these fixtures (the
        // parameter operands are scalars, so it is not broadcast up).
        let mut pairs: Vec<(&str, AttrValue)> = vec![(
            "shape",
            AttrValue::Shape(result.1.iter().map(|dim| Dim::int(*dim as i64)).collect()),
        )];
        pairs.extend(attrs.iter().map(|(key, val)| (*key, val.clone())));
        spec_attrs(name, Attrs::from_pairs(pairs), result)
    }

    fn spec_attrs(name: &str, attrs: Attrs, result: (DType, Vec<usize>)) -> OpSpec {
        OpSpec {
            name: name.to_string(),
            attrs,
            operand_types: Vec::new(),
            result_types: vec![ValueType::new(
                result.0,
                result.1.iter().map(|dim| Dim::int(*dim as i64)).collect(),
            )],
            regions: Vec::new(),
            location: None,
        }
    }

    fn shape_attr(dims: &[usize]) -> AttrValue {
        AttrValue::Shape(dims.iter().map(|dim| Dim::int(*dim as i64)).collect())
    }

    fn algorithm(name: &str) -> (&'static str, AttrValue) {
        ("algorithm", AttrValue::Str(name.to_string()))
    }

    fn f64_of(results: &[Value]) -> Vec<f64> {
        scalar_values(&results[0]).into_iter().map(as_f64).collect()
    }

    fn i64_of(results: &[Value]) -> Vec<i64> {
        scalar_values(&results[0]).into_iter().map(as_i64).collect()
    }

    fn f64_bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    fn f32_bits(values: &[f64]) -> Vec<u32> {
        values.iter().map(|v| (*v as f32).to_bits()).collect()
    }

    fn f16_bits(results: &[Value]) -> Vec<u16> {
        match &results[0].storage {
            Storage::F16(values) => values.iter().map(|v| v.to_bits()).collect(),
            other => panic!("expected float16 storage, got {other:?}"),
        }
    }

    /// Assert a float32 result is within `ulps` of the pinned numpy bits.
    ///
    /// The f32 Box–Muller fast path is the ONE place where bit-equality with
    /// numpy is not attainable: numpy's float32 `log`/`cos` run through its
    /// own SIMD kernels, which are not correctly rounded (measured on numpy
    /// 2.5: `np.log(np.float32(x))` differs from the correctly rounded value
    /// by 1 ulp for some inputs, while glibc's `logf`/`cosf` — what Rust's
    /// `f32::ln`/`f32::cos` lower to — agree with the correctly rounded
    /// result). The word stream, the word→f32 conversion, the 2^-32/2^-53
    /// scaling and the u1 guard ARE bit-exact and pinned exactly elsewhere.
    fn assert_f32_within(actual: &[f64], expected: &[u32], ulps: u32) {
        let got = f32_bits(actual);
        assert_eq!(got.len(), expected.len(), "result length");
        for (index, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            let distance = (*a as i64 - *b as i64).abs();
            assert!(
                distance <= ulps as i64,
                "element {index}: got {a:#010X}, expected {b:#010X} ({distance} ulp away)",
            );
        }
    }

    // --- ciphers ----------------------------------------------------------

    #[test]
    fn splitmix_matches_reference_stream() {
        let reference = ref_splitmix(0xDEAD_BEEF, 32);
        assert_eq!(splitmix_words(0xDEAD_BEEF, 32), reference);
        // A different seed yields a different stream.
        assert_ne!(
            splitmix_words(0xDEAD_BEEF, 32),
            splitmix_words(0xDEAD_BEEE, 32)
        );
        // mix_scalar is the stream's first word (state starts at `seed`).
        assert_eq!(mix_scalar(0xDEAD_BEEF), reference[0]);
    }

    #[test]
    fn threefry_matches_kat_and_reference() {
        // Random123 known-answer test: key 0, counter 0.
        let (x0, x1) = threefry2x32(&[0], 0, 0);
        assert_eq!((x0[0], x1[0]), (0x6B20_0159, 0x99BA_4EFE));
        for (counter, (k0, k1)) in [(0u32, (0u32, 0u32)), (1, (1, 2)), (7, (0xFFFF_FFFF, 3))] {
            let (x0, x1) = threefry2x32(&[counter], k0, k1);
            assert_eq!([x0[0], x1[0]], ref_threefry([counter, 0], [k0, k1]));
        }
    }

    #[test]
    fn philox_matches_kat_and_reference() {
        // Random123 known-answer test: zero key, zero counter (the evoxc
        // schedule reduces to the standard one at key 0).
        let blocks = philox4x32(&[0], [0, 0, 0, 0]);
        assert_eq!(
            [blocks[0][0], blocks[1][0], blocks[2][0], blocks[3][0]],
            [0x6627_E8D5, 0xE169_C58D, 0xBC57_AC4C, 0x9B00_DBD8]
        );
        let key = [1u32, 2, 3, 4];
        for counter in [0u32, 1, 2] {
            let blocks = philox4x32(&[counter], key);
            assert_eq!(
                [blocks[0][0], blocks[1][0], blocks[2][0], blocks[3][0]],
                ref_philox([counter, 0, 0, 0], key)
            );
        }
    }

    // --- word streams (hard-coded numpy reference vectors) ----------------

    #[test]
    fn splitmix_word_stream_vector() {
        // numpy: _words(seed=0, count=8)
        let expected: [u64; 8] = [
            0xE220_A839_7B1D_CDAF,
            0x6E78_9E6A_A1B9_65F4,
            0x06C4_5D18_8009_454F,
            0xF88B_B8A8_724C_81EC,
            0x1B39_896A_51A8_749B,
            0x53CB_9F0C_747E_A2EA,
            0x2C82_9ABE_1F45_32E1,
            0xC584_133A_C916_AB3C,
        ];
        assert_eq!(splitmix_words(0, 8), expected);
    }

    #[test]
    fn threefry_word_stream_vector() {
        let key = Key::Words(vec![1, 2]);
        let Words::W32(words) =
            stream_words(Algorithm::ThreeFry2x32, &key, SALT_UNIFORM, 10).expect("stream")
        else {
            panic!("threefry words are 32-bit");
        };
        assert_eq!(
            words,
            vec![
                0x2E3D_96D2,
                0x53B1_5DC2,
                0xF532_3862,
                0x9975_089D,
                0xAC7A_4DE2,
                0xFC6D_1697,
                0x3BC5_F543,
                0x014F_66B0,
                0xDA04_D5EC,
                0x9369_F540,
            ]
        );
    }

    #[test]
    fn philox_word_stream_vector() {
        let key = Key::Words(vec![1, 2, 3, 4]);
        let Words::W32(words) =
            stream_words(Algorithm::Philox4x32Ten, &key, SALT_UNIFORM, 10).expect("stream")
        else {
            panic!("philox words are 32-bit");
        };
        assert_eq!(
            words,
            vec![
                0x0CFF_3564,
                0x318F_514A,
                0x33D3_E1D8,
                0x9FFE_01E7,
                0x8E87_3BE8,
                0x6CF1_373B,
                0x5934_92E3,
                0x655E_CB1A,
                0x85A4_AD0A,
                0x3C47_B406,
            ]
        );
    }

    #[test]
    fn splitmix_salt_folds_into_the_key() {
        let key = Key::State(0x0123_4567_89AB_CDEF);
        let Words::W64(words) =
            stream_words(Algorithm::SplitMix64, &key, SALT_UNIFORM, 8).expect("stream")
        else {
            panic!("splitmix words are 64-bit");
        };
        assert_eq!(
            words,
            vec![
                0xE2F9_80B0_4779_2D88,
                0x7117_877F_C7B9_86B3,
                0x7B3C_E6D7_641B_8D63,
                0x4782_29BD_E57D_A5C7,
                0xBEF5_8BEA_9B7D_7FAD,
                0xF81C_3187_FFFD_F513,
                0x22E0_38CE_DD45_6EE3,
                0x0989_6448_1256_C2E8,
            ]
        );
        // The salt changes the stream (key folding, never the counter).
        let Words::W64(other) =
            stream_words(Algorithm::SplitMix64, &key, SALT_NORMAL, 8).expect("stream")
        else {
            panic!("splitmix words are 64-bit");
        };
        assert_ne!(words, other);
    }

    #[test]
    fn normal_word_pairs_vectors() {
        let cases: [(Algorithm, Key, [u32; 5], [u32; 5]); 2] = [
            (
                Algorithm::ThreeFry2x32,
                Key::Words(vec![7, -9i32 as u32]),
                [
                    0x6EC9_36D0,
                    0x943F_9B77,
                    0x899C_3976,
                    0x346B_0537,
                    0x8A60_DB43,
                ],
                [
                    0xC656_3720,
                    0xF448_BEF1,
                    0xB4CC_3912,
                    0x322C_F50A,
                    0x2D0E_4FD1,
                ],
            ),
            (
                Algorithm::Philox4x32Ten,
                Key::Words(vec![11, -13i32 as u32, 17, -19i32 as u32]),
                [
                    0x5FF4_6D9C,
                    0x14E2_8149,
                    0x5590_4213,
                    0x9915_92FB,
                    0x5706_BC8E,
                ],
                [
                    0x1F0D_F1FE,
                    0xB26D_7EB5,
                    0x1141_D401,
                    0x1198_36FD,
                    0x2512_187A,
                ],
            ),
        ];
        for (algorithm, key, expected_u1, expected_u2) in cases {
            let Words::W32(flat) =
                normal_word_pairs(algorithm, &key, SALT_NORMAL, 5).expect("pairs")
            else {
                panic!("counter algorithms produce 32-bit word pairs");
            };
            for index in 0..5 {
                assert_eq!(flat[2 * index], expected_u1[index], "u1 word {index}");
                assert_eq!(flat[2 * index + 1], expected_u2[index], "u2 word {index}");
            }
        }
        let Words::W64(flat) = normal_word_pairs(
            Algorithm::SplitMix64,
            &Key::State(0xDEAD_BEEF),
            SALT_NORMAL,
            5,
        )
        .expect("pairs") else {
            panic!("splitmix produces 64-bit word pairs");
        };
        let expected_u1: [u64; 5] = [
            0x0DC0_DCFF_60A0_9DAE,
            0x3F0C_BBCA_36D2_DA37,
            0xB882_4D4F_7ED2_240D,
            0xC37F_4E98_F97D_3DFF,
            0xFC06_09B0_8CD5_7AD4,
        ];
        let expected_u2: [u64; 5] = [
            0x9498_BF8C_CB91_20C6,
            0x6E15_810A_530C_FD2A,
            0x0A6C_821D_FF82_2E28,
            0xB86D_DC9C_F255_66A1,
            0xF2C3_D602_0D1F_1D80,
        ];
        for index in 0..5 {
            assert_eq!(flat[2 * index], expected_u1[index], "u1 word {index}");
            assert_eq!(flat[2 * index + 1], expected_u2[index], "u2 word {index}");
        }
    }

    #[test]
    fn child_key_words_vectors() {
        let threefry = child_words(&Key::Words(vec![1, 2]), SALT_NORMAL).expect("child");
        assert_eq!(threefry, vec![0xA74D_C5CD, 0xEEB1_1ED7]);
        let philox = child_words(&Key::Words(vec![1, 2, 3, 4]), SALT_NORMAL).expect("child");
        assert_eq!(
            philox,
            vec![0xADA3_E3BE, 0x7A98_E516, 0xC97A_4B02, 0xFD11_3791]
        );
        // The child key is cipher call 0 with the salt folded in.
        assert_eq!(
            child_words(&Key::Words(vec![0, 0]), 0).expect("child"),
            vec![0x6B20_0159, 0x99BA_4EFE]
        );
    }

    // --- op kernels (hard-coded numpy reference vectors) ------------------

    #[test]
    fn key_mix_vectors() {
        let program = Program::new("t");
        let mut ctx = test_ctx(&program);
        // splitmix64: mix(key ^ salt), rank-0 int64.
        let spec_splitmix = spec(
            "random_key_mix",
            &[algorithm("splitmix64"), ("salt", AttrValue::Int(0))],
            (DType::Int64, vec![]),
        );
        let out = key_mix_kernel(&mut ctx, &spec_splitmix, &[key_splitmix(12345)]).expect("mix");
        assert_eq!(i64_of(&out), vec![2_454_886_589_211_414_944]);

        // threefry2x32: two child words as int32.
        let spec_threefry = spec(
            "random_key_mix",
            &[
                algorithm("threefry2x32"),
                ("salt", AttrValue::Int(SALT_NORMAL as i64)),
            ],
            (DType::Int32, vec![2]),
        );
        let out = key_mix_kernel(&mut ctx, &spec_threefry, &[key_words(&[1, 2])]).expect("mix");
        assert_eq!(i64_of(&out), vec![-1_488_075_315, -290_382_121]);

        // philox4x32_10: four child words as int32.
        let spec_philox = spec(
            "random_key_mix",
            &[
                algorithm("philox4x32_10"),
                ("salt", AttrValue::Int(0xDEAD_BEEF)),
            ],
            (DType::Int32, vec![4]),
        );
        let out = key_mix_kernel(&mut ctx, &spec_philox, &[key_words(&[1, 2, 3, 4])]).expect("mix");
        assert_eq!(
            i64_of(&out),
            vec![-694_148_709, 1_168_582_945, 1_923_107_854, -494_943_585]
        );
    }

    #[test]
    fn uniform_vectors() {
        let program = Program::new("t");
        let mut ctx = test_ctx(&program);

        let spec_splitmix_f32 = spec(
            "random_uniform",
            &[
                algorithm("splitmix64"),
                ("dtype", AttrValue::DType(DType::Float32)),
            ],
            (DType::Float32, vec![2, 3]),
        );
        let out = uniform_kernel(
            &mut ctx,
            &spec_splitmix_f32,
            &[key_splitmix(0), scalar_f64(0.0), scalar_f64(1.0)],
        )
        .expect("uniform");

        assert_eq!(
            f32_bits(&f64_of(&out)),
            vec![
                0x3E32_C3DA,
                0x3F14_1703,
                0x3F5D_5559,
                0x3F5B_AFB1,
                0x3EFC_F136,
                0x3F74_1E56,
            ]
        );

        let spec_splitmix_f64 = spec(
            "random_uniform",
            &[
                algorithm("splitmix64"),
                ("dtype", AttrValue::DType(DType::Float64)),
            ],
            (DType::Float64, vec![2, 3]),
        );
        let out = uniform_kernel(
            &mut ctx,
            &spec_splitmix_f64,
            &[key_splitmix(0), scalar_f64(0.0), scalar_f64(1.0)],
        )
        .expect("uniform");
        assert_eq!(
            f64_bits(&f64_of(&out)),
            vec![
                0x3FC6_587B_4FA5_5F50,
                0x3FE2_82E0_68E4_6291,
                0x3FEB_AAAB_2A0C_13BF,
                0x3FEB_75F6_2A1B_D625,
                0x3FDF_9E26_CB9B_110A,
                0x3FEE_83CA_C6D8_FC9F,
            ]
        );

        // threefry2x32 with a non-unit range (low=-1, high=2).
        let spec_threefry = spec(
            "random_uniform",
            &[
                algorithm("threefry2x32"),
                ("dtype", AttrValue::DType(DType::Float64)),
            ],
            (DType::Float64, vec![4]),
        );
        let out = uniform_kernel(
            &mut ctx,
            &spec_threefry,
            &[key_words(&[7, 9]), scalar_f64(-1.0), scalar_f64(2.0)],
        )
        .expect("uniform");
        assert_eq!(
            f64_bits(&f64_of(&out)),
            vec![
                0x3FFC_91D9_6E90_0000,
                0x3FDE_DA47_9AC0_0000,
                0xBFEA_791D_2C40_0000,
                0x3FDE_3EB8_C980_0000,
            ]
        );

        // philox4x32_10 into float16.
        let spec_philox = spec(
            "random_uniform",
            &[
                algorithm("philox4x32_10"),
                ("dtype", AttrValue::DType(DType::Float16)),
            ],
            (DType::Float16, vec![3]),
        );
        let out = uniform_kernel(
            &mut ctx,
            &spec_philox,
            &[key_words(&[1, 2, 3, 4]), scalar_f64(0.0), scalar_f64(1.0)],
        )
        .expect("uniform");
        assert_eq!(
            f64_of(&out),
            vec![0.05078125, 0.193603515625, 0.202392578125]
        );
    }

    #[test]
    fn normal_vectors_f64_and_f32() {
        let program = Program::new("t");
        let mut ctx = test_ctx(&program);

        // f64 path, splitmix64.
        let spec_splitmix_f64 = spec(
            "random_normal",
            &[
                algorithm("splitmix64"),
                ("dtype", AttrValue::DType(DType::Float64)),
            ],
            (DType::Float64, vec![4]),
        );
        let out = normal_kernel(
            &mut ctx,
            &spec_splitmix_f64,
            &[key_splitmix(42), scalar_f64(0.5), scalar_f64(2.0)],
        )
        .expect("normal");
        assert_eq!(
            f64_bits(&f64_of(&out)),
            vec![
                0xBFF8_1B6F_2CF0_45B2,
                0x3FF3_D2A8_7017_708A,
                0x4000_CF34_519F_B996,
                0xC007_917E_057D_AE3B,
            ]
        );

        // f32 fast path, splitmix64 (same words 2i/2i+1, f32 tail).
        let spec_splitmix_f32 = spec(
            "random_normal",
            &[
                algorithm("splitmix64"),
                ("dtype", AttrValue::DType(DType::Float32)),
            ],
            (DType::Float32, vec![4]),
        );
        let out = normal_kernel(
            &mut ctx,
            &spec_splitmix_f32,
            &[key_splitmix(42), scalar_f64(0.5), scalar_f64(2.0)],
        )
        .expect("normal");
        assert_f32_within(
            &f64_of(&out),
            &[0xBFC0_DB7A, 0x3F9E_9544, 0x4006_79A2, 0xC03C_8BF1],
            1,
        );

        // f64 and f32, threefry2x32.
        let spec_threefry_f64 = spec(
            "random_normal",
            &[
                algorithm("threefry2x32"),
                ("dtype", AttrValue::DType(DType::Float64)),
            ],
            (DType::Float64, vec![4]),
        );
        let out = normal_kernel(
            &mut ctx,
            &spec_threefry_f64,
            &[key_words(&[1, 2]), scalar_f64(0.5), scalar_f64(2.0)],
        )
        .expect("normal");
        assert_eq!(
            f64_bits(&f64_of(&out)),
            vec![
                0x4001_7229_E10E_C3C2,
                0x3FFF_B655_C6FD_B107,
                0x3FF1_F2F0_282E_6AF6,
                0x4012_94E6_F048_F00C,
            ]
        );

        let spec_threefry_f32 = spec_threefry_f64.clone();
        let mut spec_threefry_f32 = spec_threefry_f32;
        spec_threefry_f32.result_types[0].dtype = DType::Float32;
        let out = normal_kernel(
            &mut ctx,
            &spec_threefry_f32,
            &[key_words(&[1, 2]), scalar_f64(0.5), scalar_f64(2.0)],
        )
        .expect("normal");
        // Element 2 is the measured 1-ulp numpy SIMD-log divergence
        // (correctly rounded: 0x3F8F_9780).
        assert_f32_within(
            &f64_of(&out),
            &[0x400B_9150, 0x3FFD_B2AE, 0x3F8F_9781, 0x4094_A738],
            1,
        );

        // f32, philox4x32_10.
        let spec_philox_f32 = spec(
            "random_normal",
            &[
                algorithm("philox4x32_10"),
                ("dtype", AttrValue::DType(DType::Float32)),
            ],
            (DType::Float32, vec![4]),
        );
        let out = normal_kernel(
            &mut ctx,
            &spec_philox_f32,
            &[key_words(&[5, 6, 7, 8]), scalar_f64(0.5), scalar_f64(2.0)],
        )
        .expect("normal");
        assert_f32_within(
            &f64_of(&out),
            &[0x3FA4_3120, 0x3FC7_9CF1, 0xBEB4_50B0, 0x3F03_CF11],
            1,
        );
    }

    #[test]
    fn randint_vectors() {
        let program = Program::new("t");
        let mut ctx = test_ctx(&program);

        let spec_splitmix = spec(
            "random_randint",
            &[
                algorithm("splitmix64"),
                ("dtype", AttrValue::DType(DType::Int32)),
            ],
            (DType::Int32, vec![8]),
        );
        let out = randint_kernel(
            &mut ctx,
            &spec_splitmix,
            &[key_splitmix(5), scalar_i64(-3), scalar_i64(7)],
        )
        .expect("randint");
        assert_eq!(i64_of(&out), vec![5, 6, 4, 0, 5, 5, 4, 6]);

        let spec_threefry = spec(
            "random_randint",
            &[
                algorithm("threefry2x32"),
                ("dtype", AttrValue::DType(DType::Int64)),
            ],
            (DType::Int64, vec![5]),
        );
        let out = randint_kernel(
            &mut ctx,
            &spec_threefry,
            &[key_words(&[1, 2]), scalar_i64(0), scalar_i64(1_000_000)],
        )
        .expect("randint");
        assert_eq!(
            i64_of(&out),
            vec![711_382, 831_656, 779_912, 239_786, 125_157]
        );
    }

    #[test]
    fn permutation_vectors() {
        let program = Program::new("t");
        let mut ctx = test_ctx(&program);

        let spec_splitmix = spec(
            "random_permutation",
            &[
                algorithm("splitmix64"),
                ("dtype", AttrValue::DType(DType::Int32)),
            ],
            (DType::Int32, vec![8]),
        );
        let out = permutation_kernel(&mut ctx, &spec_splitmix, &[key_splitmix(3), scalar_i64(8)])
            .expect("permutation");
        assert_eq!(i64_of(&out), vec![6, 2, 7, 0, 3, 1, 4, 5]);

        let spec_threefry = spec(
            "random_permutation",
            &[
                algorithm("threefry2x32"),
                ("dtype", AttrValue::DType(DType::Int64)),
            ],
            (DType::Int64, vec![9]),
        );
        let out = permutation_kernel(
            &mut ctx,
            &spec_threefry,
            &[key_words(&[4, 5]), scalar_i64(9)],
        )
        .expect("permutation");
        assert_eq!(i64_of(&out), vec![0, 2, 3, 5, 4, 8, 6, 1, 7]);

        let spec_philox = spec(
            "random_permutation",
            &[
                algorithm("philox4x32_10"),
                ("dtype", AttrValue::DType(DType::Int32)),
            ],
            (DType::Int32, vec![10]),
        );
        let out = permutation_kernel(
            &mut ctx,
            &spec_philox,
            &[key_words(&[1, 2, 3, 4]), scalar_i64(10)],
        )
        .expect("permutation");
        assert_eq!(i64_of(&out), vec![8, 2, 0, 6, 3, 9, 5, 7, 4, 1]);
    }

    #[test]
    fn multinomial_vectors() {
        let program = Program::new("t");
        let mut ctx = test_ctx(&program);
        let probabilities = value(
            DType::Float64,
            vec![4],
            &[0.1, 0.2, 0.3, 0.4]
                .iter()
                .map(|p| ScalarValue::F(*p))
                .collect::<Vec<_>>(),
        );
        let spec_splitmix = spec(
            "random_multinomial",
            &[
                algorithm("splitmix64"),
                ("num_samples", AttrValue::Int(5)),
                ("dtype", AttrValue::DType(DType::Int32)),
            ],
            (DType::Int32, vec![5]),
        );
        let out = multinomial_kernel(&mut ctx, &spec_splitmix, &[key_splitmix(11), probabilities])
            .expect("multinomial");
        assert_eq!(i64_of(&out), vec![1, 3, 2, 3, 2]);

        let probabilities = value(
            DType::Float32,
            vec![3],
            &[0.25, 0.25, 0.5]
                .iter()
                .map(|p| ScalarValue::F(*p))
                .collect::<Vec<_>>(),
        );
        let spec_threefry = spec(
            "random_multinomial",
            &[
                algorithm("threefry2x32"),
                ("num_samples", AttrValue::Int(6)),
                ("dtype", AttrValue::DType(DType::Int64)),
            ],
            (DType::Int64, vec![6]),
        );
        let out = multinomial_kernel(
            &mut ctx,
            &spec_threefry,
            &[key_words(&[2, 3]), probabilities],
        )
        .expect("multinomial");
        assert_eq!(i64_of(&out), vec![2, 2, 0, 1, 2, 1]);
    }

    #[test]
    fn draws_broadcast_from_the_attribute_shape() {
        // The word stream is drawn for the `shape` attribute and then
        // broadcast against the parameter operands (the reference kernel
        // computes `low + u * (high - low)` with numpy broadcasting).
        let program = Program::new("t");
        let mut ctx = test_ctx(&program);

        let low = value(
            DType::Float64,
            vec![2, 3],
            &[0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
                .iter()
                .map(|v| ScalarValue::F(*v))
                .collect::<Vec<_>>(),
        );
        let spec_uniform = spec_attrs(
            "random_uniform",
            Attrs::from_pairs([
                ("shape", shape_attr(&[3])),
                ("dtype", AttrValue::DType(DType::Float64)),
                algorithm("splitmix64"),
            ]),
            (DType::Float64, vec![2, 3]),
        );
        let out = uniform_kernel(
            &mut ctx,
            &spec_uniform,
            &[key_splitmix(0), low, scalar_f64(1.0)],
        )
        .expect("uniform");
        assert_eq!(
            f64_bits(&f64_of(&out)),
            vec![
                0x3FE2_CB0F_69F4_ABEA,
                0x3FF3_5F47_E5C6_E75C,
                0x3FF3_3FFF_A076_F131,
                0x4008_822C_BB8E_291C,
                0x4003_CD7B_A438_29C1,
                0x3FF9_BFFE_E164_D392,
            ]
        );

        let mean = value(
            DType::Float64,
            vec![2, 2],
            &[0.5, -0.5, 1.5, -1.5]
                .iter()
                .map(|v| ScalarValue::F(*v))
                .collect::<Vec<_>>(),
        );
        let spec_normal = spec_attrs(
            "random_normal",
            Attrs::from_pairs([
                ("shape", shape_attr(&[2])),
                ("dtype", AttrValue::DType(DType::Float64)),
                algorithm("splitmix64"),
            ]),
            (DType::Float64, vec![2, 2]),
        );
        let out = normal_kernel(
            &mut ctx,
            &spec_normal,
            &[key_splitmix(7), mean, scalar_f64(1.0)],
        )
        .expect("normal");
        assert_eq!(
            f64_bits(&f64_of(&out)),
            vec![
                0x3FF7_19B7_82C5_69EF,
                0xC000_1CA3_E4E5_CB96,
                0x4003_8CDB_C162_B4F8,
                0xC008_1CA3_E4E5_CB96,
            ]
        );

        let low = value(
            DType::Int64,
            vec![2, 2],
            &[0, 10, 20, 30]
                .iter()
                .map(|v| ScalarValue::I(*v))
                .collect::<Vec<_>>(),
        );
        let high = value(
            DType::Int64,
            vec![2, 2],
            &[5, 15, 25, 35]
                .iter()
                .map(|v| ScalarValue::I(*v))
                .collect::<Vec<_>>(),
        );
        let spec_randint = spec_attrs(
            "random_randint",
            Attrs::from_pairs([
                ("shape", shape_attr(&[2])),
                ("dtype", AttrValue::DType(DType::Int32)),
                algorithm("threefry2x32"),
            ]),
            (DType::Int32, vec![2, 2]),
        );
        let out = randint_kernel(&mut ctx, &spec_randint, &[key_words(&[1, 2]), low, high])
            .expect("randint");
        assert_eq!(i64_of(&out), vec![3, 14, 23, 34]);
    }

    #[test]
    fn float16_and_integer_bound_vectors() {
        let program = Program::new("t");
        let mut ctx = test_ctx(&program);

        // float16 output takes the exact f64 path and casts at the end.
        let spec_f16 = spec(
            "random_normal",
            &[
                algorithm("splitmix64"),
                ("dtype", AttrValue::DType(DType::Float16)),
            ],
            (DType::Float16, vec![5]),
        );
        let out = normal_kernel(
            &mut ctx,
            &spec_f16,
            &[key_splitmix(42), scalar_f64(0.5), scalar_f64(2.0)],
        )
        .expect("normal");
        assert_eq!(f16_bits(&out), vec![0xBE07, 0x3CF5, 0x4034, 0xC1E4, 0x36B6]);

        // Integer bounds: `high - low` is integer arithmetic (7) before the
        // float64 scaling.
        let spec_int_bounds = spec(
            "random_uniform",
            &[
                algorithm("splitmix64"),
                ("dtype", AttrValue::DType(DType::Float64)),
            ],
            (DType::Float64, vec![4]),
        );
        let out = uniform_kernel(
            &mut ctx,
            &spec_int_bounds,
            &[key_splitmix(0), scalar_i64(3), scalar_i64(10)],
        )
        .expect("uniform");
        assert_eq!(
            f64_bits(&f64_of(&out)),
            vec![
                0x4010_E35A_F96C_2CDA,
                0x401C_3284_5BC7_D63F,
                0x4022_1AAA_E265_48A4,
                0x4022_039B_B26C_2DB0,
            ]
        );

        // Mixed float32 low / float64 high: the span is promoted to float64.
        let low_f32 = value(DType::Float32, vec![], &[ScalarValue::F(-0.5)]);
        let spec_mixed = spec(
            "random_uniform",
            &[
                algorithm("threefry2x32"),
                ("dtype", AttrValue::DType(DType::Float32)),
            ],
            (DType::Float32, vec![3]),
        );
        let out = uniform_kernel(
            &mut ctx,
            &spec_mixed,
            &[key_words(&[9, 8]), low_f32, scalar_f64(1.0)],
        )
        .expect("uniform");
        assert_eq!(
            f32_bits(&f64_of(&out)),
            vec![0xBD0D_A861, 0xBEA7_B716, 0xBE90_3C20]
        );
    }

    // --- determinism and explicit errors ---------------------------------

    #[test]
    fn same_key_is_bit_identical_and_keys_differ() {
        let program = Program::new("t");
        let mut ctx = test_ctx(&program);
        let spec_threefry = spec(
            "random_normal",
            &[
                algorithm("threefry2x32"),
                ("dtype", AttrValue::DType(DType::Float32)),
            ],
            (DType::Float32, vec![6]),
        );
        let run = |ctx: &mut RunCtx<'_>, key: Value| {
            let out = normal_kernel(
                ctx,
                &spec_threefry,
                &[key, scalar_f64(0.0), scalar_f64(1.0)],
            )
            .expect("normal");
            f64_of(&out)
        };
        let first = run(&mut ctx, key_words(&[3, 4]));
        let second = run(&mut ctx, key_words(&[3, 4]));
        let other = run(&mut ctx, key_words(&[3, 5]));
        assert_eq!(f32_bits(&first), f32_bits(&second), "same key must replay");
        assert_ne!(
            f32_bits(&first),
            f32_bits(&other),
            "a different key is a new stream"
        );
    }

    #[test]
    fn explicit_errors_name_the_op_and_algorithm() {
        let program = Program::new("t");
        let mut ctx = test_ctx(&program);

        // Unknown algorithm.
        let bad = spec(
            "random_uniform",
            &[
                algorithm("xoshiro"),
                ("dtype", AttrValue::DType(DType::Float32)),
            ],
            (DType::Float32, vec![2]),
        );
        let err = uniform_kernel(
            &mut ctx,
            &bad,
            &[key_splitmix(0), scalar_f64(0.0), scalar_f64(1.0)],
        )
        .expect_err("unknown algorithm");
        assert!(err.to_string().contains("xoshiro"), "{err}");

        // Counter-based key with the wrong shape.
        let bad = spec(
            "random_uniform",
            &[
                algorithm("threefry2x32"),
                ("dtype", AttrValue::DType(DType::Float32)),
            ],
            (DType::Float32, vec![2]),
        );
        let err = uniform_kernel(
            &mut ctx,
            &bad,
            &[key_words(&[1, 2, 3]), scalar_f64(0.0), scalar_f64(1.0)],
        )
        .expect_err("wrong key shape");
        assert!(err.to_string().contains("threefry2x32"), "{err}");

        // Counter-based key with the wrong dtype (right shape, float dtype).
        let float_key = value(
            DType::Float32,
            vec![2],
            &[ScalarValue::F(1.0), ScalarValue::F(2.0)],
        );
        let err = uniform_kernel(
            &mut ctx,
            &bad,
            &[float_key, scalar_f64(0.0), scalar_f64(1.0)],
        )
        .expect_err("wrong key dtype");
        assert!(err.to_string().contains("int32"), "{err}");

        // Non-floating uniform output dtype.
        let bad = spec(
            "random_uniform",
            &[
                algorithm("splitmix64"),
                ("dtype", AttrValue::DType(DType::Int32)),
            ],
            (DType::Int32, vec![2]),
        );
        let err = uniform_kernel(
            &mut ctx,
            &bad,
            &[key_splitmix(0), scalar_f64(0.0), scalar_f64(1.0)],
        )
        .expect_err("integer uniform output");
        assert!(err.to_string().contains("floating"), "{err}");

        // randint with a non-positive span.
        let bad = spec(
            "random_randint",
            &[
                algorithm("splitmix64"),
                ("dtype", AttrValue::DType(DType::Int32)),
            ],
            (DType::Int32, vec![3]),
        );
        let err = randint_kernel(
            &mut ctx,
            &bad,
            &[key_splitmix(0), scalar_i64(7), scalar_i64(7)],
        )
        .expect_err("empty range");
        assert!(err.to_string().contains("greater than low"), "{err}");

        // multinomial with a probability vector that does not sum to 1.
        let probabilities = value(
            DType::Float64,
            vec![2],
            &[0.2, 0.2]
                .iter()
                .map(|p| ScalarValue::F(*p))
                .collect::<Vec<_>>(),
        );
        let bad = spec(
            "random_multinomial",
            &[
                algorithm("splitmix64"),
                ("num_samples", AttrValue::Int(2)),
                ("dtype", AttrValue::DType(DType::Int32)),
            ],
            (DType::Int32, vec![2]),
        );
        let err = multinomial_kernel(&mut ctx, &bad, &[key_splitmix(0), probabilities])
            .expect_err("unnormalized probabilities");
        assert!(err.to_string().contains("sum to 1"), "{err}");
    }
}
