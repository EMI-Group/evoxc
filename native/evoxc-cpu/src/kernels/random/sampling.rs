//! Sampling kernels: word→value conversion and the `random_key_mix`,
//! `random_uniform`, `random_normal` and `random_randint` draws.
use evoxc_core::{DType, Error, Result};
use half::f16;

use crate::buffer::{broadcast_to, Buffer};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util::numel;
use crate::plan::OpSpec;
use crate::scalar::ScalarValue;

use super::bits::{mix_scalar, normal_word_pairs, stream_words};
use super::key::{child_words, extract_key, Algorithm, Key, Words};
use super::{
    algorithm_of, as_f64, as_i64, as_u64, build, draw_shape, int_attr, require_kind, result_dtype,
    result_shape, scalar_values, SALT_NORMAL, SALT_RANDINT, SALT_UNIFORM,
};

// --- stream-conversion scales ----------------------------------------------

const INV_2P53: f64 = 1.0 / 9_007_199_254_740_992.0; // 2^-53, exact
const INV_2P32: f64 = 1.0 / 4_294_967_296.0; // 2^-32, exact
const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// f32 fast-path constants (the exact f32 values the numpy kernel uses).
const INV_2P53_F32: f32 = 1.0 / 9_007_199_254_740_992.0; // 2^-53 as f32
const INV_2P32_F32: f32 = 1.0 / 4_294_967_296.0; // 2^-32 as f32
const NEG_TWO_F32: f32 = -2.0;

// --- word conversion -------------------------------------------------------

/// float64 uniforms in [0, 1) from stream words, by word width.
pub(super) fn word_uniforms(words: &Words) -> Vec<f64> {
    match words {
        Words::W64(values) => values
            .iter()
            .map(|word| ((word >> 11) as f64) * INV_2P53)
            .collect(),
        Words::W32(values) => values.iter().map(|word| *word as f64 * INV_2P32).collect(),
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
pub(super) fn key_mix_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
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
pub(super) fn uniform_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
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
pub(super) fn normal_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
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
pub(super) fn randint_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
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

#[cfg(test)]
mod tests {
    //! Hard-coded numpy reference vectors for the sampler kernels.
    use super::*;
    use crate::kernels::random::permutation::multinomial_kernel;
    use crate::kernels::random::test_util::*;
    use evoxc_core::{AttrValue, Attrs, DType, Program};

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
