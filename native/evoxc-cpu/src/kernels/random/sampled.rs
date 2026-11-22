//! Typed, contiguous fast paths for the draws whose declared `shape` attribute
//! already equals the result shape — the overwhelmingly common case.
//!
//! The generic path in `sampling.rs` builds a whole-array chain per call:
//! `stream_words` -> `word_uniforms` -> `broadcast_offsets` -> `gather` ->
//! dtype-erased `ScalarValue` output -> `build`.  When the draw shape IS the
//! result shape that chain is pure overhead: the broadcast is the identity,
//! every parameter is either a scalar or already result-shaped, and each value
//! can go straight from a stream word into a typed `Vec<T>`.
//!
//! Bit-exactness is the binding contract, so the expressions here are the SAME
//! expressions, in the same order and the same dtypes, as `sampling.rs`; only
//! the buffer plumbing differs.  Every case the typed path does not cover (a
//! parameter with any other broadcast shape, a draw shape that differs from the
//! result shape, a dtype the op rejects) returns `Ok(None)` and the generic
//! kernel runs unchanged.
use evoxc_core::{DType, Error, Result};
use half::f16;

use crate::buffer::{Buffer, Storage};
use crate::scalar::{Scalar, ScalarValue};
use crate::with_type;

use super::bits::{for_each_normal_pair, for_each_word, words_are_64_bit};
use super::key::{Algorithm, Key};
use super::sampling::{
    broadcast_f64, broadcast_offsets, gather, promoted_span, span_scalar, INV_2P32, INV_2P32_F32,
    INV_2P53, INV_2P53_F32, NEG_TWO_F32, TWO_PI,
};
use super::{as_f64, as_i64, scalar_values, SALT_NORMAL, SALT_RANDINT, SALT_UNIFORM};

/// `TWO_PI` as the exact `f32` the generic f32 fast path uses.
const TWO_PI_F32: f32 = TWO_PI as f32;
#[cfg(test)]
thread_local! {
    /// Test-only bypass that forces the generic kernels, so a test can compare
    /// a typed fast path against the untouched generic path byte for byte.
    /// Thread-local: `cargo test` runs each test on its own thread.
    pub(super) static FORCE_GENERIC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
/// False only inside a test that set [`FORCE_GENERIC`].
#[inline]
fn fast_path_enabled() -> bool {
    #[cfg(test)]
    {
        !FORCE_GENERIC.with(std::cell::Cell::get)
    }
    #[cfg(not(test))]
    {
        true
    }
}

// --- parameters ------------------------------------------------------------

/// One real-valued operand in its pre-broadcast form.
enum Param {
    /// The operand has exactly one element: its single value.
    Scalar(ScalarValue),
    /// The operand already has the result shape: its raw per-element values
    /// (kept dtype-erased so the promoted span arithmetic stays integer-exact).
    Dense(Vec<ScalarValue>),
    /// Any other broadcast shape: the materialized f64 expansion.
    Flat(Vec<f64>),
}

impl Param {
    /// The raw value at `index` (a `Scalar` repeats; `Flat` has lost its raw
    /// form and must not reach the span arithmetic — see [`span_param`]).
    fn raw(&self, index: usize) -> ScalarValue {
        match self {
            Param::Scalar(value) => *value,
            Param::Dense(values) => values[index],
            Param::Flat(values) => ScalarValue::F(values[index]),
        }
    }
}

/// A parameter resolved for the sampling loop: a single value, or the
/// result-shaped f64 values.
enum Line {
    Scalar(f64),
    Slice(Vec<f64>),
}

impl Line {
    #[inline]
    fn at(&self, index: usize) -> f64 {
        match self {
            Line::Scalar(value) => *value,
            Line::Slice(values) => values[index],
        }
    }
}

/// A parameter resolved to int64 (the `random_randint` bound convention).
enum IntLine {
    Scalar(i64),
    Slice(Vec<i64>),
}

impl IntLine {
    #[inline]
    fn at(&self, index: usize) -> i64 {
        match self {
            IntLine::Scalar(value) => *value,
            IntLine::Slice(values) => values[index],
        }
    }
}

/// `high - low` per element in the operands' promoted dtype.
enum SpanLine {
    Scalar(i64),
    Slice(Vec<i64>),
}

impl SpanLine {
    #[inline]
    fn at(&self, index: usize) -> i64 {
        match self {
            SpanLine::Scalar(value) => *value,
            SpanLine::Slice(values) => values[index],
        }
    }
}

/// The operand's single element (a `numel() == 1` operand is never empty).
fn first_scalar(operand: &Buffer) -> ScalarValue {
    with_type!(operand, |values| {
        values
            .first()
            .map(|value| value.to_scalar())
            .unwrap_or(ScalarValue::F(0.0))
    })
}

/// Resolve one broadcast operand without materializing a whole-array
/// `ScalarValue` vector when it is a scalar or already result-shaped.
fn param(op: &str, operand: &Buffer, shape: &[usize]) -> Result<Param> {
    // A single-element operand broadcasts onto any shape of at least its own
    // rank; anything else keeps the generic (validated) broadcast path.
    if operand.numel() == 1 && operand.rank() <= shape.len() {
        return Ok(Param::Scalar(first_scalar(operand)));
    }
    if operand.shape == shape {
        return Ok(Param::Dense(scalar_values(operand)));
    }
    Ok(Param::Flat(broadcast_f64(op, operand, shape)?))
}

/// `high - low` for `random_uniform`: the same promoted-dtype subtraction
/// `promoted_span` performs, minus the broadcast when both operands fit the
/// fast forms.
fn span_param(
    op: &str,
    low: &Buffer,
    high: &Buffer,
    low_param: &Param,
    high_param: &Param,
    shape: &[usize],
    count: usize,
) -> Result<Param> {
    let dtype = DType::promote(low.dtype, high.dtype);
    match (low_param, high_param) {
        (Param::Scalar(low_value), Param::Scalar(high_value)) => Ok(Param::Scalar(
            ScalarValue::F(span_scalar(*high_value, *low_value, dtype, op)?),
        )),
        // A materialized broadcast has lost its raw values: let the generic
        // path recompute the span from the operands themselves.
        (Param::Flat(_), _) | (_, Param::Flat(_)) => {
            Ok(Param::Flat(promoted_span(op, low, high, shape)?))
        }
        (low_param, high_param) => {
            let mut span = Vec::with_capacity(count);
            for index in 0..count {
                span.push(span_scalar(
                    high_param.raw(index),
                    low_param.raw(index),
                    dtype,
                    op,
                )?);
            }
            Ok(Param::Flat(span))
        }
    }
}

/// Resolve a bound operand to int64 values on the result shape, exactly like
/// the generic path's `scalar_values` + broadcast-offset gather.
fn int_line(operand: &Buffer, shape: &[usize]) -> IntLine {
    if operand.numel() == 1 && operand.rank() <= shape.len() {
        return IntLine::Scalar(as_i64(first_scalar(operand)));
    }
    let raw: Vec<i64> = scalar_values(operand)
        .into_iter()
        .map(as_i64)
        .collect();
    if operand.shape == shape {
        return IntLine::Slice(raw);
    }
    IntLine::Slice(gather(&raw, &broadcast_offsets(&operand.shape, shape)))
}

/// `high - low` per element with the generic path's `span <= 0` check (the
/// first offending element reports the bounds, in the same words).
fn int_span(op: &str, low: &IntLine, high: &IntLine, count: usize) -> Result<SpanLine> {
    if let (IntLine::Scalar(low_value), IntLine::Scalar(high_value)) = (low, high) {
        let span = high_value.wrapping_sub(*low_value);
        check_span(op, span, *low_value, *high_value)?;
        return Ok(SpanLine::Scalar(span));
    }
    let mut span = Vec::with_capacity(count);
    for index in 0..count {
        let low_value = low.at(index);
        let high_value = high.at(index);
        let value = high_value.wrapping_sub(low_value);
        check_span(op, value, low_value, high_value)?;
        span.push(value);
    }
    Ok(SpanLine::Slice(span))
}

fn check_span(op: &str, span: i64, low: i64, high: i64) -> Result<()> {
    if span <= 0 {
        return Err(Error::backend(format!(
            "op '{op}': random.randint high must be greater than low \
             (got span <= 0 for low={low} high={high})"
        )));
    }
    Ok(())
}

// --- word conversion -------------------------------------------------------

/// One stream word as a float64 uniform in `[0, 1)` (the generic
/// `word_uniforms` conversion, by word width).
#[inline]
fn unit_from_word(word: u64, wide: bool) -> f64 {
    if wide {
        ((word >> 11) as f64) * INV_2P53
    } else {
        (word as f64) * INV_2P32
    }
}

/// One stream word as a float32 uniform (the generic f32 fast path's own
/// conversion: the word converts to f32 FIRST).
#[inline]
fn unit_from_word_f32(word: u64, scale: f32, wide: bool) -> f32 {
    if wide {
        ((word >> 11) as f32) * scale
    } else {
        (word as f32) * scale
    }
}

// --- draws -----------------------------------------------------------------

/// `random_uniform` on a result-shaped draw; `None` keeps the generic kernel.
// One argument per kernel input plus the resolved dtype/shape context —
// the same shape as the kernel it short-circuits.
#[allow(clippy::too_many_arguments)]
pub(super) fn uniform(
    op: &str,
    algorithm: Algorithm,
    key: &Key,
    dtype: DType,
    draw: &[usize],
    shape: &[usize],
    low: &Buffer,
    high: &Buffer,
) -> Result<Option<Buffer>> {
    if !fast_path_enabled() || draw != shape {
        return Ok(None);
    }
    let count = shape.iter().product();
    let low_param = param(op, low, shape)?;
    let high_param = param(op, high, shape)?;
    let span = span_param(op, low, high, &low_param, &high_param, shape, count)?;
    let low = line(low_param);
    let span = line(span);
    let storage = match dtype {
        DType::Float16 => uniform_into::<f16>(algorithm, key, count, &low, &span)?,
        DType::Float32 => uniform_into::<f32>(algorithm, key, count, &low, &span)?,
        DType::Float64 => uniform_into::<f64>(algorithm, key, count, &low, &span)?,
        // `require_kind` already rejected every other dtype.
        _ => return Ok(None),
    };
    Ok(Some(Buffer::new(dtype, shape.to_vec(), storage)?))
}

/// `random_normal` on a result-shaped draw; `None` keeps the generic kernel.
// One argument per kernel input plus the resolved dtype/shape context —
// the same shape as the kernel it short-circuits.
#[allow(clippy::too_many_arguments)]
pub(super) fn normal(
    op: &str,
    algorithm: Algorithm,
    key: &Key,
    dtype: DType,
    draw: &[usize],
    shape: &[usize],
    mean: &Buffer,
    std: &Buffer,
) -> Result<Option<Buffer>> {
    if !fast_path_enabled() || draw != shape {
        return Ok(None);
    }
    let count = shape.iter().product();
    let mean = line(param(op, mean, shape)?);
    let std = line(param(op, std, shape)?);
    let storage = match dtype {
        DType::Float32 => normal_f32_into::<f32>(algorithm, key, count, &mean, &std)?,
        DType::Float16 => normal_f64_into::<f16>(algorithm, key, count, &mean, &std)?,
        DType::Float64 => normal_f64_into::<f64>(algorithm, key, count, &mean, &std)?,
        // `require_kind` already rejected every other dtype.
        _ => return Ok(None),
    };
    Ok(Some(Buffer::new(dtype, shape.to_vec(), storage)?))
}

/// `random_randint` on a result-shaped draw; `None` keeps the generic kernel.
// One argument per kernel input plus the resolved dtype/shape context —
// the same shape as the kernel it short-circuits.
#[allow(clippy::too_many_arguments)]
pub(super) fn randint(
    op: &str,
    algorithm: Algorithm,
    key: &Key,
    dtype: DType,
    draw: &[usize],
    shape: &[usize],
    low: &Buffer,
    high: &Buffer,
) -> Result<Option<Buffer>> {
    if !fast_path_enabled() || draw != shape {
        return Ok(None);
    }
    let count = shape.iter().product();
    let low = int_line(low, shape);
    let high = int_line(high, shape);
    let span = int_span(op, &low, &high, count)?;
    let storage = match dtype {
        DType::Int8 => randint_into::<i8>(algorithm, key, count, &low, &span)?,
        DType::Int16 => randint_into::<i16>(algorithm, key, count, &low, &span)?,
        DType::Int32 => randint_into::<i32>(algorithm, key, count, &low, &span)?,
        DType::Int64 => randint_into::<i64>(algorithm, key, count, &low, &span)?,
        DType::UInt8 => randint_into::<u8>(algorithm, key, count, &low, &span)?,
        DType::UInt16 => randint_into::<u16>(algorithm, key, count, &low, &span)?,
        DType::UInt32 => randint_into::<u32>(algorithm, key, count, &low, &span)?,
        DType::UInt64 => randint_into::<u64>(algorithm, key, count, &low, &span)?,
        // `require_kind` already rejected every other dtype.
        _ => return Ok(None),
    };
    Ok(Some(Buffer::new(dtype, shape.to_vec(), storage)?))
}

fn line(param: Param) -> Line {
    match param {
        Param::Scalar(value) => Line::Scalar(as_f64(value)),
        Param::Dense(values) => Line::Slice(values.into_iter().map(as_f64).collect()),
        Param::Flat(values) => Line::Slice(values),
    }
}

/// `low + u * span`, written straight into the result dtype.
fn uniform_into<T: Scalar>(
    algorithm: Algorithm,
    key: &Key,
    count: usize,
    low: &Line,
    span: &Line,
) -> Result<Storage> {
    let wide = words_are_64_bit(algorithm);
    let mut values: Vec<T> = Vec::with_capacity(count);
    let mut index = 0usize;
    for_each_word(algorithm, key, SALT_UNIFORM, count, |word| {
        let u = unit_from_word(word, wide);
        let value = low.at(index) + u * span.at(index);
        values.push(T::from_scalar(ScalarValue::F(value)));
        index += 1;
    })?;
    Ok(T::into_storage(values))
}

/// The f32 Box–Muller fast path: the word fractions convert to f32 FIRST, then
/// the 2^-53/2^-32 u1 guard, the sqrt/log/cos tail and the mean/std combine all
/// run in float32 (the f64 mean/std round to f32 first).
fn normal_f32_into<T: Scalar>(
    algorithm: Algorithm,
    key: &Key,
    count: usize,
    mean: &Line,
    std: &Line,
) -> Result<Storage> {
    let wide = words_are_64_bit(algorithm);
    let (scale, eps) = if wide {
        (INV_2P53_F32, INV_2P53_F32)
    } else {
        (INV_2P32_F32, INV_2P32_F32)
    };
    let mut values: Vec<T> = Vec::with_capacity(count);
    let mut index = 0usize;
    for_each_normal_pair(algorithm, key, SALT_NORMAL, count, |word1, word2| {
        let u1 = unit_from_word_f32(word1, scale, wide);
        let u2 = unit_from_word_f32(word2, scale, wide);
        let u1 = u1.max(eps);
        let rt = (NEG_TWO_F32 * u1.ln()).sqrt();
        let z = rt * (TWO_PI_F32 * u2).cos();
        let value = (mean.at(index) as f32) + z * (std.at(index) as f32);
        values.push(T::from_scalar(ScalarValue::F(value as f64)));
        index += 1;
    })?;
    Ok(T::into_storage(values))
}

/// The exact f64 Box–Muller chain (f64 and f16 outputs — f16 casts the f64
/// result, exactly like the generic path's `build`).
fn normal_f64_into<T: Scalar>(
    algorithm: Algorithm,
    key: &Key,
    count: usize,
    mean: &Line,
    std: &Line,
) -> Result<Storage> {
    let wide = words_are_64_bit(algorithm);
    let step = if wide { INV_2P53 } else { INV_2P32 };
    let mut values: Vec<T> = Vec::with_capacity(count);
    let mut index = 0usize;
    for_each_normal_pair(algorithm, key, SALT_NORMAL, count, |word1, word2| {
        let u1 = unit_from_word(word1, wide).max(step);
        let u2 = unit_from_word(word2, wide);
        let z = (-2.0 * u1.ln()).sqrt() * (TWO_PI * u2).cos();
        values.push(T::from_scalar(ScalarValue::F(
            mean.at(index) + z * std.at(index),
        )));
        index += 1;
    })?;
    Ok(T::into_storage(values))
}

/// `low + floor(u * span)`, cast to the result dtype like the generic path.
fn randint_into<T: Scalar>(
    algorithm: Algorithm,
    key: &Key,
    count: usize,
    low: &IntLine,
    span: &SpanLine,
) -> Result<Storage> {
    let wide = words_are_64_bit(algorithm);
    let mut values: Vec<T> = Vec::with_capacity(count);
    let mut index = 0usize;
    for_each_word(algorithm, key, SALT_RANDINT, count, |word| {
        let u = unit_from_word(word, wide);
        let offset = (u * (span.at(index) as f64)).floor();
        let value = low.at(index) as f64 + offset;
        values.push(T::from_scalar(ScalarValue::F(value)));
        index += 1;
    })?;
    Ok(T::into_storage(values))
}

#[cfg(test)]
mod tests {
    //! Fast-path agreement: every typed path must produce the SAME bytes as the
    //! generic kernel path it short-circuits, and the same errors.
    //!
    //! The typed path is validated against hard-coded numpy vectors elsewhere
    //! (`sampling.rs`), so this module only has to prove the short-circuit is
    //! behaviour-preserving: each case runs the real kernel twice, once with
    //! [`FORCE_GENERIC`] set, and compares dtype, shape and every byte.
    use super::*;
    use crate::exec::{RunCtx, Value};
    use crate::kernels::random::sampling::{normal_kernel, randint_kernel, uniform_kernel};
    use crate::kernels::random::test_util::*;
    use crate::plan::OpSpec;
    use evoxc_core::{AttrValue, DType, Program};
    type Kernel = fn(&mut RunCtx<'_>, &OpSpec, &[Value]) -> Result<Vec<Value>>;

    /// Run `kernel` on the fast path and on the generic path; assert they agree.
    fn agree(kernel: Kernel, spec: &OpSpec, operands: &[Value]) -> Storage {
        let program = Program::new("t");
        let fast = {
            let mut ctx = test_ctx(&program);
            kernel(&mut ctx, spec, operands).expect("fast path")
        };
        let generic = {
            FORCE_GENERIC.with(|flag| flag.set(true));
            let mut ctx = test_ctx(&program);
            let out = kernel(&mut ctx, spec, operands);
            FORCE_GENERIC.with(|flag| flag.set(false));
            out.expect("generic path")
        };
        assert_eq!(fast.len(), 1, "result count");
        assert_eq!(fast[0].dtype, generic[0].dtype, "dtype");
        assert_eq!(fast[0].shape, generic[0].shape, "shape");
        assert_eq!(
            fast[0].storage, generic[0].storage,
            "storage differs for op '{}' on {:?} {:?}",
            spec.name, spec.result_types[0], operands.len(),
        );
        fast[0].storage.clone()
    }

    /// Run `kernel` on both paths and assert BOTH raise the same message.
    fn agree_error(kernel: Kernel, spec: &OpSpec, operands: &[Value]) {
        let program = Program::new("t");
        let mut ctx = test_ctx(&program);
        let fast = kernel(&mut ctx, spec, operands).expect_err("fast path must fail");
        FORCE_GENERIC.with(|flag| flag.set(true));
        let mut ctx = test_ctx(&program);
        let generic = kernel(&mut ctx, spec, operands);
        FORCE_GENERIC.with(|flag| flag.set(false));
        let generic = generic.expect_err("generic path must fail");
        assert_eq!(fast.to_string(), generic.to_string(), "error text");
    }

    fn float_param(shape: &[usize], dtype: DType, base: f64) -> Value {
        let count: usize = shape.iter().product();
        let values: Vec<ScalarValue> = (0..count)
            .map(|index| ScalarValue::F(base + index as f64 * 0.25))
            .collect();
        value(dtype, shape.to_vec(), &values)
    }

    fn int_param(shape: &[usize], dtype: DType, base: i64) -> Value {
        let count: usize = shape.iter().product();
        let values: Vec<ScalarValue> = (0..count)
            .map(|index| ScalarValue::I(base + index as i64))
            .collect();
        value(dtype, shape.to_vec(), &values)
    }

    fn keys(algorithm: &str) -> Value {
        match algorithm {
            "splitmix64" => key_splitmix(0x1234_5678),
            "threefry2x32" => key_words(&[1, 2]),
            _ => key_words(&[1, 2, 3, 4]),
        }
    }

    const ALGORITHMS: [&str; 3] = ["splitmix64", "threefry2x32", "philox4x32_10"];

    fn shapes() -> Vec<Vec<usize>> {
        vec![
            vec![0],
            vec![1],
            vec![7],
            vec![2, 3],
            vec![2, 3, 4],
        ]
    }

    #[test]
    fn uniform_fast_path_matches_the_generic_path() {
        for dtype in [DType::Float16, DType::Float32, DType::Float64] {
            for alg in ALGORITHMS {
                for shape in shapes() {
                    let count: usize = shape.iter().product();
                    let spec = spec(
                        "random_uniform",
                        &[
                            algorithm(alg),
                            ("dtype", AttrValue::DType(dtype)),
                        ],
                        (dtype, shape.clone()),
                    );
                    let key = keys(alg);
                    // rank-0, single-element, result-shaped and
                    // other-broadcast (the last axis) parameter forms.
                    let scalars = [scalar_f64(0.0), scalar_f64(1.0)];
                    let bytes = agree(
                        uniform_kernel,
                        &spec,
                        &[key.clone(), scalars[0].clone(), scalars[1].clone()],
                    );
                    if count > 0 {
                        assert_eq!(bytes.len(), count);
                    }
                    let wide: Vec<usize> = if shape.is_empty() {
                        Vec::new()
                    } else {
                        shape[..1].to_vec()
                    };
                    let _ = wide;
                    let dense = agree(
                        uniform_kernel,
                        &spec,
                        &[
                            key.clone(),
                            float_param(&shape, dtype, -2.0),
                            float_param(&shape, dtype, 3.0),
                        ],
                    );
                    if count > 0 {
                        assert_eq!(dense.len(), count);
                    }
                    agree(
                        uniform_kernel,
                        &spec,
                        &[
                            key.clone(),
                            float_param(&[1], dtype, -1.5),
                            float_param(&[1], dtype, 2.5),
                        ],
                    );
                    if shape.len() >= 2 {
                        let tail = shape[shape.len() - 1..].to_vec();
                        agree(
                            uniform_kernel,
                            &spec,
                            &[
                                key,
                                float_param(&tail, dtype, 0.5),
                                float_param(&tail, dtype, 4.0),
                            ],
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn normal_fast_path_matches_the_generic_path() {
        for dtype in [DType::Float16, DType::Float32, DType::Float64] {
            for alg in ALGORITHMS {
                for shape in shapes() {
                    let count: usize = shape.iter().product();
                    let spec = spec(
                        "random_normal",
                        &[
                            algorithm(alg),
                            ("dtype", AttrValue::DType(dtype)),
                        ],
                        (dtype, shape.clone()),
                    );
                    let key = keys(alg);
                    agree(
                        normal_kernel,
                        &spec,
                        &[key.clone(), scalar_f64(0.0), scalar_f64(1.0)],
                    );
                    agree(
                        normal_kernel,
                        &spec,
                        &[
                            key.clone(),
                            float_param(&shape, dtype, -3.0),
                            float_param(&shape, dtype, 2.0),
                        ],
                    );
                    agree(
                        normal_kernel,
                        &spec,
                        &[
                            key.clone(),
                            float_param(&[1], dtype, 0.75),
                            float_param(&[1], dtype, 0.5),
                        ],
                    );
                    if shape.len() >= 2 && count > 0 {
                        let tail = shape[shape.len() - 1..].to_vec();
                        agree(
                            normal_kernel,
                            &spec,
                            &[
                                key,
                                float_param(&tail, dtype, 0.25),
                                float_param(&tail, dtype, 1.5),
                            ],
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn randint_fast_path_matches_the_generic_path() {
        let dtypes = [
            DType::Int8,
            DType::Int16,
            DType::Int32,
            DType::Int64,
            DType::UInt8,
            DType::UInt16,
            DType::UInt32,
            DType::UInt64,
        ];
        for dtype in dtypes {
            for alg in ALGORITHMS {
                for shape in shapes() {
                    let count: usize = shape.iter().product();
                    let spec = spec(
                        "random_randint",
                        &[
                            algorithm(alg),
                            ("dtype", AttrValue::DType(dtype)),
                        ],
                        (dtype, shape.clone()),
                    );
                    let key = keys(alg);
                    agree(
                        randint_kernel,
                        &spec,
                        &[key.clone(), scalar_i64(-3), scalar_i64(11)],
                    );
                    agree(
                        randint_kernel,
                        &spec,
                        &[
                            key.clone(),
                            int_param(&shape, DType::Int64, -2),
                            int_param(&shape, DType::Int64, 5),
                        ],
                    );
                    if shape.len() >= 2 && count > 0 {
                        let tail = shape[shape.len() - 1..].to_vec();
                        agree(
                            randint_kernel,
                            &spec,
                            &[
                                key,
                                int_param(&tail, DType::Int64, 0),
                                int_param(&tail, DType::Int64, 4),
                            ],
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn randint_span_errors_match_the_generic_path() {
        for dtype in [DType::Int32, DType::UInt8] {
            for alg in ALGORITHMS {
                let spec = spec(
                    "random_randint",
                    &[
                        algorithm(alg),
                        ("dtype", AttrValue::DType(dtype)),
                    ],
                    (dtype, vec![2, 3]),
                );
                let key = keys(alg);
                // high == low and high < low, scalar and per-element.
                agree_error(
                    randint_kernel,
                    &spec,
                    &[key.clone(), scalar_i64(7), scalar_i64(7)],
                );
                agree_error(
                    randint_kernel,
                    &spec,
                    &[key.clone(), scalar_i64(9), scalar_i64(-9)],
                );
                let shape = vec![2, 3];
                let mut low = vec![ScalarValue::I(0); 6];
                low[4] = ScalarValue::I(50);
                let high = vec![ScalarValue::I(10); 6];
                agree_error(
                    randint_kernel,
                    &spec,
                    &[
                        key,
                        value(DType::Int64, shape.clone(), &low),
                        value(DType::Int64, shape, &high),
                    ],
                );
            }
        }
    }

    #[test]
    fn fast_paths_decline_a_draw_shape_that_differs_from_the_result() {
        // A draw shape smaller than the result shape is the generic kernel's
        // business (it broadcasts); the typed path must return `None`.
        let key = Key::State(9);
        let draw = vec![2];
        let shape = vec![2, 3];
        let low = scalar_f64(0.0);
        let high = scalar_f64(1.0);
        assert!(uniform(
            "random_uniform",
            Algorithm::SplitMix64,
            &key,
            DType::Float32,
            &draw,
            &shape,
            &low,
            &high,
        )
        .expect("uniform")
        .is_none());
        assert!(normal(
            "random_normal",
            Algorithm::SplitMix64,
            &key,
            DType::Float64,
            &draw,
            &shape,
            &scalar_f64(0.0),
            &scalar_f64(1.0),
        )
        .expect("normal")
        .is_none());
        assert!(randint(
            "random_randint",
            Algorithm::SplitMix64,
            &key,
            DType::Int32,
            &draw,
            &shape,
            &scalar_i64(0),
            &scalar_i64(4),
        )
        .expect("randint")
        .is_none());
    }
}
