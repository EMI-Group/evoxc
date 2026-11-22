//! Shared test harness for the random kernels: a bare `RunCtx`, value/attribute
//! builders and the bit-comparison helpers.
//!
//! Two independent checks back every pin these modules write:
//! 1. the kernel cores are compared against a *second*, straightforward
//!    transcription of the published ciphers (`bits.rs`), written from the
//!    Random123 / SplitMix64 definitions — not a copy of the kernel code;
//! 2. the op kernels are compared against HARD-CODED vectors produced once by
//!    the authoritative numpy reference
//!    (`evoxc/backends/numpy/kernels/random.py`), bit-for-bit (f32/f64
//!    compared by raw bits).
use std::rc::Rc;
use std::sync::Arc;

use evoxc_core::{AttrValue, Attrs, DType, Dim, Program, ValueType};

use crate::buffer::Storage;
use crate::exec::{RunCtx, RunOptions, Value};
use crate::plan::{OpSpec, Plan};
use crate::scalar::ScalarValue;

use super::{as_f64, as_i64, build, scalar_values};

pub(super) fn test_ctx(program: &Program) -> RunCtx<'_> {
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

pub(super) fn value(dtype: DType, shape: Vec<usize>, values: &[ScalarValue]) -> Value {
    Rc::new(build(dtype, shape, values).expect("test buffer"))
}

pub(super) fn key_splitmix(state: i64) -> Value {
    value(DType::Int64, vec![], &[ScalarValue::I(state)])
}

pub(super) fn key_words(words: &[i32]) -> Value {
    let values: Vec<ScalarValue> = words.iter().map(|w| ScalarValue::I(*w as i64)).collect();
    value(DType::Int32, vec![words.len()], &values)
}

pub(super) fn scalar_f64(number: f64) -> Value {
    value(DType::Float64, vec![], &[ScalarValue::F(number)])
}

pub(super) fn scalar_i64(number: i64) -> Value {
    value(DType::Int64, vec![], &[ScalarValue::I(number)])
}

pub(super) fn spec(name: &str, attrs: &[(&str, AttrValue)], result: (DType, Vec<usize>)) -> OpSpec {
    // The draw shape is the declared result shape in these fixtures (the
    // parameter operands are scalars, so it is not broadcast up).
    let mut pairs: Vec<(&str, AttrValue)> = vec![(
        "shape",
        AttrValue::Shape(result.1.iter().map(|dim| Dim::int(*dim as i64)).collect()),
    )];
    pairs.extend(attrs.iter().map(|(key, val)| (*key, val.clone())));
    spec_attrs(name, Attrs::from_pairs(pairs), result)
}

pub(super) fn spec_attrs(name: &str, attrs: Attrs, result: (DType, Vec<usize>)) -> OpSpec {
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

pub(super) fn shape_attr(dims: &[usize]) -> AttrValue {
    AttrValue::Shape(dims.iter().map(|dim| Dim::int(*dim as i64)).collect())
}

pub(super) fn algorithm(name: &str) -> (&'static str, AttrValue) {
    ("algorithm", AttrValue::Str(name.to_string()))
}

pub(super) fn f64_of(results: &[Value]) -> Vec<f64> {
    scalar_values(&results[0]).into_iter().map(as_f64).collect()
}

pub(super) fn i64_of(results: &[Value]) -> Vec<i64> {
    scalar_values(&results[0]).into_iter().map(as_i64).collect()
}

pub(super) fn f64_bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

pub(super) fn f32_bits(values: &[f64]) -> Vec<u32> {
    values.iter().map(|v| (*v as f32).to_bits()).collect()
}

pub(super) fn f16_bits(results: &[Value]) -> Vec<u16> {
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
pub(super) fn assert_f32_within(actual: &[f64], expected: &[u32], ulps: u32) {
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
