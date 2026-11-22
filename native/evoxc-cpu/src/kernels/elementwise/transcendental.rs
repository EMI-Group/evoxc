//! `erf`, `gelu` and `nan_to_num`.
use evoxc_core::{AttrValue, DType, Error, Result};
use half::f16;
use num_complex::Complex;
use serde_json::Value as Json;

use crate::buffer::{Buffer, Storage};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::plan::OpSpec;

pub(super) fn erf_kernel(_ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let input = &operands[0];
    util::reject_complex("erf", input.dtype)?;
    let out_dtype = spec.result_types[0].dtype;
    let storage = match (&input.storage, out_dtype) {
        (Storage::F64(values), DType::Float64) => Storage::F64(
            values.iter().map(|value| libm::erf(*value)).collect(),
        ),
        (Storage::F32(values), DType::Float32) => Storage::F32(
            values.iter().map(|value| libm::erff(*value)).collect(),
        ),
        (Storage::F16(values), DType::Float16) => Storage::F16(
            values
                .iter()
                .map(|value| f16::from_f32(libm::erff(value.to_f32())))
                .collect(),
        ),
        _ => {
            let cast = input.cast(out_dtype)?;
            return erf_kernel(_ctx, spec, &[std::rc::Rc::new(cast)]);
        }
    };
    single(Buffer::new(out_dtype, input.shape.clone(), storage)?)
}

pub(super) fn gelu_kernel(_ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let input = &operands[0];
    util::reject_complex("gelu", input.dtype)?;
    let out_dtype = spec.result_types[0].dtype;
    if !out_dtype.is_float() {
        return Err(Error::backend(format!(
            "op 'gelu': expected a floating dtype, got {}",
            out_dtype.name()
        )));
    }
    let source = if input.dtype == out_dtype {
        None
    } else {
        Some(input.cast(out_dtype)?)
    };
    let source = source.as_ref().unwrap_or(input);
    let sqrt_two = std::f64::consts::SQRT_2;
    let storage = match &source.storage {
        Storage::F64(values) => Storage::F64(
            values
                .iter()
                .map(|value| {
                    0.5 * value * (1.0 + libm::erf(value / sqrt_two))
                })
                .collect(),
        ),
        Storage::F32(values) => Storage::F32(
            values
                .iter()
                .map(|value| {
                    let x = *value as f64;
                    (0.5 * x * (1.0 + libm::erf(x / sqrt_two))) as f32
                })
                .collect(),
        ),
        Storage::F16(values) => Storage::F16(
            values
                .iter()
                .map(|value| {
                    let x = value.to_f64();
                    f16::from_f64(0.5 * x * (1.0 + libm::erf(x / sqrt_two)))
                })
                .collect(),
        ),
        _ => {
            return Err(Error::backend(format!(
                "op 'gelu': unsupported dtype {}",
                source.dtype.name()
            )))
        }
    };
    single(Buffer::new(out_dtype, input.shape.clone(), storage)?)
}

/// `nan_to_num`: replace NaN / +inf / -inf entries (numpy semantics).
///
/// `nan`'s registry default is `0.0`, so an ABSENT attribute means "replace
/// NaN with 0"; an explicit `null` (`nan=None` in the frontend) means "leave
/// NaN unchanged", which is numpy's `nan_to_num(arr, nan=None)`.  The
/// infinities are `any` attributes defaulting to `None` = the dtype's max/
/// min finite value (numpy's own default), so `null` and absent both mean
/// that.  Non-float storages (int/bool) have no NaN/inf and pass through, as
/// numpy's does.
pub(super) fn nan_to_num_kernel(
    _ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    let input = &operands[0];
    let nan = match spec.attrs.get("nan") {
        None => Some(0.0),
        _ => any_float(&spec.attrs, "nan")?,
    };
    let posinf = any_float(&spec.attrs, "posinf")?;
    let neginf = any_float(&spec.attrs, "neginf")?;
    let storage = match &input.storage {
        Storage::F32(values) => Storage::F32(
            values
                .iter()
                .map(|value| fix_f32(*value, nan, posinf, neginf, f32::MAX, f32::MIN))
                .collect(),
        ),
        Storage::F64(values) => Storage::F64(
            values
                .iter()
                .map(|value| fix_f64(*value, nan, posinf, neginf, f64::MAX, f64::MIN))
                .collect(),
        ),
        Storage::F16(values) => Storage::F16(
            values
                .iter()
                .map(|value| {
                    let fixed = fix_f32(
                        value.to_f32(),
                        nan,
                        posinf,
                        neginf,
                        f16::MAX.to_f32(),
                        -f16::MAX.to_f32(),
                    );
                    f16::from_f32(fixed)
                })
                .collect(),
        ),
        Storage::C64(values) => Storage::C64(
            values
                .iter()
                .map(|value| {
                    Complex::new(
                        fix_f32(value.re, nan, posinf, neginf, f32::MAX, f32::MIN),
                        fix_f32(value.im, nan, posinf, neginf, f32::MAX, f32::MIN),
                    )
                })
                .collect(),
        ),
        Storage::C128(values) => Storage::C128(
            values
                .iter()
                .map(|value| {
                    Complex::new(
                        fix_f64(value.re, nan, posinf, neginf, f64::MAX, f64::MIN),
                        fix_f64(value.im, nan, posinf, neginf, f64::MAX, f64::MIN),
                    )
                })
                .collect(),
        ),
        _ => return Ok(vec![operands[0].clone()]),
    };
    single(Buffer::new(input.dtype, input.shape.clone(), storage)?)
}

/// Replace one f32 value per numpy's `nan_to_num` (a `None` replacement leaves
/// the value unchanged).
fn fix_f32(
    value: f32,
    nan: Option<f64>,
    posinf: Option<f64>,
    neginf: Option<f64>,
    max: f32,
    min: f32,
) -> f32 {
    if value.is_nan() {
        nan.map(|v| v as f32).unwrap_or(value)
    } else if value == f32::INFINITY {
        posinf.map(|v| v as f32).unwrap_or(max)
    } else if value == f32::NEG_INFINITY {
        neginf.map(|v| v as f32).unwrap_or(min)
    } else {
        value
    }
}

/// Replace one f64 value per numpy's `nan_to_num` (a `None` replacement leaves
/// the value unchanged).
fn fix_f64(
    value: f64,
    nan: Option<f64>,
    posinf: Option<f64>,
    neginf: Option<f64>,
    max: f64,
    min: f64,
) -> f64 {
    if value.is_nan() {
        nan.unwrap_or(value)
    } else if value == f64::INFINITY {
        posinf.unwrap_or(max)
    } else if value == f64::NEG_INFINITY {
        neginf.unwrap_or(min)
    } else {
        value
    }
}

/// Read an optional numeric attribute.
///
/// `posinf`/`neginf` are `any` attributes whose registry default is `None`, so
/// the frontend always sends an explicit JSON `null` for "the numpy default";
/// the envelope decodes that to [`AttrValue::Null`], and an absent attribute
/// reads the same way.  Integers are accepted too (a replacement may be written
/// `nan=1`), so the reader matches what the numpy reference reads through
/// `op.attributes`.
fn any_float(attrs: &evoxc_core::Attrs, name: &str) -> Result<Option<f64>> {
    match attrs.get(name) {
        None | Some(AttrValue::Null) | Some(AttrValue::Any(Json::Null)) => Ok(None),
        Some(AttrValue::Any(Json::Number(number))) => number.as_f64().map(Some).ok_or_else(|| {
            Error::verification(format!(
                "op 'nan_to_num': attribute {name:?} is not representable as a float: {number}"
            ))
        }),
        Some(AttrValue::Int(value)) => Ok(Some(*value as f64)),
        Some(AttrValue::Float(value)) => Ok(Some(*value)),
        Some(other) => Err(Error::verification(format!(
            "op 'nan_to_num': attribute {name:?} must be a number or null, got {other}"
        ))),
    }
}
