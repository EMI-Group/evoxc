//! Control-flow and host-dispatch kernels: `constant`, `if`, `while`, `call`,
//! `block_call`, `external_call`, `runtime_call`.
//!
//! Region conventions match `evoxc/backends/numpy/kernels/control_flow.py`:
//!
//! * `if` — operand 0 is a 0-d `bool` predicate, `regions[0]` is "then",
//!   `regions[1]` is "else", and EVERY operand (predicate included) binds to
//!   the selected region's entry block arguments.
//! * `while` — the operands are the initial loop-carried values, `regions[0]`
//!   is the condition (exactly one 0-d `bool`) and `regions[1]` the body
//!   (one next-carried value per operand).  No iteration limit is invented.
//! * `call` — runs another function of the module (`callee`); its entry block
//!   is bound to the operands.
//!
//! `block_call` / `external_call` / `runtime_call` are host dispatches: the
//! engine is Python-free, so the PyO3 layer implements [`Callbacks`].  A plan
//! that needs a callback and runs without one is an explicit `BackendError`.
use std::rc::Rc;

use evoxc_core::{DType, Error, Result};

use crate::buffer::{shape_text, Buffer, Storage};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::KernelMap;
use crate::plan::OpSpec;

pub fn register(map: &mut KernelMap) {
    map.insert("constant", constant_kernel);
    map.insert("if", if_kernel);
    map.insert("while", while_kernel);
    map.insert("call", call_kernel);
    map.insert("block_call", block_call_kernel);
    map.insert("external_call", external_call_kernel);
    map.insert("runtime_call", runtime_call_kernel);
}

fn constant_kernel(
    _ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    _operands: &[Value],
) -> Result<Vec<Value>> {
    let payload = spec.attrs.array("value")?;
    single(Buffer::from_payload(payload)?)
}

/// Read a 0-d `bool` value (never a silent coercion).
fn predicate(op: &str, value: &Buffer, where_: &str) -> Result<bool> {
    if value.dtype != DType::Bool || !value.shape.is_empty() {
        return Err(Error::backend(format!(
            "kernel for op '{op}': {where_} must be a 0-d bool tensor, got dtype {} with shape {}",
            value.dtype.name(),
            shape_text(&value.shape)
        )));
    }
    match &value.storage {
        Storage::Bool(values) => Ok(values[0]),
        other => Err(Error::backend(format!(
            "kernel for op '{op}': {where_} has a non-bool storage variant ({})",
            other.dtype().name()
        ))),
    }
}

fn if_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    if spec.regions.len() != 2 {
        return Err(Error::verification(format!(
            "op 'if' has {} region(s); exactly 2 (then, else) are required",
            spec.regions.len()
        )));
    }
    let choice = predicate("if", &operands[0], "the predicate (operand 0)")?;
    let region = if choice {
        &spec.regions[0]
    } else {
        &spec.regions[1]
    };
    ctx.run_region(region, operands.to_vec())
}

fn while_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    if spec.regions.len() != 2 {
        return Err(Error::verification(format!(
            "op 'while' has {} region(s); exactly 2 (condition, body) are required",
            spec.regions.len()
        )));
    }
    let mut carried = operands.to_vec();
    loop {
        let condition = ctx.run_region(&spec.regions[0], carried.clone())?;
        if condition.len() != 1 {
            return Err(Error::backend(format!(
                "kernel for op 'while': condition region produced {} result(s), expected exactly 1 (a 0-d bool)",
                condition.len()
            )));
        }
        if !predicate("while", &condition[0], "the condition region result")? {
            break;
        }
        let body = ctx.run_region(&spec.regions[1], carried.clone())?;
        if body.len() != operands.len() {
            return Err(Error::backend(format!(
                "kernel for op 'while': body region produced {} result(s), expected {} next-carried value(s)",
                body.len(),
                operands.len()
            )));
        }
        carried = body;
    }
    Ok(carried)
}

fn call_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let callee = spec.attrs.str("callee")?;
    let entry = ctx
        .plan
        .functions
        .iter()
        .find(|function| function.name == callee)
        .map(|function| function.entry)
        .ok_or_else(|| {
            Error::backend(format!(
                "kernel for op 'call': module '{}' has no function named '{callee}'",
                ctx.program.name
            ))
        })?;
    ctx.run_block(entry, operands.to_vec())
}

fn missing_callbacks(kind: &str) -> Error {
    Error::backend(format!(
        "op '{kind}' requires a host callback but no resolver was supplied — pass a resolver with {kind} support when running this program"
    ))
}

fn block_call_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    let name = spec.attrs.str("block_name")?.to_string();
    let static_args = spec
        .attrs
        .get("static_args")
        .map(|value| value.to_json())
        .unwrap_or(serde_json::Value::Null);
    let callbacks = ctx.callbacks.ok_or_else(|| missing_callbacks("block_call"))?;
    let outputs = callbacks.call_block(&name, operands, &static_args)?;
    Ok(outputs.into_iter().map(Rc::new).collect())
}

fn external_call_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    let name = spec.attrs.str("name")?.to_string();
    let callbacks = ctx
        .callbacks
        .ok_or_else(|| missing_callbacks("external_call"))?;
    let outputs = callbacks.call_external(&name, operands)?;
    Ok(outputs.into_iter().map(Rc::new).collect())
}

fn runtime_call_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    let callback = spec.attrs.str("callback")?.to_string();
    let callbacks = ctx
        .callbacks
        .ok_or_else(|| missing_callbacks("runtime_call"))?;
    let outputs = callbacks.call_runtime(&callback, operands)?;
    Ok(outputs.into_iter().map(Rc::new).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use num_complex::Complex;

    #[test]
    fn predicate_requires_zero_dimensional_bool() {
        let scalar = Buffer::new(DType::Bool, vec![], Storage::Bool(vec![true])).unwrap();
        assert!(predicate("if", &scalar, "predicate").unwrap());
        let vector = Buffer::new(DType::Bool, vec![1], Storage::Bool(vec![true])).unwrap();
        assert!(predicate("if", &vector, "predicate").is_err());
        let int = Buffer::new(DType::Int32, vec![], Storage::I32(vec![1])).unwrap();
        assert!(predicate("if", &int, "predicate").is_err());
    }

    #[test]
    fn payload_round_trips_every_dtype() {
        // A small guard that the storage variants all have a decode path.
        let payload = evoxc_core::ArrayPayload {
            dtype: DType::Float16,
            shape: vec![2],
            data: vec![0, 0x3c, 0, 0x40],
        };
        let buffer = Buffer::from_payload(&payload).unwrap();
        assert_eq!(buffer.storage, Storage::F16(vec![f16::from_f32(1.0), f16::from_f32(2.0)]));
        let payload = evoxc_core::ArrayPayload {
            dtype: DType::Complex64,
            shape: vec![1],
            data: [1.0f32.to_le_bytes(), (-2.0f32).to_le_bytes()].concat(),
        };
        let buffer = Buffer::from_payload(&payload).unwrap();
        assert_eq!(buffer.storage, Storage::C64(vec![Complex::new(1.0, -2.0)]));
    }
}
