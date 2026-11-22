//! Elementwise, comparison, cast/select/broadcast and rounding kernels.
//!
//! All arithmetic is executed in the op's DECLARED result dtype (the frontend
//! already inserted compensation casts); operands are converted only when
//! their dtype differs from it, so the common same-dtype case is a flat
//! broadcast loop with no allocation beyond the result.  The two exception
//! shapes are comparisons (computed in the promoted INPUT dtype, producing
//! `bool`) and `abs` of a complex value (magnitude, a real result).
//! `power`'s scalar-exponent specialisation lives in [`crate::kernels::pow`].
//!
//! The family is split by op group: [`binary`] owns the arithmetic/bitwise/
//! logical binary ops and their in-place accumulation path, [`compare`] the
//! comparisons and `select`, [`unary`] the pointwise unary ops and
//! [`transcendental`] `erf`/`gelu`/`nan_to_num`; this module keeps `register`
//! plus the ops that belong to no family (`stop_gradient`, `cast`,
//! `broadcast`).
use evoxc_core::{Error, Result};

use crate::buffer::{broadcast_to, shape_text};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::KernelMap;
use crate::plan::OpSpec;

mod binary;
mod compare;
mod transcendental;
mod unary;

pub use binary::in_place_kernel;
pub use crate::plan::InPlaceFn;
// The allocating entry point stays crate-visible at the family's root: the
// sibling `super::pow` module's unit tests call it, which is the only consumer
// outside this module (hence the allow — a no-test build imports nothing here).
#[allow(unused_imports)]
pub(crate) use binary::binary;

use binary::binary_kernel;
use compare::select_kernel;
use transcendental::{erf_kernel, gelu_kernel, nan_to_num_kernel};
use unary::unary_kernel;

pub fn register(map: &mut KernelMap) {
    for name in [
        "add",
        "subtract",
        "multiply",
        "divide",
        "power",
        "remainder",
        "maximum",
        "minimum",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "bitwise_left_shift",
        "bitwise_right_shift",
        "logical_and",
        "logical_or",
    ] {
        map.insert(name, binary_kernel);
    }
    for name in ["equal", "not_equal", "less", "less_equal", "greater", "greater_equal"] {
        map.insert(name, binary_kernel);
    }
    for name in [
        "abs",
        "acos",
        "negate",
        "square",
        "sqrt",
        "exp",
        "log",
        "log1p",
        "sin",
        "cos",
        "tan",
        "tanh",
        "sigmoid",
        "relu",
        "sign",
        "logical_not",
        "floor",
        "ceil",
        "round",
    ] {
        map.insert(name, unary_kernel);
    }
    map.insert("erf", erf_kernel);
    map.insert("gelu", gelu_kernel);
    map.insert("cast", cast_kernel);
    map.insert("select", select_kernel);
    map.insert("broadcast", broadcast_kernel);
    map.insert("stop_gradient", stop_gradient_kernel);
    map.insert("nan_to_num", nan_to_num_kernel);
}

fn stop_gradient_kernel(
    _ctx: &mut RunCtx<'_>,
    _spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    Ok(vec![operands[0].clone()])
}

fn cast_kernel(_ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let target = spec.attrs.dtype("dtype")?;
    let input = &operands[0];
    if input.dtype == target {
        return Ok(vec![operands[0].clone()]);
    }
    single(input.cast(target)?)
}

fn broadcast_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    let target = spec.attrs.shape("shape")?;
    let shape = ctx.eval_shape_strict(target).map_err(|err| {
        err.context("op 'broadcast': cannot evaluate the target shape")
    })?;
    let input = &operands[0];
    let out = broadcast_to(input, &shape).map_err(|err| {
        Error::shape(format!(
            "op 'broadcast': operand shape {} cannot be broadcast to target shape {}: {}",
            shape_text(&input.shape),
            shape_text(&shape),
            err.message
        ))
    })?;
    single(out)
}

/// Helpers shared by the family's unit-test modules.
#[cfg(test)]
pub(super) mod test_support {
    use evoxc_core::{Attrs, DType, Result, ValueType};

    use crate::buffer::{Buffer, Storage};
    use crate::exec::Value;
    use crate::plan::OpSpec;

    pub(super) fn spec(name: &str, out: ValueType) -> OpSpec {
        OpSpec {
            name: name.to_string(),
            attrs: Attrs::new(),
            operand_types: Vec::new(),
            result_types: vec![out],
            regions: Vec::new(),
            location: None,
        }
    }

    pub(super) fn buffer(dtype: DType, shape: &[usize], storage: Storage) -> Buffer {
        Buffer::new(dtype, shape.to_vec(), storage).unwrap()
    }

    /// The kernel takes shared values (a `power(x, 1.0)` result aliases its
    /// base); the tests keep handing it plain buffers, so this local `binary`
    /// shadows the glob-imported one.
    pub(super) fn binary(spec: &OpSpec, a: &Buffer, b: &Buffer) -> Result<Vec<Value>> {
        super::binary::binary(spec, &std::rc::Rc::new(a.clone()), &std::rc::Rc::new(b.clone()))
    }
}
