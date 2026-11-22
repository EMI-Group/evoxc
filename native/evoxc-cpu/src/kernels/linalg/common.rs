//! Attribute, arity and index helpers shared by the linear-algebra families.
//!
//! Every helper is a defensive re-check of what the plan build already verified,
//! so it can only fire on a hand-built `OpSpec` and never changes an op's
//! semantics.

use evoxc_core::{AttrValue, Attrs, Error, Result};
use serde_json::Value as Json;

use crate::buffer::shape_text;
use crate::exec::Value;
use crate::kernels::util;

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

/// Reject an operand count the op schema never produces (defensive: arity is
/// verified at plan build, so this can only fire on a hand-built `OpSpec`).
pub(super) fn expect_arity(name: &str, operands: &[Value], expected: usize) -> Result<()> {
    if operands.len() != expected {
        return Err(Error::verification(format!(
            "op '{name}': expected {expected} operand(s), got {}",
            operands.len()
        )));
    }
    Ok(())
}

/// Visit every C-order multi-index of `shape` (one visit for the empty shape,
/// none for a shape with a zero extent).
pub(super) fn each_index(shape: &[usize], mut body: impl FnMut(&[usize]) -> Result<()>) -> Result<()> {
    let total = util::numel(shape);
    if total == 0 {
        return Ok(());
    }
    let mut index = vec![0usize; shape.len()];
    for _ in 0..total {
        body(&index)?;
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    Ok(())
}

/// One attribute as JSON, unwrapping an explicit tuple (`any` attributes may
/// arrive either way; mirrors `infer::common::attr_json`).
pub(super) fn attr_json(value: &AttrValue) -> Json {
    match value {
        AttrValue::Tuple(items) => Json::Array(items.iter().map(attr_json).collect()),
        other => other.to_json(),
    }
}

/// A `bool` attribute falling back to `default` when absent or `null`.
pub(super) fn attr_bool(attrs: &Attrs, name: &str, default: bool) -> Result<bool> {
    match attrs.get(name) {
        None | Some(AttrValue::Null) => Ok(default),
        Some(value) => value.as_bool(),
    }
}

/// An `int` attribute falling back to `default` when absent (`null` is an
/// explicit error, exactly like the inference hooks).
pub(super) fn attr_int(attrs: &Attrs, name: &str, default: i64) -> Result<i64> {
    match attrs.get(name) {
        None => Ok(default),
        Some(AttrValue::Null) => Err(Error::shape(format!("{name}: expected an int, got None"))),
        Some(value) => value.as_i64(),
    }
}

/// The declared result shape of a one-result op, cross-checked against the
/// operand-derived geometry (a mismatch means the program is inconsistent).
pub(super) fn check_result_shape(op: &str, out_shape: &[usize], expected: &[usize]) -> Result<()> {
    if out_shape != expected {
        return Err(Error::shape(format!(
            "op '{op}': declared result shape {} disagrees with the operand geometry {}",
            shape_text(out_shape),
            shape_text(expected)
        )));
    }
    Ok(())
}
