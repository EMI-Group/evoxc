//! Random-category shape inference (the six `random_*` ops).
//!
//! Stub: the dispatcher is wired into [`super::infer_op`] and its handled op
//! names live in [`super::RANDOM_HANDLED`]; the bodies are ported by a later
//! worker.  Until then every op of this category falls through to the explicit
//! "no native shape function is registered" error — never a silent empty
//! result.

use super::Ctx;
use crate::attr::Attrs;
use crate::error::Result;
use crate::value_type::ValueType;

pub(crate) fn infer(
    _op: &str,
    _operands: &[ValueType],
    _attrs: &Attrs,
    _ctx: &Ctx,
) -> Option<Result<Vec<ValueType>>> {
    None
}
