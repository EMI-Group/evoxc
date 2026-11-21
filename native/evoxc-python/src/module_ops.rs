//! The `infer` / `transform` entry points.
//!
//! Both follow the same shape: convert the Python arguments into the envelope's
//! native form while the GIL is held, hand the pure native work to
//! `Python::detach` (GIL released — `native/CONTEXT.md` "Performance rules"),
//! then convert the result back.  No native code ever sees a `Bound<PyAny>`,
//! and no core type is kept alive across the GIL boundary by a Python
//! reference.
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::{Map, Value as Json};
use crate::convert::{json_from_py, py_from_json};
use crate::error;
/// The transform kinds `native/CONTEXT.md` defines; anything else names the kind.
pub const TRANSFORM_KINDS: [&str; 5] = ["vectorize", "vmap", "grad", "jvp", "vjp"];
/// `infer(op_name, operand_types, attributes, result_types=None, module=None)`.
pub fn infer(
    py: Python<'_>,
    op_name: &str,
    operand_types: &Bound<'_, PyAny>,
    attributes: Option<&Bound<'_, PyAny>>,
    result_types: Option<&Bound<'_, PyAny>>,
    module: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    // ---- GIL held: Python objects -> envelope JSON -> native types ---------
    let operand_types = decode_types(py, operand_types, "operand_types")?;
    let result_types = match result_types.filter(|value| !value.is_none()) {
        Some(value) => Some(decode_types(py, value, "result_types")?),
        None => None,
    };
    let module = match module.filter(|value| !value.is_none()) {
        Some(value) => {
            let envelope = json_from_py(value, "module").map_err(|err| {
                error::object_error(py, err, "infer: module")
            })?;
            Some(
                evoxc_core::envelope::decode(&envelope)
                    .map_err(|err| error::from_core(py, err.context("module")))?,
            )
        }
        None => None,
    };
    let attrs = decode_attrs(py, op_name, attributes)?;
    // ---- GIL released: pure native inference ------------------------------
    let outputs = py
        .detach(|| {
            evoxc_core::infer_op(
                op_name,
                &operand_types,
                &attrs,
                result_types.as_deref(),
                module.as_ref(),
            )
        })
        .map_err(|err| error::from_core(py, err))?;
    // ---- GIL held: native types -> envelope JSON -> Python ----------------
    let json = evoxc_core::wire::value_types_to_json(&outputs);
    Ok(py_from_json(py, &json)?.unbind())
}
/// `transform(kind, module, config)`.
///
/// The signature and the kind validation are final; the engine itself lives in
/// `evoxc-compiler` and is wired in a follow-up.  Until then every valid kind
/// raises `TransformError` naming the kind — an explicit failure, never a
/// silent Python fallback.
pub fn transform(
    py: Python<'_>,
    kind: &str,
    module: &Bound<'_, PyAny>,
    config: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    if !TRANSFORM_KINDS.contains(&kind) {
        return Err(error::raise(
            py,
            "TransformError",
            format!(
                "unknown transform kind {kind:?}; expected one of {}",
                TRANSFORM_KINDS.join(", ")
            ),
        ));
    }
    // Validate the envelope arguments now so the check does not move when the
    // engine is wired: a wrong type is a TypeError, not a TransformError.
    for (name, value) in [("module", module), ("config", config)] {
        if value.cast::<PyDict>().is_err() {
            return Err(error::type_error(format!(
                "transform: {name} must be a dict, got {}",
                value.get_type().name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
            )));
        }
    }
    Err(error::raise(
        py,
        "TransformError",
        format!("native transform '{kind}' is not wired in this build"),
    ))
}
fn decode_types(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Vec<evoxc_core::ValueType>> {
    let json = json_from_py(value, name)
        .map_err(|err| error::object_error(py, err, &format!("infer: {name}")))?;
    evoxc_core::wire::value_types_from_json(&json)
        .map_err(|err| error::from_core(py, err.context(name)))
}
fn decode_attrs(
    py: Python<'_>,
    op_name: &str,
    attributes: Option<&Bound<'_, PyAny>>,
) -> PyResult<evoxc_core::Attrs> {
    let schema = evoxc_core::schema::require(op_name).map_err(|err| error::from_core(py, err))?;
    let json = match attributes.filter(|value| !value.is_none()) {
        Some(value) => json_from_py(value, "attributes")
            .map_err(|err| error::object_error(py, err, "infer: attributes"))?,
        None => Json::Object(Map::new()),
    };
    let map = json.as_object().ok_or_else(|| {
        error::type_error(format!(
            "infer: attributes must be a dict of op attributes, got {}",
            match &json {
                Json::Array(_) => "a sequence",
                Json::String(_) => "a str",
                Json::Number(_) => "a number",
                Json::Bool(_) => "a bool",
                Json::Null => "None",
                Json::Object(_) => "an object",
            }
        ))
    })?;
    // `attrs_from_json` is the envelope's tolerant decoder: an INTS attribute
    // whose declared default is `null` accepts JSON `null`, exactly as the
    // Python builder serializes it.
    evoxc_core::envelope::attrs_from_json(map, schema)
        .map_err(|err| error::from_core(py, err))
}
#[cfg(test)]
mod tests {
    use super::*;
    /// The kind list is the contract's closed set, in a stable order.
    #[test]
    fn transform_kinds_match_the_contract() {
        assert_eq!(TRANSFORM_KINDS, ["vectorize", "vmap", "grad", "jvp", "vjp"]);
    }
}
