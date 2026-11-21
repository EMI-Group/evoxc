//! The `infer` / `transform` entry points.
//!
//! Both follow the same shape: convert the Python arguments into the envelope's
//! native form while the GIL is held, hand the pure native work to
//! `Python::detach` (GIL released — `native/CONTEXT.md` "Performance rules"),
//! then convert the result back.  No native code ever sees a `Bound<PyAny>`,
//! and no core type is kept alive across the GIL boundary by a Python
//! reference.
//!
//! `transform` is the thin shell around `evoxc_compiler::transform`: the
//! config dict is decoded by [`transform_config`] (interpreter-free), the
//! `extension_rule` callable is kept as a [`PyExtensionRule`] handle, and the
//! engine runs detached — the bridge re-attaches to the interpreter only when
//! an extension rule is actually consulted (`native/CONTEXT.md`
//! "Extension-rule bridge").
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyString};
use serde_json::{Map, Value as Json};

use evoxc_compiler::{ExtensionRequest, ExtensionResponse, ExtensionRule, TransformKind};
use evoxc_core::graph::{Function, Program, ValueId};

use crate::convert::{json_from_py, py_from_json};
use crate::error;
use crate::transform_config;

/// The transform kinds `native/CONTEXT.md` defines; anything else names the kind.
#[allow(dead_code)] // the closed set, asserted by the unit test below
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
        Some(value) => Some(decode_module(py, value)?),
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

/// Decode the `infer` `module` argument.
///
/// Two shapes are accepted, both ending in the `Program` that `infer_op` needs
/// for `call`-callee resolution (`module.function(callee)?.output_types`):
///
/// * a full `evoxc-native-ir` envelope — recognised by its `"format"` tag and
///   decoded by [`evoxc_core::envelope::decode`], exactly as before;
/// * a signature-only context — a dict carrying `"functions"`, a list of
///   `[name, [output types]]`, `[name, [input types], [output types]]` or
///   `{"name": …, "input_types": …, "output_types": …}` entries.  A `Program`
///   with one signature-only `Function` per entry is synthesized, so nothing
///   downstream changes.
///
/// Types use the same `{"dtype": str, "shape": [dim, …]}` dicts as
/// `operand_types` ([`decode_types`]' decoder, [`evoxc_core::wire`]).  A
/// non-dict module, a missing `"functions"` list, a malformed entry or an
/// unknown dict key is a `TypeError` (argument shape, like every other
/// decoder); a malformed type is the type decoder's own `VerificationError`
/// tagged with its path.
fn decode_module(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Program> {
    let json = json_from_py(value, "module")
        .map_err(|err| error::object_error(py, err, "infer: module"))?;
    decode_module_json(&json).map_err(|err| match err {
        ContextError::Shape(message) => error::type_error(message),
        ContextError::Core(err) => error::from_core(py, err),
    })
}

/// The interpreter-free half of [`decode_module`] (unit-tested without a
/// Python interpreter); the caller maps [`ContextError`] onto `TypeError` /
/// `NativeError`.
fn decode_module_json(json: &Json) -> std::result::Result<Program, ContextError> {
    let map = json.as_object().ok_or_else(|| {
        ContextError::Shape(format!(
            "infer: module must be a dict — an evoxc-native-ir envelope or a \
             signature-only context {{\"functions\": [[name, [output types]], ...]}} — \
             got {}",
            json_kind(json)
        ))
    })?;
    if map.contains_key("format") {
        return evoxc_core::envelope::decode(json)
            .map_err(|err| ContextError::Core(err.context("module")));
    }
    let functions = map.get("functions").ok_or_else(|| {
        ContextError::Shape(format!(
            "infer: module must be an evoxc-native-ir envelope (a \"format\" tag) \
             or a signature-only context with a \"functions\" list; got keys: {}",
            json_keys(map)
        ))
    })?;
    signature_program(functions)
}

/// Why a signature-only module context was rejected: an argument-shape problem
/// (a `TypeError`, like every other decoder) or a native judgment (a
/// `NativeError` from `evoxc_core`).
#[derive(Debug)]
enum ContextError {
    Shape(String),
    Core(evoxc_core::Error),
}

/// Synthesize a `Program` holding one signature-only `Function` per entry.
fn signature_program(functions: &Json) -> std::result::Result<Program, ContextError> {
    let entries = functions.as_array().ok_or_else(|| {
        ContextError::Shape(format!(
            "infer: module.functions must be a list of signature entries, got {}",
            json_kind(functions)
        ))
    })?;
    // The synthetic name shows up in `infer_op`'s unknown-callee message, which
    // keeps naming the callee itself (`no function named 'x' in module '…'`).
    let mut program = Program::new("signature-context");
    for (index, entry) in entries.iter().enumerate() {
        let at = format!("module.functions[{index}]");
        let (name, input_types, output_types) = signature_entry(entry, &at)?;
        let mut function = Function::new(name, input_types);
        function.output_types = output_types;
        program
            .add_function(function)
            .map_err(|err| ContextError::Core(err.context(&at)))?;
    }
    Ok(program)
}

/// Decode one `module.functions[i]` signature entry into
/// `(name, input types, output types)`.
fn signature_entry(
    entry: &Json,
    at: &str,
) -> std::result::Result<(String, Vec<evoxc_core::ValueType>, Vec<evoxc_core::ValueType>), ContextError>
{
    match entry {
        Json::Array(items) => {
            if !(2..=3).contains(&items.len()) {
                return Err(ContextError::Shape(format!(
                    "infer: {at} must be [name, [output types]] or \
                     [name, [input types], [output types]], got {} element(s)",
                    items.len()
                )));
            }
            let name = signature_name(&items[0], at)?;
            let (input, output) = match items.as_slice() {
                [_name, output] => (Vec::new(), output),
                [_name, input, output] => (signature_types(input, at)?, output),
                _ => unreachable!("length checked above"),
            };
            let output_types = signature_types(output, at)?;
            Ok((name, input, output_types))
        }
        Json::Object(map) => {
            let name = match map.get("name") {
                Some(value) => signature_name(value, at)?,
                None => {
                    return Err(ContextError::Shape(format!(
                        "infer: {at} is missing its \"name\""
                    )))
                }
            };
            let input_types = match map.get("input_types") {
                Some(value) => signature_types(value, at)?,
                None => Vec::new(),
            };
            let output_types = match map.get("output_types") {
                Some(value) => signature_types(value, at)?,
                None => {
                    return Err(ContextError::Shape(format!(
                        "infer: {at} is missing its \"output_types\""
                    )))
                }
            };
            if let Some(key) = map
                .keys()
                .find(|key| !matches!(key.as_str(), "name" | "input_types" | "output_types"))
            {
                return Err(ContextError::Shape(format!(
                    "infer: {at} has an unknown key {key:?}; accepted keys are \
                     name, input_types, output_types"
                )));
            }
            Ok((name, input_types, output_types))
        }
        other => Err(ContextError::Shape(format!(
            "infer: {at} must be a list or a dict, got {}",
            json_kind(other)
        ))),
    }
}

/// One entry's function name.
fn signature_name(value: &Json, at: &str) -> std::result::Result<String, ContextError> {
    match value {
        Json::String(name) if !name.is_empty() => Ok(name.clone()),
        Json::String(_) => Err(ContextError::Shape(format!(
            "infer: {at} has an empty function name"
        ))),
        other => Err(ContextError::Shape(format!(
            "infer: {at} function names must be strings, got {}",
            json_kind(other)
        ))),
    }
}

/// Decode one signature type list with the `operand_types` decoder.
fn signature_types(
    value: &Json,
    at: &str,
) -> std::result::Result<Vec<evoxc_core::ValueType>, ContextError> {
    evoxc_core::wire::value_types_from_json(value)
        .map_err(|err| ContextError::Core(err.context(format!("{at} types"))))
}

/// The JSON kind of a value, for argument-shape messages.
fn json_kind(value: &Json) -> &'static str {
    match value {
        Json::Array(_) => "a sequence",
        Json::String(_) => "a str",
        Json::Number(_) => "a number",
        Json::Bool(_) => "a bool",
        Json::Null => "None",
        Json::Object(_) => "an object",
    }
}

/// The keys of a JSON object, for argument-shape messages.
fn json_keys(map: &Map<String, Json>) -> String {
    if map.is_empty() {
        return "<none>".to_string();
    }
    let mut keys: Vec<&str> = map.keys().map(String::as_str).collect();
    keys.sort_unstable();
    keys.join(", ")
}

/// `transform(kind, module, config)`.
///
/// Runs one of the five native graph transforms (`vectorize` / `vmap` / `grad` /
/// `jvp` / `vjp`) and returns the transformed `evoxc-native-ir` envelope plus a
/// `"transform"` object carrying the kind, the transformed return/block-argument
/// value ids and the fresh batch `Dim` name (or `None` for the AD transforms).
///
/// The config keys are documented in `native/CONTEXT.md` ("Transform configs")
/// and accepted by [`parse_config`]; an unknown key is a `TransformError`, never
/// silently ignored.
pub fn transform(
    py: Python<'_>,
    kind: &str,
    module: &Bound<'_, PyAny>,
    config: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    // Unknown kinds are an explicit `TransformError` naming the accepted set.
    let kind = TransformKind::parse(kind).map_err(|err| error::from_core(py, err))?;
    // A wrong argument type is a TypeError, not a graph error.
    let module = module.cast::<PyDict>().map_err(|_| {
        error::type_error(format!(
            "transform: module must be a dict, got {}",
            type_name(module)
        ))
    })?;
    let config = config.cast::<PyDict>().map_err(|_| {
        error::type_error(format!(
            "transform: config must be a dict, got {}",
            type_name(config)
        ))
    })?;

    // ---- GIL held: envelope JSON -> Program, Python config -> native config
    let envelope = json_from_py(module.as_any(), "module")
        .map_err(|err| error::object_error(py, err, "transform: module"))?;
    let program = evoxc_core::envelope::decode(&envelope)
        .map_err(|err| error::from_core(py, err.context("module")))?;
    let (config, extension) = parse_config(py, config)?;

    // ---- GIL released: the native transform (re-attaching for extension
    // rules only) ------------------------------------------------------------
    let output = py
        .detach(move || {
            let bridge = extension
                .as_ref()
                .map(|rule| rule as &dyn ExtensionRule);
            evoxc_compiler::transform(kind, &program, &config, bridge)
        })
        .map_err(|err| error::from_core(py, err))?;

    // ---- GIL held: native -> envelope JSON (plus the transform metadata) ---
    let mut envelope = evoxc_core::envelope::encode(&output.program);
    if let Json::Object(map) = &mut envelope {
        let mut meta = Map::new();
        meta.insert("kind".to_string(), Json::String(kind.as_str().to_string()));
        meta.insert(
            "output_ids".to_string(),
            Json::Array(output.output_ids.iter().map(|id| Json::from(id.0)).collect()),
        );
        meta.insert(
            "input_ids".to_string(),
            Json::Array(output.input_ids.iter().map(|id| Json::from(id.0)).collect()),
        );
        meta.insert(
            "batch_dim".to_string(),
            output.batch_dim.map_or(Json::Null, Json::String),
        );
        map.insert("transform".to_string(), Json::Object(meta));
    }
    Ok(py_from_json(py, &envelope)?.unbind())
}

// ---------------------------------------------------------------------------
// Config decoding (GIL held) and the extension-rule bridge
// ---------------------------------------------------------------------------

/// A Python callable kept alive for the whole transform run and consulted by the
/// engine through [`ExtensionRule`].
#[derive(Debug)]
struct PyExtensionRule {
    callback: Py<PyAny>,
}

impl ExtensionRule for PyExtensionRule {
    fn call(&self, request: &ExtensionRequest) -> evoxc_core::Result<Option<ExtensionResponse>> {
        // The engine runs detached; the bridge is the only place that needs the
        // interpreter, so it re-attaches here and nowhere else.
        Python::attach(|py| call_python_rule(py, &self.callback, request))
    }
}

/// Build the neutral request dict, call the Python callback, parse the answer.
fn call_python_rule(
    py: Python<'_>,
    callback: &Py<PyAny>,
    request: &ExtensionRequest,
) -> evoxc_core::Result<Option<ExtensionResponse>> {
    let request_json = transform_config::request_to_json(request);
    let request_py = py_from_json(py, &request_json)
        .map_err(|err| python_failure(py, &err, &request.op_name))?;
    let result = callback
        .bind(py)
        .call1((request_py,))
        .map_err(|err| python_failure(py, &err, &request.op_name))?;
    if result.is_none() {
        return Ok(None);
    }
    let response_json = json_from_py(&result, "extension response")
        .map_err(|err| python_failure(py, &err, &request.op_name))?;
    transform_config::response_from_json(&response_json, &request.op_name)
}

/// A Python failure inside the bridge is a `TransformError` that preserves the
/// interpreter's message text (and names the op being transformed).
fn python_failure(py: Python<'_>, err: &PyErr, op_name: &str) -> evoxc_core::Error {
    let text = match err.value(py).str() {
        Ok(text) => text.to_string_lossy().into_owned(),
        Err(_) => String::from("<unprintable Python error>"),
    };
    evoxc_core::Error::transform(format!("extension rule for op '{op_name}': {text}"))
}

/// Decode the Python config dict into the engine's [`TransformConfig`] plus an
/// optional extension-rule bridge.
///
/// Unknown keys are a `TransformError` naming the key and the accepted set; a
/// missing or inapplicable key is left to the engine's own diagnostics.
fn parse_config(
    py: Python<'_>,
    config: &Bound<'_, PyDict>,
) -> PyResult<(evoxc_compiler::TransformConfig, Option<PyExtensionRule>)> {
    let mut out = evoxc_compiler::TransformConfig::default();
    let mut extension: Option<PyExtensionRule> = None;
    for (key, value) in config.iter() {
        let name = key.cast::<PyString>().map_err(|_| {
            error::type_error(format!(
                "transform: config keys must be strings, got {}",
                type_name(&key)
            ))
        })?;
        let name = name
            .to_str()
            .map_err(|err| error::object_error(py, err, "transform: config key"))?;
        match name {
            "axes" => apply_axes_from_py(py, &value, &mut out)?,
            "batch_dim_name" => {
                out.batch_dim_name = transform_config::batch_dim_name_from_json(&convert_value(
                    py,
                    &value,
                    "batch_dim_name",
                )?)
                .map_err(|err| error::from_core(py, err))?;
            }
            "out_axes" => {
                let json = convert_value(py, &value, "out_axes")?;
                out.out_axes = Some(
                    transform_config::out_axes_from_json(&json)
                        .map_err(|err| error::from_core(py, err))?,
                );
            }
            "argnums" => {
                let json = convert_value(py, &value, "argnums")?;
                out.argnums = Some(
                    transform_config::int_list_from_json(&json, "config.argnums")
                        .map_err(|err| error::from_core(py, err))?,
                );
            }
            "tangent_inputs" => {
                let json = convert_value(py, &value, "tangent_inputs")?;
                out.tangent_inputs = Some(
                    transform_config::int_list_from_json(&json, "config.tangent_inputs")
                        .map_err(|err| error::from_core(py, err))?,
                );
            }
            "cotangent_inputs" => {
                let json = convert_value(py, &value, "cotangent_inputs")?;
                out.cotangent_inputs = Some(
                    transform_config::int_list_from_json(&json, "config.cotangent_inputs")
                        .map_err(|err| error::from_core(py, err))?,
                );
            }
            "extension_rule" => {
                if value.is_none() {
                    extension = None;
                } else if value.is_callable() {
                    extension = Some(PyExtensionRule {
                        callback: value.unbind(),
                    });
                } else {
                    return Err(error::type_error(format!(
                        "transform: config.extension_rule must be callable, got {}",
                        type_name(&value)
                    )));
                }
            }
            other => {
                return Err(error::raise(
                    py,
                    "TransformError",
                    format!(
                        "transform: unknown config key {other:?}; accepted keys are {}",
                        transform_config::CONFIG_KEYS.join(", ")
                    ),
                ))
            }
        }
    }
    Ok((out, extension))
}

/// Decode `config.axes`.  A Python dict is read directly so its keys may be
/// value-id ints or decimal strings; every other shape goes through the
/// envelope JSON conversion (`transform_config::apply_axes`).
fn apply_axes_from_py(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    config: &mut evoxc_compiler::TransformConfig,
) -> PyResult<()> {
    if let Ok(dict) = value.cast::<PyDict>() {
        let mut entries = Vec::with_capacity(dict.len());
        for (key, axis_value) in dict.iter() {
            let id = python_value_id(py, &key)?;
            let axis_json = json_from_py(&axis_value, "config.axes")
                .map_err(|err| error::object_error(py, err, "transform: config.axes"))?;
            let axis = transform_config::axis_from_json(&axis_json, "config.axes")
                .map_err(|err| error::from_core(py, err))?;
            entries.push((id, axis));
        }
        config.axes = entries;
        return Ok(());
    }
    let json = json_from_py(value, "config.axes")
        .map_err(|err| error::object_error(py, err, "transform: config.axes"))?;
    transform_config::apply_axes(config, &json).map_err(|err| error::from_core(py, err))
}

/// A value-id dict key: an int or a decimal string.
fn python_value_id(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<ValueId> {
    if let Ok(text) = value.cast::<PyString>() {
        let text = text
            .to_str()
            .map_err(|err| error::object_error(py, err, "transform: config.axes"))?;
        return transform_config::value_id_from_text(text, "config.axes")
            .map_err(|err| error::from_core(py, err));
    }
    if value.is_instance_of::<PyInt>() {
        let number: i64 = value.extract().map_err(|_| {
            error::raise(
                py,
                "TransformError",
                "config.axes: a value id does not fit a 64-bit integer",
            )
        })?;
        if number < 0 || number > u32::MAX as i64 {
            return Err(error::raise(
                py,
                "TransformError",
                format!("config.axes: value id {number} is outside the module's u32 id space"),
            ));
        }
        return Ok(ValueId(number as u32));
    }
    Err(error::raise(
        py,
        "TransformError",
        format!(
            "config.axes: value-id keys must be ints or decimal strings, got {}",
            type_name(value)
        ),
    ))
}

/// Convert one config value through the envelope JSON conversion, tagging any
/// conversion failure with the config key.
fn convert_value(py: Python<'_>, value: &Bound<'_, PyAny>, key: &str) -> PyResult<Json> {
    json_from_py(value, &format!("config.{key}")).map_err(|err| {
        error::object_error(py, err, &format!("transform: config.{key}"))
    })
}

fn type_name(value: &Bound<'_, PyAny>) -> String {
    match value.get_type().name() {
        Ok(name) => name.to_string_lossy().into_owned(),
        Err(_) => String::from("<unknown type>"),
    }
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
            json_kind(&json)
        ))
    })?;
    // `attrs_from_json` is the envelope's tolerant decoder: an INTS attribute
    // whose declared default is `null` accepts JSON `null`, exactly as the
    // Python builder serializes it.
    evoxc_core::envelope::attrs_from_json(map, schema)
        .map_err(|err| error::from_core(py, err))
}

/// Fixtures shared by both test modules: a type dict exactly as
/// `operand_types` carries it and the minimal valid full envelope
/// (`callee: (f32[2]) -> f32[2]` returning its argument).
#[cfg(test)]
mod test_support {
    use super::*;
    use evoxc_core::{Dim, Shape};
    use serde_json::json;

    /// A static shape, spelled the way `Dim` wants it.
    pub(super) fn shape(dims: &[i64]) -> Shape {
        dims.iter().map(|&n| Dim::int(n)).collect()
    }

    /// A type dict exactly as `operand_types` carries it.
    pub(super) fn ty(dtype: &str, shape: &[i64]) -> Json {
        json!({
            "dtype": dtype,
            "shape": shape.iter().map(|&n| json!({"int": n})).collect::<Vec<_>>(),
        })
    }

    /// The minimal valid full envelope: `callee: (f32[2]) -> f32[2]` returns
    /// its argument.
    pub(super) fn envelope() -> Json {
        json!({
            "format": evoxc_core::envelope::FORMAT,
            "version": evoxc_core::envelope::VERSION,
            "module": {"name": "m"},
            "functions": [{
                "name": "callee",
                "input_types": [ty("float32", &[2])],
                "output_types": [ty("float32", &[2])],
                "block": {
                    "arguments": [{"id": 0, "type": ty("float32", &[2])}],
                    "ops": [{"id": 0, "ref": "0"}],
                },
            }],
            "ops": [{"id": 0, "name": "return", "operands": [0], "results": []}],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{envelope, shape, ty};
    use super::*;
    use evoxc_core::{AttrValue, Attrs, DType, ValueType};
    use serde_json::json;

    /// The kind list is the contract's closed set, in a stable order.
    #[test]
    fn transform_kinds_match_the_contract() {
        assert_eq!(TRANSFORM_KINDS, ["vectorize", "vmap", "grad", "jvp", "vjp"]);
    }

    // --- the `infer` module argument ---------------------------------------

    fn call_attrs(callee: &str) -> Attrs {
        Attrs::from_pairs([("callee", AttrValue::Str(callee.to_string()))])
    }

    /// `infer_op("call", ..)` against a decoded module, as `infer` runs it.
    fn call_outputs(module: &Program) -> evoxc_core::Result<Vec<ValueType>> {
        evoxc_core::infer_op(
            "call",
            &[ValueType::new(DType::Float32, shape(&[2]))],
            &call_attrs("callee"),
            None,
            Some(module),
        )
    }

    /// The argument-shape message of a rejected context.
    fn shape_message(err: ContextError) -> String {
        match err {
            ContextError::Shape(message) => message,
            ContextError::Core(err) => {
                panic!("expected an argument-shape error, got {}: {}", err.kind_str(), err.message)
            }
        }
    }

    #[test]
    fn a_full_envelope_module_still_decodes() {
        let program = decode_module_json(&envelope()).unwrap();
        let callee = program.function("callee").unwrap();
        assert_eq!(
            callee.input_types,
            vec![ValueType::new(DType::Float32, shape(&[2]))]
        );
        assert_eq!(
            call_outputs(&program).unwrap(),
            vec![ValueType::new(DType::Float32, shape(&[2]))]
        );
    }

    #[test]
    fn a_full_envelope_keeps_its_persistence_diagnostics() {
        let mut broken = envelope();
        broken["format"] = json!("not-an-envelope");
        let err = decode_module_json(&broken).unwrap_err();
        match err {
            ContextError::Core(err) => {
                assert_eq!(err.kind_str(), "PersistenceError");
                assert!(err.message.starts_with("module: "), "{}", err.message);
            }
            ContextError::Shape(message) => panic!("expected a native error, got {message}"),
        }
    }

    #[test]
    fn a_signature_only_context_resolves_the_callee_output_types() {
        let context = json!({"functions": [["callee", [ty("float32", &[4, 3])]]]});
        let program = decode_module_json(&context).unwrap();
        assert_eq!(
            call_outputs(&program).unwrap(),
            vec![ValueType::new(DType::Float32, shape(&[4, 3]))]
        );
    }

    #[test]
    fn a_signature_only_context_accepts_the_three_element_and_dict_forms() {
        // [name, input types, output types] — inputs are carried, not inferred.
        let context = json!({"functions": [
            ["callee", [ty("float32", &[2])], [ty("int64", &[])]],
        ]});
        let program = decode_module_json(&context).unwrap();
        assert_eq!(
            program.function("callee").unwrap().input_types,
            vec![ValueType::new(DType::Float32, shape(&[2]))]
        );
        assert_eq!(call_outputs(&program).unwrap(), vec![ValueType::scalar(DType::Int64)]);

        // {"name", "input_types"?, "output_types"}.
        let context = json!({"functions": [{
            "name": "callee",
            "output_types": [ty("float32", &[7])],
        }]});
        let program = decode_module_json(&context).unwrap();
        assert!(program.function("callee").unwrap().input_types.is_empty());
        assert_eq!(
            call_outputs(&program).unwrap(),
            vec![ValueType::new(DType::Float32, shape(&[7]))]
        );

        // An empty context is a valid (callee-less) module.
        assert!(decode_module_json(&json!({"functions": []}))
            .unwrap()
            .functions
            .is_empty());
    }

    #[test]
    fn an_unknown_callee_stays_a_verification_error_naming_it() {
        let program = decode_module_json(&json!({"functions": [["callee", []]]})).unwrap();
        let err = evoxc_core::infer_op(
            "call",
            &[],
            &call_attrs("nope"),
            None,
            Some(&program),
        )
        .unwrap_err();
        assert_eq!(err.kind_str(), "VerificationError");
        assert!(err.message.contains("no function named 'nope'"), "{}", err.message);
    }

    #[test]
    fn a_malformed_module_context_is_an_argument_shape_error() {
        // Not a dict at all.
        let message = shape_message(decode_module_json(&json!([["callee", []]])).unwrap_err());
        assert!(message.contains("module must be a dict"), "{message}");

        // A dict with neither a format tag nor a functions list.
        let message = shape_message(decode_module_json(&json!({"callees": []})).unwrap_err());
        assert!(message.contains("got keys: callees"), "{message}");
        let message = shape_message(decode_module_json(&json!({})).unwrap_err());
        assert!(message.contains("got keys: <none>"), "{message}");

        // "functions" must be a list.
        let message = shape_message(decode_module_json(&json!({"functions": 3})).unwrap_err());
        assert!(message.contains("module.functions must be a list"), "{message}");

        // A wrong entry element count.
        let message = shape_message(
            decode_module_json(&json!({"functions": [["callee"]]})).unwrap_err(),
        );
        assert!(message.contains("module.functions[0]"), "{message}");
        assert!(message.contains("1 element(s)"), "{message}");
        let message = shape_message(
            decode_module_json(&json!({"functions": [["callee", [], [], 4]]})).unwrap_err(),
        );
        assert!(message.contains("4 element(s)"), "{message}");

        // A non-string / empty name, a non-list non-dict entry, a dict without
        // its output types, and an unknown dict key.
        let message = shape_message(
            decode_module_json(&json!({"functions": [[7, []]]})).unwrap_err(),
        );
        assert!(message.contains("function names must be strings"), "{message}");
        let message = shape_message(
            decode_module_json(&json!({"functions": [["", []]]})).unwrap_err(),
        );
        assert!(message.contains("empty function name"), "{message}");
        let message =
            shape_message(decode_module_json(&json!({"functions": ["callee"]})).unwrap_err());
        assert!(message.contains("must be a list or a dict"), "{message}");
        let message = shape_message(
            decode_module_json(&json!({"functions": [{"name": "callee"}]})).unwrap_err(),
        );
        assert!(message.contains("missing its \"output_types\""), "{message}");
        let message = shape_message(
            decode_module_json(&json!({"functions": [
                {"name": "callee", "output_types": [], "extra": 1},
            ]}))
            .unwrap_err(),
        );
        assert!(message.contains("unknown key \"extra\""), "{message}");
    }

    #[test]
    fn a_malformed_signature_type_is_a_native_error_tagged_with_its_path() {
        let context = json!({"functions": [["callee", [{"dtype": "nope", "shape": []}]]]});
        match decode_module_json(&context).unwrap_err() {
            ContextError::Core(err) => {
                assert_eq!(err.kind_str(), "DTypeError");
                assert!(err.message.starts_with("module.functions[0] types: "), "{}", err.message);
            }
            ContextError::Shape(message) => panic!("expected a native error, got {message}"),
        }
    }
}

/// Interpreter-backed tests: config decoding and the extension-rule bridge.  They
/// need a linked libpython, so they only build with
/// `--no-default-features --features auto-init`.
#[cfg(all(test, feature = "auto-init"))]
mod py_tests {
    use super::test_support::envelope;
    use super::*;
    use pyo3::types::PyList;

    fn kind_of(err: &PyErr, py: Python<'_>) -> String {
        err.value(py)
            .getattr("kind")
            .expect("a NativeError carries .kind")
            .extract()
            .expect("kind is a str")
    }

    #[test]
    fn config_parses_every_key_and_both_axes_shapes() {
        Python::attach(|py| {
            let config = PyDict::new(py);
            let axes = PyDict::new(py);
            axes.set_item(0_i64, 0_i64).unwrap();
            axes.set_item("1", py.None()).unwrap();
            config.set_item("axes", axes).unwrap();
            config.set_item("batch_dim_name", "sample").unwrap();
            config.set_item("out_axes", 1_i64).unwrap();
            config.set_item("argnums", PyList::new(py, [0_i64, 2]).unwrap()).unwrap();
            config.set_item("tangent_inputs", PyList::new(py, [0_i64]).unwrap()).unwrap();
            config.set_item("cotangent_inputs", PyList::new(py, [0_i64]).unwrap()).unwrap();
            let (parsed, extension) = parse_config(py, &config).unwrap();
            assert!(extension.is_none());
            assert_eq!(
                parsed.axes,
                vec![(ValueId(0), Some(0)), (ValueId(1), None)]
            );
            assert_eq!(parsed.batch_dim_name.as_deref(), Some("sample"));
            assert_eq!(parsed.out_axes, Some(evoxc_compiler::OutAxes::All(Some(1))));
            assert_eq!(parsed.argnums, Some(vec![0, 2]));
            assert_eq!(parsed.tangent_inputs, Some(vec![0]));
            assert_eq!(parsed.cotangent_inputs, Some(vec![0]));

            // The pair-list and per-position forms.
            let config = PyDict::new(py);
            let pairs = PyList::new(
                py,
                [
                    PyList::new(py, [0_i64, 0]).unwrap(),
                    PyList::new(py, [1_i64, 0]).unwrap(),
                ],
            )
            .unwrap();
            config.set_item("axes", pairs).unwrap();
            let (parsed, _) = parse_config(py, &config).unwrap();
            assert_eq!(
                parsed.axes,
                vec![(ValueId(0), Some(0)), (ValueId(1), Some(0))]
            );

            let config = PyDict::new(py);
            let positional = PyList::new(py, [0_i64, 0]).unwrap();
            config.set_item("axes", positional).unwrap();
            let (parsed, _) = parse_config(py, &config).unwrap();
            assert_eq!(parsed.input_axes, vec![Some(0), Some(0)]);
            assert!(parsed.axes.is_empty());
        });
    }

    #[test]
    fn config_rejects_unknown_keys_and_bad_values() {
        Python::attach(|py| {
            let config = PyDict::new(py);
            config.set_item("axis", 0_i64).unwrap();
            let err = parse_config(py, &config).unwrap_err();
            assert_eq!(kind_of(&err, py), "TransformError");
            assert!(err.to_string().contains("unknown config key \"axis\""));

            let config = PyDict::new(py);
            config.set_item("axes", 3_i64).unwrap();
            let err = parse_config(py, &config).unwrap_err();
            assert_eq!(kind_of(&err, py), "TransformError");
            assert!(err.to_string().contains("config.axes"));

            let config = PyDict::new(py);
            config.set_item("extension_rule", 3_i64).unwrap();
            let err = parse_config(py, &config).unwrap_err();
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert!(err.to_string().contains("must be callable"));
        });
    }

    #[test]
    fn infer_accepts_a_signature_only_module_and_rejects_a_malformed_one() {
        Python::attach(|py| {
            let operands = py.eval(c"[]", None, None).unwrap();
            let attrs = py.eval(c"{'callee': 'callee'}", None, None).unwrap();

            // `call` resolves its outputs from the condition context's signature.
            let module = py
                .eval(c"{'functions': [('callee', [{'dtype': 'int64', 'shape': []}])]}", None, None)
                .unwrap();
            let results = infer(py, "call", &operands, Some(&attrs), None, Some(&module)).unwrap();
            let results = json_from_py(results.bind(py), "results").unwrap();
            assert_eq!(results, serde_json::json!([{"dtype": "int64", "shape": []}]));

            // A full envelope still works through the same entry point.
            let module = py_from_json(py, &envelope()).unwrap();
            let results = infer(py, "call", &operands, Some(&attrs), None, Some(&module)).unwrap();
            let results = json_from_py(results.bind(py), "results").unwrap();
            assert_eq!(
                results,
                serde_json::json!([{"dtype": "float32", "shape": [{"int": 2}]}])
            );

            // An unknown callee is still a VerificationError naming it.
            let unknown = py.eval(c"{'callee': 'nope'}", None, None).unwrap();
            let err = infer(py, "call", &operands, Some(&unknown), None, Some(&module))
                .unwrap_err();
            assert_eq!(kind_of(&err, py), "VerificationError");
            assert!(err.to_string().contains("no function named 'nope'"), "{err}");

            // A malformed context is an argument-shape TypeError, not a
            // silent fallback to "no module".
            let bad = py.eval(c"{'callees': []}", None, None).unwrap();
            let err =
                infer(py, "call", &operands, Some(&attrs), None, Some(&bad)).unwrap_err();
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert!(err.to_string().contains("got keys: callees"), "{err}");

            // A non-dict module too.
            let bad = py.eval(c"[['callee', []]]", None, None).unwrap();
            let err =
                infer(py, "call", &operands, Some(&attrs), None, Some(&bad)).unwrap_err();
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert!(err.to_string().contains("module must be a dict"), "{err}");
        });
    }

    #[test]
    fn a_python_extension_rule_returns_none_or_a_response() {
        Python::attach(|py| {
            let mut request = ExtensionRequest::new(
                evoxc_compiler::ExtensionKind::Batching,
                &evoxc_compiler::RuleOp {
                    name: "block_call".to_string(),
                    attributes: {
                        let mut attrs = evoxc_core::Attrs::new();
                        attrs.insert("block_name", evoxc_core::AttrValue::Str("foo".to_string()));
                        attrs
                    },
                    original_operand_types: vec![evoxc_core::ValueType::scalar(
                        evoxc_core::DType::Float32,
                    )],
                    original_result_types: vec![evoxc_core::ValueType::scalar(
                        evoxc_core::DType::Float32,
                    )],
                    operands: Vec::new(),
                    operand_types: vec![evoxc_core::ValueType::scalar(
                        evoxc_core::DType::Float32,
                    )],
                    results: Vec::new(),
                    result_types: Vec::new(),
                    location: None,
                },
            );
            request.axes = Some(vec![Some(0)]);

            // None means "no rule".
            let callback: Py<PyAny> = py
                .eval(c"lambda request: None", None, None)
                .unwrap()
                .unbind();
            assert!(call_python_rule(py, &callback, &request).unwrap().is_none());

            // A dict is a response, with the request shape checked by the lambda.
            let callback: Py<PyAny> = py
                .eval(
                    c"lambda request: {
                        'module': {'format': 'evoxc-native-ir', 'version': 1,
                                   'module': {'name': 'rule', 'metadata': {}},
                                   'functions': [], 'ops': []},
                        'entry': 'rule',
                        'output_ids': [request['operand_types'] and 2],
                        'output_axes': [0],
                    }",
                    None,
                    None,
                )
                .unwrap()
                .unbind();
            let response = call_python_rule(py, &callback, &request)
                .unwrap()
                .expect("a dict is a rule");
            assert_eq!(response.entry, "rule");
            assert_eq!(response.output_ids, vec![ValueId(2)]);
            assert_eq!(response.output_axes, Some(vec![Some(0)]));

            // An exception keeps its text and becomes a TransformError.
            let callback: Py<PyAny> = py
                .eval(c"lambda request: (_ for _ in ()).throw(ValueError('boom'))", None, None)
                .unwrap()
                .unbind();
            let err = call_python_rule(py, &callback, &request).unwrap_err();
            assert_eq!(err.kind_str(), "TransformError");
            assert!(err.message.contains("block_call"), "{}", err.message);
            assert!(err.message.contains("boom"), "{}", err.message);
        });
    }
}
