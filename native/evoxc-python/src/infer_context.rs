//! The `infer` module argument: full-envelope vs signature-only decoding.
//!
//! `infer_op` resolves a `call`'s callee types from
//! `module.function(callee)?.output_types`, so the `infer` `module` argument has
//! to end in a [`Program`].  [`decode_module`] accepts the two shapes of
//! `native/CONTEXT.md` and produces exactly that `Program` — either a full
//! `evoxc-native-ir` envelope, or a signature-only context synthesized into one
//! body-less `Function` per entry — so every downstream `call` path is
//! untouched.
//!
//! Argument-shape problems are plain `TypeError`s (never a silent fallback to
//! "no module" — `native/CONTEXT.md`), while a malformed *type* keeps the type
//! decoder's own `evoxc::core::Error` family tagged with its path.
use pyo3::prelude::*;
use serde_json::{Map, Value as Json};

use evoxc::core::graph::{Function, Program};

use crate::convert::json_from_py;
use crate::error;

/// Decode the `infer` `module` argument.
///
/// Two shapes are accepted, both ending in the `Program` that `infer_op` needs
/// for `call`-callee resolution (`module.function(callee)?.output_types`):
///
/// * a full `evoxc-native-ir` envelope — recognised by its `"format"` tag and
///   decoded by [`evoxc::core::envelope::decode`], exactly as before;
/// * a signature-only context — a dict carrying `"functions"`, a list of
///   `[name, [output types]]`, `[name, [input types], [output types]]` or
///   `{"name": …, "input_types": …, "output_types": …}` entries.  A `Program`
///   with one signature-only `Function` per entry is synthesized, so nothing
///   downstream changes.
///
/// Types use the same `{"dtype": str, "shape": [dim, …]}` dicts as
/// `operand_types` (`decode_types`' decoder, [`evoxc::core::wire`]).  A
/// non-dict module, a missing `"functions"` list, a malformed entry or an
/// unknown dict key is a `TypeError` (argument shape, like every other
/// decoder); a malformed type is the type decoder's own `VerificationError`
/// tagged with its path.
pub(crate) fn decode_module(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Program> {
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
        return evoxc::core::envelope::decode(json)
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
/// `NativeError` from `evoxc::core`).
#[derive(Debug)]
enum ContextError {
    Shape(String),
    Core(evoxc::core::Error),
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

/// One decoded signature entry: `(name, input types, output types)`.
type Signature = (String, Vec<evoxc::core::ValueType>, Vec<evoxc::core::ValueType>);

/// Decode one `module.functions[i]` signature entry.
fn signature_entry(entry: &Json, at: &str) -> std::result::Result<Signature, ContextError> {
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
) -> std::result::Result<Vec<evoxc::core::ValueType>, ContextError> {
    evoxc::core::wire::value_types_from_json(value)
        .map_err(|err| ContextError::Core(err.context(format!("{at} types"))))
}

/// The JSON kind of a value, for argument-shape messages.
pub(crate) fn json_kind(value: &Json) -> &'static str {
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

/// Fixtures shared by the module-context tests and the interpreter-backed
/// entry-point test: a type dict exactly as `operand_types` carries it, the
/// minimal valid full envelope (`callee: (f32[2]) -> f32[2]` returning its
/// argument), and the `NativeError.kind` reader.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use evoxc::core::{Dim, Shape};
    use serde_json::json;

    /// A static shape, spelled the way `Dim` wants it.
    pub(crate) fn shape(dims: &[i64]) -> Shape {
        dims.iter().map(|&n| Dim::int(n)).collect()
    }

    /// A type dict exactly as `operand_types` carries it.
    pub(crate) fn ty(dtype: &str, shape: &[i64]) -> Json {
        json!({
            "dtype": dtype,
            "shape": shape.iter().map(|&n| json!({"int": n})).collect::<Vec<_>>(),
        })
    }

    /// The minimal valid full envelope: `callee: (f32[2]) -> f32[2]` returns
    /// its argument.
    pub(crate) fn envelope() -> Json {
        json!({
            "format": evoxc::core::envelope::FORMAT,
            "version": evoxc::core::envelope::VERSION,
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

    /// A `NativeError`'s `.kind` family (needs a linked libpython, so it lives
    /// behind the `auto-init` feature).
    #[cfg(feature = "auto-init")]
    pub(crate) fn kind_of(err: &PyErr, py: Python<'_>) -> String {
        err.value(py)
            .getattr("kind")
            .expect("a NativeError carries .kind")
            .extract()
            .expect("kind is a str")
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{envelope, shape, ty};
    use super::*;
    use evoxc::core::{AttrValue, Attrs, DType, ValueType};
    use serde_json::json;

    // --- the `infer` module argument ---------------------------------------

    fn call_attrs(callee: &str) -> Attrs {
        Attrs::from_pairs([("callee", AttrValue::Str(callee.to_string()))])
    }

    /// `infer_op("call", ..)` against a decoded module, as `infer` runs it.
    fn call_outputs(module: &Program) -> evoxc::core::Result<Vec<ValueType>> {
        evoxc::core::infer_op(
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
        let err = evoxc::core::infer_op(
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

/// Interpreter-backed test: the `infer` entry point over both module shapes.
/// It needs a linked libpython, so it only builds with
/// `--no-default-features --features auto-init`.
#[cfg(all(test, feature = "auto-init"))]
mod py_tests {
    use super::test_support::{envelope, kind_of};
    use super::*;
    use crate::convert::{json_from_py, py_from_json};
    use crate::module_ops::infer;

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
}
