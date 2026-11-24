//! The `batch_functions` entry point (`evoxc._evoxc_native.batch_functions`).
//!
//! Mirrors [`crate::module_ops::transform`]: decode every input program as an
//! `evoxc-native-ir` envelope while the GIL is held, decode the config dict into
//! the engine's [`BatchFunctionsConfig`] (see [`transform_config`]), hand the
//! pure native merge to `Python::detach` with the very same extension-rule
//! bridge `transform` builds, then encode the merged program and attach the
//! `BatchReport`.
//!
//! The FFI call shape is the one
//! `native/evoxc-compiler/tests/parity_transform/run_batch_parity.py` drives:
//!
//! ```text
//! batch_functions(programs: list[dict], config: dict) -> dict
//! ```
//!
//! Returns `{"module": <merged evoxc-native-ir envelope>, "batch_report": {...}}`
//! where the report is exactly the 12 keys of `BatchReport::to_json`.
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use serde_json::{Map, Value as Json};

use evoxc::compiler::batch_functions::batch_functions_with_env;
use evoxc::compiler::{ExtensionRule, TransformEnv};
use evoxc::core::graph::Program;

use crate::convert::{json_from_py, py_from_json};
use crate::error;
use crate::module_ops::{extension_rule_from_value, PyExtensionRule};
use crate::transform_config;

/// `batch_functions(programs, config)`.
///
/// `programs` is a list/tuple of `evoxc-native-ir` envelopes (one per function
/// of the population); `config` accepts `axes`, `out_axes`, `switch_ops`,
/// `bands`, `alignment` and the optional `extension_rule` callable.  Unknown
/// keys and malformed values are explicit errors, never silently ignored.
pub fn batch_functions(
    py: Python<'_>,
    programs: &Bound<'_, PyAny>,
    config: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    // ---- GIL held: programs -> Program, config -> typed native config ------
    let programs = decode_programs(py, programs)?;
    let config = config.cast::<PyDict>().map_err(|_| {
        error::type_error(format!(
            "batch_functions: config must be a dict, got {}",
            type_name(config)
        ))
    })?;
    let (n_inputs, n_outputs) = arity(&programs);
    let (config, extension) = parse_batch_config(py, config, n_inputs, n_outputs)?;

    // ---- GIL released: the native merge (the bridge re-attaches for host
    // rules only) -----------------------------------------------------------
    let output = py
        .detach(move || {
            let env = TransformEnv::builtin();
            let env = match extension.as_ref() {
                Some(rule) => env.with_extension(rule as &dyn ExtensionRule),
                None => env,
            };
            batch_functions_with_env(&programs, &config, &env)
        })
        .map_err(|err| error::from_core(py, err))?;

    // ---- GIL held: merged program -> `{"module": envelope, "batch_report"}`
    //
    // The result nests the merged `evoxc-native-ir` envelope under `"module"`:
    // that is the shape `run_batch_parity.py` drives (`convert.from_native`
    // consumes `result["module"]`, the report is `result["batch_report"]`).
    let envelope = evoxc::core::envelope::encode(&output.program);
    let mut result = Map::new();
    result.insert("module".to_string(), envelope);
    result.insert("batch_report".to_string(), output.report.to_json());
    Ok(py_from_json(py, &Json::Object(result))?.unbind())
}

/// Decode `programs` into [`Program`]s, preserving the envelope diagnostics
/// (each failure is tagged with its `programs[i]` position).
fn decode_programs(py: Python<'_>, programs: &Bound<'_, PyAny>) -> PyResult<Vec<Program>> {
    let items: Vec<Bound<'_, PyAny>> = if let Ok(list) = programs.cast::<PyList>() {
        list.iter().collect()
    } else if let Ok(tuple) = programs.cast::<PyTuple>() {
        tuple.iter().collect()
    } else {
        return Err(error::type_error(format!(
            "batch_functions: programs must be a list/tuple of dicts (evoxc-native-ir \
             envelopes), got {}",
            type_name(programs)
        )));
    };
    let mut decoded = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        let envelope = item.cast::<PyDict>().map_err(|_| {
            error::type_error(format!(
                "batch_functions: programs[{index}] must be a dict (an evoxc-native-ir \
                 envelope), got {}",
                type_name(item)
            ))
        })?;
        let path = format!("programs[{index}]");
        let json = json_from_py(envelope.as_any(), &path)
            .map_err(|err| error::object_error(py, err, &format!("batch_functions: {path}")))?;
        let program = evoxc::core::envelope::decode(&json)
            .map_err(|err| error::from_core(py, err.context(&path)))?;
        decoded.push(program);
    }
    Ok(decoded)
}

/// The entry function's `(tensor inputs, tensor outputs)` leaf counts.
///
/// `axes` / `out_axes` are flattened to one entry per leaf, so a bare option is
/// broadcast to this many entries.  A malformed program yields `(0, 0)`; the
/// engine then raises its own explicit diagnostic before the counts matter.
fn arity(programs: &[Program]) -> (usize, usize) {
    let Some(program) = programs.first() else {
        return (0, 0);
    };
    let Some(function) = program.functions.first() else {
        return (0, 0);
    };
    let Ok(block) = program.entry_block(function) else {
        return (0, 0);
    };
    let inputs = block.arguments.len();
    let outputs = program
        .terminator(block.id)
        .ok()
        .flatten()
        .map(|op| op.operands.len())
        .unwrap_or(0);
    (inputs, outputs)
}

/// Decode the Python config dict into the engine's config plus the optional
/// extension-rule bridge (the same bridge `transform` builds).
fn parse_batch_config(
    py: Python<'_>,
    config: &Bound<'_, PyDict>,
    n_inputs: usize,
    n_outputs: usize,
) -> PyResult<(evoxc::compiler::batch_functions::BatchFunctionsConfig, Option<PyExtensionRule>)> {
    let mut json = Map::new();
    let mut extension: Option<PyExtensionRule> = None;
    for (key, value) in config.iter() {
        let name = key.cast::<PyString>().map_err(|_| {
            error::type_error(format!(
                "batch_functions: config keys must be strings, got {}",
                type_name(&key)
            ))
        })?;
        let name = name
            .to_str()
            .map_err(|err| error::object_error(py, err, "batch_functions: config key"))?;
        match name {
            "extension_rule" => {
                extension = extension_rule_from_value(&value, "batch_functions")?;
            }
            "axes" | "out_axes" | "switch_ops" | "bands" | "alignment" => {
                let path = format!("config.{name}");
                let encoded = json_from_py(&value, &path).map_err(|err| {
                    error::object_error(py, err, &format!("batch_functions: {path}"))
                })?;
                json.insert(name.to_string(), encoded);
            }
            other => {
                return Err(error::raise(
                    py,
                    "TransformError",
                    format!(
                        "batch_functions: unknown config key {other:?}; accepted keys are {}",
                        transform_config::BATCH_CONFIG_KEYS.join(", ")
                    ),
                ))
            }
        }
    }
    let config = transform_config::parse_batch_config(&json, n_inputs, n_outputs)
        .map_err(|err| error::from_core(py, err))?;
    Ok((config, extension))
}

/// The Python type name for argument-shape messages.
fn type_name(value: &Bound<'_, PyAny>) -> String {
    match value.get_type().name() {
        Ok(name) => name.to_string_lossy().into_owned(),
        Err(_) => String::from("<unknown type>"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arity_of_an_empty_population_is_zero() {
        assert_eq!(arity(&[]), (0, 0));
    }

    #[cfg(all(test, feature = "auto-init"))]
    #[test]
    fn non_dict_programs_and_configs_are_type_errors() {
        Python::attach(|py| {
            // A non-dict program element names its position.
            let programs = PyList::new(py, [1_i64]).unwrap();
            let config = PyDict::new(py);
            let err = batch_functions(py, programs.as_any(), config.as_any()).unwrap_err();
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py), "{err}");
            assert!(err.to_string().contains("programs[0]"), "{err}");

            // A non-dict config.
            let programs = PyList::empty(py);
            let err = batch_functions(py, programs.as_any(), programs.as_any()).unwrap_err();
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py), "{err}");
            assert!(err.to_string().contains("config must be a dict"), "{err}");
        });
    }

    #[cfg(all(test, feature = "auto-init"))]
    #[test]
    fn unknown_config_keys_are_transform_errors() {
        Python::attach(|py| {
            let programs = PyList::empty(py);
            let config = PyDict::new(py);
            config.set_item("junk", 1).unwrap();
            let err = batch_functions(py, programs.as_any(), config.as_any()).unwrap_err();
            assert!(err.is_instance_of::<crate::error::NativeError>(py), "{err}");
            assert!(err.to_string().contains("unknown config key"), "{err}");
        });
    }
}
