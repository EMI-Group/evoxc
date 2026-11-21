//! Python objects ⇄ envelope JSON.
//!
//! `native/CONTEXT.md` makes the envelope the *whole* FFI payload: one JSON-able
//! dict per call.  This module is the only place the binding touches Python
//! values, and it converts each call **once** — a per-node Python callback would
//! cross the GIL boundary once per op instead of once per call, and would make
//! the native dispatch loop depend on Python object lifetimes.
//!
//! The accepted Python surface is deliberately the JSON data model plus the
//! sequences the contract spells out:
//!
//! * `None`, `bool`, `int`, `float`, `str`
//! * `dict` with `str` keys
//! * `list` **and** `tuple` (the contract's `operand_types` / `result_types` are
//!   sequences, and callers legitimately pass tuples)
//!
//! Anything else (a `set`, a generator, a numpy array, a complex number) is
//! rejected with a `TypeError` naming the path — never coerced silently.
//!
//! Conversion never mutates its inputs and never allocates a Python string per
//! element: paths are built in one reused `String`.
use std::convert::Infallible;
use std::fmt::Write as _;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde_json::{Map, Number, Value as Json};
use crate::error;
/// Decode one Python value into envelope JSON, tagging messages with `root`.
pub fn json_from_py(value: &Bound<'_, PyAny>, root: &str) -> PyResult<Json> {
    let mut path = String::with_capacity(root.len() + 16);
    path.push_str(root);
    encode_value(value, &mut path)
}
/// Encode one envelope JSON value into a fresh Python object.
pub fn py_from_json<'py>(py: Python<'py>, value: &Json) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Json::Null => py.None().into_bound(py),
        Json::Bool(flag) => PyBool::new(py, *flag).to_owned().into_any(),
        Json::Number(number) => number_to_py(py, number),
        Json::String(text) => PyString::new(py, text).into_any(),
        Json::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(py_from_json(py, item)?)?;
            }
            list.into_any()
        }
        Json::Object(map) => {
            let dict = PyDict::new(py);
            for (key, item) in map {
                dict.set_item(key, py_from_json(py, item)?)?;
            }
            dict.into_any()
        }
    })
}
/// Numbers keep their JSON integer-ness: an `i64`/`u64` becomes a Python `int`,
/// anything else a `float`.  (Python `bool` is never reached — it is checked
/// before `int` on the way in and encoded as JSON `true`/`false`.)
fn number_to_py<'py>(py: Python<'py>, number: &Number) -> Bound<'py, PyAny> {
    if let Some(int) = number.as_i64() {
        return never(int.into_pyobject(py)).into_any();
    }
    if let Some(uint) = number.as_u64() {
        return never(uint.into_pyobject(py)).into_any();
    }
    let float = number.as_f64().unwrap_or(f64::NAN);
    never(float.into_pyobject(py)).into_any()
}
/// `Result<T, Infallible>` carries no failure; spelling that out keeps the
/// conversion free of `unwrap`/`expect` (the crate never panics on user input).
fn never<T>(value: Result<T, Infallible>) -> T {
    match value {
        Ok(value) => value,
        Err(never) => match never {},
    }
}
fn encode_value(value: &Bound<'_, PyAny>, path: &mut String) -> PyResult<Json> {
    if value.is_none() {
        return Ok(Json::Null);
    }
    // `bool` is an `int` subclass in Python: test it first or `True` becomes 1.
    if value.is_instance_of::<PyBool>() {
        let flag: bool = value.extract()?;
        return Ok(Json::Bool(flag));
    }
    if value.is_instance_of::<PyInt>() {
        if let Ok(int) = value.extract::<i64>() {
            return Ok(Json::from(int));
        }
        if let Ok(uint) = value.extract::<u64>() {
            return Ok(Json::from(uint));
        }
        return Err(error::type_error(format!(
            "{path}: integer {} does not fit the envelope's 64-bit number range",
            python_text(value)
        )));
    }
    if value.is_instance_of::<PyFloat>() {
        let float: f64 = value.extract()?;
        return Number::from_f64(float).map(Json::Number).ok_or_else(|| {
            error::type_error(format!(
                "{path}: {float} is not a finite JSON number (NaN and infinity have no \
                 envelope form; pass them as an ndarray attribute payload instead)"
            ))
        });
    }
    if let Ok(text) = value.cast::<PyString>() {
        let text = text.to_str().map_err(|err| {
            error::object_error(value.py(), err, &format!("{path}: undecodable string"))
        })?;
        return Ok(Json::String(text.to_string()));
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        let mut map = Map::new();
        for (key, item) in dict.iter() {
            let key = key.cast::<PyString>().map_err(|_| {
                error::type_error(format!(
                    "{path}: dictionary keys must be strings, got {}",
                    type_name(&key)
                ))
            })?;
            let key = key.to_str().map_err(|err| {
                error::object_error(value.py(), err, &format!("{path}: undecodable key"))
            })?;
            let mark = path.len();
            let _ = write!(path, ".{key}");
            let encoded = encode_value(&item, path);
            path.truncate(mark);
            map.insert(key.to_string(), encoded?);
        }
        return Ok(Json::Object(map));
    }
    if let Ok(list) = value.cast::<PyList>() {
        return encode_sequence(list.iter().collect(), path).map(Json::Array);
    }
    if let Ok(tuple) = value.cast::<PyTuple>() {
        return encode_sequence(tuple.iter().collect(), path).map(Json::Array);
    }
    Err(error::type_error(format!(
        "{path}: expected dict, list, tuple, str, bool, int, float or None, got {}",
        type_name(value)
    )))
}
fn encode_sequence(items: Vec<Bound<'_, PyAny>>, path: &mut String) -> PyResult<Vec<Json>> {
    let mut out = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        let mark = path.len();
        let _ = write!(path, "[{index}]");
        let encoded = encode_value(item, path);
        path.truncate(mark);
        out.push(encoded?);
    }
    Ok(out)
}
fn type_name(value: &Bound<'_, PyAny>) -> String {
    match value.get_type().name() {
        Ok(name) => name.to_string_lossy().into_owned(),
        Err(_) => String::from("<unknown type>"),
    }
}
fn python_text(value: &Bound<'_, PyAny>) -> String {
    match value.str() {
        Ok(text) => text.to_string_lossy().into_owned(),
        Err(_) => String::from("<unprintable>"),
    }
}
#[cfg(all(test, feature = "auto-init"))]
mod tests {
    use super::*;
    use pyo3::types::PySet;
    fn encode<'py>(
        py: Python<'py>,
        build: impl FnOnce(Python<'py>) -> Bound<'py, PyAny>,
    ) -> Json {
        let value = build(py);
        json_from_py(&value, "attributes").expect("conversion must succeed")
    }
    #[test]
    fn scalars_keep_their_json_kind() {
        Python::attach(|py| {
            assert_eq!(encode(py, |py| py.None().into_bound(py)), Json::Null);
            assert_eq!(encode(py, |py| PyBool::new(py, true).to_owned().into_any()), Json::Bool(true));
            let int = encode(py, |py| never(2_i64.into_pyobject(py)).into_any());
            assert_eq!(int, Json::from(2));
            assert!(int.as_i64().is_some(), "int must not become a float");
            let float = encode(py, |py| never(2.5_f64.into_pyobject(py)).into_any());
            assert_eq!(float, Json::from(2.5));
            assert!(float.as_f64().is_some() && float.as_i64().is_none());
            let text = encode(py, |py| PyString::new(py, "f32").into_any());
            assert_eq!(text, Json::String("f32".to_string()));
        });
    }
    #[test]
    fn nested_attributes_dicts_lists_and_tuples_round_trip() {
        Python::attach(|py| {
            let inner = PyDict::new(py);
            inner.set_item("dtype", "float32").unwrap();
            inner.set_item("shape", PyTuple::new(py, [2_i64, 3]).unwrap()).unwrap();
            let attrs = PyDict::new(py);
            attrs.set_item("value", inner).unwrap();
            attrs.set_item("permutation", PyList::new(py, [1_i64, 0]).unwrap()).unwrap();
            attrs.set_item("keepdims", false).unwrap();
            let encoded = json_from_py(attrs.as_any(), "attributes").unwrap();
            let object = encoded.as_object().expect("a dict encodes as an object");
            // Tuples decode as arrays: the envelope has one sequence form.
            assert_eq!(
                object["value"]["shape"],
                Json::Array(vec![Json::from(2), Json::from(3)])
            );
            assert_eq!(object["permutation"], Json::Array(vec![Json::from(1), Json::from(0)]));
            assert_eq!(object["keepdims"], Json::Bool(false));
            // JSON -> Python -> JSON is the identity.
            let back = py_from_json(py, &encoded).unwrap();
            assert_eq!(json_from_py(&back, "attributes").unwrap(), encoded);
        });
    }
    #[test]
    fn an_ndarray_payload_survives_the_python_crossing() {
        // Two little-endian f32 values (1.0, 2.0) as the Python builder emits them.
        let payload = Json::Object(Map::from_iter([(
            "__evoxc_ndarray__".to_string(),
            Json::Object(Map::from_iter([
                ("dtype".to_string(), Json::String("float32".to_string())),
                ("shape".to_string(), Json::Array(vec![Json::from(2)])),
                ("data_b64".to_string(), Json::String("AACAPwAAAEA=".to_string())),
            ])),
        )]));
        Python::attach(|py| {
            let py_value = py_from_json(py, &payload).unwrap();
            let back = json_from_py(&py_value, "attributes.value").unwrap();
            assert_eq!(back, payload);
            let array = evoxc_core::wire::array_from_json(&back).unwrap();
            assert_eq!(array.shape, vec![2]);
            assert_eq!(array.dtype.name(), "float32");
            let floats: Vec<f32> = array
                .data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            assert_eq!(floats, vec![1.0, 2.0]);
        });
    }
    #[test]
    fn numbers_outside_the_envelope_range_are_type_errors() {
        Python::attach(|py| {
            // 2**64 exceeds both the i64 and the u64 envelope range.
            let huge = never((i128::from(u64::MAX) + 1).into_pyobject(py)).into_any();
            let err = json_from_py(&huge, "attributes.big").unwrap_err();
            assert!(err.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert!(err.to_string().contains("attributes.big"));
            let nan = never(f64::NAN.into_pyobject(py)).into_any();
            let err = json_from_py(&nan, "attributes.nan").unwrap_err();
            assert!(err.to_string().contains("not a finite JSON number"));
        });
    }
    #[test]
    fn unsupported_types_name_the_path() {
        Python::attach(|py| {
            let attrs = PyDict::new(py);
            let nested = PyDict::new(py);
            let set = PySet::empty(py).unwrap();
            set.add("x").unwrap();
            nested.set_item("bad", set).unwrap();
            attrs.set_item("axes", nested).unwrap();
            let err = json_from_py(attrs.as_any(), "attributes").unwrap_err();
            let text = err.to_string();
            assert!(text.contains("attributes.axes.bad"), "got {text}");
            assert!(text.contains("set"), "the offending type must be named: {text}");
        });
    }
    #[test]
    fn non_string_dict_keys_are_rejected_with_the_path() {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item(3_i64, "x").unwrap();
            let err = json_from_py(dict.as_any(), "module").unwrap_err();
            assert!(err.to_string().contains("module: dictionary keys must be strings"));
        });
    }
    #[test]
    fn index_paths_point_at_the_offending_element() {
        Python::attach(|py| {
            let items = PyList::empty(py);
            items.append(1_i64).unwrap();
            let bad = PySet::empty(py).unwrap();
            bad.add("x").unwrap();
            items.append(bad).unwrap();
            let err = json_from_py(items.as_any(), "operand_types").unwrap_err();
            assert!(
                err.to_string()
                    .contains("operand_types[1]: expected dict, list, tuple, str"),
                "got {err}"
            );
        });
    }
}
