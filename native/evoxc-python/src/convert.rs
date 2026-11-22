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
use std::collections::HashMap;
use std::convert::Infallible;
use std::fmt::Write as _;
use std::hash::{BuildHasherDefault, Hasher};

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde_json::{Map, Number, Value as Json};

use crate::error;

/// Decode one Python value into envelope JSON, tagging messages with `root`.
pub fn json_from_py(value: &Bound<'_, PyAny>, root: &str) -> PyResult<Json> {
    let path = Path {
        parent: None,
        segment: Segment::Root(root),
    };
    encode_value(value, &path)
}

/// One step of a decoded value's path (`attributes.axes[1]`).
///
/// The path used to be rendered into a `String` at every key and element — tens
/// of thousands of `write!` calls per transform for a path that is only ever
/// read by an error message.  It is a parent-linked chain instead: one stack
/// frame per recursion level, rendered exactly once, and only on failure.
enum Segment<'a> {
    Root(&'a str),
    Key(&'a str),
    Index(usize),
}

struct Path<'a> {
    parent: Option<&'a Path<'a>>,
    segment: Segment<'a>,
}

impl Path<'_> {
    /// A path is formatted as `root.key[index]…`, exactly as before.
    fn render(&self) -> String {
        let mut text = String::with_capacity(32);
        self.write(&mut text);
        text
    }

    fn write(&self, out: &mut String) {
        if let Some(parent) = self.parent {
            parent.write(out);
        }
        match &self.segment {
            Segment::Root(root) => out.push_str(root),
            Segment::Key(key) => {
                out.push('.');
                out.push_str(key);
            }
            Segment::Index(index) => {
                let _ = write!(out, "[{index}]");
            }
        }
    }
}

/// Encode one envelope JSON value into a fresh Python object.
///
/// A transform returns tens of thousands of small Python objects, and almost all
/// of them are *repeats*: a few dozen spellings, a handful of dtypes/shapes and
/// source locations, the same value ids and dims.  Per-call caches therefore
/// share those leaves and records (see [`PyCache`]); every other container
/// (`dict`, `list`) is still allocated per node, because a caller may mutate a
/// container in the returned envelope and aliasing one would be observable.
pub fn py_from_json<'py>(py: Python<'py>, value: &Json) -> PyResult<Bound<'py, PyAny>> {
    let mut cache = PyCache::new();
    py_from_json_inner(py, value, &mut cache)
}

/// Largest `int` kept in the dense per-call cache.
///
/// Value ids, op ids and dims of a large graph are well below this; the wider
/// the table the more memory each call touches, so it stops where repeated ids
/// stop being the common case.  Integers in `-5..=256` are already immortal
/// singletons in CPython and never enter the table.
const INT_CACHE_LIMIT: i64 = 8192;

/// FNV-1a, for the per-call cache tables.
///
/// Every key inserted into the returned envelope is looked up here — tens of
/// thousands of short strings and 64-bit fingerprints per call — and the default
/// SipHash costs more than the objects it saves.  The hasher is never exposed
/// to untrusted keys (a collision only costs one extra comparison), so a fast,
/// non-cryptographic hash is the right trade for an in-process cache.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
#[derive(Default)]
struct FnvHasher(u64);

impl Hasher for FnvHasher {
    fn write(&mut self, bytes: &[u8]) {
        if self.0 == 0 {
            self.0 = FNV_OFFSET_BASIS;
        }
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x100_0000_01b3);
        }
    }

    fn finish(&self) -> u64 {
        if self.0 == 0 {
            FNV_OFFSET_BASIS
        } else {
            self.0
        }
    }
}

/// Hash map with [`FnvHasher`]; `Hasher::default()` starts at FNV's offset basis.
type FnvMap<K, V> = HashMap<K, V, BuildHasherDefault<FnvHasher>>;

/// Per-call caches for the immutable leaves of the returned envelope.
///
/// * `strings` — dict keys and string values (`"id"`, `"dtype"`, `"float32"`,
///   every op name).  Reusing the object also lets `PyDict_SetItem` use the hash
///   cached on it.
/// * `ints` — value/op ids, dims and small attribute counts.  Python `int`s are
///   immutable, so sharing one is unobservable.
/// * `types` / `locations` — the two repeated *structural* records of the
///   envelope: `{"dtype": …, "shape": […]}` type dicts and
///   `{"file": …, "line": …, "col": …, "code_snippet": …}` locations.  Both are
///   keyed by the *content* of the native record (a 64-bit FNV fingerprint,
///   collisions resolved by comparing the JSON).  A graph repeats a handful of
///   types and locations across thousands of values, so one Python dict serves
///   every occurrence.  Both are only ever read downstream (`evoxc.ir`
///   deserialization reads them by field into its own `ValueType` / `Location`),
///   but they are *mutable* containers: they are only shared between the record
///   positions of one call, never across calls, and never handed to arbitrary
///   payloads — see [`is_type_object`] / [`is_location_object`].
struct PyCache<'py, 'json> {
    strings: FnvMap<Box<str>, Bound<'py, PyString>>,
    ints: Vec<Option<Bound<'py, PyInt>>>,
    records: FnvMap<u64, Vec<(&'json Json, Bound<'py, PyAny>)>>,
}

impl<'py, 'json> PyCache<'py, 'json> {
    fn new() -> PyCache<'py, 'json> {
        PyCache {
            strings: FnvMap::default(),
            ints: Vec::new(),
            records: FnvMap::default(),
        }
    }

    /// One interned `str` per spelling, for the duration of one call.
    fn string(&mut self, py: Python<'py>, text: &str) -> Bound<'py, PyString> {
        if let Some(hit) = self.strings.get(text) {
            return hit.clone();
        }
        let value = PyString::new(py, text);
        self.strings.insert(text.into(), value.clone());
        value
    }

    /// One shared `int` per value (CPython's own small-int singletons are used
    /// as-is rather than copied into the table).
    ///
    /// The table is indexed by the value itself, so only the strictly positive
    /// band above CPython's `-5..=256` singletons has a slot; every other value
    /// (a negative one, or one at/above [`INT_CACHE_LIMIT`]) goes straight
    /// through `PyInt::new`.
    fn int(&mut self, py: Python<'py>, value: i64) -> Bound<'py, PyInt> {
        if (257..INT_CACHE_LIMIT).contains(&value) {
            let index = value as usize;
            if index >= self.ints.len() {
                let grown = (index + 1).max(self.ints.len() * 2).max(64);
                self.ints.resize(grown, None);
            }
            if let Some(hit) = &self.ints[index] {
                return hit.clone();
            }
            let created = PyInt::new(py, value);
            self.ints[index] = Some(created.clone());
            return created;
        }
        PyInt::new(py, value)
    }
}

/// Is this JSON value exactly an envelope `Type` (`{"dtype": str, "shape": […]}`)?
///
/// The type cache only engages on values that answer yes, so an arbitrary
/// payload that happens to sit under a `"type"` attribute key is converted (and
/// kept distinct) like any other dict.
fn is_type_object(value: &Json) -> bool {
    matches!(
        value,
        Json::Object(map)
            if map.len() == 2
                && map.get("dtype").is_some_and(Json::is_string)
                && map.get("shape").is_some_and(Json::is_array)
    )
}

/// Is this JSON value exactly an envelope `Location`
/// (`{"file": str, "line": int, "col": int, "code_snippet": str | null}`)?
fn is_location_object(value: &Json) -> bool {
    matches!(
        value,
        Json::Object(map)
            if map.len() == 4
                && map.get("file").is_some_and(Json::is_string)
                && map.get("line").is_some_and(Json::is_number)
                && map.get("col").is_some_and(Json::is_number)
    )
}

/// 64-bit FNV-1a over a JSON subtree: structural markers keep `[1]` and `"1"`
/// apart, numbers hash by their integer (or float-bit) representation.
fn json_fingerprint(value: &Json) -> u64 {
    fn write(hash: &mut u64, bytes: &[u8]) {
        for byte in bytes {
            *hash ^= u64::from(*byte);
            *hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    fn walk(hash: &mut u64, value: &Json) {
        match value {
            Json::Null => write(hash, b"n"),
            Json::Bool(flag) => write(hash, if *flag { b"t" } else { b"f" }),
            Json::Number(number) => {
                if let Some(int) = number.as_i64() {
                    write(hash, b"i");
                    write(hash, &int.to_le_bytes());
                } else if let Some(uint) = number.as_u64() {
                    write(hash, b"u");
                    write(hash, &uint.to_le_bytes());
                } else {
                    write(hash, b"d");
                    write(
                        hash,
                        &number.as_f64().unwrap_or(f64::NAN).to_bits().to_le_bytes(),
                    );
                }
            }
            Json::String(text) => {
                write(hash, b"s");
                write(hash, &(text.len() as u64).to_le_bytes());
                write(hash, text.as_bytes());
            }
            Json::Array(items) => {
                write(hash, b"[");
                write(hash, &(items.len() as u64).to_le_bytes());
                for item in items {
                    walk(hash, item);
                }
                write(hash, b"]");
            }
            Json::Object(map) => {
                write(hash, b"{");
                write(hash, &(map.len() as u64).to_le_bytes());
                for (key, item) in map {
                    write(hash, &(key.len() as u64).to_le_bytes());
                    write(hash, key.as_bytes());
                    walk(hash, item);
                }
                write(hash, b"}");
            }
        }
    }
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    walk(&mut hash, value);
    hash
}

/// Convert one envelope type, reusing the Python dict of an identical type.
fn py_from_type<'py, 'json>(
    py: Python<'py>,
    value: &'json Json,
    cache: &mut PyCache<'py, 'json>,
) -> PyResult<Bound<'py, PyAny>> {
    if !is_type_object(value) {
        return py_from_json_inner(py, value, cache);
    }
    py_from_record(py, value, cache)
}

/// Convert one envelope location, reusing the Python dict of an identical
/// location (a large graph's ops share a handful of source lines).
fn py_from_location<'py, 'json>(
    py: Python<'py>,
    value: &'json Json,
    cache: &mut PyCache<'py, 'json>,
) -> PyResult<Bound<'py, PyAny>> {
    if !is_location_object(value) {
        return py_from_json_inner(py, value, cache);
    }
    py_from_record(py, value, cache)
}

/// Reuse the Python object built for an identical structural record.
fn py_from_record<'py, 'json>(
    py: Python<'py>,
    value: &'json Json,
    cache: &mut PyCache<'py, 'json>,
) -> PyResult<Bound<'py, PyAny>> {
    let key = json_fingerprint(value);
    if let Some(bucket) = cache.records.get(&key) {
        for (source, object) in bucket {
            if *source == value {
                return Ok(object.clone());
            }
        }
    }
    let built = py_from_json_inner(py, value, cache)?;
    cache
        .records
        .entry(key)
        .or_default()
        .push((value, built.clone()));
    Ok(built)
}

fn py_from_json_inner<'py, 'json>(
    py: Python<'py>,
    value: &'json Json,
    cache: &mut PyCache<'py, 'json>,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Json::Null => py.None().into_bound(py),
        Json::Bool(flag) => PyBool::new(py, *flag).to_owned().into_any(),
        Json::Number(number) => number_to_py(py, number, cache),
        Json::String(text) => cache.string(py, text).into_any(),
        Json::Array(items) => {
            let mut converted = Vec::with_capacity(items.len());
            for item in items {
                converted.push(py_from_json_inner(py, item, cache)?);
            }
            PyList::new(py, converted)?.into_any()
        }
        Json::Object(map) => {
            let dict = PyDict::new(py);
            for (key, item) in map {
                // The repeated *containers* of the envelope: type positions
                // (`op.results[].type`, `block.arguments[].type`,
                // `function.input_types` / `output_types`) and op locations.
                // Everything else is rebuilt per node.
                let converted = match key.as_str() {
                    "type" => py_from_type(py, item, cache)?,
                    "location" => py_from_location(py, item, cache)?,
                    "input_types" | "output_types" => match item {
                        Json::Array(types) => {
                            let mut converted = Vec::with_capacity(types.len());
                            for entry in types {
                                converted.push(py_from_type(py, entry, cache)?);
                            }
                            PyList::new(py, converted)?.into_any()
                        }
                        _ => py_from_json_inner(py, item, cache)?,
                    },
                    _ => py_from_json_inner(py, item, cache)?,
                };
                dict.set_item(cache.string(py, key), converted)?;
            }
            dict.into_any()
        }
    })
}

/// Numbers keep their JSON integer-ness: an `i64`/`u64` becomes a Python `int`,
/// anything else a `float`.  (Python `bool` is never reached — it is checked
/// before `int` on the way in and encoded as JSON `true`/`false`.)
fn number_to_py<'py, 'json>(
    py: Python<'py>,
    number: &Number,
    cache: &mut PyCache<'py, 'json>,
) -> Bound<'py, PyAny> {
    if let Some(int) = number.as_i64() {
        return cache.int(py, int).into_any();
    }
    if let Some(uint) = number.as_u64() {
        return match i64::try_from(uint) {
            Ok(int) => cache.int(py, int).into_any(),
            Err(_) => never(uint.into_pyobject(py)).into_any(),
        };
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

fn encode_value(value: &Bound<'_, PyAny>, path: &Path<'_>) -> PyResult<Json> {
    if value.is_none() {
        return Ok(Json::Null);
    }
    // Exact/`Py*_Check` casts (subclasses accepted, as before).  Dicts, lists,
    // tuples and strings are tried first: the envelope is almost entirely made
    // of them, and a cast is a cheaper check than the generic `is_instance_of`.
    if let Ok(dict) = value.cast::<PyDict>() {
        let mut map = Map::new();
        for (key, item) in dict.iter() {
            let key = key.cast::<PyString>().map_err(|_| {
                error::type_error(format!(
                    "{}: dictionary keys must be strings, got {}",
                    path.render(),
                    type_name(&key)
                ))
            })?;
            let key = key.to_str().map_err(|err| {
                error::object_error(
                    value.py(),
                    err,
                    &format!("{}: undecodable key", path.render()),
                )
            })?;
            let child = Path {
                parent: Some(path),
                segment: Segment::Key(key),
            };
            let encoded = encode_value(&item, &child);
            map.insert(key.to_string(), encoded?);
        }
        return Ok(Json::Object(map));
    }
    if let Ok(list) = value.cast::<PyList>() {
        let mut out = Vec::with_capacity(list.len());
        encode_items(list.iter(), path, &mut out)?;
        return Ok(Json::Array(out));
    }
    if let Ok(tuple) = value.cast::<PyTuple>() {
        let mut out = Vec::with_capacity(tuple.len());
        encode_items(tuple.iter(), path, &mut out)?;
        return Ok(Json::Array(out));
    }
    if let Ok(text) = value.cast::<PyString>() {
        let text = text.to_str().map_err(|err| {
            error::object_error(
                value.py(),
                err,
                &format!("{}: undecodable string", path.render()),
            )
        })?;
        return Ok(Json::String(text.to_string()));
    }
    // `bool` is an `int` subclass in Python: test it first or `True` becomes 1.
    if value.cast::<PyBool>().is_ok() {
        let flag: bool = value.extract()?;
        return Ok(Json::Bool(flag));
    }
    if value.cast::<PyInt>().is_ok() {
        if let Ok(int) = value.extract::<i64>() {
            return Ok(Json::from(int));
        }
        if let Ok(uint) = value.extract::<u64>() {
            return Ok(Json::from(uint));
        }
        return Err(error::type_error(format!(
            "{}: integer {} does not fit the envelope's 64-bit number range",
            path.render(),
            python_text(value)
        )));
    }
    if value.cast::<PyFloat>().is_ok() {
        let float: f64 = value.extract()?;
        return Number::from_f64(float).map(Json::Number).ok_or_else(|| {
            error::type_error(format!(
                "{}: {float} is not a finite JSON number (NaN and infinity have no \
                 envelope form; pass them as an ndarray attribute payload instead)",
                path.render()
            ))
        });
    }
    Err(error::type_error(format!(
        "{}: expected dict, list, tuple, str, bool, int, float or None, got {}",
        path.render(),
        type_name(value)
    )))
}

fn encode_items<'py>(
    items: impl Iterator<Item = Bound<'py, PyAny>>,
    path: &Path<'_>,
    out: &mut Vec<Json>,
) -> PyResult<()> {
    for (index, item) in items.enumerate() {
        let child = Path {
            parent: Some(path),
            segment: Segment::Index(index),
        };
        out.push(encode_value(&item, &child)?);
    }
    Ok(())
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
    fn encode<'py>(py: Python<'py>, build: impl FnOnce(Python<'py>) -> Bound<'py, PyAny>) -> Json {
        let value = build(py);
        json_from_py(&value, "attributes").expect("conversion must succeed")
    }
    #[test]
    fn scalars_keep_their_json_kind() {
        Python::attach(|py| {
            assert_eq!(encode(py, |py| py.None().into_bound(py)), Json::Null);
            assert_eq!(
                encode(py, |py| PyBool::new(py, true).to_owned().into_any()),
                Json::Bool(true)
            );
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
    fn a_negative_int_bypasses_the_shared_int_table() {
        Python::attach(|py| {
            let mut cache = PyCache::new();
            // The boundary values around CPython's small-int singletons and the
            // table's own limit, plus the extremes: the table is indexed by the
            // value, so a negative one must never reach it.
            for value in [-8_i64, -6, -5, -1, 0, 256, 257, 8_191, 8_192, i64::MIN, i64::MAX] {
                let bound = number_to_py(py, &Number::from(value), &mut cache);
                assert_eq!(bound.extract::<i64>().unwrap(), value);
            }
            let first = number_to_py(py, &Number::from(300_i64), &mut cache);
            let second = number_to_py(py, &Number::from(300_i64), &mut cache);
            assert!(first.is(&second), "an in-band value is one shared object");
        });
    }
    #[test]
    fn nested_attributes_dicts_lists_and_tuples_round_trip() {
        Python::attach(|py| {
            let inner = PyDict::new(py);
            inner.set_item("dtype", "float32").unwrap();
            inner
                .set_item("shape", PyTuple::new(py, [2_i64, 3]).unwrap())
                .unwrap();
            let attrs = PyDict::new(py);
            attrs.set_item("value", inner).unwrap();
            attrs
                .set_item("permutation", PyList::new(py, [1_i64, 0]).unwrap())
                .unwrap();
            attrs.set_item("keepdims", false).unwrap();
            let encoded = json_from_py(attrs.as_any(), "attributes").unwrap();
            let object = encoded.as_object().expect("a dict encodes as an object");
            // Tuples decode as arrays: the envelope has one sequence form.
            assert_eq!(
                object["value"]["shape"],
                Json::Array(vec![Json::from(2), Json::from(3)])
            );
            assert_eq!(
                object["permutation"],
                Json::Array(vec![Json::from(1), Json::from(0)])
            );
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
                (
                    "data_b64".to_string(),
                    Json::String("AACAPwAAAEA=".to_string()),
                ),
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
            assert!(
                text.contains("set"),
                "the offending type must be named: {text}"
            );
        });
    }
    #[test]
    fn non_string_dict_keys_are_rejected_with_the_path() {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item(3_i64, "x").unwrap();
            let err = json_from_py(dict.as_any(), "module").unwrap_err();
            assert!(err
                .to_string()
                .contains("module: dictionary keys must be strings"));
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
