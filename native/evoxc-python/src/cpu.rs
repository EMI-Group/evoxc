//! The native CPU execution engine behind `native_compile` / `NativeProgram`.
//!
//! `native_compile(envelope) -> NativeProgram` decodes one `evoxc-native-ir`
//! envelope with [`evoxc_core::envelope::decode`] and hands the program to
//! [`evoxc_cpu::plan::compile`] (detached — the closure captures only `Send`
//! data).  The returned object is immutable: it owns the decoded `Program`,
//! the `Arc<Plan>` and the measured compile time, and `run()` executes it on
//! typed host buffers through [`evoxc_cpu::run_program`].
//!
//! ## numpy interop (copy through bytes, deliberately)
//!
//! The newest `numpy` crate (0.28) requires pyo3 0.28 while this crate pins
//! pyo3 0.30, so there is no `numpy` crate to use here and the binding crosses
//! the boundary as raw little-endian bytes:
//!
//! * **in** — `arr.dtype.name` → [`evoxc_core::DType::parse`], `arr.shape` →
//!   dims, `arr.flags.c_contiguous` decides whether `numpy.ascontiguousarray`
//!   is applied first (a strided array is never silently misread, and the
//!   caller's array is only borrowed).  The bytes come from a `memoryview`
//!   cast to `B` read through [`pyo3::buffer::PyBuffer<u8>`], then
//!   [`evoxc_cpu::Buffer::from_payload`] decodes them into the right
//!   [`evoxc_cpu::Storage`] variant (and validates the byte length).
//! * **out** — the storage is flattened to little-endian bytes, copied into a
//!   fresh `PyByteArray` and turned into an array with `numpy.frombuffer` +
//!   `reshape`.  `frombuffer` on a bytearray owns its bytes, so every result is
//!   a fresh writable C-contiguous array that shares nothing with the engine.
//!
//! ## Host callbacks
//!
//! `block_call` / `external_call` / `runtime_call` are resolved through an
//! optional Python `resolver` whose `call_block(name, args[, static_args])`,
//! `call_external(name, args)` and `call_runtime(name, args)` methods receive a
//! Python list of numpy arrays and return a list/tuple of numpy arrays.  The
//! engine runs detached, so [`PyCallbacks`] only owns a `Py<PyAny>` and
//! re-attaches with [`Python::attach`] per call — the same discipline as
//! `PyExtensionRule`.  A resolver exception, a non-sequence answer or an
//! unconvertible array becomes an explicit `BackendError` preserving the
//! interpreter's message text.
use std::sync::Arc;
use std::time::Instant;

use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyByteArray, PyDict, PyList, PyModule};
use serde_json::{Map, Value as Json};

use evoxc_core::graph::Program;
use evoxc_core::{ArrayPayload, DType};
use evoxc_cpu::plan::Plan;
use evoxc_cpu::{Buffer, Callbacks, RunOptions, Storage, Value};

use crate::convert::{json_from_py, py_from_json};
use crate::error;

/// The `numpy` module, imported once per process (lazily — `import evoxc`
/// never needs it, only a `run`/`native_compile` conversion does).
static NUMPY: PyOnceLock<Py<PyModule>> = PyOnceLock::new();

fn numpy(py: Python<'_>) -> PyResult<&Bound<'_, PyModule>> {
    NUMPY
        .get_or_try_init(py, || -> PyResult<Py<PyModule>> {
            Ok(py.import("numpy")?.unbind())
        })
        .map(|module| module.bind(py))
}

// ---------------------------------------------------------------------------
// `native_compile`
// ---------------------------------------------------------------------------

/// Decode one envelope into a `Program` (interpreter-free, unit-tested).
fn program_from_json(envelope: &Json) -> evoxc_core::Result<Program> {
    evoxc_core::envelope::decode(envelope).map_err(|err| err.context("envelope"))
}

/// Compile an `evoxc-native-ir` envelope for native CPU execution.
///
/// A non-dict argument is a `TypeError`; an envelope violation keeps its
/// native family (`PersistenceError` / `VerificationError` / `DTypeError` /
/// `ShapeError`), and an op without a native kernel raises the engine's
/// explicit `BackendError` from `compile`.
#[pyfunction]
pub fn native_compile(py: Python<'_>, envelope: &Bound<'_, PyAny>) -> PyResult<NativeProgram> {
    let envelope = envelope.cast::<PyDict>().map_err(|_| {
        error::type_error(format!(
            "native_compile: envelope must be a dict, got {}",
            type_name(envelope)
        ))
    })?;
    let json = json_from_py(envelope.as_any(), "envelope")
        .map_err(|err| error::object_error(py, err, "native_compile: envelope"))?;
    let program = program_from_json(&json).map_err(|err| error::from_core(py, err))?;

    let started = Instant::now();
    let plan = py
        .detach(|| evoxc_cpu::plan::compile(&program))
        .map_err(|err| error::from_core(py, err))?;
    let compile_ms = started.elapsed().as_secs_f64() * 1000.0;

    Ok(NativeProgram {
        program,
        plan: Arc::new(plan),
        compile_ms,
    })
}

/// A compiled native CPU program returned by [`native_compile`].
///
/// Immutable: construction is the only mutation, and every method is a read
/// (plus `run`, which executes on a scratch context and keeps no state).
#[pyclass(module = "evoxc._evoxc_native")]
pub struct NativeProgram {
    program: Program,
    plan: Arc<Plan>,
    compile_ms: f64,
}

#[pymethods]
impl NativeProgram {
    /// The entry function's name.
    fn entry(&self) -> String {
        self.plan.entry_function().name.clone()
    }

    /// The human-readable dispatch plan (`evoxc_cpu::Plan::text`).
    fn plan_text(&self) -> String {
        self.plan.text()
    }

    /// Plan statistics: op/block/plan-entry counts, the measured compile time
    /// and the resolver/unsupported-op inventories.
    ///
    /// `dispatched_ops` equals `ops` — this engine dispatches every planned op.
    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(py_from_json(py, &self.stats_json())?.unbind())
    }

    /// The entry function's output types as
    /// `{"dtype": name, "shape": [Dim...]}` dicts.
    fn output_types(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let json = evoxc_core::wire::value_types_to_json(&self.plan.entry_function().output_types);
        Ok(py_from_json(py, &json)?.unbind())
    }

    /// The host callbacks the plan needs, split by kind.
    fn required_ops(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(py_from_json(py, &self.required_json())?.unbind())
    }

    /// Execute one call over flat numpy inputs.
    ///
    /// `flat_inputs` is a sequence of numpy arrays in the entry function's
    /// input order; the result is a Python list of fresh numpy arrays (no
    /// aliasing with the engine).  `resolver` is required exactly when the plan
    /// needs host callbacks and must expose the matching
    /// `call_block` / `call_external` / `call_runtime` methods.
    #[pyo3(signature = (flat_inputs, resolver=None))]
    fn run(
        &self,
        py: Python<'_>,
        flat_inputs: &Bound<'_, PyAny>,
        resolver: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        // ---- GIL held: numpy arrays -> typed buffers, resolver preflight ----
        let inputs = decode_inputs(py, flat_inputs)?;
        let resolver = resolver.filter(|value| !value.is_none());
        preflight(py, &self.plan, resolver)?;
        let callbacks = resolver.map(|resolver| PyCallbacks {
            resolver: resolver.clone().unbind(),
        });

        // ---- GIL released: the native interpreter (the bridge re-attaches
        // for host callbacks only) -------------------------------------------
        let program = &self.program;
        let plan = Arc::clone(&self.plan);
        let outputs = py
            .detach(|| {
                let callbacks = callbacks
                    .as_ref()
                    .map(|callbacks| callbacks as &dyn Callbacks);
                evoxc_cpu::run_program(program, &plan, inputs, RunOptions::default(), callbacks)
            })
            .map_err(|err| error::from_core(py, err))?;

        // ---- GIL held: buffers -> fresh numpy arrays -----------------------
        let list = PyList::empty(py);
        for (index, buffer) in outputs.iter().enumerate() {
            let array = buffer_to_array(py, buffer)
                .map_err(|err| error::object_error(py, err, &format!("run: output {index}")))?;
            list.append(array)?;
        }
        Ok(list.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "<NativeProgram entry={:?} ops={} blocks={}>",
            self.plan.entry_function().name,
            self.plan.op_count,
            self.plan.block_count
        )
    }
}

impl NativeProgram {
    fn stats_json(&self) -> Json {
        Json::Object(Map::from_iter([
            ("ops".to_string(), Json::from(self.plan.op_count)),
            ("dispatched_ops".to_string(), Json::from(self.plan.op_count)),
            ("blocks".to_string(), Json::from(self.plan.block_count)),
            ("plan_entries".to_string(), Json::from(self.plan.op_count)),
            ("compile_ms".to_string(), Json::from(self.compile_ms)),
            (
                "unsupported".to_string(),
                string_array(&self.plan.unsupported),
            ),
            (
                "required_blocks".to_string(),
                string_array(&self.plan.required_blocks),
            ),
            (
                "required_external".to_string(),
                string_array(&self.plan.required_external),
            ),
            (
                "required_runtime".to_string(),
                string_array(&self.plan.required_runtime),
            ),
        ]))
    }

    fn required_json(&self) -> Json {
        Json::Object(Map::from_iter([
            (
                "blocks".to_string(),
                string_array(&self.plan.required_blocks),
            ),
            (
                "external".to_string(),
                string_array(&self.plan.required_external),
            ),
            (
                "runtime_calls".to_string(),
                string_array(&self.plan.required_runtime),
            ),
        ]))
    }
}

fn string_array<S: AsRef<str>>(items: &[S]) -> Json {
    Json::Array(items.iter().map(|item| Json::from(item.as_ref())).collect())
}

// ---------------------------------------------------------------------------
// numpy ⇄ Buffer
// ---------------------------------------------------------------------------

/// Decode one Python sequence of numpy arrays into typed buffers.
fn decode_inputs(py: Python<'_>, flat_inputs: &Bound<'_, PyAny>) -> PyResult<Vec<Buffer>> {
    let items = flat_inputs.try_iter().map_err(|_| {
        error::type_error(format!(
            "run: flat_inputs must be a sequence of numpy arrays, got {}",
            type_name(flat_inputs)
        ))
    })?;
    let mut inputs = Vec::new();
    for item in items {
        let item = item.map_err(|err| error::object_error(py, err, "run: flat_inputs"))?;
        let path = format!("flat_inputs[{}]", inputs.len());
        inputs.push(array_to_buffer(py, &item, &path)?);
    }
    Ok(inputs)
}

/// numpy array → [`Buffer`], naming `path` (`flat_inputs[2]`) in a `TypeError`.
fn array_to_buffer(py: Python<'_>, obj: &Bound<'_, PyAny>, path: &str) -> PyResult<Buffer> {
    let not_array = || {
        error::type_error(format!(
            "{path}: expected a numpy array, got {}",
            type_name(obj)
        ))
    };
    let dtype_obj = obj.getattr("dtype").map_err(|_| not_array())?;
    let dtype_name = dtype_obj
        .getattr("name")
        .and_then(|name| name.extract::<String>())
        .map_err(|_| not_array())?;
    let native_order = dtype_obj
        .getattr("isnative")
        .and_then(|value| value.extract::<bool>())
        .map_err(|_| not_array())?;
    let dtype = DType::parse(&dtype_name).map_err(|err| error::from_core(py, err))?;
    if !native_order {
        // The bytes below are the array's raw bytes, so a swapped byte order
        // would be silently misread; refuse it explicitly.
        return Err(error::raise(
            py,
            "BackendError",
            format!(
                "{path}: non-native byte order {dtype_name} is not supported — convert the array with `.astype(...)` first"
            ),
        ));
    }
    let shape = obj
        .getattr("shape")
        .and_then(|shape| shape.extract::<Vec<i64>>())
        .map_err(|_| not_array())?;
    let contiguous = obj
        .getattr("flags")
        .and_then(|flags| flags.getattr("c_contiguous"))
        .and_then(|value| value.extract::<bool>())
        .map_err(|_| not_array())?;

    // A strided view is copied, never misread; the caller's array stays untouched.
    let copied = if contiguous {
        None
    } else {
        let ascontiguousarray = numpy(py)?.getattr("ascontiguousarray")?;
        Some(ascontiguousarray.call1((obj,)).map_err(|err| {
            error::object_error(py, err, &format!("{path}: numpy.ascontiguousarray"))
        })?)
    };
    let array = copied.as_ref().unwrap_or(obj);
    let data = array_bytes(py, array, path)?;

    Buffer::from_payload(&ArrayPayload { dtype, shape, data })
        .map_err(|err| error::from_core(py, err))
}

/// The C-order little-endian bytes of a C-contiguous numpy array.
fn array_bytes(py: Python<'_>, array: &Bound<'_, PyAny>, path: &str) -> PyResult<Vec<u8>> {
    let view = array.getattr("data").map_err(|err| {
        error::type_error(format!(
            "{path}: expected a numpy array ({})",
            message(err.value(py))
        ))
    })?;
    let cast = view.call_method1("cast", ("B",)).map_err(|err| {
        error::type_error(format!(
            "{path}: cannot take a byte view ({})",
            message(err.value(py))
        ))
    })?;
    let buffer = PyBuffer::<u8>::get(&cast).map_err(|err| {
        error::raise(
            py,
            "BackendError",
            format!(
                "{path}: numpy refused a contiguous byte buffer ({})",
                message(err.value(py))
            ),
        )
    })?;
    buffer.to_vec(py).map_err(|err| {
        error::raise(
            py,
            "BackendError",
            format!(
                "{path}: reading the numpy bytes failed ({})",
                message(err.value(py))
            ),
        )
    })
}

/// [`Buffer`] → a fresh, writable, C-contiguous numpy array (owns its bytes).
fn buffer_to_array<'py>(py: Python<'py>, buffer: &Buffer) -> PyResult<Bound<'py, PyAny>> {
    let bytes = storage_to_bytes(&buffer.storage);
    let owner = PyByteArray::new(py, &bytes);
    let frombuffer = numpy(py)?.getattr("frombuffer")?;
    let flat = frombuffer.call1((owner, buffer.dtype.name()))?;
    if buffer.shape.is_empty() {
        // A 0-d buffer is one element; `frombuffer` yields shape (1,).
        flat.call_method1("reshape", ((),))
    } else {
        flat.call_method1("reshape", (buffer.shape.clone(),))
    }
}

/// Flatten storage to little-endian bytes (bool as `0x00`/`0x01`, f16 via its
/// bit pattern, complex as interleaved real/imaginary).
fn storage_to_bytes(storage: &Storage) -> Vec<u8> {
    let itemsize = match storage {
        Storage::Bool(_) | Storage::I8(_) | Storage::U8(_) => 1,
        Storage::I16(_) | Storage::U16(_) | Storage::F16(_) => 2,
        Storage::I32(_) | Storage::U32(_) | Storage::F32(_) | Storage::C64(_) => 4,
        Storage::I64(_) | Storage::U64(_) | Storage::F64(_) | Storage::C128(_) => 8,
    };
    let mut bytes = Vec::with_capacity(storage.len() * itemsize);
    match storage {
        Storage::Bool(values) => bytes.extend(values.iter().map(|value| u8::from(*value))),
        Storage::I8(values) => bytes.extend(values.iter().flat_map(|value| value.to_le_bytes())),
        Storage::I16(values) => bytes.extend(values.iter().flat_map(|value| value.to_le_bytes())),
        Storage::I32(values) => bytes.extend(values.iter().flat_map(|value| value.to_le_bytes())),
        Storage::I64(values) => bytes.extend(values.iter().flat_map(|value| value.to_le_bytes())),
        Storage::U8(values) => bytes.extend_from_slice(values),
        Storage::U16(values) => bytes.extend(values.iter().flat_map(|value| value.to_le_bytes())),
        Storage::U32(values) => bytes.extend(values.iter().flat_map(|value| value.to_le_bytes())),
        Storage::U64(values) => bytes.extend(values.iter().flat_map(|value| value.to_le_bytes())),
        Storage::F16(values) => bytes.extend(
            values
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes()),
        ),
        Storage::F32(values) => bytes.extend(values.iter().flat_map(|value| value.to_le_bytes())),
        Storage::F64(values) => bytes.extend(values.iter().flat_map(|value| value.to_le_bytes())),
        Storage::C64(values) => {
            for value in values {
                bytes.extend_from_slice(&value.re.to_le_bytes());
                bytes.extend_from_slice(&value.im.to_le_bytes());
            }
        }
        Storage::C128(values) => {
            for value in values {
                bytes.extend_from_slice(&value.re.to_le_bytes());
                bytes.extend_from_slice(&value.im.to_le_bytes());
            }
        }
    }
    bytes
}

// ---------------------------------------------------------------------------
// Resolver preflight and the host-callback bridge
// ---------------------------------------------------------------------------

/// Reject a run that needs callbacks the resolver cannot serve, before any op
/// executes (`BackendError` naming the missing methods and what the plan needs).
fn preflight(py: Python<'_>, plan: &Plan, resolver: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    let mut required: Vec<(&str, &[String])> = Vec::new();
    if !plan.required_blocks.is_empty() {
        required.push(("call_block", plan.required_blocks.as_slice()));
    }
    if !plan.required_external.is_empty() {
        required.push(("call_external", plan.required_external.as_slice()));
    }
    if !plan.required_runtime.is_empty() {
        required.push(("call_runtime", plan.required_runtime.as_slice()));
    }
    if required.is_empty() {
        return Ok(());
    }

    let mut missing: Vec<&str> = Vec::new();
    match resolver {
        None => missing.extend(required.iter().map(|(method, _)| *method)),
        Some(resolver) => {
            for (method, _) in &required {
                let callable = resolver
                    .getattr(*method)
                    .map(|value| value.is_callable())
                    .unwrap_or(false);
                if !callable {
                    missing.push(method);
                }
            }
        }
    }
    if missing.is_empty() {
        return Ok(());
    }

    let needs = required
        .iter()
        .map(|(method, names)| format!("{method} for {names:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    let problem = match resolver {
        None => "no resolver was supplied".to_string(),
        Some(_) => format!("the resolver is missing method(s) {}", missing.join(", ")),
    };
    Err(error::raise(
        py,
        "BackendError",
        format!("run: the graph requires host callbacks ({needs}) but {problem}"),
    ))
}

/// The engine's [`Callbacks`] over a Python resolver object.
///
/// Holds only the `Py<PyAny>` (never a borrowed handle) so it can cross the
/// detach boundary; each call re-attaches for its own duration.
struct PyCallbacks {
    resolver: Py<PyAny>,
}

impl Callbacks for PyCallbacks {
    fn call_block(
        &self,
        name: &str,
        args: &[Value],
        static_args: &Json,
    ) -> evoxc_core::Result<Vec<Buffer>> {
        Python::attach(|py| {
            invoke(
                py,
                &self.resolver,
                "call_block",
                name,
                args,
                Some(static_args),
            )
        })
    }

    fn call_external(&self, name: &str, args: &[Value]) -> evoxc_core::Result<Vec<Buffer>> {
        Python::attach(|py| invoke(py, &self.resolver, "call_external", name, args, None))
    }

    fn call_runtime(&self, callback: &str, args: &[Value]) -> evoxc_core::Result<Vec<Buffer>> {
        Python::attach(|py| invoke(py, &self.resolver, "call_runtime", callback, args, None))
    }
}

/// One resolver call: `method(name, args)` — plus `static_args` as a third
/// positional argument for `call_block` when the engine's is non-empty.
fn invoke(
    py: Python<'_>,
    resolver: &Py<PyAny>,
    method: &str,
    name: &str,
    args: &[Value],
    static_args: Option<&Json>,
) -> evoxc_core::Result<Vec<Buffer>> {
    let function = resolver
        .bind(py)
        .getattr(method)
        .map_err(|err| callback_error(py, method, name, &err))?;

    let list = PyList::empty(py);
    for value in args {
        let array =
            buffer_to_array(py, value).map_err(|err| callback_error(py, method, name, &err))?;
        list.append(array)
            .map_err(|err| callback_error(py, method, name, &err))?;
    }

    let result = match static_args.filter(|value| !is_absent(value)) {
        Some(static_args) => {
            let static_py = py_from_json(py, static_args)
                .map_err(|err| callback_error(py, method, name, &err))?;
            function.call1((name, list.clone(), static_py))
        }
        None => function.call1((name, list)),
    }
    .map_err(|err| callback_error(py, method, name, &err))?;

    let items = result
        .try_iter()
        .map_err(|err| callback_error(py, method, name, &err))?;
    let mut buffers = Vec::new();
    for item in items {
        let item = item.map_err(|err| callback_error(py, method, name, &err))?;
        let path = format!("resolver.{method}({name}) result[{}]", buffers.len());
        let buffer = array_to_buffer(py, &item, &path)
            .map_err(|err| callback_error(py, method, name, &err))?;
        buffers.push(buffer);
    }
    Ok(buffers)
}

/// A Python failure inside the bridge is a `BackendError` naming the method and
/// the op, preserving the interpreter's message text.
fn callback_error(py: Python<'_>, method: &str, name: &str, err: &PyErr) -> evoxc_core::Error {
    evoxc_core::Error::backend(format!(
        "resolver.{method}({name}) raised: {}",
        message(err.value(py))
    ))
}

/// The `null` / empty-object / empty-array "no static args" spellings.
fn is_absent(value: &Json) -> bool {
    match value {
        Json::Null => true,
        Json::Object(map) => map.is_empty(),
        Json::Array(items) => items.is_empty(),
        _ => false,
    }
}

/// A Python error's text, never panicking on an unprintable exception.
fn message(value: &Bound<'_, PyAny>) -> String {
    match value.str() {
        Ok(text) => text.to_string_lossy().into_owned(),
        Err(_) => String::from("<unprintable Python error>"),
    }
}

/// A Python object's type name for argument-shape messages.
fn type_name(value: &Bound<'_, PyAny>) -> String {
    value
        .get_type()
        .name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|_| String::from("<unknown>"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use evoxc_core::ErrorKind;
    use half::f16;
    use num_complex::Complex;

    #[test]
    fn dtype_names_round_trip() {
        for dtype in evoxc_core::dtype::ALL {
            assert_eq!(DType::parse(dtype.name()).unwrap(), dtype);
        }
        assert_eq!(
            DType::parse("not_a_dtype").unwrap_err().kind,
            ErrorKind::DType
        );
    }

    #[test]
    fn integers_and_floats_flatten_little_endian() {
        let storage = Storage::I32(vec![1, -2]);
        assert_eq!(
            storage_to_bytes(&storage),
            vec![1, 0, 0, 0, 0xfe, 0xff, 0xff, 0xff]
        );
        let storage = Storage::F32(vec![1.0, -1.0]);
        assert_eq!(
            storage_to_bytes(&storage),
            1.0f32
                .to_le_bytes()
                .iter()
                .copied()
                .chain((-1.0f32).to_le_bytes())
                .collect::<Vec<u8>>()
        );
    }

    #[test]
    fn bool_f16_and_complex_flatten_to_their_numpy_layouts() {
        let storage = Storage::Bool(vec![true, false]);
        assert_eq!(storage_to_bytes(&storage), vec![1, 0]);

        let value = f16::from_f32(0.5);
        let storage = Storage::F16(vec![value]);
        assert_eq!(
            storage_to_bytes(&storage),
            value.to_bits().to_le_bytes().to_vec()
        );

        let storage = Storage::C64(vec![Complex::new(1.5f32, -2.5f32)]);
        let mut expected = 1.5f32.to_le_bytes().to_vec();
        expected.extend_from_slice(&(-2.5f32).to_le_bytes());
        assert_eq!(storage_to_bytes(&storage), expected);

        let storage = Storage::C128(vec![Complex::new(1.0f64, 2.0f64)]);
        let mut expected = 1.0f64.to_le_bytes().to_vec();
        expected.extend_from_slice(&2.0f64.to_le_bytes());
        assert_eq!(storage_to_bytes(&storage), expected);
    }

    #[test]
    fn a_bad_discriminator_is_a_persistence_error() {
        let envelope = Json::Object(Map::from_iter([
            ("format".to_string(), Json::from("something-else")),
            ("version".to_string(), Json::from(1)),
        ]));
        let err = program_from_json(&envelope).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Persistence);
        assert!(err.message.contains("format"), "{}", err.message);
    }

    #[test]
    fn a_valid_empty_envelope_decodes() {
        let envelope = Json::Object(Map::from_iter([
            ("format".to_string(), Json::from("evoxc-native-ir")),
            ("version".to_string(), Json::from(1)),
            (
                "module".to_string(),
                Json::Object(Map::from_iter([("name".to_string(), Json::from("main"))])),
            ),
            ("functions".to_string(), Json::Array(Vec::new())),
            ("ops".to_string(), Json::Array(Vec::new())),
        ]));
        let program = program_from_json(&envelope).expect("an empty module decodes");
        assert_eq!(program.name, "main");
    }
}
