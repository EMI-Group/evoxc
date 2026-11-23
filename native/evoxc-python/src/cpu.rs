//! The native CPU execution engine behind `native_compile` / `NativeProgram`.
//!
//! `native_compile(envelope) -> NativeProgram` decodes one `evoxc-native-ir`
//! envelope with [`evoxc_core::envelope::decode`] and hands the program to
//! [`evoxc_cpu::plan::compile`] (detached — the closure captures only `Send`
//! data).  The returned object is immutable: it owns the decoded `Program`,
//! the `Arc<Plan>` and the measured compile time, and `run()` executes it on
//! typed host buffers through [`evoxc_cpu::run_program`].
//!
//! ## numpy interop (typed views in, typed ownership out)
//!
//! Payloads cross through the `numpy` 0.30 crate as TYPED arrays — never as
//! hand-rolled little-endian byte slices:
//!
//! * **in** — [`array_to_buffer`] downcasts to a `numpy::PyUntypedArray`, takes
//!   the dtype name from `dtype.name` ([`evoxc_core::DType::parse`]), refuses a
//!   swapped byte order (`dtype.is_native_byteorder() == Some(false)`) BEFORE
//!   any typed view could misread it, copies `shape()`, and then reads the
//!   payload through `PyReadonlyArrayDyn<T>`: a C-contiguous array is one
//!   `memcpy` (`ArrayView::as_slice().to_vec()`), while a Fortran-ordered,
//!   transposed or negative-stride view is walked in logical order by
//!   `ndarray`'s iterator.  A strided array is therefore never misread, no
//!   `numpy.ascontiguousarray` call happens anywhere, and the caller's array is
//!   only borrowed (never mutated).  `bool` is the one special case: a numpy
//!   bool array may hold any byte (only `!= 0` is meaningful) and reading byte
//!   `2` as a Rust `bool` would be undefined behaviour, so it is read through a
//!   zero-copy `uint8` view and normalized to `byte != 0`.
//! * **out** — [`buffer_to_array`] CONSUMES the engine's `Buffer`: each
//!   `Storage` variant hands its `Vec<T>` to numpy as the array's buffer
//!   (`ndarray::ArrayD::from_shape_vec` + `IntoPyArray`), i.e. zero copy, and
//!   the result is fresh, writable and C-contiguous.  `bool` cannot move (a
//!   `Vec<bool>` is bit-packed, one BIT per element) and gets a zero-filled
//!   numpy allocation filled elementwise.  Resolver arguments are borrowed
//!   (`Rc<Buffer>`), so [`buffer_to_array_copy`] clones the typed payload once
//!   and reuses the owned path.
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

use half::f16;
use ndarray::{ArrayD, IxDyn};
use num_complex::Complex;
use numpy::{
    IntoPyArray, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn,
    PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::{Map, Value as Json};

use evoxc_core::graph::Program;
use evoxc_core::DType;
use evoxc_cpu::plan::Plan;
use evoxc_cpu::{Buffer, Callbacks, RunOptions, Storage, Value};

use crate::convert::{json_from_py, py_from_json};
use crate::error;

/// The `numpy` module, imported once per process.  Only the tests need it (the
/// conversion paths below use the typed `numpy` crate), and `import evoxc` never
/// runs any of this.
#[cfg(all(test, feature = "auto-init"))]
static NUMPY: pyo3::sync::PyOnceLock<Py<pyo3::types::PyModule>> = pyo3::sync::PyOnceLock::new();

#[cfg(all(test, feature = "auto-init"))]
fn numpy(py: Python<'_>) -> PyResult<&Bound<'_, pyo3::types::PyModule>> {
    NUMPY
        .get_or_try_init(py, || -> PyResult<Py<pyo3::types::PyModule>> {
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

        // ---- GIL held: buffers -> fresh numpy arrays (ownership moves) ------
        let list = PyList::empty(py);
        for (index, buffer) in outputs.into_iter().enumerate() {
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
///
/// The array is only BORROWED: values are copied out of its view (one memcpy for
/// a C-contiguous array, a logical-order walk for any other layout) into the
/// engine's own typed storage, so the caller's array is never mutated.
fn array_to_buffer(py: Python<'_>, obj: &Bound<'_, PyAny>, path: &str) -> PyResult<Buffer> {
    let not_array = || {
        error::type_error(format!(
            "{path}: expected a numpy array, got {}",
            type_name(obj)
        ))
    };
    // `PyUntypedArray` accepts ndarray subclasses (`numpy.matrix`, masked-free
    // subclasses) exactly like `PyArray_Check` does.
    let array = obj.cast::<PyUntypedArray>().map_err(|_| not_array())?;
    let descr = array.dtype();
    let dtype_name = descr
        .getattr("name")
        .and_then(|name| name.extract::<String>())
        .map_err(|_| not_array())?;
    let dtype = DType::parse(&dtype_name).map_err(|err| error::from_core(py, err))?;
    // The typed views below read native-endian values, so a swapped byte order
    // would be silently misread; refuse it before any conversion.  A `|`/1-byte
    // dtype reports `None`, which IS native.
    if descr.is_native_byteorder() == Some(false) {
        return Err(error::raise(
            py,
            "BackendError",
            format!(
                "{path}: non-native byte order {dtype_name} is not supported — convert the array with `.astype(...)` first"
            ),
        ));
    }
    let dims: Vec<usize> = array.shape().to_vec();
    let storage = match dtype {
        DType::Bool => Storage::Bool(bool_values(py, obj, path)?),
        DType::Int8 => Storage::I8(collect_values::<i8>(py, obj, path)?),
        DType::Int16 => Storage::I16(collect_values::<i16>(py, obj, path)?),
        DType::Int32 => Storage::I32(collect_values::<i32>(py, obj, path)?),
        DType::Int64 => Storage::I64(collect_values::<i64>(py, obj, path)?),
        DType::UInt8 => Storage::U8(collect_values::<u8>(py, obj, path)?),
        DType::UInt16 => Storage::U16(collect_values::<u16>(py, obj, path)?),
        DType::UInt32 => Storage::U32(collect_values::<u32>(py, obj, path)?),
        DType::UInt64 => Storage::U64(collect_values::<u64>(py, obj, path)?),
        DType::Float16 => Storage::F16(collect_values::<f16>(py, obj, path)?),
        DType::Float32 => Storage::F32(collect_values::<f32>(py, obj, path)?),
        DType::Float64 => Storage::F64(collect_values::<f64>(py, obj, path)?),
        DType::Complex64 => Storage::C64(collect_values::<Complex<f32>>(py, obj, path)?),
        DType::Complex128 => Storage::C128(collect_values::<Complex<f64>>(py, obj, path)?),
    };
    Buffer::new(dtype, dims, storage).map_err(|err| error::from_core(py, err))
}

/// Copy a typed numpy view into an owned `Vec<T>` in logical (C) order.
///
/// `ArrayView::as_slice` succeeds only for a standard-layout array — that arm is
/// one `memcpy`; every other layout (Fortran order, a transpose, a
/// negative-stride slice) falls back to `ndarray`'s logical-order iterator, which
/// visits the same elements numpy's own `ascontiguousarray` would produce.
fn collect_values<T: numpy::Element + Copy>(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    path: &str,
) -> PyResult<Vec<T>> {
    let view = obj
        .extract::<PyReadonlyArrayDyn<'_, T>>()
        .map_err(|err| borrow_error(py, path, &err))?;
    let array = view.as_array();
    // `ArrayView::as_slice` is `Some` only for a standard (C) layout; any
    // other layout is walked in logical index order by `iter`.
    Ok(match array.as_slice() {
        Some(values) => values.to_vec(),
        None => array.iter().copied().collect(),
    })
}

/// A typed `bool` view would be unsound (numpy bool pixels may hold any byte),
/// so read the array through a zero-copy `uint8` view and normalize.
///
/// `view('uint8')` keeps the itemsize at 1 byte, which numpy allows for ANY
/// strides, and `byte != 0` is the same normalization the engine applies.
fn bool_values(py: Python<'_>, obj: &Bound<'_, PyAny>, path: &str) -> PyResult<Vec<bool>> {
    let bytes = obj.call_method1("view", ("uint8",)).map_err(|err| {
        error::raise(
            py,
            "BackendError",
            format!(
                "{path}: cannot view the bool array as uint8 ({})",
                message(err.value(py))
            ),
        )
    })?;
    Ok(collect_values::<u8>(py, &bytes, path)?
        .into_iter()
        .map(|byte| byte != 0)
        .collect())
}

/// Borrow a typed numpy view and turn a refusal into a `BackendError` naming
/// `path` (the preflight downcast already proved the object is an ndarray, so
/// this only reports a borrow-flag conflict or an equivalent FFI failure).
fn borrow_error(py: Python<'_>, path: &str, err: &PyErr) -> PyErr {
    error::raise(
        py,
        "BackendError",
        format!("{path}: cannot read the array ({})", message(err.value(py))),
    )
}

/// [`Buffer`] → a fresh numpy array, ZERO copy for every dtype but `bool`.
///
/// The `Buffer` is CONSUMED: its `Vec<T>` becomes numpy's own buffer
/// (`ndarray::ArrayD::from_shape_vec` + `IntoPyArray`, which hands the
/// allocation to a `PySliceContainer` base object), so no element is copied and
/// the result is writable and C-contiguous.  Nothing is shared with the engine,
/// so writing into it never reaches a caller's input array.
fn buffer_to_array<'py>(py: Python<'py>, buffer: Buffer) -> PyResult<Bound<'py, PyAny>> {
    let Buffer {
        dtype,
        shape,
        mut storage,
    } = buffer;
    // `ArrayD::from_shape_vec` is the only fallible step (it checks that the
    // element count matches the shape, which `Buffer::new` has already done) and
    // it never reallocates, so the transfer stays zero copy.  The payload is
    // TAKEN out of its `Storage` variant (`Storage` has a `Drop` impl for the
    // engine's byte pool, so the variant itself cannot be moved out); the
    // emptied storage then drops without touching the pool.
    macro_rules! taken {
        ($values:expr) => {{
            let values = std::mem::take($values);
            let array = ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(|err| {
                error::raise(
                    py,
                    "ShapeError",
                    format!(
                        "run: cannot build a numpy {} array of shape {shape:?}: {err}",
                        dtype.name()
                    ),
                )
            })?;
            array.into_pyarray(py).into_any()
        }};
    }
    Ok(match &mut storage {
        Storage::Bool(values) => bool_array(py, &shape, values)?,
        Storage::I8(values) => taken!(values),
        Storage::I16(values) => taken!(values),
        Storage::I32(values) => taken!(values),
        Storage::I64(values) => taken!(values),
        Storage::U8(values) => taken!(values),
        Storage::U16(values) => taken!(values),
        Storage::U32(values) => taken!(values),
        Storage::U64(values) => taken!(values),
        Storage::F16(values) => taken!(values),
        Storage::F32(values) => taken!(values),
        Storage::F64(values) => taken!(values),
        Storage::C64(values) => taken!(values),
        Storage::C128(values) => taken!(values),
    })
}

/// [`Buffer`] → a numpy array for a BORROWED buffer.
///
/// Resolver arguments arrive as `Rc<Buffer>` and cannot be moved out of the
/// engine, so the typed payload is cloned once (a single typed memcpy) and the
/// owned path does the rest.
fn buffer_to_array_copy<'py>(py: Python<'py>, buffer: &Buffer) -> PyResult<Bound<'py, PyAny>> {
    buffer_to_array(py, buffer.clone())
}

/// The `bool` output path: a `Vec<bool>` is bit-packed (one BIT per element), so
/// its allocation can never become numpy's byte-per-element buffer — allocate the
/// numpy array and write the values into it instead.
fn bool_array<'py>(
    py: Python<'py>,
    shape: &[usize],
    values: &[bool],
) -> PyResult<Bound<'py, PyAny>> {
    let array = PyArrayDyn::<bool>::zeros(py, IxDyn(shape), false);
    {
        let mut writable = array.try_readwrite().map_err(|err| {
            error::raise(
                py,
                "BackendError",
                format!("run: cannot write the bool output ({err})"),
            )
        })?;
        let slots = writable.as_slice_mut().map_err(|err| {
            error::raise(
                py,
                "BackendError",
                format!("run: the bool output is not contiguous ({err})"),
            )
        })?;
        if slots.len() != values.len() {
            return Err(error::raise(
                py,
                "ShapeError",
                format!(
                    "run: a bool buffer of shape {shape:?} needs {} element(s), got {}",
                    slots.len(),
                    values.len()
                ),
            ));
        }
        for (slot, value) in slots.iter_mut().zip(values.iter()) {
            *slot = *value;
        }
    }
    Ok(array.into_any())
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
        let array = buffer_to_array_copy(py, value)
            .map_err(|err| callback_error(py, method, name, &err))?;
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

/// The numpy bridging surface, exercised through a real interpreter (run with
/// `cargo test -p evoxc-python --no-default-features --features auto-init`).
///
/// The linked interpreter must be able to `import numpy` — the root project's
/// `.venv` provides it — because every case here builds real numpy arrays.
#[cfg(all(test, feature = "auto-init"))]
mod marshalling {
    use super::*;
    use pyo3::types::{PyByteArray, PyTuple};

    /// A deterministic byte pattern: every bit pattern is a valid value of
    /// every evoxc dtype (except `bool`, which is covered separately).
    fn payload(len: usize) -> Vec<u8> {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 33) as u8
            })
            .collect()
    }

    /// A writable, C-contiguous 1-D numpy array of `count` elements of `dtype`
    /// over a reproducible byte pattern (plus the bytes it was built from).
    fn make_array<'py>(
        py: Python<'py>,
        dtype: DType,
        count: usize,
    ) -> (Bound<'py, PyAny>, Vec<u8>) {
        let bytes = payload(count * dtype.itemsize());
        let owner = PyByteArray::new(py, &bytes);
        let array = numpy(py)
            .unwrap()
            .getattr("frombuffer")
            .unwrap()
            .call1((owner, dtype.name()))
            .unwrap();
        (array, bytes)
    }

    fn bytes_of(array: &Bound<'_, PyAny>) -> Vec<u8> {
        array
            .call_method0("tobytes")
            .unwrap()
            .extract::<Vec<u8>>()
            .unwrap()
    }

    fn dtype_name(array: &Bound<'_, PyAny>) -> String {
        array
            .getattr("dtype")
            .unwrap()
            .getattr("name")
            .unwrap()
            .extract::<String>()
            .unwrap()
    }

    fn shape_of(array: &Bound<'_, PyAny>) -> Vec<usize> {
        array
            .getattr("shape")
            .unwrap()
            .extract::<Vec<usize>>()
            .unwrap()
    }

    fn flag(array: &Bound<'_, PyAny>, name: &str) -> bool {
        array
            .getattr("flags")
            .unwrap()
            .getattr(name)
            .unwrap()
            .extract::<bool>()
            .unwrap()
    }

    /// numpy's own answer to "what does this view mean" — the C-order values of
    /// an arbitrarily strided array — so a strided input can be compared exactly
    /// without the binding's help.
    fn contiguous_bytes(py: Python<'_>, array: &Bound<'_, PyAny>) -> Vec<u8> {
        let copy = numpy(py)
            .unwrap()
            .getattr("ascontiguousarray")
            .unwrap()
            .call1((array,))
            .unwrap();
        bytes_of(&copy)
    }

    /// `Result::expect` with the dtype in the panic message (clippy's
    /// `expect_fun_call` rejects `expect(dtype.name())` outright).
    fn ok<T, E: std::fmt::Display>(result: Result<T, E>, dtype: DType) -> T {
        result.unwrap_or_else(|err| panic!("{}: {err}", dtype.name()))
    }

    #[test]
    fn every_dtype_round_trips_bit_identically() {
        Python::attach(|py| {
            for dtype in evoxc_core::dtype::ALL {
                if dtype == DType::Bool {
                    continue; // non-canonical byte values normalize (see below)
                }
                let (array, bytes) = make_array(py, dtype, 1024);
                let buffer = ok(array_to_buffer(py, &array, "test"), dtype);
                assert_eq!(buffer.dtype, dtype);
                assert_eq!(buffer.shape, vec![1024]);
                let out = ok(buffer_to_array(py, buffer), dtype);
                assert_eq!(dtype_name(&out), dtype.name());
                assert_eq!(shape_of(&out), vec![1024]);
                assert_eq!(bytes_of(&out), bytes, "{} round trip", dtype.name());
            }
        });
    }

    /// The typed outbound path per dtype: a `Buffer`'s payload becomes a numpy
    /// array of the matching dtype and shape, and survives the trip back.
    #[test]
    fn typed_buffers_become_numpy_arrays_of_their_dtype() {
        Python::attach(|py| {
            let cases: Vec<(DType, Vec<usize>, Storage)> = vec![
                (DType::Int32, vec![2], Storage::I32(vec![1, -2])),
                (DType::Float32, vec![2], Storage::F32(vec![1.0, -1.0])),
                (DType::Float64, vec![2, 2], Storage::F64(vec![0.5, 1.5, -2.5, 3.0])),
                (DType::Int8, vec![3], Storage::I8(vec![-1, 0, 127])),
                (DType::Bool, vec![2], Storage::Bool(vec![true, false])),
                (
                    DType::Float16,
                    vec![1],
                    Storage::F16(vec![f16::from_f32(0.5)]),
                ),
                (
                    DType::Complex64,
                    vec![1],
                    Storage::C64(vec![Complex::new(1.5f32, -2.5f32)]),
                ),
                (
                    DType::Complex128,
                    vec![1],
                    Storage::C128(vec![Complex::new(1.0f64, 2.0f64)]),
                ),
            ];
            for (dtype, shape, storage) in cases {
                let expected = storage.clone();
                let buffer = ok(Buffer::new(dtype, shape.clone(), storage), dtype);
                let array = ok(buffer_to_array(py, buffer), dtype);
                assert_eq!(dtype_name(&array), dtype.name());
                assert_eq!(shape_of(&array), shape);
                assert!(flag(&array, "c_contiguous"));

                // ...and back into the engine: identical dtype, shape and values.
                let back = ok(array_to_buffer(py, &array, "test"), dtype);
                assert_eq!(back.dtype, dtype);
                assert_eq!(back.shape, shape);
                assert_eq!(back.storage, expected, "{} values", dtype.name());
            }
        });
    }

    #[test]
    fn bool_bytes_normalize_to_zero_or_one() {
        Python::attach(|py| {
            let owner = PyByteArray::new(py, &[0u8, 1, 2, 255]);
            let array = numpy(py)
                .unwrap()
                .getattr("frombuffer")
                .unwrap()
                .call1((owner, "bool"))
                .unwrap();
            // A numpy bool array may hold any byte (`only != 0` is meaningful),
            // so the uint8 view normalizes exactly like `Buffer::from_payload`.
            let buffer = array_to_buffer(py, &array, "test").unwrap();
            assert_eq!(buffer.storage, Storage::Bool(vec![false, true, true, true]));
            let out = buffer_to_array(py, buffer).unwrap();
            assert_eq!(bytes_of(&out), vec![0u8, 1, 1, 1]);
        });
    }

    #[test]
    fn a_fortran_ordered_input_reads_in_logical_order() {
        Python::attach(|py| {
            let (source, _) = make_array(py, DType::Float32, 24);
            let c_order = source.call_method1("reshape", ((4, 6),)).unwrap();
            let fortran = numpy(py)
                .unwrap()
                .getattr("asfortranarray")
                .unwrap()
                .call1((&c_order,))
                .unwrap();
            assert!(flag(&fortran, "f_contiguous"));
            assert!(!flag(&fortran, "c_contiguous"));

            let buffer = array_to_buffer(py, &fortran, "test").unwrap();
            assert_eq!(buffer.shape, vec![4, 6]);
            let reference = array_to_buffer(py, &c_order, "test").unwrap();
            assert_eq!(buffer.storage, reference.storage);
            // ...and the same values numpy itself reads out of the view.
            let out = buffer_to_array(py, buffer).unwrap();
            assert!(flag(&out, "c_contiguous"));
            assert_eq!(bytes_of(&out), contiguous_bytes(py, &fortran));
        });
    }

    #[test]
    fn negative_stride_views_read_their_logical_values() {
        Python::attach(|py| {
            let (source, _) = make_array(py, DType::Int32, 12);
            let matrix = source.call_method1("reshape", ((3, 4),)).unwrap();

            // A reversed vector (`arr[::-1]`) and a flipped transpose: both are
            // negative-stride views, which `ndarray` offsets and inverts.
            for view in [
                numpy(py)
                    .unwrap()
                    .getattr("flip")
                    .unwrap()
                    .call1((&source,))
                    .unwrap(),
                numpy(py)
                    .unwrap()
                    .getattr("flip")
                    .unwrap()
                    .call1((matrix.getattr("T").unwrap(),))
                    .unwrap(),
            ] {
                let strides: Vec<isize> = view.getattr("strides").unwrap().extract().unwrap();
                assert!(strides.iter().any(|stride| *stride < 0), "{strides:?}");
                assert!(!flag(&view, "c_contiguous"));

                let buffer = array_to_buffer(py, &view, "test").unwrap();
                assert_eq!(buffer.shape, shape_of(&view));
                let out = buffer_to_array(py, buffer).unwrap();
                assert_eq!(bytes_of(&out), contiguous_bytes(py, &view));
            }
        });
    }

    #[test]
    fn a_read_only_input_is_accepted() {
        Python::attach(|py| {
            let (array, bytes) = make_array(py, DType::Float64, 8);
            array.call_method1("setflags", (false,)).unwrap();
            assert!(!flag(&array, "writeable"));
            let buffer = array_to_buffer(py, &array, "test").unwrap();
            let out = buffer_to_array(py, buffer).unwrap();
            assert_eq!(bytes_of(&out), bytes);
            assert!(flag(&out, "writeable"));
        });
    }

    #[test]
    fn zero_dim_and_empty_arrays_round_trip() {
        Python::attach(|py| {
            // A 0-d array is one element with an empty shape.
            let (flat, bytes) = make_array(py, DType::Float32, 1);
            let scalar = flat.call_method1("reshape", (PyTuple::empty(py),)).unwrap();
            let buffer = array_to_buffer(py, &scalar, "test").unwrap();
            assert!(buffer.shape.is_empty());
            assert!(buffer.storage.len() == 1);
            let out = buffer_to_array(py, buffer).unwrap();
            assert_eq!(out.getattr("shape").unwrap().len().unwrap(), 0);
            assert_eq!(bytes_of(&out), bytes);

            // A zero-length array round trips with no bytes at all.
            let empty = numpy(py)
                .unwrap()
                .getattr("empty")
                .unwrap()
                .call1(((0,), "float64"))
                .unwrap();
            let buffer = array_to_buffer(py, &empty, "test").unwrap();
            assert_eq!(buffer.shape, vec![0]);
            assert!(buffer.storage.is_empty());
            let out = buffer_to_array(py, buffer).unwrap();
            assert_eq!(shape_of(&out), vec![0]);
            assert_eq!(dtype_name(&out), "float64");
            assert_eq!(bytes_of(&out), Vec::<u8>::new());
        });
    }

    #[test]
    fn a_non_native_byte_order_is_refused_with_the_pinned_message() {
        Python::attach(|py| {
            for (dtype, swapped) in [(DType::Float32, ">f4"), (DType::Complex128, ">c16")] {
                let (array, _) = make_array(py, dtype, 4);
                let big_endian = array.call_method1("astype", (swapped,)).unwrap();
                assert!(!big_endian
                    .getattr("dtype")
                    .unwrap()
                    .getattr("isnative")
                    .unwrap()
                    .extract::<bool>()
                    .unwrap());
                let err = array_to_buffer(py, &big_endian, "flat_inputs[0]").unwrap_err();
                let value = err.value(py);
                assert_eq!(
                    value.getattr("kind").unwrap().extract::<String>().unwrap(),
                    "BackendError"
                );
                // numpy reports the canonical name, so the pinned text names the
                // dtype rather than the `>f4` spelling.
                assert_eq!(
                    value.str().unwrap().to_string_lossy(),
                    format!(
                        "flat_inputs[0]: non-native byte order {} is not supported — convert the array with `.astype(...)` first",
                        dtype.name()
                    )
                );
            }
        });
    }

    #[test]
    fn the_returned_array_is_fresh_writable_and_does_not_alias_the_input() {
        Python::attach(|py| {
            let (array, bytes) = make_array(py, DType::Float32, 4096);
            let buffer = array_to_buffer(py, &array, "test").unwrap();
            let out = buffer_to_array(py, buffer).unwrap();
            for name in ["c_contiguous", "writeable"] {
                assert!(flag(&out, name), "the result must be {name}");
            }
            // `IntoPyArray` moves the Rust allocation into a `PySliceContainer`
            // BASE object, and numpy never reports `owndata` for an array with a
            // base — the memory is still exclusively this array's (that base IS
            // the engine's `Vec<T>`), and it is writable and C-contiguous.
            assert!(!flag(&out, "owndata"));
            let base = out.getattr("base").unwrap();
            assert!(!base.is_none());
            assert_eq!(
                base.get_type().name().unwrap().to_string_lossy(),
                "PySliceContainer"
            );
            // Writing the result must not reach the caller's array.
            out.call_method1("fill", (7.0,)).unwrap();
            assert_eq!(bytes_of(&array), bytes);
            assert!(bytes_of(&out)
                .chunks_exact(4)
                .all(|word| word == 7.0f32.to_le_bytes()));
        });
    }

    #[test]
    fn a_transposed_input_is_read_in_logical_order() {
        Python::attach(|py| {
            let (source, _) = make_array(py, DType::Float32, 24);
            let matrix = source.call_method1("reshape", ((4, 6),)).unwrap();
            let transposed = matrix.call_method0("transpose").unwrap();
            assert!(!flag(&transposed, "c_contiguous"));
            let buffer = array_to_buffer(py, &transposed, "test").unwrap();
            assert_eq!(buffer.shape, vec![6, 4]);
            let out = buffer_to_array(py, buffer).unwrap();
            assert_eq!(bytes_of(&out), contiguous_bytes(py, &transposed));
        });
    }

    #[test]
    fn an_ndarray_subclass_is_accepted() {
        Python::attach(|py| {
            // `numpy.matrix` is an `ndarray` subclass; the downcast uses
            // `PyArray_Check`, so it is accepted like a plain array.
            let (source, _) = make_array(py, DType::Float64, 6);
            let matrix = numpy(py)
                .unwrap()
                .getattr("matrix")
                .unwrap()
                .call1((source.call_method1("reshape", ((2, 3),)).unwrap(),))
                .unwrap();
            let buffer = array_to_buffer(py, &matrix, "test").unwrap();
            assert_eq!(buffer.shape, vec![2, 3]);
            let reference = array_to_buffer(
                py,
                &source.call_method1("reshape", ((2, 3),)).unwrap(),
                "test",
            )
            .unwrap();
            assert_eq!(buffer.storage, reference.storage);
        });
    }

    #[test]
    fn a_non_array_argument_names_its_type() {
        Python::attach(|py| {
            let (array, _) = make_array(py, DType::Float32, 16);
            assert_eq!(
                array_to_buffer(py, &array, "test").unwrap().storage.len(),
                16
            );
            let err = array_to_buffer(py, &py.None().into_bound(py), "flat_inputs[0]").unwrap_err();
            assert!(
                err.to_string()
                    .contains("flat_inputs[0]: expected a numpy array, got NoneType"),
                "{err}"
            );
        });
    }

    #[test]
    fn a_buffer_numpy_buffer_chain_round_trips_bit_identically() {
        Python::attach(|py| {
            // buffer -> numpy array -> buffer again, twice through the bridge.
            // The comparison is over the numpy BYTES (not `Storage`, whose
            // float equality would trip on the NaNs the random payload carries).
            for dtype in evoxc_core::dtype::ALL {
                let (array, _) = make_array(py, dtype, 64);
                let first = ok(array_to_buffer(py, &array, "test"), dtype);
                let out = ok(buffer_to_array(py, first), dtype);
                let bytes = bytes_of(&out);
                let second = ok(array_to_buffer(py, &out, "test"), dtype);
                assert_eq!(second.dtype, dtype);
                assert_eq!(second.shape, vec![64]);
                let out_again = ok(buffer_to_array(py, second), dtype);
                assert_eq!(dtype_name(&out_again), dtype.name());
                assert_eq!(bytes_of(&out_again), bytes, "{} chain", dtype.name());
            }
        });
    }

    /// The committed FFI floor: 8 MB in + 8 MB out through the typed bridge.
    #[test]
    #[ignore = "timing probe: `--ignored --nocapture` refreshes the 8 MB round-trip number"]
    fn an_eight_megabyte_round_trip_is_measured() {
        Python::attach(|py| {
            const COUNT: usize = 2 << 20; // 2M float32 = 8 MB
            let (array, bytes) = make_array(py, DType::Float32, COUNT);
            // One untimed warm-up (numpy's C-API init, the allocator).
            let warm = array_to_buffer(py, &array, "test").unwrap();
            let warm = buffer_to_array(py, warm).unwrap();
            assert_eq!(bytes_of(&warm), bytes);

            const REPS: u32 = 20;
            let started = Instant::now();
            for _ in 0..REPS {
                let buffer = array_to_buffer(py, &array, "test").unwrap();
                let out = buffer_to_array(py, buffer).unwrap();
                std::hint::black_box(&out);
            }
            let per_round_trip = started.elapsed().as_secs_f64() / f64::from(REPS);
            println!(
                "8 MB round trip (2M f32 in + 2M f32 out): {:.3} ms/round trip, {:.3} ms/direction",
                per_round_trip * 1000.0,
                per_round_trip * 500.0
            );
        });
    }
}
