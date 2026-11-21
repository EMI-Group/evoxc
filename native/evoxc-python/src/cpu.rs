//! The native CPU execution engine behind `native_compile` / `NativeProgram`.
//!
//! `native_compile(envelope) -> NativeProgram` decodes one `evoxc-native-ir`
//! envelope with [`evoxc_core::envelope::decode`] and hands the program to
//! [`evoxc_cpu::plan::compile`] (detached — the closure captures only `Send`
//! data).  The returned object is immutable: it owns the decoded `Program`,
//! the `Arc<Plan>` and the measured compile time, and `run()` executes it on
//! typed host buffers through [`evoxc_cpu::run_program`].
//!
//! ## numpy interop (borrow on the way in, one copy out)
//!
//! The newest `numpy` crate (0.28) requires pyo3 0.28 while this crate pins
//! pyo3 0.30, so there is no `numpy` crate to use here and the binding crosses
//! the boundary as raw little-endian bytes.  Both directions are ONE pass over
//! the payload — a marshalling memcpy, not an element-at-a-time conversion
//! through an intermediate `Vec<u8>`:
//!
//! * **in** — `arr.dtype.name` → [`evoxc_core::DType::parse`], `arr.shape` →
//!   dims, `arr.flags.c_contiguous` decides whether `numpy.ascontiguousarray`
//!   is applied first (a strided array is never silently misread, and the
//!   caller's array is only borrowed).  The bytes are BORROWED from the array
//!   through a `memoryview` cast to `B` ([`pyo3::buffer::PyBuffer<u8>`] +
//!   `as_slice_ptr`) and decoded straight into the typed
//!   [`evoxc_cpu::Storage`] (one pass, no intermediate byte vector); the
//!   caller's array is never mutated.
//! * **out** — the destination array is allocated ONCE with
//!   `numpy.empty(shape, dtype)` (so it owns its bytes, is writable and
//!   C-contiguous) and the storage's little-endian bytes are bulk-copied into
//!   it with [`pyo3::buffer::PyBuffer::copy_from_slice`] (one `memcpy`).  On a
//!   little-endian target the typed storage's own bytes ARE the numpy layout
//!   for every dtype (f16 2 bytes, complex re/im contiguous, bool 1 byte), so
//!   the copy reads them through a zero-copy `&[u8]` view; a big-endian host
//!   keeps the per-element `to_le_bytes` flattening.
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
use num_complex::Complex;
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyList, PyModule, PyTuple};
use serde_json::{Map, Value as Json};

use evoxc_core::graph::Program;
use evoxc_core::DType;
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

    // Borrow the array's bytes and decode them straight into the typed storage:
    // one pass over the payload, no intermediate `Vec<u8>`.  `view` owns the
    // borrow, so it must outlive `bytes` (and no Python code runs in between).
    let view = byte_view(py, array, path)?;
    let bytes = borrowed_bytes(py, &view, path)?;

    let dims = shape
        .iter()
        .map(|dim| {
            usize::try_from(*dim).map_err(|_| {
                error::raise(
                    py,
                    "ShapeError",
                    format!("{path}: negative dimension ({dim}) in the array shape"),
                )
            })
        })
        .collect::<PyResult<Vec<usize>>>()?;
    let expected = dims.iter().product::<usize>() * dtype.itemsize();
    if bytes.len() != expected {
        return Err(error::raise(
            py,
            "VerificationError",
            format!(
                "{path}: the array's shape {dims:?} declares {expected} byte(s) but it carries {}",
                bytes.len()
            ),
        ));
    }
    let storage = decode_bytes(dtype, bytes, split_threads(bytes.len()));
    Buffer::new(dtype, dims, storage).map_err(|err| error::from_core(py, err))
}

/// The `B`-cast memoryview of a numpy array, as a [`PyBuffer`] borrow.
fn byte_view(py: Python<'_>, array: &Bound<'_, PyAny>, path: &str) -> PyResult<PyBuffer<u8>> {
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
    PyBuffer::<u8>::get(&cast).map_err(|err| {
        error::raise(
            py,
            "BackendError",
            format!(
                "{path}: numpy refused a contiguous byte buffer ({})",
                message(err.value(py))
            ),
        )
    })
}

/// The borrowed bytes of a contiguous byte view.
///
/// The slice borrows the exporter's memory, so the caller must keep `view`
/// alive for as long as it uses the slice and must not run Python code
/// meanwhile (a Python call could resize the exporting array).
fn borrowed_bytes<'a>(py: Python<'_>, view: &'a PyBuffer<u8>, path: &str) -> PyResult<&'a [u8]> {
    match view.as_slice_ptr() {
        // SAFETY: the pointer addresses the buffer's own `item_count()` bytes,
        // which `view` keeps exported for `'a`; `u8` has no validity or
        // alignment requirements, and nothing writes through this shared slice.
        Some(ptr) => Ok(unsafe { &*ptr.as_ptr() }),
        None => Err(error::raise(
            py,
            "BackendError",
            format!("{path}: numpy refused a contiguous byte buffer (non-contiguous strides)"),
        )),
    }
}

/// Decode raw little-endian bytes into the typed storage (bool as `!= 0`).
///
/// Byte-for-byte what [`evoxc_cpu::Buffer::from_payload`] decodes, just read
/// from a borrowed slice instead of a copied `Vec<u8>`, and split across
/// `threads` for a payload large enough to pay for them.
fn decode_bytes(dtype: DType, bytes: &[u8], threads: usize) -> Storage {
    let count = bytes.len() / dtype.itemsize();
    match dtype {
        DType::Bool => Storage::Bool(decode_words::<bool, 1>(bytes, count, threads, |word| {
            word[0] != 0
        })),
        DType::Int8 => Storage::I8(decode_words::<i8, 1>(bytes, count, threads, |word| {
            word[0] as i8
        })),
        DType::UInt8 => Storage::U8(decode_words::<u8, 1>(bytes, count, threads, |word| word[0])),
        DType::Int16 => Storage::I16(decode_words(bytes, count, threads, i16::from_le_bytes)),
        DType::UInt16 => Storage::U16(decode_words(bytes, count, threads, u16::from_le_bytes)),
        DType::Float16 => Storage::F16(decode_words::<f16, 2>(bytes, count, threads, |word| {
            f16::from_bits(u16::from_le_bytes(word))
        })),
        DType::Int32 => Storage::I32(decode_words(bytes, count, threads, i32::from_le_bytes)),
        DType::UInt32 => Storage::U32(decode_words(bytes, count, threads, u32::from_le_bytes)),
        DType::Float32 => Storage::F32(decode_words(bytes, count, threads, f32::from_le_bytes)),
        DType::Int64 => Storage::I64(decode_words(bytes, count, threads, i64::from_le_bytes)),
        DType::UInt64 => Storage::U64(decode_words(bytes, count, threads, u64::from_le_bytes)),
        DType::Float64 => Storage::F64(decode_words(bytes, count, threads, f64::from_le_bytes)),
        DType::Complex64 => Storage::C64(decode_words(bytes, count, threads, |word: [u8; 8]| {
            Complex::new(
                f32::from_le_bytes([word[0], word[1], word[2], word[3]]),
                f32::from_le_bytes([word[4], word[5], word[6], word[7]]),
            )
        })),
        DType::Complex128 => {
            Storage::C128(decode_words(bytes, count, threads, |word: [u8; 16]| {
                Complex::new(
                    f64::from_le_bytes([
                        word[0], word[1], word[2], word[3], word[4], word[5], word[6], word[7],
                    ]),
                    f64::from_le_bytes([
                        word[8], word[9], word[10], word[11], word[12], word[13], word[14],
                        word[15],
                    ]),
                )
            }))
        }
    }
}

/// The output address shared with [`decode_words`]' helper threads.
///
/// The threads write disjoint index ranges through it, so sharing the raw
/// address is sound (`*mut T` is neither `Send` nor `Sync` on its own).
struct OutPtr<T>(*mut T);
// SAFETY: `decode_words` hands out disjoint output ranges only.
unsafe impl<T> Send for OutPtr<T> {}
unsafe impl<T> Sync for OutPtr<T> {}

impl<T> OutPtr<T> {
    /// Write one element at `index` (each index is written exactly once).
    ///
    /// # Safety
    /// `index` must be inside the allocation this pointer came from, and no
    /// other thread may write the same index.
    unsafe fn write(&self, index: usize, value: T) {
        self.0.add(index).write(value);
    }
}

/// Decode `count` little-endian words of `N` bytes each into a vector of `T`,
/// splitting the work across `threads` when asked.
///
/// The values are written through raw pointers, one disjoint range per thread,
/// so no reference to uninitialized memory is created and the trailing
/// `set_len` is sound; `bytes` must hold at least `count * N` bytes.
fn decode_words<T: Copy + Send, const N: usize>(
    bytes: &[u8],
    count: usize,
    threads: usize,
    decode: impl Fn([u8; N]) -> T + Sync,
) -> Vec<T> {
    let mut values: Vec<T> = Vec::with_capacity(count);
    let chunk = count.div_ceil(threads.max(1));
    // SAFETY: `values` has room for `count` elements and is empty, so its
    // allocation is exclusively ours; every thread writes its own
    // `[start, start + len)` range through the raw pointer (`ptr::write` into
    // uninitialized memory is fine for a `Copy` type — no reference to it is
    // ever formed), and `set_len(count)` only runs after the join, when every
    // element is initialized.  `count * N` bytes are guaranteed by the caller.
    unsafe {
        let out = OutPtr(values.as_mut_ptr());
        let body = |start: usize, len: usize| {
            let mut offset = start * N;
            for index in start..start + len {
                let mut word = [0u8; N];
                word.copy_from_slice(&bytes[offset..offset + N]);
                out.write(index, decode(word));
                offset += N;
            }
        };
        if threads <= 1 || chunk == 0 || chunk >= count {
            body(0, count);
        } else {
            std::thread::scope(|scope| {
                let mut next = chunk;
                while next < count {
                    let len = chunk.min(count - next);
                    let body = &body;
                    scope.spawn(move || body(next, len));
                    next += len;
                }
                // The calling thread takes the leading chunk instead of idling.
                body(0, chunk);
            });
        }
        values.set_len(count);
    }
    values
}

/// Copy `src` into `dst`, splitting the work across `threads` when asked.
fn copy_bytes(dst: &mut [u8], src: &[u8], threads: usize) {
    if threads <= 1 || dst.len() <= threads {
        dst.copy_from_slice(src);
        return;
    }
    let chunk = dst.len().div_ceil(threads);
    std::thread::scope(|scope| {
        let mut chunks = dst.chunks_mut(chunk);
        let mut offset = 0;
        let first = chunks.next();
        for chunk in chunks {
            let start = offset + chunk.len();
            let window = &src[start..start + chunk.len()];
            scope.spawn(move || chunk.copy_from_slice(window));
            offset = start;
        }
        // The calling thread takes the leading chunk instead of idling.
        if let Some(chunk) = first {
            chunk.copy_from_slice(&src[..chunk.len()]);
        }
    });
}

/// Element count a flat copy of `bytes` bytes should split across, mirroring
/// the engine's own threshold: one thread below 1 MiB, then one more per
/// further 512 KiB up to four (a scoped spawn costs ~12 µs here, so spawning
/// for a smaller payload loses more than it gains).
const SPLIT_MIN_BYTES: usize = 1 << 20;
const SPLIT_PER_THREAD: usize = 1 << 19;
const SPLIT_MAX_THREADS: usize = 4;

fn split_threads(bytes: usize) -> usize {
    if bytes < SPLIT_MIN_BYTES {
        return 1;
    }
    static CORES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let cores = *CORES.get_or_init(|| {
        std::thread::available_parallelism()
            .map(|threads| threads.get())
            .unwrap_or(1)
            .clamp(1, SPLIT_MAX_THREADS)
    });
    cores.min((bytes / SPLIT_PER_THREAD).max(1))
}

/// [`Buffer`] → a fresh, writable, C-contiguous numpy array (owns its bytes).
///
/// One allocation (`numpy.empty`) and one bulk copy: nothing is shared with the
/// engine, and the caller's inputs are untouched.
fn buffer_to_array<'py>(py: Python<'py>, buffer: &Buffer) -> PyResult<Bound<'py, PyAny>> {
    let shape = PyTuple::new(py, &buffer.shape)?;
    let empty = numpy(py)?.getattr("empty")?;
    let array = empty.call1((shape, buffer.dtype.name()))?;
    let view = array.getattr("data")?.call_method1("cast", ("B",))?;
    let dest = PyBuffer::<u8>::get(&view)?;

    // On a little-endian target the typed storage's bytes already are the numpy
    // layout for every dtype, so a borrowed view is enough (no flattening); a
    // big-endian host keeps the per-element `to_le_bytes` path.
    let threads = split_threads(buffer.storage.len() * buffer.dtype.itemsize());
    if cfg!(target_endian = "little") {
        fill_bytes(py, &dest, storage_bytes_le(&buffer.storage), threads)?;
    } else {
        let bytes = storage_to_bytes(&buffer.storage);
        fill_bytes(py, &dest, &bytes, threads)?;
    }
    Ok(array)
}

/// Write `bytes` into a freshly allocated numpy array's memory.
fn fill_bytes(py: Python<'_>, dest: &PyBuffer<u8>, bytes: &[u8], threads: usize) -> PyResult<()> {
    if threads <= 1 {
        return dest.copy_from_slice(py, bytes);
    }
    let Some(ptr) = dest.as_slice_ptr() else {
        return Err(pyo3::exceptions::PyBufferError::new_err(
            "the destination array is not C-contiguous",
        ));
    };
    // SAFETY: `dest` is the `B`-cast view of an array this call just allocated
    // with `numpy.empty` and has not returned or shared yet, so this exclusive
    // borrow of its `bytes.len()` bytes cannot alias anything; no Python code
    // runs while it lives, and `copy_bytes` splits it into disjoint chunks.
    let dst: &mut [u8] = unsafe { &mut *ptr.as_ptr() };
    if dst.len() != bytes.len() {
        return Err(pyo3::exceptions::PyBufferError::new_err(format!(
            "the destination array holds {} byte(s) but {} were produced",
            dst.len(),
            bytes.len()
        )));
    }
    copy_bytes(dst, bytes, threads);
    Ok(())
}

/// The little-endian bytes of a storage, borrowed without a copy.
///
/// Only valid on a little-endian target: there the in-memory bytes of every
/// typed storage are the numpy layout (`f16` is its 2-byte bit pattern,
/// `Complex` is `repr(C) {re, im}`, `bool` is one byte holding 0 or 1).
fn storage_bytes_le(storage: &Storage) -> &[u8] {
    /// A byte view of any typed slice: sound because `u8` has no validity or
    /// alignment requirements and `size_of_val` is exactly the slice's byte
    /// range (the classic `bytemuck::cast_slice`, made safe by the `u8` target).
    fn view<T>(values: &[T]) -> &[u8] {
        // SAFETY: see above — a shared, initialized `&[T]` reinterpreted as a
        // `&[u8]` of exactly its bytes; nothing writes through it and the
        // lifetime is the source's.
        unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        }
    }
    match storage {
        Storage::Bool(values) => view(values),
        Storage::I8(values) => view(values),
        Storage::I16(values) => view(values),
        Storage::I32(values) => view(values),
        Storage::I64(values) => view(values),
        Storage::U8(values) => view(values),
        Storage::U16(values) => view(values),
        Storage::U32(values) => view(values),
        Storage::U64(values) => view(values),
        Storage::F16(values) => view(values),
        Storage::F32(values) => view(values),
        Storage::F64(values) => view(values),
        Storage::C64(values) => view(values),
        Storage::C128(values) => view(values),
    }
}

/// Flatten storage to little-endian bytes (bool as `0x00`/`0x01`, f16 via its
/// bit pattern, complex as interleaved real/imaginary).
///
/// This is the big-endian host fallback; on a little-endian target
/// [`storage_bytes_le`] hands out the same bytes as a borrowed view.
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

/// The numpy marshalling surface, exercised through a real interpreter (run
/// with `cargo test -p evoxc-python --no-default-features --features auto-init`).
#[cfg(all(test, feature = "auto-init"))]
mod marshalling {
    use super::*;
    use pyo3::types::PyByteArray;

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

    #[test]
    fn every_dtype_round_trips_bit_identically() {
        Python::attach(|py| {
            for dtype in evoxc_core::dtype::ALL {
                if dtype == DType::Bool {
                    continue; // non-canonical byte values normalize (see below)
                }
                let (array, bytes) = make_array(py, dtype, 1024);
                let buffer = array_to_buffer(py, &array, "test").expect(dtype.name());
                assert_eq!(buffer.dtype, dtype);
                assert_eq!(buffer.shape, vec![1024]);
                let out = buffer_to_array(py, &buffer).expect(dtype.name());
                assert_eq!(bytes_of(&out), bytes, "{} round trip", dtype.name());
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
            // The previous path (`Buffer::from_payload`) also decoded any
            // non-zero byte as `true` and re-encoded it as 1.
            let buffer = array_to_buffer(py, &array, "test").unwrap();
            assert_eq!(buffer.storage, Storage::Bool(vec![false, true, true, true]));
            let out = buffer_to_array(py, &buffer).unwrap();
            assert_eq!(bytes_of(&out), vec![0u8, 1, 1, 1]);
        });
    }

    #[test]
    fn large_payloads_take_the_threaded_path_bit_identically() {
        Python::attach(|py| {
            // Above the 1 MiB split threshold, so the decode and the copy are
            // both chunked across threads.
            for (dtype, count) in [
                (DType::Float32, 1 << 20),
                (DType::Int64, 1 << 19),
                (DType::Complex128, 1 << 18),
                (DType::Float16, 1 << 21),
                (DType::UInt8, 1 << 21),
                (DType::Int8, 1 << 21),
                (DType::Bool, 1 << 21),
            ] {
                assert!(count * dtype.itemsize() > SPLIT_MIN_BYTES);
                let (array, bytes) = make_array(py, dtype, count);
                let buffer = array_to_buffer(py, &array, "test").expect(dtype.name());
                assert_eq!(buffer.storage.len(), count);
                let out = buffer_to_array(py, &buffer).expect(dtype.name());
                let expected = if dtype == DType::Bool {
                    bytes.iter().map(|byte| u8::from(*byte != 0)).collect()
                } else {
                    bytes
                };
                assert_eq!(bytes_of(&out), expected, "{} round trip", dtype.name());
            }
        });
    }

    #[test]
    fn the_returned_array_is_fresh_writable_and_does_not_alias_the_input() {
        Python::attach(|py| {
            let (array, bytes) = make_array(py, DType::Float32, 4096);
            let buffer = array_to_buffer(py, &array, "test").unwrap();
            let out = buffer_to_array(py, &buffer).unwrap();
            let flags = out.getattr("flags").unwrap();
            for flag in ["owndata", "c_contiguous", "writeable"] {
                assert!(
                    flags.getattr(flag).unwrap().extract::<bool>().unwrap(),
                    "the result must be {flag}"
                );
            }
            // Writing the result must not reach the caller's array.
            out.call_method1("fill", (7.0,)).unwrap();
            assert_eq!(bytes_of(&array), bytes);
            assert!(bytes_of(&out)
                .chunks_exact(4)
                .all(|word| word == 7.0f32.to_le_bytes()));
        });
    }

    #[test]
    fn a_strided_input_is_copied_contiguous_not_misread() {
        Python::attach(|py| {
            let (source, _) = make_array(py, DType::Float32, 24);
            let matrix = source.call_method1("reshape", ((4, 6),)).unwrap();
            let transposed = matrix.call_method0("transpose").unwrap();
            assert!(!transposed
                .getattr("flags")
                .unwrap()
                .getattr("c_contiguous")
                .unwrap()
                .extract::<bool>()
                .unwrap());
            let buffer = array_to_buffer(py, &transposed, "test").unwrap();
            assert_eq!(buffer.shape, vec![6, 4]);
            let contiguous = numpy(py)
                .unwrap()
                .getattr("ascontiguousarray")
                .unwrap()
                .call1((&transposed,))
                .unwrap();
            assert_eq!(
                storage_bytes_le(&buffer.storage),
                bytes_of(&contiguous).as_slice()
            );
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
            let out = buffer_to_array(py, &buffer).unwrap();
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
            let out = buffer_to_array(py, &buffer).unwrap();
            assert_eq!(bytes_of(&out), Vec::<u8>::new());
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
            assert!(err.to_string().contains("expected a numpy array"), "{err}");
        });
    }

    #[test]
    fn split_threads_only_splits_large_payloads() {
        assert_eq!(split_threads(0), 1);
        assert_eq!(split_threads(SPLIT_MIN_BYTES - 1), 1);
        assert!(split_threads(SPLIT_MIN_BYTES) >= 1);
        assert_eq!(split_threads(64 << 20), split_threads(1 << 22).max(1));
        assert!(split_threads(64 << 20) <= SPLIT_MAX_THREADS);
    }
}
