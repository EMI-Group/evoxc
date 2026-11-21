//! Executables: loaded-program metadata, the flat ABI, and the enqueue path.
//!
//! An [`Executable`] owns the metadata of a loaded program (a name and an
//! ordered input/output [`ExecutableAbi`]) plus a lease on the
//! [`DeviceContext`] it was loaded into.  [`Executable::submit`] is the
//! `enqueue`-style run: it validates the runtime inputs against the ABI and the
//! placement rules, returns output buffer handles that are `Pending` on the
//! submission, and hands back the [`Completion`] that reports the outcome.
//!
//! No stage of this path copies data.  A wrong device, a same-device foreign
//! session, a wrong dtype or a wrong shape is an error — never a staging or a
//! reinterpretation — exactly as the Python run boundary behaves.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use evoxc_core::{Dim, Error, Result, ValueType};

use crate::buffer::Buffer;
use crate::device::{Device, DeviceContext};
use crate::session::{BackendTag, SessionId};
use crate::submission::{Completion, Submission, SubmissionTracker};

static NEXT_EXECUTABLE_ID: AtomicU64 = AtomicU64::new(1);

/// Longest accepted program name.
pub const MAX_PROGRAM_NAME_LEN: usize = 256;

/// Process-unique executable identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExecutableId(u64);

impl ExecutableId {
    #[inline]
    pub fn value(self) -> u64 {
        self.0
    }

    pub(crate) fn allocate() -> ExecutableId {
        ExecutableId(NEXT_EXECUTABLE_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Display for ExecutableId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The flat call ABI of a loaded program: ordered input and output types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutableAbi {
    inputs: Vec<ValueType>,
    outputs: Vec<ValueType>,
}

impl ExecutableAbi {
    pub fn new(inputs: Vec<ValueType>, outputs: Vec<ValueType>) -> ExecutableAbi {
        ExecutableAbi { inputs, outputs }
    }

    pub fn inputs(&self) -> &[ValueType] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[ValueType] {
        &self.outputs
    }

    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }
}

/// The metadata a backend reports after loading a program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramMetadata {
    name: String,
    abi: ExecutableAbi,
}

impl ProgramMetadata {
    /// Build metadata, validating the program name (non-empty, bounded,
    /// printable) — the name is quoted in every diagnostic.
    pub fn new(name: impl Into<String>, abi: ExecutableAbi) -> Result<ProgramMetadata> {
        let name = name.into();
        if name.is_empty() {
            return Err(Error::general(
                "program metadata needs a non-empty name (it is quoted in run diagnostics)",
            ));
        }
        if name.len() > MAX_PROGRAM_NAME_LEN {
            return Err(Error::general(format!(
                "program name is {} bytes; the maximum is {MAX_PROGRAM_NAME_LEN}",
                name.len()
            )));
        }
        if name.chars().any(char::is_control) {
            return Err(Error::general(
                "program name must not contain control characters (it is quoted in diagnostics)",
            ));
        }
        Ok(ProgramMetadata { name, abi })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn abi(&self) -> &ExecutableAbi {
        &self.abi
    }
}

struct ExecutableInner {
    id: ExecutableId,
    metadata: ProgramMetadata,
    context: DeviceContext,
}

/// A loaded program that can be submitted against one device context.
#[derive(Clone)]
pub struct Executable {
    inner: Arc<ExecutableInner>,
}

impl Executable {
    /// Attach loaded-program metadata to the context the program was loaded on.
    ///
    /// Validates the ABI: every output type must be statically sized (the
    /// skeleton allocates output buffers itself) and no input may declare a
    /// negative dimension.  Input types may keep symbolic dimensions; those are
    /// resolved at run time by the backend, so `submit` only checks the dims
    /// that are already known.
    pub fn load(metadata: ProgramMetadata, context: DeviceContext) -> Result<Executable> {
        for (position, ty) in metadata.abi.inputs.iter().enumerate() {
            validate_input_type(position, ty)?;
        }
        for (position, ty) in metadata.abi.outputs.iter().enumerate() {
            crate::buffer::static_extent(ty).map_err(|error| {
                error.context(format!(
                    "output {position} of program {:?} decomposes to a runtime output type that this \
                     runtime cannot allocate",
                    metadata.name
                ))
            })?;
        }
        Ok(Executable {
            inner: Arc::new(ExecutableInner {
                id: ExecutableId::allocate(),
                metadata,
                context,
            }),
        })
    }

    pub fn id(&self) -> ExecutableId {
        self.inner.id
    }

    pub fn name(&self) -> &str {
        self.inner.metadata.name()
    }

    pub fn abi(&self) -> &ExecutableAbi {
        &self.inner.metadata.abi
    }

    /// The device the program was loaded on.
    pub fn device(&self) -> &Device {
        self.inner.context.device()
    }

    pub fn backend(&self) -> &BackendTag {
        self.inner.context.backend()
    }

    pub fn session_id(&self) -> SessionId {
        self.inner.context.session_id()
    }

    /// The context this executable leases.
    pub fn context(&self) -> &DeviceContext {
        &self.inner.context
    }

    /// Enqueue one execution over `inputs`.
    ///
    /// Validates arity, dtype, shape and placement, then returns the output
    /// buffer handles (pending on the submission) and its completion.  The
    /// submission retains `inputs` until it settles.
    pub fn submit(&self, inputs: Vec<Buffer>) -> Result<(Vec<Buffer>, Completion)> {
        let session = self.inner.context.session();
        session.ensure_open()?;

        let abi = self.abi();
        if inputs.len() != abi.input_count() {
            return Err(Error::shape(format!(
                "executable {} ({:?}) expects {} input buffer(s), got {}",
                self.id(),
                self.name(),
                abi.input_count(),
                inputs.len()
            )));
        }

        for (position, (buffer, expected)) in inputs.iter().zip(abi.inputs()).enumerate() {
            self.check_input_type(position, buffer, expected)?;
            self.check_input_placement(position, buffer)?;
        }

        let mut outputs = Vec::with_capacity(abi.output_count());
        for (position, ty) in abi.outputs().iter().enumerate() {
            let output = Buffer::alloc(ty.clone(), self.inner.context.clone()).map_err(|error| {
                error.context(format!(
                    "output {position} of executable {} ({:?})",
                    self.id(),
                    self.name()
                ))
            })?;
            outputs.push(output);
        }

        let tracker: Arc<SubmissionTracker> = self.inner.context.tracker();
        let submission = Submission::new(self.id(), inputs, outputs, &tracker);
        let completion = submission.completion();
        Ok((submission.outputs().to_vec(), completion))
    }

    /// Wait for the submissions issued through this executable before the call.
    ///
    /// The scope is a snapshot: later submissions are not covered.  A pending
    /// submission is an explicit `BackendError` (the skeleton has no driver to
    /// block on) and the first failed submission surfaces with its own error
    /// family.
    pub fn synchronize(&self) -> Result<()> {
        self.inner.context.tracker().wait_scope(Some(self.id()))
    }

    /// Submissions of this executable still pending inside the retained window.
    pub fn pending_submissions(&self) -> usize {
        self.inner.context.tracker().pending_scope(Some(self.id()))
    }

    fn check_input_type(&self, position: usize, buffer: &Buffer, expected: &ValueType) -> Result<()> {
        let actual = buffer.ty();
        if actual.dtype != expected.dtype {
            return Err(Error::dtype(format!(
                "input {position} of executable {} ({:?}) expects dtype {}, got {} (buffer {})",
                self.id(),
                self.name(),
                expected.dtype,
                actual.dtype,
                buffer.id()
            )));
        }
        if actual.rank() != expected.rank() {
            return Err(Error::shape(format!(
                "input {position} of executable {} ({:?}) expects rank {}, got rank {} (buffer {}: {})",
                self.id(),
                self.name(),
                expected.rank(),
                actual.rank(),
                buffer.id(),
                actual
            )));
        }
        for (axis, dim) in expected.shape.iter().enumerate() {
            let Some(wanted) = dim.known_size() else {
                // Symbolic/dynamic ABI dims are resolved by the backend at run
                // time; the runtime checks only what is already known.
                continue;
            };
            let got = actual.shape.get(axis).and_then(Dim::known_size);
            if got != Some(wanted) {
                return Err(Error::shape(format!(
                    "input {position} of executable {} ({:?}) expects axis {axis} of size {wanted}, got \
                     {} (buffer {}: {})",
                    self.id(),
                    self.name(),
                    got.map_or_else(|| "a non-static extent".to_owned(), |size| size.to_string()),
                    buffer.id(),
                    actual
                )));
            }
        }
        Ok(())
    }

    fn check_input_placement(&self, position: usize, buffer: &Buffer) -> Result<()> {
        let buffer_device = buffer.device();
        if !buffer_device.same_physical_device(self.device()) {
            return Err(Error::device(format!(
                "input {position} lives on {buffer_device} but executable {} ({:?}) runs on {}; there is \
                 no implicit device transfer — stage the data explicitly (DeviceContext::check_transfer) \
                 and resubmit",
                self.id(),
                self.name(),
                self.device()
            )));
        }
        if !buffer.context().compatible_with(self.context()) {
            return Err(Error::device(format!(
                "input {position} is on {buffer_device} but belongs to session {} (backend '{}', context \
                 {}), while executable {} ({:?}) belongs to session {} (backend '{}', context {}); \
                 same-device buffers from another session or backend cannot be consumed without an \
                 explicit cross-context import — copy through a cpu context",
                buffer.context().session_id(),
                buffer.backend(),
                buffer.context().id(),
                self.id(),
                self.name(),
                self.session_id(),
                self.backend(),
                self.context().id()
            )));
        }
        Ok(())
    }
}

impl fmt::Debug for Executable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Executable")
            .field("id", &self.inner.id)
            .field("name", &self.inner.metadata.name())
            .field("device", &self.device())
            .field("backend", &self.backend())
            .field("session", &self.session_id())
            .field("inputs", &self.abi().input_count())
            .field("outputs", &self.abi().output_count())
            .finish()
    }
}

fn validate_input_type(position: usize, ty: &ValueType) -> Result<()> {
    for (axis, dim) in ty.shape.iter().enumerate() {
        if let Some(size) = dim.known_size() {
            if size < 0 {
                return Err(Error::shape(format!(
                    "input {position} declares a negative dimension {size} at axis {axis} ({ty})"
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use evoxc_core::DType;

    use crate::buffer::Readiness;
    use crate::session::{Session, SessionConfig};

    fn session(backend: &str, devices: &[&str]) -> Session {
        let mut config = SessionConfig::new(BackendTag::parse(backend).expect("valid tag"));
        for device in devices {
            config = config.with_device(device.parse().expect("valid device"));
        }
        Session::open(config).expect("session opens")
    }

    fn f32(shape: &[i64]) -> ValueType {
        ValueType::of(DType::Float32, shape.iter().copied().map(Dim::int))
    }

    fn metadata(inputs: Vec<ValueType>, outputs: Vec<ValueType>) -> ProgramMetadata {
        ProgramMetadata::new("forward", ExecutableAbi::new(inputs, outputs)).expect("metadata")
    }

    fn executable_on(session: &Session, device: &str) -> Executable {
        let context = session
            .device_context(&device.parse().expect("valid device"))
            .expect("served device");
        Executable::load(metadata(vec![f32(&[2, 3])], vec![f32(&[2, 3])]), context).expect("loaded")
    }

    fn host_input(session: &Session, device: &str) -> Buffer {
        let context = session
            .device_context(&device.parse().expect("valid device"))
            .expect("served device");
        Buffer::host_zeroed(f32(&[2, 3]), context).expect("host input")
    }

    fn device_input(session: &Session, device: &str) -> Buffer {
        let context = session
            .device_context(&device.parse().expect("valid device"))
            .expect("served device");
        Buffer::alloc(f32(&[2, 3]), context).expect("device input")
    }

    #[test]
    fn load_exposes_the_flat_abi_and_placement() {
        let session = session("iree", &["rocm:0"]);
        let executable = executable_on(&session, "rocm:0");
        assert_eq!(executable.name(), "forward");
        assert_eq!(executable.abi().input_count(), 1);
        assert_eq!(executable.abi().output_count(), 1);
        assert_eq!(executable.device(), &"rocm:0".parse::<Device>().expect("valid"));
        assert_eq!(executable.backend().as_str(), "iree");
        assert_eq!(executable.session_id(), session.id());
        assert!(executable.id().value() > 0);
    }

    #[test]
    fn load_requires_a_program_name() {
        let abi = ExecutableAbi::new(vec![], vec![]);
        let empty = ProgramMetadata::new("", abi.clone()).expect_err("empty name");
        assert_eq!(empty.kind_str(), "EVOXCError");
        assert!(empty.message.contains("non-empty name"));

        let control = ProgramMetadata::new("a\tb", abi.clone()).expect_err("control char");
        assert!(control.message.contains("control characters"));

        let long = ProgramMetadata::new("x".repeat(MAX_PROGRAM_NAME_LEN + 1), abi).expect_err("long name");
        assert!(long.message.contains("maximum"));

        assert_eq!(ProgramMetadata::new("ok", ExecutableAbi::new(vec![], vec![])).expect("valid").name(), "ok");
    }

    #[test]
    fn load_rejects_output_types_it_cannot_allocate() {
        let session = session("iree", &["rocm:0"]);
        let context = session
            .device_context(&"rocm:0".parse().expect("valid"))
            .expect("served");
        let symbolic_output = ProgramMetadata::new(
            "forward",
            ExecutableAbi::new(vec![], vec![ValueType::of(DType::Float32, [Dim::named("B")])]),
        )
        .expect("metadata");
        let error = Executable::load(symbolic_output, context).expect_err("symbolic output");
        assert_eq!(error.kind_str(), "ShapeError");
        assert!(error.message.contains("output 0 of program"));
        assert!(error.message.contains("static"));
    }

    #[test]
    fn load_rejects_negative_input_dims() {
        let session = session("numpy", &["cpu:0"]);
        let context = session
            .device_context(&"cpu:0".parse().expect("valid"))
            .expect("served");
        let bad = ProgramMetadata::new(
            "forward",
            ExecutableAbi::new(vec![ValueType::of(DType::Float32, [Dim::int(-2)])], vec![]),
        )
        .expect("metadata");
        let error = Executable::load(bad, context).expect_err("negative dim");
        assert_eq!(error.kind_str(), "ShapeError");
        assert!(error.message.contains("negative dimension"));
    }

    #[test]
    fn submit_validates_arity() {
        let session = session("numpy", &["cpu:0"]);
        let executable = executable_on(&session, "cpu:0");
        let error = executable.submit(Vec::new()).expect_err("missing input");
        assert_eq!(error.kind_str(), "ShapeError");
        assert!(error.message.contains("expects 1 input buffer(s), got 0"));

        let error = executable
            .submit(vec![host_input(&session, "cpu:0"), host_input(&session, "cpu:0")])
            .expect_err("extra input");
        assert!(error.message.contains("got 2"));
    }

    #[test]
    fn submit_validates_dtype() {
        let session = session("numpy", &["cpu:0"]);
        let context = session
            .device_context(&"cpu:0".parse().expect("valid"))
            .expect("served");
        let executable = executable_on(&session, "cpu:0");
        // Same shape, different dtype.
        let wrong_dtype = Buffer::host_zeroed(
            ValueType::of(DType::Int32, [Dim::int(2), Dim::int(3)]),
            context,
        )
        .expect("int buffer");
        let error = executable.submit(vec![wrong_dtype]).expect_err("dtype mismatch");
        assert_eq!(error.kind_str(), "DTypeError");
        assert!(error.message.contains("expects dtype float32"));
        assert!(error.message.contains("got int32"));
    }

    #[test]
    fn submit_validates_shape() {
        let session = session("numpy", &["cpu:0"]);
        let context = session
            .device_context(&"cpu:0".parse().expect("valid"))
            .expect("served");
        let executable = executable_on(&session, "cpu:0");

        let wrong_rank = Buffer::host_zeroed(f32(&[6]), context.clone()).expect("rank 1");
        let error = executable.submit(vec![wrong_rank]).expect_err("rank mismatch");
        assert_eq!(error.kind_str(), "ShapeError");
        assert!(error.message.contains("expects rank 2"));
        assert!(error.message.contains("got rank 1"));

        let wrong_axis = Buffer::host_zeroed(f32(&[2, 4]), context).expect("wrong axis");
        let error = executable.submit(vec![wrong_axis]).expect_err("axis mismatch");
        assert_eq!(error.kind_str(), "ShapeError");
        assert!(error.message.contains("axis 1 of size 3"));
        assert!(error.message.contains("got 4"));
    }

    #[test]
    fn submit_rejects_a_wrong_device() {
        let session = session("iree", &["cpu:0", "rocm:0"]);
        let executable = executable_on(&session, "rocm:0");
        let error = executable
            .submit(vec![host_input(&session, "cpu:0")])
            .expect_err("host input into a device executable");
        assert_eq!(error.kind_str(), "DeviceError");
        assert!(error.message.contains("lives on cpu:0"));
        assert!(error.message.contains("rocm:0"));
        assert!(error.message.contains("no implicit device transfer"));
    }

    #[test]
    fn submit_rejects_a_same_device_foreign_session() {
        let first = session("iree", &["rocm:0"]);
        let second = session("iree", &["rocm:0"]);
        let executable = executable_on(&first, "rocm:0");
        let foreign = device_input(&second, "rocm:0");
        let error = executable.submit(vec![foreign]).expect_err("foreign session");
        assert_eq!(error.kind_str(), "DeviceError");
        assert!(error.message.contains(&first.id().to_string()));
        assert!(error.message.contains(&second.id().to_string()));
        assert!(error.message.contains("cross-context import"));
    }

    #[test]
    fn submit_rejects_a_same_device_foreign_backend() {
        let iree = session("iree", &["rocm:0"]);
        let xla = session("xla", &["rocm:0"]);
        let executable = executable_on(&iree, "rocm:0");
        let error = executable
            .submit(vec![device_input(&xla, "rocm:0")])
            .expect_err("foreign backend");
        assert_eq!(error.kind_str(), "DeviceError");
        assert!(error.message.contains("backend 'xla'"));
        assert!(error.message.contains("backend 'iree'"));
    }

    #[test]
    fn submit_enqueues_pending_outputs_on_the_executable_device() {
        let session = session("numpy", &["cpu:0"]);
        let executable = executable_on(&session, "cpu:0");
        let (outputs, completion) = executable
            .submit(vec![host_input(&session, "cpu:0")])
            .expect("submitted");
        assert_eq!(outputs.len(), 1);
        assert_eq!(completion.poll(), crate::submission::Status::Pending);
        assert_eq!(completion.outputs().len(), 1);
        let output = &outputs[0];
        assert_eq!(output.device(), executable.device());
        assert!(matches!(output.readiness(), Readiness::Pending(token) if token.value() == completion.id().value()));
        assert_eq!(output.ty(), &f32(&[2, 3]));
        assert!(executable.pending_submissions() == 1);

        let error = executable.synchronize().expect_err("no driver, still pending");
        assert_eq!(error.kind_str(), "BackendError");

        assert!(completion.complete_success());
        assert!(output.is_ready());
        assert_eq!(output.host_bytes().expect("ready bytes").len(), 24);
        executable.synchronize().expect("executable is quiescent");
        session.synchronize().expect("session is quiescent");
        completion.wait().expect("succeeded");
        assert_eq!(executable.pending_submissions(), 0);
    }

    #[test]
    fn synchronize_scope_is_the_calling_executable() {
        let session = session("iree", &["rocm:0"]);
        let context = session
            .device_context(&"rocm:0".parse().expect("valid"))
            .expect("served");
        let abi = ExecutableAbi::new(vec![], vec![f32(&[4])]);
        let first = Executable::load(ProgramMetadata::new("first", abi.clone()).expect("meta"), context.clone())
            .expect("loaded");
        let second = Executable::load(ProgramMetadata::new("second", abi).expect("meta"), context)
            .expect("loaded");

        let (_, first_completion) = first.submit(Vec::new()).expect("submit first");
        let (_, second_completion) = second.submit(Vec::new()).expect("submit second");
        assert!(first_completion.complete_success());

        first.synchronize().expect("first is quiescent");
        let error = second.synchronize().expect_err("second is pending");
        assert_eq!(error.kind_str(), "BackendError");
        assert!(error.message.contains(&second_completion.id().to_string()));
        let error = session.synchronize().expect_err("session scope sees second");
        assert_eq!(error.kind_str(), "BackendError");

        assert!(second_completion.complete_success());
        second.synchronize().expect("now quiescent");
        session.synchronize().expect("session quiescent");
    }

    #[test]
    fn synchronize_surfaces_an_asynchronous_failure() {
        let session = session("numpy", &["cpu:0"]);
        let executable = executable_on(&session, "cpu:0");
        let (_, completion) = executable
            .submit(vec![host_input(&session, "cpu:0")])
            .expect("submitted");
        assert!(completion.complete_failure(Error::backend("kernel aborted")));

        let error = executable.synchronize().expect_err("failure is recorded");
        assert_eq!(error.kind_str(), "BackendError");
        assert!(error.message.contains("kernel aborted"));
        assert!(error.message.contains(&completion.id().to_string()));
        let error = session.synchronize().expect_err("session scope too");
        assert_eq!(error.kind_str(), "BackendError");
    }

    #[test]
    fn a_closed_session_refuses_new_submissions() {
        let session = session("numpy", &["cpu:0"]);
        let executable = executable_on(&session, "cpu:0");
        let submission = executable
            .submit(vec![host_input(&session, "cpu:0")])
            .expect("submitted");
        let pending_close = session.close().expect_err("pending work blocks close");
        assert_eq!(pending_close.kind_str(), "BackendError");
        assert!(pending_close.message.contains("still pending"));
        assert!(!session.is_closed());

        assert!(submission.1.complete_success());
        session.close().expect("quiescent session closes");
        let error = executable
            .submit(vec![host_input(&session, "cpu:0")])
            .expect_err("closed session");
        assert_eq!(error.kind_str(), "DeviceError");
        assert!(error.message.contains("is closed"));
    }
}
