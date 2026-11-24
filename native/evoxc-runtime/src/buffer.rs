//! Buffers: storage, type metadata, an owning context lease, and readiness.
//!
//! A [`Buffer`] is the runtime's only opaque data handle.  It always reports
//! *where* its storage actually lives ([`BufferStorage::Host`] bytes on a
//! cpu-kind context, or an opaque [`DeviceHandle`] minted by a backend), it
//! always leases the [`DeviceContext`] that can use it (which in turn leases
//! the session, so dropping a `Session` handle never invalidates a live
//! buffer), and it is either [`Readiness::Ready`] or
//! [`Readiness::Pending`] on a dependency token.
//!
//! Two invariants are enforced here rather than documented and hoped for:
//!
//! 1. **Host bytes belong to the host device.**  [`Buffer::host`] refuses a
//!    non-cpu context, so numpy host memory can never masquerade as a device
//!    payload (proposal §13: "NumPy host memory is CPU memory, never a
//!    label-only CUDA, ROCm, or Vulkan tensor").
//! 2. **A pending buffer is never readable.**  Reading one is an explicit
//!    error naming the dependency, never zeroed or stale bytes.
//!
//! The skeleton sizes storage statically: a symbolic or runtime-dynamic shape
//! is rejected with a `ShapeError` instead of guessing a byte length.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use evoxc_core::{Dim, Error, Result, ValueType};

use crate::device::DeviceContext;
use crate::session::BackendTag;

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

/// Process-unique buffer identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferId(u64);

impl BufferId {
    #[inline]
    pub fn value(self) -> u64 {
        self.0
    }
}

impl fmt::Display for BufferId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The submission (or other asynchronous dependency) a pending buffer waits on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DependencyToken(u64);

impl DependencyToken {
    #[inline]
    pub fn new(value: u64) -> DependencyToken {
        DependencyToken(value)
    }

    #[inline]
    pub fn value(self) -> u64 {
        self.0
    }
}

impl fmt::Display for DependencyToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Whether a buffer's contents may be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Readiness {
    /// Contents are final.
    Ready,
    /// Some asynchronous work (`DependencyToken`) must settle first.
    Pending(DependencyToken),
}

impl Readiness {
    pub fn is_ready(self) -> bool {
        matches!(self, Readiness::Ready)
    }

    pub fn is_pending(self) -> bool {
        matches!(self, Readiness::Pending(_))
    }

    /// The dependency a pending buffer waits on.
    pub fn dependency(self) -> Option<DependencyToken> {
        match self {
            Readiness::Pending(token) => Some(token),
            Readiness::Ready => None,
        }
    }
}

/// An opaque handle to backend-owned device memory.
///
/// The runtime records the backend that minted the handle and, when the
/// backend has bound a resource, its identifier.  A placeholder handle
/// (`resource() == None`) is what the skeleton mints for an output buffer no
/// driver has filled yet; nothing in the runtime dereferences a handle, and a
/// real adapter replaces it at load time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceHandle {
    backend: BackendTag,
    resource: Option<u64>,
}

impl DeviceHandle {
    /// A handle bound to a backend resource id.
    pub fn bound(backend: BackendTag, resource: u64) -> DeviceHandle {
        DeviceHandle { backend, resource: Some(resource) }
    }

    /// An unbound handle: storage the runtime has not been given yet.
    pub fn placeholder(backend: BackendTag) -> DeviceHandle {
        DeviceHandle { backend, resource: None }
    }

    pub fn backend(&self) -> &BackendTag {
        &self.backend
    }

    pub fn resource(&self) -> Option<u64> {
        self.resource
    }

    pub fn is_placeholder(&self) -> bool {
        self.resource.is_none()
    }
}

/// Where a buffer's bytes live.
///
/// The enum is the explicit boundary between host data and device payloads:
/// there is no third "probably both" state, and no variant is reachable through
/// an implicit transfer.
#[derive(Debug)]
pub enum BufferStorage {
    /// C-order host bytes (numpy-compatible) on a cpu-kind context.
    Host(Vec<u8>),
    /// Device memory behind an opaque backend handle.
    Device(DeviceHandle),
}

impl BufferStorage {
    pub fn is_host(&self) -> bool {
        matches!(self, BufferStorage::Host(_))
    }
}

/// Metadata + storage + lease + readiness of one runtime buffer.
pub(crate) struct BufferInner {
    id: BufferId,
    ty: ValueType,
    storage: BufferStorage,
    context: DeviceContext,
    readiness: Mutex<Readiness>,
    element_count: usize,
    byte_len: usize,
}

/// A runtime buffer handle.
///
/// Cloning a `Buffer` shares the same storage: a buffer is an identity, not a
/// copy, exactly like the Python `Tensor` handle it backs.
#[derive(Clone)]
pub struct Buffer {
    inner: Arc<BufferInner>,
}

impl Buffer {
    /// Build a host buffer from C-order bytes on a cpu-kind context.
    ///
    /// Validates that `ty` has a fully static, non-negative shape and that
    /// `data.len()` is exactly `element_count * itemsize`; both failures are
    /// `ShapeError`s naming what was expected and what arrived.
    pub fn host(ty: ValueType, data: Vec<u8>, context: DeviceContext) -> Result<Buffer> {
        if !context.is_cpu() {
            return Err(Error::device(format!(
                "host bytes cannot live on {} (session {}): a host buffer requires a cpu-kind device \
                 context; upload the data to a device context explicitly instead of relabelling it",
                context.device(),
                context.session_id()
            )));
        }
        let (element_count, byte_len) = static_extent(&ty)?;
        if data.len() != byte_len {
            return Err(Error::shape(format!(
                "host buffer for {ty} needs exactly {byte_len} bytes ({element_count} elements), got {}; \
                 storage must match the value type (no truncation, no padding)",
                data.len()
            )));
        }
        Ok(Buffer::from_parts(ty, BufferStorage::Host(data), context, element_count, byte_len, Readiness::Ready))
    }

    /// Build a zero-initialized host buffer of the given static type.
    pub fn host_zeroed(ty: ValueType, context: DeviceContext) -> Result<Buffer> {
        let (_, byte_len) = static_extent(&ty)?;
        Buffer::host(ty, vec![0u8; byte_len], context)
    }

    /// Build a device buffer around a backend handle.
    ///
    /// The handle must have been minted by the context's backend; the buffer
    /// then leases that context (and through it the session) for as long as it
    /// lives.  Named `from_handle` because [`Buffer::device`] is the accessor.
    pub fn from_handle(ty: ValueType, handle: DeviceHandle, context: DeviceContext) -> Result<Buffer> {
        if handle.backend != *context.backend() {
            return Err(Error::device(format!(
                "device handle belongs to backend '{}' but the buffer is placed in a '{}' context \
                 (session {}, {}); a handle is only usable inside the backend that minted it",
                handle.backend,
                context.backend(),
                context.session_id(),
                context.device()
            )));
        }
        let (element_count, byte_len) = static_extent(&ty)?;
        Ok(Buffer::from_parts(
            ty,
            BufferStorage::Device(handle),
            context,
            element_count,
            byte_len,
            Readiness::Ready,
        ))
    }

    /// Allocate an output buffer in `context`.
    ///
    /// A cpu-kind context gets zeroed host bytes (the skeleton has no driver to
    /// write them), any other kind gets a placeholder device handle.  Outputs
    /// produced by a submission are armed `Pending` on that submission; a
    /// standalone `alloc` is `Ready`, which is the honest state for "nothing
    /// asynchronous owns these bytes".
    pub fn alloc(ty: ValueType, context: DeviceContext) -> Result<Buffer> {
        let (element_count, byte_len) = static_extent(&ty)?;
        let storage = if context.is_cpu() {
            BufferStorage::Host(vec![0u8; byte_len])
        } else {
            BufferStorage::Device(DeviceHandle::placeholder(context.backend().clone()))
        };
        Ok(Buffer::from_parts(ty, storage, context, element_count, byte_len, Readiness::Ready))
    }

    fn from_parts(
        ty: ValueType,
        storage: BufferStorage,
        context: DeviceContext,
        element_count: usize,
        byte_len: usize,
        readiness: Readiness,
    ) -> Buffer {
        Buffer {
            inner: Arc::new(BufferInner {
                id: BufferId(NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed)),
                ty,
                storage,
                context,
                readiness: Mutex::new(readiness),
                element_count,
                byte_len,
            }),
        }
    }

    pub fn id(&self) -> BufferId {
        self.inner.id
    }

    /// The buffer's value type (dtype + static shape).
    pub fn ty(&self) -> &ValueType {
        &self.inner.ty
    }

    /// The physical device the storage lives on.
    pub fn device(&self) -> &crate::device::Device {
        self.inner.context.device()
    }

    /// The backend whose handle/bytes these are.
    pub fn backend(&self) -> &BackendTag {
        self.inner.context.backend()
    }

    /// The context this buffer leases (and through it, the session).
    pub fn context(&self) -> &DeviceContext {
        &self.inner.context
    }

    /// Number of elements, computed once from the static type.
    pub fn element_count(&self) -> usize {
        self.inner.element_count
    }

    /// C-order byte length, computed once from dtype × element count.
    pub fn byte_len(&self) -> usize {
        self.inner.byte_len
    }

    /// True for host storage on a cpu-kind context.
    pub fn is_host_resident(&self) -> bool {
        self.inner.storage.is_host()
    }

    pub fn storage(&self) -> &BufferStorage {
        &self.inner.storage
    }

    pub fn readiness(&self) -> Readiness {
        *crate::lock(&self.inner.readiness)
    }

    pub fn is_ready(&self) -> bool {
        self.readiness().is_ready()
    }

    /// Host bytes, when this is a ready host buffer.
    ///
    /// A pending buffer raises an explicit `EVOXCError` naming the dependency;
    /// a device buffer raises a `DeviceError` (readback is an explicit
    /// transfer, never `.host_bytes()` on device memory).
    pub fn host_bytes(&self) -> Result<&[u8]> {
        if let Readiness::Pending(token) = self.readiness() {
            return Err(Error::general(format!(
                "buffer {} is pending on dependency {token}: its contents are not final; wait on the \
                 producing submission's completion before reading",
                self.id()
            )));
        }
        match &self.inner.storage {
            BufferStorage::Host(data) => Ok(data),
            BufferStorage::Device(_) => Err(Error::device(format!(
                "buffer {} holds device storage on {} (session {}); reading it needs an explicit transfer \
                 to a cpu context — there is no implicit device-to-host copy",
                self.id(),
                self.device(),
                self.context().session_id()
            ))),
        }
    }

    /// The opaque device handle, when this is a device buffer.
    pub fn device_handle(&self) -> Result<&DeviceHandle> {
        match &self.inner.storage {
            BufferStorage::Device(handle) => Ok(handle),
            BufferStorage::Host(_) => Err(Error::device(format!(
                "buffer {} is host-resident ({}), not a device buffer; there is no device handle to export",
                self.id(),
                self.device()
            ))),
        }
    }

    /// Driver hook: mark the buffer ready when it waits on exactly `token`.
    ///
    /// Returns whether the state changed.  A buffer armed on a different token
    /// is left untouched: one submission never settles another's output.
    pub(crate) fn arm(&self, token: DependencyToken) {
        *crate::lock(&self.inner.readiness) = Readiness::Pending(token);
    }

    pub(crate) fn mark_ready(&self, token: DependencyToken) -> bool {
        let mut readiness = crate::lock(&self.inner.readiness);
        match *readiness {
            Readiness::Pending(current) if current == token => {
                *readiness = Readiness::Ready;
                true
            }
            _ => false,
        }
    }

    /// Test support: how many handles share this buffer's allocation.
    #[cfg(test)]
    pub(crate) fn lease_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    /// Test support: a weak handle used to prove retention/release.
    #[cfg(test)]
    pub(crate) fn downgrade(&self) -> std::sync::Weak<BufferInner> {
        Arc::downgrade(&self.inner)
    }
}

impl fmt::Debug for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Buffer")
            .field("id", &self.inner.id)
            .field("ty", &self.inner.ty)
            .field("device", &self.inner.context.device())
            .field("backend", &self.inner.context.backend())
            .field("host_resident", &self.is_host_resident())
            .field("byte_len", &self.inner.byte_len)
            .field("readiness", &self.readiness())
            .finish()
    }
}

/// Element count and byte length of a fully static value type.
pub(crate) fn static_extent(ty: &ValueType) -> Result<(usize, usize)> {
    let mut elements: usize = 1;
    for (axis, dim) in ty.shape.iter().enumerate() {
        let size = match dim {
            Dim::Int(size) => *size,
            other => {
                return Err(Error::shape(format!(
                    "buffer type {ty} has a non-static dimension at axis {axis} ({other:?}); buffer \
                     storage is sized statically, so resolve the dimension before allocating"
                )));
            }
        };
        if size < 0 {
            return Err(Error::shape(format!(
                "buffer type {ty} has a negative dimension {size} at axis {axis}"
            )));
        }
        let size = usize::try_from(size).map_err(|_| {
            Error::shape(format!("buffer type {ty} has a dimension at axis {axis} that does not fit usize"))
        })?;
        elements = elements.checked_mul(size).ok_or_else(|| {
            Error::shape(format!("buffer type {ty} has more than usize::MAX elements"))
        })?;
    }
    let byte_len = elements.checked_mul(ty.dtype.itemsize()).ok_or_else(|| {
        Error::shape(format!("buffer type {ty} needs more than usize::MAX bytes"))
    })?;
    Ok((elements, byte_len))
}

#[cfg(test)]
mod tests {
    use super::*;

    use evoxc_core::DType;

    use crate::device::Device;
    use crate::session::{BackendTag, Session, SessionConfig};

    fn context(backend: &str, device: &str) -> DeviceContext {
        let config = SessionConfig::new(BackendTag::parse(backend).expect("valid tag"))
            .with_device(device.parse().expect("valid device"));
        let session = Session::open(config).expect("session opens");
        session
            .device_context(&device.parse().expect("valid device"))
            .expect("served device")
    }

    fn f32_matrix() -> ValueType {
        ValueType::of(DType::Float32, [Dim::int(3), Dim::int(4)])
    }

    #[test]
    fn host_buffer_sizes_come_from_the_value_type() {
        let buffer = Buffer::host(f32_matrix(), vec![0u8; 48], context("numpy", "cpu:0")).expect("sized");
        assert_eq!(buffer.element_count(), 12);
        assert_eq!(buffer.byte_len(), 48);
        assert!(buffer.is_host_resident());
        assert!(buffer.is_ready());
        assert_eq!(buffer.readiness(), Readiness::Ready);
        assert_eq!(buffer.device(), &Device::host());
        assert_eq!(buffer.backend().as_str(), "numpy");
        assert_eq!(buffer.ty(), &f32_matrix());
        assert_eq!(buffer.host_bytes().expect("host bytes").len(), 48);
        assert!(buffer.id().value() > 0);
    }

    #[test]
    fn host_buffer_byte_length_is_checked() {
        let short = Buffer::host(f32_matrix(), vec![0u8; 47], context("numpy", "cpu:0"))
            .expect_err("byte length must match");
        assert_eq!(short.kind_str(), "ShapeError");
        assert!(short.message.contains("48 bytes"));
        assert!(short.message.contains("47"));

        let long = Buffer::host(f32_matrix(), vec![0u8; 49], context("numpy", "cpu:0"))
            .expect_err("byte length must match");
        assert_eq!(long.kind_str(), "ShapeError");
    }

    #[test]
    fn host_storage_requires_a_cpu_context() {
        let error = Buffer::host(f32_matrix(), vec![0u8; 48], context("iree", "rocm:0"))
            .expect_err("host bytes on a device");
        assert_eq!(error.kind_str(), "DeviceError");
        assert!(error.message.contains("rocm:0"));
        assert!(error.message.contains("cpu-kind"));
    }

    #[test]
    fn symbolic_and_negative_shapes_are_rejected() {
        let symbolic = ValueType::of(DType::Float32, [Dim::int(2), Dim::named("B")]);
        let error = Buffer::host(symbolic, vec![0u8; 8], context("numpy", "cpu:0")).expect_err("symbolic");
        assert_eq!(error.kind_str(), "ShapeError");
        assert!(error.message.contains("non-static dimension at axis 1"));

        let dynamic = ValueType::of(DType::Float32, [Dim::dynamic()]);
        let error = Buffer::host(dynamic, Vec::new(), context("numpy", "cpu:0")).expect_err("dynamic");
        assert_eq!(error.kind_str(), "ShapeError");

        let negative = ValueType::of(DType::Float32, [Dim::int(-1)]);
        let error = Buffer::host(negative, Vec::new(), context("numpy", "cpu:0")).expect_err("negative");
        assert!(error.message.contains("negative dimension"));
    }

    #[test]
    fn zero_length_buffers_are_legal() {
        let ty = ValueType::of(DType::Int64, [Dim::int(0)]);
        let buffer = Buffer::host(ty, Vec::new(), context("numpy", "cpu:0")).expect("empty buffer");
        assert_eq!(buffer.element_count(), 0);
        assert_eq!(buffer.byte_len(), 0);
        assert!(buffer.host_bytes().expect("host bytes").is_empty());
    }

    #[test]
    fn host_zeroed_fills_the_declared_extent() {
        let buffer = Buffer::host_zeroed(f32_matrix(), context("numpy", "cpu:0")).expect("zeroed");
        assert_eq!(buffer.byte_len(), 48);
        assert!(buffer.host_bytes().expect("bytes").iter().all(|byte| *byte == 0));
    }

    #[test]
    fn device_buffers_require_a_matching_backend_and_are_not_host_readable() {
        let context = context("iree", "rocm:0");
        let handle = DeviceHandle::bound(BackendTag::parse("iree").expect("valid"), 7);
        let buffer = Buffer::from_handle(f32_matrix(), handle, context.clone()).expect("device buffer");
        assert!(!buffer.is_host_resident());
        assert_eq!(buffer.device_handle().expect("handle").resource(), Some(7));
        let error = buffer.host_bytes().expect_err("device storage is not host bytes");
        assert_eq!(error.kind_str(), "DeviceError");
        assert!(error.message.contains("explicit transfer"));

        let foreign = DeviceHandle::bound(BackendTag::parse("xla").expect("valid"), 9);
        let error = Buffer::from_handle(f32_matrix(), foreign, context).expect_err("foreign handle");
        assert_eq!(error.kind_str(), "DeviceError");
        assert!(error.message.contains("'xla'"));
        assert!(error.message.contains("'iree'"));
    }

    #[test]
    fn host_buffers_have_no_device_handle() {
        let buffer = Buffer::host_zeroed(f32_matrix(), context("numpy", "cpu:0")).expect("host buffer");
        let error = buffer.device_handle().expect_err("host storage");
        assert_eq!(error.kind_str(), "DeviceError");
        assert!(error.message.contains("host-resident"));
    }

    #[test]
    fn pending_buffers_are_never_readable() {
        let context = context("numpy", "cpu:0");
        let buffer = Buffer::host_zeroed(f32_matrix(), context).expect("host buffer");
        buffer.arm(DependencyToken::new(31));
        assert_eq!(buffer.readiness(), Readiness::Pending(DependencyToken::new(31)));
        assert_eq!(buffer.readiness().dependency(), Some(DependencyToken::new(31)));
        assert!(!buffer.is_ready());

        let error = buffer.host_bytes().expect_err("pending data has no final contents");
        assert_eq!(error.kind_str(), "EVOXCError");
        assert!(error.message.contains("pending on dependency 31"));

        assert!(!buffer.mark_ready(DependencyToken::new(32)), "a foreign token does not settle it");
        assert!(!buffer.is_ready());
        assert!(buffer.mark_ready(DependencyToken::new(31)));
        assert!(buffer.is_ready());
        assert!(buffer.host_bytes().expect("ready").iter().all(|byte| *byte == 0));
    }

    #[test]
    fn alloc_uses_host_bytes_on_host_and_a_placeholder_handle_on_devices() {
        let host = Buffer::alloc(f32_matrix(), context("numpy", "cpu:0")).expect("host alloc");
        assert!(host.is_host_resident());
        assert_eq!(host.host_bytes().expect("bytes").len(), 48);

        let device = Buffer::alloc(f32_matrix(), context("iree", "rocm:0")).expect("device alloc");
        assert!(!device.is_host_resident());
        let handle = device.device_handle().expect("handle");
        assert!(handle.is_placeholder());
        assert_eq!(handle.backend().as_str(), "iree");
        assert_eq!(device.readiness(), Readiness::Ready);
    }

    #[test]
    fn a_buffer_keeps_its_context_and_session_lease_alive() {
        let config = SessionConfig::new(BackendTag::parse("iree").expect("valid"))
            .with_device("rocm:0".parse().expect("valid device"));
        let session = Session::open(config).expect("opens");
        let context = session.device_context(&"rocm:0".parse().expect("valid")).expect("served");
        let weak_session = session.downgrade();
        let buffer = Buffer::alloc(f32_matrix(), context.clone()).expect("alloc");

        drop(session);
        assert!(weak_session.upgrade().is_some(), "the buffer leases its context, which leases the session");
        assert_eq!(buffer.context().id(), context.id());
        assert_eq!(buffer.context().lease_count(), 2, "the buffer holds a second context lease");

        drop(context);
        assert!(weak_session.upgrade().is_some(), "the buffer still holds the session open");
        drop(buffer);
        assert!(weak_session.upgrade().is_none(), "the last lease releases the session");
    }

    #[test]
    fn cloning_a_buffer_shares_its_storage() {
        let buffer = Buffer::host_zeroed(f32_matrix(), context("numpy", "cpu:0")).expect("host buffer");
        let clone = buffer.clone();
        assert_eq!(clone.id(), buffer.id());
        assert_eq!(clone.lease_count(), 2);
        clone.arm(DependencyToken::new(5));
        assert_eq!(buffer.readiness(), Readiness::Pending(DependencyToken::new(5)));
        assert!(clone.mark_ready(DependencyToken::new(5)));
        assert!(buffer.is_ready(), "readiness is shared, not copied");
    }
}
