//! Physical devices and the runtime contexts that can use their memory.
//!
//! A [`Device`] is a *placement*: a kind plus a per-kind enumeration index,
//! exactly the Python `core.Device` value model.  A [`DeviceContext`] is the
//! runtime identity that owns an allocator/queue/plugin lease for one device
//! inside one [`Session`](crate::Session): two clients on the same physical
//! GPU may own mutually incompatible buffers, so the runtime keeps the two
//! identities distinct and diagnoses placement errors instead of reconciling
//! them (`native/CONTEXT.md`, proposal §12/§13).
//!
//! Kinds are a validated string, never a second enum: the accepted spellings
//! are the ones the Python contract publishes (`cpu`, `cuda`, `rocm`, `hip`,
//! `vulkan`), and `hip` is preserved as spelled while canonicalizing to
//! `rocm` only for physical-identity comparison.

use std::fmt;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use evoxc_core::{Error, Result};

use crate::placement::{check_transfer, Placement};
use crate::session::{BackendTag, SessionId, SessionResources};
use crate::submission::SubmissionTracker;

/// Every device-kind spelling the Python contract accepts.
pub const KNOWN_DEVICE_KINDS: [&str; 5] = ["cpu", "cuda", "rocm", "hip", "vulkan"];

/// Model bound on a per-kind device index.
///
/// Indices are per-kind enumeration positions (never HAL ids), so a runtime
/// that enumerates this many devices already exceeds what evoxc models; the
/// bound keeps `Device` a small copyable value and rejects typos such as
/// `rocm:4294967295`.
pub const MAX_DEVICE_INDEX: u32 = 64;

/// A validated device-kind spelling.
///
/// Construction is the only validation gate, so a `DeviceKind` always spells
/// one of [`KNOWN_DEVICE_KINDS`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeviceKind(String);

impl DeviceKind {
    /// Parse a kind spelling; anything outside the known set is a
    /// `DeviceError` naming the accepted spellings.
    pub fn parse(kind: &str) -> Result<DeviceKind> {
        if KNOWN_DEVICE_KINDS.contains(&kind) {
            Ok(DeviceKind(kind.to_owned()))
        } else {
            Err(Error::device(format!(
                "unknown device kind {kind:?}; accepted kinds are {:?} (kind spellings come from the Python core.Device contract)",
                KNOWN_DEVICE_KINDS
            )))
        }
    }

    /// The host kind (`cpu`).
    pub fn cpu() -> DeviceKind {
        DeviceKind("cpu".to_owned())
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[inline]
    pub fn is_cpu(&self) -> bool {
        self.0 == "cpu"
    }

    /// The canonical physical kind: `hip` is the AMD alias of `rocm`.
    #[inline]
    pub fn canonical(&self) -> &str {
        if self.0 == "hip" {
            "rocm"
        } else {
            &self.0
        }
    }
}

impl fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A physical placement: a validated kind plus a per-kind index.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Device {
    kind: DeviceKind,
    index: u32,
}

impl Device {
    /// Build a device, validating the kind and the index.
    pub fn new(kind: &str, index: u32) -> Result<Device> {
        let kind = DeviceKind::parse(kind)?;
        if kind.is_cpu() && index != 0 {
            return Err(Error::device(format!(
                "cpu has a single host device (cpu:0); got cpu:{index} — host bytes always belong to cpu:0"
            )));
        }
        if index >= MAX_DEVICE_INDEX {
            return Err(Error::device(format!(
                "device index {index} for kind {kind:?} is out of range: indices are per-kind enumeration positions in 0..{MAX_DEVICE_INDEX}"
            )));
        }
        Ok(Device { kind, index })
    }

    /// The only host placement: `cpu:0`.
    pub fn host() -> Device {
        Device { kind: DeviceKind::cpu(), index: 0 }
    }

    #[inline]
    pub fn kind(&self) -> &DeviceKind {
        &self.kind
    }

    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }

    #[inline]
    pub fn is_cpu(&self) -> bool {
        self.kind.is_cpu()
    }

    /// True when both values name the same physical device.
    ///
    /// `rocm:0` and `hip:0` are the same AMD device spelled two ways; every
    /// other comparison is kind-string plus index.  Device *equality* stays
    /// spelling-sensitive (the Python contract preserves `hip` as spelled).
    pub fn same_physical_device(&self, other: &Device) -> bool {
        self.index == other.index && self.kind.canonical() == other.kind.canonical()
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.kind, self.index)
    }
}

impl FromStr for Device {
    type Err = Error;

    /// Parse `kind` (index 0) or `kind:index`.
    fn from_str(text: &str) -> Result<Device> {
        match text.split_once(':') {
            None => Device::new(text, 0),
            Some((kind, index)) => {
                if kind.is_empty() || index.is_empty() || index.contains(':') {
                    return Err(Error::device(format!(
                        "invalid device spelling {text:?}: expected 'kind' or 'kind:index'"
                    )));
                }
                let index: u32 = index.parse().map_err(|_| {
                    Error::device(format!(
                        "invalid device index in {text:?}: expected a non-negative integer"
                    ))
                })?;
                Device::new(kind, index)
            }
        }
    }
}

static NEXT_CONTEXT_ID: AtomicU64 = AtomicU64::new(1);

/// The runtime context able to use one device's buffers.
///
/// A context leases its session's resources (`Arc<SessionResources>`), so
/// dropping the public [`Session`](crate::Session) handle never invalidates a
/// context, buffer or executable that still leases them.  Context *identity*
/// is allocation identity: `device_context()` hands out clones of one shared
/// inner value, and [`DeviceContext::same_context`] is the compatibility test
/// consumers and [`check_transfer`] use.
#[derive(Clone)]
pub struct DeviceContext {
    inner: Arc<DeviceContextInner>,
}

pub(crate) struct DeviceContextInner {
    id: u64,
    device: Device,
    backend: BackendTag,
    session: Arc<SessionResources>,
}

impl DeviceContext {
    pub(crate) fn new(device: Device, backend: BackendTag, session: Arc<SessionResources>) -> DeviceContext {
        DeviceContext {
            inner: Arc::new(DeviceContextInner {
                id: NEXT_CONTEXT_ID.fetch_add(1, Ordering::Relaxed),
                device,
                backend,
                session,
            }),
        }
    }

    /// Process-unique context id (distinct from the session id: one session
    /// serves one context per device it opened).
    pub fn id(&self) -> u64 {
        self.inner.id
    }

    #[inline]
    pub fn device(&self) -> &Device {
        &self.inner.device
    }

    #[inline]
    pub fn backend(&self) -> &BackendTag {
        &self.inner.backend
    }

    pub fn session_id(&self) -> SessionId {
        self.inner.session.id()
    }

    #[inline]
    pub fn is_cpu(&self) -> bool {
        self.inner.device.is_cpu()
    }

    /// True when both handles refer to the very same context (same session,
    /// same backend, same device, same lease).
    pub fn same_context(&self, other: &DeviceContext) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// True when a buffer of this context may be consumed by `other` without
    /// any transfer: identical context identity.
    pub fn compatible_with(&self, other: &DeviceContext) -> bool {
        self.same_context(other)
    }

    /// Classify an explicit transfer from this context into `to`.
    ///
    /// There is no implicit movement anywhere in the runtime: this is the
    /// helper a caller consults before staging data, and it returns a
    /// `DeviceError` when no non-copy route is available.
    pub fn check_transfer(&self, to: &DeviceContext) -> Result<Placement> {
        check_transfer(self, to)
    }

    pub(crate) fn session(&self) -> &Arc<SessionResources> {
        &self.inner.session
    }

    pub(crate) fn tracker(&self) -> Arc<SubmissionTracker> {
        self.inner.session.tracker().clone()
    }

    /// Test support: how many owners lease this context (crate unit tests
    /// assert that buffers and executables keep their lease alive).
    #[cfg(test)]
    pub(crate) fn lease_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }
}

impl fmt::Debug for DeviceContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceContext")
            .field("id", &self.inner.id)
            .field("device", &self.inner.device)
            .field("backend", &self.inner.backend)
            .field("session", &self.inner.session.id())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Weak;

    use evoxc_core::{DType, ValueType};

    use crate::session::{Session, SessionConfig};

    fn session_on(backend: &str, device: &str) -> Session {
        let config = SessionConfig::new(BackendTag::parse(backend).expect("valid backend tag"))
            .with_device(device.parse().expect("valid device"));
        Session::open(config).expect("session opens")
    }

    #[test]
    fn parses_and_displays_devices() {
        let rocm: Device = "rocm:1".parse().expect("valid device");
        assert_eq!(rocm.to_string(), "rocm:1");
        assert_eq!(rocm.kind().as_str(), "rocm");
        assert_eq!(rocm.index(), 1);
        assert!(!rocm.is_cpu());

        let bare: Device = "cuda".parse().expect("valid device");
        assert_eq!(bare, Device::new("cuda", 0).expect("valid device"));
        assert_eq!(Device::host(), Device::new("cpu", 0).expect("valid device"));
        assert_eq!(Device::host().to_string(), "cpu:0");
    }

    #[test]
    fn hip_is_the_rocm_alias_only_for_physical_identity() {
        let hip: Device = "hip:0".parse().expect("valid device");
        let rocm: Device = "rocm:0".parse().expect("valid device");
        assert_eq!(hip.kind().as_str(), "hip", "hip is preserved as spelled");
        assert_eq!(hip.kind().canonical(), "rocm");
        assert_ne!(hip, rocm, "Device equality stays spelling-sensitive");
        assert!(hip.same_physical_device(&rocm));
        assert!(!hip.same_physical_device(&Device::new("hip", 1).expect("valid device")));
    }

    #[test]
    fn rejects_unknown_kinds_and_bad_indices() {
        let unknown = Device::new("tpu", 0).expect_err("tpu is not a device kind");
        assert_eq!(unknown.kind_str(), "DeviceError");
        assert!(unknown.message.contains("tpu"));
        assert!(unknown.message.contains("cpu"));

        let cpu_index = Device::new("cpu", 1).expect_err("cpu has one device");
        assert_eq!(cpu_index.kind_str(), "DeviceError");
        assert!(cpu_index.message.contains("cpu:0"));

        let overflow = Device::new("rocm", MAX_DEVICE_INDEX).expect_err("out of range");
        assert!(overflow.message.contains("out of range"));

        let bad = "rocm:not-a-number".parse::<Device>().expect_err("bad index");
        assert_eq!(bad.kind_str(), "DeviceError");
        assert!("rocm:1:1".parse::<Device>().is_err(), "a second colon is invalid");
        assert!("".parse::<Device>().is_err());
    }

    #[test]
    fn context_identity_is_allocation_identity() {
        let session = session_on("numpy", "cpu:0");
        let first = session.device_context(&Device::host()).expect("served");
        let second = session.device_context(&Device::host()).expect("served");
        assert_eq!(first.id(), second.id());
        assert!(first.same_context(&second));
        assert!(first.compatible_with(&second));
        assert_eq!(first.session_id(), session.id());
        assert_eq!(first.device(), &Device::host());
        assert!(first.is_cpu());
    }

    #[test]
    fn contexts_of_two_sessions_are_distinct() {
        let a = session_on("iree", "rocm:0");
        let b = session_on("iree", "rocm:0");
        let ca = a.device_context(&"rocm:0".parse().expect("valid")).expect("served");
        let cb = b.device_context(&"rocm:0".parse().expect("valid")).expect("served");
        assert!(!ca.same_context(&cb));
        assert!(!ca.compatible_with(&cb));
        assert_ne!(ca.id(), cb.id());
        assert_ne!(ca.session_id(), cb.session_id());
    }

    #[test]
    fn dropping_the_session_handle_keeps_a_leased_context_alive() {
        let session = session_on("numpy", "cpu:0");
        let context = session.device_context(&Device::host()).expect("served");
        let strong_count: Weak<SessionResources> = session.downgrade();
        drop(session);
        assert!(
            strong_count.upgrade().is_some(),
            "a context lease keeps the session's resources alive"
        );
        drop(context);
        assert!(
            strong_count.upgrade().is_none(),
            "releasing the last lease releases the session resources"
        );
    }

    #[test]
    fn context_lease_count_tracks_buffers() {
        let session = session_on("numpy", "cpu:0");
        let context = session.device_context(&Device::host()).expect("served");
        let baseline = context.lease_count();
        let ty = ValueType::of(DType::Float32, [evoxc_core::Dim::int(2)]);
        let buffer = crate::buffer::Buffer::host(ty, vec![0u8; 8], context.clone()).expect("host bytes fit");
        assert_eq!(context.lease_count(), baseline + 1, "a buffer leases its context");
        drop(buffer);
        assert_eq!(context.lease_count(), baseline);
    }
}
