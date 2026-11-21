//! Sessions: the resource owner behind device contexts.
//!
//! A session owns one device context per device it serves and a session-scoped
//! submission tracker (the scope of [`Session::synchronize`]).  The public
//! `Session` handle is a thin lease over [`SessionResources`]: cloning it
//! shares the same resources, and *dropping* it does not unload anything a
//! context, buffer or executable still leases — those objects hold an
//! `Arc<SessionResources>` of their own (proposal §12: "A session dropping its
//! public handle cannot unload resources still leased by those objects").
//!
//! Sessions never invent a backend enumeration: a session carries an opaque,
//! validated [`BackendTag`] string (the Python adapter name) and a list of
//! physical devices.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use evoxc_core::{Error, Result};

use crate::device::{Device, DeviceContext};
use crate::submission::{SubmissionTracker, DEFAULT_COMPLETION_RETENTION, MAX_COMPLETION_RETENTION};

/// Longest accepted session label.
pub const MAX_LABEL_LEN: usize = 128;

static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);

/// Process-unique session identifier.
///
/// Stable for the whole lifetime of a session's resources, including after the
/// public handle is dropped, so diagnostics can name a session that still has
/// live leases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SessionId(u64);

impl SessionId {
    #[inline]
    pub fn value(self) -> u64 {
        self.0
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// An opaque, validated backend identifier (`numpy`, `iree`, `xla`, `tvm`, …).
///
/// The runtime does not enumerate backends: the tag names whichever adapter
/// loaded the program.  Validation only rejects spellings that cannot be a
/// backend name, so two contexts are comparable with `==` and a diagnostic can
/// quote them.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BackendTag(String);

impl BackendTag {
    /// Longest accepted backend tag.
    pub const MAX_LEN: usize = 32;

    /// Parse a backend tag: non-empty, at most [`BackendTag::MAX_LEN`] bytes,
    /// lowercase ASCII letters, digits, `_`, `.` or `-`.
    pub fn parse(tag: &str) -> Result<BackendTag> {
        let valid = !tag.is_empty()
            && tag.len() <= BackendTag::MAX_LEN
            && tag
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'_' | b'.' | b'-'));
        if valid {
            Ok(BackendTag(tag.to_owned()))
        } else {
            Err(Error::device(format!(
                "invalid backend tag {tag:?}: expected 1..={} lowercase letters, digits, '_', '.' or '-' \
                 (the tag is the adapter name, e.g. 'numpy', 'iree', 'xla', 'tvm')",
                BackendTag::MAX_LEN
            )))
        }
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for BackendTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Static configuration of a session.
///
/// A plain data value: [`Session::open`] is the single validation gate (device
/// list, label, retention), so building a config never fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionConfig {
    backend: BackendTag,
    devices: Vec<Device>,
    label: Option<String>,
    completion_retention: usize,
}

impl SessionConfig {
    /// A session config for one backend, serving the host device (`cpu:0`)
    /// unless [`SessionConfig::with_devices`] replaces the list.
    pub fn new(backend: BackendTag) -> SessionConfig {
        SessionConfig {
            backend,
            devices: Vec::new(),
            label: None,
            completion_retention: DEFAULT_COMPLETION_RETENTION,
        }
    }

    /// Append a device to the session's device list.
    pub fn with_device(mut self, device: Device) -> SessionConfig {
        self.devices.push(device);
        self
    }

    /// Replace the device list (`open` resolves an empty list to `[cpu:0]`).
    pub fn with_devices(mut self, devices: impl IntoIterator<Item = Device>) -> SessionConfig {
        self.devices = devices.into_iter().collect();
        self
    }

    /// A human-readable label carried by the session (diagnostics only).
    pub fn with_label(mut self, label: impl Into<String>) -> SessionConfig {
        self.label = Some(label.into());
        self
    }

    /// How many settled submissions the session tracker retains (`1..=4096`).
    pub fn with_completion_retention(mut self, capacity: usize) -> SessionConfig {
        self.completion_retention = capacity;
        self
    }

    pub fn backend(&self) -> &BackendTag {
        &self.backend
    }

    pub fn devices(&self) -> &[Device] {
        &self.devices
    }

    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    pub fn completion_retention(&self) -> usize {
        self.completion_retention
    }
}

/// The resources a session owns: its id, backend, tracker and closed flag.
///
/// Leased by every [`DeviceContext`]; this is what keeps a session alive after
/// the public [`Session`] handle is dropped.
pub(crate) struct SessionResources {
    id: SessionId,
    backend: BackendTag,
    tracker: Arc<SubmissionTracker>,
    closed: Mutex<bool>,
}

impl SessionResources {
    fn new(id: SessionId, backend: BackendTag, retention: usize) -> Result<Arc<SessionResources>> {
        let tracker = SubmissionTracker::new(retention)?;
        Ok(Arc::new(SessionResources {
            id,
            backend,
            tracker: Arc::new(tracker),
            closed: Mutex::new(false),
        }))
    }

    pub(crate) fn id(&self) -> SessionId {
        self.id
    }

    pub(crate) fn backend(&self) -> &BackendTag {
        &self.backend
    }

    pub(crate) fn tracker(&self) -> &Arc<SubmissionTracker> {
        &self.tracker
    }

    fn closed_guard(&self) -> MutexGuard<'_, bool> {
        crate::lock(&self.closed)
    }

    /// True once [`Session::close`] succeeded.
    pub(crate) fn is_closed(&self) -> bool {
        *self.closed_guard()
    }

    /// Fail explicitly when the session has been closed.
    pub(crate) fn ensure_open(&self) -> Result<()> {
        if *self.closed_guard() {
            return Err(Error::device(format!(
                "session {} (backend '{}') is closed; open a new session instead of reusing its executables",
                self.id, self.backend
            )));
        }
        Ok(())
    }
}

/// A handle to a session: the owner of device contexts and a submission scope.
#[derive(Clone)]
pub struct Session {
    resources: Arc<SessionResources>,
    config: SessionConfig,
    contexts: Vec<DeviceContext>,
}

impl Session {
    /// Open a session, creating one device context per served device.
    ///
    /// Validates the device list (known kinds, in-range indices, no duplicate
    /// physical device), the label and the completion retention.  An empty
    /// device list resolves to `[cpu:0]`, the always-available host device.
    pub fn open(config: SessionConfig) -> Result<Session> {
        let mut devices = if config.devices.is_empty() {
            vec![Device::host()]
        } else {
            config.devices.clone()
        };
        validate_label(config.label.as_deref())?;
        validate_retention(config.completion_retention)?;
        validate_devices(&devices)?;
        // Keep the resolved list in the config the session reports.
        devices.shrink_to_fit();
        let resolved = SessionConfig { devices: devices.clone(), ..config };

        let id = SessionId(NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed));
        let resources = SessionResources::new(id, resolved.backend.clone(), resolved.completion_retention)?;
        let contexts = devices
            .into_iter()
            .map(|device| DeviceContext::new(device, resolved.backend.clone(), resources.clone()))
            .collect();
        Ok(Session { resources, config: resolved, contexts })
    }

    pub fn id(&self) -> SessionId {
        self.resources.id()
    }

    pub fn backend(&self) -> &BackendTag {
        self.resources.backend()
    }

    /// The resolved config this session was opened with (devices filled in).
    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    pub fn label(&self) -> Option<&str> {
        self.config.label.as_deref()
    }

    /// The device contexts this session owns, in config order.
    pub fn contexts(&self) -> &[DeviceContext] {
        &self.contexts
    }

    /// The served devices, in config order.
    pub fn devices(&self) -> Vec<Device> {
        self.contexts.iter().map(|context| context.device().clone()).collect()
    }

    /// Look up the context the session serves for `device`.
    ///
    /// Lookup is spelling-sensitive (`hip:0` and `rocm:0` are one physical
    /// device but two `Device` values), so a request the session was not
    /// opened for is refused rather than silently reinterpreted.
    pub fn device_context(&self, device: &Device) -> Result<DeviceContext> {
        self.contexts
            .iter()
            .find(|context| context.device() == device)
            .cloned()
            .ok_or_else(|| {
                let served: Vec<String> = self.devices().iter().map(ToString::to_string).collect();
                Error::device(format!(
                    "session {} (backend '{}') does not serve {device}; it serves {:?}",
                    self.id(),
                    self.backend(),
                    served
                ))
            })
    }

    /// True once [`Session::close`] succeeded.
    pub fn is_closed(&self) -> bool {
        self.resources.is_closed()
    }

    /// Request quiescence and mark the session closed.
    ///
    /// Refuses to close while a submission is still pending (wait on its
    /// completion first) and is idempotent once closed.  Buffers and
    /// executables that lease the session keep working for reads and
    /// transfers; new submissions through this session's executables fail with
    /// an explicit `DeviceError`.
    pub fn close(&self) -> Result<()> {
        let mut closed = self.resources.closed_guard();
        if *closed {
            return Ok(());
        }
        let pending = self.resources.tracker().pending_count();
        if pending > 0 {
            return Err(Error::backend(format!(
                "session {} cannot close: {pending} submission(s) are still pending; \
                 wait on their completions (Completion::wait) first",
                self.id()
            )));
        }
        *closed = true;
        Ok(())
    }

    /// Number of submissions the session tracker still holds as pending.
    pub fn pending_submissions(&self) -> usize {
        self.resources.tracker().pending_count()
    }

    /// Wait for the submissions issued through this session before the call.
    ///
    /// The scope is a snapshot taken at call time; later submissions are not
    /// covered.  Errors are recorded for explicit observation, so the first
    /// failed submission in the snapshot surfaces here.
    pub fn synchronize(&self) -> Result<()> {
        self.resources.tracker().wait_scope(None)
    }

    /// Test support: a weak handle used to prove leases keep resources alive.
    #[cfg(test)]
    pub(crate) fn downgrade(&self) -> std::sync::Weak<SessionResources> {
        Arc::downgrade(&self.resources)
    }
}

impl fmt::Debug for Session {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Session")
            .field("id", &self.id())
            .field("backend", &self.backend())
            .field("label", &self.label())
            .field("devices", &self.devices())
            .field("closed", &self.is_closed())
            .finish()
    }
}

fn validate_label(label: Option<&str>) -> Result<()> {
    let Some(label) = label else { return Ok(()) };
    if label.is_empty() {
        return Err(Error::device("session label must not be empty (omit it instead)"));
    }
    if label.len() > MAX_LABEL_LEN {
        return Err(Error::device(format!(
            "session label is {} bytes; the maximum is {MAX_LABEL_LEN}",
            label.len()
        )));
    }
    if label.chars().any(char::is_control) {
        return Err(Error::device(
            "session label must not contain control characters (it is quoted in diagnostics)",
        ));
    }
    Ok(())
}

fn validate_retention(capacity: usize) -> Result<()> {
    if capacity == 0 || capacity > MAX_COMPLETION_RETENTION {
        return Err(Error::backend(format!(
            "completion retention must be in 1..={MAX_COMPLETION_RETENTION}; got {capacity}"
        )));
    }
    Ok(())
}

fn validate_devices(devices: &[Device]) -> Result<()> {
    for (position, device) in devices.iter().enumerate() {
        if let Some(duplicate) = devices[..position].iter().find(|other| *other == device) {
            return Err(Error::device(format!(
                "session config lists {duplicate} twice; open one context per device"
            )));
        }
        if let Some(same_device) = devices[..position].iter().find(|other| other.same_physical_device(device)) {
            return Err(Error::device(format!(
                "session config lists {device} and {same_device}: they are the same physical device \
                 spelled two ways; pick one spelling (open one context per physical device)"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceKind;

    fn config(backend: &str) -> SessionConfig {
        SessionConfig::new(BackendTag::parse(backend).expect("valid tag"))
    }

    fn cpu() -> Device {
        Device::host()
    }

    fn rocm(index: u32) -> Device {
        Device::new("rocm", index).expect("valid device")
    }

    #[test]
    fn backend_tags_are_validated_opaque_strings() {
        assert_eq!(BackendTag::parse("numpy").expect("valid").as_str(), "numpy");
        assert_eq!(BackendTag::parse("iree").expect("valid").to_string(), "iree");
        for bad in ["", "IREE", "iree/x", "iree x", "ir ee"] {
            let error = BackendTag::parse(bad).expect_err("invalid tag");
            assert_eq!(error.kind_str(), "DeviceError");
            assert!(error.message.contains("backend tag"));
        }
        let long = "a".repeat(BackendTag::MAX_LEN + 1);
        assert!(BackendTag::parse(&long).is_err());
    }

    #[test]
    fn open_resolves_the_default_host_device() {
        let session = Session::open(config("numpy")).expect("opens");
        assert_eq!(session.devices(), vec![cpu()]);
        assert_eq!(session.contexts().len(), 1);
        assert_eq!(session.backend().as_str(), "numpy");
        assert_eq!(session.config().devices(), &[cpu()]);
        assert!(!session.is_closed());
        assert!(session.label().is_none());

        let context = session.device_context(&cpu()).expect("served");
        assert!(context.is_cpu());
        assert_eq!(context.session_id(), session.id());
        assert_eq!(context.backend(), session.backend());
    }

    #[test]
    fn open_serves_every_configured_device_in_order() {
        let session = Session::open(
            config("iree")
                .with_device(cpu())
                .with_device(rocm(0))
                .with_device(rocm(1))
                .with_label("rocm-session"),
        )
        .expect("opens");
        assert_eq!(session.devices(), vec![cpu(), rocm(0), rocm(1)]);
        assert_eq!(session.label(), Some("rocm-session"));
        let context = session.device_context(&rocm(1)).expect("served");
        assert_eq!(context.device(), &rocm(1));
        assert_eq!(context.device().kind(), &DeviceKind::parse("rocm").expect("valid"));
    }

    #[test]
    fn open_rejects_duplicate_and_aliased_devices() {
        let duplicate = Session::open(config("iree").with_device(cpu()).with_device(cpu()))
            .expect_err("duplicate device");
        assert_eq!(duplicate.kind_str(), "DeviceError");
        assert!(duplicate.message.contains("twice"));

        let hip = Device::new("hip", 0).expect("valid device");
        let aliased = Session::open(config("iree").with_device(rocm(0)).with_device(hip))
            .expect_err("aliased device");
        assert_eq!(aliased.kind_str(), "DeviceError");
        assert!(aliased.message.contains("same physical device"));
    }

    #[test]
    fn open_rejects_config_errors() {
        let empty_label = Session::open(config("numpy").with_label("")).expect_err("empty label");
        assert_eq!(empty_label.kind_str(), "DeviceError");

        let long_label = Session::open(config("numpy").with_label("x".repeat(MAX_LABEL_LEN + 1)))
            .expect_err("long label");
        assert!(long_label.message.contains("maximum"));

        let control = Session::open(config("numpy").with_label("a\nb")).expect_err("control char");
        assert!(control.message.contains("control characters"));

        let zero_retention = Session::open(config("numpy").with_completion_retention(0))
            .expect_err("zero retention");
        assert_eq!(zero_retention.kind_str(), "BackendError");
        assert!(zero_retention.message.contains("1..="));

        let huge = Session::open(config("numpy").with_completion_retention(MAX_COMPLETION_RETENTION + 1))
            .expect_err("retention too large");
        assert!(huge.message.contains("1..="));
    }

    #[test]
    fn device_context_refuses_a_device_the_session_does_not_serve() {
        let session = Session::open(config("numpy")).expect("opens");
        let cuda = Device::new("cuda", 0).expect("valid device");
        let error = session.device_context(&cuda).expect_err("not served");
        assert_eq!(error.kind_str(), "DeviceError");
        assert!(error.message.contains("cuda:0"));
        assert!(error.message.contains("cpu:0"), "the message lists what the session serves");
        assert!(error.message.contains(&session.id().to_string()));
    }

    #[test]
    fn device_context_lookup_is_spelling_sensitive() {
        let session = Session::open(config("iree").with_device(rocm(0))).expect("opens");
        let hip = Device::new("hip", 0).expect("valid device");
        let error = session.device_context(&hip).expect_err("spelling mismatch");
        assert!(error.message.contains("hip:0"));
        assert!(error.message.contains("rocm:0"));
    }

    #[test]
    fn clone_shares_one_identity_and_drop_keeps_resources_alive() {
        let session = Session::open(config("numpy")).expect("opens");
        let clone = session.clone();
        assert_eq!(clone.id(), session.id());
        assert_eq!(clone.devices(), session.devices());
        let context = session.device_context(&cpu()).expect("served");
        let weak = session.downgrade();

        drop(session);
        assert!(weak.upgrade().is_some(), "the clone and context lease the session");
        assert_eq!(clone.id(), context.session_id());
        drop(clone);
        assert!(weak.upgrade().is_some(), "the context still leases the session");
        drop(context);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn close_is_idempotent_and_blocks_new_contexts_only_after_it_succeeds() {
        let session = Session::open(config("numpy")).expect("opens");
        let context = session.device_context(&cpu()).expect("served");
        session.close().expect("closes");
        assert!(session.is_closed());
        session.close().expect("closing twice is fine");
        // Reads of leased resources stay valid; only new work is refused.
        assert_eq!(context.device(), &cpu());
        assert_eq!(context.session().id(), session.id());
        assert!(context.session().ensure_open().is_err());
    }

    #[test]
    fn synchronize_with_no_submissions_is_a_noop() {
        let session = Session::open(config("numpy")).expect("opens");
        session.synchronize().expect("nothing to wait for");
        assert_eq!(session.pending_submissions(), 0);
    }
}
