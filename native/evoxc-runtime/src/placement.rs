//! Explicit placement and transfer classification.
//!
//! The runtime never moves data on its own.  Ask [`check_transfer`] what an
//! explicit transfer would be, and let the caller decide whether to perform it.
//! The four outcomes mirror the Python contract's rule that "same-context
//! no-op, cross-client import, host upload, and device readback are
//! distinguishable transfer outcomes" (proposal §13).
//!
//! ```text
//! SameContext         no movement: identical context identity
//! CrossContextImport  zero-copy import proven compatible (same physical
//!                     device, same backend tag), still explicit
//! HostUpload          host bytes -> device memory of the target context
//! HostReadback        device memory -> host memory of the target context
//! ```

use evoxc_core::{Error, Result};

use crate::device::DeviceContext;

/// What an explicit transfer from one context into another would involve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Placement {
    /// Both handles are the same context; the transfer is a no-op.
    SameContext,
    /// A different context on the same physical device and backend: the
    /// receiver may import the source's device memory without copying.
    CrossContextImport,
    /// Host bytes entering device memory.
    HostUpload,
    /// Device memory leaving for host memory.
    HostReadback,
}

impl Placement {
    /// False only for [`Placement::SameContext`].
    pub fn moves_data(self) -> bool {
        !matches!(self, Placement::SameContext)
    }

    /// True for the two host-boundary outcomes.
    pub fn is_host_staged(self) -> bool {
        matches!(self, Placement::HostUpload | Placement::HostReadback)
    }

    /// Short stable spelling used in diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            Placement::SameContext => "same-context",
            Placement::CrossContextImport => "cross-context-import",
            Placement::HostUpload => "host-upload",
            Placement::HostReadback => "host-readback",
        }
    }
}

/// Classify an explicit transfer from `from` into `to`.
///
/// Returns an error — never a fallback — when no non-copy route exists: a
/// cross-device import or a cross-backend import must be staged through a cpu
/// context by the caller.
pub fn check_transfer(from: &DeviceContext, to: &DeviceContext) -> Result<Placement> {
    if from.same_context(to) {
        return Ok(Placement::SameContext);
    }

    let target = to.device();
    let source = from.device();

    // Anything landing in host memory is a readback, including a host-to-host
    // copy between two cpu-kind contexts (the destination is host memory).
    if target.is_cpu() {
        return Ok(Placement::HostReadback);
    }
    if source.is_cpu() {
        return Ok(Placement::HostUpload);
    }

    if !source.same_physical_device(target) {
        return Err(Error::device(format!(
            "cannot transfer {source} data (session {}, context {}) into {target} (session {}, context {}): \
             a cross-device import needs the same physical device; stage through a cpu context explicitly \
             (source -> cpu -> {target})",
            from.session_id(),
            from.id(),
            to.session_id(),
            to.id(),
        )));
    }
    if from.backend() != to.backend() {
        return Err(Error::device(format!(
            "cannot import {target} data between backends '{}' (session {}, context {}) and '{}' \
             (session {}, context {}): a cross-context import is only possible inside one backend; \
             stage through a cpu context explicitly",
            from.backend(),
            from.session_id(),
            from.id(),
            to.backend(),
            to.session_id(),
            to.id(),
        )));
    }
    Ok(Placement::CrossContextImport)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::session::{BackendTag, Session, SessionConfig};

    fn session(backend: &str, devices: &[&str]) -> Session {
        let mut config = SessionConfig::new(BackendTag::parse(backend).expect("valid tag"));
        for device in devices {
            config = config.with_device(device.parse().expect("valid device"));
        }
        Session::open(config).expect("session opens")
    }

    #[test]
    fn identical_contexts_need_no_movement() {
        let session = session("iree", &["cpu:0", "rocm:0"]);
        let host = session.device_context(&"cpu:0".parse().expect("valid")).expect("served");
        let host_again = session.device_context(&"cpu:0".parse().expect("valid")).expect("served");
        let placement = check_transfer(&host, &host_again).expect("same context");
        assert_eq!(placement, Placement::SameContext);
        assert!(!placement.moves_data());
        assert!(!placement.is_host_staged());
        assert_eq!(host.check_transfer(&host_again).expect("same"), placement);

        let device = session.device_context(&"rocm:0".parse().expect("valid")).expect("served");
        assert_eq!(check_transfer(&device, &device).expect("same"), Placement::SameContext);
    }

    #[test]
    fn host_and_device_boundaries_are_classified() {
        let session = session("iree", &["cpu:0", "rocm:0"]);
        let host = session.device_context(&"cpu:0".parse().expect("valid")).expect("served");
        let device = session.device_context(&"rocm:0".parse().expect("valid")).expect("served");

        let upload = check_transfer(&host, &device).expect("upload");
        assert_eq!(upload, Placement::HostUpload);
        assert!(upload.moves_data());
        assert!(upload.is_host_staged());
        assert_eq!(upload.as_str(), "host-upload");

        let readback = check_transfer(&device, &host).expect("readback");
        assert_eq!(readback, Placement::HostReadback);
        assert!(readback.moves_data());
        assert!(readback.is_host_staged());
    }

    #[test]
    fn a_host_to_host_copy_between_contexts_is_a_readback() {
        let a = session("numpy", &["cpu:0"]);
        let b = session("numpy", &["cpu:0"]);
        let host_a = a.device_context(&"cpu:0".parse().expect("valid")).expect("served");
        let host_b = b.device_context(&"cpu:0".parse().expect("valid")).expect("served");
        assert_eq!(
            check_transfer(&host_a, &host_b).expect("host copy"),
            Placement::HostReadback,
            "the destination is host memory, so the explicit copy is a readback outcome"
        );
    }

    #[test]
    fn same_device_same_backend_across_sessions_is_an_import() {
        let a = session("iree", &["rocm:0"]);
        let b = session("iree", &["rocm:0"]);
        let from = a.device_context(&"rocm:0".parse().expect("valid")).expect("served");
        let to = b.device_context(&"rocm:0".parse().expect("valid")).expect("served");
        let placement = check_transfer(&from, &to).expect("compatible import");
        assert_eq!(placement, Placement::CrossContextImport);
        assert!(placement.moves_data());
        assert!(!placement.is_host_staged());
        assert_eq!(placement.as_str(), "cross-context-import");
    }

    #[test]
    fn a_different_device_requires_host_staging() {
        let session = session("iree", &["rocm:0", "rocm:1"]);
        let zero = session.device_context(&"rocm:0".parse().expect("valid")).expect("served");
        let one = session.device_context(&"rocm:1".parse().expect("valid")).expect("served");
        let error = check_transfer(&zero, &one).expect_err("no direct cross-device route");
        assert_eq!(error.kind_str(), "DeviceError");
        assert!(error.message.contains("rocm:0"));
        assert!(error.message.contains("rocm:1"));
        assert!(error.message.contains("stage through a cpu context"));
    }

    #[test]
    fn a_different_backend_requires_host_staging() {
        let iree = session("iree", &["rocm:0"]);
        let xla = session("xla", &["rocm:0"]);
        let from = iree.device_context(&"rocm:0".parse().expect("valid")).expect("served");
        let to = xla.device_context(&"rocm:0".parse().expect("valid")).expect("served");
        let error = check_transfer(&from, &to).expect_err("cross-backend import is impossible");
        assert_eq!(error.kind_str(), "DeviceError");
        assert!(error.message.contains("'iree'"));
        assert!(error.message.contains("'xla'"));
    }

    #[test]
    fn hip_and_rocm_spell_the_same_physical_device() {
        let iree = session("iree", &["hip:0"]);
        let xla = session("iree", &["rocm:0"]);
        let from = iree.device_context(&"hip:0".parse().expect("valid")).expect("served");
        let to = xla.device_context(&"rocm:0".parse().expect("valid")).expect("served");
        assert_eq!(
            check_transfer(&from, &to).expect("same physical device, same backend"),
            Placement::CrossContextImport
        );
    }
}
