//! `evoxc-runtime`: the native runtime skeleton for EvoX Compiler.
//!
//! # What this crate models
//!
//! The ownership graph the Python runtime contract requires, in Rust, without
//! any device driver:
//!
//! ```text
//! Session          resource owner: id + backend tag + device contexts + tracker
//!   └ DeviceContext  one device inside one session (allocator/queue/plugin lease)
//!       └ Buffer     storage + ValueType + context lease + readiness
//! Executable       loaded-program metadata (flat ABI) + context lease
//! Submission       retained inputs + produced outputs + one outcome
//! Completion       observation handle for that outcome
//! ```
//!
//! A [`Session`] handle is a lease over [`session::SessionResources`]: cloning
//! it shares the same resources, and dropping it never unloads anything a
//! context, buffer or executable still leases (those hold an `Arc` of their
//! own).  That is the skeleton's version of the proposal §12 rule "A session
//! dropping its public handle cannot unload resources still leased by those
//! objects".
//!
//! # Invariants
//!
//! * **One numerical authority, one runtime identity model.**  The crate
//!   depends on `evoxc-core` only — never on `evoxc-compiler`, never on PyO3 —
//!   and device kinds stay a validated string set (`cpu`, `cuda`, `rocm`,
//!   `hip`, `vulkan`) rather than a second enumeration.
//! * **No implicit movement.**  [`Buffer::host`] refuses a non-cpu context, a
//!   device buffer is never readable as host bytes, and every non-trivial
//!   transfer is classified up front by [`check_transfer`] as
//!   [`Placement::SameContext`], [`Placement::CrossContextImport`],
//!   [`Placement::HostUpload`] or [`Placement::HostReadback`].
//! * **No implicit host copies.**  [`Executable::submit`] returns output
//!   handles on the executing device; nothing is downloaded for the caller.
//! * **Readiness is explicit.**  A submission's outputs are
//!   [`Readiness::Pending`] on a dependency token until it settles; reading a
//!   pending buffer is an explicit error, never invalid data.
//! * **Retention is scoped and bounded.**  A submission retains its inputs
//!   until it settles and releases them afterwards; the session tracker keeps a
//!   bounded window of settled submissions (never evicting pending work).
//! * **User errors are values.**  Every fallible path returns
//!   `Result<_, evoxc_core::Error>`; there is no `unsafe`, no panic in `Drop`,
//!   and no `unwrap`/`expect` on user input.
//!
//! # What it is not
//!
//! There are no device drivers, no memory allocators, no queues and no
//! kernels: buffers hold host bytes or an opaque backend handle, and a
//! submission is settled by the side that created it
//! (`Submission::complete_success` / `complete_failure`, also reachable through
//! the returned [`Completion`]).  `synchronize` therefore reports an explicit
//! `BackendError` for work that is still pending instead of blocking or
//! pretending.  The Python `Tensor`/`Executable` contract remains authoritative;
//! this crate is the native model it will be ported onto.

use std::sync::{Mutex, MutexGuard};

pub mod buffer;
pub mod device;
pub mod executable;
pub mod placement;
pub mod session;
pub mod submission;

pub use buffer::{
    Buffer, BufferId, BufferStorage, DependencyToken, DeviceHandle, Readiness,
};
pub use device::{Device, DeviceContext, DeviceKind, KNOWN_DEVICE_KINDS, MAX_DEVICE_INDEX};
pub use evoxc_core::{Error, ErrorKind, Result, ValueType};
pub use executable::{Executable, ExecutableAbi, ExecutableId, ProgramMetadata};
pub use placement::{check_transfer, Placement};
pub use session::{BackendTag, Session, SessionConfig, SessionId};
pub use submission::{
    Completion, Status, Submission, SubmissionId, SubmissionTracker, DEFAULT_COMPLETION_RETENTION,
    MAX_COMPLETION_RETENTION,
};

/// Crate version (`version()` in the Python binding reports the workspace one).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Lock a runtime mutex, recovering from poisoning instead of panicking.
///
/// A poisoned lock means another thread panicked while holding it; the runtime
/// model's state is still readable, and `Drop`/cleanup must never panic, so the
/// guard is recovered rather than unwrapped.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}
