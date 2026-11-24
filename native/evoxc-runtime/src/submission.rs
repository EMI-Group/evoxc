//! Submissions, completions and bounded completion tracking.
//!
//! A [`Submission`] is one enqueued execution: it **retains its input buffers**
//! (so a caller may drop its tensor handles immediately), produces output
//! buffer handles that are `Pending(dependency token)` until the submission
//! settles, and reports its outcome through a [`Completion`].
//!
//! Readiness is the whole point of the type: an output buffer that is still
//! pending cannot be read (that is an explicit error, never invalid data), and
//! a failure leaves the outputs pending forever while the failure itself is
//! observable through [`Completion::wait`] / [`Completion::poll`].
//!
//! [`SubmissionTracker`] is the session-scoped, **bounded** bookkeeping behind
//! `synchronize`: it retains the newest settled submissions up to a capacity
//! and never evicts a pending one, so synchronization cannot require an
//! ever-growing historical list (proposal §12).

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use evoxc_core::{Error, Result};

use crate::buffer::{Buffer, DependencyToken};
use crate::executable::ExecutableId;

/// Default number of settled submissions a session tracker retains.
pub const DEFAULT_COMPLETION_RETENTION: usize = 64;

/// Upper bound accepted for a tracker capacity.
pub const MAX_COMPLETION_RETENTION: usize = 4096;

static NEXT_SUBMISSION_ID: AtomicU64 = AtomicU64::new(1);

/// Process-unique submission identifier (also the dependency token of the
/// submission's outputs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubmissionId(u64);

impl SubmissionId {
    #[inline]
    pub fn value(self) -> u64 {
        self.0
    }
}

impl fmt::Display for SubmissionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The state of a submitted execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    /// Enqueued; its outputs are not readable yet.
    Pending,
    /// Every output is ready and every input has been released.
    Succeeded,
    /// The submission failed; its outputs stay pending and the error is the
    /// recorded outcome.
    Failed(Error),
}

impl Status {
    pub fn is_pending(&self) -> bool {
        matches!(self, Status::Pending)
    }

    /// True for `Succeeded` and `Failed` (a settled submission).
    pub fn is_settled(&self) -> bool {
        !self.is_pending()
    }

    pub fn is_failed(&self) -> bool {
        matches!(self, Status::Failed(_))
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Status::Pending => f.write_str("pending"),
            Status::Succeeded => f.write_str("succeeded"),
            Status::Failed(error) => write!(f, "failed ({error})"),
        }
    }
}

/// One enqueued execution: retained inputs, produced outputs, one outcome.
///
/// Handles are `Arc`-backed, so a [`Completion`] obtained from
/// [`Submission::completion`] observes the same state.
pub struct Submission {
    inner: Arc<SubmissionInner>,
}

pub(crate) struct SubmissionInner {
    id: SubmissionId,
    executable: ExecutableId,
    inputs: Mutex<Vec<Buffer>>,
    outputs: Vec<Buffer>,
    status: Mutex<Status>,
    token: DependencyToken,
}

impl SubmissionInner {
    fn id(&self) -> SubmissionId {
        self.id
    }

    fn status(&self) -> Status {
        crate::lock(&self.status).clone()
    }
}

impl Submission {
    /// Create a submission, arm its outputs with the dependency token of the
    /// new submission id, and register it with the session tracker.
    ///
    /// Runtime-internal: submissions are created by
    /// [`Executable::submit`](crate::Executable::submit), which has already
    /// validated the ABI and placement.
    pub(crate) fn new(
        executable: ExecutableId,
        inputs: Vec<Buffer>,
        outputs: Vec<Buffer>,
        tracker: &SubmissionTracker,
    ) -> Submission {
        let id = SubmissionId(NEXT_SUBMISSION_ID.fetch_add(1, Ordering::Relaxed));
        let token = DependencyToken::new(id.value());
        for output in &outputs {
            output.arm(token);
        }
        let inner = Arc::new(SubmissionInner {
            id,
            executable,
            inputs: Mutex::new(inputs),
            outputs,
            status: Mutex::new(Status::Pending),
            token,
        });
        tracker.register(Arc::clone(&inner));
        Submission { inner }
    }

    pub fn id(&self) -> SubmissionId {
        self.inner.id
    }

    /// The executable that issued this submission (the `synchronize` scope).
    pub fn executable(&self) -> ExecutableId {
        self.inner.executable
    }

    /// The dependency token arming this submission's outputs.
    pub fn dependency_token(&self) -> DependencyToken {
        self.inner.token
    }

    pub fn status(&self) -> Status {
        self.inner.status()
    }

    pub fn completion(&self) -> Completion {
        Completion { inner: Arc::clone(&self.inner) }
    }

    /// The produced buffer handles (still pending until the submission
    /// settles).
    pub fn outputs(&self) -> &[Buffer] {
        &self.inner.outputs
    }

    /// Input buffers the submission currently retains.
    pub fn retained_inputs(&self) -> Vec<Buffer> {
        crate::lock(&self.inner.inputs).clone()
    }

    /// How many input buffers are still retained.
    pub fn retained_input_count(&self) -> usize {
        crate::lock(&self.inner.inputs).len()
    }

    /// Driver hook: record success, make the outputs ready, release inputs.
    ///
    /// Returns `false` when the submission was already settled (the first
    /// outcome wins; a late success cannot overwrite a recorded failure).
    pub fn complete_success(&self) -> bool {
        self.inner.settle(Status::Succeeded)
    }

    /// Driver hook: record a failure, release inputs, leave the outputs
    /// pending (reading them keeps raising explicitly rather than returning
    /// bytes no driver wrote).
    pub fn complete_failure(&self, error: Error) -> bool {
        self.inner.settle(Status::Failed(error))
    }
}

impl SubmissionInner {
    fn settle(&self, outcome: Status) -> bool {
        let previous = {
            let mut status = crate::lock(&self.status);
            if !status.is_pending() {
                return false;
            }
            std::mem::replace(&mut *status, outcome)
        };
        let succeeded = matches!(previous, Status::Pending) && matches!(self.status(), Status::Succeeded);
        if succeeded {
            for output in &self.outputs {
                output.mark_ready(self.token);
            }
        }
        // Submitted work no longer reads the inputs once the submission is
        // settled, so retention ends here — on both outcomes.
        crate::lock(&self.inputs).clear();
        true
    }
}

impl fmt::Debug for Submission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Submission")
            .field("id", &self.inner.id)
            .field("executable", &self.inner.executable)
            .field("status", &self.inner.status())
            .field("retained_inputs", &self.retained_input_count())
            .field("outputs", &self.inner.outputs.len())
            .finish()
    }
}

/// An observation handle for one submission's outcome.
///
/// `poll` never blocks; `wait` reports the submission-specific status, or an
/// explicit `BackendError` when the submission is still pending (this skeleton
/// has no execution driver, so a submission is settled by the side that
/// created it).
#[derive(Clone)]
pub struct Completion {
    inner: Arc<SubmissionInner>,
}

impl Completion {
    pub fn id(&self) -> SubmissionId {
        self.inner.id
    }

    pub fn executable(&self) -> ExecutableId {
        self.inner.executable
    }

    /// Non-blocking status snapshot.
    pub fn poll(&self) -> Status {
        self.inner.status()
    }

    pub fn status(&self) -> Status {
        self.inner.status()
    }

    pub fn is_done(&self) -> bool {
        self.inner.status().is_settled()
    }

    /// The produced buffer handles (pending until the submission settles).
    pub fn outputs(&self) -> Vec<Buffer> {
        self.inner.outputs.clone()
    }

    /// Report this submission's outcome.
    ///
    /// `Ok(())` for a succeeded submission, the recorded error for a failed
    /// one, and an explicit `BackendError` while it is still pending.
    pub fn wait(&self) -> Result<()> {
        match self.inner.status() {
            Status::Succeeded => Ok(()),
            Status::Failed(error) => Err(error.context(format!("submission {}", self.id()))),
            Status::Pending => Err(Error::backend(format!(
                "submission {} is still pending: this runtime skeleton has no execution driver, so the \
                 creating backend must settle it (Submission::complete_success / complete_failure) before waiting",
                self.id()
            ))),
        }
    }

    /// Driver hook: see [`Submission::complete_success`].
    pub fn complete_success(&self) -> bool {
        self.inner.settle(Status::Succeeded)
    }

    /// Driver hook: see [`Submission::complete_failure`].
    pub fn complete_failure(&self, error: Error) -> bool {
        self.inner.settle(Status::Failed(error))
    }
}

impl fmt::Debug for Completion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Completion")
            .field("id", &self.inner.id)
            .field("status", &self.inner.status())
            .finish()
    }
}

/// Bounded, session-scoped bookkeeping of submissions.
///
/// Retention is newest-first up to `capacity` settled entries; pending entries
/// are never evicted (losing one would lose a failure and break
/// `synchronize`).  Evicted settled submissions stay observable through any
/// [`Completion`] handle the caller still holds.
pub struct SubmissionTracker {
    capacity: usize,
    entries: Mutex<VecDeque<Arc<SubmissionInner>>>,
    pruned: AtomicU64,
}

impl SubmissionTracker {
    /// Build a tracker retaining at most `capacity` settled submissions.
    pub fn new(capacity: usize) -> Result<SubmissionTracker> {
        if capacity == 0 || capacity > MAX_COMPLETION_RETENTION {
            return Err(Error::backend(format!(
                "completion retention must be in 1..={MAX_COMPLETION_RETENTION}; got {capacity}"
            )));
        }
        Ok(SubmissionTracker {
            capacity,
            entries: Mutex::new(VecDeque::new()),
            pruned: AtomicU64::new(0),
        })
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Entries currently retained (pending and settled).
    pub fn len(&self) -> usize {
        crate::lock(&self.entries).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Submissions currently pending inside the retained window.
    pub fn pending_count(&self) -> usize {
        let entries = crate::lock(&self.entries);
        entries.iter().filter(|entry| entry.status().is_pending()).count()
    }

    /// Pending submissions inside the retained window, optionally scoped to one
    /// executable.
    pub(crate) fn pending_scope(&self, executable: Option<ExecutableId>) -> usize {
        let entries = crate::lock(&self.entries);
        entries
            .iter()
            .filter(|entry| {
                executable.is_none_or(|id| entry.executable == id) && entry.status().is_pending()
            })
            .count()
    }

    /// How many settled entries have been evicted by the retention bound.
    pub fn pruned(&self) -> u64 {
        self.pruned.load(Ordering::Relaxed)
    }

    pub(crate) fn register(&self, submission: Arc<SubmissionInner>) {
        let mut entries = crate::lock(&self.entries);
        entries.push_back(submission);
        let mut pruned = 0u64;
        while entries.len() > self.capacity {
            let oldest_settled = entries.front().map(|entry| entry.status().is_settled()).unwrap_or(false);
            if !oldest_settled {
                // Pending work is never evicted: its failure must stay
                // observable and `synchronize` must still see it.
                break;
            }
            entries.pop_front();
            pruned += 1;
        }
        if pruned > 0 {
            self.pruned.fetch_add(pruned, Ordering::Relaxed);
        }
    }

    fn snapshot(&self) -> Vec<Arc<SubmissionInner>> {
        crate::lock(&self.entries).iter().cloned().collect()
    }

    /// Wait for a snapshot of the retained submissions; `None` means the whole
    /// session, `Some(id)` only the submissions of one executable.
    ///
    /// The scope is a snapshot taken at call time.  A pending submission
    /// cannot be waited for without a driver, so it is an explicit
    /// `BackendError`; the first failed submission in issue order surfaces
    /// with its own error family.
    pub(crate) fn wait_scope(&self, executable: Option<ExecutableId>) -> Result<()> {
        for entry in self.snapshot() {
            if let Some(id) = executable {
                if entry.executable != id {
                    continue;
                }
            }
            match entry.status() {
                Status::Succeeded => {}
                Status::Failed(error) => {
                    return Err(error.context(format!("submission {}", entry.id())));
                }
                Status::Pending => {
                    return Err(Error::backend(format!(
                        "submission {} is still pending: this runtime skeleton has no execution driver to wait \
                         on; settle it (Submission::complete_success / complete_failure) first",
                        entry.id()
                    )));
                }
            }
        }
        Ok(())
    }
}

impl fmt::Debug for SubmissionTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubmissionTracker")
            .field("capacity", &self.capacity)
            .field("retained", &self.len())
            .field("pending", &self.pending_count())
            .field("pruned", &self.pruned())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use evoxc_core::{DType, Dim, ValueType};

    use crate::buffer::Buffer;
    use crate::device::{Device, DeviceContext};
    use crate::session::{BackendTag, Session, SessionConfig};

    fn context(backend: &str, device: &str) -> DeviceContext {
        let config = SessionConfig::new(BackendTag::parse(backend).expect("valid tag"))
            .with_device(device.parse().expect("valid device"));
        let session = Session::open(config).expect("session opens");
        session
            .device_context(&device.parse().expect("valid device"))
            .expect("served device")
    }

    fn host_buffer(context: &DeviceContext, elements: usize) -> Buffer {
        let ty = ValueType::of(DType::Float32, [Dim::int(elements as i64)]);
        Buffer::host(ty, vec![7u8; elements * 4], context.clone()).expect("host buffer")
    }

    fn device_buffer(context: &DeviceContext, elements: usize) -> Buffer {
        let ty = ValueType::of(DType::Float32, [Dim::int(elements as i64)]);
        Buffer::alloc(ty, context.clone()).expect("device buffer")
    }

    fn submission(tracker: &SubmissionTracker, outputs: Vec<Buffer>) -> Submission {
        Submission::new(ExecutableId::allocate(), Vec::new(), outputs, tracker)
    }

    #[test]
    fn submission_retains_inputs_until_completion() {
        let tracker = SubmissionTracker::new(8).expect("tracker");
        let context = context("numpy", "cpu:0");
        let input = host_buffer(&context, 4);
        let weak = input.downgrade();
        assert_eq!(input.lease_count(), 1);

        let submission = Submission::new(ExecutableId::allocate(), vec![input.clone()], Vec::new(), &tracker);
        assert_eq!(submission.retained_input_count(), 1, "the submission retains its input");
        assert_eq!(input.lease_count(), 2);

        drop(input);
        assert!(weak.upgrade().is_some(), "dropping the caller's handle keeps the input alive");

        assert!(submission.complete_success());
        assert_eq!(submission.retained_input_count(), 0, "completion releases the retention");
        assert!(
            weak.upgrade().is_none(),
            "once settled, nothing keeps the input buffer alive"
        );
        assert_eq!(submission.status(), Status::Succeeded);
    }

    #[test]
    fn failure_also_releases_inputs_and_keeps_outputs_pending() {
        let tracker = SubmissionTracker::new(8).expect("tracker");
        let context = context("iree", "cpu:0");
        let input = host_buffer(&context, 2);
        let output = device_buffer(&context, 3);
        let submission =
            Submission::new(ExecutableId::allocate(), vec![input], vec![output.clone()], &tracker);

        assert!(output.readiness().is_pending());
        assert!(submission.complete_failure(Error::backend("kernel aborted")));
        assert_eq!(submission.retained_input_count(), 0);
        assert!(
            output.readiness().is_pending(),
            "a failed submission never makes its outputs readable"
        );
        match submission.status() {
            Status::Failed(error) => {
                assert_eq!(error.kind_str(), "BackendError");
                assert!(error.message.contains("kernel aborted"));
            }
            other => panic!("expected a failed status, got {other}"),
        }
        let error = submission.completion().wait().expect_err("failure is reported");
        assert_eq!(error.kind_str(), "BackendError");
        assert!(error.message.contains(&format!("submission {}", submission.id())));
    }

    #[test]
    fn completion_makes_outputs_ready_and_readable() {
        let tracker = SubmissionTracker::new(8).expect("tracker");
        let context = context("numpy", "cpu:0");
        let output = host_buffer(&context, 3);
        let submission = submission(&tracker, vec![output.clone()]);

        let error = output.host_bytes().expect_err("pending data is not readable");
        assert_eq!(error.kind_str(), "EVOXCError");
        assert!(error.message.contains("pending"));
        assert!(error.message.contains(&submission.id().value().to_string()));

        assert!(submission.complete_success());
        assert!(output.is_ready());
        assert_eq!(output.host_bytes().expect("ready host bytes").len(), 12);
        submission.completion().wait().expect("succeeded");
    }

    #[test]
    fn the_first_outcome_wins() {
        let tracker = SubmissionTracker::new(8).expect("tracker");
        let context = context("numpy", "cpu:0");
        let submission = submission(&tracker, vec![host_buffer(&context, 1)]);

        assert!(submission.complete_success());
        assert!(!submission.complete_failure(Error::backend("late failure")));
        assert!(!submission.complete_success());
        assert_eq!(submission.status(), Status::Succeeded);
        assert_eq!(submission.completion().poll(), Status::Succeeded);
        assert!(submission.completion().is_done());
    }

    #[test]
    fn waiting_on_a_pending_submission_is_explicit() {
        let tracker = SubmissionTracker::new(8).expect("tracker");
        let context = context("numpy", "cpu:0");
        let submission = submission(&tracker, vec![host_buffer(&context, 1)]);

        assert_eq!(submission.completion().poll(), Status::Pending);
        assert!(!submission.completion().is_done());
        let error = submission.completion().wait().expect_err("cannot wait without a driver");
        assert_eq!(error.kind_str(), "BackendError");
        assert!(error.message.contains("still pending"));
        assert!(error.message.contains("no execution driver"));
    }

    #[test]
    fn retention_is_bounded_and_keeps_the_newest() {
        let tracker = SubmissionTracker::new(4).expect("tracker");
        let context = context("numpy", "cpu:0");
        let executable = ExecutableId::allocate();
        let mut completions = Vec::new();
        for _ in 0..10 {
            let submission =
                Submission::new(executable, Vec::new(), vec![host_buffer(&context, 1)], &tracker);
            assert!(submission.complete_success());
            completions.push((submission.id(), submission.completion()));
        }

        assert_eq!(tracker.len(), 4, "settled retention does not grow without bound");
        assert_eq!(tracker.pruned(), 6);
        assert_eq!(tracker.pending_count(), 0);
        // Evicted submissions stay observable through the handles the caller holds.
        assert!(completions[0].1.is_done());
        assert_eq!(completions[0].1.poll(), Status::Succeeded);
        // And `synchronize` still succeeds over the retained window.
        tracker.wait_scope(None).expect("all settled");
    }

    #[test]
    fn pending_submissions_are_never_evicted() {
        let tracker = SubmissionTracker::new(2).expect("tracker");
        let context = context("iree", "cpu:0");
        let submissions: Vec<Submission> = (0..5)
            .map(|_| submission(&tracker, vec![device_buffer(&context, 1)]))
            .collect();

        assert_eq!(tracker.len(), 5, "pending work is retained regardless of capacity");
        assert_eq!(tracker.pruned(), 0);
        assert_eq!(tracker.pending_count(), 5);

        // Settling the oldest ones lets the tracker prune down to capacity.
        for submission in &submissions {
            assert!(submission.complete_success());
        }
        tracker.register(Arc::clone(&submissions[0].inner));
        assert!(tracker.len() <= 2);
        assert_eq!(tracker.pending_count(), 0);
    }

    #[test]
    fn tracker_rejects_invalid_capacities() {
        for bad in [0, MAX_COMPLETION_RETENTION + 1] {
            let error = SubmissionTracker::new(bad).expect_err("invalid capacity");
            assert_eq!(error.kind_str(), "BackendError");
            assert!(error.message.contains("1..="));
        }
    }

    #[test]
    fn scope_filters_by_executable_and_surfaces_failures() {
        let tracker = SubmissionTracker::new(16).expect("tracker");
        let a = ExecutableId::allocate();
        let b = ExecutableId::allocate();
        let first = Submission::new(a, Vec::new(), Vec::new(), &tracker);
        let second = Submission::new(b, Vec::new(), Vec::new(), &tracker);
        assert!(first.complete_success());

        tracker.wait_scope(Some(a)).expect("executable a is quiescent");
        let error = tracker.wait_scope(Some(b)).expect_err("b is pending");
        assert_eq!(error.kind_str(), "BackendError");
        assert!(error.message.contains(&second.id().to_string()));

        assert!(second.complete_failure(Error::device("device lost")));
        let error = tracker.wait_scope(Some(b)).expect_err("b failed");
        assert_eq!(error.kind_str(), "DeviceError");
        assert!(error.message.contains("device lost"));
        let error = tracker.wait_scope(None).expect_err("session scope sees b");
        assert_eq!(error.kind_str(), "DeviceError");
        // The failure stays observable through the submission's own handle too.
        assert!(second.completion().wait().is_err());
    }

    #[test]
    fn arming_replaces_the_dependency_token_of_fresh_outputs() {
        let tracker = SubmissionTracker::new(8).expect("tracker");
        let context = context("iree", "cpu:0");
        let output = device_buffer(&context, 2);
        assert!(output.is_ready(), "a freshly allocated buffer is ready");
        let submission = submission(&tracker, vec![output.clone()]);
        assert_eq!(
            output.readiness(),
            crate::buffer::Readiness::Pending(submission.dependency_token())
        );
        assert_eq!(submission.dependency_token().value(), submission.id().value());
        assert!(submission.complete_success());
        assert!(output.is_ready());
    }

    #[test]
    fn a_submission_on_a_rocm_context_produces_device_outputs() {
        let tracker = SubmissionTracker::new(8).expect("tracker");
        let context = context("iree", "rocm:0");
        assert_eq!(context.device().kind().as_str(), "rocm");
        let output = device_buffer(&context, 4);
        assert!(!output.is_host_resident());
        assert_eq!(output.device(), &Device::new("rocm", 0).expect("valid device"));
        let submission = submission(&tracker, vec![output.clone()]);
        let error = output
            .host_bytes()
            .expect_err("a device output must not be readable as host bytes");
        assert_eq!(error.kind_str(), "EVOXCError");
        assert!(submission.complete_success());
        let error = output
            .host_bytes()
            .expect_err("no implicit device-to-host copy, even once ready");
        assert_eq!(error.kind_str(), "DeviceError");
    }
}
