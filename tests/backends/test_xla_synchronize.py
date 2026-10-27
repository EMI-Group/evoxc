"""xla adapter EXPLICIT device synchronization (``Executable.synchronize``).

Pins the completion-event lifecycle of ``XlaExecutable.synchronize``
(``etl/backends/adapters/xla.py``) and of the driver's event sink
(``etl/backends/adapters/xla_util.py`` — ``execute(..., events=[...])``),
exercised against the TEST-ONLY fake PJRT plugin built from C at test time
(the same fake as ``test_pjrt_ctypes_plugin.py``; zero-filled outputs, ONE
addressable device, no computation):

* a successful DEVICE-RESIDENT ``run`` records the plugin's per-device
  ``PJRT_Event*`` handle and NOTHING synchronizes implicitly — many runs
  accumulate handles, and one ``synchronize()`` AWAITS then DESTROYS every
  one of them in that order and clears the list; a second call performs
  zero plugin FFI (the idle fast path) and is therefore idempotent/reusable;
* an await failure still awaits + destroys every remaining handle and
  re-raises the FIRST error (no handle leaked on the error path); the call
  is retryable afterwards (nothing stays pending);
* a destroy failure is explicit the same way;
* a plugin that returns NO event handle (``0``/NULL) makes ``run`` record a
  reason and ``synchronize`` raise an explicit ``core.BackendError`` ONCE —
  never a silent completion claim — after which synchronization is quiet;
* a closed executable with still-pending handles raises an explicit
  ``core.BackendError`` naming the situation;
* ``close()`` DESTROYS pending handles WITHOUT awaiting and never raises on
  best-effort destroy failures;
* a failed execute destroys a stray handle best-effort and appends nothing;
* the CPU host-staging path requests no event at all (its ``ToHostBuffer``
  already awaits), so its synchronization state stays empty.

SEMANTICS ONLY — the fake performs no computation, so numerical parity is
out of scope (real-plugin parity lives in ``test_adapter_xla.py``).
"""

import numpy as np
import pytest

import etl
from etl.backends.adapters import xla
from etl.backends.adapters import xla_util

from tests.backends import _adapter_utils as u
from tests.backends.test_pjrt_ctypes_plugin import _build_plugin, _compile_fake

NAME = "xla"
CPU = etl.core.Device("cpu", 0)
CUDA0 = etl.core.Device("cuda", 0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _activate(plugin_path, monkeypatch):
    """Point plugin discovery at the fake .so and (re)activate the adapter."""
    monkeypatch.setenv("ETL_PJRT_PLUGIN", str(plugin_path))
    xla.register()  # idempotent; also (re)installs the "cuda" placement provider


def _device_resident_executable(tmp_path, so_name, monkeypatch):
    """A device-resident ``XlaExecutable`` + its (device-placed) inputs.

    Each test builds its OWN plugin path (the ``_load_plugin`` cache and the
    process-global shared client are keyed by path), so the tests are
    isolated from each other and from the sibling fake-plugin suites.
    """
    plug = _build_plugin(tmp_path, so_name)
    _activate(plug, monkeypatch)
    fn, specs = u.matmul_relu_sum()
    artifact = _compile_fake(fn, specs, plug)
    raw = etl.backends.get(NAME).load(artifact, device=CUDA0)
    a, b = u.matmul_relu_sum_args()
    inputs = [etl.core.Tensor(a).to(CUDA0), etl.core.Tensor(b).to(CUDA0)]
    return raw, inputs


class _EventLog:
    """Recording pass-through spies over the plugin's event entry points."""

    def __init__(self):
        self.awaited: list[int] = []
        self.destroyed: list[int] = []
        self.await_errors: dict[int, BaseException] = {}
        self.destroy_errors: dict[int, BaseException] = {}


def _spy_events(monkeypatch):
    """Wrap ``PjrtPlugin.await_event``/``destroy_event`` with a recorder.

    The wrappers still perform the REAL plugin FFI (they call the captured
    original), so event ordering/ownership stays real; injecting an entry
    into ``await_errors``/``destroy_errors`` makes that handle's call raise
    the given exception BEFORE the FFI (the failure-path pins).
    """
    log = _EventLog()
    real_await = xla_util.PjrtPlugin.await_event
    real_destroy = xla_util.PjrtPlugin.destroy_event

    def await_event(self, handle):
        log.awaited.append(handle)
        error = log.await_errors.get(handle)
        if error is not None:
            raise error
        return real_await(self, handle)

    def destroy_event(self, handle):
        log.destroyed.append(handle)
        error = log.destroy_errors.get(handle)
        if error is not None:
            raise error
        return real_destroy(self, handle)

    monkeypatch.setattr(xla_util.PjrtPlugin, "await_event", await_event)
    monkeypatch.setattr(xla_util.PjrtPlugin, "destroy_event", destroy_event)
    return log


# ---------------------------------------------------------------------------
# 1. success: record -> await+destroy in order -> clear -> idle fast path
# ---------------------------------------------------------------------------


def test_run_records_events_and_synchronize_awaits_then_destroys(
    tmp_path, monkeypatch
):
    """One sync awaits + destroys every pending handle IN ORDER, then is idle.

    Also pins that nothing synchronizes implicitly: the handles stay pending
    across successive ``run`` calls (a run never waits for the device).
    """
    raw, inputs = _device_resident_executable(
        tmp_path, "fake_sync_success.so", monkeypatch
    )
    log = _spy_events(monkeypatch)
    assert raw._pending_events == []  # nothing outstanding before any run

    handles = []
    for step in range(3):
        (out,) = raw.run(inputs)
        assert isinstance(out.data, xla.XlaDevicePayload)
        # run() did NOT synchronize: each dispatch's handle is still pending
        # (the recorded reason stays unset — the plugin did emit events).
        assert len(raw._pending_events) == step + 1
        assert raw._sync_unavailable_reason is None
        handles.append(raw._pending_events[-1])
        assert log.awaited == [] and log.destroyed == []
    assert len(set(handles)) == len(handles)  # one distinct handle per run

    # One synchronize waits for ALL of them, destroys each right after its
    # await, keeps the per-run order, and clears the pending list.
    assert raw.synchronize() is None
    assert log.awaited == handles
    assert log.destroyed == handles
    assert raw._pending_events == []

    # Idle fast path: a second call performs ZERO plugin FFI (no await, no
    # destroy) and returns None — reusable/idempotent.
    assert raw.synchronize() is None
    assert log.awaited == handles
    assert log.destroyed == handles


def test_pipeline_executable_synchronize_delegates_to_xla(tmp_path, monkeypatch):
    """``etl.load(...).synchronize()`` reaches the adapter's real sync."""
    raw, inputs = _device_resident_executable(
        tmp_path, "fake_sync_pipeline.so", monkeypatch
    )
    log = _spy_events(monkeypatch)
    fn, specs = u.matmul_relu_sum()
    artifact = _compile_fake(fn, specs, raw._plugin.path)
    exe = etl.load(artifact, device=CUDA0)

    xa, xb = inputs
    out = etl.run(exe, xa, xb)
    assert isinstance(out.data, xla.XlaDevicePayload)
    backend_exe = exe.backend_executable
    assert isinstance(backend_exe, xla.XlaExecutable)
    assert len(backend_exe._pending_events) == 1  # etl.run never syncs

    handle = backend_exe._pending_events[0]
    assert exe.synchronize() is None
    assert log.awaited == [handle] and log.destroyed == [handle]
    assert backend_exe._pending_events == []


# ---------------------------------------------------------------------------
# 2. error paths: first error re-raised AFTER the rest is cleaned up
# ---------------------------------------------------------------------------


def test_await_failure_cleans_up_every_handle_and_reraises_first(
    tmp_path, monkeypatch
):
    """A failing await never leaks: all handles awaited+destroyed, FIRST error."""
    raw, inputs = _device_resident_executable(
        tmp_path, "fake_sync_await_fail.so", monkeypatch
    )
    log = _spy_events(monkeypatch)
    for _ in range(2):
        raw.run(inputs)
    first, second = raw._pending_events
    log.await_errors[first] = RuntimeError("await-boom-first")
    log.await_errors[second] = RuntimeError("await-boom-second")

    with pytest.raises(RuntimeError, match="await-boom-first"):
        raw.synchronize()

    # Every handle was still awaited AND destroyed (no leak on the error
    # path) and the pending list was cleared.
    assert log.awaited == [first, second]
    assert log.destroyed == [first, second]
    assert raw._pending_events == []

    # Retryable: nothing stays pending, so a repeat call is the quiet idle
    # fast path (no further FFI) and never re-raises the consumed error.
    assert raw.synchronize() is None
    assert log.awaited == [first, second]
    assert log.destroyed == [first, second]


def test_destroy_failure_is_explicit(tmp_path, monkeypatch):
    """A destroy failure surfaces explicitly (after the await) and is retryable."""
    raw, inputs = _device_resident_executable(
        tmp_path, "fake_sync_destroy_fail.so", monkeypatch
    )
    log = _spy_events(monkeypatch)
    raw.run(inputs)
    (handle,) = raw._pending_events
    log.destroy_errors[handle] = RuntimeError("destroy-boom")

    with pytest.raises(RuntimeError, match="destroy-boom"):
        raw.synchronize()

    assert log.awaited == [handle]  # awaited first, then the destroy failed
    assert log.destroyed == [handle]
    assert raw._pending_events == []

    assert raw.synchronize() is None  # retry: quiet
    assert log.awaited == [handle]
    assert log.destroyed == [handle]


# ---------------------------------------------------------------------------
# 3. a plugin that violates the event contract (no handle) is never silent
# ---------------------------------------------------------------------------


def test_missing_event_handle_raises_backenderror_once(tmp_path, monkeypatch):
    """``0``/NULL from the plugin -> run is quiet, sync raises ONCE (loud)."""
    raw, inputs = _device_resident_executable(
        tmp_path, "fake_sync_no_event.so", monkeypatch
    )
    log = _spy_events(monkeypatch)
    monkeypatch.setenv("ETL_FAKE_PJRT_NO_DEVICE_EVENT", "1")

    # run itself never raises for the contract violation and records the
    # reason instead; no handle joins the pending list.
    (out,) = raw.run(inputs)
    assert isinstance(out.data, xla.XlaDevicePayload)
    assert raw._pending_events == []
    assert raw._sync_unavailable_reason is not None

    # synchronize refuses to claim completion silently.
    with pytest.raises(etl.BackendError, match="returned no device-complete event"):
        raw.synchronize()
    # ... exactly once: the reason is consumed, so the next call is quiet.
    assert raw.synchronize() is None
    assert log.awaited == [] and log.destroyed == []  # zero FFI on both calls


# ---------------------------------------------------------------------------
# 4. closed executable / close() lifecycle
# ---------------------------------------------------------------------------


def test_synchronize_after_plugin_release_raises_explicitly(tmp_path, monkeypatch):
    """Pending handles + a released plugin cannot be awaited -> BackendError."""
    raw, inputs = _device_resident_executable(
        tmp_path, "fake_sync_released.so", monkeypatch
    )
    log = _spy_events(monkeypatch)
    raw.run(inputs)
    assert len(raw._pending_events) == 1

    # Simulate the plugin being released while handles are still pending
    # (the state ``synchronize`` refuses to paper over): close() clears the
    # list, so a handle can only outlive the plugin when the reference is
    # dropped elsewhere.
    raw._plugin = None
    with pytest.raises(
        etl.BackendError, match="cannot synchronize this closed XlaExecutable"
    ):
        raw.synchronize()
    # The handles are dropped (they can be neither awaited nor destroyed)
    # and the error is not sticky: a repeat call is quiet.
    assert raw._pending_events == []
    assert raw.synchronize() is None
    assert log.awaited == [] and log.destroyed == []


def test_close_destroys_pending_handles_without_awaiting(tmp_path, monkeypatch):
    """``close()`` destroys (never awaits) pending handles and clears the state."""
    raw, inputs = _device_resident_executable(
        tmp_path, "fake_sync_close.so", monkeypatch
    )
    log = _spy_events(monkeypatch)
    for _ in range(2):
        raw.run(inputs)
    handles = list(raw._pending_events)
    assert len(handles) == 2

    raw.close()

    assert log.awaited == []  # close never blocks on device work
    assert sorted(log.destroyed) == sorted(handles)
    assert raw._pending_events == []
    assert raw._sync_unavailable_reason is None
    assert raw._client is None
    # Post-close synchronization is the quiet idle path (nothing pending).
    assert raw.synchronize() is None
    assert log.awaited == []  # unchanged


def test_close_swallows_destroy_failures(tmp_path, monkeypatch):
    """Best-effort cleanup: a failing destroy can never make close() fail."""
    raw, inputs = _device_resident_executable(
        tmp_path, "fake_sync_close_fail.so", monkeypatch
    )
    log = _spy_events(monkeypatch)
    raw.run(inputs)
    (handle,) = raw._pending_events
    log.destroy_errors[handle] = RuntimeError("destroy-boom-on-close")

    raw.close()  # must NOT raise

    assert log.destroyed == [handle]  # attempted
    assert log.awaited == []
    assert raw._pending_events == []
    assert raw._client is None


# ---------------------------------------------------------------------------
# 5. driver-level: a stray handle on a failed execute never leaks
# ---------------------------------------------------------------------------


def test_failed_execute_destroys_stray_handle_and_records_nothing(
    tmp_path, monkeypatch
):
    """Execute failure after the plugin wrote a handle: destroyed, not recorded."""
    raw, inputs = _device_resident_executable(
        tmp_path, "fake_sync_stray.so", monkeypatch
    )
    log = _spy_events(monkeypatch)
    monkeypatch.setenv("ETL_FAKE_PJRT_FAIL_AFTER_DEVICE_EVENT", "1")

    with pytest.raises(
        etl.BackendError,
        match="injected fake failure after writing a device-complete event",
    ):
        raw.run(inputs)

    assert raw._pending_events == []  # the caller got nothing
    assert raw._sync_unavailable_reason is None  # an execute failure is not a
    # "no event" contract violation — it surfaces from run itself
    assert log.awaited == []
    assert len(log.destroyed) == 1 and log.destroyed[0]  # the stray handle
    stray = log.destroyed[0]

    # The next successful run records normally and synchronizes cleanly
    # (the freed stray may be re-used by the allocator, so compare positions,
    # never identity).
    monkeypatch.delenv("ETL_FAKE_PJRT_FAIL_AFTER_DEVICE_EVENT")
    raw.run(inputs)
    (handle,) = raw._pending_events
    assert raw.synchronize() is None
    assert log.awaited == [handle]
    assert len(log.destroyed) == 2
    assert log.destroyed[0] == stray
    assert log.destroyed[1] == handle


# ---------------------------------------------------------------------------
# 6. the CPU host-staging path requests no event at all
# ---------------------------------------------------------------------------


def test_cpu_host_staging_run_requests_no_event(tmp_path, monkeypatch):
    """CPU runs are synchronous: no event requested, sync stays a quiet no-op."""
    plug = _build_plugin(tmp_path, "fake_sync_cpu.so")
    _activate(plug, monkeypatch)
    fn, specs = u.matmul_relu_sum()
    artifact = _compile_fake(fn, specs, plug)
    raw = etl.backends.get(NAME).load(artifact)  # device None = host staging
    assert raw.device == CPU
    log = _spy_events(monkeypatch)

    a, b = u.matmul_relu_sum_args()
    (out,) = raw.run([etl.core.Tensor(a), etl.core.Tensor(b)])
    assert isinstance(out.data, np.ndarray)  # already a host copy
    assert raw._pending_events == []
    assert raw._sync_unavailable_reason is None

    # The host-staging path's OWN awaits (``ToHostBuffer`` completion
    # events) happened inside run; synchronization adds nothing.
    awaited, destroyed = len(log.awaited), len(log.destroyed)
    assert raw.synchronize() is None
    assert len(log.awaited) == awaited
    assert len(log.destroyed) == destroyed
