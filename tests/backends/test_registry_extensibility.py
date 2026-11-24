"""Backend-registry extensibility: replace, unregister, and entry-point discovery.

Pins the registration contract of ``evoxc/backends/registry.py`` beyond the
basic round-trip already covered by ``test_backend_interface.py``:

* ``register(backend, replace=True)`` swaps a DIFFERENT instance under an
  existing name, while the default still raises ``BackendError`` (never a
  silent clobber);
* ``unregister(name)`` returns the removed instance and raises
  ``BackendError`` (naming the name) for an unknown one;
* third-party backends published in the ``evoxc.backends`` entry-point group
  are discovered on a registry miss, covering all three load shapes — a MODULE
  with a ``register()`` function, a plain CALLABLE returning a ``Backend``,
  and a ``Backend`` INSTANCE;
* the entry-point group is scanned ONCE per process (the private
  ``_reset_entry_point_cache`` hook allows a re-scan);
* a failing entry point surfaces as ``BackendError`` naming the entry point,
  chaining the original exception as ``__cause__``.

``importlib.metadata.entry_points`` is monkeypatched (never the real installed
set) so the tests are environment-independent. The process-global registry and
the scan flag are snapshotted/restored around every test.
"""

import importlib.metadata
import types

import pytest

import evoxc
from evoxc.backends import Backend, register, unregister
from evoxc.backends import registry as registry_module
from evoxc.backends.registry import ENTRY_POINT_GROUP, get


# ---------------------------------------------------------------------------
# minimal concrete Backend
# ---------------------------------------------------------------------------


def _make_backend(name):
    def lower(self, graph, options=None):
        return object()

    def compile(self, lowered, options=None):
        return object()

    def load(self, artifact, device=None):
        return object()

    cls = type(
        f"RegistryExtBackend_{name}",
        (Backend,),
        {"lower": lower, "compile": compile, "load": load, "name": name},
    )
    return cls()


class _FakeEntryPoint:
    """Stand-in for ``importlib.metadata.EntryPoint`` (name/value/load)."""

    def __init__(self, name, value, loader):
        self.name = name
        self.value = value
        self._loader = loader

    def load(self):
        return self._loader()


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Snapshot the process-global registry + scan flag; restore afterwards."""
    saved = dict(registry_module._registry)
    registry_module._reset_entry_point_cache()
    yield
    registry_module._registry.clear()
    registry_module._registry.update(saved)
    registry_module._reset_entry_point_cache()


def _patch_entry_points(monkeypatch, entry_points, scans):
    """Install a fake ``importlib.metadata.entry_points`` for the group scan."""

    def fake_entry_points(*, group):
        assert group == ENTRY_POINT_GROUP
        scans.append(group)
        return list(entry_points)

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)


# ---------------------------------------------------------------------------
# 1. replace / unregister
# ---------------------------------------------------------------------------


def test_replace_true_swaps_the_instance():
    first = _make_backend("registry-ext-replace")
    second = _make_backend("registry-ext-replace")
    assert register(first) is first
    assert register(second, replace=True) is second
    assert get("registry-ext-replace") is second


def test_duplicate_without_replace_still_raises():
    register(_make_backend("registry-ext-dup"))
    with pytest.raises(evoxc.BackendError, match="already registered"):
        register(_make_backend("registry-ext-dup"))


def test_unregister_returns_the_instance():
    backend = _make_backend("registry-ext-unregister")
    register(backend)
    assert unregister("registry-ext-unregister") is backend
    with pytest.raises(evoxc.BackendError, match="unknown backend"):
        unregister("registry-ext-unregister")


def test_unregister_unknown_name_raises_naming_it():
    with pytest.raises(evoxc.BackendError) as excinfo:
        unregister("registry-ext-never-registered")
    assert "registry-ext-never-registered" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 2. entry-point discovery: all three load shapes
# ---------------------------------------------------------------------------


def test_entry_point_module_with_register_function(monkeypatch):
    backend = _make_backend("registry-ext-ep-module")
    module = types.ModuleType("registry_ext_ep_module")
    module.register = lambda: register(backend)
    scans = []
    _patch_entry_points(
        monkeypatch,
        [_FakeEntryPoint("ep_module", "fakepkg:mod", lambda: module)],
        scans,
    )
    # A registry miss triggers the scan, which registers the module's backend.
    assert get("registry-ext-ep-module") is backend
    assert scans == [ENTRY_POINT_GROUP]


def test_entry_point_callable_returning_a_backend(monkeypatch):
    backend = _make_backend("registry-ext-ep-callable")
    scans = []
    _patch_entry_points(
        monkeypatch,
        [_FakeEntryPoint("ep_callable", "fakepkg:factory", lambda: (lambda: backend))],
        scans,
    )
    assert get("registry-ext-ep-callable") is backend


def test_entry_point_backend_instance(monkeypatch):
    backend = _make_backend("registry-ext-ep-instance")
    scans = []
    _patch_entry_points(
        monkeypatch,
        [_FakeEntryPoint("ep_instance", "fakepkg:instance", lambda: backend)],
        scans,
    )
    assert get("registry-ext-ep-instance") is backend


def test_entry_points_loaded_in_name_order(monkeypatch):
    order = []

    def _ordered(name):
        order.append(name)
        return _make_backend(name)

    scans = []
    _patch_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint("zeta", "fakepkg:zeta", lambda: _ordered("zeta")),
            _FakeEntryPoint("alpha", "fakepkg:alpha", lambda: _ordered("alpha")),
        ],
        scans,
    )

    with pytest.raises(evoxc.BackendError, match="unknown backend"):
        get("registry-ext-force-scan")
    assert order == ["alpha", "zeta"]


# ---------------------------------------------------------------------------
# 3. a single scan per process
# ---------------------------------------------------------------------------


def test_entry_point_group_is_scanned_once_per_process(monkeypatch):
    backend = _make_backend("registry-ext-ep-once")
    scans = []
    _patch_entry_points(
        monkeypatch,
        [_FakeEntryPoint("ep_once", "fakepkg:mod", lambda: backend)],
        scans,
    )
    # First miss scans and registers; a second miss must NOT scan again.
    assert get("registry-ext-ep-once") is backend
    with pytest.raises(evoxc.BackendError, match="unknown backend"):
        get("registry-ext-still-missing")
    assert len(scans) == 1
    # The private reset hook allows a re-scan.
    registry_module._reset_entry_point_cache()
    with pytest.raises(evoxc.BackendError, match="unknown backend"):
        get("registry-ext-still-missing")
    assert len(scans) == 2


# ---------------------------------------------------------------------------
# 4. a failing entry point surfaces (naming it, chaining the cause)
# ---------------------------------------------------------------------------


def test_failing_entry_point_raises_backenderror_naming_it(monkeypatch):
    def boom():
        raise RuntimeError("kaboom from the entry point")

    scans = []
    _patch_entry_points(
        monkeypatch,
        [_FakeEntryPoint("ep_broken", "fakepkg:broken", boom)],
        scans,
    )
    with pytest.raises(evoxc.BackendError) as excinfo:
        get("registry-ext-trigger-broken")
    message = str(excinfo.value)
    assert "ep_broken" in message
    assert "fakepkg:broken" in message
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "kaboom" in str(excinfo.value.__cause__)


def test_loaded_object_of_unsupported_shape_raises(monkeypatch):
    # An object that is neither a Backend, a module with register(), nor a
    # callable — the scan raises an explicit BackendError, never silently
    # ignoring the entry point.
    scans = []
    _patch_entry_points(
        monkeypatch,
        [_FakeEntryPoint("ep_bad_shape", "fakepkg:x", lambda: 42)],
        scans,
    )
    with pytest.raises(evoxc.BackendError, match="ep_bad_shape"):
        get("registry-ext-trigger-bad-shape")
