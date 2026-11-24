"""tests/test_pipeline_backend_instance.py — a passed ``Backend`` INSTANCE
reaches every staging hop.

``evoxc.build`` / ``evoxc.build_cached`` resolve their backend ONCE
(``evoxc.pipeline._resolve_backend_device``) and the resolved ``Backend``
instance is the one used at the ``lower``, ``compile`` AND ``load`` hops — an
explicitly passed instance is never dropped and never re-resolved by name.
``evoxc.load`` likewise uses a passed instance for the load hop (after the
existing name-match ``PersistenceError`` check), not a registry lookup.

Proof strategy (binding for this file): the instance's name is ALSO registered
to a DIFFERENT, recording decoy backend, so a name lookup would silently hit
the decoy. Every assertion therefore checks the passed instance recorded the
hop AND the decoy recorded nothing — the pin is not trivially true by
resolution. Nothing here touches a compiler adapter; the recording backend
delegates staging to the numpy reference backend and works on CPU only.
"""

import numpy as np
import pytest

import evoxc
from evoxc import backends
from evoxc.backends import registry as _registry

#: The recording backend's name; a decoy backend is registered under it for
#: the duration of each test so name resolution can never masquerade as
#: instance passing.
_NAME = "recording-instance-backend"

_SPEC = evoxc.TensorSpec((4,), evoxc.float32)
_X = np.array([0.5, -1.0, 2.0, 3.5], dtype=np.float32)


class _DelegateExecutable:
    """Backend executable: records runs, delegates behaviour to numpy's."""

    def __init__(self, inner, backend_name):
        self._inner = inner
        self.backend_name = backend_name
        self.runs = 0

    @property
    def functions(self):
        return self._inner.functions

    @property
    def device(self):
        return self._inner.device

    def run(self, flat_input_tensors, options=None):
        self.runs += 1
        return self._inner.run(flat_input_tensors, options)

    def save(self, path):
        return self._inner.save(path)


class RecordingBackend(backends.Backend):
    """Minimal recording backend: delegates staging to the numpy backend but
    stamps its own ``name`` into the staged objects and records every hop
    (stage + option dict) plus the executable it produced."""

    name = _NAME
    capabilities = backends.numpy_backend.capabilities

    def __init__(self):
        self.calls = []
        self.executable = None

    def lower(self, graph, options=None):
        self.calls.append(("lower", dict(options or {})))
        inner = backends.numpy_backend.lower(graph, options)
        return backends.LoweredProgram(
            backend=self.name,
            signature=inner.signature,
            payload=inner.payload,
        )

    def compile(self, lowered, options=None):
        self.calls.append(("compile", dict(options or {})))
        if lowered.backend != self.name:
            raise evoxc.BackendError(
                f"cannot compile a LoweredProgram produced by backend "
                f"{lowered.backend!r} with the {self.name} backend"
            )
        inner = backends.numpy_backend.compile(
            backends.LoweredProgram(
                backend="numpy",
                signature=lowered.signature,
                payload=lowered.payload,
            ),
            options,
        )
        return backends.CompiledArtifact(
            backend=self.name,
            signature=inner.signature,
            target=inner.target,
            payload=inner.payload,
            required_custom_ops=inner.required_custom_ops,
            runtime_dependencies=inner.runtime_dependencies,
        )

    def load(self, artifact, device=None, options=None):
        self.calls.append(("load", dict(options or {})))
        if artifact.backend != self.name:
            raise evoxc.PersistenceError(
                f"artifact was produced by backend {artifact.backend!r}; "
                f"cannot load it with backend {self.name!r}"
            )
        inner = backends.numpy_backend.load(
            backends.CompiledArtifact(
                backend="numpy",
                signature=artifact.signature,
                target=artifact.target,
                payload=artifact.payload,
                required_custom_ops=artifact.required_custom_ops,
                runtime_dependencies=artifact.runtime_dependencies,
            ),
            device,
        )
        self.executable = _DelegateExecutable(inner, self.name)
        return self.executable

    def stages(self):
        """The recorded stage names, in call order."""
        return [stage for stage, _ in self.calls]


@pytest.fixture
def recording():
    """``(passed_instance, decoy)`` with the decoy registered under ``_NAME``.

    The registry entry is saved/restored by direct dict access so the fixture
    is order-independent and never leaks into another test.
    """
    instance = RecordingBackend()
    decoy = RecordingBackend()
    previous = _registry._registry.get(_NAME)
    _registry._registry[_NAME] = decoy
    try:
        yield instance, decoy
    finally:
        if previous is None:
            _registry._registry.pop(_NAME, None)
        else:
            _registry._registry[_NAME] = previous


def _scale(x):
    """The workhorse graph: x * 2.0 (a tiny elementwise program)."""
    return evoxc.multiply(x, 2.0)


# ---------------------------------------------------------------------------
# 1. The fixture is meaningful: the name resolves to the decoy, not the pass
# ---------------------------------------------------------------------------


def test_registered_name_resolves_to_the_decoy(recording):
    """Sanity check for every other pin: ``backends.get(_NAME)`` returns the
    DECOY, so a name lookup can never be mistaken for instance passing."""
    instance, decoy = recording
    assert instance is not decoy
    assert backends.get(_NAME) is decoy


# ---------------------------------------------------------------------------
# 2. evoxc.build — the passed instance is used at lower, compile AND load
# ---------------------------------------------------------------------------


def test_build_uses_the_passed_instance_at_every_hop(recording):
    instance, decoy = recording

    executable = evoxc.build(_scale, _SPEC, backend=instance)

    assert instance.stages() == ["lower", "compile", "load"]
    assert decoy.calls == []
    # The staged objects carry the instance's name all the way through.
    assert executable.backend == _NAME
    assert executable.device == evoxc.Device("cpu", 0)
    # ... and the produced executable really runs on that instance.
    out = evoxc.run(executable, _X)
    np.testing.assert_array_equal(out.numpy(), _X * 2.0)
    assert instance.executable.runs == 1


def test_build_forwards_the_same_options_to_all_three_hops(recording):
    """The build docstring's expansion forwards ONE options dict to lower,
    compile AND load — pinned on the observable per-hop option snapshots."""
    instance, decoy = recording

    evoxc.build(_scale, _SPEC, backend=instance, opt_level="O2")

    assert instance.stages() == ["lower", "compile", "load"]
    assert [options for _, options in instance.calls] == [
        {"opt_level": "O2"},
        {"opt_level": "O2"},
        {"opt_level": "O2"},
    ]
    assert decoy.calls == []


def test_build_traces_before_lower_and_follows_the_documented_order(recording):
    """The documented expansion is ``trace -> lower -> compile -> load``: the
    trace body runs BEFORE the backend's first hop, and the backend hops then
    occur in exactly that order (build's docstring stays accurate)."""
    instance, decoy = recording
    events = []

    def recorded_fn(x):
        events.append("trace")
        return evoxc.multiply(x, 2.0)

    executable = evoxc.build(recorded_fn, _SPEC, backend=instance)

    assert events == ["trace"]
    assert instance.stages() == ["lower", "compile", "load"]
    assert decoy.calls == []
    assert executable.signature is not None


# ---------------------------------------------------------------------------
# 3. evoxc.build_cached — same instance on a miss, identical executable on a hit
# ---------------------------------------------------------------------------


def test_build_cached_uses_the_instance_on_a_miss(recording):
    """The documented miss expansion is ``trace -> lower -> compile -> load``:
    the trace body runs first and the SAME instance then performs the three
    backend hops."""
    instance, decoy = recording
    cache = evoxc.CompileCache()
    events = []

    def recorded_fn(x):
        events.append("trace")
        return evoxc.multiply(x, 2.0)

    executable = evoxc.build_cached(recorded_fn, _SPEC, backend=instance, cache=cache)

    assert events == ["trace"]
    assert instance.stages() == ["lower", "compile", "load"]
    assert decoy.calls == []
    assert cache.misses == 1 and cache.hits == 0
    assert executable.backend == _NAME


def test_build_cached_hit_returns_the_identical_executable_and_no_new_hops(
    recording,
):
    instance, decoy = recording
    cache = evoxc.CompileCache()

    first = evoxc.build_cached(_scale, _SPEC, backend=instance, cache=cache)
    hops_after_miss = list(instance.calls)
    second = evoxc.build_cached(_scale, _SPEC, backend=instance, cache=cache)

    # A hit is the SAME fully loaded Executable with zero pipeline work.
    assert second is first
    assert cache.hits == 1 and cache.misses == 1
    assert instance.calls == hops_after_miss
    assert decoy.calls == []


# ---------------------------------------------------------------------------
# 4. evoxc.load — a passed instance loads, never a name lookup
# ---------------------------------------------------------------------------


def test_load_uses_the_passed_instance_not_the_registered_decoy(recording):
    """A DIFFERENT instance sharing the artifact's backend name performs the
    load hop — proving ``load`` honours the instance rather than resolving the
    name (which would hit the registered decoy)."""
    instance, decoy = recording
    graph = evoxc.trace(_scale, _SPEC)
    lowered = evoxc.lower(graph, backend=instance)
    artifact = evoxc.compile(lowered, backend=instance)
    assert instance.stages() == ["lower", "compile"]

    loader = RecordingBackend()
    executable = evoxc.load(artifact, backend=loader)

    assert loader.stages() == ["load"]
    assert decoy.calls == []
    # The first instance was NOT asked to load (no name re-resolution).
    assert instance.stages() == ["lower", "compile"]
    assert executable.backend == _NAME
    np.testing.assert_array_equal(
        evoxc.run(executable, _X).numpy(), _X * 2.0
    )


def test_load_name_mismatch_still_raises_persistence_error(recording):
    """The existing name-mismatch ``PersistenceError`` is unchanged for a
    passed instance AND for a name string; neither reaches any backend."""
    instance, decoy = recording
    graph = evoxc.trace(_scale, _SPEC)
    artifact = evoxc.compile(
        evoxc.lower(graph, backend=instance), backend=instance
    )

    other = RecordingBackend()
    other.name = "some-other-backend"
    with pytest.raises(evoxc.PersistenceError) as excinfo:
        evoxc.load(artifact, backend=other)
    assert "never silently recompile" in str(excinfo.value)
    assert other.calls == []

    with pytest.raises(evoxc.PersistenceError) as excinfo:
        evoxc.load(artifact, backend="numpy")
    assert "never silently recompile" in str(excinfo.value)

    assert decoy.calls == []
    assert instance.stages() == ["lower", "compile"]
