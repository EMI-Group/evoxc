"""The ``native`` backend WITHOUT the extension — the missing-engine contract.

Contract under test: ``../../evoxc/backends/native/CONTEXT.md`` — the backend
is OPTIONAL and lazily activated, and a missing/broken ``evoxc._evoxc_native``
extension (or one built without the Rust CPU engine's ``native_compile`` entry
point) is an ACTIONABLE ``core.BackendError`` naming
``scripts/build-native.sh`` and ``EVOXC_NATIVE_LIB`` — NEVER a silent fallback
to the numpy interpreter and never an ``ImportError``/``SystemExit`` escaping
the staging API.

This is the ONE native-backend suite with no module-level skip marker: it must
run in BOTH an engine-present and an engine-absent environment. The seam is
:func:`tests.backends._native_utils.without_engine`, which points
``EVOXC_NATIVE_LIB`` at a non-existent file and clears the loader cache for the
duration of a block, then restores the previous state exactly (including the
loader cache) so no other test sees a poisoned loader.
"""
import numpy as np
import pytest

import evoxc
from evoxc import core
from evoxc.backends.native import engine
from tests.backends import _native_utils as nu


@evoxc.defn
def doubled(x):
    """Elementwise square: a graph the engine COULD run when it is built."""
    return evoxc.multiply(x, x)


#: One concrete input spec (tracing needs no engine).
SPECS = (core.TensorSpec((3,), np.float32),)

#: The traced graph (block-free, so a failed ``lower`` mutates nothing).
GRAPH = evoxc.trace(doubled, *SPECS)

#: The host array both backends can compute on.
ARRAY = np.arange(3, dtype=np.float32)

#: The staging entry points that must fail explicitly without the engine.
_ACTIONS = {
    "build": lambda: evoxc.build(doubled, *SPECS, backend="native"),
    "lower": lambda: evoxc.lower(GRAPH, backend="native"),
    "register": lambda: evoxc.backends.native.register(),
}


def test_without_engine_disables_and_restores_the_loader():
    """The seam reports the extension unavailable, then restores the state."""
    assert engine.engine_available() == nu.ENGINE_AVAILABLE
    with nu.without_engine():
        assert engine.engine_available() is False
        with pytest.raises(core.BackendError):
            engine.engine_module()
    assert engine.engine_available() == nu.ENGINE_AVAILABLE


def test_build_raises_actionable_backend_error():
    """``evoxc.build(..., backend="native")`` names the build recipe + override."""
    with nu.without_engine():
        with pytest.raises(core.BackendError) as info:
            evoxc.build(doubled, *SPECS, backend="native")
    message = str(info.value)
    assert "scripts/build-native.sh" in message
    assert "EVOXC_NATIVE_LIB" in message
    assert "native_compile" in message


def test_build_never_falls_back_to_numpy():
    """Without the engine the build RAISES — it never returns a numpy result.

    Positive control: the very same program still evaluates on the explicit
    numpy backend, so the failure above is the native backend refusing, not a
    broken graph.
    """
    with nu.without_engine():
        try:
            result = evoxc.build(doubled, *SPECS, backend="native")
        except core.BackendError as exc:
            assert "never falls back" in str(exc)
        else:  # pragma: no cover - the contract violation this test exists for
            pytest.fail(
                "evoxc.build(backend='native') returned "
                f"{result!r} without the native extension — a silent fallback"
            )

    reference = evoxc.evaluate(doubled, ARRAY, backend="numpy").numpy()
    np.testing.assert_array_equal(reference, ARRAY * ARRAY)


def test_register_probes_the_extension_and_raises():
    """``register()`` probes the extension on every activation attempt.

    Asserted on ``register()`` directly (not ``backends.get``): the process
    registry may already hold the singleton from an earlier test, in which case
    ``get`` returns it without re-probing.
    """
    with nu.without_engine():
        with pytest.raises(core.BackendError) as info:
            evoxc.backends.native.register()
    assert "scripts/build-native.sh" in str(info.value)


def test_lower_raises_naming_the_build_script():
    """``evoxc.lower(..., backend="native")`` fails at the staging boundary."""
    with nu.without_engine():
        with pytest.raises(core.BackendError) as info:
            evoxc.lower(GRAPH, backend="native")
    message = str(info.value)
    assert "scripts/build-native.sh" in message
    assert "EVOXC_NATIVE_LIB" in message


def test_engine_version_without_engine_is_explicit():
    """Even the version query used to stamp artifacts is an explicit error."""
    with nu.without_engine():
        with pytest.raises(core.BackendError) as info:
            engine.engine_version()
    assert "scripts/build-native.sh" in str(info.value)


@pytest.mark.parametrize("action", sorted(_ACTIONS))
def test_failures_are_evoxc_errors_not_systemexit_or_importerror(action):
    """No ``SystemExit``/``ImportError`` may leak out of the staging API."""
    raised = None
    with nu.without_engine():
        try:
            _ACTIONS[action]()
        except BaseException as exc:  # noqa: BLE001 — classify, do not swallow
            raised = exc
    assert raised is not None, f"{action}() succeeded without the extension"
    assert isinstance(raised, core.EVOXCError), (
        f"{action}() raised {type(raised).__name__}, expected an evoxc error: "
        f"{raised!r}"
    )
    assert not isinstance(raised, (SystemExit, ImportError))


@pytest.mark.skipif(not nu.ENGINE_AVAILABLE, reason=nu.SKIP_REASON)
def test_loader_state_is_fully_restored_afterwards():
    """After the seam closes, a native build runs again (no poisoned cache)."""
    with nu.without_engine():
        with pytest.raises(core.BackendError):
            evoxc.build(doubled, *SPECS, backend="native")

    assert engine.engine_available() is True
    executable = nu.build_native(doubled, *SPECS)
    nu.assert_tree_parity(
        nu.run_native(executable, ARRAY),
        nu.evaluate_numpy(doubled, ARRAY),
        label="restored loader",
    )
