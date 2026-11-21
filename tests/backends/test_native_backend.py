"""The ``native`` backend (in-process Rust CPU engine) — staging and runtime contract.

Contract under test: ``../../evoxc/backends/native/CONTEXT.md`` — the FOUR-stage
contract shared with the numpy reference backend (``lower`` → ``compile`` →
``load`` → ``run``), the ``Signature``/``LoweredProgram``/``CompiledArtifact``
objects, the host-data device story (CPU only, no staging tier, no implicit
transfer) and the engine diagnostics (``plan_text`` / ``stats`` /
``required_ops`` / ``summary``).

The engine is the Rust ``evoxc-cpu`` crate reached through the
``evoxc._evoxc_native`` PyO3 extension, so this module is gated on it: an
unbuilt extension skips the file with the shared
:data:`tests.backends._native_utils.SKIP_REASON`. The missing-extension
contract (the actionable ``BackendError``) lives in the companion
``test_native_backend_missing_engine.py``, which runs in BOTH configurations.

The numpy backend is the numerical oracle throughout (``nu.evaluate_numpy``),
compared with ``nu.assert_tree_parity`` (exact for bool/integer, a per-width
float budget otherwise). CPU only, tiny shapes, no network.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import evoxc
from evoxc import core
from evoxc.backends.native import engine
from tests.backends import _native_utils as nu

pytestmark = pytest.mark.skipif(not nu.ENGINE_AVAILABLE, reason=nu.SKIP_REASON)


# ---------------------------------------------------------------------------
# Shared program: one matmul + one reduction (two engine dispatches)
# ---------------------------------------------------------------------------

@evoxc.defn
def matmul_reduce(x, w):
    """``dot(x, w)`` then a per-row sum — two ops, one block, one output."""
    return evoxc.sum(evoxc.dot(x, w), axes=1)


#: Named input specs (``name=`` is what ``evoxc.bind`` resolves).
SPECS = (
    core.TensorSpec((4, 3), np.float32, name="x"),
    core.TensorSpec((3, 2), np.float32, name="w"),
)

_RNG = np.random.default_rng(20260101)
X = _RNG.normal(size=(4, 3)).astype(np.float32)
W = _RNG.normal(size=(3, 2)).astype(np.float32)

#: Repository root — the cwd for the fresh-interpreter subprocess legs.
ROOT = Path(__file__).resolve().parents[2]


def _python(script, *args):
    """Run ``script`` in a fresh interpreter (cwd = repo root, env inherited)."""
    return subprocess.run(
        [sys.executable, "-c", script, *args],
        cwd=str(ROOT),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.fixture(scope="module")
def graph():
    """The traced graph (block-free: ``lower`` never mutates it)."""
    return evoxc.trace(matmul_reduce, *SPECS)


@pytest.fixture(scope="module")
def staged(graph):
    """One full ``lower`` → ``compile`` → ``load`` staging of that graph."""
    lowered = evoxc.lower(graph, backend="native")
    artifact = evoxc.compile(lowered, backend="native")
    executable = evoxc.load(artifact, backend="native")
    return {
        "lowered": lowered,
        "artifact": artifact,
        "executable": executable,
    }


def _reference():
    return nu.evaluate_numpy(matmul_reduce, X, W)


# ---------------------------------------------------------------------------
# 1. The explicit pipeline and its objects
# ---------------------------------------------------------------------------


def test_explicit_pipeline_stage_attributes(graph, staged):
    """Every stage has the documented type and backend/target metadata."""
    assert isinstance(graph, evoxc.Graph)
    lowered, artifact, executable = (
        staged["lowered"],
        staged["artifact"],
        staged["executable"],
    )
    assert isinstance(lowered, evoxc.backends.LoweredProgram)
    assert isinstance(artifact, evoxc.backends.CompiledArtifact)
    assert isinstance(executable, evoxc.Executable)

    assert lowered.backend == "native"
    assert artifact.backend == "native"
    assert artifact.target == "cpu"
    # A plain graph needs no custom ops, and the runtime dependencies are the
    # host numpy version plus the extension's own version.
    assert artifact.required_custom_ops == ()
    assert set(artifact.runtime_dependencies) == {"numpy", "evoxc-native"}
    assert artifact.runtime_dependencies["numpy"] == np.__version__

    assert executable.backend == "native"
    assert executable.device == core.Device("cpu", 0)
    assert executable.functions == ("main",)


def test_explicit_pipeline_values_match_numpy(staged):
    """The engine's numbers equal the pure-Python interpreter's numbers."""
    got = nu.as_numpy(evoxc.run(staged["executable"], X, W))
    nu.assert_tree_parity(got, _reference(), label="explicit pipeline")


# ---------------------------------------------------------------------------
# 2/3. The documented sugar (build / evaluate)
# ---------------------------------------------------------------------------


def test_build_matches_explicit_pipeline(staged):
    """``evoxc.build(..., backend="native")`` == the explicit pipeline."""
    built = nu.build_native(matmul_reduce, *SPECS)
    assert built.backend == "native"
    assert built.device == core.Device("cpu", 0)
    got = nu.run_native(built, X, W)
    nu.assert_tree_parity(
        got, nu.run_native(staged["executable"], X, W), label="build vs explicit"
    )
    nu.assert_tree_parity(got, _reference(), label="build vs numpy")


def test_evaluate_matches_numpy_backend():
    """``evoxc.evaluate(..., backend="native")`` == ``backend="numpy"``."""
    got = nu.evaluate_native(matmul_reduce, X, W)
    expected = nu.evaluate_numpy(matmul_reduce, X, W)
    nu.assert_tree_parity(got, expected, label="evaluate")


# ---------------------------------------------------------------------------
# 4. bind (argument-passing sugar) + synchronize
# ---------------------------------------------------------------------------


def test_bind_supplies_named_input(staged):
    """Binding ``w`` removes it from the run signature; values are unchanged."""
    bound = evoxc.bind(staged["executable"], w=evoxc.from_numpy(W))
    assert isinstance(bound, evoxc.BoundExecutable)
    assert bound.summary()["bound_count"] == 1
    got = nu.as_numpy(evoxc.run(bound, X))
    nu.assert_tree_parity(got, _reference(), label="bound")


def test_bound_executable_synchronize_is_callable(staged):
    """``BoundExecutable.synchronize()`` delegates and returns ``None``."""
    bound = evoxc.bind(staged["executable"], w=W)
    assert callable(bound.synchronize)
    assert bound.synchronize() is None


# ---------------------------------------------------------------------------
# 5. Persistence: every stage round-trips
# ---------------------------------------------------------------------------


def test_lowered_program_save_load_round_trip(graph, staged, tmp_path):
    path = tmp_path / "lowered.evoxc"
    staged["lowered"].save(path)
    reloaded = evoxc.backends.LoweredProgram.load(path)
    assert reloaded.backend == "native"
    artifact = evoxc.compile(reloaded, backend="native")
    executable = evoxc.load(artifact, backend="native")
    nu.assert_tree_parity(
        nu.run_native(executable, X, W), _reference(), label="reloaded lowered"
    )


def test_compiled_artifact_save_load_round_trip(staged, tmp_path):
    path = tmp_path / "artifact.evoxc"
    staged["artifact"].save(path)
    reloaded = evoxc.backends.CompiledArtifact.load(path)
    assert reloaded.backend == "native"
    assert reloaded.target == "cpu"
    # A reloaded artifact always recompiles from its payload (the in-process
    # program hint is never serialized).
    executable = evoxc.load(reloaded, backend="native")
    nu.assert_tree_parity(
        nu.run_native(executable, X, W), _reference(), label="reloaded artifact"
    )


def test_executable_save_and_pipeline_load_round_trip(staged, tmp_path):
    path = tmp_path / "executable.evoxc"
    staged["executable"].save(path)
    reloaded = evoxc.Executable.load(path)
    assert reloaded.backend == "native"
    assert reloaded.device == core.Device("cpu", 0)
    nu.assert_tree_parity(
        nu.run_native(reloaded, X, W), _reference(), label="reloaded executable"
    )


def test_native_executable_load_round_trip(staged, tmp_path):
    """The backend-level ``NativeExecutable.load`` reconstructs the same program."""
    path = tmp_path / "backend_executable.evoxc"
    staged["executable"].save(path)
    backend_executable = type(staged["executable"].backend_executable).load(path)
    assert type(backend_executable).__name__ == "NativeExecutable"
    assert backend_executable.device == core.Device("cpu", 0)
    executable = evoxc.Executable(backend_executable, staged["artifact"].signature)
    nu.assert_tree_parity(
        nu.run_native(executable, X, W), _reference(), label="NativeExecutable.load"
    )


#: Fresh-interpreter round trip: compile the saved artifact and run it again.
_SUBPROCESS_ROUND_TRIP = """
import sys

import numpy as np

import evoxc

artifact_path, inputs_path, output_path = sys.argv[1:4]
artifact = evoxc.backends.CompiledArtifact.load(artifact_path)
executable = evoxc.load(artifact, backend="native")
with np.load(inputs_path) as payload:
    x = payload["x"]
    w = payload["w"]
out = evoxc.run(executable, x, w)
np.save(output_path, out.numpy())
"""


def test_fresh_subprocess_artifact_round_trip(staged, tmp_path):
    """A saved artifact recompiles and runs in a DIFFERENT process."""
    artifact_path = tmp_path / "artifact.evoxc"
    staged["artifact"].save(artifact_path)
    inputs_path = tmp_path / "inputs.npz"
    np.savez(inputs_path, x=X, w=W)
    output_path = tmp_path / "out.npy"

    proc = _python(
        _SUBPROCESS_ROUND_TRIP,
        str(artifact_path),
        str(inputs_path),
        str(output_path),
    )
    assert proc.returncode == 0, (
        f"fresh-interpreter artifact run failed:\nstdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    nu.assert_parity(
        np.load(output_path), _reference(), label="subprocess round trip"
    )


# ---------------------------------------------------------------------------
# 7. Backend-name mismatches are persistence/backend errors
# ---------------------------------------------------------------------------


def test_pipeline_load_rejects_cross_backend_artifacts(graph, staged, tmp_path):
    """``Executable.load`` refuses a recorded backend that differs from the ask."""
    numpy_artifact = evoxc.compile(evoxc.lower(graph, backend="numpy"), backend="numpy")
    numpy_path = tmp_path / "numpy_artifact.evoxc"
    numpy_artifact.save(numpy_path)
    with pytest.raises(core.PersistenceError) as info:
        evoxc.Executable.load(numpy_path, backend="native")
    assert "numpy" in str(info.value)

    native_path = tmp_path / "native_artifact.evoxc"
    staged["artifact"].save(native_path)
    with pytest.raises(core.PersistenceError) as info:
        evoxc.Executable.load(native_path, backend="numpy")
    assert "native" in str(info.value)


def test_backend_load_rejects_foreign_artifact(graph):
    """The backend's own ``load`` rejects an artifact it did not produce."""
    numpy_artifact = evoxc.compile(evoxc.lower(graph, backend="numpy"), backend="numpy")
    with pytest.raises(core.PersistenceError) as info:
        evoxc.backends.get("native").load(numpy_artifact)
    assert "numpy" in str(info.value)


def test_compile_backend_mismatch_names_both_backends(graph, staged):
    """``evoxc.compile`` never silently re-lowers across backends."""
    numpy_lowered = evoxc.lower(graph, backend="numpy")
    with pytest.raises(core.BackendError) as info:
        evoxc.compile(numpy_lowered, backend="native")
    assert "numpy" in str(info.value) and "native" in str(info.value)

    with pytest.raises(core.BackendError) as info:
        evoxc.compile(staged["lowered"], backend="numpy")
    assert "native" in str(info.value) and "numpy" in str(info.value)


# ---------------------------------------------------------------------------
# 8. Device rules at load (v1: CPU only)
# ---------------------------------------------------------------------------


def test_load_rejects_non_cpu_device(staged):
    """A non-CPU ``Device`` is an explicit ``BackendError`` naming the kind."""
    with pytest.raises(core.BackendError) as info:
        evoxc.backends.get("native").load(staged["artifact"], core.Device("cuda", 0))
    message = str(info.value)
    assert "cuda" in message
    assert "CPU" in message

    # … and through the pipeline wrapper, which forwards the same device.
    with pytest.raises(core.BackendError):
        evoxc.load(
            staged["artifact"], backend="native", device=core.Device("cuda", 0)
        )


def test_load_rejects_non_device_object(staged):
    """A bare string is NOT a ``Device`` (the pipeline normalizes it; the
    backend-level contract demands a ``core.Device``)."""
    with pytest.raises(core.DeviceError):
        evoxc.backends.get("native").load(staged["artifact"], "cpu")


def test_load_accepts_none_and_explicit_cpu_device(staged):
    """``device=None`` and an explicit ``cpu:0`` both load and run."""
    backend = evoxc.backends.get("native")
    default_executable = backend.load(staged["artifact"])
    assert default_executable.device == core.Device("cpu", 0)
    cpu_executable = backend.load(staged["artifact"], core.Device("cpu", 0))
    assert cpu_executable.device == core.Device("cpu", 0)
    for backend_executable in (default_executable, cpu_executable):
        wrapped = evoxc.Executable(backend_executable, staged["artifact"].signature)
        nu.assert_tree_parity(
            nu.run_native(wrapped, X, W), _reference(), label="cpu load"
        )


# ---------------------------------------------------------------------------
# 9. Invalid inputs at the run boundary keep their documented error KINDS
# ---------------------------------------------------------------------------


def test_run_wrong_shape_raises_shape_error(staged):
    with pytest.raises(core.ShapeError):
        evoxc.run(staged["executable"], np.zeros((5, 3), np.float32), W)


def test_run_wrong_dtype_raises_dtype_error(staged):
    with pytest.raises(core.DTypeError):
        evoxc.run(staged["executable"], X.astype(np.float64), W)


def test_run_wrong_structure_raises_trace_error(staged):
    """A pytree structure mismatch is the pipeline's ``TraceError``."""
    with pytest.raises(core.TraceError) as info:
        evoxc.run(staged["executable"], {"x": X}, W)
    assert "pytree path" in str(info.value)


def test_engine_unsupported_ops_reader_is_tolerant():
    """The engine's unsupported-op report tolerates spelling/record drift."""
    assert engine.unsupported_ops({}) == ()
    assert engine.unsupported_ops({"unsupported_ops": ["foo", {"op": "bar"}]}) == (
        "bar",
        "foo",
    )


def test_known_options_are_empty_and_compiler_options_are_loud(graph):
    """v1 takes NO pipeline options — a compiler option fails, never ignored."""
    backend = evoxc.backends.get("native")
    assert set(backend.KNOWN_OPTIONS) == {"lower", "compile", "load", "run"}
    assert all(value == frozenset() for value in backend.KNOWN_OPTIONS.values())

    with pytest.raises(core.BackendError) as info:
        evoxc.lower(graph, backend="native", opt_level="O1")
    assert "opt_level" in str(info.value)


def test_capability_dtype_gate_rejects_undeclared_dtypes():
    """The dtype gate is a private seam (no public op produces an undeclared
    dtype), so it is pinned directly: declared dtypes pass, anything else is a
    ``BackendError`` naming the dtype."""
    backend = evoxc.backends.get("native")
    assert np.dtype(np.float32) in backend.capabilities.dtypes
    assert backend._check_value_dtype(np.dtype(np.float32), "probe") is None
    with pytest.raises(core.BackendError) as info:
        backend._check_value_dtype(np.dtype("U3"), "probe")
    assert "cannot execute dtype" in str(info.value)


# ---------------------------------------------------------------------------
# 10. Diagnostics: summary / plan text / stats / required ops / synchronize
# ---------------------------------------------------------------------------

_SUMMARY_KEYS = {
    "backend",
    "device",
    "functions",
    "input_count",
    "output_count",
    "input_specs",
    "output_specs",
    "static_value_count",
    "required_ops",
    "capabilities",
}


def test_summary_surface(staged):
    summary = staged["executable"].summary()
    assert _SUMMARY_KEYS <= set(summary)
    assert summary["backend"] == "native"
    assert summary["device"] == str(core.Device("cpu", 0))
    assert summary["functions"] == ["main"]
    assert summary["input_count"] == 2
    assert summary["output_count"] == 1
    assert summary["static_value_count"] == 0
    assert summary["required_ops"] == []
    assert summary["capabilities"]["dtype_count"] == 14
    assert summary["capabilities"]["sparse_ops"] is False
    assert summary["capabilities"]["dynamic_shapes"] is True


def test_plan_text_names_entry_and_ops(staged):
    text = staged["executable"].plan_text()
    assert isinstance(text, str) and text
    assert "entry 'main'" in text
    assert "dot" in text
    assert "reduce_sum" in text


def test_engine_stats_surface(staged):
    stats = staged["executable"].backend_executable.stats()
    assert {
        "ops",
        "dispatched_ops",
        "blocks",
        "plan_entries",
        "compile_ms",
        "unsupported",
    } <= set(stats)
    assert stats["ops"] >= stats["dispatched_ops"] >= 1
    assert stats["blocks"] >= 1
    assert stats["plan_entries"] >= 1
    assert stats["compile_ms"] >= 0
    assert stats["unsupported"] == []


def test_required_ops_report(staged):
    assert staged["executable"].backend_executable.required_ops() == {
        "blocks": (),
        "external": (),
        "runtime_calls": (),
    }


def test_synchronize_is_a_value_neutral_noop(staged):
    """``run`` blocks, so ``synchronize`` is the documented no-op."""
    before = nu.as_numpy(evoxc.run(staged["executable"], X, W))
    assert staged["executable"].synchronize() is None
    assert staged["executable"].synchronize() is None
    after = nu.as_numpy(evoxc.run(staged["executable"], X, W))
    nu.assert_tree_parity(after, before, label="synchronize")
    nu.assert_tree_parity(after, _reference(), label="synchronize vs numpy")


# ---------------------------------------------------------------------------
# 11. Lazy activation: ``import evoxc`` never loads the extension
# ---------------------------------------------------------------------------

_IMPORT_PROBE = "import sys, evoxc; print('evoxc._evoxc_native' in sys.modules)"

_ACTIVATION_PROBE = """
import sys

import numpy as np

import evoxc
from evoxc import core

print("before", "evoxc._evoxc_native" in sys.modules)
backend = evoxc.backends.get("native")
assert backend.name == "native"
print("after", "evoxc._evoxc_native" in sys.modules)


@evoxc.defn
def square_sum(x):
    return evoxc.sum(evoxc.multiply(x, x))


executable = evoxc.build(
    square_sum, core.TensorSpec((3,), np.float32), backend="native"
)
print("value", float(evoxc.run(executable, np.arange(3, dtype=np.float32)).numpy()))
"""


def test_import_evoxc_does_not_load_the_extension():
    proc = _python(_IMPORT_PROBE)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False", proc.stdout


def test_native_activation_is_lazy_and_works():
    """Touching ``backend="native"`` loads the extension and executes."""
    proc = _python(_ACTIVATION_PROBE)
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert lines[0] == "before False"
    assert lines[1] == "after True"
    assert lines[2] == "value 5.0"


# ---------------------------------------------------------------------------
# 12. Registry surface
# ---------------------------------------------------------------------------


def test_backend_registered_under_native_name():
    backend = evoxc.backends.get("native")
    assert isinstance(backend, evoxc.backends.Backend)
    assert backend.name == "native"
    # ``get`` is idempotent: the registered singleton, every time.
    assert evoxc.backends.get("native") is backend

    from evoxc.backends import native as native_package

    # ``register()`` is idempotent too (it re-probes, then re-registers the
    # very same instance).
    assert native_package.register() is native_package.native_backend
    assert evoxc.backends.get("native") is native_package.native_backend


def test_capability_matrix():
    """The declared v1 capability matrix (CPU host data, no sparse, sync)."""
    capabilities = evoxc.backends.get("native").capabilities
    assert capabilities.dynamic_shapes is True
    assert capabilities.collectives is True
    assert capabilities.runtime_calls is True
    assert capabilities.external_calls is True
    assert capabilities.custom_blocks is True
    assert capabilities.async_collectives is False
    assert capabilities.sparse_ops is False
    assert np.dtype(np.bool_) in capabilities.dtypes
    assert np.dtype(np.complex128) in capabilities.dtypes
