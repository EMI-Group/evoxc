"""Inspection / introspection surface of the public pipeline objects.

The contract under test is ``evoxc/CONTEXT.md`` ("Inspection (pure reads)"):

* ``Graph`` — ``repr``, ``summary()`` (documented key set, JSON-safe),
  ``digest()`` (deterministic in-process, stable across processes for the same
  source, sensitive to real graph changes) and ``text()``.
* ``LoweredProgram`` / ``CompiledArtifact`` — ``repr``, ``summary()``,
  ``source_text()`` / ``text()`` and the ``save``/``load`` round trip.
* ``Executable`` / ``BoundExecutable`` — ``repr``, ``summary()``,
  ``plan_text()`` (numpy interpreter plan and the native engine plan).
* ``Defn`` — metadata-only ``repr`` that never traces.
* ``Tensor`` — ``repr`` is inspection only: it shows shape/dtype/device and
  never materializes device memory nor raises.

Every helper exercised here is a PURE read: nothing traces, lowers, compiles,
executes, transfers or populates a cache. The rendering lives in
``evoxc/_inspect.py`` (pipeline wrappers), ``evoxc/trace/graph.py``,
``evoxc/trace/defn.py``, ``evoxc/backends/program.py`` and
``evoxc/core/tensor.py``.

The native leg is guarded: it needs the optional ``evoxc._evoxc_native``
extension (built by ``scripts/build-native.sh`` from the ``native/`` Cargo
workspace) and is skipped with build instructions when that extension is
absent.
"""
from __future__ import annotations

import functools
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import evoxc
from evoxc import core
from evoxc.backends.native import engine as native_engine

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Shared graphs, specs and documented key sets
# ---------------------------------------------------------------------------


@evoxc.defn
def _dot_relu(x, w):
    """dot(x, w) → relu — two tensor inputs, one output, no static values."""
    return evoxc.relu(evoxc.dot(x, w))


@evoxc.defn
def _pair(x):
    """One tensor input, two outputs (two-op graph)."""
    return (x, evoxc.add(x, x))


@evoxc.defn
def _with_statics(x, n, tag):
    """A graph specialized on an int and a (long) string static value."""
    if n > 3:
        return evoxc.multiply(x, 2.0)
    return x


def _dot_relu_specs():
    """Named specs for ``_dot_relu`` (names let ``evoxc.bind`` address leaves)."""
    return (
        evoxc.TensorSpec((2, 3), evoxc.float32, name="x"),
        evoxc.TensorSpec((3, 4), evoxc.float32, name="w"),
    )


def _graph():
    return evoxc.trace(_dot_relu, *_dot_relu_specs())


def _lowered():
    return evoxc.lower(_graph())


def _artifact():
    return evoxc.compile(_lowered())


def _executable():
    return evoxc.load(_artifact())


@evoxc.defn
def _digest_probe(x):
    """The graph a fresh subprocess re-traces for the cross-process digest pin."""
    y = evoxc.add(x, x)
    return evoxc.sum(evoxc.relu(y), axes=(0,))


def _digest_probe_graph():
    return evoxc.trace(_digest_probe, evoxc.TensorSpec((4,), evoxc.float32))


#: ``Graph.summary()``'s documented key set (evoxc/CONTEXT.md "Inspection").
GRAPH_SUMMARY_KEYS = {
    "functions",
    "op_count",
    "op_counts_by_category",
    "tensor_inputs",
    "tensor_outputs",
    "static_values",
    "source_location_count",
    "has_batch_report",
    "digest",
}

#: ``LoweredProgram``/``CompiledArtifact.summary()``'s documented key set.
PROGRAM_SUMMARY_KEYS = {
    "backend",
    "target",
    "payload_kind",
    "payload_bytes",
    "input_count",
    "output_count",
    "input_shapes",
    "output_shapes",
    "input_dtypes",
    "static_value_count",
    "required_custom_ops",
    "runtime_dependencies",
}

#: ``Executable.summary()``'s documented key set.
EXECUTABLE_SUMMARY_KEYS = {
    "backend",
    "device",
    "functions",
    "input_count",
    "output_count",
    "input_specs",
    "output_specs",
    "input_tree",
    "output_tree",
    "static_value_count",
    "required_ops",
    "capabilities",
}

#: Every dtype ``evoxc`` exposes (``evoxc/core/dtypes.py``).
ALL_DTYPES = [
    evoxc.float16,
    evoxc.float32,
    evoxc.float64,
    evoxc.int8,
    evoxc.int16,
    evoxc.int32,
    evoxc.int64,
    evoxc.uint8,
    evoxc.uint16,
    evoxc.uint32,
    evoxc.uint64,
    evoxc.bool_,
    evoxc.complex64,
    evoxc.complex128,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_python(script, *args):
    """Run ``script`` in a fresh interpreter at the repo root (no network)."""
    env = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
    return subprocess.run(
        [sys.executable, "-c", script, *args],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )


class _NoHostPayload:
    """Device payload that FORBIDS host materialization.

    Satisfies the duck-typed payload protocol (``.shape`` / ``.dtype`` /
    ``.device``) but every host-materialization entry point raises
    ``AssertionError`` and counts the attempt, so a ``Tensor.__repr__`` that
    touched device memory would fail loudly and observably.
    """

    def __init__(self, shape, dtype, device):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.device = device
        self.host_calls = 0

    def _refuse(self):
        self.host_calls += 1
        raise AssertionError("Tensor.__repr__ must not materialize device memory")

    def to_host(self):
        self._refuse()

    def __array__(self, dtype=None):
        self._refuse()

    def __dlpack__(self, *args, **kwargs):
        self._refuse()

    def __dlpack_device__(self):
        self._refuse()


# ---------------------------------------------------------------------------
# 1. Graph: repr / summary / digest / text
# ---------------------------------------------------------------------------


def test_graph_repr_reports_counts():
    """``repr`` is the documented one-line structural summary."""
    graph = _graph()
    assert repr(graph) == (
        "<Graph 3 ops, 1 function, 2 tensor inputs, 0 static values, 1 output>"
    )
    pair_graph = evoxc.trace(_pair, evoxc.TensorSpec((3,), evoxc.float32))
    assert repr(pair_graph) == (
        "<Graph 2 ops, 1 function, 1 tensor inputs, 0 static values, 2 outputs>"
    )


def test_graph_summary_key_set_and_json_safety():
    """``summary()`` has exactly the documented keys and is JSON-safe."""
    summary = _graph().summary()
    assert set(summary) == GRAPH_SUMMARY_KEYS
    # JSON-safe: the dumps/loads round trip reproduces the dict exactly.
    assert json.loads(json.dumps(summary)) == summary


def test_graph_summary_describes_io_and_categories():
    """Tensor inputs/outputs, op counts and their per-category split."""
    graph = _graph()
    summary = graph.summary()

    assert summary["functions"] == ["main"]
    assert summary["op_count"] == 3
    assert summary["op_counts_by_category"] == {
        "elementwise": 1,
        "linalg": 1,
        "terminator": 1,
    }
    # Sorted by category, and the categories sum to the op count.
    categories = list(summary["op_counts_by_category"])
    assert categories == sorted(categories)
    assert sum(summary["op_counts_by_category"].values()) == summary["op_count"]

    assert summary["tensor_inputs"] == [
        {"path": [0], "shape": ["2", "3"], "dtype": "float32"},
        {"path": [1], "shape": ["3", "4"], "dtype": "float32"},
    ]
    assert summary["tensor_outputs"] == [
        {"path": [], "shape": ["2", "4"], "dtype": "float32"},
    ]


def test_graph_summary_static_values_truncate_value_repr():
    """Static leaves report path/kind and a ``repr`` truncated to 80 chars."""
    graph = evoxc.trace(
        _with_statics, evoxc.TensorSpec((3,), evoxc.float32), 5, "z" * 200
    )
    summary = graph.summary()
    assert [record["path"] for record in summary["static_values"]] == [[1], [2]]
    assert [record["kind"] for record in summary["static_values"]] == ["int", "str"]
    assert summary["static_values"][0]["value_repr"] == "5"
    long_repr = summary["static_values"][1]["value_repr"]
    assert len(long_repr) == 80
    assert long_repr.startswith("'zzz")


def test_graph_summary_locations_and_batch_report_flags():
    """``source_location_count`` matches the recorded locations; no batch report."""
    graph = _graph()
    summary = graph.summary()
    assert summary["source_location_count"] == len(graph.source_locations)
    assert summary["source_location_count"] > 0
    assert summary["has_batch_report"] is False


def test_graph_inspection_is_repeatable_and_non_mutating():
    """Repeated inspection returns equal values and never changes the graph."""
    graph = _graph()
    digest = graph.digest()
    first = graph.summary()
    second = graph.summary()
    assert first == second
    assert first["digest"] == digest
    assert repr(graph) == repr(graph)
    assert graph.digest() == digest
    assert graph.text() == graph.text()


def test_graph_digest_is_a_sha256_hex_string():
    digest = _graph().digest()
    assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_graph_digest_deterministic_in_process():
    """Two traces of the same defn + specs digest identically."""
    first = _graph().digest()
    second = _graph().digest()
    assert first == second
    assert first == _graph().digest()


def test_graph_digest_changes_when_the_graph_changes():
    """A different op, dtype or static value is a different digest."""
    base = _graph().digest()

    @evoxc.defn
    def _dot_neg(x, w):
        return evoxc.negate(evoxc.dot(x, w))

    assert evoxc.trace(_dot_neg, *_dot_relu_specs()).digest() != base

    other_dtype = (
        evoxc.TensorSpec((2, 3), evoxc.float64, name="x"),
        evoxc.TensorSpec((3, 4), evoxc.float64, name="w"),
    )
    assert evoxc.trace(_dot_relu, *other_dtype).digest() != base

    spec = evoxc.TensorSpec((3,), evoxc.float32)
    assert (
        evoxc.trace(_with_statics, spec, 5, "a").digest()
        != evoxc.trace(_with_statics, spec, 6, "a").digest()
    )


_DIGEST_SUBPROCESS_SCRIPT = (
    "import tests.test_inspection as m\n"
    "print(m._digest_probe_graph().digest())\n"
)


def test_graph_digest_stable_across_processes():
    """The same source re-traced in fresh interpreters digests identically."""
    local = _digest_probe_graph().digest()
    digests = []
    for _ in range(2):
        result = _run_python(_DIGEST_SUBPROCESS_SCRIPT)
        assert result.returncode == 0, result.stderr
        digests.append(result.stdout.strip())
    assert digests[0] == digests[1] == local


# BUG(evoxc): Graph.digest() hashes ir.serialize_module(), whose op entries carry
# each op's source location (file/line/col), so the documented "structure + I/O
# signature only" digest changes when the IDENTICAL graph is defined in another
# file — contradicting "Identical graphs in different processes digest
# identically" (evoxc/trace/graph.py digest docstring). Minimal repro: this test
# writes the same module source to two paths and compares the digests.
@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG(evoxc): the digest embeds per-op source locations (file path), so "
        "an identical graph defined in a different file digests differently"
    ),
)
def test_graph_digest_ignores_source_file_path(tmp_path):
    source = (
        "import evoxc\n"
        "@evoxc.defn\n"
        "def probe(x):\n"
        "    return evoxc.relu(evoxc.add(x, x))\n"
        "g = evoxc.trace(probe, evoxc.TensorSpec((4,), evoxc.float32))\n"
        "print(g.digest())\n"
    )
    digests = []
    for name in ("first_location.py", "second_location.py"):
        script = tmp_path / name
        script.write_text(source)
        result = _run_python(str(script))
        assert result.returncode == 0, result.stderr
        digests.append(result.stdout.strip())
    assert digests[0] == digests[1]


def test_graph_text_is_the_pretty_printed_ir():
    """``text()`` is the pretty-printed IR that ``print()`` writes."""
    graph = _graph()
    text = graph.text()
    assert isinstance(text, str)
    assert text == evoxc.ir.pretty_print(graph.module)
    assert "func @main" in text
    for name in ("evoxc.dot", "evoxc.relu", "evoxc.return"):
        assert name in text


# ---------------------------------------------------------------------------
# 2. LoweredProgram / CompiledArtifact
# ---------------------------------------------------------------------------


def test_lowered_program_repr_and_summary():
    """``repr`` carries backend/payload/leaf counts; ``summary`` its keys."""
    lowered = _lowered()
    assert re.fullmatch(
        r"LoweredProgram\(backend='numpy', payload=serialized-ir \d+ B, "
        r"inputs=2, outputs=1\)",
        repr(lowered),
    )
    summary = lowered.summary()
    assert set(summary) == PROGRAM_SUMMARY_KEYS
    assert json.loads(json.dumps(summary)) == summary


def test_lowered_program_summary_contents():
    """The summary reports the payload kind/size and the recorded signature."""
    summary = _lowered().summary()
    assert summary["backend"] == "numpy"
    assert summary["target"] is None
    assert summary["payload_kind"] == "serialized-ir"
    assert isinstance(summary["payload_bytes"], int) and summary["payload_bytes"] > 0
    assert summary["input_count"] == 2
    assert summary["output_count"] == 1
    assert summary["input_shapes"] == [["2", "3"], ["3", "4"]]
    assert summary["output_shapes"] == [["2", "4"]]
    assert summary["input_dtypes"] == ["float32", "float32"]
    assert summary["static_value_count"] == 0
    assert summary["required_custom_ops"] == []
    assert summary["runtime_dependencies"] == {}


def test_lowered_program_source_text_and_text():
    """``source_text()`` returns exactly what ``text()`` renders (str)."""
    lowered = _lowered()
    source = lowered.source_text()
    assert isinstance(source, str)
    assert source == lowered.text()
    assert "func @main" in source and "evoxc.dot" in source


def test_lowered_program_save_load_roundtrip(tmp_path):
    """The persist round trip preserves the summary and repr-relevant fields."""
    lowered = _lowered()
    path = tmp_path / "lowered.evoxc"
    lowered.save(path)
    loaded = evoxc.LoweredProgram.load(path)
    assert loaded.summary() == lowered.summary()
    assert repr(loaded) == repr(lowered)


def test_lowered_program_source_text_is_none_without_a_text_payload():
    """A payload with no text form: ``source_text`` → None, ``text`` raises."""
    lowered = evoxc.LoweredProgram(backend="numpy", payload={"opaque": 1})
    assert lowered.summary()["payload_kind"] == "json"
    assert "payload=json" in repr(lowered)
    assert lowered.source_text() is None
    with pytest.raises(evoxc.BackendError, match="cannot render lowered payload"):
        lowered.text()


def test_lowered_program_without_a_payload():
    """An absent payload reports ``None`` kind/size instead of inventing one."""
    lowered = evoxc.LoweredProgram(backend="numpy")
    summary = lowered.summary()
    assert summary["payload_kind"] is None
    assert summary["payload_bytes"] is None
    assert summary["input_count"] == 0 and summary["output_count"] == 0
    assert "payload=None" in repr(lowered)
    assert lowered.source_text() is None


def test_compiled_artifact_repr_and_summary():
    """``repr`` adds the compile target; ``summary`` has the documented keys."""
    artifact = _artifact()
    assert re.fullmatch(
        r"CompiledArtifact\(backend='numpy', target='cpu', "
        r"payload=serialized-ir \d+ B, inputs=2, outputs=1\)",
        repr(artifact),
    )
    summary = artifact.summary()
    assert set(summary) == PROGRAM_SUMMARY_KEYS
    assert json.loads(json.dumps(summary)) == summary
    assert summary["backend"] == "numpy"
    assert summary["target"] == "cpu"
    assert summary["payload_kind"] == "serialized-ir"
    assert summary["input_shapes"] == [["2", "3"], ["3", "4"]]
    assert summary["input_dtypes"] == ["float32", "float32"]


def test_compiled_artifact_source_text():
    """``source_text()`` renders the artifact payload's human-readable form."""
    artifact = _artifact()
    source = artifact.source_text()
    assert isinstance(source, str)
    # Same payload, same rendering as the lowered program's ``text()``.
    assert source == _lowered().text()
    assert "func @main" in source and "evoxc.dot" in source


def test_compiled_artifact_save_load_roundtrip(tmp_path):
    """``save`` + ``load`` preserves the summary/repr-relevant fields."""
    artifact = _artifact()
    path = tmp_path / "artifact.evoxc"
    artifact.save(path)
    loaded = evoxc.CompiledArtifact.load(path)
    assert loaded.summary() == artifact.summary()
    assert repr(loaded) == repr(artifact)
    assert loaded.source_text() == artifact.source_text()


def test_compiled_artifact_summary_without_a_signature():
    """A missing signature yields zeroed counts and empty spec lists."""
    artifact = evoxc.CompiledArtifact(backend="numpy", target="cpu")
    summary = artifact.summary()
    assert summary["input_count"] == 0 and summary["output_count"] == 0
    assert summary["input_shapes"] == [] and summary["output_shapes"] == []
    assert summary["input_dtypes"] == []
    assert summary["required_custom_ops"] == []
    assert summary["runtime_dependencies"] == {}


# ---------------------------------------------------------------------------
# 3. Executable: repr / summary / plan_text (numpy + native)
# ---------------------------------------------------------------------------


def test_executable_repr_numpy():
    assert repr(_executable()) == (
        "<Executable backend=numpy device=cpu:0 inputs=2 outputs=1>"
    )


def test_executable_summary_key_set_and_json_safety():
    """``summary()`` has exactly the documented keys and is JSON-safe."""
    summary = _executable().summary()
    assert set(summary) == EXECUTABLE_SUMMARY_KEYS
    assert json.loads(json.dumps(summary)) == summary


def test_executable_summary_contents_numpy():
    """The snapshot reports the backend, device, I/O side and capabilities."""
    summary = _executable().summary()
    assert summary["backend"] == "numpy"
    assert summary["device"] == str(evoxc.Device("cpu", 0))
    assert summary["functions"] == ["main"]
    assert summary["input_count"] == 2 and summary["output_count"] == 1
    assert summary["input_specs"] == [
        {"shape": ["2", "3"], "dtype": "float32", "device": None},
        {"shape": ["3", "4"], "dtype": "float32", "device": None},
    ]
    assert summary["output_specs"] == [
        {"shape": ["2", "4"], "dtype": "float32", "device": None}
    ]
    assert summary["input_tree"] == "tuple of length 2"
    assert summary["output_tree"] == "tensor output leaf"
    assert summary["static_value_count"] == 0
    assert summary["required_ops"] == []

    capabilities = summary["capabilities"]
    assert isinstance(capabilities, dict)
    # ``dtypes`` is reported as a count, never as the concrete dtype set.
    assert capabilities["dtype_count"] == len(ALL_DTYPES)
    assert "dtypes" not in capabilities
    assert isinstance(capabilities["rng_bit_generator"], list)
    assert capabilities["dynamic_shapes"] is True


def test_executable_plan_text_numpy():
    """The numpy interpreter's plan renders as text and repeats identically."""
    executable = _executable()
    text = executable.plan_text()
    assert isinstance(text, str)
    assert text.startswith("function @main")
    for marker in ("dot", "relu", "<return>"):
        assert marker in text
    assert executable.plan_text() == text


def test_executable_inspection_is_pure_numpy(monkeypatch):
    """repr/summary/plan_text run nothing and populate no fast-path cache."""
    executable = _executable()
    backend_executable = executable.backend_executable
    calls = []
    real_run = backend_executable.run

    def spy(*args, **kwargs):
        calls.append(args)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(backend_executable, "run", spy)

    def snapshot():
        return (repr(executable), executable.summary(), executable.plan_text())

    assert executable._input_plan is None and executable._output_plan is None
    first = snapshot()
    second = snapshot()
    assert first == second
    assert calls == []
    assert executable._input_plan is None and executable._output_plan is None
    interpreter = getattr(backend_executable, "_interpreter", None)
    if interpreter is not None:
        assert interpreter._plans == {}

    # A real run afterwards still works and leaves the inspection unchanged.
    out = evoxc.run(
        executable,
        evoxc.tensor(np.zeros((2, 3), dtype=np.float32)),
        evoxc.tensor(np.zeros((3, 4), dtype=np.float32)),
    )
    assert out.shape == (2, 4)
    assert len(calls) == 1
    assert snapshot() == first
    if interpreter is not None:
        assert interpreter._plans  # the run built its own plans


def test_executable_summary_capabilities_none_for_unresolvable_backend():
    """An unknown backend name reports ``capabilities=None`` and never raises."""
    fake = types.SimpleNamespace(
        backend_name="not-a-registered-backend",
        functions=(),
        device=evoxc.Device("cpu", 0),
        run=lambda *args, **kwargs: None,
    )
    executable = evoxc.Executable(fake, None)
    summary = executable.summary()
    assert summary["backend"] == "not-a-registered-backend"
    assert summary["capabilities"] is None


def test_executable_plan_text_none_without_a_backend_plan():
    """A backend executable with no ``plan_text`` renders ``None``."""
    fake = types.SimpleNamespace(
        backend_name="numpy",
        functions=("main",),
        device=evoxc.Device("cpu", 0),
        run=lambda *args, **kwargs: None,
    )
    executable = evoxc.Executable(fake, None)
    assert executable.plan_text() is None
    assert repr(executable) == "<Executable backend=numpy device=cpu:0 inputs=0 outputs=0>"
    summary = executable.summary()
    assert summary["input_tree"] is None and summary["output_tree"] is None
    assert summary["input_specs"] == [] and summary["output_specs"] == []


_NATIVE_SKIP_REASON = (
    "the optional evoxc._evoxc_native extension is not built in this "
    "environment; build it with `scripts/build-native.sh` (the native/ Cargo "
    "workspace) to exercise the native-backend inspection leg"
)

requires_native = pytest.mark.skipif(
    not native_engine.engine_available(), reason=_NATIVE_SKIP_REASON
)


def _native_executable():
    return evoxc.build(_dot_relu, *_dot_relu_specs(), backend="native")


@requires_native
def test_executable_repr_native():
    assert repr(_native_executable()) == (
        "<Executable backend=native device=cpu:0 inputs=2 outputs=1>"
    )


@requires_native
def test_executable_summary_native():
    summary = _native_executable().summary()
    assert set(summary) == EXECUTABLE_SUMMARY_KEYS
    assert json.loads(json.dumps(summary)) == summary
    assert summary["backend"] == "native"
    assert summary["device"] == str(evoxc.Device("cpu", 0))
    assert summary["functions"] == ["main"]
    assert summary["input_count"] == 2 and summary["output_count"] == 1
    assert summary["output_specs"] == [
        {"shape": ["2", "4"], "dtype": "float32", "device": None}
    ]
    assert summary["capabilities"]["dtype_count"] == len(ALL_DTYPES)


@requires_native
def test_executable_plan_text_native():
    """The native engine's dispatch plan renders as text and repeats identically."""
    executable = _native_executable()
    text = executable.plan_text()
    assert isinstance(text, str)
    assert "entry 'main'" in text
    assert "dot" in text and "relu" in text
    assert executable.plan_text() == text


@requires_native
def test_executable_inspection_is_pure_native(monkeypatch):
    """Native inspection calls into the engine's PLAN only, never into ``run``."""
    executable = _native_executable()
    backend_executable = executable.backend_executable
    calls = []

    def spy(*args, **kwargs):
        calls.append(args)
        raise AssertionError("inspection must not execute the program")

    monkeypatch.setattr(backend_executable, "run", spy)
    first = (repr(executable), executable.summary(), executable.plan_text())
    second = (repr(executable), executable.summary(), executable.plan_text())
    assert first == second
    assert calls == []


# ---------------------------------------------------------------------------
# 4. Defn
# ---------------------------------------------------------------------------


def test_defn_repr_names_qualname_and_module():
    assert repr(_dot_relu) == "<Defn _dot_relu (tests.test_inspection)>"


def test_defn_repr_never_traces():
    """``repr`` is metadata only — a body that would explode is never entered."""

    @evoxc.defn
    def _explodes(x):
        raise RuntimeError("repr must never trace this function")

    assert repr(_explodes) == (
        "<Defn test_defn_repr_never_traces.<locals>._explodes "
        "(tests.test_inspection)>"
    )
    # The repr path never calls the Defn either (a call is always a TraceError).
    with pytest.raises(evoxc.TraceError):
        _explodes(evoxc.TensorSpec((3,), evoxc.float32))


def test_defn_repr_falls_back_to_the_function_repr():
    """A callable without a qualname renders through its own repr."""
    wrapped = evoxc.defn(functools.partial(len))
    text = repr(wrapped)
    assert text.startswith("<Defn functools.partial(")
    assert "(functools)>" in text


# ---------------------------------------------------------------------------
# 5. Tensor: repr is inspection only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ALL_DTYPES, ids=[str(d) for d in ALL_DTYPES])
def test_tensor_repr_covers_every_dtype(dtype):
    """All 14 dtypes render shape/dtype/device; host data gets a preview."""
    values = np.zeros((3,), dtype=dtype)
    host = core.Tensor(values)
    text = repr(host)
    assert text.startswith(f"Tensor(shape=(3,), dtype={np.dtype(dtype)}, device=cpu:0")
    assert "data=" in text

    payload = _NoHostPayload((3,), dtype, evoxc.Device("cuda", 0))
    device_tensor = core.Tensor(payload)
    device_text = repr(device_tensor)
    assert device_text == (
        f"Tensor(shape=(3,), dtype={np.dtype(dtype)}, device=cuda:0, "
        "payload=_NoHostPayload)"
    )
    # Metadata only: the payload was never asked for a host copy.
    assert payload.host_calls == 0
    assert device_tensor.data is payload


def test_tensor_repr_zero_dim():
    assert repr(core.Tensor(np.array(5.0, dtype=np.float32))) == (
        "Tensor(shape=(), dtype=float32, device=cpu:0, data=[5.])"
    )
    payload = _NoHostPayload((), evoxc.float32, evoxc.Device("cuda", 1))
    assert repr(core.Tensor(payload)) == (
        "Tensor(shape=(), dtype=float32, device=cuda:1, payload=_NoHostPayload)"
    )


@pytest.mark.parametrize("shape", [(0, 3), (0,)], ids=["zero-by-three", "empty"])
def test_tensor_repr_zero_element_shapes(shape):
    text = repr(core.Tensor(np.zeros(shape, dtype=np.int64)))
    assert text.startswith(
        f"Tensor(shape={shape}, dtype=int64, device=cpu:0"
    )
    assert "data=[]" in text


def test_tensor_repr_high_rank_has_no_preview():
    text = repr(core.Tensor(np.zeros((1, 2, 3, 4, 5), dtype=np.float16)))
    assert text == "Tensor(shape=(1, 2, 3, 4, 5), dtype=float16, device=cpu:0)"


@pytest.mark.parametrize(
    "size,preview",
    [(8, True), (9, False)],
    ids=["at-threshold", "above-threshold"],
)
def test_tensor_repr_preview_threshold(size, preview):
    text = repr(core.Tensor(np.zeros(size, dtype=np.float32)))
    assert ("data=" in text) is preview


@pytest.mark.parametrize(
    "values", [np.array(["a", "b"]), np.array([{"a": 1}, {"b": 2}], dtype=object)],
    ids=["unicode", "object"],
)
def test_tensor_repr_non_numeric_dtypes_have_no_preview(values):
    text = repr(core.Tensor(values))
    assert "data=" not in text
    assert f"shape={values.shape}" in text


def test_tensor_repr_never_materializes_a_device_payload():
    """A payload that refuses host access still renders — metadata only."""
    payload = _NoHostPayload((2, 3), evoxc.float64, evoxc.Device("rocm", 2))
    tensor = core.Tensor(payload)
    text = repr(tensor)
    assert text == (
        "Tensor(shape=(2, 3), dtype=float64, device=rocm:2, payload=_NoHostPayload)"
    )
    assert payload.host_calls == 0


def test_tensor_repr_never_raises():
    """An unrepresentable payload falls back to metadata-free text."""
    payload = _NoHostPayload((2,), evoxc.float32, evoxc.Device("cuda", 0))
    tensor = core.Tensor(payload)
    payload.shape = None  # break the payload AFTER construction
    assert repr(tensor) == "Tensor(<unrepresentable _NoHostPayload>)"


# ---------------------------------------------------------------------------
# 6. BoundExecutable
# ---------------------------------------------------------------------------


def _bound_executable():
    executable = _executable()
    bound = evoxc.bind(executable, w=evoxc.ones((3, 4), dtype=evoxc.float32))
    return executable, bound


def test_bound_executable_repr_reports_remaining_inputs():
    _, bound = _bound_executable()
    assert repr(bound) == (
        "<BoundExecutable backend=numpy device=cpu:0 inputs=1 outputs=1 bound=1>"
    )
    all_bound = evoxc.bind(
        _executable(),
        x=evoxc.zeros((2, 3), dtype=evoxc.float32),
        w=evoxc.ones((3, 4), dtype=evoxc.float32),
    )
    assert repr(all_bound) == (
        "<BoundExecutable backend=numpy device=cpu:0 inputs=0 outputs=1 bound=2>"
    )


def test_bound_executable_summary_is_the_delegated_snapshot_plus_bindings():
    """The wrapped snapshot, with ``bound_count`` and the REDUCED input side."""
    executable, bound = _bound_executable()
    expected = dict(executable.summary())
    expected["bound_count"] = 1
    expected["input_count"] = 1
    expected["input_specs"] = expected["input_specs"][:1]
    summary = bound.summary()
    assert summary == expected
    # The full recorded input tree stays (only the spec list is reduced).
    assert summary["input_tree"] == executable.summary()["input_tree"]
    assert json.loads(json.dumps(summary)) == summary
    assert set(summary) == EXECUTABLE_SUMMARY_KEYS | {"bound_count"}


def test_bound_executable_plan_text_delegates():
    executable, bound = _bound_executable()
    assert bound.plan_text() == executable.plan_text()
    assert isinstance(bound.plan_text(), str)


def test_bound_executable_inspection_is_pure(monkeypatch):
    """Bound repr/summary/plan_text never run the program either."""
    executable, bound = _bound_executable()
    calls = []
    real_run = executable.backend_executable.run

    def spy(*args, **kwargs):
        calls.append(args)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(executable.backend_executable, "run", spy)
    first = (repr(bound), bound.summary(), bound.plan_text())
    second = (repr(bound), bound.summary(), bound.plan_text())
    assert first == second
    assert calls == []
    assert bound._input_plan is None
