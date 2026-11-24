"""The numpy backend's SINGLE-DECODE staging property (``decode_stats()`` seam).

Contract under test (``evoxc/backends/numpy/CONTEXT.md``, "Single-decode staging
(in-process module reuse)" and the docstrings of ``_decode_payload`` /
``decode_stats`` / ``reset_decode_stats``):

* The in-process explicit pipeline ``trace -> lower -> compile -> load`` performs
  ZERO ``ir.deserialize_module`` calls — ``lower`` attaches the LIVE module (plus
  the payload object and the collected custom-op names) to the
  ``LoweredProgram`` and ``compile``/``load`` reuse it (and the artifact's
  private module attachment) instead of decoding the payload back.
* A disk round-trip decodes EXACTLY ONCE, at the first stage that needs the IR:
  ``LoweredProgram.load(p)`` + ``compile`` => one decode in ``compile``;
  ``CompiledArtifact.load(p)`` + ``load`` => one decode in ``load``. Later hops
  (``load`` after ``compile``, ``run`` after ``load``) add none.
* ``_decode_payload`` is the SINGLE ``ir.deserialize_module`` call site, so the
  counter is exact; the private attachments are never declared dataclass fields
  (``save()`` persists only declared fields, so the containers stay
  byte-identical) and a disk-loaded object carries none of them.
* The persisted payload is the snapshot of record: mutating the LIVE module
  after ``lower`` neither changes the saved bytes nor what a disk-loaded
  artifact computes.

Every assertion is about that documented contract — never about timing. This
module needs numpy only (no compiler, no GPU, no PJRT plugin).
"""

import dataclasses
import os
import subprocess
import sys

import numpy as np
import pytest

import evoxc
from evoxc import core
from evoxc.backends import numpy as numpy_backend

STATS = numpy_backend.decode_stats
RESET = numpy_backend.reset_decode_stats

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(evoxc.__file__)))

#: Private, runtime-only attachments the numpy backend puts on its staging
#: objects. They are NEVER declared dataclass fields and NEVER persisted.
_ATTACHMENTS = ("_numpy_module", "_numpy_payload", "_numpy_required_custom_ops")


@pytest.fixture(autouse=True)
def _clean_decode_window(monkeypatch):
    """Zero the process-global counter around every test (order independence).

    The counter is process-global and the staging defaults can be overridden by
    ``EVOXC_BACKEND``/``EVOXC_DEVICE``/``EVOXC_TARGET_BACKENDS``; clearing those
    keeps the file's "numpy, in-process" statements true on any ambient box.
    """
    for var in ("EVOXC_BACKEND", "EVOXC_DEVICE", "EVOXC_TARGET_BACKENDS"):
        monkeypatch.delenv(var, raising=False)
    RESET()
    try:
        yield
    finally:
        RESET()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def fn(x, w):
    """The tiny graph under test: ``sum(tanh(dot(x, w)), axes=1)``."""
    return evoxc.sum(evoxc.tanh(evoxc.dot(x, w)), axes=1)


def specs():
    """TensorSpec tree for ``fn`` (static shapes, f32)."""
    return (
        core.TensorSpec(shape=(3, 4), dtype=np.float32),
        core.TensorSpec(shape=(4, 2), dtype=np.float32),
    )


def inputs():
    """Fixed-seed host inputs, distinct enough that a wrong graph is visible."""
    rng = np.random.default_rng(0)
    x = evoxc.Tensor(rng.standard_normal((3, 4)).astype(np.float32))
    w = evoxc.Tensor(rng.standard_normal((4, 2)).astype(np.float32))
    return x, w


def reference(x, w):
    """The pure-numpy reference (re-implements the kernels' formulas)."""
    return np.sum(np.tanh(x.numpy() @ w.numpy()), axis=1, dtype=np.float32)


def assert_parity(out, expected):
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)


def traced_graph():
    return evoxc.trace(fn, *specs())


# ---------------------------------------------------------------------------
# 1-4: the IN-PROCESS pipeline never deserializes
# ---------------------------------------------------------------------------


def test_explicit_pipeline_decodes_nothing():
    """``trace -> lower -> compile -> load -> run`` makes ZERO decodes."""
    graph = traced_graph()
    assert STATS()["deserialize"] == 0  # nothing staged yet

    lowered = evoxc.lower(graph, backend="numpy")
    assert STATS() == {"deserialize": 0}, "lower must not deserialize"

    artifact = evoxc.compile(lowered, backend="numpy")
    assert STATS() == {"deserialize": 0}, "compile reuses lower's live module"

    executable = evoxc.load(artifact, backend="numpy")
    assert STATS() == {"deserialize": 0}, "load reuses the artifact's module"

    assert isinstance(executable.backend_executable, numpy_backend.NumpyExecutable)
    assert executable.backend == "numpy"
    x, w = inputs()
    out = evoxc.run(executable, x, w)
    assert_parity(out, reference(x, w))
    assert STATS() == {"deserialize": 0}, "run executes the live module"


@pytest.mark.parametrize("backend", ["numpy", None])
def test_build_decodes_nothing(backend):
    """``evoxc.build`` (explicit numpy or the default) adds no decode."""
    executable = evoxc.build(fn, *specs(), backend=backend)
    assert STATS() == {"deserialize": 0}
    x, w = inputs()
    assert_parity(evoxc.run(executable, x, w), reference(x, w))
    assert STATS() == {"deserialize": 0}, "a run never decodes"


def test_evaluate_decodes_nothing():
    """``evoxc.evaluate`` (build+run shorthand) makes ZERO decodes."""
    x, w = inputs()
    out = evoxc.evaluate(fn, x, w, backend="numpy")
    assert STATS() == {"deserialize": 0}
    assert_parity(out, reference(x, w))
    assert STATS() == {"deserialize": 0}


# ---------------------------------------------------------------------------
# 5-6: a disk round-trip decodes EXACTLY ONCE, at the first stage that needs IR
# ---------------------------------------------------------------------------


def test_lowered_disk_round_trip_decodes_once_at_compile(tmp_path):
    """``LoweredProgram.load`` decodes nothing; ``compile`` decodes exactly once."""
    lowered = evoxc.lower(traced_graph(), backend="numpy")
    path = tmp_path / "lowered.evox"
    lowered.save(path)

    RESET()
    reloaded = numpy_backend.LoweredProgram.load(path)
    assert STATS() == {"deserialize": 0}, (
        "loading the container must not decode the IR payload"
    )

    artifact = evoxc.compile(reloaded, backend="numpy")
    assert STATS() == {"deserialize": 1}, "compile is the FIRST stage needing the IR"

    executable = evoxc.load(artifact, backend="numpy")
    assert STATS() == {"deserialize": 1}, "load must not decode a second time"

    x, w = inputs()
    assert_parity(evoxc.run(executable, x, w), reference(x, w))
    assert STATS() == {"deserialize": 1}, "run must not decode"


def test_artifact_disk_round_trip_decodes_once_at_load(tmp_path):
    """``CompiledArtifact.load`` decodes nothing; ``load`` decodes exactly once."""
    artifact = evoxc.compile(
        evoxc.lower(traced_graph(), backend="numpy"), backend="numpy"
    )
    path = tmp_path / "artifact.evox"
    artifact.save(path)

    RESET()
    reloaded = numpy_backend.CompiledArtifact.load(path)
    assert STATS() == {"deserialize": 0}

    executable = evoxc.load(reloaded, backend="numpy")
    assert STATS() == {"deserialize": 1}, "load is the FIRST stage needing the IR"

    x, w = inputs()
    assert_parity(evoxc.run(executable, x, w), reference(x, w))
    assert STATS() == {"deserialize": 1}


# ---------------------------------------------------------------------------
# 7-9: the private attachments live only on the in-process objects
# ---------------------------------------------------------------------------


def test_private_attachments_are_not_dataclass_fields():
    """``save()`` writes only declared fields — the attachments are not fields."""
    lowered = evoxc.lower(traced_graph(), backend="numpy")
    artifact = evoxc.compile(lowered, backend="numpy")

    lowered_fields = {field.name for field in dataclasses.fields(lowered)}
    artifact_fields = {field.name for field in dataclasses.fields(artifact)}
    assert lowered_fields == {"backend", "signature", "payload"}
    assert artifact_fields == {
        "backend",
        "signature",
        "target",
        "payload",
        "required_custom_ops",
        "runtime_dependencies",
    }
    for name in _ATTACHMENTS:
        assert name not in lowered_fields
        assert name not in artifact_fields


def test_in_process_objects_carry_the_attachments():
    """The in-process objects DO have the live module + the exact payload."""
    graph = traced_graph()
    lowered = evoxc.lower(graph, backend="numpy")
    artifact = evoxc.compile(lowered, backend="numpy")

    assert lowered._numpy_module is graph.module
    assert lowered._numpy_payload is lowered.payload
    assert isinstance(lowered._numpy_required_custom_ops, tuple)

    assert artifact._numpy_module is graph.module
    assert artifact._numpy_payload is artifact.payload
    assert artifact._numpy_payload is lowered.payload


def test_disk_loaded_objects_have_no_attachment(tmp_path):
    """A container rebuilt from disk carries NO live module attachment."""
    lowered = evoxc.lower(traced_graph(), backend="numpy")
    artifact = evoxc.compile(lowered, backend="numpy")
    lowered_path = tmp_path / "lowered.evox"
    artifact_path = tmp_path / "artifact.evox"
    lowered.save(lowered_path)
    artifact.save(artifact_path)

    for reloaded in (
        numpy_backend.LoweredProgram.load(lowered_path),
        numpy_backend.CompiledArtifact.load(artifact_path),
    ):
        for name in _ATTACHMENTS:
            assert getattr(reloaded, name, None) is None, (
                f"{type(reloaded).__name__} loaded from disk must not carry {name}"
            )


# ---------------------------------------------------------------------------
# 10-11: the persisted container is byte-identical across a round-trip
# ---------------------------------------------------------------------------


def test_lowered_container_is_byte_identical_after_round_trip(tmp_path):
    lowered = evoxc.lower(traced_graph(), backend="numpy")
    first = tmp_path / "first.evox"
    second = tmp_path / "second.evox"
    lowered.save(first)
    numpy_backend.LoweredProgram.load(first).save(second)
    assert first.read_bytes() == second.read_bytes()


def test_artifact_container_is_byte_identical_after_round_trip(tmp_path):
    artifact = evoxc.compile(
        evoxc.lower(traced_graph(), backend="numpy"), backend="numpy"
    )
    first = tmp_path / "first.evox"
    second = tmp_path / "second.evox"
    artifact.save(first)
    numpy_backend.CompiledArtifact.load(first).save(second)
    assert first.read_bytes() == second.read_bytes()


# ---------------------------------------------------------------------------
# 12: the payload is the snapshot of record (live-module mutation is invisible)
# ---------------------------------------------------------------------------


def test_live_module_mutation_does_not_change_saved_artifact(tmp_path):
    """Mutating the live module after ``lower`` changes neither bytes nor values."""
    graph = traced_graph()
    lowered = evoxc.lower(graph, backend="numpy")

    before = tmp_path / "before.evox"
    after = tmp_path / "after.evox"
    lowered.save(before)

    assert graph.module.functions[0].name == "main"
    graph.module.functions[0].name = "renamed_main"
    lowered.save(after)

    assert before.read_bytes() == after.read_bytes(), (
        "the saved payload is the snapshot taken at lower() time"
    )

    # The disk-loaded artifact still runs the ORIGINAL program.
    reloaded = numpy_backend.LoweredProgram.load(before)
    executable = evoxc.load(
        evoxc.compile(reloaded, backend="numpy"), backend="numpy"
    )
    x, w = inputs()
    assert_parity(evoxc.run(executable, x, w), reference(x, w))
    assert STATS() == {"deserialize": 1}, "one decode for the disk round-trip"


# ---------------------------------------------------------------------------
# 13: a saved artifact loads and runs in a FRESH interpreter
# ---------------------------------------------------------------------------


def test_saved_artifact_runs_in_a_fresh_subprocess(tmp_path):
    artifact = evoxc.compile(
        evoxc.lower(traced_graph(), backend="numpy"), backend="numpy"
    )
    artifact_path = tmp_path / "artifact.evox"
    artifact.save(artifact_path)

    x, w = inputs()
    # Keys sort into the graph's input leaf order (x, w).
    input_path = tmp_path / "inputs.npz"
    np.savez(input_path, input_0=x.numpy(), input_1=w.numpy())
    output_path = tmp_path / "out.npy"

    script = f"""
import json
import numpy as np
import evoxc

art = evoxc.backends.CompiledArtifact.load({str(artifact_path)!r})
exe = evoxc.load(art, backend="numpy")
z = np.load({str(input_path)!r})
out = evoxc.run(exe, *[z[k] for k in sorted(z.files)])
np.save({str(output_path)!r}, out.numpy())
print(json.dumps(evoxc.backends.numpy.decode_stats()))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=os.environ.copy(),
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"fresh-process load/run failed (rc={proc.returncode})\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert '"deserialize": 1' in proc.stdout, (
        f"a fresh-process disk load must decode exactly once; stdout:\n{proc.stdout}"
    )
    np.testing.assert_allclose(
        np.load(output_path), reference(x, w), rtol=1e-6, atol=1e-6
    )


# ---------------------------------------------------------------------------
# 14-15: the counter tracks the seam exactly and is never reset behind us
# ---------------------------------------------------------------------------


def test_two_disk_cycles_accumulate_two_decodes(tmp_path):
    artifact = evoxc.compile(
        evoxc.lower(traced_graph(), backend="numpy"), backend="numpy"
    )
    path = tmp_path / "artifact.evox"
    artifact.save(path)

    RESET()
    x, w = inputs()
    for expected in (1, 2):
        reloaded = numpy_backend.CompiledArtifact.load(path)
        assert STATS() == {"deserialize": expected - 1}
        executable = evoxc.load(reloaded, backend="numpy")
        assert STATS() == {"deserialize": expected}, (
            "each disk load cycle decodes exactly once"
        )
        assert_parity(evoxc.run(executable, x, w), reference(x, w))
        assert STATS() == {"deserialize": expected}


def test_decode_payload_is_the_single_counting_call_site():
    """``_decode_payload`` is the one counting call site (the seam is exact)."""
    lowered = evoxc.lower(traced_graph(), backend="numpy")
    RESET()

    module = numpy_backend._decode_payload(lowered.payload)

    assert STATS() == {"deserialize": 1}
    assert module.functions[0].name == "main"

    RESET()
    assert STATS() == {"deserialize": 0}
