"""xla PJRT-plugin AUTODISCOVERY at the ADAPTER / PIPELINE boundary.

Companion of ``test_xla_plugin_discovery.py``: that file pins the resolution
surface (precedence, candidate hygiene, caching, the opt-out, no jax imports,
the platform->kind rule) ONE STEP AT A TIME. This file pins the three
end-to-end consequences that only appear once a whole ``evoxc.build`` runs
against a discovered plugin — all of them regressions of the work that made
discovery automatic:

* **Registration is unconditional.** ``xla.register()`` performs NO plugin
  probe: it is a registry insert only, because ``plugin_path`` is a PER-CALL
  option a classmethod probe cannot see. ``evoxc.backends.get("xla")`` must
  therefore succeed in a process where NO plugin is discoverable (no
  ``plugin_path`` — registration cannot see one —, no ``EVOXC_PJRT_PLUGIN``,
  empty well-known paths, ``EVOXC_PJRT_PLUGIN_DISCOVERY=0``), and the
  actionable four-step ``core.BackendError`` message must come from
  ``XlaBackend.check_available()`` (the explicit option-less preflight) and
  from the PIPELINE (``evoxc.build``) — never from registration.
* **``plugin_path`` alone is enough.** With the same neutered environment,
  ``evoxc.build(fn, spec, backend="xla", device="cpu", plugin_path=<.so>)``
  must build AND run. This is the regression pin for the option-forwarding
  bug: ``build`` used to drop its resolved options on the ``load`` hop, so
  the load stage re-discovered the plugin with NO options and failed even
  though ``compile`` had just succeeded with the explicit path.
* **Every stage receives the resolved options.** A recording stub backend
  pins that ``evoxc.build`` — and ``evoxc.build_cached`` on its ONE miss —
  forwards the same resolved option dict to ``lower``, ``compile`` AND
  ``load`` (the bug was structural in the pipeline, so it is pinned without
  a compiler: the stub is the whole backend).

It also pins the DRIVER-level failure cleanup that the device-complete-event
work needed: a failed ``PJRT_LoadedExecutable_Execute`` may already have
written REAL output pointers into the caller's REUSED scaffolding, so
``_LoadedExecutable.execute`` must zero every output slot before raising —
otherwise the next call could wrap a stale pointer as a ``_Buffer``. The
TEST-ONLY fake plugin (``_fake_pjrt_plugin.c``, gcc at test time) grew the
``EVOXC_FAKE_PJRT_FAIL_AFTER_OUTPUTS`` knob for exactly this: it fills N
caller slots and THEN returns a ``PJRT_Error``.

Isolation (binding): the two environment-sensitive cases run in a SUBPROCESS
that neuters discovery itself (empty ``_DEFAULT_PLUGIN_PATHS``, the opt-out
env var, a cache reset, ``EVOXC_PJRT_PLUGIN`` unset, every ambient
``EVOXC_*`` variable stripped), so this file behaves identically on a
developer box with a REAL ``jax_plugins`` plugin installed. The in-process
cases use explicit ``plugin_path`` / a throwaway stub backend and never
depend on — or disturb — the ambient plugin state.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import evoxc
from evoxc.backends.adapters import xla_util

from tests.backends.test_pjrt_ctypes_plugin import _build_plugin

#: Repository root: the subprocesses import ``evoxc`` from the checkout (the
#: pytest conftest's ``sys.path`` shim does not reach a child process).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The plugin the fake computes: ``(2, 3)`` f32 ``add(x, x)``. The fake
#: performs NO computation (zero-filled outputs of the declared shape/dtype),
#: so shape/dtype/plumbing are what these pins assert — numerical parity lives
#: in ``test_adapter_xla.py`` against a real plugin.
SHAPE = (2, 3)

#: The env knobs this file drives (module-level so a typo names the wrong
#: variable loudly instead of silently passing).
PLUGIN_ENV = xla_util._ENV_PLUGIN
DISCOVERY_ENV = xla_util._ENV_PLUGIN_DISCOVERY
FAIL_AFTER_OUTPUTS_ENV = "EVOXC_FAKE_PJRT_FAIL_AFTER_OUTPUTS"
FAIL_AFTER_OUTPUTS_MESSAGE = (
    "injected fake failure after writing output buffers"
)


# ---------------------------------------------------------------------------
# subprocess harness (ambient-box isolation)
# ---------------------------------------------------------------------------


def _run_script(script, *args):
    """Run a discovery script in a child interpreter; return its summary dict.

    The child gets a CLEAN ``EVOXC_*`` environment (every ambient option /
    discovery variable stripped), this checkout on ``PYTHONPATH`` and the
    repository root as cwd, so nothing the parent process inherited — a real
    ``jax_plugins`` plugin, an ``EVOXC_PJRT_PLUGIN`` from the developer's
    shell, an ``EVOXC_DEVICE`` default — can influence the outcome. Each
    script neuters whatever discovery sources remain and prints one
    ``SUMMARY:<json>`` line; an assertion failure inside the child surfaces
    here as a non-zero exit with its traceback in stderr.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("EVOXC_")
    }
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPO_ROOT), env.get("PYTHONPATH")) if part
    )
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), *[str(a) for a in args]],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=600,
    )
    assert proc.returncode == 0, (
        "the xla autodiscovery subprocess failed "
        f"(rc={proc.returncode})\n--- stdout ---\n{proc.stdout}"
        f"\n--- stderr ---\n{proc.stderr}"
    )
    for line in proc.stdout.splitlines():
        if line.startswith("SUMMARY:"):
            return json.loads(line[len("SUMMARY:"):])
    raise AssertionError(
        f"the subprocess printed no SUMMARY line; stdout:\n{proc.stdout}"
    )


def _neuter_discovery_prelude():
    """The child-side ``import os, sys`` + discovery-neutering prologue.

    Kept as text (not a helper module) so the child runs the EXACT lines the
    docstring describes: an empty ``_DEFAULT_PLUGIN_PATHS`` tuple, the
    ``EVOXC_PJRT_PLUGIN_DISCOVERY=0`` opt-out and a discovery-cache reset,
    with ``EVOXC_PJRT_PLUGIN`` additionally asserted absent.
    """
    return """
        import json
        import os
        import sys

        import numpy as np

        assert "EVOXC_PJRT_PLUGIN" not in os.environ, (
            "the harness must run with EVOXC_PJRT_PLUGIN unset"
        )

        import evoxc
        from evoxc.backends.adapters import xla, xla_util

        # Neuter EVERY discovery source registration/compile/load could see:
        # no ambient plugin env var (asserted above), no well-known paths, no
        # jax_plugins scan. ``plugin_path`` is the only remaining source — and
        # a classmethod probe cannot see it, which is the whole point.
        xla_util._DEFAULT_PLUGIN_PATHS = ()
        os.environ["EVOXC_PJRT_PLUGIN_DISCOVERY"] = "0"
        xla_util._reset_plugin_discovery_cache()

        def fn(x):
            return evoxc.add(x, x)

        spec = evoxc.TensorSpec((2, 3), evoxc.float32)
    """


_MISSING_PLUGIN_TOKENS = (
    "GetPjRtApi",
    "plugin_path",
    "EVOXC_PJRT_PLUGIN",
    "jax-rocm7-pjrt",
    "bazel build",
)


def _assert_actionable_message(message):
    """The four-step "no plugin" message, token by token."""
    for token in _MISSING_PLUGIN_TOKENS:
        assert token in message, f"{token!r} missing from: {message}"
    assert message == xla_util._PLUGIN_MISSING_MESSAGE


# ---------------------------------------------------------------------------
# (1) registration is UNCONDITIONAL; the preflight and the pipeline are loud
# ---------------------------------------------------------------------------


def test_registration_is_unconditional_and_check_available_is_the_preflight():
    """``get("xla")`` registers without a plugin; the probes raise the message.

    A registry insert must never depend on a plugin being discoverable:
    ``plugin_path`` is a per-call option a classmethod probe cannot see, so a
    registration-time refusal would make that option unusable. The actionable
    error belongs to ``XlaBackend.check_available()`` (the documented
    option-less preflight) — and, for a real pipeline, to the stage that
    actually has the options.
    """
    summary = _run_script(
        _neuter_discovery_prelude()
        + """
        summary = {}

        backend = evoxc.backends.get("xla")  # MUST NOT raise
        summary["backend_name"] = backend.name
        summary["backend_class"] = type(backend).__name__
        summary["is_xla_backend"] = isinstance(backend, xla.XlaBackend)

        try:
            backend.check_available()
        except evoxc.BackendError as exc:
            summary["check_available"] = str(exc)
        else:
            raise AssertionError(
                "check_available() must raise when no plugin is discoverable"
            )

        print("SUMMARY:" + json.dumps(summary))
        """
    )
    assert summary["backend_name"] == "xla"
    assert summary["backend_class"] == "XlaBackend"
    assert summary["is_xla_backend"] is True
    _assert_actionable_message(summary["check_available"])


def test_pipeline_reports_the_actionable_missing_plugin_message():
    """``evoxc.build`` (no option) raises the SAME message — from the pipeline.

    The failure must come from a staging call that can carry options (here
    ``compile``), not from registration and not from a silent fallback.
    """
    summary = _run_script(
        _neuter_discovery_prelude()
        + """
        summary = {}
        try:
            evoxc.build(fn, spec, backend="xla", device="cpu")
        except evoxc.BackendError as exc:
            summary["build"] = str(exc)
        else:
            raise AssertionError(
                "evoxc.build(..., backend='xla') must raise when no plugin is "
                "discoverable"
            )
        print("SUMMARY:" + json.dumps(summary))
        """
    )
    _assert_actionable_message(summary["build"])


# ---------------------------------------------------------------------------
# (2) plugin_path ALONE is enough (the option-forwarding regression pin)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fake_plugin(tmp_path_factory):
    """The standard fake PJRT plugin (zero-filled outputs, one device).

    Built at test time with gcc; skips like the sibling fake-plugin suites
    when no C compiler is available.
    """
    return _build_plugin(
        tmp_path_factory.mktemp("xla_autodiscovery_plugin"),
        "fake_pjrt_plugin.so",
    )


def test_plugin_path_option_alone_builds_and_runs(fake_plugin):
    """``build(..., plugin_path=...)`` succeeds with NO other plugin source.

    Regression pin (binding): ``evoxc.build`` forwards its resolved options to
    every stage, INCLUDING ``load``. When the load hop dropped them, the load
    stage re-discovered the plugin without ``plugin_path`` and raised the
    "no plugin" error even though ``compile`` had just succeeded — with no
    env var, no well-known path and the scan disabled, the explicit option is
    the only thing that can make this build work.

    The fake performs no computation, so the run output is the fake's
    zero-fill of the graph's declared shape/dtype (exactly what the sibling
    ``test_pjrt_ctypes_plugin`` asserts); shape/dtype/plumbing prove the whole
    discovered-plugin path ran.
    """
    summary = _run_script(
        _neuter_discovery_prelude()
        + """
        plugin_path = sys.argv[1]
        summary = {"plugin_path": plugin_path}

        # Anti-vacuity: without the option NOTHING is discoverable here, so a
        # success below can only come from the option reaching the stages.
        try:
            evoxc.build(fn, spec, backend="xla", device="cpu")
        except evoxc.BackendError as exc:
            summary["without_option"] = str(exc)
        else:
            raise AssertionError(
                "without plugin_path no plugin must be discoverable — the "
                "environment was supposed to be neutered"
            )

        exe = evoxc.build(
            fn, spec, backend="xla", device="cpu", plugin_path=plugin_path
        )
        summary["functions"] = list(exe.functions)
        summary["device"] = str(exe.device)
        out = evoxc.run(exe, np.zeros((2, 3), np.float32))
        summary["shape"] = list(out.shape)
        summary["dtype"] = out.dtype.name
        summary["values"] = np.asarray(out.numpy()).tolist()
        print("SUMMARY:" + json.dumps(summary))
        """,
        fake_plugin,
    )
    _assert_actionable_message(summary["without_option"])
    assert summary["functions"] == ["main"]
    assert summary["device"] == "Device(kind='cpu', index=0)"
    assert summary["shape"] == list(SHAPE)
    assert summary["dtype"] == "float32"
    np.testing.assert_array_equal(
        np.asarray(summary["values"], dtype="float32"), np.zeros(SHAPE, "float32")
    )


# ---------------------------------------------------------------------------
# (3) build / build_cached forward the resolved options to EVERY stage
# ---------------------------------------------------------------------------


class _RecordingBackend(evoxc.backends.Backend):
    """Pass-through stub recording the option dict each stage receives.

    ``lower``/``compile``/``load`` delegate to the numpy reference backend
    (renaming the staged objects) so the returned executable actually runs —
    the subject under test is WHICH options reach each hop, not any compiler.
    The name is deliberately NOT ``iree``-family, so the pipeline adds no
    ``target_backends`` inference and the recorded dicts are exactly the
    caller's options.
    """

    name = "xla-options-recorder"
    capabilities = evoxc.backends.numpy_backend.capabilities

    def __init__(self):
        self.calls = {"lower": [], "compile": [], "load": []}

    def lower(self, graph, options=None):
        self.calls["lower"].append(dict(options or {}))
        lowered = evoxc.backends.numpy_backend.lower(graph, options)
        return evoxc.backends.LoweredProgram(
            backend=self.name,
            signature=lowered.signature,
            payload=lowered.payload,
        )

    def compile(self, lowered, options=None):
        self.calls["compile"].append(dict(options or {}))
        inner = evoxc.backends.numpy_backend.compile(
            evoxc.backends.LoweredProgram(
                backend="numpy",
                signature=lowered.signature,
                payload=lowered.payload,
            ),
            options,
        )
        return evoxc.backends.CompiledArtifact(
            backend=self.name,
            signature=inner.signature,
            target=inner.target,
            payload=inner.payload,
            required_custom_ops=inner.required_custom_ops,
            runtime_dependencies=inner.runtime_dependencies,
        )

    def load(self, artifact, device=None, options=None):
        self.calls["load"].append(dict(options or {}))
        inner = evoxc.backends.CompiledArtifact(
            backend="numpy",
            signature=artifact.signature,
            target=artifact.target,
            payload=artifact.payload,
            required_custom_ops=artifact.required_custom_ops,
            runtime_dependencies=artifact.runtime_dependencies,
        )
        return evoxc.backends.numpy_backend.load(inner, device, options)


@pytest.fixture
def recording_backend():
    """Register the recording stub for one test and restore the registry.

    Direct registry assignment (save/restore of the previous entry, exactly
    like the pipeline-env-defaults suite) keeps this file independent of
    whatever a real ``xla`` adapter registered earlier in the process. The
    stub name is unique to this module, so no other backend is shadowed.
    """
    from evoxc.backends import registry as _registry

    stub = _RecordingBackend()
    previous = _registry._registry.get(stub.name)
    _registry._registry[stub.name] = stub
    try:
        yield stub
    finally:
        if previous is None:
            _registry._registry.pop(stub.name, None)
        else:
            _registry._registry[stub.name] = previous


#: The sentinel keys every stage must see. A plain str / int so the same dict
#: is also a valid ``build_cached`` key component (codec-encodable).
SENTINEL_OPTIONS = {"marker_option": "sentinel-value", "marker_level": 7}

SPEC = evoxc.TensorSpec(SHAPE, evoxc.float32)


def _double(x):
    return evoxc.add(x, x)


def _assert_sentinel_reached_every_stage(backend):
    """Each stage got ONE call carrying the sentinel keys, all identical."""
    recorded = {}
    for stage in ("lower", "compile", "load"):
        assert backend.calls[stage], f"the {stage} stage was never called"
        recorded[stage] = backend.calls[stage][0]
        for key, value in SENTINEL_OPTIONS.items():
            assert recorded[stage].get(key) == value, (
                f"the {stage} stage did not receive {key}={value!r} — "
                f"got {recorded[stage]!r}"
            )
    assert recorded["lower"] == recorded["compile"] == recorded["load"], (
        "every stage must receive the SAME resolved option dict — got "
        f"{recorded!r}"
    )
    return recorded


def test_build_forwards_options_to_lower_compile_and_load(recording_backend):
    """``evoxc.build`` hands its resolved options to all three stages.

    The load hop is the load-bearing one: it used to be dropped, which broke
    every option that only the load stage consumes (the xla ``plugin_path``
    above is the user-visible symptom).
    """
    exe = evoxc.build(
        _double,
        SPEC,
        backend=recording_backend.name,
        device="cpu",
        **SENTINEL_OPTIONS,
    )
    recorded = _assert_sentinel_reached_every_stage(recording_backend)
    assert all(len(recording_backend.calls[stage]) == 1 for stage in recorded)
    assert exe.device == evoxc.core.Device("cpu", 0)
    x = np.arange(6, dtype="float32").reshape(SHAPE)
    np.testing.assert_array_equal(evoxc.run(exe, x).numpy(), x + x)


def test_build_cached_forwards_options_to_load_and_reuses_the_hit(
    recording_backend,
):
    """``build_cached`` forwards the options on its ONE miss, then hits.

    The memoized shorthand expands to the same stage sequence as
    ``evoxc.build`` (its whole contract), so it inherits the load-hop
    forwarding; the SECOND call must perform zero pipeline work and return
    the identical executable.
    """
    cache = evoxc.CompileCache()
    keyword = dict(
        backend=recording_backend.name, device="cpu", cache=cache
    )
    first = evoxc.build_cached(_double, SPEC, **keyword, **SENTINEL_OPTIONS)
    recorded = _assert_sentinel_reached_every_stage(recording_backend)

    second = evoxc.build_cached(_double, SPEC, **keyword, **SENTINEL_OPTIONS)

    assert second is first  # a cache hit returns the SAME executable
    assert (cache.hits, cache.misses) == (1, 1)
    for stage in ("lower", "compile", "load"):
        assert recording_backend.calls[stage] == [recorded[stage]], (
            f"the cache hit must not re-run the {stage} stage — got "
            f"{recording_backend.calls[stage]!r}"
        )
    x = np.arange(6, dtype="float32").reshape(SHAPE)
    np.testing.assert_array_equal(evoxc.run(second, x).numpy(), x + x)


# ---------------------------------------------------------------------------
# (4) a FAILED execute clears the REUSED output slots
# ---------------------------------------------------------------------------


def _tiny_mlir_text():
    """The StableHLO text of ``add(x, x)`` — what the fake "compiles"."""
    graph = evoxc.trace(_double, SPEC)
    lowered = evoxc.lower(graph, backend="xla")
    return lowered.payload["mlir_text"]


@pytest.fixture
def driver(fake_plugin):
    """Driver-level ``(plugin, loaded executable, input buffer)`` on the fake.

    Built with the EXPLICIT ``plugin_path`` option, so an ambient installed
    plugin (a real ``jax_plugins`` one on a developer box) can never be
    picked up here.
    """
    plugin = xla_util._load_plugin({"plugin_path": str(fake_plugin)})
    client = xla_util.acquire_client(plugin)
    loaded = client.compile(_tiny_mlir_text())
    buffer = client.buffer_from_host(np.zeros(SHAPE, "float32"))
    try:
        yield plugin, loaded, buffer
    finally:
        buffer.close()
        loaded.close()
        xla_util.release_client(plugin)


@pytest.mark.parametrize(
    "events",
    ["no_device_complete_event", "with_device_complete_event"],
    ids=["no_events", "with_events"],
)
def test_failed_execute_clears_reused_output_slots(driver, monkeypatch, events):
    """A failed execute never leaves stale output pointers in the scaffolding.

    ``_LoadedExecutable.execute`` reuses ONE ctypes scaffolding per
    ``(executable, input arity, output count)`` — its ``output_lists`` slots
    are written THROUGH by the plugin, and a later successful call overwrites
    them. A FAILED call is the dangerous case: the plugin may have written
    real ``PJRT_Buffer*`` values before returning its error, so the driver
    must zero every slot before re-raising or the next call on this
    scaffolding could wrap a dangling pointer as a ``_Buffer``.

    The C fake's ``EVOXC_FAKE_PJRT_FAIL_AFTER_OUTPUTS`` knob writes real
    output buffers into the caller's slots and THEN fails, which is exactly
    that state — both with the plugin-requested device-complete event (the
    stray handle must be destroyed and nothing handed to the caller) and
    without it.
    """
    plugin, loaded, buffer = driver
    with_events = events == "with_device_complete_event"

    # Anti-vacuity: a successful execute leaves REAL pointers in the slots the
    # driver reuses for every later call.
    outputs = loaded.execute([buffer])
    assert [out.shape for out in outputs] == [SHAPE]
    for out in outputs:
        out.close()
    scaffolding = loaded._execute_scaffolding(1)
    out_ptrs = scaffolding[1]
    assert len(out_ptrs) == 1
    assert out_ptrs[0] is not None

    destroyed = []
    real_destroy = xla_util.PjrtPlugin.destroy_event

    def _spy_destroy(self, handle):
        destroyed.append(handle)
        return real_destroy(self, handle)

    monkeypatch.setattr(xla_util.PjrtPlugin, "destroy_event", _spy_destroy)
    monkeypatch.setenv(FAIL_AFTER_OUTPUTS_ENV, "all")
    # ``None`` leaves ``device_complete_events`` NULL (the zero-plugin-cost
    # path); a list asks the plugin for ONE per-device handle.
    sink = [] if with_events else None
    with pytest.raises(evoxc.BackendError, match=FAIL_AFTER_OUTPUTS_MESSAGE):
        loaded.execute([buffer], events=sink)
    # The plugin wrote a real buffer pointer into the slot and the call
    # failed: the scaffolding must hold NO pointer...
    assert [ptr for ptr in out_ptrs] == [None]
    # ... the caller was handed no completion handle ...
    if sink is not None:
        assert sink == []
    # ... and a handle the plugin DID write was destroyed, not leaked.
    if with_events:
        assert len(destroyed) == 1 and destroyed[0]
    else:
        assert destroyed == []

    # The SAME scaffolding still serves a later successful execute.
    monkeypatch.delenv(FAIL_AFTER_OUTPUTS_ENV)
    outputs = loaded.execute([buffer])
    assert [out.shape for out in outputs] == [SHAPE]
    assert out_ptrs[0] is not None
    for out in outputs:
        out.close()
