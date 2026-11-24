"""The custom IREE module options: ``iree_compiler_module`` / ``iree_runtime_module``.

``evoxc/backends/adapters/iree.py`` resolves the IREE compiler module at the
COMPILE stage (``iree_compiler_module``) and the runtime module at the LOAD
stage (``iree_runtime_module``) — for a custom IREE build whose Python packages
are not named ``iree.compiler`` / ``iree.runtime``. Pinned here:

* both keys are part of the adapter's declared option surface and pass stage
  validation (no unrecognized-option error);
* a MODULE OBJECT supplied for ``iree_compiler_module`` is the module
  ``compile_str`` is invoked on: a delegating fake that records the call proves
  the compile hop is routed through it, and the resulting executable still runs;
* ``iree_runtime_module`` likewise: the runtime module is used at load;
* a bogus dotted module path raises ``core.BackendError`` naming the option and
  the offending value at the stage that resolves it (compile / load).

The module skips cleanly when iree is absent; the fakes DELEGATE to the real
iree modules so the compiled program is genuinely produced/loaded.
"""

import types

import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import iree.compiler
import iree.runtime

import evoxc
from evoxc.backends import options as options_module
from evoxc.backends.adapters import iree as iree_adapt

_SPEC = evoxc.TensorSpec((4,), evoxc.float32)
_INPUT = np.arange(4, dtype=np.float32)


def _fn(x):
    return evoxc.add(x, x)


@pytest.fixture(autouse=True)
def _clear_module_env(monkeypatch):
    """The env-var form must not leak into the option tests."""
    monkeypatch.delenv("EVOXC_IREE_COMPILER_MODULE", raising=False)
    monkeypatch.delenv("EVOXC_IREE_RUNTIME_MODULE", raising=False)


def _fake_compiler_module():
    """A module object that delegates ``compile_str`` to the real iree compiler."""
    calls = []

    module = types.ModuleType("fake_iree_compiler")
    real_compile_str = iree.compiler.compile_str

    def compile_str(text, **kwargs):
        calls.append((text, kwargs))
        return real_compile_str(text, **kwargs)

    module.compile_str = compile_str
    module.CompilerToolError = iree.compiler.CompilerToolError
    return module, calls


def _fake_runtime_module():
    """A module object delegating to the real iree runtime, recording ``get_driver``."""
    calls = {"get_driver": 0}

    module = types.ModuleType("fake_iree_runtime")
    for name in dir(iree.runtime):
        if name.startswith("__"):
            continue
        try:
            setattr(module, name, getattr(iree.runtime, name))
        except Exception:  # pragma: no cover - defensively skip odd attrs
            pass
    real_get_driver = iree.runtime.get_driver

    def get_driver(*args, **kwargs):
        calls["get_driver"] += 1
        return real_get_driver(*args, **kwargs)

    module.get_driver = get_driver
    return module, calls


# ---------------------------------------------------------------------------
# 1. option surface / stage validation
# ---------------------------------------------------------------------------


def test_module_options_are_declared_and_validated():
    known = iree_adapt.IreeBackend.KNOWN_OPTIONS
    assert "iree_compiler_module" in known["compile"]
    assert "iree_runtime_module" in known["load"]
    # Accepts a dotted path / module object (value validation is at resolve
    # time, not here — no unrecognized-KEY error).
    options_module.validate_options(
        {"iree_compiler_module": "iree.compiler"}, known, "iree", "compile"
    )
    options_module.validate_options(
        {"iree_runtime_module": "iree.runtime"}, known, "iree", "load"
    )
    # Union semantics: each key is accepted-and-ignored at the other's stage.
    options_module.validate_options(
        {"iree_compiler_module": "iree.compiler"}, known, "iree", "load"
    )
    options_module.validate_options(
        {"iree_runtime_module": "iree.runtime"}, known, "iree", "compile"
    )


# ---------------------------------------------------------------------------
# 2. the compiler module routes the compile hop
# ---------------------------------------------------------------------------


def test_iree_compiler_module_routes_compile_through_it():
    module, calls = _fake_compiler_module()
    exe = evoxc.build(_fn, _SPEC, backend="iree", iree_compiler_module=module)
    # The fake saw exactly the one compile call and the StableHLO program.
    assert len(calls) == 1
    assert "stablehlo.add" in calls[0][0]
    # ... and the artifact it produced is a real, runnable executable.
    out = evoxc.run(exe, _INPUT)
    np.testing.assert_allclose(out.numpy(), _INPUT + _INPUT, rtol=1e-6, atol=1e-6)


def test_iree_compiler_module_by_dotted_path():
    # The dotted-path form resolves the DEFAULT published module and works.
    exe = evoxc.build(
        _fn, _SPEC, backend="iree", iree_compiler_module="iree.compiler"
    )
    out = evoxc.run(exe, _INPUT)
    np.testing.assert_allclose(out.numpy(), _INPUT + _INPUT, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. the runtime module routes the load hop
# ---------------------------------------------------------------------------


def test_iree_runtime_module_routes_load_through_it():
    module, calls = _fake_runtime_module()
    graph = evoxc.trace(_fn, _SPEC)
    lowered = evoxc.lower(graph, backend="iree")
    artifact = evoxc.compile(lowered)
    exe = evoxc.load(artifact, iree_runtime_module=module)
    assert calls["get_driver"] >= 1
    out = evoxc.run(exe, _INPUT)
    np.testing.assert_allclose(out.numpy(), _INPUT + _INPUT, rtol=1e-6, atol=1e-6)


def test_both_module_options_forward_through_build():
    # build() forwards ONE options dict to lower/compile/load, so both module
    # options reach their stage through the sugar.
    compiler_module, compile_calls = _fake_compiler_module()
    runtime_module, runtime_calls = _fake_runtime_module()
    exe = evoxc.build(
        _fn,
        _SPEC,
        backend="iree",
        iree_compiler_module=compiler_module,
        iree_runtime_module=runtime_module,
    )
    assert len(compile_calls) == 1
    assert runtime_calls["get_driver"] >= 1
    out = evoxc.run(exe, _INPUT)
    np.testing.assert_allclose(out.numpy(), _INPUT + _INPUT, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. bogus values fail loudly at the resolving stage
# ---------------------------------------------------------------------------


def test_bogus_compiler_module_path_names_option_and_value():
    with pytest.raises(evoxc.BackendError) as excinfo:
        evoxc.build(
            _fn,
            _SPEC,
            backend="iree",
            iree_compiler_module="definitely_not_a_real_module_xyz",
        )
    message = str(excinfo.value)
    assert "iree_compiler_module" in message
    assert "definitely_not_a_real_module_xyz" in message


def test_bogus_runtime_module_path_names_option_and_value():
    graph = evoxc.trace(_fn, _SPEC)
    lowered = evoxc.lower(graph, backend="iree")
    artifact = evoxc.compile(lowered)
    with pytest.raises(evoxc.BackendError) as excinfo:
        evoxc.load(
            artifact, iree_runtime_module="definitely_not_a_real_runtime_module_xyz"
        )
    message = str(excinfo.value)
    assert "iree_runtime_module" in message
    assert "definitely_not_a_real_runtime_module_xyz" in message
