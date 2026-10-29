"""Contract tests for the shared compiler backend framework.

Contract under test: ``evoxc/backends/compiler.py`` — the shared
``CompilerBackend`` base that the three optional external-compiler adapters
(``evoxc/backends/adapters/iree``, ``xla``, ``tvm``) build on. Pinned here with
a TEST-ONLY stub subclass defined in this file (test-only code never goes
into the package):

* ``lower(graph)`` — the shared implementation: (1) ``graph.verify()``,
  (2) capability pre-check (a rejection names the missing feature:
  ``collectives`` / ``runtime_calls`` / ``dynamic_shapes`` / ``dtypes`` /
  ``custom_blocks``), (3) portable block inlining, (4) StableHLO export, and
  (5) a ``LoweredProgram`` whose payload is ``{"format": "stablehlo",
  "format_version": 1, "mlir_text": <str>, "entry_functions": [<fn names>]}``.
* ``compile(lowered)`` — the adapter side validates ``lowered.backend ==
  self.name`` (never cross-backend compilation) and wraps the payload into a
  ``CompiledArtifact``.
* ``load(artifact)`` — validates ``artifact.backend == self.name`` and
  returns an executable (never re-compiling).
* the verification-pass COUNT (``ir.verify`` walks): exactly ONE for a graph
  with nothing to inline (the post-inline re-verification is provably
  redundant, and the export uses ``stablehlo._emit_verified`` instead of the
  verifying public ``export``), exactly TWO when a portable decomposition was
  spliced; the public ``export`` still verifies its input once.

Every test builds its stub through ``_make_stub`` with a DISTINCT backend
name, so the process-global registry never collides across tests. CPU only;
tiny shapes; everything imported from ``evoxc`` directly.
"""

import json

import numpy as np
import pytest

import evoxc
from evoxc.backends import Capabilities, register
from evoxc.backends.compiler import CompilerBackend

# ---------------------------------------------------------------------------
# test-only stub backend
# ---------------------------------------------------------------------------


class _StubExecutable:
    """Executable returned by the stub backend's ``load``.

    Mirrors the ``Executable`` protocol surface (``functions`` / ``device`` /
    ``run``) without doing real work: compiler-backed executables are
    produced by the real adapters, never by this stub.
    """

    def __init__(self, artifact):
        self.functions = tuple(artifact.payload.get("entry_functions", ()))
        self.device = None

    def run(self, flat_input_tensors):
        raise evoxc.BackendError(
            "stub compiler backend produces non-executable artifacts"
        )


class _StubCompilerBackend(CompilerBackend):
    """Minimal concrete ``CompilerBackend`` for contract tests.

    ``lower`` is inherited from the shared framework; ``compile``/``load``
    are the adapter-side staging steps and are stubbed here.
    """

    name = "compiler-stub-base"
    capabilities = Capabilities(
        dynamic_shapes=True,
        dtypes=frozenset({np.dtype("float32"), np.dtype("float64")}),
        collectives=True,
        runtime_calls=True,
        custom_blocks=True,
    )

    def compile(self, lowered, options=None):
        if lowered.backend != self.name:
            raise evoxc.BackendError(
                "cannot compile a LoweredProgram produced by backend "
                f"{lowered.backend!r} with backend {self.name!r}"
            )
        return evoxc.backends.CompiledArtifact(
            backend=self.name,
            signature=lowered.signature,
            target="cpu",
            payload=dict(lowered.payload),
        )

    def load(self, artifact, device=None):
        if artifact.backend != self.name:
            raise evoxc.PersistenceError(
                "cannot load an artifact produced by backend "
                f"{artifact.backend!r} with backend {self.name!r}"
            )
        return _StubExecutable(artifact)


#: Full capability set for the stub (overridable per test).
_DEFAULT_STUB_CAPS = {
    "dynamic_shapes": True,
    "dtypes": frozenset({np.dtype("float32"), np.dtype("float64")}),
    "collectives": True,
    "runtime_calls": True,
    "custom_blocks": True,
}


def _make_stub(name, **cap_overrides):
    """Build a fresh ``_StubCompilerBackend`` subclass with a distinct name.

    Capabilities start from the full default set and are updated with the
    overrides; every test uses a unique ``name`` so the process-global
    backend registry never collides.
    """
    caps = dict(_DEFAULT_STUB_CAPS)
    caps.update(cap_overrides)
    cls = type(
        f"_Stub_{name.replace('-', '_')}",
        (_StubCompilerBackend,),
        {"name": name, "capabilities": Capabilities(**caps)},
    )
    return cls()


# ---------------------------------------------------------------------------
# graph builders
# ---------------------------------------------------------------------------


def _add_graph():
    def fn(x):
        return evoxc.add(x, x)

    return evoxc.trace(fn, evoxc.TensorSpec((2, 3), evoxc.float32))


def _collective_graph():
    def fn(x):
        return evoxc.dist.all_reduce(x)

    return evoxc.trace(fn, evoxc.TensorSpec((2, 3), evoxc.float32))


def _double_callback(x):
    return x * 2.0


def _runtime_call_graph():
    def fn(x):
        return evoxc.runtime_call(
            _double_callback, x, result=evoxc.TensorSpec((2, 3), evoxc.float32)
        )

    return evoxc.trace(fn, evoxc.TensorSpec((2, 3), evoxc.float32))


def _symbolic_graph():
    def fn(x):
        return evoxc.add(x, x)

    return evoxc.trace(fn, evoxc.TensorSpec((evoxc.dim("B"),), evoxc.float32))


def _float64_graph():
    def fn(x):
        return evoxc.add(x, x)

    return evoxc.trace(fn, evoxc.TensorSpec((2, 3), evoxc.float64))


@evoxc.block(outputs=[evoxc.TensorSpec((2, 3), evoxc.float32)])
@evoxc.defn
def compiler_framework_portable_double(x: evoxc.TensorSpec((2, 3), evoxc.float32)):
    return evoxc.add(x, x)


def _block_graph():
    def fn(x):
        return compiler_framework_portable_double(x)

    return evoxc.trace(fn, evoxc.TensorSpec((2, 3), evoxc.float32))


# ---------------------------------------------------------------------------
# 1. shared lower(): stablehlo payload contract
# ---------------------------------------------------------------------------


def test_shared_lower_produces_stablehlo_payload():
    stub = _make_stub("payload-contract")
    graph = _add_graph()
    # Negative control: the graph is valid IR — the assertions below pin the
    # shared lower() output, not verification behavior.
    graph.verify()

    lowered = stub.lower(graph)

    assert isinstance(lowered, evoxc.backends.LoweredProgram)
    assert lowered.backend == "payload-contract"
    payload = lowered.payload
    assert isinstance(payload, dict)
    assert payload["format"] == "stablehlo"
    assert payload["format_version"] == 1
    mlir = payload["mlir_text"]
    assert isinstance(mlir, str)
    assert mlir
    # StableHLO shell + a mapped op (same substrings pinned in
    # test_stablehlo.py).
    assert "module {" in mlir
    assert "func.func @main" in mlir
    assert "stablehlo.add" in mlir
    assert "func.return" in mlir
    entry = payload["entry_functions"]
    assert isinstance(entry, list)
    assert entry
    assert all(isinstance(name, str) for name in entry)
    assert "main" in entry
    # Persistence containers require JSON-serializable payloads.
    json.dumps(payload)


# ---------------------------------------------------------------------------
# 2. capability pre-check (valid IR rejected by the check, naming the feature)
# ---------------------------------------------------------------------------

_CAPABILITY_REJECT_CASES = [
    ("collective", _collective_graph, {"collectives": False}),
    ("runtime_call", _runtime_call_graph, {"runtime_calls": False}),
    ("dynamic", _symbolic_graph, {"dynamic_shapes": False}),
    ("float64", _float64_graph, {"dtypes": frozenset({np.dtype("float32")})}),
    ("block", _block_graph, {"custom_blocks": False}),
]


@pytest.mark.parametrize(
    "feature, graph_fn, overrides",
    _CAPABILITY_REJECT_CASES,
    ids=[case[0] for case in _CAPABILITY_REJECT_CASES],
)
def test_capability_pre_check_rejects_naming_feature(feature, graph_fn, overrides):
    graph = graph_fn()
    # Negative control: the graph is valid IR — the rejection must come from
    # the capability pre-check, not from verification.
    graph.verify()
    stub = _make_stub(f"reject-{feature}", **overrides)
    with pytest.raises(evoxc.BackendError) as excinfo:
        stub.lower(graph)
    assert feature in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# 3. full-capability stub lowers every exportable graph
# ---------------------------------------------------------------------------

_FULL_CAP_GRAPHS = [
    ("add", _add_graph),
    ("collective", _collective_graph),
    ("symbolic", _symbolic_graph),
    ("float64", _float64_graph),
    ("block", _block_graph),
]


@pytest.mark.parametrize(
    "name, graph_fn", _FULL_CAP_GRAPHS, ids=[case[0] for case in _FULL_CAP_GRAPHS]
)
def test_full_capability_stub_lowers_exportable_graphs(name, graph_fn):
    # runtime_call is deliberately NOT in this list: it has no StableHLO
    # mapping and the export rejects it regardless of the capability flags
    # (pinned in test_stablehlo.py).
    stub = _make_stub("full-caps")
    graph = graph_fn()
    graph.verify()
    lowered = stub.lower(graph)
    assert isinstance(lowered, evoxc.backends.LoweredProgram)
    assert lowered.backend == "full-caps"
    assert lowered.payload["format"] == "stablehlo"


# ---------------------------------------------------------------------------
# 4. compile(): backend-name validation (never cross-backend compilation)
# ---------------------------------------------------------------------------


def test_compile_validates_lowered_backend_name():
    stub = _make_stub("compile-validation")
    foreign = evoxc.backends.LoweredProgram(backend="some-other-backend", payload={})
    with pytest.raises(evoxc.BackendError) as excinfo:
        stub.compile(foreign)
    msg = str(excinfo.value)
    assert "some-other-backend" in msg
    assert "compile-validation" in msg

    # The matching path wraps the payload into a CompiledArtifact verbatim.
    lowered = stub.lower(_add_graph())
    artifact = stub.compile(lowered)
    assert artifact.backend == "compile-validation"
    assert artifact.target == "cpu"
    assert artifact.payload == lowered.payload


# ---------------------------------------------------------------------------
# 5. registry + pipeline integration (string-name resolution)
# ---------------------------------------------------------------------------


def test_stub_registers_and_resolves_via_pipeline():
    stub = _make_stub("registry-stub")
    assert register(stub) is stub
    assert evoxc.backends.get("registry-stub") is stub

    graph = _add_graph()
    # String-name resolution through the pipeline (_resolve_backend).
    lowered = evoxc.lower(graph, backend="registry-stub")
    assert lowered.backend == "registry-stub"
    # evoxc.compile resolves the backend from lowered.backend via the registry.
    artifact = evoxc.compile(lowered)
    assert artifact.backend == "registry-stub"

    exe = evoxc.load(artifact)
    assert isinstance(exe.functions, tuple)
    assert "main" in exe.functions


# ---------------------------------------------------------------------------
# 6. save/load round-trips through the persist container
# ---------------------------------------------------------------------------


def test_lowered_and_artifact_save_load_roundtrip(tmp_path):
    stub = _make_stub("persist-stub")
    register(stub)

    lowered = stub.lower(_add_graph())
    lpath = tmp_path / "lowered_stablehlo.evoxc"
    lowered.save(lpath)
    restored = evoxc.backends.LoweredProgram.load(lpath)
    assert restored.backend == lowered.backend == "persist-stub"
    assert restored.payload == lowered.payload

    artifact = stub.compile(lowered)
    apath = tmp_path / "artifact_stablehlo.evoxc"
    artifact.save(apath)
    restored_artifact = evoxc.backends.CompiledArtifact.load(apath)
    assert restored_artifact.backend == artifact.backend == "persist-stub"
    assert restored_artifact.target == "cpu"
    assert restored_artifact.payload == artifact.payload


# ---------------------------------------------------------------------------
# 7. verification-pass count (the redundancy proof at the ``inline_portables``
#    call site must stay TRUE — never a silent correctness relaxation)
# ---------------------------------------------------------------------------
def _count_verifications(monkeypatch):
    """Count full ``ir.verify`` walks from BOTH reachable call sites.

    ``CompilerBackend.lower`` reaches verification through exactly two names:
    ``Graph.verify()`` (``evoxc.trace.graph`` calls ``ir.verify`` via the
    ``evoxc.ir`` module attribute, resolved at CALL time) and
    ``stablehlo.export`` (which bound ``verify`` into its own module namespace
    at import). Patching both keeps the count honest wherever a walk happens.
    """
    import evoxc.ir as ir_mod
    from evoxc.backends import stablehlo as stablehlo_mod

    calls = []
    real = ir_mod.verify

    def counting(module):
        calls.append(module)
        return real(module)

    monkeypatch.setattr(ir_mod, "verify", counting)
    monkeypatch.setattr(stablehlo_mod, "verify", counting)
    return calls


def test_lower_verifies_exactly_once_without_inlining(monkeypatch):
    # A graph with nothing to splice is verified ONCE: the defensive
    # post-inline ``graph.verify()`` is provably redundant when
    # ``inline_portables`` reports zero expansions, and the export uses the
    # private ``stablehlo._emit_verified`` fast path instead of the public
    # (verifying) ``export``.
    stub = _make_stub("verify-count-plain")
    graph = _add_graph()
    calls = _count_verifications(monkeypatch)
    lowered = stub.lower(graph)
    assert len(calls) == 1
    assert calls[0] is graph.module
    assert "stablehlo.add" in lowered.payload["mlir_text"]


def test_lower_verifies_twice_when_a_portable_is_inlined(monkeypatch):
    # When a portable decomposition IS spliced into the module, the module has
    # changed since the first walk, so the defensive re-verification runs: 2
    # walks total (pre-inline + post-inline) — and STILL not a third from the
    # export fast path.
    stub = _make_stub("verify-count-inline")
    graph = _block_graph()
    calls = _count_verifications(monkeypatch)
    lowered = stub.lower(graph)
    assert len(calls) == 2
    assert calls[0] is calls[1] is graph.module
    mlir_text = lowered.payload["mlir_text"]
    assert "block_call" not in mlir_text
    assert "stablehlo.add" in mlir_text


def test_public_export_still_verifies_its_input(monkeypatch):
    # Contrast pin: the PUBLIC entry point never takes the fast path — its
    # contract is "verify first", so it performs one walk (which is exactly
    # why ``lower`` calls the private entry point instead).
    from evoxc.backends import stablehlo as stablehlo_mod

    calls = _count_verifications(monkeypatch)
    graph = _add_graph()
    text = stablehlo_mod.export(graph)
    assert len(calls) == 1
    assert calls[0] is graph.module
    assert text == stablehlo_mod._emit_verified(graph.module)
