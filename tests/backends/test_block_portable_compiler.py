"""Portable block inlining on compiler backends.

``CompilerBackend.lower`` splices a block's PORTABLE decomposition (a block
with a registered ``@evoxc.defn`` decomposition) through the shared
``inline_portables`` machinery — no ``block_call`` survives into the exported
StableHLO. A block with only a BACKEND IMPL (no portable) cannot be inlined
and has no compiler-side meaning, so it is rejected at the capability
pre-check with an explicit ``core.BackendError`` naming the block.

Pinned through three backends:

* the numpy reference interpreter (a portable block runs by inlining; a
  backend-only block runs through its registered numpy impl);
* a test-only ``CompilerBackend`` stub (the exported MLIR drops ``block_call``
  for a portable block; a backend-only block raises the explicit error);
* the ``iree`` adapter (a portable block compiles and runs with numpy parity;
  a backend-only block is rejected) — guarded so the file stays green without
  iree.

Block names are process-global (declaration == registration), so every block
here is prefixed ``block_portable_compiler_``.
"""

import numpy as np
import pytest

import evoxc
from evoxc.backends import Capabilities, CompiledArtifact, register
from evoxc.backends.compiler import CompilerBackend

_SPEC = evoxc.TensorSpec((2, 3), evoxc.float32)
_INPUT = np.arange(6, dtype=np.float32).reshape(2, 3)


# ---------------------------------------------------------------------------
# blocks (declaration IS registration)
# ---------------------------------------------------------------------------


@evoxc.block(outputs=[evoxc.TensorSpec((2, 3), evoxc.float32)])
@evoxc.defn
def block_portable_compiler_double(x: evoxc.TensorSpec((2, 3), evoxc.float32)):
    return evoxc.add(x, x)


block_portable_compiler_backend_only = evoxc.block(
    "block_portable_compiler_backend_only",
    inputs=[evoxc.TensorSpec((2, 3), evoxc.float32)],
    outputs=[evoxc.TensorSpec((2, 3), evoxc.float32)],
)


@block_portable_compiler_backend_only.impl("numpy")
def _backend_only_numpy_impl(x):
    return x * 2.0


def _portable_fn(x):
    return block_portable_compiler_double(x)


def _backend_only_fn(x):
    return block_portable_compiler_backend_only(x)


# ---------------------------------------------------------------------------
# test-only compiler-backend stub (compile/load are trivial wrappers)
# ---------------------------------------------------------------------------


class _StubExecutable:
    def __init__(self, artifact):
        self.functions = tuple(artifact.payload.get("entry_functions", ()))
        self.device = None

    def run(self, flat_input_tensors):
        raise evoxc.BackendError("stub compiler backend cannot run")


class _StubCompilerBackend(CompilerBackend):
    name = "block-portable-compiler-stub"
    capabilities = Capabilities(
        dynamic_shapes=True,
        dtypes=frozenset({np.dtype("float32")}),
        # A real compiler adapter declares custom_blocks=False: it has no
        # per-backend block impls, only portable inlining.
        custom_blocks=False,
    )

    def compile(self, lowered, options=None):
        return CompiledArtifact(
            backend=self.name,
            signature=lowered.signature,
            target="cpu",
            payload=dict(lowered.payload),
        )

    def load(self, artifact, device=None):
        return _StubExecutable(artifact)


@pytest.fixture(scope="module")
def stub_backend():
    backend = _StubCompilerBackend()
    register(backend)
    return backend


# ---------------------------------------------------------------------------
# numpy interpreter (the reference)
# ---------------------------------------------------------------------------


def test_numpy_runs_a_portable_block_by_inlining():
    out = evoxc.evaluate(_portable_fn, _INPUT)
    np.testing.assert_array_equal(out.numpy(), _INPUT + _INPUT)


def test_numpy_runs_a_backend_only_block_through_its_impl():
    out = evoxc.evaluate(_backend_only_fn, _INPUT)
    np.testing.assert_array_equal(out.numpy(), _INPUT * 2.0)


# ---------------------------------------------------------------------------
# compiler backend stub: portable inlining / explicit rejection
# ---------------------------------------------------------------------------


def test_compiler_backend_inlines_a_portable_block(stub_backend):
    graph = evoxc.trace(_portable_fn, _SPEC)
    lowered = stub_backend.lower(graph)
    mlir = lowered.payload["mlir_text"]
    assert "block_call" not in mlir
    assert "block_portable_compiler_double" not in mlir
    assert "stablehlo.add" in mlir


def test_compiler_backend_compiles_the_inlined_program(stub_backend):
    graph = evoxc.trace(_portable_fn, _SPEC)
    artifact = stub_backend.compile(stub_backend.lower(graph))
    assert artifact.backend == stub_backend.name
    assert "stablehlo.add" in artifact.payload["mlir_text"]


def test_compiler_backend_rejects_a_backend_only_block(stub_backend):
    graph = evoxc.trace(_backend_only_fn, _SPEC)
    with pytest.raises(evoxc.BackendError) as excinfo:
        stub_backend.lower(graph)
    message = str(excinfo.value)
    assert "block_call" in message
    assert "block_portable_compiler_backend_only" in message
    assert "portable" in message


# ---------------------------------------------------------------------------
# iree adapter (optional)
# ---------------------------------------------------------------------------


def test_iree_compiles_and_runs_a_portable_block():
    pytest.importorskip("iree.compiler")
    pytest.importorskip("iree.runtime")
    exe = evoxc.build(_portable_fn, _SPEC, backend="iree")
    out = evoxc.run(exe, _INPUT)
    assert out.device == evoxc.Device("cpu", 0)
    np.testing.assert_allclose(out.numpy(), _INPUT + _INPUT, rtol=1e-6, atol=1e-6)


def test_iree_rejects_a_backend_only_block():
    pytest.importorskip("iree.compiler")
    pytest.importorskip("iree.runtime")
    with pytest.raises(evoxc.BackendError) as excinfo:
        evoxc.build(_backend_only_fn, _SPEC, backend="iree")
    message = str(excinfo.value)
    assert "block_call" in message
    assert "block_portable_compiler_backend_only" in message
