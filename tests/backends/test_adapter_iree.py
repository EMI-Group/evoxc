"""IREE adapter contract tests (``evoxc/backends/adapters/iree.py``).

Contract under test: the shared ``CompilerBackend`` framework
(``evoxc/backends/compiler.py``) plus the IREE adapter's REAL
lower/compile/load/run through ``iree.compiler`` / ``iree.runtime`` (CPU).
One contract, three dependencies: this file, ``test_adapter_xla.py`` and
``test_adapter_tvm.py`` are structurally identical and share the graph
builders, tolerances and parity helper in ``tests/backends/_adapter_utils``.

Pinned here:

* registration + lazy name resolution: ``iree.register()`` installs the
  singleton ``iree_backend`` (name ``"iree"``) and every pipeline entry point
  accepting a backend name string (``evoxc.lower`` / ``evoxc.build`` /
  ``evoxc.evaluate``, resolved through ``evoxc.backends.get``) reaches the same
  instance.
* numerical parity vs the default numpy backend (matmul+relu+sum and
  reshape/broadcast/transpose) at the cross-compiler fp32 tolerance.
* symbolic dims iff ``capabilities.dynamic_shapes`` — otherwise an explicit
  ``evoxc.BackendError`` NAMING the feature (never a silent fallback).
* artifact save/load round-trips at pipeline level (``evoxc.load``) and raw
  backend level (``backend.load(...).run([flat tensors])``); load-time
  backend-name mismatch raises ``evoxc.PersistenceError`` — never silently
  recompile.
* capability rejections (``runtime_call`` / collectives when undeclared)
  raise ``evoxc.BackendError`` NAMING the feature.
* the lowered payload is the shared StableHLO contract: ``format ==
  "stablehlo"``, ``format_version == 1``, MLIR text, entry functions.

IREE runs the real MLIR compiler (seconds per compile), so the 2s-per-file
convention does not apply here; the module keeps the compile count low via
one shared module-scoped artifact fixture plus one compile per distinct
graph.
"""

import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import evoxc
from evoxc.backends.adapters import iree

from tests.backends import _adapter_utils as u

NAME = "iree"


# ---------------------------------------------------------------------------
# module fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def adapter_backend():
    """The registered singleton backend (``iree_backend``)."""
    iree.register()
    backend = evoxc.backends.get(NAME)
    assert backend.name == NAME
    return backend


@pytest.fixture(scope="module")
def adapter_artifact(adapter_backend):
    """One shared compiled matmul+relu+sum artifact for the whole module."""
    fn, specs = u.matmul_relu_sum()
    _, lowered, artifact = u.stage(NAME, fn, specs)
    assert lowered.backend == NAME
    assert artifact.backend == NAME
    return artifact


# ---------------------------------------------------------------------------
# 1. registration / singleton
# ---------------------------------------------------------------------------


def test_registered_singleton(adapter_backend):
    assert adapter_backend is iree.iree_backend
    assert evoxc.backends.get(NAME) is iree.iree_backend
    assert isinstance(adapter_backend.capabilities, evoxc.backends.Capabilities)


# ---------------------------------------------------------------------------
# 2. full pipeline + numerical parity
# ---------------------------------------------------------------------------


def test_matmul_relu_sum_parity(adapter_artifact):
    fn, specs = u.matmul_relu_sum()
    a, b = u.matmul_relu_sum_args()
    exe = evoxc.load(adapter_artifact)
    u.assert_parity(fn, (a, b), exe)


def test_reshape_broadcast_transpose_parity_via_build_name(adapter_backend):
    fn, specs = u.reshape_broadcast_transpose()
    x = u.standard_normal((4, 8))
    exe = evoxc.build(fn, *specs, backend=NAME)
    u.assert_parity(fn, (x,), exe)


def test_evaluate_with_backend_name(adapter_backend):
    fn, specs = u.matmul_relu_sum()
    a, b = u.matmul_relu_sum_args()
    expected = evoxc.evaluate(fn, a, b)  # default numpy reference
    actual = evoxc.evaluate(fn, a, b, backend=NAME)
    assert isinstance(actual, evoxc.Tensor)
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(
        actual.numpy(), expected.numpy(), rtol=u.FP32_RTOL, atol=u.FP32_ATOL
    )


# ---------------------------------------------------------------------------
# 3. symbolic dims
# ---------------------------------------------------------------------------


def test_symbolic_dims(adapter_backend):
    fn, specs = u.symbolic_scale()
    graph = evoxc.trace(fn, *specs)
    graph.verify()
    if adapter_backend.capabilities.dynamic_shapes is False:
        # Capability rejection must name the feature — no silent fallback.
        with pytest.raises(evoxc.BackendError) as excinfo:
            evoxc.lower(graph, backend=NAME)
        assert "dynamic" in str(excinfo.value).lower()
        return

    numpy_exe = evoxc.build(fn, *specs)  # default numpy reference
    adapter_exe = evoxc.build(fn, *specs, backend=NAME)
    for x in (
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
    ):
        expected = evoxc.run(numpy_exe, x)
        actual = evoxc.run(adapter_exe, x)
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(
            actual.numpy(), expected.numpy(), rtol=u.FP32_RTOL, atol=u.FP32_ATOL
        )


def test_symbolic_dynamic_reshape_rejected_at_lower(adapter_backend):
    # Writer regression: a softmax-style graph over a symbolic batch contains
    # a dynamic keepdims reshape ((B,) -> (B, 1)). Writer._reject_dynamic_dims
    # must reject it at lower() time with evoxc's OWN BackendError — never
    # invalid MLIR that dies later with an obscure iree-compile parse error.
    fn, specs = u.symbolic_softmax()
    graph = evoxc.trace(fn, *specs)
    graph.verify()  # negative control: valid IR — rejection is export-time
    with pytest.raises(evoxc.BackendError) as excinfo:
        evoxc.lower(graph, backend=NAME)
    msg = str(excinfo.value)
    assert "op 'reshape'" in msg
    assert "dynamic" in msg


# ---------------------------------------------------------------------------
# 4. payload contract + name resolution
# ---------------------------------------------------------------------------


def test_lower_name_resolution_and_payload_contract(adapter_backend):
    fn, specs = u.matmul_relu_sum()
    graph = evoxc.trace(fn, *specs)
    graph.verify()
    lowered = evoxc.lower(graph, backend=NAME)
    assert lowered.backend == NAME
    payload = lowered.payload
    assert payload["format"] == "stablehlo"
    assert payload["format_version"] == 1
    mlir = payload["mlir_text"]
    assert isinstance(mlir, str) and mlir
    assert "module {" in mlir
    assert "func.func" in mlir
    assert "stablehlo." in mlir
    entry = payload["entry_functions"]
    assert isinstance(entry, list)
    assert "main" in entry
    # Direct backend invocation agrees with string-name resolution.
    assert adapter_backend.lower(graph).payload == payload


# ---------------------------------------------------------------------------
# 5. artifact persistence
# ---------------------------------------------------------------------------


def test_artifact_save_load_roundtrip(tmp_path, adapter_backend, adapter_artifact):
    path = tmp_path / "artifact.evoxc"
    adapter_artifact.save(path)
    loaded = evoxc.backends.CompiledArtifact.load(path)
    assert loaded.backend == NAME

    # Backend-level reload: raw run over flat evoxc.Tensor inputs.
    raw = adapter_backend.load(loaded)
    fn, specs = u.matmul_relu_sum()
    a, b = u.matmul_relu_sum_args()
    expected = evoxc.evaluate(fn, a, b)
    (out,) = raw.run([evoxc.tensor(a), evoxc.tensor(b)])
    np.testing.assert_allclose(
        out.numpy(), expected.numpy(), rtol=u.FP32_RTOL, atol=u.FP32_ATOL
    )

    # Pipeline-level reload runs through the structured signature.
    exe = evoxc.load(loaded)
    u.assert_parity(fn, (a, b), exe)


def test_backend_name_mismatch_on_load(adapter_backend, adapter_artifact):
    # An adapter artifact can never be loaded by another backend: "numpy"
    # stands in for any mismatching registered backend (always present,
    # order-independent).
    with pytest.raises(evoxc.PersistenceError, match="never silently recompile"):
        evoxc.load(adapter_artifact, backend="numpy")

    # The reverse direction: an adapter never loads another backend's
    # artifact.

    def fn(x):
        return evoxc.add(x, x)

    numpy_artifact = evoxc.compile(
        evoxc.lower(evoxc.trace(fn, evoxc.TensorSpec((2, 3), evoxc.float32)))
    )
    with pytest.raises(evoxc.PersistenceError):
        adapter_backend.load(numpy_artifact)


# ---------------------------------------------------------------------------
# 6. executable attributes
# ---------------------------------------------------------------------------


def test_executable_attributes(adapter_artifact):
    exe = evoxc.load(adapter_artifact)
    assert isinstance(exe.functions, tuple)
    assert "main" in exe.functions
    assert exe.device is None or exe.device == evoxc.Device("cpu", 0)


# ---------------------------------------------------------------------------
# 7. capability rejections
# ---------------------------------------------------------------------------


def test_unsupported_features_rejected_naming_feature(adapter_backend):
    cases = []
    if adapter_backend.capabilities.runtime_calls is False:
        cases.append(("runtime_call", u.runtime_call_graph()))
    if adapter_backend.capabilities.collectives is False:
        cases.append(("collective", u.collective_graph()))
    if not cases:
        pytest.skip("adapter declares runtime_calls and collectives — nothing to reject")

    for feature, graph in cases:
        graph.verify()  # negative control: valid IR — rejection is capability-only
        with pytest.raises(evoxc.BackendError) as excinfo:
            adapter_backend.lower(graph)
        assert feature in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# 8. batched dot_general with mismatched batch ranks (writer regression)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def batched_dot_artifacts(adapter_backend):
    """One compiled iree artifact per static batched-dot emission path."""
    return {
        name: u.stage(NAME, fn, specs)[2]
        for name, (fn, specs, _args) in u.batched_dot_cases().items()
    }


@pytest.mark.parametrize(
    "name", ["rank3_rank2", "rhs_higher_rank", "size1_rhs_batch"]
)
def test_batched_dot_mismatched_batch_ranks_parity(name, batched_dot_artifacts):
    # The pre-fix writer emitted unbalanced batching dims for these shapes
    # (invalid MLIR that iree-compile rejected). Compile+run must now match
    # numpy matmul exactly in shape and value.
    fn, _specs, args = u.batched_dot_cases()[name]
    exe = evoxc.load(batched_dot_artifacts[name])
    expected = np.matmul(args[0], args[1])
    actual = evoxc.run(exe, *args).numpy()
    assert actual.shape == expected.shape
    # K=768 fp32 accumulation-order differences across compilers measure
    # below 2e-4, so the cross-compiler bound is 1e-3 (same bound the bench
    # harness uses for large fp32 reductions).
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)


def test_batched_dot_symbolic_batch_parity(adapter_backend):
    # One compiled executable serves every concrete batch size: the rhs
    # matrix is broadcast up to the runtime batch via
    # dynamic_broadcast_in_dim + get_dimension_size.
    fn, specs = u.batched_dot_symbolic()
    exe = evoxc.build(fn, *specs, backend=NAME)
    for b in (2, 5):
        a = u.standard_normal((b, 512, 768))
        w = u.standard_normal((768, 2304))
        expected = np.matmul(a, w)
        actual = evoxc.run(exe, a, w).numpy()
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)


@evoxc.defn
def _vmap_shared_weight_fn(x, w, b):
    """``relu(dot(x, w) + b)`` — the vmap shared-weight repro below."""
    return evoxc.relu(evoxc.add(evoxc.dot(x, w), b))


def test_batched_dot_vmap_shared_weight_parity(adapter_backend):
    # Bug-report repro: vmap with in_axes=(0, None, None) — x is mapped
    # over its leading axis, w and b are unmapped (shared), so the traced
    # graph's dot is (B,1,16) @ (1,16,8): a size-1-batch rhs under a
    # SYMBOLIC batch. The writer's size-1-batch squeeze path is gated on
    # fully static shapes (fix 16ced05), so the dot must fall through to
    # the batched dynamic-broadcast path (dynamic_broadcast_in_dim +
    # batched dot_general) and COMPILE + RUN on iree. Before the fix the
    # squeeze emitted a non-batched dot_general with a dynamic lhs and
    # iree-compile failed: "failed to legalize operation
    # 'stablehlo.dynamic_reshape'".
    graph = evoxc.vmap(_vmap_shared_weight_fn, in_axes=(0, None, None), out_axes=0)(
        evoxc.TensorSpec((32, 1, 16), evoxc.float32),
        evoxc.TensorSpec((16, 8), evoxc.float32),
        evoxc.TensorSpec((8,), evoxc.float32),
    )
    graph.verify()
    numpy_exe = evoxc.load(evoxc.compile(evoxc.lower(graph)))
    adapter_exe = evoxc.load(evoxc.compile(evoxc.lower(graph, backend=NAME)))
    x = u.standard_normal((32, 1, 16))
    w = u.standard_normal((16, 8))
    b = u.standard_normal((8,))
    expected = evoxc.run(numpy_exe, x, w, b)
    actual = evoxc.run(adapter_exe, x, w, b)
    assert isinstance(actual, evoxc.Tensor)
    assert actual.dtype == expected.dtype
    assert actual.numpy().shape == expected.numpy().shape
    # Cross-compiler fp32 bound, same as the other batched-dot parity tests.
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-3, atol=1e-3)
