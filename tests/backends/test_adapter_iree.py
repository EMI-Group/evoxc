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
* the per-dispatch buffer-binding-limit rejection (HIP/CUDA HAL cap of 16,
  checked at executable LOAD — ``adapters/CONTEXT.md`` Known Issue #19):
  ``_raise_if_dispatch_binding_limit`` rewrites EXACTLY that runtime text into
  an actionable ``core.BackendError`` (observed count, cap, compile target,
  the three remedies; the original ``ValueError`` as ``__cause__``), annotates
  the split/external-kernel ``segment_index``, and leaves every other iree
  load failure — the llvm-cpu command-buffer cap, the mixed-install trap —
  with its previous wording; ``TestIreeBindingLimitError`` also drives the real
  ROCm/HIP wall (K=15 pieces ⇒ 16 bindings loads and runs, K=16 ⇒ 17 bindings
  is rejected), GPU-guarded.

IREE runs the real MLIR compiler (seconds per compile), so the 2s-per-file
convention does not apply here; the module keeps the compile count low via
one shared module-scoped artifact fixture plus one compile per distinct
graph.
"""

import gc

import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import evoxc
from evoxc.backends.adapters import iree

from tests.backends import _adapter_utils as u
from tests.backends import _rocm_guard as rocm_guard

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


# ---------------------------------------------------------------------------
# 9. per-dispatch binding limit (HIP/CUDA HAL cap, raised at executable LOAD)
# ---------------------------------------------------------------------------

#: The documented runtime rejection text of the HIP/CUDA verifier
#: (``runtime/src/iree/hal/drivers/{hip,cuda}/native_executable.c``; see
#: ``evoxc/backends/adapters/CONTEXT.md`` Known Issue #19).
HIP_BINDING_LIMIT_TEXT = (
    "hip/native_executable.c:223: INVALID_ARGUMENT; failed to create "
    "executable; exports[0] binding_flags count 21 exceeds maximum of 16 "
    "(IREE_HAL_HIP_MAX_DISPATCH_BINDING_COUNT)"
)

#: The llvm-cpu command-buffer dispatch cap — a DIFFERENT runtime message
#: that ``_raise_if_dispatch_binding_limit`` deliberately does NOT rewrite
#: (Known Issue #19: the local-task caps are out of its scope).
LOCAL_TASK_COMMAND_BUFFER_TEXT = (
    "hal/module.c:1077: OUT_OF_RANGE; binding count 47 > 32"
)

#: The mixed-install trap text (Known Issue #7) — unrelated to bindings and
#: therefore never rewritten.
MIXED_INSTALL_TEXT = (
    "required module features [Ch] are not available in this runtime "
    "configuration"
)


def _install_fake_iree_runtime(monkeypatch, load_vm_module, flatbuffer_loader=None):
    """Patch the ``iree.runtime`` entry points ``_load_vm_module`` calls.

    ``_load_vm_module`` imports ``iree.runtime as rt`` INSIDE its body and then
    calls ``rt.Config`` / ``rt.VmModule.copy_buffer`` / ``rt.load_vm_module``
    (GPU kinds) or ``rt.load_vm_flatbuffer`` (CPU), so patching the attributes
    of the real module reaches it. The device/VM-module scaffolding is faked so
    no real flatbuffer is parsed; only the loader under test is real (a stub).
    ``monkeypatch`` restores every attribute on teardown.
    """
    import types

    import iree.runtime as rt

    class _FakeVmModule:
        @staticmethod
        def copy_buffer(vm_instance, vmfb):
            return object()

    monkeypatch.setattr(
        rt, "Config", lambda device=None: types.SimpleNamespace(vm_instance=object())
    )
    monkeypatch.setattr(rt, "VmModule", _FakeVmModule)
    monkeypatch.setattr(rt, "load_vm_module", load_vm_module)
    if flatbuffer_loader is not None:
        monkeypatch.setattr(rt, "load_vm_flatbuffer", flatbuffer_loader)


def _binding_wall_fn(k):
    """K ``reshape(divide(reduce_sum(x_i), N), (1,))`` pieces, concatenated.

    IREE rewrites the ``stablehlo.concatenate`` into a ``flow.tensor.update``
    chain and fuses it (plus each piece's trailing elementwise ``divide``)
    into ONE dispatch reading the K reduce outputs — K+1 bindings (Known
    Issue #19's measured worst case).
    """

    def fn(*xs):
        pieces = [
            evoxc.reshape(evoxc.divide(evoxc.sum(x), _BINDING_WALL_DIVISOR), (1,))
            for x in xs
        ]
        return evoxc.concatenate(pieces, axis=0)

    return fn


#: Reduced-row length and the divisor of the synthetic binding-wall graph.
_BINDING_WALL_ROWS = 4
_BINDING_WALL_DIVISOR = 4.0


def _binding_wall_specs(k):
    return tuple(
        evoxc.TensorSpec((_BINDING_WALL_ROWS,), evoxc.float32) for _ in range(k)
    )


def _binding_wall_args(k, seed=0):
    """Host inputs whose reduce sums are exactly representable in fp32."""
    rng = np.random.default_rng(seed)
    return [
        (rng.integers(0, 4, size=(_BINDING_WALL_ROWS,)) / 4.0).astype(np.float32)
        for _ in range(k)
    ]


class TestIreeBindingLimitError:
    """The per-dispatch 16-binding HAL cap -> actionable ``core.BackendError``.

    Pins ``evoxc/backends/adapters/iree.py::_raise_if_dispatch_binding_limit``
    and its wiring inside the shared ``_load_vm_module`` (Known Issue #19):
    ONLY the HIP/CUDA binding-limit rejection is rewritten (with the original
    ``ValueError`` as ``__cause__``); every other iree load failure keeps its
    previous wording. GPU-free except the last pin, which needs a real ROCm
    device (one small compile + load, no timing — the box's GPU is shared).
    """

    # -- the message contract (pure unit) ----------------------------------

    def test_hip_binding_limit_becomes_actionable_backend_error(self):
        exc = ValueError(HIP_BINDING_LIMIT_TEXT)
        with pytest.raises(evoxc.core.BackendError) as excinfo:
            iree._raise_if_dispatch_binding_limit(exc, "rocm")
        message = str(excinfo.value)
        # The observed count and the cap, both spelled out.
        assert "needs 21 bindings" in message
        assert "cap of 16" in message
        # The device kind AND the compile target it corresponds to.
        assert "target 'rocm'" in message
        assert "ROCm/HIP" in message
        # The three documented remedies, verbatim.
        assert "FEWER bands" in message
        assert "target_backends=['llvm-cpu']" in message
        assert "numpy" in message
        # Nothing was silently retried/recompiled/fallen back.
        assert "Nothing was retried" in message
        # The original runtime error is preserved as the cause.
        assert excinfo.value.__cause__ is exc
        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_segment_index_names_the_external_kernel_segment(self):
        exc = ValueError(HIP_BINDING_LIMIT_TEXT)
        with pytest.raises(evoxc.core.BackendError) as excinfo:
            iree._raise_if_dispatch_binding_limit(exc, "rocm", segment_index=2)
        message = str(excinfo.value)
        assert "external-kernel segment 2" in message
        assert "needs 21 bindings" in message
        assert excinfo.value.__cause__ is exc

    @pytest.mark.parametrize("kind", ["cuda", "cpu", "vulkan"])
    def test_kind_dependent_label_and_default_cap(self, kind):
        # No numeric cap in the text => the per-kind table supplies it; the
        # label/target follow the kind. Vulkan has NO cap (bindings become
        # descriptor-set layouts), so its message names "the HAL's cap".
        text = "hip/native_executable.c:223: exports[0] binding_flags exceed"
        exc = ValueError(text)
        with pytest.raises(evoxc.core.BackendError) as excinfo:
            iree._raise_if_dispatch_binding_limit(exc, kind)
        message = str(excinfo.value)
        assert "an unstated number of bindings" in message
        expected_target = iree._EXPECTED_TARGET_BY_KIND[kind]
        assert f"target {expected_target!r}" in message
        if kind == "vulkan":
            assert "the HAL's cap" in message
        else:
            assert f"cap of {iree._DISPATCH_BINDING_LIMITS[kind]}" in message

    def test_non_binding_limit_errors_are_left_alone(self):
        # Returns ``None`` (no raise) for both non-binding texts.
        assert (
            iree._raise_if_dispatch_binding_limit(
                ValueError(LOCAL_TASK_COMMAND_BUFFER_TEXT), "cpu"
            )
            is None
        )
        assert (
            iree._raise_if_dispatch_binding_limit(ValueError(MIXED_INSTALL_TEXT), "cpu")
            is None
        )
        # The HIP text only counts when the kind-independent message pattern
        # OR the runtime source marker matches — an unrelated ValueError with
        # the word "binding" is not enough.
        assert (
            iree._raise_if_dispatch_binding_limit(
                ValueError("binding a python callback failed"), "rocm"
            )
            is None
        )

    # -- the shared load path ----------------------------------------------

    def test_load_vm_module_rewrites_the_hip_binding_limit(self, monkeypatch):
        def _boom(vm_module, config):
            raise ValueError(HIP_BINDING_LIMIT_TEXT)

        _install_fake_iree_runtime(monkeypatch, _boom)
        with pytest.raises(evoxc.core.BackendError) as excinfo:
            iree._load_vm_module(b"\x00vmfb", "rocm", runtime_device=None)
        assert "needs 21 bindings" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_load_vm_module_passes_unrelated_errors_through(self, monkeypatch):
        original = ValueError("totally unrelated runtime failure")

        def _boom(vm_module, config):
            raise original

        _install_fake_iree_runtime(monkeypatch, _boom)
        with pytest.raises(ValueError) as excinfo:
            iree._load_vm_module(b"\x00vmfb", "cuda", runtime_device=None)
        # Same exception object, unchanged type and message (never wrapped).
        assert excinfo.value is original
        assert str(excinfo.value) == "totally unrelated runtime failure"

    def test_cpu_load_prioritises_the_binding_limit_over_the_mixed_install_wrapper(
        self, monkeypatch
    ):
        # On the CPU path the binding-limit recognition runs BEFORE the
        # mixed-install wrapper, so the message is never misattributed.

        def _boom(vmfb, driver=None):
            raise ValueError(HIP_BINDING_LIMIT_TEXT)

        _install_fake_iree_runtime(monkeypatch, None, flatbuffer_loader=_boom)
        with pytest.raises(evoxc.core.BackendError) as excinfo:
            iree._load_vm_module(b"\x00vmfb", "cpu", runtime_device=None)
        message = str(excinfo.value)
        assert "needs 21 bindings" in message
        assert "MIXED IREE INSTALL" not in message

    def test_cpu_load_keeps_the_mixed_install_wrapper(self, monkeypatch):
        def _boom(vmfb, driver=None):
            raise ValueError(MIXED_INSTALL_TEXT)

        _install_fake_iree_runtime(monkeypatch, None, flatbuffer_loader=_boom)
        with pytest.raises(evoxc.core.BackendError) as excinfo:
            iree._load_vm_module(b"\x00vmfb", "cpu", runtime_device=None)
        assert "MIXED IREE INSTALL" in str(excinfo.value)

    # -- the real driver shape (GPU-guarded) -------------------------------

    def test_real_rocm_driver_binding_wall(self):
        # K=15 pieces => 16 bindings (the cap): compiles, LOADS and runs.
        # K=16 pieces => 17 bindings: the HAL verifier rejects the module at
        # load and evoxc converts it into the actionable BackendError.
        # ONE compile per K on the shared MI300X, no timing.
        device = rocm_guard.require_rocm()
        specs15 = _binding_wall_specs(15)
        args15 = _binding_wall_args(15)
        exe = evoxc.build(
            _binding_wall_fn(15),
            *specs15,
            backend=NAME,
            target_backends=["rocm"],
            device=device,
        )
        out = evoxc.run(
            exe, *[evoxc.core.Tensor(a).to(device) for a in args15]
        )
        host = np.asarray(out.to(evoxc.core.Device("cpu", 0)).numpy())
        # Exactly-representable fp32 data ⇒ a bit-exact reduce + divide.
        expected = np.array(
            [a.astype(np.float64).sum() / _BINDING_WALL_DIVISOR for a in args15],
            dtype=np.float32,
        )
        assert host.shape == (15,)
        np.testing.assert_array_equal(host, expected)

        with pytest.raises(evoxc.core.BackendError) as excinfo:
            evoxc.build(
                _binding_wall_fn(16),
                *_binding_wall_specs(16),
                backend=NAME,
                target_backends=["rocm"],
                device=device,
            )
        message = str(excinfo.value)
        assert "needs 17 bindings" in message
        assert "cap of 16" in message
        # Release the rocm executable / device tensor HERE instead of leaving
        # their teardown to a later collection point in an unrelated suite:
        # with these finalizers still pending, the extra garbage shifts WHEN
        # the xla suites' payload/client finalizers run, and
        # ``xla_util.acquire_client`` holds the non-reentrant
        # ``_shared_clients_lock`` across ``plugin.create_client()`` — a
        # finalizer firing there self-deadlocks the process (measured: with
        # this test leaving its rocm objects alive, a full
        # ``pytest tests/backends`` run hung inside
        # ``test_xla_device_resident.py::test_load_device_validation``; the
        # same suite is green with the in-test collection below). The
        # lock-ordering bug itself is a separate product issue — see this
        # node's CONTEXT.md notes.
        del exe, out, host
        gc.collect()
