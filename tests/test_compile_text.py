"""tests/test_compile_text.py — ``evoxc.compile_text``: the explicit-staging
escape hatch that compiles hand-written / hand-EDITED StableHLO text.

``compile_text(mlir_text, input_specs, output_specs, *, backend=None,
device=None, entry_functions=("main",), **options) -> Executable`` builds the
shared StableHLO ``LoweredProgram`` record directly (there is NO tracing and
NO ``Graph``: the text IS the program), then forwards ``device``/``**options``
unchanged to the normal ``compile`` + ``load`` stages (no stage bypassed,
nothing re-lowered). ``backend`` must be a ``CompilerBackend`` — the numpy
interpreter and the native CPU engine reject the call with a ``BackendError``
naming the backend.

The whole file is iree-guarded: a compiler-free host skips it cleanly. The
real compiles are cached module-wide (one ``compile_str`` per distinct text /
option set), so the suite stays a correctness check rather than a compiler
benchmark. CPU only (``target_backends=["llvm-cpu"]``) — no GPU anywhere.
"""

import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import evoxc
from evoxc.backends import stablehlo

# ---------------------------------------------------------------------------
# Hand-written StableHLO programs (the text IS the program)
# ---------------------------------------------------------------------------

#: A dynamic-batch ``tanh`` — proves a symbolic dim (``tensor<?x4xf32>``)
#: travels through the summary/signature without tracing.
_TANH_TEXT = """
module {
  func.func public @main(%arg0: tensor<?x4xf32>) -> tensor<?x4xf32> {
    %0 = stablehlo.tanh %arg0 : tensor<?x4xf32>
    return %0 : tensor<?x4xf32>
  }
}
"""

#: Two inputs, two outputs: the flat-tuple output signature in one program.
_SUM_DIFF_TEXT = """
module {
  func.func public @main(%lhs: tensor<2x3xf32>, %rhs: tensor<2x3xf32>)
      -> (tensor<2x3xf32>, tensor<2x3xf32>) {
    %sum = stablehlo.add %lhs, %rhs : tensor<2x3xf32>
    %diff = stablehlo.subtract %lhs, %rhs : tensor<2x3xf32>
    return %sum, %diff : tensor<2x3xf32>, tensor<2x3xf32>
  }
}
"""

_DYNAMIC_SPEC = evoxc.TensorSpec((evoxc.Dim("N"), 4), evoxc.float32)
_STATIC_SPEC = evoxc.TensorSpec((4,), evoxc.float32)
_PAIR_SPEC = evoxc.TensorSpec((2, 3), evoxc.float32)

#: One compiled Executable per distinct (text, specs, options) cell.
_EXECUTABLES: dict = {}


def _compile_once(key, text, inputs, outputs, **kwargs):
    """Compile ``text`` once per process and reuse the executable."""
    executable = _EXECUTABLES.get(key)
    if executable is None:
        executable = evoxc.compile_text(
            text, inputs, outputs, backend="iree", **kwargs
        )
        _EXECUTABLES[key] = executable
    return executable


def _numpy_of(value):
    """Tensor / ndarray -> host ndarray."""
    return np.asarray(value.numpy() if isinstance(value, evoxc.Tensor) else value)


# ---------------------------------------------------------------------------
# 1. A hand-written dynamic-shape module compiles and runs
# ---------------------------------------------------------------------------


def test_handwritten_tanh_dynamic_dim_matches_numpy():
    executable = _compile_once(
        "tanh", _TANH_TEXT, [_DYNAMIC_SPEC], [_DYNAMIC_SPEC]
    )
    # The recorded signature carries the symbolic dim.
    (input_spec,) = executable.signature.input_specs
    assert input_spec.shape[0] == evoxc.Dim("N")
    assert input_spec.shape[1] == 4
    # The SAME executable serves two different batch sizes (dynamic dim).
    for rows in (3, 5):
        x = np.linspace(-1.5, 1.5, rows * 4, dtype=np.float32).reshape(rows, 4)
        (out,) = evoxc.run(executable, x)
        assert isinstance(out, evoxc.Tensor)
        np.testing.assert_allclose(_numpy_of(out), np.tanh(x), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Two inputs / two outputs -> evoxc.run(exe, a, b) and a FLAT output tuple
# ---------------------------------------------------------------------------


def test_two_input_text_runs_and_returns_a_flat_tuple():
    executable = _compile_once(
        "sum-diff",
        _SUM_DIFF_TEXT,
        [_PAIR_SPEC, _PAIR_SPEC],
        [_PAIR_SPEC, _PAIR_SPEC],
    )
    lhs = np.arange(6, dtype=np.float32).reshape(2, 3)
    rhs = np.ones((2, 3), dtype=np.float32)

    result = evoxc.run(executable, lhs, rhs)

    # Binding signature: outputs are ALWAYS one flat tuple of tensors, even
    # for a single output (see the one-output cases above).
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(leaf, evoxc.Tensor) for leaf in result)
    np.testing.assert_array_equal(_numpy_of(result[0]), lhs + rhs)
    np.testing.assert_array_equal(_numpy_of(result[1]), lhs - rhs)


# ---------------------------------------------------------------------------
# 3. entry_functions / device handling and options forwarding
# ---------------------------------------------------------------------------


def test_entry_functions_and_device_are_forwarded():
    executable = _compile_once(
        "tanh-entry-device",
        _TANH_TEXT,
        [_DYNAMIC_SPEC],
        [_DYNAMIC_SPEC],
        entry_functions=("main",),
        device=evoxc.Device("cpu", 0),
    )
    assert executable.device == evoxc.Device("cpu", 0)
    # The default device is the v1 cpu:0 default, unchanged by compile_text.
    default = _compile_once("tanh", _TANH_TEXT, [_DYNAMIC_SPEC], [_DYNAMIC_SPEC])
    assert default.device == evoxc.Device("cpu", 0)
    x = np.arange(8, dtype=np.float32).reshape(2, 4) / 8.0
    (out,) = evoxc.run(executable, x)
    np.testing.assert_allclose(_numpy_of(out), np.tanh(x), rtol=1e-5, atol=1e-6)


def test_options_are_forwarded_to_the_compiler_stage():
    """``target_backends`` reaches the iree compiler (explicit llvm-cpu) and
    an invalid target fails there with the adapter's ``BackendError``."""
    executable = _compile_once(
        "tanh-llvm-cpu",
        _TANH_TEXT,
        [_DYNAMIC_SPEC],
        [_DYNAMIC_SPEC],
        target_backends=["llvm-cpu"],
    )
    x = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(2, 4)
    (out,) = evoxc.run(executable, x)
    np.testing.assert_allclose(_numpy_of(out), np.tanh(x), rtol=1e-5, atol=1e-6)

    with pytest.raises(evoxc.BackendError) as excinfo:
        evoxc.compile_text(
            _TANH_TEXT,
            [_DYNAMIC_SPEC],
            [_DYNAMIC_SPEC],
            backend="iree",
            target_backends=["no-such-target"],
        )
    assert "no-such-target" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 4. Error paths (no compiler involved)
# ---------------------------------------------------------------------------


def test_non_string_text_rejected():
    for bad in (None, 42, b"module {}"):
        with pytest.raises(TypeError, match="mlir_text must be a non-empty"):
            evoxc.compile_text(bad, [], [_STATIC_SPEC], backend="iree")


def test_empty_or_whitespace_text_rejected():
    for bad in ("", "   \n\t "):
        with pytest.raises(TypeError, match="empty/whitespace-only"):
            evoxc.compile_text(bad, [], [_STATIC_SPEC], backend="iree")


def test_non_tensor_spec_sequences_rejected():
    with pytest.raises(TypeError, match="input_specs\\[0\\] must be a core.TensorSpec"):
        evoxc.compile_text(_TANH_TEXT, [(4,)], [_STATIC_SPEC], backend="iree")
    with pytest.raises(TypeError, match="output_specs\\[1\\] must be a core.TensorSpec"):
        evoxc.compile_text(
            _TANH_TEXT, [], [_STATIC_SPEC, "nope"], backend="iree"
        )
    # A bare string is a sequence, but never a spec sequence.
    with pytest.raises(TypeError, match="input_specs must be a sequence"):
        evoxc.compile_text(_TANH_TEXT, "main", [_STATIC_SPEC], backend="iree")


def test_empty_output_specs_rejected():
    with pytest.raises(TypeError, match="output_specs must contain at least one"):
        evoxc.compile_text(_TANH_TEXT, [_STATIC_SPEC], [], backend="iree")


def test_bad_entry_functions_rejected():
    with pytest.raises(TypeError, match="entry_functions must be a non-empty"):
        evoxc.compile_text(
            _TANH_TEXT, [], [_STATIC_SPEC], backend="iree", entry_functions=()
        )
    with pytest.raises(TypeError, match="entry_functions must be a non-empty"):
        evoxc.compile_text(
            _TANH_TEXT, [], [_STATIC_SPEC], backend="iree", entry_functions="main"
        )
    with pytest.raises(TypeError, match="entry_functions\\[1\\] must be a non-empty"):
        evoxc.compile_text(
            _TANH_TEXT,
            [],
            [_STATIC_SPEC],
            backend="iree",
            entry_functions=("main", 7),
        )


def test_interpreter_backends_rejected_with_the_backend_named():
    from evoxc.backends.native import engine as native_engine

    for backend in ("numpy", "native"):
        if backend == "native" and not native_engine.engine_available():
            # Host guard: without the built Rust extension ``get("native")``
            # raises its own actionable BackendError before compile_text can
            # gate it (pinned by test_native_backend_missing_engine.py).
            pytest.skip("the native engine extension is not built")
        with pytest.raises(evoxc.BackendError) as excinfo:
            evoxc.compile_text(_TANH_TEXT, [], [_STATIC_SPEC], backend=backend)
        message = str(excinfo.value)
        assert f"backend {backend!r}" in message
        assert "compiler backend such as backend='iree'" in message


# ---------------------------------------------------------------------------
# 5. Round-trip edit flow: export -> compile_text -> same values
# ---------------------------------------------------------------------------


def _tanh_plus_one(x):
    return evoxc.add(evoxc.tanh(x), 1.0)


def test_exported_text_recompiles_to_the_same_values():
    spec = evoxc.TensorSpec((2, 4), evoxc.float32)
    graph = evoxc.trace(_tanh_plus_one, spec)
    exported = stablehlo.export(graph)

    original = evoxc.build(_tanh_plus_one, spec, backend="iree")
    recompiled = evoxc.compile_text(
        exported, [spec], [spec], backend="iree"
    )

    x = np.linspace(-2.0, 2.0, 8, dtype=np.float32).reshape(2, 4)
    # ``build`` returns the defn's own output tree (a bare Tensor here); the
    # text path's output tree is the flat spec tuple, so its result is a
    # 1-tuple — the documented compile_text signature.
    wanted = evoxc.run(original, x)
    (got,) = evoxc.run(recompiled, x)
    np.testing.assert_allclose(
        _numpy_of(got), _numpy_of(wanted), rtol=1e-6, atol=1e-6
    )
