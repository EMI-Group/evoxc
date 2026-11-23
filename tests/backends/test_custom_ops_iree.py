"""End-to-end compiler escape hatches through the ``iree`` adapter.

Pinned here through ``backend="iree"`` on llvm-cpu:

* an ``inline_stablehlo``-only graph (a raw ``stablehlo.tanh`` snippet)
  compiles and RUNS with numpy parity, host-resident (cpu:0), exactly like any
  other iree graph — the exporter forwards the snippet verbatim and the
  compiler consumes it;
* a ``custom_call`` graph reaches IREE and is REJECTED by legalization (not by
  the parser): the compiler diagnostic names ``stablehlo.custom_call`` and the
  target, documenting that IREE only legalizes its own custom-call target set
  (evoxc forwards the op; the compiler decides whether the target exists).

IREE runs the real MLIR compiler, so the 2s-per-file convention does not apply.
The module skips cleanly when iree is absent (``importorskip``).
"""

import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import evoxc

_SPEC = evoxc.TensorSpec((2, 3), evoxc.float32)
_INPUT = np.array(
    [[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]], dtype=np.float32
)


def _inline_tanh(x):
    return evoxc.inline_stablehlo(
        "%out0 = stablehlo.tanh %in0 : tensor<2x3xf32>",
        x,
        result=evoxc.TensorSpec((2, 3), evoxc.float32),
    )


def test_inline_stablehlo_compiles_and_runs_with_numpy_parity():
    exe = evoxc.build(_inline_tanh, _SPEC, backend="iree")
    out = evoxc.run(exe, _INPUT)
    assert isinstance(out, evoxc.Tensor)
    # Host-resident, like every other llvm-cpu iree result.
    assert out.device == evoxc.Device("cpu", 0)
    assert out.shape == (2, 3)
    assert out.dtype == np.dtype("float32")
    np.testing.assert_allclose(out.numpy(), np.tanh(_INPUT), rtol=1e-6, atol=1e-6)


def test_inline_stablehlo_binds_operand_to_preceding_op():
    # The snippet consumes an op result (not just a function argument), so the
    # substituted SSA name is the preceding op's — the snippet is spliced into
    # the surrounding program, not isolated.
    def fn(x):
        y = evoxc.add(x, x)
        return evoxc.inline_stablehlo(
            "%out0 = stablehlo.tanh %in0 : tensor<2x3xf32>",
            y,
            result=evoxc.TensorSpec((2, 3), evoxc.float32),
        )

    exe = evoxc.build(fn, _SPEC, backend="iree")
    out = evoxc.run(exe, _INPUT).numpy()
    np.testing.assert_allclose(out, np.tanh(_INPUT + _INPUT), rtol=1e-6, atol=1e-6)


def test_custom_call_is_rejected_by_iree_legalization():
    def fn(x):
        return evoxc.custom_call(
            "my_unregistered_target",
            x,
            result=evoxc.TensorSpec((2, 3), evoxc.float32),
        )

    with pytest.raises(evoxc.BackendError) as excinfo:
        evoxc.build(fn, _SPEC, backend="iree")
    message = str(excinfo.value)
    # The op and its target reach IREE intact ...
    assert "stablehlo.custom_call" in message
    assert "my_unregistered_target" in message
    # ... and the failure is the LEGALIZATION verdict, never a parse/syntax
    # error (which would mean the exporter emitted malformed MLIR).
    assert "failed to legalize" in message
    lowered = message.lower()
    assert "syntax error" not in lowered
    assert "parse error" not in lowered
