"""Local CPU backends reject the compiler escape hatches at ``lower()``.

``custom_call`` / ``inline_stablehlo`` are forwarded to an EXTERNAL compiler by
the compiler adapters; a local interpreter has nothing to execute, so both the
numpy reference backend and the native Rust CPU engine reject them at
``lower()`` with an explicit ``core.BackendError`` naming the op — never a
silent pass-through and never a generic "no kernel" drift message.

The graphs are valid IR (a ``verify()`` negative control proves the rejection
comes from the backend capability gate, not verification). The native leg skips
when the optional extension is not built.
"""

import numpy as np
import pytest

import evoxc
from evoxc import core

from tests.backends import _native_utils as nu

_SPEC = evoxc.TensorSpec((2, 3), evoxc.float32)


def _custom_call_fn(x):
    return evoxc.custom_call(
        "some_target", x, result=evoxc.TensorSpec((2, 3), evoxc.float32)
    )


def _inline_fn(x):
    return evoxc.inline_stablehlo(
        "%out0 = stablehlo.tanh %in0 : tensor<2x3xf32>",
        x,
        result=evoxc.TensorSpec((2, 3), evoxc.float32),
    )


_CASES = [
    ("custom_call", _custom_call_fn),
    ("inline_stablehlo", _inline_fn),
]


@pytest.mark.parametrize(
    "op_name, fn", _CASES, ids=[case[0] for case in _CASES]
)
def test_numpy_lower_rejects_custom_ops_naming_the_op(op_name, fn):
    graph = evoxc.trace(fn, _SPEC)
    # Negative control: the graph is valid IR — the rejection is lower-time.
    graph.verify()
    with pytest.raises(core.BackendError) as excinfo:
        evoxc.lower(graph, backend="numpy")
    message = str(excinfo.value)
    assert op_name in message
    assert "numpy" in message
    # The message points at the sanctioned alternatives (a Python kernel or a
    # compiler backend), never a silent fallback.
    assert "compiler backend" in message or "iree" in message


@pytest.mark.parametrize(
    "op_name, fn", _CASES, ids=[case[0] for case in _CASES]
)
def test_numpy_backend_lower_rejects_custom_ops_directly(op_name, fn):
    # The same gate is reachable through the backend object itself.
    graph = evoxc.trace(fn, _SPEC)
    with pytest.raises(core.BackendError) as excinfo:
        evoxc.numpy_backend.lower(graph)
    assert op_name in str(excinfo.value)


@pytest.mark.skipif(not nu.ENGINE_AVAILABLE, reason=nu.SKIP_REASON)
@pytest.mark.parametrize(
    "op_name, fn", _CASES, ids=[case[0] for case in _CASES]
)
def test_native_lower_rejects_custom_ops_naming_the_op(op_name, fn):
    graph = evoxc.trace(fn, _SPEC)
    graph.verify()
    with pytest.raises(core.BackendError) as excinfo:
        evoxc.lower(graph, backend="native")
    message = str(excinfo.value)
    assert op_name in message
    assert "native" in message
