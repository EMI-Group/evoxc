"""Contract tests for the two compiler escape hatches
(``evoxc.custom_call`` / ``evoxc.inline_stablehlo``).

Authority: ``evoxc/ops/custom.py`` and ``evoxc/ops/CONTEXT.md`` ("Custom-call /
raw-StableHLO ops"). Both ops are OPAQUE compiler declarations — evoxc never
executes, interprets or validates the target/snippet; it carries them into the
emitted program. Pinned here:

- trace-time attributes (``call_target_name`` / ``backend_config`` /
  ``api_version`` / ``has_side_effect`` for ``custom_call``; ``body`` /
  ``has_side_effect`` for ``inline_stablehlo``) plus the declared
  ``result_specs``-derived result types and the ``callback`` effect;
- single ``TensorSpec`` result -> one ``SymbolicTensor``; tuple/list of specs
  -> a tuple (one per spec);
- Python-scalar operands auto-promoted to 0-d ``constant`` ops;
- the ``custom`` IR category op count;
- every frontend ``TypeError`` (bad target/body/config/api-version/flag/result)
  and the ``api_version >= 1`` value constraint enforced by ``ir.verify``;
- the ``inline_stablehlo`` token-level SSA contract (``%in<i>`` / ``%out<i>``
  ranges, every declared result assigned, and ``%in0x`` NOT read as ``%in0``);
- the no-active-trace / concrete-``Tensor``-operand ``TraceError`` paths;
- graph save/load round-trip (attributes + digest) and ``graph.text()``;
- explicit backend rejection on the numpy interpreter and the native CPU
  engine (``BackendError`` naming the op — never a silent fallback).
"""
from __future__ import annotations

import contextlib
import importlib

import numpy as np
import pytest

import evoxc

from tests.ops.conftest import ops_of

_F32 = evoxc.float32
_I64 = evoxc.int64

#: A valid single-result snippet: %in0 used twice, %out0 assigned.
_BODY_ADD = "%out0 = stablehlo.add %in0, %in0 : tensor<4xf32>"


def _spec(shape=(4,), dtype=_F32):
    return evoxc.TensorSpec(shape, dtype)


def _one_op(graph, name):
    ops = ops_of(graph, name)
    assert len(ops) == 1, f"expected exactly one {name!r} op, got {len(ops)}"
    return ops[0]


def _custom_call_fn(result=None, **kwargs):
    """A traceable fn issuing one ``custom_call`` on its ``x`` operand."""
    result = _spec() if result is None else result

    def f(x):
        return evoxc.custom_call("my.target", x, result=result, **kwargs)

    return f


def _inline_fn(body=_BODY_ADD, result=None, **kwargs):
    result = _spec() if result is None else result

    def f(x):
        return evoxc.inline_stablehlo(body, x, result=result, **kwargs)

    return f


# ---------------------------------------------------------------------------
# Trace-time attributes and result types
# ---------------------------------------------------------------------------

def test_custom_call_attributes_and_result_types():
    @evoxc.defn
    def f(x):
        return evoxc.custom_call(
            "my.target",
            x,
            result=_spec((2, 3)),
            backend_config='cfg="v"',
            api_version=2,
            has_side_effect=True,
        )

    graph = evoxc.trace(f, _spec())
    op = _one_op(graph, "custom_call")
    attrs = op.attributes

    assert op.effect == "callback"
    assert attrs["call_target_name"] == "my.target"
    assert attrs["backend_config"] == 'cfg="v"'
    assert attrs["api_version"] == 2
    assert attrs["has_side_effect"] is True

    # Result types resolve from result_specs (no shape_fn).
    (spec,) = attrs["result_specs"]
    assert spec.dtype == np.dtype("float32")
    assert spec.shape == (2, 3)
    assert op.results[0].type.dtype == np.dtype("float32")
    assert op.results[0].type.shape == (2, 3)


def test_custom_call_defaults():
    graph = evoxc.trace(_custom_call_fn(), _spec())
    attrs = _one_op(graph, "custom_call").attributes

    assert attrs["backend_config"] == ""
    assert attrs["api_version"] == 1
    assert attrs["has_side_effect"] is False
    # The default attribute values verify cleanly.
    assert graph.verify() is None


def test_inline_stablehlo_attributes_and_result_types():
    body = "%out0:2 = stablehlo.multiply %in0, %in1 : tensor<3xf32>"

    @evoxc.defn
    def f(x, y):
        return evoxc.inline_stablehlo(
            body, x, y, result=_spec((3,)), has_side_effect=True
        )

    graph = evoxc.trace(f, _spec((3,)), _spec((3,)))
    op = _one_op(graph, "inline_stablehlo")
    attrs = op.attributes

    assert op.effect == "callback"
    assert attrs["body"] == body
    assert attrs["has_side_effect"] is True
    (spec,) = attrs["result_specs"]
    assert spec.dtype == np.dtype("float32")
    assert spec.shape == (3,)
    assert op.results[0].type.dtype == np.dtype("float32")
    assert op.results[0].type.shape == (3,)


# ---------------------------------------------------------------------------
# Single spec vs tuple/list of specs (return types)
# ---------------------------------------------------------------------------

def test_single_spec_returns_one_symbolic_tensor():
    captured = {}

    def f(x):
        y = evoxc.custom_call("t", x, result=_spec((4,)))
        captured["y"] = y
        return y

    evoxc.trace(f, _spec())
    y = captured["y"]
    assert isinstance(y, evoxc.SymbolicTensor)
    assert not isinstance(y, tuple)
    assert y.dtype == np.dtype("float32")
    assert y.shape == (4,)


@pytest.mark.parametrize("container", [tuple, list], ids=["tuple", "list"])
def test_multi_spec_returns_tuple_one_per_spec(container):
    specs = container([_spec((3,), _I64), _spec((2, 3), _F32)])
    captured = {}

    def f(x):
        out = evoxc.custom_call("t", x, result=specs)
        captured["out"] = out
        return out

    graph = evoxc.trace(f, _spec())
    out = captured["out"]
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].dtype == np.dtype("int64") and out[0].shape == (3,)
    assert out[1].dtype == np.dtype("float32") and out[1].shape == (2, 3)

    op = _one_op(graph, "custom_call")
    assert len(op.results) == 2
    assert op.results[0].type.dtype == np.dtype("int64")
    assert op.results[1].type.shape == (2, 3)


def test_inline_multi_spec_returns_tuple():
    body = (
        "%out0 = stablehlo.add %in0, %in0 : tensor<4xf32>\n"
        "%out1 = stablehlo.multiply %in0, %in0 : tensor<4xf32>"
    )

    def f(x):
        return evoxc.inline_stablehlo(
            body, x, result=(_spec(), _spec())
        )

    graph = evoxc.trace(f, _spec())
    op = _one_op(graph, "inline_stablehlo")
    assert len(op.results) == 2


# ---------------------------------------------------------------------------
# Python-scalar operand promotion
# ---------------------------------------------------------------------------

def test_custom_call_promotes_scalar_operands_to_zero_d_constants():
    def f(x):
        return evoxc.custom_call("t", x, 2, result=_spec())

    graph = evoxc.trace(f, _spec())
    op = _one_op(graph, "custom_call")
    assert len(op.operands) == 2

    constants = ops_of(graph, "constant")
    assert len(constants) == 1
    assert constants[0].results[0].type.shape == ()
    assert constants[0].results[0].type.dtype == np.dtype("int64")


def test_inline_promotes_scalar_operands_to_zero_d_constants():
    body = "%out0 = stablehlo.add %in0, %in1 : tensor<4xf32>"

    def f(x):
        return evoxc.inline_stablehlo(body, x, 1.5, result=_spec())

    graph = evoxc.trace(f, _spec())
    op = _one_op(graph, "inline_stablehlo")
    assert len(op.operands) == 2

    (constant,) = ops_of(graph, "constant")
    assert constant.results[0].type.shape == ()
    assert constant.results[0].type.dtype == np.dtype("float64")


# ---------------------------------------------------------------------------
# IR category / op counts
# ---------------------------------------------------------------------------

def test_both_ops_count_in_the_custom_category():
    def f(x):
        a = evoxc.custom_call("t", x, result=_spec())
        b = evoxc.inline_stablehlo(_BODY_ADD, x, result=_spec())
        return a, b

    graph = evoxc.trace(f, _spec())
    counts = graph.summary()["op_counts_by_category"]
    assert counts["custom"] == 2
    assert len(ops_of(graph, "custom_call")) == 1
    assert len(ops_of(graph, "inline_stablehlo")) == 1


# ---------------------------------------------------------------------------
# Frontend TypeErrors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad,kind",
    [(42, "int"), ("", "empty str"), (None, "NoneType")],
    ids=["int", "empty-str", "none"],
)
def test_custom_call_rejects_bad_call_target_name(bad, kind):
    def f(x):
        return evoxc.custom_call(bad, x, result=_spec())

    with pytest.raises(
        TypeError,
        match=rf"custom_call: call_target_name must be a non-empty str, got {kind}",
    ):
        evoxc.trace(f, _spec())


def test_custom_call_rejects_non_str_backend_config():
    def f(x):
        return evoxc.custom_call("t", x, result=_spec(), backend_config=42)

    with pytest.raises(
        TypeError, match="custom_call: backend_config must be a str, got int"
    ):
        evoxc.trace(f, _spec())


@pytest.mark.parametrize(
    "bad,kind",
    [(True, "bool"), ("1", "str"), (1.0, "float")],
    ids=["bool", "str", "float"],
)
def test_custom_call_rejects_non_int_api_version(bad, kind):
    def f(x):
        return evoxc.custom_call("t", x, result=_spec(), api_version=bad)

    with pytest.raises(
        TypeError,
        match=rf"custom_call: api_version must be an int, got {kind}",
    ):
        evoxc.trace(f, _spec())


@pytest.mark.parametrize("bad", [1, "yes", None], ids=["int", "str", "none"])
def test_custom_call_rejects_non_bool_has_side_effect(bad):
    def f(x):
        return evoxc.custom_call("t", x, result=_spec(), has_side_effect=bad)

    with pytest.raises(
        TypeError, match="custom_call: has_side_effect must be a bool"
    ):
        evoxc.trace(f, _spec())


@pytest.mark.parametrize(
    "bad,msg",
    [
        (42, "result must be a TensorSpec"),
        ((), "non-empty"),
        ((_spec(), 42), "non-empty tuple/list of TensorSpecs"),
    ],
    ids=["not-a-spec", "empty-tuple", "tuple-with-non-spec"],
)
def test_custom_call_rejects_bad_result_specs(bad, msg):
    def f(x):
        return evoxc.custom_call("t", x, result=bad)

    with pytest.raises(TypeError, match=msg):
        evoxc.trace(f, _spec())


@pytest.mark.parametrize(
    "bad,kind",
    [(42, "int"), ("", "empty str"), (None, "NoneType")],
    ids=["int", "empty-str", "none"],
)
def test_inline_rejects_bad_body(bad, kind):
    def f(x):
        return evoxc.inline_stablehlo(bad, x, result=_spec())

    with pytest.raises(
        TypeError,
        match=rf"inline_stablehlo: body must be a non-empty str, got {kind}",
    ):
        evoxc.trace(f, _spec())


@pytest.mark.parametrize("bad", [1, "yes", None], ids=["int", "str", "none"])
def test_inline_rejects_non_bool_has_side_effect(bad):
    def f(x):
        return evoxc.inline_stablehlo(
            _BODY_ADD, x, result=_spec(), has_side_effect=bad
        )

    with pytest.raises(
        TypeError, match="inline_stablehlo: has_side_effect must be a bool"
    ):
        evoxc.trace(f, _spec())


@pytest.mark.parametrize(
    "bad,msg",
    [
        (42, "result must be a TensorSpec"),
        ((), "non-empty"),
        ((_spec(), 42), "non-empty tuple/list of TensorSpecs"),
    ],
    ids=["not-a-spec", "empty-tuple", "tuple-with-non-spec"],
)
def test_inline_rejects_bad_result_specs(bad, msg):
    def f(x):
        return evoxc.inline_stablehlo(_BODY_ADD, x, result=bad)

    with pytest.raises(TypeError, match=msg):
        evoxc.trace(f, _spec())


def test_custom_call_api_version_below_one_fails_ir_verification():
    """``api_version``'s ``>= 1`` VALUE constraint is enforced by
    ``ir.verify`` (the frontend only rejects the wrong TYPE)."""
    def f(x):
        return evoxc.custom_call("t", x, result=_spec(), api_version=0)

    graph = evoxc.trace(f, _spec())  # tracing itself does not verify
    with pytest.raises(
        evoxc.VerificationError,
        match=r"api_version' must be an int >= 1, got 0",
    ):
        graph.verify()


# ---------------------------------------------------------------------------
# inline_stablehlo SSA contract (token-level)
# ---------------------------------------------------------------------------

def test_inline_rejects_in_operand_out_of_range():
    body = "%out0 = stablehlo.add %in1, %in1 : tensor<4xf32>"

    def f(x):
        return evoxc.inline_stablehlo(body, x, result=_spec())

    with pytest.raises(
        evoxc.TraceError,
        match=r"%in1 is out of range \(1 operand\(s\) available\)",
    ):
        evoxc.trace(f, _spec())


def test_inline_rejects_out_operand_out_of_range():
    body = (
        "%out0 = stablehlo.add %in0, %in0 : tensor<4xf32>\n"
        "%out5 = stablehlo.add %in0, %in0 : tensor<4xf32>"
    )

    def f(x):
        return evoxc.inline_stablehlo(body, x, result=_spec())

    with pytest.raises(
        evoxc.TraceError,
        match=r"%out5 is out of range \(1 result\(s\) declared\)",
    ):
        evoxc.trace(f, _spec())


def test_inline_rejects_declared_result_never_assigned():
    body = "%out0 = stablehlo.add %in0, %in0 : tensor<4xf32>"

    def f(x):
        return evoxc.inline_stablehlo(body, x, result=(_spec(), _spec()))

    with pytest.raises(
        evoxc.TraceError,
        match=r"declared result %out1 is never assigned",
    ):
        evoxc.trace(f, _spec())


def test_inline_in0x_is_not_misread_as_in0():
    """The trailing lookahead keeps ``%in0x`` from being read as ``%in0``:
    with ZERO operands this snippet is valid."""
    body = "%out0 = stablehlo.add %in0x, %in0x : tensor<4xf32>"

    def f(x):
        out = evoxc.inline_stablehlo(body, result=_spec())
        return out

    graph = evoxc.trace(f, _spec())
    op = _one_op(graph, "inline_stablehlo")
    assert op.attributes["body"] == body
    assert op.results[0].type.shape == (4,)


def test_inline_out0x_is_not_counted_as_an_assignment():
    """The same lookahead applies to assignments: ``%out0x = ...`` does NOT
    define declared result ``%out0``."""
    body = "%out0x = stablehlo.add %in0, %in0 : tensor<4xf32>"

    def f(x):
        return evoxc.inline_stablehlo(body, x, result=_spec())

    with pytest.raises(
        evoxc.TraceError,
        match=r"declared result %out0 is never assigned",
    ):
        evoxc.trace(f, _spec())


# ---------------------------------------------------------------------------
# No-active-trace / concrete-Tensor-operand errors
# ---------------------------------------------------------------------------

def test_both_ops_outside_trace_raise_no_active_trace():
    for call in (
        lambda: evoxc.custom_call("t", 1.0, result=_spec()),
        lambda: evoxc.inline_stablehlo(_BODY_ADD, 1.0, result=_spec()),
    ):
        with pytest.raises(evoxc.TraceError) as exc:
            call()
        message = str(exc.value)
        assert message.startswith(
            "No active trace: tensor ops can only be called while tracing"
        )
        assert "evoxc.trace" in message
        assert "evoxc.evaluate" in message


@pytest.mark.parametrize(
    "call", ["custom_call", "inline_stablehlo"], ids=["custom_call", "inline"]
)
def test_both_ops_reject_concrete_tensor_operand(call):
    tensor = evoxc.tensor(np.array([1.0, 2.0], dtype=np.float32))

    def f(x):
        if call == "custom_call":
            return evoxc.custom_call("t", tensor, result=_spec((2,)))
        return evoxc.inline_stablehlo(
            "%out0 = stablehlo.add %in0, %in0 : tensor<2xf32>",
            tensor,
            result=_spec((2,)),
        )

    with pytest.raises(evoxc.TraceError) as exc:
        evoxc.trace(f, _spec((2,)))
    message = str(exc.value)
    assert message.startswith(
        "Concrete Tensor operands are not allowed in graph ops "
        "(evoxc has no eager mode)"
    )
    assert "explicit input" in message
    assert "evoxc.constant" in message
    assert "evoxc.evaluate" in message


# ---------------------------------------------------------------------------
# Graph round-trip + text rendering
# ---------------------------------------------------------------------------

def _mixed_graph():
    def f(x):
        a = evoxc.custom_call(
            "my.target",
            x,
            result=_spec((2, 3)),
            backend_config="c=1",
            api_version=2,
            has_side_effect=True,
        )
        b = evoxc.inline_stablehlo(
            _BODY_ADD, x, result=_spec(), has_side_effect=False
        )
        return a, b

    return evoxc.trace(f, _spec())


def test_graph_roundtrip_preserves_attributes_and_digest(tmp_path):
    graph = _mixed_graph()
    digest = graph.digest()

    path = tmp_path / "custom_ops.evoxcgraph"
    graph.save(str(path))
    loaded = evoxc.Graph.load(str(path))

    assert loaded.digest() == digest

    original = _one_op(graph, "custom_call").attributes
    restored = _one_op(loaded, "custom_call").attributes
    for key in (
        "call_target_name",
        "backend_config",
        "api_version",
        "has_side_effect",
    ):
        assert restored[key] == original[key]
    assert tuple(restored["result_specs"]) == tuple(original["result_specs"])

    original_inline = _one_op(graph, "inline_stablehlo").attributes
    restored_inline = _one_op(loaded, "inline_stablehlo").attributes
    assert restored_inline["body"] == original_inline["body"]
    assert (
        restored_inline["has_side_effect"]
        == original_inline["has_side_effect"]
    )
    assert tuple(restored_inline["result_specs"]) == tuple(
        original_inline["result_specs"]
    )


def test_graph_text_renders_both_calls():
    text = _mixed_graph().text()
    assert "custom_call" in text
    assert '"my.target"' in text
    assert '"c=1"' in text
    assert "inline_stablehlo" in text
    assert "stablehlo.add" in text
    assert "%out0" in text


# ---------------------------------------------------------------------------
# Backend rejection (explicit, never a silent fallback)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _native_backend_registered_without_probe():
    """Expose the REAL ``native`` backend singleton through the registry
    without probing the optional Rust extension.

    ``registry.get("native")`` calls ``register()``, which probes
    ``native_compile`` and raises when the extension is absent. The module
    itself imports no extension at module scope, and ``NativeBackend.lower``
    runs its Python capability pre-check BEFORE the engine gate — so the
    custom-op rejection is reachable in a compiler/engine-free environment
    (same save/restore pattern as ``tests/ops/test_random.py``).
    """
    from evoxc.backends import registry as _registry

    module = importlib.import_module(_registry.OPTIONAL_ADAPTERS["native"])
    backend = getattr(module, "native_backend")
    table = _registry._registry
    previous = table.get("native")
    table["native"] = backend
    try:
        yield backend
    finally:
        if previous is None:
            table.pop("native", None)
        else:
            table["native"] = previous


@pytest.mark.parametrize(
    "name,make",
    [
        ("custom_call", _custom_call_fn),
        ("inline_stablehlo", _inline_fn),
    ],
    ids=["custom_call", "inline_stablehlo"],
)
def test_numpy_backend_rejects_compiler_escape_hatches(name, make):
    with pytest.raises(evoxc.BackendError) as exc:
        evoxc.build(make(), _spec(), backend="numpy")
    message = str(exc.value)
    assert "numpy backend cannot execute" in message
    assert f"'{name}'" in message


@pytest.mark.parametrize(
    "name,make",
    [
        ("custom_call", _custom_call_fn),
        ("inline_stablehlo", _inline_fn),
    ],
    ids=["custom_call", "inline_stablehlo"],
)
def test_native_backend_rejects_compiler_escape_hatches(name, make):
    with _native_backend_registered_without_probe():
        with pytest.raises(evoxc.BackendError) as exc:
            evoxc.build(make(), _spec(), backend="native")
    message = str(exc.value)
    assert f"the native backend cannot execute '{name}'" in message
