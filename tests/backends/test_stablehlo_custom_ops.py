"""StableHLO export of the compiler escape hatches.

Contract under test: ``evoxc/backends/stablehlo/writer.py``'s
``_emit_custom_call`` / ``_emit_inline_stablehlo`` (the two IR ops declared in
``evoxc/ir/op_defs/custom.py`` and built by ``evoxc.custom_call`` /
``evoxc.inline_stablehlo``). Pinned here:

* ``custom_call`` golden text: single / multiple / ZERO results; the target
  printed as a bare ``@name`` or a quoted ``@"..."`` symbol (with ``\\``/``"``
  escaped); ``backend_config`` only when non-empty (its own quote/backslash
  escaping); ``api_version`` ALWAYS emitted; ``has_side_effect = true`` only
  when true; the ``// evoxc:custom_call`` marker line.
* ``inline_stablehlo`` golden text: ``%in<i>`` substituted with the operand's
  real SSA name and ``%out<i>`` with a FRESH result name; a multi-line body
  keeps its inner indentation and blank lines; a nested ``stablehlo.while``
  body is preserved verbatim; the ``// evoxc:inline_stablehlo`` marker.
* the exporter-side token validation (``core.BackendError`` naming the op and
  the offending ``%in``/``%out`` token): out-of-range ``%in``, ``%out``
  beyond the result count, a result never assigned, and a result assigned
  TWICE (the writer rejects duplicates — the frontend does not).
* a callback-effect snippet is NEVER hoisted or reordered: a pure-looking
  ``stablehlo.tanh`` snippet emitted after another op keeps program order.
* ``stablehlo.ops.status``: both escape hatches are ``"v1"`` while
  ``external_call`` remains ``"deferred"``.

Zero-result ``custom_call`` cannot be built through the frontend (``result``
must be a non-empty spec), so that one golden is built with the raw
``ir.Builder`` exactly as the IR layer allows. Everything else goes through
the public ``evoxc`` ops. CPU only; tiny shapes.
"""

import numpy as np
import pytest

import evoxc
from evoxc import ir
from evoxc.backends import stablehlo
from evoxc.backends.stablehlo import ops as stablehlo_ops

_SPEC = evoxc.TensorSpec((2, 3), evoxc.float32)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _trace(fn):
    return evoxc.trace(fn, _SPEC)


def _export(fn):
    return stablehlo.export(_trace(fn))


def _raw_module(custom_attrs, *, operands=1):
    """Build a one-op module with a ``custom_call`` op via the raw IR Builder.

    Used only for result-arity cases the frontend rejects (a zero-result
    custom call). ``custom_attrs`` must carry ``result_specs`` matching the
    declared result types.
    """
    builder = ir.Builder()
    builder.build_module()
    input_types = tuple(
        ir.ValueType(np.dtype("float32"), (2, 3)) for _ in range(operands)
    )
    function = builder.build_function("main", input_types)
    args = function.entry_block.arguments
    op = builder.create(
        "custom_call", operands=tuple(args), attributes=custom_attrs
    )
    builder.set_terminator(
        function.entry_block, "return", tuple(v for v in op.results)
    )
    return builder.module


def _mutate_inline_body(graph, bad_body):
    """Replace the (single) ``inline_stablehlo`` op's ``body`` attribute.

    The frontend validates the ``%in``/``%out`` token contract at trace time,
    so the WRITER's own validation is exercised by mutating the traced op's
    attribute — the result types (and therefore ``ir.verify``) are unchanged.
    """
    op = graph.module.functions[0].entry_block.ops[0]
    assert op.name == "inline_stablehlo"
    op.attributes["body"] = bad_body
    return graph


# ---------------------------------------------------------------------------
# 1. custom_call golden text
# ---------------------------------------------------------------------------


def test_custom_call_single_result_golden():
    def fn(x):
        return evoxc.custom_call(
            "my_target", x, result=evoxc.TensorSpec((2, 3), evoxc.float32)
        )

    mlir = _export(fn)
    assert "// evoxc:custom_call" in mlir
    assert (
        "    %0 = stablehlo.custom_call @my_target(%arg0) "
        "{api_version = 1 : i32} : (tensor<2x3xf32>) -> tensor<2x3xf32>"
        in mlir
    )
    # api_version is always present; backend_config/has_side_effect are not.
    assert "backend_config" not in mlir
    assert "has_side_effect" not in mlir


def test_custom_call_multiple_results_golden():
    def fn(x):
        return evoxc.custom_call(
            "multi_target",
            x,
            x,
            result=(
                evoxc.TensorSpec((2, 3), evoxc.float32),
                evoxc.TensorSpec((2, 3), evoxc.float32),
            ),
        )

    mlir = _export(fn)
    assert (
        "    %0, %1 = stablehlo.custom_call @multi_target(%arg0, %arg0) "
        "{api_version = 1 : i32} : (tensor<2x3xf32>, tensor<2x3xf32>) -> "
        "(tensor<2x3xf32>, tensor<2x3xf32>)" in mlir
    )
    assert "func.return %0, %1 : tensor<2x3xf32>, tensor<2x3xf32>" in mlir


def test_custom_call_zero_results_golden():
    module = _raw_module(
        {
            "call_target_name": "no_result_target",
            "result_specs": (),
            "backend_config": "",
            "api_version": 1,
            "has_side_effect": False,
        }
    )
    mlir = stablehlo.export(module)
    assert (
        "    stablehlo.custom_call @no_result_target(%arg0) "
        "{api_version = 1 : i32} : (tensor<2x3xf32>) -> ()" in mlir
    )
    assert "func.return\n" in mlir
    # A result-less call binds no SSA name.
    assert "%0 =" not in mlir


def test_custom_call_target_quoting_and_escaping():
    # A target with a space is quoted; the quote/backslash inside are escaped.
    def spaced(x):
        return evoxc.custom_call(
            "my target", x, result=evoxc.TensorSpec((2, 3), evoxc.float32)
        )

    assert 'stablehlo.custom_call @"my target"(%arg0)' in _export(spaced)

    def odd(x):
        return evoxc.custom_call(
            'a"b\\c', x, result=evoxc.TensorSpec((2, 3), evoxc.float32)
        )

    mlir = _export(odd)
    assert 'stablehlo.custom_call @"a\\"b\\\\c"(%arg0)' in mlir
    # A conservative bare identifier stays unquoted.
    def bare(x):
        return evoxc.custom_call(
            "ns.v1$op_2", x, result=evoxc.TensorSpec((2, 3), evoxc.float32)
        )

    assert "stablehlo.custom_call @ns.v1$op_2(%arg0)" in _export(bare)


def test_custom_call_backend_config_only_when_non_empty_and_escaped():
    def with_config(x):
        return evoxc.custom_call(
            "t",
            x,
            result=evoxc.TensorSpec((2, 3), evoxc.float32),
            backend_config='{"a": "b\\c"}',
        )

    mlir = _export(with_config)
    assert 'backend_config = "{\\"a\\": \\"b\\\\c\\"}"' in mlir

    def empty_config(x):
        return evoxc.custom_call(
            "t", x, result=evoxc.TensorSpec((2, 3), evoxc.float32),
            backend_config="",
        )

    assert "backend_config" not in _export(empty_config)


def test_custom_call_has_side_effect_and_api_version():
    def fn(x):
        return evoxc.custom_call(
            "t",
            x,
            result=evoxc.TensorSpec((2, 3), evoxc.float32),
            api_version=3,
            has_side_effect=True,
        )

    mlir = _export(fn)
    assert "api_version = 3 : i32" in mlir
    assert "has_side_effect = true" in mlir
    # Attribute order: backend_config (absent), api_version, has_side_effect.
    assert "{api_version = 3 : i32, has_side_effect = true}" in mlir

    def no_side_effect(x):
        return evoxc.custom_call(
            "t", x, result=evoxc.TensorSpec((2, 3), evoxc.float32),
            has_side_effect=False,
        )

    assert "has_side_effect" not in _export(no_side_effect)


def test_custom_call_marker_line_is_indented_with_the_body():
    def fn(x):
        return evoxc.custom_call(
            "t", x, result=evoxc.TensorSpec((2, 3), evoxc.float32)
        )

    lines = _export(fn).splitlines()
    assert "    // evoxc:custom_call" in lines


# ---------------------------------------------------------------------------
# 2. inline_stablehlo golden text
# ---------------------------------------------------------------------------


def test_inline_stablehlo_substitutes_operand_and_result_names():
    def fn(x):
        return evoxc.inline_stablehlo(
            "%out0 = stablehlo.tanh %in0 : tensor<2x3xf32>",
            x,
            result=evoxc.TensorSpec((2, 3), evoxc.float32),
        )

    mlir = _export(fn)
    assert "    // evoxc:inline_stablehlo" in mlir
    # %in0 -> the function arg, %out0 -> a fresh counter name.
    assert "    %0 = stablehlo.tanh %arg0 : tensor<2x3xf32>" in mlir
    assert "%in0" not in mlir
    assert "%out0" not in mlir


def test_inline_stablehlo_multi_operand_order():
    def fn(a, b):
        return evoxc.inline_stablehlo(
            "%out0 = stablehlo.add %in0, %in1 : tensor<2x3xf32>",
            a,
            b,
            result=evoxc.TensorSpec((2, 3), evoxc.float32),
        )

    graph = evoxc.trace(fn, _SPEC, _SPEC)
    mlir = stablehlo.export(graph)
    assert "    %0 = stablehlo.add %arg0, %arg1 : tensor<2x3xf32>" in mlir


def test_inline_stablehlo_preserves_multiline_body_indentation_and_blank_lines():
    body = (
        "%r = stablehlo.tanh %in0 : tensor<2x3xf32>\n"
        "\n"
        "%out0 = stablehlo.add %r, %in0 : tensor<2x3xf32>"
    )

    def fn(x):
        return evoxc.inline_stablehlo(
            body, x, result=evoxc.TensorSpec((2, 3), evoxc.float32)
        )

    lines = _export(fn).splitlines()
    marker = lines.index("    // evoxc:inline_stablehlo")
    emitted_body = lines[marker + 1 : marker + 1 + len(body.splitlines())]
    # Every original line keeps its content and inner indentation, prefixed by
    # the function-body indent; the blank line survives as a blank (indented)
    # line. Only the reserved tokens are substituted (%in0 -> %arg0, %out0 ->
    # the fresh result name %0).
    expected = body.replace("%in0", "%arg0").replace("%out0", "%0")
    assert emitted_body == ["    " + line for line in expected.splitlines()]
    assert emitted_body[1].strip() == ""


def test_inline_stablehlo_preserves_nested_while_region():
    body = (
        '%res = "stablehlo.while"(%in0) ({\n'
        "^bb0(%a: tensor<2x3xf32>):\n"
        "  %cond = stablehlo.constant dense<true> : tensor<i1>\n"
        "  stablehlo.return %cond : tensor<i1>\n"
        "}, {\n"
        "^bb0(%a: tensor<2x3xf32>):\n"
        "  %tanh = stablehlo.tanh %a : tensor<2x3xf32>\n"
        "  stablehlo.return %tanh : tensor<2x3xf32>\n"
        "}) : (tensor<2x3xf32>) -> tensor<2x3xf32>\n"
        "%out0 = stablehlo.add %res, %in0 : tensor<2x3xf32>"
    )

    def fn(x):
        return evoxc.inline_stablehlo(
            body, x, result=evoxc.TensorSpec((2, 3), evoxc.float32)
        )

    mlir = _export(fn)
    lines = mlir.splitlines()
    marker = lines.index("    // evoxc:inline_stablehlo")
    emitted_body = lines[marker + 1 : marker + 1 + len(body.splitlines())]
    expected = body.replace("%in0", "%arg0").replace("%out0", "%0")
    assert emitted_body == ["    " + line for line in expected.splitlines()]
    # The nested region header / block args / terminator survive verbatim.
    assert any('"stablehlo.while"' in line for line in emitted_body)
    assert sum(line.strip().startswith("^bb0(") for line in emitted_body) == 2
    assert sum("stablehlo.return" in line for line in emitted_body) == 2


# ---------------------------------------------------------------------------
# 3. exporter-side token validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_body, expected",
    [
        (
            "%out0 = stablehlo.tanh %in5 : tensor<2x3xf32>",
            "%in5",
        ),
        (
            "%out3 = stablehlo.tanh %in0 : tensor<2x3xf32>",
            "%out3",
        ),
        (
            "%x = stablehlo.tanh %in0 : tensor<2x3xf32>",
            "%out0",
        ),
        (
            "%out0 = stablehlo.tanh %in0 : tensor<2x3xf32>\n"
            "%out0 = stablehlo.tanh %in0 : tensor<2x3xf32>",
            "%out0",
        ),
    ],
    ids=["in-out-of-range", "out-out-of-range", "never-assigned", "assigned-twice"],
)
def test_inline_stablehlo_writer_validation_errors(bad_body, expected):
    def fn(x):
        return evoxc.inline_stablehlo(
            "%out0 = stablehlo.tanh %in0 : tensor<2x3xf32>",
            x,
            result=evoxc.TensorSpec((2, 3), evoxc.float32),
        )

    graph = _mutate_inline_body(_trace(fn), bad_body)
    with pytest.raises(evoxc.BackendError) as excinfo:
        stablehlo.export(graph)
    message = str(excinfo.value)
    assert "inline_stablehlo" in message
    assert expected in message


def test_inline_stablehlo_duplicate_assignment_names_the_count():
    def fn(x):
        return evoxc.inline_stablehlo(
            "%out0 = stablehlo.tanh %in0 : tensor<2x3xf32>",
            x,
            result=evoxc.TensorSpec((2, 3), evoxc.float32),
        )

    graph = _mutate_inline_body(
        _trace(fn),
        "%out0 = stablehlo.tanh %in0 : tensor<2x3xf32>\n"
        "%out0 = stablehlo.tanh %in0 : tensor<2x3xf32>",
    )
    with pytest.raises(evoxc.BackendError, match=r"found 2"):
        stablehlo.export(graph)


def test_inline_stablehlo_empty_body_rejected_by_writer():
    # The frontend rejects an empty body too, but the WRITER re-validates (a
    # directly constructed op must fail with a naming BackendError, never
    # malformed MLIR).
    def fn(x):
        return evoxc.inline_stablehlo(
            "%out0 = stablehlo.tanh %in0 : tensor<2x3xf32>",
            x,
            result=evoxc.TensorSpec((2, 3), evoxc.float32),
        )

    graph = _mutate_inline_body(_trace(fn), "   ")
    with pytest.raises(evoxc.BackendError, match="inline_stablehlo"):
        stablehlo.export(graph)


# ---------------------------------------------------------------------------
# 4. ordering: a pure-looking snippet is never hoisted/reordered
# ---------------------------------------------------------------------------


def test_inline_snippet_keeps_program_order_after_another_op():
    def fn(x):
        y = evoxc.add(x, x)
        return evoxc.inline_stablehlo(
            "%out0 = stablehlo.tanh %in0 : tensor<2x3xf32>",
            y,
            result=evoxc.TensorSpec((2, 3), evoxc.float32),
        )

    mlir = _export(fn)
    add_at = mlir.index("stablehlo.add %arg0, %arg0")
    marker_at = mlir.index("// evoxc:inline_stablehlo")
    tanh_at = mlir.index("stablehlo.tanh")
    assert add_at < marker_at < tanh_at
    # The snippet consumes the add's result, so the add is not moved after it.
    assert "stablehlo.tanh %0" in mlir


# ---------------------------------------------------------------------------
# 5. status(): v1 vs deferred
# ---------------------------------------------------------------------------


def test_custom_ops_are_v1_and_external_call_stays_deferred():
    assert stablehlo_ops.status("custom_call") == "v1"
    assert stablehlo_ops.status("inline_stablehlo") == "v1"
    assert stablehlo_ops.is_supported("custom_call")
    assert stablehlo_ops.is_supported("inline_stablehlo")
    assert stablehlo_ops.status("external_call") == "deferred"
    assert not stablehlo_ops.is_supported("external_call")
