"""Contract tests for the two opaque custom IR ops.

``custom_call`` (a named StableHLO custom-call target) and ``inline_stablehlo``
(a raw StableHLO/MLIR snippet emitted verbatim) are declared in
``evoxc/ir/op_defs/custom.py``. Both mirror ``external_call``: ``category
"custom"``, ``effect "callback"``, variadic operands (``arity (0, None)``),
``result_count None`` and ``shape_fn None`` — their result types resolve from
the declared ``result_specs`` attribute (or explicit ``result_types``), and
``verify`` checks the two agree.

The package is READ-ONLY from here: a failing assertion means evoxc violates
its documented contract, never that the test should be weakened.
CPU only, no GPU/network; every module is tiny.
"""

import numpy as np
import pytest

from evoxc import ir
from evoxc.core import Dim, VerificationError

F32 = np.dtype("float32")
I32 = np.dtype("int32")

_CUSTOM_OPS = ("custom_call", "inline_stablehlo")


def _vt(shape, dtype=F32):
    return ir.ValueType(dtype, shape)


# Distinctive, non-default attribute payloads used across the file.
TARGET = "evoxc.test.target"
BACKEND_CONFIG = "cfg:key=value"
TRICKY_BODY = (
    'stablehlo.add %a, %b : tensor<2x3xf32>  // "double" and \\backslash\n'
    "second line"
)


def _custom_attrs(**overrides):
    """Valid ``custom_call`` attributes (overridable per test)."""
    attrs = {
        "call_target_name": TARGET,
        "result_specs": (_vt((2, 3)),),
    }
    attrs.update(overrides)
    return attrs


def _inline_attrs(**overrides):
    """Valid ``inline_stablehlo`` attributes (overridable per test)."""
    attrs = {
        "body": TRICKY_BODY,
        "result_specs": (_vt((Dim("B"), None)), _vt((), I32)),
    }
    attrs.update(overrides)
    return attrs


def _module_with(op_name, attributes):
    """A one-op module: ``main() { %0 = op(...); return %0 }``."""
    module = ir.Module(name="custom")
    b = ir.Builder(module)
    f = b.build_function("main", ())
    op = b.create(op_name, (), attributes=attributes)
    b.set_terminator(f.entry_block, "return", tuple(op.results))
    return module, op


def _both_ops_module():
    """``main() { custom_call; inline_stablehlo; return ... }``."""
    module = ir.Module(name="custom")
    b = ir.Builder(module)
    f = b.build_function("main", ())
    call = b.create(
        "custom_call",
        (),
        attributes=_custom_attrs(
            backend_config=BACKEND_CONFIG,
            api_version=2,
            has_side_effect=True,
        ),
    )
    inline = b.create("inline_stablehlo", (), attributes=_inline_attrs())
    b.set_terminator(
        f.entry_block, "return", (call.result, *inline.results)
    )
    return module, call, inline


def _ops_named(module, name):
    return [
        op
        for op in module.functions[0].entry_block.ops
        if op.name == name
    ]


# ---------------------------------------------------------------------------
# 1. OpDef contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _CUSTOM_OPS)
def test_opdef_contract(name):
    opdef = ir.opdef(name)
    assert opdef.category == "custom"
    assert opdef.effect == "callback"
    assert opdef.effect == ir.EFFECT_CALLBACK
    assert opdef.arity == (0, None)
    assert opdef.result_count is None
    assert opdef.shape_fn is None
    assert opdef.regions == 0
    assert opdef.is_terminator is False


def test_custom_call_attribute_schema():
    attrs = ir.opdef("custom_call").attributes
    assert tuple((a.name, a.type, a.required) for a in attrs) == (
        ("call_target_name", ir.ATTR_STR, True),
        ("result_specs", ir.ATTR_ANY, True),
        ("backend_config", ir.ATTR_STR, False),
        ("api_version", ir.ATTR_INT, False),
        ("has_side_effect", ir.ATTR_BOOL, False),
    )
    by_name = {a.name: a for a in attrs}
    assert by_name["backend_config"].default == ""
    assert by_name["api_version"].default == 1
    assert by_name["has_side_effect"].default is False


def test_inline_stablehlo_attribute_schema():
    attrs = ir.opdef("inline_stablehlo").attributes
    assert tuple((a.name, a.type, a.required) for a in attrs) == (
        ("body", ir.ATTR_STR, True),
        ("result_specs", ir.ATTR_ANY, True),
        ("has_side_effect", ir.ATTR_BOOL, False),
    )
    by_name = {a.name: a for a in attrs}
    assert by_name["has_side_effect"].default is False


def test_optional_attributes_materialize_their_defaults():
    _, call = _module_with("custom_call", _custom_attrs())
    assert call.attributes["backend_config"] == ""
    assert call.attributes["api_version"] == 1
    assert call.attributes["has_side_effect"] is False

    _, inline = _module_with("inline_stablehlo", _inline_attrs())
    assert inline.attributes["has_side_effect"] is False


@pytest.mark.parametrize("name", _CUSTOM_OPS)
def test_required_attributes_are_required(name):
    required = "call_target_name" if name == "custom_call" else "body"
    attrs = _custom_attrs() if name == "custom_call" else _inline_attrs()
    del attrs[required]
    with pytest.raises(VerificationError, match=required):
        _module_with(name, attrs)


# ---------------------------------------------------------------------------
# 2. Builder.create resolves result types from result_specs
# ---------------------------------------------------------------------------


def test_create_resolves_single_result_from_result_specs():
    # No explicit result_types — the declared result_specs is the only source.
    _, op = _module_with("custom_call", _custom_attrs())
    assert len(op.results) == 1
    assert op.result.type == _vt((2, 3))


def test_create_resolves_multiple_results_including_symbolic_and_dynamic_dims():
    _, op = _module_with("inline_stablehlo", _inline_attrs())
    assert tuple(result.type for result in op.results) == _inline_attrs()[
        "result_specs"
    ]
    symbolic_shape = op.results[0].type.shape
    assert symbolic_shape[0] == Dim("B")
    assert isinstance(symbolic_shape[0], Dim)
    assert symbolic_shape[1] is None
    assert op.results[1].type == _vt((), I32)


def test_create_accepts_an_empty_result_specs_sequence():
    shape_module, op = _module_with(
        "custom_call", _custom_attrs(result_specs=())
    )
    assert op.results == ()
    assert ir.verify(shape_module) is None


# ---------------------------------------------------------------------------
# 3. ir.verify success and failures
# ---------------------------------------------------------------------------


def test_verify_accepts_both_ops():
    module, call, inline = _both_ops_module()
    assert ir.verify(module) is None
    assert call.result is not None
    assert len(inline.results) == 2


def test_verify_rejects_empty_call_target_name():
    module, _ = _module_with("custom_call", _custom_attrs(call_target_name=""))
    with pytest.raises(VerificationError, match="call_target_name"):
        ir.verify(module)


@pytest.mark.parametrize("api_version", [0, -1, True, "1"])
def test_verify_rejects_bad_api_version(api_version):
    # Api version 0/-1 pass the ATTR_INT tag but fail the value check; True and
    # non-ints are caught by the tag check (bools are never ints here).
    module, op = _module_with("custom_call", _custom_attrs())
    op.attributes["api_version"] = api_version
    with pytest.raises(VerificationError, match="api_version"):
        ir.verify(module)


def test_verify_rejects_empty_body():
    module, _ = _module_with("inline_stablehlo", _inline_attrs(body=""))
    with pytest.raises(VerificationError, match="body"):
        ir.verify(module)


def test_verify_rejects_non_bool_has_side_effect():
    module, op = _module_with("custom_call", _custom_attrs())
    op.attributes["has_side_effect"] = 1
    with pytest.raises(VerificationError, match="has_side_effect"):
        ir.verify(module)


@pytest.mark.parametrize(
    "bad_result_specs",
    [
        (_vt((2, 3)), _vt((2, 3))),  # a different result COUNT
        (_vt((2, 3), I32),),  # same count, a different dtype
        (_vt((3, 2)),),  # same count, a different static shape
    ],
    ids=["count", "dtype", "shape"],
)
def test_verify_rejects_result_specs_that_disagree_with_results(bad_result_specs):
    module, op = _module_with("custom_call", _custom_attrs())
    op.attributes["result_specs"] = bad_result_specs
    with pytest.raises(VerificationError, match="do not match"):
        ir.verify(module)


@pytest.mark.parametrize("bad_specs", [42, "not-specs", (np.dtype("float32"),)])
def test_verify_rejects_result_specs_not_a_valuetype_sequence(bad_specs):
    module, op = _module_with("custom_call", _custom_attrs())
    op.attributes["result_specs"] = bad_specs
    with pytest.raises(VerificationError, match="result_specs"):
        ir.verify(module)


# ---------------------------------------------------------------------------
# 4. Serialization round-trip (byte-identical)
# ---------------------------------------------------------------------------


def test_serialize_round_trip_is_byte_identical_and_preserves_attributes():
    module, _, _ = _both_ops_module()
    first = ir.serialize_module(module)
    restored = ir.deserialize_module(first)
    assert ir.verify(restored) is None
    assert ir.serialize_module(restored) == first

    call = _ops_named(restored, "custom_call")[0]
    assert call.attributes["call_target_name"] == TARGET
    assert call.attributes["backend_config"] == BACKEND_CONFIG
    assert call.attributes["api_version"] == 2
    assert call.attributes["has_side_effect"] is True

    inline = _ops_named(restored, "inline_stablehlo")[0]
    # The tricky body (quotes, backslash, newline) survives verbatim.
    assert inline.attributes["body"] == TRICKY_BODY
    specs = inline.attributes["result_specs"]
    assert all(isinstance(spec, ir.ValueType) for spec in specs)
    assert specs[0].shape[0] == Dim("B")
    assert specs[0].shape[1] is None
    assert specs[1] == _vt((), I32)
    # Normalized result specs still agree with the results after the trip.
    assert tuple(specs) == tuple(result.type for result in inline.results)


# ---------------------------------------------------------------------------
# 5. pretty_print renders both ops with their attributes
# ---------------------------------------------------------------------------


def test_pretty_print_renders_custom_ops():
    module, _, _ = _both_ops_module()
    text = ir.pretty_print(module)

    assert "evoxc.custom_call()" in text
    assert f'call_target_name = "{TARGET}"' in text
    assert f'backend_config = "{BACKEND_CONFIG}"' in text
    assert "api_version = 2" in text
    assert "has_side_effect = True" in text

    assert "evoxc.inline_stablehlo()" in text
    assert "body = " in text
    # The body is JSON-quoted on one line, so its embedded newline is escaped.
    assert "\\n" in text
    assert "result_specs = " in text
    assert "tensor<" not in text.split("result_specs = ", 1)[1].split(
        "}", 1
    )[0]  # ValueType repr, not tensor<> spelling
    assert "ValueType(" in text
