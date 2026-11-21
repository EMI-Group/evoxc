"""Round-trip pins for the private native envelope codec.

The codec is the in-process bridge to the optional Rust/PyO3 core; these tests
pin the ``evoxc-native-ir`` v1 envelope without requiring the extension to
exist.  The envelope is flat at the top level — ``format``, ``version``,
``module``, ``functions`` and ``ops`` — with every ndarray payload INLINE
(``{"__evoxc_ndarray__": {"dtype", "shape", "data_b64"}}``), so there is no
``constants`` table and no ``sha256``.  ``"module"`` here is the envelope body's
module-info object; the façade's ``{"module": ir.Module, ...}`` result key is a
different thing.
"""

from __future__ import annotations

import base64
import copy
import importlib

import numpy as np
import pytest

import evoxc
from evoxc import core, ir
from evoxc._native_codec import (
    ENVELOPE_FORMAT,
    decode_module,
    decode_native_module,
    decode_type,
    encode_module,
    encode_type,
)
from evoxc.core import Dim, DimExpr


def _simple_module() -> ir.Module:
    """One function: ``main(x) = x + x``."""
    m = ir.Module(name="simple")
    b = ir.Builder(m)
    f = b.build_function("main", (ir.ValueType(np.float32, (4,)),))
    x = f.entry_block.arguments[0]
    result = b.emit("add", (x, x))
    b.set_terminator(f.entry_block, "return", (result,))
    return m


def _empty_module() -> ir.Module:
    """A zero-input function that returns nothing."""
    m = ir.Module(name="empty")
    b = ir.Builder(m)
    f = b.build_function("main", ())
    b.set_terminator(f.entry_block, "return", ())
    return m


def _multi_feature_module() -> ir.Module:
    """Two functions exercising dims, regions, effects, constants and attrs."""
    B = Dim("B", 8)
    N = Dim("N")
    m = ir.Module(name="m")
    b = ir.Builder(m)
    f = b.build_function(
        "main",
        (
            ir.ValueType(np.float32, (B, N)),
            ir.ValueType(np.float32, (B, N)),
            ir.ValueType(np.int32, (DimExpr("mul", B, 2), None)),
        ),
    )
    x, y, _z = f.entry_block.arguments
    loc = ir.Location("model.py", 12, 8, code_snippet="x + y * 2")
    two = b.emit(
        "constant", (), attributes={"value": np.array(2.0, dtype=np.float32)}
    )
    y2 = b.emit("multiply", (y, two))
    s = b.emit("add", (x, y2), location=loc)
    b.emit("rank", (), attributes={"group": "world"})
    b.emit("all_reduce", (s,), attributes={"reduce_op": "sum", "group": "g1"})
    b.create(
        "runtime_call",
        (),
        attributes={
            "callback": "cb_scale",
            "result_specs": [ir.ValueType(np.float32, (B, N))],
        },
    )
    b.create(
        "block_call",
        (s,),
        attributes={
            "block_name": "my_block",
            "static_args": ((1, 2), "mode"),
            "result_specs": [ir.ValueType(np.float64, (N, 4))],
        },
    )
    pred = b.emit("equal", (x, y))
    b.emit("broadcast", (x,), attributes={"shape": (DimExpr("mul", B, 2), 8, N)})

    r_true = b.build_region((pred.type, x.type))
    b.push_region(r_true)
    xb = r_true.entry.arguments[1]
    taken = b.emit("add", (xb, xb))
    b.set_terminator(r_true.entry, "return", (taken,))
    b.pop_region()
    r_false = b.build_region((pred.type, x.type))
    b.push_region(r_false)
    fb = r_false.entry.arguments[1]
    b.set_terminator(r_false.entry, "return", (fb,))
    b.pop_region()
    iff = b.create("if", (pred, x), regions=(r_true, r_false), location=loc)
    b.set_terminator(f.entry_block, "return", (s, iff.result))

    hf = b.build_function("helper", (ir.ValueType(np.float64, (2, 3)),))
    h = hf.entry_block.arguments[0]
    const_int = b.emit(
        "constant",
        (),
        attributes={"value": np.arange(6, dtype=np.int32).reshape(2, 3)},
    )
    const_f64 = b.emit(
        "constant",
        (),
        attributes={"value": np.linspace(1.0, 4.0, 4, dtype=np.float64).reshape(2, 2)},
    )
    h2 = b.emit("add", (h, h))
    hr = b.emit("reshape", (h2,), attributes={"shape": (3, 2)})
    b.set_terminator(hf.entry_block, "return", (hr, const_int, const_f64))
    return m


def _control_flow_module() -> ir.Module:
    """A traced graph with both a ``cond`` and a ``while_loop`` region."""

    def f(x):
        def true_fn(y):
            return evoxc.add(y, 1.0)

        def false_fn(y):
            return evoxc.subtract(y, 1.0)

        def cond_fn(state):
            return evoxc.less(evoxc.sum(state), 10.0)

        def body_fn(state):
            return evoxc.add(state, 1.0)

        branch = evoxc.cond(evoxc.less(evoxc.sum(x), 0.5), true_fn, false_fn, x)
        looped = evoxc.while_loop(cond_fn, body_fn, branch)
        return evoxc.add(looped, x)

    return evoxc.trace(f, evoxc.TensorSpec((2, 3), evoxc.float32)).module


def _symbolic_module() -> ir.Module:
    B = evoxc.dim("B")

    def f(x, y):
        return evoxc.add(evoxc.reshape(x, (3, B)), y)

    return evoxc.trace(
        f,
        evoxc.TensorSpec((B, 3), evoxc.float32),
        evoxc.TensorSpec((3, B * 2), evoxc.float32),
    ).module


def _assert_dim_equal(a, b) -> None:
    assert (a is None) == (b is None)
    if a is None:
        return
    if isinstance(a, int) or isinstance(b, int):
        assert a == b
        return
    assert type(a) is type(b), (a, b)
    if isinstance(a, Dim):
        assert (a.name, a.size) == (b.name, b.size)
    else:
        assert a.op == b.op
        _assert_dim_equal(a.left, b.left)
        _assert_dim_equal(a.right, b.right)


def _assert_type_equal(a: ir.ValueType, b: ir.ValueType) -> None:
    assert a.dtype.name == b.dtype.name, (a.dtype, b.dtype)
    assert len(a.shape) == len(b.shape)
    for da, db in zip(a.shape, b.shape):
        _assert_dim_equal(da, db)


def _assert_attr_equal(actual, expected) -> None:
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.dtype.name == expected.dtype.name
        assert actual.shape == expected.shape
        assert np.array_equal(actual, expected)
        return
    if isinstance(expected, tuple) and isinstance(actual, tuple):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _assert_attr_equal(a, e)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _assert_attr_equal(a, e)
        return
    # `result_specs` is schema-decoded to a TUPLE (a builder-normalized
    # attribute), so a list expectation compares element-wise against it.
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _assert_attr_equal(a, e)
        return
    if isinstance(expected, dict) and isinstance(actual, dict):
        assert list(actual.keys()) == list(expected.keys())
        for key in expected:
            _assert_attr_equal(actual[key], expected[key])
        return
    if isinstance(expected, ir.ValueType) and isinstance(actual, ir.ValueType):
        _assert_type_equal(actual, expected)
        return
    if isinstance(expected, np.dtype) and isinstance(actual, np.dtype):
        assert actual.name == expected.name
        return
    if isinstance(expected, (Dim, DimExpr)) and isinstance(actual, (Dim, DimExpr)):
        _assert_dim_equal(actual, expected)
        return
    assert type(actual) is type(expected), (actual, expected)
    assert actual == expected


def _walk_blocks(module: ir.Module):
    def walk_block(block):
        yield block
        for op in block.ops:
            for region in op.regions:
                for nested in region.blocks:
                    yield from walk_block(nested)

    for function in module.functions:
        for block in function.region.blocks:
            yield from walk_block(block)


def _assert_block_equal(actual: ir.Block, expected: ir.Block) -> None:
    assert [arg.id for arg in actual.arguments] == [arg.id for arg in expected.arguments]
    for a_arg, e_arg in zip(actual.arguments, expected.arguments):
        _assert_type_equal(a_arg.type, e_arg.type)
        assert a_arg.owner is actual and e_arg.owner is expected
        assert a_arg.index == e_arg.index
    assert len(actual.ops) == len(expected.ops)
    for a_op, e_op in zip(actual.ops, expected.ops):
        assert (a_op.id, a_op.name) == (e_op.id, e_op.name)
        assert [v.id for v in a_op.operands] == [v.id for v in e_op.operands]
        assert [v.id for v in a_op.results] == [v.id for v in e_op.results]
        assert a_op.parent is actual and e_op.parent is expected
        assert list(a_op.attributes) == list(e_op.attributes)
        for name, expected_attr in e_op.attributes.items():
            _assert_attr_equal(a_op.attributes[name], expected_attr)
        assert len(a_op.regions) == len(e_op.regions)
        for a_region, e_region in zip(a_op.regions, e_op.regions):
            assert a_region.parent is a_op
            assert len(a_region.blocks) == len(e_region.blocks)
            for a_block, e_block in zip(a_region.blocks, e_region.blocks):
                assert a_block.parent is a_region
                _assert_block_equal(a_block, e_block)
        if a_op.location is None:
            assert e_op.location is None
        else:
            assert a_op.location == e_op.location
        for value in a_op.results:
            assert value.owner is a_op


def _assert_module_equal(actual: ir.Module, expected: ir.Module) -> None:
    assert actual.name == expected.name
    assert len(actual.functions) == len(expected.functions)
    for a_fn, e_fn in zip(actual.functions, expected.functions):
        assert a_fn.name == e_fn.name
        assert a_fn.parent is actual
        assert len(a_fn.input_types) == len(e_fn.input_types)
        for a_t, e_t in zip(a_fn.input_types, e_fn.input_types):
            _assert_type_equal(a_t, e_t)
        assert a_fn.region.parent is a_fn
        assert len(a_fn.region.blocks) == len(e_fn.region.blocks)
        for a_block, e_block in zip(a_fn.region.blocks, e_fn.region.blocks):
            _assert_block_equal(a_block, e_block)


def _all_values(module: ir.Module):
    for block in _walk_blocks(module):
        yield from block.arguments
        for op in block.ops:
            yield from op.results


def _all_ops(module: ir.Module):
    for block in _walk_blocks(module):
        yield from block.ops


def _envelope_value_types(envelope: dict) -> dict:
    """The envelope's INLINE value types, keyed by decimal SSA id string.

    The envelope has no ``value_types`` map: every value's type travels on its
    block-argument entry or on its op's ``results`` entry.  Nested op regions
    carry their own block arguments, so the walk is recursive.
    """
    types = {}

    def walk_block(block: dict) -> None:
        for argument in block["arguments"]:
            types[str(argument["id"])] = argument["type"]
        for entry in block["ops"]:
            # Block op entries are refs into the flat module ops table.
            op_id = entry.get("id", entry.get("ref"))
            results = ops_by_id.get(str(op_id), {}).get("results", ())
            for result in results:
                types[str(result["id"])] = result["type"]

    def walk_region(region: dict) -> None:
        if "blocks" in region:
            for block in region["blocks"]:
                walk_block(block)
        else:
            walk_block(region)

    ops_by_id = {str(op["id"]): op for op in envelope["ops"]}
    for function in envelope["functions"]:
        walk_block(function["block"])
    for op in envelope["ops"]:
        for region in op["regions"]:
            walk_region(region)
    return types


def _assert_uses_consistent(module: ir.Module) -> None:
    for value in _all_values(module):
        expected = [
            (op.id, idx)
            for block in _walk_blocks(module)
            for op in block.ops
            for idx, operand in enumerate(op.operands)
            if operand.id == value.id
        ]
        assert [(use.owner.id, use.operand_index) for use in value.uses] == expected


def test_envelope_format_version():
    assert ENVELOPE_FORMAT == 1


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3),
        (None, 4),
        (Dim("B"), Dim("B", 8)),
        (DimExpr("mul", Dim("B"), 2) + 1,),
        (DimExpr("min", DimExpr("floordiv", Dim("N"), 2), 7), None),
    ],
)
def test_type_round_trip(shape):
    value_type = ir.ValueType(np.float32, shape)
    decoded = decode_type(encode_type(value_type))
    assert isinstance(decoded, ir.ValueType)
    _assert_type_equal(decoded, value_type)
    assert decoded == value_type


def test_type_envelope_shape_uses_object_dim_forms():
    payload = encode_type(ir.ValueType(np.float32, (Dim("B"), None, 3)))
    assert payload == {
        "dtype": "float32",
        "shape": [{"dim": "B"}, None, {"int": 3}],
    }
    expr = encode_type(ir.ValueType(np.int64, (DimExpr("mul", Dim("B", 4), 2),)))
    assert expr["shape"] == [
        {
            "expr": {
                "op": "mul",
                "args": [{"dim": "B", "size": 4}, {"int": 2}],
            }
        }
    ]


def test_envelope_is_flat_with_inline_arrays():
    """The wire envelope is flat: module-info + inline arrays, no side tables.

    ``"module"`` here is the module-info object (``{"name", "metadata"}``); the
    façade's ``{"module": ir.Module, ...}`` result key is a different thing.
    Array attributes are inline rather than hoisted into a ``constants`` table,
    and an in-process envelope carries no ``sha256``.
    """
    envelope = encode_module(_simple_module())
    assert set(envelope) == {
        "format",
        "version",
        "module",
        "functions",
        "ops",
    }
    assert envelope["format"] == "evoxc-native-ir"
    assert envelope["version"] == ENVELOPE_FORMAT
    assert isinstance(envelope["module"], dict)
    assert set(envelope["module"]) == {"name", "metadata"}
    assert "value_types" not in envelope
    assert "constants" not in envelope
    assert "sha256" not in envelope



def test_empty_function_round_trip():
    module = _empty_module()
    ir.verify(module)
    decoded = decode_module(encode_module(module))
    ir.verify(decoded)
    _assert_module_equal(decoded, module)
    assert decoded.main.input_types == ()


def test_simple_function_round_trip_preserves_ids():
    module = _simple_module()
    ir.verify(module)
    envelope = encode_module(module)
    decoded = decode_module(envelope)
    ir.verify(decoded)
    _assert_module_equal(decoded, module)
    value_ids = {value.id for value in _all_values(module)}
    assert set(_envelope_value_types(envelope)) == {str(i) for i in value_ids}
    assert decoded.new_op_id() > max(op.id for op in decoded.main.entry_block.ops)
    assert decoded.new_value_id() > max(value_ids)


def test_value_types_cover_every_ssa_value():
    module = _multi_feature_module()
    envelope = encode_module(module)
    ids = {str(value.id) for value in _all_values(module)}
    assert set(_envelope_value_types(envelope)) == ids
    assert len(ids) == len({value.id for value in _all_values(module)})


def test_multi_feature_module_round_trip():
    module = _multi_feature_module()
    ir.verify(module)
    decoded = decode_module(encode_module(module))
    ir.verify(decoded)
    _assert_module_equal(decoded, module)
    _assert_uses_consistent(decoded)
    assert len(decoded.functions) == 2
    assert decoded.functions[1].name == "helper"


def test_control_flow_regions_round_trip():
    module = _control_flow_module()
    ir.verify(module)
    decoded = decode_module(encode_module(module))
    ir.verify(decoded)
    _assert_module_equal(decoded, module)
    _assert_uses_consistent(decoded)
    names = {op.name for block in _walk_blocks(decoded) for op in block.ops}
    assert {"if", "while"} <= names


def test_symbolic_dims_and_dimexpr_round_trip():
    module = _symbolic_module()
    ir.verify(module)
    envelope = encode_module(module)
    decoded = decode_module(envelope)
    ir.verify(decoded)
    _assert_module_equal(decoded, module)
    shapes = [
        dim
        for function in envelope["functions"]
        for value_type in function["input_types"]
        for dim in value_type["shape"]
    ]
    assert {"dim": "B"} in shapes
    assert {"int": 3} in shapes
    assert {
        "expr": {"op": "mul", "args": [{"dim": "B"}, {"int": 2}]}
    } in shapes


def test_attribute_variety_round_trip():
    m = ir.Module(name="attrs")
    b = ir.Builder(m)
    f = b.build_function("main", (ir.ValueType(np.float32, (2,)),))
    x = f.entry_block.arguments[0]
    # `block_call.static_args` is the generic (ATTR_ANY) attribute slot, so it
    # is the contract-bearing place for the JSON-able value variety.  (A
    # schema-typed op such as `add` rejects unknown attributes at verify time.)
    payload = {
        "bool_true": True,
        "int": 7,
        "float": 1.5,
        "str": "mode",
        "none": None,
        "tuple": (1, "a", None),
        "list": [1, 2, 3],
        "dim": Dim("B", 8),
        "expr": DimExpr("add", Dim("B"), 2),
        "nested": {"inner": (True, 2.5), "three": "three"},
        "nested_ints": ((1, 2), (3, 4, 5)),
        "shape": (Dim("B"), None, 4),
        "value_type": ir.ValueType(np.float64, (Dim("B"), 3)),
        "array": np.arange(6, dtype=np.int16).reshape(2, 3),
    }
    call = b.create(
        "block_call",
        (x,),
        attributes={
            "block_name": "my_block",
            "static_args": payload,
            "result_specs": [ir.ValueType(np.float32, (2,))],
        },
    )
    b.set_terminator(f.entry_block, "return", (call.result,))
    decoded = decode_module(encode_module(m))
    attrs = decoded.main.entry_block.ops[0].attributes["static_args"]
    assert list(attrs) == list(payload)
    for key, expected in payload.items():
        if key == "value_type":
            continue  # asserted structurally below (generic ValueTypes are dicts)
        _assert_attr_equal(attrs[key], expected)
    assert isinstance(attrs["tuple"], tuple)
    assert isinstance(attrs["list"], list)
    assert isinstance(attrs["dim"], Dim)
    assert isinstance(attrs["expr"], DimExpr)
    assert isinstance(attrs["nested"]["inner"], tuple)
    assert attrs["nested"]["three"] == "three"
    assert attrs["nested_ints"] == ((1, 2), (3, 4, 5))
    # A generic `ValueType` round-trips structurally as a plain type dict (the
    # `result_specs` slot is the schema-driven one that restores `ValueType`s).
    # Symbolic dims restore to `Dim`; an `{"int": n}` entry stays a plain dict
    # because int dims only exist inside schema-decoded shape lists.
    assert attrs["value_type"] == {
        "dtype": "float64",
        "shape": [Dim("B"), {"int": 3}],
    }
    assert attrs["array"].dtype == np.dtype("int16")
    assert np.array_equal(attrs["array"], payload["array"])


def test_builder_normalized_attributes_round_trip():
    module = _multi_feature_module()
    decoded = decode_module(encode_module(module))
    by_name = {
        op.name: op for block in _walk_blocks(decoded) for op in block.ops
    }
    assert by_name["reshape"].attributes["shape"] == (3, 2)
    assert by_name["block_call"].attributes["static_args"] == ((1, 2), "mode")
    assert by_name["runtime_call"].attributes["callback"] == "cb_scale"
    spec = by_name["runtime_call"].attributes["result_specs"][0]
    assert isinstance(spec, ir.ValueType)


def test_constants_round_trip_including_byte_order_and_zero_dim():
    m = ir.Module(name="consts")
    b = ir.Builder(m)
    f = b.build_function("main", ())
    big_endian = np.array([1.5, -2.0, 3.25], dtype=">f4")
    zero_dim = np.array(42.0, dtype=np.float64)
    scalars = np.arange(6, dtype=np.int16).reshape(2, 3)
    c1 = b.emit("constant", (), attributes={"value": big_endian})
    c2 = b.emit("constant", (), attributes={"value": zero_dim})
    c3 = b.emit("constant", (), attributes={"value": scalars})
    dup = b.emit("constant", (), attributes={"value": scalars.copy()})
    b.set_terminator(f.entry_block, "return", (c1, c2, c3, dup))
    envelope = encode_module(m)
    # Every array is INLINE (no constants table, no side reference).
    assert "constants" not in envelope
    markers = [op["attributes"]["value"] for op in envelope["ops"][:4]]
    assert all(set(marker) == {"__evoxc_ndarray__"} for marker in markers)
    inline = [marker["__evoxc_ndarray__"] for marker in markers]
    assert [(entry["dtype"], entry["shape"]) for entry in inline] == [
        ("float32", [3]),
        ("float64", []),
        ("int16", [2, 3]),
        ("int16", [2, 3]),
    ]
    assert all(isinstance(entry["data_b64"], str) for entry in inline)
    # Raw little-endian C-order payload: 3 float32s, 1 float64, 6 int16s.
    assert [len(base64.b64decode(entry["data_b64"])) for entry in inline] == [
        12,
        8,
        12,
        12,
    ]
    # Duplicate payloads encode identically (there is deliberately no dedup).
    assert inline[2] == inline[3]
    decoded = decode_module(envelope)
    ops = decoded.main.entry_block.ops
    be = ops[0].attributes["value"]
    assert be.dtype.name == "float32" and be.shape == (3,)
    assert be.dtype.isnative and np.array_equal(be, big_endian)
    zd = ops[1].attributes["value"]
    assert zd.shape == () and zd.dtype.name == "float64" and float(zd) == 42.0
    arr = ops[2].attributes["value"]
    assert arr.shape == (2, 3) and arr.dtype.name == "int16"
    assert np.array_equal(arr, scalars)
    assert np.array_equal(ops[3].attributes["value"], scalars)


def test_multi_block_region_round_trip():
    B = Dim("B")
    m = ir.Module(name="multiblock")
    b = ir.Builder(m)
    f = b.build_function("main", (ir.ValueType(np.float32, (B,)),))
    x = f.entry_block.arguments[0]
    pred = b.emit("less", (x, x))
    true_region = b.build_region((pred.type, x.type))
    b.push_region(true_region)
    b.set_terminator(true_region.entry, "return", (true_region.entry.arguments[1],))
    b.pop_region()
    true_region.append_block(ir.Block())
    second = true_region.blocks[-1]
    ret = ir.Op(name="return", id=m.new_op_id())
    second.append(ret)
    false_region = b.build_region((pred.type, x.type))
    b.push_region(false_region)
    b.set_terminator(false_region.entry, "return", (false_region.entry.arguments[1],))
    b.pop_region()
    iff = b.create("if", (pred, x), regions=(true_region, false_region))
    b.set_terminator(f.entry_block, "return", (iff.result,))
    decoded = decode_module(encode_module(m))
    _assert_module_equal(decoded, m)
    decoded_if = decoded.main.entry_block.ops[-2]
    assert [len(region.blocks) for region in decoded_if.regions] == [2, 1]
    assert decoded_if.regions[0].blocks[1].ops[0].name == "return"


def test_locations_round_trip_and_none_stays_none():
    m = ir.Module(name="locs")
    b = ir.Builder(m)
    f = b.build_function("main", (ir.ValueType(np.float32, (2,)),))
    x = f.entry_block.arguments[0]
    located = b.emit("add", (x, x), location=ir.Location("model.py", 83, 5, "x = y"))
    plain = b.emit("multiply", (located, x))
    b.set_terminator(f.entry_block, "return", (plain,))
    envelope = encode_module(m)
    assert envelope["ops"][0]["location"] == {
        "file": "model.py",
        "line": 83,
        "col": 5,
        "code_snippet": "x = y",
    }
    assert envelope["ops"][1]["location"] is None
    decoded = decode_module(envelope)
    add_op, mul_op = decoded.main.entry_block.ops[:2]
    assert add_op.location.file == "model.py"
    assert (add_op.location.line, add_op.location.col) == (83, 5)
    # The full location, including the captured source snippet, round-trips.
    assert add_op.location.code_snippet == "x = y"
    # An absent location stays absent (never invented).
    assert mul_op.location is None


def test_encode_module_does_not_mutate_input():
    module = _multi_feature_module()
    before_ops = repr(module._op_ids)
    before_values = repr(module._value_ids)

    def snapshot():
        return [
            (
                op.id,
                op.name,
                [v.id for v in op.operands],
                [v.id for v in op.results],
                repr(op.attributes),
                [len(v.uses) for v in op.results],
                None if op.location is None else (op.location.file, op.location.line),
            )
            for block in _walk_blocks(module)
            for op in block.ops
        ]

    before = snapshot()
    encode_module(module)
    encode_module(module)
    assert snapshot() == before
    assert repr(module._op_ids) == before_ops
    assert repr(module._value_ids) == before_values
    _assert_uses_consistent(module)


@pytest.mark.parametrize(
    "mutate, error_cls",
    [
        # Discriminator / version problems are explicit codec errors.
        (lambda env: env.__setitem__("format", 2), ValueError),
        (lambda env: env.__setitem__("version", 99), ValueError),
        # Structural body problems are VerificationErrors from `serialize`.
        (lambda env: env.pop("functions"), core.VerificationError),
        (lambda env: env.pop("module"), core.VerificationError),
        (lambda env: env.pop("ops"), core.VerificationError),
        (
            lambda env: env["ops"][0]["results"][0].__setitem__("id", 999999),
            core.VerificationError,
        ),
        (
            lambda env: env["ops"][0].__setitem__("operands", [999999, 999999]),
            core.VerificationError,
        ),
        (
            lambda env: env["ops"][0]["attributes"].__setitem__("bogus", 1),
            core.VerificationError,
        ),
        (
            lambda env: env["functions"][0]["input_types"][0]["shape"].__setitem__(
                0, ["dim", "B"]
            ),
            core.VerificationError,
        ),
    ],
)
def test_malformed_envelopes_raise_documented_errors(mutate, error_cls):
    """Structural problems are ``VerificationError``, never the old ``ValueError``."""
    envelope = encode_module(_simple_module())
    mutate(envelope)
    with pytest.raises(error_cls):
        decode_module(envelope)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda env: env.pop("functions"),
        lambda env: env.pop("module"),
        lambda env: env.pop("ops"),
    ],
)
def test_malformed_payloads_name_the_missing_field(mutate):
    """Structural body errors keep the documented ``malformed IR payload`` prefix."""
    envelope = encode_module(_simple_module())
    mutate(envelope)
    with pytest.raises(core.VerificationError, match="malformed IR payload"):
        decode_module(envelope)


def test_duplicate_value_id_rejected():
    """A value id may appear exactly once (a block argument reusing a result id)."""
    envelope = encode_module(_simple_module())
    block = envelope["functions"][0]["block"]
    result_id = envelope["ops"][0]["results"][0]["id"]
    block["arguments"][0]["id"] = result_id
    envelope["ops"][0]["operands"] = [result_id, result_id]
    with pytest.raises(core.VerificationError, match="duplicate value id"):
        decode_module(envelope)


# ---------------------------------------------------------------------------
# The TRUSTED fast decoder used by the transform seam
# ---------------------------------------------------------------------------


def test_trusted_decode_matches_the_validated_decode():
    """`decode_native_module` is structurally identical to `decode_module`."""
    module = _multi_feature_module()
    envelope = encode_module(module)
    fast = decode_native_module(envelope)
    validated = decode_module(encode_module(module))
    ir.verify(fast)
    _assert_module_equal(fast, validated)
    _assert_uses_consistent(fast)
    assert fast.new_op_id() > max(op.id for op in _all_ops(module))
    assert fast.new_value_id() > max(value.id for value in _all_values(module))


def test_trusted_decode_skips_verify_but_can_cross_check():
    """The fast path trusts the envelope; `verify_module=True` re-checks it.

    A recorded result type the fast decode accepts (it trusts the wire) is
    caught by the development-only `ir.verify` cross-check — the same one
    `EVOXC_NATIVE_DECODE_CHECK=1` turns on.
    """
    envelope = encode_module(_simple_module())
    envelope["ops"][0]["results"][0]["type"] = {
        "dtype": "float64",
        "shape": [{"int": 4}],
    }
    trusted = decode_native_module(envelope, verify_module=False)
    assert trusted.main.entry_block.ops[0].results[0].type.dtype == np.dtype("float64")
    with pytest.raises(core.VerificationError):
        decode_native_module(envelope, verify_module=True)


def test_native_decode_check_env_flag_enables_the_cross_check(monkeypatch):
    """`EVOXC_NATIVE_DECODE_CHECK=1` (read at import) turns the verify on."""
    from evoxc import _native_codec as codec_module

    envelope = encode_module(_simple_module())
    monkeypatch.setenv("EVOXC_NATIVE_DECODE_CHECK", "1")
    try:
        importlib.reload(codec_module)
        assert codec_module._VERIFY_NATIVE_DECODE is True
        envelope["ops"][0]["results"][0]["type"] = {
            "dtype": "float64",
            "shape": [{"int": 4}],
        }
        with pytest.raises(core.VerificationError):
            codec_module.decode_native_module(envelope)
    finally:
        monkeypatch.delenv("EVOXC_NATIVE_DECODE_CHECK", raising=False)
        importlib.reload(codec_module)
