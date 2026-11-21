"""End-to-end verification of the native `transform` entry point.

A manual dev check (not run by cargo or pytest): it builds `evoxc-native-ir`
envelopes by hand and exercises all five transforms — plus the extension-rule
bridge and the config diagnostics — through the installed PyO3 extension.

    source scripts/dev-env.sh
    scripts/build-native.sh --debug
    .venv/bin/python native/evoxc-python/verify_transform.py
"""
import importlib.util
import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
so = _REPO_ROOT / "evoxc" / "_evoxc_native.abi3.so"
spec = importlib.util.spec_from_file_location("evoxc._evoxc_native", so)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def ty(dtype, shape):
    return {"dtype": dtype, "shape": [{"int": d} for d in shape]}


def op_entry(oid, name, operands, result_ids, result_types, attrs, regions):
    return {
        "id": oid,
        "name": name,
        "operands": list(operands),
        "results": [
            {"id": vid, "type": t} for vid, t in zip(result_ids, result_types)
        ],
        "attributes": attrs or {},
        "regions": regions or [],
        "location": None,
    }


class Builder:
    def __init__(self, name="main"):
        self.name = name
        self.value_types = {}
        self.next_value = 0
        self.next_op = 0
        self.ops = {}
        self.main_args = []
        self.main_ops = []
        self.returns = []

    def add_arg(self, t):
        vid = self.next_value
        self.next_value += 1
        self.value_types[vid] = t
        self.main_args.append({"id": vid, "type": t})
        return vid

    def add_op(self, name, operands, result_types, attrs=None, regions=None, in_main=True):
        oid = self.next_op
        self.next_op += 1
        ids = []
        for t in result_types:
            vid = self.next_value
            self.next_value += 1
            self.value_types[vid] = t
            ids.append(vid)
        self.ops[oid] = op_entry(oid, name, operands, ids, result_types, attrs, regions)
        if in_main:
            self.main_ops.append({"id": oid, "ref": str(oid)})
            if name == "return":
                self.returns = list(operands)
        return oid, ids

    def finish(self):
        return {
            "format": "evoxc-native-ir",
            "version": 1,
            "module": {"name": "m", "metadata": {}},
            "functions": [
                {
                    "name": self.name,
                    "input_types": [arg["type"] for arg in self.main_args],
                    "output_types": [self.value_types[v] for v in self.returns],
                    "metadata": {},
                    "block": {"arguments": self.main_args, "ops": self.main_ops},
                }
            ],
            "ops": [self.ops[k] for k in sorted(self.ops)],
        }


def square_graph():
    b = Builder()
    x = b.add_arg(ty("float32", [3]))
    _, (sq,) = b.add_op("multiply", [x, x], [ty("float32", [3])])
    b.add_op("return", [sq], [])
    return b.finish(), x


def sum_square_graph(dtype="float32"):
    b = Builder()
    x = b.add_arg(ty(dtype, [3]))
    _, (sq,) = b.add_op("multiply", [x, x], [ty(dtype, [3])])
    _, (s,) = b.add_op(
        "reduce_sum",
        [sq],
        [ty(dtype, [])],
        attrs={"axes": [0], "keepdims": False, "reduce_op": "sum"},
    )
    b.add_op("return", [s], [])
    return b.finish(), x


def int_input_scalar_graph():
    b = Builder()
    x = b.add_arg(ty("int32", [3]))
    _, (f,) = b.add_op("cast", [x], [ty("float32", [3])], attrs={"dtype": "float32"})
    _, (s,) = b.add_op(
        "reduce_sum",
        [f],
        [ty("float32", [])],
        attrs={"axes": [0], "keepdims": False, "reduce_op": "sum"},
    )
    b.add_op("return", [s], [])
    return b.finish(), x


def if_graph():
    b = Builder()
    x = b.add_arg(ty("float32", []))
    _, (cond,) = b.add_op("less", [x, x], [ty("bool", [])])
    then_op, _ = b.add_op("return", [x], [], in_main=False)
    else_op, _ = b.add_op("return", [x], [], in_main=False)
    then_block = {"arguments": [], "ops": [{"id": then_op, "ref": str(then_op)}]}
    else_block = {"arguments": [], "ops": [{"id": else_op, "ref": str(else_op)}]}
    _, (y,) = b.add_op("if", [cond], [ty("float32", [])], regions=[then_block, else_block])
    b.add_op("return", [y], [])
    return b.finish(), x


def block_call_graph():
    b = Builder()
    x = b.add_arg(ty("float32", [3]))
    y = b.add_arg(ty("float32", [3]))
    _, (z,) = b.add_op(
        "block_call",
        [x, y],
        [ty("float32", [3])],
        attrs={
            "block_name": "foo",
            "result_specs": [ty("float32", [3])],
            "static_args": [],
        },
    )
    b.add_op("return", [z], [])
    return b.finish(), x, y


def names(envelope):
    return [op["name"] for op in envelope["ops"]]


def show(label, envelope, fields=()):
    fn = envelope["functions"][0]
    print(f"{label}: inputs={fn['input_types']}")
    print(f"{label}: outputs={fn['output_types']} ops={names(envelope)}")
    for field in fields:
        print(f"{label}: {field}={envelope.get(field)}")


def expect_error(label, call):
    try:
        call()
    except native.NativeError as err:
        print(f"{label}: NativeError kind={err.kind!r} message={str(err)!r}")
        return err
    raise AssertionError(f"{label}: expected a NativeError")


# 1. vectorize f(x) = x*x ---------------------------------------------------
env, x = square_graph()
res = native.transform("vectorize", env, {"axes": {x: 0}})
show("1 vectorize x*x", res)
assert res["functions"][0]["input_types"][0] == {
    "dtype": "float32",
    "shape": [{"dim": "batch"}, {"int": 3}],
}
assert res["functions"][0]["output_types"] == res["functions"][0]["input_types"]
assert names(res)[0] == "multiply"
assert res["transform"]["kind"] == "vectorize"
assert res["transform"]["batch_dim"] == "batch"
assert res["transform"]["output_ids"] and res["transform"]["input_ids"]

# 2. vmap -------------------------------------------------------------------
res = native.transform("vmap", env, {"axes": {x: 0}, "out_axes": 0})
show("2a vmap out_axes=0", res)
assert res["functions"][0]["output_types"] == [
    {"dtype": "float32", "shape": [{"dim": "batch"}, {"int": 3}]}
]
expect_error(
    "2b vmap out_axes=None on a mapped output",
    lambda: native.transform("vmap", env, {"axes": {x: 0}, "out_axes": None}),
)
# out_axes=None on an UNMAPPED output is the legal case.
env2, x2 = square_graph()
res = native.transform("vmap", env2, {"axes": {x2: None}, "out_axes": None})
show("2c vmap out_axes=None on an unmapped output", res)
assert res["functions"][0]["output_types"] == [ty("float32", [3])]
expect_error(
    "2d vmap out_axes=1 (engine defers non-leading output axes)",
    lambda: native.transform("vmap", env, {"axes": {x: 0}, "out_axes": 1}),
)

# 3. grad -------------------------------------------------------------------
env, x = sum_square_graph()
res = native.transform("grad", env, {"argnums": [0]})
show("3a grad sum(x*x)", res)
assert res["functions"][0]["output_types"] == [ty("float32", [3])]
assert res["transform"]["kind"] == "grad"
assert res["transform"]["batch_dim"] is None
env, x = square_graph()
err = expect_error(
    "3b grad of a non-scalar output",
    lambda: native.transform("grad", env, {"argnums": [0]}),
)
assert err.kind == "ShapeError" and "scalar" in str(err)
env, x = int_input_scalar_graph()
err = expect_error(
    "3c grad of an int input",
    lambda: native.transform("grad", env, {"argnums": [0]}),
)
assert err.kind == "TransformError" and "int32" in str(err)

# 4. jvp / vjp --------------------------------------------------------------
env, x = square_graph()
res = native.transform("jvp", env, {"tangent_inputs": [0]})
show("4a jvp tangent_inputs=[0]", res)
fn = res["functions"][0]
assert len(fn["input_types"]) == 2
assert fn["output_types"] == [ty("float32", [3]), ty("float32", [3])]
res = native.transform("vjp", env, {"cotangent_inputs": [0]})
show("4b vjp cotangent_inputs=[0]", res)
fn = res["functions"][0]
assert len(fn["input_types"]) == 2
assert fn["output_types"] == [ty("float32", [3]), ty("float32", [3])]

# 5. control flow is not vectorizable --------------------------------------
env, x = if_graph()
err = expect_error(
    "5 vectorize an if graph",
    lambda: native.transform("vectorize", env, {"axes": {x: 0}}),
)
assert "if" in str(err)

# 6. extension rule bridge --------------------------------------------------
env, x, y = block_call_graph()


def add_rule(request):
    batched = request["operand_types"][0]
    rule = {
        "format": "evoxc-native-ir",
        "version": 1,
        "module": {"name": "rule", "metadata": {}},
        "functions": [
            {
                "name": "rule",
                "input_types": request["operand_types"],
                "output_types": [batched],
                "metadata": {},
                "block": {
                    "arguments": [
                        {"id": 0, "type": request["operand_types"][0]},
                        {"id": 1, "type": request["operand_types"][1]},
                    ],
                    "ops": [{"id": 0, "ref": "0"}, {"id": 1, "ref": "1"}],
                },
            }
        ],
        "ops": [
            op_entry(0, "add", [0, 1], [2], [batched], {}, []),
            op_entry(1, "return", [2], [], [], {}, []),
        ],
    }
    add_rule.requests.append(request)
    return {
        "module": rule,
        "entry": "rule",
        "output_ids": [2],
        "output_axes": [0],
    }


add_rule.requests = []
res = native.transform(
    "vectorize", env, {"axes": {x: 0, y: 0}, "extension_rule": add_rule}
)
show("6a extension rule (block:foo -> add)", res)
assert names(res) == ["add", "return"], names(res)
assert res["functions"][0]["input_types"] == [
    {"dtype": "float32", "shape": [{"dim": "batch"}, {"int": 3}]},
    {"dtype": "float32", "shape": [{"dim": "batch"}, {"int": 3}]},
]
assert res["functions"][0]["output_types"] == [
    {"dtype": "float32", "shape": [{"dim": "batch"}, {"int": 3}]}
]
request = dict(add_rule.requests[0])
print("6a request keys:", sorted(request))
assert request["kind"] == "batching"
assert request["op_name"] == "block_call"
assert request["rule_key"] == "block:foo"
assert request["axes"] == [0, 0] and request["tangents"] is None
assert len(request["operand_types"]) == 2 and len(request["result_types"]) == 1

env, x, y = block_call_graph()
err = expect_error(
    "6b extension rule returning None",
    lambda: native.transform(
        "vectorize", env, {"axes": {x: 0, y: 0}, "extension_rule": lambda request: None}
    ),
)
assert "block:foo" in str(err)

# 7. config validation ------------------------------------------------------
env, x = square_graph()
err = expect_error(
    "7a unknown transform kind",
    lambda: native.transform("jax", env, {"axes": {x: 0}}),
)
assert "jax" in str(err) and "vectorize" in str(err)
err = expect_error(
    "7b unknown config key",
    lambda: native.transform("vectorize", env, {"axis": {x: 0}}),
)
assert "axis" in str(err)
err = expect_error(
    "7c axes list form (pair list)",
    lambda: native.transform("vectorize", env, {"axes": [[x, 1]]}),
)
assert "must be 0" in str(err)  # engine rejects a non-leading mapped axis

# The pair-list form with a leading axis succeeds.
env, x = square_graph()
res = native.transform("vectorize", env, {"axes": [[x, 0]]})
show("7d axes pair-list form", res)
assert res["transform"]["batch_dim"] == "batch"

print("ALL CHECKS PASSED")
