#!/usr/bin/env python3
"""Generate the committed differential parity fixture for native shape inference.

The native shape table (`evoxc-core`) is a port of the Python authority
`evoxc/ir/inference.py`.  This script runs every op that declares a shape hook
through the Python authority and records the outcome, so the Rust integration
test `native/evoxc-core/tests/parity.rs` can pin the port without needing a
Python interpreter at test time.

Usage (from the repository root, with an interpreter that can import evoxc):

    .venv/bin/python native/evoxc-core/tests/parity/generate.py

Outputs (both committed, both deterministic):

* `cases.json` — `{"generated_by": ..., "skipped": [...], "cases": [...]}`.
  Each case is `{key, op, operands, attrs, ok, results | error_kind, error}`.
  Cases are ordered by op name, then by generation index within the op.
* `expected.json` — `{case_key: {"ok": bool, "results" | "error_kind"/"error"}}`.

Types are encoded exactly as `evoxc-core/src/wire.rs` decodes them:
`{"dtype": <numpy dtype name>, "shape": [dim, ...]}` with
`dim = {"int": n} | {"dim": name, "size": n?} | {"expr": {"op": ..., "args": [...]}} | null`.

Re-run this script whenever `evoxc/ir/op_defs/**` or `evoxc/ir/inference.py`
changes; the fixture is the differential record of the Python authority.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import numpy as np

from evoxc.core import Dim, DimExpr
from evoxc.ir.op_defs import all_opdefs
from evoxc.ir.types import ValueType

HERE = os.path.dirname(os.path.abspath(__file__))

F16 = "float16"
F32 = "float32"
F64 = "float64"
I32 = "int32"
I64 = "int64"
BOOL = "bool"
C64 = "complex64"
C128 = "complex128"

# --- JSON encoding (mirrors evoxc-core/src/wire.rs) --------------------------


def dim_json(dim: Any) -> Any:
    """Encode one shape dim exactly like `wire::dim_to_json`."""
    if dim is None:
        return None
    if isinstance(dim, bool):
        raise TypeError(f"bool is not a shape dim: {dim!r}")
    if isinstance(dim, int):
        return {"int": int(dim)}
    if isinstance(dim, Dim):
        out: dict[str, Any] = {"dim": dim.name}
        if dim.size is not None:
            out["size"] = int(dim.size)
        return out
    if isinstance(dim, DimExpr):
        return {
            "expr": {"op": dim.op, "args": [dim_json(dim.left), dim_json(dim.right)]}
        }
    raise TypeError(f"unsupported shape dim {dim!r}")


def type_json(dtype: Any, shape: Any) -> dict:
    """Encode one `ValueType` exactly like `wire::value_type_to_json`."""
    return {
        "dtype": np.dtype(dtype).name,
        "shape": [dim_json(d) for d in tuple(shape)],
    }


def encode_attr(spec: Any, value: Any) -> Any:
    """Encode one Python attribute value in its wheel-declared JSON form."""
    if spec.type == "shape":
        return [dim_json(d) for d in tuple(value)]
    if spec.type == "nested_ints":
        return [list(row) for row in value]
    if spec.type in ("ints", "floats", "strs"):
        return list(value)
    if spec.type == "dtype":
        return np.dtype(value).name
    if spec.type == "any":
        # `any` is JSON-able by contract; only Dim-free values are used here.
        return value
    if spec.type in ("bool", "int", "float", "str"):
        return value
    raise TypeError(f"attribute {spec.name!r}: unencodable type tag {spec.type!r}")


# --- curated operand patterns ------------------------------------------------

ONE_PATTERNS = [(), (3,), (2, 3), (2, 3, 4), (2, 2)]

TWO_PATTERNS = [
    ((2, 3), (2, 3)),
    ((2, 3), (3,)),
    ((2, 3), (3, 4)),
    ((2, 3, 4), (3, 4)),
    ((4,), (2, 3, 4)),
]

THREE_PATTERNS = [
    ((2, 3), (2, 3), (2, 3)),
    ((2, 3), (3,), (3, 4)),
]

# Operand dtypes per op (index = operand position); float32 otherwise.
DTYPE_RULES: dict[str, dict[int, str]] = {
    "bitwise_and": {0: I32, 1: I32},
    "bitwise_or": {0: I32, 1: I32},
    "bitwise_xor": {0: I32, 1: I32},
    "bitwise_left_shift": {0: I32, 1: I32},
    "bitwise_right_shift": {0: I32, 1: I32},
    "logical_and": {0: BOOL, 1: BOOL},
    "logical_or": {0: BOOL, 1: BOOL},
    "logical_not": {0: BOOL},
    "select": {0: BOOL},
    "scatter": {1: I64},
    "gather": {1: I32},
}

REDUCE_OPS = {
    "reduce_sum": "sum",
    "reduce_prod": "prod",
    "reduce_mean": "mean",
    "reduce_max": "max",
    "reduce_min": "min",
}


def _prod(shape: tuple) -> int:
    out = 1
    for d in shape:
        if not isinstance(d, int) or isinstance(d, bool):
            raise TypeError(f"non-int dim {d!r} in a reshape target")
        out *= d
    return out


def attr_variants(name: str, shapes: list[tuple]) -> list[dict]:
    """Curated attribute variants for one operand shape tuple."""
    rank = len(shapes[0])
    if name in REDUCE_OPS:
        return [
            {"axes": [0], "reduce_op": REDUCE_OPS[name], "keepdims": False},
            {"axes": [0], "reduce_op": REDUCE_OPS[name], "keepdims": True},
            {"axes": [], "reduce_op": REDUCE_OPS[name], "keepdims": False},
        ]
    if name in ("sort", "argsort"):
        return [{"axis": 0}, {}]
    if name in ("argmax", "argmin"):
        return [{"axis": 0}, {"axis": 0, "keepdims": True}, {}]
    if name == "broadcast":
        return [{"shape": [2] + list(shapes[0])}]
    if name == "cast":
        return [{"dtype": "float64"}]
    if name in ("cumprod", "cumsum"):
        return [{"axis": 0}]
    if name == "flip":
        return [{"axes": [0]}, {"axes": list(range(rank))}]
    if name == "pad":
        return [{"padding_config": [[1, 0] for _ in range(rank)]}]
    if name == "reshape":
        return [{"shape": [_prod(shapes[0])]}, {"shape": [-1]}]
    if name == "roll":
        return [{"shift": [1], "axis": 0}]
    if name == "slice":
        full = {
            "start_indices": [0] * rank,
            "limit_indices": list(shapes[0]),
        }
        partial = {
            "start_indices": [0] * rank,
            "limit_indices": [max(1, d - 1) for d in shapes[0]],
            "strides": [2] * rank,
        }
        return [full, partial]
    if name == "tile":
        return [{"reps": [2] * max(1, rank)}]
    if name == "transpose":
        return [{"permutation": list(reversed(range(rank)))}]
    if name in ("tril", "triu"):
        return [{"k": 0}]
    return [{}]


# --- explicit cases for ops whose operand layout is op-specific ---------------
#
# `SPECIAL[op]` is a list of `{"operands": [(dtype, shape), ...], "attrs": {...}}`
# entries.  They are the only cases generated for that op.

SPECIAL: dict[str, list[dict]] = {
    # --- elementwise dtype edge: complex magnitude is real -------------------
    "abs": [
        {"operands": [(C64, (2, 3))]},
        {"operands": [(C128, (2,))]},
    ],
    # --- symbolic dims: identity / broadcast / DimExpr arithmetic ------------
    "add": [
        {"operands": [(F32, (Dim("B"), 3)), (F32, (Dim("B"), 3))]},
        {"operands": [(F32, (Dim("B", 4), 3)), (F32, (1, 3))]},
        {"operands": [(I32, (Dim("B"),)), (F32, (Dim("B"),))]},
    ],
    "multiply": [
        {"operands": [(F32, (Dim("B"), 3)), (F32, (3,))]},
    ],
    "concatenate": [
        {"operands": [(F32, (2, 3)), (F32, (2, 3))], "attrs": {"axis": 0}},
        {"operands": [(F32, (3,)), (I32, (3,))], "attrs": {"axis": 0}},
        {"operands": [(F32, (2, 3, 4)), (F32, (2, 3, 4))], "attrs": {"axis": 2}},
        {"operands": [(F32, (Dim("B"), 3)), (F32, (Dim("C"), 3))], "attrs": {"axis": 0}},
        {"operands": [(F32, (2, 3)), (F32, (2, 4))], "attrs": {"axis": 0}},
    ],
    "tile": [
        {"operands": [(F32, (Dim("B"), 3))], "attrs": {"reps": [2, 1]}},
        {"operands": [(F32, (2, 3))], "attrs": {"reps": [2, 1]}},
        {"operands": [(F32, (3,))], "attrs": {"reps": [2, 3]}},
    ],
    "transpose": [
        {"operands": [(F32, (Dim("B"), 3))], "attrs": {"permutation": [1, 0]}},
        {"operands": [(F32, (2, 3, 4))], "attrs": {"permutation": [2, 0, 1]}},
        {"operands": [(F32, (2, 3))], "attrs": {"permutation": [0, 0]}},
    ],
    "reduce_sum": [
        {"operands": [(F32, (Dim("B"), 3))],
         "attrs": {"axes": [1], "reduce_op": "sum", "keepdims": False}},
    ],
    # --- linalg -------------------------------------------------------------
    "conv": [
        {"operands": [(F32, (1, 3, 8, 8)), (F32, (4, 3, 3, 3))],
         "attrs": {"padding": "SAME"}},
        {"operands": [(F32, (1, 3, 8, 8)), (F32, (4, 3, 3, 3))],
         "attrs": {"padding": "VALID"}},
        {"operands": [(F32, (2, 3, 8, 8)), (F32, (6, 3, 3, 3))],
         "attrs": {"strides": [2, 2], "padding": "SAME"}},
        {"operands": [(F32, (1, 3, 8, 8)), (F32, (4, 3, 3, 3))],
         "attrs": {"padding": [[1, 1], [1, 1]]}},
        {"operands": [(F32, (1, 6, 8, 8)), (F32, (8, 3, 3, 3))],
         "attrs": {"padding": "VALID", "feature_group_count": 2}},
        {"operands": [(F32, (1, 3, 8, 8)), (F32, (4, 2, 3, 3))], "attrs": {}},
    ],
    "diagonal": [
        {"operands": [(F32, (2, 3, 4))],
         "attrs": {"offset": 0, "axis1": 1, "axis2": 2}},
        {"operands": [(F32, (2, 3))], "attrs": {"offset": 1, "axis1": 0, "axis2": 1}},
        {"operands": [(F32, (2, 3))], "attrs": {"axis1": 0, "axis2": 0}},
    ],
    "solve": [
        {"operands": [(F32, (2, 2)), (F32, (2, 3))]},
        {"operands": [(F32, (2, 2)), (F32, (2,))]},
        {"operands": [(I32, (2, 2)), (F32, (2, 3))]},
        {"operands": [(F32, (2, 2)), (F32, (3, 3))]},
    ],
    "eigh": [
        {"operands": [(F32, (2, 2))]},
        {"operands": [(F32, (2, 3, 3))]},
        {"operands": [(F32, (2, 3))]},
    ],
    "cholesky": [
        {"operands": [(F32, (2, 2))]},
        {"operands": [(F32, (2, 3, 3))]},
        {"operands": [(F32, (2, 3, 4))]},
    ],
    "qr": [
        {"operands": [(F32, (2, 3))]},
        {"operands": [(F32, (4, 2))]},
        {"operands": [(F32, (5,))]},
    ],
    "svd": [
        {"operands": [(F32, (2, 3))]},
        {"operands": [(F32, (4, 2))]},
        {"operands": [(C64, (3, 3))]},
        {"operands": [(F32, (5,))]},
    ],
    "matrix_rank": [
        {"operands": [(F32, (2, 3))]},
        {"operands": [(F32, (2, 3, 4))]},
        {"operands": [(F32, (4,))]},
    ],
    "matrix_exp": [
        {"operands": [(F32, (2, 2))]},
        {"operands": [(F32, (2, 3, 3))]},
        {"operands": [(F32, (2, 3))]},
    ],
    # --- collectives / graph scalars ---------------------------------------
    "all_gather": [
        {"operands": [(F32, (2, 6))], "attrs": {"group": "world", "group_size": 2}},
        {"operands": [(F32, (2, 6))],
         "attrs": {"axis": 1, "group": "world", "group_size": 3}},
        {"operands": [(F32, (6,))], "attrs": {"group": "world", "group_size": None}},
    ],
    "all_reduce": [
        {"operands": [(F32, (2, 3))], "attrs": {"reduce_op": "sum", "group": "world"}},
        {"operands": [(F32, (2, 3))],
         "attrs": {"reduce_op": "sum", "group": "dp", "group_size": 4}},
    ],
    "all_to_all": [
        {"operands": [(F32, (2, 6))],
         "attrs": {"split_axis": 0, "concat_axis": 1, "group": "world",
                   "group_size": 2}},
        {"operands": [(F32, (2, 6))],
         "attrs": {"split_axis": 0, "concat_axis": 0, "group": "world",
                   "group_size": 2}},
        {"operands": [(F32, (4, 6))],
         "attrs": {"split_axis": 0, "concat_axis": 1, "group": "world",
                   "group_size": None}},
    ],
    "broadcast_collective": [
        {"operands": [(F32, (2, 3))], "attrs": {"group": "world"}},
        {"operands": [(F32, (2, 3))],
         "attrs": {"group": "world", "src_rank": 1, "group_size": 2}},
    ],
    "collective_permute": [
        {"operands": [(F32, (2, 3))],
         "attrs": {"source_target_pairs": [[0, 1], [1, 0]], "group": "world",
                   "group_size": 2}},
    ],
    "reduce_scatter": [
        {"operands": [(F32, (2, 6))],
         "attrs": {"reduce_op": "sum", "axis": 0, "group": "world",
                   "group_size": 2}},
        {"operands": [(F32, (2, 6))],
         "attrs": {"reduce_op": "max", "axis": 0, "group": "world",
                   "group_size": None}},
    ],
    "rank": [{"operands": [], "attrs": {"group": "world"}}],
    "world_size": [{"operands": [], "attrs": {"group": "world"}}],
    # --- key-based random ---------------------------------------------------
    "random_key_mix": [
        {"operands": [(I64, ())], "attrs": {"algorithm": "splitmix64", "salt": 1}},
        {"operands": [(I32, (2,))], "attrs": {"algorithm": "threefry2x32", "salt": 7}},
        {"operands": [(I32, (4,))], "attrs": {"algorithm": "philox4x32_10", "salt": 3}},
        {"operands": [(F32, (2,))], "attrs": {"salt": 1}},
    ],
    "random_uniform": [
        {"operands": [(I64, ()), (F32, ()), (F32, ())],
         "attrs": {"shape": [2, 3], "dtype": "float32"}},
        {"operands": [(I32, (2,)), (F32, (2, 3)), (F32, ())],
         "attrs": {"algorithm": "threefry2x32", "shape": [2, 3], "dtype": "float64"}},
        {"operands": [(I64, ()), (F32, ()), (F32, ())],
         "attrs": {"shape": [2, 3], "dtype": "float32", "algorithm": "bogus"}},
    ],
    "random_normal": [
        {"operands": [(I64, ()), (F32, ()), (F32, ())],
         "attrs": {"shape": [2, 3], "dtype": "float32"}},
        {"operands": [(I32, (4,)), (F32, ()), (F32, (2, 3))],
         "attrs": {"algorithm": "philox4x32_10", "shape": [2, 3],
                   "dtype": "float32"}},
    ],
    "random_randint": [
        {"operands": [(I64, ()), (I64, ()), (I64, ())],
         "attrs": {"shape": [2, 3], "dtype": "int64"}},
    ],
    "random_permutation": [
        {"operands": [(I64, ()), (I32, ())], "attrs": {"n": 4, "dtype": "int64"}},
        {"operands": [(I64, ()), (I32, ())], "attrs": {"dtype": "int64"}},
        {"operands": [(I64, ()), (I32, ())], "attrs": {"n": -1, "dtype": "int64"}},
    ],
    "random_multinomial": [
        {"operands": [(I64, ()), (F32, (2, 3))],
         "attrs": {"num_samples": 3, "dtype": "int64"}},
        {"operands": [(I64, ()), (F32, (2, 3))],
         "attrs": {"num_samples": -1, "dtype": "int64"}},
    ],
    # --- sparse -------------------------------------------------------------
    "sparse_from_dense": [
        {"operands": [(F32, (2, 3))], "attrs": {"dense_shape": [2, 3], "dtype": "float32"}},
        {"operands": [(F64, (4,))], "attrs": {"dense_shape": [4], "dtype": "float64"}},
        {"operands": [(F32, (2, 3))], "attrs": {"dense_shape": [2, 3, 4], "dtype": "float32"}},
    ],
    "sparse_to_dense": [
        {"operands": [(I64, (4, 2)), (F32, (4,))],
         "attrs": {"dense_shape": [3, 2], "dtype": "float32"}},
        {"operands": [(I64, (2, 4, 2)), (F32, (2, 4))],
         "attrs": {"dense_shape": [3, 2], "dtype": "float32"}},
    ],
    "sparse_coo_to_csr": [
        {"operands": [(I64, (4, 2)), (F32, (4,))],
         "attrs": {"dense_shape": [3, 2], "dtype": "float32"}},
        {"operands": [(I64, (2, 4, 2)), (F32, (2, 4))],
         "attrs": {"dense_shape": [3, 2], "dtype": "float32"}},
        {"operands": [(I64, (4, 2)), (F32, (4,))],
         "attrs": {"dense_shape": [3, 2, 4], "dtype": "float32"}},
    ],
    "sparse_coo_to_csc": [
        {"operands": [(I64, (4, 2)), (F32, (4,))],
         "attrs": {"dense_shape": [3, 2], "dtype": "float32"}},
    ],
    "sparse_csr_to_coo": [
        {"operands": [(I64, (4,)), (I64, (3,)), (F32, (3,))],
         "attrs": {"dense_shape": [3, 2], "dtype": "float32"}},
        {"operands": [(I64, (2, 4)), (I64, (2, 3)), (F32, (2, 3))],
         "attrs": {"dense_shape": [3, 2], "dtype": "float32"}},
    ],
    "sparse_csc_to_coo": [
        {"operands": [(I64, (4,)), (I64, (3,)), (F32, (3,))],
         "attrs": {"dense_shape": [3, 2], "dtype": "float32"}},
    ],
    "sparse_negate": [
        {"operands": [(I64, (4, 2)), (F32, (4,))],
         "attrs": {"dense_shape": [3, 2], "dtype": "float32"}},
    ],
    "sparse_add": [
        {"operands": [(I64, (4, 2)), (F32, (4,)), (I64, (4, 2)), (F32, (4,))],
         "attrs": {"dense_shape": [3, 2], "dtype": "float32"}},
        {"operands": [(I64, (2, 4, 2)), (F32, (2, 4)), (I64, (2, 4, 2)),
                      (F32, (2, 4))],
         "attrs": {"dense_shape": [3, 2], "dtype": "float32"}},
    ],
    "sparse_multiply": [
        {"operands": [(I64, (4, 2)), (F32, (4,)), (I64, (4, 2)), (F32, (4,))],
         "attrs": {"dense_shape": [3, 2], "dtype": "float32"}},
    ],
    "sparse_multiply_dense": [
        {"operands": [(I64, (4, 2)), (F32, (4,)), (F32, (3, 2))],
         "attrs": {"dense_shape": [3, 2], "dtype": "float32"}},
    ],
    "sparse_reduce_sum": [
        {"operands": [(I64, (4, 2)), (F32, (4,))],
         "attrs": {"dense_shape": [3, 2], "dtype": "float32", "axes": [1],
                   "keepdims": False}},
        {"operands": [(I64, (4, 2)), (F32, (4,))],
         "attrs": {"dense_shape": [3, 2], "dtype": "float32", "axes": [0],
                   "keepdims": True}},
        {"operands": [(I64, (4, 2)), (F32, (4,))],
         "attrs": {"dense_shape": [3, 2], "dtype": "float32", "axes": [],
                   "keepdims": False}},
    ],
    "sparse_transpose": [
        {"operands": [(I64, (4, 2)), (F32, (4,))],
         "attrs": {"dense_shape": [3, 2], "dtype": "float32", "perm": [1, 0]}},
        {"operands": [(I64, (4, 2)), (F32, (4,))],
         "attrs": {"dense_shape": [3, 2], "dtype": "float32", "perm": [0, 0]}},
    ],
    "sparse_reshape": [
        {"operands": [(I64, (4, 2)), (F32, (4,))],
         "attrs": {"dense_shape": [6], "dtype": "float32", "old_shape": [3, 2]}},
        {"operands": [(I64, (4, 2)), (F32, (4,))],
         "attrs": {"dense_shape": [5], "dtype": "float32", "old_shape": [3, 2]}},
    ],
    "sparse_concatenate": [
        {"operands": [(I64, (4, 2)), (F32, (4,)), (I64, (3, 2)), (F32, (3,))],
         "attrs": {"dense_shape": [6, 2], "dtype": "float32", "axis": 0,
                   "operand_extents": [3, 3]}},
        {"operands": [(I64, (4, 2)), (F32, (4,)), (I64, (3, 2)), (F32, (3,))],
         "attrs": {"dense_shape": [3, 4], "dtype": "float32", "axis": 1,
                   "operand_extents": [2, 2]}},
        {"operands": [(I64, (4, 2)), (F32, (4,))],
         "attrs": {"dense_shape": [3, 2], "dtype": "float32", "axis": 0,
                   "operand_extents": [3]}},
    ],
    "sparse_dot_dense": [
        {"operands": [(I64, (4, 2)), (F32, (4,)), (F32, (2, 3))],
         "attrs": {"dense_shape": [3, 2], "dtype": "float32"}},
    ],
    "dense_dot_sparse": [
        {"operands": [(F32, (3, 2)), (I64, (4, 2)), (F32, (4,))],
         "attrs": {"dense_shape": [2, 3], "dtype": "float32"}},
    ],
    # --- control ------------------------------------------------------------
    "return": [
        {"operands": []},
        {"operands": [(F32, (2, 3))]},
        {"operands": [(F32, (2,)), (I64, (3,))]},
    ],
    "while": [
        {"operands": [(F32, (2, 3))]},
        {"operands": [(F32, ()), (I32, (3,))]},
    ],
}


# --- generation --------------------------------------------------------------


def operands_for(name: str, count: int, shapes: tuple) -> list[tuple[str, tuple]]:
    """Operand `(dtype, shape)` list with the op's curated dtype rules."""
    rules = DTYPE_RULES.get(name, {})
    return [(rules.get(i, F32), tuple(shapes[i])) for i in range(count)]


def cases_for(op_def: Any) -> tuple[list[dict], str | None]:
    """Generate `(operand specs, attrs)` cases for one op.

    Returns `(cases, skip_reason)`; `skip_reason` is `None` unless a required
    attribute could not be filled (the op is then recorded as skipped).
    """
    name = op_def.name
    if name in SPECIAL:
        return [dict(entry) for entry in SPECIAL[name]], None
    arity = op_def.arity
    if isinstance(arity, int):
        count = arity
    else:
        return [], f"variadic op without a curated operand layout"
    patterns = {1: ONE_PATTERNS, 2: TWO_PATTERNS, 3: THREE_PATTERNS}.get(count)
    if patterns is None:
        return [], f"no curated operand pattern for arity {count}"
    out = []
    for pattern in patterns:
        shape_list = [pattern] if count == 1 else list(pattern)
        for attrs in attr_variants(name, shape_list):
            out.append(
                {
                    "operands": operands_for(name, count, shape_list),
                    "attrs": attrs,
                }
            )
    return out, None


def main() -> int:
    cases: list[dict] = []
    skipped: list[dict] = []
    counts = {"total": 0, "ok": 0, "error": 0}

    for op_def in all_opdefs():
        if op_def.shape_fn is None:
            continue
        raw_cases, skip_reason = cases_for(op_def)
        if skip_reason is not None:
            skipped.append({"op": op_def.name, "reason": skip_reason})
            print(f"SKIP {op_def.name}: {skip_reason}")
            continue
        if not raw_cases:
            skipped.append({"op": op_def.name, "reason": "no generated cases"})
            print(f"SKIP {op_def.name}: no generated cases")
            continue
        specs = {spec.name: spec for spec in op_def.attributes}
        for index, raw in enumerate(raw_cases):
            operands = raw["operands"]
            attrs_py = dict(raw.get("attrs", {}))
            # A required attribute we do not fill would fail for the wrong
            # reason (a missing-attribute error, not a shape answer).
            missing = [
                spec.name
                for spec in op_def.attributes
                if spec.required and spec.name not in attrs_py
            ]
            if missing:
                skipped.append(
                    {
                        "op": op_def.name,
                        "reason": "unfilled required attribute(s): "
                        + ", ".join(missing),
                    }
                )
                print(f"SKIP {op_def.name}: unfilled required attrs {missing}")
                break
            try:
                encoded_attrs = {
                    key: encode_attr(specs[key], value)
                    for key, value in attrs_py.items()
                }
            except KeyError as exc:
                raise SystemExit(
                    f"{op_def.name}: attribute {exc} is not declared by the OpDef"
                ) from exc
            value_types = tuple(
                ValueType(np.dtype(dtype), tuple(shape)) for dtype, shape in operands
            )
            # The Python hooks receive the *effective* attribute map, i.e. the
            # declared defaults with the case's explicit values layered on top
            # (some hooks read `attributes["axis"]` directly).  The fixture
            # still records only the explicit attributes, so the native
            # default-fill path is exercised (and pinned) as well.
            effective_attrs = {
                spec.name: spec.default
                for spec in op_def.attributes
                if not spec.required
            }
            effective_attrs.update(attrs_py)
            try:
                results = op_def.shape_fn(value_types, dict(effective_attrs))
                outcome = {
                    "ok": True,
                    "results": [type_json(rt.dtype, rt.shape) for rt in results],
                }
                counts["ok"] += 1
            except Exception as exc:  # noqa: BLE001 - the outcome IS the record
                outcome = {
                    "ok": False,
                    "error_kind": type(exc).__name__,
                    "error": str(exc),
                }
                counts["error"] += 1
            counts["total"] += 1
            key = f"{op_def.name}#{index}"
            cases.append(
                {
                    "key": key,
                    "op": op_def.name,
                    "operands": [type_json(dtype, shape) for dtype, shape in operands],
                    "attrs": encoded_attrs,
                    **outcome,
                }
            )

    expected = {
        case["key"]: (
            {"ok": True, "results": case["results"]}
            if case["ok"]
            else {
                "ok": False,
                "error_kind": case["error_kind"],
                "error": case["error"],
            }
        )
        for case in cases
    }

    payload = {
        "generated_by": "native/evoxc-core/tests/parity/generate.py",
        "authority": "evoxc/ir/inference.py",
        "skipped": skipped,
        "cases": cases,
    }
    cases_path = os.path.join(HERE, "cases.json")
    expected_path = os.path.join(HERE, "expected.json")
    with open(cases_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")
    with open(expected_path, "w", encoding="utf-8") as handle:
        json.dump(expected, handle, indent=1, sort_keys=True)
        handle.write("\n")

    ops_covered = {case["op"] for case in cases}
    print(
        f"\n{counts['total']} cases over {len(ops_covered)} ops "
        f"({counts['ok']} ok, {counts['error']} errors), {len(skipped)} skipped"
    )
    print(f"wrote {cases_path}")
    print(f"wrote {expected_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
