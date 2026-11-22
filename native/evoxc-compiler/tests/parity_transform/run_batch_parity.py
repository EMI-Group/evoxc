"""Differential parity harness: the native `batch_functions` FFI vs the Python merge.

The native merge engine lives in `evoxc-compiler` (`src/batch_functions/`) and is
exposed to Python as ``evoxc._evoxc_native.batch_functions`` by
``native/evoxc-python``; this harness drives that live entry point against the
``evoxc.batch_functions`` frontend.

Every case runs the SAME merge through
``merge_native.batch_functions(envelopes, config)`` and the Python frontend and
compares

(a) the STRUCTURE of the resulting module (function name/metadata,
    block-argument types, op-name sequence, canonicalized attributes with
    ndarray payloads by dtype/shape/raw bytes, operand/result references
    remapped to defining-position-or-block-argument indices, terminator
    operand references),
(b) the NUMBERS (both graphs through the explicit numpy pipeline on identical
    seeded arrays, leaf by leaf), and
(c) the ``BatchReport`` (the exact 12-key payload).

``config`` mirrors `native/evoxc-compiler/src/batch_functions::BatchFunctionsConfig`:
``{"axes": [0, null, ...], "out_axes": [0, ...], "switch_ops": ..., "bands":
..., "alignment": "..."}``.  The Python public API takes its axes/out_axes as
PYTREES, but the FFI only sees programs (no Python tree), so the harness
flattens them to ONE ENTRY PER TENSOR LEAF — ``0`` maps the leading function
axis, ``null`` leaves the value unmapped — exactly the native
``Vec<Option<i64>>`` shape.  ``switch_ops`` is ``null``/``true`` or the
group list (``["add", "subtract"]`` / ``["elementwise"]``), ``bands`` is
``null``/an int/``"auto"``, ``alignment`` its canonical string.

The comparison self-checks prove the comparator can say "no" (a changed op
name, attribute, operand reference, dtype or output reference is detected; a
perturbed / retyped / reshaped / re-arity'd numeric leaf is detected).

The only SKIP is a missing prerequisite: when the built
``evoxc/_evoxc_native.abi3.so`` (or the ``.venv`` it is loaded through) is
absent, the harness prints ``SKIP: ...`` and exits 0 — build it with
``source scripts/dev-env.sh && scripts/build-native.sh --debug``.  Should a
loaded extension expose no ``batch_functions`` symbol (a stale build), the
corpus still runs the Python merge checks and the run reports the cases as
skipped instead of passing silently.

Run it directly (the cargo gate does exactly this)::

    source scripts/dev-env.sh
    .venv/bin/python native/evoxc-compiler/tests/parity_transform/run_batch_parity.py

``-k SUBSTRING`` filters cases by name.  Exit status is 1 on any mismatch and 2
on a usage error; the last lines are::

    BATCH PARITY: <cases> cases, <checks> checks, <mismatches> mismatches

and, in the missing-symbol fallback only, a trailing
``BATCH PARITY: <n> cases skipped (native FFI not implemented)`` plus a
``SKIP:`` line.
"""
from __future__ import annotations

import copy
import json
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import evoxc
from evoxc import core, ir
from evoxc.trace import Graph
from evoxc.transforms import batching
from evoxc.transforms.batch_functions import _normalize_axes, _normalize_out_axes

import batch_cases
import convert

#: The exact key set of `BatchReport.as_dict()` (and the native `to_json`).
REPORT_KEYS = (
    "n_functions",
    "value_classes",
    "shared_classes",
    "switched",
    "stacked_constants",
    "output_muxes",
    "emitted_ops",
    "bands",
    "band_sizes",
    "band_strategy",
    "cost_estimate",
    "flat_cost_estimate",
)

BAND_STRATEGIES = {"flat", "explicit", "auto"}


# ---------------------------------------------------------------------------
# Options / native config
# ---------------------------------------------------------------------------
def _python_options(case) -> dict:
    """The public `batch_functions` keyword arguments for one case."""
    options = {
        "axes": 0,
        "out_axes": 0,
        "switch_ops": None,
        "bands": None,
        "alignment": "variant",
    }
    options.update(case.options)
    return options


def _json_value(value):
    """Normalize a Python option to its JSON shape (tuples become lists)."""
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _raw_option(option) -> list:
    """Per-leaf entries for an option the Python normalizer REFUSED.

    Only the v1 deferral cases take this path, and every one of them drives a
    single tensor leaf (the option is a bare ``1``/``None``), so the raw reading
    is exact.  A structured option that fails normalization would be a harness
    bug, not a deferral, so it is rejected loudly.
    """
    if option is None or isinstance(option, int) and not isinstance(option, bool):
        return [option]
    if isinstance(option, (list, tuple, dict)):
        leaves, _tree = core.flatten(option)
        return list(leaves)
    raise AssertionError(f"unsupported raw axes/out_axes option {option!r}")


def _length_matches(label: str, entries: Sequence, expected: int) -> None:
    if entries and len(entries) > expected:
        raise AssertionError(
            f"{label} has {len(entries)} entries for {expected} tensor leaves"
        )


def _native_config(graphs: Sequence[Graph], options: dict) -> dict:
    """The JSON config the future FFI consumes (per-tensor-leaf option lists).

    ``axes``/``out_axes`` are flattened exactly like the Python entry point
    normalizes them (via `_normalize_axes` / `_normalize_out_axes`); when the
    Python normalizer REFUSES an option (a v1 deferral such as a non-leading
    axis) the raw reading is used, so the native engine receives the bad option
    and must raise its own `TransformError`.
    """
    graph = graphs[0]
    try:
        axes = list(batching._tensor_input_axes(graph, _normalize_axes(graph, options["axes"])))
    except core.TransformError:
        axes = _raw_option(options["axes"])
        _length_matches("axes", axes, len(graph.tensor_specs))
    try:
        out_axes = list(_normalize_out_axes(graph, options["out_axes"]))
    except core.TransformError:
        out_axes = _raw_option(options["out_axes"])
        _length_matches("out_axes", out_axes, _output_count(graph))
    return {
        "axes": [_json_value(entry) for entry in axes],
        "out_axes": [_json_value(entry) for entry in out_axes],
        "switch_ops": _json_value(options["switch_ops"]),
        "bands": _json_value(options["bands"]),
        "alignment": _json_value(options["alignment"]),
    }


def _output_count(graph: Graph) -> int:
    return len(graph.module.main.entry_block.terminator.operands)


def _as_dict(value, label: str):
    """Accept a dict or a JSON string (the FFI may hand back either)."""
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict):
        return value
    raise AssertionError(f"{label} is a {type(value).__name__}, expected a dict")


# ---------------------------------------------------------------------------
# The structural comparison (id-allocation independent)
# ---------------------------------------------------------------------------
def _array_key(array: np.ndarray) -> tuple:
    """Canonical ndarray identity: dtype name, shape and raw little-endian bytes."""
    native = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return (
        "ndarray",
        array.dtype.name,
        tuple(int(dim) for dim in array.shape),
        np.ascontiguousarray(native).tobytes(),
    )


def _attr_key(value) -> tuple:
    """A canonical, mutation-friendly view of one attribute value."""
    if isinstance(value, np.ndarray):
        return _array_key(value)
    if isinstance(value, np.generic):
        return _attr_key(value.item())
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value)
    if isinstance(value, complex):
        return ("complex", value.real, value.imag)
    if isinstance(value, str):
        return ("str", value)
    if value is None:
        return ("none",)
    if isinstance(value, np.dtype):
        return ("dtype", value.name)
    if isinstance(value, (list, tuple)):
        return ("seq", tuple(_attr_key(item) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                sorted((str(key), _attr_key(item)) for key, item in value.items())
            ),
        )
    return ("repr", type(value).__name__, repr(value))


def _dim_key(dim) -> tuple:
    if isinstance(dim, core.Dim):
        return ("dim", dim.name, None if dim.size is None else int(dim.size))
    if isinstance(dim, core.DimExpr):
        return ("expr", dim.op, _dim_key(dim.left), _dim_key(dim.right))
    if dim is None:
        return ("dynamic",)
    return ("int", int(dim))


def _type_key(value_type) -> tuple:
    return (
        np.dtype(value_type.dtype).name,
        tuple(_dim_key(dim) for dim in value_type.shape),
    )


def _structural_key(module) -> dict:
    """Canonical module view with value ids remapped to structural positions.

    Every operand reference becomes ``("arg", block_argument_index)`` or
    ``("op", defining_op_position, result_index)``, so the two engines may
    allocate ids differently without a false mismatch — while a changed
    reference, op name, attribute, dtype or terminator operand still differs.
    """
    function = module.main
    block = function.entry_block
    arg_index = {id(value): index for index, value in enumerate(block.arguments)}
    result_index: Dict[int, Tuple[int, int]] = {}
    for op_index, op in enumerate(block.ops):
        for index, value in enumerate(op.results):
            result_index[id(value)] = (op_index, index)

    def ref(value) -> tuple:
        key = id(value)
        if key in arg_index:
            return ("arg", arg_index[key])
        if key in result_index:
            op_index, index = result_index[key]
            return ("op", op_index, index)
        raise AssertionError(
            "structural key: a value is neither a block argument nor an op result"
        )

    return {
        "module_name": module.name,
        "module_metadata": _attr_key(module.metadata),
        "function_name": function.name,
        "function_metadata": _attr_key(function.metadata),
        "block_arguments": [_type_key(value.type) for value in block.arguments],
        "ops": [
            {
                "name": op.name,
                "attributes": _attr_key(op.attributes),
                "operands": [ref(value) for value in op.operands],
                "results": [_type_key(value.type) for value in op.results],
            }
            for op in block.ops
        ],
        "terminator_operands": [
            ref(value) for value in block.terminator.operands
        ],
    }


def _first_diff(expected, actual, path: str = "<root>") -> Optional[str]:
    """The first canonical-view divergence, or ``None``."""
    if type(expected) is not type(actual):
        return f"{path}: type {type(expected).__name__} vs {type(actual).__name__}"
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            return f"{path}: keys {sorted(expected)} vs {sorted(actual)}"
        for key in expected:
            detail = _first_diff(expected[key], actual[key], f"{path}.{key}")
            if detail is not None:
                return detail
        return None
    if isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            return f"{path}: length {len(expected)} vs {len(actual)}"
        for index, (want, got) in enumerate(zip(expected, actual)):
            detail = _first_diff(want, got, f"{path}[{index}]")
            if detail is not None:
                return detail
        return None
    if expected != actual:
        return f"{path}: {expected!r} vs {actual!r}"
    return None


def _structural_self_check() -> str:
    """Prove the comparator rejects every structural change it must catch."""
    graphs = [
        evoxc.trace(batch_cases.identity_program, *batch_cases.IDENTICAL_SPECS)
        for _ in range(2)
    ]
    module = evoxc.batch_functions(graphs).module
    key = _structural_key(module)
    if _first_diff(key, copy.deepcopy(key)) is not None:
        raise AssertionError("structural self-check: identical views reported a diff")

    def detect(label: str, mutate) -> None:
        perturbed = copy.deepcopy(key)
        mutate(perturbed)
        if _first_diff(key, perturbed) is None:
            raise AssertionError(
                f"structural self-check: a changed {label} was not detected"
            )

    ops = key["ops"]
    multiply = next(i for i, op in enumerate(ops) if op["name"] == "multiply")
    reshape = next(i for i, op in enumerate(ops) if op["name"] == "reshape")
    detect("op name", lambda k: k["ops"][multiply].__setitem__("name", "bogus_op"))
    detect(
        "attribute",
        lambda k: k["ops"][reshape].__setitem__("attributes", ("changed",)),
    )
    detect(
        "operand reference",
        lambda k: k["ops"][multiply]["operands"].__setitem__(0, ("arg", 999)),
    )
    detect(
        "dtype",
        lambda k: k["block_arguments"].__setitem__(
            0, ("float64", k["block_arguments"][0][1])
        ),
    )
    detect(
        "output reference",
        lambda k: k["terminator_operands"].__setitem__(0, ("arg", 0)),
    )
    return "the structural comparator detects op-name/attribute/reference/dtype changes"


# ---------------------------------------------------------------------------
# Execution / numeric comparison
# ---------------------------------------------------------------------------
def _concrete_shape(spec) -> tuple:
    shape = []
    for dim in spec.shape:
        if isinstance(dim, core.Dim):
            if dim.size is None:
                raise AssertionError(
                    "the batch parity corpus only merges static shapes "
                    f"(symbolic dim {dim.name!r} has no concrete extent)"
                )
            shape.append(int(dim.size))
        elif isinstance(dim, core.DimExpr) or dim is None:
            raise AssertionError(f"the batch parity corpus is static; got dim {dim!r}")
        else:
            shape.append(int(dim))
    return tuple(shape)


def _default_array(spec, rng: np.random.Generator) -> np.ndarray:
    """A seeded runtime array for one merged tensor input."""
    shape = _concrete_shape(spec)
    dtype = np.dtype(spec.dtype)
    if dtype.kind == "f":
        return rng.uniform(0.5, 1.5, size=shape).astype(dtype)
    if dtype.kind == "c":
        return (
            rng.uniform(0.5, 1.5, size=shape) + 1j * rng.uniform(-0.5, 0.5, size=shape)
        ).astype(dtype)
    if dtype.kind == "b":
        return rng.integers(0, 2, size=shape).astype(dtype)
    if dtype.kind in "iu":
        return rng.integers(1, 4, size=shape).astype(dtype)
    raise AssertionError(f"the harness cannot generate a {dtype} input")


def _args(graph: Graph, arrays: Sequence[np.ndarray]):
    """A runtime argument tree from generated tensor leaves + recorded statics."""
    static = {record.index: record.value for record in graph.static_values}
    pending = iter(arrays)
    leaves = [
        static[index] if index in static else next(pending)
        for index in range(graph.input_specs.num_leaves)
    ]
    return core.unflatten(leaves, graph.input_specs)


def _execute(graph: Graph, args) -> list:
    """lower → compile → load → run on the numpy backend; flat numpy leaves."""
    lowered = evoxc.lower(graph, backend="numpy")
    artifact = evoxc.compile(lowered, backend="numpy")
    executable = evoxc.load(artifact, backend="numpy")
    out = evoxc.run(executable, *args)
    leaves, _tree = core.flatten(out)
    return [leaf.numpy() if isinstance(leaf, evoxc.Tensor) else leaf for leaf in leaves]


def _leaf_detail(want, got) -> Optional[str]:
    """Exact dtype/shape; ``array_equal`` for int/bool; ``allclose`` for floats."""
    want = np.asarray(want)
    got = np.asarray(got)
    if want.dtype != got.dtype:
        return f"dtype {want.dtype} vs {got.dtype}"
    if want.shape != got.shape:
        return f"shape {want.shape} vs {got.shape}"
    if want.dtype.kind in "biu":
        if not np.array_equal(want, got):
            return "integer/bool values differ"
        return None
    if not np.array_equal(np.isnan(want), np.isnan(got)):
        return "NaN positions differ"
    if not np.allclose(want, got, rtol=1e-6, atol=1e-6, equal_nan=True):
        diff = np.abs(want - got)
        finite = np.isfinite(diff)
        worst = float(np.max(diff[finite])) if finite.any() else float("inf")
        return f"values differ (max |expected-actual| = {worst:e})"
    return None


def _compare_leaf_lists(expected: Sequence, actual: Sequence) -> Optional[str]:
    if len(expected) != len(actual):
        return f"output leaf count {len(expected)} vs {len(actual)}"
    for index, (want, got) in enumerate(zip(expected, actual)):
        detail = _leaf_detail(want, got)
        if detail is not None:
            return f"leaf {index}: {detail}"
    return None


def _numeric_self_check() -> str:
    """The numeric comparison must be able to say "no"."""
    reference = [np.arange(6, dtype=np.float32).reshape(2, 3)]
    if _compare_leaf_lists(reference, [reference[0].copy()]) is not None:
        raise AssertionError("numeric self-check: identical leaves reported a mismatch")
    perturbed = reference[0].copy()
    perturbed[0, 0] += 1.0
    checks = [
        (perturbed, "perturbed value"),
        (reference[0].astype(np.float64), "dtype change"),
        (reference[0][:1], "shape change"),
    ]
    for leaf, label in checks:
        if _compare_leaf_lists(reference, [leaf]) is None:
            raise AssertionError(f"numeric self-check: a {label} was not detected")
    if _compare_leaf_lists(reference, [reference[0], reference[0]]) is None:
        raise AssertionError("numeric self-check: a leaf-count change was not detected")
    return "the numeric comparison detects value/dtype/shape/arity changes"


# ---------------------------------------------------------------------------
# Per-case checks
# ---------------------------------------------------------------------------
def _check_report(graph: Graph, n_functions: int) -> List[str]:
    """The 12-key report with the documented key types + band invariants."""
    failures: List[str] = []
    report = getattr(graph, "batch_report", None)
    if report is None:
        return ["the merged graph carries no batch_report"]
    data = report.as_dict()
    if not isinstance(data, dict):
        return [f"batch_report.as_dict() is a {type(data).__name__}, expected dict"]
    if set(data) != set(REPORT_KEYS):
        failures.append(
            f"report keys {sorted(data)} != the documented 12 {sorted(REPORT_KEYS)}"
        )
    for key in (
        "n_functions",
        "value_classes",
        "shared_classes",
        "stacked_constants",
        "output_muxes",
        "emitted_ops",
        "bands",
    ):
        if not isinstance(data.get(key), int) or isinstance(data.get(key), bool):
            failures.append(f"report[{key!r}] is not an int: {data.get(key)!r}")
    if data.get("n_functions") != n_functions:
        failures.append(
            f"report n_functions {data.get('n_functions')!r} != {n_functions}"
        )
    sizes = data.get("band_sizes")
    if not isinstance(sizes, list) or not all(
        isinstance(size, int) and not isinstance(size, bool) for size in sizes
    ):
        failures.append(f"report['band_sizes'] is not a list of ints: {sizes!r}")
    elif sizes and sum(sizes) != data.get("n_functions"):
        failures.append(
            f"report band_sizes {sizes} do not sum to n_functions {data.get('n_functions')}"
        )
    if data.get("band_strategy") not in BAND_STRATEGIES:
        failures.append(f"report['band_strategy'] is not one of {sorted(BAND_STRATEGIES)}")
    for key in ("cost_estimate", "flat_cost_estimate"):
        value = data.get(key)
        if value is not None and not isinstance(value, (int, float)):
            failures.append(f"report[{key!r}] is neither None nor a number: {value!r}")
    switched = data.get("switched")
    if not isinstance(switched, list):
        failures.append(f"report['switched'] is not a list: {switched!r}")
    else:
        for index, record in enumerate(switched):
            if not isinstance(record, dict):
                failures.append(f"report['switched'][{index}] is not a dict")
                continue
            if set(record) != {"group", "variants", "rows_per_variant"}:
                failures.append(
                    f"report['switched'][{index}] keys {sorted(record)} are not "
                    "{group, variants, rows_per_variant}"
                )
            if not isinstance(record.get("group"), str):
                failures.append(f"report['switched'][{index}]['group'] is not a str")
            variants = record.get("variants")
            if not isinstance(variants, list) or not all(
                isinstance(name, str) for name in variants
            ):
                failures.append(
                    f"report['switched'][{index}]['variants'] is not a list of str"
                )
            rows = record.get("rows_per_variant")
            if not isinstance(rows, list) or not all(
                isinstance(row, list)
                and all(isinstance(value, int) and not isinstance(value, bool) for value in row)
                for row in rows
            ):
                failures.append(
                    f"report['switched'][{index}]['rows_per_variant'] is not a "
                    "list of int lists"
                )
    return failures


def _row_exact(
    case,
    merged: Graph,
    graphs: Sequence[Graph],
    options: dict,
) -> Optional[str]:
    """Row i of the merged result equals `graphs[i]` on the per-function inputs."""
    graph = graphs[0]
    entries = list(
        batching._tensor_input_axes(graph, _normalize_axes(graph, options["axes"]))
    )
    out_entries = list(_normalize_out_axes(graph, options["out_axes"]))
    rng = np.random.default_rng(case.seed)
    merged_arrays = [_default_array(spec, rng) for spec in merged.tensor_specs]
    merged_leaves = _execute(merged, _args(merged, merged_arrays))
    if len(out_entries) != len(merged_leaves):
        return (
            f"internal: {len(out_entries)} out_axes entries for "
            f"{len(merged_leaves)} merged output leaves"
        )
    for index, program in enumerate(graphs):
        row_arrays = [
            array[index] if entry == 0 else array
            for entry, array in zip(entries, merged_arrays)
        ]
        per_function = _execute(program, _args(program, row_arrays))
        if len(per_function) != len(merged_leaves):
            return (
                f"row {index}: function output arity {len(per_function)} vs "
                f"merged {len(merged_leaves)}"
            )
        for leaf_index, (want, got) in enumerate(zip(per_function, merged_leaves)):
            actual = got[index] if out_entries[leaf_index] == 0 else got
            detail = _leaf_detail(want, actual)
            if detail is not None:
                return f"row {index} leaf {leaf_index}: {detail}"
    return None


def _op_names(graph: Graph) -> list:
    return [op.name for op in graph.module.main.entry_block.ops]


def _normalize_report(payload) -> dict:
    data = dict(_as_dict(payload, "the native batch_report"))
    if set(data) != set(REPORT_KEYS):
        raise AssertionError(
            f"native report keys {sorted(data)} != {sorted(REPORT_KEYS)}"
        )
    data["band_sizes"] = [int(size) for size in data["band_sizes"]]
    data["switched"] = [
        {
            "group": record["group"],
            "variants": list(record["variants"]),
            "rows_per_variant": [list(rows) for rows in record["rows_per_variant"]],
        }
        for record in data["switched"]
    ]
    for key in ("cost_estimate", "flat_cost_estimate"):
        value = data[key]
        data[key] = None if value is None else float(value)
    return data


def _report_diff(expected: dict, actual: dict) -> Optional[str]:
    if set(expected) != set(actual):
        return f"report keys {sorted(expected)} vs {sorted(actual)}"
    for key in REPORT_KEYS:
        if expected[key] != actual[key]:
            return f"report[{key!r}] {expected[key]!r} (python) vs {actual[key]!r} (native)"
    return None


# ---------------------------------------------------------------------------
# Running one case
# ---------------------------------------------------------------------------
@dataclass
class Stats:
    cases: int = 0
    checks: int = 0
    mismatches: int = 0
    deferrals: int = 0


def _report_status(status: str, name: str, detail: str = "") -> None:
    line = f"[{status}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)


def _run_checks(stats: Stats, name: str, checks: Sequence[Tuple[str, Any]]) -> None:
    """Run ``(label, callable)`` checks; a callable returns ``None`` or a detail."""
    failures: List[str] = []
    for label, check in checks:
        stats.checks += 1
        try:
            detail = check()
        except Exception:  # noqa: BLE001 - a harness bug must not hide results
            detail = "harness error:\n" + traceback.format_exc()
        if detail is not None:
            failures.append(f"{label}: {detail}")
    if failures:
        stats.mismatches += len(failures)
        detail = failures[0]
        if len(failures) > 1:
            detail += f" (+{len(failures) - 1} more failed check(s))"
        _report_status("FAIL", name, detail)
    else:
        _report_status("OK", name, f"{len(checks)} check(s)")


def _run_python_only_case(case, stats: Stats) -> Optional[Graph]:
    """Build + execute the case entirely on the Python side (skip mode)."""
    try:
        graphs = case.build()
        options = _python_options(case)
        merged = evoxc.batch_functions(graphs, **options)
    except Exception:  # noqa: BLE001
        stats.checks += 1
        stats.mismatches += 1
        _report_status("FAIL", case.name, "case build failed:\n" + traceback.format_exc())
        return None

    def verify():
        ir.verify(merged.module)
        return None

    def report():
        failures = _check_report(merged, len(graphs))
        return "; ".join(failures) if failures else None

    def numerics():
        return _row_exact(case, merged, graphs, options)

    _run_checks(
        stats,
        case.name,
        [("verify", verify), ("report", report), ("row-exact", numerics)],
    )
    return merged


def _run_native_case(case, merge, stats: Stats) -> Optional[Graph]:
    """Compare the Python merge against the native FFI (structural / numeric / report)."""
    try:
        graphs = case.build()
        options = _python_options(case)
        expected = evoxc.batch_functions(graphs, **options)
        envelope = [convert.to_native(graph.module) for graph in graphs]
        config = _native_config(graphs, options)
    except Exception:  # noqa: BLE001
        stats.checks += 1
        stats.mismatches += 1
        _report_status("FAIL", case.name, "case build failed:\n" + traceback.format_exc())
        return None

    try:
        result = merge(envelope, config)
    except Exception as exc:  # noqa: BLE001
        stats.checks += 1
        stats.mismatches += 1
        _report_status(
            "FAIL",
            case.name,
            f"native batch_functions raised {_error_label(exc)}: {exc}",
        )
        return expected

    native_module = None

    def structural():
        nonlocal native_module
        native_module = convert.from_native(_as_dict(result["module"], "result['module']"))
        return _first_diff(
            _structural_key(expected.module), _structural_key(native_module)
        )

    def verify():
        ir.verify(expected.module)
        if native_module is None:
            return "the structural check did not produce a native module"
        ir.verify(native_module)
        return None

    def numerics():
        if native_module is None:
            return "the structural check did not produce a native module"
        native_graph = Graph(
            native_module,
            expected.input_specs,
            expected.tensor_specs,
            expected.output_tree,
            expected.static_values,
            expected.output_static_values,
        )
        rng = np.random.default_rng(case.seed)
        arrays = [_default_array(spec, rng) for spec in expected.tensor_specs]
        args = _args(expected, arrays)
        expected_leaves = _execute(expected, args)
        actual_leaves = _execute(native_graph, args)
        return _compare_leaf_lists(expected_leaves, actual_leaves)

    def report():
        native_report = _normalize_report(result["batch_report"])
        python_report = _normalize_report(expected.batch_report.as_dict())
        return _report_diff(python_report, native_report)

    _run_checks(
        stats,
        case.name,
        [
            ("structural", structural),
            ("verify", verify),
            ("numeric", numerics),
            ("report", report),
        ],
    )
    return expected


def _error_label(exc: BaseException) -> str:
    kind = getattr(exc, "kind", None)
    if kind:
        return f"{type(exc).__name__} (kind={kind})"
    return type(exc).__name__


def _is_transform_error(exc: BaseException, expected: type) -> bool:
    if isinstance(exc, expected):
        return True
    kind = getattr(exc, "kind", None)
    return "TransformError" in type(exc).__name__ or kind == "TransformError"


def _run_deferral(case, merge, stats: Stats) -> None:
    """Both engines must raise the documented `TransformError`."""
    stats.cases += 1
    stats.checks += 1
    try:
        graphs = case.build()
        options = _python_options(case)
        config = _native_config(graphs, options)
        envelope = [convert.to_native(graph.module) for graph in graphs]
    except Exception:  # noqa: BLE001
        stats.mismatches += 1
        _report_status("FAIL", case.name, "case build failed:\n" + traceback.format_exc())
        return

    python_exc = None
    try:
        evoxc.batch_functions(graphs, **options)
    except Exception as exc:  # noqa: BLE001
        python_exc = exc
    if python_exc is None:
        stats.mismatches += 1
        _report_status("FAIL", case.name, "the Python merge did NOT raise")
        return
    if not _is_transform_error(python_exc, case.expected):
        stats.mismatches += 1
        _report_status(
            "FAIL",
            case.name,
            f"the Python merge raised {type(python_exc).__name__}, expected "
            f"{case.expected.__name__}: {python_exc}",
        )
        return

    if merge is None:
        stats.deferrals += 1
        _report_status(
            "DEFERRED",
            case.name,
            f"python raises {case.expected.__name__} ({case.note})",
        )
        return

    native_exc = None
    try:
        merge(envelope, config)
    except Exception as exc:  # noqa: BLE001
        native_exc = exc
    if native_exc is None:
        stats.mismatches += 1
        _report_status("FAIL", case.name, "the native merge did NOT raise")
        return
    if not _is_transform_error(native_exc, case.expected):
        stats.mismatches += 1
        _report_status(
            "FAIL",
            case.name,
            f"the native merge raised {_error_label(native_exc)}, expected "
            f"{case.expected.__name__}: {native_exc}",
        )
        return
    stats.deferrals += 1
    _report_status(
        "DEFERRED",
        case.name,
        f"both engines raise {case.expected.__name__} ({case.note})",
    )


# ---------------------------------------------------------------------------
# Cross-case invariants (the pinned observations)
# ---------------------------------------------------------------------------
def _cross_case_checks(merged_by_name: Dict[str, Graph], stats: Stats) -> None:
    """Identical-programs / bands-1-vs-None / the two pinned auto costs."""

    def pinned_costs():
        failures = []
        identical = merged_by_name.get("bands_auto_identical")
        if identical is not None:
            report = identical.batch_report
            if (report.bands, report.band_sizes) != (1, (5,)):
                failures.append(
                    f"identical auto bands/sizes {(report.bands, report.band_sizes)} != (1, (5,))"
                )
            if report.cost_estimate != 1232.0 or report.flat_cost_estimate != 1232.0:
                failures.append(
                    "identical auto cost "
                    f"({report.cost_estimate}, {report.flat_cost_estimate}) != "
                    "(1232.0, 1232.0)"
                )
        two_group = merged_by_name.get("bands_auto")
        if two_group is not None:
            report = two_group.batch_report
            if report.cost_estimate != 39168.0 or report.flat_cost_estimate != 43136.0:
                failures.append(
                    "two-group auto cost "
                    f"({report.cost_estimate}, {report.flat_cost_estimate}) != "
                    "(39168.0, 43136.0)"
                )
            elif report.cost_estimate > report.flat_cost_estimate:
                failures.append("the auto partition costs more than the flat merge")
        return "; ".join(failures) if failures else None

    def identical_has_no_select():
        failures = []
        for name in ("identical_programs", "bands_auto_identical"):
            graph = merged_by_name.get(name)
            if graph is None:
                continue
            select_count = _op_names(graph).count("select")
            if select_count:
                failures.append(f"{name} emits {select_count} select op(s)")
            if graph.batch_report.output_muxes:
                failures.append(
                    f"{name} reports {graph.batch_report.output_muxes} output mux(es)"
                )
        return "; ".join(failures) if failures else None

    def bands_one_matches_flat():
        flat = merged_by_name.get("bands_flat")
        one = merged_by_name.get("bands_one")
        if flat is None or one is None:
            return None
        if _op_names(flat) != _op_names(one):
            return "bands=1 and bands=None emit different op-name sequences"
        return None

    checks = []
    if any(name in merged_by_name for name in ("bands_auto", "bands_auto_identical")):
        checks.append(("pinned auto costs", pinned_costs))
    if any(name in merged_by_name for name in ("identical_programs", "bands_auto_identical")):
        checks.append(("identical merges emit no select", identical_has_no_select))
    if "bands_flat" in merged_by_name and "bands_one" in merged_by_name:
        checks.append(("bands=1 matches bands=None", bands_one_matches_flat))
    for label, check in checks:
        stats.checks += 1
        try:
            detail = check()
        except Exception:  # noqa: BLE001
            detail = "harness error:\n" + traceback.format_exc()
        if detail is not None:
            stats.mismatches += 1
            _report_status("FAIL", f"cross-case: {label}", detail)
        else:
            _report_status("OK", f"cross-case: {label}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    only = None
    if args:
        if args[0] not in ("-k", "--filter"):
            print(f"usage: run_batch_parity.py [-k SUBSTRING]", file=sys.stderr)
            return 2
        if len(args) < 2:
            print("usage: run_batch_parity.py [-k SUBSTRING]", file=sys.stderr)
            return 2
        only = args[1]

    selected_cases = [
        case for case in batch_cases.CASES if only is None or only in case.name
    ]
    selected_deferrals = [
        case for case in batch_cases.DEFERRALS if only is None or only in case.name
    ]
    if only is not None and not selected_cases and not selected_deferrals:
        print(f"no case matches {only!r}", file=sys.stderr)
        return 2

    try:
        native = convert.load_native()
    except FileNotFoundError as exc:
        print(f"SKIP: {exc}")
        print(
            "SKIP: build it with `source scripts/dev-env.sh && "
            "scripts/build-native.sh --debug`"
        )
        return 0
    merge_native = getattr(native, "batch_functions", None)

    stats = Stats()
    stats.checks += 1
    try:
        detail = _structural_self_check()
        _report_status("OK", "self-check: structural comparator", detail)
    except Exception:  # noqa: BLE001
        stats.mismatches += 1
        _report_status(
            "FAIL", "self-check: structural comparator", traceback.format_exc()
        )
    stats.checks += 1
    try:
        detail = _numeric_self_check()
        _report_status("OK", "self-check: numeric comparison", detail)
    except Exception:  # noqa: BLE001
        stats.mismatches += 1
        _report_status("FAIL", "self-check: numeric comparison", traceback.format_exc())

    merged_by_name: Dict[str, Graph] = {}
    for case in selected_cases:
        stats.cases += 1
        if merge_native is None:
            merged = _run_python_only_case(case, stats)
        else:
            merged = _run_native_case(case, merge_native, stats)
        if merged is not None:
            merged_by_name[case.name] = merged

    for case in selected_deferrals:
        _run_deferral(case, merge_native, stats)

    _cross_case_checks(merged_by_name, stats)

    print(
        f"BATCH PARITY: {stats.cases} cases, {stats.checks} checks, "
        f"{stats.mismatches} mismatches"
    )
    if merge_native is None:
        print(
            f"BATCH PARITY: {stats.cases} cases skipped "
            "(native FFI not implemented)"
        )
        print(
            "SKIP: native evoxc._evoxc_native.batch_functions is not implemented "
            f"yet; {stats.cases} cases built and Python-side checked"
        )
    return 1 if stats.mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
