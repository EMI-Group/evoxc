#!/usr/bin/env python3
"""FFI-level inference parity: `evoxc._evoxc_native.infer` vs the Python authority.

A manual dev check (not run by cargo or pytest).  It replays the committed
differential fixture `native/evoxc-core/tests/parity/cases.json` (518 cases
over all 113 shape-function ops, generated from `evoxc/ir/inference.py` by
`tests/parity/generate.py`) through the **installed PyO3 extension** and
compares every answer with the recorded expectation:

* a successful case must return exactly the recorded result types — dtype
  name plus the full shape JSON, so symbolic `Dim` identity/`size`, `DimExpr`
  operator/operand structure and a runtime-dynamic `None` dim are compared
  structurally;
* a rejection case must raise `NativeError` with the recorded public error
  family (`NativeError.kind`).

Where a rejection case's family differs from the fixture record, the script
does not silently tolerate it: it re-runs the case through the live Python
authority at both layers and only classifies the difference as a *documented
divergence* when the record still matches the authority's `shape_fn` AND either
the authority's own pipeline (`evoxc.ir.builder.Builder.create`, the layer that
corresponds to the native `infer`) agrees with the native family (the fixture
recorded the deeper `shape_fn` error) or the recorded family is not an EVOXC
family at all while the message is identical (no native family can express it).
Everything else is a mismatch, and a stale record never passes as one.

It also replays a small `infer` module-argument section (signature-only callee
contexts in all three entry forms plus a full envelope resolving `call`, the
unknown-callee `VerificationError` and the `TypeError` argument-shape
rejections) — extension-only, so it needs no `evoxc` import.
It additionally cross-checks **every** successful case against the live Python
authority — `evoxc.ir.op_defs.opdef(name).shape_fn(operand_types, attrs)` with
the op's declared defaults layered under the case's explicit attributes, the
same call `generate.py` makes — so a stale fixture is caught too.

Usage (from the repository root, with an interpreter that can import evoxc):

    source scripts/dev-env.sh
    scripts/build-native.sh --debug
    .venv/bin/python native/evoxc-python/verify_infer_parity.py

Exit codes: 0 all cases matched, 1 a mismatch (or a stale/absent fixture, or
too few cross-checked cases), 2 the extension is not installed (only when
`--allow-missing` was not passed — with it the script reports the skip and
exits 0).
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Any

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FIXTURE_DIR = REPO_ROOT / "native" / "evoxc-core" / "tests" / "parity"
EXTENSION = REPO_ROOT / "evoxc" / "_evoxc_native.abi3.so"

#: The public error families of `native/CONTEXT.md` ("Error mapping").  A
#: `NativeError.kind` is always one of these; the fixture may also record a
#: plain Python exception raised by a `shape_fn` (e.g. `ValueError`), which no
#: native family can express.
EVOXC_FAMILIES = frozenset(
    {
        "TraceError",
        "ShapeError",
        "TransformError",
        "BackendError",
        "PersistenceError",
        "DeviceError",
        "DTypeError",
        "VerificationError",
        "EVOXCError",
    }
)

#: Default minimum number of successful cases that must be re-derived from the
#: live Python authority (the objective's floor; the script cross-checks all).
MIN_LIVE_CROSS_CHECK = 50


# --- loading the extension ---------------------------------------------------


def load_native() -> Any | None:
    """Import the installed extension module, or print why it is missing."""
    try:
        import evoxc._evoxc_native as native
    except (ImportError, OSError) as exc:
        print("SKIP: the native extension evoxc._evoxc_native is not importable.")
        print(f"  expected: {EXTENSION}")
        print(f"  present:  {EXTENSION.exists()}")
        print(f"  error:    {type(exc).__name__}: {exc}")
        print("  build it with: scripts/build-native.sh   (or --debug for a")
        print("  fast unoptimized build), then re-run this script.")
        return None
    return native


# --- fixture decoding --------------------------------------------------------
#
# The fixture encodes a type exactly like `evoxc-core/src/wire.rs` decodes it:
# `{"dtype": <numpy dtype name>, "shape": [dim, ...]}` with
# `dim = {"int": n} | {"dim": name, "size": n?} | {"expr": {...}} | None`.
# These helpers are the inverse of `generate.py`'s `dim_json`/`type_json`.


def dim_from_json(encoded: Any) -> Any:
    """Decode one fixture dim into `None`/`int`/`Dim`/`DimExpr`."""
    from evoxc.core import Dim, DimExpr

    if encoded is None:
        return None
    if isinstance(encoded, int) and not isinstance(encoded, bool):
        return encoded
    if not isinstance(encoded, dict):
        raise TypeError(f"unsupported fixture dim {encoded!r}")
    if "int" in encoded:
        return int(encoded["int"])
    if "dim" in encoded:
        size = encoded.get("size")
        return Dim(encoded["dim"], None if size is None else int(size))
    if "expr" in encoded:
        expr = encoded["expr"]
        return DimExpr(
            expr["op"],
            dim_from_json(expr["args"][0]),
            dim_from_json(expr["args"][1]),
        )
    raise TypeError(f"unsupported fixture dim {encoded!r}")


def dim_to_json(dim: Any) -> Any:
    """Encode one dim exactly like `generate.py`'s `dim_json`."""
    from evoxc.core import Dim, DimExpr

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
            "expr": {
                "op": dim.op,
                "args": [dim_to_json(dim.left), dim_to_json(dim.right)],
            }
        }
    raise TypeError(f"unsupported shape dim {dim!r}")


def value_type_from_json(encoded: Any) -> Any:
    """Decode one fixture type dict into a Python `ValueType`."""
    from evoxc.ir.types import ValueType

    return ValueType(
        np.dtype(encoded["dtype"]),
        tuple(dim_from_json(dim) for dim in encoded["shape"]),
    )


def value_type_to_json(value_type: Any) -> dict:
    """Encode one Python `ValueType` into the fixture's JSON form."""
    return {
        "dtype": np.dtype(value_type.dtype).name,
        "shape": [dim_to_json(dim) for dim in tuple(value_type.shape)],
    }


# --- attribute decoding ------------------------------------------------------


def decode_attr(spec_type: str, value: Any) -> Any:
    """Decode one fixture attribute value back into its Python form.

    Only needed for the live-authority cross-checks: `native.infer` takes the
    fixture's JSON attributes verbatim.
    """
    if value is None:
        return None
    if spec_type == "shape":
        return tuple(dim_from_json(dim) for dim in value)
    if spec_type == "nested_ints":
        return [list(row) for row in value]
    if spec_type in ("ints", "floats", "strs"):
        return list(value)
    if spec_type == "dtype":
        return np.dtype(value)
    if spec_type in ("any", "bool", "int", "float", "str"):
        return value
    raise TypeError(f"unencodable attribute type tag {spec_type!r}")


def attr_spec(op_def: Any, name: str) -> Any:
    """The `AttrSpec` called ``name`` on ``op_def``."""
    for spec in op_def.attributes:
        if spec.name == name:
            return spec
    raise KeyError(f"op '{op_def.name}' declares no attribute {name!r}")


def effective_attrs(op_def: Any, encoded_attrs: dict) -> dict:
    """The attribute map the Python `shape_fn` receives.

    Mirrors `generate.py`: declared defaults for every non-required attribute,
    with the case's explicit values layered on top.
    """
    attrs = {spec.name: spec.default for spec in op_def.attributes if not spec.required}
    for key, value in encoded_attrs.items():
        attrs[key] = decode_attr(attr_spec(op_def, key).type, value)
    return attrs


# --- comparison --------------------------------------------------------------


_MISSING = object()


def first_difference(expected: Any, got: Any, path: str = "results") -> str | None:
    """Describe the first structural difference between two JSON-shaped values."""
    if isinstance(expected, dict) and isinstance(got, dict):
        for key in sorted(set(expected) | set(got)):
            sub = first_difference(
                expected.get(key, _MISSING), got.get(key, _MISSING), f"{path}.{key}"
            )
            if sub is not None:
                return sub
        return None
    if isinstance(expected, list) and isinstance(got, list):
        if len(expected) != len(got):
            return f"{path}: expected {len(expected)} entr(ies), got {len(got)}"
        for index, (want, actual) in enumerate(zip(expected, got)):
            sub = first_difference(want, actual, f"{path}[{index}]")
            if sub is not None:
                return sub
        return None
    if expected is _MISSING or got is _MISSING:
        want = "<absent>" if expected is _MISSING else repr(expected)
        actual = "<absent>" if got is _MISSING else repr(got)
        return f"{path}: expected {want}, got {actual}"
    if isinstance(expected, bool) != isinstance(got, bool) or expected != got:
        return f"{path}: expected {expected!r}, got {got!r}"
    return None


# --- the two authorities -----------------------------------------------------


def python_shape_fn_results(case: dict) -> tuple[Any, ...]:
    """The live Python authority's answer for one case (`shape_fn` call)."""
    from evoxc.ir.op_defs import opdef

    op_def = opdef(case["op"])
    operand_types = tuple(value_type_from_json(t) for t in case["operands"])
    return op_def.shape_fn(
        operand_types, effective_attrs(op_def, case.get("attrs") or {})
    )


def python_shape_fn_family(case: dict) -> str | None:
    """The family the bare `shape_fn` raises for one case (the fixture's layer).

    `generate.py` called `shape_fn` directly, so this is what a fixture record
    must still agree with — it is the staleness control for a recorded family.
    """
    try:
        python_shape_fn_results(case)
    except BaseException as exc:  # noqa: BLE001 - the family IS the observation
        return type(exc).__name__
    return None


def python_pipeline_family(case: dict) -> str | None:
    """The family the Python authority's own pipeline raises for one case.

    `Builder.create` runs the same checks the native `infer` runs (op lookup,
    arity, region count, attribute schema, then the op's `shape_fn`), so its
    answer is the layer `evoxc._evoxc_native.infer` must match.  Returns the
    exception class name, or `None` when the authority accepts the case.
    """
    from evoxc.ir.builder import Builder
    from evoxc.ir.op_defs import opdef

    op_def = opdef(case["op"])
    operand_types = tuple(value_type_from_json(t) for t in case["operands"])
    builder = Builder()
    builder.build_module("parity")
    builder.build_function("main", operand_types)
    arguments = tuple(builder.current_block.arguments)
    attrs = {
        key: decode_attr(attr_spec(op_def, key).type, value)
        for key, value in (case.get("attrs") or {}).items()
    }
    try:
        builder.create(case["op"], arguments, attrs)
    except BaseException as exc:  # noqa: BLE001 - the family IS the observation
        return type(exc).__name__
    return None


# --- the harness -------------------------------------------------------------


class Report:
    """Counters for one run."""

    def __init__(self) -> None:
        self.cases = 0
        self.passed = 0
        self.mismatched = 0
        self.skipped = 0
        self.cross_checked = 0
        self.cross_check_skipped = 0
        self.mismatch_by_op: collections.Counter[str] = collections.Counter()
        self.mismatches: list[str] = []
        self.divergences: list[str] = []

    def fail(self, op: str, message: str) -> None:
        self.mismatched += 1
        self.mismatch_by_op[op] += 1
        self.mismatches.append(message)


def run_case(native: Any, case: dict) -> dict:
    """Call the extension for one case; return `{"ok", "results"|"kind"/"message"}`."""
    request = {
        "op_name": case["op"],
        "operand_types": case["operands"],
        "attributes": case.get("attrs") or {},
        "result_types": case.get("result_types"),
        "module": case.get("module"),
    }
    try:
        results = native.infer(
            request["op_name"],
            request["operand_types"],
            request["attributes"],
            result_types=request["result_types"],
            module=request["module"],
        )
    except native.NativeError as err:
        return {"ok": False, "kind": err.kind, "message": str(err)}
    except Exception as err:  # noqa: BLE001 - a non-NativeError is a binding bug
        return {
            "ok": False,
            "kind": f"<{type(err).__name__}>",
            "message": str(err),
            "not_native_error": True,
        }
    return {"ok": True, "results": [dict(result) for result in results]}


def check_case(native: Any, case: dict, expected: dict, report: Report) -> None:
    """Compare one case against the fixture (and, for ok cases, the authority)."""
    key = case["key"]
    op = case["op"]
    report.cases += 1

    expected_ok = expected.get("ok")
    if expected_ok is not True and expected_ok is not False:
        report.fail(op, f'{key} ({op}): fixture entry has no boolean "ok"')
        return
    if case.get("ok") != expected_ok:
        report.fail(op, f'{key} ({op}): cases.json and expected.json disagree on "ok"')
        return

    outcome = run_case(native, case)

    if "not_native_error" in outcome:
        report.fail(
            op,
            f"{key} ({op}): extension raised {outcome['kind']} instead of "
            f"NativeError: {outcome['message']}",
        )
        return

    if expected_ok:
        if not outcome["ok"]:
            report.fail(
                op,
                f"{key} ({op}): the authority returns results, but native inference "
                f"failed with {outcome['kind']}: {outcome['message']}",
            )
            return
        difference = first_difference(expected["results"], outcome["results"])
        if difference is not None:
            report.fail(op, f"{key} ({op}): result type mismatch: {difference}")
            return
        report.passed += 1
        check_live_authority(case, outcome["results"], report)
        return

    # --- rejection case ------------------------------------------------------
    if outcome["ok"]:
        report.fail(
            op,
            f"{key} ({op}): the authority raises {expected.get('error_kind')}, but "
            f"native inference returned {len(outcome['results'])} result type(s): "
            f"{json.dumps(outcome['results'])}",
        )
        return

    want_family = expected.get("error_kind")
    got_family = outcome["kind"]
    if got_family == want_family:
        report.passed += 1
        return

    # The families differ: decide whether this is a genuine divergence between
    # the extension and the Python authority, or a fixture record at a deeper
    # layer than the native `infer` (which validates before calling `shape_fn`).
    # Both classifications also require the fixture record to still match the
    # live authority at its own layer, so a stale record never passes as one.
    pipeline_family = python_pipeline_family(case)
    shape_fn_family = python_shape_fn_family(case)
    if pipeline_family == got_family and shape_fn_family == want_family:
        report.passed += 1
        report.divergences.append(
            f"{key} ({op}): fixture records {want_family} from the op's shape_fn; "
            f"the authority's own pipeline and native infer both raise "
            f"{got_family} first (schema-level validation) — same contract layer"
        )
        return
    if (
        want_family not in EVOXC_FAMILIES
        and shape_fn_family == want_family
        and expected.get("error") == outcome["message"]
    ):
        report.passed += 1
        report.divergences.append(
            f"{key} ({op}): fixture records {want_family} (not an EVOXC error "
            f"family, so no native kind can express it); native raises "
            f"{got_family} with the identical message — message pinned"
        )
        return
    report.fail(
        op,
        f"{key} ({op}): expected {want_family} {expected.get('error')!r}, got "
        f"{got_family} {outcome['message']!r}; the live authority's pipeline raises "
        f"{pipeline_family} and its shape_fn raises {shape_fn_family} — fix the "
        f"native side, or regenerate the fixture if the authority changed",
    )


# --- the `infer` module argument --------------------------------------------
#
# `infer` accepts the `module` argument in two shapes: a full `evoxc-native-ir`
# envelope (the fixture path above) and a signature-only context that carries
# just the callee signatures `call` resolves against.  These cases exercise the
# second shape (and the argument-shape rejections) directly through the
# extension; they need no `evoxc` import, so they always run.
SIGNATURE_ONLY_F32 = {"dtype": "float32", "shape": [{"int": 4}, {"int": 3}]}
SIGNATURE_ONLY_I64 = {"dtype": "int64", "shape": []}
SIGNATURE_ONLY_ENVELOPE = {
    "format": "evoxc-native-ir",
    "version": 1,
    "module": {"name": "m"},
    "functions": [
        {
            "name": "callee",
            "input_types": [SIGNATURE_ONLY_F32],
            "output_types": [SIGNATURE_ONLY_F32],
            "block": {
                "arguments": [{"id": 0, "type": SIGNATURE_ONLY_F32}],
                "ops": [{"id": 0, "ref": "0"}],
            },
        }
    ],
    "ops": [{"id": 0, "name": "return", "operands": [0], "results": []}],
}


def check_module_contexts(native: Any, report: Report) -> None:
    """Check the signature-only `infer` module contexts and their rejections."""

    def run(module: Any, callee: str = "callee") -> dict:
        try:
            return {
                "ok": True,
                "results": native.infer(
                    "call", [], {"callee": callee}, module=module
                ),
            }
        except native.NativeError as err:
            return {"ok": False, "kind": err.kind, "message": str(err)}
        except Exception as err:  # noqa: BLE001 - classified as the finding below
            return {"ok": False, "kind": f"<{type(err).__name__}>", "message": str(err)}

    # (label, module, expected results | (family, message substring) | TypeError)
    cases: list[tuple[str, Any, Any]] = [
        (
            "module#signature-2",
            {"functions": [["callee", [SIGNATURE_ONLY_F32]]]},
            [SIGNATURE_ONLY_F32],
        ),
        (
            "module#signature-3",
            {"functions": [["callee", [SIGNATURE_ONLY_F32], [SIGNATURE_ONLY_I64]]]},
            [SIGNATURE_ONLY_I64],
        ),
        (
            "module#signature-dict",
            {"functions": [{"name": "callee", "output_types": [SIGNATURE_ONLY_I64]}]},
            [SIGNATURE_ONLY_I64],
        ),
        ("module#envelope", SIGNATURE_ONLY_ENVELOPE, [SIGNATURE_ONLY_F32]),
        (
            "module#unknown-callee",
            {"functions": [["callee", [SIGNATURE_ONLY_F32]]]},
            ("VerificationError", "no function named 'nope'"),
        ),
        ("module#missing-functions", {"callees": []}, TypeError),
        ("module#non-dict", [["callee", []]], TypeError),
        ("module#functions-not-a-list", {"functions": "nope"}, TypeError),
        ("module#wrong-entry-length", {"functions": [["callee"]]}, TypeError),
        (
            "module#unknown-entry-key",
            {"functions": [{"name": "callee", "output_types": [], "extra": 1}]},
            TypeError,
        ),
        (
            "module#bad-type",
            {"functions": [["callee", [{"dtype": "nope", "shape": []}]]]},
            ("DTypeError", "module.functions[0] types"),
        ),
    ]
    for label, module, expected in cases:
        report.cases += 1
        callee = "nope" if "unknown-callee" in label else "callee"
        outcome = run(module, callee)
        if expected is TypeError:
            if outcome["ok"]:
                report.fail("call", f"{label}: expected a TypeError, got results")
            elif outcome["kind"] != "<TypeError>":
                report.fail(
                    "call",
                    f"{label}: expected a TypeError, got {outcome['kind']}: "
                    f"{outcome['message']}",
                )
            else:
                report.passed += 1
            continue
        if isinstance(expected, tuple):
            family, needle = expected
            if outcome["ok"]:
                report.fail("call", f"{label}: expected {family}, got results")
            elif outcome["kind"] != family or needle not in outcome["message"]:
                report.fail(
                    "call",
                    f"{label}: expected {family} containing {needle!r}, got "
                    f"{outcome['kind']}: {outcome['message']}",
                )
            else:
                report.passed += 1
            continue
        if not outcome["ok"]:
            report.fail(
                "call",
                f"{label}: expected {json.dumps(expected)}, got {outcome['kind']}: "
                f"{outcome['message']}",
            )
            continue
        difference = first_difference(expected, outcome["results"])
        if difference is not None:
            report.fail("call", f"{label}: result type mismatch: {difference}")
        else:
            report.passed += 1


def check_live_authority(case: dict, native_results: list, report: Report) -> None:
    """Re-derive one successful case from the live Python authority."""
    op = case["op"]
    if case.get("result_types") is not None or case.get("module") is not None:
        # `shape_fn` cannot see explicit result types or a module envelope, and
        # an op with neither a shape hook nor a derivable answer is not part of
        # the fixture.  Nothing to re-derive.
        report.cross_check_skipped += 1
        return
    try:
        derived = [
            value_type_to_json(value_type)
            for value_type in python_shape_fn_results(case)
        ]
    except Exception as err:  # noqa: BLE001 - a raise here IS the finding
        report.cross_check_skipped += 1
        report.fail(
            op,
            f"{case['key']} ({op}): the live authority's shape_fn raised "
            f"{type(err).__name__}: {err} while native inference succeeded — the "
            f"fixture is stale, re-run native/evoxc-core/tests/parity/generate.py",
        )
        return
    report.cross_checked += 1
    difference = first_difference(derived, native_results, path="live-authority")
    if difference is not None:
        report.fail(
            op,
            f"{case['key']} ({op}): native inference disagrees with the live Python "
            f"authority: {difference} (fixture is stale or the port diverged)",
        )


def load_fixture(directory: pathlib.Path) -> tuple[dict, dict]:
    cases_path = directory / "cases.json"
    expected_path = directory / "expected.json"
    for path in (cases_path, expected_path):
        if not path.exists():
            raise SystemExit(f"missing parity fixture {path}")
    cases_doc = json.loads(cases_path.read_text(encoding="utf-8"))
    expected_doc = json.loads(expected_path.read_text(encoding="utf-8"))
    return cases_doc, expected_doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="exit 0 instead of 2 when the native extension is not installed",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print every case as it is checked"
    )
    parser.add_argument(
        "--fixture-dir",
        type=pathlib.Path,
        default=FIXTURE_DIR,
        help=f"directory holding cases.json + expected.json (default {FIXTURE_DIR})",
    )
    parser.add_argument(
        "--min-cross-check",
        type=int,
        default=MIN_LIVE_CROSS_CHECK,
        help="minimum successful cases re-derived from the live authority "
        f"(default {MIN_LIVE_CROSS_CHECK})",
    )
    args = parser.parse_args(argv)

    native = load_native()
    if native is None:
        return 0 if args.allow_missing else 2

    cases_doc, expected_doc = load_fixture(args.fixture_dir)
    cases = cases_doc.get("cases") or []
    skipped_ops = cases_doc.get("skipped") or []

    print(f"extension: {native.__file__} (version {native.version()})")
    fixture_label = (
        args.fixture_dir.relative_to(REPO_ROOT)
        if args.fixture_dir.is_relative_to(REPO_ROOT)
        else args.fixture_dir
    )
    print(
        f"fixture:   {fixture_label} "
        f"({len(cases)} cases, {len({case['op'] for case in cases})} ops, "
        f"{len(skipped_ops)} fixture-recorded skips)"
    )
    print(f"authority: {cases_doc.get('authority')}")
    print()

    report = Report()
    for case in cases:
        key = case["key"]
        expected = expected_doc.get(key)
        if expected is None:
            report.cases += 1
            report.fail(case["op"], f"{key} ({case['op']}): no entry in expected.json")
            continue
        before = report.mismatched
        check_case(native, case, expected, report)
        if args.verbose:
            status = "ok" if report.mismatched == before else "MISMATCH"
            print(f"{status:8} {key:24} {case['op']}")

    seen = {case["key"] for case in cases}
    for key in expected_doc:
        if key not in seen:
            report.cases += 1
            report.fail("<fixture>", f"{key}: expected.json entry has no case")

    for entry in skipped_ops:
        report.skipped += 1
        print(f"skipped (fixture): {entry.get('op')}: {entry.get('reason')}")
    check_module_contexts(native, report)

    # --- summary -------------------------------------------------------------
    print()
    print(f"cases           {report.cases}")
    print(f"passed          {report.passed}")
    print(f"mismatches      {report.mismatched}")
    print(
        f"live checks     {report.cross_checked} successful cases re-derived from "
        f"evoxc.ir.op_defs.opdef(op).shape_fn "
        f"({report.cross_check_skipped} not re-derivable)"
    )
    print(f"documented      {len(report.divergences)} layered family divergences")
    print(f"skipped         {report.skipped}")

    if report.divergences:
        print()
        print("documented divergences (not failures):")
        for line in report.divergences:
            print(f"  - {line}")

    if report.mismatches:
        print()
        print("mismatches by op:")
        for op, count in sorted(report.mismatch_by_op.items()):
            print(f"  {op}: {count}")
        print()
        print("mismatch detail (fix the native side; do not weaken the fixture):")
        for line in report.mismatches:
            print(f"  - {line}")
        return 1

    if report.cross_checked < args.min_cross_check:
        print()
        print(
            f"FAIL: only {report.cross_checked} successful cases were re-derived from "
            f"the live authority; at least {args.min_cross_check} are required"
        )
        return 1

    print()
    print("ALL CASES MATCH THE PYTHON AUTHORITY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
