"""Pins for the optional native extension loader and typed façade.

No test here requires the Rust build: the absent path is exercised for real and
the present path is exercised with an injected fake module via the documented
``_MODULE`` / ``_reset`` seams.
"""

from __future__ import annotations

import importlib
import types

import numpy as np
import pytest

from evoxc import core, ir
from evoxc import _native


class FakeNativeError(Exception):
    """Stand-in for the extension's ``NativeError`` (has a ``kind``)."""

    def __init__(self, kind, message):
        self.kind = kind
        super().__init__(message)


def _simple_module() -> ir.Module:
    m = ir.Module(name="simple")
    b = ir.Builder(m)
    f = b.build_function("main", (ir.ValueType(np.float32, (4,)),))
    x = f.entry_block.arguments[0]
    out = b.emit("add", (x, x))
    b.set_terminator(f.entry_block, "return", (out,))
    return m


def _fake_module(**overrides):
    module = types.ModuleType("evoxc._evoxc_native")
    module.NativeError = FakeNativeError
    module.version = lambda: "9.9.9"
    module.infer = lambda *args: []
    module.transform = lambda *args: {}
    for name, value in overrides.items():
        setattr(module, name, value)
    return module


@pytest.fixture(autouse=True)
def _clean_loader_state():
    _native._reset()
    yield
    _native._reset()


def test_absent_extension_is_available_false_and_stable(monkeypatch):
    monkeypatch.delenv(_native.NATIVE_LIB_ENV, raising=False)
    calls = []

    def discover():
        calls.append(1)
        return []

    monkeypatch.setattr(_native, "_discover_paths", discover)
    assert _native.available() is False
    assert _native.available() is False
    assert len(calls) == 1


def test_absent_extension_require_names_the_remedy(monkeypatch):
    monkeypatch.delenv(_native.NATIVE_LIB_ENV, raising=False)
    monkeypatch.setattr(_native, "_discover_paths", lambda: [])
    with pytest.raises(ImportError) as excinfo:
        _native.require()
    message = str(excinfo.value)
    assert "scripts/build-native.sh" in message
    assert "_evoxc_native.abi3.so" in message
    assert _native.NATIVE_LIB_ENV in message
    assert excinfo.value.__cause__ is not None


def test_absent_extension_version_raises(monkeypatch):
    monkeypatch.delenv(_native.NATIVE_LIB_ENV, raising=False)
    monkeypatch.setattr(_native, "_discover_paths", lambda: [])
    with pytest.raises(ImportError, match="build-native.sh"):
        _native.version()


def test_env_override_missing_path_is_actionable(monkeypatch, tmp_path):
    missing = tmp_path / "missing_native.so"
    monkeypatch.setenv(_native.NATIVE_LIB_ENV, str(missing))
    with pytest.raises(ImportError) as excinfo:
        _native.require()
    message = str(excinfo.value)
    assert _native.NATIVE_LIB_ENV in message
    assert str(missing) in message


def test_env_override_broken_file_is_actionable(monkeypatch, tmp_path):
    broken = tmp_path / "broken_native.so"
    broken.write_bytes(b"not a shared object")
    monkeypatch.setenv(_native.NATIVE_LIB_ENV, str(broken))
    assert _native.available() is False
    with pytest.raises(ImportError) as excinfo:
        _native.require()
    assert str(broken) in str(excinfo.value) or "native extension" in str(
        excinfo.value
    )


def test_present_extension_reports_version_and_stable_availability(monkeypatch):
    _native._MODULE = _fake_module()
    _native._LOAD_ERROR = None
    assert _native.available() is True
    assert _native.version() == "9.9.9"
    monkeypatch.setattr(
        _native, "_discover_paths", lambda: (_ for _ in ()).throw(AssertionError)
    )
    assert _native.available() is True


def test_infer_encodes_and_decodes_types():
    recorded = {}

    def fake_infer(op_name, operand_types, attributes, result_types, module):
        recorded.update(
            op_name=op_name,
            operand_types=operand_types,
            attributes=attributes,
            result_types=result_types,
            module=module,
        )
        return [{"dtype": "float64", "shape": [{"int": 4}]}]

    _native._MODULE = _fake_module(infer=fake_infer)
    spec = ir.ValueType(np.float32, (2, 3))
    context = {"functions": [("main", [spec])], "tag": 7}
    attributes = {"alpha": (1, 2), "flag": True}
    result = _native.infer("add", [spec], attributes, module=context)

    assert recorded["op_name"] == "add"
    assert recorded["operand_types"] == [
        {"dtype": "float32", "shape": [{"int": 2}, {"int": 3}]}
    ]
    assert recorded["attributes"] is attributes
    assert recorded["result_types"] is None
    # An already-encoded envelope mapping passes through VERBATIM (a shallow
    # copy: the façade never re-encodes a mapping the caller built).
    assert recorded["module"] == context
    assert recorded["module"] is not context
    assert result == [ir.ValueType(np.dtype("float64"), (4,))]


def test_infer_forwards_explicit_result_types():
    recorded = {}

    def fake_infer(op_name, operand_types, attributes, result_types, module):
        recorded["result_types"] = result_types
        return list(result_types)

    _native._MODULE = _fake_module(infer=fake_infer)
    rt = ir.ValueType(np.int32, (ir.ValueType(np.float32, ()).shape))
    out = _native.infer("external_call", [], {}, result_types=[rt])
    assert recorded["result_types"] == [{"dtype": "int32", "shape": []}]
    assert out[0].dtype == np.dtype("int32")


def test_transform_round_trips_module_and_preserves_extra_keys():
    sentinel = object()
    seen = {}

    def fake_transform(kind, envelope, config):
        seen["kind"] = kind
        seen["envelope"] = envelope
        seen["config"] = config
        return {
            # The extension returns the transformed envelope FLAT (its own
            # ``module`` entry is the module-info object), plus the metadata.
            **envelope,
            "transform": {
                "kind": kind,
                "output_ids": [1],
                "input_ids": [0],
                "batch_dim": "batch",
            },
            "output_types": [{"dtype": "float32", "shape": [{"int": 4}]}],
            "report": sentinel,
        }

    _native._MODULE = _fake_module(transform=fake_transform)
    module = _simple_module()
    rule = lambda *args: None
    config = {"axes": 0, "extension_rule": rule}
    result = _native.transform("vectorize", module, config)

    assert seen["kind"] == "vectorize"
    assert seen["config"] is config
    assert seen["config"]["extension_rule"] is rule
    assert isinstance(result["module"], ir.Module)
    ir.verify(result["module"])
    assert [value.id for value in result["module"].main.entry_block.ops[1].results] == [
        value.id for value in module.main.entry_block.ops[1].results
    ]
    assert result["transform"] == {
        "kind": "vectorize",
        "output_ids": [1],
        "input_ids": [0],
        "batch_dim": "batch",
    }
    # Any non-envelope top-level key is preserved VERBATIM.
    assert result["output_types"] == [{"dtype": "float32", "shape": [{"int": 4}]}]
    assert result["report"] is sentinel


@pytest.mark.parametrize(
    "kind, error_cls",
    [
        ("ShapeError", core.ShapeError),
        ("VerificationError", core.VerificationError),
        ("DTypeError", core.DTypeError),
        ("TransformError", core.TransformError),
        ("BackendError", core.BackendError),
        ("DeviceError", core.DeviceError),
        ("TraceError", core.TraceError),
    ],
)
def test_native_error_kind_mapping_via_infer(kind, error_cls):
    message = f"native failure for {kind}"

    def fake_infer(*args):
        raise FakeNativeError(kind, message)

    _native._MODULE = _fake_module(infer=fake_infer)
    with pytest.raises(error_cls) as excinfo:
        _native.infer("add", [], {})
    assert str(excinfo.value) == message
    assert isinstance(excinfo.value.__cause__, FakeNativeError)


def test_native_error_kind_mapping_via_transform():
    message = "transform exploded"

    def fake_transform(kind, envelope, config):
        raise FakeNativeError("TransformError", message)

    _native._MODULE = _fake_module(transform=fake_transform)
    with pytest.raises(core.TransformError) as excinfo:
        _native.transform("vmap", _simple_module(), {})
    assert str(excinfo.value) == message


def test_unknown_native_error_kind_is_reraised_unchanged():
    def fake_infer(*args):
        raise FakeNativeError("MysteryError", "keep me")

    _native._MODULE = _fake_module(infer=fake_infer)
    with pytest.raises(FakeNativeError) as excinfo:
        _native.infer("add", [], {})
    assert excinfo.value.kind == "MysteryError"
    assert str(excinfo.value) == "keep me"


def test_non_native_error_is_reraised_unchanged():
    def fake_infer(*args):
        raise ValueError("host bug")

    _native._MODULE = _fake_module(infer=fake_infer)
    with pytest.raises(ValueError, match="host bug"):
        _native.infer("add", [], {})


def test_import_evoxc_and_loader_do_not_require_the_extension(monkeypatch):
    monkeypatch.delenv(_native.NATIVE_LIB_ENV, raising=False)
    importlib.import_module("evoxc")
    module = importlib.import_module("evoxc._native")
    assert module.EXTENSION_MODULE_NAME == "evoxc._evoxc_native"
    assert module.NATIVE_LIB_ENV == "EVOXC_NATIVE_LIB"
