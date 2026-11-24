"""Pins the engine-authoritative half of the `batch_functions` batchability gate.

`batch_functions._check_ops` accepts an op when `batching.batching_rules` holds
its key OR the native engine's compiled-in builtin set does
(`evoxc._native.batching_rule_keys()`). These tests pin both halves and their
relationship:

* the native query is a sorted, duplicate-free list of builtin rule keys with
  no host-namespace key (`block:*` / `external:*` / `sparse_*` come from the
  extension-rule bridge, never from the builtin registry);
* the known v1 gaps (`sort`, `random_uniform`, `runtime_call`, …) are absent,
  so the standard "no batching rule" error still fires for them;
* the Python mirror covers every native key — a builtin rule can therefore
  never be gated out by a stale mirror;
* the gate really accepts a native-only key: deleting the Python entry and
  keeping the native key still lets `_check_ops` pass (the OR path).
"""
from __future__ import annotations
import importlib
import pytest
import evoxc
from evoxc import _native, core
from evoxc.transforms import batching

# The package re-exports the public `batch_functions` FUNCTION under the same
# name, so the module itself is reached through the import system.
batch_functions = importlib.import_module("evoxc.transforms.batch_functions")
@pytest.fixture(scope="module")
def native_keys() -> tuple[str, ...]:
    _native.require()
    return tuple(_native.batching_rule_keys())
def test_native_keys_are_sorted_and_unique(native_keys):
    assert native_keys
    assert list(native_keys) == sorted(native_keys)
    assert len(set(native_keys)) == len(native_keys)
def test_native_keys_cover_representative_builtins(native_keys):
    keys = set(native_keys)
    for name in ("add", "multiply", "dot", "reduce_sum", "reshape", "gather"):
        assert name in keys, name
def test_native_keys_exclude_host_namespaces(native_keys):
    for key in native_keys:
        assert not key.startswith("block:"), key
        assert not key.startswith("external:"), key
        assert not key.startswith("sparse_"), key
        assert key != "dense_dot_sparse", key
def test_native_keys_exclude_v1_gaps(native_keys):
    keys = set(native_keys)
    for name in (
        "sort",
        "argsort",
        "tile",
        "flip",
        "roll",
        "diag",
        "cumprod",
        "nan_to_num",
        "random_uniform",
        "random_normal",
        "runtime_call",
        "cond",
        "while_loop",
        "scan",
    ):
        assert name not in keys, name
def test_python_mirror_covers_every_native_key(native_keys):
    """The registry mirror is a superset: no native rule is gated out."""
    missing = sorted(set(native_keys) - set(batching.batching_rules))
    assert not missing, missing
def test_gate_accepts_a_native_only_key(monkeypatch):
    """A key the engine knows but the mirror lacks still passes `_check_ops`."""

    @evoxc.defn
    def f(x):
        return x + x

    graph = evoxc.trace(f, evoxc.TensorSpec((4,), "float32"))
    assert any(op.name == "add" for op in graph.module.functions[0].entry_block.ops)
    monkeypatch.delitem(batching.batching_rules, "add")
    monkeypatch.setattr(batch_functions, "_NATIVE_BATCHABLE", frozenset({"add"}))
    batch_functions._check_ops([graph])  # the OR path: no TransformError
def test_gate_rejects_a_key_in_neither_table(monkeypatch):
    """With the mirror entry gone and the native set empty, the gate names it."""

    @evoxc.defn
    def f(x):
        return x + x

    graph = evoxc.trace(f, evoxc.TensorSpec((4,), "float32"))
    monkeypatch.delitem(batching.batching_rules, "add")
    monkeypatch.setattr(batch_functions, "_NATIVE_BATCHABLE", frozenset())
    with pytest.raises(core.TransformError, match="no batching rule registered for key 'add'"):
        batch_functions._check_ops([graph])
def test_gate_keeps_the_python_mirror_alone(monkeypatch):
    """The untouched mirror passes the gate with an empty native set."""

    @evoxc.defn
    def f(x):
        return x * x

    graph = evoxc.trace(f, evoxc.TensorSpec((4,), "float32"))
    monkeypatch.setattr(batch_functions, "_NATIVE_BATCHABLE", frozenset())
    batch_functions._check_ops([graph])
def test_native_batchable_is_the_engine_set():
    keys = batch_functions._native_batchable()
    assert keys == frozenset(_native.batching_rule_keys())
