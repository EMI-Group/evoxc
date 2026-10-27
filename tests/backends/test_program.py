"""LoweredProgram + Signature contract tests (numpy backend).

Covers the staging objects owned by ``evoxc.backends`` (see
``evoxc/backends/program.py`` and ``evoxc/backends/CONTEXT.md``):

* ``Signature``: frozen, field layout matching ``Graph.signature_info()``.
* numpy ``LoweredProgram``: backend name, self-describing serialized-IR
  payload, ``text()``, ``save()``/``load()`` round-trip via the persist
  container (full corruption/mismatch coverage lives in
  ``test_artifact_persistence.py``).
* Cross-stage signature equality: graph -> lower -> compile.

CPU only; shapes kept tiny.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pytest

import evoxc
from evoxc import persist
from evoxc.backends import LoweredProgram, Signature

#: Keys of ``Graph.signature_info()`` == ``Signature`` field names.
_SIGNATURE_KEYS = (
    "input_tree",
    "output_tree",
    "input_specs",
    "output_specs",
    "static_values",
    "output_static_values",
)


def _double(x):
    """x -> x * 2 (the canonical tiny program for these tests)."""
    return evoxc.multiply(x, 2)


def _simple_graph():
    return evoxc.trace(_double, evoxc.TensorSpec((3,), evoxc.float32))


def _decoded_info(graph):
    """``graph.signature_info()`` with every encoded value decoded back."""
    info = graph.signature_info()
    assert set(info) == set(_SIGNATURE_KEYS)
    return {key: persist.decode_value(value) for key, value in info.items()}


# ---------------------------------------------------------------------------
# 1. Simple program: lower via the numpy backend
# ---------------------------------------------------------------------------


def test_lower_simple_program_numpy_backend():
    graph = _simple_graph()
    lowered = evoxc.lower(graph, backend=evoxc.numpy_backend)

    assert isinstance(lowered, LoweredProgram)
    assert lowered.backend == "numpy"
    assert isinstance(lowered.signature, Signature)
    # Payload: self-describing serialized ir.Module (version + module keys).
    assert isinstance(lowered.payload, dict)
    assert "version" in lowered.payload
    assert "module" in lowered.payload
    # text(): pretty-printed deserialized module, non-empty, names the function.
    text = lowered.text()
    assert isinstance(text, str)
    assert text.strip()
    assert "main" in text


def test_lower_with_backend_name_string():
    graph = _simple_graph()
    lowered = evoxc.lower(graph, backend="numpy")
    assert lowered.backend == "numpy"
    assert lowered.text()


# ---------------------------------------------------------------------------
# 2. Signature structure
# ---------------------------------------------------------------------------


def test_signature_tree_and_spec_structure():
    lowered = evoxc.lower(_simple_graph())
    sig = lowered.signature

    assert isinstance(sig.input_tree, evoxc.core.TreeSpec)
    assert isinstance(sig.output_tree, evoxc.core.TreeSpec)
    # Leaf counts are coherent across the signature.
    assert sig.input_tree.num_leaves == len(sig.input_specs)
    assert sig.output_tree.num_leaves == len(sig.output_specs)
    # Each input leaf carries the traced shape/dtype.
    assert len(sig.input_specs) == 1
    in_spec = sig.input_specs[0]
    assert isinstance(in_spec, evoxc.core.TensorSpec)
    assert in_spec.shape == (3,)
    assert in_spec.dtype == evoxc.float32
    # The output spec mirrors the traced result type.
    assert len(sig.output_specs) == 1
    out_spec = sig.output_specs[0]
    assert isinstance(out_spec, evoxc.core.TensorSpec)
    assert out_spec.shape == (3,)
    assert out_spec.dtype == evoxc.float32


def test_signature_is_a_frozen_dataclass():
    assert dataclasses.is_dataclass(Signature)
    sig = Signature()
    assert sig.output_static_values == ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        sig.input_specs = ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        sig.static_values = (1.0,)


# ---------------------------------------------------------------------------
# 3. Structured inputs / outputs and static values
# ---------------------------------------------------------------------------


def test_dict_input_and_output_structure():
    def fn(xs):
        return {"sum": evoxc.add(xs["a"], xs["b"])}

    graph = evoxc.trace(
        fn,
        {
            "a": evoxc.TensorSpec((2,), evoxc.float32),
            "b": evoxc.TensorSpec((2,), evoxc.float32),
        },
    )
    lowered = evoxc.lower(graph)
    sig = lowered.signature

    # Input tree = args tuple -> the single dict argument: dict node with
    # sorted keys and two tensor leaves.
    in_tree = sig.input_tree
    assert in_tree.type is tuple
    assert len(in_tree.children) == 1
    arg_tree = in_tree.children[0]
    assert arg_tree.type is dict
    assert arg_tree.node_data == ["a", "b"]
    assert len(arg_tree.children) == 2
    assert in_tree.num_leaves == 2
    assert len(sig.input_specs) == 2
    # Output tree encodes the returned dict structure.
    out_tree = sig.output_tree
    assert out_tree.type is dict
    assert out_tree.node_data == ["sum"]
    assert out_tree.num_leaves == 1
    assert len(sig.output_specs) == 1

    exe = evoxc.load(evoxc.compile(lowered))
    out = evoxc.run(
        exe,
        {
            "a": evoxc.tensor(np.array([1.0, 2.0], dtype=np.float32)),
            "b": evoxc.tensor(np.array([3.0, 4.0], dtype=np.float32)),
        },
    )
    assert isinstance(out, dict)
    assert set(out) == {"sum"}
    np.testing.assert_allclose(out["sum"].numpy(), [4.0, 6.0])


@dataclass
class Pair:
    """Structured input/output container for trace tests."""

    left: object
    right: object


def test_dataclass_input_and_output_structure():
    def fn(xs):
        return Pair(evoxc.add(xs.left, xs.right), evoxc.multiply(xs.left, xs.right))

    exe = evoxc.build(
        fn,
        Pair(evoxc.TensorSpec((2,), evoxc.float32), evoxc.TensorSpec((2,), evoxc.float32)),
    )
    sig = exe.signature

    in_tree = sig.input_tree
    assert in_tree.type is tuple
    assert len(in_tree.children) == 1
    arg_tree = in_tree.children[0]
    assert arg_tree.type is Pair
    assert arg_tree.node_data == ["left", "right"]
    assert in_tree.num_leaves == 2
    assert len(sig.input_specs) == 2
    out_tree = sig.output_tree
    assert out_tree.type is Pair
    assert out_tree.node_data == ["left", "right"]
    assert out_tree.num_leaves == 2
    assert len(sig.output_specs) == 2

    out = evoxc.run(
        exe,
        Pair(
            evoxc.tensor(np.array([1.0, 2.0], dtype=np.float32)),
            evoxc.tensor(np.array([3.0, 4.0], dtype=np.float32)),
        ),
    )
    assert isinstance(out, Pair)
    np.testing.assert_allclose(out.left.numpy(), [4.0, 6.0])
    np.testing.assert_allclose(out.right.numpy(), [3.0, 8.0])


def test_static_input_values_recorded_pre_order():
    def fn(x, scale):
        return evoxc.multiply(x, scale)

    graph = evoxc.trace(fn, evoxc.TensorSpec((3,), evoxc.float32), 2.5)
    lowered = evoxc.lower(graph)
    sig = lowered.signature

    assert sig.static_values == (2.5,)
    # All outputs are tensors -> no recorded static outputs (the default).
    assert sig.output_static_values == ()
    assert len(sig.input_specs) == 1
    assert sig.input_tree.num_leaves == 2  # one tensor leaf + one static leaf

    # The static value travels with the run-time arguments and specializes.
    exe = evoxc.load(evoxc.compile(lowered))
    out = evoxc.run(exe, evoxc.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32)), 2.5)
    np.testing.assert_allclose(out.numpy(), [2.5, 5.0, 7.5])


def test_static_output_values_recorded_pre_order():
    def fn(x):
        return (evoxc.multiply(x, 2), "done")

    graph = evoxc.trace(fn, evoxc.TensorSpec((3,), evoxc.float32))
    lowered = evoxc.lower(graph)
    sig = lowered.signature

    assert sig.output_static_values == ("done",)
    assert sig.output_tree.num_leaves == 2
    assert len(sig.output_specs) == 1

    exe = evoxc.load(evoxc.compile(lowered))
    out = evoxc.run(exe, evoxc.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32)))
    assert isinstance(out, tuple) and len(out) == 2
    assert out[1] == "done"
    np.testing.assert_allclose(out[0].numpy(), [2.0, 4.0, 6.0])


# ---------------------------------------------------------------------------
# 4. Signature equality across the staging chain
# ---------------------------------------------------------------------------


def test_lowered_signature_matches_graph_signature_info():
    graph = _simple_graph()
    lowered = evoxc.lower(graph)
    info = _decoded_info(graph)

    assert lowered.signature.input_tree == info["input_tree"]
    assert lowered.signature.output_tree == info["output_tree"]
    assert lowered.signature.input_specs == info["input_specs"]
    assert lowered.signature.output_specs == info["output_specs"]
    assert lowered.signature.static_values == info["static_values"]
    assert lowered.signature.output_static_values == info["output_static_values"]


@pytest.mark.parametrize("structured", [False, True])
def test_artifact_signature_matches_lowered_signature(structured):
    if structured:
        graph = evoxc.trace(
            lambda xs: {"d": evoxc.multiply(xs["a"], 2)},
            {"a": evoxc.TensorSpec((3,), evoxc.float32)},
        )
    else:
        graph = _simple_graph()
    lowered = evoxc.lower(graph)
    artifact = evoxc.compile(lowered)
    assert isinstance(artifact.signature, Signature)
    for field in _SIGNATURE_KEYS:
        assert getattr(artifact.signature, field) == getattr(lowered.signature, field)
    # Field-for-field equality of the whole signature objects.
    assert artifact.signature == lowered.signature


# ---------------------------------------------------------------------------
# 5. LoweredProgram save/load round-trip
# ---------------------------------------------------------------------------


def test_lowered_program_save_load_roundtrip(tmp_path):
    lowered = evoxc.lower(_simple_graph())
    path = tmp_path / "lowered_program.evoxcbin"

    lowered.save(path)
    assert path.exists()
    loaded = LoweredProgram.load(path)

    assert isinstance(loaded, LoweredProgram)
    assert loaded.backend == "numpy"
    assert loaded.backend == lowered.backend
    assert loaded.payload == lowered.payload
    assert loaded.text() == lowered.text()
    assert isinstance(loaded.signature, Signature)
    for field in _SIGNATURE_KEYS:
        assert getattr(loaded.signature, field) == getattr(lowered.signature, field)
    # The reloaded signature stays structurally coherent.
    assert loaded.signature.input_tree.num_leaves == len(loaded.signature.input_specs)
    assert loaded.signature.output_tree.num_leaves == len(loaded.signature.output_specs)


def test_lowered_program_roundtrip_static_values(tmp_path):
    def fn(x, scale):
        return evoxc.multiply(x, scale)

    lowered = evoxc.lower(
        evoxc.trace(fn, evoxc.TensorSpec((3,), evoxc.float32), 2.5)
    )
    path = tmp_path / "lowered_static.evoxcbin"
    lowered.save(path)
    loaded = LoweredProgram.load(path)
    assert loaded.signature.static_values == (2.5,)
    assert loaded.signature.input_specs == lowered.signature.input_specs
    assert loaded.signature.input_tree == lowered.signature.input_tree
