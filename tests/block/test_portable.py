"""Portable-decomposition inlining tests for ``evoxc.block`` (decorator form).

Covers the binding contract from ``evoxc/block/CONTEXT.md``:

* ``@evoxc.block @evoxc.defn`` registers a portable implementation (name = function
  name; ``has_portable`` True; resolved batching policy ``batching_rule``;
  the decomposition fallbacks land in the transforms registries).
* The numpy backend INLINES the portable decomposition at ``evoxc.lower`` time
  (graph->graph splice), so the interpreter never sees the ``block_call`` and
  numerical results equal the decomposition's computation.
* Errors: non-defn portables fail at declaration (``BlockError``); a portable
  returning a non-tensor or a shape that mismatches the declared outputs
  fails loudly at lower time (``BackendError``).
* Structured portable returns: a portable body may return a namedtuple or a
  dataclass of SymbolicTensors (including a namedtuple nested around plain
  tuples). The return annotation is NOT part of the pinned contract, so
  structured-return blocks declare ``outputs=`` explicitly. Direct
  evaluate/lowering works for both shapes; vmap works through the
  namedtuple return, and through the dataclass return once the parallel
  ``evoxc/block`` implementation lands — at base the dataclass-through-vmap
  case fails with ``BlockError`` from ``evoxc/block/rules.py::_flatten_outputs``
  (see the ``BUG(evoxc)`` marker on ``test_vmap_through_dataclass_portable``).

Notes:

* Block annotations are eagerly evaluated (``evoxc.defn``/``evoxc.block`` do not
  evaluate string annotations), so this module deliberately has NO
  ``from __future__ import annotations`` and multi-output return annotations
  are spelled as literal tuples of ``TensorSpec``.
* The block registry and the transforms rule registries are process-wide:
  every block declared here uses the ``port_`` prefix to stay unique.
* A rank-0 spec cannot accept a rank-1 input (``ShapeError`` at call time),
  so vector-numerics tests declare rank-1 specs with a runtime-dynamic
  ``None`` dim (``TensorSpec((None,), ...)``) — the objective's scalar-spec +
  linspace pairing is unsatisfiable as written.
"""

from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import pytest

import evoxc
from evoxc.block.errors import BlockError

BATCH = 8
DIM = 4

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _walk_blocks(block):
    """Yield every op of a block, nested-region ops first (bottom-up)."""
    for op in block.ops:
        for region in op.regions:
            for nested in region.blocks:
                yield from _walk_blocks(nested)
        yield op


def _count_block_calls(module):
    """Number of ``block_call`` ops anywhere in a (de)serialized ir.Module."""
    count = 0
    for fn in module.functions:
        for block in fn.region.blocks:
            for op in _walk_blocks(block):
                if op.name == "block_call":
                    count += 1
    return count


def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def _run(graph, *args):
    """Explicit pipeline: lower -> compile -> load -> run (returns structure)."""
    return evoxc.run(evoxc.load(evoxc.compile(evoxc.lower(graph))), *args)


def _batched_rows():
    """BATCH rows of DIM floats (module-level BATCH/DIM constants)."""
    return np.linspace(-1, 1, BATCH * DIM, dtype=np.float32).reshape(BATCH, DIM)


# ---------------------------------------------------------------------------
# Block declarations (module scope, unique ``port_`` names)
# ---------------------------------------------------------------------------


@evoxc.block
@evoxc.defn
def port_swish(x: evoxc.TensorSpec((None,), evoxc.float32)) -> evoxc.TensorSpec((None,), evoxc.float32):
    return evoxc.sigmoid(x) * x


@evoxc.defn
def port_swish_fn(x):
    return port_swish(x)


@evoxc.block
@evoxc.defn
def port_pair(
    x: evoxc.TensorSpec((None,), evoxc.float32),
) -> (evoxc.TensorSpec((None,), evoxc.float32), evoxc.TensorSpec((None,), evoxc.float32)):
    return evoxc.sigmoid(x) * x, evoxc.tanh(x)


@evoxc.defn
def port_pair_fn(x):
    return port_pair(x)


# Portable returning a non-tensor: the return annotation is omitted and the
# output spec is declared explicitly, so the declaration succeeds and the
# failure surfaces at LOWER time (where the decomposition is inlined).
@evoxc.block(outputs=[evoxc.TensorSpec((), evoxc.float32)])
@evoxc.defn
def port_badtensor(x: evoxc.TensorSpec((), evoxc.float32)):
    return 2.0


@evoxc.defn
def port_badtensor_fn(x):
    return port_badtensor(x)


# Declared output is a scalar, the portable returns rank 1 (a rank-vs-dim
# variant of the declared-vs-portable mismatch).
@evoxc.block(outputs=[evoxc.TensorSpec((), evoxc.float32)])
@evoxc.defn
def port_shape_mismatch(x: evoxc.TensorSpec((2,), evoxc.float32)):
    return evoxc.reshape(x, (2,))


@evoxc.defn
def port_shape_mismatch_fn(x):
    return port_shape_mismatch(x)


# --- Structured portable returns (namedtuple / dataclass) -------------------
#
# The return annotation of a namedtuple/dataclass-returning portable is NOT
# part of the pinned contract (the spec-derivation path cannot see through
# these annotations), so output specs are declared EXPLICITLY via
# ``outputs=``, exactly like port_badtensor above.

_PortNT = namedtuple("_PortNT", "a b")


@evoxc.block(
    outputs=[
        evoxc.TensorSpec((None,), evoxc.float32),
        evoxc.TensorSpec((None,), evoxc.float32),
    ]
)
@evoxc.defn
def port_nt(x: evoxc.TensorSpec((None,), evoxc.float32)):
    return _PortNT(evoxc.sigmoid(x) * x, evoxc.tanh(x))


@evoxc.defn
def port_nt_fn(x):
    return port_nt(x)


@dataclass
class _PortDC:
    a: object
    b: object


@evoxc.block(
    outputs=[
        evoxc.TensorSpec((None,), evoxc.float32),
        evoxc.TensorSpec((None,), evoxc.float32),
    ]
)
@evoxc.defn
def port_dc(x: evoxc.TensorSpec((None,), evoxc.float32)):
    return _PortDC(evoxc.sigmoid(x) * x, evoxc.tanh(x))


@evoxc.defn
def port_dc_fn(x):
    return port_dc(x)


# A namedtuple whose ``tail`` field holds a plain tuple: three tensor leaves.
_PortHeadTail = namedtuple("_PortHeadTail", "head tail")


@evoxc.block(
    outputs=[
        evoxc.TensorSpec((None,), evoxc.float32),
        evoxc.TensorSpec((None,), evoxc.float32),
        evoxc.TensorSpec((None,), evoxc.float32),
    ]
)
@evoxc.defn
def port_nt_nested(x: evoxc.TensorSpec((None,), evoxc.float32)):
    return _PortHeadTail(evoxc.sigmoid(x) * x, (evoxc.tanh(x), evoxc.sigmoid(x)))


@evoxc.defn
def port_nt_nested_fn(x):
    return port_nt_nested(x)


# ---------------------------------------------------------------------------
# Inlining + numerics
# ---------------------------------------------------------------------------


def test_decorator_block_inlined_with_correct_numerics():
    """The portable decomposition replaces the block_call and computes it."""
    x = np.linspace(-2, 2, 32, dtype=np.float32)
    y = evoxc.evaluate(port_swish_fn, x)
    assert isinstance(y, evoxc.Tensor)
    assert np.allclose(y.numpy(), _sigmoid_np(x) * x, rtol=1e-6)


def test_decorator_block_attributes():
    """Name = fn name; portable registered; default policy = batching_rule."""
    assert port_swish.name == "port_swish"
    assert port_swish.has_portable is True
    assert port_swish.batching_policy == "batching_rule"


def test_output_specs_derived_from_return_annotation():
    """No declared outputs: output_specs derive from the TensorSpec annotation."""
    (spec,) = port_swish.output_specs
    assert isinstance(spec, evoxc.TensorSpec)
    assert spec.shape == (None,)
    assert spec.dtype == np.dtype(np.float32)


def test_lowering_inlines_the_block_call():
    """After lower() the serialized payload contains no block_call ops."""
    graph = evoxc.trace(port_swish_fn, evoxc.TensorSpec((None,), evoxc.float32))
    assert _count_block_calls(graph.module) == 1  # the traced graph has it

    lowered = evoxc.lower(graph)
    module = evoxc.ir.deserialize_module(lowered.payload)
    assert _count_block_calls(module) == 0


def test_multi_output_portable():
    """A tuple-returning portable inlines into two numerically-correct results."""
    x = np.linspace(-2, 2, 32, dtype=np.float32)
    out = evoxc.evaluate(port_pair_fn, x)
    assert isinstance(out, tuple) and len(out) == 2
    a, b = out
    assert np.allclose(a.numpy(), _sigmoid_np(x) * x, rtol=1e-6)
    assert np.allclose(b.numpy(), np.tanh(x), rtol=1e-6)


# ---------------------------------------------------------------------------
# Structured portable returns (namedtuple / dataclass)
# ---------------------------------------------------------------------------


def test_namedtuple_portable_runs_with_correct_fields():
    """A namedtuple-returning portable runs with both fields correct."""
    x = np.linspace(-2, 2, 32, dtype=np.float32)
    out = evoxc.evaluate(port_nt_fn, x)
    assert isinstance(out, tuple) and len(out) == 2
    a, b = out
    assert np.allclose(a.numpy(), _sigmoid_np(x) * x, rtol=1e-6)
    assert np.allclose(b.numpy(), np.tanh(x), rtol=1e-6)


def test_dataclass_portable_runs_with_correct_fields():
    """A dataclass-returning portable runs with both fields correct."""
    x = np.linspace(-2, 2, 32, dtype=np.float32)
    out = evoxc.evaluate(port_dc_fn, x)
    assert isinstance(out, tuple) and len(out) == 2
    a, b = out
    assert np.allclose(a.numpy(), _sigmoid_np(x) * x, rtol=1e-6)
    assert np.allclose(b.numpy(), np.tanh(x), rtol=1e-6)


def test_namedtuple_portable_lowering_inlines_the_block_call():
    """lower() inlines a namedtuple-returning portable like any other."""
    graph = evoxc.trace(port_nt_fn, evoxc.TensorSpec((None,), evoxc.float32))
    assert _count_block_calls(graph.module) == 1  # the traced graph has it

    lowered = evoxc.lower(graph)
    module = evoxc.ir.deserialize_module(lowered.payload)
    assert _count_block_calls(module) == 0


def test_nested_namedtuple_portable_mixed_return():
    """A namedtuple field holding a plain tuple flattens to 3 outputs."""
    x = np.linspace(-2, 2, 32, dtype=np.float32)
    out = evoxc.evaluate(port_nt_nested_fn, x)
    assert isinstance(out, tuple) and len(out) == 3
    head, tail1, tail2 = out
    assert np.allclose(head.numpy(), _sigmoid_np(x) * x, rtol=1e-6)
    assert np.allclose(tail1.numpy(), np.tanh(x), rtol=1e-6)
    assert np.allclose(tail2.numpy(), _sigmoid_np(x), rtol=1e-6)


def test_vmap_through_namedtuple_portable():
    """vmap through a namedtuple-returning portable == per-row reference."""
    tf = evoxc.vmap(port_nt_fn, out_axes=(0, 0))
    graph = tf(evoxc.TensorSpec((BATCH, DIM), evoxc.float32))
    x = _batched_rows()
    out = _run(graph, x)
    assert isinstance(out, tuple) and len(out) == 2

    # Reference: the unvectorized computation, row by row (build once, run
    # per row — the documented explicit form of evoxc.evaluate).
    exe = evoxc.build(port_nt_fn, evoxc.TensorSpec((DIM,), evoxc.float32))
    ref_a = np.stack([evoxc.run(exe, row)[0].numpy() for row in x])
    ref_b = np.stack([evoxc.run(exe, row)[1].numpy() for row in x])
    a, b = out
    assert np.allclose(a.numpy(), ref_a, rtol=1e-6)
    assert np.allclose(b.numpy(), ref_b, rtol=1e-6)


# BUG(evoxc): dataclass portable returns are rejected by the portable batching
# fallback in evoxc/block/rules.py::_flatten_outputs — it only accepts
# SymbolicTensor / tuple / list / dict, so a dataclass raises BlockError at
# vmap time. The pinned contract requires dataclass returns to work through
# vmap; this test passes once the parallel evoxc implementation lands. Do NOT
# skip/xfail/weaken it.
def test_vmap_through_dataclass_portable():
    """vmap through a dataclass-returning portable == per-row reference."""
    tf = evoxc.vmap(port_dc_fn, out_axes=(0, 0))
    graph = tf(evoxc.TensorSpec((BATCH, DIM), evoxc.float32))
    x = _batched_rows()
    out = _run(graph, x)
    assert isinstance(out, tuple) and len(out) == 2

    # Reference: the unvectorized computation, row by row.
    exe = evoxc.build(port_dc_fn, evoxc.TensorSpec((DIM,), evoxc.float32))
    ref_a = np.stack([evoxc.run(exe, row)[0].numpy() for row in x])
    ref_b = np.stack([evoxc.run(exe, row)[1].numpy() for row in x])
    a, b = out
    assert np.allclose(a.numpy(), ref_a, rtol=1e-6)
    assert np.allclose(b.numpy(), ref_b, rtol=1e-6)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_factory_portable_must_be_a_defn():
    """A non-defn portable (lambda) fails the declaration with BlockError."""
    with pytest.raises(BlockError, match="must be an evoxc.defn"):
        evoxc.block(
            "port_factory_bad",
            inputs=[evoxc.TensorSpec((), evoxc.float32)],
            outputs=[evoxc.TensorSpec((), evoxc.float32)],
            portable=lambda v: v,
        )


def test_decorator_over_non_defn_raises():
    """Decorating a plain (non-defn) function with @evoxc.block raises."""
    with pytest.raises(BlockError, match="must be an evoxc.defn"):

        @evoxc.block
        def port_decorator_bad(x: evoxc.TensorSpec((), evoxc.float32)) -> evoxc.TensorSpec((), evoxc.float32):
            return x


def test_portable_returning_non_tensor_fails_at_lower():
    """Trace succeeds; lower inlines the portable and fails loudly.

    The portable returns a Python float: the traced decomposition has zero
    tensor results, so the numpy backend reports a BackendError at lower time
    (the objective's "must return symbolic" BlockError is the transforms-side
    message; the lower-time contract is "produces 0 result(s)").
    """
    graph = evoxc.trace(port_badtensor_fn, evoxc.TensorSpec((), evoxc.float32))
    with pytest.raises(evoxc.BackendError, match="produces 0 result"):
        evoxc.lower(graph)


def test_declared_vs_portable_shape_mismatch():
    """Declared scalar output vs rank-1 portable output -> BackendError."""
    graph = evoxc.trace(port_shape_mismatch_fn, evoxc.TensorSpec((2,), evoxc.float32))
    with pytest.raises(evoxc.BackendError, match="but the block_call declares"):
        evoxc.lower(graph)
