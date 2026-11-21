"""Differential parity harness: native transforms vs the Python transforms.

Every case traces ONE Python graph, converts its IR to the native envelope
(``convert.to_native``), runs the SAME transform through both engines and
executes both results through the explicit numpy pipeline on IDENTICAL seeded
inputs:

    python transform  ->  Graph  --lower/compile/load--> run --+
                                                               +--> compare
    native transform  ->  Module -> Graph --lower/compile/load--> run --+

The comparison is exact on dtype and shape, ``np.array_equal`` for bool and
integer leaves, ``np.allclose(rtol=1e-6, atol=1e-6, equal_nan=True)`` for
floating/complex leaves (plus identical NaN positions).

Modes
-----
``vectorize``      one entry per input position (``0`` mapped, ``None`` shared)
``vmap``           the same axes plus ``out_axes`` (a bare int or one entry per
                   output).  Python's public ``vmap`` only supports {0, None};
                   a non-leading ``1``/``-1``/per-output entry is compared
                   against a NUMPY oracle applied to the ``out_axes=0`` result
                   (``np.moveaxis`` for a mapped output, ``expand_dims`` +
                   broadcast for an unmapped one).  If the native engine still
                   defers non-leading axes, the mode is reported as DEFERRED
                   (both engines must then raise ``TransformError``) — the same
                   mode becomes a full comparison automatically once the
                   engine accepts it.
``grad``           ``argnums`` over the flattened tensor inputs
``jvp``            ``tangent_inputs`` = input indices with an explicit tangent
``vjp``            ``cotangent_inputs`` = output indices with an explicit
                   cotangent

Deferral assertions pin the documented v1 gaps on BOTH engines: the Python
transform and the native transform must raise ``TransformError``.

Run it directly (the cargo gate does exactly this)::

    source scripts/dev-env.sh
    .venv/bin/python native/evoxc-compiler/tests/parity_transform/run_parity.py

Exit status is 1 on any mismatch; the last line is always::

    PARITY: <cases> cases, <modes> mode-runs, <mismatches> mismatches
"""
from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

import evoxc
from evoxc import core
from evoxc.trace import Graph

import convert

native = convert.load_native()


# ---------------------------------------------------------------------------
# Mode descriptors
# ---------------------------------------------------------------------------
@dataclass
class Mode:
    """One (transform kind, config) exercised against one traced graph."""

    label: str
    kind: str
    config: dict
    build_python: Callable[[Graph], Graph]
    oracle: Optional[Callable[..., list]] = None
    deferral: bool = False
    note: str = ""
    #: The native engine may legitimately defer this mode (the non-leading
    #: `out_axes` entries the Python public API also refuses); every other mode
    #: must be compared, so a native-only deferral there is a mismatch.
    native_deferral_ok: bool = False


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _entries(out_axes, outputs: int) -> list:
    """Normalize ``out_axes`` to one entry per output."""
    if isinstance(out_axes, (list, tuple)):
        entries = list(out_axes)
    else:
        entries = [out_axes] * outputs
    if len(entries) != outputs:
        raise AssertionError(
            f"harness out_axes has {len(entries)} entries for {outputs} outputs"
        )
    return entries


def _all_leading(out_axes) -> bool:
    if isinstance(out_axes, (list, tuple)):
        return all(entry in (0, None) for entry in out_axes)
    return out_axes in (0, None)


def _axes_pytree(graph: Graph, entries: Sequence) -> tuple:
    """The public-API axes pytree: one entry per leaf, `None` for statics."""
    static = {record.index for record in graph.static_values}
    pending = iter(entries)
    leaves = [
        None if index in static else next(pending)
        for index in range(graph.input_specs.num_leaves)
    ]
    return core.unflatten(leaves, graph.input_specs)


def vectorize_mode(in_axes: Sequence) -> Mode:
    in_axes = tuple(in_axes)
    return Mode(
        label=f"vectorize(axes={in_axes})",
        kind="vectorize",
        config={"axes": list(in_axes)},
        build_python=lambda g: evoxc.vectorize(g, _axes_pytree(g, in_axes)),
    )


def vmap_mode(in_axes: Sequence, out_axes: Any = 0) -> Mode:
    in_axes = tuple(in_axes)
    if isinstance(out_axes, (list, tuple)) and len(out_axes) == 1:
        # A one-output graph takes the bare form in the public Python API.
        out_axes = out_axes[0]
    config = {"axes": list(in_axes), "out_axes": list(out_axes) if isinstance(out_axes, (list, tuple)) else out_axes}
    if _all_leading(out_axes):
        return Mode(
            label=f"vmap(axes={in_axes}, out_axes={out_axes})",
            kind="vmap",
            config=config,
            build_python=lambda g: evoxc.vmap(
                g, _axes_pytree(g, in_axes), out_axes
            ),
        )

    # Non-leading entries: Python's public vmap raises, so the Python side is
    # vmap(out_axes=0-masked) + the documented numpy rearrangement.
    def build(g: Graph) -> Graph:
        outputs = len(g.module.main.entry_block.terminator.operands)
        entries = _entries(out_axes, outputs)
        masked = tuple(0 for _ in entries)
        axes = _axes_pytree(g, in_axes)
        return evoxc.vmap(g, axes, masked[0] if len(masked) == 1 else masked)

    def oracle(leaves, base_graph, input_arrays):
        outputs = base_graph.module.main.entry_block.terminator.operands
        entries = _entries(out_axes, len(outputs))
        from evoxc.transforms.vmap import _collect_batch_dims, _leading_batch_count

        batch_dims = _collect_batch_dims(base_graph)
        mapped = [
            _leading_batch_count(value.type.shape, batch_dims) > 0
            for value in outputs
        ]
        batch = _batch_extent(in_axes, input_arrays)
        moved = list(leaves)
        for index, (entry, is_mapped) in enumerate(zip(entries, mapped)):
            if entry is None or entry == 0:
                continue
            array = moved[index]
            if is_mapped:
                moved[index] = np.moveaxis(array, 0, entry)
            else:
                expanded = np.expand_dims(array, entry)
                target = list(expanded.shape)
                target[entry] = batch
                moved[index] = np.broadcast_to(expanded, target).copy()
        return moved

    return Mode(
        label=f"vmap(axes={in_axes}, out_axes={out_axes}) [{_oracle_label(out_axes)}]",
        kind="vmap",
        config=config,
        build_python=build,
        oracle=oracle,
        note="numpy out_axes oracle",
        native_deferral_ok=True,
    )


def _oracle_label(out_axes) -> str:
    horizontal = out_axes if not isinstance(out_axes, (list, tuple)) else list(out_axes)
    return f"numpy oracle out_axes={horizontal}"


def _batch_extent(in_axes: Sequence, input_arrays: Sequence) -> int:
    for entry, array in zip(in_axes, input_arrays):
        if entry == 0:
            return int(np.shape(array)[0])
    raise AssertionError("an out_axes oracle needs at least one mapped input")


def grad_mode(argnums: Optional[Sequence]) -> Mode:
    """``grad``; ``None`` selects every flattened tensor input (the default)."""
    if argnums is None:
        return Mode(
            label="grad(argnums=None)",
            kind="grad",
            config={},
            build_python=lambda g: evoxc.grad(g, None),
        )
    argnums = tuple(argnums)
    return Mode(
        label=f"grad(argnums={argnums})",
        kind="grad",
        config={"argnums": list(argnums)},
        build_python=lambda g: evoxc.grad(g, argnums),
    )


def jvp_mode(tangent_inputs: Sequence) -> Mode:
    tangent_inputs = tuple(tangent_inputs)

    def build(g: Graph) -> Graph:
        static = {record.index for record in g.static_values}
        pending = iter(range(len(g.tensor_specs)))
        leaves = []
        for index in range(g.input_specs.num_leaves):
            if index in static:
                leaves.append(None)
                continue
            position = next(pending)
            spec = g.tensor_specs[position]
            leaves.append(
                core.TensorSpec(tuple(spec.shape), spec.dtype)
                if position in tangent_inputs
                else None
            )
        return evoxc.jvp(g, core.unflatten(leaves, g.input_specs))

    return Mode(
        label=f"jvp(tangent_inputs={tangent_inputs})",
        kind="jvp",
        config={"tangent_inputs": list(tangent_inputs)},
        build_python=build,
    )


def vjp_mode(cotangent_inputs: Sequence) -> Mode:
    cotangent_inputs = tuple(cotangent_inputs)

    def build(g: Graph) -> Graph:
        outputs = g.module.main.entry_block.terminator.operands
        cotangents = tuple(
            core.TensorSpec(tuple(value.type.shape), value.type.dtype)
            if index in cotangent_inputs
            else None
            for index, value in enumerate(outputs)
        )
        return evoxc.vjp(g, cotangents if len(outputs) > 1 else cotangents[0])

    return Mode(
        label=f"vjp(cotangent_inputs={cotangent_inputs})",
        kind="vjp",
        config={"cotangent_inputs": list(cotangent_inputs)},
        build_python=build,
    )


def deferral_mode(mode: Mode, reason: str) -> Mode:
    mode.deferral = True
    mode.note = reason
    mode.label += " [deferral]"
    return mode


# ---------------------------------------------------------------------------
# Case descriptors
# ---------------------------------------------------------------------------
@dataclass
class Case:
    name: str
    fn: Callable
    specs: tuple
    modes: tuple
    seed: int = 0


F32 = evoxc.float32
F64 = evoxc.float64
I32 = evoxc.int32


def _spec(shape, dtype=F32):
    return evoxc.TensorSpec(tuple(shape), dtype)


CASES: list[Case] = [
    Case(
        "elementwise_chain",
        lambda x, y: evoxc.sum(evoxc.relu(x * y + x - y) * y + x * y),
        (_spec((3,)), _spec((3,))),
        (
            vectorize_mode((0, None)),
            vmap_mode((0, 0), 0),
            jvp_mode((0, 1)),
        ),
        seed=1,
    ),
    Case(
        "unary_exp_log_log1p",
        lambda x: evoxc.sum(evoxc.exp(x) + evoxc.log(x) + evoxc.log1p(x)),
        (_spec((2, 3)),),
        (grad_mode((0,)), jvp_mode((0,)), vjp_mode((0,))),
        seed=2,
    ),
    Case(
        "unary_sqrt_square_abs",
        lambda x: evoxc.sum(evoxc.sqrt(x) + evoxc.square(x) + evoxc.abs(x)),
        (_spec((4,)),),
        (grad_mode((0,)), vjp_mode((0,))),
        seed=3,
    ),
    Case(
        "unary_sigmoid_tanh",
        lambda x: evoxc.sum(evoxc.sigmoid(x) * evoxc.tanh(x)),
        (_spec((5,)),),
        (grad_mode((0,)), jvp_mode((0,))),
        seed=4,
    ),
    Case(
        "unary_trig",
        lambda x: evoxc.sum(evoxc.sin(x) + evoxc.cos(x) + evoxc.tan(x)),
        (_spec((4,)),),
        (grad_mode((0,)), vectorize_mode((0,))),
        seed=5,
    ),
    Case(
        "unary_erf_gelu",
        lambda x: evoxc.sum(evoxc.erf(x) + evoxc.gelu(x)),
        (_spec((6,)),),
        (grad_mode((0,)), jvp_mode((0,))),
        seed=6,
    ),
    Case(
        "arithmetic_binary",
        lambda a, b: evoxc.sum(
            a
            + b
            + (a - b)
            + a * b
            + a / b
            + evoxc.power(a, 2.0)
            + evoxc.remainder(a, b)
            + evoxc.maximum(a, b)
            + evoxc.minimum(a, b)
        ),
        (_spec((3,)), _spec((3,))),
        (
            vectorize_mode((0, 0)),
            grad_mode((0, 1)),
            jvp_mode((0, 1)),
        ),
        seed=7,
    ),
    Case(
        "select_case",
        lambda x, y: evoxc.sum(evoxc.select(x > y, x + y, x * y)),
        (_spec((4,)), _spec((4,))),
        (vectorize_mode((0, 0)), grad_mode((0, 1))),
        seed=8,
    ),
    Case(
        "cast_case",
        lambda x: evoxc.sum(evoxc.cast(x, F64) * 2.0),
        (_spec((4,), F32),),
        (vectorize_mode((0,)), jvp_mode((0,))),
        seed=9,
    ),
    Case(
        "reductions_keepdims",
        lambda x: evoxc.sum(
            evoxc.sum(x, axes=(0,), keepdims=True) * 2.0
            + evoxc.max(x, axes=(1,), keepdims=True)
        ),
        (_spec((2, 3)),),
        (
            vectorize_mode((0,)),
            grad_mode((0,)),
            vmap_mode((0,), 1),
        ),
        seed=10,
    ),
    Case(
        "reduction_mean_prod",
        lambda x: evoxc.sum(
            evoxc.mean(x, axes=(0,), keepdims=True)
            * evoxc.prod(x, axes=(-1,), keepdims=True)
        ),
        (_spec((2, 3)),),
        (grad_mode((0,)), jvp_mode((0,))),
        seed=11,
    ),
    Case(
        "dot_2d",
        lambda a, b: evoxc.sum(evoxc.dot(a, b)),
        (_spec((2, 3)), _spec((3, 4))),
        (
            vectorize_mode((0, None)),
            grad_mode((0,)),
            jvp_mode((0,)),
        ),
        seed=12,
    ),
    Case(
        "dot_batched",
        lambda a, b: evoxc.sum(evoxc.dot(a, b) * 2.0),
        (_spec((3, 2)), _spec((2, 4))),
        (vectorize_mode((0, 0)),),
        seed=13,
    ),
    Case(
        "conv1d",
        lambda x, w: evoxc.sum(
            evoxc.conv(x, w, strides=2, padding="SAME") * 2.0
        ),
        (_spec((1, 4, 8)), _spec((2, 4, 3))),
        (
            deferral_mode(grad_mode((0, 1)), "conv VJP is a v1 deferral"),
            vectorize_mode((0, None)),
        ),
        seed=14,
    ),
    Case(
        "conv2d",
        lambda x, w: evoxc.sum(
            evoxc.conv(
                x,
                w,
                strides=(1, 2),
                padding=((1, 0), (0, 1)),
                kernel_dilation=(2, 1),
            )
        ),
        (_spec((1, 2, 6, 6)), _spec((3, 2, 3, 3))),
        (
            deferral_mode(grad_mode((0,)), "conv VJP is a v1 deferral"),
            vectorize_mode((0, None)),
        ),
        seed=15,
    ),
    Case(
        "reshape_wildcard",
        lambda x: evoxc.sum(evoxc.reshape(x, (-1, 3)) * 2.0),
        (_spec((2, 6)),),
        (vectorize_mode((0,)), jvp_mode((0,))),
        seed=16,
    ),
    Case(
        "transpose_perm",
        lambda x: evoxc.sum(evoxc.transpose(x, (0, 2, 1)) * 2.0),
        (_spec((2, 3, 4)),),
        (vectorize_mode((0,)), grad_mode((0,))),
        seed=17,
    ),
    Case(
        "slice_case",
        lambda x: evoxc.sum(evoxc.slice(x, 1, 3) * 2.0),
        (_spec((6,)),),
        (vectorize_mode((0,)), grad_mode((0,))),
        seed=18,
    ),
    Case(
        "pad_case",
        lambda x: evoxc.sum(evoxc.pad(x, ((1, 0), (0, 2)), 0.5)),
        (_spec((2, 3)),),
        (vectorize_mode((0,)), grad_mode((0,))),
        seed=19,
    ),
    Case(
        "concatenate_case",
        lambda x, y: evoxc.sum(evoxc.concatenate([x, y], axis=1) * 2.0),
        (_spec((2, 3)), _spec((2, 3))),
        (vectorize_mode((0, 0)), grad_mode((0, 1))),
        seed=20,
    ),
    Case(
        "gather_case",
        lambda x, idx: evoxc.sum(evoxc.gather(x, idx, axis=0) * 2.0),
        (_spec((4, 3)), _spec((2,), I32)),
        (grad_mode((0,)),),
        seed=21,
    ),
    Case(
        "scatter_case",
        lambda x, idx, upd: evoxc.sum(
            evoxc.scatter(x, idx, upd, axis=0) * 2.0
        ),
        (_spec((4,)), _spec((2,), I32), _spec((2,))),
        (
            vectorize_mode((0, 0, 0)),
            grad_mode((0,)),
        ),
        seed=22,
    ),
    Case(
        "tile_case",
        lambda x: evoxc.sum(evoxc.tile(x, (2, 1)) * 2.0),
        (_spec((2, 3)),),
        (
            deferral_mode(vectorize_mode((0,)), "tile has no batching rule"),
            deferral_mode(grad_mode((0,)), "tile has no derivative rule"),
        ),
        seed=23,
    ),
    Case(
        "diag_case",
        lambda x: evoxc.sum(evoxc.diag(x) * 2.0),
        (_spec((4,)),),
        (
            deferral_mode(vectorize_mode((0,)), "diag has no batching rule"),
            deferral_mode(grad_mode((0,)), "diag has no derivative rule"),
        ),
        seed=24,
    ),
    Case(
        "tril_triu",
        lambda x: evoxc.sum(evoxc.tril(x) + evoxc.triu(x)),
        (_spec((3, 3)),),
        (vectorize_mode((0,)), grad_mode((0,))),
        seed=25,
    ),
    Case(
        "cumsum_case",
        lambda x: evoxc.sum(evoxc.cumsum(x) * 2.0),
        (_spec((5,)),),
        (vectorize_mode((0,)), grad_mode((0,)), jvp_mode((0,))),
        seed=26,
    ),
    Case(
        "vmap_multi_out",
        lambda x, y: (x * y, evoxc.sum(x * y)),
        (_spec((3,)), _spec((3,))),
        (
            vmap_mode((0, 0), (0, 0)),
            vmap_mode((0, 0), (1, 0)),
        ),
        seed=27,
    ),
    Case(
        "vmap_unmapped_output",
        lambda x, y: (x * x, y * y),
        (_spec((3,)), _spec((2, 3))),
        (
            vmap_mode((0, None), (0, 0)),
            vmap_mode((0, None), (1, 0)),
            vmap_mode((0, None), (0, None)),
        ),
        seed=28,
    ),
    Case(
        "vmap_out_minus_one",
        lambda x: evoxc.sum(evoxc.tanh(x) * x, axes=(1,)),
        (_spec((2, 3)),),
        (vmap_mode((0,), -1), vmap_mode((0,), 0)),
        seed=29,
    ),
    Case(
        "two_input_broadcast",
        lambda a, b: evoxc.sum(a * b + b),
        (_spec((2, 1)), _spec((3,))),
        (vectorize_mode((0, None)), vmap_mode((0, None), 1)),
        seed=30,
    ),
    Case(
        "float64_chain",
        lambda x: evoxc.sum(evoxc.sigmoid(x) * evoxc.tanh(x)),
        (_spec((4,), F64),),
        (grad_mode((0,)), jvp_mode((0,))),
        seed=31,
    ),
    Case(
        "int_cast",
        lambda x: evoxc.sum(evoxc.cast(x, F32) * 2.0),
        (_spec((4,), I32),),
        (vectorize_mode((0,)), jvp_mode(())),
        seed=32,
    ),
    Case(
        "stop_gradient",
        lambda x: evoxc.sum(evoxc.stop_gradient(x) * x),
        (_spec((4,)),),
        (grad_mode((0,)), vectorize_mode((0,))),
        seed=33,
    ),
    Case(
        "vmap_comparison_bool_output",
        lambda x, y: (x > y, x + y),
        (_spec((3,)), _spec((3,))),
        (vmap_mode((0, 0), (0, 0)), vectorize_mode((0, 0))),
        seed=34,
    ),
    Case(
        "transpose_default_permutation",
        lambda x: evoxc.sum(evoxc.transpose(x) * 2.0),
        (_spec((2, 3, 4)),),
        (vectorize_mode((0,)), grad_mode((0,))),
        seed=35,
    ),
    Case(
        "conv1d_default_attributes",
        lambda x, w: evoxc.sum(evoxc.conv(x, w) * 2.0),
        (_spec((1, 3, 6)), _spec((2, 3, 3))),
        (vectorize_mode((0, None)),),
        seed=36,
    ),
    Case(
        "jvp_partial_tangents",
        lambda a, b: evoxc.sum(a * b + evoxc.sin(a)),
        (_spec((4,)), _spec((4,))),
        (jvp_mode((0,)),),
        seed=37,
    ),
    Case(
        "vjp_multi_output",
        lambda x: (evoxc.sum(x), x * x),
        (_spec((3,)),),
        (vjp_mode((0, 1)), vjp_mode((1,))),
        seed=38,
    ),
    Case(
        "grad_default_argnums",
        lambda a, b: evoxc.sum(a * b + evoxc.square(a)),
        (_spec((3,)), _spec((3,))),
        (grad_mode(None),),
        seed=39,
    ),
    Case(
        "vmap_three_outputs",
        lambda x, y: (x * y, x + y, x - y),
        (_spec((3,)), _spec((3,))),
        (vmap_mode((0, 0), (0, 0, 0)), vectorize_mode((0, 0))),
        seed=40,
    ),
    Case(
        "broadcasting_power_divide",
        lambda x, y: evoxc.sum(evoxc.power(x, y) + x / y),
        (_spec((3, 1)), _spec((1, 4))),
        (vectorize_mode((0, None)), vectorize_mode((0, 0)), grad_mode((0, 1))),
        seed=41,
    ),
    Case(
        "jvp_two_outputs",
        lambda x: (evoxc.sigmoid(x), evoxc.tanh(x)),
        (_spec((4,)),),
        (jvp_mode((0,)),),
        seed=42,
    ),
    Case(
        # A value read twice: the reverse sweep must ACCUMULATE both terms.
        "reuse_accumulation",
        lambda x: evoxc.sum((y := evoxc.square(x)) * y + y),
        (_spec((4,)),),
        (grad_mode((0,)), vjp_mode((0,)), jvp_mode((0,))),
        seed=44,
    ),
    Case(
        "reduce_max_grad",
        lambda x: evoxc.sum(evoxc.max(x, axes=(1,)) * evoxc.min(x, axes=(1,))),
        (_spec((2, 3)),),
        (grad_mode((0,)), jvp_mode((0,)), vectorize_mode((0,))),
        seed=45,
    ),
    Case(
        "slice_strided_grad",
        lambda x: evoxc.sum(evoxc.slice(x, 0, 4, strides=2) * 2.0),
        (_spec((8,)),),
        (vectorize_mode((0,)), grad_mode((0,)), jvp_mode((0,))),
        seed=46,
    ),
    Case(
        "mean_multi_axes",
        lambda x: evoxc.sum(evoxc.mean(x, axes=(1, 2), keepdims=False) * 2.0),
        (_spec((2, 3, 4)),),
        (vectorize_mode((0,)), grad_mode((0,))),
        seed=47,
    ),
    Case(
        "cumsum_axis_grad",
        lambda x: evoxc.sum(evoxc.cumsum(x, axis=1) * 2.0),
        (_spec((2, 5)),),
        (vectorize_mode((0,)), grad_mode((0,)), jvp_mode((0,))),
        seed=48,
    ),
    Case(
        "scatter_vjp",
        lambda x, idx, upd: evoxc.sum(evoxc.scatter(x, idx, upd, axis=0) * 2.0),
        (_spec((4,)), _spec((2,), I32), _spec((2,))),
        (vjp_mode((0,)),),
        seed=49,
    ),
    Case(
        "dot_vmap_shared_rhs",
        lambda a, b: evoxc.dot(a, b),
        (_spec((2, 3)), _spec((3, 4))),
        (vmap_mode((0, None), 0), vmap_mode((0, None), 1)),
        seed=50,
    ),
    Case(
        "transpose_vmap_trailing_axis",
        lambda x: evoxc.transpose(x, (0, 2, 1)) * 2.0,
        (_spec((2, 3, 4)),),
        (vmap_mode((0,), -1), vmap_mode((0,), 1)),
        seed=51,
    ),
    Case(
        # The trace specializes the Python float and the dict of floats into
        # graph statics, so the runtime argument tree mixes tensors and statics.
        "static_leaves",
        lambda x, y, scale, weights: evoxc.sum(evoxc.tanh(x * scale) * y)
        + weights["a"] * evoxc.sum(y)
        + weights["b"],
        (_spec((3,)), _spec((3,)), 0.5, {"a": 2.0, "b": -1.0}),
        (
            vectorize_mode((0, None)),
            grad_mode((0,)),
            grad_mode(None),
            jvp_mode((0,)),
        ),
        seed=43,
    ),
]


# ---------------------------------------------------------------------------
# Deferral assertions (both engines must raise TransformError)
# ---------------------------------------------------------------------------
@dataclass
class Deferral:
    name: str
    fn: Callable
    specs: tuple
    build_python: Callable[[Graph], Graph]
    kind: str
    config: dict
    seed: int = 100


def _cond_graph(x):
    return evoxc.cond(
        evoxc.sum(x) > 0.0,
        lambda v: v * 2.0,
        lambda v: v * -2.0,
        x,
    )


def _random_graph(x):
    key = evoxc.random.key(7)
    return evoxc.sum(evoxc.random.uniform(key, (4,)) * x)


def _sort_graph(x):
    return evoxc.sum(evoxc.sort(x) * 2.0)


def _square_graph(x):
    return evoxc.sum(x * x)


DEFERRALS: list[Deferral] = [
    Deferral(
        "vectorize_control_flow_cond",
        _cond_graph,
        (_spec((3,)),),
        lambda g: evoxc.vectorize(g, (0,)),
        "vectorize",
        {"axes": [0]},
    ),
    Deferral(
        "grad_random_op",
        _random_graph,
        (_spec((4,)),),
        lambda g: evoxc.grad(g, (0,)),
        "grad",
        {"argnums": [0]},
    ),
    Deferral(
        "vectorize_sort",
        _sort_graph,
        (_spec((5,)),),
        lambda g: evoxc.vectorize(g, (0,)),
        "vectorize",
        {"axes": [0]},
    ),
    Deferral(
        "vmap_out_axes_none_on_mapped_output",
        _square_graph,
        (_spec((3,)),),
        lambda g: evoxc.vmap(g, (0,), None),
        "vmap",
        {"axes": [0], "out_axes": None},
    ),
    Deferral(
        "vectorize_non_leading_in_axes",
        _square_graph,
        (_spec((3, 4)),),
        lambda g: evoxc.vectorize(g, (1,)),
        "vectorize",
        {"axes": [1]},
    ),
]


# ---------------------------------------------------------------------------
# Input generation / execution / comparison
# ---------------------------------------------------------------------------
def _default_array(spec, rng, batch_sizes=None):
    shape = _concrete_shape(spec.shape, batch_sizes if batch_sizes is not None else {})
    dtype = np.dtype(spec.dtype)
    if dtype.kind == "f":
        return rng.uniform(0.5, 1.5, size=shape).astype(dtype)
    if dtype.kind == "c":
        return (
            rng.uniform(0.5, 1.5, size=shape)
            + 1j * rng.uniform(-0.5, 0.5, size=shape)
        ).astype(dtype)
    if dtype.kind == "b":
        return rng.integers(0, 2, size=shape).astype(dtype)
    if dtype.kind in "iu":
        return rng.integers(1, 4, size=shape).astype(dtype)
    raise AssertionError(f"harness cannot generate a {dtype} input")


def _tangent_array(spec, rng, batch_sizes=None):
    shape = _concrete_shape(spec.shape, batch_sizes if batch_sizes is not None else {})
    dtype = np.dtype(spec.dtype)
    if dtype.kind == "f":
        return rng.standard_normal(size=shape).astype(dtype)
    if dtype.kind == "c":
        return (
            rng.standard_normal(size=shape) + 1j * rng.standard_normal(size=shape)
        ).astype(dtype)
    return _default_array(spec, rng, batch_sizes)


#: Concrete extent used for every fresh symbolic batch `Dim` in the harness.
BATCH_EXTENT = 3


def _dim_value(dim, batch_sizes: dict):
    if isinstance(dim, core.Dim):
        if dim.size is not None:
            return int(dim.size)
        return batch_sizes.setdefault(dim.name, BATCH_EXTENT)
    if isinstance(dim, core.DimExpr):
        left = _dim_value(dim.left, batch_sizes)
        right = _dim_value(dim.right, batch_sizes)
        return {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "floordiv": lambda a, b: a // b,
            "mod": lambda a, b: a % b,
            "min": min,
            "max": max,
        }[dim.op](left, right)
    if dim is None:
        raise AssertionError(
            "the parity harness only traces graphs with static shapes "
            "(a runtime-dynamic dim has no concrete extent)"
        )
    return int(dim)


def _concrete_shape(shape, batch_sizes: dict) -> tuple:
    return tuple(_dim_value(dim, batch_sizes) for dim in shape)


def _execute(graph: Graph, args: tuple) -> list:
    """lower → compile → load → run on the numpy backend; flat numpy leaves."""
    lowered = evoxc.lower(graph, backend="numpy")
    artifact = evoxc.compile(lowered, backend="numpy")
    executable = evoxc.load(artifact, backend="numpy")
    out = evoxc.run(executable, *args)
    leaves, _tree = core.flatten(out)
    return [leaf.numpy() if isinstance(leaf, evoxc.Tensor) else leaf for leaf in leaves]


def _native_graph(module, template: Graph) -> Graph:
    """A Graph around a native-produced module, reusing the Python trees."""
    return Graph(
        module,
        template.input_specs,
        template.tensor_specs,
        template.output_tree,
        template.static_values,
        template.output_static_values,
    )


def _plain_tree(spec, arrays: list, static: dict, base: int = 0):
    """Rebuild a runtime argument tree from generated leaves + recorded statics.

    Leaves are filled in `core.flatten` order: a leaf the trace specialized to a
    static value (a Python scalar, a dict of them, …) gets the recorded value
    back, every other leaf gets the next generated tensor (`base` is the flat
    leaf index of `spec` inside the graph's whole input tree).
    """
    pending = iter(arrays)
    leaves = [
        static[base + index] if (base + index) in static else next(pending)
        for index in range(spec.num_leaves)
    ]
    return core.unflatten(leaves, spec)


def _plain_args(template: Graph, arrays: list) -> tuple:
    """Runtime arguments matching a non-sweep graph's input tree."""
    static = {record.index: record.value for record in template.static_values}
    return _plain_tree(template.input_specs, arrays, static)


def _args_for(
    mode: Mode, template: Graph, primal: list, extras: list, n_outputs: int
) -> tuple:
    """Runtime arguments matching the transformed graph's input tree.

    The sweep graphs take a 2-tuple `(primal structure, flat extra tuple)`;
    `None` entries are the recorded static leaves for zero tangents/cotangents.
    """
    if mode.kind in ("jvp", "vjp"):
        static = {record.index: record.value for record in template.static_values}
        primal_part = _plain_tree(template.input_specs.children[0], list(primal), static)
        if mode.kind == "jvp":
            selected = set(mode.config["tangent_inputs"])
            pending = list(extras)
            extra_part = tuple(
                pending.pop(0) if index in selected else None
                for index in range(len(primal))
            )
        else:
            selected = set(mode.config["cotangent_inputs"])
            pending = list(extras)
            extra_part = tuple(
                pending.pop(0) if index in selected else None
                for index in range(n_outputs)
            )
        return (primal_part, extra_part)
    return _plain_args(template, list(primal))


def _compare(mode: Mode, expected: list, actual: list) -> Optional[str]:
    if len(expected) != len(actual):
        return f"output leaf count {len(expected)} (python) vs {len(actual)} (native)"
    for index, (want, got) in enumerate(zip(expected, actual)):
        want = np.asarray(want)
        got = np.asarray(got)
        if want.dtype != got.dtype:
            return f"leaf {index}: dtype {want.dtype} (python) vs {got.dtype} (native)"
        if want.shape != got.shape:
            return f"leaf {index}: shape {want.shape} (python) vs {got.shape} (native)"
        if want.dtype.kind in "biu":
            if not np.array_equal(want, got):
                return f"leaf {index}: integer/bool values differ ({want!r} vs {got!r})"
            continue
        if not np.array_equal(np.isnan(want), np.isnan(got)):
            return f"leaf {index}: NaN positions differ"
        if not np.allclose(want, got, rtol=1e-6, atol=1e-6, equal_nan=True):
            diff = np.abs(want - got)
            finite = np.isfinite(diff)
            worst = float(np.max(diff[finite])) if finite.any() else float("inf")
            return (
                f"leaf {index}: values differ (max |python-native| = {worst:e}; "
                f"python[0]={np.ravel(want)[0]!r} native[0]={np.ravel(got)[0]!r})"
            )
    return None


def _error_kind(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, native.NativeError):
        return "NativeError", exc.kind
    return type(exc).__name__, ""


def _is_transform_error(exc: BaseException) -> bool:
    class_name, kind = _error_kind(exc)
    return "TransformError" in class_name or kind == "TransformError"


def _self_check() -> None:
    """The comparison must be able to say "no" (a vacuous parity is worthless)."""
    reference = [np.arange(6, dtype=np.float32).reshape(2, 3)]
    if _compare(None, reference, [reference[0].copy()]) is not None:
        raise AssertionError("harness self-check: identical leaves reported a mismatch")
    perturbed = reference[0].copy()
    perturbed[0, 0] += 1.0
    if _compare(None, reference, [perturbed]) is None:
        raise AssertionError("harness self-check: a perturbed leaf was not detected")
    if _compare(None, reference, [reference[0].astype(np.float64)]) is None:
        raise AssertionError("harness self-check: a dtype change was not detected")
    if _compare(None, reference, [reference[0][:1]]) is None:
        raise AssertionError("harness self-check: a shape change was not detected")
    if _compare(None, reference, [reference[0], reference[0]]) is None:
        raise AssertionError("harness self-check: a leaf-count change was not detected")
    print("SELF-CHECK: the comparison detects perturbed values, dtypes, shapes and arity")


# ---------------------------------------------------------------------------
# One case, one mode
# ---------------------------------------------------------------------------
@dataclass
class Stats:
    cases: int = 0
    modes: int = 0
    mismatches: int = 0
    deferred: int = 0


def _report(status: str, case: str, mode: Mode, detail: str = "") -> None:
    suffix = f" — {mode.note}" if mode.note else ""
    line = f"[{status}] {case} :: {mode.label}{suffix}"
    if detail:
        line += f" — {detail}"
    print(line)


def run_mode(case: Case, traced: Graph, envelope: dict, mode: Mode, stats: Stats) -> None:
    stats.modes += 1
    rng = np.random.default_rng(case.seed)
    n_primal = len(traced.tensor_specs)
    n_outputs = len(traced.module.main.entry_block.terminator.operands)
    if mode.deferral:
        _check_deferral(case.name, traced, envelope, mode, stats)
        return
    try:
        python_graph = mode.build_python(traced)
    except Exception as exc:  # noqa: BLE001 - parity reports every failure
        stats.mismatches += 1
        _report(
            "FAIL",
            case.name,
            mode,
            f"python transform raised {type(exc).__name__}: {exc}",
        )
        return

    try:
        native_envelope = native.transform(mode.kind, envelope, mode.config)
    except Exception as exc:  # noqa: BLE001
        if mode.native_deferral_ok and _is_transform_error(exc) and "deferred" in str(exc):
            stats.deferred += 1
            _report(
                "DEFERRED",
                case.name,
                mode,
                f"native engine defers this mode: {exc}",
            )
            return
        stats.mismatches += 1
        stats_text = f"{type(exc).__name__}"
        if isinstance(exc, native.NativeError):
            stats_text += f" kind={exc.kind}"
        _report("FAIL", case.name, mode, f"native transform raised {stats_text}: {exc}")
        return

    native_graph = _native_graph(convert.from_native(native_envelope), python_graph)
    try:
        arrays = [
            _default_array(spec, rng) for spec in python_graph.tensor_specs[:n_primal]
        ]
        extras = [
            _tangent_array(spec, rng)
            for spec in python_graph.tensor_specs[n_primal:]
        ]
    except Exception as exc:  # noqa: BLE001
        stats.mismatches += 1
        _report(
            "FAIL",
            case.name,
            mode,
            f"harness could not build inputs: {type(exc).__name__}: {exc}",
        )
        return
    args = _args_for(mode, python_graph, arrays, extras, n_outputs)

    try:
        expected = _execute(python_graph, args)
    except Exception as exc:  # noqa: BLE001
        stats.mismatches += 1
        _report(
            "FAIL",
            case.name,
            mode,
            f"python graph failed to run: {type(exc).__name__}: {exc}",
        )
        return
    if mode.oracle is not None:
        try:
            expected = mode.oracle(expected, python_graph, arrays)
        except Exception as exc:  # noqa: BLE001
            stats.mismatches += 1
            _report(
                "FAIL",
                case.name,
                mode,
                f"numpy out_axes oracle failed: {type(exc).__name__}: {exc}",
            )
            return

    try:
        actual = _execute(native_graph, args)
    except Exception as exc:  # noqa: BLE001
        stats.mismatches += 1
        _report(
            "FAIL",
            case.name,
            mode,
            f"native graph failed to run: {type(exc).__name__}: {exc}",
        )
        return

    detail = _compare(mode, expected, actual)
    if detail is None:
        _report("PASS", case.name, mode)
    else:
        stats.mismatches += 1
        _report("FAIL", case.name, mode, detail)


def _check_deferral(
    name: str, traced: Graph, envelope: dict, mode: Mode, stats: Stats
) -> None:
    """Both engines must reject a documented v1 deferral with TransformError."""
    python_exc = None
    try:
        mode.build_python(traced)
    except Exception as exc:  # noqa: BLE001
        python_exc = exc
    native_exc = None
    try:
        native.transform(mode.kind, envelope, mode.config)
    except Exception as exc:  # noqa: BLE001
        native_exc = exc
    if python_exc is None:
        stats.mismatches += 1
        _report("FAIL", name, mode, "the Python transform did NOT raise")
        return
    if not _is_transform_error(python_exc):
        stats.mismatches += 1
        _report(
            "FAIL",
            name,
            mode,
            f"the Python transform raised {type(python_exc).__name__}, "
            f"expected TransformError: {python_exc}",
        )
        return
    if native_exc is None:
        stats.mismatches += 1
        _report("FAIL", name, mode, "the native transform did NOT raise")
        return
    if not _is_transform_error(native_exc):
        stats.mismatches += 1
        _report(
            "FAIL",
            name,
            mode,
            f"the native transform raised {type(native_exc).__name__}, "
            f"expected TransformError: {native_exc}",
        )
        return
    stats.deferred += 1
    _report(
        "DEFERRED",
        name,
        mode,
        f"both engines raise TransformError ({mode.note})",
    )


def run_case(case: Case, stats: Stats) -> None:
    stats.cases += 1
    traced = evoxc.trace(case.fn, *case.specs)
    envelope = convert.to_native(traced.module)
    for mode in case.modes:
        try:
            run_mode(case, traced, envelope, mode, stats)
        except Exception:  # noqa: BLE001 - a harness bug must not hide results
            stats.mismatches += 1
            _report("FAIL", case.name, mode, "harness error:\n" + traceback.format_exc())


def run_deferral(deferral: Deferral, stats: Stats) -> None:
    stats.cases += 1
    stats.modes += 1
    traced = evoxc.trace(deferral.fn, *deferral.specs)
    envelope = convert.to_native(traced.module)
    mode = Mode(
        label=f"{deferral.kind}({deferral.config})",
        kind=deferral.kind,
        config=deferral.config,
        build_python=deferral.build_python,
        deferral=True,
    )
    _check_deferral(deferral.name, traced, envelope, mode, stats)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    only = None
    if args and args[0] in ("-k", "--filter"):
        if len(args) < 2:
            print("usage: run_parity.py [-k SUBSTRING]", file=sys.stderr)
            return 2
        only = args[1]

    stats = Stats()
    selected_cases = [case for case in CASES if only is None or only in case.name]
    selected_deferrals = [
        item for item in DEFERRALS if only is None or only in item.name
    ]
    if only is not None and not selected_cases and not selected_deferrals:
        print(f"no case matches {only!r}", file=sys.stderr)
        return 2

    _self_check()
    for case in selected_cases:
        run_case(case, stats)
    for deferral in selected_deferrals:
        run_deferral(deferral, stats)

    total_modes = stats.modes
    print(
        f"PARITY: {stats.cases} cases, {total_modes} mode-runs, "
        f"{stats.mismatches} mismatches"
    )
    if stats.deferred:
        print(
            f"PARITY: {stats.deferred} mode-runs are documented deferrals "
            "(both engines raise TransformError)"
        )
    return 1 if stats.mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
