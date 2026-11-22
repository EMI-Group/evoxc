"""Case descriptors for the native `batch_functions` differential parity harness.

Each case is a small POPULATION of programs built with ``evoxc.trace(fn, *specs)``
directly — never through the public ``batch_functions`` callable path — so every
program of a population is traced against the EXACT same argument specs.  The
descriptor also carries the public keyword arguments the merge is driven with
(``axes``/``out_axes``/``switch_ops``/``bands``/``alignment``) plus the seed the
harness derives its runtime arrays from.

Two lists are exported:

* ``CASES`` — populations the Python merge must accept and the native FFI must
  reproduce structurally, numerically and in its report.
* ``DEFERRALS`` — populations/option combinations the documented v1 contract
  rejects with ``core.TransformError`` on BOTH engines.

The banding and cost model under test are native-only, owned by
``native/evoxc-compiler/src/batch_functions/bands.rs``.

`run_batch_parity.py` owns the execution/comparison; this module only builds
programs, so it stays import-light and deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Tuple

import evoxc
from evoxc import core

F32 = evoxc.float32


def spec(shape, dtype=F32) -> "evoxc.TensorSpec":
    """A static ``TensorSpec`` (the parity corpus never uses symbolic dims)."""
    return evoxc.TensorSpec(tuple(shape), dtype)


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BatchCase:
    """One accepted merge population.

    Attributes:
        name: Stable case id (the ``-k`` filter matches a substring of it).
        fns: The per-function callables; every one is traced against `specs`.
        specs: The argument specs shared by EVERY program of the population
            (a traced program is ``evoxc.trace(fn, *specs)``).
        options: The public ``batch_functions`` keyword arguments.
        seed: The ``np.random.default_rng`` seed for the runtime arrays.
        note: A one-line description of what the case pins.
    """

    name: str
    fns: Tuple[Callable, ...]
    specs: Tuple[Any, ...]
    options: dict = field(default_factory=dict)
    seed: int = 0
    note: str = ""

    def build(self) -> list:
        """Trace every program of the population (identical specs)."""
        return [evoxc.trace(fn, *self.specs) for fn in self.fns]


@dataclass(frozen=True)
class DeferralCase:
    """One documented v1 deferral: both engines must raise `expected`.

    Attributes:
        name: Stable case id.
        fns: The per-function callables of the (rejected) population.
        specs: The shared argument specs.
        options: The public keyword arguments that trigger the deferral.
        expected: The public error class both engines must raise.
        note: A one-line description of the deferred capability.
    """

    name: str
    fns: Tuple[Callable, ...]
    specs: Tuple[Any, ...]
    options: dict = field(default_factory=dict)
    expected: type = core.TransformError
    note: str = ""

    def build(self) -> list:
        return [evoxc.trace(fn, *self.specs) for fn in self.fns]


# ---------------------------------------------------------------------------
# Populations
# ---------------------------------------------------------------------------
def identity_program(x):
    """``(x * 2) + 2`` — the zero-select baseline (used by the cost pins)."""
    return evoxc.add(evoxc.multiply(x, 2.0), 2.0)


def divergent_topology_fns() -> Tuple[Callable, ...]:
    """Four rewrites of the same elementwise chain — divergent op topology."""
    return (
        lambda x: evoxc.add(evoxc.multiply(x, 2.0), 1.0),
        lambda x: evoxc.multiply(evoxc.add(x, 1.0), 2.0),
        lambda x: evoxc.multiply(x, 3.0),
        lambda x: evoxc.subtract(x, 1.0),
    )


def switch_variant_fns() -> Tuple[Callable, ...]:
    """One skeleton, three elementwise variants: add / subtract / multiply."""
    return (
        lambda x: evoxc.add(x, 1.0),
        lambda x: evoxc.subtract(x, 1.0),
        lambda x: evoxc.multiply(x, 2.0),
    )


def shared_constant_fns() -> Tuple[Callable, ...]:
    """Three identical programs: their ``1.0`` constant is shared, never stacked."""
    return (lambda x: evoxc.add(x, 1.0),) * 3


def stacked_constant_fns() -> Tuple[Callable, ...]:
    """Three programs with a DIFFERENT ``c`` each: the payload stacks into ``(3, ...)``."""
    return tuple((lambda c: (lambda x: evoxc.add(x, c)))(float(index)) for index in range(3))


def two_group_fns() -> Tuple[Callable, ...]:
    """5 × ``(x + c) + c`` + 5 × ``(x * c) - c`` over one ``(1024,)`` spec.

    The two structural groups give two fingerprints, which is what the
    ``bands="auto"`` cost model prices at 39168.0 against a flat 43136.0.
    """
    def group_a(constant):
        def fn(x):
            return evoxc.add(evoxc.add(x, constant), constant)

        return fn

    def group_b(constant):
        def fn(x):
            return evoxc.subtract(evoxc.multiply(x, constant), constant)

        return fn

    return tuple(group_a(float(index + 1)) for index in range(5)) + tuple(
        group_b(float(index + 1)) for index in range(5)
    )


def dict_input_fns() -> Tuple[Callable, ...]:
    """Two programs over a dict pytree: ``x`` is rowwise, ``w`` is shared."""
    return (
        lambda d: evoxc.add(evoxc.multiply(d["x"], 2.0), d["w"]),
        lambda d: evoxc.subtract(evoxc.multiply(d["x"], 3.0), d["w"]),
    )


def multi_output_fns() -> Tuple[Callable, ...]:
    """Two 2-output LIST programs (a list output tree takes ``out_axes=[0, 0]``)."""
    return (
        lambda x: [evoxc.add(evoxc.multiply(x, 2.0), 1.0), evoxc.multiply(x, 3.0)],
        lambda x: [evoxc.subtract(evoxc.multiply(x, 4.0), 1.0), evoxc.multiply(x, 5.0)],
    )


def multi_output_unmapped_fns() -> Tuple[Callable, ...]:
    """A mapped output plus an UNMAPPED one (``sum`` over the shared input)."""
    return (
        lambda x, w: [evoxc.add(x, 1.0), evoxc.sum(w)],
        lambda x, w: [evoxc.subtract(x, 1.0), evoxc.sum(w)],
    )


def sort_program(x):
    """`sort` carries no batching rule in v1 (the ruleless-op deferral)."""
    return evoxc.sort(x)


def cond_program(x):
    """A region-bearing control-flow body is never batchable (v1)."""
    return evoxc.cond(
        evoxc.sum(x) > 0.0,
        lambda value: value * 2.0,
        lambda value: value * -2.0,
        x,
    )


# ---------------------------------------------------------------------------
# The accepted corpus
# ---------------------------------------------------------------------------
IDENTICAL_SPECS = (spec((8,)),)
SWITCH_SPECS = (spec((6,)),)
CONSTANT_SPECS = (spec((4,)),)
TWO_GROUP_SPEC = (spec((1024,)),)

CASES: list = [
    BatchCase(
        "identical_programs",
        tuple(identity_program for _ in range(5)),
        IDENTICAL_SPECS,
        {},
        seed=1,
        note="5 identical (x*2)+2 programs — the zero-select baseline",
    ),
    BatchCase(
        "divergent_topology",
        divergent_topology_fns(),
        IDENTICAL_SPECS,
        {"alignment": "occurrence"},
        seed=2,
        note="four rewrites of one chain — output muxing, occurrence alignment",
    ),
    BatchCase(
        "switch_variant_alignment",
        switch_variant_fns(),
        SWITCH_SPECS,
        {"switch_ops": True, "alignment": "variant"},
        seed=3,
        note="one skeleton, add/subtract/multiply variants, variant alignment",
    ),
    BatchCase(
        "switch_occurrence_alignment",
        switch_variant_fns(),
        SWITCH_SPECS,
        {"switch_ops": True, "alignment": "occurrence"},
        seed=4,
        note="the same switch population under occurrence alignment",
    ),
    BatchCase(
        "switch_auto_alignment",
        switch_variant_fns(),
        SWITCH_SPECS,
        {"switch_ops": True, "alignment": "auto"},
        seed=5,
        note="the same switch population under auto alignment",
    ),
    BatchCase(
        "switch_explicit_ops",
        (lambda x: evoxc.add(x, 1.0), lambda x: evoxc.subtract(x, 1.0)),
        SWITCH_SPECS,
        {"switch_ops": [["add", "subtract"]]},
        seed=6,
        note="an explicit op-name merge group",
    ),
    BatchCase(
        "switch_category",
        (lambda x: evoxc.add(x, 1.0), lambda x: evoxc.multiply(x, 2.0)),
        SWITCH_SPECS,
        {"switch_ops": ["elementwise"]},
        seed=7,
        note="a category-only merge group (a one-element list holding the name)",
    ),
    BatchCase(
        "shared_constant",
        shared_constant_fns(),
        CONSTANT_SPECS,
        {},
        seed=8,
        note="one constant shared by every program (never stacked)",
    ),
    BatchCase(
        "stacked_constant",
        stacked_constant_fns(),
        CONSTANT_SPECS,
        {},
        seed=9,
        note="a different constant per program (one stacked payload)",
    ),
    BatchCase(
        "bands_flat",
        two_group_fns(),
        TWO_GROUP_SPEC,
        {"bands": None},
        seed=10,
        note="the two-group 10-function population, flat merge",
    ),
    BatchCase(
        "bands_one",
        two_group_fns(),
        TWO_GROUP_SPEC,
        {"bands": 1},
        seed=11,
        note="bands=1 — the flat merge with explicit band fields",
    ),
    BatchCase(
        "bands_two",
        two_group_fns(),
        TWO_GROUP_SPEC,
        {"bands": 2},
        seed=12,
        note="two explicit bands (5 + 5)",
    ),
    BatchCase(
        "bands_five",
        two_group_fns(),
        TWO_GROUP_SPEC,
        {"bands": 5},
        seed=13,
        note="five explicit bands of two",
    ),
    BatchCase(
        "bands_auto",
        two_group_fns(),
        TWO_GROUP_SPEC,
        {"bands": "auto"},
        seed=14,
        note="auto partition — pinned at 39168.0 <= flat 43136.0",
    ),
    BatchCase(
        "bands_auto_identical",
        tuple(identity_program for _ in range(5)),
        IDENTICAL_SPECS,
        {"switch_ops": True, "bands": "auto"},
        seed=15,
        note="auto keeps identical programs in one band — pinned 1232.0 / 1232.0",
    ),
    BatchCase(
        "axes_none_shared",
        (lambda x: evoxc.add(x, 1.0), lambda x: evoxc.multiply(x, 2.0)),
        CONSTANT_SPECS,
        {"axes": None},
        seed=16,
        note="an unmapped (shared) input with a broadcast output",
    ),
    BatchCase(
        "axes_dict_mixed",
        dict_input_fns(),
        ({"x": spec((4,)), "w": spec((4,))},),
        {"axes": ({"x": 0, "w": None},)},
        seed=17,
        note="a dict input tree: x rowwise, w shared",
    ),
    BatchCase(
        "multi_output_mapped",
        multi_output_fns(),
        CONSTANT_SPECS,
        {"out_axes": [0, 0]},
        seed=18,
        note="a 2-output list tree, both outputs on the function axis",
    ),
    BatchCase(
        "multi_output_unmapped_second",
        multi_output_unmapped_fns(),
        (spec((4,)), spec((3,))),
        {"axes": (0, None), "out_axes": [0, None]},
        seed=19,
        note="a mapped first output and an unmapped second output",
    ),
]


# ---------------------------------------------------------------------------
# The documented v1 deferrals (both engines raise `expected`)
# ---------------------------------------------------------------------------
DEFERRALS: list = [
    DeferralCase(
        "defer_bands_with_unmapped_output",
        multi_output_unmapped_fns(),
        (spec((4,)), spec((3,))),
        {"axes": (0, None), "bands": 2, "out_axes": [0, None]},
        note="bands != None rejects an unmapped tensor output",
    ),
    DeferralCase(
        "defer_non_leading_axes",
        (lambda x: evoxc.add(x, 1.0), lambda x: evoxc.multiply(x, 2.0)),
        (spec((3, 4)),),
        {"axes": (1,)},
        note="a non-zero axes entry is a v1 deferral",
    ),
    DeferralCase(
        "defer_non_leading_out_axes",
        (lambda x: evoxc.add(x, 1.0), lambda x: evoxc.multiply(x, 2.0)),
        (spec((3, 4)),),
        {"out_axes": 1},
        note="a non-zero out_axes entry is a v1 deferral",
    ),
    DeferralCase(
        "defer_out_axes_none_on_mapped_output",
        (lambda x: evoxc.add(x, 1.0), lambda x: evoxc.multiply(x, 2.0)),
        CONSTANT_SPECS,
        {"out_axes": None},
        note="out_axes=None never silently drops a mapped function axis",
    ),
    DeferralCase(
        "defer_sort_without_batching_rule",
        (sort_program, sort_program),
        (spec((5,)),),
        {},
        note="sort has no batching rule in v1",
    ),
    DeferralCase(
        "defer_cond_region_bearing",
        (cond_program, cond_program),
        (spec((3,)),),
        {},
        note="region-bearing control flow is never batchable",
    ),
]
