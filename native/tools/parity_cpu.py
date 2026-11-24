"""Numpy-parity harness: the native CPU engine vs the numpy backend.

Every case is a graph plus concrete numpy inputs; the SAME graph runs twice::

    fn --trace--> Graph --encode_module--> envelope --native_compile--> run --+
                                                                             +--> compare
    fn --evoxc.evaluate(fn, *inputs, backend="numpy")  (or the explicit
         lower/compile/load/run pipeline for a hand-built Graph) -------------->

Run it (needs the built extension — ``scripts/build-native.sh``)::

    source scripts/dev-env.sh
    .venv/bin/python native/tools/parity_cpu.py [--filter SUBSTR] [--verbose]

Comparison contract
-------------------
What is ACTUALLY asserted, leaf by leaf:
* Output-leaf count, order (the entry function's return operands), dtype and
  shape must match EXACTLY.
* bool / integer leaves: ``np.array_equal`` (exact).  NaN positions must match
  exactly wherever NaN is produced.
* float / complex leaves: ``np.allclose`` with ``rtol=1e-5, atol=1e-6`` for
  f16/f32/c64 and ``rtol=1e-12, atol=1e-13`` for f64/c128, plus an exact
  NaN-position match and INF agreement (same positions, same sign).  The table
  always reports the max absolute and max relative error (``-`` = an
  exact-comparison case).  f16 is held to the f32 tolerance: its own epsilon
  (~1e-3) would otherwise fail on rounding alone.
* FLOAT BIT-IDENTITY IS NOT ASSERTED HERE (and cannot be for every leaf).  It is
  asserted ONLY for the cases that flag it explicitly: the ``exact`` key-based
  RNG cases compare floats bit-for-bit with ``np.array_equal`` (the algorithm is
  documented bit-identical against numpy).  Two families legitimately differ in
  their last bits and would fail a bit-exact contract: an f32 ``dot`` (faer's
  gemm reassociates the K-sum differently from numpy's threaded OpenBLAS
  ``sgemm`` — ~5e-5 relative where cancellation leaves a near-zero result) and
  any transcendental elementwise op (``sin``/``cos``/``tan``/``tanh``/``exp``/
  ``log``/``log1p``/``erf``/``gelu`` call the platform libm, while numpy uses
  its own SIMD kernels — 1-2 ulp windows).
* ``known_roundoff: (rtol, atol)`` on a case REPLACES the default float
  tolerance for that case's tolerance-mode (float/complex) leaves and marks the
  row ``PASS~``, the same visible flag style as ``rng_normal`` below.  It exists
  for a measured, understood round-off gap rather than a defect:
  ``dot_f32_large`` uses ``(1e-4, 1e-5)`` (measured: 580/3072 elements differ,
  max abs 9.5e-6), while ``transcendental_f32`` runs on the DEFAULT tolerance
  (measured: ≤1-2 ulp).
* ``rng_normal`` relaxes ONLY the f32 ``normal`` leaf to the tolerance path
  (the engine documents a 1-ulp f32 window) and marks the row ``PASS~``.
* A case whose NUMPY side raises passes when the native side raises too
  (error parity; both families are printed) and fails when it does not.  A
  case may declare ``native_error=(kind, ...)``: "native must raise this kind
  while numpy succeeds" (input validation, the resolver preflight).
* Documented divergence (not a failure): with ``stable=False`` and DUPLICATE
  values the native engine returns the stable index order, while the numpy
  backend returns numpy's quicksort order — the native kernel documents one
  stable code path for both kinds (numpy's unstable tie order is unspecified).
  The corpus therefore pins tie behaviour with ``stable=True`` and keeps a
  tie-free ``stable=False`` case.

Exit: 0 all passed, 1 at least one failure, 2 the extension is not built.
The last lines are::

    PARITY: <cases> cases, <passed> passed, <failed> failed, <skipped> skipped
    PARITY: <per-category passed/total>
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

import evoxc
from evoxc import core, ir
from evoxc._native_codec import encode_module

#: The envelope's widest integer (see :func:`reduce_wide_int_attrs`).
U64_MAX = 2**64 - 1
#: (rtol, atol) per floating dtype family (see the module docstring).
TOL_F32 = (1e-5, 1e-6)
TOL_F64 = (1e-12, 1e-13)


# ---------------------------------------------------------------------------
# Cases and outcomes
# ---------------------------------------------------------------------------


@dataclass
class Case:
    """One graph plus its runtime inputs.

    ``fn`` is normally an ``@evoxc.defn`` traced with ``specs``; ``graph``
    overrides the trace with a pre-built ``Graph`` (the ``call`` case, whose
    two-function module has no frontend builder).
    """

    name: str
    category: str
    fn: Optional[Callable[..., Any]]
    specs: tuple = ()
    inputs: tuple = ()
    graph: Any = None
    #: ``exact`` = floats bit-exact; ``rng_normal`` = the documented 1-ulp f32
    #: ``normal`` window (tolerance pass, flagged ``PASS~``). Never both on one
    #: leaf: ``rng_normal`` relaxes only the f32 leaves.
    flags: frozenset = frozenset()
    #: ``(rtol, atol)`` REPLACING the default float tolerance for this case's
    #: tolerance-mode leaves, for a measured round-off gap (see the module
    #: docstring); the row prints ``PASS~``. Ignored by bit-exact leaves.
    known_roundoff: Optional[tuple] = None
    #: Error kinds the NATIVE side must raise while the numpy side succeeds.
    native_error: tuple = ()
    #: Optional resolver passed to ``NativeProgram.run``.
    resolver: Any = None
    note: str = ""


@dataclass
class Outcome:
    case: Case
    status: str  # PASS / PASS~ (tolerance-flagged) / FAIL
    detail: str = ""
    native_desc: str = "-"
    max_abs: Optional[float] = None
    max_rel: Optional[float] = None


# ---------------------------------------------------------------------------
# Envelope preparation
# ---------------------------------------------------------------------------


def reduce_wide_int_attrs(envelope: dict) -> dict:
    """Reduce an ``int``-tagged attribute wider than ``u64`` modulo ``2**64``.

    The frontend stamps the per-op RNG salts as plain Python ints; a single
    salt (the SplitMix64 golden gamma ``0x9E3779B97F4A7C15``) now travels
    verbatim, because ``evoxc-core`` decodes an ``int`` attribute with
    ``as_u64`` (``as_i64`` fallback) and the engine reads it back through an
    ``as u64`` cast.

    ``random_split_n`` derives its salts as ``i * 0x9E3779B97F4A7C15``, which
    for ``i >= 2`` exceeds ``u64`` (``2 * GOLDEN`` = 22801429638646396970) and
    so has no 64-bit envelope encoding at all.  The numpy backend applies the
    salt with ``np.uint64`` arithmetic, i.e. modulo ``2**64``, so reducing here
    is the same 64-bit salt and keeps the comparison honest.  Harness-side
    workaround for a frontend encoding gap — the fix belongs to
    ``evoxc/ops/random.py`` (out of this task's write scope).
    """
    from evoxc.ir.op_defs import opdef
    for op in envelope.get("ops", ()):
        attrs = op.get("attributes") or {}
        if not attrs:
            continue
        tags = {spec.name: spec.type for spec in opdef(op["name"]).attributes}
        for key, value in attrs.items():
            if (
                tags.get(key) == "int"
                and isinstance(value, int)
                and not isinstance(value, bool)
                and value > U64_MAX
            ):
                attrs[key] = value % (1 << 64)
    return envelope
def encode_graph(graph) -> dict:
    """The native envelope for ``graph``, with the >u64 attribute reduction."""
    return reduce_wide_int_attrs(encode_module(graph.module))


# ---------------------------------------------------------------------------
# Running the two sides
# ---------------------------------------------------------------------------


def _error_kind(exc: BaseException) -> str:
    """The public error family: the evoxc class name or ``NativeError.kind``."""
    kind = getattr(exc, "kind", None)
    return kind if isinstance(kind, str) else type(exc).__name__


def _describe(exc: BaseException) -> str:
    lines = str(exc).strip().splitlines()
    return f"{_error_kind(exc)}: {lines[0][:120] if lines else ''}"


def _flatten_outputs(value: Any) -> list:
    """Flatten a pipeline output pytree into ndarrays (leaves, in order)."""
    return [np.asarray(leaf.numpy()) for leaf in core.tree_leaves(value)]


def run_numpy(case: Case) -> list:
    """The numpy-backend reference (``evoxc.evaluate`` / the explicit pipeline)."""
    if case.graph is not None:
        lowered = evoxc.lower(case.graph, backend="numpy")
        artifact = evoxc.compile(lowered, backend="numpy")
        executable = evoxc.load(artifact, backend="numpy", device="cpu")
        return _flatten_outputs(evoxc.run(executable, *case.inputs))
    return _flatten_outputs(evoxc.evaluate(case.fn, *case.inputs, backend="numpy"))


def compile_native(extension, case: Case):
    """Trace (or take) the graph and native-compile it."""
    graph = case.graph if case.graph is not None else evoxc.trace(case.fn, *case.specs)
    return extension.native_compile(encode_graph(graph))


def run_native(program, case: Case) -> list:
    flat = [np.ascontiguousarray(value) for value in case.inputs]
    return [np.asarray(out) for out in program.run(flat, resolver=case.resolver)]


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _compare_float(
    native: np.ndarray,
    ref: np.ndarray,
    dtype: np.dtype,
    override: Optional[tuple] = None,
) -> tuple:
    # A per-case override replaces the dtype-family default (the measured
    # round-off gaps in the module docstring); nothing else changes.
    if override is not None:
        rtol, atol = override
    else:
        rtol, atol = TOL_F64 if dtype.itemsize >= 8 else TOL_F32
    nan_mismatch = int(np.count_nonzero(np.isnan(native) ^ np.isnan(ref)))
    inf_mismatch = int(
        np.count_nonzero(np.isinf(native) ^ np.isinf(ref))
        + np.count_nonzero(np.isinf(native) & np.isinf(ref) & (native != ref))
    )
    finite = np.isfinite(native) & np.isfinite(ref)
    max_abs = max_rel = 0.0
    if finite.any():
        delta = np.abs(native[finite] - ref[finite])
        max_abs = float(delta.max())
        scale = np.abs(ref[finite])
        rel = np.divide(delta, scale, out=np.zeros_like(delta), where=scale > 0)
        max_rel = float(rel.max()) if rel.size else 0.0
    ok = (
        nan_mismatch == 0
        and inf_mismatch == 0
        and bool(np.allclose(native, ref, rtol=rtol, atol=atol, equal_nan=True))
    )
    detail = ""
    if nan_mismatch:
        detail += f"{nan_mismatch} NaN-position mismatch; "
    if inf_mismatch:
        detail += f"{inf_mismatch} inf mismatch; "
    if not ok and not detail:
        detail = f"tolerance rtol={rtol} atol={atol} exceeded"
    return ok, max_abs, max_rel, detail


def _leaf_mode(case: Case, dtype: np.dtype) -> str:
    """``exact`` (bit-exact) or ``tolerance`` for a floating leaf."""
    if "exact" not in case.flags:
        return "tolerance"
    if "rng_normal" in case.flags and dtype == np.dtype("float32"):
        return "tolerance"
    return "exact"


def _compare_leaf(native: np.ndarray, ref: np.ndarray, case: Case) -> tuple:
    """Compare one output leaf → ``(ok, max_abs, max_rel, detail)``."""
    if native.dtype != ref.dtype:
        return False, None, None, f"dtype {native.dtype} != {ref.dtype}"
    if native.shape != ref.shape:
        return False, None, None, f"shape {native.shape} != {ref.shape}"
    kind = native.dtype.kind
    if kind not in "fc":
        ok = bool(np.array_equal(native, ref))
        return ok, (0.0 if ok else None), (0.0 if ok else None), "" if ok else "values differ (exact)"
    if _leaf_mode(case, native.dtype) == "exact":
        ok = bool(np.array_equal(native, ref, equal_nan=True))
        return ok, (0.0 if ok else None), (0.0 if ok else None), "" if ok else "values differ (bit-exact)"
    return _compare_float(native, ref, native.dtype, case.known_roundoff)


def _describe_outputs(outputs: Sequence[np.ndarray]) -> str:
    parts = [f"{out.dtype.name}({', '.join(str(d) for d in out.shape)})" for out in outputs]
    return " + ".join(parts) if parts else "<no outputs>"


def compare(case: Case, native: list, ref: list) -> Outcome:
    if len(native) != len(ref):
        return Outcome(case, "FAIL", f"output count {len(native)} != {len(ref)}", _describe_outputs(native))
    max_abs: Optional[float] = 0.0
    max_rel: Optional[float] = 0.0
    problems = []
    for index, (got, want) in enumerate(zip(native, ref)):
        ok, abs_err, rel_err, detail = _compare_leaf(got, want, case)
        max_abs = None if abs_err is None or max_abs is None else max(max_abs, abs_err)
        max_rel = None if rel_err is None or max_rel is None else max(max_rel, rel_err)
        if not ok:
            problems.append(f"output[{index}]: {detail}")
    roundoff_flagged = case.known_roundoff is not None or "rng_normal" in case.flags
    status = "FAIL" if problems else ("PASS~" if roundoff_flagged else "PASS")
    return Outcome(case, status, "; ".join(problems), _describe_outputs(native), max_abs, max_rel)


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------


def check_case(extension, case: Case) -> Outcome:
    """Run both sides and classify the case (never swallows an error)."""
    numpy_error: Optional[BaseException] = None
    ref: Optional[list] = None
    try:
        ref = run_numpy(case)
    except Exception as exc:  # noqa: BLE001 - both sides are compared, so an
        numpy_error = exc  # exception is data here, not a swallowed failure.

    native_error: Optional[BaseException] = None
    native: Optional[list] = None
    try:
        native = run_native(compile_native(extension, case), case)
    except Exception as exc:  # noqa: BLE001
        native_error = exc

    if numpy_error is not None and native_error is not None:
        return Outcome(
            case,
            "PASS",
            f"error parity: numpy {_describe(numpy_error)} | native {_describe(native_error)}",
        )
    if numpy_error is not None:
        return Outcome(case, "FAIL", f"numpy side raised without a native match: {_describe(numpy_error)}")
    if native_error is not None:
        kind = _error_kind(native_error)
        if kind in case.native_error:
            return Outcome(case, "PASS", f"native raised {_describe(native_error)}")
        expected = f" (expected {case.native_error})" if case.native_error else ""
        return Outcome(case, "FAIL", f"native side raised {_describe(native_error)}{expected}")
    return compare(case, native, ref or [])


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------


def _spec(shape, dtype=evoxc.float32) -> "evoxc.TensorSpec":
    return evoxc.TensorSpec(tuple(shape), dtype)


def _resolve(rng, shape, dtype):
    """A deterministic input array of ``shape``/``dtype``."""
    dtype = np.dtype(dtype.name)
    if dtype.kind == "c":
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    if dtype.kind in "iu":
        return rng.integers(-8, 9, shape).astype(dtype)
    return rng.uniform(-1.5, 1.5, shape).astype(dtype)


def _build_call_graph(template_fn, template_specs):
    """A two-function module (``helper`` + ``main`` calling it) as a Graph.

    A ``Defn`` is deliberately not callable and the frontend has no ``call``
    builder, so the module is built with ``ir.Builder`` and attached to a
    freshly traced Graph of the same signature — every other case gets that
    same module + pytree wiring from ``evoxc.trace``.
    """
    builder = ir.Builder()
    builder.build_module("call_graph")

    def value_type(shape):
        return ir.ValueType(core.dtype("float32"), tuple(shape))

    helper = builder.build_function("helper", (value_type((3,)),))
    (helper_arg,) = helper.entry_block.arguments
    two = builder.create("constant", (), attributes={"value": np.array(2.0, np.float32)})
    doubled = builder.create("multiply", (helper_arg, two.results[0]))
    builder.set_terminator(helper.entry_block, "return", (doubled.results[0],))

    main = builder.build_function("main", (value_type((3,)),))
    (main_arg,) = main.entry_block.arguments
    called = builder.create("call", (main_arg,), attributes={"callee": "helper"})
    total = builder.create("add", (called.results[0], main_arg))
    builder.set_terminator(main.entry_block, "return", (total.results[0],))

    module = builder.module
    module.functions = [f for f in module.functions if f.name == "main"] + [
        f for f in module.functions if f.name != "main"
    ]
    graph = evoxc.trace(template_fn, *template_specs)
    graph.module = module
    return graph


class _ExternalResolver:
    """The host-callback resolver the native engine needs for ``external_call``."""

    @staticmethod
    def call_external(name, args):
        if name != "parity_double":
            raise core.BackendError(f"resolver: unknown external kernel {name!r}")
        return [np.asarray(args[0]) * 2.0]

    call_block = call_external
    call_runtime = call_external


# --- the graphs -------------------------------------------------------------


ew_chain = evoxc.defn(lambda x, y: evoxc.add(evoxc.multiply(x, y), evoxc.power(x, 2.0)))

ew_max_nan = evoxc.defn(lambda x, y: evoxc.maximum(x, y))

ew_rounding = evoxc.defn(lambda x: evoxc.add(evoxc.add(evoxc.round(x), evoxc.floor(x)), evoxc.ceil(x)))

ew_abs_complex = evoxc.defn(lambda x: evoxc.abs(x))

ew_activations = evoxc.defn(lambda x: evoxc.add(evoxc.erf(x), evoxc.add(evoxc.gelu(x), evoxc.add(evoxc.sigmoid(x), evoxc.tanh(x)))))

ew_casts = evoxc.defn(lambda x, i: (evoxc.cast(x, evoxc.bool_), evoxc.cast(x, evoxc.int32), evoxc.cast(i, evoxc.float32), evoxc.cast(i, evoxc.float64)))

ew_select = evoxc.defn(lambda x, y: evoxc.select(evoxc.less(x, y), x, y))

ew_remainder = evoxc.defn(lambda x, y: evoxc.remainder(x, y))

@evoxc.defn
def ew_int_divide(x, y):
    # Integral operands promote to float64 (the documented true-division rule),
    # so this is a VALUE case, not an error case.
    return evoxc.divide(x, y)


ew_int_power_negative = evoxc.defn(lambda x, y: evoxc.power(x, y))

@evoxc.defn
def red_axis(x):
    return (
        evoxc.reduce_sum(x, axes=(1,)),
        evoxc.reduce_mean(x, axes=(0,), keepdims=True),
        evoxc.reduce_max(x, axes=None),
        evoxc.reduce_min(x, axes=(0, 1), keepdims=True),
    )


red_prod = evoxc.defn(lambda x: (evoxc.reduce_prod(x, axes=(1,)), evoxc.reduce_prod(x, axes=None, keepdims=True)))

red_arg_ties = evoxc.defn(lambda x: (evoxc.argmax(x, axis=1), evoxc.argmin(x, axis=1), evoxc.argmax(x), evoxc.argmin(x, axis=1, keepdims=True)))

@evoxc.defn
def red_cum(x):
    return (
        evoxc.cumsum(x, axis=0),
        evoxc.cumsum(x, axis=1, reverse=True),
        evoxc.cumprod(x, axis=0),
        evoxc.cumprod(x, axis=1, reverse=True),
    )


idx_reshape = evoxc.defn(lambda x: (evoxc.reshape(x, (3, 4)), evoxc.reshape(x, (4, 3))))

idx_transpose = evoxc.defn(lambda x: (evoxc.transpose(x), evoxc.transpose(x, (2, 0, 1))))

idx_slice = evoxc.defn(lambda x: (evoxc.slice(x, 1, (2, 3), (2, 1)), evoxc.slice(x, (0, 0), (4, 3), (1, 2))))

idx_slice_negative_stride = evoxc.defn(lambda x: evoxc.slice(x, 0, (4,), (-1,)))

idx_gather = evoxc.defn(lambda x, i0, i1: (evoxc.gather(x, i0, axis=0), evoxc.gather(x, i1, axis=1)))

idx_scatter = evoxc.defn(lambda x, indices, updates: evoxc.scatter(x, indices, updates, axis=0))

idx_concatenate = evoxc.defn(lambda x, y: evoxc.concatenate([x, y], axis=1))

idx_tile = evoxc.defn(lambda x: evoxc.tile(x, (2, 3)))

idx_flip = evoxc.defn(lambda x: (evoxc.flip(x, (0,)), evoxc.flip(x, (0, 1))))
# Nullable-attribute graphs: every `None` parameter below is serialized as an
# explicit JSON `null` on the op (the registry's documented `default=None`).
ew_transcendental = evoxc.defn(
    lambda x: (evoxc.sin(x), evoxc.cos(x), evoxc.tanh(x), evoxc.exp(x))
)
ew_nan_to_num = evoxc.defn(lambda x: evoxc.nan_to_num(x))
ew_nan_to_num_explicit = evoxc.defn(lambda x: evoxc.nan_to_num(x, nan=7, posinf=1.0, neginf=-1.0))
idx_flip_all_axes = evoxc.defn(lambda x: evoxc.flip(x, None))
idx_roll_flat = evoxc.defn(lambda x: evoxc.roll(x, 3))
idx_slice_default_strides = evoxc.defn(lambda x: evoxc.slice(x, (0, 1), (2, 3)))
red_argmax_flat = evoxc.defn(lambda x: (evoxc.argmax(x), evoxc.argmin(x)))


idx_roll = evoxc.defn(lambda x: (evoxc.roll(x, 2), evoxc.roll(x, (1, 2), axis=(0, 1))))

idx_pad = evoxc.defn(lambda x: evoxc.pad(x, ((1, 2), (0, 3)), value=1.5))

idx_diag = evoxc.defn(lambda x: evoxc.diag(x))

idx_tril_triu = evoxc.defn(lambda x: (evoxc.tril(x, -1), evoxc.triu(x, 1)))

idx_diagonal = evoxc.defn(lambda x: evoxc.diagonal(x, offset=1))

lin_dot = evoxc.defn(lambda a, b: evoxc.dot(a, b))

lin_conv_valid = evoxc.defn(lambda x, w: evoxc.conv(x, w, padding="VALID"))

lin_conv_same = evoxc.defn(lambda x, w: evoxc.conv(x, w, padding="SAME"))

lin_solve = evoxc.defn(lambda a, b: evoxc.solve(a, b))

lin_cholesky = evoxc.defn(lambda a: evoxc.cholesky(a))

@evoxc.defn
def lin_qr(x):
    q, r = evoxc.qr(x)
    return evoxc.dot(q, r)  # reconstruct: the factors are sign-ambiguous, x is not


@evoxc.defn
def lin_svd(x):
    u, s, vh = evoxc.svd(x)
    return evoxc.dot(u * s, vh)  # reconstruct: only u @ diag(s) @ vh is unique


@evoxc.defn
def lin_eigh(a):
    w, v = evoxc.eigh(a)
    return (w, evoxc.reduce_sum(evoxc.dot(a, v) - v * w, axes=None))


lin_matrix_rank = evoxc.defn(lambda x: evoxc.matrix_rank(x))

lin_matrix_exp = evoxc.defn(lambda a: evoxc.matrix_exp(a))

@evoxc.defn
def lin_sort(x):
    return (
        evoxc.sort(x, axis=-1, descending=True),
        evoxc.argsort(x, axis=1, stable=True),
        evoxc.sort(x, axis=0, stable=True),
    )


lin_argsort_no_ties = evoxc.defn(lambda x: (evoxc.argsort(x, axis=1), evoxc.argsort(x, axis=0, descending=True)))

@evoxc.defn
def cf_cond(x):
    pred = evoxc.less(evoxc.reduce_sum(x, axes=None), 0.0)
    return evoxc.cond(pred, lambda a: evoxc.add(a, 10.0), lambda a: evoxc.multiply(a, -2.0), x)


@evoxc.defn
def cf_while_counted(x):
    # The trip count is a RUNTIME value (x[0] units), so the engine's while
    # op must execute a data-dependent number of iterations.
    limit = evoxc.reduce_sum(evoxc.slice(x, 0, 1), axes=None)

    def cond_fn(carried):
        i, _ = carried
        return evoxc.less(i, limit)

    def body(carried):
        i, acc = carried
        return (evoxc.add(i, 1.0), evoxc.add(acc, x))

    return evoxc.while_loop(cond_fn, body, (evoxc.multiply(limit, 0.0), x))


@evoxc.defn
def cf_while_consumes(x):
    def cond_fn(carried):
        i, _, _ = carried
        return evoxc.less(i, 6.0)

    def body(carried):
        i, acc, w = carried
        return (evoxc.add(i, 1.0), evoxc.add(evoxc.multiply(acc, 0.5), w), evoxc.multiply(w, 0.9))

    zero = evoxc.multiply(evoxc.reduce_sum(x, axes=None), 0.0)
    return evoxc.while_loop(cond_fn, body, (zero, evoxc.multiply(x, 0.0), x))


cf_call_template = evoxc.defn(lambda x: evoxc.add(x, x))


@evoxc.defn
def cf_cond_with_while(x):
    def cond_fn(carried):
        i, _ = carried
        return evoxc.less(i, 4.0)

    def body(carried):
        i, acc = carried
        return (evoxc.add(i, 1.0), evoxc.add(acc, 0.25))

    def true_fn(a):
        carried = evoxc.while_loop(cond_fn, body, (evoxc.multiply(evoxc.reduce_sum(x, axes=None), 0.0), a))
        return carried[1]

    pred = evoxc.greater(evoxc.reduce_sum(x, axes=None), 0.0)
    return evoxc.cond(pred, true_fn, lambda a: evoxc.multiply(a, 3.0), x)

def _rng_graph(algorithm: str, kind: str):
    """One RNG graph per (algorithm, kind) — keys / sampling / normal."""

    @evoxc.defn
    def keys():
        # A key is a plain integer tensor whose static shape encodes the
        # algorithm (splitmix64: rank-0, threefry2x32: (2,), philox: (4,)),
        # so split/split_n rounds need no inputs and compare bit-exactly.
        key = evoxc.random.key(7, algorithm=algorithm)
        a, b = evoxc.random.split(key)
        c, d = evoxc.random.split(a)
        stacked = evoxc.stack(evoxc.random.split_n(b, 3))
        return (key, a, b, c, d, stacked, evoxc.reduce_sum(stacked, axes=None))

    @evoxc.defn
    def sampling(probs):
        key = evoxc.random.key(11, algorithm=algorithm)
        return (
            evoxc.random.uniform(key, (6,), dtype=evoxc.float32),
            evoxc.random.uniform(key, (4,), low=-2.0, high=5.0, dtype=evoxc.float64),
            evoxc.cast(evoxc.random.randint(key, (5,), 0, 17), evoxc.int64),
            evoxc.cast(evoxc.random.permutation(key, 8), evoxc.int64),
            evoxc.random.multinomial(key, probs, 4),
        )

    @evoxc.defn
    def normal():
        key = evoxc.random.key(3, algorithm=algorithm)
        return (
            evoxc.random.normal(key, (8,), dtype=evoxc.float32),
            evoxc.random.normal(key, (8,), mean=2.0, std=0.5, dtype=evoxc.float64),
        )

    return {"keys": keys, "sampling": sampling, "normal": normal}[kind]


def build_cases() -> list:
    """The parity corpus — one :class:`Case` per graph, grouped by category.

    Each row is ``(name, category, fn, specs[, extra])``; ``extra`` may carry
    ``inputs`` (else random concrete inputs are derived from ``specs``),
    ``graph``, ``flags``, ``native_error``, ``resolver`` and ``note``.
    """
    rng = np.random.default_rng(20260101)
    # A SEPARATE seed for the round-off cases added below, so their (large)
    # draws cannot shift the shared `rng` sequence every other case derives from.
    rng_roundoff = np.random.default_rng(20260202)
    f32, f64, c64 = evoxc.float32, evoxc.float64, evoxc.complex64
    i32, i64 = evoxc.int32, evoxc.int64
    S = _spec  # S((3, 4)) / S((5,), i64)
    rows: list = []

    def rnd(*shape, dtype=f32):
        return _resolve(rng, shape, dtype)

    # ---- 1. elementwise ---------------------------------------------------
    nan_x = np.array([[np.nan, 1.0, -2.0, 3.5], [0.5, np.nan, 7.0, -1.0]], np.float32)
    nan_inf = np.array([np.nan, np.inf, -np.inf, 1.0, -0.5], np.float32)
    nan_y = np.array([[2.0, np.nan, 1.0, -3.5], [np.nan, 2.5, -7.0, 1.0]], np.float32)
    rows += [
        ("ew_chain", "elementwise", ew_chain, (S((4, 5)), S((4, 5)))),
        ("ew_max_nan", "elementwise", ew_max_nan, (S((2, 4)), S((2, 4))), {"inputs": (nan_x, nan_y)}),
        ("ew_round_floor_ceil", "elementwise", ew_rounding, (S((7,)),)),
        ("ew_abs_complex", "elementwise", ew_abs_complex, (S((6,), c64),)),
        ("ew_activations", "elementwise", ew_activations, (S((6,)),)),
        ("ew_casts", "elementwise", ew_casts, (S((5,), f32), S((5,), i32)), {
            "inputs": (np.array([0.0, 0.5, -1.5, 2.7, -0.0], np.float32),
                       np.array([1, -2, 3, 4, -5], np.int32))}),
        ("ew_select", "elementwise", ew_select, (S((4, 5)), S((4, 5)))),
        ("ew_remainder_negative_divisor", "elementwise", ew_remainder, (S((5,), i64), S((5,), i64)), {
            "inputs": (np.array([-7, 7, -7, 7, 0], np.int64), np.array([3, -3, -3, 3, -5], np.int64))}),
        ("ew_int_divide", "elementwise", ew_int_divide, (S((3,), i32), S((3,), i32)), {
            "inputs": (np.array([7, -7, 0], np.int32), np.array([2, -2, 3], np.int32)),
            "note": "integral divide promotes to float64 (no error)"}),
        ("ew_nan_to_num_defaults", "elementwise", ew_nan_to_num, (S((5,)),), {
            "inputs": (nan_inf,),
            "note": "posinf/neginf arrive as JSON null: the dtype's max/min finite value"}),
        ("ew_nan_to_num_defaults_f16", "elementwise", ew_nan_to_num, (S((5,), evoxc.float16),), {
            "inputs": (nan_inf.astype(np.float16),), "note": "f16 dtype bounds"}),
        ("ew_nan_to_num_defaults_f64", "elementwise", ew_nan_to_num, (S((5,), f64),), {
            "inputs": (nan_inf.astype(np.float64),)}),
        ("ew_nan_to_num_explicit", "elementwise", ew_nan_to_num_explicit, (S((5,)),), {
            "inputs": (nan_inf,)}),
        ("err_int_power_negative", "elementwise", ew_int_power_negative, (S((2,), i64), S((2,), i64)), {
            "inputs": (np.array([2, 3], np.int64), np.array([-1, -2], np.int64)),
            "note": "integer power with a negative exponent: both sides raise"}),
        ("transcendental_f32", "elementwise", ew_transcendental, (S((4000,)),), {
            "inputs": (np.linspace(-0.9, 0.9, 4000, dtype=np.float32),),
            "note": "sin/cos/tanh/exp over 4000 samples: DEFAULT tolerance (measured ≤1-2 ulp, "
                    "platform libm vs numpy's SIMD kernels — bit-identity is not asserted)"}),
    ]

    # ---- 2. reductions ----------------------------------------------------
    rows += [
        ("red_axis_keepdims", "reductions", red_axis, (S((3, 4)),)),
        ("red_prod", "reductions", red_prod, (S((2, 5)),)),
        ("red_argmax_argmin_ties", "reductions", red_arg_ties, (S((3, 4), i32),), {
            "inputs": (np.array([[1, 3, 3, 2], [5, 5, 5, 5], [0, -1, 0, -1]], np.int32),)}),
        ("red_cumsum_cumprod", "reductions", red_cum, (S((4, 5)),)),
    ]

    # ---- 3. indexing / structure -----------------------------------------
    rows += [
        ("idx_reshape", "indexing", idx_reshape, (S((2, 6)),)),
        ("idx_transpose", "indexing", idx_transpose, (S((2, 3, 4)),)),
        ("idx_slice_strides", "indexing", idx_slice, (S((5, 6)),)),
        ("err_slice_negative_stride", "indexing", idx_slice_negative_stride, (S((5,)),),
         {"note": "negative strides are rejected at trace time by both sides"}),
        ("idx_gather", "indexing", idx_gather, (S((5, 4)), S((3,), i32), S((2,), i32)), {
            "inputs": (rnd(5, 4), np.array([4, 0, 2], np.int32), np.array([3, 1], np.int32))}),
        ("idx_scatter", "indexing", idx_scatter, (S((5, 4)), S((2,), i32), S((2, 4))), {
            "inputs": (rnd(5, 4), np.array([3, 0], np.int32), rnd(2, 4))}),
        ("idx_concatenate", "indexing", idx_concatenate, (S((3, 2)), S((3, 3)))),
        ("idx_tile", "indexing", idx_tile, (S((2, 3)),)),
        ("idx_flip", "indexing", idx_flip, (S((3, 4)),)),
        ("idx_flip_all_axes", "indexing", idx_flip_all_axes, (S((3, 4)),),
         {"note": "flip axes=null reverses every axis"}),
        ("idx_roll_flat", "indexing", idx_roll_flat, (S((3, 4)),),
         {"note": "roll axis=null rolls the flattened array"}),
        ("idx_slice_default_strides", "indexing", idx_slice_default_strides, (S((4, 5)),),
         {"note": "slice strides=null walks with stride 1"}),
        ("red_argmax_argmin_flat", "reductions", red_argmax_flat, (S((3, 4)),),
         {"note": "argmax/argmin axis=null reduce over the flattened operand"}),
        ("idx_roll", "indexing", idx_roll, (S((3, 4)),)),
        ("idx_pad", "indexing", idx_pad, (S((2, 3)),)),
        ("idx_diag", "indexing", idx_diag, (S((3,)),), {"inputs": (np.array([1.0, -2.0, 3.5], np.float32),)}),
        ("idx_tril_triu", "indexing", idx_tril_triu, (S((4, 4)),)),
        ("idx_diagonal_offset", "indexing", idx_diagonal, (S((3, 5)),)),
    ]

    # ---- 4. linalg --------------------------------------------------------
    conv_x, conv_w = rnd(1, 2, 6), rnd(3, 2, 3)
    solve_a = np.eye(4) * 4.0 + rng.standard_normal((4, 4)) * 0.1
    spd = solve_a @ solve_a.T + np.eye(4) * 4.0
    sym = spd + spd.T
    rank_x = np.outer(np.array([1.0, 2.0, 0.5, -1.0]), np.array([2.0, -1.0, 0.0]))
    sort_x = np.array([[3.0, 1.0, 3.0, -2.0, 0.5, 0.5], [0.5, 9.0, -1.0, 0.0, 3.0, 3.0],
                       [-4.0, 2.0, 2.0, 7.0, 1.0, 0.0]], np.float32)
    rows += [
        ("lin_dot_matmul", "linalg", lin_dot, (S((3, 4), f64), S((4, 5), f64))),
        ("lin_dot_batched", "linalg", lin_dot, (S((2, 3, 4)), S((2, 4, 5)))),
        ("lin_conv_valid", "linalg", lin_conv_valid, (S((1, 2, 6)), S((3, 2, 3))), {"inputs": (conv_x, conv_w)}),
        ("lin_conv_same", "linalg", lin_conv_same, (S((1, 2, 6)), S((3, 2, 3))), {"inputs": (conv_x, conv_w)}),
        ("lin_solve", "linalg", lin_solve, (S((4, 4), f64), S((4,), f64)),
         {"inputs": (solve_a, rng.standard_normal(4))}),
        ("lin_cholesky", "linalg", lin_cholesky, (S((4, 4), f64),), {"inputs": (spd,)}),
        ("lin_qr_reconstruct", "linalg", lin_qr, (S((5, 3), f64),)),
        ("lin_svd_reconstruct", "linalg", lin_svd, (S((4, 3), f64),)),
        ("lin_eigh_eigenvalues", "linalg", lin_eigh, (S((4, 4), f64),), {
            "inputs": (sym,),
            "note": "eigenvalues + the A@v - v*w residual (eigenvectors are sign-ambiguous)"}),
        ("lin_matrix_rank", "linalg", lin_matrix_rank, (S((4, 3), f64),), {"inputs": (rank_x,)}),
        ("lin_matrix_exp", "linalg", lin_matrix_exp, (S((3, 3), f64),),
         {"inputs": (rng.standard_normal((3, 3)) * 0.25,)}),
        ("lin_sort_argsort", "linalg", lin_sort, (S((3, 6)),),
         {"inputs": (sort_x,), "note": "tie-broken argsort via stable=True (see the documented divergence)"}),
        ("lin_argsort_no_ties", "linalg", lin_argsort_no_ties, (S((4, 5)),),
         {"inputs": (rng.permutation(20).reshape(4, 5).astype(np.float32),),
          "note": "stable=False over distinct values"}),
        ("dot_f32_large", "linalg", lin_dot, (S((64, 96)), S((96, 48))), {
            "inputs": (rng_roundoff.standard_normal((64, 96)).astype(np.float32),
                       rng_roundoff.standard_normal((96, 48)).astype(np.float32)),
            "known_roundoff": (1e-4, 1e-5),
            "note": "faer gemm vs OpenBLAS sgemm reassociate the K-sum: measured 580/3072 "
                    "elements differ, max abs 9.5e-6"}),
    ]

    # ---- 5. control flow --------------------------------------------------
    rows += [
        ("cf_cond", "control-flow", cf_cond, (S((3,)),), {"inputs": (np.array([-1.0, 0.5, -2.0], np.float32),)}),
        ("cf_while_counted", "control-flow", cf_while_counted, (S((3,)),), {
            "inputs": (np.array([3.0, 1.0, 0.5], np.float32),),
            "note": "the trip count comes from the input value x[0]"}),
        ("cf_while_consumes_input", "control-flow", cf_while_consumes, (S((4,)),)),
        ("cf_call", "control-flow", None, (), {
            "inputs": (np.array([1.0, -2.0, 0.5], np.float32),),
            "graph": _build_call_graph(cf_call_template, (S((3,)),)),
            "note": "hand-built two-function module: main calls helper"}),
        ("cf_cond_with_while", "control-flow", cf_cond_with_while, (S((3,)),),
         {"inputs": (np.array([0.5, 1.5, -0.5], np.float32),)}),
    ]

    # ---- 6. random (bit-exact) -------------------------------------------
    probs = np.array([0.1, 0.2, 0.4, 0.05, 0.25], np.float32)
    for algorithm in ("splitmix64", "threefry2x32", "philox4x32_10"):
        rows += [
            (f"rng_{algorithm}_keys", "random", _rng_graph(algorithm, "keys"), (),
             {"flags": frozenset({"exact"})}),
            (f"rng_{algorithm}_sampling", "random", _rng_graph(algorithm, "sampling"), (S((5,)),),
             {"inputs": (probs,), "flags": frozenset({"exact"})}),
            (f"rng_{algorithm}_normal", "random", _rng_graph(algorithm, "normal"), (),
             {"flags": frozenset({"exact", "rng_normal"}),
              "note": "f32 normal has a documented 1-ulp window; f64 is bit-exact"}),
        ]

    # ---- 7. dynamic shapes + input validation ----------------------------
    dyn_n = evoxc.dim("n")
    for size in (3, 7):
        rows += [
            (f"dyn_reduce_n{size}", "dynamic", dyn_reduce, (S((dyn_n, 4)),), {
                "inputs": (np.arange(size * 4, dtype=np.float32).reshape(size, 4),),
                "note": "symbolic (n, 4) spec bound at run time"}),
            (f"dyn_matmul_n{size}", "dynamic", dyn_matmul, (S((dyn_n, 3)), S((3, 2))),
             {"inputs": (rnd(size, 3), rnd(3, 2))}),
        ]
    evoxc.register_external_kernel("parity_double", lambda x: [np.asarray(x) * 2.0], backend="numpy")
    ext_x = np.array([1.0, 2.0, 3.0], np.float32)
    rows += [
        ("err_wrong_dtype", "dynamic", plain_scale, (S((3, 4)),), {
            "inputs": (np.zeros((3, 4), np.float64),), "native_error": ("DTypeError",),
            "note": "float64 input against a float32 spec"}),
        ("err_wrong_rank", "dynamic", plain_scale, (S((3, 4)),), {
            "inputs": (np.zeros((3, 4, 1), np.float32),), "native_error": ("ShapeError",),
            "note": "rank-3 input against a rank-2 spec"}),
        ("err_resolver_required", "dynamic", external_graph, (S((3,)),), {
            "inputs": (ext_x,), "native_error": ("BackendError",),
            "note": "external_call needs a resolver; run with resolver=None"}),
        ("ext_resolver_provided", "dynamic", external_graph, (S((3,)),),
         {"inputs": (ext_x,), "resolver": _ExternalResolver()}),
    ]

    cases: list = []
    for row in rows:
        name, category, fn, specs = row[:4]
        extra = row[4] if len(row) > 4 else {}
        given = extra.get("inputs")
        inputs = given if given is not None else tuple(rnd(*spec.shape, dtype=spec.dtype) for spec in specs)
        cases.append(Case(name, category, fn, tuple(specs), tuple(inputs),
                          **{k: v for k, v in extra.items() if k != "inputs"}))
    return cases


dyn_reduce = evoxc.defn(lambda x: evoxc.reduce_sum(evoxc.multiply(x, x), axes=(1,)))

dyn_matmul = evoxc.defn(lambda x, w: evoxc.dot(x, w))

plain_scale = evoxc.defn(lambda x: evoxc.multiply(x, 2.0))

external_graph = evoxc.defn(lambda x: evoxc.external_call("parity_double", x, result=evoxc.TensorSpec((3,), evoxc.float32)))

# ---------------------------------------------------------------------------
# Table + entry point
# ---------------------------------------------------------------------------


def _fmt(value: Optional[float]) -> str:
    return "-" if value is None else ("0" if value == 0.0 else f"{value:.3e}")


def print_table(outcomes: Sequence[Outcome], verbose: bool) -> None:
    header = f"{'case':38} {'category':13} {'backend=native dtype/shape':40} {'max_abs':>10} {'max_rel':>10} status"
    print(header)
    print("-" * len(header))
    for outcome in outcomes:
        print(
            f"{outcome.case.name:38} {outcome.case.category:13} {outcome.native_desc:40} "
            f"{_fmt(outcome.max_abs):>10} {_fmt(outcome.max_rel):>10} {outcome.status}"
        )
        if outcome.detail and (verbose or outcome.status == "FAIL"):
            print(f"    {outcome.detail}")
        if verbose and outcome.case.note:
            print(f"    note: {outcome.case.note}")


def summarise(outcomes: Sequence[Outcome]) -> int:
    passed = sum(1 for outcome in outcomes if outcome.status.startswith("PASS"))
    failed = [outcome for outcome in outcomes if outcome.status == "FAIL"]
    print()
    print(f"PARITY: {len(outcomes)} cases, {passed} passed, {len(failed)} failed, 0 skipped")
    flagged = [outcome.case.name for outcome in outcomes if outcome.status == "PASS~"]
    if flagged:
        print(f"PARITY: {len(flagged)} tolerance-passed (flagged): {', '.join(flagged)}")
    per_category: dict = {}
    for outcome in outcomes:
        ok, total = per_category.get(outcome.case.category, (0, 0))
        per_category[outcome.case.category] = (ok + (not outcome.status.startswith("FAIL")), total + 1)
    print("PARITY: " + ", ".join(f"{name} {ok}/{total}" for name, (ok, total) in per_category.items()))
    if failed:
        print("PARITY: failures:")
        for outcome in failed:
            print(f"  - {outcome.case.name}: {outcome.detail}")
    return 1 if failed else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--filter", default=None, help="run only cases whose name contains this substring")
    parser.add_argument("--verbose", action="store_true", help="print notes, every detail line and failing plans")
    args = parser.parse_args(argv)

    import evoxc._native as native_module

    if not native_module.available():
        print("PARITY: the native extension is not built — run scripts/build-native.sh", file=sys.stderr)
        return 2
    extension = native_module.require()

    cases = build_cases()
    if args.filter:
        cases = [case for case in cases if args.filter in case.name]
        if not cases:
            print(f"PARITY: no case matches --filter {args.filter!r}", file=sys.stderr)
            return 2

    outcomes = []
    for case in cases:
        outcome = check_case(extension, case)
        outcomes.append(outcome)
        if args.verbose and outcome.status == "FAIL":
            try:
                print(f"[verbose] {case.name} plan:\n{compile_native(extension, case).plan_text()}")
            except Exception:  # noqa: BLE001 - diagnostics only
                pass

    print_table(outcomes, args.verbose)
    return summarise(outcomes)


if __name__ == "__main__":
    raise SystemExit(main())
