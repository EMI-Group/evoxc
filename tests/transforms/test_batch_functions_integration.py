"""``evoxc.batch_functions`` — backend integration (IREE + StableHLO export).

`evoxc.batch_functions(fns, axes, out_axes, switch_ops=...)` merges N traced
`Graph`s with identical input/output signatures into ONE graph whose leading
axis is a **static** function axis of length N (row i evaluates function i).
Because that axis is a static Python `int` — never a symbolic `Dim` — the
merged graph must be an ordinary, fully static graph that compiler backends
can lower. This module pins exactly that end-to-end, and nothing about the
merge's bookkeeping (the pure merge semantics live with the transform itself).

Coverage:
* **IREE / llvm-cpu round-trip** — a switched merge (``switch_ops=True`` over
  ``[x + 1.0, x * 2.0, x - 3.0]``, so the same graph position holds three
  different elementwise ops and the emitter builds a static per-row ``select``
  mask cascade) and an unmapped merge (``axes=None`` over ``[sin(x), exp(x)]``,
  where the shared input is broadcast and the divergent outputs are muxed).
  Each merged graph is compiled ONCE, run through ``evoxc.run``, and compared
  (``rtol=atol=1e-5``) against BOTH the numpy-backend result of the very same
  merged graph (the semantic ground truth) AND the stacked per-function numpy
  references (the contract: row i == function i applied to row i).
* **StableHLO export sanity** (compiler-free) — ``stablehlo.export`` of the
  merged graphs yields MLIR text, and the switched merge's mask cascade really
  is a ``stablehlo.select``.
* **vmap cross-check** (numpy backend) — with three IDENTICAL rule-rich
  programs the merge degenerates to one ``vectorize`` pass, so it must agree
  numerically with ``evoxc.vmap`` on the same stacked input and emit NO
  ``select`` at all.

Staging is always explicit (``lower → compile → load → run``; ``evoxc.build``
takes a callable/`Defn`, never an already-traced `Graph`, so a merged graph is
staged through the explicit pipeline exactly like the sibling adapter suites
do). The IREE test is guarded so the file is green compiler-free as well; the
other two tests never touch iree.
"""

import importlib.util

import numpy as np
import pytest

import evoxc
from evoxc import TensorSpec
from evoxc.backends.stablehlo import export

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def run_graph(graph, *args):
    """Run a graph through the explicit staging pipeline (numpy backend).

    ``lower -> compile -> load -> run`` — the same composition the sibling
    adapter suites use; nothing is staged implicitly.
    """
    return evoxc.run(evoxc.load(evoxc.compile(evoxc.lower(graph))), *args)


def to_np(value):
    """Recursively replace `evoxc.Tensor` leaves with numpy arrays."""
    if isinstance(value, evoxc.Tensor):
        return value.numpy()
    if isinstance(value, (tuple, list)):
        return type(value)(to_np(v) for v in value)
    if isinstance(value, dict):
        return {k: to_np(v) for k, v in value.items()}
    return value


def all_ops(graph):
    """Every op in every block/region of the graph's module."""
    found = []

    def walk(block):
        for op in block.ops:
            found.append(op)
            for region in op.regions:
                for nested_block in region.blocks:
                    walk(nested_block)

    walk(graph.module.main.entry_block)
    return found


def op_names(graph):
    return [op.name for op in all_ops(graph)]


def _stacked(arrays):
    """Stack per-function numpy references along the function axis."""
    return np.stack(list(arrays))


#: The per-row input spec of every merged graph here (small + rank 2, so the
#: compiled lowering stays cheap).
ROW_SPEC = TensorSpec((3, 4), dtype="float32")


# --- the programs ----------------------------------------------------------


def add_one(x):
    return x + 1.0


def double(x):
    return x * 2.0


def minus_three(x):
    return x - 3.0


def sin_of(x):
    return evoxc.sin(x)


def exp_of(x):
    return evoxc.exp(x)


def tanh_rich(x):
    """Three distinct rule-carrying ops (grad/vmap rules exist for all)."""
    return evoxc.tanh(evoxc.multiply(x, 2.0)) + evoxc.exp(-x)


#: The switched programs and their plain-numpy references (row i == program i
#: applied to row i of the mapped input).
SWITCH_PROGRAMS = (add_one, double, minus_three)
SWITCH_REFS = (
    lambda row: row + np.float32(1.0),
    lambda row: row * np.float32(2.0),
    lambda row: row - np.float32(3.0),
)

#: The unmapped programs: every function sees the SAME (unmapped) input.
SHARED_PROGRAMS = (sin_of, exp_of)
SHARED_REFS = (np.sin, np.exp)


def trace_all(fns, spec=ROW_SPEC):
    """Trace every program once over the identical per-row spec."""
    return [evoxc.trace(fn, spec) for fn in fns]


def switched_merge():
    """``switch_ops=True`` merge of the three elementwise programs."""
    merged = evoxc.batch_functions(
        trace_all(SWITCH_PROGRAMS), axes=0, switch_ops=True
    )
    assert isinstance(merged, evoxc.Graph)
    return merged


def shared_merge():
    """``axes=None`` merge: one shared input, muxed (divergent) outputs."""
    merged = evoxc.batch_functions(trace_all(SHARED_PROGRAMS), axes=None)
    assert isinstance(merged, evoxc.Graph)
    return merged


#: ``key -> Executable`` — the whole module shares one compile per merged graph
#: (IREE compiles cost seconds; the merged graphs are deterministic).
_IREE_EXECUTABLES = {}


def iree_executable(key, graph):
    """Compile the merged graph ONCE for iree/llvm-cpu and cache the handle."""
    exe = _IREE_EXECUTABLES.get(key)
    if exe is None:
        lowered = evoxc.lower(
            graph, backend="iree", target_backends=["llvm-cpu"]
        )
        exe = evoxc.load(evoxc.compile(lowered))
        _IREE_EXECUTABLES[key] = exe
    return exe


#: Module-level availability probe (never an import): the two non-iree tests
#: must still RUN in a compiler-free environment, so the iree leg is guarded
#: per test instead of skipping the whole module.
def _module_available(name):
    """`find_spec` that treats a blocker / broken parent package as absent."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


HAVE_IREE = _module_available("iree.compiler") and _module_available(
    "iree.runtime"
)

needs_iree = pytest.mark.skipif(
    not HAVE_IREE, reason="iree compiler/runtime not installed"
)


#: Deterministic rank-3 mapped inputs (leading axis == the static function
#: axis, length == number of merged programs) and the rank-2 shared input.
_RNG = np.random.default_rng(20241104)
MAPPED_INPUT = _RNG.standard_normal((3, 3, 4)).astype(np.float32)
SHARED_INPUT = _RNG.standard_normal((3, 4)).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. IREE / llvm-cpu: the merged graph compiles and runs
# ---------------------------------------------------------------------------


@needs_iree
def test_iree_llvm_cpu_switched_merge_parity():
    """A switch_ops merge (select cascade) compiles on iree and stays exact.

    The function axis is static, so this is an ordinary static graph — if the
    merge ever leaked a symbolic dimension the adapter would reject it at
    ``lower()`` time.
    """
    pytest.importorskip("iree.compiler")
    pytest.importorskip("iree.runtime")

    merged = switched_merge()
    assert merged.verify() is None
    # Anti-vacuity: the elementwise variants really were switched (otherwise
    # the export test's `stablehlo.select` pin would be meaningless).
    assert merged.batch_report.switched, "expected a switched elementwise class"

    assert merged.tensor_specs[0].shape == (3, 3, 4)  # static function axis
    assert merged.tensor_specs[0].dtype == evoxc.float32

    exe = iree_executable("switched", merged)
    compiled = to_np(evoxc.run(exe, MAPPED_INPUT))
    reference = to_np(run_graph(merged, MAPPED_INPUT))

    assert compiled.shape == (3, 3, 4)
    # (a) identical to the numpy backend on the very same merged graph …
    np.testing.assert_allclose(compiled, reference, rtol=1e-5, atol=1e-5)
    # (b) … and to the stacked per-function references.
    expected = _stacked(
        ref(MAPPED_INPUT[i]) for i, ref in enumerate(SWITCH_REFS)
    )
    np.testing.assert_allclose(compiled, expected, rtol=1e-5, atol=1e-5)


@needs_iree
def test_iree_llvm_cpu_unmapped_merge_parity():
    """An ``axes=None`` merge (shared input, muxed outputs) compiles on iree."""
    pytest.importorskip("iree.compiler")
    pytest.importorskip("iree.runtime")

    merged = shared_merge()
    assert merged.verify() is None
    # The input is shared (no function axis); the OUTPUT is muxed, hence mapped.
    assert merged.tensor_specs[0].shape == (3, 4)
    assert merged.batch_report.output_muxes == 1

    exe = iree_executable("shared", merged)
    compiled = to_np(evoxc.run(exe, SHARED_INPUT))
    reference = to_np(run_graph(merged, SHARED_INPUT))

    assert compiled.shape == (2, 3, 4)
    np.testing.assert_allclose(compiled, reference, rtol=1e-5, atol=1e-5)
    expected = _stacked(ref(SHARED_INPUT) for ref in SHARED_REFS)
    np.testing.assert_allclose(compiled, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. StableHLO export sanity (no compiler needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build", [switched_merge, shared_merge], ids=["switched", "unmapped"]
)
def test_stablehlo_export_merged_graph(build):
    """A merged graph exports to StableHLO MLIR text without any special case.

    ``export`` imports only core/ir (always importable), so this pin needs no
    compiler.
    """
    merged = build()
    assert merged.verify() is None

    text = export(merged)
    assert isinstance(text, str)
    assert text.strip(), "export produced empty MLIR text"
    assert "stablehlo." in text
    assert "func.func" in text


def test_stablehlo_export_switched_merge_emits_select():
    """The switched merge's per-row mask cascade is a real ``stablehlo.select``.

    Never a Python-level branch and never per-row evaluation: the cascade is
    static IR the compiler lowers like any other elementwise select.
    """
    merged = switched_merge()
    assert merged.verify() is None
    assert merged.batch_report.switched
    assert "stablehlo.select" in export(merged)


# ---------------------------------------------------------------------------
# 3. vmap cross-check (numpy backend): identical programs == one vectorize
# ---------------------------------------------------------------------------


def test_identical_programs_merge_matches_vmap():
    """``batch_functions([g, g, g])`` == ``vmap(g)`` + no select ops.

    With identical programs every class has a single variant, so the emitter
    takes the zero-overhead path (exactly one ``vectorize`` pass). That is the
    strongest statement this transform can make about its own output: the
    merged graph IS the vectorized graph, numerically and structurally.
    """
    batched_spec = TensorSpec((3, 3, 4), dtype="float32")

    merged = evoxc.batch_functions([tanh_rich, tanh_rich, tanh_rich], axes=0)(
        batched_spec
    )
    assert isinstance(merged, evoxc.Graph)
    assert merged.verify() is None

    vectorized = evoxc.vmap(tanh_rich, 0)(batched_spec)
    assert isinstance(vectorized, evoxc.Graph)
    assert vectorized.verify() is None

    got = to_np(run_graph(merged, MAPPED_INPUT))
    want = to_np(run_graph(vectorized, MAPPED_INPUT))
    assert got.shape == (3, 3, 4)
    np.testing.assert_allclose(got, want, rtol=1e-6)

    # Identical programs emit NO mask cascade at all (and no report switches).
    assert "select" not in op_names(merged)
    assert not merged.batch_report.switched
    assert merged.batch_report.output_muxes == 0

    # Row i is still the program applied to row i of the mapped input (three
    # identical programs, three different rows).
    row_graph = evoxc.trace(tanh_rich, TensorSpec((3, 4), "float32"))
    row_refs = _stacked(
        to_np(run_graph(row_graph, MAPPED_INPUT[i])) for i in range(3)
    )
    np.testing.assert_allclose(got, row_refs, rtol=1e-6)
