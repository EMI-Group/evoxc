"""Native CPU engine — fast-path SHAPE pins (cheap, load-robust perf regression net).

The performance contract lives in ``native/evoxc-cpu/CONTEXT.md`` ("Performance
(measured on this host, shared CPU)"), which records the strided-run
slice/gather/scatter paths, the typed random sampling path and the broadcast
elementwise run driver, together with the per-op medians the ``tn_step_pop500``
attribution table implies (``slice f32 (500,50,1)`` 2.471 ms / 364 calls ≈
6.8 µs a call, broadcast ``multiply`` ≈ 0.1 ms a call, ``random_uniform``
"already 2-3× faster than numpy").

What this module pins — deliberately the SHAPE of those fast paths instead of
an absolute throughput number, because this host is shared and a wall-clock
assert with real headroom cannot be made tight:

* **Behaviour** (the primary assertion): the three fast paths together produce
  the numpy backend's values — a fast path that silently drops elements or
  mis-resolves a strided run fails here, on any host, at any load.
* **Plan shape**: the compiled native plan is ONE flat block whose steps carry
  the static ``(dtype, shape)`` result markers the plan-time specialisation
  writes (``static float32[500, 50, 1]``, ``static float32[500, 17]``), i.e. the
  static-result fast path is engaged rather than a symbolic re-evaluation.
* **A very generous ceiling**: the whole graph is timed as the median of warm
  ``run`` calls and must stay under :data:`SLICE_ADD_UNIFORM_CEILING_MS`. The
  recorded median is ~0.05 ms (numpy's own leg ~0.08 ms), so the ceiling is
  ~90x the recorded number — it can only fire on a REGIME change (the recorded
  losses are floors of 1.4–14.5 ns/element; losing the strided-run or typed
  sampling path would cost 10–100x), never on scheduler noise. Measured
  worst-case single call in a cold process: 0.24 ms.
* **Harness inventory**: ``native/tools/bench_cpu.py`` still carries the eight
  workloads including the authoritative ``tn_step_pop500`` approximation, so
  the recorded table stays reproducible.

CPU only, fixed shapes, no network, no GPU. Gated on the compiled extension.
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path

import numpy as np
import pytest

import evoxc
from evoxc import core

from tests.backends import _native_utils as nu

pytestmark = pytest.mark.skipif(not nu.ENGINE_AVAILABLE, reason=nu.SKIP_REASON)

#: Shape of the strided-run slice workload (the TensorNEAT/GP generation shape).
ROWS, LANES, WIDE = 500, 50, 4

#: Width of the typed ``random_uniform`` draw (a GP/GA parameter block).
UNIFORM_COLS = 17

#: Wall-clock ceiling for the graph below, in milliseconds. The recorded median
#: is ~0.054 ms and numpy's own leg ~0.079 ms on the recording host, so this is
#: ~90x headroom: a generous ceiling that still catches a lost fast path.
SLICE_ADD_UNIFORM_CEILING_MS = 5.0

#: Warm iterations after the warmup; the MEDIAN is compared against the ceiling.
ITERATIONS = 21

#: Repo root (``tests/backends/<this file>`` → three levels up).
REPO_ROOT = Path(__file__).resolve().parents[2]


def _fast_paths(rows, lanes, cols):
    """The perf graph: strided slice + broadcast add + typed uniform, all reduced.

    ``(rows, lanes, 4) → slice (rows, lanes, 1)`` exercises a non-innermost
    strided-run copy (not a numpy view), the ``add`` against a
    ``(1, lanes, 1)`` constant exercises the broadcast elementwise run driver,
    and ``random_uniform`` draws a typed ``rows x cols`` f32 block. Both halves
    are reduced so neither can be dead code.
    """

    @evoxc.defn
    def fast_paths(x):
        sliced = evoxc.slice(x, (0, 0, 0), (rows, lanes, 1))
        bias = evoxc.constant(
            evoxc.from_numpy(np.arange(lanes, dtype=np.float32).reshape(1, lanes, 1))
        )
        shifted = evoxc.add(sliced, bias)
        key = evoxc.random.key(7)
        noise = evoxc.random.uniform(key, (rows, cols))
        return evoxc.add(
            evoxc.reduce_sum(shifted, axes=(0, 1)),
            evoxc.reduce_sum(noise),
        )

    return fast_paths


#: The recorded workload: the TensorNEAT/GP generation shapes.
STRIDED_FAST_PATHS = _fast_paths(ROWS, LANES, UNIFORM_COLS)


def _input(rows=ROWS, lanes=LANES, wide=WIDE):
    """Deterministic f32 (rows, lanes, wide) input (no RNG)."""
    size = rows * lanes * wide
    return (np.arange(size, dtype=np.float32) / 7.0).reshape(rows, lanes, wide)


def _spec(rows=ROWS, lanes=LANES, wide=WIDE):
    return core.TensorSpec(shape=(rows, lanes, wide), dtype=core.float32)


def _build(backend, rows=ROWS, lanes=LANES, wide=WIDE, cols=UNIFORM_COLS):
    """Build the perf graph for ``backend`` at one static size."""
    if (rows, lanes, cols) == (ROWS, LANES, UNIFORM_COLS):
        graph = STRIDED_FAST_PATHS  # the module-level (recorded) graph
    else:
        graph = _fast_paths(rows, lanes, cols)
    return evoxc.build(graph, _spec(rows, lanes, wide), backend=backend)


# ---------------------------------------------------------------------------
# behaviour: the fast paths compute the numpy answer
# ---------------------------------------------------------------------------
def test_fast_paths_match_the_numpy_backend():
    """A wrong strided run / broadcast / typed draw shows up as a value mismatch."""
    x = _input()
    got = np.asarray(evoxc.run(_build("native"), x).numpy())
    expected = np.asarray(evoxc.run(_build("numpy"), x).numpy())

    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    # f32 reductions over 25 000 elements in a different association order, so
    # a defensible relative budget rather than bit-exactness (the ELEMENTWISE
    # fast paths themselves are pinned bit-exact in
    # ``test_native_backend_parity_core.py``).
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-3)


def test_fast_paths_hold_for_another_static_size():
    """The same fast paths are correct at a second static shape (no size special-case)."""
    rows, lanes, wide = 37, 5, 3
    x = _input(rows, lanes, wide)
    got = np.asarray(evoxc.run(_build("native", rows, lanes, wide), x).numpy())
    expected = np.asarray(evoxc.run(_build("numpy", rows, lanes, wide), x).numpy())
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-4)


# ---------------------------------------------------------------------------
# plan shape: one flat block with the static-result fast-path markers
# ---------------------------------------------------------------------------
def test_plan_carries_static_result_markers_and_no_region():
    """The compiled plan is flat and every step pre-computes its static type."""
    plan = _build("native").plan_text()

    # One region-free block: the fast paths must not need a nested region.
    assert "1 block(s)" in plan, plan
    assert "block #0" in plan
    # The plan-time static (dtype, shape) marker the run's fast path reads.
    assert "static float32[500, 50, 1]" in plan, plan  # the strided slice result
    assert "static float32[500, 17]" in plan  # the typed uniform draw result
    # The op set the shape exists for, in plan order.
    for op_name in ("slice", "add", "random_uniform", "reduce_sum"):
        assert op_name in plan, f"{op_name} missing from the native plan:\n{plan}"


# ---------------------------------------------------------------------------
# a very generous wall-clock ceiling (the median of warm runs)
# ---------------------------------------------------------------------------
def test_slice_add_uniform_run_under_generous_ceiling():
    """Warm ``run`` median under a ~90x-generous ceiling (regime-change detector)."""
    x = _input()
    executable = _build("native")
    for _ in range(5):  # warmup: page in the plan, the buffers and the pool
        evoxc.run(executable, x)

    samples = []
    for _ in range(ITERATIONS):
        started = time.perf_counter()
        evoxc.run(executable, x)
        samples.append((time.perf_counter() - started) * 1e3)

    median = statistics.median(samples)
    assert median < SLICE_ADD_UNIFORM_CEILING_MS, (
        f"median warm native run {median:.3f} ms exceeds the "
        f"{SLICE_ADD_UNIFORM_CEILING_MS:.1f} ms ceiling "
        f"(fast paths lost? min {min(samples):.3f} / max {max(samples):.3f} ms)"
    )


# ---------------------------------------------------------------------------
# the bench harness the recorded table comes from still exists
# ---------------------------------------------------------------------------
def test_bench_harness_carries_the_expected_workloads():
    """``native/tools/bench_cpu.py`` still defines the authoritative workloads.

    The registry has grown past the eight rows of the CONTEXT table (the
    large-tensor rows came later), so the pin is: every original name is still
    there AND the closed registry still carries at least 17 ``Workload(...)``
    entries.
    """
    text = (REPO_ROOT / "native" / "tools" / "bench_cpu.py").read_text(encoding="utf-8")

    assert "def tn_step_pop500(" in text, (
        "native/tools/bench_cpu.py no longer defines the tn_step_pop500 workload "
        "that native/evoxc-cpu/CONTEXT.md's authoritative table is measured on"
    )
    # Every original workload name the table registers (the eight base rows).
    for name in (
        "ew_chain_1e6",
        "ew_transcendental_1e6",
        "dot_sum_256",
        "gp_gather_4096",
        "gp_sort_4096",
        "tf_softmax_8x128x384",
        "chain_2000",
        "tn_step_pop500",
    ):
        assert f'"{name}"' in text, f"workload {name} missing from bench_cpu.py"
    # The registry itself, not just the defns.
    assert "def build_workloads()" in text
    assert "Workload(" in text
    # The closed registry is the ``Workload(...)`` list ``build_workloads()``
    # returns (there is no module-level ``WORKLOADS`` table to read); all of
    # them live inside that function, so counting there is exact.
    registry = text.split("def build_workloads()", 1)[1]
    assert registry.count("Workload(") >= 17, (
        "native/tools/bench_cpu.py's build_workloads() registry shrank below the "
        "17 workloads its docstring advertises"
    )
