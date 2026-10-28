"""Pins for `batch_functions`' TWO INTERNAL PERFORMANCE HOOKS — deterministically.

Contract under test (binding): `evoxc/transforms/_emit.py`
(`emit_merged_graph`, `_stamp_locations`, `finish_merged_graph`),
`evoxc/transforms/_bands.py` (`emit_banded_graph`, `_combine_bands`),
`evoxc/ops/_utils.py` (`get_location`, `_suppress_location_capture`), summarised
in `evoxc/transforms/CONTEXT.md` § "Phase 2 — emission (`_emit.py`)".

A merge is an EMITTER: it builds a whole graph of ordinary ops in one go, so the
per-op call-site capture of `ops` would be pure overhead. Instead the merged
graph is re-stamped afterwards with its contributing ORIGINAL ops' locations
(`_stamp_locations`), and the helper ops the mux / band machinery emits itself
get ONE call site captured once per merge. Two hooks implement that, and this
module pins BOTH without ever measuring wall-clock time:

1. **Suppression hook.** `_suppress_location_capture()` is a private,
   re-entrant context manager that makes `get_location()` return
   `ir.Location.unknown()` WITHOUT the `inspect.stack()` walk. Both drivers do
   `call_site = get_location()` exactly ONCE, OUTSIDE the suppression block, and
   then emit every op (rules, stacked constants, band slices/gathers/
   concatenations, mux `select`s) INSIDE it. So **`inspect.stack` is called
   exactly ONCE per merge — independently of how many ops the merge emits**.
   Loud caveat: `get_location` wraps its walk in `try/except Exception`, so a
   monkeypatched `inspect.stack` that RAISES is silently swallowed — "the merge
   still succeeds" alone proves nothing. The CALL COUNT is the real contract, so
   every case below installs a counting `inspect.stack` that ALSO raises and
   pins the exact count (`== 1`).

2. **Location preservation.** Suppressing capture must not lose source
   information: every non-terminator op of the merged graph carries a REAL
   `ir.Location` drawn from the contributing programs' source lines, the
   `return` terminator keeps `location is None`, and `graph.source_locations`
   carries the contributing values' locations forward.

Determinism: numpy backend, CPU, small shapes, fixed populations, and NO timing
assertions whatsoever — no `time` / `perf_counter` import anywhere in this
module. The pins are exact structural constants (call counts and emitted-op
counts), so they hold identically on a slow machine and a fast one.
"""
import inspect

import numpy as np
import pytest

import evoxc
from evoxc import core
from evoxc import ir

F32 = evoxc.float32

# ---------------------------------------------------------------------------
# helpers (module level — no fixtures, no conftest changes)
# ---------------------------------------------------------------------------


def run_graph(graph, *args):
    """Explicit staging pipeline: lower → compile → load → run."""
    return evoxc.run(evoxc.load(evoxc.compile(evoxc.lower(graph))), *args)


def to_np_tree(value):
    """Recursively replace `evoxc.Tensor` leaves with numpy arrays."""
    if isinstance(value, core.Tensor):
        return value.numpy()
    if isinstance(value, dict):
        return {key: to_np_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)) and not isinstance(value, core.Tensor):
        return type(value)(to_np_tree(item) for item in value)
    return value


def entry_ops(graph):
    """The graph's entry-block ops, in emission order (`return` included)."""
    return list(graph.module.main.entry_block.ops)


def non_terminator_ops(graph):
    """Entry-block ops without the `return` terminator (the location carriers)."""
    return [op for op in entry_ops(graph) if op.name != "return"]


def verifies(graph):
    """`batch_functions` never verifies (staging stays explicit) — do it here."""
    assert isinstance(graph, evoxc.Graph)
    graph.verify()  # raises core.VerificationError on bad IR
    return graph


class _StackCounter:
    """A counting `inspect.stack` stand-in that ALSO raises on every call.

    Raising is the loud half of the pin: `get_location`'s `except Exception`
    swallows it, so a merge that walked the stack would still "succeed" while
    silently losing every location. The `calls` counter is the half that
    actually proves the suppression hook works.
    """

    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("stack walk attempted during the merge")


def counting_merge(monkeypatch, graphs, **kwargs):
    """`batch_functions(graphs, **kwargs)` + the number of `inspect.stack` calls.

    The counter is installed ONLY around the merge — tracing happens before, so
    the count is the merge's own stack-walk cost (exactly the quantity the
    suppression hook exists to keep at one).
    """
    counter = _StackCounter()
    monkeypatch.setattr(inspect, "stack", counter)
    graph = evoxc.batch_functions(graphs, **kwargs)
    return graph, counter.calls


# --- populations (fixed, deterministic) ------------------------------------


def _constant_chain(length, step=0.5):
    """A `length`-op chain of identical `x + step` adds (one shared constant)."""

    def program(x):
        for _ in range(length):
            x = x + step
        return x

    return program


def chain_population(length, n_functions):
    """`n_functions` IDENTICAL long-chain programs over a `(4,)` float32 spec."""
    spec = evoxc.TensorSpec((4,), F32)
    fn = _constant_chain(length)
    return [evoxc.trace(fn, spec) for _ in range(n_functions)]


def _plus_minus(x, c):
    return (x + c) * (x - c)


def _minus_plus(x, c):
    return (x - c) * (x + c)


def _plus_plus(x, c):
    return (x + c) * (x + c)


def _times_plus(x, c):
    return (x * c) + c


#: The three-way divergent population: same topology, different operand order /
#: operand identity at the `multiply`, with a DISTINCT constant per program.
_DIVERGENT_VARIANTS = (_plus_minus, _minus_plus, _plus_plus)


def divergent_population(variants=_DIVERGENT_VARIANTS, n_functions=10):
    """10 divergent programs over a `(1024,)` float32 spec, constant `i + 1`."""
    spec = evoxc.TensorSpec((1024,), F32)
    graphs = []
    for index in range(n_functions):
        constant = float(index + 1)
        variant = variants[index % len(variants)]

        def program(x, _constant=constant, _variant=variant):
            return _variant(x, _constant)

        graphs.append(evoxc.trace(program, spec))
    return graphs


def three_op_program(x):
    """Three ops on three DISTINCT source lines (the location-set anchor)."""
    y = x + 1.0
    z = y - 2.0
    return z * 3.0


# ---------------------------------------------------------------------------
# 1. Suppression hook — `inspect.stack` is called EXACTLY ONCE per merge
# ---------------------------------------------------------------------------

#: `(id, population factory, merge kwargs, canonical emitted-op count)`.
#: Every count below is the measured value at this HEAD for the `alignment`
#: policy its `kwargs` STATE (`"occurrence"` unless the merge omits the option,
#: in which case the default `"variant"` applies) — the class-alignment policy
#: changes how many classes a switched position emits, so a count is only
#: meaningful next to its policy. The assertion that matters for the hook is the
#: invariant ALSO pinned for all of them: the `inspect.stack` call count stays
#: exactly 1, whatever the alignment emits.
SUPPRESSION_CASES = [
    pytest.param(
        lambda: chain_population(12, 32),
        {"switch_ops": True},
        25,
        id="flat-identical-32x12op-chain",
    ),
    pytest.param(
        divergent_population,
        {"switch_ops": True, "alignment": "occurrence"},
        17,
        id="flat-divergent-10-occurrence",
    ),
    pytest.param(
        divergent_population,
        {"switch_ops": True, "bands": 3, "alignment": "occurrence"},
        55,
        id="banded-explicit-3-occurrence",
    ),
    pytest.param(
        divergent_population,
        {"switch_ops": True, "bands": "auto", "alignment": "occurrence"},
        17,
        id="banded-auto-flat-partition-occurrence",
    ),
    pytest.param(
        lambda: divergent_population(variants=(_plus_minus, _plus_plus, _times_plus)),
        {"switch_ops": True, "bands": "auto"},
        21,
        id="banded-auto-two-bands",
    ),
    # The DEFAULT policy (`alignment` omitted → `_merge.DEFAULT_ALIGNMENT`, i.e.
    # `"variant"`): the same `bands=3` merge as `banded-explicit-3-occurrence`
    # emits 40 ops instead of 55, because sibling variants share one class per
    # skeleton. Both policies therefore stay pinned.
    pytest.param(
        divergent_population,
        {"switch_ops": True, "bands": 3},
        40,
        id="banded-explicit-3-default-variant",
    ),
]


@pytest.mark.parametrize("build, kwargs, emitted_ops", SUPPRESSION_CASES)
def test_merge_walks_the_stack_exactly_once(monkeypatch, build, kwargs, emitted_ops):
    """ONE `inspect.stack` call per merge, whatever the emission size.

    The fake stack raises on EVERY call; the merge must not even notice (the
    raise is swallowed by `get_location`) while the counter proves the
    suppression block covered the whole emission.
    """
    graphs = build()
    graph, calls = counting_merge(monkeypatch, graphs, **kwargs)
    verifies(graph)

    assert calls == 1, (
        "batch_functions must capture its ONE call-site location outside the "
        f"suppression block and never walk the stack again, got {calls} "
        f"inspect.stack() calls (kwargs={kwargs!r})"
    )
    assert graph.batch_report.emitted_ops == emitted_ops
    # The count is a small CONSTANT — it never scales with the emission.
    assert graph.batch_report.emitted_ops >= 17
    assert len(entry_ops(graph)) == emitted_ops + 1  # + the `return`


def test_long_chain_merge_emits_many_ops_for_one_stack_walk(monkeypatch):
    """`emitted_ops >> 1` and `calls == 1`: the hook is size-independent.

    A 32-program population of 12-op chains emits 25 ops for ONE stack walk; the
    ratio is the point — the per-op capture of `ops` is what the suppression
    block removes.
    """
    graphs = chain_population(12, 32)
    graph, calls = counting_merge(monkeypatch, graphs, switch_ops=True)
    verifies(graph)

    report = graph.batch_report
    assert report.emitted_ops == 25
    assert report.n_functions == 32
    assert calls == 1
    assert report.emitted_ops > 10 * calls  # a small constant vs a big emission


@pytest.mark.parametrize("length, emitted_ops", [(12, 25), (24, 49), (48, 97)])
def test_call_count_does_not_scale_with_the_emitted_op_count(
    monkeypatch, length, emitted_ops
):
    """4× the emitted ops ⇒ the SAME single stack walk (the hook's whole point).

    The same 32 identical programs are merged at three chain lengths; only the
    emission grows (2·length + 1 ops), the `inspect.stack` count stays at 1.
    """
    graphs = chain_population(length, 32)
    graph, calls = counting_merge(monkeypatch, graphs, switch_ops=True)
    verifies(graph)
    assert graph.batch_report.emitted_ops == emitted_ops
    assert calls == 1


def test_constant_stacking_merge_walks_the_stack_once(monkeypatch):
    """The stacked-constant path emits a `(N, ...)` constant per class — 1 walk."""
    spec = evoxc.TensorSpec((1024,), F32)
    graphs = []
    for index in range(6):
        constant = float(index + 1)

        def program(x, _constant=constant):
            return _times_plus(x, _constant)

        graphs.append(evoxc.trace(program, spec))

    graph, calls = counting_merge(monkeypatch, graphs, switch_ops=True)
    verifies(graph)
    assert calls == 1
    assert graph.batch_report.stacked_constants == 1
    assert graph.batch_report.emitted_ops == 5


def test_suppression_context_manager_returns_unknown_without_a_walk(monkeypatch):
    """Direct pin of the MECHANISM behind the counts above (internal API).

    While a `_suppress_location_capture()` block is active, `get_location()`
    answers `ir.Location.unknown()` immediately — no `inspect.stack` call at all
    — and the block is RE-ENTRANT (nested blocks restore the previous depth).
    """
    from evoxc.ops._utils import _suppress_location_capture, get_location

    counter = _StackCounter()
    monkeypatch.setattr(inspect, "stack", counter)

    assert counter.calls == 0
    with _suppress_location_capture():
        assert get_location() == ir.Location.unknown()
        with _suppress_location_capture():  # re-entrant: still suppressed
            assert get_location() == ir.Location.unknown()
        assert get_location() == ir.Location.unknown()
    assert counter.calls == 0, (
        "get_location() must short-circuit on the suppression depth before "
        "touching inspect.stack()"
    )

    # Positive control: outside the block the walk is consulted again (the fake
    # raises, and `get_location` degrades to `unknown` — never propagates).
    assert get_location() == ir.Location.unknown()
    assert counter.calls == 1


# ---------------------------------------------------------------------------
# 2. Location preservation — suppression must not cost source information
# ---------------------------------------------------------------------------


def test_merged_ops_inherit_the_contributing_source_locations():
    """Set equality: the merged emission covers EXACTLY the source lines.

    Three identical programs, each with three ops on three distinct source
    lines. The merged graph inserts alignment `reshape` ops, but every one of
    them is stamped with its contributing op's location, so the merged
    `(file, line)` set equals the source program's — no line invented, none
    lost.
    """
    spec = evoxc.TensorSpec((4,), F32)
    graphs = [evoxc.trace(three_op_program, spec) for _ in range(3)]
    graph = verifies(evoxc.batch_functions(graphs, switch_ops=True))

    source_ops = non_terminator_ops(graphs[0])
    source_lines = {(op.location.file, op.location.line) for op in source_ops}
    assert len(source_lines) == 3, "the probe program must span three lines"

    merged_ops = non_terminator_ops(graph)
    assert merged_ops, "the merge emitted nothing"
    assert {(op.location.file, op.location.line) for op in merged_ops} == source_lines
    # Every merged op points into THIS module, on one of the program's lines.
    for op in merged_ops:
        assert op.location is not None, f"op '{op.name}' lost its location"
        assert op.location.file == __file__
        assert op.location.line in {line for _file, line in source_lines}
        assert op.location.col == 0


def test_alignment_helper_ops_are_stamped_with_their_contributing_op():
    """`_stamp_locations` gives the merge's OWN helper ops the source location.

    Constant/broadcast alignment inserts `reshape` ops; each one is stamped with
    the contributing op's location rather than the merge call site (which is a
    different, later source line from `evoxc.batch_functions`).
    """
    spec = evoxc.TensorSpec((4,), F32)
    graphs = [evoxc.trace(three_op_program, spec) for _ in range(2)]
    graph = verifies(evoxc.batch_functions(graphs, switch_ops=True))

    ops = non_terminator_ops(graph)
    reshapes = [op for op in ops if op.name == "reshape"]
    assert reshapes, "this population must exercise the alignment reshape"
    for index, op in enumerate(ops):
        if op.name != "reshape":
            continue
        previous = ops[index - 1]
        assert op.location == previous.location, (
            "a merge-inserted helper op must inherit the contributing op's "
            f"location, got {op.location} vs {previous.location}"
        )


def test_terminator_location_and_the_source_location_map():
    """The `return` keeps `location is None`; `source_locations` is carried over.

    `graph.source_locations` records the CONTRIBUTING values — the input args
    plus one entry per re-stamped original op — with real `ir.Location`s that
    point into this module; the helper ops the merge inserts itself are not
    recorded, and nothing is re-captured from the merge call site.
    """
    spec = evoxc.TensorSpec((4,), F32)
    graphs = [evoxc.trace(three_op_program, spec) for _ in range(2)]
    graph = verifies(evoxc.batch_functions(graphs, switch_ops=True))

    # The `return` terminator carries no location (tracing does not stamp it).
    terminator = entry_ops(graph)[-1]
    assert terminator.name == "return"
    assert terminator.location is None

    source_locations = graph.source_locations
    assert source_locations, "the merge dropped every source location"
    for value_id, location in source_locations.items():
        assert isinstance(location, ir.Location), (
            f"source_locations[{value_id}] is {type(location).__name__}, "
            "expected an ir.Location"
        )
        assert location.file == __file__
        assert location.line > 0

    # One entry per CONTRIBUTING value: every tensor input leaf (carried over
    # from the traced programs) plus the original op results the merge
    # re-stamped — the merge's own helper ops add none.
    contributing = len(graphs[0].source_locations) + len(
        non_terminator_ops(graphs[0])
    )
    assert len(source_locations) == contributing

    # Every recorded `(file, line)` is a real source position: one of the
    # program's own op lines, or the line the program was TRACED from (the
    # input-arg location, which `evoxc.trace` records separately).
    op_lines = {op.location.line for op in non_terminator_ops(graphs[0])}
    argument_lines = {
        location.line for location in graphs[0].source_locations.values()
    }
    for location in source_locations.values():
        assert location.line in op_lines | argument_lines
    assert op_lines <= {location.line for location in source_locations.values()}


# ---------------------------------------------------------------------------
# 3. `EVOXC_DISABLE_LOCATIONS=1` — the env-var half of the same hook
# ---------------------------------------------------------------------------


def test_disable_locations_env_forces_unknown_locations(monkeypatch):
    """With the env var set, tracing AND merging produce `Location.unknown()`.

    The variable is read at TRACE time too, so the programs are traced inside
    the env context; the merged graph then carries no real location at all —
    while still verifying as ordinary IR.
    """
    monkeypatch.setenv("EVOXC_DISABLE_LOCATIONS", "1")
    spec = evoxc.TensorSpec((4,), F32)
    graphs = [evoxc.trace(three_op_program, spec) for _ in range(2)]

    for op in non_terminator_ops(graphs[0]):
        assert op.location == ir.Location.unknown()

    graph = verifies(evoxc.batch_functions(graphs, switch_ops=True))
    merged_ops = non_terminator_ops(graph)
    assert merged_ops
    for op in merged_ops:
        assert op.location == ir.Location.unknown(), (
            f"op '{op.name}' captured {op.location} under "
            "EVOXC_DISABLE_LOCATIONS=1"
        )


def test_merge_under_disable_locations_still_runs_row_exact(monkeypatch):
    """Disabling capture changes locations only — the merged graph is correct."""
    monkeypatch.setenv("EVOXC_DISABLE_LOCATIONS", "1")
    graphs = divergent_population()
    graph = verifies(evoxc.batch_functions(graphs, switch_ops=True))

    rows = np.stack(
        [np.linspace(-1.0, 1.0, 1024, dtype=np.float32) + index for index in range(10)]
    )
    out = to_np_tree(run_graph(graph, rows))
    reference = np.stack(
        [to_np_tree(run_graph(program, row)) for program, row in zip(graphs, rows)]
    )
    assert out.shape == reference.shape == (10, 1024)
    np.testing.assert_allclose(out, reference, rtol=1e-5, atol=1e-5)
