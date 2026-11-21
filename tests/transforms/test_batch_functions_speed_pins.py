"""Pins for `batch_functions`' NATIVE merge contract — deterministically.

Contract under test (binding): `evoxc/transforms/CONTEXT.md`
§ "`batch_functions` frontend", `native/evoxc-compiler/src/batch_functions/CONTEXT.md`
§ "Emission contract" / "Deviations from the Python contract" /
"Bands merged graph", plus `evoxc/transforms/_report.py` (`BatchReport`, the
12 keys) and `evoxc/ops/_utils.py` (`get_location`).

`batch_functions` merges in the NATIVE engine
(`evoxc._evoxc_native.batch_functions`): Rust builds the merged module, stamps
every rule-emitted op with its contributing source op's `Location` (carried
through the native envelope from TRACING) and returns the 12-key report.
Python keeps only validation, `Graph` reconstruction and report rehydration.
The pre-native Python emitters — `evoxc.transforms._emit`, `_bands` and their
`_suppress_location_capture()` hook — are gone, so the old "one call site per
merge / suppression block" instrumentation is structurally obsolete; what
remains true, and what this module pins, is the native contract:

1. **Zero merge-time Python work.** A merge calls
   `evoxc.ops._utils.get_location` ZERO times and never reaches
   `inspect.stack()`: every source location was captured at TRACE time and
   survives the native round trip. A positive control shows the same live
   counter DOES fire during tracing, so the zero is a real interception and not
   a patch that misses its target. The cost is therefore independent of the
   emission size (chain lengths 12/24/48 → 25/49/97 emitted ops, 0 captures).
2. **The report is the instrumentation.** Emitted-op counts, class counts,
   switch/constant/mux counts and the band/cost fields are read from
   `graph.batch_report` (the documented 12-key `BatchReport`) and cross-checked
   against the merged IR's entry block — never from private emitter helpers.
   Every count in `MERGE_CASES` is the MEASURED native value at this HEAD for a
   fixed population, stated next to the `alignment` policy its kwargs select.
3. **Locations are inherited, never invented.** Merged rule-emitted ops carry
   their contributing program op's real `(file, line)`; ops the merge machinery
   emits itself (switch/mask constants, `select`s, band `slice`s, the output
   `concatenate`) and the `return` terminator carry `location is None` (Rust
   has no frame-capture call site); programs traced with capture disabled keep
   the `<unknown>` placeholder; `graph.source_locations` still holds one entry
   per merged input, carried from the source graphs.
4. **`EVOXC_DISABLE_LOCATIONS=1` changes locations only** — the merged graph is
   still row-exact against the per-program runs.

Determinism: numpy backend, CPU, small shapes, fixed populations, and NO timing
assertions whatsoever — no `time` / `perf_counter` import anywhere in this
module. The pins are exact structural constants (call counts and report
fields), so they hold identically on a slow machine and a fast one.
"""

import inspect

import numpy as np
import pytest

import evoxc
from evoxc import core
from evoxc import ir
from evoxc.ops import _utils as ops_utils

F32 = evoxc.float32

#: The exact 12-key `BatchReport` surface, pinned independently of
#: `evoxc.transforms._report.REPORT_KEYS` (a contract, not a mirror of the
#: implementation — `test_batch_functions.py::test_public_surface` pins it too).
REPORT_KEYS = (
    "n_functions",
    "value_classes",
    "shared_classes",
    "switched",
    "stacked_constants",
    "output_muxes",
    "emitted_ops",
    "bands",
    "band_sizes",
    "band_strategy",
    "cost_estimate",
    "flat_cost_estimate",
)

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

    Location capture is a `sys._getframe` walk, so `inspect.stack` must never be
    reached: the count is pinned at ZERO for every case (and the raise makes an
    accidental traversal loud instead of merely slow).
    """

    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("inspect.stack() reached during the merge")


class _CaptureCounter:
    """A DELEGATING counter for `evoxc.ops._utils.get_location`.

    The op modules reach `get_location` through the `_utils` module attribute
    (`_utils.get_location(depth=2)`), so replacing that attribute intercepts
    every capture attempt. The stand-in delegates verbatim, keeping the captured
    location byte-identical to the real one's (`depth` is not shifted).
    """

    def __init__(self, real):
        self._real = real
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self._real(*args, **kwargs)


class _MergeCounters:
    """Both merge-time seams: the Python capture hook and `inspect.stack`.

    Installed for the duration of a single merge (the `monkeypatch` fixture
    restores both bindings afterwards). Tracing happens BEFORE the merge, so
    `locations.calls` is the merge's own capture cost — the quantity the native
    engine keeps at zero.
    """

    def __init__(self, monkeypatch):
        self.locations = _CaptureCounter(ops_utils.get_location)
        self.stack = _StackCounter()
        monkeypatch.setattr(ops_utils, "get_location", self.locations)
        monkeypatch.setattr(inspect, "stack", self.stack)


def counting_merge(monkeypatch, graphs, **kwargs):
    """`batch_functions(graphs, **kwargs)` + the merge-time counters."""
    counters = _MergeCounters(monkeypatch)
    graph = evoxc.batch_functions(graphs, **kwargs)
    return graph, counters


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


def stacked_constant_population(n_functions=6):
    """Every program carries its OWN constant payload in one shared skeleton."""
    spec = evoxc.TensorSpec((1024,), F32)
    graphs = []
    for index in range(n_functions):
        constant = float(index + 1)

        def program(x, _constant=constant):
            return _times_plus(x, _constant)

        graphs.append(evoxc.trace(program, spec))
    return graphs


def three_op_program(x):
    """Three ops on three DISTINCT source lines (the location-set anchor)."""
    y = x + 1.0
    z = y - 2.0
    return z * 3.0


def two_input_program(x, y):
    """Two tensor inputs on one source line (the `source_locations` anchor)."""
    return (x * y) + 1.0


# ---------------------------------------------------------------------------
# 1. Zero Python location/introspection work during a merge
# ---------------------------------------------------------------------------

#: `(population factory, merge kwargs, expected report fields)`.
#: Every count is the MEASURED native value at this HEAD for that fixed
#: population. `alignment` is stated whenever a count depends on it (the class
#: alignment policy changes how many classes a switched position emits, so a
#: count is only meaningful next to its policy); an omitted `alignment` means
#: the default `"variant"` applies.
MERGE_CASES = [
    pytest.param(
        lambda: chain_population(12, 32),
        {"switch_ops": True},
        {
            "n_functions": 32,
            "value_classes": 13,
            "shared_classes": 13,
            "switch_classes": 0,
            "stacked_constants": 0,
            "output_muxes": 0,
            "emitted_ops": 25,
            "bands": 1,
            "band_sizes": (32,),
            "band_strategy": "flat",
            "costs": None,
        },
        id="flat-identical-32x12op-chain",
    ),
    pytest.param(
        divergent_population,
        {"switch_ops": True, "alignment": "occurrence"},
        {
            "n_functions": 10,
            "value_classes": 5,
            "shared_classes": 5,
            "switch_classes": 2,
            "stacked_constants": 1,
            "output_muxes": 1,
            "emitted_ops": 17,
            "bands": 1,
            "band_sizes": (10,),
            "band_strategy": "flat",
            "costs": None,
        },
        id="flat-divergent-10-occurrence",
    ),
    pytest.param(
        divergent_population,
        {"switch_ops": True, "bands": 3, "alignment": "occurrence"},
        {
            "n_functions": 10,
            "value_classes": 15,
            "shared_classes": 12,
            "switch_classes": 6,
            "stacked_constants": 3,
            "output_muxes": 3,
            "emitted_ops": 55,
            "bands": 3,
            "band_sizes": (4, 3, 3),
            "band_strategy": "explicit",
            "costs": None,
        },
        id="banded-explicit-3-occurrence",
    ),
    pytest.param(
        divergent_population,
        {"switch_ops": True, "bands": "auto", "alignment": "occurrence"},
        {
            "n_functions": 10,
            "value_classes": 5,
            "shared_classes": 5,
            "switch_classes": 2,
            "stacked_constants": 1,
            "output_muxes": 1,
            "emitted_ops": 17,
            "bands": 1,
            "band_sizes": (10,),
            "band_strategy": "auto",
            # The model ran and found nothing cheaper than the flat merge.
            "costs": (86144.0, 86144.0),
        },
        id="banded-auto-flat-partition-occurrence",
    ),
    pytest.param(
        lambda: divergent_population(
            variants=(_plus_minus, _plus_plus, _times_plus)
        ),
        {"switch_ops": True, "bands": "auto"},
        {
            "n_functions": 10,
            "value_classes": 8,
            "shared_classes": 8,
            "switch_classes": 0,
            "stacked_constants": 2,
            "output_muxes": 1,
            "emitted_ops": 21,
            "bands": 2,
            "band_sizes": (7, 3),
            "band_strategy": "auto",
            # The auto plan's exact static-cost output for this population.
            "costs": (54528.0, 75392.0),
        },
        id="banded-auto-two-bands",
    ),
    # The DEFAULT policy (`alignment` omitted → `"variant"`): the same `bands=3`
    # merge as `banded-explicit-3-occurrence` emits 40 ops instead of 55,
    # because sibling variants share one class per skeleton. Both policies stay
    # pinned.
    pytest.param(
        divergent_population,
        {"switch_ops": True, "bands": 3},
        {
            "n_functions": 10,
            "value_classes": 18,
            "shared_classes": 10,
            "switch_classes": 0,
            "stacked_constants": 3,
            "output_muxes": 3,
            "emitted_ops": 40,
            "bands": 3,
            "band_sizes": (4, 3, 3),
            "band_strategy": "explicit",
            "costs": None,
        },
        id="banded-explicit-3-default-variant",
    ),
]


@pytest.mark.parametrize("build, kwargs, expected", MERGE_CASES)
def test_merge_does_no_python_location_or_introspection_work(
    monkeypatch, build, kwargs, expected
):
    """A merge captures NO Python location and never walks `inspect.stack`.

    All source locations are captured while TRACING and travel through the
    native envelope, so the merge itself must make zero `get_location()` calls —
    whatever the emission size — and must never reach `inspect.stack()`.
    """
    graphs = build()
    graph, counters = counting_merge(monkeypatch, graphs, **kwargs)
    verifies(graph)

    assert counters.locations.calls == 0, (
        "the native merge must capture no Python call site, got "
        f"{counters.locations.calls} get_location() call(s)"
    )
    assert counters.stack.calls == 0, (
        "location capture is a sys._getframe walk — it must never reach "
        f"inspect.stack(), got {counters.stack.calls} call(s)"
    )
    # The merged IR really does carry source locations: they came from the
    # traced programs, not from a capture the merge would have made.
    assert any(
        op.location is not None and op.location.file == __file__
        for op in non_terminator_ops(graph)
    )
    # …and the report still describes the merge.
    assert graph.batch_report.emitted_ops == expected["emitted_ops"]


def test_tracing_is_where_locations_are_captured(monkeypatch):
    """Positive control: the live counter fires while TRACING, not while merging.

    Without this leg the zero counts above could come from a patch that misses
    the ops' lookup. Tracing three ops increments the counter; the merge that
    follows leaves it untouched.
    """
    counter = _CaptureCounter(ops_utils.get_location)
    stack = _StackCounter()
    monkeypatch.setattr(ops_utils, "get_location", counter)
    monkeypatch.setattr(inspect, "stack", stack)

    spec = evoxc.TensorSpec((4,), F32)
    traced = evoxc.trace(three_op_program, spec)
    assert counter.calls > 0, "tracing must capture its op call sites"
    assert stack.calls == 0, "tracing uses sys._getframe, never inspect.stack"
    for op in non_terminator_ops(traced):
        assert op.location is not None and op.location.file == __file__

    captured_at_trace_time = counter.calls
    merged = verifies(evoxc.batch_functions([traced, traced], switch_ops=True))
    assert counter.calls == captured_at_trace_time
    assert stack.calls == 0
    assert merged.batch_report.emitted_ops == len(non_terminator_ops(merged))


@pytest.mark.parametrize("length, emitted_ops", [(12, 25), (24, 49), (48, 97)])
def test_merge_cost_does_not_scale_with_the_emitted_op_count(
    monkeypatch, length, emitted_ops
):
    """4× the emitted ops ⇒ the SAME zero captures (the native contract's point).

    The same 32 identical programs are merged at three chain lengths; only the
    emission grows (2·length + 1 ops), the merge-time capture count stays at 0.
    """
    graphs = chain_population(length, 32)
    graph, counters = counting_merge(monkeypatch, graphs, switch_ops=True)
    verifies(graph)

    assert graph.batch_report.emitted_ops == emitted_ops
    assert graph.batch_report.emitted_ops > 10
    assert counters.locations.calls == 0
    assert counters.stack.calls == 0
    assert len(entry_ops(graph)) == emitted_ops + 1  # + the `return`


# ---------------------------------------------------------------------------
# 2. Emitted-op pins read the 12-key `graph.batch_report`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build, kwargs, expected", MERGE_CASES)
def test_report_counts_describe_the_merge(build, kwargs, expected):
    """Every structural count comes from `graph.batch_report`, cross-checked.

    The report is plain transform metadata (never persisted), but it is the
    documented description of what one merge did — these pins read it instead
    of any private emit helper.
    """
    graph = verifies(evoxc.batch_functions(build(), **kwargs))
    report = graph.batch_report

    assert isinstance(report, evoxc.BatchReport)
    assert tuple(report.as_dict()) == REPORT_KEYS  # the 12 keys, in order
    assert len(REPORT_KEYS) == 12

    for field in (
        "n_functions",
        "value_classes",
        "shared_classes",
        "stacked_constants",
        "output_muxes",
        "emitted_ops",
        "bands",
        "band_sizes",
        "band_strategy",
    ):
        assert getattr(report, field) == expected[field], field

    # Switch records are the group/variant/rows description of a switched
    # position; their rows partition the functions they cover.
    assert isinstance(report.switched, tuple)
    assert len(report.switched) == expected["switch_classes"]
    for record in report.switched:
        assert isinstance(record, evoxc.SwitchRecord)
        assert record.variants
        assert len(record.variants) == len(record.rows_per_variant)
        rows = [
            row for variant_rows in record.rows_per_variant for row in variant_rows
        ]
        assert len(rows) == len(set(rows)), "a function appears twice in one class"
        assert all(0 <= row < report.n_functions for row in rows)

    # The class counters describe a subset relation, the band fields a partition.
    assert 0 <= report.shared_classes <= report.value_classes
    assert len(report.band_sizes) == report.bands
    assert sum(report.band_sizes) == report.n_functions

    # The report's op count IS the entry block minus its terminator, and these
    # populations all have a rank-1 output leaf, so the entry block holds
    # exactly `emitted_ops + 1` ops.
    assert report.emitted_ops == len(non_terminator_ops(graph))
    assert len(entry_ops(graph)) == report.emitted_ops + 1
    assert entry_ops(graph)[-1].name == "return"

    # Cost estimates are the static model's — reported only for `"auto"`.
    if expected["costs"] is None:
        assert report.cost_estimate is None
        assert report.flat_cost_estimate is None
    else:
        assert (
            report.cost_estimate,
            report.flat_cost_estimate,
        ) == expected["costs"]
        assert report.cost_estimate <= report.flat_cost_estimate


def test_stacked_constants_report_one_row_per_function():
    """`stacked_constants` counts the ONE `(N, …)` constant folding N payloads.

    Distinct per-program payloads in one shared skeleton collapse into a single
    stacked constant whose row i is program i's payload — visible both in the
    report and in the merged IR (the old emitter helper is not consulted).
    """
    graphs = stacked_constant_population()
    graph = verifies(evoxc.batch_functions(graphs, switch_ops=True))
    report = graph.batch_report

    assert (report.n_functions, report.stacked_constants) == (6, 1)
    assert report.output_muxes == 0
    assert report.switched == ()
    assert report.emitted_ops == 5
    assert len(entry_ops(graph)) == report.emitted_ops + 1

    constants = [
        op for op in non_terminator_ops(graph) if op.name == "constant"
    ]
    assert len(constants) == 1, "one stacked constant, no mask constants"
    payload = constants[0].attributes["value"]
    assert payload.shape == (6,)
    np.testing.assert_array_equal(
        payload, np.arange(1, 7, dtype=payload.dtype)
    )


# ---------------------------------------------------------------------------
# 3. Locations — inherited from the contributing ops, never invented
# ---------------------------------------------------------------------------


def test_merged_ops_inherit_the_contributing_source_locations():
    """Set equality: the merged emission covers EXACTLY the source lines.

    Three identical programs, each with three ops on three distinct source
    lines. The merged graph inserts alignment `reshape` ops, but every ruled
    op is stamped by the native engine with its contributing op's location, so
    the merged `(file, line)` set equals the source program's — no line
    invented (in particular not the `batch_functions` call site), none lost.
    """
    spec = evoxc.TensorSpec((4,), F32)
    graphs = [evoxc.trace(three_op_program, spec) for _ in range(3)]
    graph = verifies(evoxc.batch_functions(graphs, switch_ops=True))

    source_ops = non_terminator_ops(graphs[0])
    source_lines = {(op.location.file, op.location.line) for op in source_ops}
    assert len(source_lines) == 3, "the probe program must span three lines"

    stamped = [
        op for op in non_terminator_ops(graph) if op.location is not None
    ]
    assert stamped, "the merge emitted nothing with a contributing location"
    for op in stamped:
        assert isinstance(op.location, ir.Location)
        assert (op.location.file, op.location.line) in source_lines
        assert op.location.col == 0
    assert {(op.location.file, op.location.line) for op in stamped} == source_lines
    assert all(op.location.file == __file__ for op in stamped)


def test_merge_owned_helper_ops_and_the_terminator_carry_no_location():
    """Mask/stack/`select`/band helpers and the `return` have `location is None`.

    Rust has no frame-capture call site, so the ops the merge machinery emits
    itself carry no location at all — distinct from the inherited locations of
    the rule-emitted ops above.
    """
    graphs = divergent_population()
    graph = verifies(
        evoxc.batch_functions(graphs, switch_ops=True, alignment="occurrence")
    )
    ops = entry_ops(graph)
    assert ops[-1].name == "return"
    assert ops[-1].location is None

    unlocated = [op for op in ops[:-1] if op.location is None]
    assert unlocated, "this switched merge must emit mask/select helpers"
    assert {op.name for op in unlocated} >= {"constant", "select"}

    # The banded machinery's band views and output glue are merge-owned too.
    banded = verifies(
        evoxc.batch_functions(divergent_population(), switch_ops=True, bands=3)
    )
    banded_ops = entry_ops(banded)
    assert banded_ops[-1].name == "return"
    assert banded_ops[-1].location is None
    assert {op.name for op in banded_ops[:-1] if op.location is None} >= {
        "slice",
        "concatenate",
    }


def test_uncaptured_program_locations_stay_the_unknown_placeholder(monkeypatch):
    """A contributor's `<unknown>` placeholder is kept, never upgraded.

    With capture disabled the traced programs carry `ir.Location.unknown()`
    (file `<unknown>`, line 0); the merge does not invent a call site for them —
    every stamped merged op carries that SAME placeholder, and the merge-owned
    helper ops / `return` stay `None`.
    """
    monkeypatch.setenv("EVOXC_DISABLE_LOCATIONS", "1")
    spec = evoxc.TensorSpec((4,), F32)
    graphs = [evoxc.trace(three_op_program, spec) for _ in range(2)]
    for op in non_terminator_ops(graphs[0]):
        assert op.location == ir.Location.unknown()

    graph = verifies(evoxc.batch_functions(graphs, switch_ops=True))
    unknown = ir.Location.unknown()
    stamped = [
        op for op in non_terminator_ops(graph) if op.location is not None
    ]
    assert stamped
    for op in stamped:
        assert op.location == unknown, (
            f"op '{op.name}' was rewritten with location {op.location} instead "
            "of the contributor's <unknown> placeholder"
        )
    for op in non_terminator_ops(graph):
        assert op.location is None or op.location == unknown
    assert entry_ops(graph)[-1].name == "return"
    assert entry_ops(graph)[-1].location is None


def test_source_locations_survive_the_merge():
    """`graph.source_locations` keeps one entry per MERGED input argument.

    The entries are the contributing programs' recorded input locations (the
    `evoxc.trace` call sites), remapped onto the merged block arguments; the
    merge's own ops add none.
    """
    spec = evoxc.TensorSpec((4,), F32)
    graphs = [evoxc.trace(three_op_program, spec) for _ in range(3)]
    graph = verifies(evoxc.batch_functions(graphs, switch_ops=True))

    block_args = list(graph.module.main.entry_block.arguments)
    assert len(block_args) == 1
    assert len(graphs[0].source_locations) == 1
    assert set(graph.source_locations) == {argument.id for argument in block_args}

    location = next(iter(graph.source_locations.values()))
    assert isinstance(location, ir.Location)
    assert location.file == __file__
    assert location.line > 0
    assert location == next(iter(graphs[0].source_locations.values()))

    # The map describes INPUTS: no merged op result gets an entry.
    result_ids = {
        value.id for op in non_terminator_ops(graph) for value in op.results
    }
    assert not (set(graph.source_locations) & result_ids)


def test_every_merged_input_keeps_a_source_location_entry():
    """A two-input population carries BOTH input entries through the merge."""
    spec = evoxc.TensorSpec((4,), F32)
    graphs = [evoxc.trace(two_input_program, spec, spec) for _ in range(2)]
    graph = verifies(
        evoxc.batch_functions(graphs, axes=(0, None), switch_ops=True)
    )

    block_args = list(graph.module.main.entry_block.arguments)
    assert len(block_args) == 2
    assert set(graph.source_locations) == {argument.id for argument in block_args}
    for location in graph.source_locations.values():
        assert isinstance(location, ir.Location)
        assert location.file == __file__
        assert location.line > 0


# ---------------------------------------------------------------------------
# 4. `EVOXC_DISABLE_LOCATIONS=1` — locations only, never semantics
# ---------------------------------------------------------------------------


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
