"""Tests for the ``bands=`` partition of `evoxc.batch_functions`.

Binding contract: `evoxc/transforms/CONTEXT.md` § "Bands — clustered merge
(`bands=`, `_bands.py`)" + the `bands` half of the `batch_functions` docstring
and `_merge_graphs` validation order.

`bands` partitions the N programs BEFORE merging: ``None`` (the flat merge,
byte-identical to the pre-`bands` output), ``int K`` with ``1 <= K <= N``
(`_bands.even_partition` — K contiguous bands, sizes as even as possible,
LARGER bands first: ``N=10, K=3`` → ``(4, 3, 3)``), or ``"auto"`` (the static
cost model picks among the flat, structural-fingerprint and
fingerprint-with-coalesced-singletons partitions). Bands merge independently
and are glued with one `concatenate` along the static function axis (plus one
static inverse-permutation `gather` when the bands are not in function order).

Coverage (numpy backend unless stated; staging is always explicit — never
`evoxc.build`/`evoxc.evaluate`, which trace a callable rather than accept a
merged `Graph`):

* **A. Validation** — normalized BEFORE any tracing (``bool`` → `TypeError`,
  out-of-range ``K`` / unknown string → `core.TransformError` naming the
  accepted forms), `bands` + ``out_axes=None`` → v1 `core.TransformError`, and
  the standard merge validation still runs FIRST (a `bands=2` merge containing
  `evoxc.sort` reports the op deferral naming the function index).
* **B. Row exactness** — ``run(merged)[i]`` is function i's own run for
  ``bands in {1, 2, 3, 10}``: mapped chains, an unmapped shared input, a
  multi-output program, a rank-0 per-row output, a bool-dtype output. The
  populations use small DYADIC float32 values, so the comparison is
  ``np.array_equal`` — bit-exact, never an allclose tolerance.
* **C. `bands=1` ≡ `bands=None`** — byte-identical StableHLO export text and
  entry-block op-name sequence; only the report's band fields differ.
* **D. `bands="auto"`** — identical programs resolve to ONE band, two disjoint
  structural groups to TWO bands (never costlier than flat), and `switch_ops`
  collapses the fingerprints into a single band.
* **E. Global switch rows** — a banded `SwitchRecord.rows_per_variant` holds
  GLOBAL function indices, never band-local ones.
* **F. Persistence** — `Graph.save`/`load` round-trips the numbers and drops
  `batch_report`.
* **G. IREE / llvm-cpu** — one banded merge lowered/compiled/run on iree
  matches the numpy backend (guarded: the file is green compiler-free).
"""
import importlib.util

import numpy as np
import pytest

import evoxc
from evoxc import core
from evoxc.backends.stablehlo import export

# --- helpers (module level — no fixtures, no conftest changes) --------------


def run_graph(graph, *args):
    """Explicit staging pipeline: lower → compile → load → run."""
    return evoxc.run(evoxc.load(evoxc.compile(evoxc.lower(graph))), *args)


def to_np_tree(value):
    """Recursively replace `evoxc.Tensor` leaves with numpy arrays."""
    if isinstance(value, core.Tensor):
        return value.numpy()
    if isinstance(value, dict):
        return {key: to_np_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_np_tree(item) for item in value)
    return value


def op_names(graph):
    """Entry-block op names, in emission order (`return` terminator included)."""
    return [op.name for op in graph.module.main.entry_block.ops]


def merged(fns, **kwargs):
    """`evoxc.batch_functions(...)` plus the caller's `verify()` obligation.

    The transform deliberately does NOT call `ir.verify` (staging stays
    explicit), so every merged graph in this file goes through this helper.
    """
    graph = evoxc.batch_functions(fns, **kwargs)
    assert isinstance(graph, evoxc.Graph)
    graph.verify()  # raises core.VerificationError on bad IR
    return graph


def stack_reference(graphs, row_args):
    """Stack every program's OWN run on its row of the mapped inputs."""
    return np.stack([to_np_tree(run_graph(g, *a)) for g, a in zip(graphs, row_args)])


def assert_row_exact(out, graphs, row_args):
    """Row i of `out` is program i's own run — BIT-exactly (`array_equal`).

    Small dyadic values + switching-by-`select` + band slicing/gathering keep
    every row's arithmetic identical to the standalone program, so float32
    rounding cannot differ: exact equality is the honest assertion.
    """
    actual = to_np_tree(out)
    expected = stack_reference(graphs, row_args)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected), "banded rows are not bit-exact"


#: ``N -> {K: the band sizes `even_partition(N, K)` must produce}``: contiguous,
#: larger bands first, a permutation of the function axis.
BAND_SIZES = {10: {1: (10,), 2: (5, 5), 3: (4, 3, 3), 10: (1,) * 10},
              4: {1: (4,), 2: (2, 2), 4: (1,) * 4}}

# ---------------------------------------------------------------------------
# A. Validation — before any tracing, and after the standard merge checks
# ---------------------------------------------------------------------------


def _three_simple_graphs():
    """Three same-signature programs (``K`` is validated against this N = 3)."""
    spec = evoxc.TensorSpec((3,), evoxc.float32)
    return [evoxc.trace(fn, spec) for fn in
            (lambda x: x + 1.0, lambda x: x - 1.0, lambda x: x * 1.0)]


@pytest.mark.parametrize("bands", [0, -1, 4])
def test_band_count_out_of_range(bands):
    """``K`` outside ``1..N`` is a `core.TransformError` naming the range."""
    with pytest.raises(core.TransformError) as excinfo:
        evoxc.batch_functions(_three_simple_graphs(), bands=bands)  # N = 3
    message = str(excinfo.value)
    assert "out of range" in message and f"bands={bands}" in message
    assert "1 <= K <= N" in message


def test_unknown_bands_string_is_rejected():
    """An unknown string (``"auto"`` is the only one) names the accepted forms."""
    with pytest.raises(core.TransformError) as excinfo:
        evoxc.batch_functions(_three_simple_graphs(), bands="banana")
    assert "unknown bands value 'banana'" in str(excinfo.value)
    assert "'auto'" in str(excinfo.value)


@pytest.mark.parametrize(
    "bands, fragment",
    [
        pytest.param(2.5, "got float", id="float"),
        # a `bool` is an `int` subclass but is NEVER a band count
        pytest.param(True, "booleans are not accepted", id="bool"),
        pytest.param(["auto"], "got list", id="list"),
        pytest.param({"bands": 2}, "got dict", id="dict"),
    ],
)
def test_bands_type_errors(bands, fragment):
    """A non-`None`/`int`/`str` spelling is a `TypeError` (never a fallback)."""
    with pytest.raises(TypeError) as excinfo:
        evoxc.batch_functions(_three_simple_graphs(), bands=bands)
    assert fragment in str(excinfo.value)


def test_bands_is_validated_before_any_tracing():
    """A bad `bands` on the CALLABLE path raises before a program is traced."""
    calls = []

    def program(x):
        calls.append(x)  # tracing executes the Python body once per spec
        return x + 1.0

    with pytest.raises(core.TransformError, match="out of range"):
        evoxc.batch_functions([program, program], bands=0)
    assert calls == [], "the band count must be validated before any tracing"
    # A VALID `bands` traces nothing either: the callable path only returns a
    # TransformCallable, whose ``(*specs)`` invocation is what traces.
    assert evoxc.batch_functions([program, program], bands=2).kind == "batch_functions"
    assert calls == []


@pytest.mark.parametrize("bands", [1, 2], ids=["bands-one", "bands-two"])
def test_bands_with_out_axes_none_is_rejected(bands):
    """`out_axes=None` on a tensor output is a banded-merge v1 restriction.

    A band produces a per-band value, which is concatenated along the function
    axis — there is no single unmapped output to produce. Checked before the
    partition, so ``bands=1`` obeys the same contract as any other K.
    """
    with pytest.raises(core.TransformError) as excinfo:
        evoxc.batch_functions(_three_simple_graphs(), bands=bands, out_axes=None)
    message = str(excinfo.value)
    assert "out_axes=None is not supported with bands=" in message
    assert "out_axes=0 for every tensor output" in message


def test_banded_out_axes_check_runs_after_the_op_check():
    """The standard merge validation still wins over the banded-out_axes check.

    `_merge_graphs` validates program shape, the cross-function contract, op
    batchability and output pytree nodes BEFORE resolving the partition, so a
    ``bands=2`` + ``out_axes=None`` merge containing `evoxc.sort` reports the
    op deferral (naming the function index), exactly like the flat merge.
    """
    spec = evoxc.TensorSpec((5,), evoxc.float32)
    graphs = [evoxc.trace(lambda x: evoxc.sort(x), spec),
              evoxc.trace(lambda x: x + 1.0, spec)]
    for bands, kwargs in ((None, {}), (2, {"out_axes": None})):
        with pytest.raises(core.TransformError) as excinfo:
            evoxc.batch_functions(graphs, bands=bands, **kwargs)
        message = str(excinfo.value)
        assert "cannot batch op 'sort' from function 0" in message
        assert "no batching rule registered for key 'sort'" in message


# ---------------------------------------------------------------------------
# B. Row exactness — row i of the merged run == program i's own run (bit-exact)
# ---------------------------------------------------------------------------

#: The per-row spec of the section-B populations (rank 1, tiny) and an
#: exactly-representable input: every value and intermediate below is a small
#: dyadic rational, so float32 arithmetic on them is EXACT.
ROW_SPEC = evoxc.TensorSpec((4,), evoxc.float32)
ROW_INPUT = np.array([-3.0, -1.0, 2.0, 5.0], dtype=np.float32)

#: Ten programs over ``ROW_SPEC``: some differ in exactly ONE op at a shared
#: skeleton position, some are two-op chains, and several distinct constant
#: payloads are in play (so the flat merge really has to stack constants).
CHAIN_PROGRAMS = (
    ("add", 1.0), ("subtract", 1.0), ("multiply", 2.0), ("chain", (1.0, 2.0)),
    ("multiply", 0.5), ("chain", (4.0, 4.0)), ("add", 4.0), ("subtract", 2.0),
    ("multiply", 0.25), ("chain", (0.5, 0.5)),
)


def _program(kind, argument):
    """One same-signature elementwise program (a fresh closure per payload)."""
    if kind == "chain":
        inner, outer = argument

        def chained(x):
            return evoxc.multiply(evoxc.add(x, inner), outer)

        return chained
    op = {"add": evoxc.add, "subtract": evoxc.subtract,
          "multiply": evoxc.multiply}[kind]

    def elementwise(x):
        return op(x, argument)

    return elementwise


def chain_graphs():
    """Trace the 10-program population once per call."""
    return [evoxc.trace(_program(kind, arg), ROW_SPEC) for kind, arg in CHAIN_PROGRAMS]


def mapped_rows(n_functions, row=ROW_INPUT):
    """The ``(N, *row.shape)`` mapped input (every row is the same vector)."""
    return np.stack([row] * n_functions).astype(np.float32)


def single_output_check(graphs, bands, args, row_args, **kwargs):
    """Merge, run on `*args`, assert row exactness + the explicit-partition report."""
    graph = merged(graphs, bands=bands, **kwargs)
    assert_row_exact(run_graph(graph, *args), graphs, row_args)
    report = graph.batch_report
    assert report.band_sizes == BAND_SIZES[len(graphs)][bands]
    assert report.bands == len(report.band_sizes)
    assert sum(report.band_sizes) == report.n_functions == len(graphs)
    assert report.band_strategy == "explicit"
    return graph


@pytest.mark.parametrize("switch_ops", [None, True], ids=["switch-none", "switch-true"])
@pytest.mark.parametrize("bands", [1, 2, 3, 10])
def test_row_exact_mapped_chains(bands, switch_ops):
    """A 10-program elementwise population, every band count, both policies."""
    graphs, rows = chain_graphs(), mapped_rows(10)
    graph = single_output_check(
        graphs, bands, (rows,), [(rows[i],) for i in range(10)], switch_ops=switch_ops
    )
    assert graph.tensor_specs[0].shape == (10, 4)  # static function axis


def _two_input(kind, argument, op):
    """``(x, w) ↦ op(program(x), w)`` — ``x`` mapped, ``w`` unmapped/shared."""
    inner = _program(kind, argument)

    def fn(x, w):
        return op(inner(x), w)

    return fn


@pytest.mark.parametrize("bands", [1, 2, 3, 10])
def test_row_exact_unmapped_shared_input(bands):
    """``axes=(0, None)``: one mapped input plus a shared (unmapped) one.

    Every band reads the SAME unmapped block argument (no per-band view), so
    the shared value never gains the function axis.
    """
    two_input_ops = (evoxc.multiply, evoxc.add, evoxc.subtract)
    graphs = [
        evoxc.trace(_two_input(kind, arg, two_input_ops[i % 3]), ROW_SPEC, ROW_SPEC)
        for i, (kind, arg) in enumerate(CHAIN_PROGRAMS)
    ]
    rows, shared = mapped_rows(10), np.array([1.0, 2.0, 4.0, 0.5], np.float32)
    graph = single_output_check(
        graphs, bands, (rows, shared), [(rows[i], shared) for i in range(10)],
        axes=(0, None), switch_ops=True,
    )
    assert graph.tensor_specs[0].shape == (10, 4)
    assert graph.tensor_specs[1].shape == (4,)  # never gains the function axis


@pytest.mark.parametrize("bands", [1, 2, 3, 10])
def test_row_exact_multi_output(bands):
    """A tuple output with ``out_axes=(0, 0)``: BOTH leaves are concatenated."""
    def program(constant):
        def fn(x):
            return (evoxc.add(x, constant), evoxc.multiply(x, constant))

        return fn

    graphs = [evoxc.trace(program(float(i + 1)), ROW_SPEC) for i in range(10)]
    graph, rows = merged(graphs, out_axes=(0, 0), bands=bands), mapped_rows(10)
    out = run_graph(graph, rows)
    reference = [to_np_tree(run_graph(g, rows[i])) for i, g in enumerate(graphs)]
    for leaf_index in range(2):
        actual = to_np_tree(out[leaf_index])
        expected = np.stack([row[leaf_index] for row in reference])
        assert actual.shape == expected.shape == (10, 4)
        assert np.array_equal(actual, expected)
    assert graph.batch_report.band_sizes == BAND_SIZES[10][bands]


def _four_program_population(factory):
    """Trace ``factory(0.0 .. 3.0)`` over ``ROW_SPEC`` (4 programs)."""
    return [evoxc.trace(factory(float(i)), ROW_SPEC) for i in range(4)]


def _single_output_equal(graphs, bands):
    """Merge at `bands`, run rows, and return ``(out, expected)`` as numpy."""
    rows = mapped_rows(len(graphs))
    out = to_np_tree(run_graph(merged(graphs, bands=bands), rows))
    expected = np.stack(
        [to_np_tree(run_graph(g, rows[i])) for i, g in enumerate(graphs)]
    )
    assert out.shape == expected.shape
    assert out.dtype == expected.dtype
    assert np.array_equal(out, expected)
    return out, expected


@pytest.mark.parametrize("bands", [1, 2, 4])
def test_row_exact_rank0_output(bands):
    """A rank-0 per-row output is broadcast to ``(N,)`` — banding included.

    A band's per-row reduction value is rank 0 and unmapped, so the band
    combiner broadcasts it to ``(band_size,)`` before the concatenation.
    """
    def factory(constant):
        def fn(x):
            return evoxc.sum(evoxc.multiply(x, constant))

        return fn

    graphs = _four_program_population(factory)
    out, expected = _single_output_equal(graphs, bands)
    assert out.shape == expected.shape == (4,)
    assert merged(graphs, bands=bands).batch_report.band_sizes == BAND_SIZES[4][bands]


@pytest.mark.parametrize("bands", [1, 2, 4])
def test_row_exact_bool_output(bands):
    """A bool-dtype output (per-program ``greater`` constants) stays exact."""
    def factory(constant):
        def fn(x):
            return evoxc.greater(x, constant)

        return fn

    graphs = _four_program_population(factory)
    out, expected = _single_output_equal(graphs, bands)
    assert out.dtype == expected.dtype == np.bool_
    assert out.shape == expected.shape == (4, 4)


# ---------------------------------------------------------------------------
# C. `bands=1` is BYTE-IDENTICAL to the flat merge
# ---------------------------------------------------------------------------


def flat_population_graphs():
    """16 programs, each ONE different elementwise op with its own payload."""
    spec, ops = evoxc.TensorSpec((4,), evoxc.float32), (
        evoxc.add, evoxc.subtract, evoxc.multiply, evoxc.divide)
    graphs = []
    for index in range(16):
        op, constant = ops[index % 4], float(index + 1)

        def fn(x, op=op, constant=constant):
            return op(x, constant)

        graphs.append(evoxc.trace(fn, spec))
    return graphs


def test_bands_one_is_byte_identical_to_the_flat_merge():
    """A single band delegates to the flat emitter — only the report differs."""
    graphs = flat_population_graphs()
    flat, single = merged(graphs, bands=None), merged(graphs, bands=1)
    # (a) byte-identical MLIR text and entry-block op sequence …
    assert export(single) == export(flat)
    assert op_names(single) == op_names(flat)
    assert "stablehlo." in export(single)
    # (b) … bit-identical numbers …
    rows = np.random.default_rng(5).standard_normal((16, 4)).astype(np.float32)
    assert np.array_equal(
        to_np_tree(run_graph(single, rows)), to_np_tree(run_graph(flat, rows))
    )
    # (c) … identical specs, and ONLY the band fields of the report differ.
    assert single.tensor_specs[0].shape == flat.tensor_specs[0].shape == (16, 4)
    report, flat_report = single.batch_report, flat.batch_report
    assert (report.bands, report.band_sizes) == (1, (16,))
    assert report.band_strategy == "explicit"
    assert flat_report.band_strategy == "flat"
    assert report.emitted_ops == flat_report.emitted_ops
    assert report.cost_estimate is None and report.flat_cost_estimate is None


# ---------------------------------------------------------------------------
# D. `bands="auto"` — the static cost model's partition
# ---------------------------------------------------------------------------

#: The "two disjoint structural groups" population: group A is ``(x + c) + c``
#: (two adds), group B is ``(x * c) - c`` (multiply then subtract) — the SAME
#: skeleton length and op category, so only a switch-group-free fingerprint
#: tells them apart.
GROUP_VECTOR_SPEC = evoxc.TensorSpec((1024,), evoxc.float32)


def _group_a(constant):
    def fn(x):
        return evoxc.add(evoxc.add(x, constant), constant)

    return fn


def _group_b(constant):
    def fn(x):
        return evoxc.subtract(evoxc.multiply(x, constant), constant)

    return fn


def two_group_graphs():
    """5 group-A + 5 group-B programs over the same (1024,) spec."""
    first = [evoxc.trace(_group_a(float(i + 1)), GROUP_VECTOR_SPEC) for i in range(5)]
    second = [evoxc.trace(_group_b(float(i + 1)), GROUP_VECTOR_SPEC) for i in range(5)]
    return first + second


def test_auto_keeps_identical_programs_in_one_band():
    """Identical programs have ONE fingerprint, so auto resolves to one band."""
    spec = evoxc.TensorSpec((8,), evoxc.float32)

    def program(x):
        return evoxc.add(evoxc.multiply(x, 2.0), 2.0)

    graphs = [evoxc.trace(program, spec) for _ in range(5)]
    graph = merged(graphs, switch_ops=True, bands="auto")
    report = graph.batch_report
    assert (report.bands, report.band_sizes) == (1, (5,))
    assert report.band_strategy == "auto"
    # The model ran and found nothing cheaper than the flat merge.
    assert report.cost_estimate == report.flat_cost_estimate == 1232.0
    # A single-band auto resolution emits the flat graph byte for byte.
    assert export(graph) == export(merged(graphs, switch_ops=True, bands=None))
    rows = np.stack([np.arange(8, dtype=np.float32)] * 5)
    assert_row_exact(run_graph(graph, rows), graphs, [(rows[i],) for i in range(5)])


def test_auto_splits_two_disjoint_groups():
    """Two structural groups → two fingerprint bands, never costlier than flat."""
    graphs = two_group_graphs()
    graph = merged(graphs, bands="auto")
    report = graph.batch_report
    assert (report.bands, report.band_sizes) == (2, (5, 5))
    assert sum(report.band_sizes) == report.n_functions == 10
    assert report.band_strategy == "auto"
    # The model's own verdict: the chosen partition is not costlier than flat.
    assert report.cost_estimate is not None and report.flat_cost_estimate is not None
    assert report.cost_estimate <= report.flat_cost_estimate
    # … and it really is row-exact (same op sequence per row → bit-identical).
    rows = np.random.default_rng(3).standard_normal((10, 1024)).astype(np.float32)
    assert_row_exact(run_graph(graph, rows), graphs, [(rows[i],) for i in range(10)])


def test_auto_with_switch_ops_collapses_the_fingerprints():
    """`switch_ops` erases the structural difference the fingerprints key on.

    With switch groups every elementwise position fingerprints as
    ``("group", key)`` whatever op sits there, so the two structural groups above
    become ONE fingerprint bucket: the fingerprint partition equals the flat
    partition, auto resolves to a SINGLE band and reports the flat cost. (A
    property of the partition, not a heuristic bug — once positions are
    interchangeable there is nothing left for a band to save.)
    """
    report = merged(two_group_graphs(), switch_ops=True, bands="auto").batch_report
    assert (report.bands, report.band_sizes) == (1, (10,))
    assert report.band_strategy == "auto"
    assert report.cost_estimate == report.flat_cost_estimate
    # Switching makes the FLAT partition itself pricier (each switch position
    # pays one op per variant plus its `select` mux on all 10 rows).
    unswitched = merged(two_group_graphs(), bands="auto")
    assert unswitched.batch_report.flat_cost_estimate < report.flat_cost_estimate


# ---------------------------------------------------------------------------
# E. `SwitchRecord.rows_per_variant` stays GLOBAL under banding
# ---------------------------------------------------------------------------


def test_switch_record_rows_are_global_function_indices():
    """A band's masks use band-local rows; the REPORT keeps global indices."""
    spec = evoxc.TensorSpec((3,), evoxc.float32)
    programs = (
        lambda x: evoxc.add(x, 1.0),
        lambda x: evoxc.subtract(x, 1.0),
        lambda x: evoxc.add(x, 2.0),
        lambda x: evoxc.multiply(x, 1.0),
    )
    graphs = [evoxc.trace(fn, spec) for fn in programs]
    graph = merged(graphs, switch_ops=True, bands=2)
    report = graph.batch_report
    assert report.band_sizes == (2, 2)  # band 0 = functions 0,1; band 1 = 2,3
    assert report.emitted_ops == 17
    assert [(r.group, r.variants, r.rows_per_variant) for r in report.switched] == [
        ("elementwise", ("add", "subtract"), ((0,), (1,))),
        ("elementwise", ("add", "multiply"), ((2,), (3,))),
    ]
    # The decisive pin: the multiply variant lives in band 1 at band-local row 1,
    # but the report must say GLOBAL function 3 (never the band-local 1).
    record = next(r for r in report.switched if "multiply" in r.variants)
    rows_of = record.rows_per_variant[record.variants.index("multiply")]
    assert rows_of == (3,)
    assert rows_of != (1,)
    # … and the merged graph is still row-correct.
    rows = np.array([[-1.0, 0.0, 1.0]] * 4, dtype=np.float32)
    assert_row_exact(run_graph(graph, rows), graphs, [(rows[i],) for i in range(4)])


# ---------------------------------------------------------------------------
# F. Persistence — a banded merge saves and loads like any other graph
# ---------------------------------------------------------------------------


def test_banded_merge_save_load_round_trip(tmp_path):
    """`Graph.save`/`load` keeps the numbers and drops the (non-persisted) report."""
    graph = merged(flat_population_graphs(), bands=4)  # sizes (4, 4, 4, 4)
    assert graph.batch_report.band_sizes == (4, 4, 4, 4)
    path = tmp_path / "banded.evoxc"
    graph.save(str(path))
    loaded = evoxc.Graph.load(str(path))
    loaded.verify()
    assert not hasattr(loaded, "batch_report"), "batch_report is never persisted"
    rows = np.random.default_rng(9).standard_normal((16, 4)).astype(np.float32)
    assert np.array_equal(
        to_np_tree(run_graph(graph, rows)), to_np_tree(run_graph(loaded, rows))
    )
    assert loaded.tensor_specs[0].shape == (16, 4)


# ---------------------------------------------------------------------------
# G. IREE / llvm-cpu — a banded merge is an ordinary static graph
# ---------------------------------------------------------------------------


def _module_available(name):
    """`find_spec` that treats a blocker / broken parent package as absent."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


HAVE_IREE = _module_available("iree.compiler") and _module_available("iree.runtime")
needs_iree = pytest.mark.skipif(
    not HAVE_IREE, reason="iree compiler/runtime not installed"
)

#: ``key -> Executable``: one compile per merged graph for the whole module.
_IREE_EXECUTABLES = {}


def iree_executable(key, graph):
    """Compile the merged graph ONCE for iree/llvm-cpu and cache the handle."""
    exe = _IREE_EXECUTABLES.get(key)
    if exe is None:
        lowered = evoxc.lower(graph, backend="iree", target_backends=["llvm-cpu"])
        exe = _IREE_EXECUTABLES[key] = evoxc.load(evoxc.compile(lowered))
    return exe


@needs_iree
def test_iree_llvm_cpu_banded_merge_parity():
    """A `bands=3` switched merge compiles on iree and matches numpy.

    Every band size is a static Python `int` and no symbolic `Dim` is ever
    introduced, so this is an ordinary static graph the adapter lowers
    unchanged (the band slicing/gathering and the output `concatenate` are
    plain `slice`/`gather`/`concatenate` ops).
    """
    pytest.importorskip("iree.compiler")
    pytest.importorskip("iree.runtime")
    spec, elements = evoxc.TensorSpec((3, 4), evoxc.float32), (
        evoxc.add, evoxc.subtract, evoxc.multiply)
    graphs = []
    for index in range(6):
        op, constant = elements[index % 3], float(index + 1)

        def fn(x, op=op, constant=constant):
            return op(x, constant)

        graphs.append(evoxc.trace(fn, spec))
    graph = merged(graphs, bands=3, switch_ops=True)
    report = graph.batch_report
    assert report.band_sizes == (2, 2, 2)
    assert report.switched, "expected the elementwise positions to be switched"
    assert graph.tensor_specs[0].shape == (6, 3, 4)  # static function axis

    rows = np.random.default_rng(7).standard_normal((6, 3, 4)).astype(np.float32)
    compiled = to_np_tree(evoxc.run(iree_executable("banded", graph), rows))
    reference = to_np_tree(run_graph(graph, rows))
    assert compiled.shape == (6, 3, 4)
    # (a) identical to the numpy backend on the very same banded graph …
    np.testing.assert_allclose(compiled, reference, rtol=1e-5, atol=1e-5)
    # (b) … and to the stacked per-function references.
    expected = stack_reference(graphs, [(rows[i],) for i in range(6)])
    np.testing.assert_allclose(compiled, expected, rtol=1e-5, atol=1e-5)
