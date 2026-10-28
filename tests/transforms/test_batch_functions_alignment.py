"""Tests for `evoxc.batch_functions(..., alignment=...)` — the class-alignment
policy of `switch_ops` positions.
Contract under test (binding): `evoxc/transforms/CONTEXT.md` § "Alignment of
switched positions (`alignment=`, `_merge.py`)", plus `_merge.ALIGNMENTS` /
`_merge.DEFAULT_ALIGNMENT` / `_merge.normalize_alignment` and the public
`evoxc.batch_functions` signature + docstring.

`alignment` chooses WHICH class a program's op joins at a skeleton covered by a
`switch_ops` merge group — the only positions where two programs can put
DIFFERENT variants at one skeleton:
  * `"occurrence"` — a program's k-th distinct variant at a skeleton joins that
    skeleton's k-th occurrence class (byte-identical to the output without the
    option: same class numbering, `SwitchRecord` order, emitted op sequence);
  * `"variant"` — an EXISTING class of that skeleton that already holds THIS
    variant (and does not yet hold this program) wins over the occurrence
    position, so sibling branches `(x + c) * (x - c)` vs `(x - c) * (x + c)`
    get ONE class per op instead of two mixed classes;
  * `"auto"` — `"variant"` plus a last-resort reuse of any other free class of
    the skeleton, so it never creates more classes than `"occurrence"`.

Sections: A default + byte-identity of omitted `alignment`; B `"variant"`/
`"auto"` accept a valid graph; C the MEASURED op/select counts of the branchy
population; D `"auto"` never emits more than either mode; E the correctness
invariant — every mode is row-correct and `alignment` NEVER changes a number
(`np.array_equal`-level pin); F constants still stack BY OCCURRENCE under every
alignment; G validation/errors (non-str → `TypeError`, unknown spelling →
`core.TransformError`, both before any tracing) and inertness outside
`switch_ops` groups.

DEFAULT-ALIGNMENT NOTE: the default is `"variant"` (`_merge.DEFAULT_ALIGNMENT`
and the public signature default) — exactly what `evoxc/transforms/CONTEXT.md`
documents, and what the default pins here assert. `"occurrence"` remains an
accepted EXPLICIT value: it is the byte-stable LEGACY policy (the pre-`alignment`
output — same class numbering, `SwitchRecord` order, emitted op sequence), so
every count pin below that states occurrence numbers passes
`alignment="occurrence"` explicitly.

`alignment` is a PURE performance heuristic — it can only change the op/select
COUNT, never a number — so these tests assert OP COUNTS and never timings.
Execution is explicit staging only (`lower → compile → load → run`): numpy
backend, CPU, deterministic seeded inputs.
"""
import inspect
import numpy as np
import pytest
import evoxc
from evoxc import core
from evoxc.backends import stablehlo
from evoxc.transforms import _merge
F32 = evoxc.float32
# ---------------------------------------------------------------------------
# helpers (module level, copied locally — test modules never import each
# other's helpers: `test_batch_functions*.py` are edited independently)
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
    if isinstance(value, tuple) and hasattr(value, "_fields"):  # namedtuple
        return type(value)(*(to_np_tree(item) for item in value))
    if isinstance(value, list):
        return [to_np_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_np_tree(item) for item in value)
    return value
def entry_ops(graph):
    """The merged graph's entry-block ops (the `return` terminator included)."""
    return list(graph.module.main.entry_block.ops)
def op_names(graph):
    """Entry-block op names, in emission order (terminator included)."""
    return [op.name for op in entry_ops(graph)]
def emitted_count(graph):
    """Ops emitted in the merged entry block (`return` excluded)."""
    return len(entry_ops(graph)) - 1
def select_count(graph):
    """`select` ops in the merged entry block (the mux/mask cascade cost)."""
    return op_names(graph).count("select")
def counts(graph):
    """``(emitted ops, select ops)`` of a merged graph — the alignment cost."""
    return emitted_count(graph), select_count(graph)
def merged(fns, **kwargs):
    """`batch_functions(...)` + the caller's `verify()` obligation (the merge
    deliberately does not run `ir.verify` — staging stays explicit)."""
    graph = evoxc.batch_functions(fns, **kwargs)
    assert isinstance(graph, evoxc.Graph)
    graph.verify()  # raises core.VerificationError on bad IR
    return graph
def row_reference(graphs, row_args):
    """Stack each program's OWN run on its row of the mapped inputs."""
    return np.stack(
        [
            to_np_tree(run_graph(graph, *args))
            for graph, args in zip(graphs, row_args)
        ]
    )
def assert_rowwise(out, reference, rtol=1e-5, atol=1e-6):
    """Row i of a merged output equals the stacked per-program reference."""
    out_np = to_np_tree(out)
    reference = np.asarray(reference, dtype=out_np.dtype)
    assert out_np.shape == reference.shape
    np.testing.assert_allclose(out_np, reference, rtol=rtol, atol=atol)
#: The three accepted spellings, in `_merge.ALIGNMENTS` preference order.
ALIGNMENT_MODES = ("occurrence", "variant", "auto")
#: Length-1024 row spec of the measured "branchy" population. The population
#: uses the SAME constant payload (``c = 2.0``) in BOTH branches, which is what
#: makes alignment matter: distinct payloads would put the two branch ops in two
#: constant classes anyway (constants are deliberately NOT aligned).
BRANCHY_SPEC = evoxc.TensorSpec((1024,), F32)
def branchy_program(program_index, constant=2.0, reduce=True):
    """One program of the measured branchy population: ``(x + c) * (x - c)``,
    with the two branch ops created in OPPOSITE order on odd programs, so a pair
    of programs reaches the same two skeletons with the same two variants
    (`add`, `subtract`) in swapped positions — the sibling-branch shape
    `alignment="variant"` is about. ``reduce`` adds the shared `sum` (primary).
    """
    def fn(x):
        if program_index % 2 == 0:
            left = evoxc.add(x, constant)
            right = evoxc.subtract(x, constant)
        else:
            right = evoxc.subtract(x, constant)
            left = evoxc.add(x, constant)
        product = evoxc.multiply(left, right)
        return evoxc.sum(product) if reduce else product
    return fn
def branchy_graphs(n_functions=16, reduce=True):
    """The traced branchy population (per-function `(1024,)` float32 rows)."""
    return [
        evoxc.trace(branchy_program(index, reduce=reduce), BRANCHY_SPEC)
        for index in range(n_functions)
    ]
def branchy_inputs(n_functions=16, seed=0):
    """Deterministic rowwise inputs: `(n_functions, 1024)` float32."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_functions, 1024)).astype(np.float32)
def constant_programs(n_functions=6, spec=BRANCHY_SPEC):
    """`n_functions` programs ``x ↦ (x * c) + c`` with DISTINCT `c` (1..N).

    The stacked-constant pin's population: a constant class folds its members'
    per-row payloads into ONE `(N, ...)` constant — `alignment` must not touch it.
    """
    graphs = []
    for index in range(n_functions):
        payload = float(index + 1)
        def fn(x, c=payload):
            return evoxc.add(evoxc.multiply(x, c), c)
        graphs.append(evoxc.trace(fn, spec))
    return graphs
#: The elementwise op family used to build "same skeleton, DIFFERENT variant"
#: programs; one constant payload per program keeps the constants distinct.
OPERATIONS = (evoxc.add, evoxc.subtract, evoxc.multiply, evoxc.divide)
def op_variant_graphs(spec):
    """One program per elementwise op — the maximal-divergence population."""
    graphs = []
    for index, op_fn in enumerate(OPERATIONS):
        payload = float(index + 1)
        def fn(x, op=op_fn, c=payload):
            return op(x, c)
        graphs.append(evoxc.trace(fn, spec))
    return graphs
#: MEASURED at this HEAD (16-program branchy population, `switch_ops=True`,
#: flat merge): ``alignment -> (emitted ops, select ops)``. Op/select counts are
#: static IR properties (never timings), so they are pinned exactly.
BRANCHY_COUNTS = {
    "occurrence": (19, 3),
    "variant": (7, 0),
    "auto": (7, 0),
}
#: Same population WITHOUT the shared downstream `sum` (`multiply` is then the
#: output, so one op fewer on every mode; the select cascade is unchanged).
UNREDUCED_BRANCHY_COUNTS = {
    "occurrence": (17, 3),
    "variant": (6, 0),
    "auto": (6, 0),
}
# --- A. The default: `"variant"`, byte-identical to omitting the option
def test_signature_default_is_variant():
    """The public `alignment` default is `"variant"`.

    Documented default (`evoxc/transforms/CONTEXT.md` § "Alignment of switched
    positions"): `"variant"`. `"occurrence"` remains an accepted EXPLICIT
    value — the byte-stable LEGACY policy.
    """
    parameter = inspect.signature(evoxc.batch_functions).parameters["alignment"]
    assert parameter.default == "variant"
    # The implementation's single source of truth agrees with the public
    # signature, and `DEFAULT_ALIGNMENT` is one of the accepted spellings (the
    # `ALIGNMENTS` listing order only shapes the validation error text).
    assert _merge.ALIGNMENT_VARIANT == "variant"
    assert _merge.DEFAULT_ALIGNMENT == _merge.ALIGNMENT_VARIANT
    assert _merge.DEFAULT_ALIGNMENT in _merge.ALIGNMENTS
    assert _merge.normalize_alignment(_merge.DEFAULT_ALIGNMENT) == "variant"
    assert _merge.ALIGNMENTS == ("occurrence", "variant", "auto")
    # `"occurrence"` stays a valid explicit spelling (the legacy policy).
    assert _merge.ALIGNMENT_OCCURRENCE == "occurrence"
    assert _merge.normalize_alignment("occurrence") == "occurrence"
def test_omitted_alignment_is_byte_identical_to_variant():
    """Omitting `alignment` == `alignment="variant"`, byte for byte.

    `"variant"` is the documented default, so the omitted option must emit
    exactly the explicit-`"variant"` graph; `"occurrence"` (the byte-stable
    legacy policy) is reachable only by asking for it.
    """
    graphs = branchy_graphs()
    omitted = merged(graphs, switch_ops=True)
    explicit = merged(graphs, switch_ops=True, alignment="variant")
    assert stablehlo.export(omitted) == stablehlo.export(explicit)
    assert op_names(omitted) == op_names(explicit)
    assert omitted.batch_report.as_dict() == explicit.batch_report.as_dict()
    # Non-vacuity: the pinned graphs ARE the switched variant shape (a default
    # that happened to emit the `"occurrence"` DAG would fail here) ...
    assert counts(omitted) == BRANCHY_COUNTS["variant"]
    assert counts(omitted) == counts(explicit)
    # ... and the legacy policy is a genuinely DIFFERENT graph on this
    # population, so the pin above compares two distinguishable merges.
    assert counts(merged(graphs, switch_ops=True, alignment="occurrence")) == (
        BRANCHY_COUNTS["occurrence"]
    )
    # Without a `switch_ops` merge group there is no switch position to align,
    # so `alignment` is inert: the omitted default and all three explicit modes
    # emit the identical graph (alignment only picks among SWITCHED classes).
    unswitched = [merged(graphs)] + [
        merged(graphs, alignment=mode) for mode in ALIGNMENT_MODES
    ]
    assert len({stablehlo.export(graph) for graph in unswitched}) == 1
    assert len({tuple(op_names(graph)) for graph in unswitched}) == 1
# --- B. `"variant"` / `"auto"` are accepted and build a valid graph
@pytest.mark.parametrize("alignment", ["variant", "auto"])
def test_variant_and_auto_produce_a_valid_graph(alignment):
    """Both aligned modes merge the population into a VERIFIED graph."""
    graphs = branchy_graphs()
    graph = merged(graphs, switch_ops=True, alignment=alignment)
    report = graph.batch_report
    assert isinstance(report, evoxc.BatchReport)
    assert report.n_functions == len(graphs)
    # The report's op count is the entry block's emitted-op count.
    assert report.emitted_ops == emitted_count(graph)
    assert counts(graph) == BRANCHY_COUNTS[alignment]
    # ONE class per variant at the two branch skeletons, no mixed class left to
    # mux: the switched records are gone entirely in the aligned modes.
    assert report.switched == ()
def test_variant_and_auto_reports_agree_where_the_dags_coincide():
    """On this sibling population `"auto"` resolves to the `"variant"` DAG."""
    graphs = branchy_graphs()
    variant = merged(graphs, switch_ops=True, alignment="variant")
    auto = merged(graphs, switch_ops=True, alignment="auto")
    assert variant.batch_report.as_dict() == auto.batch_report.as_dict()
    assert op_names(variant) == op_names(auto)
    assert stablehlo.export(variant) == stablehlo.export(auto)
    assert variant.batch_report.switched == auto.batch_report.switched == ()
# --- C. Measured op / select counts of the branchy population
def test_branchy_population_op_and_select_counts():
    """`occurrence` 19 ops / 3 selects vs `variant` & `auto` 7 ops / 0 selects."""
    graphs = branchy_graphs()
    merged_graphs = {
        mode: merged(graphs, switch_ops=True, alignment=mode)
        for mode in ALIGNMENT_MODES
    }
    measured = {mode: counts(graph) for mode, graph in merged_graphs.items()}
    assert measured == BRANCHY_COUNTS
    # The point of the option: strictly fewer ops AND strictly fewer selects.
    for mode in ("variant", "auto"):
        for axis in (0, 1):
            assert measured[mode][axis] < measured["occurrence"][axis], mode
    # Occurrence alignment keeps both branch positions as SWITCHED groups (one
    # op per variant + a per-row select cascade per position) ...
    occurrence_report = merged_graphs["occurrence"].batch_report
    assert len(occurrence_report.switched) == 2
    first = occurrence_report.switched[0]
    assert isinstance(first, evoxc.SwitchRecord)
    assert first.group == "elementwise"
    assert first.variants == ("add", "subtract")
    # ... in GLOBAL function indices, one row per program.
    assert first.rows_per_variant == (
        tuple(range(0, 16, 2)),
        tuple(range(1, 16, 2)),
    )
    # ... while the aligned modes emit one class per variant, so nothing is left
    # to mux (see `test_variant_and_auto_produce_a_valid_graph`).
    assert merged_graphs["variant"].batch_report.switched == ()
    assert merged_graphs["auto"].batch_report.switched == ()
def test_unreduced_branchy_population_op_and_select_counts():
    """Without the shared `sum`: `occurrence` 17 ops / 3 selects vs 6 / 0."""
    graphs = branchy_graphs(reduce=False)
    measured = {
        mode: counts(merged(graphs, switch_ops=True, alignment=mode))
        for mode in ALIGNMENT_MODES
    }
    assert measured == UNREDUCED_BRANCHY_COUNTS
    assert measured == {"occurrence": (17, 3), "variant": (6, 0), "auto": (6, 0)}
# --- D. `"auto"` never emits more than either mode
@pytest.mark.parametrize("reduce", [True, False], ids=["reduced", "unreduced"])
def test_auto_emits_no_more_than_either_mode(reduce):
    """`auto <= min(occurrence, variant)` in ops and in selects.

    `"auto"` is `"variant"` plus a last-resort reuse of any other free class of
    the skeleton, so it can never create MORE classes than `"variant"` — and the
    docstring's promise is that it never creates more than `"occurrence"`.
    """
    graphs = branchy_graphs(reduce=reduce)
    measured = {
        mode: counts(merged(graphs, switch_ops=True, alignment=mode))
        for mode in ALIGNMENT_MODES
    }
    for axis, label in enumerate(("emitted_ops", "selects")):
        best_other = min(
            measured["occurrence"][axis], measured["variant"][axis]
        )
        assert measured["auto"][axis] <= best_other, label
    # ... and the same on the maximal-divergence population, where no reuse is
    # available to any mode (every program has its own variant).
    divergence = op_variant_graphs(evoxc.TensorSpec((3,), F32))
    divergent_counts = {
        mode: counts(merged(divergence, switch_ops=True, alignment=mode))
        for mode in ALIGNMENT_MODES
    }
    assert divergent_counts["auto"][0] <= min(
        divergent_counts["occurrence"][0], divergent_counts["variant"][0]
    )
# --- E. Correctness: `alignment` is a PURE heuristic, never a semantic choice
@pytest.mark.parametrize("alignment", list(ALIGNMENT_MODES))
def test_branchy_population_is_row_correct_in_every_mode(alignment):
    """Row i of every mode's merged run equals program i's own run.

    The merge itself guarantees bit-exactness (`alignment` changes class
    ASSIGNMENT only — a class's operands are shared by slot equality); the
    assertion below is the documented float tolerance (`rtol=1e-5,
    atol=1e-6`) because the population ends in a float reduction. The strict
    `np.array_equal` form is pinned by
    `test_alignment_never_changes_numbers`.
    """
    graphs = branchy_graphs()
    graph = merged(graphs, switch_ops=True, alignment=alignment)
    xs = branchy_inputs()
    reference = row_reference(
        graphs, [(xs[index],) for index in range(len(graphs))]
    )
    assert_rowwise(run_graph(graph, xs), reference)
    assert to_np_tree(run_graph(graph, xs)).shape == (len(graphs),)
#: Populations of `test_alignment_never_changes_numbers`: factories returning
#: `(graphs, rowwise inputs)`. Each one puts DIFFERENT op variants at one switch
#: position across programs.
NUMBER_INVARIANCE_POPULATIONS = [
    pytest.param(
        lambda: (branchy_graphs(), branchy_inputs(seed=0)),
        id="branchy-reduced",
    ),
    pytest.param(
        lambda: (branchy_graphs(reduce=False), branchy_inputs(seed=1)),
        id="branchy-unreduced",
    ),
    pytest.param(
        lambda: (
            op_variant_graphs(evoxc.TensorSpec((3,), F32)),
            (np.random.default_rng(2).normal(size=(4, 3)) + 5.0).astype(F32),
        ),
        id="op-variants",
    ),
]
@pytest.mark.parametrize("population", NUMBER_INVARIANCE_POPULATIONS)
def test_alignment_never_changes_numbers(population):
    """Every mode is `np.array_equal` to the per-program references.

    `alignment` may only change HOW MANY ops/selects are emitted — a class value
    at row p is that row's variant applied to that row's own operands, so the
    numbers are identical across all three modes (`np.array_equal`, the strict
    form: different class DAGs must not even reorder float accumulation here).
    """
    graphs, xs = population()
    reference = row_reference(
        graphs, [(xs[index],) for index in range(len(graphs))]
    )
    merged_graphs = {
        mode: merged(graphs, switch_ops=True, alignment=mode)
        for mode in ALIGNMENT_MODES
    }
    results = {}
    for mode, graph in merged_graphs.items():
        out = to_np_tree(run_graph(graph, xs))
        assert out.shape == reference.shape
        assert np.array_equal(out, reference), mode
        results[mode] = out
    # Cross-mode equality: the three modes agree with each other too.
    assert np.array_equal(results["occurrence"], results["variant"])
    assert np.array_equal(results["occurrence"], results["auto"])
    # Only the COST differs (strictly, on the branchy populations; the
    # maximal-divergence population has no reuse available to any mode).
    emitted = {
        mode: emitted_count(graph) for mode, graph in merged_graphs.items()
    }
    assert emitted["auto"] == emitted["variant"] <= emitted["occurrence"]
# --- F. Constants still stack BY OCCURRENCE under every alignment
@pytest.mark.parametrize("alignment", list(ALIGNMENT_MODES))
def test_constants_stack_by_occurrence_under_every_alignment(alignment):
    """N distinct constant payloads fold into ONE stacked constant.

    Constants are deliberately EXCLUDED from class alignment: a constant class
    stacks its members' payloads by occurrence, which is what folds N distinct
    (e.g. GP weight) payloads into one `(N, ...)` constant. Variant aligning
    them would explode one class into N constants and undo the batching.
    """
    graphs = constant_programs(6)
    graph = merged(graphs, switch_ops=True, alignment=alignment)
    report = graph.batch_report
    assert report.stacked_constants == 1
    assert op_names(graph).count("constant") == 1
    # Correctness of the stacked payload (one class, one op, six rows).
    xs = branchy_inputs(n_functions=6, seed=7)
    reference = row_reference(graphs, [(xs[index],) for index in range(6)])
    assert_rowwise(run_graph(graph, xs), reference)
# --- G. Validation, spelling normalization and inertness outside switch groups
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(1, id="int"),
        pytest.param(None, id="none"),
        pytest.param(3.5, id="float"),
        pytest.param(["occurrence"], id="list"),
        pytest.param(("variant",), id="tuple"),
        pytest.param(b"variant", id="bytes"),
    ],
)
def test_non_string_alignment_is_a_type_error(value):
    """A non-`str` `alignment` is a `TypeError` naming the accepted forms."""
    graphs = branchy_graphs(n_functions=2)
    with pytest.raises(TypeError) as excinfo:
        evoxc.batch_functions(graphs, switch_ops=True, alignment=value)
    message = str(excinfo.value)
    assert "alignment must be a string" in message
    for spelling in ALIGNMENT_MODES:
        assert spelling in message
    # A `TypeError`, never a `TransformError`/`EVOXCError`.
    assert not isinstance(excinfo.value, core.EVOXCError)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param("banana", id="unknown"),
        pytest.param("", id="empty"),
        pytest.param("occurence", id="typo"),
        pytest.param("variants", id="plural"),
    ],
)
def test_unknown_alignment_spelling_is_a_transform_error(value):
    """An unknown spelling is a `core.TransformError` naming the accepted forms."""
    graphs = branchy_graphs(n_functions=2)
    with pytest.raises(core.TransformError) as excinfo:
        evoxc.batch_functions(graphs, switch_ops=True, alignment=value)
    message = str(excinfo.value)
    assert "unknown alignment" in message
    assert repr(value) in message
    for spelling in ALIGNMENT_MODES:
        assert spelling in message
@pytest.mark.parametrize(
    "spelling, canonical",
    [
        pytest.param("  Occurrence  ", "occurrence", id="padded-occurrence"),
        pytest.param("VARIANT", "variant", id="upper-variant"),
        pytest.param("Auto", "auto", id="title-auto"),
    ],
)
def test_alignment_spelling_is_normalized(spelling, canonical):
    """Spellings are case-insensitive and whitespace-stripped (byte-identical)."""
    assert _merge.normalize_alignment(spelling) == canonical
    graphs = branchy_graphs()
    normalized = merged(graphs, switch_ops=True, alignment=spelling)
    reference = merged(graphs, switch_ops=True, alignment=canonical)
    assert stablehlo.export(normalized) == stablehlo.export(reference)
    assert op_names(normalized) == op_names(reference)
    assert counts(normalized) == BRANCHY_COUNTS[canonical]
def test_bad_alignment_is_rejected_before_any_tracing():
    """`alignment` is validated BEFORE tracing — the callable path included."""
    calls = []
    def f_add(x):
        calls.append("add")
        return evoxc.add(x, 1.0)
    def f_multiply(x):
        calls.append("multiply")
        return evoxc.multiply(x, 2.0)
    # No `TransformCallable` is produced and no program is traced.
    with pytest.raises(core.TransformError):
        evoxc.batch_functions([f_add, f_multiply], alignment="banana")
    assert calls == []
    with pytest.raises(TypeError):
        evoxc.batch_functions([f_add, f_multiply], alignment=1)
    assert calls == []
    # Positive control: the same callable list with a valid alignment builds a
    # `TransformCallable` — tracing still happens only when it is CALLED.
    transform = evoxc.batch_functions([f_add, f_multiply], alignment="variant")
    assert transform.kind == "batch_functions"
    assert calls == []
    graph = transform(evoxc.TensorSpec((2, 3), F32))
    assert calls == ["add", "multiply"]
    graphs = [
        evoxc.trace(f_add, evoxc.TensorSpec((3,), F32)),
        evoxc.trace(f_multiply, evoxc.TensorSpec((3,), F32)),
    ]
    xs = np.arange(6, dtype=np.float32).reshape(2, 3)
    reference = row_reference(graphs, [(xs[0],), (xs[1],)])
    assert_rowwise(run_graph(graph, xs), reference)
def test_alignment_is_inert_without_switch_ops():
    """Without `switch_ops` there is no switched position to align.

    Every skeleton then has ONE variant per program by construction, so the
    three alignments must emit byte-identical graphs (the option only chooses
    among classes of a `switch_ops` merge group).
    """
    graphs = branchy_graphs()
    texts = {}
    for mode in ALIGNMENT_MODES:
        graph = merged(graphs, alignment=mode)  # switch_ops defaults to None
        texts[mode] = stablehlo.export(graph)
        assert counts(graph) == (7, 0)
    assert len(set(texts.values())) == 1
    # The aligned labels differ from the SWITCHED occurrence DAG, proving the
    # inertness assertion above is not comparing an unrelated baseline.
    assert counts(merged(graphs, switch_ops=True, alignment="occurrence")) == (
        BRANCHY_COUNTS["occurrence"]
    )
