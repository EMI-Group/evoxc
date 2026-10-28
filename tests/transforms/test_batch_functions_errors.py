"""Error-table tests for `evoxc.batch_functions` — every rejection is explicit.

Contract under test (binding): `evoxc/transforms/CONTEXT.md` "Batch functions"
and the `evoxc/transforms/batch_functions.py` docstring.
`batch_functions(fns, axes=0, out_axes=0, *, switch_ops=None)` merges N
same-signature programs onto one static leading function axis; there is never
a silent Python-loop fallback and never a silent axis drop. The rejections are
split across four error types:

* `TypeError` — malformed ARGUMENTS (not a non-empty sequence, mixed
  Graph/callable kinds, an unsupported element, malformed `switch_ops`);
* `core.TransformError` — v1 DEFERRALS (non-leading/out-of-range axes, a
  static input mapped to an axis, an op without a batching rule or with
  regions, `out_axes=None` on a mapped output, an axes/out_axes pytree that
  does not match the graph structure);
* `core.TraceError` — cross-function STRUCTURE violations (input/output tree
  shape, differing static leaves);
* `core.DTypeError` / `core.ShapeError` / `core.DeviceError` — per-leaf
  cross-function dtype/shape/device violations.

One test class per rejection family; messages are matched by a distinctive
fragment (the lead-in wording is owned by `batch_functions`/`vectorize`, so the
fragments below avoid the prefixes).
"""
import collections

import pytest

import evoxc
from evoxc import core

# The canonical signature every helper program in this module shares: one
# rank-1 float32 tensor input, one rank-1 float32 tensor output.
SPEC = core.TensorSpec((3,), core.float32)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _add_one(x):
    return evoxc.add(x, 1.0)


def _mul_two(x):
    return evoxc.multiply(x, 2.0)


def _add_graph(spec=SPEC):
    return evoxc.trace(_add_one, spec)


def _mul_graph(spec=SPEC):
    return evoxc.trace(_mul_two, spec)


def _expect(exc_type, fragment, fns, **kwargs):
    """Assert `batch_functions(fns, **kwargs)` raises exactly `exc_type`.

    Both halves matter: the TYPE pins which error family a rejection belongs
    to (argument `TypeError` vs v1 `TransformError` vs contract `TraceError`/
    dtype/shape/device errors), and the message fragment pins WHY.
    """
    with pytest.raises(exc_type) as excinfo:
        evoxc.batch_functions(fns, **kwargs)
    assert type(excinfo.value) is exc_type, (
        f"expected exactly {exc_type.__name__}, got "
        f"{type(excinfo.value).__name__}: {excinfo.value}"
    )
    assert fragment in str(excinfo.value), (
        f"expected the message to contain {fragment!r}, got: {excinfo.value}"
    )
    return excinfo.value


def _while_graph(spec=SPEC):
    """A region-bearing `while_loop` program (one carried vector)."""

    def fn(x):
        def cond(carried):
            keep = evoxc.sum(
                evoxc.cast(evoxc.less(carried[0], 3.0), core.int32), axes=(0,)
            )
            return keep > 0

        def body(carried):
            return (evoxc.add(carried[0], 1.0),)

        return evoxc.while_loop(cond, body, (x,))[0]

    return evoxc.trace(fn, spec)


def _cond_graph(spec=SPEC):
    """A region-bearing `cond` program."""

    def fn(x):
        pred = evoxc.sum(
            evoxc.cast(evoxc.less(x, 0.0), core.int32), axes=(0,)
        ) > 0
        return evoxc.cond(pred, lambda: evoxc.negate(x), lambda: x)

    return evoxc.trace(fn, spec)


def _random_graph():
    """A `random_uniform` program — random ops have no batching rule."""

    key_spec = core.TensorSpec((), core.int64)
    return evoxc.trace(lambda k: evoxc.random.uniform(k, (4,)), key_spec)


# ---------------------------------------------------------------------------
# 1-3. `fns` must be a non-empty sequence (TypeError)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fns, fragment",
    [
        (
            [],
            "fns must be a non-empty sequence of traced Graphs or of "
            "callables — got an empty sequence",
        ),
        (
            (),
            "fns must be a non-empty sequence of traced Graphs or of "
            "callables — got an empty sequence",
        ),
        (
            42,
            "fns must be a non-empty sequence (list/tuple) of traced Graphs "
            "or of callables/@evoxc.defn functions, got int",
        ),
        (
            "ab",
            "fns must be a non-empty sequence (list/tuple) of traced Graphs "
            "or of callables/@evoxc.defn functions, got str",
        ),
        (
            b"ab",
            "fns must be a non-empty sequence (list/tuple) of traced Graphs "
            "or of callables/@evoxc.defn functions, got bytes",
        ),
    ],
)
def test_fns_must_be_a_non_empty_sequence(fns, fragment):
    """Empty / non-sequence `fns` → TypeError naming the offending type."""
    _expect(TypeError, fragment, fns)


# ---------------------------------------------------------------------------
# 4-5. Graph/callable kinds are never mixed; elements must be programs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fns, fragment",
    [
        (
            # The MIX guard reports the first non-Graph element, whatever its
            # type — even an int that is not a callable at all.
            [_add_graph(), _add_one],
            "must not mix traced Graphs with callables — element 0 is a "
            "Graph while element 1 is a function",
        ),
        (
            [_add_graph(), 42],
            "must not mix traced Graphs with callables — element 0 is a "
            "Graph while element 1 is a int",
        ),
    ],
)
def test_graphs_and_callables_must_not_be_mixed(fns, fragment):
    """A Graph/callable mix is unrepresentable — TypeError naming both kinds."""
    _expect(TypeError, fragment, fns)


def test_callable_list_rejects_a_non_callable_element():
    """On the pure-callable path each element must be callable/`@defn`."""
    _expect(
        TypeError,
        "fns[1] must be a traced Graph or a callable/@evoxc.defn function, "
        "got int",
        [lambda x: x, 42],
    )


# ---------------------------------------------------------------------------
# 6. `axes` / `out_axes` must be leading (0) or unmapped (None)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entry, fragment",
    [
        (1, "mapped axis 1 is out of range for tensor leaf 0 of rank 1"),
        (2, "mapped axis 2 is out of range for tensor leaf 0 of rank 1"),
    ],
)
def test_axes_entry_out_of_range(entry, fragment):
    """A mapped axis outside the leaf's rank → TransformError (range check)."""
    _expect(
        core.TransformError,
        fragment,
        [_add_graph(), _mul_graph()],
        axes=entry,
    )


def test_axes_non_leading_axis_is_deferred():
    """A rank-2 leaf with `axes=1` hits the v1 non-leading-axes deferral."""
    spec = core.TensorSpec((3, 4), core.float32)
    graphs = [
        evoxc.trace(_add_one, spec),
        evoxc.trace(_mul_two, spec),
    ]
    _expect(
        core.TransformError,
        "mapped axis 1 for tensor leaf 0 must be 0 (the leading axis) in v1 "
        "— non-leading axes are deferred",
        graphs,
        axes=1,
    )


def test_out_axes_non_leading_axis_is_deferred():
    """`out_axes` mirrors the input rule: only 0 (or None) is supported."""
    _expect(
        core.TransformError,
        "mapped axis 1 for tensor leaf 0 must be 0 (the leading axis) in v1 "
        "— non-leading axes are deferred",
        [_add_graph(), _mul_graph()],
        out_axes=1,
    )


# ---------------------------------------------------------------------------
# 7. Static inputs must be unmapped
# ---------------------------------------------------------------------------


def test_axes_must_be_none_for_a_static_input():
    """A recorded static input leaf mapped to an axis → TransformError."""
    spec = core.TensorSpec((3,), core.float32)
    graphs = [
        evoxc.trace(lambda x, s: evoxc.add(x, s), spec, 2.0),
        evoxc.trace(lambda x, s: evoxc.multiply(x, s), spec, 2.0),
    ]
    _expect(
        core.TransformError,
        "static leaf 1 must map to None, got 0",
        graphs,
        axes=(0, 0),
    )
    # Positive control: the same pair merges once the static slot is unmapped.
    merged = evoxc.batch_functions(graphs, axes=(0, None))
    assert isinstance(merged, evoxc.Graph)
    assert merged.batch_report.n_functions == 2


# ---------------------------------------------------------------------------
# 8. Cross-function INPUT contract (dtype / shape / device / tree / static)
# ---------------------------------------------------------------------------


def test_input_dtype_mismatch_is_a_dtype_error():
    """Every function must take the same per-leaf dtype as function 0."""
    float64_spec = core.TensorSpec((3,), core.float64)
    _expect(
        core.DTypeError,
        "input leaf 0 at pytree path [0] has dtype float64 but function 0 "
        "has float32",
        [_add_graph(), _add_graph(float64_spec)],
    )


def test_input_shape_mismatch_is_a_shape_error():
    """Every function must take the same per-leaf shape as function 0."""
    wider_spec = core.TensorSpec((4,), core.float32)
    _expect(
        core.ShapeError,
        "input leaf 0 at pytree path [0] has shape (4,) but function 0 has "
        "(3,)",
        [_add_graph(), _add_graph(wider_spec)],
    )


def test_input_device_mismatch_is_a_device_error():
    """Every function must take the same per-leaf device as function 0."""
    cuda_spec = core.TensorSpec(
        (3,), core.float32, device=evoxc.Device("cuda", 0)
    )
    _expect(
        core.DeviceError,
        "input leaf 0 at pytree path [0] lives on device "
        "Device(kind='cuda', index=0)",
        [_add_graph(), _add_graph(cuda_spec)],
    )


def test_input_tree_structure_mismatch_is_a_trace_error():
    """Different input pytrees → TraceError with the first mismatch path."""
    two_input_graph = evoxc.trace(lambda x, y: evoxc.add(x, y), SPEC, SPEC)
    _expect(
        core.TraceError,
        "the input tree structure differs from function 0 — first mismatch "
        "at pytree path",
        [_add_graph(), two_input_graph],
    )


def test_static_input_leaf_mismatch_is_a_trace_error():
    """Static inputs specialize the graph — they must match exactly."""
    spec = core.TensorSpec((3,), core.float32)
    two = evoxc.trace(lambda x, s: evoxc.add(x, s), spec, 2.0)
    three = evoxc.trace(lambda x, s: evoxc.add(x, s), spec, 3.0)
    _expect(
        core.TraceError,
        "static input leaf at pytree path [1] differs from function 0",
        [two, three],
        axes=(0, None),
    )


# ---------------------------------------------------------------------------
# 9. Cross-function OUTPUT contract (dtype / shape / static / tree)
# ---------------------------------------------------------------------------


def test_output_dtype_mismatch_is_a_dtype_error():
    """Every function must RETURN the same per-leaf dtype as function 0."""
    casted = evoxc.trace(lambda x: evoxc.cast(x, core.float64), SPEC)
    _expect(
        core.DTypeError,
        "output leaf 0 at pytree path () has dtype float64",
        [_add_graph(), casted],
    )


def test_output_shape_mismatch_is_a_shape_error():
    """Every function must RETURN the same per-leaf shape as function 0."""
    reduced = evoxc.trace(lambda x: evoxc.sum(x, axes=(0,)), SPEC)
    _expect(
        core.ShapeError,
        "output leaf 0 at pytree path () has shape ()",
        [_add_graph(), reduced],
    )


def test_static_output_leaf_mismatch_is_a_trace_error():
    """Recorded static OUTPUT leaves must match exactly across functions."""
    seven = evoxc.trace(lambda x: (evoxc.add(x, 1.0), 7), SPEC)
    eight = evoxc.trace(lambda x: (evoxc.multiply(x, 2.0), 8), SPEC)
    _expect(
        core.TraceError,
        "static output leaf at pytree path [1] differs from function 0",
        [seven, eight],
    )


def test_output_tree_structure_mismatch_is_a_trace_error():
    """Different output pytrees → TraceError with the first mismatch path."""
    pair_out = evoxc.trace(
        lambda x: (evoxc.add(x, 1.0), evoxc.add(x, 2.0)), SPEC
    )
    _expect(
        core.TraceError,
        "the output tree structure differs from function 0 — first mismatch "
        "at pytree path",
        [_add_graph(), pair_out],
    )


# ---------------------------------------------------------------------------
# 10. Ops without a batching rule / region-bearing control flow (v1 deferral)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build_graph, axes, fragment",
    [
        # `sort` has no batching rule in v1 — with a mapped rank-1 input and
        # with every input unmapped.
        (
            lambda: evoxc.trace(lambda x: evoxc.sort(x), SPEC),
            0,
            "cannot batch op 'sort' from function 0",
        ),
        (
            lambda: evoxc.trace(lambda x: evoxc.sort(x), SPEC),
            None,
            "cannot batch op 'sort' from function 0",
        ),
        # Region-bearing control flow is never batchable.
        (
            lambda: _while_graph(),
            0,
            "cannot batch op 'while' from function 0: region-bearing "
            "control-flow ops (cond/while_loop/scan) are not supported",
        ),
        (
            lambda: _cond_graph(),
            0,
            "cannot batch op 'if' from function 0: region-bearing "
            "control-flow ops (cond/while_loop/scan) are not supported",
        ),
        # Random ops carry no batching rule either. The key is rank 0, so the
        # axes have to be all-None (axes=0 would fail the rank check first).
        (
            _random_graph,
            None,
            "cannot batch op 'random_uniform' from function 0",
        ),
    ],
)
def test_unbatchable_ops_are_explicit_deferrals(build_graph, axes, fragment):
    """Ruleless / region-bearing ops → TransformError naming op + function."""
    graph = build_graph()
    exc = _expect(core.TransformError, fragment, [graph, graph], axes=axes)
    # The message names the function index and is an explicit v1 deferral
    # (the ruleless-op wording additionally points at the rule registry).
    assert "function 0" in str(exc)
    assert "v1 deferral of batch_functions" in str(exc)


# ---------------------------------------------------------------------------
# 11. Sparse programs
# ---------------------------------------------------------------------------
# The sparse ops DO have batching rules registered (`evoxc.sparse.rules`), so
# the "no batching rule" sparse path is unreachable: a program that builds and
# densifies a sparse value merges normally (see the positive control below).


def test_sparse_ops_are_batchable_so_the_no_rule_path_is_unreachable():
    """A sparse build+dense program merges like any other op stream."""
    spec = core.TensorSpec((3, 3), core.float32)
    graph = evoxc.trace(
        lambda x: evoxc.sparse.to_dense(evoxc.sparse.from_dense(x)), spec
    )
    merged = evoxc.batch_functions([graph, graph])
    assert isinstance(merged, evoxc.Graph)
    assert merged.batch_report.n_functions == 2


def test_sparse_output_is_rejected_explicitly():
    """A graph RETURNING a sparse value is rejected — the merge never guesses.

    At this HEAD the rejection arrives from `_normalize_out_axes`: the output
    tree's root IS the registered `SparseTensor` node (6 leaves — 2 tensor
    leaves + dense_shape/dtype/format statics), and no plain `0`/`None`/
    int-pytree can match a registered node's `TreeSpec`, so the out_axes
    structure check fires before the registered-node guard below.
    """
    spec = core.TensorSpec((3, 3), core.float32)
    graph = evoxc.trace(lambda x: evoxc.sparse.from_dense(x), spec)
    assert graph.output_tree.num_leaves == 6
    _expect(
        core.TransformError,
        "the out_axes pytree does not match the output structure",
        [graph, graph],
        out_axes=(0,) * graph.output_tree.num_leaves,
    )


# BUG(evoxc): `_check_output_nodes` — the documented registered-output-node
# v1 deferral ("returns a registered pytree node with static leaves ... batched
# output-aux remapping is a v1 deferral of batch_functions") is UNREACHABLE
# through the public API. `_normalize_out_axes` runs first and no axes pytree
# can structurally match a registered node (`first_mismatch_path` requires the
# node type/node_data to match), so the user-visible error is the out_axes
# structure mismatch instead of the guard message.
# Minimal repro (no matching out_axes exists — `(0,)*6`, `(None,)*6`, a bare
# `0` (2 tensor outputs → bare-int error) and a dict/namedtuple wrapper all
# fail earlier):
#   g = evoxc.trace(lambda x: evoxc.sparse.from_dense(x), TensorSpec((3,3), float32))
#   evoxc.batch_functions([g, g], out_axes=(None,) * 6)
#   -> TransformError: "the out_axes pytree does not match the output structure"
# Never the documented guard message. The merge IS still rejected explicitly
# (a TransformError), so this is a message/reachability gap, not a silent
# merge — the assertions below pin the DOCUMENTED contract.
@pytest.mark.xfail(
    strict=False,
    reason=(
        "BUG(evoxc): the registered-output-node guard (_check_output_nodes) "
        "is shadowed by out_axes normalization, so the documented message is "
        "never raised"
    ),
)
def test_sparse_output_reaches_the_registered_node_guard():
    """Documented v1 deferral: registered output node with static leaves."""
    spec = core.TensorSpec((3, 3), core.float32)
    graph = evoxc.trace(lambda x: evoxc.sparse.from_dense(x), spec)
    # Reason for the axes argument's shape: probe the SINGLE traced graph's
    # output leaf count (6) and pass an all-None pytree of that width — the
    # shape a registered-node-aware normalization would accept.
    _expect(
        core.TransformError,
        "returns a registered pytree node with static leaves",
        [graph, graph],
        out_axes=(None,) * graph.output_tree.num_leaves,
    )


# ---------------------------------------------------------------------------
# 12. `switch_ops` validation (TypeError for malformed, TransformError for
#     unknown categories/op names/empty groups)
# ---------------------------------------------------------------------------

_SWITCH_CATEGORIES = (
    "['collective', 'comparison', 'control', 'elementwise', 'linalg', "
    "'random', 'reduction', 'sorting', 'sparse', 'structure', 'terminator']"
)


@pytest.mark.parametrize(
    "switch_ops, exc_type, fragment",
    [
        (
            ["nope"],
            core.TransformError,
            "unknown switch_ops category 'nope'; valid categories are "
            + _SWITCH_CATEGORIES,
        ),
        (
            [{"nope"}],
            core.TransformError,
            "unknown op name 'nope' in a switch_ops merge group",
        ),
        (
            [[]],
            core.TransformError,
            "empty switch_ops merge group",
        ),
        (
            "elementwise",
            TypeError,
            "switch_ops must be None, False, True, or an iterable of merge "
            "groups (category names / iterables of op names); got str",
        ),
        (
            3,
            TypeError,
            "switch_ops must be None, False, True, or an iterable of merge "
            "groups (category names / iterables of op names); got int",
        ),
        (
            [b"add"],
            TypeError,
            "switch_ops entries must be a category name (str) or an iterable "
            "of op names, got bytes",
        ),
        (
            [[1]],
            TypeError,
            "switch_ops op-name entries must be strings, got 1",
        ),
    ],
)
def test_switch_ops_validation(switch_ops, exc_type, fragment):
    """Malformed `switch_ops` → TypeError; unknown names → TransformError."""
    _expect(
        exc_type,
        fragment,
        [_add_graph(), _mul_graph()],
        switch_ops=switch_ops,
    )


# ---------------------------------------------------------------------------
# 13. `out_axes=None` never silently drops a batch axis
# ---------------------------------------------------------------------------


def test_out_axes_none_on_a_mapped_output_is_rejected():
    """Identical programs: the (mapped) shared output cannot be un-batched."""
    _expect(
        core.TransformError,
        "out_axes=None at output leaf 0 requires the output to be unmapped, "
        "but it carries a mapped function axis — a batch axis is never "
        "silently dropped",
        [_add_graph(), _add_graph()],
        out_axes=None,
    )


def test_out_axes_none_on_a_muxed_output_is_rejected():
    """A muxed output is ALWAYS mapped (row i = function i's value)."""
    _expect(
        core.TransformError,
        "out_axes=None at output leaf 0 requires the output to be unmapped, "
        "but the output is muxed from 2 different function results",
        [_add_graph(), _mul_graph()],
        out_axes=None,
    )


def test_bare_out_axes_needs_at_most_one_tensor_output():
    """A bare int/None `out_axes` cannot cover two tensor outputs."""
    pair_out = evoxc.trace(
        lambda x: (evoxc.add(x, 1.0), evoxc.add(x, 2.0)), SPEC
    )
    _expect(
        core.TransformError,
        "a bare int/None out_axes applies only when the graph has at most "
        "ONE tensor output; got 2 tensor outputs",
        [pair_out, pair_out],
        out_axes=0,
    )


# ---------------------------------------------------------------------------
# 14. `axes` pytree must match the graph's (argument-wrapped) input structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build, axes",
    [
        # The traced input tree wraps EACH positional argument in the root
        # tuple, so a bare dict/namedtuple (without the root wrapper) and a
        # positional int tuple both diverge.
        (
            lambda: evoxc.trace(
                lambda d: evoxc.add(d["a"], d["b"]),
                {"a": SPEC, "b": SPEC},
            ),
            {"a": 0, "b": 0},
        ),
        (
            lambda: evoxc.trace(
                lambda d: evoxc.add(d["a"], d["b"]),
                {"a": SPEC, "b": SPEC},
            ),
            (0, 0),
        ),
        (
            lambda: evoxc.trace(
                lambda p: evoxc.add(p.a, p.b),
                collections.namedtuple("Pair", ["a", "b"])(SPEC, SPEC),
            ),
            (0, 0),
        ),
    ],
)
def test_axes_pytree_must_match_the_input_structure(build, axes):
    """A structurally wrong axes pytree → TransformError with the path."""
    _expect(
        core.TransformError,
        "the axes pytree does not match the graph's input structure — first "
        "mismatch at pytree path",
        [build(), build()],
        axes=axes,
    )


def test_wrapped_axes_pytree_is_accepted():
    """Positive control: the root-wrapped axes pytree merges (the guard is
    not vacuous)."""
    graph = evoxc.trace(
        lambda d: evoxc.add(d["a"], d["b"]), {"a": SPEC, "b": SPEC}
    )
    merged = evoxc.batch_functions(
        [graph, graph], axes=({"a": 0, "b": None},)
    )
    assert isinstance(merged, evoxc.Graph)
    assert merged.batch_report.n_functions == 2
