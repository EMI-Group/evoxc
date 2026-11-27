"""Error contract of the structural transform `evoxc.split_by_inputs`.

`split_by_inputs` slices one traced graph into a `prepare` / `apply` pair at
its input structure. Every invalid invocation and every graph the pass cannot
split must fail EXPLICITLY — `TypeError` for wrong argument kinds,
`core.TransformError` for semantics (selector range/duplicates, mask structure,
static-leaf entries, trivial splits, gated graphs), `core.TraceError` for
concrete tensors fed to the callable form — never a silent fallback and never
a half-built plan.

Each case pins the error TYPE plus a distinguishing substring of the message
(the offending index / pytree path / op name and the actionable hint), so
wording polish does not break the suite while a lost diagnostic does.
Authority: `evoxc/transforms/CONTEXT.md` ("split_by_inputs (structural pass,
binding)" + "v1 scope vs deferred"), `evoxc/transforms/split.py` (the mask /
`fixed_leaves` validation) and `evoxc/transforms/_split_core.py` (`gates` /
`_gate_op` / `split_graph`).
"""
import numpy as np
import pytest

import evoxc

SPEC = evoxc.TensorSpec((4, 3), evoxc.float32)
SMALL = evoxc.TensorSpec((4,), evoxc.float32)


# ---------------------------------------------------------------------------
# traceable functions (small shapes; nothing here executes)
# ---------------------------------------------------------------------------


def f_two(x, y):
    return x + y


def f_three(a, b, c):
    return a + b + c


def f_one(x):
    return x * 2.0


def f_static(x, scale):
    """`scale` is a Python value — a STATIC input leaf, never a boundary."""
    return x * scale


def f_static_two(x, y, scale):
    """Two tensor inputs beside a STATIC `scale` (tree leaf position 2)."""
    return (x + y) * scale


def f_dict(d):
    """One argument that is a dict of two tensors (root = a 1-child tuple)."""
    return d["a"] + d["b"]


def f_nested(x, d):
    """A tensor argument beside a nested dict argument."""
    return x + d["a"] + d["b"]


def f_cond(x, y):
    """Region-bearing `if` (the op `evoxc.cond` lowers to)."""
    return evoxc.sum(evoxc.cond(evoxc.sum(x) > 0, lambda: x * 2, lambda: -x)) + y


def f_while(x, y):
    """Region-bearing `while` (the op `evoxc.while_loop` lowers to)."""
    def body(s):
        return s + 1

    init = evoxc.sum(x) * 0
    return evoxc.while_loop(lambda s: s < 4, body, init) + y


def f_all_reduce(x, y):
    """An effectful `collective` op."""
    return evoxc.sum(evoxc.dist.all_reduce(x, op="sum")) + y


def f_custom_call(x, y):
    """An effectful `callback` op (the compiler escape hatch)."""
    call = evoxc.custom_call("tx_split_target", x, result=SPEC)
    return call + y


def g_two():
    return evoxc.trace(f_two, SPEC, SPEC)


def g_three():
    return evoxc.trace(f_three, SMALL, SMALL, SMALL)


def _msg(exc_info):
    return str(exc_info.value)


# ---------------------------------------------------------------------------
# 1. selector argument kinds and the both/neither rule
# ---------------------------------------------------------------------------


class TestSelectorArguments:
    """Exactly one of `fixed` / `fixed_leaves`, with the right kind."""

    def test_neither_selector(self):
        with pytest.raises(TypeError) as exc:
            evoxc.split_by_inputs(g_two())
        msg = _msg(exc)
        assert "neither" in msg
        assert "`fixed`" in msg
        assert "`fixed_leaves`" in msg

    def test_both_selectors(self):
        with pytest.raises(TypeError) as exc:
            evoxc.split_by_inputs(
                g_two(), fixed=(True, False), fixed_leaves=[0]
            )
        msg = _msg(exc)
        assert "both" in msg
        assert "`fixed`" in msg
        assert "`fixed_leaves`" in msg

    @pytest.mark.parametrize("bad", ["ab", b"ab"], ids=["str", "bytes"])
    def test_fixed_leaves_string_rejected(self, bad):
        with pytest.raises(TypeError) as exc:
            evoxc.split_by_inputs(g_two(), fixed_leaves=bad)
        msg = _msg(exc)
        assert "`fixed_leaves`" in msg
        assert type(bad).__name__ in msg  # names the offending kind

    def test_fixed_leaves_non_iterable_rejected(self):
        with pytest.raises(TypeError) as exc:
            evoxc.split_by_inputs(g_two(), fixed_leaves=3)
        msg = _msg(exc)
        assert "`fixed_leaves`" in msg
        assert "int" in msg

    @pytest.mark.parametrize(
        "entry", [0.5, True, np.int64(1)], ids=["float", "bool", "numpy_int"]
    )
    def test_fixed_leaves_non_plain_int_entry(self, entry):
        # `_is_int` accepts only true Python ints — a bool (an int subclass), a
        # float and a numpy integer are all refused as "not plain ints".
        with pytest.raises(TypeError) as exc:
            evoxc.split_by_inputs(g_two(), fixed_leaves=[entry])
        msg = _msg(exc)
        assert "plain ints" in msg
        assert repr(entry) in msg

    @pytest.mark.parametrize(
        "target", [f_two, evoxc.defn(f_two)], ids=["callable", "defn"]
    )
    def test_callable_path_validates_kinds_up_front(self, target):
        """The argument-KIND check runs before any tracing (the range and
        duplicate checks need the traced graph and run per call)."""
        with pytest.raises(TypeError) as exc:
            evoxc.split_by_inputs(target, fixed_leaves="x")
        msg = _msg(exc)
        assert "`fixed_leaves`" in msg
        assert "str" in msg

    def test_callable_path_defers_the_range_check_to_the_call(self):
        sc = evoxc.split_by_inputs(f_three, fixed_leaves=[7])  # no trace yet
        assert sc.kind == "split_by_inputs"
        with pytest.raises(evoxc.TransformError, match="out of range"):
            sc(SMALL, SMALL, SMALL)

    def test_callable_path_rejects_concrete_tensors(self):
        """Building a plan never executes: a concrete `Tensor` is a
        `TraceError`, never a silent eager split."""
        sc = evoxc.split_by_inputs(f_two, fixed_leaves=[0])
        concrete = evoxc.tensor(np.ones((4, 3), dtype=np.float32))
        with pytest.raises(evoxc.TraceError, match="never executes"):
            sc(concrete, concrete)


# ---------------------------------------------------------------------------
# 2. `fixed_leaves` range and duplicates (tensor-index axis)
# ---------------------------------------------------------------------------


class TestFixedLeavesRange:
    """Indices index `graph.tensor_specs`; out-of-range/duplicate fail."""

    @pytest.mark.parametrize("index", [7, -1], ids=["too_large", "negative"])
    def test_out_of_range_names_index_count_and_range(self, index):
        with pytest.raises(evoxc.TransformError) as exc:
            evoxc.split_by_inputs(g_three(), fixed_leaves=[index])
        msg = _msg(exc)
        assert f"index {index}" in msg
        assert "3 tensor inputs" in msg
        assert "0..2" in msg  # the valid range

    def test_duplicate_index_rejected(self):
        with pytest.raises(evoxc.TransformError) as exc:
            evoxc.split_by_inputs(g_two(), fixed_leaves=[1, 1])
        msg = _msg(exc)
        assert "duplicate" in msg
        assert "index 1" in msg


# ---------------------------------------------------------------------------
# 3. mask structure mismatches (pytree-path diagnostics)
# ---------------------------------------------------------------------------


class TestMaskStructureMismatch:
    """A `fixed` mask must mirror the input structure; a mismatch names the
    FIRST offending pytree path (`core.format_path`)."""

    def test_dict_mask_against_tuple_inputs(self):
        # Root input structure is the args tuple (x, y); the mask is a dict.
        with pytest.raises(evoxc.TransformError) as exc:
            evoxc.split_by_inputs(g_two(), fixed={"a": True, "b": False})
        msg = _msg(exc)
        assert "first mismatch at pytree path" in msg
        assert "does not match the graph's input structure" in msg

    def test_tuple_mask_against_dict_input(self):
        # The dict argument's root is a 1-child tuple; a 2-leaf tuple is not it.
        graph = evoxc.trace(f_dict, {"a": SMALL, "b": SMALL})
        with pytest.raises(evoxc.TransformError) as exc:
            evoxc.split_by_inputs(graph, fixed=(True, False))
        assert "first mismatch at pytree path" in _msg(exc)

    @pytest.mark.parametrize(
        "mask", [(True, False, True), (True,)], ids=["too_long", "too_short"]
    )
    def test_wrong_arity(self, mask):
        with pytest.raises(evoxc.TransformError) as exc:
            evoxc.split_by_inputs(g_two(), fixed=mask)
        assert "first mismatch at pytree path" in _msg(exc)

    def test_nested_mismatch_names_the_inner_path(self):
        graph = evoxc.trace(f_nested, SMALL, {"a": SMALL, "b": SMALL})
        with pytest.raises(evoxc.TransformError) as exc:
            evoxc.split_by_inputs(graph, fixed=(True, {"a": True, "c": False}))
        msg = _msg(exc)
        assert "first mismatch at pytree path [1]" in msg
        assert "same container structure" in msg


# ---------------------------------------------------------------------------
# 4. a mask entry on a STATIC leaf
# ---------------------------------------------------------------------------


class TestStaticLeafMaskEntry:
    """A static value specializes the graph and is never a boundary input, so
    a non-`None` mask entry there is a `TransformError` naming the path and
    pointing at `None`."""

    @pytest.mark.parametrize("entry", [True, False], ids=["true", "false"])
    def test_non_none_entry_at_static_leaf(self, entry):
        graph = evoxc.trace(f_static, SMALL, 3)  # `scale` -> static leaf [1]
        with pytest.raises(evoxc.TransformError) as exc:
            evoxc.split_by_inputs(graph, fixed=(True, entry))
        msg = _msg(exc)
        assert "STATIC" in msg
        assert "pytree path [1]" in msg
        assert "None" in msg

    def test_none_at_static_leaf_is_accepted(self):
        # The documented spelling: True/False on the tensor leaves, None on the
        # static leaf (the split itself stays legal).
        graph = evoxc.trace(f_static_two, SMALL, SMALL, 3)
        plan = evoxc.split_by_inputs(graph, fixed=(True, False, None))
        assert plan.fixed_leaves == (0,)
        assert plan.varying_leaves == (1,)


# ---------------------------------------------------------------------------
# 5. mask entry kind at a TENSOR leaf
# ---------------------------------------------------------------------------


class TestMaskEntryKind:
    """A tensor-leaf entry must be `True` / `False` / `None`.

    NOTE: the mask is flattened with `core.flatten` and only the STRUCTURE is
    compared against the input tree, so a leaf's VALUE type is never matched
    against the spec — a bare `1` / `"yes"` reaches the entry-kind check and
    raises `TransformError` (not a `TypeError`) naming the path and the value.
    """

    @pytest.mark.parametrize("entry", [1, "yes"], ids=["int", "str"])
    def test_non_bool_entry_at_tensor_leaf(self, entry):
        with pytest.raises(evoxc.TransformError) as exc:
            evoxc.split_by_inputs(g_two(), fixed=(entry, False))
        msg = _msg(exc)
        assert "must be True, False or None" in msg
        assert "pytree path [0]" in msg
        assert repr(entry) in msg
        assert "tensor input 0" in msg


# ---------------------------------------------------------------------------
# 6. trivial splits: both halves must be non-trivial
# ---------------------------------------------------------------------------


class TestTrivialSplits:
    """Fixing no tensor leaf and fixing ALL tensor leaves are both refused."""

    @pytest.mark.parametrize(
        "mask", [False, (False, False), (None, None)], ids=["bare", "explicit", "none"]
    )
    def test_nothing_fixed(self, mask):
        with pytest.raises(evoxc.TransformError) as exc:
            evoxc.split_by_inputs(g_two(), fixed=mask)
        msg = _msg(exc)
        assert "no tensor input is fixed" in msg
        assert "nothing to prepare" in msg

    @pytest.mark.parametrize("mask", [True, (True, True)], ids=["bare", "explicit"])
    def test_everything_fixed(self, mask):
        with pytest.raises(evoxc.TransformError) as exc:
            evoxc.split_by_inputs(g_two(), fixed=mask)
        msg = _msg(exc)
        assert "all 2 tensor inputs are fixed" in msg
        assert "nothing to split" in msg
        assert "varying" in msg

    def test_single_tensor_input_index_zero_is_all_fixed(self):
        graph = evoxc.trace(f_one, SMALL)
        with pytest.raises(evoxc.TransformError) as exc:
            evoxc.split_by_inputs(graph, fixed_leaves=[0])
        msg = _msg(exc)
        assert "all 1 tensor inputs are fixed" in msg
        assert "nothing to split" in msg


# ---------------------------------------------------------------------------
# 7. gated graphs: regions and effectful ops (never a silent fallback)
# ---------------------------------------------------------------------------


class TestGatedRegions:
    """Region-bearing control flow is not splittable in v1; the message names
    the IR op (`if` / `while`) and the nested-region reason."""

    def test_cond_region(self):
        graph = evoxc.trace(f_cond, SPEC, SPEC)
        with pytest.raises(evoxc.TransformError) as exc:
            evoxc.split_by_inputs(graph, fixed_leaves=[0])
        msg = _msg(exc)
        assert "op 'if'" in msg
        assert "region(s)" in msg
        assert "not yet splittable" in msg
        assert "blocked in v1" in msg

    def test_while_loop_region(self):
        graph = evoxc.trace(f_while, SPEC, SPEC)
        with pytest.raises(evoxc.TransformError) as exc:
            evoxc.split_by_inputs(graph, fixed_leaves=[0])
        msg = _msg(exc)
        assert "op 'while'" in msg
        assert "not yet splittable" in msg

    def test_the_same_functions_still_trace(self):
        # The restriction is SPLIT-only: tracing (and `verify`) is unaffected.
        for fn in (f_cond, f_while):
            graph = evoxc.trace(fn, SPEC, SPEC)
            assert graph.verify() is None


class TestGatedEffects:
    """Effectful ops are refused: "does not depend on the varying inputs" is
    not "safe to run once". The message names the op, its effect, and why."""

    @pytest.mark.parametrize(
        "fn,op,effect",
        [
            (f_all_reduce, "all_reduce", "collective"),
            (f_custom_call, "custom_call", "callback"),
        ],
        ids=["collective", "custom_call"],
    )
    def test_effectful_op_rejected(self, fn, op, effect):
        graph = evoxc.trace(fn, SPEC, SPEC)
        with pytest.raises(evoxc.TransformError) as exc:
            evoxc.split_by_inputs(graph, fixed_leaves=[0])
        msg = _msg(exc)
        assert f"op '{op}'" in msg
        assert f"effect '{effect}'" in msg
        assert "cannot be split" in msg
        # An actionable reason, not a bare re-raise: it names the varying
        # inputs and the only effects v1 accepts.
        assert "varying inputs" in msg
        assert "'pure'" in msg and "'read'" in msg


# ---------------------------------------------------------------------------
# 8/9. non-Graph inputs (callable form is exempt — that is not an error)
# ---------------------------------------------------------------------------


class TestNonGraphInput:
    """A non-Graph object needs a selector to get past the both/neither
    check; then the type mismatch is a `TypeError` with the trace hint."""

    @pytest.mark.parametrize(
        "bad,type_name",
        [
            (123, "int"),
            (np.zeros(3), "ndarray"),
            (None, "NoneType"),
        ],
        ids=["int", "ndarray", "none"],
    )
    def test_non_graph_object_rejected(self, bad, type_name):
        with pytest.raises(TypeError) as exc:
            evoxc.split_by_inputs(bad, fixed_leaves=[0])
        msg = _msg(exc)
        assert type_name in msg
        assert "evoxc.trace first" in msg

    def test_callable_form_is_not_rejected(self):
        # A plain callable / @evoxc.defn function IS a valid input form.
        sc = evoxc.split_by_inputs(f_two, fixed_leaves=[0])
        assert sc.kind == "split_by_inputs"
        plan = sc(SPEC, SPEC)
        assert plan.fixed_leaves == (0,)


class TestAnalyzeInputsNonGraph:
    """`analyze_inputs` is pure inspection of a TRACED graph only."""

    @pytest.mark.parametrize("bad", [123, None], ids=["int", "none"])
    def test_non_graph_rejected(self, bad):
        with pytest.raises(TypeError) as exc:
            evoxc.analyze_inputs(bad)
        msg = _msg(exc)
        assert "evoxc.trace" in msg
        assert type(bad).__name__ in msg
