"""Inspection-facts contract of ``evoxc.analyze_inputs``.

Pins the binding statements in ``evoxc/transforms/CONTEXT.md`` ("split_by_inputs
(structural pass, binding)" — the ``InputAnalysis`` / ``InputInfo`` bullets):
`analyze_inputs` is pure inspection over a traced graph that reports, per tensor
input, the direct consumers, the transitively reachable op count, the influenced
tensor outputs and what fixing that input ALONE would cost as a boundary.

Every expected number below is derived BY HAND from the traced IR (the op list is
printed in the comment above each assertion) — never from the implementation under
test. The graph used for the hand-known facts is::

    def f(x, y):
        return (evoxc.sum(evoxc.tanh(x)),
                evoxc.sum(evoxc.tanh(y) + evoxc.sin(x)))

whose entry block (non-terminator ops, in order) is::

    0 tanh(x)          depends on x
    1 reduce_sum(0)    depends on x        -> out0
    2 tanh(y)          depends on y
    3 sin(x)           depends on x
    4 add(2, 3)        depends on x, y
    5 reduce_sum(4)    depends on x, y     -> out1
    (terminator) return(1, 5)
"""

from __future__ import annotations

import json
import re

import pytest

import evoxc
from evoxc import core
from evoxc.transforms import analyze_inputs

# (3, 4) float32 -> 3 * 4 * 4 bytes.
NBYTES_3x4_f32 = 48


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _two_output_graph():
    """The hand-known graph: two inputs, two tensor outputs."""

    def f(x, y):
        return (evoxc.sum(evoxc.tanh(x)), evoxc.sum(evoxc.tanh(y) + evoxc.sin(x)))

    return evoxc.trace(
        f,
        evoxc.TensorSpec((3, 4), evoxc.float32),
        evoxc.TensorSpec((3, 4), evoxc.float32),
    )


def _cond_graph():
    def f(x, y):
        pred = evoxc.sum(x) > 0.0
        return evoxc.cond(
            pred,
            lambda: evoxc.sum(evoxc.tanh(x)),
            lambda: evoxc.sum(evoxc.abs(y)),
        )

    return evoxc.trace(
        f,
        evoxc.TensorSpec((3, 4), evoxc.float32),
        evoxc.TensorSpec((3, 4), evoxc.float32),
    )


def _while_graph():
    def f(x, y):
        def cond_fn(state):
            return state < 10.0

        def body_fn(state):
            return state * 0.5

        return evoxc.while_loop(cond_fn, body_fn, evoxc.sum(x)), evoxc.sum(y)

    return evoxc.trace(
        f,
        evoxc.TensorSpec((3, 4), evoxc.float32),
        evoxc.TensorSpec((3, 4), evoxc.float32),
    )


def _all_reduce_graph():
    def f(x, y):
        return evoxc.sum(evoxc.dist.all_reduce(x)), evoxc.sum(y)

    return evoxc.trace(
        f,
        evoxc.TensorSpec((3, 4), evoxc.float32),
        evoxc.TensorSpec((3, 4), evoxc.float32),
    )


def _entry_ops(graph):
    block = graph.module.functions[0].region.blocks[0]
    return block, [op for op in block.ops if not op.is_terminator]


# ---------------------------------------------------------------------------
# 1. hand-known facts
# ---------------------------------------------------------------------------


class TestHandKnownFacts:
    def test_entry_block_op_list_justifies_the_counts(self):
        graph = _two_output_graph()
        _block, ops = _entry_ops(graph)
        assert [op.name for op in ops] == [
            "tanh",
            "reduce_sum",
            "tanh",
            "sin",
            "add",
            "reduce_sum",
        ]

    def test_graph_level_counts(self):
        graph = _two_output_graph()
        analysis = analyze_inputs(graph)
        _block, ops = _entry_ops(graph)
        # 6 non-terminator entry-block ops, 2 tensor output leaves.
        assert analysis.op_count == len(ops) == 6
        assert analysis.output_count == 2
        assert analysis.splittable is True
        assert analysis.blocked_reason is None
        assert analysis.source is graph

    def test_input_x_facts(self):
        graph = _two_output_graph()
        analysis = analyze_inputs(graph)
        (x, _y) = analysis.inputs
        assert x.index == 0
        assert x.path == "[0]"
        assert x.spec is graph.tensor_specs[0]
        assert x.nbytes == NBYTES_3x4_f32  # 3 * 4 elements * 4 bytes
        # consumers tanh(0) and sin(3) -> 2 distinct ops
        assert x.direct_consumers == 2
        # ops whose dependence mask carries x's bit: tanh, reduce_sum(0),
        # sin, add, reduce_sum(1) -> 5
        assert x.op_count == 5
        # both return operands depend on x
        assert x.outputs == (0, 1)
        assert isinstance(x.outputs, tuple)
        # fixing x alone crosses reduce_sum(0)'s scalar (4 B) and sin(x) (48 B)
        assert x.boundary_bytes == 4 + NBYTES_3x4_f32 == 52
        assert x.boundary_count == 2

    def test_input_y_facts(self):
        graph = _two_output_graph()
        analysis = analyze_inputs(graph)
        (_x, y) = analysis.inputs
        assert y.index == 1
        assert y.path == "[1]"
        assert y.spec is graph.tensor_specs[1]
        assert y.nbytes == NBYTES_3x4_f32
        # only tanh(y) consumes y directly
        assert y.direct_consumers == 1
        # tanh(y), add, reduce_sum(1) -> 3
        assert y.op_count == 3
        # only out1 depends on y
        assert y.outputs == (1,)
        # fixing y alone crosses tanh(y)'s result (48 B) into the mixed `add`
        assert y.boundary_bytes == NBYTES_3x4_f32 == 48
        assert y.boundary_count == 1

    def test_inputs_are_in_tensor_spec_order(self):
        graph = _two_output_graph()
        analysis = analyze_inputs(graph)
        assert [info.index for info in analysis.inputs] == [0, 1]
        assert [info.spec for info in analysis.inputs] == list(graph.tensor_specs)

    def test_path_counts_static_leaves_but_index_does_not(self):
        def f(x, k, y):
            return evoxc.sum(evoxc.tanh(x) * k) + evoxc.sum(y)

        graph = evoxc.trace(
            f,
            evoxc.TensorSpec((3, 4), evoxc.float32),
            2,  # a static int at tree-leaf position 1
            evoxc.TensorSpec((5,), evoxc.float32),
        )
        analysis = analyze_inputs(graph)
        assert len(graph.static_values) == 1
        first, second = analysis.inputs
        # tensor-input indices are 0 and 1 ...
        assert first.index == 0
        assert second.index == 1
        # ... but the rendered paths count the static leaf, so y sits at [2]
        assert first.path == "[0]"
        assert second.path == "[2]"
        assert first.spec is graph.tensor_specs[0]
        assert second.spec is graph.tensor_specs[1]
        assert second.nbytes == 5 * 4


# ---------------------------------------------------------------------------
# 2. InputAnalysis.split is documented sugar over split_by_inputs
# ---------------------------------------------------------------------------


class TestSplitSugar:
    @pytest.mark.parametrize(
        "kwargs",
        [{"fixed": (True, False)}, {"fixed_leaves": [0]}],
        ids=["mask", "fixed_leaves"],
    )
    def test_split_matches_split_by_inputs(self, kwargs):
        graph = _two_output_graph()
        analysis = analyze_inputs(graph)

        sugar = analysis.split(**kwargs)
        direct = evoxc.split_by_inputs(graph, **kwargs)

        assert sugar.prepare.digest() == direct.prepare.digest()
        assert sugar.apply.digest() == direct.apply.digest()
        assert sugar.fixed_leaves == direct.fixed_leaves
        assert sugar.varying_leaves == direct.varying_leaves
        assert [value.summary() for value in sugar.boundary] == [
            value.summary() for value in direct.boundary
        ]

    def test_split_of_the_hand_known_boundary(self):
        graph = _two_output_graph()
        analysis = analyze_inputs(graph)
        plan = analysis.split(fixed_leaves=[0])
        # fixing x alone: reduce_sum(0)'s scalar then sin(x), per the report
        assert len(plan.boundary) == analysis.inputs[0].boundary_count == 2
        assert plan.boundary_bytes == analysis.inputs[0].boundary_bytes == 52
        assert sorted(value.defining_op for value in plan.boundary) == [
            "reduce_sum",
            "sin",
        ]


# ---------------------------------------------------------------------------
# 3. blocked graphs
# ---------------------------------------------------------------------------

BLOCKED_BUILDERS = [
    pytest.param(_cond_graph, "if", id="cond"),
    pytest.param(_while_graph, "while", id="while_loop"),
    pytest.param(_all_reduce_graph, "all_reduce", id="all_reduce"),
]


class TestBlockedGraphs:
    @pytest.mark.parametrize("build, op_name", BLOCKED_BUILDERS)
    def test_blocked_graph_reports_reason_and_none_boundaries(self, build, op_name):
        analysis = analyze_inputs(build())
        assert analysis.splittable is False
        assert isinstance(analysis.blocked_reason, str) and analysis.blocked_reason
        assert f"'{op_name}'" in analysis.blocked_reason
        for info in analysis.inputs:
            # the gate blocks the SPLIT, not the per-op counting
            assert isinstance(info.direct_consumers, int)
            assert isinstance(info.op_count, int)
            assert isinstance(info.outputs, tuple)
            assert all(isinstance(out, int) for out in info.outputs)
            assert info.boundary_bytes is None
            assert info.boundary_count is None

    @pytest.mark.parametrize("build, op_name", BLOCKED_BUILDERS)
    def test_blocked_split_raises_the_same_message(self, build, op_name):
        analysis = analyze_inputs(build())
        with pytest.raises(core.TransformError) as excinfo:
            analysis.split(fixed_leaves=[0])
        assert str(excinfo.value) == analysis.blocked_reason

    def test_single_tensor_input_is_splittable_but_alone_is_not_legal(self):
        def f(x):
            return evoxc.sum(evoxc.tanh(x))

        graph = evoxc.trace(f, evoxc.TensorSpec((3, 4), evoxc.float32))
        analysis = analyze_inputs(graph)
        assert analysis.splittable is True
        assert analysis.blocked_reason is None
        (info,) = analysis.inputs
        # per-op facts are still reported ...
        assert info.direct_consumers == 1
        assert info.op_count == 2  # tanh + reduce_sum
        assert info.outputs == (0,)
        # ... but "fixing the only input" means "all fixed", not a legal split
        assert info.boundary_bytes is None
        assert info.boundary_count is None

    def test_fixing_a_dead_input_crosses_nothing(self):
        def f(x, y):
            evoxc.tanh(x)  # dead: its result is never returned
            return evoxc.sum(y)

        graph = evoxc.trace(
            f,
            evoxc.TensorSpec((3, 4), evoxc.float32),
            evoxc.TensorSpec((3, 4), evoxc.float32),
        )
        analysis = analyze_inputs(graph)
        assert analysis.splittable is True
        dead, used = analysis.inputs

        # x feeds only the dead tanh: fixable alone and it crosses nothing
        assert dead.direct_consumers == 1
        assert dead.op_count == 1  # just the dead tanh
        assert dead.outputs == ()
        assert dead.boundary_bytes == 0
        assert dead.boundary_count == 0

        # y feeds the returned reduce_sum: a scalar boundary
        assert used.boundary_count == 1
        assert used.boundary_bytes == 4

        plan = analysis.split(fixed_leaves=[0])
        assert len(plan.boundary) == 0


# ---------------------------------------------------------------------------
# 4. inspection conventions
# ---------------------------------------------------------------------------

# Exact key sets, asserted by equality (never a superset check).
SUMMARY_KEYS = {
    "input_count",
    "output_count",
    "op_count",
    "splittable",
    "blocked_reason",
}
AS_DICT_KEYS = SUMMARY_KEYS | {"inputs"}
INFO_SUMMARY_KEYS = {
    "index",
    "path",
    "shape",
    "dtype",
    "nbytes",
    "direct_consumers",
    "op_count",
    "outputs",
    "boundary_bytes",
    "boundary_count",
}
INFO_AS_DICT_KEYS = INFO_SUMMARY_KEYS

_RE_ANALYSIS_REPR = re.compile(
    r"<InputAnalysis \d+ inputs, \d+ ops, \d+ outputs, (splittable|blocked)>"
)
_RE_INFO_REPR = re.compile(
    r"<InputInfo \d+ at .+? shape=\(.*?\) dtype=\S+ .+?: "
    r"consumers=\d+ ops=\d+ outputs=\[[\d, ]*\] boundary=.+>"
)


class TestInspectionConventions:
    def test_summary_key_set(self):
        analysis = analyze_inputs(_two_output_graph())
        assert set(analysis.summary()) == SUMMARY_KEYS

    def test_as_dict_key_set(self):
        analysis = analyze_inputs(_two_output_graph())
        assert set(analysis.as_dict()) == AS_DICT_KEYS

    def test_info_summary_key_set(self):
        analysis = analyze_inputs(_two_output_graph())
        for info in analysis.inputs:
            assert set(info.summary()) == INFO_SUMMARY_KEYS

    def test_info_as_dict_key_set(self):
        analysis = analyze_inputs(_two_output_graph())
        for info in analysis.inputs:
            assert set(info.as_dict()) == INFO_AS_DICT_KEYS

    def test_summary_is_json_safe(self):
        analysis = analyze_inputs(_two_output_graph())
        summary = analysis.summary()
        assert isinstance(summary["splittable"], bool)
        assert isinstance(summary["op_count"], int)
        assert summary["blocked_reason"] is None
        assert json.loads(json.dumps(summary)) == summary

    def test_as_dict_is_json_safe(self):
        analysis = analyze_inputs(_two_output_graph())
        data = analysis.as_dict()
        # the per-input ``outputs`` are LISTS here (not tuples)
        for entry in data["inputs"]:
            assert isinstance(entry["outputs"], list)
        assert json.loads(json.dumps(data)) == data

    def test_info_fields_json_safe(self):
        analysis = analyze_inputs(_two_output_graph())
        for info in analysis.inputs:
            entry = info.as_dict()
            assert isinstance(entry["boundary_bytes"], int)
            assert isinstance(entry["boundary_count"], int)
            assert isinstance(entry["nbytes"], int)
            assert json.loads(json.dumps(entry)) == entry

    @pytest.mark.parametrize(
        "build, state",
        [
            pytest.param(_two_output_graph, "splittable", id="splittable"),
            pytest.param(_cond_graph, "blocked", id="blocked"),
        ],
    )
    def test_repr_matches_the_analysis_format(self, build, state):
        analysis = analyze_inputs(build())
        assert _RE_ANALYSIS_REPR.fullmatch(repr(analysis))
        assert repr(analysis).endswith(f"{state}>")

    def test_repr_matches_the_info_format(self):
        analysis = analyze_inputs(_two_output_graph())
        for info in analysis.inputs:
            assert _RE_INFO_REPR.fullmatch(repr(info)), repr(info)

    def test_text_has_one_row_per_tensor_input(self):
        analysis = analyze_inputs(_two_output_graph())
        text = analysis.text()
        lines = text.splitlines()
        # the header carries the graph-level counts
        assert "tensor inputs" in text
        assert "entry-block ops" in text
        assert "tensor outputs" in text
        # one row per TENSOR input, each starting with its index field
        for info in analysis.inputs:
            prefix = f"  {info.index:>4}  "
            assert any(line.startswith(prefix) for line in lines), (info.index, text)

    def test_text_reports_a_blocked_graph(self):
        analysis = analyze_inputs(_cond_graph())
        text = analysis.text()
        assert "splittable: no" in text
        assert analysis.blocked_reason in text

    def test_analysis_never_mutates_the_graph(self):
        graph = _two_output_graph()
        before = graph.digest()
        analysis = analyze_inputs(graph)
        assert graph.digest() == before
        assert analysis.source is graph
        # a second pass sees the identical graph
        assert analyze_inputs(graph).as_dict() == analysis.as_dict()
