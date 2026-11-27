"""Tests for `evoxc.split_by_inputs` — the structural `prepare` / `apply` split.

Contract under test (binding): `evoxc/transforms/CONTEXT.md` "split_by_inputs
(structural pass, binding)". `split_by_inputs` slices ONE traced graph into TWO
ordinary graphs at the boundary between inputs that stay FIXED across calls and
inputs that VARY per call: `prepare` computes the fixed inputs' contribution
once and returns a FLAT tuple of boundary tensors, `apply` evaluates the varying
inputs on top of that tuple and returns the source output tree. The source graph
is never mutated; both halves are ordinary graphs of cloned ordinary ops.

CORRECTNESS LAW: ``apply(prepare(f), v)`` agrees with ``f(f, v)`` under the
backend's normal numerical contract — the ops are the SAME ops, cloned, in the
same order, so on one backend the agreement is BIT-EXACT. The boundary is a
SNAPSHOT of the fixed half's contribution (never re-derived per call), so a
changed fixed input means re-running `prepare`.

Runtime call shapes (binding): ``evoxc.run(prepare, *fixed_args)`` → a flat
tuple; ``evoxc.run(apply, varying_tree, prepared_tuple)`` → the source output
tree. `apply`'s root argument is the synthetic 2-child tuple ``(varying tree,
prepared tuple)``, and the varying tree is the source input tree PRUNED to the
varying tensor leaves (so a 3-input graph with one varying leaf yields a
1-tuple). `evoxc.build` rejects an already-traced `Graph`, so both halves are
staged with the explicit `lower → compile → load` chain.
"""

import json
import re

import numpy as np
import pytest

import evoxc
from evoxc import core
from tests.backends import _native_utils as nu


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

#: Small static shapes; `(dataset, params, candidate)` is the fitness layout.
DATASET = (8, 4)
PARAMS = (4, 3)
CANDIDATE = (3,)


def stage(graph, backend="numpy"):
    """Stage one half through the explicit pipeline: lower → compile → load."""
    return evoxc.load(
        evoxc.compile(evoxc.lower(graph, backend=backend), backend=backend),
        backend=backend,
    )


def to_np(value):
    """Unwrap `evoxc.run` structured outputs (bare Tensor or tuple) to numpy."""
    if isinstance(value, core.Tensor):
        return value.numpy()
    if isinstance(value, (tuple, list)):
        return tuple(to_np(item) for item in value)
    return value


def entry_ops(graph):
    """The graph's NON-TERMINATOR entry-block ops, in program order."""
    return tuple(
        op for op in graph.module.main.entry_block.ops if not op.is_terminator
    )


def tensor_output_count(graph):
    """Number of TENSOR leaves of a graph's output tree."""
    return graph.output_tree.num_leaves - len(graph.output_static_values)


def return_ids(graph):
    """The SSA ids of the graph's `return` operands."""
    return tuple(
        value.id for value in graph.module.main.entry_block.terminator.operands
    )


def as_json(value):
    """`value` with every tuple rendered as a list (the `as_dict()` shape)."""
    if isinstance(value, dict):
        return {key: as_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [as_json(item) for item in value]
    return value


# --- the graph under test ---------------------------------------------------
#
# A fitness-style graph: the fixed part does REAL work (standardize the
# dataset, then `tanh(Xn @ W1)`), the varying part is cheap (a reduction that
# combines a per-call `candidate`). Fixing `dataset` + `params` puts 8 ops in
# `prepare` and leaves 3 in `apply`.


def fitness(dataset, params, candidate):
    mean = evoxc.mean(dataset, axes=(0,))
    centered = dataset - mean
    std = evoxc.sqrt(evoxc.mean(centered * centered, axes=(0,)))
    normalized = centered / std
    hidden = evoxc.tanh(evoxc.dot(normalized, params))
    return evoxc.mean(evoxc.sum(hidden * candidate, axes=(1,)), axes=(0,))


def fitness_graph():
    return evoxc.trace(
        fitness,
        evoxc.TensorSpec(DATASET, evoxc.float32),  # 0: dataset
        evoxc.TensorSpec(PARAMS, evoxc.float32),  # 1: params
        evoxc.TensorSpec(CANDIDATE, evoxc.float32),  # 2: candidate
    )


def _rng(seed=0):
    return np.random.default_rng(seed)


def _fitness_inputs(seed=0):
    rng = _rng(seed)
    dataset = rng.normal(size=DATASET).astype(np.float32)
    params = rng.normal(size=PARAMS).astype(np.float32)
    candidates = [
        rng.normal(size=CANDIDATE).astype(np.float32) for _ in range(3)
    ]
    return dataset, params, candidates


# ---------------------------------------------------------------------------
# 1. correctness law — bit-exact, one prepared result reused across candidates
# ---------------------------------------------------------------------------


def test_correctness_law_is_bit_exact_across_candidates():
    """`apply(prepare(f), v)` == `f(f, v)` bit-exactly, one boundary reused."""
    graph = fitness_graph()
    plan = evoxc.split_by_inputs(graph, fixed_leaves=[0, 1])
    assert plan.fixed_leaves == (0, 1)
    assert plan.varying_leaves == (2,)

    full, prepare, apply = stage(graph), stage(plan.prepare), stage(plan.apply)
    dataset, params, candidates = _fitness_inputs()

    # ONE prepare result, reused for every candidate: the boundary is the
    # snapshot of the fixed half's contribution, not a per-turn computation.
    prepared = evoxc.run(prepare, dataset, params)
    assert isinstance(prepared, tuple) and len(prepared) == len(plan.boundary)

    for candidate in candidates:
        got = evoxc.run(apply, (candidate,), prepared).numpy()
        ref = evoxc.run(full, dataset, params, candidate).numpy()
        assert got.shape == ref.shape == ()
        assert got.dtype == ref.dtype == np.float32
        assert np.array_equal(got, ref), (
            "the split must be bit-exact: both halves carry the SAME ops "
            "cloned in the same order"
        )


def test_the_boundary_is_a_snapshot_of_the_fixed_half():
    """A changed fixed input needs a NEW `prepare` run — nothing re-derives it."""
    graph = fitness_graph()
    plan = evoxc.split_by_inputs(graph, fixed_leaves=[0, 1])
    full, prepare, apply = stage(graph), stage(plan.prepare), stage(plan.apply)

    dataset, params, candidates = _fitness_inputs()
    other_dataset = (dataset + np.float32(3.0)).astype(np.float32)
    candidate = candidates[0]

    first = evoxc.run(prepare, dataset, params)
    second = evoxc.run(prepare, other_dataset, params)
    # A different dataset changes the boundary: it IS the fixed half's work.
    assert not np.array_equal(first[0].numpy(), second[0].numpy())

    assert np.array_equal(
        evoxc.run(apply, (candidate,), second).numpy(),
        evoxc.run(full, other_dataset, params, candidate).numpy(),
    )
    # The stale snapshot still describes the OLD dataset — it is never
    # refreshed implicitly, which is what "snapshot" means here.
    assert np.array_equal(
        evoxc.run(apply, (candidate,), first).numpy(),
        evoxc.run(full, dataset, params, candidate).numpy(),
    )


def test_params_only_fixed_prepare_forwards_the_input_spec():
    """Fixing just `params` makes `prepare` a pure pass-through of that input."""
    graph = fitness_graph()
    plan = evoxc.split_by_inputs(graph, fixed_leaves=[1])
    assert plan.fixed_leaves == (1,)
    assert plan.varying_leaves == (0, 2)

    summary = plan.summary()
    assert summary["prepare_ops"] == 0  # nothing to compute, only to forward
    assert len(plan.boundary) == 1
    (value,) = plan.boundary
    assert value.is_input is True and value.defining_op is None
    assert value.spec is graph.tensor_specs[1]

    full, prepare, apply = stage(graph), stage(plan.prepare), stage(plan.apply)
    dataset, params, candidates = _fitness_inputs(seed=1)
    prepared = evoxc.run(prepare, params)
    for candidate in candidates:
        got = evoxc.run(apply, (dataset, candidate), prepared).numpy()
        ref = evoxc.run(full, dataset, params, candidate).numpy()
        assert np.array_equal(got, ref)


@pytest.mark.skipif(not nu.ENGINE_AVAILABLE, reason=nu.SKIP_REASON)
def test_correctness_law_holds_on_the_native_backend():
    """The same bit-exact split on the native engine (native is self-consistent).

    The pin is SPLIT vs UNSPLIT on the SAME backend: numpy-vs-native float
    reassociation (`faer` vs OpenBLAS `dot`, the platform libm transcendentals)
    is a documented cross-backend residual and is not what this test measures.
    """
    graph = fitness_graph()
    plan = evoxc.split_by_inputs(graph, fixed_leaves=[0, 1])
    full = stage(graph, backend="native")
    prepare = stage(plan.prepare, backend="native")
    apply = stage(plan.apply, backend="native")

    dataset, params, candidates = _fitness_inputs(seed=2)
    prepared = evoxc.run(prepare, dataset, params)
    for candidate in candidates:
        got = evoxc.run(apply, (candidate,), prepared).numpy()
        ref = evoxc.run(full, dataset, params, candidate).numpy()
        assert np.array_equal(got, ref)


# ---------------------------------------------------------------------------
# 2. plan.verify() + the wiring law
# ---------------------------------------------------------------------------


def test_plan_verify_and_the_wiring_law():
    graph = fitness_graph()
    plan = evoxc.split_by_inputs(graph, fixed_leaves=[0, 1])
    plan.verify()  # raises VerificationError on any wiring bug

    boundary_count = len(plan.boundary)
    assert boundary_count == 1
    assert tensor_output_count(plan.prepare) == boundary_count
    assert len(plan.apply.tensor_specs) - len(plan.varying_leaves) == boundary_count

    # Per-position shape/dtype agreement across the three descriptions of the
    # same boundary tensor: the plan's record, `prepare`'s output type and
    # `apply`'s prepared input spec.
    prepare_outputs = plan.prepare.module.main.output_types
    prepared_inputs = plan.apply.tensor_specs[len(plan.varying_leaves):]
    assert len(prepare_outputs) == len(prepared_inputs) == boundary_count
    assert plan.boundary_specs == tuple(value.spec for value in plan.boundary)
    for index, value in enumerate(plan.boundary):
        assert tuple(prepare_outputs[index].shape) == tuple(value.shape)
        assert prepare_outputs[index].dtype == value.dtype
        assert tuple(prepared_inputs[index].shape) == tuple(value.shape)
        assert prepared_inputs[index].dtype == value.dtype
        assert tuple(plan.boundary_specs[index].shape) == tuple(value.shape)

    # `apply` hands the SOURCE output tree back, untouched.
    assert plan.apply.output_tree.num_leaves == plan.source.output_tree.num_leaves
    assert plan.apply.output_tree is plan.source.output_tree

    # `apply`'s signature is the synthetic 2-child root (varying tree,
    # prepared tuple); `prepare`'s is the fixed pruned tree.
    assert len(plan.apply.input_specs.children) == 2
    varying_tree, prepared_tree = plan.apply.input_specs.children
    assert varying_tree.num_leaves == len(plan.varying_leaves)
    assert prepared_tree.num_leaves == boundary_count
    assert plan.prepare.input_specs.num_leaves == len(plan.fixed_leaves)


# ---------------------------------------------------------------------------
# 3. a pass-through fixed input consumed by varying work
# ---------------------------------------------------------------------------


def test_fixed_input_consumed_by_varying_work_crosses_by_identity():
    def scale_then_reduce(scale, x):
        return evoxc.sum(scale * x)

    graph = evoxc.trace(
        scale_then_reduce,
        evoxc.TensorSpec((4,), evoxc.float32),
        evoxc.TensorSpec((4,), evoxc.float32),
    )
    plan = evoxc.split_by_inputs(graph, fixed_leaves=[0])
    assert plan.summary()["prepare_ops"] == 0

    (value,) = plan.boundary
    assert value.is_input is True
    assert value.defining_op is None
    assert value.producer == "<input>"
    # The SOURCE spec OBJECT survives — identity, not equality.
    assert value.spec is graph.tensor_specs[0]
    assert value.source_value_id == graph.module.main.entry_block.arguments[0].id
    assert value.shape == (4,) and value.dtype == evoxc.float32
    assert value.nbytes == 16

    full, prepare, apply = stage(graph), stage(plan.prepare), stage(plan.apply)
    rng = _rng(3)
    scale = rng.normal(size=(4,)).astype(np.float32)
    x = rng.normal(size=(4,)).astype(np.float32)
    prepared = evoxc.run(prepare, scale)
    assert np.array_equal(
        evoxc.run(apply, (x,), prepared).numpy(),
        evoxc.run(full, scale, x).numpy(),
    )


# ---------------------------------------------------------------------------
# 4. an output that depends only on fixed inputs
# ---------------------------------------------------------------------------


def test_fixed_only_output_crosses_the_boundary():
    def two_outputs(w, x):
        fixed_only = evoxc.sum(evoxc.tanh(w))
        mixed = evoxc.sum(evoxc.tanh(w) * x)
        return (fixed_only, mixed)

    graph = evoxc.trace(
        two_outputs,
        evoxc.TensorSpec((4,), evoxc.float32),
        evoxc.TensorSpec((4,), evoxc.float32),
    )
    plan = evoxc.split_by_inputs(graph, fixed_leaves=[0])
    ids = return_ids(graph)
    assert len(ids) == 2

    # A fixed-only value returned DIRECTLY is a boundary value: `apply` cannot
    # recompute it, it must receive it.
    crossing = [value for value in plan.boundary if value.source_value_id in ids]
    assert [value.source_value_id for value in crossing] == [ids[0]]
    assert crossing[0].is_input is False
    assert crossing[0].defining_op == "reduce_sum"
    assert crossing[0].producer == "reduce_sum"
    assert len(plan.boundary) == 2  # ... plus the fixed-only `tanh` operand

    full, prepare, apply = stage(graph), stage(plan.prepare), stage(plan.apply)
    rng = _rng(4)
    w = rng.normal(size=(4,)).astype(np.float32)
    x = rng.normal(size=(4,)).astype(np.float32)
    got = to_np(evoxc.run(apply, (x,), evoxc.run(prepare, w)))
    ref = to_np(evoxc.run(full, w, x))
    assert len(got) == len(ref) == 2
    assert np.array_equal(got[0], ref[0])
    assert np.array_equal(got[1], ref[1])


# ---------------------------------------------------------------------------
# 5. dead ops are dropped from BOTH halves
# ---------------------------------------------------------------------------


def test_dead_ops_are_dropped_from_both_halves():
    def with_dead_branch(w, x):
        live = evoxc.tanh(w)
        _dead_fixed = evoxc.sin(w)  # feeds nothing
        _dead_mixed = evoxc.multiply(x, 3.0)  # feeds nothing
        return live * x

    graph = evoxc.trace(
        with_dead_branch,
        evoxc.TensorSpec((4,), evoxc.float32),
        evoxc.TensorSpec((4,), evoxc.float32),
    )
    plan = evoxc.split_by_inputs(graph, fixed_leaves=[0])
    summary = plan.summary()

    assert summary["dropped_ops"] > 0
    assert (
        summary["prepare_ops"] + summary["apply_ops"] + summary["dropped_ops"]
        == summary["source_ops"]
    )
    # The counts are the halves' real non-terminator op counts, not estimates.
    assert summary["source_ops"] == len(entry_ops(graph))
    assert summary["prepare_ops"] == len(entry_ops(plan.prepare))
    assert summary["apply_ops"] == len(entry_ops(plan.apply))

    # Dropping is legal only because the gates admit pure/read ops only.
    prepare_names = [op.name for op in entry_ops(plan.prepare)]
    apply_names = [op.name for op in entry_ops(plan.apply)]
    assert "sin" not in prepare_names + apply_names

    full, prepare, apply = stage(graph), stage(plan.prepare), stage(plan.apply)
    rng = _rng(5)
    w = rng.normal(size=(4,)).astype(np.float32)
    x = rng.normal(size=(4,)).astype(np.float32)
    assert np.array_equal(
        evoxc.run(apply, (x,), evoxc.run(prepare, w)).numpy(),
        evoxc.run(full, w, x).numpy(),
    )


# ---------------------------------------------------------------------------
# 6. a varying input consumed only by dead ops
# ---------------------------------------------------------------------------


def test_varying_input_only_feeding_dead_ops_is_kept_in_apply():
    def with_dead_varying(w, x, junk):
        live = evoxc.tanh(w)
        _dead = evoxc.multiply(junk, 2.0)  # `junk` influences nothing live
        return live * x

    graph = evoxc.trace(
        with_dead_varying,
        evoxc.TensorSpec((4,), evoxc.float32),
        evoxc.TensorSpec((4,), evoxc.float32),
        evoxc.TensorSpec((3,), evoxc.float32),
    )
    plan = evoxc.split_by_inputs(graph, fixed_leaves=[0])
    assert plan.varying_leaves == (1, 2)
    assert plan.unused_varying_leaves == (2,)
    assert plan.summary()["unused_varying_leaves"] == (2,)
    # Kept in `apply`'s signature for stability: all varying specs survive.
    assert [tuple(spec.shape) for spec in plan.apply.tensor_specs] == [
        (4,),
        (3,),
        (4,),
    ]

    full, prepare, apply = stage(graph), stage(plan.prepare), stage(plan.apply)
    rng = _rng(6)
    w = rng.normal(size=(4,)).astype(np.float32)
    x = rng.normal(size=(4,)).astype(np.float32)
    junk = rng.normal(size=(3,)).astype(np.float32)
    prepared = evoxc.run(prepare, w)
    # `apply` still ACCEPTS the unused varying leaf at run time.
    got = evoxc.run(apply, (x, junk), prepared).numpy()
    ref = evoxc.run(full, w, x, junk).numpy()
    assert np.array_equal(got, ref)
    # ... and the value is genuinely inert.
    other = evoxc.run(apply, (x, -junk), prepared).numpy()
    assert np.array_equal(got, other)


# ---------------------------------------------------------------------------
# 7. determinism and purity
# ---------------------------------------------------------------------------


def test_repeat_splits_are_digest_identical():
    graph = fitness_graph()
    first = evoxc.split_by_inputs(graph, fixed_leaves=[0, 1])
    second = evoxc.split_by_inputs(graph, fixed_leaves=[0, 1])
    assert first.prepare.digest() == second.prepare.digest()
    assert first.apply.digest() == second.apply.digest()
    assert first.summary() == second.summary()


def test_mask_form_and_fixed_leaves_form_agree():
    graph = fitness_graph()
    by_leaves = evoxc.split_by_inputs(graph, fixed_leaves=[0, 1])
    by_mask = evoxc.split_by_inputs(graph, fixed=(True, True, False))
    assert by_mask.fixed_leaves == by_leaves.fixed_leaves == (0, 1)
    assert by_mask.varying_leaves == by_leaves.varying_leaves == (2,)
    assert by_mask.prepare.digest() == by_leaves.prepare.digest()
    assert by_mask.apply.digest() == by_leaves.apply.digest()


def test_source_graph_is_not_mutated():
    graph = fitness_graph()
    digest_before = graph.digest()
    specs_before = tuple(graph.tensor_specs)
    ids_before = tuple(id(spec) for spec in graph.tensor_specs)
    ops_before = tuple((op.name, tuple(r.id for r in op.results)) for op in entry_ops(graph))

    plan = evoxc.split_by_inputs(graph, fixed_leaves=[0, 1])

    assert graph.digest() == digest_before
    assert plan.source is graph
    assert tuple(graph.tensor_specs) == specs_before
    assert tuple(id(spec) for spec in graph.tensor_specs) == ids_before
    assert (
        tuple((op.name, tuple(r.id for r in op.results)) for op in entry_ops(graph))
        == ops_before
    )
    # The halves are fresh modules, not views onto the source.
    assert plan.prepare.module is not graph.module
    assert plan.apply.module is not graph.module


# ---------------------------------------------------------------------------
# 8. inspection: summary / as_dict / repr / text
# ---------------------------------------------------------------------------


SUMMARY_KEYS = {
    "fixed_leaves",
    "varying_leaves",
    "boundary_values",
    "boundary_bytes",
    "source_ops",
    "prepare_ops",
    "apply_ops",
    "dropped_ops",
    "unused_varying_leaves",
    "prepare_digest",
    "apply_digest",
}


def test_summary_has_the_eleven_documented_keys():
    plan = evoxc.split_by_inputs(fitness_graph(), fixed_leaves=[0, 1])
    summary = plan.summary()
    assert set(summary) == SUMMARY_KEYS
    assert len(summary) == 11
    assert summary["fixed_leaves"] == (0, 1)
    assert summary["varying_leaves"] == (2,)
    assert summary["boundary_values"] == len(plan.boundary) == 1
    assert summary["boundary_bytes"] == plan.boundary_bytes == 8 * 3 * 4
    assert summary["prepare_digest"] == plan.prepare.digest()
    assert summary["apply_digest"] == plan.apply.digest()


def test_as_dict_survives_a_json_round_trip():
    plan = evoxc.split_by_inputs(fitness_graph(), fixed_leaves=[0, 1])
    data = plan.as_dict()
    # `as_dict()` = `summary()` (tuples as lists) + a per-value boundary list.
    assert set(data) == SUMMARY_KEYS | {"boundary"}
    assert data["fixed_leaves"] == [0, 1]
    assert isinstance(data["boundary"], list)
    assert len(data["boundary"]) == len(plan.boundary)
    assert data["boundary"][0]["is_input"] is False
    assert data["boundary"][0]["defining_op"] == "tanh"

    round_tripped = json.loads(json.dumps(data))
    assert round_tripped == as_json(data)


def test_repr_matches_the_documented_shape():
    plan = evoxc.split_by_inputs(fitness_graph(), fixed_leaves=[0, 1])
    pattern = (
        r"<SplitPlan fixed=\d+ varying=\d+ boundary=\d+ values / .+, "
        r"prepare \d+ ops, apply \d+ ops>"
    )
    assert re.fullmatch(pattern, repr(plan)) is not None, repr(plan)
    assert repr(plan) == (
        f"<SplitPlan fixed={len(plan.fixed_leaves)} "
        f"varying={len(plan.varying_leaves)} "
        f"boundary={len(plan.boundary)} values / "
        f"{8 * 3 * 4} B, "
        f"prepare {len(entry_ops(plan.prepare))} ops, "
        f"apply {len(entry_ops(plan.apply))} ops>"
    )


def test_text_has_one_row_per_boundary_value(capsys):
    graph = fitness_graph()
    plan = evoxc.split_by_inputs(graph, fixed_leaves=[0, 1])
    text = plan.text()
    rows = [line for line in text.splitlines() if re.match(r"^    \[\d+\] ", line)]
    assert len(rows) == len(plan.boundary)
    assert rows[0].startswith("    [0] tanh")
    assert f"boundary ({len(plan.boundary)} values," in text
    assert plan.summary()["prepare_digest"] in text
    assert plan.summary()["apply_digest"] in text
    # `print()` is `text()` on stdout, nothing else.
    plan.print()
    captured = capsys.readouterr()
    assert captured.out == text + "\n"
    assert captured.err == ""
