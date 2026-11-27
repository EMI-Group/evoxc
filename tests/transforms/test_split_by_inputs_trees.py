"""Tests for `evoxc.split_by_inputs` — pytree, static, symbolic, persistence
and public-surface pins.

Contract under test (binding): `evoxc/transforms/CONTEXT.md`
"split_by_inputs (structural pass, binding)" plus its "Design Decisions".

`split_by_inputs` slices ONE traced graph into a `prepare` half (the FIXED
inputs' contribution) and an `apply` half (the VARYING inputs on top of the
prepared boundary); both halves are ordinary graphs of ordinary ops, so the
whole runtime contract is

    prepared = evoxc.run(prepare_executable, *fixed_args)     # a flat tuple
    outputs  = evoxc.run(apply_executable, varying_tree, prepared)

with `apply`'s root being the synthetic 2-child tuple ``(varying inputs,
prepared tuple)``. `prepare`'s input tree is the SOURCE tree pruned to the
fixed tensor leaves (statics preserved and re-indexed), `apply`'s element 0 is
the same tree pruned to the varying tensor leaves.

The pass is structural only — it never executes and never caches — and
`evoxc.build` deliberately rejects an already-traced `Graph`, so every
numerical check here stages the halves through the explicit
`lower` → `compile` → `load` pipeline. Two nested-pytree pins are kept as
strict `xfail`s against a documented prune bug (see their `BUG(evoxc)`
markers and this suite's Known Issues entry).
"""

import numpy as np
import pytest

import evoxc
from evoxc import core

from tests.transforms._fd_utils import to_np

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_W = evoxc.TensorSpec((3, 4), evoxc.float32)
_X = evoxc.TensorSpec((3, 4), evoxc.float32)
_Z = evoxc.TensorSpec((4,), evoxc.float32)
_SCALE = 3


def _stage(graph):
    """Explicit staging pipeline (lower → compile → load) for ONE half."""
    return evoxc.load(evoxc.compile(evoxc.lower(graph)))


def _rand(shape, seed=0):
    """A deterministic float32 array (no global RNG state)."""
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _static_records(graph):
    """`(value, kind, index, path)` for every static record of a graph."""
    return [
        (record.value, record.kind, record.index, record.path)
        for record in graph.static_values
    ]


def _nested(batch, z, scale):
    """A nested-dict batch whose two leaves feed ONE fixed-only subgraph."""
    pre = evoxc.sum(evoxc.tanh(evoxc.multiply(batch["x"], batch["y"])))
    return evoxc.multiply(evoxc.add(z, pre), float(scale))


def _flat(w, x, scale):
    """A flat signature with a trailing static Python value."""
    return evoxc.multiply(evoxc.multiply(x, evoxc.tanh(w)), float(scale))


def _mixed_nested(inputs, scale):
    """A nested container whose two leaves split ACROSS the boundary."""
    return evoxc.multiply(
        evoxc.multiply(inputs["y"], evoxc.tanh(inputs["x"])), float(scale)
    )


# ---------------------------------------------------------------------------
# 1. nested pytree inputs + a static leaf (prune at container granularity)
# ---------------------------------------------------------------------------


def test_nested_container_leaves_split_at_the_argument_boundary():
    """A whole nested container fixed + a varying arg + a static leaf."""
    graph = evoxc.trace(_nested, {"x": _W, "y": _X}, _Z, _SCALE)
    assert len(graph.tensor_specs) == 3
    assert len(graph.static_values) == 1

    plan = evoxc.split_by_inputs(graph, fixed=({"x": True, "y": True}, False, None))
    plan.verify()

    # --- the leaves ------------------------------------------------
    assert plan.fixed_leaves == (0, 1)
    assert plan.varying_leaves == (2,)
    assert plan.source is graph

    # --- prepare: the pruned FIXED tree, source spec objects by identity
    assert len(plan.prepare.tensor_specs) == 2
    assert plan.prepare.tensor_specs[0] is graph.tensor_specs[0]
    assert plan.prepare.tensor_specs[1] is graph.tensor_specs[1]
    # the static leaf survives, re-indexed into the pruned tree
    assert _static_records(plan.prepare) == [(_SCALE, "int", 2, (1,))]
    assert plan.prepare.input_specs.num_leaves == (
        len(plan.prepare.tensor_specs) + len(plan.prepare.static_values)
    )

    # --- apply: the pruned VARYING tree, then the boundary tuple ----
    assert plan.apply.tensor_specs[0] is graph.tensor_specs[2]
    assert len(plan.apply.tensor_specs) == 1 + len(plan.boundary)
    assert _static_records(plan.apply) == [(_SCALE, "int", 1, (0, 1))]
    assert plan.apply.input_specs.num_leaves == (
        len(plan.apply.tensor_specs) + len(plan.apply.static_values)
    )
    # the synthetic 2-child root: (varying inputs, prepared tuple)
    root = plan.apply.input_specs
    assert root.type is tuple
    assert len(root.children) == 2
    varying_tree, boundary_tree = root.children
    assert varying_tree.type is tuple
    assert boundary_tree.type is tuple
    assert boundary_tree.num_leaves == len(plan.boundary)

    # --- the fixed half crosses as ONE rank-0 value -----------------
    assert len(plan.boundary) == 1
    assert plan.boundary_specs[0].shape == ()
    assert plan.boundary[0].is_input is False
    assert plan.boundary[0].defining_op is not None

    # --- the correctness law, bit-exactly ---------------------------
    xv, yv, zv = _rand((3, 4), 1), _rand((3, 4), 2), _rand((4,), 3)
    reference = to_np(evoxc.run(_stage(graph), {"x": xv, "y": yv}, zv, _SCALE))
    prepared = evoxc.run(_stage(plan.prepare), {"x": xv, "y": yv}, _SCALE)
    assert isinstance(prepared, tuple)
    assert len(prepared) == len(plan.boundary)
    got = to_np(evoxc.run(_stage(plan.apply), (zv, _SCALE), prepared))
    assert np.array_equal(got, reference)

    # --- the mask axis and the tensor-index axis describe the SAME split
    by_leaves = evoxc.split_by_inputs(graph, fixed_leaves=[0, 1])
    assert by_leaves.fixed_leaves == plan.fixed_leaves
    assert by_leaves.prepare.digest() == plan.prepare.digest()
    assert by_leaves.apply.digest() == plan.apply.digest()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG(evoxc): prune_input_tree reuses an ancestor TreeSpec verbatim when "
        "its child count is unchanged, so a per-leaf prune INSIDE a nested "
        "container keeps the dropped leaf and rebuild_static_values raises "
        "VerificationError"
    ),
)
def test_nested_partial_prune_keeps_only_the_fixed_tensor_leaves():
    """Splitting INSIDE a nested container: the pruned signature is minimal.

    BUG(evoxc): `_split_core.prune_input_tree` returns an ancestor `TreeSpec`
    VERBATIM whenever its direct child count is unchanged, even when a
    DESCENDANT was pruned — so the "pruned" tree still carries the dropped
    varying leaf and `rebuild_static_values` raises `VerificationError`
    ("the pruned input tree has 3 leaves but 2 source leaf positions were
    kept"). The documented contract (`evoxc/transforms/CONTEXT.md` Known
    Issues: "plain tuple/list/dict prune freely") says a per-leaf prune inside
    a plain dict/tuple container is exactly what this pass must support.

    Repro::

        graph = evoxc.trace(_mixed_nested, {"x": s, "y": s}, 2)
        evoxc.split_by_inputs(graph, fixed=({"x": True, "y": False}, None))
        # -> VerificationError: the pruned input tree has 3 leaves but 2
        #    source leaf positions were kept (+0 appended leaves)
    """
    graph = evoxc.trace(_mixed_nested, {"x": _W, "y": _X}, _SCALE)
    plan = evoxc.split_by_inputs(graph, fixed=({"x": True, "y": False}, None))
    plan.verify()

    assert plan.fixed_leaves == (0,)
    assert plan.varying_leaves == (1,)

    # `prepare` keeps the fixed dict leaf only (the static leaf stays)
    assert len(plan.prepare.tensor_specs) == 1
    assert plan.prepare.tensor_specs[0] is graph.tensor_specs[0]
    assert _static_records(plan.prepare) == [(_SCALE, "int", 1, (1,))]
    assert plan.prepare.input_specs.num_leaves == 2

    # `apply` takes the varying dict leaf only, then the prepared tuple
    assert plan.apply.tensor_specs[0] is graph.tensor_specs[1]
    assert len(plan.apply.tensor_specs) == 1 + len(plan.boundary)
    assert plan.apply.input_specs.num_leaves == (
        len(plan.apply.tensor_specs) + len(plan.apply.static_values)
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG(evoxc): the same prune_input_tree ancestor-reuse bug blocks the "
        "plan for a per-leaf prune inside a nested container, so the split can "
        "never be constructed and run"
    ),
)
def test_nested_partial_prune_end_to_end_matches_the_unsplit_graph():
    """The same nested split, run: `apply(prepare(f), v) == f(f, v)`.

    BUG(evoxc): blocked by the same `prune_input_tree` ancestor-reuse bug as
    `test_nested_partial_prune_keeps_only_the_fixed_tensor_leaves` — the plan
    cannot even be constructed (see that test's repro).
    """
    graph = evoxc.trace(_mixed_nested, {"x": _W, "y": _X}, _SCALE)
    plan = evoxc.split_by_inputs(graph, fixed_leaves=[0])
    plan.verify()

    xv, yv = _rand((3, 4), 4), _rand((3, 4), 5)
    reference = to_np(evoxc.run(_stage(graph), {"x": xv, "y": yv}, _SCALE))
    prepared = evoxc.run(_stage(plan.prepare), {"x": xv}, _SCALE)
    got = to_np(evoxc.run(_stage(plan.apply), ({"y": yv}, _SCALE), prepared))
    assert np.array_equal(got, reference)

    # the mask form and the tensor-index form agree digest-for-digest
    by_mask = evoxc.split_by_inputs(graph, fixed=({"x": True, "y": False}, None))
    assert by_mask.prepare.digest() == plan.prepare.digest()
    assert by_mask.apply.digest() == plan.apply.digest()


# ---------------------------------------------------------------------------
# 1b. static leaves in a flat signature (preserved + re-indexed)
# ---------------------------------------------------------------------------


def test_static_leaf_survives_in_both_halves_and_both_accept_it():
    graph = evoxc.trace(_flat, _W, _X, _SCALE)
    assert _static_records(graph) == [(_SCALE, "int", 2, (2,))]

    plan = evoxc.split_by_inputs(graph, fixed=(True, False, None))
    plan.verify()

    assert plan.fixed_leaves == (0,)
    assert plan.varying_leaves == (1,)

    # a static value specializes BOTH halves instead of crossing the boundary
    assert _static_records(plan.prepare) == [(_SCALE, "int", 1, (1,))]
    assert _static_records(plan.apply) == [(_SCALE, "int", 1, (0, 1))]
    for half in (plan.prepare, plan.apply):
        assert half.input_specs.num_leaves == (
            len(half.tensor_specs) + len(half.static_values)
        )
    assert len(plan.prepare.tensor_specs) == 1
    assert len(plan.apply.tensor_specs) == 1 + len(plan.boundary)

    # both halves still TAKE it at run time, and the law holds bit-exactly
    wv, xv = _rand((3, 4), 6), _rand((3, 4), 7)
    reference = to_np(evoxc.run(_stage(graph), wv, xv, _SCALE))
    prepared = evoxc.run(_stage(plan.prepare), wv, _SCALE)
    assert isinstance(prepared, tuple)
    assert prepared[0].shape == (3, 4)
    got = to_np(evoxc.run(_stage(plan.apply), (xv, _SCALE), prepared))
    assert np.array_equal(got, reference)

    # the mask and the tensor-index forms are the same split
    by_leaves = evoxc.split_by_inputs(graph, fixed_leaves=[0])
    assert by_leaves.prepare.digest() == plan.prepare.digest()
    assert by_leaves.apply.digest() == plan.apply.digest()


def test_none_at_a_tensor_leaf_means_varying():
    """`None` behaves like `False` at a tensor leaf (only statics REQUIRE it)."""
    graph = evoxc.trace(_flat, _W, _X, _SCALE)
    by_bools = evoxc.split_by_inputs(graph, fixed=(True, False, None))
    by_nones = evoxc.split_by_inputs(graph, fixed=(True, None, None))

    assert by_nones.fixed_leaves == by_bools.fixed_leaves == (0,)
    assert by_nones.varying_leaves == by_bools.varying_leaves == (1,)
    assert by_nones.prepare.digest() == by_bools.prepare.digest()
    assert by_nones.apply.digest() == by_bools.apply.digest()


# ---------------------------------------------------------------------------
# 1c. the mask/tensor-index axes through a nested tree (validation only)
# ---------------------------------------------------------------------------


def test_mask_at_a_static_leaf_raises_in_a_nested_tree():
    """A non-`None` mask entry at a static leaf is refused at ANY depth."""
    graph = evoxc.trace(_mixed_nested, {"x": _W, "y": _X}, _SCALE)

    with pytest.raises(core.TransformError) as excinfo:
        evoxc.split_by_inputs(graph, fixed=({"x": True, "y": False}, False))
    message = str(excinfo.value)
    assert "STATIC input leaf (False)" in message
    assert "use None there" in message

    # the tensor-index axis does not depend on the tree's depth either
    with pytest.raises(core.TransformError) as excinfo:
        evoxc.split_by_inputs(graph, fixed_leaves=[0, 3])
    assert "out of range" in str(excinfo.value)
    assert "2 tensor inputs" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 2. symbolic dims across the boundary
# ---------------------------------------------------------------------------


def test_symbolic_varying_dim_runs_at_two_extents_on_one_apply():
    """One staged `apply` serves several extents; the boundary is `n`-free."""
    n = core.Dim("n")
    weight_spec = evoxc.TensorSpec((4, 3), evoxc.float32)
    data_spec = evoxc.TensorSpec((n, 4), evoxc.float32)

    def fn(weight, data):
        return evoxc.sum(evoxc.tanh(evoxc.matmul(data, weight)))

    graph = evoxc.trace(fn, weight_spec, data_spec)
    plan = evoxc.split_by_inputs(graph, fixed_leaves=[0])
    plan.verify()

    assert plan.fixed_leaves == (0,)
    assert plan.varying_leaves == (1,)
    # the varying half keeps the SOURCE spec (and its `Dim` object) by identity
    assert plan.apply.tensor_specs[0] is graph.tensor_specs[1]
    assert plan.apply.tensor_specs[0].shape[0] is n

    # the prepared boundary is the fixed half's contribution: shape-static
    assert len(plan.boundary) == 1
    assert plan.boundary_specs[0].shape == (4, 3)
    assert not any(
        isinstance(dim, (core.Dim, core.DimExpr))
        for dim in plan.boundary_specs[0].shape
    )

    prepare_executable = _stage(plan.prepare)
    apply_executable = _stage(plan.apply)
    unsplit_executable = _stage(graph)

    weight = _rand((4, 3), 11)
    for extent in (2, 5):
        data = _rand((extent, 4), 100 + extent)
        prepared = evoxc.run(prepare_executable, weight)
        assert isinstance(prepared, tuple) and len(prepared) == 1
        assert prepared[0].shape == (4, 3)  # `n`-free at run time too
        got = to_np(evoxc.run(apply_executable, (data,), prepared))
        reference = to_np(evoxc.run(unsplit_executable, weight, data))
        assert np.array_equal(got, reference)


def test_pass_through_fixed_input_keeps_its_source_spec_object():
    """A fixed input that IS the boundary value keeps its spec and `Dim`s."""
    n = core.Dim("n")
    scale_spec = evoxc.TensorSpec((n, 4), evoxc.float32)
    data_spec = evoxc.TensorSpec((n, 4), evoxc.float32)

    def fn(scale_tensor, data):
        return evoxc.sum(evoxc.multiply(data, scale_tensor))

    graph = evoxc.trace(fn, scale_spec, data_spec)
    plan = evoxc.split_by_inputs(graph, fixed_leaves=[0])
    plan.verify()

    assert len(plan.boundary) == 1
    value = plan.boundary[0]
    assert value.is_input is True
    assert value.defining_op is None
    assert value.producer == "<input>"
    assert value.spec is graph.tensor_specs[0]
    assert value.spec.shape[0] is n
    assert plan.boundary_specs[0] is graph.tensor_specs[0]

    # a pass-through boundary computes nothing: `prepare` is a bare return
    scale = _rand((3, 4), 12)
    data = _rand((3, 4), 13)
    prepared = evoxc.run(_stage(plan.prepare), scale)
    assert len(prepared) == 1 and prepared[0].shape == (3, 4)
    got = to_np(evoxc.run(_stage(plan.apply), (data,), prepared))
    reference = to_np(evoxc.run(_stage(graph), scale, data))
    assert np.array_equal(got, reference)


# ---------------------------------------------------------------------------
# 3. persistence round-trip of both halves
# ---------------------------------------------------------------------------


def test_split_halves_round_trip_through_graph_save_and_load(tmp_path):
    graph = evoxc.trace(_flat, _W, _X, _SCALE)
    plan = evoxc.split_by_inputs(graph, fixed_leaves=[0])

    prepare_path = str(tmp_path / "prepare.evoxc")
    apply_path = str(tmp_path / "apply.evoxc")
    plan.prepare.save(prepare_path)
    plan.apply.save(apply_path)

    loaded_prepare = evoxc.Graph.load(prepare_path)
    loaded_apply = evoxc.Graph.load(apply_path)

    # `digest()` excludes source locations, so it survives the container
    assert loaded_prepare.digest() == plan.prepare.digest()
    assert loaded_apply.digest() == plan.apply.digest()

    wv, xv = _rand((3, 4), 8), _rand((3, 4), 9)
    reference = to_np(evoxc.run(_stage(graph), wv, xv, _SCALE))
    prepared = evoxc.run(_stage(loaded_prepare), wv, _SCALE)
    assert _static_records(loaded_prepare) == _static_records(plan.prepare)
    assert _static_records(loaded_apply) == _static_records(plan.apply)
    got = to_np(evoxc.run(_stage(loaded_apply), (xv, _SCALE), prepared))
    assert np.array_equal(got, reference)


# ---------------------------------------------------------------------------
# 4. the callable / `Defn` form
# ---------------------------------------------------------------------------


def test_split_callable_kind_repr_and_defn_form():
    callable_plan = evoxc.split_by_inputs(_flat, fixed_leaves=[0])
    assert isinstance(callable_plan, evoxc.SplitCallable)
    assert callable_plan.kind == "split_by_inputs"
    assert "SplitCallable" in repr(callable_plan)

    plan = callable_plan(_W, _X, _SCALE)
    assert isinstance(plan, evoxc.SplitPlan)

    @evoxc.defn
    def flat_defn(w, x, scale):
        return evoxc.multiply(evoxc.multiply(x, evoxc.tanh(w)), float(scale))

    defn_callable = evoxc.split_by_inputs(flat_defn, fixed=(True, False, None))
    assert isinstance(defn_callable, evoxc.SplitCallable)
    assert defn_callable.kind == "split_by_inputs"
    assert "SplitCallable" in repr(defn_callable)

    defn_plan = defn_callable(_W, _X, _SCALE)
    assert isinstance(defn_plan, evoxc.SplitPlan)
    assert defn_plan.fixed_leaves == plan.fixed_leaves
    assert defn_plan.prepare.digest() == plan.prepare.digest()
    assert defn_plan.apply.digest() == plan.apply.digest()


def test_split_callable_traces_fresh_each_call():
    callable_plan = evoxc.split_by_inputs(_flat, fixed_leaves=[0])
    first = callable_plan(_W, _X, _SCALE)
    second = callable_plan(_W, _X, _SCALE)

    assert first is not second
    assert first.source is not second.source
    assert first.prepare is not second.prepare
    assert first.apply is not second.apply
    # same program, so the structural digests agree
    assert first.prepare.digest() == second.prepare.digest()
    assert first.apply.digest() == second.apply.digest()


def test_split_callable_rejects_concrete_tensors():
    callable_plan = evoxc.split_by_inputs(_flat, fixed_leaves=[0])
    concrete = core.Tensor(_rand((3, 4), 10))

    with pytest.raises(core.TraceError) as excinfo:
        callable_plan(concrete, concrete, _SCALE)

    message = str(excinfo.value)
    assert "never executes" in message
    assert "explicitly" in message


# ---------------------------------------------------------------------------
# 5. public-surface identity pins
# ---------------------------------------------------------------------------


def test_top_level_identity_pins():
    assert evoxc.split_by_inputs is evoxc.transforms.split_by_inputs
    assert evoxc.analyze_inputs is evoxc.transforms.analyze_inputs
    assert evoxc.SplitPlan is evoxc.transforms.SplitPlan
    assert evoxc.InputAnalysis is evoxc.transforms.InputAnalysis


def test_transforms_all_exposes_the_split_surface():
    names = (
        "split_by_inputs",
        "analyze_inputs",
        "SplitPlan",
        "BoundaryValue",
        "InputAnalysis",
        "InputInfo",
        "SplitCallable",
    )
    for name in names:
        assert name in evoxc.transforms.__all__
        assert hasattr(evoxc.transforms, name)
        assert hasattr(evoxc, name)
        assert getattr(evoxc, name) is getattr(evoxc.transforms, name)
