"""Design-principle compliance tests — the executable spec of `../CONTEXT.md`.

These tests pin the *contracts* that make evoxc an explicit, minimal tensor
graph runtime (see the root CONTEXT.md design principles and the package
contract in `../evoxc/CONTEXT.md`). Groups (one class per group, in file order):

1. `TestNoImplicitTracing`   — `@evoxc.defn` is NOT JIT/eager; no mode switching.
2. `TestClosureCapture`      — captured Tensors are errors; `evoxc.constant` opts in.
3. `TestSymbolicPurity`      — SymbolicTensor has no concrete-data escape hatches.
4. `TestStagingExplicitness` — distinct stage types; shorthand = documented
                               composition only; wrong-stage objects fail clearly.
5. `TestBindIsSugar`         — `evoxc.bind` never alters the graph, validates names.
6. `TestVmapIsSugar`         — `vmap(f, in_axes, out_axes=0)` ≡ `vectorize(trace(f))`.
7. `TestConcreteCreators`    — `zeros`/`ones`/... return concrete Tensors; DLPack.
8. `TestEnpGraphOps`         — `evoxc.numpy.zeros` inside a defn is a GRAPH op.
9. `TestSerializationRoundtrips` — Graph/LoweredProgram/CompiledArtifact save/load.
10. `TestPythonSemantics`    — Python values stay Python; no silent specialization.
11. `TestLocalTensorSemantics` — collectives are explicit ops, never implicit.
12. `TestErrorLocations`     — error messages carry `file.py:line` locations.
13. `TestTreeMapIsComposition` — `tree_map` is transparent sugar over
                               `flatten`/`unflatten` (structure-preserving).
14. `TestSparseExplicitness` — sparse tensors obey the SAME discipline:
                               graph-time computation ops (no eager mode),
                               polymorphic creators, pure symbolic leaves,
                               static-leaf snapshotting, explicit deferrals.
15. `TestExplicitPlacement` — no implicit device transfer: placement is
                               explicit (R1/R2/R3/R5/R6) — host memory cannot
                               be relabeled, `.numpy()` on a non-cpu payload
                               raises, `Tensor.to` is the only transfer API,
                               and run/bind/constant boundaries reject
                               foreign-device tensors.
16. `TestIoExplicitness`      — `evoxc.io` async host IO is documented sugar
                               over the ONE explicit transfer primitive:
                               prefetch/sink schedule the same explicit
                               `Tensor.to` copies on a background worker —
                               never implicit copying, never auto-waiting
                               handles, failures surface at wait/result.

Conventions: small shapes, CPU only, fast. Tests assert the documented
contract; a test exposing a contract violation stays failing with a
`# BUG(evoxc): ...` marker (see `tests/CONTEXT.md`).
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import re
import warnings
from collections import namedtuple

import numpy as np
import pytest

import evoxc

#: Source-location pattern from the root error strategy (`model.py:83`).
_LOCATION_RE = re.compile(r"[A-Za-z0-9_./]+\.py:\d+")

#: A tiny spec reused across groups (concrete 3-vector of float32).
_SPEC3 = evoxc.TensorSpec((3,), evoxc.float32)


# ---------------------------------------------------------------------------
# Shared helpers / tiny graph definitions
# ---------------------------------------------------------------------------


def _serialize(graph_or_module) -> dict:
    """The verified, self-describing `ir.serialize_module` payload."""
    module = getattr(graph_or_module, "module", graph_or_module)
    return evoxc.ir.serialize_module(module)


def _op_names(graph) -> list:
    """Op names of the entry function's entry block, in program order."""
    return [op.name for op in graph.module.main.entry_block.ops]


def _capture_symbolic() -> evoxc.SymbolicTensor:
    """Trace a tiny defn and return the SymbolicTensor it produced."""
    captured = {}

    @evoxc.defn
    def f(x):
        captured["out"] = evoxc.add(x, 1.0)
        return captured["out"]

    evoxc.trace(f, _SPEC3)
    return captured["out"]


@evoxc.defn
def _add_one(x):
    """`x + 1.0` — used by serialization / python-semantics / collective tests."""
    return evoxc.add(x, 1.0)


@evoxc.defn
def _mul_add(x):
    """`2.0 * x + 1.0` — used by the vmap≡vectorize sugar tests."""
    return evoxc.add(evoxc.multiply(x, 2.0), 1.0)


@evoxc.defn
def _dot(x, w):
    """`dot(x, w)` — used by staging / bind tests."""
    return evoxc.dot(x, w)


def _dot_pipeline(names=("x", "w")):
    """trace→lower→compile→load a named `dot(x, w)` pipeline.

    Returns `(defn, CompiledArtifact, Executable)`.
    """
    x_spec = evoxc.TensorSpec((2, 3), evoxc.float32, name=names[0])
    w_spec = evoxc.TensorSpec((3, 4), evoxc.float32, name=names[1])
    artifact = evoxc.compile(evoxc.lower(evoxc.trace(_dot, x_spec, w_spec)))
    return _dot, artifact, evoxc.load(artifact)


def _dot_inputs():
    """Concrete inputs for `_dot_pipeline`: x (2,3) and w (3,4) float32."""
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    w = np.full((3, 4), 0.5, dtype=np.float32)
    return x, w


#: Tiny COO spec: (3, 4) float32 — the shared sparse trace-input spec.
_SPARSE_SPEC = evoxc.sparse.SparseTensorSpec(
    evoxc.TensorSpec((None, 2), evoxc.int64),
    evoxc.TensorSpec((None,), evoxc.float32),
    dense_shape=(3, 4),
    format="coo",
)


def _sparse_coo():
    """A tiny concrete COO (3, 4) float32 with two stored entries."""
    indices = np.array([[0, 1], [2, 3]], dtype=np.int64)
    values = np.array([1.5, -2.5], dtype=np.float32)
    return evoxc.sparse.coo(indices, values, (3, 4))


def _sparse_csr():
    """A tiny concrete CSR (3, 4) float32 storing the same entries as
    `_sparse_coo` — used for the format static-leaf mismatch."""
    indptr = np.array([0, 1, 2, 2], dtype=np.int64)
    indices = np.array([1, 3], dtype=np.int64)
    values = np.array([1.5, -2.5], dtype=np.float32)
    return evoxc.sparse.csr(indptr, indices, values, (3, 4))


@evoxc.defn
def _sparse_to_dense(x):
    """`evoxc.sparse.to_dense(x)` — used by the sparse sugar / static-leaf
    / vjp-deferral tests."""
    return evoxc.sparse.to_dense(x)


def _capture_symbolic_sparse():
    """Trace a tiny defn and return the symbolic sparse it produced."""
    captured = {}

    @evoxc.defn
    def f(x):
        captured["out"] = evoxc.sparse.negate(x)
        return evoxc.sparse.to_dense(captured["out"])

    evoxc.trace(f, _SPARSE_SPEC)
    return captured["out"]


# ===========================================================================
# 1. @evoxc.defn is NOT JIT/eager — no implicit tracing, no mode switching
# ===========================================================================


class TestNoImplicitTracing:
    def test_defn_direct_call_raises(self):
        """Calling a Defn with concrete tensors raises TraceError directing
        to evoxc.trace / evoxc.evaluate (principle: explicit staging only)."""

        @evoxc.defn
        def f(x):
            return evoxc.add(x, 1.0)

        with pytest.raises(
            evoxc.TraceError,
            match=r"`evoxc\.trace\(defn, \*specs\)`.*`evoxc\.evaluate\(defn, \*args\)`",
        ):
            f(evoxc.tensor([1.0, 2.0]))

    def test_defn_call_raises_even_with_specs(self):
        """A Defn never silently traces either — even TensorSpec inputs fail."""

        @evoxc.defn
        def f(x):
            return evoxc.add(x, 1.0)

        with pytest.raises(evoxc.TraceError, match="evoxc\\.trace"):
            f(_SPEC3)

    def test_ops_raise_outside_any_trace(self):
        """Ops on concrete tensors outside a trace fail — there is no eager
        mode to fall back into."""

        with pytest.raises(evoxc.TraceError, match="No active trace"):
            evoxc.add(evoxc.tensor([1.0]), evoxc.tensor([2.0]))

        # ... even with Python scalars only: ops are graph-only operations.
        with pytest.raises(evoxc.TraceError, match="No active trace"):
            evoxc.add(1.0, 2.0)

    def test_no_eager_fallback_after_tracing(self):
        """Tracing a function once does not give it a hidden eager mode: the
        same function called outside a trace still fails loudly."""

        def plain(x):
            return evoxc.add(x, 1.0)

        evoxc.trace(plain, _SPEC3)  # tracing works...

        with pytest.raises(evoxc.TraceError, match="No active trace"):
            plain(evoxc.tensor([1.0, 2.0, 3.0]))  # ...direct calls do not execute

    def test_tracing_yields_symbolic_tensors(self):
        """Graph values come from tracing and are SymbolicTensor, not Tensor."""
        captured = {}

        @evoxc.defn
        def f(x):
            captured["y"] = evoxc.add(x, 1.0)
            return captured["y"]

        graph = evoxc.trace(f, _SPEC3)
        assert isinstance(graph, evoxc.Graph)
        assert isinstance(captured["y"], evoxc.SymbolicTensor)
        assert not isinstance(captured["y"], evoxc.Tensor)


# ===========================================================================
# 2. Closure capture: captured Tensors are errors, evoxc.constant opts in
# ===========================================================================


class TestClosureCapture:
    def test_closure_captured_tensor_raises(self):
        """A concrete Tensor captured from a closure is a TraceError whose
        message names all three sanctioned paths: explicit input, evoxc.constant,
        evoxc.evaluate."""

        w = evoxc.tensor([1.0, 2.0, 3.0])

        @evoxc.defn
        def f(x):
            return evoxc.add(x, w)

        with pytest.raises(
            evoxc.TraceError,
            match=r"explicit input.*evoxc\.constant.*evoxc\.evaluate",
        ):
            evoxc.trace(f, _SPEC3)

    def test_constant_is_the_explicit_opt_in(self):
        """`evoxc.constant(w)` embeds the data explicitly and the graph runs."""
        w = evoxc.tensor([1.0, 2.0, 3.0])

        @evoxc.defn
        def f(x):
            return evoxc.add(x, evoxc.constant(w))

        y = evoxc.evaluate(f, evoxc.tensor([10.0, 20.0, 30.0]))
        np.testing.assert_array_equal(y.numpy(), [11.0, 22.0, 33.0])

    def test_constant_snapshots_data(self):
        """`evoxc.constant` snapshots the buffer at graph-construction time;
        mutating the source tensor afterwards must not change the graph."""
        w = evoxc.tensor([1.0, 2.0, 3.0], dtype=evoxc.float32)

        @evoxc.defn
        def f(x):
            return evoxc.add(x, evoxc.constant(w))

        graph = evoxc.trace(f, _SPEC3)
        w.numpy()[0] = 100.0  # mutate AFTER the graph was built

        exe = evoxc.load(evoxc.compile(evoxc.lower(graph)))
        y = evoxc.run(exe, np.array([10.0, 20.0, 30.0], dtype=np.float32))
        np.testing.assert_array_equal(y.numpy(), [11.0, 22.0, 33.0])

    def test_constant_warns_above_threshold(self, monkeypatch):
        """Embedding data above `EVOXC_LARGE_CONSTANT_BYTES` warns (UserWarning).

        The threshold is documented as env-tunable via `EVOXC_LARGE_CONSTANT_BYTES`;
        it is read once at import time (evoxc/ops/constant.py), so the test sets
        the env var AND refreshes the cached module constant.
        """
        monkeypatch.setenv("EVOXC_LARGE_CONSTANT_BYTES", "16")
        const_mod = importlib.import_module("evoxc.ops.constant")
        monkeypatch.setattr(const_mod, "EVOXC_LARGE_CONSTANT_BYTES", 16)

        @evoxc.defn
        def big(x):
            # 16 float32 = 64 bytes > 16-byte threshold
            return evoxc.add(x, evoxc.constant(evoxc.ones((16,), dtype=evoxc.float32)))

        with pytest.warns(UserWarning, match="EVOXC_LARGE_CONSTANT_BYTES"):
            evoxc.trace(big, evoxc.TensorSpec((16,), evoxc.float32))

    def test_small_constant_does_not_warn(self, monkeypatch):
        """Constants at or below the threshold embed silently."""
        monkeypatch.setenv("EVOXC_LARGE_CONSTANT_BYTES", "16")
        const_mod = importlib.import_module("evoxc.ops.constant")
        monkeypatch.setattr(const_mod, "EVOXC_LARGE_CONSTANT_BYTES", 16)

        @evoxc.defn
        def small(x):
            # 2 float32 = 8 bytes <= 16-byte threshold
            return evoxc.add(x, evoxc.constant(evoxc.ones((2,), dtype=evoxc.float32)))

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            evoxc.trace(small, evoxc.TensorSpec((2,), evoxc.float32))
        assert not [w for w in recorded if issubclass(w.category, UserWarning)]


# ===========================================================================
# 3. SymbolicTensor purity: no concrete-data escape hatches
# ===========================================================================


class TestSymbolicPurity:
    def test_no_concrete_interop_attributes(self):
        """A SymbolicTensor must never be mistaken for concrete data: no
        `.numpy()`, no DLPack, no array protocol, no raw data pointer."""
        symbolic = _capture_symbolic()
        for attr in ("numpy", "data_ptr", "__dlpack__", "__array__", "item"):
            assert not hasattr(symbolic, attr), f"SymbolicTensor must not expose {attr!r}"

    def test_bool_raises_trace_error(self):
        """`bool(symbolic)` raises TraceError directing to evoxc.cond/while_loop/
        scan (principle 4: runtime control flow is explicit)."""
        symbolic = _capture_symbolic()
        with pytest.raises(evoxc.TraceError, match="evoxc\\.cond"):
            bool(symbolic)


# ===========================================================================
# 4. Staging explicitness: distinct types, shorthand = documented composition
# ===========================================================================


class TestStagingExplicitness:
    def test_pipeline_types_are_distinct(self):
        """Defn / Graph / LoweredProgram / CompiledArtifact / Executable are
        five distinct public types; nothing silently morphs across stages."""
        graph = evoxc.trace(_add_one, _SPEC3)
        lowered = evoxc.lower(graph)
        artifact = evoxc.compile(lowered)
        executable = evoxc.load(artifact)

        stages = [evoxc.Defn, graph, lowered, artifact, executable]
        assert len({type(stage) for stage in stages}) == 5

        assert isinstance(graph, evoxc.Graph)
        assert not isinstance(graph, (evoxc.LoweredProgram, evoxc.CompiledArtifact, evoxc.Executable))
        assert isinstance(lowered, evoxc.LoweredProgram)
        assert not isinstance(lowered, (evoxc.Graph, evoxc.CompiledArtifact, evoxc.Executable))
        assert isinstance(artifact, evoxc.CompiledArtifact)
        assert not isinstance(artifact, (evoxc.Graph, evoxc.LoweredProgram, evoxc.Executable))
        assert isinstance(executable, evoxc.Executable)
        assert not isinstance(executable, (evoxc.Graph, evoxc.LoweredProgram, evoxc.CompiledArtifact))

    def test_wrong_stage_objects_rejected(self):
        """Passing a Graph where a later stage is expected raises a clear
        TypeError — stages are never silently consumed (actual behavior:
        compile/load/run all raise TypeError naming the expected type)."""
        graph = evoxc.trace(_add_one, _SPEC3)

        with pytest.raises(TypeError, match="LoweredProgram"):
            evoxc.compile(graph)

        with pytest.raises(TypeError, match="CompiledArtifact"):
            evoxc.load(graph)

        with pytest.raises(TypeError, match="Executable"):
            evoxc.run(graph)

    def test_build_docstring_documents_expansion(self):
        """`evoxc.build`'s docstring states its exact expansion (trace→lower→
        compile→load) — shorthand must be documented composition only."""
        doc = evoxc.build.__doc__
        assert doc, "evoxc.build must carry a docstring documenting its expansion"
        for stage in ("trace", "lower", "compile", "load"):
            assert stage in doc

    def test_evaluate_docstring_documents_expansion(self):
        """`evoxc.evaluate`'s docstring states its exact expansion (derive specs
        → build → run)."""
        doc = evoxc.evaluate.__doc__
        assert doc, "evoxc.evaluate must carry a docstring documenting its expansion"
        for stage in ("build", "run"):
            assert stage in doc

    def test_build_equals_explicit_pipeline(self):
        """`evoxc.build(f, *specs)` produces the same outputs as the explicit
        trace→lower→compile→load→run pipeline."""
        _, _, executable = _dot_pipeline()
        x, w = _dot_inputs()

        graph = evoxc.trace(_dot, evoxc.TensorSpec((2, 3), evoxc.float32),
                          evoxc.TensorSpec((3, 4), evoxc.float32))
        explicit = evoxc.load(evoxc.compile(evoxc.lower(graph)))

        y_short = evoxc.run(executable, x, w)
        y_long = evoxc.run(explicit, x, w)
        np.testing.assert_array_equal(y_short.numpy(), y_long.numpy())

    def test_evaluate_equals_explicit_pipeline(self):
        """`evoxc.evaluate(f, *tensors)` equals deriving specs from the tensors
        (shape + dtype only) then building and running."""
        x, w = _dot_inputs()
        y_short = evoxc.evaluate(_dot, x, w)

        graph = evoxc.trace(
            _dot,
            evoxc.TensorSpec(shape=x.shape, dtype=x.dtype),
            evoxc.TensorSpec(shape=w.shape, dtype=w.dtype),
        )
        y_long = evoxc.run(evoxc.load(evoxc.compile(evoxc.lower(graph))), x, w)
        np.testing.assert_array_equal(y_short.numpy(), y_long.numpy())

    def test_build_cached_docstring_documents_expansion(self):
        """`evoxc.build_cached`'s docstring documents its exact expansion — a
        MEMOIZED `evoxc.build` (the trace→lower→compile→load composition runs
        at most once per distinct key): documented shorthand, no other
        behavior — never an eager fallback, no cache=False escape."""
        from evoxc.build_cache import build_cached

        doc = build_cached.__doc__
        assert doc, ("build_cached must carry a docstring documenting its "
                     "memoized-build expansion")
        for stage in ("trace", "lower", "compile", "load"):
            assert stage in doc
        assert "MEMOIZED" in doc
        assert "no other behavior" in doc

    def test_build_cached_equals_explicit_pipeline(self):
        """`build_cached(f, *specs)` returns a NORMAL Executable — the same
        program the explicit trace→lower→compile→load→run composition
        produces (a memoized build, nothing more)."""
        from evoxc.build_cache import CompileCache, build_cached

        x_spec = evoxc.TensorSpec((2, 3), evoxc.float32)
        w_spec = evoxc.TensorSpec((3, 4), evoxc.float32)
        cached = build_cached(_dot, x_spec, w_spec, cache=CompileCache())
        assert isinstance(cached, evoxc.Executable)

        graph = evoxc.trace(_dot, x_spec, w_spec)
        explicit = evoxc.load(evoxc.compile(evoxc.lower(graph)))
        x, w = _dot_inputs()
        np.testing.assert_array_equal(
            evoxc.run(cached, x, w).numpy(),
            evoxc.run(explicit, x, w).numpy(),
        )


# ===========================================================================
# 5. evoxc.bind is argument-passing sugar — never alters the graph
# ===========================================================================


class TestBindIsSugar:
    def test_bind_never_alters_the_graph(self):
        """Binding must not change the compiled program. The numpy
        CompiledArtifact payload IS the serialized IR module, so deep-equal
        payloads before/after bind prove the graph is untouched."""
        _, artifact, executable = _dot_pipeline()

        before = copy.deepcopy(artifact.payload)
        bound = evoxc.bind(executable, w=evoxc.ones((3, 4), dtype=evoxc.float32))
        after = copy.deepcopy(artifact.payload)

        assert before == after
        # bind wraps the SAME executable, just pre-supplying arguments.
        assert bound.executable is executable

    def test_bind_run_equals_direct_run(self):
        """`run(bind(exe, w=w), x)` equals `run(exe, x, w)`."""
        _, _, executable = _dot_pipeline()
        x, w = _dot_inputs()

        bound = evoxc.bind(executable, w=evoxc.from_numpy(w))
        y_bound = evoxc.run(bound, x)
        y_direct = evoxc.run(executable, x, w)
        np.testing.assert_array_equal(y_bound.numpy(), y_direct.numpy())

    def test_bind_validates_names(self):
        """Binding an unknown input name fails clearly (no silent guessing)."""
        _, _, executable = _dot_pipeline()
        with pytest.raises(evoxc.TraceError, match="unknown input name"):
            evoxc.bind(executable, z=evoxc.ones((3, 4), dtype=evoxc.float32))


# ===========================================================================
# 6. vmap is transparent sugar over vectorize
# ===========================================================================


class TestVmapIsSugar:
    def test_vmap_fn_equals_vectorize_of_trace(self):
        """`vmap(f, in_axes, out_axes=0)` applied to specs produces IR equal
        to `vectorize(trace(f, *stripped_specs), in_axes)`.

        No naming normalization is needed: both paths go through `evoxc.trace`,
        which names the entry function "main", and with `out_axes=0` the
        vmap output-rearrangement step adds no ops.
        """
        batched = evoxc.TensorSpec((4, 5), evoxc.float32)
        via_vmap = evoxc.vmap(_mul_add, in_axes=0, out_axes=0)(batched)

        unbatched = evoxc.TensorSpec((5,), evoxc.float32)  # leading dim stripped
        via_vectorize = evoxc.vectorize(evoxc.trace(_mul_add, unbatched), 0)

        assert _serialize(via_vmap) == _serialize(via_vectorize)

    def test_vmap_graph_equals_vectorize(self):
        """`vmap(graph, in_axes=0, out_axes=0)` ≡ `vectorize(graph, 0)`."""
        graph = evoxc.trace(_mul_add, evoxc.TensorSpec((5,), evoxc.float32))
        via_vmap = evoxc.vmap(graph, in_axes=0, out_axes=0)
        via_vectorize = evoxc.vectorize(graph, 0)
        assert _serialize(via_vmap) == _serialize(via_vectorize)

    def test_vmap_callable_rejects_concrete_tensors(self):
        """Transforms build graphs and never execute: calling a `vmap(f)`
        callable with concrete tensors raises TraceError."""
        with pytest.raises(evoxc.TraceError, match="never execute"):
            evoxc.vmap(_mul_add)(evoxc.tensor(np.ones((4, 5), dtype=np.float32)))

    def test_vectorized_graph_runs_on_numpy_backend(self):
        """The sugar produces an ordinary graph of ordinary ops that the
        numpy backend runs without any special vectorize support."""
        batched = evoxc.TensorSpec((4, 5), evoxc.float32)
        graph = evoxc.vmap(_mul_add, in_axes=0, out_axes=0)(batched)
        exe = evoxc.load(evoxc.compile(evoxc.lower(graph)))

        x = np.arange(20, dtype=np.float32).reshape(4, 5)
        y = evoxc.run(exe, x)
        np.testing.assert_array_equal(y.numpy(), x * 2.0 + 1.0)


# ===========================================================================
# 7. Concrete creators return concrete Tensors (DLPack interop)
# ===========================================================================


class TestConcreteCreators:
    @pytest.mark.parametrize(
        "make",
        [
            pytest.param(lambda: evoxc.zeros((2, 3)), id="zeros"),
            pytest.param(lambda: evoxc.ones((2,)), id="ones"),
            pytest.param(lambda: evoxc.full((2,), 7.0), id="full"),
            pytest.param(lambda: evoxc.empty((2,)), id="empty"),
            pytest.param(lambda: evoxc.tensor([1, 2, 3]), id="tensor"),
            pytest.param(lambda: evoxc.from_numpy(np.arange(4)), id="from_numpy"),
        ],
    )
    def test_creators_return_concrete_tensors(self, make):
        """Every concrete creator returns a materialized Tensor with
        `.numpy()`, dtype, shape, device and DLPack export — never a
        SymbolicTensor."""
        t = make()
        assert isinstance(t, evoxc.Tensor)
        assert not isinstance(t, evoxc.SymbolicTensor)
        assert callable(t.numpy)
        assert callable(t.__dlpack__)
        assert isinstance(t.dtype, np.dtype)
        assert isinstance(t.shape, tuple)
        assert t.device.kind == "cpu"

    def test_creator_values(self):
        """Creators produce the documented numpy-convention values."""
        np.testing.assert_array_equal(evoxc.zeros((2, 2)).numpy(), np.zeros((2, 2)))
        np.testing.assert_array_equal(evoxc.ones((3,)).numpy(), np.ones((3,)))
        np.testing.assert_array_equal(evoxc.full((2,), 7.0).numpy(), np.full((2,), 7.0))
        np.testing.assert_array_equal(evoxc.tensor([1, 2, 3]).numpy(), np.array([1, 2, 3]))

    def test_dlpack_roundtrip(self):
        """`evoxc.from_dlpack(evoxc.tensor(...))` round-trips values unchanged."""
        original = evoxc.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=evoxc.float32)
        imported = evoxc.from_dlpack(original)
        assert isinstance(imported, evoxc.Tensor)
        np.testing.assert_array_equal(imported.numpy(), original.numpy())

    def test_torch_interop(self):
        """torch tensor → evoxc.from_dlpack and evoxc Tensor → torch
        (via torch.utils.dlpack.from_dlpack) both preserve values."""
        torch = pytest.importorskip("torch")

        t_torch = torch.arange(6, dtype=torch.float32).reshape(2, 3)

        as_evoxc = evoxc.from_dlpack(t_torch)
        assert isinstance(as_evoxc, evoxc.Tensor)
        np.testing.assert_array_equal(as_evoxc.numpy(), t_torch.numpy())

        back = torch.utils.dlpack.from_dlpack(evoxc.tensor(t_torch.numpy()))
        np.testing.assert_array_equal(back.numpy(), t_torch.numpy())


# ===========================================================================
# 8. evoxc.numpy (enp) creation ops are GRAPH ops
# ===========================================================================


class TestEnpGraphOps:
    def test_enp_zeros_builds_a_constant_op(self):
        """`evoxc.numpy.zeros` inside a defn builds an IR op (the same
        `constant` op kind `evoxc.constant` builds) — it does not leak a
        concrete Tensor into the graph."""
        @evoxc.defn
        def f(x):
            return evoxc.add(x, evoxc.numpy.zeros((3,)))

        graph = evoxc.trace(f, _SPEC3)
        names = _op_names(graph)
        assert "constant" in names

        constants = [op for op in graph.module.main.entry_block.ops if op.name == "constant"]
        assert constants, "enp.zeros must be materialized as a graph op"
        # The embedded payload is a plain numpy array, never an evoxc.Tensor.
        for op in constants:
            assert isinstance(op.attributes["value"], np.ndarray)
            assert not isinstance(op.attributes["value"], evoxc.Tensor)

    def test_enp_zeros_has_no_eager_fallback(self):
        """Outside a trace, `evoxc.numpy.zeros` raises — there is no silent
        concrete creation path (it is a graph op)."""
        with pytest.raises(evoxc.TraceError, match="No active trace"):
            evoxc.numpy.zeros((3,))

    def test_enp_zeros_evaluates(self):
        """The graph op executes with correct values."""
        @evoxc.defn
        def f(x):
            return evoxc.add(x, evoxc.numpy.zeros((3,)))

        y = evoxc.evaluate(f, evoxc.tensor([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(y.numpy(), [1.0, 2.0, 3.0])


# ===========================================================================
# 9. Serialization round-trips (Graph / LoweredProgram / CompiledArtifact)
# ===========================================================================


def _corrupt(path, kind):
    """Corrupt a saved container in place: flip a payload byte or truncate."""
    data = path.read_bytes()
    if kind == "flip-byte":
        mutated = bytearray(data)
        mutated[len(mutated) // 2] ^= 0xFF
        path.write_bytes(bytes(mutated))
    elif kind == "truncate":
        path.write_bytes(data[: max(1, len(data) // 2)])
    else:
        raise ValueError(f"unknown corruption kind: {kind}")


class TestSerializationRoundtrips:
    def test_graph_roundtrip_lowers(self, tmp_path):
        """A loaded Graph is a working graph: it can be lowered and run."""
        graph = evoxc.trace(_add_one, _SPEC3)
        path = tmp_path / "g.evoxcgraph"
        graph.save(str(path))

        loaded = evoxc.Graph.load(str(path))
        assert isinstance(loaded, evoxc.Graph)

        exe = evoxc.load(evoxc.compile(evoxc.lower(loaded)))
        y = evoxc.run(exe, np.ones(3, dtype=np.float32))
        np.testing.assert_array_equal(y.numpy(), [2.0, 2.0, 2.0])

    def test_lowered_roundtrip_compiles(self, tmp_path):
        """A loaded LoweredProgram compiles and runs; never re-lowered."""
        lowered = evoxc.lower(evoxc.trace(_add_one, _SPEC3))
        path = tmp_path / "p.evoxc"
        lowered.save(str(path))

        loaded = evoxc.LoweredProgram.load(str(path))
        assert isinstance(loaded, evoxc.LoweredProgram)
        assert loaded.backend == "numpy"

        exe = evoxc.load(evoxc.compile(loaded))
        y = evoxc.run(exe, np.ones(3, dtype=np.float32))
        np.testing.assert_array_equal(y.numpy(), [2.0, 2.0, 2.0])

    def test_artifact_roundtrip_runs(self, tmp_path):
        """A loaded CompiledArtifact loads into a runnable executable."""
        artifact = evoxc.compile(evoxc.lower(evoxc.trace(_add_one, _SPEC3)))
        path = tmp_path / "a.evoxcartifact"
        artifact.save(str(path))

        loaded = evoxc.CompiledArtifact.load(str(path))
        assert isinstance(loaded, evoxc.CompiledArtifact)
        assert loaded.backend == "numpy"

        exe = evoxc.load(loaded)
        y = evoxc.run(exe, np.ones(3, dtype=np.float32))
        np.testing.assert_array_equal(y.numpy(), [2.0, 2.0, 2.0])

    @pytest.mark.parametrize("kind", ["flip-byte", "truncate"])
    def test_corrupt_graph_file_fails(self, tmp_path, kind):
        """Byte-flipped or truncated containers fail with PersistenceError —
        never a silent reinterpretation."""
        graph = evoxc.trace(_add_one, _SPEC3)
        path = tmp_path / "g.evoxcgraph"
        graph.save(str(path))
        _corrupt(path, kind)

        with pytest.raises(evoxc.PersistenceError):
            evoxc.Graph.load(str(path))

    @pytest.mark.parametrize("kind", ["flip-byte", "truncate"])
    def test_corrupt_lowered_file_fails(self, tmp_path, kind):
        lowered = evoxc.lower(evoxc.trace(_add_one, _SPEC3))
        path = tmp_path / "p.evoxc"
        lowered.save(str(path))
        _corrupt(path, kind)

        with pytest.raises(evoxc.PersistenceError):
            evoxc.LoweredProgram.load(str(path))

    @pytest.mark.parametrize("kind", ["flip-byte", "truncate"])
    def test_corrupt_artifact_file_fails(self, tmp_path, kind):
        artifact = evoxc.compile(evoxc.lower(evoxc.trace(_add_one, _SPEC3)))
        path = tmp_path / "a.evoxcartifact"
        artifact.save(str(path))
        _corrupt(path, kind)

        with pytest.raises(evoxc.PersistenceError):
            evoxc.CompiledArtifact.load(str(path))


# ===========================================================================
# 10. Python value → Python semantics (no silent specialization)
# ===========================================================================


class TestPythonSemantics:
    def test_python_values_stay_python(self):
        """Plain Python containers and scalars behave exactly like Python —
        the graph is never involved implicitly."""
        values = [1, 2, 3]
        assert sum(values) == 6
        assert [v * 2 for v in values] == [2, 4, 6]
        assert min(values) == 1 and max(values) == 3

    def test_static_values_specialize_the_graph(self):
        """A static Python value specializes the graph at trace time: two
        different values produce two different graphs, and run validates the
        value instead of silently recompiling."""

        @evoxc.defn
        def f(x, scale):  # scale: static Python value
            if scale > 0:
                return evoxc.multiply(x, scale)
            return evoxc.negate(x)

        g_positive = evoxc.trace(f, _SPEC3, 2.0)
        g_negative = evoxc.trace(f, _SPEC3, -1.0)
        assert _serialize(g_positive) != _serialize(g_negative)

        exe_pos = evoxc.load(evoxc.compile(evoxc.lower(g_positive)))
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = evoxc.run(exe_pos, x, 2.0)
        np.testing.assert_array_equal(y.numpy(), [2.0, 4.0, 6.0])

        exe_neg = evoxc.load(evoxc.compile(evoxc.lower(g_negative)))
        y = evoxc.run(exe_neg, x, -1.0)
        np.testing.assert_array_equal(y.numpy(), [-1.0, -2.0, -3.0])

        # Changing the static value at run time is a new graph — fail clearly.
        with pytest.raises(evoxc.TraceError, match="specialized on"):
            evoxc.run(exe_pos, x, 3.0)

    def test_symbolic_if_fails_at_trace_time(self):
        """`if evoxc.sum(x) > 0:` over a graph value fails at trace time with a
        clear error — runtime control flow is `evoxc.cond` (principle 4: no
        silent specialization on tensor data)."""

        @evoxc.defn
        def f(x):
            if evoxc.sum(x) > 0:
                return x
            return evoxc.negate(x)

        with pytest.raises(evoxc.TraceError, match="evoxc\\.cond"):
            evoxc.trace(f, _SPEC3)


# ===========================================================================
# 11. Local tensors + explicit communication
# ===========================================================================


class TestLocalTensorSemantics:
    def test_collectives_appear_explicitly_in_ir(self):
        """`evoxc.dist.all_reduce` appears as an explicit `all_reduce` op with
        effect `collective` in the serialized module — collectives are part of
        the program, never implicit."""

        @evoxc.defn
        def f(x):
            return evoxc.dist.all_reduce(x, op="sum")

        graph = evoxc.trace(f, evoxc.TensorSpec((4,), evoxc.float32))
        payload = _serialize(graph)

        op_names = [op["name"] for op in payload["ops"]]
        assert "all_reduce" in op_names
        assert evoxc.ir.opdef("all_reduce").effect == "collective"

    def test_plain_defn_has_no_collective_ops(self):
        """A defn without collectives contains NO collective-effect ops — the
        compiler/tracer never invents communication."""
        graph = evoxc.trace(_add_one, _SPEC3)
        payload = _serialize(graph)
        assert all(
            evoxc.ir.opdef(op["name"]).effect != "collective" for op in payload["ops"]
        )

    def test_all_reduce_runs_with_default_executor(self):
        """The reference backend's default single-rank executor is the
        identity: all_reduce on one rank returns the local tensor."""
        @evoxc.defn
        def f(x):
            return evoxc.dist.all_reduce(x, op="sum")

        y = evoxc.evaluate(f, evoxc.tensor([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_array_equal(y.numpy(), [1.0, 2.0, 3.0, 4.0])


# ===========================================================================
# 12. Error messages include source locations (`model.py:83`)
# ===========================================================================


class TestErrorLocations:
    def test_bool_error_includes_source_location(self):
        """The TraceError from using a symbolic as a Python boolean names the
        op's source location."""
        symbolic = _capture_symbolic()
        with pytest.raises(evoxc.TraceError, match=_LOCATION_RE) as excinfo:
            bool(symbolic)
        assert "test_spec_compliance.py" in str(excinfo.value)

    def test_invalid_trace_output_error_includes_location(self):
        """Returning a concrete tensor from a traced function fails with the
        trace location in the message."""

        @evoxc.defn
        def f(x):
            return evoxc.tensor([1.0, 2.0, 3.0])

        with pytest.raises(evoxc.TraceError, match=_LOCATION_RE) as excinfo:
            evoxc.trace(f, _SPEC3)
        assert "test_spec_compliance.py" in str(excinfo.value)

    @pytest.mark.parametrize("kind", ["add", "dot"])
    def test_shape_errors_include_source_location(self, kind):
        if kind == "add":
            @evoxc.defn
            def fn(a, b):
                return evoxc.add(a, b)

            specs = (evoxc.TensorSpec((2,), evoxc.float32), evoxc.TensorSpec((3,), evoxc.float32))
        else:
            @evoxc.defn
            def fn(a, b):
                return evoxc.dot(a, b)

            specs = (evoxc.TensorSpec((2, 3), evoxc.float32), evoxc.TensorSpec((4, 2), evoxc.float32))

        with pytest.raises(evoxc.ShapeError) as excinfo:
            evoxc.trace(fn, *specs)
        message = str(excinfo.value)
        assert _LOCATION_RE.search(message), (
            f"shape error must carry a source location like `model.py:83`, got: {message}"
        )


# ===========================================================================
# 13. tree_map is transparent sugar over flatten/unflatten
# ===========================================================================


_Point = namedtuple("_Point", ["x", "y"])


@dataclasses.dataclass(frozen=True)
class _Box:
    """A user-defined dataclass pytree node with two scalar fields."""

    lo: object
    hi: object


#: Nested mixed structure: dict → tuple / list / namedtuple / dataclass / leaf.
_MIXED_TREE = {
    "pair": (1, 2),
    "seq": [3, 4],
    "point": _Point(5, 6),
    "box": _Box(7, 8),
    "scalar": 9,
}


def _double_leaf(leaf):
    """Type-preserving leaf transform (int → int), so `tree_map`'s structure
    is unaffected whether `tree_structure` records leaf types or not."""
    return leaf * 2


class TestTreeMapIsComposition:
    """`tree_map` is exactly the documented composition — ``flatten``, map
    the leaves, ``unflatten`` — and preserves the input structure exactly."""

    def test_tree_map_equals_flatten_map_unflatten(self):
        """`tree_map(f, t) == unflatten([f(l) for l in flatten(t)[0]],
        flatten(t)[1])`: mapping a nested mixed structure is precisely
        leaf-mapping plus rebuild — sugar over the explicit primitives, never
        new semantics."""

        leaves, spec = evoxc.flatten(_MIXED_TREE)
        expected = evoxc.unflatten([_double_leaf(leaf) for leaf in leaves], spec)
        assert evoxc.tree_map(_double_leaf, _MIXED_TREE) == expected

    def test_tree_map_preserves_structure(self):
        """`tree_structure(tree_map(f, t)) == tree_structure(t)`: the mapped
        tree keeps the exact container skeleton — dict keys, tuple/list
        arity, namedtuple fields, dataclass fields."""

        assert evoxc.tree_structure(
            evoxc.tree_map(_double_leaf, _MIXED_TREE)
        ) == evoxc.tree_structure(_MIXED_TREE)


# ===========================================================================
# 14. Sparse tensors: the same explicitness discipline, in all three phases
# ===========================================================================


class TestSparseExplicitness:
    """Sparse tensors (evoxc.sparse) obey the SAME explicit-staging discipline
    as the dense core (sparse/CONTEXT.md is the authoritative contract):
    graph-time computation ops with no eager mode, polymorphic creators
    (concrete -> eager value, symbolic -> in-graph assembly), pure symbolic
    leaves, dense_shape/dtype/format as static snapshotted leaves, and
    explicit v1 deferrals — never a silent fallback."""

    def test_ops_raise_outside_any_trace(self):
        """Sparse computation ops called outside a trace raise TraceError —
        there is no eager mode to fall back into (same as dense ops)."""
        s = _sparse_coo()
        calls = [
            (evoxc.sparse.add, (s, s)),
            (evoxc.sparse.subtract, (s, s)),
            (evoxc.sparse.multiply, (s, s)),
            (evoxc.sparse.multiply_dense, (s, s)),  # builder check precedes operands
            (evoxc.sparse.negate, (s,)),
            (evoxc.sparse.sum, (s,)),
            (evoxc.sparse.transpose, (s,)),
            (evoxc.sparse.reshape, (s, (12,))),
            (evoxc.sparse.concatenate, ([s, s],)),
        ]
        for op, args in calls:
            with pytest.raises(evoxc.TraceError, match="No active trace"):
                op(*args)

    def test_matmul_with_concrete_sparse_raises_outside_trace(self):
        """`sparse.matmul` normalizes its operands before the builder check,
        so concrete sparse operands outside a trace get the three-option
        TraceError (explicit input / evoxc.constant / evoxc.evaluate)."""
        s = _sparse_coo()
        with pytest.raises(
            evoxc.TraceError,
            match=r"explicit input.*evoxc\.constant.*evoxc\.evaluate",
        ):
            evoxc.sparse.matmul(s, s)

    def test_converters_need_a_trace_for_symbolic_inputs(self):
        """Converters are polymorphic: concrete -> eager method; a SYMBOLIC
        instance outside a trace raises TraceError (graph materialization is
        trace-time only). The one exception is the documented identity:
        `to_coo` on an already-COO value is a no-op."""
        symbolic = _capture_symbolic_sparse()
        for op in (evoxc.sparse.to_dense, evoxc.sparse.to_csr, evoxc.sparse.to_csc):
            with pytest.raises(evoxc.TraceError, match="No active trace"):
                op(symbolic)
        # COO -> COO is the documented identity (no op is emitted).
        assert evoxc.sparse.to_coo(symbolic) is symbolic

    def test_converters_are_eager_for_concrete_inputs(self):
        """Converters on CONCRETE instances are pure eager layout
        conversions — the polymorphic counterpart of the graph ops."""
        s = _sparse_coo()
        dense = evoxc.sparse.to_dense(s)
        assert isinstance(dense, np.ndarray)
        np.testing.assert_array_equal(
            dense,
            [[0.0, 1.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -2.5]],
        )
        csr = evoxc.sparse.to_csr(s)
        assert isinstance(csr, evoxc.sparse.CSRTensor)
        np.testing.assert_array_equal(evoxc.sparse.to_dense(csr), dense)

    def test_ops_build_ir_inside_a_trace(self):
        """Inside a trace the sparse ops build IR ops only — `sparse_add`,
        `sparse_dot_dense`, `sparse_to_dense`, `sparse_from_dense`, ... —
        never eager results."""

        @evoxc.defn
        def f(a, b, d, x):
            added = evoxc.sparse.add(a, evoxc.sparse.from_dense(x))
            return evoxc.add(evoxc.sparse.to_dense(added), evoxc.sparse.matmul(a, d))

        graph = evoxc.trace(
            f,
            _SPARSE_SPEC,
            _SPARSE_SPEC,
            evoxc.TensorSpec((4, 4), evoxc.float32),
            evoxc.TensorSpec((3, 4), evoxc.float32),
        )
        names = _op_names(graph)
        for op_name in (
            "sparse_from_dense",
            "sparse_add",
            "sparse_to_dense",
            "sparse_dot_dense",
        ):
            assert op_name in names, f"expected {op_name!r} in graph ops {names}"

        # Results are graph values, never eager data.
        out = graph.module.main.entry_block.ops[-2].results[0]
        assert isinstance(out, evoxc.ir.Value)

    def test_no_eager_numerical_ops(self):
        """Feeding CONCRETE sparse values to a computation op inside a trace
        raises the three-option TraceError — the ops never fall back to
        eager numerics."""
        s = _sparse_coo()

        @evoxc.defn
        def f(x):
            return evoxc.sparse.add(x, s)

        with pytest.raises(
            evoxc.TraceError,
            match=r"explicit input.*evoxc\.constant.*evoxc\.evaluate",
        ):
            evoxc.trace(f, _SPARSE_SPEC)

    def test_creators_concrete_components_are_eager(self):
        """Concrete components build a validated eager sparse value — the
        `.values` / `.indices` carry concrete data, canonical form is
        enforced, never silently."""
        s = _sparse_coo()
        assert isinstance(s, evoxc.sparse.SparseTensor)
        assert isinstance(s.values, np.ndarray)
        assert isinstance(s.indices, np.ndarray)
        np.testing.assert_array_equal(s.values, [1.5, -2.5])
        np.testing.assert_array_equal(s.indices, [[0, 1], [2, 3]])
        assert s.dtype == np.dtype(np.float32)
        assert s.dense_shape == (3, 4)
        assert s.format == "coo"

        # core.Tensor components are equally valid concrete components.
        t = evoxc.sparse.coo(
            evoxc.tensor(np.array([[0, 1]], dtype=np.int64)),
            evoxc.tensor(np.array([7.0], dtype=np.float32)),
            (3, 4),
        )
        assert isinstance(t.values, evoxc.Tensor)
        np.testing.assert_array_equal(t.to_dense()[0, 1], 7.0)

        # Canonical-form validation is eager (duplicate rows -> ShapeError).
        with pytest.raises(evoxc.ShapeError):
            evoxc.sparse.coo(
                np.array([[0, 1], [0, 1]], dtype=np.int64),
                np.array([1.0, 2.0]),
                (3, 4),
            )

    def test_creators_symbolic_components_are_graph_values(self):
        """Symbolic components inside a trace assemble the in-graph sparse
        value via from_parts — the children are SymbolicTensor (no data,
        no validation)."""

        @evoxc.defn
        def f(x):
            out = evoxc.sparse.coo(x.indices, x.values, (3, 4))
            return evoxc.sparse.to_dense(out)

        evoxc.trace(f, _SPARSE_SPEC)  # traces fine: in-graph assembly

    def test_creators_reject_tensorspec_components(self):
        """`core.TensorSpec` creator components raise TypeError directing to
        `SparseTensorSpec` for trace inputs."""
        with pytest.raises(TypeError, match="SparseTensorSpec"):
            evoxc.sparse.coo(
                evoxc.TensorSpec((None, 2), evoxc.int64),
                np.array([1.0, 2.0]),
                (3, 4),
            )

    def test_creators_reject_mixed_concrete_and_symbolic(self):
        """Mixed concrete+symbolic creator components raise TypeError — a
        creator is either eager assembly or in-graph assembly, never both."""
        with pytest.raises(TypeError, match="mixed kinds"):
            evoxc.sparse.coo(
                np.array([[0, 1]], dtype=np.int64),
                _capture_symbolic(),
                (3, 4),
            )

    def test_symbolic_sparse_leaves_have_no_concrete_escape_hatches(self):
        """The indices/values leaves of an in-graph sparse value are pure
        SymbolicTensor — no `.numpy()`, no DLPack, no array protocol
        (mirror of the dense check)."""
        symbolic = _capture_symbolic_sparse()
        assert isinstance(symbolic, evoxc.sparse.SparseTensor)
        for leaf in (symbolic.indices, symbolic.values):
            assert isinstance(leaf, evoxc.SymbolicTensor)
            for attr in ("numpy", "data_ptr", "__dlpack__", "__array__", "item"):
                assert not hasattr(leaf, attr), (
                    f"symbolic sparse leaf must not expose {attr!r}"
                )

    def test_no_sparse_constant(self):
        """`evoxc.constant` embeds concrete core.Tensor data only — a concrete
        sparse tensor is rejected (v1: no sparse constant)."""

        @evoxc.defn
        def f(x):
            return evoxc.sparse.to_dense(evoxc.constant(_sparse_coo()))

        with pytest.raises(evoxc.TraceError, match="got SparseTensor"):
            evoxc.trace(f, _SPARSE_SPEC)

    def test_closure_captured_sparse_raises(self):
        """A concrete sparse captured from a closure and fed to a computation
        op is a TraceError naming the three sanctioned paths — same as the
        dense closure-capture contract."""
        s = _sparse_coo()

        @evoxc.defn
        def f(x):
            return evoxc.sparse.add(x, s)

        with pytest.raises(
            evoxc.TraceError,
            match=r"explicit input.*evoxc\.constant.*evoxc\.evaluate",
        ):
            evoxc.trace(f, _SPARSE_SPEC)

    def test_static_leaves_are_snapshotted_and_run_validated(self):
        """dense_shape/dtype/format are static leaves: tracing with a
        SparseTensorSpec then RUNNING with a concrete sparse whose
        dense_shape (or format) differs fails loudly — the graph never
        silently adapts to a different sparse structure."""
        exe = evoxc.build(_sparse_to_dense, _SPARSE_SPEC)

        # dense_shape mismatch (3, 4) vs (3, 5).
        wrong_shape = evoxc.sparse.coo(
            np.array([[0, 1], [2, 3]], dtype=np.int64),
            np.array([1.5, -2.5], dtype=np.float32),
            (3, 5),
        )
        with pytest.raises(evoxc.TraceError, match="specialized on"):
            evoxc.run(exe, wrong_shape)

        # format mismatch: the coo-vs-csr leaf layouts diverge.
        with pytest.raises(evoxc.TraceError, match="structure does not match"):
            evoxc.run(exe, _sparse_csr())

    def test_evaluate_derives_sparse_spec_and_returns_concrete(self):
        """`evoxc.evaluate` with concrete sparse args derives a
        SparseTensorSpec via from_concrete and returns concrete sparse
        results — documented shorthand composition, no new semantics."""

        @evoxc.defn
        def f(x):
            return evoxc.sparse.negate(x)

        out = evoxc.evaluate(f, _sparse_coo())
        assert isinstance(out, evoxc.sparse.SparseTensor)
        assert not isinstance(out, evoxc.sparse.SparseTensorSpec)
        np.testing.assert_array_equal(out.values.numpy(), [-1.5, 2.5])
        np.testing.assert_array_equal(out.to_dense(), -_sparse_coo().to_dense())

        # The derived spec mirrors the concrete instance's static leaves.
        derived = evoxc.sparse.SparseTensorSpec.from_concrete(_sparse_coo())
        assert isinstance(derived, evoxc.sparse.SparseTensorSpec)
        assert derived.dense_shape == (3, 4)
        assert derived.format == "coo"
        assert derived.dtype == np.dtype(np.float32)

    def test_vmap_bare_axes_on_sparse_numerics(self):
        """`vmap(graph, in_axes=0)` with a bare int axis maps a sparse input
        (registered pytree node): a batched concrete sparse runs and the
        dense output matches the per-batch references."""
        graph = evoxc.vmap(evoxc.trace(_sparse_to_dense, _SPARSE_SPEC), in_axes=0)
        exe = evoxc.load(evoxc.compile(evoxc.lower(graph)))

        # Batched concrete COO: leading batch dim on the tensor leaves
        # (validation-free assembly — the per-element canonical checks are
        # the interpreter's job).
        batched = evoxc.sparse.SparseTensor.from_parts(
            np.array([[[0, 1], [2, 3]], [[0, 3], [1, 0]]], dtype=np.int64),
            np.array([[1.5, -2.5], [3.5, -1.5]], dtype=np.float32),
            dense_shape=(3, 4),
            format="coo",
        )
        out = evoxc.run(exe, batched).numpy()
        expected = np.zeros((2, 3, 4), dtype=np.float32)
        expected[0, 0, 1] = 1.5
        expected[0, 2, 3] = -2.5
        expected[1, 0, 3] = 3.5
        expected[1, 1, 0] = -1.5
        np.testing.assert_array_equal(out, expected)

    def test_vmap_callable_bare_axes_on_sparse(self):
        """`vmap(f, in_axes=0)(sparse_spec)` — the callable sugar — traces
        the wrapped defn with the unbatched spec and vectorizes it."""
        graph = evoxc.vmap(_sparse_to_dense, in_axes=0)(_SPARSE_SPEC)
        assert isinstance(graph, evoxc.Graph)

    def test_sparse_sparse_matmul_is_a_v1_deferral(self):
        """sparse @ sparse matmul raises TraceError at trace time (densify
        one operand with evoxc.sparse.to_dense) — never a silent fallback."""

        @evoxc.defn
        def f(a, b):
            return evoxc.sparse.matmul(a, b)

        with pytest.raises(evoxc.TraceError, match="v1 deferral"):
            evoxc.trace(f, _SPARSE_SPEC, _SPARSE_SPEC)

    def test_whole_sparse_bind_is_a_v1_deferral(self):
        """Binding a WHOLE sparse value by name is unsupported in v1 — bind
        is per-leaf; the error is explicit and lists the leaf names."""
        spec = evoxc.sparse.SparseTensorSpec(
            evoxc.TensorSpec((None, 2), evoxc.int64, name="s_indices"),
            evoxc.TensorSpec((None,), evoxc.float32, name="s_values"),
            dense_shape=(3, 4),
            format="coo",
        )
        exe = evoxc.build(_sparse_to_dense, spec)
        with pytest.raises(
            evoxc.TraceError, match=r"unknown input name.*s_indices"
        ):
            evoxc.bind(exe, s=_sparse_coo())

    def test_leaf_level_bind_works(self):
        """Binding individual tensor leaves by name works: bind the indices
        leaf, run with the remaining values + static leaves, get the right
        dense output."""
        spec = evoxc.sparse.SparseTensorSpec(
            evoxc.TensorSpec((None, 2), evoxc.int64, name="s_indices"),
            evoxc.TensorSpec((None,), evoxc.float32, name="s_values"),
            dense_shape=(3, 4),
            format="coo",
        )
        exe = evoxc.build(_sparse_to_dense, spec)
        bound = evoxc.bind(
            exe,
            s_indices=evoxc.tensor(np.array([[0, 1], [2, 3]], dtype=np.int64)),
        )
        partial = evoxc.sparse.SparseTensor.from_parts(
            np.array([1.5, -2.5], dtype=np.float32),
            dense_shape=(3, 4),
            format="coo",
        )
        out = evoxc.run(bound, partial)
        np.testing.assert_array_equal(out.numpy(), _sparse_coo().to_dense())

    def test_vjp_deferrals_are_explicit(self):
        """`sparse_concatenate`, `sparse_coo_to_csc` and `sparse_csc_to_coo`
        have no VJP rule in v1 — vjp raises TransformError naming the op,
        never a silent fallback."""
        # sparse_concatenate (dense output -> plain dense cotangent spec).
        @evoxc.defn
        def cat(a, b):
            return evoxc.sparse.to_dense(evoxc.sparse.concatenate([a, b], axis=0))

        with pytest.raises(evoxc.TransformError, match="sparse_concatenate"):
            evoxc.vjp(cat, evoxc.TensorSpec((6, 4), evoxc.float32))(
                _SPARSE_SPEC, _SPARSE_SPEC
            )

        # sparse_coo_to_csc (CSC output -> sparse cotangent spec).
        @evoxc.defn
        def to_csc_fn(x):
            return evoxc.sparse.to_csc(x)

        csc_cotangent = evoxc.sparse.SparseTensorSpec(
            evoxc.TensorSpec((5,), evoxc.int64),
            evoxc.TensorSpec((None,), evoxc.int64),
            evoxc.TensorSpec((None,), evoxc.float32),
            dense_shape=(3, 4),
            format="csc",
        )
        with pytest.raises(evoxc.TransformError, match="sparse_coo_to_csc"):
            evoxc.vjp(to_csc_fn, csc_cotangent)(_SPARSE_SPEC)

        # sparse_csc_to_coo (csc input auto-converted for to_dense).
        csc_spec = evoxc.sparse.SparseTensorSpec(
            evoxc.TensorSpec((5,), evoxc.int64),
            evoxc.TensorSpec((None,), evoxc.int64),
            evoxc.TensorSpec((None,), evoxc.float32),
            dense_shape=(3, 4),
            format="csc",
        )
        with pytest.raises(evoxc.TransformError, match="sparse_csc_to_coo"):
            evoxc.vjp(_sparse_to_dense, evoxc.TensorSpec((3, 4), evoxc.float32))(
                csc_spec
            )


# ===========================================================================
# 15. Explicit placement — no implicit device transfer (R1/R2/R3/R5/R6)
# ===========================================================================
#
# Design-principle pins: "placement is explicit". A tensor is a local
# physical tensor; host (numpy) memory cannot be relabeled onto another
# device, `.numpy()` on a non-cpu device payload raises (no implicit
# device-to-host transfer), `Tensor.to(device)` is the ONLY transfer API,
# and the run/bind/constant boundaries reject tensors that are not already
# where they must be. All CPU-only: the foreign device is a local fake
# payload duck-typed to the device-payload protocol (.shape/.dtype/.device/
# .to_host) — no backend imports, no GPU.


class _FakeCudaPayload:
    """Tiny duck-typed cuda-kind device payload: shape/dtype/device
    (a core.Device of kind "cuda") + ``to_host()`` -> FRESH ndarray."""

    def __init__(self, shape=(3,), dtype=np.float32):
        self._shape = tuple(shape)
        self.dtype = dtype
        self.device = evoxc.Device("cuda", 0)
        count = int(np.prod(self._shape))
        self._data = np.arange(count, dtype=dtype).reshape(self._shape)

    @property
    def shape(self):
        return self._shape

    def to_host(self):
        return self._data.copy()


def _cuda_kind_tensor(shape=(3,), dtype=np.float32):
    """A concrete ``core.Tensor`` wrapping the fake cuda-kind payload."""
    return evoxc.Tensor(_FakeCudaPayload(shape=shape, dtype=dtype))


class TestExplicitPlacement:
    def test_host_memory_cannot_be_relabeled_cuda(self):
        """R1: ndarray-backed data IS host memory — constructing a Tensor
        with device=Device('cuda', 0) raises DeviceError (no relabeling of
        host memory), naming the explicit t.to transfer as the remedy."""
        array = np.zeros(3, dtype=np.float32)
        with pytest.raises(evoxc.DeviceError) as excinfo:
            evoxc.Tensor(array, device=evoxc.Device("cuda", 0))
        message = str(excinfo.value)
        assert "host (numpy)" in message
        assert "t.to(" in message
        assert "Device(kind='cuda', index=0)" in message

    def test_concrete_creator_rejects_non_cpu_device(self):
        """Concrete creators (evoxc.zeros) produce host memory — a cuda device
        target raises DeviceError, never a silent relabel."""
        with pytest.raises(evoxc.DeviceError) as excinfo:
            evoxc.zeros((3,), device=evoxc.Device("cuda", 0))
        message = str(excinfo.value)
        assert "evoxc.zeros" in message
        assert "host (numpy)" in message
        assert "t.to(" in message

    def test_numpy_on_non_cpu_payload_raises(self):
        """.numpy() on a cuda-kind payload raises DeviceError — there is no
        implicit device-to-host transfer; t.to(cpu) comes first."""
        with pytest.raises(evoxc.DeviceError) as excinfo:
            _cuda_kind_tensor().numpy()
        message = str(excinfo.value)
        assert "no implicit device" in message
        assert "t.to(" in message

    def test_to_cpu_materializes_fresh_host_copy(self):
        """The explicit transfer t.to(Device('cpu', 0)) on a non-cpu payload
        yields a cpu:0 tensor whose values round-trip through to_host()."""
        source = _cuda_kind_tensor(shape=(4,))
        host = source.to(evoxc.Device("cpu", 0))
        assert host.device == evoxc.Device("cpu", 0)
        assert isinstance(host.data, np.ndarray)
        np.testing.assert_array_equal(
            host.numpy(), np.arange(4, dtype=np.float32)
        )

    def test_to_same_device_returns_self(self):
        """t.to(its own device) is the identity — no copy, no error."""
        tensor = _cuda_kind_tensor()
        assert tensor.to(tensor.device) is tensor

    def test_run_boundary_rejects_foreign_device_input(self):
        """R5: run on a cpu executable rejects a cuda-kind input with a
        DeviceError naming the input path — no implicit device↔host
        transfer ever happens at the run boundary (mirrors the pipeline
        tests as a compliance statement)."""

        @evoxc.defn
        def f(x):
            return evoxc.add(x, 1.0)

        executable = evoxc.build(f, evoxc.TensorSpec((3,), evoxc.float32))
        with pytest.raises(evoxc.DeviceError) as excinfo:
            evoxc.run(executable, _cuda_kind_tensor())
        message = str(excinfo.value)
        assert "input at path [0] is on device" in message
        assert "no implicit device" in message
        assert "t.to(" in message

    def test_constant_rejects_non_cpu_tensor(self):
        """R6: embedding a non-cpu tensor via evoxc.constant raises DeviceError
        — the snapshot would be an implicit device-to-host transfer."""
        captured = {}

        @evoxc.defn
        def f():
            captured["out"] = evoxc.constant(_cuda_kind_tensor())
            return captured["out"]

        with pytest.raises(evoxc.DeviceError) as excinfo:
            evoxc.trace(f)
        message = str(excinfo.value)
        assert "requires host data" in message
        assert "t.to(" in message
        assert "Device(kind='cuda', index=0)" in message


# ===========================================================================
# 16. evoxc.io — async host IO is sugar over the ONE explicit transfer primitive
# ===========================================================================
#
# Design-principle pins: async host IO introduces NO implicit copying.
# `prefetch` (host → device) and `sink` (device → host) are documented as
# scheduling the SAME explicit `Tensor.to` copies the caller could perform
# synchronously, merely on a background worker — schedule-time validation is
# synchronous, runtime failures surface at wait()/result() (never swallowed),
# and async handles never auto-wait (no blocking destructor). All CPU-only.


class TestIoExplicitness:
    def test_io_module_documents_sugar_over_explicit_tensor_to(self):
        """`evoxc.io`'s module docstring states its sugar contract: prefetch/
        sink are the same explicit `Tensor.to` transfers, merely executed on
        a background worker — no hidden staging, no implicit copies; errors
        surface at wait()/result()/wait_all(), never swallowed; GC never
        auto-waits."""
        import evoxc.io

        doc = evoxc.io.__doc__
        assert doc, "evoxc.io must carry a module docstring documenting its contract"
        assert "Tensor.to" in doc
        assert "no hidden staging" in doc
        # NB: "background" and "worker." sit on different docstring lines.
        assert "background" in doc and "worker" in doc
        for word in ("wait()", "wait_all()", "GC"):
            assert word in doc
