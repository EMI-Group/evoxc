"""Error paths of the graph transforms.

v1 restrictions and invalid arguments must fail explicitly with the
contract-mandated error types — `TransformError` (missing rules, axis
mismatches), `ShapeError` (grad of non-scalar output, bad None-cotangents),
`TypeError` (wrong argument kinds), `TraceError` (concrete tensors fed to a
`TransformCallable`) — never with silent fallbacks. Boolean/int-producing ops
(`argmax`, comparisons, ...) are the documented exception: their builtin
rules yield `ZeroTangent`, so gradients through them are zero, NOT an error
(see `./evoxc/transforms/CONTEXT.md` "AD semantics" and "v1 scope vs deferred").
"""

import numpy as np
import pytest

import evoxc

SPEC = evoxc.TensorSpec((4, 3), evoxc.float32)
BATCHED_SPEC = evoxc.TensorSpec((8, 4, 3), evoxc.float32)


def run_graph(graph, *args):
    """Explicit staging: trace→lower→compile→load→run."""
    return evoxc.run(evoxc.load(evoxc.compile(evoxc.lower(graph))), *args)


def f_scale(x):
    return x * 2.0


def f_two(x, y):
    return x + y


def f_static(x, c):
    return x * c


def f_rc(x):
    """Scalar output using a runtime callback (Python escape hatch — no AD or
    batching rules; transforms must reject it by name)."""
    return evoxc.sum(
        evoxc.runtime_call(lambda t: t * 2, x, result=evoxc.TensorSpec((4, 3), evoxc.float32))
        + x
    )


def f_all_reduce(x):
    return evoxc.sum(evoxc.dist.all_reduce(x, op="sum"))


def f_sq(x):
    return evoxc.sum(x * x)


def f_multi(x):
    return x * 2, x + 1


class TestVectorizeAxisErrors:
    """`vectorize(graph, axes)` validates axes against the graph inputs."""

    def test_nonzero_mapped_axis_rejected(self):
        graph = evoxc.trace(f_scale, SPEC)
        with pytest.raises(
            evoxc.TransformError, match=r"must be 0 \(the leading axis\) in v1"
        ):
            evoxc.vectorize(graph, 1)

    def test_mapped_axis_out_of_range(self):
        graph = evoxc.trace(f_scale, SPEC)  # rank 2
        with pytest.raises(evoxc.TransformError, match="out of range"):
            evoxc.vectorize(graph, 2)

    def test_axes_pytree_structure_mismatch(self):
        graph = evoxc.trace(f_two, SPEC, SPEC)
        with pytest.raises(
            evoxc.TransformError, match="does not match the graph's input structure"
        ):
            evoxc.vectorize(graph, (0, 0, 0))

    def test_static_leaf_mapped(self):
        graph = evoxc.trace(f_static, SPEC, 2.0)
        with pytest.raises(evoxc.TransformError, match="static leaf"):
            evoxc.vectorize(graph, (0, 0))

    def test_bare_int_with_multiple_tensor_inputs(self):
        graph = evoxc.trace(f_two, SPEC, SPEC)
        with pytest.raises(evoxc.TransformError, match="exactly ONE tensor input"):
            evoxc.vectorize(graph, 0)

    @pytest.mark.parametrize(
        "bad",
        [f_scale, evoxc.defn(f_scale)],
        ids=["callable", "defn"],
    )
    def test_non_graph_raises_typeerror(self, bad):
        with pytest.raises(TypeError, match="evoxc.vmap"):
            evoxc.vectorize(bad, 0)


class TestVmapInAxesErrors:
    """`vmap(f, in_axes)` mirrors vectorize's axes validation (sugar over the
    same machinery)."""

    def test_nonzero_in_axes_rejected(self):
        tf = evoxc.vmap(f_scale, in_axes=1)
        with pytest.raises(
            evoxc.TransformError, match=r"must be 0 \(the leading axis\) in v1"
        ):
            tf(BATCHED_SPEC)

    def test_in_axes_out_of_range(self):
        tf = evoxc.vmap(f_scale, in_axes=3)  # batched spec rank 3
        with pytest.raises(evoxc.TransformError, match="out of range"):
            tf(BATCHED_SPEC)

    def test_in_axes_structure_mismatch(self):
        tf = evoxc.vmap(f_two, in_axes=(0, 0, 0))
        with pytest.raises(
            evoxc.TransformError, match="does not match the argument structure"
        ):
            tf(BATCHED_SPEC, SPEC)

    def test_static_leaf_mapped(self):
        tf = evoxc.vmap(f_static, in_axes=(0, 0))
        with pytest.raises(evoxc.TransformError, match="static leaf"):
            tf(BATCHED_SPEC, 2.0)

    def test_bare_in_axes_with_multiple_tensor_specs(self):
        tf = evoxc.vmap(f_two)  # default in_axes=0 is a bare entry
        with pytest.raises(evoxc.TransformError, match="exactly ONE tensor spec"):
            tf(BATCHED_SPEC, SPEC)


class TestNonDifferentiableOps:
    """Ops with NO rule (`runtime_call`, collectives) raise `TransformError`
    naming the op — never a silent fallback."""

    def test_grad_runtime_call(self):
        with pytest.raises(
            evoxc.TransformError, match=r"no VJP rule for op 'runtime_call'"
        ):
            evoxc.grad(f_rc)(SPEC)

    def test_grad_runtime_call_graph_form(self):
        graph = evoxc.trace(f_rc, SPEC)
        with pytest.raises(
            evoxc.TransformError, match=r"no VJP rule for op 'runtime_call'"
        ):
            evoxc.grad(graph)

    def test_jvp_runtime_call(self):
        with pytest.raises(
            evoxc.TransformError, match=r"no JVP rule for op 'runtime_call'"
        ):
            evoxc.jvp(f_rc, SPEC)(SPEC)

    def test_vjp_runtime_call(self):
        with pytest.raises(
            evoxc.TransformError, match=r"no VJP rule for op 'runtime_call'"
        ):
            evoxc.vjp(f_rc)(SPEC)

    def test_grad_collective(self):
        with pytest.raises(
            evoxc.TransformError, match=r"no VJP rule for op 'all_reduce'"
        ):
            evoxc.grad(f_all_reduce)(SPEC)

    def test_vmap_runtime_call(self):
        with pytest.raises(
            evoxc.TransformError, match=r"no batching rule for op 'runtime_call'"
        ):
            evoxc.vmap(f_rc)(BATCHED_SPEC)

    def test_vectorize_runtime_call_graph(self):
        graph = evoxc.trace(f_rc, SPEC)
        with pytest.raises(
            evoxc.TransformError, match=r"no batching rule for op 'runtime_call'"
        ):
            evoxc.vectorize(graph, 0)


class TestArgValidation:
    """grad/jvp/vjp argument validation errors."""

    def test_grad_argnums_out_of_range(self):
        with pytest.raises(evoxc.TransformError, match="out of range"):
            evoxc.grad(f_sq, argnums=1)(SPEC)

    def test_grad_argnums_non_int(self):
        with pytest.raises(evoxc.TransformError, match="must be None, an int"):
            evoxc.grad(f_sq, argnums="x")(SPEC)

    @pytest.mark.parametrize("entry", [0.5, "a", True], ids=["float", "str", "bool"])
    def test_grad_argnums_bad_entries(self, entry):
        with pytest.raises(evoxc.TransformError, match="entries must be ints"):
            evoxc.grad(f_sq, argnums=(entry,))(SPEC)

    def test_grad_non_float_input(self):
        with pytest.raises(evoxc.TransformError, match="cannot be differentiated"):
            evoxc.grad(lambda x: evoxc.sum(x * 1))(evoxc.TensorSpec((4, 3), evoxc.int32))

    def test_jvp_tangent_shape_mismatch(self):
        with pytest.raises(evoxc.TransformError, match="does not match the primal shape"):
            evoxc.jvp(f_sq, evoxc.TensorSpec((5, 3), evoxc.float32))(SPEC)

    def test_jvp_tangent_dtype_mismatch(self):
        with pytest.raises(evoxc.TransformError, match="does not match the primal dtype"):
            evoxc.jvp(f_sq, evoxc.TensorSpec((4, 3), evoxc.int32))(SPEC)

    def test_jvp_tangent_structure_mismatch(self):
        with pytest.raises(
            evoxc.TransformError, match="the tangent structure does not match"
        ):
            evoxc.jvp(
                f_sq,
                (evoxc.TensorSpec((4, 3), evoxc.float32), evoxc.TensorSpec((4, 3), evoxc.float32)),
            )(SPEC)

    def test_jvp_tangent_invalid_leaf(self):
        with pytest.raises(
            evoxc.TransformError, match=r"entries must be a core.TensorSpec"
        ):
            evoxc.jvp(f_two, (evoxc.TensorSpec((4, 3), evoxc.float32), 42))(SPEC, SPEC)

    def test_vjp_cotangent_shape_mismatch(self):
        graph = evoxc.trace(f_sq, SPEC)
        with pytest.raises(evoxc.TransformError, match="does not match the primal shape"):
            evoxc.vjp(graph, evoxc.TensorSpec((2, 2), evoxc.float32))

    def test_vjp_cotangent_dtype_mismatch(self):
        graph = evoxc.trace(f_sq, SPEC)
        with pytest.raises(evoxc.TransformError, match="does not match the primal dtype"):
            evoxc.vjp(graph, evoxc.TensorSpec((), evoxc.int32))

    def test_vjp_cotangent_structure_mismatch(self):
        graph = evoxc.trace(f_sq, SPEC)
        with pytest.raises(
            evoxc.TransformError, match="the cotangent structure does not match"
        ):
            evoxc.vjp(
                graph,
                (evoxc.TensorSpec((), evoxc.float32), evoxc.TensorSpec((), evoxc.float32)),
            )

    def test_vjp_none_cotangent_on_nonscalar_output(self):
        with pytest.raises(evoxc.ShapeError, match="requires a scalar output"):
            evoxc.vjp(f_scale)(SPEC)

    def test_vjp_none_cotangent_on_multi_output(self):
        with pytest.raises(evoxc.ShapeError, match="requires exactly one tensor output"):
            evoxc.vjp(f_multi)(SPEC)

    def test_vjp_none_entry_cotangent_on_nonscalar_output(self):
        graph = evoxc.trace(f_scale, SPEC)
        with pytest.raises(evoxc.ShapeError, match="seeds a scalar-one"):
            evoxc.vjp(graph, (None,))

    def test_grad_nonscalar_output(self):
        with pytest.raises(evoxc.ShapeError, match="scalar"):
            evoxc.grad(f_scale)(SPEC)

    def test_grad_multi_output(self):
        with pytest.raises(evoxc.ShapeError, match="exactly one tensor output"):
            evoxc.grad(f_multi)(SPEC)


class TestTypeErrors:
    """Wrong argument kinds raise `TypeError`; concrete tensors fed to a
    `TransformCallable` raise `TraceError` (transforms never execute)."""

    @pytest.mark.parametrize(
        "transform,kwargs",
        [
            (evoxc.vmap, {}),
            (evoxc.grad, {}),
            (evoxc.jvp, {"tangents": None}),
            (evoxc.vjp, {}),
        ],
        ids=["vmap", "grad", "jvp", "vjp"],
    )
    def test_invalid_object_raises_typeerror(self, transform, kwargs):
        with pytest.raises(TypeError, match="expects an evoxc.Graph"):
            transform(42, **kwargs)

    @pytest.mark.parametrize(
        "tc",
        [
            evoxc.vmap(f_scale),
            evoxc.grad(f_sq),
            evoxc.jvp(f_sq, SPEC),
            evoxc.vjp(f_sq),
        ],
        ids=["vmap", "grad", "jvp", "vjp"],
    )
    def test_transform_callable_with_concrete_tensor(self, tc):
        concrete = evoxc.tensor(np.ones((4, 3), dtype=np.float32))
        with pytest.raises(evoxc.TraceError, match="never execute"):
            tc(concrete)


class TestZeroGradientContract:
    """Boolean/int-producing ops (argmax, comparisons, ...) have builtin rules
    yielding `ZeroTangent` — a zero gradient, documented, NOT an error."""

    def test_grad_through_argmax_is_zero(self):
        grad_fn = evoxc.grad(lambda x: evoxc.cast(evoxc.argmax(x), evoxc.float32), argnums=0)
        graph = grad_fn(evoxc.TensorSpec((3, 4), evoxc.float32))
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        result = run_graph(graph, x)
        assert result.dtype == evoxc.float32
        assert np.allclose(result.numpy(), 0.0)

    def test_grad_through_comparison_is_zero(self):
        grad_fn = evoxc.grad(
            lambda x: evoxc.sum(evoxc.cast(evoxc.equal(x, x), evoxc.float32)), argnums=0
        )
        graph = grad_fn(evoxc.TensorSpec((3, 4), evoxc.float32))
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        result = run_graph(graph, x)
        assert result.dtype == evoxc.float32
        assert np.allclose(result.numpy(), 0.0)


# ---------------------------------------------------------------------------
# external_call with no rule and no portable: the external:<name> key format
# ---------------------------------------------------------------------------


def _ext_none_graph():
    """A scalar-loss graph over an external_call with NO registered rule and
    NO portable decomposition (the kernel is never even registered)."""
    @evoxc.defn
    def f(x):
        return evoxc.sum(
            evoxc.external_call(
                "tx_ext_nerr", x, result=evoxc.TensorSpec((4, 3), evoxc.float32)
            )
        )

    return evoxc.trace(f, SPEC)


class TestExternalCallRuleErrors:
    """external_call with neither an explicit rule nor a portable
    decomposition raises `TransformError` naming the op AND the
    `external:<name>` key (the namespaced message format), hinting at the
    external-kernel handle — never a silent fallback."""

    def test_grad_external_call_names_op_and_key(self):
        with pytest.raises(
            evoxc.TransformError,
            match=(
                r"no VJP rule for op 'external_call' "
                r"\(key 'external:tx_ext_nerr'\)"
            ),
        ) as exc:
            evoxc.grad(_ext_none_graph())
        assert "evoxc.register_external_kernel('tx_ext_nerr', fn)" in str(exc.value)

    def test_vjp_external_call_names_op_and_key(self):
        with pytest.raises(
            evoxc.TransformError,
            match=(
                r"no VJP rule for op 'external_call' "
                r"\(key 'external:tx_ext_nerr'\)"
            ),
        ):
            evoxc.vjp(_ext_none_graph())

    def test_jvp_external_call_names_op_and_key(self):
        with pytest.raises(
            evoxc.TransformError,
            match=(
                r"no JVP rule for op 'external_call' "
                r"\(key 'external:tx_ext_nerr'\)"
            ),
        ):
            evoxc.jvp(
                _ext_none_graph(),
                tangents=(evoxc.TensorSpec((4, 3), evoxc.float32),),
            )

    def test_vmap_external_call_names_op_and_key(self):
        with pytest.raises(
            evoxc.TransformError,
            match=(
                r"no batching rule for op 'external_call' "
                r"\(key 'external:tx_ext_nerr'\)"
            ),
        ):
            evoxc.vmap(_ext_none_graph(), in_axes=0)

    def test_vectorize_external_call_names_op_and_key(self):
        with pytest.raises(
            evoxc.TransformError,
            match=(
                r"no batching rule for op 'external_call' "
                r"\(key 'external:tx_ext_nerr'\)"
            ),
        ):
            evoxc.vectorize(_ext_none_graph(), 0)
