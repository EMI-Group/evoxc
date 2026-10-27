"""Shared construction helpers for the evoxc.sparse test suite.

The suite mirrors the `evoxc.sparse` contract (`../../evoxc/sparse/CONTEXT.md` —
sibling, read-only). Helpers here are minimal and stable; per-file extras
belong in the test modules themselves. Import package-qualified:
`from tests.sparse.conftest import ...` (never `from conftest import ...`).
"""

import re

import numpy as np
import pytest

import evoxc
from evoxc import core
from evoxc import sparse

# ---------------------------------------------------------------------------
# Canonical example: the SAME 3x4 dense matrix in all three formats.
# ---------------------------------------------------------------------------


def dense_example():
    """The dense reference of the canonical examples (float32, 3x4)."""
    return np.array(
        [[0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 2.0, 0.0],
         [3.0, 0.0, 0.0, 4.0]],
        dtype=np.float32,
    )


def coo_example():
    """A small canonical COO (numpy leaves, float32 values)."""
    return sparse.coo(
        np.array([[0, 1], [1, 2], [2, 0], [2, 3]], dtype=np.int64),
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        (3, 4),
    )


def csr_example():
    """The same data in canonical CSR form."""
    return sparse.csr(
        np.array([0, 1, 2, 4], dtype=np.int64),
        np.array([1, 2, 0, 3], dtype=np.int64),
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        (3, 4),
    )


def csc_example():
    """The same data in canonical CSC form (column-major ordering)."""
    return sparse.csc(
        np.array([0, 1, 2, 3, 4], dtype=np.int64),
        np.array([2, 0, 1, 2], dtype=np.int64),
        np.array([3.0, 1.0, 2.0, 4.0], dtype=np.float32),
        (3, 4),
    )


# ---------------------------------------------------------------------------
# Matching SparseTensorSpecs (unbatched, trace-time inputs).
# ---------------------------------------------------------------------------


def coo_spec(dtype=evoxc.float32):
    """The SparseTensorSpec matching `coo_example()`."""
    return sparse.SparseTensorSpec(
        evoxc.TensorSpec((None, 2), evoxc.int64),
        evoxc.TensorSpec((None,), dtype),
        dense_shape=(3, 4),
    )


def csr_spec(dtype=evoxc.float32):
    """The SparseTensorSpec matching `csr_example()` (indptr static (4,))."""
    return sparse.SparseTensorSpec(
        evoxc.TensorSpec((4,), evoxc.int64),
        evoxc.TensorSpec((None,), evoxc.int64),
        evoxc.TensorSpec((None,), dtype),
        dense_shape=(3, 4),
        format="csr",
    )


def csc_spec(dtype=evoxc.float32):
    """The SparseTensorSpec matching `csc_example()` (indptr static (5,))."""
    return sparse.SparseTensorSpec(
        evoxc.TensorSpec((5,), evoxc.int64),
        evoxc.TensorSpec((None,), evoxc.int64),
        evoxc.TensorSpec((None,), dtype),
        dense_shape=(3, 4),
        format="csc",
    )


# ---------------------------------------------------------------------------
# Execution helpers.
# ---------------------------------------------------------------------------


def run_graph(graph, *args):
    """Explicit staging: graph -> lower -> compile -> load -> run."""
    return evoxc.run(evoxc.load(evoxc.compile(evoxc.lower(graph))), *args)


def materialize(value):
    """Sparse -> `.to_dense()` ndarray; `evoxc.Tensor` -> `.numpy()`; recurse
    into tuples/lists. Lets tests compare graph outputs against pure-numpy
    references regardless of the result's phase."""
    if sparse.is_sparse(value):
        return value.to_dense()
    if isinstance(value, evoxc.Tensor):
        return value.numpy()
    if isinstance(value, (tuple, list)):
        return type(value)(materialize(v) for v in value)
    return value


def eval_dense(fn, *args):
    """`evoxc.evaluate(fn, *args)` (concrete sparse args derive
    SparseTensorSpecs automatically), then `materialize` the result."""
    return materialize(evoxc.evaluate(fn, *args))


def batched_coo_example():
    """A 2-element batched COO input for vectorize/vmap runs.

    Leaves are (B, nnz, ndim) / (B, nnz) with dense_shape UNCHANGED (3, 4) —
    the input-side batched convention (batch lives in the flat leaves only).
    Built via `SparseTensor.from_parts`: the validating constructors reject
    batched leaves (rank-2 canonical form only) in v1.
    """
    idx = np.array([[0, 1], [1, 2], [2, 0], [2, 3]], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    return sparse.SparseTensor.from_parts(
        evoxc.core.from_numpy(np.stack([idx, idx], axis=0)),
        evoxc.core.from_numpy(np.stack([vals, vals * 2.0], axis=0)),
        dense_shape=(3, 4),
        format="coo",
    )


# ---------------------------------------------------------------------------
# Compiler-backend capability-drift assertion (shared env-gating convention).
# ---------------------------------------------------------------------------


def assert_capability_drift(graph, backend, op_name):
    """Assert the compiler-backend capability-drift contract for `graph`.

    Env gating: activating the adapter itself requires its compiler
    dependency (`registry.get` -> adapter `register()` -> `check_available`),
    so in a compiler-free env `evoxc.lower` raises the adapter-absence
    `BackendError` before the drift pre-check is reached — an environment
    property, not a contract violation. Skip ONLY on that genuine
    adapter-absence signature (the `pip install evoxc[...]` install hint, as in
    `test_backends.py::test_sparse_op_defers_on_tvm`); anything else must
    fail the strict assertions below.
    """
    with pytest.raises(core.BackendError) as excinfo:
        evoxc.lower(graph, backend=backend)
    msg = str(excinfo.value)
    if "capability drift" not in msg and "pip install evoxc[" in msg:
        # Genuine adapter absence in this env — the drift pre-check is
        # unreachable. Any OTHER message falls through to the pins below.
        pytest.skip(f"{backend} adapter not available in this env: {msg}")
    assert re.search("capability drift", msg)
    assert op_name in msg
    assert "evoxc.sparse.to_dense" in msg
