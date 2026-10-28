"""evoxc — EvoX Compiler.

A minimal, explicit, compiler-neutral tensor graph compiler and runtime.

The public surface is the contract documented in this package's CONTEXT.md
(read it). Everything here is an explicit staging step or a transparent
composition of explicit steps — nothing silently traces, compiles, moves, or
communicates.

Typical explicit workflow::

    @evoxc.defn
    def f(x, w):
        return evoxc.dot(x, w)

    graph = evoxc.trace(f, evoxc.TensorSpec((B, D), evoxc.float32), evoxc.TensorSpec((D, H), evoxc.float32))
    graph = evoxc.vmap(graph, ...)          # optional
    lowered = evoxc.lower(graph)
    artifact = evoxc.compile(lowered)
    executable = evoxc.load(artifact, device="cpu")
    y = evoxc.run(executable, x, w)

Convenience shorthands (documented compositions only)::

    exe = evoxc.build(f, *specs, device="cpu")   # trace→lower→compile→load
    y = evoxc.evaluate(f, x, w)                  # derive specs → build → run

Pytree utilities (pure sugar over flatten/unflatten — no new semantics):

    evoxc.tree_map(fn, *trees)        # map over leaves, structure-validated
    evoxc.tree_leaves(tree)           # all leaves in pre-order
    evoxc.tree_structure(tree)        # the TreeSpec of a tree
    evoxc.tree_flatten(tree)          # (leaves, treespec)
    evoxc.tree_unflatten(leaves, spec)  # rebuild a tree from leaves + spec
"""
from __future__ import annotations

__version__ = "0.1.0"

# --- Value model (evoxc.core) ------------------------------------------------
from . import core  # noqa: F401
from .core import (  # noqa: F401
    BackendError,
    DTypeError,
    Device,
    DeviceError,
    Dim,
    DimExpr,
    EVOXCError,
    PersistenceError,
    ShapeError,
    SymbolicTensor,
    Tensor,
    TensorSpec,
    TraceError,
    TransformError,
    TreeSpec,
    VerificationError,
    bool_,
    complex128,
    complex64,
    devices,
    dim,
    dtype,
    empty,
    flatten,
    float16,
    float32,
    float64,
    from_dlpack,
    from_numpy,
    full,
    int8,
    int16,
    int32,
    int64,
    ones,
    register_pytree_node,
    replicate_tensor,
    split_tensor,
    tensor,
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_structure,
    tree_unflatten,
    uint8,
    uint16,
    uint32,
    uint64,
    unflatten,
    zeros,
)

# --- Tensor operations (evoxc.ops) -------------------------------------------
from . import ops  # noqa: F401
from .ops import (  # noqa: F401
    abs,
    acos,
    add,
    argmax,
    argmin,
    argsort,
    bitwise_and,
    bitwise_left_shift,
    bitwise_or,
    bitwise_right_shift,
    bitwise_xor,
    broadcast,
    cast,
    clamp,
    ceil,
    cholesky,
    concatenate,
    constant,
    conv,
    cos,
    cumprod,
    cumsum,
    diag,
    diagonal,

    divide,
    dot,
    eigh,
    equal,
    erf,
    exp,
    eye,
    flip,
    floor,

    gather,
    gelu,
    greater,
    greater_equal,
    isnan,
    less,
    less_equal,
    linspace,
    log,
    log1p,
    logical_and,
    logical_not,
    logical_or,
    matmul,
    matrix_exp,
    matrix_rank,
    max,
    maximum,
    mean,
    median,
    min,
    minimum,
    multiply,
    nan_to_num,
    nansum,

    negate,
    norm,
    not_equal,
    pad,
    power,
    prod,
    qr,
    reduce_max,
    reduce_mean,
    reduce_min,
    reduce_prod,
    reduce_sum,
    relu,
    remainder,
    reshape,
    roll,
    round,

    runtime_call,
    scatter,
    select,
    sigmoid,
    sign,
    sin,
    slice,
    sort,
    solve,
    sqrt,
    square,
    stack,
    std,
    stop_gradient,
    subtract,
    sum,
    svd,
    tan,
    tanh,
    tile,
    topk,
    # NOTE: the matrix-trace op is NOT exported at the top level — `evoxc.trace`
    # is the tracing function (foundational API). The op lives at
    # `evoxc.ops.trace`.

    transpose,
    tril,
    triu,
    var,
    external_call,
)

# --- Tracing (evoxc.trace) ---------------------------------------------------
# NOTE: `evoxc.trace` (the function) intentionally shadows the `evoxc.trace`
# submodule as an attribute — `import evoxc.trace` / `from evoxc.trace import ...`
# still work through the import system. Same applies to `evoxc.numpy` below.
from . import trace  # noqa: F401  (submodule)
from .trace import Defn, Graph, cond, defn, scan, trace, while_loop  # noqa: F401

# --- Graph transforms (evoxc.transforms) -------------------------------------
from . import transforms  # noqa: F401
from .transforms import (  # noqa: F401
    BatchReport,
    SwitchRecord,
    batch_functions,
    grad,
    jvp,
    vectorize,
    vjp,
    vmap,
)

# --- Backends (evoxc.backends) -----------------------------------------------
from . import backends  # noqa: F401
from .backends import (  # noqa: F401
    Backend,
    Capabilities,
    CompiledArtifact,
    LoweredProgram,
    numpy_backend,
)

# --- Distributed collectives (evoxc.dist) ------------------------------------
from . import dist  # noqa: F401

# --- Sparse tensors (evoxc.sparse) -------------------------------------------
from . import sparse  # noqa: F401

# --- Custom blocks (evoxc.block) ---------------------------------------------
from . import block  # noqa: F401  (submodule)
from .block import BlockOp, block, get_block  # noqa: F401

# --- External kernels (evoxc.external) ----------------------------------------
from . import external  # noqa: F401  (submodule; the kernel registry)
from .external import (  # noqa: F401
    ExternalKernel,
    get_external_kernel,
    get_portable,
    register_external_kernel,
    register_portable,
    unregister_external_kernel,
)

# --- NumPy-like namespace (evoxc.numpy) --------------------------------------
from . import numpy  # noqa: F401
from . import numpy as enp  # noqa: F401  (documented alias)
from .numpy import arange  # noqa: F401  (GRAPH creation op — same object as
# enp.arange; top-level evoxc.zeros/ones/full/empty are the CONCRETE core
# creators, arange has no concrete counterpart)

# --- Key-based functional RNG (evoxc.random) ----------------------------------
from . import random  # noqa: F401  (submodule; the evoxc.random API surface)

# --- Host data IO (evoxc.io) -------------------------------------------------
from . import io  # noqa: F401  (submodule; the evoxc.io.prefetch/sink/AsyncCopy/wait_all surface)

# --- Persistence / cache (evoxc.persist) -------------------------------------
from . import persist  # noqa: F401
from .persist import Cache, FileCache  # noqa: F401

# --- Execution pipeline (evoxc.pipeline) -------------------------------------
from . import pipeline  # noqa: F401
from .pipeline import (  # noqa: F401
    BoundExecutable,
    Executable,
    bind,
    build,
    compile,
    evaluate,
    load,
    lower,
    run,
)

# --- Memoized build sugar (evoxc.build_cache) ---------------------------------
from .build_cache import CompileCache, build_cached  # noqa: F401
