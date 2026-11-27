"""Result types for the `split_by_inputs` structural graph split.

`split_by_inputs` is the FIRST structural (non-numeric) pass in this package:
it slices the graph DATA STRUCTURE into a `prepare` half (the fixed inputs'
contribution) and an `apply` half (the varying inputs on top of a prepared
boundary). It carries NO numerical authority — no derivatives, no batching,
no computation — so it lives in Python here rather than in the Rust engine
(see `./CONTEXT.md`, "split_by_inputs").

Contents:

* `BoundaryValue` — one tensor that crosses from `prepare` to `apply`.
* `SplitPlan` — the two graphs plus the leaves/boundary bookkeeping, with
  inspection helpers (`summary`/`as_dict`/`text`/`print`/`repr`) and a
  `verify()` that checks the wiring law.
* `SplitCallable` — the spec→plan convention for a function/`Defn` input
  (mirrors `_wrappers.TransformCallable`, but returns a `SplitPlan`).
* `InputAnalysis` / `InputInfo` — the per-input inspection report returned by
  `analyze_inputs` (see `analyze.py`).

This module is deliberately free of machinery: the analysis/clone/build code
lives in `_split_deps.py` / `_split_build.py` (re-exported by `_split_core.py`),
the public entry points in `split.py` (the pass) and `analyze.py` (the report).
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

from evoxc import core
from evoxc.trace import Graph
from evoxc.trace._tree import _iter_leaf_paths

__all__ = [
    "BoundaryValue",
    "SplitPlan",
    "SplitCallable",
    "InputAnalysis",
    "InputInfo",
]


# ---------------------------------------------------------------------------
# static byte accounting
# ---------------------------------------------------------------------------


def _known_extent(dim: Any) -> Optional[int]:
    """The statically known size of one shape dim, or ``None``.

    ``int`` dims are static; a `core.Dim` with a known ``size`` and an
    evaluable `core.DimExpr` are known too. A ``None`` dim (runtime-dynamic),
    a `core.Dim` without a size, and a `core.DimExpr` that needs run-time
    bindings are all unknown.
    """
    if dim is None or isinstance(dim, bool):
        return None
    if isinstance(dim, int):
        return dim
    if isinstance(dim, core.Dim):
        return dim.size
    if isinstance(dim, core.DimExpr):
        try:
            return dim.evaluate()
        except core.ShapeError:
            return None
    return None


def static_nbytes(shape: Sequence[Any], dtype: Any) -> Optional[int]:
    """Static byte size of a tensor described by ``shape``/``dtype``.

    The product of the known extents times ``dtype.itemsize``; ``None`` when
    ANY extent is unknown (a runtime-dynamic dim, an unresolved `Dim`, a
    `DimExpr` needing bindings). Never raises — this is a reporting estimate.
    """
    total = 1
    for dim in shape:
        extent = _known_extent(dim)
        if extent is None:
            return None
        total *= int(extent)
    return total * int(np.dtype(dtype).itemsize)


def format_bytes(count: Optional[int]) -> str:
    """Render a byte count compactly (``1.2 MB``); ``"?"`` when unknown."""
    if count is None:
        return "?"
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"  # unreachable (the loop returns at GB)


# ---------------------------------------------------------------------------
# BoundaryValue
# ---------------------------------------------------------------------------


class BoundaryValue:
    """One tensor crossing the `prepare` → `apply` boundary.

    Attributes:
        source_value_id: The SSA id of the corresponding value in the SOURCE
            graph (`ir.Value.id`) — a block argument for a pass-through
            input, an op result otherwise.
        spec: The `core.TensorSpec` the boundary tensor is described by. For
            a pass-through input this is the SOURCE spec OBJECT (its identity
            and symbolic `Dim` objects survive); for an op result it is a
            spec built from the result's `ir.ValueType` (no device, no name).
        shape: ``tuple(spec.shape)``.
        dtype: ``spec.dtype``.
        nbytes: Static product of the known extents times the dtype's item
            size, or ``None`` when an extent is unknown.
        defining_op: The producing op's NAME (``None`` for a pass-through
            input).
        location: The producing op's source `ir.Location` (the input's
            recorded source location for a pass-through input), or ``None``.
        is_input: True iff the boundary value is a pass-through graph input.
    """

    def __init__(
        self,
        source_value_id: int,
        spec: "core.TensorSpec",
        *,
        defining_op: Optional[str] = None,
        location: Any = None,
        is_input: bool = False,
    ) -> None:
        self.source_value_id = int(source_value_id)
        self.spec = spec
        self.shape = tuple(spec.shape)
        self.dtype = spec.dtype
        self.nbytes = static_nbytes(self.shape, self.dtype)
        self.defining_op = defining_op
        self.location = location
        self.is_input = bool(is_input)

    @property
    def producer(self) -> str:
        """A short producer label: the op name or ``"<input>"``."""
        return "<input>" if self.is_input else str(self.defining_op)

    def summary(self) -> dict:
        """A JSON-safe dict describing this boundary value."""
        return {
            "source_value_id": self.source_value_id,
            "defining_op": self.defining_op,
            "is_input": self.is_input,
            "shape": [str(dim) for dim in self.shape],
            "dtype": str(self.dtype),
            "nbytes": self.nbytes,
            "location": None if self.location is None else str(self.location),
        }

    def as_dict(self) -> dict:
        """Alias of `summary()` (the record is already JSON-safe)."""
        return self.summary()

    def __repr__(self) -> str:
        return (
            f"<BoundaryValue %{self.source_value_id} producer={self.producer} "
            f"shape=({', '.join(str(dim) for dim in self.shape)}) "
            f"dtype={self.dtype} {format_bytes(self.nbytes)}>"
        )


# ---------------------------------------------------------------------------
# SplitPlan
# ---------------------------------------------------------------------------


def _entry_ops(graph: Graph) -> Tuple[Any, ...]:
    """The graph's NON-TERMINATOR entry-block ops, in program order."""
    return tuple(op for op in graph.module.main.entry_block.ops if not op.is_terminator)


def _tensor_output_count(graph: Graph) -> int:
    """Number of TENSOR leaves of a graph's output tree."""
    return graph.output_tree.num_leaves - len(graph.output_static_values)


class SplitPlan:
    """The result of `split_by_inputs`: a `prepare` / `apply` graph pair.

    Attributes:
        source: the graph that was split (never mutated).
        prepare: the graph computing the boundary from the FIXED inputs only
            — ``evoxc.run(prepare, *fixed_args)`` returns a flat tuple of
            prepared tensors.
        apply: the graph computing the source outputs from the VARYING inputs
            plus that prepared tuple — ``evoxc.run(apply, varying_tree,
            prepared_tuple)`` returns the source output tree.
        fixed_leaves: flat TENSOR-input indices (into `source.tensor_specs`)
            that stay in `prepare`, ascending.
        varying_leaves: flat TENSOR-input indices that move to `apply`,
            ascending.
        boundary: the `BoundaryValue` records, in the deterministic order
            `prepare` returns them.
        unused_varying_leaves: varying tensor positions that no needed op
            consumes (kept in `apply`'s signature for stability).
    """

    def __init__(
        self,
        source: Graph,
        prepare: Graph,
        apply: Graph,
        fixed_leaves: Sequence[int],
        varying_leaves: Sequence[int],
        boundary: Sequence[BoundaryValue],
        unused_varying_leaves: Sequence[int] = (),
        prepare_op_ids: Optional[Sequence[int]] = None,
        apply_op_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.source = source
        self.prepare = prepare
        self.apply = apply
        self.fixed_leaves = tuple(int(index) for index in fixed_leaves)
        self.varying_leaves = tuple(int(index) for index in varying_leaves)
        self.boundary = tuple(boundary)
        self.unused_varying_leaves = tuple(int(index) for index in unused_varying_leaves)
        if prepare_op_ids is None or apply_op_ids is None:
            # No recorded provenance (a plan assembled outside `split_graph`):
            # re-derive the SOURCE op partition from the plan's own inputs, so
            # `summary()`'s op counts stay exact instead of falling back to
            # clone counts that double-count a shared op.
            from evoxc.transforms import _split_core  # local: it imports this module

            prepare_op_ids, apply_op_ids = _split_core.op_id_partition(
                source, self.fixed_leaves
            )
        # SOURCE op ids each half emits — private accounting for `summary()`,
        # so a `constant` emitted by BOTH halves is one id in the union.
        self._prepare_op_ids = frozenset(int(op_id) for op_id in prepare_op_ids)
        self._apply_op_ids = frozenset(int(op_id) for op_id in apply_op_ids)

    # --- boundary -------------------------------------------------------

    @property
    def boundary_specs(self) -> Tuple["core.TensorSpec", ...]:
        """`core.TensorSpec` for each boundary value, in boundary order."""
        return tuple(value.spec for value in self.boundary)

    @property
    def boundary_bytes(self) -> Optional[int]:
        """Sum of the statically known boundary byte sizes (``None`` if any
        value's size is unknown)."""
        total = 0
        for value in self.boundary:
            if value.nbytes is None:
                return None
            total += value.nbytes
        return total

    # --- op accounting --------------------------------------------------

    def _source_ops(self) -> int:
        return len(_entry_ops(self.source))

    def _op_accounting(self) -> Tuple[int, int, int]:
        """``(prepare_ops, apply_ops, dropped_ops)`` from DISTINCT SOURCE ops.

        Each half clones every SOURCE op it emits exactly once, so a half's op
        count is the size of its source-op-id set (== its entry-block op
        count). ``dropped_ops`` is the residual
        ``source_ops - len(prepare_ids | apply_ids)``: the source ops in
        NEITHER half. An op emitted by BOTH halves — today only a `constant`
        whose result feeds a fixed-only producer of the boundary AND a mixed
        op — is counted in each half's count and ONCE in the union, so
        ``prepare_ops + apply_ops >= source_ops - dropped_ops`` with equality
        iff no op is shared, and ``dropped_ops`` is never negative (both id
        sets are subsets of the source's op ids).
        """

        prepare_ops = len(self._prepare_op_ids)
        apply_ops = len(self._apply_op_ids)
        shared = self._prepare_op_ids | self._apply_op_ids
        return prepare_ops, apply_ops, self._source_ops() - len(shared)

    # --- inspection -----------------------------------------------------

    def summary(self) -> dict:
        """A JSON-safe structural summary of the plan.

        Keys (exactly): ``fixed_leaves``, ``varying_leaves``,
        ``boundary_values``, ``boundary_bytes``, ``source_ops``,
        ``prepare_ops``, ``apply_ops``, ``dropped_ops``,
        ``unused_varying_leaves``, ``prepare_digest``, ``apply_digest``.
        Op counts are NON-TERMINATOR ops. ``prepare_ops`` / ``apply_ops`` are
        the numbers of SOURCE ops each half emits, and ``dropped_ops`` counts
        the source ops in NEITHER half (dead or folded into the boundary).
        An op emitted by BOTH halves — today only a `constant` whose result
        feeds a fixed-only producer of the boundary AND a mixed op — is
        counted in each half's count and once in the union ``dropped_ops`` is
        derived from, so ``prepare_ops + apply_ops`` exceeds
        ``source_ops - dropped_ops`` by the number of shared ops (they are
        equal when nothing is shared) and ``dropped_ops`` is never negative.
        ``boundary_bytes`` is ``None`` when any boundary value's static size is
        unknown.
        """

        source_ops = self._source_ops()
        prepare_ops, apply_ops, dropped_ops = self._op_accounting()
        return {
            "fixed_leaves": self.fixed_leaves,
            "varying_leaves": self.varying_leaves,
            "boundary_values": len(self.boundary),
            "boundary_bytes": self.boundary_bytes,
            "source_ops": source_ops,
            "prepare_ops": prepare_ops,
            "apply_ops": apply_ops,
            "dropped_ops": dropped_ops,
            "unused_varying_leaves": self.unused_varying_leaves,
            "prepare_digest": self.prepare.digest(),
            "apply_digest": self.apply.digest(),
        }

    def as_dict(self) -> dict:
        """`summary()` with tuples rendered as lists plus a per-value
        ``boundary`` list (everything is JSON-safe)."""
        data = self.summary()
        for key in ("fixed_leaves", "varying_leaves", "unused_varying_leaves"):
            data[key] = list(data[key])
        data["boundary"] = [value.as_dict() for value in self.boundary]
        return data

    def text(self) -> str:
        """Human-readable report: the leaves with their pytree paths, the
        boundary table and the per-graph op counts."""
        # Every plan field indexes TENSOR inputs, while `_iter_leaf_paths`
        # enumerates TREE leaves, so drop the static positions (a static value
        # is never a boundary input and has no tensor position).
        static_positions = {record.index for record in self.source.static_values}
        paths = [
            path
            for index, path in enumerate(_iter_leaf_paths(self.source.input_specs))
            if index not in static_positions
        ]
        lines = [f"SplitPlan over {self.source!r}"]
        lines.append(
            f"  fixed leaves ({len(self.fixed_leaves)}): "
            + self._render_leaves(paths, self.fixed_leaves)
        )
        lines.append(
            f"  varying leaves ({len(self.varying_leaves)}): "
            + self._render_leaves(paths, self.varying_leaves)
        )
        if self.unused_varying_leaves:
            lines.append(
                "  unused varying leaves: "
                + self._render_leaves(paths, self.unused_varying_leaves)
            )
        lines.append(
            f"  boundary ({len(self.boundary)} values, "
            f"{format_bytes(self.boundary_bytes)}):"
        )
        for index, value in enumerate(self.boundary):
            lines.append(
                f"    [{index}] {value.producer:<24} "
                f"shape=({', '.join(str(dim) for dim in value.shape)}) "
                f"dtype={value.dtype} {format_bytes(value.nbytes)}"
            )
        summary = self.summary()
        lines.append(
            f"  ops: source {summary['source_ops']}, "
            f"prepare {summary['prepare_ops']}, apply {summary['apply_ops']}, "
            f"dropped {summary['dropped_ops']}"
        )
        lines.append(f"  prepare digest {summary['prepare_digest']}")
        lines.append(f"  apply digest   {summary['apply_digest']}")
        return "\n".join(lines)

    def print(self) -> None:
        """Print `text()` to stdout."""
        print(self.text())

    def _render_leaves(self, paths, leaves) -> str:
        if not leaves:
            return "<none>"
        return ", ".join(
            f"{index} at {core.format_path(paths[index])} "
            f"({', '.join(str(dim) for dim in self.source.tensor_specs[index].shape)}"
            f"{self.source.tensor_specs[index].dtype})"
            for index in leaves
        )

    def __repr__(self) -> str:
        return (
            f"<SplitPlan fixed={len(self.fixed_leaves)} "
            f"varying={len(self.varying_leaves)} "
            f"boundary={len(self.boundary)} values / "
            f"{format_bytes(self.boundary_bytes)}, "
            f"prepare {len(_entry_ops(self.prepare))} ops, "
            f"apply {len(_entry_ops(self.apply))} ops>"
        )

    # --- verification ---------------------------------------------------

    def verify(self) -> None:
        """Verify both graphs and the plan's WIRING LAW.

        Checks (raising `core.VerificationError` with a precise message):

        * `prepare.verify()` and `apply.verify()` (IR invariants);
        * ``prepare``'s tensor-output leaf count == ``len(boundary)`` ==
          ``apply``'s prepared-input count;
        * each position agrees on shape/dtype/device (the IR side carries no
          device, so a device is compared whenever both sides declare one);
        * ``apply``'s tensor-output leaf count == the source's.

        Argument-kind problems raise `core.TransformError`; this method is
        for plan-construction bugs.
        """
        self.prepare.verify()
        self.apply.verify()

        boundary_count = len(self.boundary)
        prepare_outputs = _tensor_output_count(self.prepare)
        if prepare_outputs != boundary_count:
            raise core.VerificationError(
                f"split plan: `prepare` returns {prepare_outputs} tensor "
                f"outputs but the plan records {boundary_count} boundary "
                f"values"
            )
        prepared_inputs = len(self.apply.tensor_specs) - len(self.varying_leaves)
        if prepared_inputs != boundary_count:
            raise core.VerificationError(
                f"split plan: `apply` takes {prepared_inputs} prepared inputs "
                f"but the plan records {boundary_count} boundary values"
            )

        prepare_specs = tuple(
            core.TensorSpec(shape=value_type.shape, dtype=value_type.dtype)
            for value_type in self.prepare.module.main.output_types
        )
        apply_prepared = self.apply.tensor_specs[len(self.varying_leaves):]
        for index, value in enumerate(self.boundary):
            _check_spec_match(
                prepare_specs[index], value.spec, index, "prepare output"
            )
            _check_spec_match(
                apply_prepared[index], value.spec, index, "apply prepared input"
            )

        source_outputs = _tensor_output_count(self.source)
        apply_outputs = _tensor_output_count(self.apply)
        if apply_outputs != source_outputs:
            raise core.VerificationError(
                f"split plan: `apply` returns {apply_outputs} tensor outputs "
                f"but the source graph returns {source_outputs}"
            )


def _check_spec_match(
    actual: "core.TensorSpec", expected: "core.TensorSpec", index: int, what: str
) -> None:
    """Raise `core.VerificationError` unless two specs agree.

    Shape and dtype must match exactly. A device is compared only when BOTH
    sides declare one — IR value types carry no device, so the `prepare`
    side's spec is always device-less while a pass-through input's spec may
    name one.
    """
    if tuple(actual.shape) != tuple(expected.shape) or actual.dtype != expected.dtype:
        raise core.VerificationError(
            f"split plan: boundary value {index} ({what}) is described as "
            f"shape {tuple(actual.shape)} dtype {actual.dtype} but the plan "
            f"records shape {tuple(expected.shape)} dtype {expected.dtype}"
        )
    if (
        actual.device is not None
        and expected.device is not None
        and actual.device != expected.device
    ):
        raise core.VerificationError(
            f"split plan: boundary value {index} ({what}) is described on "
            f"device {actual.device} but the plan records device "
            f"{expected.device}"
        )


# ---------------------------------------------------------------------------
# SplitCallable
# ---------------------------------------------------------------------------


class SplitCallable:
    """Returned by `split_by_inputs` when given a callable/`Defn`.

    ``sc(*specs)`` — where ``specs`` is a structure of `core.TensorSpec`s
    (+ static values) matching the wrapped function's inputs — traces the
    function exactly ONCE and returns the `SplitPlan`. Passing concrete
    `Tensor`s raises `core.TraceError`: a plan is a graph transformation and
    never executes, so build/run the two graphs explicitly
    (`evoxc.build` / `evoxc.run`).
    """

    def __init__(self, trace_and_split: Callable[..., "SplitPlan"], doc=None) -> None:
        self._trace_and_split = trace_and_split
        #: Always ``"split_by_inputs"``.
        self.kind = "split_by_inputs"
        if doc:
            self.__doc__ = doc

    def __call__(self, *specs) -> "SplitPlan":
        self._check_no_concrete_tensors(specs)
        return self._trace_and_split(*specs)

    def _check_no_concrete_tensors(self, args) -> None:
        """Reject concrete `core.Tensor` leaves with `core.TraceError`."""
        flat, _ = core.flatten(args)
        for arg in flat:
            if isinstance(arg, core.Tensor):
                raise core.TraceError(
                    "calling an evoxc.split_by_inputs result with concrete "
                    "Tensors is not supported: it traces a function from "
                    "TensorSpecs and never executes. Pass TensorSpecs to obtain "
                    "the SplitPlan, then build/run the `prepare` and `apply` "
                    "graphs explicitly (evoxc.build / evoxc.run)."
                )

    def __repr__(self) -> str:
        return f"<evoxc.transforms.SplitCallable kind={self.kind!r}>"


# ---------------------------------------------------------------------------
# InputInfo (analyze_inputs)
# ---------------------------------------------------------------------------


class InputInfo:
    """Per-tensor-input facts of one traced graph (`analyze_inputs`).

    Attributes:
        index: The tensor-input index — into ``graph.tensor_specs``, i.e. the
            entry-block argument order, which is the SAME axis
            `split_by_inputs`'s ``fixed_leaves`` uses.
        path: The input's pytree path, rendered by `core.format_path` over its
            leaf position in ``graph.input_specs`` (that axis also counts the
            static leaves, hence the distinction from `index`).
        spec: The SOURCE `core.TensorSpec` OBJECT (its identity and symbolic
            `Dim` objects survive).
        nbytes: Static byte size (`static_nbytes`), or ``None`` when an extent
            is unknown.
        direct_consumers: Distinct non-terminator entry-block ops that take
            this block argument as an operand.
        op_count: Ops transitively depending on it (their dependence mask has
            this input's bit set).
        outputs: Indices into the graph's TENSOR output leaves whose `return`
            operand depends on this input.
        boundary_bytes: Summed static byte size of the boundary that fixing
            this input ALONE would produce; ``None`` when the graph is not
            splittable, fixing it alone is not a legal split, or any boundary
            value's size is unknown.
        boundary_count: How many boundary values fixing this input alone
            produces (``None`` under exactly the conditions of
            `boundary_bytes`).
    """

    def __init__(
        self,
        index: int,
        path: str,
        spec: "core.TensorSpec",
        *,
        direct_consumers: int = 0,
        op_count: int = 0,
        outputs: Sequence[int] = (),
        boundary_bytes: Optional[int] = None,
        boundary_count: Optional[int] = None,
    ) -> None:
        self.index = int(index)
        self.path = path
        self.spec = spec
        self.nbytes = static_nbytes(tuple(spec.shape), spec.dtype)
        self.direct_consumers = int(direct_consumers)
        self.op_count = int(op_count)
        self.outputs = tuple(int(output) for output in outputs)
        self.boundary_bytes = (
            None if boundary_bytes is None else int(boundary_bytes)
        )
        self.boundary_count = (
            None if boundary_count is None else int(boundary_count)
        )

    def summary(self) -> dict:
        """A JSON-safe dict of these facts (``outputs`` as a tuple)."""
        return {
            "index": self.index,
            "path": self.path,
            "shape": [str(dim) for dim in self.spec.shape],
            "dtype": str(self.spec.dtype),
            "nbytes": self.nbytes,
            "direct_consumers": self.direct_consumers,
            "op_count": self.op_count,
            "outputs": self.outputs,
            "boundary_bytes": self.boundary_bytes,
            "boundary_count": self.boundary_count,
        }

    def as_dict(self) -> dict:
        """`summary()` with ``outputs`` rendered as a list (JSON-safe)."""
        data = self.summary()
        data["outputs"] = list(data["outputs"])
        return data

    def __repr__(self) -> str:
        if self.boundary_count is None:
            boundary = "?"
        else:
            boundary = (
                f"{format_bytes(self.boundary_bytes)} / {self.boundary_count}"
            )
        return (
            f"<InputInfo {self.index} at {self.path} "
            f"shape=({', '.join(str(dim) for dim in self.spec.shape)}) "
            f"dtype={self.spec.dtype} {format_bytes(self.nbytes)}: "
            f"consumers={self.direct_consumers} ops={self.op_count} "
            f"outputs={list(self.outputs)} boundary={boundary}>"
        )


# ---------------------------------------------------------------------------
# InputAnalysis (analyze_inputs)
# ---------------------------------------------------------------------------


class InputAnalysis:
    """The result of `analyze_inputs`: the per-input facts of one graph.

    Pure inspection — this object holds no graph transformation, executes
    nothing and caches nothing; `split()` builds a FRESH `SplitPlan` each
    call.

    Attributes:
        source: the graph analyzed (never mutated).
        inputs: one `InputInfo` per TENSOR input, in ``graph.tensor_specs``
            order (tensor-input index order).
        op_count: NON-TERMINATOR entry-block ops (nested region bodies are the
            gate's business, never counted here).
        output_count: number of TENSOR leaves in the graph's output tree.
        splittable: True iff `evoxc.split_by_inputs` accepts this graph.
        blocked_reason: the gate's message verbatim (naming the offending op
            and its source location) when `splittable` is False, else ``None``.
    """

    def __init__(
        self,
        source: Graph,
        inputs: Sequence[InputInfo],
        *,
        op_count: int = 0,
        output_count: int = 0,
        splittable: bool = True,
        blocked_reason: Optional[str] = None,
    ) -> None:
        self.source = source
        self.inputs = tuple(inputs)
        self.op_count = int(op_count)
        self.output_count = int(output_count)
        self.splittable = bool(splittable)
        self.blocked_reason = blocked_reason

    # --- the pass this report explains ----------------------------------

    def split(self, fixed=None, *, fixed_leaves=None) -> SplitPlan:
        """Split `self.source` — thin sugar over `evoxc.split_by_inputs`.

        Arguments are forwarded verbatim (``fixed`` / ``fixed_leaves``); on a
        blocked graph this raises exactly the gate's `core.TransformError`.
        """
        # Local import: `split.py` imports this module, so a module-level
        # import here would be circular.
        from evoxc.transforms.split import split_by_inputs

        return split_by_inputs(self.source, fixed, fixed_leaves=fixed_leaves)

    # --- inspection -----------------------------------------------------

    def summary(self) -> dict:
        """A JSON-safe summary of the analysis.

        Keys (exactly): ``input_count``, ``output_count``, ``op_count``,
        ``splittable``, ``blocked_reason``. `as_dict()` adds the per-input
        detail.
        """
        return {
            "input_count": len(self.inputs),
            "output_count": self.output_count,
            "op_count": self.op_count,
            "splittable": self.splittable,
            "blocked_reason": self.blocked_reason,
        }

    def as_dict(self) -> dict:
        """`summary()` plus ``inputs`` — one dict per tensor input."""
        data = self.summary()
        data["inputs"] = [info.as_dict() for info in self.inputs]
        return data

    def text(self) -> str:
        """Human-readable report: a header with the graph-level counts and
        the blocked reason, then one row per tensor input."""
        static_count = len(self.source.static_values)
        lines = [
            f"InputAnalysis over {self.source!r}",
            f"  {len(self.inputs)} tensor inputs, {static_count} static values, "
            f"{self.op_count} entry-block ops, {self.output_count} tensor outputs",
        ]
        if self.splittable:
            lines.append("  splittable: yes")
        else:
            lines.append(f"  splittable: no — {self.blocked_reason}")
        if self.inputs:
            lines.append(
                "   idx  path           shape             dtype    bytes    "
                "consumers  ops  outputs    boundary(alone)"
            )
            for info in self.inputs:
                shape = "(" + ", ".join(str(dim) for dim in info.spec.shape) + ")"
                if info.boundary_count is None:
                    boundary = "?"
                else:
                    boundary = (
                        f"{format_bytes(info.boundary_bytes)} / "
                        f"{info.boundary_count} values"
                    )
                lines.append(
                    f"  {info.index:>4}  {info.path:<14} {shape:<17} "
                    f"{str(info.spec.dtype):<8} {format_bytes(info.nbytes):>7}  "
                    f"{info.direct_consumers:>9}  {info.op_count:>4}  "
                    f"{str(list(info.outputs)):<9}  {boundary}"
                )
        return "\n".join(lines)

    def print(self) -> None:
        """Print `text()` to stdout."""
        print(self.text())

    def __repr__(self) -> str:
        state = "splittable" if self.splittable else "blocked"
        return (
            f"<InputAnalysis {len(self.inputs)} inputs, {self.op_count} ops, "
            f"{self.output_count} outputs, {state}>"
        )
