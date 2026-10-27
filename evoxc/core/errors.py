"""Error hierarchy for evoxc.

All public errors raised by evoxc derive from :class:`EVOXCError` so callers can
catch a single base class. Each subclass identifies a distinct failure domain
(see the root ``CONTEXT.md`` error strategy — binding for all modules).

Convention (binding): error messages include a source location (e.g.
``model.py:83``) whenever a graph location exists. No evoxc API silently
swallows or works around errors.
"""

from __future__ import annotations

__all__ = [
    "EVOXCError",
    "TraceError",
    "ShapeError",
    "TransformError",
    "BackendError",
    "PersistenceError",
    "DeviceError",
    "DTypeError",
    "VerificationError",
]


class EVOXCError(Exception):
    """Base class for all evoxc errors."""


class TraceError(EVOXCError):
    """Raised when a graph-building operation is used outside a valid trace.

    Examples: calling an op function outside ``evoxc.trace``/``@evoxc.defn``,
    passing a concrete :class:`~evoxc.core.Tensor` to an op function (no eager
    mode), calling a ``Defn`` with concrete tensors, or using a
    ``SymbolicTensor`` as a Python boolean / with unregistered operator
    handlers.
    """


class ShapeError(EVOXCError):
    """Raised on shape/rank mismatches and failed shape inference.

    Includes symbolic-dimension unification failures reported by
    ``Dim``/``DimExpr`` evaluation.
    """


class TransformError(EVOXCError):
    """Raised by graph→graph transforms (vectorize/vmap/grad/jvp/vjp).

    E.g. axis mismatch, unsupported batching rule, missing derivative rule.
    Transforms must never silently fall back when a rule is missing.
    """


class BackendError(EVOXCError):
    """Raised when a backend cannot lower/compile/load/execute a program.

    Backend capability limitations (dynamic shapes, collectives, runtime
    calls, custom blocks, unsupported dtypes) fail explicitly.
    """


class PersistenceError(EVOXCError):
    """Raised on save/load failures: format mismatch, integrity failure,
    backend/device/ABI incompatibility. Loading never silently re-traces or
    recompiles."""


class DeviceError(EVOXCError):
    """Raised on device errors: unknown device kind/index, device mismatch,
    interop (DLPack) failures, multi-device preparation errors."""


class DTypeError(EVOXCError):
    """Raised when an object cannot be normalized to a supported dtype, or a
    dtype is unsupported by an operation/backend."""


class VerificationError(EVOXCError):
    """Raised when IR verification fails (malformed module, op operand/result
    mismatches, violated invariants)."""
