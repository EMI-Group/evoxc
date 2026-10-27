"""Error types owned by `evoxc.block`."""

from __future__ import annotations

from evoxc.core import EVOXCError

__all__ = ["BlockError"]


class BlockError(EVOXCError):
    """Invalid block declaration or static misuse of a `BlockOp`.

    Declaration-time problems: duplicate or unknown block names, malformed
    input/output specs, undeclared or wrongly typed static attributes,
    invalid effect kinds or batching policies, and structurally invalid
    declarations (e.g. a portable implementation that is not an `evoxc.defn`
    function, or a missing name).

    Graph-time problems use the standard core error classes instead:
    `TraceError` (call outside a trace / concrete `Tensor` operand),
    `ShapeError` / `DTypeError` (operand mismatch against `input_specs`),
    and `TransformError` (missing batching/derivative rule — raised by
    `evoxc.transforms`, never by block itself).
    """
