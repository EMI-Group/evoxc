"""SSA values.

A ``Value`` is a typed SSA reference inside a ``Function``:

* an *op result* — produced by exactly one op (``owner`` is the defining
  ``Op``, ``index`` is the result index), or
* a *block argument* — an input of a ``Block`` (``owner`` is the block,
  ``index`` is the argument index).

Values carry NO data: no storage, no device, no ``numpy()``, no DLPack. The
frontend ``SymbolicTensor`` (in ``evoxc.core``) wraps a ``Value``; the IR itself
is pure structure. SSA identity is ``id``, which is module-unique (assigned by
the ``Builder`` from the owning ``Module``'s counters so serialization is
stable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Union

from .types import ValueType

if TYPE_CHECKING:  # deferred: op.py/block.py import value.py
    from .block import Block
    from .op import Op


@dataclass
class Use:
    """One use of a Value: ``owner.operands[operand_index] is value``.

    Attributes:
        owner: The op using the value.
        operand_index: Index into ``owner.operands`` of the used value.
    """

    owner: "Op"
    operand_index: int

    @property
    def value(self) -> "Value":
        """The value used by this use."""
        return self.owner.operands[self.operand_index]


class _UseIndex:
    """Private identity-dedup index over a ``Value``'s ``uses`` list.

    Keys are ``(id(use.owner), use.operand_index)`` — the SSA structure a use
    record describes — so a membership test never compares owners structurally
    (``Op``/``Value`` are dataclasses whose generated ``__eq__`` walks the whole
    owner graph, which made trace-time bookkeeping quadratic in the op count).

    The index is a CACHE derived from ``uses``, not a second source of truth:
    ``length`` pins the list length it was built from, so any mutation made
    elsewhere in the IR (a caller rebinding ``uses``, the IR codec rebuilding
    it) forces a rebuild on the next ``add_use``/``remove_use`` instead of
    deduplicating against stale state. It is never part of the SSA structure:
    excluded from ``Value.__repr__`` and ``Value.__eq__``.
    """

    __slots__ = ("keys", "length")

    def __init__(self, uses: "list[Use]") -> None:
        self.keys: "set[tuple[int, int]]" = {
            (id(use.owner), use.operand_index) for use in uses
        }
        self.length = len(uses)


@dataclass
class Value:
    """An SSA value: a typed reference to a future runtime tensor in a graph.

    Attributes:
        id: Module-unique SSA identity (assigned by the ``Builder``).
        type: ``ValueType`` (dtype + symbolic shape).
        owner: The defining ``Op`` (op result) or the parent ``Block``
            (block argument).
        index: Result index (op result) or argument index (block argument).
        uses: All ``Use``s of this value; kept in sync by the ``Builder`` and
            by ``replace_all_uses_with``.
    """

    id: int
    type: ValueType
    owner: Union["Op", "Block"]
    index: int
    uses: list[Use] = field(default_factory=list)
    # Private O(1) dedup index over `uses` (see `_UseIndex`); never structural,
    # so `repr`/`__eq__` are unchanged and it is not a constructor parameter.
    _use_index: Optional["_UseIndex"] = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def is_block_arg(self) -> bool:
        """True if this value is a block argument (it has no defining op)."""
        from .block import Block  # deferred import: op.py/block.py import value.py

        return isinstance(self.owner, Block)

    @property
    def is_op_result(self) -> bool:
        """True if this value is produced by an op."""
        return not self.is_block_arg

    @property
    def defining_op(self) -> Union["Op", None]:
        """The op producing this value, or None for block arguments."""
        if self.is_block_arg:
            return None
        return self.owner  # type: ignore[return-value]

    def _dedup_index(self) -> "_UseIndex":
        """The dedup index over ``uses``, rebuilt if ``uses`` changed underneath."""
        index = self._use_index
        if index is None or index.length != len(self.uses):
            index = self._use_index = _UseIndex(self.uses)
        return index

    def add_use(self, use: Use) -> None:
        """Record one use of this value (idempotent, O(1) in the use count).

        Duplicates are detected on owner IDENTITY plus operand index — the SSA
        structure the record describes — not on structural ``Use`` equality,
        which compares owners through their generated dataclass ``__eq__`` (the
        whole owner graph, quadratic in the trace length). Two uses by the same
        owner at different operand indices (``add(x, x)``) are both recorded.
        ``uses`` stays the ordered, deduplicated list it always was.
        """
        index = self._dedup_index()
        key = (id(use.owner), use.operand_index)
        if key in index.keys:
            return
        index.keys.add(key)
        self.uses.append(use)
        index.length = len(self.uses)

    def _drop_use_at(self, position: int) -> None:
        """Remove ``uses[position]``, keeping the dedup index in sync (internal)."""
        index = self._dedup_index()
        use = self.uses.pop(position)
        index.keys.discard((id(use.owner), use.operand_index))
        index.length = len(self.uses)

    def remove_use(self, use: Use) -> None:
        """Forget one use of this value.

        Raises:
            ValueError: If ``use`` is not recorded on this value.
        """
        try:
            position = self.uses.index(use)
        except ValueError:
            raise ValueError(f"use {use} is not recorded on value %{self.id}") from None
        self._drop_use_at(position)

    def replace_all_uses_with(self, new: "Value") -> None:
        """Rewrite every use of this value to ``new`` (classic SSA RAUW).

        Mutates each using op's operand list and updates ``new.uses``. This
        value's use list is drained — it ends up unused.
        """
        for use in list(self.uses):
            use.owner._replace_operand(use.operand_index, new)
        self.uses.clear()
        self._use_index = None
