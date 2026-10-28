"""Internal class-DAG machinery for the `batch_functions` transform.

`batch_functions` merges N traced graphs with identical input/output signatures
but different implementations into ONE graph whose leading axis is a STATIC
"function axis" of length N (row i evaluates function i). This module is the
PURE BOOKKEEPING half of that transform — no IR is built here:

* attribute canonicalization (`canonical_value` / `canonical_attributes`) so an
  op's identity and variant are HASHABLE and deterministic;
* merge groups (`MergeGroup` / `parse_switch_groups`) — the ops allowed to
  share one skeleton position across DIFFERENT implementations (each variant is
  emitted, then muxed by a static per-row `select` mask cascade);
* the class DAG (`MergeDAG`): every distinct computation position is a CLASS;
  a class holds the `(program_index, op)` contributors that compute it, and its
  distinct VARIANTS (differing op identities) in first-appearance order;
* reachability from the output slots and the resulting emission order.

Design (binding — see `./CONTEXT.md` "Batch functions"):

``variant_id(op) = (op.name, canonical_attributes)``

``identity(op)`` = `variant_id(op)` normally, `("constant",)` for constants
(an ALWAYS-ON merge: the payload is deliberately NOT part of the identity), or
`("group", group_key)` when a `switch_ops` merge group covers the op (first
matching group in declaration order wins).

``skeleton = (identity, operand_slots, result_types)`` where an operand slot is
``(class_id, result_index)`` for a value produced by a program op or
``("input", tensor_leaf_index)`` for a function block argument.

``occurrence`` = per ``(program, skeleton)`` the 0-based index of the distinct
``variant_id``s seen for that skeleton in that program, in insertion order, so
``class_key = (skeleton, occurrence)``: two DIFFERENT same-skeleton ops (or two
different constants) inside ONE program land in SEPARATE classes, while an
exact repeat reuses one class. Each class therefore holds at most one distinct
variant per program.

``alignment`` (`ALIGNMENTS`) chooses HOW that class is picked for a position
covered by a `switch_ops` merge group — the only positions where two programs
can put DIFFERENT variants at one skeleton: ``"occurrence"`` (default) = the
class at the program's occurrence index, created on demand (byte-identical
output); ``"variant"`` = a class of the skeleton that already holds this
variant and does not yet hold this program, else the occurrence class;
``"auto"`` = ``"variant"`` plus a last-resort reuse of any other free class of
the skeleton before creating one.

Constants are DELIBERATELY excluded (`_align_class` aligns ``("group", ...)``
identities only): a constant class stacks its members' per-row payloads by
OCCURRENCE, which is what folds N distinct GP weight payloads into ONE stacked
``(N, ...)`` constant — variant alignment would explode them into N constants
and undo the batching.

**Correctness invariant (binding).** Every class assignment is ROW-CORRECT: a
class's operands are shared BY SLOT EQUALITY (the skeleton includes the operand
slots), so all contributors' operands are the same emitted values and the class
value at row p is that row's variant applied to p's own operands. `alignment`
is therefore a PURE PERFORMANCE HEURISTIC (op count, select count), never a
semantic choice — it cannot change any number. The ONE hard rule: a second
distinct-variant op of the same program must never join one class (its `select`
mask would overwrite row p), hence every lookup below filters on "does not
already hold this program".

`_opdef` memoizes the IR op-def lookup (the lazy `evoxc.ir.op_defs` import) for
the per-op hot path; each op's `variant_id` is computed ONCE and carried on its
contributor tuple so emission never recomputes it.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from evoxc import core
from evoxc import ir
from evoxc.trace import Graph

#: ``switch_ops=True`` merge groups: the IR op categories whose same-skeleton
#: positions are merged into ONE switch class (each variant emitted, then
#: muxed by a static per-row mask cascade). Deliberately excludes `structure`
#: (e.g. `select` itself) and `reduction`/`linalg`: those change shapes/values
#: in ways a select mux cannot express generically.
DEFAULT_SWITCH_CATEGORIES: Tuple[str, ...] = ("elementwise", "comparison")


# --- op-definition lookup (hot path) ----------------------------------------


#: Memo of ``op name -> ir.OpDef`` — avoids the per-op lazy `evoxc.ir.op_defs`
#: import inside ``Op.opdef``/``is_terminator``. Op defs are registered once at
#: `evoxc.ir` import time and never re-registered, so the mapping is immutable.
_OPDEF_CACHE: Dict[str, "ir.OpDef"] = {}


def _opdef(name: str) -> "ir.OpDef":
    """The registered `ir.OpDef` for `name`, memoized (raises `KeyError` if absent)."""
    definition = _OPDEF_CACHE.get(name)
    if definition is None:
        definition = _OPDEF_CACHE[name] = ir.opdef(name)
    return definition


# --- attribute canonicalization ---------------------------------------------


#: Memo of ``np.dtype -> str(dtype)``: `str(np.dtype)` is comparatively slow
#: and every constant attribute canonicalization needs the same few tokens.
_DTYPE_TOKENS: Dict[np.dtype, str] = {}


def _dtype_token(dtype: np.dtype) -> str:
    """``str(dtype)``, memoized (equal dtypes share one token)."""
    token = _DTYPE_TOKENS.get(dtype)
    if token is None:
        token = _DTYPE_TOKENS[dtype] = str(np.dtype(dtype))
    return token


def canonical_value(value: Any) -> Any:
    """Deeply canonicalize one attribute value into a HASHABLE form.

    Op attributes must be hashable (they are part of the class-DAG skeleton and
    variant keys), so:

    * ``np.ndarray`` → ``("ndarray", str(dtype), shape, payload bytes)`` —
      payload CONTENT is part of the key, so two constants with different data
      are different variants;
    * ``np.dtype`` → ``("dtype", str)``;
    * ``ir.ValueType`` → itself (a frozen, hashable dataclass — symbolic shapes
      included);
    * ``core.Dim`` / ``core.DimExpr`` → a structural tuple recursion;
    * ``dict`` → ``("dict", tuple of (key, canon) sorted by key)``;
    * ``list`` / ``tuple`` → ``("seq", tuple of canon)`` (both normalize to the
      same form — deterministic either way);
    * ``set`` / ``frozenset`` → ``("set", frozenset of canon)``;
    * Python scalars are TAGGED (``("int", 1)``, ``("bool", True)``,
      ``("float", 1.0)``, ...) so the classic ``1 == 1.0 == True`` hash
      collisions can never merge two semantically different variants;
    * everything else must be hashable (returned as-is), else it falls back to
      ``("repr", type name, repr)``.
    """
    if isinstance(value, np.ndarray):
        return ("ndarray", _dtype_token(value.dtype), tuple(value.shape), value.tobytes())
    if isinstance(value, np.dtype):
        return ("dtype", _dtype_token(value))
    if isinstance(value, ir.ValueType):
        return value
    if isinstance(value, core.Dim):
        return ("Dim", value.name, value.size)
    if isinstance(value, core.DimExpr):
        return (
            "DimExpr",
            value.op,
            canonical_value(value.left),
            canonical_value(value.right),
        )
    if isinstance(value, dict):
        items = sorted(
            ((k, canonical_value(v)) for k, v in value.items()),
            key=lambda kv: repr(kv[0]),
        )
        return ("dict", tuple(items))
    if isinstance(value, (list, tuple)):
        return ("seq", tuple(canonical_value(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return ("set", frozenset(canonical_value(item) for item in value))
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value)
    if isinstance(value, complex):
        return ("complex", value)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, (bytes, bytearray)):
        return ("bytes", bytes(value))
    if value is None:
        return ("none",)
    try:
        hash(value)
    except TypeError:
        return ("repr", type(value).__qualname__, repr(value))
    return value


def canonical_attributes(attributes: Optional[dict]) -> tuple:
    """Canonicalize an op's attribute dict into a sorted, hashable tuple."""
    if not attributes:
        return ()
    return tuple(
        (key, canonical_value(value))
        for key, value in sorted(attributes.items(), key=lambda kv: repr(kv[0]))
    )


def variant_id(op: ir.Op) -> tuple:
    """``(op.name, canonical_attributes(op))`` — an op's concrete VARIANT.

    Two ops share a variant iff they are the same IR op with the same
    attributes (constants compare on their payload bytes, so two constants with
    different data are different variants).
    """
    return (op.name, canonical_attributes(op.attributes))


# --- merge groups (`switch_ops`) --------------------------------------------


@dataclass(frozen=True)
class MergeGroup:
    """One `switch_ops` merge group.

    Attributes:
        kind: ``"category"`` (every op of an IR category) or ``"set"`` (an
            explicit set of op names).
        category: The category name for category groups (`""` otherwise).
        names: The op names covered by a set group (`frozenset()` for
            category groups).
        label: The human-readable group name recorded in `SwitchRecord.group`
            — the category name, or ``",".join(sorted(names))``.
    """

    kind: str
    category: str
    names: frozenset
    label: str

    @property
    def key(self) -> tuple:
        """The identity key substituting an op's `variant_id` in a skeleton."""
        if self.kind == "category":
            return ("category", self.category)
        return ("set", self.names)

    def matches(self, op: ir.Op) -> bool:
        """True when `op` belongs to this group."""
        if self.kind == "category":
            return _opdef(op.name).category == self.category
        return op.name in self.names


def _op_categories() -> frozenset:
    """The set of IR op category names (`{o.category for o in all_opdefs()}`)."""
    return frozenset(definition.category for definition in ir.all_opdefs())


def _category_group(category: str) -> MergeGroup:
    """A category merge group, validated against the IR op registry."""
    if not isinstance(category, str) or not category:
        raise TypeError(
            "batch_functions: switch_ops category entries must be non-empty "
            f"strings, got {category!r}"
        )
    valid = _op_categories()
    if category not in valid:
        raise core.TransformError(
            f"batch_functions: unknown switch_ops category {category!r}; "
            f"valid categories are {sorted(valid)}"
        )
    return MergeGroup(
        kind="category", category=category, names=frozenset(), label=category
    )


def _set_group(names) -> MergeGroup:
    """An explicit op-name merge group, validated against the IR op registry."""
    try:
        items = list(names)
    except TypeError as exc:
        raise TypeError(
            "batch_functions: a switch_ops entry must be a category name "
            "(str) or an iterable of op names, got "
            f"{type(names).__name__}"
        ) from exc
    if not items:
        raise core.TransformError(
            "batch_functions: empty switch_ops merge group — an explicit "
            "group must name at least one op"
        )
    for name in items:
        if not isinstance(name, str):
            raise TypeError(
                "batch_functions: switch_ops op-name entries must be strings, "
                f"got {name!r}"
            )
        if not ir.has_opdef(name):
            raise core.TransformError(
                f"batch_functions: unknown op name {name!r} in a switch_ops "
                "merge group; names must be canonical IR op names "
                "(see ir.op_names())"
            )
    unique = frozenset(items)
    return MergeGroup(
        kind="set",
        category="",
        names=unique,
        label=",".join(sorted(unique)),
    )


def parse_switch_groups(switch_ops) -> Tuple[MergeGroup, ...]:
    """Normalize `switch_ops` into a tuple of `MergeGroup`s.

    * ``None`` / ``False`` → no merge groups (no op switching at all);
    * ``True`` → the default groups, one per `DEFAULT_SWITCH_CATEGORIES` entry
      (``elementwise`` and ``comparison``);
    * any other iterable (a `str`/`bytes` is rejected) → explicit groups, each
      element either a category name (`str`, validated against
      ``{o.category for o in ir.all_opdefs()}``) or an iterable of op names
      (each validated against the IR registry).

    Raises:
        TypeError: `switch_ops` is a str/bytes or not iterable, or an element
            is neither a str nor an iterable of str.
        core.TransformError: an unknown category, an empty/unknown op-name
            group.
    """
    if switch_ops is None or switch_ops is False:
        return ()
    if switch_ops is True:
        return tuple(
            _category_group(category) for category in DEFAULT_SWITCH_CATEGORIES
        )
    if isinstance(switch_ops, (str, bytes, bytearray)):
        raise TypeError(
            "batch_functions: switch_ops must be None, False, True, or an "
            "iterable of merge groups (category names / iterables of op "
            f"names); got {type(switch_ops).__name__}"
        )
    try:
        items = list(switch_ops)
    except TypeError as exc:
        raise TypeError(
            "batch_functions: switch_ops must be None, False, True, or an "
            "iterable of merge groups (category names / iterables of op "
            f"names); got {type(switch_ops).__name__}"
        ) from exc
    groups: List[MergeGroup] = []
    for item in items:
        if isinstance(item, str):
            groups.append(_category_group(item))
        elif isinstance(item, (bytes, bytearray)):
            raise TypeError(
                "batch_functions: switch_ops entries must be a category name "
                f"(str) or an iterable of op names, got {type(item).__name__}"
            )
        else:
            groups.append(_set_group(item))
    return tuple(groups)


# --- the class DAG ----------------------------------------------------------


@dataclass
class ProgramInfo:
    """One merged program: its graph, entry function/block, and arg layout."""

    graph: Graph
    function: ir.Function
    block: ir.Block
    terminator: ir.Op
    #: ``id(block argument) -> flat tensor-input index``.
    arg_ids: Dict[int, int]


@dataclass
class ClassInfo:
    """One computation class of the merged graph.

    Attributes:
        class_id: Monotonically increasing class id.
        skeleton: ``(identity, operand_slots, result_types)``.
        occurrence: The occurrence index (within its skeleton) that CREATED the
            class — under ``alignment="occurrence"`` that is exactly the class's
            position; an aligned mode may reuse a class created at another
            occurrence, so treat it as creation metadata only (see the module
            docstring).
        result_count: Number of results of the contributing ops.
        contributors: ``(program_index, op, variant)`` triples, in scan order
            (one per program per distinct variant; an exact repeat within one
            program contributes twice). The op's `variant_id` is computed once
            at scan time and carried here so emission never recomputes it.
        variants: Distinct `variant_id`s in first-appearance order.
        rows: Program indices per variant, aligned with `variants`.
        variant_index: ``variant -> index in `variants``` (bookkeeping only).
    """

    class_id: int
    skeleton: tuple
    occurrence: int
    result_count: int
    contributors: List[Tuple[int, ir.Op, tuple]] = field(default_factory=list)
    variants: List[tuple] = field(default_factory=list)
    rows: List[List[int]] = field(default_factory=list)
    variant_index: Dict[tuple, int] = field(default_factory=dict)
    #: Lazily built ``variant -> first (program_index, op) contributor``.
    _representatives: Optional[Dict[tuple, Tuple[int, ir.Op]]] = field(
        default=None, repr=False, compare=False
    )

    @property
    def representative(self) -> Tuple[int, ir.Op]:
        """The first `(program_index, op)` contributor."""
        program_index, op, _variant = self.contributors[0]
        return program_index, op

    @property
    def program_indices(self) -> Tuple[int, ...]:
        """Distinct contributing program indices, in first-appearance order."""
        return tuple(dict.fromkeys(pi for pi, _op, _variant in self.contributors))

    def representative_for(self, variant: tuple) -> Tuple[int, ir.Op]:
        """The first `(program_index, op)` contributor of one variant."""
        representatives = self._representatives
        if representatives is None:
            representatives = {}
            for program_index, op, contributor_variant in self.contributors:
                representatives.setdefault(contributor_variant, (program_index, op))
            self._representatives = representatives
        entry = representatives.get(variant)
        if entry is None:
            raise core.TransformError(
                "batch_functions: internal error — no contributor for variant "
                f"{variant!r} of class {self.class_id}"
            )
        return entry


class MergeDAG:
    """The class DAG of one `batch_functions` merge (pure bookkeeping).

    Built by `build_merge_dag`; consumed by `_emit.emit_merged_graph`.
    """

    def __init__(self) -> None:
        self.programs: List[ProgramInfo] = []
        self.classes: Dict[int, ClassInfo] = {}
        #: Per program: ``id(value) -> (class_id, result_index)`` for every
        #: value produced by a program op.
        self.slots: List[Dict[int, Tuple[int, int]]] = []
        #: Class ids in first-appearance order (a valid topological order).
        self.order: List[int] = []
        #: `order` filtered to the classes reachable from the output slots.
        self.reachable: List[int] = []

    def slot_of(self, program_index: int, value: ir.Value) -> Tuple[int, int]:
        """The ``(class_id, result_index)`` slot producing `value`."""
        slot = self.slots[program_index].get(id(value))
        if slot is None:
            raise core.TransformError(
                "batch_functions: internal error — value %"
                f"{value.id} of function {program_index} has no producing "
                "class slot"
            )
        return slot

    def ref_of(self, program_index: int, value: ir.Value):
        """``("input", leaf_index)`` for a block argument, else the class slot."""
        program = self.programs[program_index]
        arg_index = program.arg_ids.get(id(value))
        if arg_index is not None:
            return ("input", arg_index)
        return self.slot_of(program_index, value)


_RESULT_TYPE = operator.attrgetter("type")


def _result_types(op: ir.Op) -> tuple:
    """``tuple(result.type for result in op.results)`` — the C-level spelling."""
    return tuple(map(_RESULT_TYPE, op.results))


def _identity(op: ir.Op, variant: tuple, groups: Sequence[MergeGroup]) -> tuple:
    """The op's skeleton identity (constants merge always; groups switch).

    `variant` is the op's already-computed `variant_id` — never recomputed here.
    """
    if op.name == "constant":
        return ("constant",)
    for group in groups:
        if group.matches(op):
            return ("group", group.key)
    return variant


# --- class alignment (`alignment=`) -----------------------------------------


#: Default: a program's k-th distinct variant at a skeleton goes to that
#: skeleton's k-th occurrence class (byte-identical to the pre-`alignment`
#: output).
ALIGNMENT_OCCURRENCE = "occurrence"
#: Variant-first: a class that already holds this variant (and not this program)
#: wins over the occurrence-position class.
ALIGNMENT_VARIANT = "variant"
#: Variant-first plus a last-resort reuse of any other free class of the
#: skeleton before creating one (never creates a class `"variant"` would reuse).
ALIGNMENT_AUTO = "auto"

#: Every accepted `alignment` spelling, in preference order (index 0 = default).
ALIGNMENTS: Tuple[str, ...] = (
    ALIGNMENT_OCCURRENCE,
    ALIGNMENT_VARIANT,
    ALIGNMENT_AUTO,
)
DEFAULT_ALIGNMENT = ALIGNMENT_OCCURRENCE


def normalize_alignment(alignment: Any) -> str:
    """Normalize `alignment` into one of `ALIGNMENTS` (case-insensitive,
    whitespace-stripped), rejecting anything else LOUDLY.

    Raises:
        TypeError: `alignment` is not a string.
        core.TransformError: `alignment` is not an accepted spelling.
    """
    if not isinstance(alignment, str):
        raise TypeError(
            "batch_functions: alignment must be a string, one of "
            f"{list(ALIGNMENTS)}; got {type(alignment).__name__}"
        )
    normalized = alignment.strip().lower()
    if normalized not in ALIGNMENTS:
        raise core.TransformError(
            f"batch_functions: unknown alignment {alignment!r}; accepted forms "
            f"are {list(ALIGNMENTS)}"
        )
    return normalized


@dataclass
class _SkeletonClasses:
    """Class bookkeeping of ONE skeleton, shared by every program of a merge.

    Attributes:
        buckets: ``occurrence index -> class ids at that position``, in creation
            order (a list because the aligned modes can append a second class
            there when the first already holds this program).
        class_ids: Every class id of this skeleton, in creation order (the
            ``"auto"`` last-resort scan order).
        variant_classes: ``variant -> class ids holding it``, in creation order
            (the aligned modes' variant lookup); populated for `switch_ops`
            positions only, the sole places two programs can differ.
    """

    buckets: Dict[int, List[int]] = field(default_factory=dict)
    class_ids: List[int] = field(default_factory=list)
    variant_classes: Dict[tuple, List[int]] = field(default_factory=dict)


def _free_class(
    candidate_ids: Sequence[int], dag: MergeDAG, program_index: int
) -> Optional[int]:
    """The first candidate class that does NOT already hold `program_index`.

    Contributors are appended in non-decreasing program order (programs are
    scanned in order), so a class holds the current program iff its LAST
    contributor has this program index — one comparison, never a scan.
    """
    for class_id in candidate_ids:
        contributors = dag.classes[class_id].contributors
        if not contributors or contributors[-1][0] != program_index:
            return class_id
    return None


def _align_class(
    classes: _SkeletonClasses,
    skeleton: tuple,
    variant: tuple,
    occurrence: int,
    n_results: int,
    dag: MergeDAG,
    program_index: int,
    alignment: str,
) -> int:
    """Pick (creating it when needed) the class for one op of `program_index`.

    Order of preference (every choice is row-correct — see the module docstring's
    correctness invariant; this is a PURE performance heuristic):
      1. `alignment` != ``"occurrence"`` AND the identity is a `switch_ops` MERGE
         GROUP: a class of this skeleton that already holds `variant` and does
         not yet hold `program_index`. Constants are excluded by construction
         (`switchable`) and exact-structure identities need no alignment (every
         program there has the same variant).
      2. The occurrence bucket's first class free for this program.
      3. ``"auto"`` only: any other free class of the skeleton (a last resort
         that only ever REDUCES the class count relative to ``"variant"``).
      4. A NEW class (appended to the occurrence bucket and to `class_ids`).
    """
    switchable = skeleton[0][0] == "group"
    if switchable and alignment != ALIGNMENT_OCCURRENCE:
        class_id = _free_class(
            classes.variant_classes.get(variant, ()), dag, program_index
        )
        if class_id is not None:
            return class_id
    bucket = classes.buckets.get(occurrence)
    if bucket is None:
        bucket = classes.buckets[occurrence] = []
    class_id = _free_class(bucket, dag, program_index)
    if class_id is None and switchable and alignment == ALIGNMENT_AUTO:
        class_id = _free_class(classes.class_ids, dag, program_index)
    if class_id is None:
        class_id = len(dag.classes)
        bucket.append(class_id)
        classes.class_ids.append(class_id)
        dag.classes[class_id] = ClassInfo(
            class_id=class_id,
            skeleton=skeleton,
            occurrence=occurrence,
            result_count=n_results,
        )
        dag.order.append(class_id)
    if switchable:
        ids = classes.variant_classes.get(variant)
        if ids is None:
            classes.variant_classes[variant] = [class_id]
        elif class_id not in ids:
            ids.append(class_id)
    return class_id


def build_merge_dag(
    graphs: Sequence[Graph],
    groups: Sequence[MergeGroup],
    alignment: str = DEFAULT_ALIGNMENT,
) -> MergeDAG:
    """Build the class DAG of `graphs` under the merge `groups` and `alignment`.

    Programs are scanned in order, each program's ops in program order
    (topological by construction), skipping the `return` terminator; the
    class ids follow first-appearance order. Afterwards reachability from the
    output slots is computed (`MergeDAG.reachable`), a valid emission order
    (with identical programs it reduces to the original program order).

    `alignment` (`normalize_alignment`'s canonical spelling) only affects
    positions covered by a `switch_ops` merge group — see `_align_class` for the
    preference order and the row-correctness invariant that makes it a pure
    performance heuristic.

    The caller is responsible for the structural/contract validation (identical
    signatures, single-function single-block programs, every op batchable) and
    for the block-arg-count check — this function assumes them.
    """
    alignment = normalize_alignment(alignment)
    dag = MergeDAG()
    #: ``skeleton -> its class bookkeeping`` (global, one hash lookup per op: the
    #: per-program `own` map below is keyed by the skeleton too).
    skeleton_classes: Dict[tuple, _SkeletonClasses] = {}
    for program_index, graph in enumerate(graphs):
        function = graph.module.functions[0]
        block = function.entry_block
        arg_ids = {id(argument): i for i, argument in enumerate(block.arguments)}
        terminator = block.terminator
        slots: Dict[int, Tuple[int, int]] = {}
        #: ``skeleton -> (variant -> class_id, variants seen, classes)``.
        own: Dict[tuple, Tuple[Dict[tuple, int], List[tuple], _SkeletonClasses]] = {}
        for op in block.ops:
            if _opdef(op.name).is_terminator:
                continue
            variant = variant_id(op)
            identity = _identity(op, variant, groups)
            operand_slots = []
            append_slot = operand_slots.append
            for operand in op.operands:
                input_index = arg_ids.get(id(operand))
                if input_index is not None:
                    append_slot(("input", input_index))
                else:
                    append_slot(slots[id(operand)])
            operand_slots = tuple(operand_slots)
            result_types = _result_types(op)
            skeleton = (identity, operand_slots, result_types)
            entry = own.get(skeleton)
            if entry is None:
                classes = skeleton_classes.get(skeleton)
                if classes is None:
                    classes = skeleton_classes[skeleton] = _SkeletonClasses()
                own[skeleton] = entry = ({}, [], classes)
            by_variant, variants_seen, classes = entry
            class_id = by_variant.get(variant)
            if class_id is None:
                occurrence = len(variants_seen)
                variants_seen.append(variant)
                class_id = _align_class(
                    classes,
                    skeleton,
                    variant,
                    occurrence,
                    len(op.results),
                    dag,
                    program_index,
                    alignment,
                )
                by_variant[variant] = class_id
            info = dag.classes[class_id]
            info.contributors.append((program_index, op, variant))
            variant_position = info.variant_index.get(variant)
            if variant_position is None:
                variant_position = len(info.variants)
                info.variant_index[variant] = variant_position
                info.variants.append(variant)
                info.rows.append([program_index])
            else:
                # Contributors arrive in non-decreasing program order (programs
                # are scanned in order), so the last entry is the only possible
                # duplicate — one comparison instead of a scan.
                rows = info.rows[variant_position]
                if rows[-1] != program_index:
                    rows.append(program_index)
            for index, result in enumerate(op.results):
                slots[id(result)] = (class_id, index)
        dag.slots.append(slots)
        dag.programs.append(
            ProgramInfo(
                graph=graph,
                function=function,
                block=block,
                terminator=terminator,
                arg_ids=arg_ids,
            )
        )
    _mark_reachable(dag)
    return dag


def _mark_reachable(dag: MergeDAG) -> None:
    """Seed reachability from the output slots, then walk operand slots.

    Every program's `return` operands are the merged graph's outputs; a class
    is REACHABLE when a path of operand slots leads to it from some output.
    Unreachable classes (e.g. dead ops left behind by a transform) are never
    emitted.
    """
    reachable = set()
    stack: List[int] = []
    for program_index, program in enumerate(dag.programs):
        for operand in program.terminator.operands:
            ref = dag.ref_of(program_index, operand)
            if ref[0] != "input":
                stack.append(ref[0])
    while stack:
        class_id = stack.pop()
        if class_id in reachable:
            continue
        reachable.add(class_id)
        info = dag.classes[class_id]
        program_index, op = info.representative
        for operand in op.operands:
            ref = dag.ref_of(program_index, operand)
            if ref[0] != "input":
                stack.append(ref[0])
    dag.reachable = [class_id for class_id in dag.order if class_id in reachable]
