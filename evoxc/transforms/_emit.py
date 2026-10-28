"""Batched emission for the `batch_functions` transform.

The IR-building half of `batch_functions`: given N traced graphs with identical
signatures plus their class DAG (`_merge.build_merge_dag`), build ONE new
`Graph` whose leading axis is a STATIC function axis of length N — row i
evaluates function i — and assemble the `BatchReport` describing the merge.

Per class, in emission order (see `./CONTEXT.md` "Batch functions"):

* **constant class** — built directly (`ops.constant`), never through a rule.
  All payloads equal ⇒ ONE shared UNMAPPED constant; payloads differing across
  the contributing functions ⇒ ONE stacked constant of shape
  ``(N, *payload_shape)`` where row i holds function i's payload and
  non-participating rows hold the lowest contributing function's payload (never
  NaN) — a MAPPED value.
* **single-variant class** (one distinct op variant) — the op is rebuilt ONCE
  through its registered batching rule, exactly like `vectorize`: this is the
  "identical functions ⇒ zero overhead" path (no `select`s at all).
* **switch class** (≥2 variants in one merge group) — every variant is emitted
  through its own rule over the SAME operand stack, then the variants are
  muxed per result with a static per-row `select` mask cascade (variant 0 needs
  no mask). A switch class is ALWAYS MAPPED: its value depends on the function
  row, so every candidate is aligned to the mapped shape first (an UNMAPPED
  candidate — shared-input variants — is broadcast along a fresh leading
  function axis, see `_align_switch_candidate`).

Outputs: one value per tensor output leaf, in output-leaf order. A single
distinct producing class passes through (a broadcast is inserted only when the
value is UNMAPPED and `out_axes` asks for the function axis); several distinct
producing classes are broadcast to the mapped ``(N, ...)`` shape and muxed with
the same mask cascade (`output_muxes`).

No execution, no caching, no Python loop over batch rows: the emitted graph is
an ordinary graph of ordinary ops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from evoxc import core
from evoxc import ir
from evoxc import ops
from evoxc.ops._utils import _suppress_location_capture, get_location
from evoxc.trace import Graph, current_builder, with_builder
from evoxc.transforms import batching
from evoxc.transforms._merge import ClassInfo, MergeDAG
from evoxc.transforms._metadata import MappedAxes, UNMAPPED, ValueEnv


# --- public report types ----------------------------------------------------


@dataclass(frozen=True)
class SwitchRecord:
    """One switched position: `group`, `variants` (first-appearance order) and
    `rows_per_variant` — the GLOBAL function indices using each variant."""

    group: str
    variants: Tuple[str, ...]
    rows_per_variant: Tuple[Tuple[int, ...], ...]


@dataclass
class BatchReport:
    """What one `batch_functions` merge actually did (the `batch_report`).

    `value_classes` counts the REACHABLE classes emitted; `shared_classes` the
    ones contributed by ≥2 functions; `stacked_constants` the constants whose
    payloads differed; `output_muxes` the outputs muxed from ≥2 classes;
    `emitted_ops` the merged entry block's ops minus the `return`.
    `bands`/`band_sizes`/`band_strategy` describe the partition (`"flat"` for
    `bands=None`, `"explicit"` for an int ``K``, `"auto"` for the automatic
    one); `cost_estimate`/`flat_cost_estimate` are the static model's numbers
    for the CHOSEN and the flat partition (``None`` unless the model ran).
    With several bands the per-class counters are TOTALS across bands
    (`shared_classes` sums a band's classes contributed by ≥2 of THAT BAND's
    functions) and `emitted_ops` counts band slices and output concatenations.
    """

    n_functions: int
    value_classes: int
    shared_classes: int
    switched: Tuple[SwitchRecord, ...] = ()
    stacked_constants: int = 0
    output_muxes: int = 0
    emitted_ops: int = 0
    bands: int = 1
    band_sizes: Tuple[int, ...] = ()
    band_strategy: str = "flat"
    cost_estimate: Optional[float] = None
    flat_cost_estimate: Optional[float] = None

    def as_dict(self) -> dict:
        """A plain-`dict` view (nested `SwitchRecord`s expanded to dicts)."""
        return {
            "n_functions": self.n_functions,
            "value_classes": self.value_classes,
            "shared_classes": self.shared_classes,
            "switched": [
                {
                    "group": record.group,
                    "variants": list(record.variants),
                    "rows_per_variant": [
                        list(rows) for rows in record.rows_per_variant
                    ],
                }
                for record in self.switched
            ],
            "stacked_constants": self.stacked_constants,
            "output_muxes": self.output_muxes,
            "emitted_ops": self.emitted_ops,
            "bands": self.bands,
            "band_sizes": list(self.band_sizes),
            "band_strategy": self.band_strategy,
            "cost_estimate": self.cost_estimate,
            "flat_cost_estimate": self.flat_cost_estimate,
        }


# --- emission helpers -------------------------------------------------------


def _sym(value: ir.Value) -> "core.SymbolicTensor":
    """Wrap an `ir.Value` as a `core.SymbolicTensor` (dtype/shape from its type)."""
    return batching._sym(value)


def _broadcast(value: ir.Value, shape: tuple) -> ir.Value:
    """Broadcast `value` to `shape` (numpy `broadcast_to`, prepends new dims).

    Uses the frontend `ops.broadcast` (the standard path). Targets containing a
    `None` (runtime-dynamic) dim are built on the raw builder instead: the
    frontend shape checker rejects wildcard targets while the IR shape
    inference accepts them.
    """
    if any(dim is None for dim in shape):
        op = current_builder().create(
            "broadcast", operands=(value,), attributes={"shape": tuple(shape)}
        )
        return op.result
    return ops.broadcast(_sym(value), tuple(shape)).value


def _row_mask(rows: Sequence[int], rank: int, n_functions: int) -> ir.Value:
    """A static bool constant selecting `rows` on the leading (function) axis.

    Shape ``(N,) + (1,) * (rank - 1)`` — one mask shape per emitted result rank
    (``rank`` includes the leading function axis; `rank` 0/1 both give
    ``(N,)``) so numpy broadcast against any ``(N, *per_row)`` operand aligns
    the function axis and never a per-row dim. Callers pass the rank of the
    ALIGNED (mapped) mux operand, i.e. ``1 + row_rank``.
    """
    shape = (n_functions,) + (1,) * max(rank - 1, 0)
    payload = np.zeros(shape, dtype=bool)
    payload[list(rows)] = True
    return ops.constant(core.tensor(payload)).value


def _align_switch_candidate(
    value: ir.Value,
    axes: MappedAxes,
    target_row_rank: int,
    n_functions: int,
    op_name: str,
    group_label: str,
) -> ir.Value:
    """Broadcast one switch candidate to the mapped ``(N, *(1,), *per_row)`` shape.

    A switch class is MAPPED, so every candidate carries the leading function
    axis before the `select` mux: an unmapped one is broadcast along a fresh
    leading axis, trailing size-one padding goes up to the largest per-row rank
    (the pointwise alignment convention), and a mapped candidate keeps its
    values (with equal per-row ranks both are returned unchanged, so the mapped
    path emits exactly the ops it always did). A non-leading-axes or
    non-broadcastable candidate raises `core.TransformError` — never a raw
    `ShapeError` from the `select`.
    """
    if axes.axes != tuple(range(axes.count)):
        raise core.TransformError(
            f"batch_functions: switch group {group_label!r} candidate of op "
            f"'{op_name}' carries non-leading batch axes {axes.axes} — "
            "batch_functions values are mapped on the leading function axis only"
        )
    row_shape = tuple(value.type.shape[axes.count:])
    target = (
        (n_functions,) + (1,) * (target_row_rank - len(row_shape)) + row_shape
    )
    if tuple(value.type.shape) == target:
        return value
    if len(value.type.shape) > len(target):
        raise core.TransformError(
            f"batch_functions: switch group {group_label!r} candidate of op "
            f"'{op_name}' has shape {tuple(value.type.shape)}, which cannot be "
            f"aligned to the mapped switch shape {target} — the variants of one "
            "switch class must agree on their per-row shape; drop that op from "
            "switch_ops"
        )
    try:
        return _broadcast(value, target)
    except core.ShapeError as exc:
        raise core.TransformError(
            f"batch_functions: switch group {group_label!r} candidate of op "
            f"'{op_name}' has shape {tuple(value.type.shape)}, which cannot be "
            f"broadcast to the mapped switch shape {target} — the variants of "
            "one switch class must have broadcast-compatible per-row shapes; "
            "drop that op from switch_ops"
        ) from exc


def _stamp_locations(new_block: ir.Block, start: int, location) -> None:
    """Give every op emitted from index `start` on the contributing op's location."""
    block_ops = new_block.ops
    for index in range(start, len(block_ops)):
        block_ops[index].location = location


def _run_rule(
    rule,
    op: ir.Op,
    operands: Tuple[ir.Value, ...],
    axes: Tuple[MappedAxes, ...],
    new_block: ir.Block,
) -> Tuple[Tuple[ir.Value, ...], Tuple[MappedAxes, ...]]:
    """Invoke one batching rule and validate/stamp its emission.

    Mirrors `batching._rewrite_op`: the ORIGINAL `ir.Op` is passed (rules read
    `op.attributes` / `op.operands[i].type`, i.e. the unbatched types) together
    with the NEW module's operand values; the rule's return value is validated
    and every emitted op inherits the original op's source location.
    """
    start = len(new_block.ops)
    new_values, new_axes = rule(op, operands, axes)
    new_values, new_axes = batching._validate_rule_result(
        op, new_values, new_axes
    )
    _stamp_locations(new_block, start, op.location)
    return new_values, new_axes


def _operand_axes(op: ir.Op, value: ir.Value, env: ValueEnv) -> MappedAxes:
    """The cached `MappedAxes` of one rewritten operand (strict)."""
    if value not in env:
        raise core.TransformError(
            "batch_functions: internal error — operand %"
            f"{value.id} of op '{op.name}' has no recorded batching metadata"
        )
    return env.get(value)


# --- emission ---------------------------------------------------------------


class _Merged:
    """Mutable state of one merge (kept out of the public surface).

    One instance drives ONE merge unit: the whole population in the flat merge,
    or one BAND of the banded merge. `n` is that unit's program count (the BAND
    size for a band — masks and stacked constants are unit-sized); the band
    hooks `band_args` (leaf index → band-local `ir.Value`, `_bands._BandInputs`)
    and `row_map` (band-local index → GLOBAL function index, for the user-facing
    `SwitchRecord.rows_per_variant`) stay `None` for the flat merge.
    """

    def __init__(
        self,
        graphs: Sequence[Graph],
        dag: MergeDAG,
        n_functions: int,
        groups: Sequence[Any],
        band_args: Any = None,
        row_map: Optional[Tuple[int, ...]] = None,
    ):
        self.graphs = graphs
        self.dag = dag
        self.n = n_functions
        self.groups = {group.key: group for group in groups}
        self.env = ValueEnv()
        self.emitted: Dict[int, Tuple[ir.Value, ...]] = {}
        self.source_locations: Dict[int, Any] = {}
        self.switched: List[SwitchRecord] = []
        self.stacked_constants = 0
        self.output_muxes = 0
        self.new_block: Optional[ir.Block] = None
        #: Band hooks (`None` = the flat merge).
        self.band_args = band_args
        self.row_map = tuple(row_map) if row_map is not None else None
        #: Source location of the `batch_functions` call, given to the helper ops
        #: the mux machinery emits itself (masks/selects/broadcasts) — they have
        #: no contributing original op to inherit a location from.
        self.call_site: Any = None

    # -- operand resolution --

    def global_rows(self, rows: Sequence[int]) -> Tuple[int, ...]:
        """Band-local program indices → GLOBAL function indices (identity when flat)."""
        if self.row_map is None:
            return tuple(rows)
        return tuple(self.row_map[row] for row in rows)

    def operand_value(self, program_index: int, value: ir.Value) -> ir.Value:
        """The merged-graph value standing for `value` of program `program_index`."""
        program = self.dag.programs[program_index]
        arg_index = program.arg_ids.get(id(value))
        if arg_index is not None:
            if self.band_args is not None:
                return self.band_args[arg_index]
            return self.new_block.arguments[arg_index]
        class_id, result_index = self.dag.slot_of(program_index, value)
        values = self.emitted.get(class_id)
        if values is None:
            raise core.TransformError(
                "batch_functions: internal error — class "
                f"{class_id} is emitted after its first use (emission order is "
                "not topological)"
            )
        return values[result_index]

    def set_emitted(
        self, info: ClassInfo, values: Tuple[ir.Value, ...], axes: Tuple[MappedAxes, ...]
    ) -> None:
        """Record a class's emitted values: env metadata + source locations."""
        self.env.update(values, axes)
        self.emitted[info.class_id] = values
        program_index, op = info.representative
        graph = self.graphs[program_index]
        for old_result, new_value in zip(op.results, values):
            location = graph.source_locations.get(old_result.id)
            if location is None:
                location = op.location
            if location is not None:
                self.source_locations[new_value.id] = location

    # -- per-class emission --

    def emit_constant(self, info: ClassInfo) -> None:
        """Emit a constant class: shared when all payloads agree, else stacked."""
        payloads: Dict[int, np.ndarray] = {}
        for program_index, op, _variant in info.contributors:
            payloads[program_index] = np.asarray(op.attributes["value"])
        first = min(payloads)
        reference = payloads[first]
        all_equal = all(
            np.array_equal(payload, reference) for payload in payloads.values()
        )
        start = len(self.new_block.ops)
        program_index, op = info.representative
        if all_equal:
            value = ops.constant(core.tensor(reference)).value
            axes = UNMAPPED
        else:
            rows = [payloads.get(i, reference) for i in range(self.n)]
            value = ops.constant(core.tensor(np.stack(rows, axis=0))).value
            axes = MappedAxes((0,))
            self.stacked_constants += 1
        _stamp_locations(self.new_block, start, op.location)
        self.env.set(value, axes)
        self.emitted[info.class_id] = (value,)
        graph = self.graphs[program_index]
        location = graph.source_locations.get(op.result.id)
        if location is None:
            location = op.location
        if location is not None:
            self.source_locations[value.id] = location

    def emit_class(self, info: ClassInfo) -> None:
        """Emit one non-constant class (single variant or switch mux)."""
        program_index, op = info.representative
        operands = tuple(
            self.operand_value(program_index, operand) for operand in op.operands
        )
        axes = tuple(_operand_axes(op, value, self.env) for value in operands)
        if len(info.variants) == 1:
            rule = _require_rule(op)
            values, out_axes = _run_rule(
                rule, op, operands, axes, self.new_block
            )
            self.set_emitted(info, values, out_axes)
            return
        group = self.groups[info.skeleton[0][1]]
        per_variant = []
        for variant in info.variants:
            variant_program, variant_op = info.representative_for(variant)
            rule = _require_rule(variant_op)
            values, out_axes = _run_rule(
                rule, variant_op, operands, axes, self.new_block
            )
            per_variant.append((variant_op, values, out_axes))
        reference_types = tuple(value.type for value in per_variant[0][1])
        for variant_op, values, _axes in per_variant[1:]:
            if tuple(value.type for value in values) != reference_types:
                raise core.TransformError(
                    "batch_functions: switch group "
                    f"{group.label!r} produced different result types across "
                    f"variants (op '{variant_op.name}' vs "
                    f"'{per_variant[0][0].name}') — a select mux needs "
                    "identical result types; drop that op from switch_ops"
                )
        # Everything from here on is emitted by the mux machinery itself (no
        # contributing original op) — it gets the `batch_functions` call site.
        helper_start = len(self.new_block.ops)
        # A switch class is ALWAYS mapped: every candidate is aligned to the
        # mapped (N, *(1,), *per_row) shape (unmapped candidates — shared inputs
        # — gain the leading function axis) and the mux result is MappedAxes((0,)).
        target_row_rank = max(
            value.type.rank - candidate_axes.count
            for _op, values, result_axes in per_variant
            for value, candidate_axes in zip(values, result_axes)
        )
        first_op, first_values, first_axes = per_variant[0]
        acc = [
            _align_switch_candidate(
                value, candidate_axes, target_row_rank, self.n,
                first_op.name, group.label,
            )
            for value, candidate_axes in zip(first_values, first_axes)
        ]
        for index, (variant_op, values, result_axes) in enumerate(
            per_variant[1:], start=1
        ):
            rows = info.rows[index]
            for result_index, (value, candidate_axes) in enumerate(
                zip(values, result_axes)
            ):
                aligned = _align_switch_candidate(
                    value, candidate_axes, target_row_rank, self.n,
                    variant_op.name, group.label,
                )
                mask = _row_mask(rows, aligned.type.rank, self.n)
                try:
                    acc[result_index] = ops.select(
                        _sym(mask), _sym(aligned), _sym(acc[result_index])
                    ).value
                except core.ShapeError as exc:
                    raise core.TransformError(
                        f"batch_functions: switch group {group.label!r} variants "
                        f"'{variant_op.name}' and '{first_op.name}' produced "
                        f"result shapes {tuple(aligned.type.shape)} and "
                        f"{tuple(acc[result_index].type.shape)} that one select "
                        "cannot mux — the variants of one switch class must "
                        "agree on their per-row shape; drop that op from "
                        "switch_ops"
                    ) from exc
        self.switched.append(
            SwitchRecord(
                group=group.label,
                variants=tuple(
                    variant_op.name for variant_op, _values, _axes in per_variant
                ),
                rows_per_variant=tuple(
                    self.global_rows(rows) for rows in info.rows
                ),
            )
        )
        _stamp_locations(self.new_block, helper_start, self.call_site)
        self.set_emitted(
            info, tuple(acc), (MappedAxes((0,)),) * len(acc)
        )

    # -- outputs --

    def emit_outputs(self, out_entries: Sequence[Optional[int]]) -> Tuple[ir.Value, ...]:
        """One merged value per tensor output leaf, in output-leaf order."""
        outputs = []
        helper_start = len(self.new_block.ops)
        leaf_count = len(self.dag.programs[0].terminator.operands)
        if len(out_entries) != leaf_count:
            raise core.TransformError(
                "batch_functions: internal error — "
                f"{len(out_entries)} out_axes entries for {leaf_count} tensor "
                "outputs"
            )
        for leaf_index in range(leaf_count):
            refs: List[tuple] = []
            rows_by_ref: Dict[tuple, List[int]] = {}
            value_by_ref: Dict[tuple, ir.Value] = {}
            for program_index, program in enumerate(self.dag.programs):
                value = program.terminator.operands[leaf_index]
                ref = self.dag.ref_of(program_index, value)
                if ref not in value_by_ref:
                    refs.append(ref)
                    rows_by_ref[ref] = []
                    value_by_ref[ref] = self.operand_value(program_index, value)
                rows_by_ref[ref].append(program_index)
            entry = out_entries[leaf_index]
            if len(refs) == 1:
                value = value_by_ref[refs[0]]
                if not self.env.get(value).is_mapped:
                    if entry is None:
                        outputs.append(value)
                        continue
                    value = _broadcast(value, (self.n,) + value.type.shape)
                    self.env.set(value, MappedAxes((0,)))
                elif entry is None:
                    raise core.TransformError(
                        f"batch_functions: out_axes=None at output leaf "
                        f"{leaf_index} requires the output to be unmapped, but "
                        "it carries a mapped function axis — a batch axis is "
                        "never silently dropped"
                    )
                outputs.append(value)
                continue
            if entry is None:
                raise core.TransformError(
                    f"batch_functions: out_axes=None at output leaf "
                    f"{leaf_index} requires the output to be unmapped, but the "
                    "output is muxed from "
                    f"{len(refs)} different function results — a batch axis is "
                    "never silently dropped"
                )
            self.output_muxes += 1
            aligned: List[Tuple[tuple, ir.Value]] = []
            for ref in refs:
                value = value_by_ref[ref]
                if not self.env.get(value).is_mapped:
                    value = _broadcast(value, (self.n,) + value.type.shape)
                    self.env.set(value, MappedAxes((0,)))
                aligned.append((ref, value))
            acc = aligned[0][1]
            for ref, value in aligned[1:]:
                mask = _row_mask(rows_by_ref[ref], acc.type.rank, self.n)
                acc = ops.select(_sym(mask), _sym(value), _sym(acc)).value
            # A muxed output is ALWAYS mapped (mapping is what makes row i take
            # function i's value); recording it keeps the env truthful for the
            # banded driver, which combines per-band output values.
            self.env.set(acc, MappedAxes((0,)))
            outputs.append(acc)
        _stamp_locations(self.new_block, helper_start, self.call_site)
        return tuple(outputs)


def _require_rule(op: ir.Op):
    """The registered batching rule for `op` (pre-flight validated; strict).

    Lookup goes through `batching._rule_key` so `block:<name>` /
    `external:<name>` namespacing works. `batch_functions` validates every op
    before emission, so a miss here is an internal error — never a silent
    fallback.
    """
    rule = batching.batching_rules.get(batching._rule_key(op))
    if rule is None:
        raise core.TransformError(
            f"batch_functions: no batching rule for op '{op.name}' (key "
            f"'{batching._rule_key(op)}') — v1 deferral of batch_functions"
        )
    return rule


def new_merged_module(
    graph0: Graph,
    entries: Sequence[Optional[int]],
    n_functions: int,
) -> Tuple[ir.Builder, ir.Module, ir.Block, List[core.TensorSpec], List[MappedAxes]]:
    """Build the merged module/function skeleton (steps 1-2 of a merge).

    Shared by the flat and the banded driver (`_bands.emit_banded_graph`). A
    mapped input leaf gains a STATIC leading `n_functions` dim (never a symbolic
    `Dim`); unmapped leaves are reused unchanged. Returns
    ``(builder, module, new_block, new_specs, input_axes)``.
    """
    new_specs = []
    input_axes = []
    for spec, entry in zip(graph0.tensor_specs, entries):
        if entry is None:
            new_specs.append(spec)
            input_axes.append(UNMAPPED)
            continue
        new_specs.append(
            core.TensorSpec(
                shape=(n_functions, *spec.shape),
                dtype=spec.dtype,
                device=spec.device,
                name=spec.name,
            )
        )
        input_axes.append(MappedAxes((0,)))

    builder = ir.Builder()
    module = builder.build_module(
        name=graph0.module.name, metadata=dict(graph0.module.metadata)
    )
    input_types = tuple(
        ir.ValueType(spec.dtype, tuple(spec.shape)) for spec in new_specs
    )
    function = builder.build_function(
        name=graph0.module.functions[0].name,
        input_types=input_types,
        metadata=dict(graph0.module.functions[0].metadata),
    )
    return builder, module, function.entry_block, new_specs, input_axes


def finish_merged_graph(
    graph0: Graph,
    builder: ir.Builder,
    new_block: ir.Block,
    new_specs: Sequence[core.TensorSpec],
    output_values: Sequence[ir.Value],
    source_locations: Dict[int, Any],
) -> Graph:
    """Terminate the merged entry block and wrap it into a new `Graph`.

    The `return` inherits `graph0`'s location; trees/static values are carried
    over verbatim. The caller attaches its own `BatchReport`.
    """
    builder.set_terminator(
        new_block,
        "return",
        operands=output_values,
        location=graph0.module.functions[0].entry_block.terminator.location,
    )
    return Graph(
        builder.module,
        graph0.input_specs,
        new_specs,
        graph0.output_tree,
        graph0.static_values,
        graph0.output_static_values,
        source_locations,
    )


def emit_merged_graph(
    graphs: Sequence[Graph],
    dag: MergeDAG,
    entries: Sequence[Optional[int]],
    out_entries: Sequence[Optional[int]],
    groups: Sequence[Any],
) -> Graph:
    """Build the merged `Graph` (and attach `graph.batch_report`) — the FLAT merge.

    `graphs` are the traced programs, `dag` their class DAG, `entries`
    (`out_entries`) one ``0``/``None`` per TENSOR input (output) leaf — ``0``
    makes that leaf's leading axis the function axis, ``None`` leaves it
    unmapped/shared — and `groups` the `switch_ops` merge groups. Raises
    `core.TransformError` for conflicting result types inside one switch group,
    `out_axes=None` on a mapped/muxed output, or an internal violation.
    """
    graph0 = graphs[0]
    n_functions = len(graphs)

    # 1-2. New input specs + the new module/function (inputs never mutated).
    builder, _module, new_block, new_specs, input_axes = new_merged_module(
        graph0, entries, n_functions
    )

    # 3. Seed the metadata env for the new block args and carry the input
    #    source locations forward (every program shares the same arg layout).
    state = _Merged(graphs, dag, n_functions, groups)
    state.new_block = new_block
    for old_arg, new_arg, axes in zip(
        dag.programs[0].block.arguments, new_block.arguments, input_axes
    ):
        state.env.set(new_arg, axes)
        location = graph0.source_locations.get(old_arg.id)
        if location is not None:
            state.source_locations[new_arg.id] = location

    # 4. Emit every reachable class in emission order, then the outputs.
    #    Location capture is suppressed for the whole emission: a rule-emitted
    #    op is re-stamped with its contributing ORIGINAL op's location
    #    (`_stamp_locations`), the helper ops the mux machinery emits itself
    #    are stamped with the ONE call site captured here, and
    #    `graph.source_locations` is carried from the original graphs — so the
    #    per-op `inspect.stack()` walk of `ops` would be pure overhead.
    state.call_site = get_location()
    with _suppress_location_capture(), with_builder(builder):
        for class_id in dag.reachable:
            info = dag.classes[class_id]
            if info.representative[1].name == "constant":
                state.emit_constant(info)
            else:
                state.emit_class(info)
        output_values = state.emit_outputs(out_entries)

    # 5. Terminate and wrap into a new Graph.
    graph = finish_merged_graph(
        graph0,
        builder,
        new_block,
        new_specs,
        output_values,
        state.source_locations,
    )
    shared = sum(
        1
        for class_id in dag.reachable
        if len(dag.classes[class_id].program_indices) >= 2
    )
    graph.batch_report = BatchReport(
        n_functions=n_functions,
        value_classes=len(dag.reachable),
        shared_classes=shared,
        switched=tuple(state.switched),
        stacked_constants=state.stacked_constants,
        output_muxes=state.output_muxes,
        emitted_ops=len(new_block.ops) - 1,
        bands=1,
        band_sizes=(n_functions,),
        band_strategy="flat",
    )
    return graph
