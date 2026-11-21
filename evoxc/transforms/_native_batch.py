"""Typed façade over ``evoxc._evoxc_native.batch_functions``.

The `batch_functions` merge itself runs in the native engine
(``native/evoxc-compiler/src/batch_functions/``, reached through the PyO3 entry
point); this module adds the config bridging the public contract needs on top
of :func:`evoxc._native.batch_functions`, which owns the envelope encode, the
extension call, the ``NativeError.kind`` translation and the trusted fast
decode of the merged envelope:
* the extension's 12-key ``"batch_report"`` payload is validated and returned
  as the plain dict :func:`evoxc.transforms._report.report_from_native`
  rehydrates;
* a rank-0 function-axis output is bridged (below);
* the merge config passes through untouched — a callable
  ``config["extension_rule"]`` is never encoded.
The result is ``{"module": ir.Module, "batch_report": {...}}``, plus an
``"input_ids"`` list when the extension reports one (the merged entry
function's block-argument ids, in input order).

Rank-0 function-axis outputs (the one config gap this seam bridges): the public
contract makes ``out_axes=0`` on a rank-0 unmapped output the legal "broadcast
to the function axis" case (exactly the rule `vmap` applies), but the native
merge front-end range-checks every mapped ``out_axes`` entry against the
DECLARED per-function rank, so ``0`` on a rank-0 output is rejected before the
(broadcast-capable) output stage runs.  The bridge is tried in two tiers, both
adding the function axis as an ORDINARY op after the merge (never per program,
so the merge's class/op counts stay exactly the merge of the given programs):

1. **Cheap tier** — the programs are passed UNTOUCHED and those leaves ask for
   an UNMAPPED output (``out_axes=None``); native accepts that whenever the
   merged value genuinely carries no function axis, and ONE post-merge
   `broadcast` to ``(N,)`` adds it.  This is the legacy emitter's exact
   composition, so counts are unchanged.
2. **Wrapped tier** — when native instead reports that such a leaf *is* mapped
   (the mapped-input case: a per-program reduction to a scalar), each program's
   output is expressed as a rank-1 ``(1,)`` value — one `reshape` inserted
   before that program's ``return`` in a decoded COPY (the caller's modules are
   never mutated) — and the merged ``(N, 1)`` output is unwrapped to ``(N,)``.
   When all programs collapse into one shared output class the merged value is
   exactly the wrapper's `reshape`, so that op is stripped again (the reported
   graph is the legacy one); only a muxed output keeps its wrapper ops and then
   inserts an unwrap `reshape`.
   Either way the wrapper classes are scaffold, never part of the requested
   merge, so the reported ``value_classes`` / ``shared_classes`` /
   ``emitted_ops`` are discounted by one per rank-0 leaf.
Nothing else about the merge changes: the edit is the same rank bookkeeping the
frontend already does for specs.
"""

from __future__ import annotations

import collections.abc
from typing import Any, Mapping, Sequence

from evoxc import _native, _native_codec, core, ir
from evoxc.transforms._report import REPORT_KEYS

__all__ = ["batch_functions", "REPORT_KEYS"]


def _decode_report(raw: Any) -> dict[str, Any]:
    """Validate the extension's ``"batch_report"`` payload (a plain mapping)."""
    if not isinstance(raw, Mapping):
        raise TypeError(
            "the native batch_functions result 'batch_report' must be a "
            f"mapping, got {type(raw).__name__}"
        )
    missing = [key for key in REPORT_KEYS if key not in raw]
    if missing:
        raise TypeError(
            "the native batch_functions 'batch_report' is missing key(s) "
            + ", ".join(repr(key) for key in missing)
        )
    return dict(raw)


def _rank0_function_axis_leaves(
    modules: Sequence[ir.Module], out_entries: Any
) -> list[int]:
    """Output leaves native cannot express: rank-0 under ``out_axes=0``.

    A missing / non-list ``out_axes`` is native's "every output mapped" default,
    so every rank-0 output leaf then qualifies.
    """
    if not modules:
        return []
    declared = modules[0].main.output_types
    if isinstance(out_entries, (list, tuple)):
        entries = list(out_entries)
        entries.extend([0] * (len(declared) - len(entries)))
    else:
        entries = [0] * len(declared)
    return [
        leaf_index
        for leaf_index, (value_type, entry) in enumerate(zip(declared, entries))
        if entry is not None and len(value_type.shape) == 0
    ]


def _insert_output_op(
    module: ir.Module, leaf_index: int, name: str, attributes: dict[str, Any]
) -> None:
    """Apply one op to output ``leaf_index`` inside ``module``.

    The op (``reshape`` / ``broadcast``) is emitted with an `ir.Builder` (schema
    and type validation, shape inference and use wiring for free) and then
    placed immediately before the ``return`` terminator — the builder always
    inserts at the block start, so the op is detached and re-inserted at the
    tail.  Only that one terminator operand is retargeted
    (`Op._replace_operand`, the IR's internal RAUW primitive — a full
    `replace_all_uses_with` would also rewrite unrelated consumers).
    """
    function = module.main
    block = function.entry_block
    terminator = block.ops[-1]
    value = terminator.operands[leaf_index]
    builder = ir.Builder(module)
    builder.push_region(function.region)  # entry block, insertion position 0
    try:
        op = builder.create(name, operands=(value,), attributes=attributes)
    finally:
        builder.pop_region()
    block.erase(op)  # detach from position 0
    block.insert(len(block.ops) - 1, op)  # …and place it just before `return`
    terminator._replace_operand(leaf_index, op.result)


def _wrap_program_envelope(envelope: Mapping, leaves: Sequence[int]) -> dict:
    """Re-encode a COPY of ``envelope`` with every leaf wrapped to ``(1,)``."""
    module = _native_codec.decode_module(dict(envelope))
    for leaf_index in leaves:
        _insert_output_op(module, leaf_index, "reshape", {"shape": (1,)})
    return _native_codec.encode_module(module)


def _out_entries(config: Mapping, modules: Sequence[ir.Module]) -> list:
    """The config's ``out_axes`` as one entry per tensor output leaf."""
    declared = modules[0].main.output_types
    raw = config.get("out_axes")
    if isinstance(raw, (list, tuple)):
        entries = list(raw)
        entries.extend([0] * (len(declared) - len(entries)))
        return entries
    return [0 if raw is None else raw] * len(declared)


def _rank0_unmapped_rejection(exc: BaseException) -> bool:
    """True for native's "this rank-0 output is actually mapped" rejection.

    Only that one message re-tries the merge in the wrapped tier; every other
    ``TransformError`` (a missing batching rule, a bad program shape, …) is the
    caller's to see.
    """
    return "requires the output to be unmapped" in str(exc)


def batch_functions(
    modules: Sequence[ir.Module], config: dict[str, Any]
) -> dict[str, Any]:
    """Run the native `batch_functions` merge over `modules`.

    Args:
        modules: The traced programs (one `ir.Module` per function of the
            population), in function-axis order.
        config: The merge config: ``axes`` (one ``int``/``None`` per tensor
            input leaf), ``out_axes`` (one per tensor output leaf),
            ``switch_ops``, ``bands``, ``alignment`` and the optional
            ``extension_rule`` callable.

    Returns:
        ``{"module": ir.Module, "batch_report": dict}`` — the merged program
        and the 12-key report payload. ``"input_ids"`` is included when the
        extension reports the merged block-argument ids.

    Raises:
        TypeError: `modules` / `config` have the wrong shape, or the extension's
            result is not the documented mapping.
        evoxc.core.EVOXCError: any native merge failure, mapped onto the public
            error family by ``NativeError.kind`` (message verbatim).
    """
    if not isinstance(config, dict):
        raise TypeError(
            f"batch_functions: 'config' must be a dict, got {type(config).__name__}"
        )
    rank0_leaves = _rank0_function_axis_leaves(modules, config.get("out_axes"))
    if not rank0_leaves:
        return _merge(modules, config)
    n_functions = len(modules)
    # Cheap tier: leave the programs untouched and ask for an UNMAPPED rank-0
    # output, then add the function axis with ONE post-merge `broadcast`. A
    # BANDED merge rejects `out_axes=None` outright (a per-band value cannot be
    # a single unmapped output), so it goes straight to the wrapped tier.
    cheap_config = dict(config)
    cheap_entries = _out_entries(config, modules)
    for leaf_index in rank0_leaves:
        cheap_entries[leaf_index] = None
    cheap_config["out_axes"] = cheap_entries
    try:
        assembled = (
            None if config.get("bands") is not None else _merge(modules, cheap_config)
        )
    except core.TransformError as exc:
        if not _rank0_unmapped_rejection(exc):
            raise
    else:
        if assembled is not None:
            module = assembled["module"]
            for leaf_index in rank0_leaves:
                if len(module.main.output_types[leaf_index].shape) == 0:
                    _insert_output_op(
                        module, leaf_index, "broadcast", {"shape": (n_functions,)}
                    )
            return assembled
    # Wrapped tier: the merged values ARE mapped, so native cannot keep a rank-0
    # output at all — express each program's output as a `(1,)` row and unwrap
    # the merged `(N, 1)` back to the documented `(N,)` function axis.
    wrapped = [
        _wrap_program_envelope(_native_codec.encode_module(module), rank0_leaves)
        for module in modules
    ]
    assembled = _merge(wrapped, config)
    module = assembled["module"]
    for leaf_index in rank0_leaves:
        if not _strip_rank0_wrapper(module, leaf_index):
            _insert_output_op(module, leaf_index, "reshape", {"shape": (n_functions,)})
    _discount_wrapper_classes(assembled["batch_report"], modules, rank0_leaves)
    return assembled


def _merge(
    programs: Sequence[Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """One façade merge: modules (or envelopes) in, decoded module + report out."""
    assembled = _native.batch_functions(list(programs), dict(config))
    assembled["batch_report"] = _decode_report(assembled["batch_report"])
    return assembled


def _strip_rank0_wrapper(module: ir.Module, leaf_index: int) -> bool:
    """Undo the `(1,)` wrapper residual of ONE output leaf, in place.
    When every program's output collapses into one shared class (the common
    case) the merged `(N, 1)` value is exactly the wrapper `reshape` of the
    merged rank-0 value, so replacing the terminator operand and erasing the
    `reshape` restores the legacy graph with no extra op and no unwrap.
    A muxed output (one wrapper per divergent class, feeding `select`s) cannot
    be undone this way; returns ``False`` and the caller inserts an unwrap.
    """
    block = module.main.entry_block
    terminator = block.ops[-1]
    op = terminator.operands[leaf_index].owner
    if op is None or op.name != "reshape":
        return False
    if tuple(op.attributes.get("shape", ())) != (1,):
        return False
    inner = op.operands[0]
    if len(inner.type.shape) != 0:
        return False
    terminator._replace_operand(leaf_index, inner)
    block.erase(op)
    return True


def _discount_wrapper_classes(
    report: dict[str, Any], modules: Sequence[ir.Module], leaves: Sequence[int]
) -> None:
    """Report the WRAPPER-FREE merge the caller asked for.
    The scaffold above exists only to satisfy native's declared-rank check, so
    its classes are not part of the documented merge: a stripped leaf's single
    shared wrapper class is subtracted from ``value_classes``/``shared_classes``
    and `emitted_ops`.  A muxed leaf keeps its scaffolding ops in the graph and
    its class is likewise discounted (its shared-ness is left as native
    reports it).
    """
    n_functions = len(modules)
    for leaf_index in leaves:
        report["value_classes"] = max(0, report["value_classes"] - 1)
        report["emitted_ops"] = max(0, report["emitted_ops"] - 1)
        if n_functions >= 2:
            report["shared_classes"] = max(0, report["shared_classes"] - 1)
