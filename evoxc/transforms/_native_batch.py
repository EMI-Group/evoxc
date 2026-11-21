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
* the merge config passes through untouched — a callable
  ``config["extension_rule"]`` is never encoded.
The native merge owns rank-0 mapped outputs directly: a mapped ``out_axes=0``
on a rank-0 per-function output becomes rank 1 in the merged graph (the static
function axis is prepended), exactly the "broadcast to the function axis" rule
`vmap` applies.
The result is ``{"module": ir.Module, "batch_report": {...}}``, plus an
``"input_ids"`` list when the extension reports one (the merged entry
function's block-argument ids, in input order).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from evoxc import _native, ir
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
    return _merge(modules, config)


def _merge(
    programs: Sequence[Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """One façade merge: modules (or envelopes) in, decoded module + report out."""
    assembled = _native.batch_functions(list(programs), dict(config))
    assembled["batch_report"] = _decode_report(assembled["batch_report"])
    return assembled
