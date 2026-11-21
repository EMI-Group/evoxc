"""The public `batch_functions` merge-report types.

`BatchReport` / `SwitchRecord` describe what one merge did; the merged `Graph`
carries them as the plain attribute ``graph.batch_report`` (documented transform
metadata — never part of the persisted payload, so a graph loaded back has no
`batch_report`).

The merge itself runs in the native engine
(``evoxc._evoxc_native.batch_functions``), which returns the report as its
12-key JSON payload; :func:`report_from_native` rehydrates that payload into
these dataclasses (tuples for `band_sizes` / `variants` /
`rows_per_variant`, and `rows_per_variant` holding GLOBAL function indices —
also under banding).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

__all__ = ["BatchReport", "SwitchRecord", "report_from_native"]

#: The exact key set of the native report payload (and of `as_dict()`).
REPORT_KEYS: Tuple[str, ...] = (
    "n_functions",
    "value_classes",
    "shared_classes",
    "switched",
    "stacked_constants",
    "output_muxes",
    "emitted_ops",
    "bands",
    "band_sizes",
    "band_strategy",
    "cost_estimate",
    "flat_cost_estimate",
)


@dataclass(frozen=True)
class SwitchRecord:
    """One switch class emitted by `batch_functions`.

    Attributes:
        group: The merge group's name — the IR category name for a category
            group, ``",".join(sorted(names))`` for an explicit op-name set.
        variants: The variant names in first-appearance order (an op's IR
            name; two variants differing only in attributes share a name).
        rows_per_variant: The function indices using each variant, aligned
            with `variants`; these are GLOBAL function indices, also when the
            merge is banded.
    """

    group: str
    variants: Tuple[str, ...]
    rows_per_variant: Tuple[Tuple[int, ...], ...]


@dataclass
class BatchReport:
    """What one `batch_functions` merge actually did (the `batch_report`).

    Attributes:
        n_functions: The number of merged functions (the function-axis length).
        value_classes: The number of REACHABLE computation classes emitted.
        shared_classes: How many of those were contributed by ≥2 functions.
        switched: The switch classes, in emission order.
        stacked_constants: How many constants were stacked into a per-row
            constant (payloads differed across functions).
        output_muxes: How many outputs were muxed from ≥2 distinct producing
            classes.
        emitted_ops: Ops in the merged entry block minus the `return`
            terminator.
        bands: The number of merge bands (1 for the flat merge and for a
            single-band merge).
        band_sizes: The band sizes, in band order (`(N,)` for a single-band
            merge).
        band_strategy: `"flat"` for `bands=None`, `"explicit"` for an int
            ``K``, `"auto"` for the automatic partition.
        cost_estimate: The static cost model's estimate for the CHOSEN
            partition (``None`` unless the model ran, i.e. ``bands="auto"``).
        flat_cost_estimate: The same model's estimate for the flat partition
            (``None`` unless the model ran).

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


def report_from_native(payload: Mapping[str, Any]) -> BatchReport:
    """Rehydrate the native engine's 12-key report payload into a `BatchReport`.

    The native side emits the exact `BatchReport::to_json` shape (JSON lists for
    `band_sizes` / `variants` / `rows_per_variant`); this converts them back to
    tuples so the Python object is identical to what the transform has always
    returned.

    Raises:
        TypeError: `payload` is not a mapping with exactly the expected keys.
    """
    if not isinstance(payload, Mapping):
        raise TypeError(
            "the native batch_functions result 'batch_report' must be a "
            f"mapping, got {type(payload).__name__}"
        )
    missing = [key for key in REPORT_KEYS if key not in payload]
    if missing:
        raise TypeError(
            "the native batch_functions 'batch_report' is missing key(s) "
            + ", ".join(repr(key) for key in missing)
        )
    switched = tuple(
        SwitchRecord(
            group=record["group"],
            variants=tuple(record["variants"]),
            rows_per_variant=tuple(tuple(rows) for rows in record["rows_per_variant"]),
        )
        for record in payload["switched"]
    )
    cost_estimate = payload["cost_estimate"]
    flat_cost_estimate = payload["flat_cost_estimate"]
    return BatchReport(
        n_functions=payload["n_functions"],
        value_classes=payload["value_classes"],
        shared_classes=payload["shared_classes"],
        switched=switched,
        stacked_constants=payload["stacked_constants"],
        output_muxes=payload["output_muxes"],
        emitted_ops=payload["emitted_ops"],
        bands=payload["bands"],
        band_sizes=tuple(payload["band_sizes"]),
        band_strategy=payload["band_strategy"],
        cost_estimate=None if cost_estimate is None else float(cost_estimate),
        flat_cost_estimate=(
            None if flat_cost_estimate is None else float(flat_cost_estimate)
        ),
    )
