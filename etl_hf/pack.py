"""Opt-in **packed weights**: a handful of concatenated graph inputs, sliced in-graph.

Why this module exists
----------------------
A whole ``etl_hf`` model forward declares one graph input per canonical weight
(196 for the 124M ``gpt2``, plus the ``input_ids`` leaf = **197 inputs**). IREE's
Python runtime SIGSEGVs when a program *consumes* more than ~161 distinct inputs
(measured bracket: 161 works, 162 crashes — see ``./etl_hf/CONTEXT.md``), so the
124M GPT-2 compiles fine and then dies at the first run for reasons that have
nothing to do with its graph.

Packing removes that cliff without touching the graphs: same-role weight tensors
are concatenated along axis 0 into a few fat graph inputs, and the graph slices
them back with **static** offsets (``etl.slice`` with static starts/lengths — no
dynamic slice, no new IR op). The 197-input graph becomes a ~5–8 input graph, the
weights stay ordinary graph inputs (never ``etl.constant``), and nothing about
the arithmetic changes: a static slice of a concatenation is exact.

This module is deliberately **generic over a ``param_specs`` dict** — it knows
nothing about llama or gpt2 (no family-specific packing code anywhere):

=====================  =====================================================
:func:`pack_plan`      canonical ``TensorSpec`` tree → :class:`PackPlan`
:meth:`PackPlan.pack`  canonical host arrays → packed host arrays
:meth:`PackPlan.unpack` packed ``SymbolicTensor`` dict → canonical dict (in-graph)
:func:`pack_params`    convenience wrapper for ``PackPlan.pack``
=====================  =====================================================

Bucketing rule
--------------
Canonical keys are grouped by ``(dtype, rank, shape[1:])``: any two arrays whose
*dtype* and *trailing* dims match can be concatenated along axis 0, so they share
one packed input of shape ``(sum(axis0),) + shape[1:]``. Buckets are ordered
deterministically (by the bucket's minimum canonical key) and named
``f"{prefix}{i}"``; keys inside a bucket are sorted and their axis-0 offsets are
prefix sums. A rank-0 *scalar* key cannot be concatenated, so it gets its own
single-key bucket of shape ``(1,)`` and is un-packed as a slice + reshape to
``()``.

Derived (transposed) entries
----------------------------
A tied-embedding config has ONE ``embed.weight`` ``(vocab, hidden)`` graph input
that also serves as the output projection, whose ``(in_features,
out_features)`` layout is ``(hidden, vocab)``. Transposing the whole table
inside the graph would materialise it on every call, so the plan declares that
copy instead — inferred from the spec tree by
:func:`etl_hf._specs.transposed_aliases` (a tree with an ``embed.weight`` and no
``lm_head.weight`` is a tied tree), or passed explicitly as
``pack_plan(..., transposed_aliases=...)`: the plan stores
``np.ascontiguousarray(params["embed.weight"].T)`` **once** as its own packed
entry under the derived key ``lm_head.weight``, and ``unpack`` slices it with the
same static :class:`~etl_hf.pack.PackEntry` (``transposed=True``,
``source="embed.weight"``, ``shape=(hidden, vocab)``). Packing therefore carries
one extra weight-sized buffer; the graph carries no weight-sized transpose.
Derived entries never appear in ``params`` — the host tree is still exactly the
family's canonical ``canonicalize_weights`` tree.

Import-time constraint: only ``numpy`` + ``etl`` + :mod:`etl_hf.errors` +
:mod:`etl_hf._specs` are imported here (no torch/transformers/iree), so
``import etl_hf`` stays numpy-only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

import etl
from etl_hf._specs import transposed_aliases as _declared_aliases
from etl_hf.errors import HFUnsupportedError, HFWeightError

__all__ = [
    "PACKED_PREFIX",
    "PackEntry",
    "PackPlan",
    "is_packed_tree",
    "pack_params",
    "pack_plan",
]

#: Default name prefix of the packed graph inputs (``"packed.0"``, ``"packed.1"``…).
PACKED_PREFIX: str = "packed."


# ---------------------------------------------------------------------------
# validation helpers (every failure names the key and the reason)
# ---------------------------------------------------------------------------
def _static_shape(name: str, spec: object) -> tuple[int, ...]:
    """Return a spec's shape as a tuple of static ints.

    Args:
        name: the canonical parameter key (used in the error message).
        spec: an :class:`etl.TensorSpec` (anything with a ``.shape`` tuple).

    Returns:
        tuple[int, ...]: the static shape.

    Raises:
        HFUnsupportedError: a dim is symbolic (``Dim``/``DimExpr``) or a runtime
            dynamic ``None`` — the in-graph unpack is a static ``etl.slice``, so
            a packed input must be fully static.
    """
    shape = getattr(spec, "shape", None)
    if shape is None:
        raise TypeError(
            f"pack_plan: parameter {name!r} is not an etl.TensorSpec (no .shape "
            f"attribute; got {type(spec).__name__})"
        )
    static: list[int] = []
    for axis, dim in enumerate(tuple(shape)):
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise HFUnsupportedError(
                f"pack_plan: parameter {name!r} has the non-static dim {dim!r} on "
                f"axis {axis} (shape {tuple(shape)}); packed weights need fully "
                f"static shapes because the in-graph unpack is a static etl.slice "
                f"(build the graph for a concrete seq_len/batch_size instead)"
            )
        if dim < 0:
            raise HFUnsupportedError(
                f"pack_plan: parameter {name!r} has a negative dim {dim} on axis "
                f"{axis} (shape {tuple(shape)})"
            )
        static.append(int(dim))
    return tuple(static)


def _bucket_key(name: str, shape: tuple[int, ...], dtype: np.dtype) -> tuple:
    """The bucketing key of one canonical key: ``(dtype, rank, shape[1:])``.

    A rank-0 scalar cannot be concatenated, so it is keyed by its own name — one
    single-key bucket per scalar.

    Args:
        name: the canonical parameter key.
        shape: the canonical tensor's static shape.
        dtype: the canonical tensor's dtype.

    Returns:
        tuple: hashable bucket key.
    """
    if len(shape) == 0:
        return ("scalar", name)
    return (np.dtype(dtype).str, len(shape), tuple(shape[1:]))


def _as_array(name: str, value: object, want_dtype: np.dtype) -> np.ndarray:
    """Convert + validate one canonical host array for packing.

    Args:
        name: the canonical parameter key (used in every error message).
        value: the caller's value (host array-like).
        want_dtype: the dtype the packed input carries — a mismatch is an error,
            never a silent upcast (``np.concatenate`` would upcast).

    Returns:
        np.ndarray: the validated array (no copy, no dtype change).

    Raises:
        HFWeightError: the value is not a numeric array, or its dtype differs
            from the packed input's dtype (e.g. a device ``Tensor``, or a
            float64 tree where the specs are float32).
    """
    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        raise HFWeightError(
            f"pack_params: parameter {name!r} is not a host numeric array (dtype "
            f"{array.dtype}, from {type(value).__name__}); packing concatenates "
            f"host numpy data — pack on the host first, then place the PACKED "
            f"tree with etl_hf.place_params(...) for a device-resident run"
        )
    if np.dtype(array.dtype) != np.dtype(want_dtype):
        raise HFWeightError(
            f"pack_params: parameter {name!r} has dtype {array.dtype}, but its "
            f"packed input is {np.dtype(want_dtype)}; dtypes are never silently "
            f"cast (concatenation would upcast the whole bucket)"
        )
    return array


def _transposed_source(source: str, array: np.ndarray, entry: "PackEntry") -> np.ndarray:
    """Validate and transpose one derived entry's source array (host side).

    Args:
        source: the canonical key the derived entry is built from (error text).
        array: the validated host array read from ``params[source]``.
        entry: the derived :class:`PackEntry` (its ``shape`` is the **stored**,
            transposed shape).

    Returns:
        np.ndarray: ``array.T`` as a fresh C-contiguous array.

    Raises:
        HFWeightError: ``array`` is not the reverse of the declared stored shape
            (i.e. the caller passed a tree packed for another config).
    """
    expected = tuple(reversed(entry.shape))
    if tuple(array.shape) != expected:
        raise HFWeightError(
            f"PackPlan.pack: parameter {source!r} feeds the derived packed entry "
            f"stored as {entry.shape} (a transposed copy), so its shape must be "
            f"{expected}, got {tuple(array.shape)}"
        )
    return np.ascontiguousarray(array.T)


# ---------------------------------------------------------------------------
# the plan value objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PackEntry:
    """One key's position inside a packed graph input.

    Attributes:
        input: the packed input name, e.g. ``"packed.3"``.
        start: the static ``etl.slice`` start tuple — zeros except axis 0 (the
            concatenation axis). A rank-0 key's packed input is rank 1 of length
            1, so its ``start`` holds that single offset while ``shape`` is
            ``()``.
        shape: the static shape of what the slice yields — the canonical tensor's
            shape, or the **transposed** shape for a derived entry (whose
            ``source`` array is stored as ``source.T``).
        source: the canonical ``params`` key whose data this entry stores, or
            ``None`` when the entry's own key (its name in
            :attr:`PackPlan.entries`) is the source — i.e. the usual case.
            A non-``None`` ``source`` marks a **derived** entry: an extra packed
            copy that ``params`` never contains and only ``unpack`` produces
            (a tied-embedding head stored pre-transposed, so the graph never
            transposes a weight-sized operand per call).
        transposed: whether the stored data is ``params[source].T`` — always
            ``True`` for a derived entry, ``False`` for every canonical one.
    """

    input: str
    start: tuple[int, ...]
    shape: tuple[int, ...]
    source: str | None = None
    transposed: bool = False

    @property
    def canonical(self) -> bool:
        """Whether ``params`` supplies this entry directly (not derived)."""
        return self.source is None


@dataclass(frozen=True)
class PackPlan:
    """A canonical params tree ⇄ packed graph inputs mapping (family-agnostic).

    Attributes:
        specs: the **packed** params graph inputs (a handful of
            :class:`etl.TensorSpec` values, e.g. ``{"packed.0": …, …}``).
        entries: key → :class:`PackEntry` (its packed input and static slice
            position) for every key the plan unpacks — the canonical keys
            (``entry.canonical``) plus any derived, pre-transposed copies
            (:attr:`derived_keys`).
    """

    specs: dict[str, etl.TensorSpec]
    entries: dict[str, PackEntry]

    # -- introspection ------------------------------------------------------
    @property
    def canonical_keys(self) -> tuple[str, ...]:
        """The canonical parameter keys :meth:`pack` takes as input, sorted."""
        return tuple(sorted(name for name, entry in self.entries.items() if entry.canonical))

    @property
    def derived_keys(self) -> tuple[str, ...]:
        """The derived keys :meth:`unpack` yields on top of the canonical ones, sorted.

        Each is stored as the transposed copy of its :attr:`PackEntry.source`
        (empty for a config that needs none, e.g. an untied head).
        """
        return tuple(sorted(name for name, entry in self.entries.items() if not entry.canonical))

    @property
    def unpacked_keys(self) -> tuple[str, ...]:
        """Every key :meth:`unpack` returns (canonical + derived), sorted."""
        return tuple(sorted(self.entries))

    def has(self, name: str) -> bool:
        """Whether :meth:`unpack` yields ``name`` (canonical or derived).

        Args:
            name: a key to look up.

        Returns:
            bool: ``True`` when the plan carries that key.
        """
        return name in self.entries

    @property
    def input_names(self) -> tuple[str, ...]:
        """The packed graph input names, in packing order."""
        return tuple(sorted(self.specs, key=_name_index))

    def names_by_input(self) -> dict[str, tuple[str, ...]]:
        """Group the canonical keys by packed input.

        Returns:
            dict[str, tuple[str, ...]]: packed input name → canonical keys sorted
            by their axis-0 offset (i.e. concatenation order).
        """
        grouped: dict[str, list[str]] = {name: [] for name in self.specs}
        for name, entry in self.entries.items():
            grouped.setdefault(entry.input, []).append(name)
        return {
            request: tuple(sorted(names, key=lambda key: self.entries[key].start[0]))
            for request, names in grouped.items()
        }

    def __len__(self) -> int:
        """Number of entries the plan unpacks (canonical + derived keys)."""
        return len(self.entries)

    # -- host packing -------------------------------------------------------
    def pack(self, params: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Concatenate a canonical host params tree into the packed tree.

        One ``np.concatenate`` per packed input, in deterministic order — no
        per-call cost once the packed tree is reused (pack at build/params time,
        never per run). A derived entry (a tied head, :attr:`PackEntry.source`)
        is written as the **transposed** copy of its source array, once, here on
        the host.

        Args:
            params: canonical params (the exact keys of the family's
                ``param_specs`` — the canonical tree, never including derived
                keys), host arrays.

        Returns:
            dict[str, np.ndarray]: the packed tree (the exact keys of
            :attr:`specs`), each array C-contiguous.

        Raises:
            TypeError: ``params`` is not a mapping.
            HFWeightError: a key is missing/extra, a value is not a numeric host
                array, or a dtype/shape disagrees with the plan.
        """
        if not isinstance(params, Mapping):
            raise TypeError(
                f"PackPlan.pack expects a mapping of canonical name -> host array, "
                f"got {type(params).__name__}"
            )
        expected = set(self.canonical_keys)
        got = set(params)
        if got != expected:
            raise HFWeightError(
                f"PackPlan.pack: params do not match this plan's canonical keys "
                f"(missing={sorted(expected - got)[:8]}, "
                f"extra={sorted(got - expected)[:8]}); the plan is built from a "
                f"family's param_specs(cfg) — pass that exact tree"
            )
        arrays: dict[str, np.ndarray] = {}
        for name, entry in self.entries.items():
            want = np.dtype(self.specs[entry.input].dtype)
            source = entry.source or name
            array = _as_array(source, params[source], want)
            if entry.transposed:
                array = _transposed_source(source, array, entry)
            elif tuple(array.shape) != entry.shape:
                raise HFWeightError(
                    f"PackPlan.pack: parameter {name!r} has shape "
                    f"{tuple(array.shape)}, expected {entry.shape}"
                )
            # A rank-0 scalar goes into its own rank-1, length-1 packed input.
            arrays[name] = array.reshape(1) if len(entry.shape) == 0 else array
        packed: dict[str, np.ndarray] = {}
        for input_name, names in self.names_by_input().items():
            packed[input_name] = np.ascontiguousarray(
                np.concatenate([arrays[name] for name in names], axis=0)
            )
        return packed

    # -- in-graph unpacking -------------------------------------------------
    def unpack(self, packed: Mapping[str, "etl.SymbolicTensor"]) -> dict[str, "etl.SymbolicTensor"]:
        """Slice the packed graph inputs back into the canonical params dict.

        Called as the FIRST statement of a family forward body when the graph was
        built with ``pack_plan=self``: every slice is **static**
        (``etl.slice(t, start, lengths)``), so the compiled program is identical
        to the unpacked one up to a concatenation-free copy.

        Derived entries (:attr:`derived_keys`, e.g. a tied head stored
        pre-transposed) come back as extra keys holding the static slice of the
        stored — already transposed — shape; no in-graph transpose is involved.

        Args:
            packed: packed params mapping (the exact keys of :attr:`specs`) of
                ``SymbolicTensor`` values inside a trace.

        Returns:
            dict[str, etl.SymbolicTensor]: the canonical params dict the family
            forward body expects, plus one entry per derived key (when any).

        Raises:
            TypeError: ``packed`` is not a mapping, or a value is not a
                ``SymbolicTensor``.
            HFWeightError: keys are missing/extra, or a packed input's shape does
                not match the plan (i.e. the caller packed a different config).
        """
        if not isinstance(packed, Mapping):
            raise TypeError(
                f"PackPlan.unpack expects a mapping of packed name -> "
                f"SymbolicTensor, got {type(packed).__name__}"
            )
        expected = set(self.specs)
        got = set(packed)
        if got != expected:
            raise HFWeightError(
                f"PackPlan.unpack: packed params do not match this plan's inputs "
                f"(missing={sorted(expected - got)[:8]}, "
                f"extra={sorted(got - expected)[:8]}); a packed executable takes "
                f"exactly the {len(expected)} packed inputs named "
                f"{sorted(expected)[:8]}"
            )
        for input_name in expected:
            tensor = packed[input_name]
            if not isinstance(tensor, etl.SymbolicTensor):
                raise TypeError(
                    f"PackPlan.unpack: packed input {input_name!r} must be an "
                    f"etl SymbolicTensor inside a trace, got "
                    f"{type(tensor).__name__}"
                )
            want = tuple(self.specs[input_name].shape)
            if tuple(tensor.shape) != want:
                raise etl.ShapeError(
                    f"PackPlan.unpack: packed input {input_name!r} has shape "
                    f"{tuple(tensor.shape)}, expected {want} — the params were "
                    f"packed for a different config"
                )
        canonical: dict[str, etl.SymbolicTensor] = {}
        for name, entry in self.entries.items():
            tensor = packed[entry.input]
            if len(entry.shape) == 0:
                # Rank-0 canonical value: slice the length-1 bucket, then drop
                # the axis (a slice can never produce a 1-element rank-0 result).
                canonical[name] = etl.reshape(
                    etl.slice(tensor, entry.start, (1,)), ()
                )
            else:
                canonical[name] = etl.slice(tensor, entry.start, entry.shape)
        return canonical


def _name_index(name: str) -> tuple[int, str]:
    """Sort key for packed input names: the numeric suffix, then the text.

    Args:
        name: a packed input name (e.g. ``"packed.10"``).

    Returns:
        tuple[int, str]: ``(numeric suffix or -1, name)`` — so ``packed.10``
        sorts after ``packed.9`` (no lexicographic 1/10 surprise).
    """
    prefix, _, suffix = name.rpartition(".")
    return (int(suffix), prefix) if suffix.isdigit() else (-1, name)


def is_packed_tree(tree: object) -> bool:
    """Whether a params tree (mapping or key iterable) uses packed input names.

    Used only for diagnostics: :func:`etl_hf.run` and :func:`etl_hf.predict` name
    the ``pack_weights`` remedy instead of surfacing a bare etl pytree error when
    a caller mixes the two tree shapes.

    Args:
        tree: a params mapping, or an iterable of keys.

    Returns:
        bool: ``True`` when the tree is non-empty and every key starts with
        :data:`PACKED_PREFIX`.
    """
    source: Iterable = tree.keys() if isinstance(tree, Mapping) else tree  # type: ignore[assignment]
    try:
        names = list(source)
    except TypeError:  # a non-iterable is never a params tree
        return False
    if not names:
        return False
    return all(
        isinstance(name, str) and name.startswith(PACKED_PREFIX) for name in names
    )


# ---------------------------------------------------------------------------
# public builders
# ---------------------------------------------------------------------------
def _alias_shapes(
    shapes: Mapping[str, tuple[int, ...]],
    transposed_aliases: Mapping[str, str],
) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Validate derived-entry declarations and return their stored shapes.

    Args:
        shapes: canonical key → static shape (already validated).
        transposed_aliases: ``{derived_key: source_key}`` — each derived key is
            stored as the transposed copy of ``source_key``.

    Returns:
        dict: derived key → ``(source key, stored shape)``.

    Raises:
        TypeError: a key or source is not a string, or ``transposed_aliases`` is
            not a mapping.
        HFWeightError: a source key is absent, or a derived key shadows a
            canonical key (the plan could not tell which data to store).
        HFUnsupportedError: the source is not a rank-2 static tensor (a transpose
            is only defined — and only needed — for a 2-D weight table).
    """
    if not isinstance(transposed_aliases, Mapping):
        raise TypeError(
            "pack_plan(transposed_aliases=...) must be a mapping of derived name "
            f"-> canonical source name, got {type(transposed_aliases).__name__}"
        )
    derived: dict[str, tuple[str, tuple[int, ...]]] = {}
    for name, source in transposed_aliases.items():
        if not isinstance(name, str) or not isinstance(source, str):
            raise TypeError(
                "pack_plan(transposed_aliases=...) must map derived weight name -> "
                f"canonical source name (both strings), got {name!r} -> {source!r}"
            )
        if name in shapes:
            raise HFWeightError(
                f"pack_plan: derived packed key {name!r} collides with a canonical "
                f"parameter of the same name; a derived entry is an EXTRA copy "
                f"(e.g. a tied head stored transposed), so it needs its own key"
            )
        if source not in shapes:
            raise HFWeightError(
                f"pack_plan: derived packed key {name!r} needs canonical source "
                f"{source!r}, which is not in the specs tree "
                f"({sorted(shapes)[:8]}…)"
            )
        source_shape = shapes[source]
        if len(source_shape) != 2:
            raise HFUnsupportedError(
                f"pack_plan: derived packed key {name!r} is a transposed copy of "
                f"{source!r}, which has rank {len(source_shape)} (shape "
                f"{source_shape}); only a rank-2 weight table has a meaningful "
                f"(in_features, out_features) transpose"
            )
        derived[name] = (source, source_shape[::-1])
    return derived


def pack_plan(
    specs: Mapping[str, etl.TensorSpec],
    *,
    prefix: str = PACKED_PREFIX,
    transposed_aliases: Mapping[str, str] | None = None,
) -> PackPlan:
    """Build the packed-input plan for one canonical ``param_specs`` tree.

    Generic over any family: the plan depends only on the spec tree's dtypes,
    ranks and trailing dims (see the module docstring's bucketing rule), plus any
    derived (pre-transposed) copies the tree declares.

    Args:
        specs: the canonical params graph inputs (e.g.
            ``FamilySpec.param_specs(cfg)``) — static shapes only, and a plain
            ``dict`` (etl's tree machinery keys on the exact container type, so
            the declaration rides on the *plan*, not on the mapping). A tied tree
            (``embed.weight`` present, ``lm_head.weight`` absent) declares its
            ``(hidden, vocab)`` head transpose automatically.
        prefix: name prefix of the packed inputs (default :data:`PACKED_PREFIX`,
            i.e. ``"packed.0"``, ``"packed.1"`` …).
        transposed_aliases: explicit ``{derived_key: source_key}`` derived-copy
            declarations, overriding the inferred ones. Each derived key is
            stored once as ``params[source_key].T`` and unpacked as its own
            static slice, so no weight-sized transpose happens in the graph.
            ``None`` (default) uses :func:`etl_hf._specs.transposed_aliases`
            (a tied-embedding tree declares its head); passing ``{}`` forces a
            plan with no derived entries.

    Returns:
        PackPlan: the packed specs plus one :class:`PackEntry` per unpacked key —
        the canonical keys of ``specs`` and any derived ones.

    Raises:
        TypeError: ``specs`` is not a mapping, a key is not a string,
            ``prefix`` is not a non-empty string, or ``transposed_aliases`` is
            malformed.
        HFUnsupportedError: a spec has a symbolic/dynamic dim (packing needs
            static shapes), or a derived source is not rank 2.
        HFWeightError: a derived key shadows a canonical key or names a missing
            source.
    """
    if not isinstance(specs, Mapping):
        raise TypeError(
            f"pack_plan expects a mapping of canonical name -> etl.TensorSpec, got "
            f"{type(specs).__name__}"
        )
    if not isinstance(prefix, str) or not prefix:
        raise TypeError(
            f"pack_plan(prefix=...) must be a non-empty string, got {prefix!r} of "
            f"type {type(prefix).__name__}"
        )
    declared = (
        _declared_aliases(specs) if transposed_aliases is None else transposed_aliases
    )
    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, np.dtype] = {}
    buckets: dict[tuple, list[str]] = {}
    for name, spec in specs.items():
        if not isinstance(name, str):
            raise TypeError(
                f"pack_plan: parameter names must be strings, got {name!r} of type "
                f"{type(name).__name__}"
            )
        shape = _static_shape(name, spec)
        shapes[name] = shape
        dtypes[name] = np.dtype(spec.dtype)
        buckets.setdefault(_bucket_key(name, shape, spec.dtype), []).append(name)

    derived = _alias_shapes(shapes, declared) if declared else {}
    for name, (source, stored_shape) in derived.items():
        shapes[name] = stored_shape
        dtypes[name] = dtypes[source]
        buckets.setdefault(
            _bucket_key(name, stored_shape, dtypes[source]), []
        ).append(name)

    entries: dict[str, PackEntry] = {}
    packed_specs: dict[str, etl.TensorSpec] = {}
    for index, names in enumerate(
        sorted(buckets.values(), key=min)  # deterministic: min bucket key
    ):
        input_name = f"{prefix}{index}"
        ordered = sorted(names)
        offset = 0
        for name in ordered:
            shape = shapes[name]
            length = shape[0] if shape else 1
            # Static starts: zeros except the concatenation axis (axis 0).
            start = (offset,) + (0,) * (len(shape) - 1) if shape else (offset,)
            source = derived[name][0] if name in derived else None
            entries[name] = PackEntry(
                input=input_name,
                start=start,
                shape=shape,
                source=source,
                transposed=source is not None,
            )
            offset += length
        first = shapes[ordered[0]]
        packed_shape = (offset,) if not first else (offset,) + first[1:]
        packed_specs[input_name] = etl.TensorSpec(
            shape=packed_shape, dtype=dtypes[ordered[0]]
        )
    return PackPlan(specs=packed_specs, entries=entries)


def pack_params(
    plan: PackPlan, params: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Pack a canonical host params tree with ``plan`` (shorthand for ``plan.pack``).

    Args:
        plan: the :class:`PackPlan` built from the same family's ``param_specs``.
        params: canonical params, host arrays (dtype/shape must match the plan).

    Returns:
        dict[str, np.ndarray]: the packed host tree a ``pack_weights=True``
        executable takes.

    Raises:
        TypeError: ``plan`` is not a :class:`PackPlan`, or ``params`` is not a
            mapping.
        HFWeightError: a key is missing/extra, or a value has the wrong
            dtype/shape (the message names the key).
    """
    if not isinstance(plan, PackPlan):
        raise TypeError(
            f"pack_params expects an etl_hf.PackPlan (build one with "
            f"etl_hf.pack_plan(param_specs)), got {type(plan).__name__}"
        )
    return plan.pack(params)
