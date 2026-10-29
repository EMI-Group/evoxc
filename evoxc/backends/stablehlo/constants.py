"""Bulk ``dense<[...]>`` elements-attribute rendering for the StableHLO writer.

`writer._constant_text` must render an arbitrary numpy payload as a
``dense<[...]>`` elements attribute.  The obvious implementation — a
per-element Python recursion that formats one numpy scalar at a time — costs
~2.2 µs per float element and *dominates* the export of a large merged graph:
a `batch_functions` population of 1000 programs stacks its masks into ONE
``(1000, 1024)`` i1 constant, ~86 % of such a graph's text is constant
literals, and constant rendering was 1.23 s of a 2.06 s
``evoxc.lower(graph, backend="iree")``.

This module produces BYTE-IDENTICAL text with C-level loops instead:

- The whole ``[a, b, ...]`` bracket/comma/space nesting is ONE
  ``str(array.tolist())`` call for integer, bool and float payloads —
  Python's ``list_repr`` emits exactly the layout the historical ``nest``
  built (one bracket pair per axis, ``", "`` between siblings), and
  ``str(float)`` IS the base spelling the historical ``_fmt_float`` started
  from (CPython's shortest round-trip repr).
- The per-element post-pass ``_fmt_float`` then applied (uppercase the
  exponent marker, pad a dot-less mantissa, spell non-finite values as IEEE
  hex for the payload's bit width) is applied to the WHOLE rendered text
  instead: two ``str.replace`` passes for the non-finite spellings — gated
  behind one ``np.isfinite`` sweep, so a finite payload pays no scan — and a
  ``str.find`` walk that visits only ACTUAL exponent occurrences (no regex
  scan over the whole text, no per-element call).
- Bool payloads are the same text with ``True``/``False`` → ``true``/
  ``false``; complex payloads format each part through the float path and
  pair them with one C-level ``map``.
- Integer payloads (incl. ``i64``/``u64``) are the raw ``str(list)`` text —
  ``tolist()`` already yields exact Python ints.

`_legacy_dense_text` keeps the historical recursion VERBATIM as the fallback
for payload dtypes the bulk paths do not cover (a 128-bit ``longdouble``,
whose ``tolist()`` yields numpy scalars, and object/string arrays whose
``str(v)`` fallback has no bulk equivalent).  It is also why the ``KeyError``
a nan/inf part of a ``complex128`` raises is unchanged: the hex tables are
keyed by ``itemsize * 8`` and only have 16/32/64 entries.

Large payloads (``>= _MEMO_MIN_ELEMENTS``) go through a small bounded LRU
keyed on ``(dtype, shape, sha256(C-order bytes))`` → the rendered
``dense<...>`` text, so re-exporting a graph (or exporting the same payload
in another graph) reuses the exact characters.  The key is the PAYLOAD, never
a graph or op identity, and rendering is a pure function of the payload, so a
hit is byte-exact by construction; hashing costs a few ms where rendering
costs tens-to-hundreds of ms.  The memo is process-global, bounded by entry
count AND total characters, and never persisted.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

import numpy as np

# Hex float spellings of IEEE special values (valid MLIR float literals;
# keyed by bit width). Used for inf/nan dense elements and extreme reduce
# init values.
_HEX_NAN = {16: "0x7E00", 32: "0x7FC00000", 64: "0x7FF8000000000000"}
_HEX_INF = {16: "0x7C00", 32: "0x7F800000", 64: "0x7FF0000000000000"}
_HEX_NINF = {16: "0xFC00", 32: "0xFF800000", 64: "0xFFF0000000000000"}

#: Payload dtypes the bulk paths format (float/complex are keyed by itemsize).
_BULK_FLOAT_ITEMSIZES = frozenset({2, 4, 8})
_BULK_COMPLEX_ITEMSIZES = frozenset({8, 16})

#: Memo tuning: payloads smaller than this are rendered directly (a per-call
#: hash would cost more than the C-level render), and the LRU is bounded by
#: BOTH entry count and total cached characters (a ``(1000, 1024)`` i1 text is
#: ~6.6 MB, so a count-only bound would be unbounded in practice).
_MEMO_MIN_ELEMENTS = 8192
_MEMO_MAX_ENTRIES = 8
_MEMO_MAX_CHARS = 24_000_000

_memo_lock = threading.Lock()
_memo: "OrderedDict[tuple, str]" = OrderedDict()
_memo_chars = 0


def _fmt_float(value: float, bits: int = 64) -> str:
    """Format a float as a valid MLIR float literal (mandatory decimal
    point; inf/nan via IEEE hex spelling for the given bit width)."""
    value = float(value)
    if np.isnan(value):
        return _HEX_NAN[bits]
    if np.isinf(value):
        return _HEX_NINF[bits] if value < 0 else _HEX_INF[bits]
    text = repr(value)
    if "e" in text or "E" in text:
        mantissa, exponent = _re_split_exp(text)
        if "." not in mantissa:
            mantissa += ".0"
        return f"{mantissa}E{exponent}"
    if "." not in text:
        text += ".0"
    return text


def _re_split_exp(text: str):
    """Split a repr-format float into (mantissa, exponent) at e/E."""
    for i, ch in enumerate(text):
        if ch in "eE":
            return text[:i], text[i + 1 :]
    return text, ""


def dense_text(array) -> str:
    """The ``dense<[...]>`` elements attribute for a numpy array payload."""
    arr = np.asarray(array)
    if arr.size == 0:
        return "dense<>"
    if arr.size >= _MEMO_MIN_ELEMENTS:
        key = _memo_key(arr)
        cached = _memo_get(key)
        if cached is not None:
            return cached
        text = _render(arr)
        _memo_put(key, text)
        return text
    return _render(arr)


def _render(arr) -> str:
    """Render `arr` (non-empty, memo-free) through the bulk or legacy path."""
    kind = arr.dtype.kind
    if kind == "b":
        return "dense<" + _bool_text(arr) + ">"
    if kind == "i" or kind == "u":
        # `tolist()` yields exact Python ints for every width (u64 included)
        # and `str(list)` renders exactly `str(int(v))` per element.
        return "dense<" + str(arr.tolist()) + ">"
    if kind == "f" and arr.dtype.itemsize in _BULK_FLOAT_ITEMSIZES:
        return "dense<" + _float_text(arr) + ">"
    if kind == "c" and arr.dtype.itemsize in _BULK_COMPLEX_ITEMSIZES:
        return "dense<" + _complex_text(arr) + ">"
    return _legacy_dense_text(arr)


def _bool_text(arr) -> str:
    """`[true, false, ...]` — `str(list)` then the two bool spellings.

    Payloads of ``_BOOL_BULK_MIN_ELEMENTS`` or more render through
    ``_bool_text_bulk`` (same characters, built with a numpy byte tile
    instead of a per-element Python ``repr``); smaller ones keep the
    historical path."""
    if arr.size >= _BOOL_BULK_MIN_ELEMENTS:
        return _bool_text_bulk(arr)
    text = str(arr.tolist())
    return text.replace("True", "true").replace("False", "false")


#: Payload size from which `_bool_text` switches to the bulk tile path (below
#: it the tile machinery costs more than the plain ``str(list)``).
_BOOL_BULK_MIN_ELEMENTS = 64


def _bool_text_bulk(arr) -> str:
    """Byte-identical twin of ``str(arr.tolist()).replace(...)`` for a bool
    payload, without the per-element Python repr.

    ``str(list)`` renders ``[True, False, ...]`` (nested ``[ ]`` per rank)
    and the two replaces only lower the leading letter, so the text is
    ``"[" + ", ".join("true"/"false" tokens) + "]"`` nested per rank. The
    innermost row's tokens are built as a numpy ``uint8`` tile —
    ``code, ',', ' '`` per element with ``code`` = ``'0'``/``'1'`` — decoded
    in one C-level call and turned into the two spellings by two
    ``str.replace`` passes (neither spelling contains a digit, so the
    replacements cannot cascade); nesting then joins the rows."""
    if arr.ndim == 0:
        # `str(arr.tolist())` of a 0-d array is the bare Python bool.
        return "true" if bool(arr) else "false"
    if arr.ndim > 1:
        return "[" + ", ".join(_bool_text_bulk(sub) for sub in arr) + "]"
    codes = np.ascontiguousarray(arr).reshape(-1).view(np.uint8) + 48
    tile = np.empty(codes.size * 3, dtype=np.uint8)
    tile[0::3] = codes
    tile[1::3] = 44  # ','
    tile[2::3] = 32  # ' '
    # Drop the trailing ", " (the `_re_split_exp`-free equivalent of the
    # separator join `str(list)` performs).
    flat = tile[:-2].tobytes().decode("latin-1")
    return "[" + flat.replace("0", "false").replace("1", "true") + "]"


def _float_text(arr, bits=None) -> str:
    """`[a, b, ...]` for a f16/f32/f64 payload, byte-identical to `_fmt_float`
    applied per element (the exponent / non-finite post-pass is bulk).

    `bits` overrides the IEEE hex spelling width — the historical renderer
    keyed it on the CONTAINER's `itemsize * 8`, so a complex64 payload (8
    bytes) spells a nan part with the 64-bit form, and a complex128 payload
    (16 bytes) raises `KeyError` for one; both are preserved."""
    text = str(arr.tolist())
    if not np.isfinite(arr).all():
        if bits is None:
            bits = arr.dtype.itemsize * 8
        # Order matters: "-inf" must be rewritten before the bare "inf" (and
        # neither spelling can occur inside a finite literal).
        if "-inf" in text:
            text = text.replace("-inf", _HEX_NINF[bits])
        if "inf" in text:
            text = text.replace("inf", _HEX_INF[bits])
        if "nan" in text:
            text = text.replace("nan", _HEX_NAN[bits])
    return _fix_exponents(text)


def _fix_exponents(text: str) -> str:
    """Turn every ``<mantissa>e<exp>`` repr spelling into the MLIR form
    ``<mantissa>.0E<exp>`` / ``<mantissa>E<exp>``.

    Exactly the transformation `_fmt_float` applies per element: split at the
    exponent marker, append ``".0"`` to a mantissa with no decimal point, and
    uppercase the marker.  Only ACTUAL occurrences are touched (``str.find``
    walks the text at memchr speed and the loop body runs once per exponent,
    never per element), so a payload with no exponent pays one scan and a
    payload with a few pays a few."""
    if "e" not in text:
        return text
    parts = []
    last = 0
    pos = text.find("e")
    while pos >= 0:
        start = pos
        while start > last and text[start - 1].isdigit():
            start -= 1
        parts.append(text[last:pos])
        if start == pos or (start > 0 and text[start - 1] == "."):
            pass  # mantissa already carries a decimal point
        else:
            parts.append(".0")
        parts.append("E")
        last = pos + 1
        pos = text.find("e", last)
    parts.append(text[last:])
    return "".join(parts)


def _float_tokens(arr, bits=None) -> "list[str]":
    """Flat element-token list of a 1-D float payload (one `_float_text`)."""
    return _float_text(arr, bits)[1:-1].split(", ")


def _complex_text(arr) -> str:
    """`[(a, b), ...]` for a complex64/complex128 payload, each part spelled
    exactly like `_fmt_float(part, itemsize * 8)` — the CONTAINER's bit width,
    as in the historical renderer."""
    shape = arr.shape
    bits = arr.dtype.itemsize * 8
    real = _float_tokens(np.ascontiguousarray(arr.real).reshape(-1), bits)
    imag = _float_tokens(np.ascontiguousarray(arr.imag).reshape(-1), bits)
    return _nest_tokens(list(map("({}, {})".format, real, imag)), shape)


def _nest_tokens(tokens, shape) -> str:
    """Bracket/comma nesting of a flat C-order token list — the `nest` layout
    (one bracket pair per axis, ``", "`` between siblings), built bottom-up
    with ONE `str.join` per array row instead of one recursion per element."""
    if not shape:
        return tokens[0]
    rows = tokens
    for size in reversed(shape):
        rows = [
            "[" + ", ".join(rows[i : i + size]) + "]"
            for i in range(0, len(rows), size)
        ]
    return rows[0]


def _legacy_dense_text(arr) -> str:
    """The historical per-element renderer, kept VERBATIM as the fallback for
    payload dtypes without a bulk path (numpy `longdouble`, object/string
    arrays)."""
    bits = arr.dtype.itemsize * 8

    def fmt_item(v):
        if isinstance(v, (bool, np.bool_)):
            return "true" if v else "false"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            return _fmt_float(float(v), bits)
        if isinstance(v, (complex, np.complexfloating)):
            return (
                f"({_fmt_float(float(v.real), bits)}, "
                f"{_fmt_float(float(v.imag), bits)})"
            )
        return str(v)

    def nest(a):
        if a.ndim == 0:
            return fmt_item(a.item())
        return "[" + ", ".join(nest(a[i]) for i in range(a.shape[0])) + "]"

    return "dense<" + nest(arr) + ">"


def _memo_key(array) -> tuple:
    """`(dtype, shape, sha256(C-order bytes))` — the payload identity of a
    constant's rendered text. `tobytes()` is C-order, matching the nesting
    traversal for any strides/contiguity."""
    return (
        array.dtype.str,
        array.shape,
        hashlib.sha256(array.tobytes()).digest(),
    )


def _memo_get(key: tuple):
    with _memo_lock:
        text = _memo.get(key)
        if text is None:
            return None
        _memo.move_to_end(key)
        return text


def _memo_put(key: tuple, text: str) -> None:
    global _memo_chars
    if len(text) > _MEMO_MAX_CHARS:
        return  # a single payload never evicts the whole memo
    with _memo_lock:
        previous = _memo.pop(key, None)
        if previous is not None:
            _memo_chars -= len(previous)
        _memo[key] = text
        _memo_chars += len(text)
        while len(_memo) > _MEMO_MAX_ENTRIES or _memo_chars > _MEMO_MAX_CHARS:
            _old_key, old_text = _memo.popitem(last=False)
            _memo_chars -= len(old_text)


def clear_memo() -> None:
    """Drop every memoised payload text (tests/diagnostics; never needed for
    correctness — a hit is byte-identical by construction)."""
    global _memo_chars
    with _memo_lock:
        _memo.clear()
        _memo_chars = 0
