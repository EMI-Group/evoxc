"""StableHLO export goldens for the ``remainder`` floormod composition.

``evoxc.remainder`` is numpy ``remainder`` = FLOORMOD (the result carries the
sign of the DIVISOR), while ``stablehlo.remainder`` is C ``fmod`` (the sign of
the DIVIDEND). The exporter therefore does NOT map ``remainder`` 1:1 — it
emits the inline correction (``writer._emit_remainder``, reached from
``_emit_decomposed``)::

    fmod   = stablehlo.remainder a, b
    adjust = (fmod != 0) & ((fmod < 0) != (b < 0))
    result = select(adjust, fmod + b, fmod)

Pinned here (pure ``lower`` / export text — no compiler, no device, small
shapes):

* the int32 ``(2, 3) % (3,)`` golden — exactly ONE ``stablehlo.remainder``
  (unquoted functional form), one ``comparison_direction NE``, two ``LT``,
  one ``stablehlo.xor``, one ``stablehlo.and``, one trailing
  ``stablehlo.select``, the ``dense<0> : tensor<i32>`` zero constant (+ its
  broadcast) and the preceding ``broadcast_in_dim`` operand equalization, in
  that ORDER;
* UNSIGNED dtypes emit ``stablehlo.remainder`` ALONE (unsigned remainder is
  already non-negative — it IS fmod): exactly one remainder and NO compare /
  select / zero constant;
* bool / complex64 / complex128 raise an explicit ``evoxc.BackendError``
  naming the dtype and "numpy has no remainder" (numpy has no remainder ufunc
  there, and complex only admits EQ/NE compares — never invalid MLIR);
* float32/float64 take the full correction (the float zero constant renders as
  ``dense<0.0> : tensor<f32>`` / ``<f64>``), i.e. ONE remainder site — the
  count is what the implementation actually emits (the dtype only changes the
  constant spelling, never the op count).

Numerical parity through a compiler backend is ``test_iree_emitters_parity.py``
(section 8); this file never compiles anything.
"""

import numpy as np
import pytest

import evoxc

#: The divisor broadcast + the 11-op floormod correction, in program order.
#: Each entry is matched against the mnemonic of one emitted line.
INT32_CORRECTION_SEQUENCE = (
    "stablehlo.broadcast_in_dim",  # the (3,) divisor equalized to (2, 3)
    "stablehlo.remainder",  # fmod
    "stablehlo.constant",  # the scalar zero
    "stablehlo.broadcast_in_dim",  # ... broadcast to the result shape
    "stablehlo.compare",  # comparison_direction NE  (fmod != 0)
    "stablehlo.compare",  # comparison_direction LT  (fmod < 0)
    "stablehlo.compare",  # comparison_direction LT  (b < 0)
    "stablehlo.xor",
    "stablehlo.and",
    "stablehlo.add",  # fmod + b
    "stablehlo.select",  # the trailing select
)


def _export(fn, *specs, options=None):
    """Trace ``fn`` over ``specs`` and export the graph as StableHLO MLIR."""
    return evoxc.backends.stablehlo.export(evoxc.trace(fn, *specs), options=options)


def _export_error(fn, *specs):
    """Trace ``fn`` and return the ``BackendError`` message raised by export."""
    graph = evoxc.trace(fn, *specs)
    with pytest.raises(evoxc.BackendError) as excinfo:
        evoxc.backends.stablehlo.export(graph)
    return str(excinfo.value)


def _op_mnemonics(mlir):
    """The emitted operation mnemonics, in program order (body lines only).

    ``stablehlo.remainder``/``stablehlo.xor``/``stablehlo.and``/``stablehlo.add``
    are emitted BARE (functional form), generic ops are quoted
    (``"stablehlo.compare"(...)``) — both forms are normalized to the
    ``stablehlo.<name>`` spelling here.
    """
    mnemonics = []
    for line in mlir.splitlines():
        stripped = line.strip()
        if not stripped.startswith("%"):
            continue  # module shell / func.func / func.return
        equals = stripped.index("=")
        rhs = stripped[equals + 1 :].strip()
        # Generic ops carry their name in quotes and a parenthesized operand
        # list (`"stablehlo.compare"(%a, %b) {...}`); bare ops are
        # space-separated (`stablehlo.remainder %a, %b : tensor<...>`).
        head = rhs.split("(")[0].strip().strip('"')
        mnemonics.append(head.split()[0])
    return mnemonics


def _remainder_fn(a, b):
    return evoxc.remainder(a, b)


# ---------------------------------------------------------------------------
# 1. signed integer: the full floormod correction
# ---------------------------------------------------------------------------


def test_int32_broadcast_remainder_golden():
    """int32 ``(2, 3) % (3,)``: the exact correction, in program order."""
    spec_a = evoxc.TensorSpec((2, 3), evoxc.int32)
    spec_b = evoxc.TensorSpec((3,), evoxc.int32)
    mlir = _export(_remainder_fn, spec_a, spec_b)

    # Order first (a trailing select after the `add`, never a 1:1 remainder).
    assert tuple(_op_mnemonics(mlir)) == INT32_CORRECTION_SEQUENCE

    # ... then the per-mnemonic counts the objective contract states.
    assert mlir.count("stablehlo.remainder") == 1
    assert mlir.count("comparison_direction NE") == 1
    assert mlir.count("comparison_direction LT") == 2
    assert mlir.count("stablehlo.xor") == 1
    assert mlir.count("stablehlo.and") == 1
    assert mlir.count("stablehlo.select") == 1
    assert mlir.count('"stablehlo.compare"') == 3
    assert mlir.count("dense<0> : tensor<i32>") == 1
    # The zero constant is a SCALAR tensor broadcast to the (2, 3) result.
    assert "tensor<i32>" in mlir
    # The last op before the return is the select.
    assert mlir.split("func.return")[0].strip().endswith("tensor<2x3xi32>")


def test_int32_remainder_is_not_a_bare_fmod():
    """The graph does NOT lower to a single ``stablehlo.remainder`` 1:1.

    Anti-vacuity for the golden above: the emitted correction really is the
    floormod composition, so the text is longer than a direct fmod mapping and
    the result is fed by a select, not by the remainder.
    """
    mlir = _export(
        _remainder_fn,
        evoxc.TensorSpec((2, 3), evoxc.int32),
        evoxc.TensorSpec((3,), evoxc.int32),
    )
    assert _op_mnemonics(mlir)[-1] == "stablehlo.select"
    assert len(_op_mnemonics(mlir)) == len(INT32_CORRECTION_SEQUENCE)
    # The compare operands are the remainder result and the divisor, and the
    # select's true branch is `fmod + b` — the shape of the standard fix.
    assert "stablehlo.remainder" in mlir
    assert mlir.count("stablehlo.add") == 1


# ---------------------------------------------------------------------------
# 2. unsigned: remainder alone (unsigned remainder IS fmod)
# ---------------------------------------------------------------------------


def test_uint8_remainder_emits_no_correction():
    """Unsigned remainder is already non-negative ⇒ single remainder, no fix."""
    mlir = _export(
        _remainder_fn,
        evoxc.TensorSpec((2, 3), evoxc.uint8),
        evoxc.TensorSpec((3,), evoxc.uint8),
    )
    assert mlir.count("stablehlo.remainder") == 1
    assert mlir.count('"stablehlo.compare"') == 0
    assert mlir.count("stablehlo.select") == 0
    assert mlir.count("stablehlo.xor") == 0
    assert mlir.count("stablehlo.and") == 0
    # No zero constant and no correction add either.
    assert mlir.count("stablehlo.constant") == 0
    assert mlir.count("stablehlo.add") == 0
    # The result is the remainder itself, in the unsigned type.
    assert "tensor<2x3xui8>" in mlir
    assert tuple(_op_mnemonics(mlir)) == (
        "stablehlo.broadcast_in_dim",  # the (3,) divisor equalized
        "stablehlo.remainder",
    )


# ---------------------------------------------------------------------------
# 3. floating point: the same correction, float zero constant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,zero_text,result_text",
    [
        (evoxc.float32, "dense<0.0> : tensor<f32>", "tensor<2x3xf32>"),
        (evoxc.float64, "dense<0.0> : tensor<f64>", "tensor<2x3xf64>"),
    ],
    ids=["f32", "f64"],
)
def test_float_remainder_golden(dtype, zero_text, result_text):
    """Float graphs export with exactly ONE remainder site (full correction)."""
    mlir = _export(
        _remainder_fn,
        evoxc.TensorSpec((2, 3), dtype),
        evoxc.TensorSpec((3,), dtype),
    )
    assert tuple(_op_mnemonics(mlir)) == INT32_CORRECTION_SEQUENCE
    assert mlir.count("stablehlo.remainder") == 1
    assert mlir.count("comparison_direction NE") == 1
    assert mlir.count("comparison_direction LT") == 2
    assert mlir.count("stablehlo.select") == 1
    assert zero_text in mlir
    assert f"-> {result_text}" in mlir
    assert "stablehlo.convert" not in mlir  # same dtype on both operands


# ---------------------------------------------------------------------------
# 4. non-numeric element types: explicit BackendError, never invalid MLIR
# ---------------------------------------------------------------------------

_UNSUPPORTED = [
    ("bool", evoxc.bool_, "bool"),
    ("complex64", evoxc.complex64, "complex64"),
    ("complex128", evoxc.complex128, "complex128"),
]


@pytest.mark.parametrize(
    "name,dtype,dtype_text", _UNSUPPORTED, ids=[case[0] for case in _UNSUPPORTED]
)
def test_remainder_rejected_for_dtypes_without_numpy_remainder(
    name, dtype, dtype_text
):
    message = _export_error(
        _remainder_fn,
        evoxc.TensorSpec((2, 3), dtype),
        evoxc.TensorSpec((3,), dtype),
    )
    assert "numpy has no remainder" in message
    assert "op 'remainder'" in message
    assert dtype_text in message
    # The message states WHY (no integer/float element type) and offers the
    # element-type frame, never a raw MLIR/compiler error.
    assert "floormod composition needs a signed integer or floating-point" in (
        message
    )


def test_remainder_error_is_a_backend_error_not_a_compiler_error():
    """The rejection is evoxc's own ``BackendError`` (an ``EVOXCError``)."""
    graph = evoxc.trace(
        _remainder_fn,
        evoxc.TensorSpec((2, 3), evoxc.bool_),
        evoxc.TensorSpec((3,), evoxc.bool_),
    )
    with pytest.raises(evoxc.BackendError) as excinfo:
        evoxc.backends.stablehlo.export(graph)
    assert isinstance(excinfo.value, evoxc.EVOXCError)
