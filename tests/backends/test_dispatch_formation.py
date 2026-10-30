"""Unit tests for the iree ``dispatch_formation`` compile option — the
dispatch-formation knob for ``batch_functions`` merged-population graphs.

Modelled on ``tests/backends/test_vulkan_target.py``: pure unit tests for the
derivation/validation contract (NO real compiles, NO GPU, compiler-free), plus
two parity tests that the option never changes RESULTS.

Covered (see ``evoxc/backends/adapters/iree.py``):

1. ``_resolve_iree_compile_args`` — the derivation matrix for
   ``dispatch_formation``: UNSET (and the explicit ``"default"`` mode, whose
   mode tuple is empty) emits NO dispatch-formation flag at all (byte-identical
   to the pre-option behavior), ``"aggressive"`` emits exactly one
   ``--iree-dispatch-creation-enable-aggressive-fusion``, and a user flag of the
   same NAME wins (the derived flag is dropped — never both, the documented
   conflict rule shared with ``opt_level``/``vulkan_target``). Interactions with
   ``opt_level_args``/``vulkan_target_args`` and the ``llvm-cpu``/``rocm``
   target-specific defaults stay unchanged, and the flag is NOT in the deny list.
2. ``IreeBackend._resolve_compile_options`` — the ``dispatch_formation`` VALUE
   validation (case-insensitive + whitespace-stripped; non-str / empty /
   unknown spelling => loud ``BackendError`` naming the option, the accepted
   values and the offending value), the caller dict never mutated.
3. ``KNOWN_OPTIONS`` — ``dispatch_formation`` is declared at the compile stage
   and reachable through the pipeline's option validation (accepted-and-ignored
   at ``lower``/``load``/``run``), while an unknown key is still rejected.
4. Parity — an llvm-cpu merged-population graph and a GPU-guarded rocm one give
   the SAME values with the option OFF and ON (the flag changes dispatch
   formation only, never results).
"""

from __future__ import annotations

import numpy as np
import pytest

import evoxc
from evoxc.core import BackendError, Device
from evoxc.backends.adapters import iree as iree_adapt
from evoxc.backends.options import validate_options

# ---------------------------------------------------------------------------
# Shared constants + the tiny graph used by the public-path tests
# ---------------------------------------------------------------------------

_OPTION = "dispatch_formation"
_FLAG_NAME = "--iree-dispatch-creation-enable-aggressive-fusion"

_ROW_SPEC = evoxc.TensorSpec((1024,), evoxc.float32)

#: Every env var that could leak into an option dict during these tests (the
#: per-stage env half of ``evoxc.pipeline_options`` + the staging-level vars).
_ENV_VARS = (
    "EVOXC_IREE_COMPILE_ARGS",
    "EVOXC_IREE_RUNTIME_ARGS",
    "EVOXC_IREE_VULKAN_TARGET",
    "EVOXC_OPT_LEVEL",
    "EVOXC_EIGH_ALGORITHM",
    "EVOXC_SPLIT_REDUCTION",
    "EVOXC_BACKEND",
    "EVOXC_DEVICE",
    "EVOXC_TARGET_BACKENDS",
)


@pytest.fixture(autouse=True)
def _clean_option_env(monkeypatch):
    """No ambient ``EVOXC_*`` option/environment leaks into these assertions."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _formation_flags(flags):
    """All flags whose NAME is the aggressive-fusion flag (name-based counting,
    mirroring the adapter's own override machinery — values never matter)."""
    return [flag for flag in flags if flag.partition("=")[0] == _FLAG_NAME]


# ---------------------------------------------------------------------------
# 1. _resolve_iree_compile_args — the derivation matrix (pure, no IREE)
# ---------------------------------------------------------------------------


def test_resolve_compile_args_formation_unset_emits_no_flag():
    """UNSET ``dispatch_formation`` => NO dispatch-formation flag at all:
    byte-identical to the pre-option argument set."""
    out = iree_adapt._resolve_iree_compile_args(("vulkan-spirv",), None)
    assert _formation_flags(out) == []
    assert out == list(iree_adapt._IREE_DEFAULT_COMPILE_ARGS)


def test_resolve_compile_args_formation_default_mode_emits_no_flag():
    """The explicit ``"default"`` mode resolves to the EMPTY tuple, so it emits
    exactly what UNSET emits."""
    assert iree_adapt._DISPATCH_FORMATION_MODES["default"] == ()
    out = iree_adapt._resolve_iree_compile_args(
        ("llvm-cpu",), None, dispatch_formation_args=()
    )
    assert _formation_flags(out) == []


def test_resolve_compile_args_formation_derived_flag_exactly_once():
    derived = iree_adapt._DISPATCH_FORMATION_MODES["aggressive"]
    assert derived == (_FLAG_NAME,)
    out = iree_adapt._resolve_iree_compile_args(
        ("llvm-cpu",), None, dispatch_formation_args=derived
    )
    assert _formation_flags(out) == [_FLAG_NAME]
    # The other evoxc defaults are untouched by the derived flag.
    for default in iree_adapt._IREE_DEFAULT_COMPILE_ARGS:
        assert default in out
    for default in iree_adapt._IREE_DEFAULT_LLVMCPU_ARGS:
        assert default in out


def test_resolve_compile_args_formation_user_flag_wins():
    """A user flag of the same NAME replaces the derived one — never both."""
    out = iree_adapt._resolve_iree_compile_args(
        ("llvm-cpu",), [_FLAG_NAME], dispatch_formation_args=(_FLAG_NAME,)
    )
    assert _formation_flags(out) == [_FLAG_NAME]


def test_resolve_compile_args_formation_user_flag_without_derived():
    """A raw aggressive-fusion flag with the option UNSET is passed through
    verbatim (the derived tuple is empty, nothing to collide with)."""
    out = iree_adapt._resolve_iree_compile_args(("llvm-cpu",), [_FLAG_NAME])
    assert _formation_flags(out) == [_FLAG_NAME]


def test_resolve_compile_args_formation_coexists_with_opt_level_and_vulkan():
    derived = iree_adapt._DISPATCH_FORMATION_MODES["aggressive"]
    out = iree_adapt._resolve_iree_compile_args(
        ("vulkan-spirv",),
        None,
        opt_level_args=("--iree-opt-level=O2",),
        vulkan_target_args=("--iree-vulkan-target=rdna3",),
        dispatch_formation_args=derived,
    )
    assert _formation_flags(out) == [_FLAG_NAME]
    assert [f for f in out if f.partition("=")[0] == "--iree-opt-level"] == [
        "--iree-opt-level=O2"
    ]
    assert [f for f in out if f.partition("=")[0] == "--iree-vulkan-target"] == [
        "--iree-vulkan-target=rdna3"
    ]


def test_resolve_compile_args_formation_rocm_default_unchanged(monkeypatch):
    """The detected ``--iree-rocm-target`` default still joins the defaults and
    the aggressive flag is additive, not a replacement."""
    monkeypatch.setattr(
        iree_adapt,
        "_rocm_target_default",
        lambda user_args: ("--iree-rocm-target=gfx942",),
    )
    derived = iree_adapt._DISPATCH_FORMATION_MODES["aggressive"]
    out = iree_adapt._resolve_iree_compile_args(
        ("rocm",), None, dispatch_formation_args=derived
    )
    assert "--iree-rocm-target=gfx942" in out
    assert _formation_flags(out) == [_FLAG_NAME]


def test_resolve_compile_args_formation_not_denied():
    """The aggressive-fusion flag is deliberately NOT in the deny list, so it
    can always be passed raw through ``iree_compile_args``."""
    out = iree_adapt._resolve_iree_compile_args(("llvm-cpu",), [_FLAG_NAME])
    assert _FLAG_NAME in out


# ---------------------------------------------------------------------------
# 2. _resolve_compile_options — the value validation
# ---------------------------------------------------------------------------


def test_resolve_compile_options_unset_emits_nothing():
    _targets, extra = iree_adapt.IreeBackend()._resolve_compile_options(None)
    assert _formation_flags(extra) == []


def test_resolve_compile_options_default_value_emits_nothing():
    _targets, extra = iree_adapt.IreeBackend()._resolve_compile_options(
        {_OPTION: "default"}
    )
    assert _formation_flags(extra) == []


def test_resolve_compile_options_aggressive_derives_one_flag():
    _targets, extra = iree_adapt.IreeBackend()._resolve_compile_options(
        {_OPTION: "aggressive"}
    )
    assert _formation_flags(extra) == [_FLAG_NAME]


@pytest.mark.parametrize("value", [" Aggressive ", "AGGRESSIVE", "aggressive "])
def test_resolve_compile_options_case_and_whitespace_insensitive(value):
    _targets, extra = iree_adapt.IreeBackend()._resolve_compile_options(
        {_OPTION: value}
    )
    assert _formation_flags(extra) == [_FLAG_NAME]


@pytest.mark.parametrize(
    "value", ["ludicrous", "", "  ", "auto", 3, True, ["aggressive"]]
)
def test_resolve_compile_options_rejects_bad_value(value):
    with pytest.raises(BackendError) as excinfo:
        iree_adapt.IreeBackend()._resolve_compile_options({_OPTION: value})
    message = str(excinfo.value)
    assert "'dispatch_formation' compile option" in message
    assert "must be one of" in message
    assert "'aggressive'" in message and "'default'" in message
    assert repr(value) in message


def test_resolve_compile_options_user_flag_wins_over_derived():
    _targets, extra = iree_adapt.IreeBackend()._resolve_compile_options(
        {_OPTION: "aggressive", "iree_compile_args": [_FLAG_NAME]}
    )
    assert _formation_flags(extra) == [_FLAG_NAME]


def test_resolve_compile_options_does_not_mutate_caller_dict():
    options = {_OPTION: "aggressive"}
    iree_adapt.IreeBackend()._resolve_compile_options(options)
    assert options == {_OPTION: "aggressive"}


# ---------------------------------------------------------------------------
# 3. KNOWN_OPTIONS — declared at compile, accepted-and-ignored elsewhere
# ---------------------------------------------------------------------------


def test_known_options_declares_dispatch_formation_at_compile():
    known = iree_adapt.IreeBackend.KNOWN_OPTIONS
    assert _OPTION in known["compile"]
    for stage in ("lower", "load", "run"):
        assert _OPTION not in known[stage]


def test_validate_options_accepts_dispatch_formation_at_every_stage():
    known = iree_adapt.IreeBackend.KNOWN_OPTIONS
    for stage in ("lower", "compile", "load", "run"):
        validate_options({_OPTION: "aggressive"}, known, "iree", stage)


def test_validate_options_still_rejects_unknown_key():
    known = iree_adapt.IreeBackend.KNOWN_OPTIONS
    with pytest.raises(BackendError) as excinfo:
        validate_options({"bogus": 1}, known, "iree", "compile")
    assert "does not recognize the compile option 'bogus'" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 4. Parity — dispatch formation never changes results
# ---------------------------------------------------------------------------


def _merged_graph():
    """A small merged population: 4 expression programs over one shared input."""

    def p0(x):
        return evoxc.add(evoxc.multiply(x, 2.0), 1.0)

    def p1(x):
        return evoxc.tanh(evoxc.subtract(x, 0.5))

    def p2(x):
        return evoxc.maximum(x, evoxc.multiply(x, x))

    def p3(x):
        return evoxc.exp(evoxc.multiply(x, -0.25))

    merged = evoxc.batch_functions([p0, p1, p2, p3], axes=None, out_axes=0)
    return merged(_ROW_SPEC)


def _reference(x):
    return np.stack(
        [
            x * np.float32(2.0) + np.float32(1.0),
            np.tanh(x - np.float32(0.5)),
            np.maximum(x, x * x),
            np.exp(x * np.float32(-0.25)),
        ]
    )


def _row_input():
    return np.linspace(-1.0, 1.0, 1024, dtype=np.float32)


def test_merged_graph_parity_llvm_cpu_both_modes():
    """The flag only changes dispatch FORMATION — llvm-cpu results are
    bit-identical with the option OFF and ON.

    Compiler-availability guarded (the local idiom of the other parity
    suites): this leg really lowers + compiles + runs through iree, so in a
    compiler-free configuration it must SKIP like every other compile leg
    rather than fail with a ``BackendError``.
    """
    pytest.importorskip("iree.compiler")
    pytest.importorskip("iree.runtime")
    graph = _merged_graph()
    x = _row_input()
    reference = _reference(x)
    outputs = []
    for mode in ("default", "aggressive"):
        lowered = evoxc.lower(graph, backend="iree", target_backends=["llvm-cpu"])
        artifact = evoxc.compile(
            lowered,
            backend="iree",
            target_backends=["llvm-cpu"],
            dispatch_formation=mode,
        )
        exe = evoxc.load(artifact, backend="iree")
        outputs.append(evoxc.run(exe, evoxc.from_numpy(x)).numpy())
        exe.synchronize()
    np.testing.assert_allclose(outputs[0], reference, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_merged_graph_parity_rocm_both_modes():
    """GPU-guarded rocm parity: the same values with the option OFF and ON."""
    from tests.backends import _rocm_guard as guard

    device = guard.require_rocm()
    graph = _merged_graph()
    x = _row_input()
    reference = _reference(x)
    outputs = []
    for mode in ("default", "aggressive"):
        lowered = evoxc.lower(graph, backend="iree", target_backends=["rocm"])
        artifact = evoxc.compile(
            lowered,
            backend="iree",
            target_backends=["rocm"],
            dispatch_formation=mode,
        )
        exe = evoxc.load(artifact, backend="iree", device=device)
        out = evoxc.run(exe, evoxc.from_numpy(x).to(device))
        outputs.append(out.to(Device("cpu", 0)).numpy())
        exe.synchronize()
    np.testing.assert_allclose(outputs[0], reference, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(outputs[0], outputs[1])
