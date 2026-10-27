"""Per-backend option env-var resolution tests (the options-override contract,
env half — ``evoxc.pipeline_options``).

Validates ``apply_env_options`` / ``ENV_OPTION_TABLE``: every per-backend
compiler flag/option must be settable at every pipeline stage — explicitly as
a kwarg to ``lower``/``compile``/``load``/``run`` (or the ``build``/
``evaluate`` sugar), or, when not passed explicitly, from an environment
variable. Precedence (binding): **explicit kwarg/option > env var > evoxc
default**.

Pinned here:

* no env set -> options unchanged; env vars read LAZILY at call time (every
  test drives them with ``monkeypatch`` after ``import evoxc``).
* stage-scoped application: ``EVOXC_IREE_COMPILE_ARGS`` applies at the compile
  stage only, ``EVOXC_IREE_RUNTIME_ARGS`` at the load stage only.
* explicit-wins-per-key; empty/whitespace env = unset; the caller's dict is
  never mutated.
* ``EVOXC_TVM_TARGET`` / ``EVOXC_TVM_PASS_CONFIGS`` (JSON object) /
  ``EVOXC_XLA_COMPILE_OPTIONS`` (base64 -> bytes) parse into the documented
  option keys; ``EVOXC_IREE_VULKAN_TARGET`` -> ``vulkan_target`` (the SPIR-V
  target profile, whitespace-stripped only — the compiler validates it).
* malformed env values raise ``core.BackendError`` naming the variable and
  the value — never a raw shlex/base64/json exception.
* the ``lower``-stage rows ``EVOXC_EIGH_ALGORITHM`` -> ``eigh_algorithm``
  (whitespace-stripped; the stablehlo Writer validates the value) and
  ``EVOXC_SPLIT_REDUCTION`` -> ``split_reduction`` (coerced to a real ``bool``)
  apply at every compiler backend's lower stage only, obey the same
  precedence, and reach the exporter end to end (final gated section).
* an unknown backend / a stage with no table entry is a no-op.
* full-path stub tests (a functional ``"iree"`` stub registered under the
  adapter name, exactly like ``pipeline_env_defaults_test.py``): the env
  compile args reach the compile stage, the env runtime args reach the load
  stage (``evoxc.load`` applies env internally even when the ``build`` sugar
  passes no explicit load options), an explicit kwarg beats the env end to
  end, run options are forwarded, the numpy backend documents ignoring
  options (``rank_context`` still flows through), and
  ``BoundExecutable.backend`` reports the stub's backend name.

Only the FINAL end-to-end export pins (the ``lower``-stage
``eigh_algorithm`` / ``split_reduction`` rows) touch the real iree adapter:
they import ``iree.compiler``/``iree.runtime`` inside their bodies, skip when
absent, and never call the import guard. Every other test stays adapter-free —
the iree-family paths use stub ``Backend`` instances (helper-level) or a
functional stub registered under the name ``"iree"`` (full-path) — and
``_assert_no_new_iree_import()`` pins it: the guard's baseline is captured
PER TEST by an autouse fixture (never at module import), so an adapter import
that another module of the same process legitimately performs can never make
the guard fire, while any import triggered by the test body itself fails it.
The backend half of the contract (per-adapter ``KNOWN_OPTIONS``
validation) is pinned by ``tests/backends/test_options_override.py``.
"""

from __future__ import annotations

import base64
import sys

import numpy as np
import pytest

import evoxc
from evoxc import core
from evoxc.backends.options import STAGES
from evoxc.pipeline_options import ENV_OPTION_TABLE, apply_env_options

#: The env vars this module drives (cleared by the autouse fixture so no test
#: ever sees a stray value from the host environment). Grouped by stage and
#: kept complete for both the vars the tests set and the vars the table
#: declares under the (backend, stage) keys these tests exercise.
OPTION_ENV_VARS = (
    # compile stage
    "EVOXC_IREE_COMPILE_ARGS",
    "EVOXC_IREE_VULKAN_TARGET",
    "EVOXC_XLA_COMPILE_OPTIONS",
    "EVOXC_TVM_TARGET",
    "EVOXC_TVM_PASS_CONFIGS",
    "EVOXC_OPT_LEVEL",
    # load stage
    "EVOXC_IREE_RUNTIME_ARGS",
    # lower stage (the reserved exporter options)
    "EVOXC_EIGH_ALGORITHM",
    "EVOXC_SPLIT_REDUCTION",
)

#: Did the real iree adapter already sit in ``sys.modules`` when the CURRENT
#: test started? Refreshed by the autouse fixture below so the guard is
#: order-independent: another module collected/run in the same process may
#: legitimately have imported the adapter already.
_iree_imported_at_test_start = "evoxc.backends.adapters.iree" in sys.modules


@pytest.fixture(autouse=True)
def _snapshot_adapter_import_state():
    global _iree_imported_at_test_start
    _iree_imported_at_test_start = "evoxc.backends.adapters.iree" in sys.modules
    yield


def _assert_no_new_iree_import():
    """This test must not be the one that imports the real iree adapter."""
    assert (
        "evoxc.backends.adapters.iree" in sys.modules
    ) == _iree_imported_at_test_start


@pytest.fixture(autouse=True)
def _clean_option_env(monkeypatch):
    """Every test starts with the option env vars unset."""
    for var in OPTION_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Shared graph definition
# ---------------------------------------------------------------------------


@evoxc.defn
def _linear(x, w, b):
    """A small linear layer: dot + bias + relu (three positional inputs)."""
    return evoxc.relu(evoxc.add(evoxc.dot(x, w), b))


_LINEAR_SPECS = (
    evoxc.TensorSpec((2, 3), evoxc.float32, name="x"),
    evoxc.TensorSpec((3, 4), evoxc.float32, name="w"),
    evoxc.TensorSpec((4,), evoxc.float32, name="b"),
)


def _linear_inputs():
    rng = np.random.default_rng(0)
    return (
        rng.random((2, 3)).astype("float32"),
        rng.random((3, 4)).astype("float32"),
        rng.random((4,)).astype("float32"),
    )


def _ref(x, w, b):
    return np.maximum(0.0, np.asarray(x) @ np.asarray(w) + np.asarray(b))


# ---------------------------------------------------------------------------
# ENV_OPTION_TABLE shape
# ---------------------------------------------------------------------------


def test_env_option_table_entries():
    """The env table maps exactly the documented (backend, stage) pairs."""
    assert set(ENV_OPTION_TABLE) == {
        ("iree", "compile"),
        ("iree", "load"),
        ("iree", "lower"),
        ("xla", "compile"),
        ("xla", "lower"),
        ("tvm", "compile"),
        ("tvm", "lower"),
    }


def test_env_option_table_keys_and_parsers():
    """The documented env var -> option key mapping (parsers are functions)."""
    iree_compile = ENV_OPTION_TABLE[("iree", "compile")]
    assert len(iree_compile) == 3
    assert {entry[1] for entry in iree_compile} == {
        "iree_compile_args",
        "opt_level",
        "vulkan_target",
    }
    args_entry = next(
        entry for entry in iree_compile if entry[1] == "iree_compile_args"
    )
    assert args_entry[:2] == ("EVOXC_IREE_COMPILE_ARGS", "iree_compile_args")
    assert callable(args_entry[2])
    vulkan_entry = next(
        entry for entry in iree_compile if entry[1] == "vulkan_target"
    )
    assert vulkan_entry[:2] == ("EVOXC_IREE_VULKAN_TARGET", "vulkan_target")
    assert callable(vulkan_entry[2])
    # The parser only strips whitespace — the compiler validates the profile.
    assert vulkan_entry[2]("EVOXC_IREE_VULKAN_TARGET", "  rtx4090  ") == "rtx4090"
    opt_level_entry = next(
        entry for entry in iree_compile if entry[1] == "opt_level"
    )
    assert opt_level_entry[:2] == ("EVOXC_OPT_LEVEL", "opt_level")
    assert callable(opt_level_entry[2])

    iree_load = ENV_OPTION_TABLE[("iree", "load")]
    assert len(iree_load) == 1
    var, key, parser = iree_load[0]
    assert (var, key) == ("EVOXC_IREE_RUNTIME_ARGS", "iree_runtime_args")
    assert callable(parser)

    compile_entries = ENV_OPTION_TABLE[("tvm", "compile")]
    assert {entry[1] for entry in compile_entries} == {
        "tvm_target",
        "tvm_pass_configs",
        "opt_level",
    }
    assert {entry[0] for entry in compile_entries} == {
        "EVOXC_TVM_TARGET",
        "EVOXC_TVM_PASS_CONFIGS",
        "EVOXC_OPT_LEVEL",
    }
    assert all(callable(entry[2]) for entry in compile_entries)
    assert ENV_OPTION_TABLE[("xla", "compile")][0][:2] == (
        "EVOXC_XLA_COMPILE_OPTIONS",
        "xla_compile_options",
    )
    xla_keys = {entry[1] for entry in ENV_OPTION_TABLE[("xla", "compile")]}
    assert xla_keys == {"xla_compile_options", "opt_level"}


def test_env_option_table_stages_are_canonical():
    """Every table stage is one of the canonical option stages."""
    for (backend, stage) in ENV_OPTION_TABLE:
        assert backend in ("iree", "xla", "tvm")
        assert stage in STAGES


#: The compiler backends whose ``lower`` row repeats the same two
#: backend-agnostic exporter options.
_LOWER_BACKENDS = ("iree", "xla", "tvm")


def test_env_option_table_lower_rows():
    """Each compiler backend's lower row carries exactly the two reserved
    exporter options — the same backend-agnostic spelling everywhere."""
    for backend in _LOWER_BACKENDS:
        entries = ENV_OPTION_TABLE[(backend, "lower")]
        assert len(entries) == 2
        assert {entry[1] for entry in entries} == {
            "eigh_algorithm",
            "split_reduction",
        }
        assert {(entry[0], entry[1]) for entry in entries} == {
            ("EVOXC_EIGH_ALGORITHM", "eigh_algorithm"),
            ("EVOXC_SPLIT_REDUCTION", "split_reduction"),
        }
        assert all(callable(entry[2]) for entry in entries)


# ---------------------------------------------------------------------------
# apply_env_options — helper-level contract (no backends involved)
# ---------------------------------------------------------------------------


def test_no_env_options_unchanged():
    """No env vars -> the options dict passes through untouched."""
    for backend, stage in (("iree", "compile"), ("iree", "load"),
                           ("xla", "compile"), ("tvm", "compile")):
        assert apply_env_options(backend, {}, stage) == {}
    assert apply_env_options("iree", {"target_backends": ["llvm-cpu"]},
                             "compile") == {"target_backends": ["llvm-cpu"]}


def test_env_read_lazily_per_call(monkeypatch):
    """Env vars are read at call time — set after import, honored on the next
    call; unset before the next call, no longer applied."""
    assert apply_env_options("iree", {}, "compile") == {}
    monkeypatch.setenv("EVOXC_IREE_COMPILE_ARGS", "--iree-llvmcpu-target-cpu=native")
    assert apply_env_options("iree", {}, "compile") == {
        "iree_compile_args": ["--iree-llvmcpu-target-cpu=native"]
    }
    monkeypatch.delenv("EVOXC_IREE_COMPILE_ARGS")
    assert apply_env_options("iree", {}, "compile") == {}


def test_explicit_wins_per_key(monkeypatch):
    """An explicit option key always wins over the env var; other keys are
    still env-supplied."""
    monkeypatch.setenv("EVOXC_IREE_COMPILE_ARGS", "--flag-from-env")
    resolved = apply_env_options(
        "iree", {"iree_compile_args": ["--flag-explicit"]}, "compile"
    )
    assert resolved == {"iree_compile_args": ["--flag-explicit"]}
    # Mixed: explicit target_backends + env-supplied iree_compile_args.
    resolved = apply_env_options(
        "iree", {"target_backends": ["cuda"]}, "compile"
    )
    assert resolved == {
        "target_backends": ["cuda"],
        "iree_compile_args": ["--flag-from-env"],
    }


@pytest.mark.parametrize("value", ["", "   ", "\t \n"])
def test_empty_or_whitespace_env_is_unset(monkeypatch, value):
    """An empty/whitespace env value is treated as unset."""
    monkeypatch.setenv("EVOXC_IREE_COMPILE_ARGS", value)
    assert apply_env_options("iree", {}, "compile") == {}


def test_per_stage_application_compile_args(monkeypatch):
    """EVOXC_IREE_COMPILE_ARGS applies at the iree compile stage only — never
    at lower, load, run, or on other backends."""
    monkeypatch.setenv("EVOXC_IREE_COMPILE_ARGS", "--iree-llvmcpu-target-cpu=native")
    assert apply_env_options("iree", {}, "compile") == {
        "iree_compile_args": ["--iree-llvmcpu-target-cpu=native"]
    }
    for stage in ("lower", "load", "run"):
        assert apply_env_options("iree", {}, stage) == {}
    assert apply_env_options("xla", {}, "compile") == {}
    assert apply_env_options("tvm", {}, "compile") == {}


def test_per_stage_application_runtime_args(monkeypatch):
    """EVOXC_IREE_RUNTIME_ARGS applies at the iree load stage only."""
    monkeypatch.setenv("EVOXC_IREE_RUNTIME_ARGS", "--cuda_async_allocations=false")
    assert apply_env_options("iree", {}, "load") == {
        "iree_runtime_args": ["--cuda_async_allocations=false"]
    }
    for stage in ("lower", "compile", "run"):
        assert apply_env_options("iree", {}, stage) == {}


def test_flag_list_parsing_quotes(monkeypatch):
    """Flag lists use shlex syntax — quoting is honored."""
    monkeypatch.setenv(
        "EVOXC_IREE_COMPILE_ARGS",
        '--iree-llvmcpu-target-cpu=native '
        '--iree-hal-dump-executable-sources-to="dir with spaces"',
    )
    assert apply_env_options("iree", {}, "compile") == {
        "iree_compile_args": [
            "--iree-llvmcpu-target-cpu=native",
            "--iree-hal-dump-executable-sources-to=dir with spaces",
        ]
    }


def test_tvm_target_and_pass_configs_mapping(monkeypatch):
    """EVOXC_TVM_TARGET and EVOXC_TVM_PASS_CONFIGS map onto the tvm compile
    options (target stripped, pass configs parsed as a JSON object)."""
    monkeypatch.setenv("EVOXC_TVM_TARGET", "  llvm -mcpu=native  ")
    monkeypatch.setenv(
        "EVOXC_TVM_PASS_CONFIGS",
        '{"tir.disable_assert": false, "relax.transform.": {"a": 1}}',
    )
    resolved = apply_env_options("tvm", {}, "compile")
    assert resolved["tvm_target"] == "llvm -mcpu=native"
    assert resolved["tvm_pass_configs"] == {
        "tir.disable_assert": False,
        "relax.transform.": {"a": 1},
    }
    # tvm has no other stages.
    for stage in ("lower", "load", "run"):
        assert apply_env_options("tvm", {}, stage) == {}


def test_xla_compile_options_base64_decode(monkeypatch):
    """EVOXC_XLA_COMPILE_OPTIONS is a base64 payload decoded to bytes."""
    payload = b"\x1a\x04\x20\x01\x28\x01"  # a serialized CompileOptionsProto
    monkeypatch.setenv("EVOXC_XLA_COMPILE_OPTIONS", base64.b64encode(payload).decode("ascii"))
    assert apply_env_options("xla", {}, "compile") == {
        "xla_compile_options": payload
    }


@pytest.mark.parametrize(
    ("var", "value", "fragment"),
    [
        ("EVOXC_IREE_COMPILE_ARGS", "--flag 'unclosed", "not a parseable"),
        ("EVOXC_XLA_COMPILE_OPTIONS", "%%%not-base64%%%", "not valid base64"),
        ("EVOXC_TVM_PASS_CONFIGS", "{bad json", "not valid JSON"),
        ("EVOXC_TVM_PASS_CONFIGS", "[1, 2]", "expected a JSON object"),
    ],
)
def test_malformed_env_values_raise_backend_error(monkeypatch, var, value, fragment):
    """A malformed env value raises core.BackendError naming the variable and
    the value — never a raw shlex/base64/json exception."""
    backend, stage = {
        "EVOXC_IREE_COMPILE_ARGS": ("iree", "compile"),
        "EVOXC_XLA_COMPILE_OPTIONS": ("xla", "compile"),
        "EVOXC_TVM_PASS_CONFIGS": ("tvm", "compile"),
    }[var]
    monkeypatch.setenv(var, value)
    with pytest.raises(core.BackendError) as excinfo:
        apply_env_options(backend, {}, stage)
    message = str(excinfo.value)
    assert var in message
    assert value in message
    assert fragment in message


# ---------------------------------------------------------------------------
# EVOXC_OPT_LEVEL — the first evoxc-defined + evoxc-validated option
# ---------------------------------------------------------------------------


def test_opt_level_env_applied_at_compile(monkeypatch):
    """EVOXC_OPT_LEVEL applies at the compile stage of every compiler backend,
    normalized to an int 0..3 by the shared validator."""
    monkeypatch.setenv("EVOXC_OPT_LEVEL", "O2")
    for backend in ("iree", "xla", "tvm"):
        assert apply_env_options(backend, {}, "compile") == {"opt_level": 2}
    # Other accepted forms normalize the same way.
    monkeypatch.setenv("EVOXC_OPT_LEVEL", "o3")
    assert apply_env_options("iree", {}, "compile") == {"opt_level": 3}
    monkeypatch.setenv("EVOXC_OPT_LEVEL", "1")
    assert apply_env_options("xla", {}, "compile") == {"opt_level": 1}


def test_opt_level_explicit_wins_env(monkeypatch):
    """An explicit opt_level key always wins over EVOXC_OPT_LEVEL; the env is
    never applied for that key."""
    monkeypatch.setenv("EVOXC_OPT_LEVEL", "O2")
    assert apply_env_options("iree", {"opt_level": "O1"}, "compile") == {
        "opt_level": "O1"
    }
    # Other keys are still env-supplied alongside an explicit opt_level.
    monkeypatch.setenv("EVOXC_IREE_COMPILE_ARGS", "--flag-from-env")
    assert apply_env_options(
        "iree", {"opt_level": 0}, "compile"
    ) == {"opt_level": 0, "iree_compile_args": ["--flag-from-env"]}


def test_opt_level_env_stage_scoped(monkeypatch):
    """EVOXC_OPT_LEVEL applies at the compile stage only — never at lower, load
    or run (the env table declares the compile rows)."""
    monkeypatch.setenv("EVOXC_OPT_LEVEL", "O3")
    for stage in ("lower", "load", "run"):
        assert apply_env_options("iree", {}, stage) == {}
        assert apply_env_options("xla", {}, stage) == {}
        assert apply_env_options("tvm", {}, stage) == {}


@pytest.mark.parametrize("value", ["", "   ", "\t \n"])
def test_opt_level_blank_env_is_unset(monkeypatch, value):
    """An empty/whitespace EVOXC_OPT_LEVEL is treated as unset (no key)."""
    monkeypatch.setenv("EVOXC_OPT_LEVEL", value)
    for backend in ("iree", "xla", "tvm"):
        assert apply_env_options(backend, {}, "compile") == {}


@pytest.mark.parametrize("value", ["banana", "O4", "4", "-1", "O"])
def test_opt_level_malformed_env_raises_backend_error(monkeypatch, value):
    """A malformed EVOXC_OPT_LEVEL raises core.BackendError naming the variable
    and the value — the option is evoxc-defined, so its value is validated here
    (never passed through to the compiler)."""
    monkeypatch.setenv("EVOXC_OPT_LEVEL", value)
    with pytest.raises(core.BackendError) as excinfo:
        apply_env_options("iree", {}, "compile")
    message = str(excinfo.value)
    assert "EVOXC_OPT_LEVEL" in message
    assert value in message
    assert "opt_level" in message


def test_opt_level_env_numpy_noop(monkeypatch):
    """The numpy backend has no env table entry by design: EVOXC_OPT_LEVEL is
    always a no-op for it."""
    monkeypatch.setenv("EVOXC_OPT_LEVEL", "O2")
    assert apply_env_options("numpy", {}, "compile") == {}


def test_unknown_backend_noop(monkeypatch):
    """An unknown backend name (or None) is a no-op — no error, no keys."""
    monkeypatch.setenv("EVOXC_IREE_COMPILE_ARGS", "--flag")
    assert apply_env_options("no-such-backend", {}, "compile") == {}
    assert apply_env_options("no-such-backend",
                             {"existing": 1}, "compile") == {"existing": 1}
    assert apply_env_options(None, {}, "compile") == {}


def test_caller_dict_never_mutated(monkeypatch):
    """apply_env_options returns a copy; the caller's dict is untouched."""
    monkeypatch.setenv("EVOXC_IREE_COMPILE_ARGS", "--flag")
    opts = {"target_backends": ["llvm-cpu"]}
    resolved = apply_env_options("iree", opts, "compile")
    assert opts == {"target_backends": ["llvm-cpu"]}
    assert resolved is not opts
    assert resolved == {
        "target_backends": ["llvm-cpu"],
        "iree_compile_args": ["--flag"],
    }


# ---------------------------------------------------------------------------
# EVOXC_IREE_VULKAN_TARGET — the SPIR-V target profile (compile stage)
# ---------------------------------------------------------------------------


def test_vulkan_target_env_parsed(monkeypatch):
    """EVOXC_IREE_VULKAN_TARGET maps to the iree compile ``vulkan_target``
    option with only the whitespace stripped — the VALUE (the SPIR-V target
    profile, e.g. ``rtx4090``) is validated by the compiler, never by evoxc."""
    monkeypatch.setenv("EVOXC_IREE_VULKAN_TARGET", "rtx4090")
    assert apply_env_options("iree", {}, "compile") == {
        "vulkan_target": "rtx4090"
    }
    # A padded value is stripped, never passed through with the whitespace.
    monkeypatch.setenv("EVOXC_IREE_VULKAN_TARGET", "  rtx4090  ")
    assert apply_env_options("iree", {}, "compile") == {
        "vulkan_target": "rtx4090"
    }


@pytest.mark.parametrize("value", ["", "   ", "\t \n"])
def test_vulkan_target_blank_env_is_unset(monkeypatch, value):
    """An empty/whitespace EVOXC_IREE_VULKAN_TARGET is treated as unset (no key,
    never an empty-string target profile)."""
    monkeypatch.setenv("EVOXC_IREE_VULKAN_TARGET", value)
    assert apply_env_options("iree", {}, "compile") == {}


def test_vulkan_target_explicit_wins_env(monkeypatch):
    """An explicit vulkan_target key always wins over the env var; other keys
    are still env-supplied."""
    monkeypatch.setenv("EVOXC_IREE_VULKAN_TARGET", "rtx4090")
    assert apply_env_options(
        "iree", {"vulkan_target": "rx7900xtx"}, "compile"
    ) == {"vulkan_target": "rx7900xtx"}
    # Mixed: explicit target_backends + env-supplied vulkan_target.
    monkeypatch.setenv("EVOXC_IREE_COMPILE_ARGS", "--flag-from-env")
    assert apply_env_options(
        "iree", {"target_backends": ["vulkan-spirv"]}, "compile"
    ) == {
        "target_backends": ["vulkan-spirv"],
        "vulkan_target": "rtx4090",
        "iree_compile_args": ["--flag-from-env"],
    }


def test_vulkan_target_env_stage_scoped(monkeypatch):
    """EVOXC_IREE_VULKAN_TARGET applies at the iree compile stage only — never
    at lower, load or run, and never for the numpy backend, an absent backend
    name or another compiler backend."""
    monkeypatch.setenv("EVOXC_IREE_VULKAN_TARGET", "rtx4090")
    assert apply_env_options("iree", {}, "compile") == {
        "vulkan_target": "rtx4090"
    }
    for stage in ("lower", "load", "run"):
        assert apply_env_options("iree", {}, stage) == {}
    for backend in ("numpy", "xla", "tvm", None):
        assert apply_env_options(backend, {}, "compile") == {}


# ---------------------------------------------------------------------------
# The lower-stage rows (reserved exporter options)
# ---------------------------------------------------------------------------


def test_eigh_algorithm_env_parsed(monkeypatch):
    """EVOXC_EIGH_ALGORITHM maps to eigh_algorithm at every lower row with only
    the whitespace stripped — the value is validated by the stablehlo Writer;
    a blank value is unset."""
    monkeypatch.setenv("EVOXC_EIGH_ALGORITHM", "  jacobi  ")
    for backend in _LOWER_BACKENDS:
        assert apply_env_options(backend, {}, "lower") == {
            "eigh_algorithm": "jacobi"
        }
    # A blank value is unset (never an empty-string option).
    for blank in ("", "   "):
        monkeypatch.setenv("EVOXC_EIGH_ALGORITHM", blank)
        for backend in _LOWER_BACKENDS:
            assert apply_env_options(backend, {}, "lower") == {}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
        (" TRUE ", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
        (" FALSE ", False),
    ],
)
def test_split_reduction_env_parsed(monkeypatch, value, expected):
    """EVOXC_SPLIT_REDUCTION coerces to a real bool at every lower row (never
    the bool("0") is True trap)."""
    monkeypatch.setenv("EVOXC_SPLIT_REDUCTION", value)
    for backend in _LOWER_BACKENDS:
        assert apply_env_options(backend, {}, "lower") == {
            "split_reduction": expected
        }


def test_lower_env_blanks_are_unset(monkeypatch):
    """A blank lower-stage env value is treated as unset for both rows."""
    for blank in ("", "   ", "\t \n"):
        monkeypatch.setenv("EVOXC_EIGH_ALGORITHM", blank)
        monkeypatch.setenv("EVOXC_SPLIT_REDUCTION", blank)
        for backend in _LOWER_BACKENDS:
            assert apply_env_options(backend, {}, "lower") == {}


def test_lower_stage_explicit_wins_env(monkeypatch):
    """An explicit lower option key always wins over its env var; the other
    key is still env-supplied."""
    monkeypatch.setenv("EVOXC_SPLIT_REDUCTION", "1")
    monkeypatch.setenv("EVOXC_EIGH_ALGORITHM", "jacobi")
    assert apply_env_options("iree", {"split_reduction": False}, "lower") == {
        "split_reduction": False,
        "eigh_algorithm": "jacobi",
    }
    assert apply_env_options(
        "iree", {"eigh_algorithm": "jacobi_batched"}, "lower"
    ) == {"eigh_algorithm": "jacobi_batched", "split_reduction": True}


def test_lower_env_stage_scoped(monkeypatch):
    """The lower rows apply at the lower stage only — never at compile, load
    or run, and never for the numpy backend or an absent backend name."""
    monkeypatch.setenv("EVOXC_EIGH_ALGORITHM", "jacobi")
    monkeypatch.setenv("EVOXC_SPLIT_REDUCTION", "1")
    for backend in _LOWER_BACKENDS:
        assert apply_env_options(backend, {}, "lower") == {
            "eigh_algorithm": "jacobi",
            "split_reduction": True,
        }
        for stage in ("compile", "load", "run"):
            assert apply_env_options(backend, {}, stage) == {}
    assert apply_env_options("numpy", {}, "lower") == {}
    assert apply_env_options(None, {}, "lower") == {}


def test_split_reduction_malformed_env_raises_backend_error(monkeypatch):
    """A non-bool EVOXC_SPLIT_REDUCTION raises core.BackendError naming the
    variable, the value and the accepted forms (the shared bool coercer's
    message)."""
    monkeypatch.setenv("EVOXC_SPLIT_REDUCTION", "banana")
    with pytest.raises(core.BackendError) as excinfo:
        apply_env_options("iree", {}, "lower")
    message = str(excinfo.value)
    assert "invalid EVOXC_SPLIT_REDUCTION='banana'" in message
    assert (
        "accepted forms: True/False, 0/1, or one of 'true', 'false', 'yes', "
        "'no', 'on', 'off' (case-insensitive)"
    ) in message


# ---------------------------------------------------------------------------
# Stub backends (no real adapter imports)
# ---------------------------------------------------------------------------


class _IreeStubExecutable:
    """Minimal backend executable wrapper reporting ``backend_name="iree"``
    while delegating everything to the numpy backend executable."""

    backend_name = "iree"

    def __init__(self, inner):
        self._inner = inner

    @property
    def functions(self):
        return self._inner.functions

    @property
    def device(self):
        return self._inner.device

    def run(self, flat_input_tensors, options=None):
        return self._inner.run(flat_input_tensors, options)

    def save(self, path):
        return self._inner.save(path)


class _RecordingIreeStub(evoxc.backends.Backend):
    """Functional ``"iree"`` stub for full-path tests: delegates everything to
    the numpy backend but stamps its own name into the staged objects, and
    records the compile/load options it receives (per stage)."""

    name = "iree"
    capabilities = evoxc.backends.numpy_backend.capabilities
    recorded_compile_options: list = []
    recorded_load_options: list = []

    def lower(self, graph, options=None):
        lp = evoxc.backends.numpy_backend.lower(graph, options)
        return evoxc.backends.LoweredProgram(
            backend=self.name, signature=lp.signature, payload=lp.payload
        )

    def compile(self, lowered, options=None):
        _RecordingIreeStub.recorded_compile_options.append(dict(options or {}))
        art = evoxc.backends.numpy_backend.compile(
            evoxc.backends.LoweredProgram(
                backend="numpy", signature=lowered.signature, payload=lowered.payload
            ),
            options,
        )
        return evoxc.backends.CompiledArtifact(
            backend=self.name,
            signature=art.signature,
            target=art.target,
            payload=art.payload,
        )

    def load(self, artifact, device=None, options=None):
        _RecordingIreeStub.recorded_load_options.append(dict(options or {}))
        art = evoxc.backends.CompiledArtifact(
            backend="numpy",
            signature=artifact.signature,
            target=artifact.target,
            payload=artifact.payload,
        )
        return _IreeStubExecutable(evoxc.backends.numpy_backend.load(art, device))


@pytest.fixture
def iree_stub_registered():
    """Register the functional ``"iree"`` stub for a full-path test and
    restore the registry afterwards.

    The stub is installed by direct registry assignment (save/restore of the
    previous entry) so these tests also run when a real ``"iree"`` backend is
    already registered in this process (e.g. the compiler extra installed):
    the stub replaces it for the duration of the test and is never visible to
    other tests. Tests run sequentially, so no cross-test interference.
    """
    from evoxc.backends import registry as _registry

    stub = _RecordingIreeStub()
    _RecordingIreeStub.recorded_compile_options = []
    _RecordingIreeStub.recorded_load_options = []
    previous = _registry._registry.get("iree")
    _registry._registry["iree"] = stub
    try:
        yield stub
    finally:
        if previous is None:
            _registry._registry.pop("iree", None)
        else:
            _registry._registry["iree"] = previous
        _RecordingIreeStub.recorded_compile_options = []
        _RecordingIreeStub.recorded_load_options = []


# ---------------------------------------------------------------------------
# Full path through a registered "iree" stub (no real adapter import)
# ---------------------------------------------------------------------------


def test_full_path_env_compile_args_reaches_compile(monkeypatch, iree_stub_registered):
    """EVOXC_IREE_COMPILE_ARGS reaches the compile stage through ``evoxc.build``."""
    monkeypatch.setenv("EVOXC_IREE_COMPILE_ARGS", "--iree-llvmcpu-target-cpu=native")
    exe = evoxc.build(_linear, *_LINEAR_SPECS, backend="iree")
    recorded = _RecordingIreeStub.recorded_compile_options[-1]
    assert recorded["iree_compile_args"] == ["--iree-llvmcpu-target-cpu=native"]
    # The build sugar also inferred the device-based target_backends.
    assert recorded["target_backends"] == ["llvm-cpu"]
    assert exe.device == core.Device("cpu", 0)
    _assert_no_new_iree_import()


def test_full_path_explicit_compile_kwarg_beats_env(monkeypatch, iree_stub_registered):
    """An explicit iree_compile_args kwarg beats EVOXC_IREE_COMPILE_ARGS."""
    monkeypatch.setenv("EVOXC_IREE_COMPILE_ARGS", "--flag-from-env")
    exe = evoxc.build(
        _linear, *_LINEAR_SPECS, backend="iree",
        iree_compile_args=["--flag-explicit"],
    )
    assert _RecordingIreeStub.recorded_compile_options[-1][
        "iree_compile_args"
    ] == ["--flag-explicit"]
    assert exe.device == core.Device("cpu", 0)
    _assert_no_new_iree_import()


def test_full_path_env_opt_level_reaches_compile(monkeypatch, iree_stub_registered):
    """EVOXC_OPT_LEVEL reaches the compile stage through ``evoxc.build``,
    normalized to the int 0..3 form."""
    monkeypatch.setenv("EVOXC_OPT_LEVEL", "O2")
    exe = evoxc.build(_linear, *_LINEAR_SPECS, backend="iree")
    compile_recorded = _RecordingIreeStub.recorded_compile_options[-1]
    assert compile_recorded["opt_level"] == 2
    assert compile_recorded["target_backends"] == ["llvm-cpu"]
    assert exe.device == core.Device("cpu", 0)
    _assert_no_new_iree_import()


def test_full_path_explicit_opt_level_beats_env(monkeypatch, iree_stub_registered):
    """An explicit opt_level kwarg beats EVOXC_OPT_LEVEL end to end (passed
    through unnormalized — the adapter validates it)."""
    monkeypatch.setenv("EVOXC_OPT_LEVEL", "O2")
    exe = evoxc.build(
        _linear, *_LINEAR_SPECS, backend="iree", opt_level="O0",
    )
    assert _RecordingIreeStub.recorded_compile_options[-1][
        "opt_level"
    ] == "O0"
    assert exe.device == core.Device("cpu", 0)
    _assert_no_new_iree_import()


def test_full_path_env_vulkan_target_reaches_compile(
    monkeypatch, iree_stub_registered
):
    """EVOXC_IREE_VULKAN_TARGET reaches the compile stage through ``evoxc.build``
    with the padding stripped (the adapter derives
    ``--iree-vulkan-target=<profile>`` from it)."""
    monkeypatch.setenv("EVOXC_IREE_VULKAN_TARGET", "  rtx4090  ")
    exe = evoxc.build(_linear, *_LINEAR_SPECS, backend="iree")
    compile_recorded = _RecordingIreeStub.recorded_compile_options[-1]
    assert compile_recorded["vulkan_target"] == "rtx4090"
    assert compile_recorded["target_backends"] == ["llvm-cpu"]
    # Compile-scoped: the load stage never sees the target profile.
    assert "vulkan_target" not in _RecordingIreeStub.recorded_load_options[-1]
    assert exe.device == core.Device("cpu", 0)
    _assert_no_new_iree_import()


def test_full_path_vulkan_target_blank_env_sends_no_key(
    monkeypatch, iree_stub_registered
):
    """A blank EVOXC_IREE_VULKAN_TARGET never reaches the compile stage."""
    monkeypatch.setenv("EVOXC_IREE_VULKAN_TARGET", "   ")
    exe = evoxc.build(_linear, *_LINEAR_SPECS, backend="iree")
    compile_recorded = _RecordingIreeStub.recorded_compile_options[-1]
    assert "vulkan_target" not in compile_recorded
    assert compile_recorded == {"target_backends": ["llvm-cpu"]}
    assert exe.device == core.Device("cpu", 0)
    _assert_no_new_iree_import()


def test_full_path_explicit_vulkan_target_beats_env(
    monkeypatch, iree_stub_registered
):
    """An explicit vulkan_target kwarg beats EVOXC_IREE_VULKAN_TARGET end to end
    (passed through unnormalized — the adapter validates it)."""
    monkeypatch.setenv("EVOXC_IREE_VULKAN_TARGET", "rtx4090")
    exe = evoxc.build(
        _linear, *_LINEAR_SPECS, backend="iree",
        vulkan_target="rx7900xtx",
    )
    compile_recorded = _RecordingIreeStub.recorded_compile_options[-1]
    assert compile_recorded["vulkan_target"] == "rx7900xtx"
    assert exe.device == core.Device("cpu", 0)
    _assert_no_new_iree_import()


def test_full_path_env_runtime_args_reaches_load(monkeypatch, iree_stub_registered):
    """EVOXC_IREE_RUNTIME_ARGS reaches the load stage: ``evoxc.load`` applies the
    env internally, so even the ``build`` sugar (which passes no explicit
    load options) delivers the env-supplied load option."""
    monkeypatch.setenv("EVOXC_IREE_RUNTIME_ARGS", "--cuda_async_allocations=false")
    exe = evoxc.build(_linear, *_LINEAR_SPECS, backend="iree")
    assert _RecordingIreeStub.recorded_load_options[-1] == {
        "iree_runtime_args": ["--cuda_async_allocations=false"]
    }
    assert exe.device == core.Device("cpu", 0)
    _assert_no_new_iree_import()


def test_load_stage_explicit_runtime_args_beat_env(monkeypatch, iree_stub_registered):
    """At the explicit load stage, an explicit iree_runtime_args kwarg beats
    EVOXC_IREE_RUNTIME_ARGS."""
    monkeypatch.setenv("EVOXC_IREE_RUNTIME_ARGS", "--flag-from-env")
    graph = evoxc.trace(_linear, *_LINEAR_SPECS)
    lowered = evoxc.lower(graph, backend="iree")
    artifact = evoxc.compile(lowered, backend="iree")
    exe = evoxc.load(
        artifact, backend="iree",
        iree_runtime_args=["--cuda_async_allocations=false"],
    )
    assert _RecordingIreeStub.recorded_load_options[-1] == {
        "iree_runtime_args": ["--cuda_async_allocations=false"]
    }
    assert exe.device == core.Device("cpu", 0)
    _assert_no_new_iree_import()


def test_full_path_no_env_stages_stay_clean(monkeypatch, iree_stub_registered):
    """With no option env vars, the compile/load stages receive no option
    keys (the device-based target_backends inference is separate, documented
    env-defaults behavior)."""
    exe = evoxc.build(_linear, *_LINEAR_SPECS, backend="iree")
    compile_recorded = _RecordingIreeStub.recorded_compile_options[-1]
    assert "iree_compile_args" not in compile_recorded
    assert compile_recorded == {"target_backends": ["llvm-cpu"]}
    assert _RecordingIreeStub.recorded_load_options[-1] == {}
    assert exe.device == core.Device("cpu", 0)
    _assert_no_new_iree_import()


def test_full_path_run_options_forwarded_and_executes(monkeypatch, iree_stub_registered):
    """The stub executable runs end to end; per-run options are forwarded to
    the backend run (the numpy interpreter reads ``rank_context`` from the
    options dict — the documented numpy behavior, verified through the
    stub)."""
    exe = evoxc.build(_linear, *_LINEAR_SPECS, backend="iree")
    x, w, b = _linear_inputs()
    out = evoxc.run(exe, x, w, b)
    np.testing.assert_allclose(out.numpy(), _ref(x, w, b))
    out_rank = evoxc.run(exe, x, w, b, rank_context=evoxc.dist.RankContext(0, 1))
    np.testing.assert_allclose(out_rank.numpy(), _ref(x, w, b))
    _assert_no_new_iree_import()


def test_numpy_documented_ignore(monkeypatch, iree_stub_registered):
    """The numpy reference backend documents ignoring options: cross-backend
    option dicts (e.g. target_backends) pass through ``evoxc.run`` without
    error, and ``rank_context`` still flows into the interpreter."""
    exe = evoxc.build(_linear, *_LINEAR_SPECS, backend="numpy")
    x, w, b = _linear_inputs()
    out = evoxc.run(exe, x, w, b)
    np.testing.assert_allclose(
        evoxc.run(exe, x, w, b, target_backends=["cuda"]).numpy(),
        _ref(x, w, b),
    )
    np.testing.assert_allclose(
        evoxc.run(exe, x, w, b, rank_context=evoxc.dist.RankContext(0, 1)).numpy(),
        _ref(x, w, b),
    )
    assert out.dtype == np.dtype("float32")
    _assert_no_new_iree_import()


def test_bound_executable_backend_property(monkeypatch, iree_stub_registered):
    """``BoundExecutable.backend`` delegates to the wrapped executable and
    reports the stub's backend name."""
    exe = evoxc.build(_linear, *_LINEAR_SPECS, backend="iree")
    assert exe.backend == "iree"
    bound = evoxc.bind(exe, x=_linear_inputs()[0])
    assert bound.backend == "iree"
    w, b = _linear_inputs()[1:]
    x = _linear_inputs()[0]
    np.testing.assert_allclose(evoxc.run(bound, w, b).numpy(), _ref(x, w, b))
    _assert_no_new_iree_import()


# ---------------------------------------------------------------------------
# End-to-end export pins through the REAL iree adapter (gated, last in module)
#
# These are the only tests here that import a real compiler adapter — imported
# inside the test bodies so they skip (rather than fail) when iree is absent.
# They never call the import guard: the guard's baseline is refreshed per test
# by the autouse fixture, so placement is grouping, not correctness.
# ---------------------------------------------------------------------------


@evoxc.defn
def _eigh10(a):
    """A (10,10) eigh — the exporter option selects the while-loop shape."""
    return evoxc.eigh(a)


@evoxc.defn
def _reduce_min_10001(x):
    """A 10001-element reduce_min — above the split-reduction threshold."""
    return evoxc.reduce_min(x)


def _iree_lower_mlir(graph):
    """Lower ``graph`` for iree-llvm-cpu (an export, not a compile) and return
    the exported StableHLO text."""
    lowered = evoxc.lower(graph, backend="iree", target_backends=["llvm-cpu"])
    return lowered.payload["mlir_text"]


def _iree_lower_mlir_with_options(graph, options):
    """The same export with EXPLICIT lower options (the explicit-kwarg half of
    the precedence contract)."""
    lowered = evoxc.lower(
        graph, backend="iree", target_backends=["llvm-cpu"], **options
    )
    return lowered.payload["mlir_text"]


def test_eigh_algorithm_invalid_env_raises_at_lower(monkeypatch):
    """An unknown EVOXC_EIGH_ALGORITHM raises core.BackendError end to end from
    the stablehlo Writer (listing the accepted values), not from the option
    framework — the value is validated by the consumer."""
    pytest.importorskip("iree.compiler")
    pytest.importorskip("iree.runtime")
    monkeypatch.setenv("EVOXC_EIGH_ALGORITHM", "banana")
    graph = evoxc.trace(_eigh10, evoxc.TensorSpec((10, 10), evoxc.float32))
    with pytest.raises(core.BackendError) as excinfo:
        evoxc.lower(graph, backend="iree", target_backends=["llvm-cpu"])
    message = str(excinfo.value)
    assert "invalid eigh_algorithm option 'banana'" in message
    assert "expected one of jacobi, jacobi_batched" in message


def test_eigh_algorithm_env_reaches_exporter(monkeypatch):
    """EVOXC_EIGH_ALGORITHM=jacobi selects the sequential composition in the
    lowered text: 132 lines instead of the 148-line batched default, and
    byte-identical to a direct export with the same option."""
    pytest.importorskip("iree.compiler")
    pytest.importorskip("iree.runtime")
    from evoxc.backends import stablehlo

    graph = evoxc.trace(_eigh10, evoxc.TensorSpec((10, 10), evoxc.float32))
    default_text = _iree_lower_mlir(graph)
    assert len(default_text.splitlines()) == 148

    monkeypatch.setenv("EVOXC_EIGH_ALGORITHM", "jacobi")
    env_text = _iree_lower_mlir(graph)
    assert len(env_text.splitlines()) == 132
    # The env half and the direct-export half are independent: the env is
    # cleared before the explicit comparison.
    monkeypatch.delenv("EVOXC_EIGH_ALGORITHM")
    explicit_text = _iree_lower_mlir_with_options(
        graph, {"eigh_algorithm": "jacobi"}
    )
    assert env_text == explicit_text
    assert explicit_text == stablehlo.export(
        graph, options={"eigh_algorithm": "jacobi"}
    )
    assert env_text != default_text


def test_split_reduction_env_reaches_exporter(monkeypatch):
    """EVOXC_SPLIT_REDUCTION is honored at lower: 0 emits the single plain
    reduce (no split marker) byte-identically to an explicit option, 1 emits
    the two-reduce split with its marker comment."""
    pytest.importorskip("iree.compiler")
    pytest.importorskip("iree.runtime")

    graph = evoxc.trace(_reduce_min_10001, evoxc.TensorSpec((10001,), evoxc.float32))

    monkeypatch.setenv("EVOXC_SPLIT_REDUCTION", "0")
    off_text = _iree_lower_mlir(graph)
    assert off_text.count("stablehlo.reduce") == 1
    assert off_text.count("// evoxc split reduction:") == 0
    # Env cleared before the explicit-option comparison, so independent.
    monkeypatch.delenv("EVOXC_SPLIT_REDUCTION")
    explicit_off = _iree_lower_mlir_with_options(graph, {"split_reduction": False})
    assert off_text == explicit_off

    monkeypatch.setenv("EVOXC_SPLIT_REDUCTION", "1")
    on_text = _iree_lower_mlir(graph)
    assert on_text.count("stablehlo.reduce") == 2
    assert on_text.count("// evoxc split reduction:") == 1