"""Optional external-compiler adapters (IREE / XLA / TVM) — spec tests.

The three adapter modules under ``evoxc/backends/adapters/`` are OPTIONAL: their
third-party dependencies are never hard requirements of evoxc, and importing
``evoxc`` must never import them. Dependency shape per adapter: iree needs the
``iree-base-compiler`` / ``iree-base-runtime`` pip packages; xla needs a PJRT
C API plugin ``.so`` (exporting ``GetPjRtApi`` — resolved from
``EVOXC_PJRT_PLUGIN``, the ``plugin_path`` compile option, the fixed
well-known paths, or an installed ``jax_plugins/*/*.so`` such as
``pip install jax-rocm7-pjrt`` / ``jax-cuda12-pjrt``; never a hard dependency
of evoxc, and the adapter dlopen()s the plugin, it never imports jax); tvm
needs ``apache-tvm`` + ``jaxlib`` (jaxlib only for its bundled MLIR python
bindings). This file runs ALWAYS — no adapter dependency is required to
collect or run it — and every test is ORDER-INDEPENDENT: it passes
regardless of pytest file ordering and regardless of whether other files in
the same session already imported or registered adapters. That is achieved by
a strict in-process / subprocess split:

* Environment-sensitive assertions (``sys.modules`` hygiene, lazy adapter
  import, install hints) run in a FRESH SUBPROCESS (``_run_subprocess``)
  whose interpreter, ``sys.modules``, and backend registry are untouched by
  the pytest session.
* In-process assertions only pin facts that hold in every ordering: an
  unknown-name ``get`` raises ``evoxc.BackendError`` listing the registered
  backends (numpy is always among them), and the default backend stays numpy
  when no backend argument is given.

Expected adapter behaviors pinned here (once ``evoxc/backends/adapters/``
lands): a missing third-party dependency makes ``get(<adapter>)`` raise
``evoxc.BackendError`` carrying an actionable install/config hint (the
``evoxc[...]`` extra name for pip-installable deps); for xla — whose
dependency is a per-call plugin artifact and whose ``register()`` is
therefore UNCONDITIONAL — the same actionable guidance comes from the
explicit option-less preflight ``XlaBackend.check_available()`` (the pipeline
stages raise it from ``compile``/``load``, where they can see the options); a
present dependency makes ``get`` lazily import the adapter module and
register its backend.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import evoxc

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _run_subprocess(script):
    """Run *script* in a fresh interpreter with the repo on PYTHONPATH.

    Returns the completed process. A non-zero exit fails the test with both
    stdout and stderr included in the assertion message.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"subprocess exited with code {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    return proc


#: Names of the optional third-party deps (and their import families).
_ADAPTER_DEP_NAMES = ("iree", "jax", "jaxlib", "tvm")


# ---------------------------------------------------------------------------
# 1. importing evoxc never imports adapter dependencies (fresh subprocess)
# ---------------------------------------------------------------------------


def test_import_evoxc_does_not_import_adapter_deps():
    script = "\n".join(
        [
            "import sys",
            "import evoxc",
            "from evoxc import backends",
            "_DEPS = ('iree', 'jax', 'jaxlib', 'tvm')",
            "_loaded = [d for d in _DEPS if d in sys.modules]",
            "assert not _loaded, 'evoxc imported an optional adapter dependency at import time: ' + repr(_loaded)",
            "try:",
            "    backends.get('unknown_name_xyz')",
            "except evoxc.BackendError as e:",
            "    _msg = str(e)",
            "    assert 'unknown_name_xyz' in _msg, _msg",
            "    assert 'numpy' in _msg, _msg",
            "else:",
            "    raise AssertionError('get() did not raise BackendError')",
            "_loaded_after = [d for d in _DEPS if d in sys.modules]",
            "assert not _loaded_after, 'an unknown-name lookup probed adapter modules: ' + repr(_loaded_after)",
            "print('IMPORT_HYGIENE_OK')",
        ]
    )
    proc = _run_subprocess(script)
    assert "IMPORT_HYGIENE_OK" in proc.stdout


# ---------------------------------------------------------------------------
# 2. a missing third-party dep: BackendError carrying an install hint
# ---------------------------------------------------------------------------

#: (adapter name, blocked module, install-hint fragments the message must
#: contain at least one of, the expression that must raise ``BackendError``)
MISSING_DEP_CASES = [
    # A pip-installable dependency: the adapter's ``register()`` probes it,
    # so the plain backend lookup already raises.
    ("iree", "iree", ("evoxc[iree]", "iree-base-compiler"), "backends.get('iree')"),
    # xla has NO hard pip dependency AND its ``register()`` is UNCONDITIONAL:
    # its dependency is a per-call ``plugin_path``/env/discovered plugin ``.so``,
    # which a classmethod probe cannot see, so ``backends.get('xla')`` SUCCEEDS
    # with no plugin. The actionable message therefore comes from the explicit
    # option-less availability preflight ``XlaBackend.check_available()`` — the
    # same one ``compile``/``load`` raise from, where they DO see the options.
    # Blocking "jaxlib" is harmless (the adapter never imports it — it
    # dlopen()s the plugin) and pins the case to the plugin-missing path. The
    # subprocess also switches the jax_plugins auto-discovery scan off, so
    # "missing" really means missing on a box that HAS an installed JAX plugin.
    (
        "xla",
        "jaxlib",
        ("EVOXC_PJRT_PLUGIN", "plugin_path", "bazel build"),
        "_xla.XlaBackend.check_available()",
    ),
    ("tvm", "tvm", ("evoxc[tvm]", "apache-tvm"), "backends.get('tvm')"),
]

_MISSING_DEP_SCRIPT = """\
import importlib.util
import os
import sys

_BLOCKED = {blocked!r}

# Block BOTH import strategies before evoxc is imported: a direct
# ``import <dep>`` (sys.modules sentinel) and find_spec-based lazy probing
# (importlib.util.find_spec patched for the blocked name and its submodules).
sys.modules[_BLOCKED] = None
_real_find_spec = importlib.util.find_spec
def _blocked_find_spec(name, *args, **kwargs):
    if name == _BLOCKED or name.startswith(_BLOCKED + "."):
        return None
    return _real_find_spec(name, *args, **kwargs)
importlib.util.find_spec = _blocked_find_spec

import evoxc
from evoxc import backends
from evoxc.backends.adapters import xla as _xla

# Determinism for the xla case: plugin discovery must not accidentally
# succeed from the ambient environment (EVOXC_PJRT_PLUGIN), from the
# well-known search paths, or from an installed jax_plugins plugin — the case
# under test is "no plugin anywhere", on every box.
os.environ.pop("EVOXC_PJRT_PLUGIN", None)
os.environ["EVOXC_PJRT_PLUGIN_DISCOVERY"] = "0"
from evoxc.backends.adapters import xla_util as _xla_util
_xla_util._DEFAULT_PLUGIN_PATHS = ()
_xla_util._reset_plugin_discovery_cache()

try:
    {probe}
except evoxc.BackendError as e:
    msg = str(e)
    print(msg)
    hints = {hints}
    assert any(h in msg for h in hints), "missing install hint in: " + msg
else:
    raise SystemExit("the availability probe did not raise BackendError")
"""


@pytest.mark.parametrize(
    "adapter, blocked_module, hints, probe",
    MISSING_DEP_CASES,
    ids=[case[0] for case in MISSING_DEP_CASES],
)
def test_missing_dep_raises_backend_error_with_install_hint(
    adapter, blocked_module, hints, probe
):
    proc = _run_subprocess(
        _MISSING_DEP_SCRIPT.format(
            blocked=blocked_module, probe=probe, hints=hints
        )
    )
    # The subprocess prints the actual error message; its zero exit already
    # proves it was an ``evoxc.BackendError`` carrying an install hint (the
    # script asserts the hint itself). The exact wording is the adapter's —
    # e.g. "the iree backend requires the IREE Python packages ... `pip
    # install evoxc[iree]`" — and must NOT be pinned to the generic "unknown
    # backend" phrasing (that is reserved for genuinely unknown names,
    # pinned elsewhere).


# ---------------------------------------------------------------------------
# 3. in-process facts that hold in every ordering
# ---------------------------------------------------------------------------


def test_get_unknown_name_raises_in_process():
    with pytest.raises(evoxc.BackendError) as excinfo:
        evoxc.backends.get("unknown_name_xyz")
    msg = str(excinfo.value)
    assert "unknown_name_xyz" in msg
    # Registered backends are listed; numpy is always among them, no matter
    # which adapters earlier tests registered.
    assert "numpy" in msg


def test_default_backend_stays_numpy_without_adapter_involvement():
    def fn(x):
        return evoxc.add(x, x)

    lowered = evoxc.lower(evoxc.trace(fn, evoxc.TensorSpec((2, 3), evoxc.float32)))
    assert lowered.backend == "numpy"
    assert evoxc.backends.get("numpy") is evoxc.backends.numpy_backend


# ---------------------------------------------------------------------------
# 4. a present dep: get() lazily imports and registers the adapter
# ---------------------------------------------------------------------------


def test_get_lazily_imports_adapter_when_dep_present():
    pytest.importorskip("iree.compiler")
    script = "\n".join(
        [
            "import sys",
            "import evoxc",
            "from evoxc import backends",
            "assert 'iree' not in sys.modules, 'iree imported eagerly'",
            "backend = backends.get('iree')",
            "assert backend.name == 'iree', backend.name",
            "assert 'iree' in sys.modules, 'get() must lazily import the adapter module'",
            "assert backends.get('iree') is backend",
            "print('LAZY_IMPORT_OK')",
        ]
    )
    proc = _run_subprocess(script)
    assert "LAZY_IMPORT_OK" in proc.stdout
