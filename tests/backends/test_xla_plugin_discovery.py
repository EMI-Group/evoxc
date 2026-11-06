"""PJRT-plugin AUTO-DISCOVERY contract of the xla adapter (device-free).

``evoxc/backends/adapters/xla_util.py`` resolves the plugin ``.so`` the xla
adapter ``dlopen``-s without any pip binding (``jax``/``jaxlib`` are never
imported). This file pins that resolution surface — no real XLA build, no
GPU, no plugin package needed: the candidates are copies of the test-only fake
plugin (``_fake_pjrt_plugin.c``, built by gcc at test time through the sibling
``test_pjrt_ctypes_plugin._build_plugin``).

Pinned contract
---------------
* **Precedence** — ``_find_plugin_path`` resolves (1) ``options['plugin_path']``
  (an existing FILE: a non-file raises ``core.BackendError`` naming the source
  and ``repr(candidate)`` and NEVER falls through to a later step), (2) the
  ``EVOXC_PJRT_PLUGIN`` environment variable, (3) the fixed
  ``_DEFAULT_PLUGIN_PATHS`` tuple (``os.path.isfile`` probe), (4) the installed
  ``jax_plugins/*/*.so`` scan. It returns an ABSOLUTE path and raises
  ``core.BackendError(xla_util._PLUGIN_MISSING_MESSAGE)`` when nothing is
  found.
* **The scan** — ``_jax_plugins_candidates`` globs
  ``<sys.path entry>/jax_plugins/*/*.so``, dedupes ``sys.path`` entries by
  ``os.path.realpath``, skips empty (``""``) and non-directory entries, dedupes
  candidates by realpath and returns them SORTED BY PATH STRING (not by
  ``sys.path`` order) — so ``jax_plugins/xla_cpu/...`` deterministically beats
  ``jax_plugins/xla_cuda12|rocm7/...``. ``_probe_plugin`` accepts a candidate
  only when ``ctypes.CDLL`` succeeds AND the library exports ``GetPjRtApi`` /
  ``GetPjrtApi``; a failed dlopen (a plain text file named ``*.so``) or a
  missing symbol simply skips that candidate and never raises.
* **Caching** — probe results (``_plugin_probe_cache``, keyed by realpath) and
  the scan outcome (``_discovery_cache['scan']``: a path, or ``None`` for
  "scanned and found nothing") are cached for the process, so a second
  ``_find_plugin_path()`` neither re-globs nor re-``dlopen``-s.
  ``_reset_plugin_discovery_cache()`` clears both; successful ``CDLL`` handles
  stay alive in ``xla_util._plugin_handles`` and are NEVER unloaded (a later
  ``_reset`` therefore does not make a library available again — never assume
  an unload).
* **Opt-out + its validation** — ``EVOXC_PJRT_PLUGIN_DISCOVERY``: unset/empty
  enables the scan (``""`` is treated like unset), ``1/true/yes/on`` enables
  it, ``0/false/no/off`` (case-insensitive, whitespace stripped) disables it
  BEFORE the cache is touched, and any other non-empty value raises
  ``core.BackendError`` naming the variable, the raw value and the accepted
  spellings.
* **No jax imports** — a discovery run and a resolved path leave ``jax``,
  ``jaxlib`` and ``jax_plugins`` absent from ``sys.modules`` (the plugin is
  only ``dlopen``-ed).
* **Platform honesty** — the xla adapter derives device-kind support from the
  platform the plugin REPORTS (``xla._PLATFORM_KINDS``: ``cpu`` -> cpu,
  ``rocm`` -> rocm/hip, ``cuda`` -> cuda; an UNLISTED platform stays
  permissive). ``XlaBackend.load`` applies it on the CPU/``device=None`` path
  AND the accelerator path, and a kind the adapter does not model at all
  (``vulkan``) is rejected BEFORE any plugin work.

Ambient-box isolation (binding): a developer/CI machine may have a REAL
``jax_plugins/xla_rocm7/xla_rocm_plugin.so`` installed, which would otherwise
win a scan. Every test therefore runs with ``EVOXC_PJRT_PLUGIN`` /
``EVOXC_PJRT_PLUGIN_DISCOVERY`` deleted, ``_DEFAULT_PLUGIN_PATHS`` emptied and
the discovery cache reset (autouse fixture), and every pure-discovery test
ALSO replaces ``sys.path`` with only its own temporary directory — a
``syspath_prepend`` would not be enough, because candidates are ordered by
path string, not by ``sys.path`` position.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
from pathlib import Path
import pytest
import evoxc
from evoxc.backends.adapters import xla_util
from tests.backends.test_pjrt_ctypes_plugin import _build_plugin
#: The two environment variables the discovery chain reads (aliased so a typo
#: in this file cannot silently point a test at the wrong variable).
PLUGIN_ENV = xla_util._ENV_PLUGIN
DISCOVERY_ENV = xla_util._ENV_PLUGIN_DISCOVERY
# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fake_plugin(tmp_path_factory):
    """The standard fake plugin: reports the unlisted, permissive ``fake``."""
    return _build_plugin(
        tmp_path_factory.mktemp("discovery_fake_plugin"),
        "fake_pjrt_plugin.so",
    )
@pytest.fixture(scope="module")
def rocm_platform_plugin(tmp_path_factory):
    """A fake reporting the LISTED platform ``rocm`` (serves rocm/hip)."""
    return _build_plugin(
        tmp_path_factory.mktemp("discovery_fake_plugin_rocm"),
        "fake_pjrt_plugin_rocm.so",
        defines=("EVOXC_FAKE_PJRT_PLATFORM=rocm",),
    )
@pytest.fixture(scope="module")
def cpu_platform_plugin(tmp_path_factory):
    """A fake reporting the LISTED platform ``cpu`` (serves cpu only)."""
    return _build_plugin(
        tmp_path_factory.mktemp("discovery_fake_plugin_cpu"),
        "fake_pjrt_plugin_cpu.so",
        defines=("EVOXC_FAKE_PJRT_PLATFORM=cpu",),
    )
@pytest.fixture(autouse=True)
def isolated_discovery(monkeypatch):
    """Make every test independent of an ambient plugin install.
    Deletes both discovery env vars, empties the fixed well-known paths and
    resets the process-wide caches before AND after the test, so neither a
    real ``jax_plugins`` plugin and/or ``EVOXC_PJRT_PLUGIN`` on the box, nor a
    cache written by an earlier test, can influence the outcome.
    """
    monkeypatch.delenv(PLUGIN_ENV, raising=False)
    monkeypatch.delenv(DISCOVERY_ENV, raising=False)
    monkeypatch.setattr(xla_util, "_DEFAULT_PLUGIN_PATHS", ())
    xla_util._reset_plugin_discovery_cache()
    yield
    xla_util._reset_plugin_discovery_cache()
def _patch_search_path(monkeypatch, *directories):
    """Point the ``jax_plugins`` scan at EXACTLY these ``sys.path`` entries.
    Replaces ``sys.path`` (never prepends): candidates are sorted by path
    string, so an ambient site-packages ``jax_plugins`` plugin could otherwise
    win. Does NOT touch the discovery cache (tests that pin the cache use it).
    """
    monkeypatch.setattr(sys, "path", [str(directory) for directory in directories])
def _use_search_path(monkeypatch, *directories):
    """``_patch_search_path`` + a cache reset (a fresh scan of that path)."""
    _patch_search_path(monkeypatch, *directories)
    xla_util._reset_plugin_discovery_cache()
def _install_jax_plugin(root, platform_dir, name, source_so):
    """Copy a built fake plugin to ``<root>/jax_plugins/<platform_dir>/<name>``."""
    plugin_dir = Path(root) / "jax_plugins" / platform_dir
    plugin_dir.mkdir(parents=True, exist_ok=True)
    target = plugin_dir / name
    shutil.copyfile(source_so, target)
    return target
def _raising_scan(*args, **kwargs):
    """A ``_jax_plugins_candidates`` stand-in proving the scan never ran."""
    raise AssertionError("the jax_plugins scan must not run in this configuration")
def _build_without_entry_point(directory, so_name):
    """Build a shared object that does NOT export the PJRT entry point.
    Used for the "a ``.so`` that is not a PJRT plugin is skipped" leg; skips
    like the real fake plugin build when no C compiler is available.
    """
    cc = shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        pytest.skip("no C compiler (gcc/cc) available — cannot build a "
                    "non-PJRT shared object")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / f"{so_name}.c"
    source.write_text("int not_a_pjrt_symbol(void) { return 0; }\n",
                      encoding="utf-8")
    target = directory / so_name
    proc = subprocess.run(
        [cc, "-shared", "-fPIC", "-std=c99", "-O1", str(source), "-o", str(target)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        pytest.skip(f"building a non-PJRT shared object failed: {proc.stderr}")
    return target
def _missing_plugin_error():
    """Context manager: the actionable "no plugin found" ``BackendError``."""
    return pytest.raises(evoxc.BackendError, match="requires a PJRT C API plugin")
# ---------------------------------------------------------------------------
# (1) the scan finds an installed ``jax_plugins`` plugin
# ---------------------------------------------------------------------------
def test_scan_finds_plugin_under_jax_plugins(tmp_path, monkeypatch, fake_plugin):
    """``<sys.path>/jax_plugins/<platform>/*.so`` is found and made absolute."""
    installed = _install_jax_plugin(
        tmp_path, "xla_fake", "foo_plugin.so", fake_plugin
    )
    _use_search_path(monkeypatch, tmp_path)
    assert xla_util._jax_plugins_candidates() == [os.path.realpath(installed)]
    found = xla_util._find_plugin_path()
    assert os.path.isabs(found)
    assert found == os.path.realpath(installed)
    # The discovered path is a usable plugin (the fake's own identity).
    plugin = xla_util._load_plugin()
    assert plugin.path == found
# ---------------------------------------------------------------------------
# (2) precedence: plugin_path > EVOXC_PJRT_PLUGIN > fixed paths > scan
# ---------------------------------------------------------------------------
def test_plugin_path_beats_env_beats_scan(tmp_path, monkeypatch, fake_plugin):
    """The three sources are consulted in order; each step wins its own."""
    scanned = _install_jax_plugin(
        tmp_path / "scan_root", "xla_fake", "scanned_plugin.so", fake_plugin
    )
    from_env = tmp_path / "env" / "env_plugin.so"
    from_env.parent.mkdir(parents=True)
    shutil.copyfile(fake_plugin, from_env)
    from_option = tmp_path / "option" / "option_plugin.so"
    from_option.parent.mkdir(parents=True)
    shutil.copyfile(fake_plugin, from_option)
    _use_search_path(monkeypatch, tmp_path / "scan_root")
    # (1) the option wins while both other sources are available.
    monkeypatch.setenv(PLUGIN_ENV, str(from_env))
    assert xla_util._find_plugin_path(
        {"plugin_path": str(from_option)}
    ) == os.path.abspath(from_option)
    # (2) without the option the environment variable wins over the scan.
    assert xla_util._find_plugin_path() == os.path.abspath(from_env)
    # (3) with neither, the scan supplies the path.
    monkeypatch.delenv(PLUGIN_ENV)
    assert xla_util._find_plugin_path() == os.path.realpath(scanned)
def test_env_alone_never_consults_the_scan(tmp_path, monkeypatch, fake_plugin):
    """A resolved ``EVOXC_PJRT_PLUGIN`` short-circuits the ``jax_plugins`` scan."""
    _install_jax_plugin(tmp_path, "xla_fake", "foo_plugin.so", fake_plugin)
    _use_search_path(monkeypatch, tmp_path)
    monkeypatch.setattr(xla_util, "_jax_plugins_candidates", _raising_scan)
    monkeypatch.setenv(PLUGIN_ENV, str(fake_plugin))
    assert xla_util._find_plugin_path() == os.path.abspath(fake_plugin)
def test_fixed_well_known_path_is_used_before_the_scan(
    tmp_path, monkeypatch, fake_plugin
):
    """Step (3): an existing ``_DEFAULT_PLUGIN_PATHS`` entry beats the scan."""
    known = tmp_path / "well_known" / "pjrt_c_api_cpu_plugin.so"
    known.parent.mkdir()
    shutil.copyfile(fake_plugin, known)
    _install_jax_plugin(tmp_path, "xla_fake", "foo_plugin.so", fake_plugin)
    _use_search_path(monkeypatch, tmp_path)
    monkeypatch.setattr(xla_util, "_DEFAULT_PLUGIN_PATHS", (str(known),))
    assert xla_util._find_plugin_path() == os.path.abspath(known)
    # A configured-but-missing well-known path is simply skipped, so the scan
    # still resolves the installed plugin (the isfile probe, never an error).
    monkeypatch.setattr(
        xla_util, "_DEFAULT_PLUGIN_PATHS", (str(tmp_path / "absent.so"),)
    )
    assert xla_util._find_plugin_path() == os.path.realpath(
        tmp_path / "jax_plugins" / "xla_fake" / "foo_plugin.so"
    )
def test_broken_plugin_path_option_raises_and_never_falls_back(
    tmp_path, monkeypatch, fake_plugin
):
    """A non-file ``plugin_path`` is loud — it never falls through to the scan."""
    _install_jax_plugin(tmp_path, "xla_fake", "foo_plugin.so", fake_plugin)
    _use_search_path(monkeypatch, tmp_path)
    monkeypatch.setattr(xla_util, "_jax_plugins_candidates", _raising_scan)
    missing = tmp_path / "not_there.so"
    with pytest.raises(evoxc.BackendError) as excinfo:
        xla_util._find_plugin_path({"plugin_path": str(missing)})
    message = str(excinfo.value)
    assert "options['plugin_path']" in message
    assert repr(str(missing)) in message
    assert "is not a file" in message
def test_broken_env_plugin_raises_and_never_falls_back(
    tmp_path, monkeypatch, fake_plugin
):
    """A non-file ``EVOXC_PJRT_PLUGIN`` is loud too (same source-named error)."""
    _install_jax_plugin(tmp_path, "xla_fake", "foo_plugin.so", fake_plugin)
    _use_search_path(monkeypatch, tmp_path)
    monkeypatch.setattr(xla_util, "_jax_plugins_candidates", _raising_scan)
    missing = tmp_path / "not_there.so"
    monkeypatch.setenv(PLUGIN_ENV, str(missing))
    with pytest.raises(evoxc.BackendError) as excinfo:
        xla_util._find_plugin_path()
    message = str(excinfo.value)
    assert PLUGIN_ENV in message
    assert repr(str(missing)) in message
# ---------------------------------------------------------------------------
# (3) candidate hygiene: entry-point probe, dlopen failure, dedupe, ordering
# ---------------------------------------------------------------------------
def test_candidate_without_entry_point_is_skipped(
    tmp_path, monkeypatch, fake_plugin
):
    """A dlopen-able ``.so`` without ``GetPjRtApi`` is probed and skipped."""
    root = tmp_path / "root"
    # Alphabetically FIRST, so the valid plugin can only win by the probe.
    dummy = _build_without_entry_point(
        root / "jax_plugins" / "xla_a", "aaa.so"
    )
    valid = _install_jax_plugin(
        root, "xla_b", "zzz_plugin.so", fake_plugin
    )
    _use_search_path(monkeypatch, root)
    candidates = xla_util._jax_plugins_candidates()
    assert candidates == [os.path.realpath(dummy), os.path.realpath(valid)]
    assert xla_util._probe_plugin(str(dummy)) is False
    assert xla_util._probe_plugin(str(valid)) is True
    assert xla_util._find_plugin_path() == os.path.realpath(valid)
def test_plain_text_file_named_so_is_skipped(tmp_path, monkeypatch):
    """A failed ``dlopen`` skips the candidate instead of raising."""
    plugin_dir = tmp_path / "root" / "jax_plugins" / "xla_fake"
    plugin_dir.mkdir(parents=True)
    bogus = plugin_dir / "bogus_plugin.so"
    bogus.write_text("this is not a shared object\n", encoding="utf-8")
    _use_search_path(monkeypatch, tmp_path / "root")
    assert xla_util._jax_plugins_candidates() == [os.path.realpath(bogus)]
    assert xla_util._probe_plugin(str(bogus)) is False
    with _missing_plugin_error() as excinfo:
        xla_util._find_plugin_path()
    assert excinfo.value.args[0] == xla_util._PLUGIN_MISSING_MESSAGE
def test_search_path_and_candidate_dedup(tmp_path, monkeypatch, fake_plugin):
    """Duplicate ``sys.path`` entries and a symlinked plugin yield ONE candidate."""
    root = tmp_path / "root"
    installed = _install_jax_plugin(root, "xla_fake", "foo_plugin.so", fake_plugin)
    # A symlinked COPY of the same library inside another platform directory.
    alias_dir = root / "jax_plugins" / "xla_alias"
    alias_dir.mkdir()
    (alias_dir / "alias_plugin.so").symlink_to(installed)
    link_root = tmp_path / "link_root"
    link_root.symlink_to(root)
    _use_search_path(
        monkeypatch,
        root,
        str(root) + os.sep,
        link_root,
        str(link_root) + os.sep,
    )
    assert xla_util._jax_plugins_candidates() == [os.path.realpath(installed)]
    assert xla_util._find_plugin_path() == os.path.realpath(installed)
def test_scan_skips_empty_and_nondirectory_entries(
    tmp_path, monkeypatch, fake_plugin
):
    """`""` (the cwd) and file entries are skipped; a real dir still scans."""
    root = tmp_path / "root"
    installed = _install_jax_plugin(root, "xla_fake", "foo_plugin.so", fake_plugin)
    plain_file = tmp_path / "not_a_directory"
    plain_file.write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(
        sys, "path", ["", str(plain_file), str(root)]
    )
    xla_util._reset_plugin_discovery_cache()
    assert xla_util._jax_plugins_candidates() == [os.path.realpath(installed)]
def test_two_valid_platform_plugins_pick_the_sorted_first(
    tmp_path, monkeypatch, fake_plugin
):
    """Candidates are sorted by PATH STRING — ``xla_cpu`` beats ``xla_rocm7``."""
    root = tmp_path / "root"
    cpu_first = _install_jax_plugin(root, "xla_cpu", "a_plugin.so", fake_plugin)
    _install_jax_plugin(root, "xla_rocm7", "b_plugin.so", fake_plugin)
    _use_search_path(monkeypatch, root)
    # Documented rule: `.../jax_plugins/xla_cpu/...` < `.../jax_plugins/xla_rocm7/...`
    # as a plain string, so the platform directory NAME decides the winner —
    # not the ``sys.path`` order (the rocm7 root is listed nowhere special).
    assert xla_util._jax_plugins_candidates() == [
        os.path.realpath(cpu_first),
        os.path.realpath(root / "jax_plugins" / "xla_rocm7" / "b_plugin.so"),
    ]
    assert xla_util._find_plugin_path() == os.path.realpath(cpu_first)
# ---------------------------------------------------------------------------
# (4) caching: the scan runs once per process (a path OR a "found nothing")
# ---------------------------------------------------------------------------
def test_successful_scan_is_cached(tmp_path, monkeypatch, fake_plugin):
    """A second lookup reuses the cached scan result (no glob, no dlopen)."""
    root = tmp_path / "root"
    installed = _install_jax_plugin(root, "xla_fake", "foo_plugin.so", fake_plugin)
    _use_search_path(monkeypatch, root)
    assert xla_util._find_plugin_path() == os.path.realpath(installed)
    # The search path is gone AND the scan is booby-trapped: only the cache
    # can answer now.
    monkeypatch.setattr(sys, "path", [])
    monkeypatch.setattr(xla_util, "_jax_plugins_candidates", _raising_scan)
    assert xla_util._find_plugin_path() == os.path.realpath(installed)
    # ... and a reset drops the cached answer, so the lookup fails again.
    monkeypatch.setattr(xla_util, "_jax_plugins_candidates", lambda: [])
    xla_util._reset_plugin_discovery_cache()
    with _missing_plugin_error():
        xla_util._find_plugin_path()
def test_empty_scan_is_cached_too(tmp_path, monkeypatch, fake_plugin):
    """A "scanned, found nothing" outcome is cached as ``None``."""
    empty = tmp_path / "empty"
    empty.mkdir()
    _use_search_path(monkeypatch, empty)
    with _missing_plugin_error():
        xla_util._find_plugin_path()
    assert xla_util._discovery_cache["scan"] is None
    # A plugin appearing afterwards does NOT revive the cached negative scan.
    root = tmp_path / "root"
    _install_jax_plugin(root, "xla_fake", "foo_plugin.so", fake_plugin)
    _patch_search_path(monkeypatch, root)
    with _missing_plugin_error() as excinfo:
        xla_util._find_plugin_path()
    assert excinfo.value.args[0] == xla_util._PLUGIN_MISSING_MESSAGE
    # A reset re-enables scanning, and then the plugin IS found.
    xla_util._reset_plugin_discovery_cache()
    assert xla_util._find_plugin_path() == os.path.realpath(
        root / "jax_plugins" / "xla_fake" / "foo_plugin.so"
    )
def test_probe_results_are_cached_per_realpath(tmp_path, monkeypatch, fake_plugin):
    """Probe outcomes are memoized by realpath (a second probe does no dlopen)."""
    plugin_dir = tmp_path / "root" / "jax_plugins" / "xla_fake"
    plugin_dir.mkdir(parents=True)
    installed = plugin_dir / "foo_plugin.so"
    installed.write_bytes(b"this is not a shared object\n")
    real = os.path.realpath(installed)
    assert xla_util._probe_plugin(str(installed)) is False
    assert xla_util._plugin_probe_cache[real] is False
    # Swap a REAL plugin in at the same path: the cached verdict stands...
    shutil.copyfile(fake_plugin, installed)
    assert xla_util._probe_plugin(str(installed)) is False
    # ... until a reset re-probes, and then it is accepted (and cached).
    xla_util._reset_plugin_discovery_cache()
    assert xla_util._probe_plugin(str(installed)) is True
    assert xla_util._plugin_probe_cache[real] is True
    assert real in xla_util._plugin_handles
# ---------------------------------------------------------------------------
# (5) EVOXC_PJRT_PLUGIN_DISCOVERY: opt-out and its validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw", ["0", "off", "no", "false", " OFF ", "False"])
def test_discovery_disabled_skips_the_scan(tmp_path, monkeypatch, fake_plugin, raw):
    """A falsey spelling disables step (4) and leaves the scan untouched."""
    root = tmp_path / "root"
    installed = _install_jax_plugin(root, "xla_fake", "foo_plugin.so", fake_plugin)
    _use_search_path(monkeypatch, root)
    scan = xla_util._jax_plugins_candidates
    monkeypatch.setattr(xla_util, "_jax_plugins_candidates", _raising_scan)
    monkeypatch.setenv(DISCOVERY_ENV, raw)
    with _missing_plugin_error() as excinfo:
        xla_util._find_plugin_path()
    assert DISCOVERY_ENV in str(excinfo.value)
    # The disabled scan touched nothing: no cached outcome, so enabling it
    # again resolves the SAME lookup immediately.
    assert "scan" not in xla_util._discovery_cache
    monkeypatch.setattr(xla_util, "_jax_plugins_candidates", scan)
    monkeypatch.setenv(DISCOVERY_ENV, "1")
    assert xla_util._find_plugin_path() == os.path.realpath(installed)
def test_discovery_disabled_still_honours_explicit_sources(
    tmp_path, monkeypatch, fake_plugin
):
    """The opt-out only removes step (4); option/env keep working."""
    _use_search_path(monkeypatch, tmp_path)
    monkeypatch.setattr(xla_util, "_jax_plugins_candidates", _raising_scan)
    monkeypatch.setenv(DISCOVERY_ENV, "0")
    assert xla_util._find_plugin_path(
        {"plugin_path": str(fake_plugin)}
    ) == os.path.abspath(fake_plugin)
    monkeypatch.setenv(PLUGIN_ENV, str(fake_plugin))
    assert xla_util._find_plugin_path() == os.path.abspath(fake_plugin)
def test_empty_discovery_value_behaves_as_unset(tmp_path, monkeypatch, fake_plugin):
    """``EVOXC_PJRT_PLUGIN_DISCOVERY=""`` = unset -> the scan runs."""
    root = tmp_path / "root"
    installed = _install_jax_plugin(root, "xla_fake", "foo_plugin.so", fake_plugin)
    _use_search_path(monkeypatch, root)
    monkeypatch.setenv(DISCOVERY_ENV, "")
    assert xla_util._plugin_discovery_enabled() is True
    assert xla_util._find_plugin_path() == os.path.realpath(installed)
def test_malformed_discovery_value_is_loud(tmp_path, monkeypatch, fake_plugin):
    """An unrecognised value raises, naming the var, the value and the set."""
    root = tmp_path / "root"
    _install_jax_plugin(root, "xla_fake", "foo_plugin.so", fake_plugin)
    _use_search_path(monkeypatch, root)
    monkeypatch.setattr(xla_util, "_jax_plugins_candidates", _raising_scan)
    monkeypatch.setenv(DISCOVERY_ENV, "maybe")
    with pytest.raises(evoxc.BackendError) as excinfo:
        xla_util._find_plugin_path()
    message = str(excinfo.value)
    assert DISCOVERY_ENV in message
    assert "'maybe'" in message
    accepted = sorted(xla_util._DISCOVERY_FALSEY | xla_util._DISCOVERY_TRUTHY)
    assert accepted.__repr__() in message
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, True),   # unset
        ("", True),     # empty = unset
        ("   ", True),  # whitespace-only = unset
        ("1", True),
        ("true", True),
        ("YES", True),
        (" on ", True),
        ("0", False),
        ("false", False),
        ("No", False),
        (" OFF ", False),
    ],
)
def test_plugin_discovery_enabled_truth_table(raw, expected, monkeypatch):
    """The ``EVOXC_PJRT_PLUGIN_DISCOVERY`` truth table (and its tolerance)."""
    if raw is None:
        monkeypatch.delenv(DISCOVERY_ENV, raising=False)
    else:
        monkeypatch.setenv(DISCOVERY_ENV, raw)
    assert xla_util._plugin_discovery_enabled() is expected
# ---------------------------------------------------------------------------
# (6) discovery never imports jax / jaxlib
# ---------------------------------------------------------------------------
def test_discovery_does_not_import_jax(tmp_path, monkeypatch, fake_plugin):
    """A discovery run and ``_find_plugin_path`` leave no jax modules behind."""
    # An earlier module in the same process may have imported them (tvm suites
    # do ``importorskip("jaxlib")``): clear FIRST, assert afterwards.
    for name in ("jax", "jaxlib", "jax_plugins"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    root = tmp_path / "root"
    installed = _install_jax_plugin(root, "xla_fake", "foo_plugin.so", fake_plugin)
    _use_search_path(monkeypatch, root)
    assert xla_util._jax_plugins_candidates() == [os.path.realpath(installed)]
    assert xla_util._find_plugin_path() == os.path.realpath(installed)
    for name in ("jax", "jaxlib", "jax_plugins"):
        assert name not in sys.modules, f"discovery imported {name!r}"
# ---------------------------------------------------------------------------
# (7) the missing-plugin message is actionable (all four steps named)
# ---------------------------------------------------------------------------
def test_missing_plugin_message_names_every_step(tmp_path, monkeypatch):
    """The error names the option, the env var, the scan and the build recipes."""
    empty = tmp_path / "empty"
    empty.mkdir()
    _use_search_path(monkeypatch, empty)
    with _missing_plugin_error() as excinfo:
        xla_util._find_plugin_path()
    message = str(excinfo.value)
    for token in (
        "EVOXC_PJRT_PLUGIN",
        "plugin_path",
        "jax_plugins",
        "jax-rocm7-pjrt",
        "bazel build",
    ):
        assert token in message, f"{token!r} missing from: {message}"
    assert message == xla_util._PLUGIN_MISSING_MESSAGE
# ---------------------------------------------------------------------------
# (8) the reported platform must physically serve the loaded device kind
# ---------------------------------------------------------------------------
#: One compiled artifact per fake plugin (the compile stage runs once).
_ARTIFACTS: dict[str, "evoxc.backends.CompiledArtifact"] = {}
def _artifact_for(plugin_path):
    """``lower`` + ``compile`` a tiny ``add(x, x)`` graph against one plugin."""
    key = str(plugin_path)
    artifact = _ARTIFACTS.get(key)
    if artifact is None:
        def fn(x):
            return evoxc.add(x, x)
        graph = evoxc.trace(fn, evoxc.TensorSpec((2, 3), evoxc.float32))
        lowered = evoxc.lower(graph, backend="xla")
        artifact = evoxc.compile(lowered, backend="xla", plugin_path=key)
        _ARTIFACTS[key] = artifact
    return artifact
@pytest.fixture
def xla_backend(monkeypatch, request):
    """The registered xla adapter, with ``EVOXC_PJRT_PLUGIN`` set to a fake.
    The env var (not ``sys.path``) is used because ``backends.get("xla")``
    probes the plugin through ``register()`` on its very first call.
    """
    plugin = request.getfixturevalue(request.param)
    monkeypatch.setenv(PLUGIN_ENV, str(plugin))
    return evoxc.backends.get("xla"), plugin
@pytest.mark.parametrize("xla_backend", ["rocm_platform_plugin"], indirect=True)
def test_rocm_platform_rejects_cpu_device(xla_backend):
    """A ``rocm`` plugin cannot back a CPU executable (honest placement)."""
    backend, plugin = xla_backend
    artifact = _artifact_for(plugin)
    with pytest.raises(evoxc.BackendError) as excinfo:
        backend.load(artifact, device=evoxc.core.Device("cpu", 0))
    message = str(excinfo.value)
    assert "reports platform 'rocm'" in message
    assert "cannot serve" in message
    assert "Device(kind='cpu'" in message
@pytest.mark.parametrize("xla_backend", ["rocm_platform_plugin"], indirect=True)
def test_rocm_platform_rejects_the_default_cpu_device(xla_backend):
    """``device=None`` means ``Device('cpu', 0)`` — and is rejected too."""
    backend, plugin = xla_backend
    artifact = _artifact_for(plugin)
    with pytest.raises(evoxc.BackendError) as excinfo:
        backend.load(artifact)
    message = str(excinfo.value)
    assert "reports platform 'rocm'" in message
    assert "cannot serve" in message
@pytest.mark.parametrize("xla_backend", ["rocm_platform_plugin"], indirect=True)
def test_rocm_platform_accepts_its_own_device(xla_backend):
    """Positive control: the plugin's own kind loads (device-resident)."""
    backend, plugin = xla_backend
    artifact = _artifact_for(plugin)
    assert artifact.runtime_dependencies["plugin"] == "rocm fake-cpu 0.0.1"
    executable = backend.load(artifact, device=evoxc.core.Device("rocm", 0))
    try:
        assert executable.device == evoxc.core.Device("rocm", 0)
    finally:
        executable.close()
@pytest.mark.parametrize("xla_backend", ["rocm_platform_plugin"], indirect=True)
def test_rocm_platform_rejects_cuda_device(xla_backend):
    """A ``rocm`` plugin does not serve ``Device('cuda', 0)`` either."""
    backend, plugin = xla_backend
    artifact = _artifact_for(plugin)
    with pytest.raises(evoxc.BackendError) as excinfo:
        backend.load(artifact, device=evoxc.core.Device("cuda", 0))
    message = str(excinfo.value)
    assert "reports platform 'rocm'" in message
    assert "('rocm', 'hip')" in message
@pytest.mark.parametrize("xla_backend", ["cpu_platform_plugin"], indirect=True)
def test_cpu_platform_rejects_accelerator_device(xla_backend):
    """A ``cpu`` plugin cannot back a device-resident accelerator executable."""
    backend, plugin = xla_backend
    artifact = _artifact_for(plugin)
    with pytest.raises(evoxc.BackendError) as excinfo:
        backend.load(artifact, device=evoxc.core.Device("rocm", 0))
    message = str(excinfo.value)
    assert "reports platform 'cpu'" in message
    assert "cannot serve" in message
    # ... while its own kind loads fine.
    executable = backend.load(artifact, device=evoxc.core.Device("cpu", 0))
    try:
        assert executable.device == evoxc.core.Device("cpu", 0)
    finally:
        executable.close()
@pytest.mark.parametrize("xla_backend", ["fake_plugin"], indirect=True)
def test_unlisted_platform_stays_permissive(xla_backend):
    """The fake's ``fake`` platform is unlisted -> any modelled kind loads."""
    backend, plugin = xla_backend
    artifact = _artifact_for(plugin)
    assert artifact.runtime_dependencies["plugin"] == "fake fake-cpu 0.0.1"
    for device in (evoxc.core.Device("cpu", 0), evoxc.core.Device("cuda", 0)):
        executable = backend.load(artifact, device=device)
        try:
            assert executable.device == device
        finally:
            executable.close()
@pytest.mark.parametrize(
    "xla_backend",
    ["fake_plugin", "rocm_platform_plugin", "cpu_platform_plugin"],
    indirect=True,
)
def test_vulkan_device_is_rejected_before_plugin_work(xla_backend):
    """``vulkan`` is not an xla kind at all — rejected before any plugin call.

    This check runs BEFORE plugin discovery/client work (``_check_kind_served``
    without a platform name), so it holds for every reported platform.
    """
    backend, plugin = xla_backend
    artifact = _artifact_for(plugin)
    with pytest.raises(evoxc.BackendError) as excinfo:
        backend.load(artifact, device=evoxc.core.Device("vulkan", 0))
    message = str(excinfo.value)
    assert "supports CPU ('cpu') and accelerator" in message
    assert "vulkan" in message
