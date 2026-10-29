"""Packaging contract: the `evoxc` distribution ships ONLY `evoxc`, never `evoxc_hf`.

`evoxc_hf` (the HuggingFace integration) is a separate top-level package living in
this repository checkout. It must never end up in a wheel/sdist of `evoxc`, and
`evoxc`'s hard runtime dependencies must stay numpy-only — the reference oracles
used to validate `evoxc_hf` (torch + transformers) live behind the optional `hf`
extra.

The distribution boundary is enforced by the **`uv_build`** backend, which ships
exactly ONE top-level module per distribution (`[tool.uv.build-backend]
module-name = "evoxc"` — a literal directory, not a glob) plus an explicit,
root-anchored `source-include` list for the sdist extras. There is no
`packages.find` glob to widen and no `evoxc.egg-info`/`SOURCES.txt` manifest that
an earlier build can poison, so the old stale-artifact trap is structurally gone
— pinned below by planting the exact artifact that used to leak `evoxc_hf/`.

These tests are numeric-free, network-free and fast: they read `pyproject.toml`
with `tomllib`, walk the checkout for real importable packages, inspect the
installed distribution's metadata, and (when `uv` is available) build a real
wheel + sdist from a throwaway copy of the checkout and inspect their contents.
"""

from __future__ import annotations

import fnmatch
import importlib.metadata
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
LICENSE = REPO_ROOT / "LICENSE"

# Directories that never contain distributable source packages.
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".genesis",
        ".eggs",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)

HF_TOP_LEVEL = "evoxc_hf"


def _pyproject() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def _build_system() -> dict:
    return _pyproject()["build-system"]


def _build_backend_config() -> dict:
    """`[tool.uv.build-backend]` — the module + sdist-extras contract."""
    return _pyproject()["tool"]["uv"]["build-backend"]


def _repo_packages() -> set[str]:
    """Every real importable package in the checkout (dirs with `__init__.py`)."""
    packages: set[str] = set()
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.endswith(".egg-info"))
        if "__init__.py" not in files:
            continue
        rel = Path(root).relative_to(REPO_ROOT)
        if rel == Path("."):  # the repo root itself is not a package
            continue
        packages.add(".".join(rel.parts))
    return packages


def _evoxc_packages(packages: Iterable[str]) -> set[str]:
    return {p for p in packages if p == "evoxc" or p.startswith("evoxc.")}


def _hf_packages(packages: Iterable[str]) -> set[str]:
    return {p for p in packages if p == HF_TOP_LEVEL or p.startswith(HF_TOP_LEVEL + ".")}


# --------------------------------------------------------------------------
# 1. the build backend: uv_build, ONE module, explicit sdist extras
# --------------------------------------------------------------------------


def test_repo_walk_sees_both_packages():
    """Guard against a vacuous test: both top-level packages exist on disk."""
    packages = _repo_packages()
    evoxc_pkgs = _evoxc_packages(packages)
    hf_pkgs = _hf_packages(packages)

    # The evoxc distribution is a real package tree (top level + subpackages).
    assert "evoxc" in evoxc_pkgs
    assert "evoxc.core" in evoxc_pkgs
    assert len(evoxc_pkgs) > 10, sorted(evoxc_pkgs)
    # ... and the HuggingFace integration is a separate top-level package tree.
    assert HF_TOP_LEVEL in hf_pkgs
    assert f"{HF_TOP_LEVEL}.models" in hf_pkgs


def test_build_backend_is_uv_build():
    """The backend is `uv_build` — no setuptools left anywhere in the config."""
    build = _build_system()
    assert build["build-backend"] == "uv_build", (
        f"build-backend is {build['build-backend']!r}: the distribution boundary is enforced by "
        "uv_build's single-module rule (a setuptools `packages.find` glob can be widened)"
    )
    requires = list(build["requires"])
    assert any(req.replace(" ", "").startswith("uv_build") for req in requires), requires
    assert not [req for req in requires if req.lower().startswith("setuptools")], requires

    # The whole `[tool.setuptools]` table is gone: leaving it behind would be
    # dead, misleading config that no backend reads.
    assert "setuptools" not in _pyproject().get("tool", {}), (
        "a stale [tool.setuptools] table survives in pyproject.toml"
    )


def test_build_backend_declares_the_flat_layout():
    """`module-root = ""` + `module-name = "evoxc"` = this repo's flat layout."""
    config = _build_backend_config()
    assert config["module-root"] == "", (
        f"module-root is {config['module-root']!r}: this repo is FLAT (`./evoxc/` at the root), "
        "and uv_build defaults to src-layout — without module-root = '' it fails with "
        "'Expected a Python module at: src/evoxc/__init__.py'"
    )
    assert config["module-name"] == "evoxc", (
        f"module-name is {config['module-name']!r}, not 'evoxc' — the distribution ships the "
        "`evoxc` module and nothing else"
    )

    module_dir = REPO_ROOT / config["module-root"] / config["module-name"]
    assert module_dir.is_dir(), f"{module_dir} (module-root/module-name) is not a directory"
    assert (module_dir / "__init__.py").is_file(), f"{module_dir} is not a Python package"

    # Anti-vacuity: the declared module really is the full `evoxc` package tree.
    packages = _evoxc_packages(_repo_packages())
    assert "evoxc" in packages
    assert len(packages) > 10, sorted(packages)


def test_module_name_cannot_reach_evoxc_hf():
    """A literal module name selects ONE directory — it has no wildcard to widen."""
    module_name = _build_backend_config()["module-name"]

    # Literal, root-level, single component: no glob, no nesting, no traversal.
    assert not any(ch in module_name for ch in "*?[]"), module_name
    assert "/" not in module_name and "\\" not in module_name, module_name
    assert module_name != "."

    hf_pkgs = _hf_packages(_repo_packages())
    assert hf_pkgs

    for name in ("evoxc_hf", "evoxc_hf.models", "evoxc_hf.tests", "evoxc_hf.tests.foo"):
        # Neither the package name nor a sample file below it is selected.
        assert not fnmatch.fnmatchcase(name, module_name), name
        assert not fnmatch.fnmatchcase(f"{name}/__init__.py", module_name), name
        # ... and the `evoxc` sibling prefix is not a prefix-match of `evoxc_hf`.
        assert not name.startswith(module_name + "/"), name


def test_source_include_covers_the_sdist_extras_only():
    """The sdist ships `tests/`, `assets/` and `conftest.py` — nothing else."""
    config = _build_backend_config()
    patterns = list(config["source-include"])

    # The test suite is the one non-negotiable extra (a built sdist must be
    # able to run `pytest tests/`).
    assert "tests/**" in patterns, (
        f"source-include={patterns} does not ship `tests/**`: the sdist of `evoxc` must contain "
        "the test suite that mirrors the package"
    )
    assert any(pat.startswith("tests/") for pat in patterns), patterns
    assert "assets/**" in patterns, patterns
    assert "conftest.py" in patterns, patterns

    # Every pattern is root-anchored and relative.
    for pattern in patterns:
        assert not pattern.startswith("/"), pattern
        parts = Path(pattern).parts
        assert ".." not in parts, pattern
        assert not pattern.startswith("."), pattern

    # Nothing may reach the sibling packages, the scratch area or a probe script.
    for pattern in patterns:
        for forbidden in (
            HF_TOP_LEVEL,
            "bench_logs",
            "probe_new_emitters.py",
            "pyproject.toml",
            ".git",
        ):
            assert not fnmatch.fnmatchcase(forbidden, pattern), (pattern, forbidden)
            assert not fnmatch.fnmatchcase(f"{forbidden}/nested_file.py", pattern), (
                pattern,
                forbidden,
            )

    # ... and the included paths really exist (a typo would silently ship nothing).
    for entry in ("tests", "assets"):
        assert (REPO_ROOT / entry).is_dir(), entry
    assert (REPO_ROOT / "conftest.py").is_file()


# --------------------------------------------------------------------------
# 2. dependency metadata: numpy-only runtime deps, optional hf oracle extra
# --------------------------------------------------------------------------


def _requirement_name(requirement: str) -> str:
    """`torch>=2.0` / `transformers[foo]>=4` / `x; python_version>'3'` -> name."""
    head = requirement.split(";", 1)[0].strip()
    for sep in "[<>=!~ (":
        head = head.split(sep, 1)[0]
    return head.strip().lower()


def test_runtime_dependencies_are_numpy_only():
    project = _pyproject()["project"]
    assert project["dependencies"] == ["numpy>=1.24"], (
        "evoxc's hard runtime dependencies must stay numpy-only (evoxc_hf and its "
        "oracles are optional)"
    )
    names = {_requirement_name(dep) for dep in project["dependencies"]}
    assert names == {"numpy"}


def test_hf_extra_declares_the_optional_hf_oracles():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "hf" in extras, "the optional `hf` extra must exist"

    names = {_requirement_name(dep) for dep in extras["hf"]}
    assert {"torch", "transformers"} <= names, extras["hf"]

    # Nothing HF-related may leak into the base install.
    base = {_requirement_name(dep) for dep in _pyproject()["project"]["dependencies"]}
    assert not ({"torch", "transformers", HF_TOP_LEVEL} | names) & base

    # `evoxc_hf` is a repo-checkout package, never a pip requirement of anything.
    all_requirements = list(_pyproject()["project"]["dependencies"])
    for deps in extras.values():
        all_requirements.extend(deps)
    assert HF_TOP_LEVEL not in {_requirement_name(dep) for dep in all_requirements}


# --------------------------------------------------------------------------
# 3. `import evoxc` works and never pulls the optional HF stack
# --------------------------------------------------------------------------


def test_import_evoxc_does_not_import_evoxc_hf_or_the_hf_oracles():
    probe = (
        "import sys, evoxc;"
        "evoxc.build, evoxc.evaluate, evoxc.lower, evoxc.trace, evoxc.tree_map;"
        "heavy = [m for m in ('evoxc_hf', 'torch', 'transformers') if m in sys.modules];"
        "assert not heavy, heavy;"
        "print(evoxc.__version__)"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env, timeout=120
    )
    assert proc.returncode == 0, proc.stderr[-4000:]


def test_installed_distribution_metadata_lists_only_evoxc():
    """The installed `evoxc` distribution must not claim `evoxc_hf` as its own."""
    try:
        dist = importlib.metadata.distribution("evoxc")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - env guard
        pytest.skip("the `evoxc` distribution is not installed in this environment")

    top_level = dist.read_text("top_level.txt")
    if top_level is not None:  # uv_build writes no top_level.txt — conditional on purpose
        names = [line.strip() for line in top_level.splitlines() if line.strip()]
        assert "evoxc" in names
        assert HF_TOP_LEVEL not in names, (
            f"installed evoxc metadata declares the top-level package {HF_TOP_LEVEL!r} "
            f"(top_level.txt={names}) — the [tool.uv.build-backend] module-name is too broad; "
            "reinstall after fixing pyproject.toml"
        )

    extras = set(dist.metadata.get_all("Provides-Extra") or ())
    assert "hf" in extras, sorted(extras)

    record = dist.read_text("RECORD")
    if record is not None:  # wheel installs only; legacy editable installs have none
        shipped = [
            line.split(",", 1)[0]
            for line in record.splitlines()
            if line.split(",", 1)[0].startswith(HF_TOP_LEVEL + "/")
        ]
        assert not shipped, f"installed evoxc distribution ships evoxc_hf files: {shipped[:5]}"


# --------------------------------------------------------------------------
# 4. release hygiene: single-sourced version, PEP 639 license, CHANGELOG
# --------------------------------------------------------------------------


def test_version_is_single_sourced_from_distribution_metadata():
    """`pyproject.toml` is the ONE version source — the package reads it back."""
    source = (REPO_ROOT / "evoxc" / "__init__.py").read_text()
    assert "importlib.metadata" in source, (
        "evoxc/__init__.py must read __version__ from the installed distribution metadata "
        "(importlib.metadata) instead of hardcoding it"
    )
    assert not re.search(r'^__version__\s*=\s*["\']', source, re.MULTILINE), (
        "evoxc/__init__.py hardcodes a __version__ literal — pyproject.toml must stay the single "
        "source of truth"
    )
    literal = _pyproject()["project"]["version"]
    assert f'"{literal}"' not in source, (
        f"the version literal {literal!r} is duplicated in evoxc/__init__.py — it would drift "
        "from pyproject.toml"
    )


def test_dunder_version_matches_pyproject():
    """`evoxc.__version__` == `[project].version` for an installed distribution."""
    expected = _pyproject()["project"]["version"]
    try:
        installed = importlib.metadata.version("evoxc")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - env guard
        pytest.skip("the `evoxc` distribution is not installed in this environment")
    assert installed == expected, (
        f"installed distribution is {installed!r} but pyproject.toml declares {expected!r} "
        "— reinstall (`uv sync`) after changing the version"
    )

    import evoxc

    assert evoxc.__version__ == expected, (
        f"evoxc.__version__ is {evoxc.__version__!r}, pyproject.toml says {expected!r}"
    )
    # The bare-checkout fallback must never be what an installed dist reports.
    assert evoxc.__version__ != "0.0.0+unknown"


def test_license_metadata_is_declared_once_as_spdx():
    """PEP 639: an SPDX expression + the license file, no legacy license table."""
    project = _pyproject()["project"]
    assert project["license"] == "Apache-2.0", project["license"]
    assert project["license-files"] == ["LICENSE"], project.get("license-files")

    assert LICENSE.is_file(), "LICENSE is missing from the repository root"
    # The canonical Apache 2.0 text is indented on its first line.
    assert LICENSE.read_text().lstrip().startswith("Apache License"), (
        "LICENSE does not look like the Apache 2.0 text"
    )

    # The deprecated form must NOT come back alongside the SPDX expression.
    assert not [c for c in project["classifiers"] if c.startswith("License ::")], project[
        "classifiers"
    ]


def test_changelog_documents_the_current_version():
    """A Keep-a-Changelog file with a release heading for `[project].version`."""
    version = _pyproject()["project"]["version"]
    assert CHANGELOG.is_file(), "CHANGELOG.md is missing from the repository root"

    text = CHANGELOG.read_text()
    assert text.startswith("# Changelog"), text[:80]
    assert f"## [{version}]" in text, (
        f"CHANGELOG.md has no `## [{version}]` release heading: "
        f"{[line for line in text.splitlines() if line.startswith('## ')]}"
    )
    assert f"[{version}]: " in text, (
        f"CHANGELOG.md has no `[{version}]: <url>` link reference for the release tag"
    )


# --------------------------------------------------------------------------
# 5. optional: build real distributions and pin their contents (skips w/o tooling)
# --------------------------------------------------------------------------


def _importable(module: str) -> bool:
    """True when `module` is a real importable module here (not a stray dir).

    A bare directory named `build/` in the repo root is importable as a
    namespace package, which would fool a plain `find_spec(...) is not None`
    check — require an actual origin file.
    """
    spec = importlib.util.find_spec(module)
    return spec is not None and spec.origin is not None


def _build_command(dist: str, out_dir: Path) -> list[str] | None:
    """Best-effort `sdist`/`wheel` build command, or None.

    `uv build` is preferred and runs ISOLATED deliberately: uv ships the
    `uv_build` backend bundled, so the build needs no network and no backend
    installed here (`--no-build-isolation` would defeat exactly that and demand
    `uv_build` in this environment). `python -m build --no-isolation` is only a
    fallback, and it needs the `uv_build` PyPI package importable.
    """
    uv = shutil.which("uv")
    if uv is not None:
        return [uv, "build", f"--{dist}", "--out-dir", str(out_dir)]
    if _importable("uv_build") and _importable("build"):
        return [
            sys.executable,
            "-m",
            "build",
            f"--{dist}",
            "--no-isolation",
            "--outdir",
            str(out_dir),
        ]
    return None


# Copied into a throwaway source tree so the build never writes into the
# checkout (build outputs and stale metadata are irrelevant to packaging anyway).
_COPY_IGNORE = shutil.ignore_patterns(
    ".git",
    ".genesis",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
    "*.egg-info",
    "__pycache__",
    ".pytest_cache",
    ".agents",
    "bench_logs",  # large scratch area, never a build input
)

_STALE_EGG_INFO = "evoxc.egg-info"


def _sanitize_source_tree(src: Path) -> list[str]:
    """Delete stale build artifacts from a copied tree (`rm -rf evoxc.egg-info build`).

    Kept as cheap hygiene for the copied tree (and because one test plants the
    artifact deliberately). Under `uv_build` these files are not build inputs
    at all — see `test_stale_egg_info_cannot_leak_evoxc_hf`.
    """
    removed: list[str] = []
    for candidate in (*src.glob("*.egg-info"), src / "build", src / "dist", src / ".eggs"):
        if candidate.is_dir():
            shutil.rmtree(candidate)
        elif candidate.exists():
            candidate.unlink()
        else:
            continue
        removed.append(candidate.name)
    return sorted(set(removed))


def _copy_checkout(dest: Path) -> Path:
    """Copy the checkout into `dest` (no VCS dirs, build artifacts or scratch)."""
    shutil.copytree(REPO_ROOT, dest, ignore=_COPY_IGNORE)
    _sanitize_source_tree(dest)  # defensive: a plant may predate the copy
    return dest


def _build_distribution(dist: str, src: Path, out_dir: Path) -> tuple[Path | None, str]:
    """Build `dist` from `src`; `(artifact, "")` or `(None, skip-reason)`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    command = _build_command(dist, out_dir)
    if command is None:
        return None, f"no build backend available (uv/uv_build+build) — {dist} leg skipped"

    proc = subprocess.run(command, cwd=str(src), capture_output=True, text=True, timeout=900)
    artifacts = sorted(out_dir.glob("evoxc-*.whl" if dist == "wheel" else "evoxc-*.tar.gz"))
    if proc.returncode != 0 or not artifacts:
        detail = (proc.stderr or proc.stdout or "no output")[-500:]
        return None, f"{dist} build unavailable: {detail}"
    return artifacts[0], ""


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory) -> Path:
    """ONE real `evoxc` wheel build, shared by every wheel leg below."""
    src = _copy_checkout(tmp_path_factory.mktemp("wheel") / "src")
    wheel, reason = _build_distribution("wheel", src, tmp_path_factory.mktemp("wheel-out"))
    if wheel is None:
        pytest.skip(reason)
    return wheel


@pytest.fixture(scope="module")
def built_sdist(tmp_path_factory) -> Path:
    """ONE real `evoxc` sdist build, shared by every sdist leg below."""
    src = _copy_checkout(tmp_path_factory.mktemp("sdist") / "src")
    sdist, reason = _build_distribution("sdist", src, tmp_path_factory.mktemp("sdist-out"))
    if sdist is None:
        pytest.skip(reason)
    return sdist


def _hf_entries(entries: Iterable[str]) -> list[str]:
    """Paths whose first component is `evoxc_hf` (a file OR a bare directory)."""
    return [e for e in entries if e == HF_TOP_LEVEL or e.startswith(HF_TOP_LEVEL + "/")]


def _relative_entries(archive: Path) -> list[str]:
    """Member paths of an sdist tarball, relative to the `evoxc-<version>/` root."""
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    root = names[0].split("/", 1)[0] if names else ""
    entries = [n.split("/", 1)[1] if n.startswith(root + "/") else n for n in names]
    return [e for e in entries if e]  # drop the archive's own root directory entry


def _artifact_entries(archive: Path) -> list[str]:
    """Member paths of a wheel (zip) or sdist (tar), normalized the same way."""
    if archive.suffix == ".whl":
        with zipfile.ZipFile(archive) as zf:
            return zf.namelist()
    return _relative_entries(archive)


_STALE_HINT = (
    "Cause and fix: the `[tool.uv.build-backend]` module-name decides what ships. `evoxc_hf` "
    "must never be reachable from it — it lives only in the checkout. Rebuild after fixing the "
    "backend config; a stale `evoxc.egg-info/` cannot cause this under uv_build."
)


def _assert_distribution_ships_no_evoxc_hf(entries: Sequence[str], kind: str) -> None:
    """Anti-vacuity (`evoxc/` + `pyproject.toml` are shipped) + zero `evoxc_hf` entries."""
    assert any(e.startswith("evoxc/") for e in entries), (
        f"{kind} does not ship the evoxc package (the evoxc_hf pin would be vacuous): {entries[:5]}"
    )
    assert "pyproject.toml" in entries, f"{kind} does not ship pyproject.toml: {entries[:5]}"
    hf_entries = _hf_entries(entries)
    assert not hf_entries, f"{kind} ships evoxc_hf source entries: {hf_entries[:5]}\n{_STALE_HINT}"


def test_wheel_contains_no_evoxc_hf(built_wheel):
    """A real `evoxc` wheel must contain no `evoxc_hf` module (skips without tooling)."""
    with zipfile.ZipFile(built_wheel) as archive:
        names = archive.namelist()
        record_name = next((n for n in names if n.endswith(".dist-info/RECORD")), None)
        assert record_name is not None, "wheel has no RECORD"
        record = archive.read(record_name).decode().splitlines()

    shipped = [line.split(",", 1)[0] for line in record if line.split(",", 1)[0]]
    assert shipped, "empty RECORD"
    assert any(n.startswith("evoxc/") for n in shipped), "wheel does not ship the evoxc package"
    hf_entries = _hf_entries(shipped)
    assert not hf_entries, f"wheel ships evoxc_hf files: {hf_entries[:5]}\n{_STALE_HINT}"


def test_wheel_ships_only_the_evoxc_module(built_wheel):
    """The wheel's root set is exactly `evoxc` + one `.dist-info` — nothing else."""
    with zipfile.ZipFile(built_wheel) as archive:
        names = archive.namelist()

    roots = sorted({n.split("/", 1)[0] for n in names})
    dist_infos = [r for r in roots if r.endswith(".dist-info")]
    modules = [r for r in roots if not r.endswith(".dist-info")]

    assert len(dist_infos) == 1, f"wheel must carry exactly one .dist-info: {roots}"
    assert modules == ["evoxc"], (
        f"wheel root set is {roots}: uv_build ships exactly the `evoxc` module — a second "
        "top-level name means the module-name/module-root config is wrong\n" + _STALE_HINT
    )
    for forbidden in (HF_TOP_LEVEL, "etl", "bench_logs", "assets", "conftest.py"):
        assert forbidden not in roots, (forbidden, roots)


def test_sdist_contains_no_evoxc_hf(built_sdist):
    """A real `evoxc` sdist must contain no `evoxc_hf` source (skips without tooling).

    The sdist is the distribution that ships `pyproject.toml` + the sources, so
    it is the one a stale build artifact could corrupt: it is built from a
    throwaway COPY of the checkout with `*.egg-info`, `build/` and `dist/`
    removed, and `uv_build` reads exactly the module dir + `source-include`.
    """
    _assert_distribution_ships_no_evoxc_hf(_relative_entries(built_sdist), "sdist")


# --------------------------------------------------------------------------
# 6. the distribution/brand name is pinned: `evoxc` (EvoX Compiler), never `etl`
# --------------------------------------------------------------------------


def _metadata_headers(text: str) -> dict[str, str]:
    """The RFC822 header block of a `METADATA` file (the body is the README)."""
    headers: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            break
        key, sep, value = line.partition(":")
        if sep:
            headers.setdefault(key.strip().lower(), value.strip())
    return headers


def test_wheel_is_named_evoxc(built_wheel):
    """The wheel is the `evoxc` distribution presenting itself as `EvoX Compiler`."""
    assert built_wheel.name.startswith("evoxc-"), (
        f"built wheel is {built_wheel.name!r}, not `evoxc-<version>-...whl`"
    )
    with zipfile.ZipFile(built_wheel) as archive:
        names = archive.namelist()
        dist_infos = sorted({n.split("/", 1)[0] for n in names if ".dist-info/" in n})
        assert len(dist_infos) == 1, f"wheel must carry exactly one .dist-info: {dist_infos}"
        assert dist_infos[0].startswith("evoxc-"), dist_infos
        metadata = _metadata_headers(archive.read(f"{dist_infos[0]}/METADATA").decode())

    assert metadata.get("name") == "evoxc", (
        f"wheel METADATA says Name: {metadata.get('name')!r}, not 'evoxc' — "
        "the rename to `evoxc` (EvoX Compiler) must not regress"
    )
    summary = metadata.get("summary", "")
    assert summary.startswith("EvoX Compiler"), (
        f"wheel METADATA says Summary: {summary!r}; it must start with 'EvoX Compiler'"
    )


def test_wheel_metadata_carries_pep639_license(built_wheel):
    """uv_build preserves the PEP 639 metadata: expression + shipped license file."""
    with zipfile.ZipFile(built_wheel) as archive:
        names = archive.namelist()
        dist_infos = sorted({n.split("/", 1)[0] for n in names if ".dist-info/" in n})
        assert len(dist_infos) == 1, dist_infos
        metadata = _metadata_headers(archive.read(f"{dist_infos[0]}/METADATA").decode())

        license_files = [n for n in names if n.endswith("/licenses/LICENSE")]
        assert license_files, (
            f"wheel ships no `<dist-info>/licenses/LICENSE` (entries: "
            f"{[n for n in names if 'licen' in n.lower()][:5]})"
        )

    assert metadata.get("license-expression") == "Apache-2.0", (
        f"wheel METADATA says License-Expression: {metadata.get('license-expression')!r}"
    )
    assert "LICENSE" in metadata.get("license-file", ""), (
        f"wheel METADATA says License-File: {metadata.get('license-file')!r}"
    )
    assert "license" not in metadata, (
        f"the legacy `License:` METADATA field must not be emitted alongside License-Expression "
        f"(saw {metadata.get('license')!r})"
    )


def test_wheel_contains_no_etl_package(built_wheel):
    """No wheel entry may sit under a top-level `etl` package (the pre-rename name)."""
    with zipfile.ZipFile(built_wheel) as archive:
        names = archive.namelist()
        top_level_name = next((n for n in names if n.endswith("top_level.txt")), None)
        top_level = None if top_level_name is None else archive.read(top_level_name).decode()

    etl_entries = [n for n in names if n.split("/")[0].rstrip("/") == "etl"]
    assert not etl_entries, (
        f"wheel ships a top-level `etl` package ({etl_entries[:5]}): the project is `evoxc` "
        "(EvoX Compiler) — a stale `etl/` source dir was picked up by the build"
    )
    if top_level is not None:  # uv_build writes no top_level.txt
        declared = [line.strip() for line in top_level.splitlines() if line.strip()]
        assert declared == ["evoxc"], (
            f"wheel top_level.txt declares {declared!r}, not ['evoxc'] — an `etl` package "
            "was reintroduced into the distribution"
        )


def test_sdist_is_named_evoxc(built_sdist):
    """The sdist has ONE `evoxc-<version>` root shipping `evoxc` + `tests`."""
    assert built_sdist.name.startswith("evoxc-"), (
        f"built sdist is {built_sdist.name!r}, not `evoxc-<version>.tar.gz`"
    )
    with tarfile.open(built_sdist) as tar:
        roots = sorted({n.split("/", 1)[0].rstrip("/") for n in tar.getnames()})
    assert len(roots) == 1, f"sdist must have a single top-level root, got {roots}"
    assert roots[0].startswith("evoxc-"), (
        f"sdist root is {roots[0]!r}, not 'evoxc-<version>' — the rename must not regress"
    )

    top = {entry.split("/", 1)[0] for entry in _relative_entries(built_sdist)}
    assert {"evoxc", "tests"} <= top, f"sdist ships {sorted(top)} (expected evoxc + tests)"
    stale = sorted(top & {"etl", HF_TOP_LEVEL})
    assert not stale, (
        f"sdist ships top-level {stale}: the project is `evoxc` (EvoX Compiler) and `evoxc_hf` "
        "lives only in the checkout — rebuild from a clean tree"
    )


# The exact stale artifact that used to poison a setuptools sdist: a leftover
# `SOURCES.txt` from an earlier build listing `evoxc_hf/` modules. One entry
# points at a file that no longer exists, pinning the old manifest pruning.
_STALE_SOURCES = (
    "evoxc_hf/__init__.py\n"
    "evoxc_hf/config.py\n"
    "evoxc_hf/stale_module_that_no_longer_exists.py\n"
    "evoxc/__init__.py\n"
    "pyproject.toml\n"
)


def test_stale_egg_info_cannot_leak_evoxc_hf(tmp_path):
    """The stale-`evoxc.egg-info` trap is STRUCTURALLY gone under `uv_build`.

    MEASURED under setuptools (68.1.2 and 84.0.0): an sdist built from a tree
    carrying a stale `evoxc.egg-info/SOURCES.txt` DID ship the listed `evoxc_hf/`
    paths that still existed on disk — even though `packages.find` excluded the
    package. `uv_build` reads no such manifest: the wheel is the module dir, and
    the sdist is the module dir + license/readme + root-anchored
    `source-include`. So this test plants the exact historical artifact into a
    real checkout copy (whose `evoxc_hf/` sources do exist, the trap's
    precondition) and asserts both artifacts still ship zero `evoxc_hf` entries.
    """
    src = _copy_checkout(tmp_path / "src")
    # The precondition a stale manifest needed to leak anything: real files.
    assert (src / HF_TOP_LEVEL).is_dir(), "the copied tree is missing evoxc_hf/"

    egg_info = src / _STALE_EGG_INFO
    egg_info.mkdir(parents=True, exist_ok=True)
    (egg_info / "PKG-INFO").write_text("Metadata-Version: 2.1\nName: evoxc\nVersion: 0.0.0\n")
    (egg_info / "SOURCES.txt").write_text(_STALE_SOURCES)

    for dist in ("wheel", "sdist"):
        archive, reason = _build_distribution(dist, src, tmp_path / f"stale-{dist}")
        if archive is None:
            pytest.skip(reason)
        entries = _artifact_entries(archive)
        shipped = _hf_entries(entries)
        assert not shipped, (
            f"{dist} built from a tree with a planted {_STALE_EGG_INFO}/SOURCES.txt ships "
            f"{shipped[:5]}: uv_build must never read a stale manifest\n{_STALE_HINT}"
        )
        # Anti-vacuity: the build succeeded and shipped the real package.
        assert any(e.startswith("evoxc/") for e in entries), f"{dist} shipped no evoxc sources"

    # The artifact is removable — proving the plant was a real, separate directory.
    removed = _sanitize_source_tree(src)
    assert _STALE_EGG_INFO in removed, (removed, sorted(p.name for p in src.iterdir()))
