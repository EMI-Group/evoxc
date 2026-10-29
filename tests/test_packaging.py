"""Packaging contract: ONE repository, ONE uv workspace, TWO published distributions.

Both are built from this single workspace (one `uv.lock` at the root):

* ``evoxc`` — the WORKSPACE-ROOT project at the repo root, **flat layout**
  (`./evoxc/` is the import package, `module-root = ""`);
* ``evoxc-hf`` — the workspace MEMBER at ``./evoxc_hf/``, **src layout**
  (``evoxc_hf/src/evoxc_hf/``, ``module-root = "src"``).

The member's distribution name (`evoxc-hf`) differs from its import name
(`evoxc_hf`) — that is what ``module-name`` declares, and since `uv_build`
packages exactly ``<module-root>/<module-name>`` neither distribution can ever
contain the sibling: a structural guarantee, not a glob that could be widened.

The HuggingFace integration lives in the member, so its reference oracles (torch
+ transformers) sit behind ``evoxc-hf[hf]``: ``evoxc`` has no ``hf`` extra, stays
numpy-only at runtime, and ``import evoxc`` never imports ``evoxc_hf``, torch or
transformers. The member hard-depends on ``evoxc`` (``evoxc = { workspace = true
}`` for development, ``evoxc>=0.1.0`` published).

Numeric- and network-free, fast, deterministic: both `pyproject.toml` files and
`uv.lock` are read with `tomllib`, the checkout is walked, installed metadata is
inspected via `importlib.metadata`, and REAL artifacts come from ONE
`uv build --all-packages` over a throwaway copy of the checkout. Every
environment-dependent step SKIPS with a clear reason; only a genuine contract
violation fails.
"""

from __future__ import annotations

import fnmatch
import importlib
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
from typing import Iterable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
MEMBER_DIR = REPO_ROOT / "evoxc_hf"
MEMBER_PYPROJECT = MEMBER_DIR / "pyproject.toml"
UV_LOCK = REPO_ROOT / "uv.lock"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
LICENSE = REPO_ROOT / "LICENSE"

# The two projects of this workspace: import module, distribution name, and the
# module `__init__.py` that reports its version.
ROOT_MODULE = ROOT_DIST = "evoxc"
ROOT_INIT = REPO_ROOT / ROOT_MODULE / "__init__.py"
MEMBER_MODULE = "evoxc_hf"
MEMBER_DIST = "evoxc-hf"
MEMBER_INIT = MEMBER_DIR / "src" / MEMBER_MODULE / "__init__.py"

# `evoxc`'s extras are EXACTLY these six; the HF oracle extra lives on the member.
ROOT_EXTRAS = frozenset({"interop", "dev", "iree", "tvm", "compilers", "bench"})

# Directories that never contain distributable source packages.
_SKIP_DIRS = frozenset({".git", ".genesis", ".eggs", ".venv", ".mypy_cache", ".pytest_cache"})
_SKIP_DIRS |= frozenset({"__pycache__", "build", "dist", "node_modules", "venv"})


# ---- helpers: read the two project files, walk the checkout ------------------


def _load(path: Path) -> dict:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _pyproject() -> dict:
    """The workspace-ROOT project (`evoxc`)."""
    return _load(PYPROJECT)


def _member_pyproject() -> dict:
    """The workspace MEMBER project (`evoxc-hf`)."""
    return _load(MEMBER_PYPROJECT)


def _project_version(path: Path) -> str:
    """The `[project].version` declared by ONE project file (never the sibling's)."""
    return _load(path)["project"]["version"]


def _build_backend_config(document: dict) -> dict:
    """`[tool.uv.build-backend]` — the one-module contract of that project."""
    return document["tool"]["uv"]["build-backend"]


def _requirement_name(requirement: str) -> str:
    """`evoxc>=0.1.0` / `torch; extra == 'hf'` / `x[foo]>=1` -> `evoxc`/`torch`/`x`."""
    head = requirement.split(";", 1)[0].strip()
    for sep in "[<>=!~ (":
        head = head.split(sep, 1)[0]
    return head.strip().lower()


def _extras_of(requirement: str) -> set[str]:
    """The extra names a `Requires-Dist` line is gated on (`; extra == 'hf'`)."""
    marker = requirement.split(";", 1)[1] if ";" in requirement else ""
    return set(re.findall(r"extra\s*==\s*['\"]([^'\"]+)['\"]", marker))


# A NUMERIC PEP 440 check — no third-party `packaging` import: release segment,
# optional pre/post/dev suffix, optional local version label.
_PEP440_RE = re.compile(
    r"^\d+(?:\.\d+)*"
    r"(?:[-_.]?(?:a|b|c|rc|alpha|beta|pre|preview)[-_.]?\d+)?"
    r"(?:(?:-\d+)|(?:[-_.]?(?:post|rev|r)[-_.]?\d+))?"
    r"(?:[-_.]?dev[-_.]?\d+)?"
    r"(?:\+[a-z0-9]+(?:[-_.][a-z0-9]+)*)?$",
    re.IGNORECASE,
)


def _package_dirs() -> set[Path]:
    """Every real importable package directory of the checkout (has `__init__.py`)."""
    packages: set[Path] = set()
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.endswith(".egg-info"))
        if "__init__.py" not in files:
            continue
        rel = Path(root).relative_to(REPO_ROOT)
        if rel != Path("."):  # the repo root itself is not a package
            packages.add(rel)
    return packages


def _packages_below(*parts: str) -> set[Path]:
    """The package dir at `parts` plus every package nested below it."""
    prefix = Path(*parts)
    return {p for p in _package_dirs() if p == prefix or prefix in p.parents}


# ---- 1. ONE uv workspace; `evoxc_hf` is a member project of it ---------------


def test_workspace_declares_the_evoxc_hf_member():
    """`[tool.uv.workspace] members = ["evoxc_hf"]` and the member really exists."""
    workspace = _pyproject()["tool"]["uv"]["workspace"]
    assert workspace["members"] == ["evoxc_hf"], workspace["members"]
    assert MEMBER_DIR.is_dir(), f"{MEMBER_DIR} (the workspace member) is not a directory"
    assert MEMBER_PYPROJECT.is_file(), f"{MEMBER_PYPROJECT} is missing"
    assert UV_LOCK.is_file(), "no `uv.lock` at the root: ONE workspace means ONE lockfile"


def test_the_single_lock_covers_both_distributions():
    """`uv.lock` is the one lockfile, and both projects are editable members of it."""
    lock = tomllib.loads(UV_LOCK.read_text())
    assert set(lock["manifest"]["members"]) == {ROOT_DIST, MEMBER_DIST}, lock["manifest"]
    packages = {pkg["name"]: pkg for pkg in lock["package"] if pkg["name"] in (ROOT_DIST, MEMBER_DIST)}
    assert set(packages) == {ROOT_DIST, MEMBER_DIST}, sorted(packages)
    assert packages[ROOT_DIST]["source"] == {"editable": "."}, packages[ROOT_DIST]["source"]
    assert packages[MEMBER_DIST]["source"] == {"editable": "evoxc_hf"}, packages[MEMBER_DIST]["source"]
    # The member is locked against the workspace root project, not a PyPI release.
    assert any(dep.get("name") == ROOT_DIST for dep in packages[MEMBER_DIST]["dependencies"]), (
        f"`{MEMBER_DIST}`'s lock entry does not depend on `{ROOT_DIST}`"
    )


def test_workspace_member_resolves_evoxc_locally_but_publishes_a_requirement():
    """Development resolves the sibling project; the published metadata still requires it."""
    member = _member_pyproject()
    sources = member["tool"]["uv"]["sources"]
    assert sources["evoxc"] == {"workspace": True}, (
        f"`[tool.uv.sources] evoxc` is {sources.get('evoxc')!r}: development must resolve the "
        "workspace-root project, never a PyPI release"
    )

    dependencies = member["project"]["dependencies"]
    assert {_requirement_name(dep) for dep in dependencies} >= {ROOT_DIST}, dependencies
    requirement = next(dep for dep in dependencies if _requirement_name(dep) == ROOT_DIST)
    assert requirement.startswith(f"{ROOT_DIST}>="), (
        f"the published requirement is {requirement!r}: it must carry the `{ROOT_DIST}>=` lower "
        "bound (`workspace = true` only affects development resolution)"
    )


def test_pytest_collects_both_suites():
    """One pytest config collects `tests/` AND the member's colocated suite."""
    ini = _pyproject()["tool"]["pytest"]["ini_options"]
    testpaths = [Path(entry) for entry in ini["testpaths"]]
    assert Path("tests") in testpaths, ini["testpaths"]
    assert Path("evoxc_hf/tests") in testpaths, (
        f"testpaths are {ini['testpaths']!r}: the member keeps its suite under `evoxc_hf/tests/`"
    )
    for entry in testpaths:
        assert (REPO_ROOT / entry).is_dir(), entry

    pythonpath = [Path(entry) for entry in ini["pythonpath"]]
    assert Path("evoxc_hf/src") in pythonpath, (
        f"pythonpath is {ini['pythonpath']!r}: the member's src layout must be importable even in "
        "a venv created by a plain `uv sync` (which does not install the member)"
    )
    assert (MEMBER_DIR / "src" / MEMBER_MODULE).is_dir()


def test_both_projects_use_the_uv_build_backend():
    """Both projects are real `uv_build` projects — no setuptools anywhere."""
    requirements = []
    for path in (PYPROJECT, MEMBER_PYPROJECT):
        document = _load(path)
        build = document["build-system"]
        assert build["build-backend"] == "uv_build", (path.name, build["build-backend"])
        requires = [req.replace(" ", "") for req in build["requires"]]
        assert any(req.startswith("uv_build") for req in requires), (path.name, requires)
        assert not [req for req in requires if req.lower().startswith("setuptools")], requires
        assert "setuptools" not in document.get("tool", {}), (
            f"{path.name}: a stale [tool.setuptools] table survives — dead config no backend reads"
        )
        requirements.append(next(req for req in requires if req.startswith("uv_build")))
    # Same backend family: both projects pin the same uv_build requirement.
    assert requirements[0] == requirements[1], requirements


def test_root_source_include_ships_only_the_sdist_extras():
    """The `evoxc` sdist adds `tests/`, `assets/`, `conftest.py`, `CHANGELOG.md` — nothing else."""
    patterns = list(_build_backend_config(_pyproject())["source-include"])
    for entry in ("tests/**", "assets/**", "conftest.py", "CHANGELOG.md"):
        assert entry in patterns, (
            f"source-include={patterns} does not ship `{entry}`: a built sdist must be able to "
            "run the packaged test suite (and its release-hygiene leg)"
        )
    assert any(pat.startswith("tests/") for pat in patterns), patterns

    # Every pattern is root-anchored and relative.
    for pattern in patterns:
        assert not pattern.startswith(("/", ".")), pattern
        assert ".." not in Path(pattern).parts, pattern

    # No pattern may reach the workspace member, the scratch area, a root probe
    # script, the packaging config or the VCS directory.
    for pattern in patterns:
        for forbidden in (MEMBER_DIR.name, "bench_logs", "probe_new_emitters.py", "pyproject.toml", ".git"):
            assert not fnmatch.fnmatchcase(forbidden, pattern), (pattern, forbidden)
            assert not fnmatch.fnmatchcase(f"{forbidden}/nested_file.py", pattern), (pattern, forbidden)

    # ... and the included paths really exist (a typo would silently ship nothing).
    for entry in ("tests", "assets"):
        assert (REPO_ROOT / entry).is_dir(), entry
    for entry in ("conftest.py", "CHANGELOG.md"):
        assert (REPO_ROOT / entry).is_file(), entry


# ---- 2. identity and dependency metadata of the two distributions ------------


def _normalized_dist_name(name: str) -> str:
    """PEP 503 name normalization: `evoxc-hf` / `Evoxc.HF` -> `evoxc_hf`."""
    return re.sub(r"[-_.]+", "_", name).lower()


def test_distribution_names_differ_from_their_import_names():
    """`evoxc` / `evoxc-hf` are the distribution names; `evoxc` / `evoxc_hf` are the modules."""
    root, member = _pyproject(), _member_pyproject()
    assert root["project"]["name"] == ROOT_DIST, root["project"]["name"]
    assert _build_backend_config(root)["module-name"] == ROOT_MODULE
    assert member["project"]["name"] == MEMBER_DIST, member["project"]["name"]
    assert _build_backend_config(member)["module-name"] == MEMBER_MODULE

    # The member is the one whose distribution name differs from its import name —
    # `module-name` is what makes `pip install evoxc-hf` provide `import evoxc_hf`.
    assert member["project"]["name"] != MEMBER_MODULE
    assert _normalized_dist_name(member["project"]["name"]) == MEMBER_MODULE, (
        f"{member['project']['name']!r} does not normalize to the declared module {MEMBER_MODULE!r}"
    )


def test_versions_are_valid_pep440_and_independent():
    """Each project is versioned on its own — valid PEP 440, never a cross-project pin."""
    root_version = _project_version(PYPROJECT)
    member_version = _project_version(MEMBER_PYPROJECT)
    assert _PEP440_RE.match(root_version), f"{PYPROJECT.name}: {root_version!r} is not PEP 440"
    assert _PEP440_RE.match(member_version), f"{MEMBER_PYPROJECT.name}: {member_version!r} is not PEP 440"

    # Each version comes from its OWN file — no dynamic version, no central table.
    for path in (PYPROJECT, MEMBER_PYPROJECT):
        project = _load(path)["project"]
        assert project["version"], path
        assert "version" not in project.get("dynamic", ()), path
        assert "version" not in _load(path).get("tool", {}).get("uv", {}), path
        assert project["version"] == _project_version(path), path

    # The check is sensitive, so the (empty) disagreements above are not vacuous ...
    assert root_version != f"{member_version}.drift"
    # ... and the two versions are NEVER required to be equal: each distribution is
    # tagged and released separately, so no `assert root == member` belongs here.


def test_both_projects_share_the_python_floor():
    root = _pyproject()["project"]["requires-python"]
    member = _member_pyproject()["project"]["requires-python"]
    assert root == member, f"`requires-python` differs between the projects ({root!r} vs {member!r})"
    assert root.startswith(">="), root


def test_evoxc_runtime_dependencies_are_numpy_only():
    project = _pyproject()["project"]
    assert project["dependencies"] == ["numpy>=1.24"], "evoxc's runtime deps must stay numpy-only"
    assert {_requirement_name(dep) for dep in project["dependencies"]} == {"numpy"}


def test_evoxc_extras_are_exactly_the_six_without_hf():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert set(extras) == set(ROOT_EXTRAS), f"evoxc extras are {sorted(extras)}, not {sorted(ROOT_EXTRAS)}"
    assert "hf" not in extras, "`evoxc` must not carry an `hf` extra: it lives on `evoxc-hf`"


def test_member_requires_evoxc_and_keeps_the_hf_oracles_optional():
    project = _member_pyproject()["project"]
    base = {_requirement_name(dep) for dep in project["dependencies"]}
    assert {ROOT_DIST, "numpy"} <= base, project["dependencies"]
    assert not ({"torch", "transformers"} & base), f"HF oracles leaked into {project['dependencies']}"
    extras = project["optional-dependencies"]
    assert {"hf", "dev"} <= set(extras), sorted(extras)
    assert {"torch", "transformers"} <= {_requirement_name(dep) for dep in extras["hf"]}, extras["hf"]


# ---- 3. layout boundary: flat `evoxc` at the root, src layout for the member ----


def test_repo_walk_sees_both_package_trees():
    """Anti-vacuity: the checkout really holds two package trees to pin."""
    packages = _package_dirs()
    assert Path(ROOT_MODULE) in packages
    assert Path(MEMBER_DIR.name, "src", MEMBER_MODULE) in packages
    assert len(_packages_below(ROOT_MODULE)) > 10, sorted(_packages_below(ROOT_MODULE))
    assert Path(MEMBER_DIR.name, "src", MEMBER_MODULE, "models") in packages


def test_root_package_is_the_flat_evoxc_package():
    """`module-root = ""` + `module-name = "evoxc"` = this repo's flat layout."""
    config = _build_backend_config(_pyproject())
    assert config["module-root"] == "", f"module-root is {config['module-root']!r}: the root project is FLAT"
    assert config["module-name"] == ROOT_MODULE, config["module-name"]
    module_dir = REPO_ROOT / config["module-root"] / config["module-name"]
    assert module_dir == REPO_ROOT / ROOT_MODULE, module_dir
    assert (module_dir / "__init__.py").is_file(), f"{module_dir} is not a Python package"
    packages = _packages_below(ROOT_MODULE)
    assert Path(ROOT_MODULE, "core") in packages, sorted(packages)
    assert len(packages) > 10, sorted(packages)


def test_member_module_dir_is_the_src_layout_package():
    """`module-root = "src"` + `module-name = "evoxc_hf"` = the member's src layout."""
    config = _build_backend_config(_member_pyproject())
    assert config["module-root"] == "src", config["module-root"]
    assert config["module-name"] == MEMBER_MODULE, config["module-name"]
    module_dir = MEMBER_DIR / config["module-root"] / config["module-name"]
    assert module_dir == MEMBER_DIR / "src" / MEMBER_MODULE, module_dir
    assert (module_dir / "__init__.py").is_file(), f"{module_dir} is not a Python package"
    packages = _packages_below(MEMBER_DIR.name, "src", MEMBER_MODULE)
    assert Path(MEMBER_DIR.name, "src", MEMBER_MODULE, "models") in packages, sorted(packages)


def test_no_flat_evoxc_hf_copy_can_shadow_the_distribution():
    """The member module exists ONLY under `evoxc_hf/src/` — never as a flat copy."""
    assert Path(MEMBER_DIR.name) not in _package_dirs(), "`evoxc_hf/__init__.py` would shadow the distribution"
    assert not (MEMBER_DIR / "__init__.py").exists()
    assert not (MEMBER_DIR / MEMBER_MODULE).exists(), "`evoxc_hf/evoxc_hf/` exists: the module is src-layout only"
    assert [p for p in _package_dirs() if p.name == MEMBER_MODULE] == [
        Path(MEMBER_DIR.name, "src", MEMBER_MODULE)
    ]


def test_member_license_is_a_symlink_to_the_root_license():
    """The member's `LICENSE` is a tracked relative symlink to the repository license."""
    member_license = MEMBER_DIR / "LICENSE"
    assert member_license.is_symlink(), f"{member_license} is not a symlink to the root license"
    assert os.readlink(member_license) == "../LICENSE", os.readlink(member_license)
    assert member_license.resolve() == LICENSE.resolve(), member_license.resolve()
    # The text is identical even if the link is ever replaced by a regular file copy.
    assert member_license.read_text() == LICENSE.read_text()


def test_member_readme_is_the_declared_long_description():
    """`readme = { file = "README.md" }` points at a real, non-trivial Markdown file."""
    readme = _member_pyproject()["project"]["readme"]
    assert readme["file"] == "README.md", readme
    path = MEMBER_DIR / readme["file"]
    assert path.is_file(), f"{path} (the member's PyPI long description) is missing"
    text = path.read_text()
    assert len(text) > 1000, f"{path} is only {len(text)} bytes — not a real long description"
    assert text.lstrip().startswith("#"), text[:80]


# ---- 4. source purity: `evoxc` never reaches the member or its oracles -------


def test_import_evoxc_does_not_import_evoxc_hf_or_the_hf_oracles():
    """A fresh interpreter importing `evoxc` pulls neither the member nor torch/transformers."""
    probe = (
        "import sys, evoxc;"
        "evoxc.build, evoxc.evaluate, evoxc.lower, evoxc.trace, evoxc.tree_map;"
        f"heavy = [m for m in ('{MEMBER_MODULE}', 'torch', 'transformers') if m in sys.modules];"
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


def test_evoxc_sources_never_reference_evoxc_hf():
    """No `*.py` under `evoxc/` mentions the member: the dependency direction is one-way."""
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted((REPO_ROOT / ROOT_MODULE).rglob("*.py"))
        if MEMBER_MODULE in path.read_text(encoding="utf-8", errors="replace")
    ]
    assert not offenders, f"`{ROOT_MODULE}` sources mention `{MEMBER_MODULE}`: {offenders[:5]}"


# ---- 5. built artifacts: ONE workspace build, inspected per distribution -----


def _importable(module: str) -> bool:
    """True when `module` is a real importable module here (not a stray namespace dir)."""
    spec = importlib.util.find_spec(module)
    return spec is not None and spec.origin is not None


def _build_plan(src: Path, out_dir: Path, project: str | None) -> list[tuple[Path, list[str]]] | None:
    """`(cwd, argv)` steps building the whole workspace (`project=None`) or one project.

    `uv build` runs ISOLATED deliberately: uv bundles the `uv_build` backend, so the
    build needs no network and nothing installed here (`--no-build-isolation` would
    defeat exactly that). The `python -m build --no-isolation` fallback needs
    `uv_build` importable, and it can only build one project at a time.
    """
    uv = shutil.which("uv")
    if uv is not None:
        scope = ["--all-packages"] if project is None else ["--package", project]
        return [(src, [uv, "build", *scope, "--out-dir", str(out_dir)])]
    if not (_importable("uv_build") and _importable("build")):
        return None
    member_dir = src / MEMBER_DIR.name
    directories = {None: [src, member_dir], ROOT_DIST: [src], MEMBER_DIST: [member_dir]}.get(project)
    if directories is None:
        return None
    per_project = [sys.executable, "-m", "build", "--sdist", "--wheel", "--no-isolation"]
    return [(directory, per_project + ["--outdir", str(out_dir)]) for directory in directories]


# Copied into a throwaway source tree so the build never writes into the checkout.
_COPY_IGNORE = shutil.ignore_patterns(
    ".git", ".genesis", ".venv", "venv", "node_modules", "build", "dist", "*.egg-info",
    "__pycache__", ".pytest_cache", ".agents",
    "bench_logs",  # large scratch area, never a build input
)

_STALE_EGG_INFO = "evoxc.egg-info"

_NO_BUILD_TOOLCHAIN = (
    "no workspace build toolchain: `uv build --all-packages` is needed (uv bundles the "
    "`uv_build` backend, so the build stays isolated; the fallback needs `uv_build` importable)"
)


def _sanitize_source_tree(src: Path) -> list[str]:
    """Delete stale build artifacts from a copied tree (`rm -rf evoxc.egg-info build`)."""
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


def _run_build(src: Path, out_dir: Path, project: str | None = None) -> tuple[list[Path], str]:
    """Build the workspace (or one `--package`) from `src`; `(artifacts, skip-reason)`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = _build_plan(src, out_dir, project)
    if plan is None:
        return [], _NO_BUILD_TOOLCHAIN
    for cwd, argv in plan:
        proc = subprocess.run(argv, cwd=str(cwd), capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "no output")[-500:]
            return [], f"build unavailable (`{' '.join(argv[1:])}`): {detail}"
    suffixes = (".whl", ".tar.gz")
    artifacts = sorted(p for p in out_dir.iterdir() if p.name.endswith(suffixes))
    if not artifacts:
        return [], f"the build produced no wheel/sdist in {out_dir}"
    return artifacts, ""


def _artifact_kind(path: Path) -> str:
    if path.suffix == ".whl":
        return "wheel"
    assert path.name.endswith(".tar.gz"), f"{path.name} is neither a wheel nor an sdist"
    return "sdist"


def _dist_stem(path: Path) -> str:
    """`evoxc_hf-0.1.0-py3-none-any.whl` -> `evoxc_hf` (the distribution's file stem)."""
    return path.name.split("-", 1)[0]


def _index_artifacts(paths: Iterable[Path]) -> dict[tuple[str, str], Path]:
    """`{(distribution stem, "wheel"|"sdist"): path}` — one artifact per pair."""
    indexed: dict[tuple[str, str], Path] = {}
    for path in paths:
        key = (_dist_stem(path), _artifact_kind(path))
        assert key not in indexed, f"two {key[1]}s built for {key[0]}: {indexed[key].name}, {path.name}"
        indexed[key] = path
    return indexed


def _artifact(artifacts: dict[tuple[str, str], Path], dist: str, kind: str) -> Path:
    key = (dist, kind)
    assert key in artifacts, f"the workspace build produced no {dist} {kind}: {sorted(artifacts)}"
    return artifacts[key]


@pytest.fixture(scope="session")
def workspace_source(tmp_path_factory) -> Path:
    """ONE throwaway copy of the checkout, shared by every build leg."""
    return _copy_checkout(tmp_path_factory.mktemp("workspace-src") / "src")


@pytest.fixture(scope="session")
def built_artifacts(workspace_source, tmp_path_factory) -> dict[tuple[str, str], Path]:
    """ONE real `uv build --all-packages` for the whole session: four artifacts."""
    artifacts, reason = _run_build(workspace_source, tmp_path_factory.mktemp("workspace-dist"))
    if not artifacts:
        pytest.skip(reason)
    return _index_artifacts(artifacts)


def _wheel_names(archive: Path) -> list[str]:
    with zipfile.ZipFile(archive) as zf:
        return zf.namelist()


def _wheel_read(archive: Path, name: str) -> str:
    with zipfile.ZipFile(archive) as zf:
        return zf.read(name).decode()


def _dist_info_dir(archive: Path) -> str:
    """The single `<name>-<version>.dist-info` directory of a wheel."""
    roots = sorted({n.split("/", 1)[0] for n in _wheel_names(archive) if ".dist-info/" in n})
    assert len(roots) == 1, f"{archive.name} must carry exactly one .dist-info: {roots}"
    return roots[0]


def _metadata_headers(text: str) -> dict[str, list[str]]:
    """The RFC822 header block of a METADATA file (the body is the README): key -> values."""
    headers: dict[str, list[str]] = {}
    for line in text.splitlines():
        if not line.strip():
            break
        key, sep, value = line.partition(":")
        if sep:
            headers.setdefault(key.strip().lower(), []).append(value.strip())
    return headers


def _header(headers: dict[str, list[str]], key: str) -> str | None:
    values = headers.get(key) or []
    assert len(values) <= 1, f"METADATA field {key!r} is repeated: {values}"
    return values[0] if values else None


def _wheel_metadata(archive: Path) -> dict[str, list[str]]:
    return _metadata_headers(_wheel_read(archive, f"{_dist_info_dir(archive)}/METADATA"))


def _relative_entries(archive: Path) -> list[str]:
    """Member paths of an sdist tarball, relative to its single `<name>-<version>/` root."""
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    root = names[0].split("/", 1)[0] if names else ""
    entries = [n.split("/", 1)[1] if n.startswith(root + "/") else n for n in names]
    return [entry for entry in entries if entry]  # drop the archive's own root entry


def _sdist_root(archive: Path) -> str:
    with tarfile.open(archive) as tar:
        roots = sorted({name.split("/", 1)[0].rstrip("/") for name in tar.getnames()})
    assert len(roots) == 1, f"{archive.name} must have a single top-level root, got {roots}"
    return roots[0]


def _artifact_entries(archive: Path) -> list[str]:
    """Member paths of a wheel (zip) or sdist (tar), normalized the same way."""
    return _wheel_names(archive) if archive.suffix == ".whl" else _relative_entries(archive)


def _entries_under(entries: Iterable[str], module: str) -> list[str]:
    """Paths whose first component is `module` (a package dir, a file, or a bare dir)."""
    return [e for e in entries if e == module or e.startswith(module + "/")]


_STALE_HINT = (
    "Cause and fix: the `[tool.uv.build-backend] module-name` decides what ships, and each "
    "distribution ships exactly ONE module. `evoxc_hf` belongs to the member distribution "
    "(`evoxc_hf/src/evoxc_hf/`), never to the root `evoxc` project. Rebuild after fixing the "
    "backend config; a stale `evoxc.egg-info/` cannot cause this under uv_build."
)


def test_workspace_build_produces_exactly_the_four_artifacts(built_artifacts):
    """ONE `uv build --all-packages` = a wheel + an sdist for EACH distribution (never `etl`)."""
    root_version = _project_version(PYPROJECT)
    member_version = _project_version(MEMBER_PYPROJECT)
    expected = {
        f"{ROOT_DIST}-{root_version}-py3-none-any.whl",
        f"{ROOT_DIST}-{root_version}.tar.gz",
        f"{MEMBER_MODULE}-{member_version}-py3-none-any.whl",
        f"{MEMBER_MODULE}-{member_version}.tar.gz",
    }
    names = sorted(path.name for path in built_artifacts.values())
    assert names == sorted(expected), (
        f"the workspace build produced {names}, expected exactly {sorted(expected)} — both "
        "projects build their own sdist + pure-python (`py3-none-any`) wheel"
    )
    for name in names:  # the brand pin: named for the two distributions, never `etl`
        assert name.startswith((f"{ROOT_DIST}-", f"{MEMBER_MODULE}-")), name
    assert len([n for n in names if n.startswith(f"{ROOT_DIST}-")]) == 2, names
    assert len([n for n in names if n.startswith(f"{MEMBER_MODULE}-")]) == 2, names


def test_each_wheel_ships_only_its_own_top_level_modules(built_artifacts):
    """Each wheel's root set is exactly {its own module, its own `.dist-info`} — no sibling."""
    versions = {ROOT_MODULE: _project_version(PYPROJECT), MEMBER_MODULE: _project_version(MEMBER_PYPROJECT)}
    for module in (ROOT_MODULE, MEMBER_MODULE):
        wheel = _artifact(built_artifacts, module, "wheel")
        roots = sorted({entry.split("/", 1)[0] for entry in _wheel_names(wheel)})
        modules = [root for root in roots if not root.endswith(".dist-info")]
        assert modules == [module], (
            f"{wheel.name} root set is {roots}: uv_build ships exactly the `{module}` module\n{_STALE_HINT}"
        )
        assert [root for root in roots if root.endswith(".dist-info")] == [
            f"{module}-{versions[module]}.dist-info"
        ], f"{wheel.name} .dist-info does not carry its own project version {versions[module]!r}"
        sibling = MEMBER_MODULE if module == ROOT_MODULE else ROOT_MODULE
        assert sibling not in roots, f"{wheel.name} ships the SIBLING module `{sibling}`: {roots}"
        for forbidden in ("etl", "bench_logs", "assets", "conftest.py"):
            assert forbidden not in roots, (wheel.name, forbidden, roots)

        # The RECORD agrees: only this module (and its .dist-info) may be shipped.
        record = next((n for n in _wheel_names(wheel) if n.endswith(".dist-info/RECORD")), None)
        assert record is not None, f"{wheel.name} has no RECORD"
        shipped = [line.split(",", 1)[0] for line in _wheel_read(wheel, record).splitlines()]
        tops = sorted({entry.split("/", 1)[0] for entry in shipped if entry.split(",", 1)[0].strip()})
        assert tops and all(top == module or top.endswith(".dist-info") for top in tops), (
            f"{wheel.name} RECORD lists {tops}: only `{module}` (and its .dist-info) may ship"
        )


def test_wheel_top_level_txt_when_present_declares_one_module(built_artifacts):
    """`top_level.txt`, WHEN the backend writes one, declares exactly that one module."""
    for module in (ROOT_MODULE, MEMBER_MODULE):
        wheel = _artifact(built_artifacts, module, "wheel")
        names = _wheel_names(wheel)
        entry = next((n for n in names if n.endswith("/top_level.txt")), None)
        if entry is None:  # uv_build writes no top_level.txt — conditional on purpose
            modules = sorted({n.split("/", 1)[0] for n in names if not n.startswith(f"{module}-")})
            assert modules == [module], wheel.name  # ... the root set still pins one module
            continue
        declared = [line.strip() for line in _wheel_read(wheel, entry).splitlines() if line.strip()]
        assert declared == [module], f"{wheel.name} top_level.txt declares {declared!r}, not ['{module}']"


def test_evoxc_wheel_metadata_pins_the_six_extras(built_artifacts):
    """The `evoxc` wheel keeps its six extras, numpy-only deps and the EvoX Compiler brand."""
    wheel = _artifact(built_artifacts, ROOT_DIST, "wheel")
    metadata = _wheel_metadata(wheel)
    assert _header(metadata, "name") == ROOT_DIST, _header(metadata, "name")
    summary = _header(metadata, "summary") or ""
    assert summary.startswith("EvoX Compiler"), f"the `evoxc` wheel's Summary is {summary!r}"
    extras = set(metadata.get("provides-extra", []))
    assert extras == set(ROOT_EXTRAS), (sorted(extras), wheel.name)
    assert "hf" not in extras, "the `hf` oracle extra moved to the `evoxc-hf` distribution"

    requires = metadata.get("requires-dist", [])
    assert any(_requirement_name(req) == "numpy" for req in requires), requires
    for oracle in ("torch", "transformers", MEMBER_MODULE, MEMBER_DIST):
        for requirement in requires:
            if _requirement_name(requirement) == oracle:
                assert ";" in requirement, (
                    f"`{requirement}` is an UNCONDITIONAL runtime dependency of `evoxc`: the "
                    "oracles and the member distribution must stay behind an extra"
                )
    assert not _entries_under(_wheel_names(wheel), MEMBER_MODULE), f"{wheel.name} ships `{MEMBER_MODULE}`"


def test_evoxc_hf_wheel_metadata_pins_the_evoxc_requirement_and_hf_extra(built_artifacts):
    """The member wheel requires `evoxc` + `numpy`, and gates torch/transformers behind `hf`."""
    wheel = _artifact(built_artifacts, MEMBER_MODULE, "wheel")
    metadata = _wheel_metadata(wheel)
    assert _header(metadata, "name") == MEMBER_DIST, _header(metadata, "name")
    assert _header(metadata, "summary"), "the member wheel ships no Summary"

    requires = metadata.get("requires-dist", [])
    assert {_requirement_name(req) for req in requires} >= {ROOT_DIST, "numpy"}, requires
    evoxc_requirement = next(req for req in requires if _requirement_name(req) == ROOT_DIST)
    assert ";" not in evoxc_requirement, (
        f"`{evoxc_requirement}` is gated on a marker: `evoxc` is a hard dependency of the member"
    )
    assert {"hf", "dev"} <= set(metadata.get("provides-extra", [])), metadata.get("provides-extra")
    hf_requires = {_requirement_name(req) for req in requires if "hf" in _extras_of(req)}
    assert {"torch", "transformers"} <= hf_requires, sorted(hf_requires)
    assert not _entries_under(_wheel_names(wheel), ROOT_MODULE), (
        f"{wheel.name} ships the sibling `{ROOT_MODULE}` package: the member's sources are "
        "`src/evoxc_hf/` only"
    )


def test_each_wheel_ships_its_licenses_directory(built_artifacts):
    """PEP 639 payload: each wheel carries `<dist-info>/licenses/LICENSE`."""
    for module in (ROOT_MODULE, MEMBER_MODULE):
        wheel = _artifact(built_artifacts, module, "wheel")
        names = _wheel_names(wheel)
        dist_info = _dist_info_dir(wheel)
        assert [n for n in names if n.startswith(f"{dist_info}/licenses/") and n.endswith("LICENSE")], (
            f"{wheel.name} ships no `{dist_info}/licenses/LICENSE` "
            f"(entries with 'licen': {[n for n in names if 'licen' in n.lower()][:5]})"
        )


def test_evoxc_sdist_ships_the_module_and_tests_only(built_artifacts):
    """The `evoxc` sdist has ONE `evoxc-<version>` root: the module, tests and sdist extras."""
    sdist = _artifact(built_artifacts, ROOT_DIST, "sdist")
    version = _project_version(PYPROJECT)
    assert sdist.name == f"{ROOT_DIST}-{version}.tar.gz", sdist.name
    assert _sdist_root(sdist) == f"{ROOT_DIST}-{version}"

    entries = _relative_entries(sdist)
    top = {entry.split("/", 1)[0] for entry in entries}
    assert {ROOT_MODULE, "tests"} <= top, f"the evoxc sdist ships {sorted(top)} (expected evoxc + tests)"
    assert {"assets", "conftest.py"} <= top, sorted(top)
    assert _entries_under(entries, ROOT_MODULE), "the evoxc sdist ships no evoxc sources"
    assert "pyproject.toml" in entries, sorted(top)  # anti-vacuity: a real sdist root
    shipped = _entries_under(entries, MEMBER_MODULE)
    assert not shipped, f"the evoxc sdist ships member entries: {shipped[:5]}\n{_STALE_HINT}"


def test_evoxc_hf_sdist_ships_only_its_src_layout(built_artifacts):
    """The member sdist has ONE `evoxc_hf-<version>` root: `src/evoxc_hf/`, never `evoxc`."""
    sdist = _artifact(built_artifacts, MEMBER_MODULE, "sdist")
    version = _project_version(MEMBER_PYPROJECT)
    assert sdist.name == f"{MEMBER_MODULE}-{version}.tar.gz", sdist.name
    assert _sdist_root(sdist) == f"{MEMBER_MODULE}-{version}"

    entries = _relative_entries(sdist)
    top = {entry.split("/", 1)[0] for entry in entries}
    assert "src" in top, f"the member sdist ships {sorted(top)} (expected an sdist of `src/`)"
    assert f"src/{MEMBER_MODULE}/__init__.py" in entries, sorted(top)
    assert "pyproject.toml" in entries, sorted(top)

    # Neither a top-level `evoxc/` nor `src/evoxc/` may ride along.
    assert not _entries_under(entries, ROOT_MODULE), sorted(top)
    nested = [entry for entry in entries if entry.startswith(f"src/{ROOT_MODULE}/")]
    assert not nested, f"the member sdist ships the sibling's sources: {nested[:5]}"


def test_package_scoped_build_yields_only_the_member_artifacts(workspace_source, tmp_path):
    """`uv build --package evoxc-hf` builds the member only (skips when the flag is unavailable)."""
    artifacts, reason = _run_build(workspace_source, tmp_path / "member-dist", project=MEMBER_DIST)
    if not artifacts:
        pytest.skip(reason)
    version = _project_version(MEMBER_PYPROJECT)
    names = sorted(path.name for path in artifacts)
    assert names == sorted(
        {f"{MEMBER_MODULE}-{version}-py3-none-any.whl", f"{MEMBER_MODULE}-{version}.tar.gz"}
    ), f"`--package {MEMBER_DIST}` produced {names}: the scoped build must not touch `evoxc`"


def test_no_archive_ships_a_top_level_etl_package(built_artifacts):
    """No wheel/sdist entry may sit under a top-level `etl` package (the pre-rename name)."""
    for kind in ("wheel", "sdist"):
        for module in (ROOT_MODULE, MEMBER_MODULE):
            archive = _artifact(built_artifacts, module, kind)
            entries = _artifact_entries(archive)
            assert entries, f"{archive.name} is empty"
            etl = [entry for entry in entries if entry.split("/", 1)[0].rstrip("/") == "etl"]
            assert not etl, (
                f"{archive.name} ships a top-level `etl` package ({etl[:5]}): the projects are "
                "`evoxc` (EvoX Compiler) and `evoxc-hf`"
            )


# ---- 6. the stale-manifest trap stays structurally impossible ----------------

# The exact stale artifact that used to poison a setuptools sdist: a leftover
# `SOURCES.txt` from an earlier build listing the member's modules (in TODAY'S
# src layout). One entry points at a file that never existed, pinning the old
# manifest pruning.
_STALE_SOURCES = (
    "evoxc_hf/src/evoxc_hf/__init__.py\n"
    "evoxc_hf/src/evoxc_hf/config.py\n"
    "evoxc_hf/src/evoxc_hf/stale_module_that_no_longer_exists.py\n"
    "evoxc/__init__.py\n"
    "pyproject.toml\n"
)


def test_stale_egg_info_cannot_leak_evoxc_hf(tmp_path):
    """A planted `evoxc.egg-info/SOURCES.txt` cannot pull the member into `evoxc` artifacts.

    MEASURED under setuptools (68.1.2 and 84.0.0): an sdist built from a tree
    carrying a stale `evoxc.egg-info/SOURCES.txt` DID ship the listed `evoxc_hf/`
    paths that still existed on disk. `uv_build` reads no such manifest: the wheel
    is the module dir, the sdist is the module dir + license/readme + root-anchored
    `source-include`. So this test plants the exact historical artifact into a real
    checkout copy (whose member sources do exist, the trap's precondition) and
    asserts both `evoxc` artifacts still ship zero `evoxc_hf` entries.
    """
    src = _copy_checkout(tmp_path / "src")
    # The precondition a stale manifest needed to leak anything: real files.
    assert (src / MEMBER_DIR.name / "src" / MEMBER_MODULE).is_dir(), "copied tree lacks the member sources"

    egg_info = src / _STALE_EGG_INFO
    egg_info.mkdir(parents=True, exist_ok=True)
    (egg_info / "PKG-INFO").write_text("Metadata-Version: 2.1\nName: evoxc\nVersion: 0.0.0\n")
    (egg_info / "SOURCES.txt").write_text(_STALE_SOURCES)

    artifacts, reason = _run_build(src, tmp_path / "stale-dist", project=ROOT_DIST)
    if not artifacts:
        pytest.skip(reason)
    indexed = _index_artifacts(artifacts)

    for kind in ("wheel", "sdist"):
        archive = _artifact(indexed, ROOT_DIST, kind)
        entries = _artifact_entries(archive)
        shipped = _entries_under(entries, MEMBER_MODULE)
        assert not shipped, (
            f"the {kind} built from a tree with a planted {_STALE_EGG_INFO}/SOURCES.txt ships "
            f"{shipped[:5]}: uv_build must never read a stale manifest\n{_STALE_HINT}"
        )
        # Anti-vacuity: the build succeeded and shipped the real package.
        assert _entries_under(entries, ROOT_MODULE), f"the {kind} shipped no evoxc sources"
        if kind == "sdist":  # only the sdist carries the project file
            assert "pyproject.toml" in entries, "the sdist shipped no pyproject.toml"

    # The artifact is removable — proving the plant was a real, separate directory.
    assert _STALE_EGG_INFO in _sanitize_source_tree(src), sorted(p.name for p in src.iterdir())


# ---- 7. release hygiene: single-sourced versions, PEP 639 license, CHANGELOG ----


@pytest.mark.parametrize(
    ("dist_name", "pyproject", "init_path"),
    [(ROOT_DIST, PYPROJECT, ROOT_INIT), (MEMBER_DIST, MEMBER_PYPROJECT, MEMBER_INIT)],
    ids=[ROOT_DIST, MEMBER_DIST],
)
def test_versions_are_single_sourced_from_distribution_metadata(dist_name, pyproject, init_path):
    """Each project's pyproject is the ONE version source; its package reads it back."""
    source = init_path.read_text()
    assert "importlib.metadata" in source, (
        f"{init_path} must read __version__ from its installed distribution metadata"
    )
    assert not re.search(r'^__version__\s*=\s*["\']', source, re.MULTILINE), (
        f"{init_path} hardcodes a __version__ literal — its pyproject must stay the single source"
    )
    literal = _project_version(pyproject)
    assert f'"{literal}"' not in source, (
        f"the version literal {literal!r} is duplicated in {init_path} — it would drift"
    )
    # Each package reads ITS OWN distribution, never the sibling's.
    assert f'_distribution_version("{dist_name}")' in source, (
        f"{init_path} does not read the `{dist_name}` distribution's version"
    )
    sibling = MEMBER_DIST if dist_name == ROOT_DIST else ROOT_DIST
    assert f'_distribution_version("{sibling}")' not in source, (
        f"{init_path} reads the SIBLING distribution `{sibling}` for its version"
    )


@pytest.mark.parametrize(
    ("dist_name", "pyproject", "module"),
    [(ROOT_DIST, PYPROJECT, ROOT_MODULE), (MEMBER_DIST, MEMBER_PYPROJECT, MEMBER_MODULE)],
    ids=[ROOT_DIST, MEMBER_DIST],
)
def test_installed_distribution_metadata_matches_its_own_module(dist_name, pyproject, module):
    """An installed distribution reports its own version and claims only its own module."""
    try:
        dist = importlib.metadata.distribution(dist_name)
    except importlib.metadata.PackageNotFoundError:
        pytest.skip(
            f"the `{dist_name}` distribution is not installed here (a plain `uv sync` installs the "
            "root only — use `uv sync --all-packages`)"
        )

    expected = _project_version(pyproject)
    assert dist.version == expected, (
        f"installed `{dist_name}` is {dist.version!r} but its pyproject declares {expected!r} — "
        "reinstall after changing the version"
    )
    installed_module = importlib.import_module(module)
    assert installed_module.__version__ == expected, (
        f"{module}.__version__ is {installed_module.__version__!r}, pyproject says {expected!r}"
    )
    assert installed_module.__version__ != "0.0.0+unknown", (
        "the bare-checkout fallback must never be what an installed distribution reports"
    )

    top_level = dist.read_text("top_level.txt")
    if top_level is not None:  # uv_build writes no top_level.txt — conditional on purpose
        names = [line.strip() for line in top_level.splitlines() if line.strip()]
        assert names == [module], f"installed `{dist_name}` declares top-level packages {names!r}"

    record = dist.read_text("RECORD")
    if record is not None:  # wheel installs only; legacy editable installs have none
        sibling = MEMBER_MODULE if module == ROOT_MODULE else ROOT_MODULE
        shipped = [line.split(",", 1)[0] for line in record.splitlines() if line.startswith(sibling + "/")]
        assert not shipped, f"installed `{dist_name}` ships the sibling's files: {shipped[:5]}"


def test_license_metadata_is_declared_once_as_spdx_in_both_projects():
    """PEP 639 in BOTH projects: an SPDX expression + the license file, no legacy form."""
    for path in (PYPROJECT, MEMBER_PYPROJECT):
        project = _load(path)["project"]
        assert project["license"] == "Apache-2.0", (path.name, project["license"])
        assert project["license-files"] == ["LICENSE"], (path.name, project.get("license-files"))
        assert not [c for c in project.get("classifiers", []) if c.startswith("License ::")], (
            path.name,
            project.get("classifiers"),
        )
    assert LICENSE.is_file(), "LICENSE is missing from the repository root"
    # The canonical Apache 2.0 text is indented on its first line.
    assert LICENSE.read_text().lstrip().startswith("Apache License"), "LICENSE is not the Apache 2.0 text"


def test_wheels_carry_pep639_license_metadata(built_artifacts):
    """uv_build preserves the PEP 639 metadata in BOTH wheels' METADATA."""
    for module in (ROOT_MODULE, MEMBER_MODULE):
        wheel = _artifact(built_artifacts, module, "wheel")
        metadata = _wheel_metadata(wheel)
        assert _header(metadata, "license-expression") == "Apache-2.0", (
            wheel.name,
            _header(metadata, "license-expression"),
        )
        assert "LICENSE" in (_header(metadata, "license-file") or ""), wheel.name
        assert "license" not in metadata, (
            f"{wheel.name} emits the legacy `License:` field alongside License-Expression "
            f"(saw {metadata.get('license')!r})"
        )


def test_changelog_documents_the_current_version():
    """A Keep-a-Changelog file with a release heading for the ROOT project's version."""
    version = _project_version(PYPROJECT)
    assert CHANGELOG.is_file(), "CHANGELOG.md is missing from the repository root"
    text = CHANGELOG.read_text()
    assert text.startswith("# Changelog"), text[:80]
    assert f"## [{version}]" in text, (
        f"CHANGELOG.md has no `## [{version}]` release heading: "
        f"{[line for line in text.splitlines() if line.startswith('## ')]}"
    )
    assert f"[{version}]: " in text, f"CHANGELOG.md has no `[{version}]: <url>` link reference for the release tag"
