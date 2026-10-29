# Releasing `evoxc` and `evoxc-hf`

Both releases are **tag-driven and fully automated by GitHub Actions**:

```
pick the package → bump its [project].version → commit → tag → push tag → publish a GitHub Release
                                                                                 │
                                                                                 ▼
     .github/workflows/release.yml: route the tag to its package → build → verify tag == version
                                    → PyPI (OIDC) → attach artifacts
```

There is no version bump and no token anywhere in the automation — the tagged
project's `pyproject.toml` is the single source of truth for its version, and
`release.yml` **fails** when the tag does not match that project's version.

Assumed repository identity (the release owner must confirm these before the first
release): **owner `EMI-Group`**, **repository `evoxc`**, workflow **`release.yml`**,
GitHub/PyPI environment **`pypi`**.

---

## Two distributions

This repository publishes two independent PyPI distributions from one
`uv` workspace (one repository, one `uv.lock`, one version per project):

| distribution | project (version lives here) | import | release tag | install |
| --- | --- | --- | --- | --- |
| `evoxc` | `pyproject.toml` (workspace root) | `evoxc` | `v<version>` | `pip install evoxc` |
| `evoxc-hf` | `evoxc_hf/pyproject.toml` (workspace member) | `evoxc_hf` | `evoxc-hf-v<version>` | `pip install evoxc-hf` |

- **The tag prefix routes the release.** `v<version>` (e.g. `v0.1.0`) builds and
  publishes `evoxc`; `evoxc-hf-v<version>` (e.g. `evoxc-hf-v0.1.0`) builds and
  publishes `evoxc-hf`. Each release publishes **exactly one** package, so
  releasing both means two tags and two GitHub Releases.
- **Versions are independent.** Bumping one project never changes the other's
  version, and the two release on their own schedule.
- **Release `evoxc` first** when `evoxc-hf`'s minimum requirement on it is new:
  `evoxc_hf/pyproject.toml` depends on `evoxc>=0.1.0`, so a published `evoxc-hf`
  must never require an `evoxc` version PyPI does not have yet.
- **Neither distribution can contain the other.** Both build with `uv_build`,
  which ships exactly one module per distribution — `evoxc` from the repository
  root, `evoxc_hf` from `evoxc_hf/src/evoxc_hf/` — pinned by
  `tests/test_packaging.py`.

---

## 1. One-time setup (do this once, before the first release)

1. **Create and push the GitHub repository** — `EMI-Group/evoxc` — and push `main`
   (with `.github/workflows/ci.yml` and `.github/workflows/release.yml`).
2. **Confirm the repository name everywhere it is hard-coded** — both
   `[project.urls]` tables (`pyproject.toml` and `evoxc_hf/pyproject.toml`), the
   pending-publisher fields in step 3 below, and the absolute
   `https://raw.githubusercontent.com/EMI-Group/evoxc/main/assets/…` image URLs in
   `README.md` (needed for the diagrams to render on the PyPI project page) must
   all name the same real repository. They currently assume **`EMI-Group/evoxc`**;
   edit them if the repository lands somewhere else.
3. **Create TWO PyPI pending trusted publishers** at
   <https://pypi.org/manage/account/publishing/> — neither project exists on PyPI
   yet, so both are *pending* publishers, and the fields must match the workflow
   exactly. They differ only in the PyPI project name:

   | Field | `evoxc` | `evoxc-hf` |
   | --- | --- | --- |
   | PyPI Project Name | `evoxc` | `evoxc-hf` |
   | Owner | `EMI-Group` | `EMI-Group` |
   | Repository name | `evoxc` | `evoxc` |
   | Workflow name | `release.yml` | `release.yml` |
   | Environment name | `pypi` | `pypi` |

4. **Create the GitHub environment** `pypi` (Settings → Environments) — one
   environment serves both projects. Optionally require reviewers — publishing
   then waits for an approval, which is a good default for a first release.
5. *(Optional, for a TestPyPI dry run)* create **two separate pending trusted
   publishers** on <https://test.pypi.org/manage/account/publishing/> (one per
   project: `evoxc`, `evoxc-hf`; owner `EMI-Group`, repo `evoxc`, workflow
   `release.yml`, environment `testpypi`) and a GitHub environment `testpypi`.
   The TestPyPI job itself is not enabled by default — see the commented block in
   the header of `.github/workflows/release.yml`.

No PyPI API token is created or stored. See "Why no token" below.

---

## 2. Per-release procedure

Pick the package you are releasing (`evoxc` → `pyproject.toml` / `v<version>`;
`evoxc-hf` → `evoxc_hf/pyproject.toml` / `evoxc-hf-v<version>`) and follow the
same steps. Release `evoxc` first when the new `evoxc-hf` needs a new `evoxc`.

0. **Local pre-flight** (catches metadata/packaging mistakes in seconds):

   ```bash
   uv sync --locked --all-packages --extra dev
   uv run pytest -q
   uv build --all-packages && uvx twine check dist/*
   ```

1. **Bump `[project].version`** in that package's project file (e.g. `0.1.0` →
   `0.1.1`). The version is static: `uv_build` requires it, and nothing else
   derives it.
2. **Add a `CHANGELOG.md` entry** for the new version (newest version at the top),
   naming which distribution it belongs to.
3. **Commit** both changes on `main`:
   ```bash
   # evoxc
   git add pyproject.toml CHANGELOG.md
   git commit -m "release: evoxc v0.1.1"

   # evoxc-hf
   git add evoxc_hf/pyproject.toml CHANGELOG.md
   git commit -m "release: evoxc-hf v0.1.1"
   ```
4. **Tag and push the tag** — the prefix selects the package:
   ```bash
   # evoxc        → git tag v0.1.1
   # evoxc-hf     → git tag evoxc-hf-v0.1.1
   git tag v0.1.1
   git push origin main
   git push origin v0.1.1
   ```
5. **Publish a GitHub Release for that tag** — this is the trigger
   (`release: types: [published]`). Drafts and pre-releases do **not** publish;
   only *publishing* the release starts `release.yml`. Releasing both packages
   means publishing two releases, one per tag.
6. **Watch the `Release` workflow**: `build` (routes the tag to its project,
   builds it, asserts the tag matches that project's version, `twine check`) →
   `publish` (PyPI via OIDC; may wait for environment approval) → `attach`
   (uploads `dist/*` to the GitHub Release).

### Afterwards, check

- the PyPI project page shows the new version — <https://pypi.org/project/evoxc/>
  and/or <https://pypi.org/project/evoxc-hf/>;
- a clean install works in a fresh venv: `pip install evoxc==X.Y.Z`,
  `pip install evoxc-hf==X.Y.Z` (and `pip install "evoxc[dev]"` /
  `pip install "evoxc-hf[hf]"` where relevant) — `pip install evoxc` must still
  pull numpy only and never `evoxc-hf`;
- the GitHub Release has the right artifacts attached (the `attach` job):
  `evoxc-X.Y.Z-py3-none-any.whl` + `evoxc-X.Y.Z.tar.gz` for `v<version>`,
  `evoxc_hf-X.Y.Z-py3-none-any.whl` + `evoxc_hf-X.Y.Z.tar.gz` for
  `evoxc-hf-v<version>`;
- each wheel contains only its own module — the `packaging` CI job pins that
  boundary, but a quick eyeball is cheap:
  `unzip -l dist/evoxc-X.Y.Z-py3-none-any.whl` shows only `evoxc/…`, and
  `unzip -l dist/evoxc_hf-X.Y.Z-py3-none-any.whl` only `evoxc_hf/…`.

---

## 3. If it goes wrong

- **`Release` fails at "Fail if the release tag does not match the project
  version".** Nothing was published. The message names the routing rule (which
  tag prefix belongs to which package), the tag it saw and the version that
  package's `pyproject.toml` declares, plus the exact tag it expects
  (`v<version>` for `evoxc`, `evoxc-hf-v<version>` for `evoxc-hf`). Fix the
  mismatch: bump that project's `[project].version` to the tagged version
  (preferred), or delete the tag/release and re-create it on the correct commit
  with the correct prefix. Re-publish the release afterwards.
- **PyPI rejects the upload because the version already exists.** PyPI versions are
  **immutable** — a published file can never be replaced or re-uploaded. Bump that
  project's `[project].version` to a new patch version, commit, tag (with that
  package's prefix), and publish a new GitHub Release. Do not try to force a
  re-upload. The two projects version independently: an `evoxc-hf` re-cut does not
  require an `evoxc` release.
- **A bad release needs to be withdrawn.** Yank it from the PyPI UI of the
  affected project (<https://pypi.org/manage/project/evoxc/releases/>,
  <https://pypi.org/manage/project/evoxc-hf/releases/>) → *Yank*. Yanking keeps
  the version installable only by exact pin (and `pip install evoxc` /
  `pip install evoxc-hf` will no longer select it); it cannot be
  deleted-and-reused. Then ship a fixed patch release of that package.
- **The upload succeeded but `attach` failed.** PyPI is already published; just
  re-run the failed `attach` job from the Actions UI (`gh release upload` uses
  `--clobber`, so re-running is safe and idempotent).
- **Run the local pre-flight** at any point to reproduce build/metadata problems
  without touching PyPI:
  ```bash
  uv build --all-packages && uvx twine check dist/*
  ```
  To check one project only: `uv build --package evoxc-hf` (this writes into
  `dist/` as well).

---

## 4. Why no token

PyPI **trusted publishing** uses OpenID Connect: GitHub mints a short-lived,
workflow-scoped identity token (`id-token: write`) at publish time, and PyPI verifies
it against the pending publisher (owner / repository / workflow / environment)
configured above — one pending publisher per project (`evoxc` and `evoxc-hf`), both
keyed to the same repository, workflow and environment. Consequences: nothing secret
lives in the repository or in Actions secrets, nothing can leak from a fork PR, and
the credential cannot be replayed outside this exact workflow + environment. The
only rotating part is the environment — which is why protecting the `pypi`
environment with required reviewers is the recommended extra guard.

`.github/workflows/ci.yml` covers the pre-merge gate (tests on Python 3.11–3.14,
a compiler-adapter job, and a packaging/`twine`/distribution-boundary job); release
automation only ever *publishes* what CI already validated.
