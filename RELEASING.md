# Releasing evoxc

The release is **tag-driven and fully automated by GitHub Actions**:

```
bump [project].version  →  commit  →  tag vX.Y.Z  →  push tag  →  publish a GitHub Release
                                                                        │
                                                                        ▼
                          .github/workflows/release.yml: build → verify tag == version
                                                              → PyPI (OIDC) → attach artifacts
```

There is no version bump and no token anywhere in the automation — `pyproject.toml`
is the single source of truth for the version, and `release.yml` **fails** when the
release tag does not match it.

Assumed repository identity (the release owner must confirm these before the first
release): **owner `EMI-Group`**, **repository `evoxc`**, workflow **`release.yml`**,
GitHub/PyPI environment **`pypi`**.

---

## 1. One-time setup (do this once, before the first release)

1. **Create and push the GitHub repository** — `EMI-Group/evoxc` — and push `main`
   (with `.github/workflows/ci.yml` and `.github/workflows/release.yml`).
2. **Confirm the repository name everywhere it is hard-coded** — `[project.urls]`
   in `pyproject.toml`, the pending-publisher fields in step 3 below, and the
   absolute `https://raw.githubusercontent.com/EMI-Group/evoxc/main/assets/…`
   image URLs in `README.md` (needed for the diagrams to render on the PyPI
   project page) must all name the same real repository. They currently assume
   **`EMI-Group/evoxc`**; edit them if the repository lands somewhere else.
3. **Create the PyPI pending trusted publisher** at
   <https://pypi.org/manage/account/publishing/> — the project `evoxc` does not exist
   on PyPI yet, so this is a *pending* publisher, and the fields must match the
   workflow exactly:

   | Field | Value |
   | --- | --- |
   | PyPI Project Name | `evoxc` |
   | Owner | `EMI-Group` |
   | Repository name | `evoxc` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

4. **Create the GitHub environment** `pypi` (Settings → Environments). Optionally
   require reviewers — publishing then waits for an approval, which is a good
   default for a first release.
5. *(Optional, for a TestPyPI dry run)* create a **separate pending trusted
   publisher** on <https://test.pypi.org/manage/account/publishing/> (owner `EMI-Group`,
   repo `evoxc`, workflow `release.yml`, environment `testpypi`) and a GitHub
   environment `testpypi`. The TestPyPI job itself is not enabled by default — see
   the commented block in the header of `.github/workflows/release.yml`.

No PyPI API token is created or stored. See "Why no token" below.

---

## 2. Per-release procedure

0. **Local pre-flight** (catches metadata/packaging mistakes in seconds):

   ```bash
   uv sync --extra dev
   uv run pytest -q
   uv build && uvx twine check dist/*
   ```

1. **Bump `[project].version`** in `pyproject.toml` (e.g. `0.1.0` → `0.1.1`).
   The version is static: `uv_build` requires it, and nothing else derives it.
2. **Add a `CHANGELOG.md` entry** for the new version (create the file if the
   project has none yet; newest version at the top).
3. **Commit** both changes on `main`:
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "release: v0.1.1"
   ```
4. **Tag and push the tag**:
   ```bash
   git tag v0.1.1
   git push origin main
   git push origin v0.1.1
   ```
5. **Publish a GitHub Release for that tag** — this is the trigger
   (`release: types: [published]`). Drafts and pre-releases do **not** publish;
   only *publishing* the release starts `release.yml`.
6. **Watch the `Release` workflow**: `build` (builds, asserts `v<version>` == tag,
   `twine check`) → `publish` (PyPI via OIDC; may wait for environment approval) →
   `attach` (uploads `dist/*` to the GitHub Release).

### Afterwards, check

- the PyPI project page shows the new version: <https://pypi.org/project/evoxc/>;
- a clean install works: `pip install evoxc==X.Y.Z` (and `pip install "evoxc[dev]"`);
- the GitHub Release has `evoxc-X.Y.Z-py3-none-any.whl` and `evoxc-X.Y.Z.tar.gz`
  attached (the `attach` job);
- `evoxc_hf` is **not** in the wheel — the `packaging` CI job pins that boundary,
  but a quick eyeball is cheap: `unzip -l dist/evoxc-X.Y.Z-py3-none-any.whl`.

---

## 3. If it goes wrong

- **`Release` fails at "Fail if the release tag does not match [project].version".**
  Nothing was published. The message names both values. Fix the mismatch: bump
  `[project].version` to the tagged version (preferred), or delete the tag/release
  and re-create it on the correct commit. Re-publish the release afterwards.
- **PyPI rejects the upload because the version already exists.** PyPI versions are
  **immutable** — a published file can never be replaced or re-uploaded. Bump
  `[project].version` to a new patch version, commit, tag, and publish a new GitHub
  Release. Do not try to force a re-upload.
- **A bad release needs to be withdrawn.** Yank it from the PyPI UI
  (<https://pypi.org/manage/project/evoxc/releases/> → *Yank*). Yanking keeps the
  version installable only by exact pin (and `pip install evoxc` will no longer
  select it); it cannot be deleted-and-reused. Then ship a fixed patch release.
- **The upload succeeded but `attach` failed.** PyPI is already published; just
  re-run the failed `attach` job from the Actions UI (`gh release upload` uses
  `--clobber`, so re-running is safe and idempotent).
- **Run the local pre-flight** at any point to reproduce build/metadata problems
  without touching PyPI:
  ```bash
  uv build && uvx twine check dist/*
  ```

---

## 4. Why no token

PyPI **trusted publishing** uses OpenID Connect: GitHub mints a short-lived,
workflow-scoped identity token (`id-token: write`) at publish time, and PyPI verifies
it against the pending publisher (owner / repository / workflow / environment)
configured above. Consequences: nothing secret lives in the repository or in
Actions secrets, nothing can leak from a fork PR, and the credential cannot be
replayed outside this exact workflow + environment. The only rotating part is the
environment — which is why protecting the `pypi` environment with required reviewers
is the recommended extra guard.

`.github/workflows/ci.yml` covers the pre-merge gate (tests on Python 3.11–3.14,
a compiler-adapter job, and a packaging/`twine`/distribution-boundary job); release
automation only ever *publishes* what CI already validated.
