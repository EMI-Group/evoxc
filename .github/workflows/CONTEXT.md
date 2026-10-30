# .github/workflows — CI and release automation

## Intent

Two GitHub Actions workflows that gate and publish the two distributions (`evoxc` from the workspace root, `evoxc-hf` from the `evoxc_hf` member) built from this one repository and one `uv.lock`.
No application code lives here — these files only orchestrate install, test, build, metadata and boundary checks.

## API Surface

- `ci.yml` — on push to `main` and on pull requests, three jobs: `test` (pytest matrix on Python 3.11–3.14 with only the `dev` extra), `compilers` (same suite on Python 3.12 with the `compilers` extra, the only job exercising the IREE/TVM adapter legs end-to-end) and `packaging` (build both distributions, check metadata, assert the four artifacts, guard the distribution boundary).
- `packaging` step order: `uv build --all-packages --out-dir dist` → `uvx twine check dist/*` → the four-artifact assertion (`evoxc` + `evoxc_hf`, wheel + sdist, at the versions read from the two `pyproject.toml` files) → the two-way archive boundary guard (each wheel/sdist ships only its own top-level package, each sdist has a single versioned root, the `evoxc_hf` sdist ships `LICENSE`).
- `release.yml` — on `release: published`, tag-driven and per package: `v<version>` routes to `evoxc`, `evoxc-hf-v<version>` routes to `evoxc-hf`; any other tag fails in `build`.
- Job chain in `release.yml`: `build` → exactly one of `publish-evoxc` / `publish-evoxc-hf` → `attach` (which tolerates the skipped sibling), so a failed build or publish keeps artifacts off the GitHub Release.

## Constraints

- Workspace installs MUST pass `--all-packages` (and `--locked`): a plain `uv sync` installs the root project only and uninstalls the member.
- `uv build` writes its own 1-byte `dist/.gitignore` (content `*`) into `--out-dir`, so artifact counts in the packaging job must match `*.whl`/`*.tar.gz` only — a bare `find dist -maxdepth 1 -type f` counts a fifth file on a clean build.
- The packaging step lists `dist` with `ls -la` so the auto-generated `.gitignore` stays visible in CI logs.
- Shell globs (`twine check dist/*`, `release.yml`'s `dist/*` upload/artifact paths) skip dotfiles, so no glob needs the same restriction.
- Built archive names are normalized: the `evoxc-hf` project becomes `evoxc_hf-<version>-py3-none-any.whl` / `evoxc_hf-<version>.tar.gz`.
- Versions are never derived or bumped by a workflow: each project's `pyproject.toml` is the single source of truth and `release.yml` fails when `<tag prefix><version>` differs from the tag.
- Publishing uses PyPI trusted publishing (OIDC); no token or password is stored in the repository or in secrets, and the `pypi` environment plus two pending trusted publishers (`evoxc`, `evoxc-hf`) must exist on the PyPI side.
- Skipped tests in CI are host/dependency guards by design (no CUDA device, no compiler / PJRT plugin / Vulkan ICD, no torch/transformers) — they are never failures, and the `evoxc-hf` oracle legs run by hand rather than in CI.

## Routing Table

| Path | Area |
|---|---|
| `./ci.yml` | Push/PR gate: test matrix, compilers job, and the packaging job (build + metadata + four-artifact assertion + distribution-boundary guard). |
| `./release.yml` | Tag-driven per-package release: route by tag, build, verify version against the tag, publish via OIDC trusted publishing, attach artifacts. |
| `../../RELEASING.md` | Human release procedure (sibling — read-only, escalate writes to parent). |
