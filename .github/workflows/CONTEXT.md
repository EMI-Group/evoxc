# .github/workflows — CI and release automation

## Intent

These workflows test and package the two distributions from one workspace, release `evoxc` from a verified tag push, and retain the independent, manually initiated GitHub Release for `evoxc-hf`.

## API Surface

- `ci.yml` is the push-to-`main`/PR gate and a reusable `workflow_call` with required `checkout_sha`; callers pin their checked-out commit.
- CI runs the Python 3.11–3.14 test matrix, the Python 3.12 compiler leg, the two-distribution packaging/boundary check, and `cargo test --locked --workspace` plus `cargo clippy --locked --workspace --all-targets --no-deps -- -D warnings` in `native/`.
- The `packaging` job runs the release helper regression suite once on Python 3.12 in both normal and reusable CI with `uv run --no-project --python 3.12 python -m unittest discover -s .github/workflows -p 'test_release_checks.py'`.
- The same `packaging` job runs the version-bump helper suite on Python 3.12 with `uv run --no-project --python 3.12 python -m unittest discover -s scripts -p 'test_*.py'`, which discovers `scripts/test_bump_version.py` from the repository root.
- `release.yml` triggers on a new `v*` tag push, no-input `workflow_dispatch`, and a published GitHub Release tagged `evoxc-hf-v*`.
- On an `evoxc` tag push, `verify` matches the peeled tag commit and tag object to the event and checkout, requires membership in `origin/main` history, and requires the exact tag `v<version>` from the root `pyproject.toml`.
- `verify` → `ci` → `wheels` is shared by the tag and the `main`-only rehearsal (`gh workflow run release.yml --ref main`); rehearsals have no release-write or OIDC jobs.
- `wheels.yml` is callable with `checkout_sha` and `version` and returns `artifact_id` for the single immutable `evoxc-dist` upload containing seven wheels and one sdist after platform smoke tests, matrix and archive verification, and strict Twine validation.
- Linux wheels run on `ubuntu-24.04` (x86-64) and `ubuntu-24.04-arm` (AArch64); on each architecture the manylinux2014/glibc 2.17 and musllinux 1.2 wheels are installed and tested inside their matching PyPA containers.
- `macos-15-intel` builds x86-64 with macOS 10.12 deployment target, `macos-15` builds Arm64 with macOS 11.0 deployment target, and `windows-2025` builds x86-64 MSVC; the smoke tests run natively on those hosts, not on old macOS runtime versions.
- The Maturin sdist runs a source-build/install smoke test on `ubuntu-24.04` with Rust available; every wheel and sdist smoke test uses CPython 3.11, a fresh venv outside the checkout, and `pip --isolated install --no-cache-dir` on the exact archive (`--only-binary=:all:` for wheel legs).
- The smoke test verifies installed metadata and package versions and extension provenance, then traces a small graph, differentiates it through the Rust transform, and executes it using the native backend.
- In `EMI-Group/evoxc`, successful gates enable `prepare` (stage and verify a GitHub draft), `publish-evoxc` (publish to PyPI), then `finalize` (make that exact checked draft public).
- The separate HF path validates `evoxc-hf-v<version>` against the member `pyproject.toml` at the release event commit and runs `build-evoxc-hf` → `publish-evoxc-hf` → `attach-evoxc-hf` using only the separately built `evoxc-hf-dist` wheel and sdist.

## Constraints

- CI uses `uv sync --locked --all-packages` and runs pytest with the same lock and workspace flags so the member remains installed and dependencies do not re-resolve.
- CI's `uv build` writes `dist/.gitignore`, which is excluded from archive counts and publication; each project must ship only its own module.
- The version-bump helper (`scripts/bump_version.py`) and its tests are stdlib-only and run in the `packaging` job, so that gate needs no project environment.
- The workflows never create or move a tag, change a version, or publish from manual-dispatch input; the two project `pyproject.toml` files are authoritative for their respective versions.
- GitHub Release writes are job-scoped, and only `publish-evoxc` and `publish-evoxc-hf` have `id-token: write` and target the `pypi` environment; both PyPI trusted publishers bind to `release.yml` in `EMI-Group/evoxc`.
- Per-tag concurrency prevents release runs from overlapping; maintainers should protect release tags against mutation and configure `pypi` deployment rules for the approved tags only.
- Publishers download the tested artifact by numeric ID, recheck the live tag, and never use skip-existing or overwrite semantics; the evoxc publisher rechecks the draft and SHA-256 assets after environment approval before upload.
- Recovery reruns failed jobs in the original run (`gh run rerun <run-id> --failed`) to retain the original artifacts, not a fresh build; an exact PyPI match skips upload, and an exact already-public GitHub release is accepted by finalization without mutation.
- A partial PyPI release blocks automated upload; after comparing hashes, a maintainer can restore only missing files from the original artifact linked in the run summary, then rerun failed jobs.
- A hash mismatch, conflicting draft, or expired original artifact requires investigation, not overwrites or rebuilding a partially published version; artifacts are retained for seven days.

## Routing Table

| Path | Area |
|---|---|
| `./ci.yml` | Normal and reusable CI: Python, compiler, packaging, and Rust gates. |
| `./wheels.yml` | Reusable gate for the supported evoxc wheel matrix, source archive and exact installs. |
| `./smoke_evoxc.py` | Fresh-venv install and compiled-extension graph smoke for one exact archive. |
| `./release.yml` | Tag preflight, rehearsal, artifact paths, GitHub draft/publication, and PyPI OIDC jobs. |
| `./release_checks.py` / `./test_release_checks.py` | Source, artifact and remote-state validation; run `python -m unittest discover -s .github/workflows -p 'test_release_checks.py'` from the repository root. |
| `../../RELEASING.md` | Human release procedure (sibling — read-only, escalate writes to parent). |
