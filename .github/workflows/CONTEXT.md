# .github/workflows — CI and release automation

## Intent

These workflows test and package the two distributions from one workspace: `ci.yml` gates pull requests and pushes to `main`, `release_gate.yml` is the quick end-to-end release gate, and `release.yml` releases `evoxc` from a verified tag push while retaining the independent, manually initiated GitHub Release for `evoxc-hf`.
A release tag never runs the full CI suite: it runs the quick gate and then the authoritative wheel matrix.

## API Surface

- `ci.yml` is the push-to-`main`/PR gate only; it is not reusable and is never called by `release.yml`.
- CI runs the Python 3.11–3.14 pytest matrix, the Python 3.12 compiler leg, the `packaging` job, and `cargo test --locked --workspace` plus `cargo clippy --locked --workspace --all-targets --no-deps -- -D warnings` in `native/`.
- The `packaging` job runs `uv build --all-packages --out-dir dist`, `uvx twine check dist/*`, the shared boundary check `python3 .github/workflows/release_checks.py dist`, and the helper unit-test steps.
- The `packaging` job runs the release helper regression suite once on Python 3.12 with `uv run --no-project --python 3.12 python -m unittest discover -s .github/workflows -p 'test_release_checks.py'`.
- The same `packaging` job runs the version-bump helper suite twice, as two adjacent steps: `uv run --no-project --python 3.12 python -m unittest discover -s scripts -p 'test_*.py'` (step `Run version-bump helper unit tests (3.12)`) and `uv run --no-project --python 3.11 python -m unittest discover -s scripts -p 'test_*.py'` (step `Run version-bump helper unit tests (3.11 floor)`), which discover `scripts/test_bump_version.py` from the repository root.
- The 3.11 leg is the tool's documented stdlib-only floor, so stdlib API newer than 3.11 (e.g. a 3.13-only keyword argument) fails the gate instead of surfacing at release time.
- `release_gate.yml` is reusable only (`workflow_call` with required string inputs `checkout_sha` and `version`, no push/PR/dispatch trigger) and runs ONE `ubuntu-latest` job `gate` on the commit the release workflow selected.
- The gate's steps in order: checkout of `checkout_sha`, `astral-sh/setup-uv@v6` on Python 3.11 with cache, `uv python install 3.11 3.14`, the release-helper unit tests on 3.12, `dtolnay/rust-toolchain@stable` plus `Swatinem/rust-cache@v2` (`workspaces: native -> target`), `uv build --python 3.11 --all-packages --out-dir dist`, `uvx twine check dist/*`, `python3 .github/workflows/release_checks.py dist`, the compiled-extension smoke on the abi3 floor (`--python 3.11`) and ceiling (`--python 3.14`), and the combined-distributions smoke on 3.12.
- `--python 3.11` makes the built `evoxc` wheel `cp311-abi3` (the release tag shape), so that single wheel installs on both the 3.11 floor and the 3.14 ceiling, and `uv build --all-packages` yields four artifacts (evoxc wheel + sdist, evoxc_hf wheel + sdist).
- The gate is deliberately quick (about five minutes) and runs NO pytest, NO `cargo test`/Clippy, NO compiler extras and NO sdist source build; `ci.yml` owns the PR/main gates and `wheels.yml` owns the authoritative seven-wheel + sdist build-and-smoke matrix.
- `release_checks.py dist` is the shared distribution-boundary check run by both the CI `packaging` job and the release gate: it validates the four built archives in `dist` (default directory, optional `sys.argv[2]` override) and each project ships only its own module.
- The verified `evoxc` wheel layout is top-level `evoxc/` and `evoxc-<version>.dist-info/` plus an OPTIONAL `evoxc.libs/` bundled shared-library directory that maturin's manylinux/musllinux repair creates (both musllinux wheels carry `evoxc.libs/libgcc_s-<hash>.so.1`, with the extension patched to an `$ORIGIN/../evoxc.libs` RPATH).
- `evoxc.libs/` is accepted because a bundled C runtime library is not a Python module and does not weaken the two-distribution boundary, and those wheels are smoke-tested inside their matching PyPA container on the target libc.
- Both the `wheels.yml` inline check and `release_checks.py::dist_check` require every `evoxc.libs/` entry to be a single file directly inside it whose basename ends in `.so`, `.so.<digits>`, `.dylib` or `.dll`; a bare `evoxc.libs/` directory entry is tolerated and any other entry fails closed with the wheel name and the offending entries in the message.
- The `evoxc_hf` wheel stays exactly `{evoxc_hf, evoxc_hf-<version>.dist-info}` with no bundled-library directory, because it is pure Python.
- `test_release_checks.py` unit-tests the helper (including `dist`) against synthetic four-artifact trees.
- `smoke_evoxc_hf.py` installs the local `evoxc` and `evoxc_hf` wheels together in one fresh venv, loads a tiny llama config, and runs `predict` on the numpy backend against `reference_logits`.
- `release.yml` triggers on a new `v*` tag push, no-input `workflow_dispatch`, and a published GitHub Release tagged `evoxc-hf-v*`.
- On an `evoxc` tag push, `verify` matches the peeled tag commit and tag object to the event and checkout, requires membership in `origin/main` history, and requires the exact tag `v<version>` from the root `pyproject.toml`.
- An `evoxc` tag runs `verify` → `gate` (the caller of `release_gate.yml`) → `wheels`; the same order is shared by the `main`-only rehearsal (`gh workflow run release.yml --ref main`), and rehearsals have no release-write or OIDC jobs.
- A `workflow_dispatch` rehearsal also runs `rehearse-stage` (needs `verify` + `wheels`, `contents: read` only), which downloads the pinned artifact by numeric ID with `merge-multiple: true` and runs `release_checks.py manifest` on the flat `dist/`.
- That rehearsal job exercises the same last-mile staging check `prepare` performs — flat `dist/`, exact expected wheel/sdist names, SHA-256s — before a tag push can fail on it.
- It performs no GitHub API call, no draft/release mutation and no upload, so a rehearsal stays read-only.
- `wheels.yml` is callable with `checkout_sha` and `version` and returns `artifact_id` for the single immutable `evoxc-dist` upload containing seven wheels and one sdist after platform smoke tests, matrix and archive verification, and strict Twine validation.
- Linux wheels run on `ubuntu-24.04` (x86-64) and `ubuntu-24.04-arm` (AArch64); on each architecture the manylinux2014/glibc 2.17 and musllinux 1.2 wheels are installed and tested inside their matching PyPA containers.
- `macos-15-intel` builds x86-64 with macOS 10.12 deployment target, `macos-15` builds Arm64 with macOS 11.0 deployment target, and `windows-2025` builds x86-64 MSVC; the smoke tests run natively on those hosts, not on old macOS runtime versions.
- The Maturin sdist runs a source-build/install smoke test on `ubuntu-24.04` with Rust available; every `wheels.yml` wheel and sdist smoke test uses CPython 3.11, a fresh venv outside the checkout, and `pip --isolated install --no-cache-dir` on the exact archive (`--only-binary=:all:` for wheel legs).
- The `smoke_evoxc.py` check verifies installed metadata, package versions and extension provenance, then traces a small graph, differentiates it through the Rust transform, executes it on the native backend AND on the default numpy backend (explicit staging), and round-trips the graph through `Graph.save`/`Graph.load` with a digest match.
- The release gate runs that same wheel smoke on both the abi3 floor (3.11) and ceiling (3.14), plus `smoke_evoxc_hf.py` on 3.12.
- In `EMI-Group/evoxc`, successful gates enable `prepare` (stage and verify a GitHub draft), `publish-evoxc` (publish to PyPI), then `finalize` (make that exact checked draft public).
- The separate HF path validates `evoxc-hf-v<version>` against the member `pyproject.toml` at the release event commit and runs `verify` → `build-evoxc-hf` → `publish-evoxc-hf` → `attach-evoxc-hf` using only the separately built `evoxc-hf-dist` wheel and sdist.

## Constraints

- CI uses `uv sync --locked --all-packages` and runs pytest with the same lock and workspace flags so the member remains installed and dependencies do not re-resolve.
- CI's `uv build` writes `dist/.gitignore`, which is excluded from archive counts and publication; each project must ship only its own module.
- The version-bump helper (`scripts/bump_version.py`) and its tests are stdlib-only and run in CI's `packaging` job, so that gate needs no project environment.
- The release helper (`release_checks.py`) and its tests stay stdlib-only in the gate too, so the release gate needs a bare `python3` for the check and a `uv`-managed interpreter only for its unit tests.
- The workflows never create or move a tag, change a version, or publish from manual-dispatch input; the two project `pyproject.toml` files are authoritative for their respective versions.
- GitHub Release writes are job-scoped, and only `publish-evoxc` and `publish-evoxc-hf` have `id-token: write` and target the `pypi` environment; both PyPI trusted publishers bind to `release.yml` in `EMI-Group/evoxc`.
- Per-tag concurrency prevents release runs from overlapping; maintainers should protect release tags against mutation and configure `pypi` deployment rules for the approved tags only.
- Publishers download the tested artifact by numeric ID, recheck the live tag, and never use skip-existing or overwrite semantics; the evoxc publisher rechecks the draft and SHA-256 assets after environment approval before upload.
- Every `actions/download-artifact@v4` step that writes into `dist/` keeps that directory FLAT: the pinned artifact is downloaded by numeric ID with `merge-multiple: true`, while a step that uses `name:` downloads one named artifact that the action extracts flat.
- `release_checks.py::manifest` treats `dist/` as a flat directory and fails closed naming the offending entries when it holds a directory such as `dist/evoxc-dist/`, because `manifest`, `stage_assets` and the Twine check all read `dist/<file>` directly.
- `test_release_checks.py` pins that contract: one test asserts the nested-directory `manifest` diagnostic, and a static workflow-wiring test requires every `dist/` download step in `.github/workflows/*.yml` to declare `name:` or `merge-multiple: true`.
- Recovery reruns failed jobs in the original run (`gh run rerun <run-id> --failed`) to retain the original artifacts, not a fresh build; an exact PyPI match skips upload, and an exact already-public GitHub release is accepted by finalization without mutation.
- A partial PyPI release blocks automated upload; after comparing hashes, a maintainer can restore only missing files from the original artifact linked in the run summary, then rerun failed jobs.
- A hash mismatch, conflicting draft, or expired original artifact requires investigation, not overwrites or rebuilding a partially published version; artifacts are retained for seven days.

## Routing Table

| Path | Area |
|---|---|
| `./ci.yml` | PR/main gate: Python pytest matrix, compiler leg, Rust tests + Clippy, and packaging. |
| `./release_gate.yml` | Reusable quick end-to-end release gate (build, boundary, abi3-floor/ceiling and combined-distribution smokes). |
| `./wheels.yml` | Reusable gate for the supported evoxc wheel matrix, source archive and exact installs. |
| `./smoke_evoxc.py` | Fresh-venv install and compiled-extension graph smoke for one exact archive. |
| `./smoke_evoxc_hf.py` | Two-distributions install and forward smoke (`evoxc` + `evoxc_hf` in one venv). |
| `./release.yml` | Tag preflight, rehearsal, artifact paths, GitHub draft/publication, and PyPI OIDC jobs. |
| `./release_checks.py` / `./test_release_checks.py` | Source, artifact (`dist` boundary) and remote-state validation; run `python -m unittest discover -s .github/workflows -p 'test_release_checks.py'` from the repository root. |
| `../../RELEASING.md` | Human release procedure (sibling — read-only, escalate writes to parent). |
