# Releasing `evoxc` and `evoxc-hf`

| Distribution | Version source | Release trigger in `EMI-Group/evoxc` |
| --- | --- | --- |
| `evoxc` | Root `pyproject.toml`, `[project].version` | Push a new `v<version>` tag on a commit already in `origin/main` history. Do not create a GitHub Release yourself. |
| `evoxc-hf` | `evoxc_hf/pyproject.toml`, `[project].version` | Publish a GitHub Release for an existing `evoxc-hf-v<version>` tag. Pushing the tag alone does not publish. |

The versions are independent. Release `evoxc` first if a new `evoxc-hf` requires an
`evoxc` version not yet on PyPI. The workflows never bump versions or create or
move tags. Configuring publishing and running a rehearsal do not publish anything.

## One-time setup

1. Keep `main` protected with code review and required CI checks. Protect the
   `v*` and `evoxc-hf-v*` release tags with GitHub rulesets: restrict creation to
   release maintainers and prevent updates and deletions. The workflow uses
   per-tag concurrency and rechecks the live tag before remote release operations;
   those checks are not a substitute for tag protection.
2. In GitHub **Settings → Environments**, create `pypi` and limit its deployment
   refs to the `v*` and `evoxc-hf-v*` tags, not branches. Configure required
   reviewers and prevent self-review where available. Approve only the intended
   release tag and run after checking the gates and the artifact link in the run
   summary. Review the verified GitHub draft for `evoxc` before approving.
3. Configure a separate [PyPI trusted publisher](https://pypi.org/manage/account/publishing/)
   for each project. Use a pending publisher if the project does not exist yet;
   otherwise configure it on the existing project.

   | PyPI project | GitHub owner | Repository | Workflow filename | Environment |
   | --- | --- | --- | --- | --- |
   | `evoxc` | `EMI-Group` | `evoxc` | `release.yml` | `pypi` |
   | `evoxc-hf` | `EMI-Group` | `evoxc` | `release.yml` | `pypi` |

Only the two PyPI publish jobs have `id-token: write` and use `pypi`. They obtain
short-lived OIDC credentials; no PyPI API token is needed in Actions secrets.
GitHub Release writes are job-scoped. The publishing paths are restricted to
`EMI-Group/evoxc`; a fork can rehearse but cannot publish through this workflow.

Make the release-triggering tag push (or the HF Release publication) as an
authenticated maintainer, not from another workflow using its default
`GITHUB_TOKEN`: GitHub normally suppresses new workflow runs from events created
by that token. The `evoxc` finalizer uses the workflow token to make its draft
public and does not rely on a second run; only `evoxc-hf-v*` is accepted on the
`release: published` path in any case.

## Rehearse without publishing

After the changes to rehearse are on `main`, run this from the repository with an
authenticated GitHub CLI (or choose **Release → Run workflow**, selecting `main`):

```bash
gh workflow run release.yml --ref main
```

There are no inputs and no release-write or OIDC jobs in a rehearsal. It pins the
selected `main` commit and follows the same `verify → ci → wheels` gates as a real
`evoxc` release. `wheels.yml` is a reusable workflow, not a separate dispatch or
PR trigger. `ci.yml` runs Python 3.11–3.14 tests on Ubuntu, the Python 3.12 compiler
leg, the two-project packaging check, and locked Rust tests and Clippy. It also
runs normally on PRs and pushes to `main`.

For local checks from the checkout (use an empty output directory):

```bash
uv sync --locked --all-packages --extra dev
uv run --locked --all-packages --extra dev pytest -q
uv build --all-packages --out-dir dist
uvx twine check dist/*
```

These commands require a Rust toolchain and a C compiler on `PATH` to build
`evoxc`. The local build produces only a wheel for the current platform for
`evoxc`, not the full release matrix.

## Release `evoxc`

1. In a reviewed change, set `[project].version` in the root `pyproject.toml`,
   update `CHANGELOG.md`, and refresh `uv.lock` with `uv lock`. Merge to `main`.
   Rehearse the candidate before tagging.
2. Fetch `origin/main` and tag the intended merged commit, not an unmerged PR
   commit or an unchecked local `HEAD`. For example, for root version `0.1.1`,
   substitute the verified merge SHA for `RELEASE_SHA`:

   ```bash
   git fetch origin &&
     git merge-base --is-ancestor RELEASE_SHA origin/main &&
     git tag -a v0.1.1 RELEASE_SHA -m 'evoxc 0.1.1' &&
     git push origin refs/tags/v0.1.1
   ```

   The commit need not still be the tip of `main`. The tag must be exactly `v`
   plus the version declared **at that commit**. The source gate checks the tag
   object and peeled commit against the event and checkout, checks ancestry, and
   passes the pinned SHA to CI and the wheel build. An update to an existing tag
   is not a release trigger. Do not create a draft or public release manually for
   this tag.
3. Watch **Release** in Actions. After `verify → ci → wheels`, `prepare` stages a
   GitHub draft with the tested archives and checks their SHA-256 digests. A
   matching draft from an earlier attempt can be resumed; a conflicting one is
   rejected. `publish-evoxc` waits for any `pypi` approval, then rechecks the live
   tag, the draft, its assets and PyPI before uploading. `finalize` confirms PyPI
   has the exact tested files and that the same draft and assets still match
   before making it public.
4. Check the version-specific [PyPI files](https://pypi.org/project/evoxc/#files)
   and the public GitHub Release for the seven wheels and one sdist below. Try a
   version-pinned `python -m pip install evoxc==0.1.1` in a fresh environment
   (use the actual version).

### Tested distribution artifacts

The `evoxc` release gate collects **seven** `evoxc-<version>-cp311-abi3-<platform>.whl`
files and **one** Maturin `evoxc-<version>.tar.gz` sdist into the immutable
`evoxc-dist` artifact. The wheel platform tags and smoke environments are:

| Platform tag | GitHub runner | Smoke environment |
| --- | --- | --- |
| `manylinux_2_17_x86_64` (`manylinux2014_x86_64`) | `ubuntu-24.04` | x86-64 PyPA manylinux2014 container (glibc 2.17) |
| `musllinux_1_2_x86_64` | `ubuntu-24.04` | x86-64 PyPA musllinux 1.2 container |
| `manylinux_2_17_aarch64` (`manylinux2014_aarch64`) | `ubuntu-24.04-arm` | AArch64 PyPA manylinux2014 container (glibc 2.17) |
| `musllinux_1_2_aarch64` | `ubuntu-24.04-arm` | AArch64 PyPA musllinux 1.2 container |
| `macosx_10_12_x86_64` | `macos-15-intel` | Native macOS 15, Intel; deployment target 10.12 |
| `macosx_11_0_arm64` | `macos-15` | Native macOS 15, Arm64; deployment target 11.0 |
| `win_amd64` | `windows-2025` | Native Windows Server 2025, x86-64/MSVC |

The exact sdist is built and installed from source on x86-64 `ubuntu-24.04` with
Rust available. All eight smoke tests use ordinary **CPython 3.11**, a fresh venv
outside the checkout and `pip --isolated install --no-cache-dir` on the exact
archive (`--only-binary=:all:` on wheel legs). They verify the installed versions
and extension provenance, trace a graph, differentiate it in Rust, and execute
it on the native backend. Collection verifies the exact matrix and archive
contents and runs `twine check --strict`. Each wheel carries `evoxc` and the
native extension, not `evoxc_hf`; the sdist carries the whole Rust workspace,
including the facade crate, but no compiled extension and no `evoxc_hf`.

`cp311-abi3` uses the CPython stable ABI with a 3.11 floor: one wheel per target
is intended for compatible, standard CPython versions, not one build per Python
minor. The 3.11–3.14 Ubuntu CI matrix is separate from the exact-archive tests;
it does not imply wheel tests on every Python/OS combination. macOS deployment
tags are not tests on older macOS versions. There are no Windows Arm64 or
free-threaded wheels, and no compatibility claim for untested interpreters or OS
versions. Optional compiler adapters have their own platform requirements.

### Source fallback

If pip cannot find a compatible wheel, it can build from the sdist automatically.
To explicitly build `evoxc` from the PyPI sdist, install a Rust toolchain and a C
compiler/linker for your platform, then run:

```bash
python -m pip install --no-binary=evoxc evoxc
```

This leaves wheels enabled for dependencies. From a checkout, `python -m pip install .`
builds the same extension with Maturin. To install a downloaded archive, pass
`evoxc-<version>.tar.gz` to pip; use the tested release asset rather than
GitHub's autogenerated source archives. A source build is a fallback, not a
support guarantee for untested targets.

## Release `evoxc-hf` separately

Set the version in `evoxc_hf/pyproject.toml`, update `CHANGELOG.md`, refresh
`uv.lock`, and merge the change. Create and push `evoxc-hf-v<version>` for that
commit, then publish a GitHub Release for that existing tag as a maintainer (for
example, `gh release create evoxc-hf-v0.1.1 --verify-tag --generate-notes`). Use
the member's actual version. A draft or a tag push alone does not trigger the HF
publication path.

The workflow checks the published release event's tag and commit against the
member's version, runs `uv build --package evoxc-hf` and checks the artifacts
and metadata, publishes through its own PyPI trusted publisher, and attaches
only `evoxc_hf-<version>-py3-none-any.whl` and `evoxc_hf-<version>.tar.gz` to
that same release. This does not run the `evoxc` wheel matrix or release `evoxc`.
Afterward, check the two files on PyPI and GitHub and try a version-pinned
`python -m pip install evoxc-hf==0.1.1` in a fresh environment.

## Recover without rebuilding or overwriting

Use the **original release run**. The summary links the immutable artifact by
numeric ID (`evoxc-dist` or `evoxc-hf-dist`, retained seven days). For a retryable
failure, use **Re-run failed jobs**, not **Re-run all jobs**, a new run or a new tag:

```bash
gh run rerun RUN_ID --failed
```

- **Source, CI or build gate failed:** `evoxc` has not reached draft or publish.
  Retry transient failures in the same run. If a committed fix is needed, rehearse
  it on `main` and use a new valid version/tag; do not move the old tag.
- **Draft staging failed:** retry with the same tested artifact. Staging fills in
  only missing assets on the matching draft and verifies the complete set by name
  and SHA-256; it never replaces an asset. A conflicting draft, release or asset
  needs maintainer investigation, not `--clobber` or manual publication.
- **PyPI upload failed or its response was lost:** preflight uploads only if the
  version is absent. If PyPI already has the exact filenames and SHA-256 digests,
  the retry skips uploading. A partial release blocks automatic upload. Download
  the original artifact by the ID linked in the run summary, hash its individual
  archives (e.g. `sha256sum` on Linux), and compare to the `filename` and
  `digests.sha256` entries in `https://pypi.org/pypi/<project>/<version>/json`.
  Only if every existing file matches and none is unexpected or yanked may an
  authorized PyPI maintainer upload **only the missing archives** from that
  artifact, for example with Twine and maintainer credentials, naming each
  missing file explicitly. Recheck that the full name/hash set now matches, then
  rerun the failed jobs. Do not use skip-existing, rebuild the artifacts, or try
  to replace a PyPI file.
- **GitHub finalization failed after PyPI succeeded:** rerun the failed jobs in
  the original run. Finalization uses the same artifact ID, verifies PyPI and
  the exact release ID, marker and asset hashes, and accepts an already-public
  exact match without changing it. It does not upload to PyPI. A failed HF
  attachment likewise rechecks PyPI and attaches only missing matching files.
- **Hashes differ, release/tag conflicts, or the original artifact has expired:**
  stop and investigate rather than overwriting anything or rebuilding a partly
  published version. If a published release must be withdrawn, yank the affected
  version on PyPI and cut a corrected release with a new version and tag.

## Publish Rust crates independently

GitHub release automation never publishes to crates.io. The Rust workspace uses
`[workspace.package].version` in `native/Cargo.toml`, independent of either Python
version. For a Rust release, bump that version, keep internal path-dependency
version requirements in sync, and update `native/Cargo.lock` in the release change.

Publish in order, waiting for each dependency to be visible on crates.io before
checking and publishing its dependents:

1. `evoxc-core`
2. `evoxc-compiler`
3. `evoxc-runtime`
4. `evoxc-cpu`
5. `evoxc` (the reusable facade)

`evoxc-python` is `publish = false`; it is the Python extension crate, not a
crates.io release. From the Cargo workspace (`native/`), gate the release and
check each crate with a dry run:

```bash
cargo test --locked --workspace
cargo clippy --locked --workspace --all-targets --no-deps -- -D warnings
cargo publish --dry-run -p evoxc-core
cargo publish -p evoxc-core
```

The last command is a real upload and requires crates.io credentials. Repeat
just the dry run and publish commands for each subsequent crate in order.
