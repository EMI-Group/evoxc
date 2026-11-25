# scripts — developer environment, native build and the release bump helper
## Intent
Developer-facing tooling for building and releasing this repository, never imported by the `evoxc` package.
`bump_version.py` exists because `.github/workflows/release_checks.py verify` requires the release tag to be exactly `v` + the root `[project].version`: the failed `v0.2.0` release was a tag pushed while `pyproject.toml` still said `0.1.0`.
The tool makes "bump the declared version everywhere + commit exactly those files + annotated tag + atomic push" one safe, explicit operation instead of a manual multi-file edit.
## API Surface
`scripts/bump_version.py` — stdlib-only Python 3.11+ CLI, run as `python scripts/bump_version.py [VERSION] [options]` from anywhere inside the git repository (the root comes from `git rev-parse --show-toplevel`).
`VERSION` accepts `0.2.0`, `v0.2.0` or `V0.2.0` (`X.Y.Z` plus an optional PEP 440-ish suffix such as `-rc1`, `.post1`, `+local`); it is prompted for when omitted, and a missing value on a non-TTY stdin is an error.
Flags: `--dry-run` (print the plan and unified diffs, write nothing), `--yes`/`-y` (answer yes to every confirmation, required for CI), `--no-commit`, `--no-tag`, `--no-push`, `--no-cargo`, `--no-changelog`, `--allow-branch`, `--remote NAME` (default `origin`), `--message TEXT` (default `chore(release): evoxc <version>`).
Files it bumps: `pyproject.toml` (`[project].version` only), `native/Cargo.toml` (`[workspace.package].version` plus the `version` of the five internal `[workspace.dependencies]` entries `evoxc-core`/`evoxc-compiler`/`evoxc-runtime`/`evoxc-cpu`/`evoxc`), `native/Cargo.lock` (the six workspace-member `[[package]]` blocks), `uv.lock` (the `[[package]]` block `name = "evoxc"` only) and `CHANGELOG.md` (release `## [Unreleased]` as `## [X.Y.Z] - <today>`, open a fresh empty `## [Unreleased]` above it, and add the `[X.Y.Z]: .../releases/tag/vX.Y.Z` link reference in descending order).
Files it deliberately never touches: `evoxc_hf/pyproject.toml` and the `evoxc-hf` `uv.lock` block (that distribution is versioned and released independently), `RELEASING.md`, any `CONTEXT.md`, the README and every doc, and every byte of the bumped files outside the listed values.
`CHANGELOG.md` patching is idempotent: an existing version heading or link reference is never duplicated, so re-running the same bump changes nothing.
`scripts/test_bump_version.py` — stdlib `unittest` suite for the pure patch functions and the full flow, run with `python -m unittest discover -s scripts -p 'test_*.py'` from the repository root (wired into CI with that exact command).
## Constraints
Standard library only (`argparse`, `difflib`, `re`, `shlex`, `subprocess`, `sys`, `tomllib`, `dataclasses`, `datetime`, `pathlib`); no uv, cargo, ruff or pytest dependency at runtime.
Never moves, reuses or deletes a tag: a bump onto an existing `v<version>` fails up front, before a single byte is written.
Commits only its own files — `git add -- <paths>` then `git commit -m <msg> -- <paths>` (a path-limited commit builds a temporary index from the listed paths) — and verifies the result against `git show --name-only --format= HEAD`, so an unrelated staged or dirty file can never ride along.
Leaves the working tree otherwise untouched: every other file keeps its bytes, and `--no-commit` / a declined commit leaves the modified files in place for inspection.
File reads go through the `_read_text` helper (`path.open("r", encoding="utf-8", newline="")`) and writes through `Path.write_text(..., encoding="utf-8", newline="")`, so a CRLF checkout is not mass-rewritten.
`Path.read_text(newline=...)` is Python 3.13+ while this script and its CI suite run on the 3.11 floor / the CI 3.12 leg, so no read may use it, and `./test_bump_version.py` guards that statically (plus a test that the helper does not translate `\r\n`).
Lock files are patched in place, so `uv sync --locked` and `cargo --locked` stay green without re-locking.
Push is `git push --atomic <remote> <branch>:refs/heads/<branch> refs/tags/v<version>` (branch-only with `--no-tag`); it never force-pushes, and git's stderr is surfaced verbatim on failure.
Non-interactive stdin without `--yes` fails closed before making changes, and the non-`main` branch guard warns that the release gate requires the tagged commit on `origin/main` (default answer NO; `--allow-branch` skips it).
`--dry-run` and `--no-push` are the safe inspection paths, and every failure returns a non-zero exit code with an actionable message.
## Routing Table
| Path | Area |
|---|---|
| `./bump_version.py` | the release-version bump CLI: pure patch helpers (`normalize_version`, `patch_pyproject`, `patch_cargo_manifest`, `patch_cargo_lock`, `patch_uv_lock`, `patch_changelog`) plus the thin `main` that plans, confirms, writes, commits, tags and pushes. |
| `./test_bump_version.py` | its stdlib `unittest` suite: fixture-based patch tests and end-to-end runs in throwaway `tempfile` git repositories (never the real checkout), plus the Python-floor guard that no `read_text` call passes `newline` (Python 3.13+) and that the `_read_text` helper preserves `\r\n`. |
| `./build-native.sh` | builds the Rust/PyO3 extension for the local dev environment. |
| `./dev-env.sh` | sets up the NixOS/uv/`CARGO_HOME` development environment. |
