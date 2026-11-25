#!/usr/bin/env python3
"""Bump the ``evoxc`` version everywhere it is declared, then commit, tag and push.

``.github/workflows/release_checks.py verify`` requires the release tag to be
exactly ``v`` + the root ``[project].version``, so a tag pushed while a
declaration still says the old version fails the release (that is how the
``v0.2.0`` release failed). This tool makes "bump + commit + tag + push" one
explicit operation::

    python scripts/bump_version.py 0.2.0 --yes

It edits exactly these files, and nothing else:

* ``pyproject.toml``: ``[project].version``.
* ``native/Cargo.toml``: ``[workspace.package].version`` plus the five internal
  ``[workspace.dependencies]`` versions.
* ``native/Cargo.lock``: the six workspace-member ``[[package]]`` blocks.
* ``uv.lock``: the ``evoxc`` block only (``evoxc-hf`` releases independently).
* ``CHANGELOG.md``: release ``## [Unreleased]``, open a fresh one, add the link.

Every byte outside those values is preserved (``newline=""`` keeps a CRLF
checkout intact), an existing tag is never moved, the commit holds exactly the
bumped files (verified against ``git show --name-only``), the branch and tag go
out in one ``--atomic`` push that is never forced, and non-interactive stdin
without ``--yes`` fails closed before anything is written. The patchers are pure
``(text, version) -> (text, edits)`` functions; ``main`` only sequences them.
"""

from __future__ import annotations

import argparse
import difflib
import re
import shlex
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Sequence

#: The five files a bump may rewrite, repo-root-relative.
PYPROJECT = Path("pyproject.toml")
CARGO_MANIFEST = Path("native/Cargo.toml")
CARGO_LOCK = Path("native/Cargo.lock")
UV_LOCK = Path("uv.lock")
CHANGELOG = Path("CHANGELOG.md")

#: The six cargo workspace members whose locked version follows the workspace.
CARGO_PACKAGES: tuple[str, ...] = (
    "evoxc", "evoxc-core", "evoxc-compiler", "evoxc-runtime", "evoxc-cpu", "evoxc-python")
#: The five internal ``[workspace.dependencies]`` entries that carry a version.
CARGO_DEPENDENCIES: tuple[str, ...] = (
    "evoxc-core", "evoxc-compiler", "evoxc-runtime", "evoxc-cpu", "evoxc")
#: Release-tag URL prefix used by the CHANGELOG link references.
CHANGELOG_RELEASES = "https://github.com/EMI-Group/evoxc/releases/tag"

#: ``X.Y.Z`` with an optional PEP 440-ish suffix (``-rc1``, ``.post1``, ``+local``).
_VERSION_RE = re.compile(
    r"^(?P<release>\d+\.\d+\.\d+)"
    r"(?P<suffix>(?:[-_.+]?[0-9A-Za-z][0-9A-Za-z.+!_-]*)?)$"
)
#: Any TOML table header, single (``[t]``) or array-of-tables (``[[t]]``).
_TABLE_HEADER_RE = re.compile(r"^\[\[?[^\[\]]+\]\]?$")
#: A CHANGELOG reference definition that names a version (``[0.2.0]: url``).
_VERSION_REF_RE = re.compile(r"^\[[0-9][^\]]*\]:\s*\S")
#: The CHANGELOG's unreleased heading.
_UNRELEASED_RE = re.compile(r"^##\s*\[Unreleased\]\s*$")
#: Any level-2 CHANGELOG heading (a release or a keep-a-changelog style section).
_SECTION_RE = re.compile(r"^##\s")


class BumpError(ValueError):
    """A user-facing failure: ``main`` prints ``error: ...`` and returns 1.

    Subclasses ``ValueError``, so a malformed version is also a plain value error.
    """


@dataclass(frozen=True)
class Edit:
    """One changed value: what the plan prints as ``key: old -> new``."""

    key: str
    old: str
    new: str


@dataclass
class FilePatch:
    """A file's original and patched text plus the edits that produced it."""

    path: Path
    original: str
    updated: str
    edits: list[Edit]

    @property
    def changed(self) -> bool:
        """True when the patch actually rewrites bytes."""
        return self.original != self.updated


def _body(line: str) -> str:
    """The line without its ``\\r\\n`` / ``\\n`` / ``\\r`` terminator."""
    return line.rstrip("\r\n")


def _eol(line: str) -> str:
    """The line's exact terminator (``""`` for an unterminated last line)."""
    return line[len(_body(line)) :]


def _set_key_value(body: str, key_pattern: str, value: str) -> tuple[str, str | None]:
    """Replace the quoted value of ``key_pattern = "..."`` inside one line body.

    Returns ``(new_body, old_value)``, with ``None`` when the line has no such
    key. Only the text between the quotes changes.
    """
    match = re.match(
        rf'^(?P<lead>\s*{key_pattern}\s*=\s*")(?P<value>[^"]*)(?P<tail>".*)$', body
    )
    if match is None:
        return body, None
    return f'{match.group("lead")}{value}{match.group("tail")}', match.group("value")


def _find_table(bodies: Sequence[str], header: str, *, label: str) -> tuple[int, int]:
    """The half-open line range of a TOML table (header and next header excluded)."""
    for start, body in enumerate(bodies):
        if body.strip() != header:
            continue
        for index in range(start + 1, len(bodies)):
            if _TABLE_HEADER_RE.match(bodies[index].strip()):
                return start, index
        return start, len(bodies)
    raise BumpError(f"{label}: no {header} table found")


def _patch_lock_packages(
    text: str, version: str, names: Sequence[str], *, label: str
) -> tuple[str, list[Edit]]:
    """Patch the ``version`` of only the named ``[[package]]`` blocks.

    Other blocks (external crates, ``evoxc-hf``, numpy) stay byte-identical; an
    expected member that is absent is an explicit error.
    """
    lines = text.splitlines(keepends=True)
    bodies = [_body(line) for line in lines]
    expected = set(names)
    found: set[str] = set()
    edits: list[Edit] = []

    index = 0
    while index < len(bodies):
        if bodies[index].strip() != "[[package]]":
            index += 1
            continue
        end = len(bodies)
        for probe in range(index + 1, len(bodies)):
            if _TABLE_HEADER_RE.match(bodies[probe].strip()):
                end = probe
                break

        name, name_index = None, -1
        for probe in range(index + 1, end):
            match = re.match(r'^\s*name\s*=\s*"([^"]*)"\s*$', bodies[probe])
            if match:
                name, name_index = match.group(1), probe
                break

        if name in expected:
            found.add(name)
            for probe in range(name_index + 1, end):
                new_body, old = _set_key_value(bodies[probe], "version", version)
                if old is not None:
                    break
            else:
                raise BumpError(f"{label}: [[package]] {name} has no version field")
            if old != version:
                lines[probe] = new_body + _eol(lines[probe])
                bodies[probe] = new_body
                edits.append(Edit(f"[[package]] {name}.version", old, version))
        index = end

    missing = sorted(expected - found)
    if missing:
        raise BumpError(f"{label}: no [[package]] block for {', '.join(missing)}")
    return "".join(lines), edits


def _patch_table_version(
    lines: list[str],
    bodies: list[str],
    header: str,
    version: str,
    key: str,
    *,
    label: str,
) -> Edit | None:
    """Set the single ``version`` key of a TOML table in place; ``None`` if unchanged."""
    start, end = _find_table(bodies, header, label=label)
    for index in range(start + 1, end):
        new_body, old = _set_key_value(bodies[index], "version", version)
        if old is None:
            continue
        if old == version:
            return None
        lines[index] = new_body + _eol(lines[index])
        bodies[index] = new_body
        return Edit(key, old, version)
    raise BumpError(f'{label}: no `version = "..."` line in {header}')


def normalize_version(text: str) -> str:
    """The bare ``X.Y.Z[+suffix]`` form; a malformed version raises :class:`BumpError`.

    Accepts ``0.2.0``/``v0.2.0``/``V0.2.0`` with surrounding whitespace, requires
    three numeric components, and rejects embedded whitespace (``1.2``, ``v``,
    ``abc`` and ``1.2.3 4`` all fail loudly).
    """
    if not isinstance(text, str):
        raise BumpError(f"invalid version {text!r}: expected a string")
    candidate = text.strip()
    if not candidate:
        raise BumpError("invalid version: empty")
    if any(character.isspace() for character in candidate):
        raise BumpError(f"invalid version {text!r}: whitespace is not allowed")
    if candidate[:1] in ("v", "V"):
        candidate = candidate[1:]
    if _VERSION_RE.match(candidate) is None:
        raise BumpError(
            f"invalid version {text!r}: expected X.Y.Z with an optional PEP 440 "
            "suffix (e.g. 0.2.0, 1.0.0-rc1)"
        )
    return candidate


def tag_for(version: str) -> str:
    """The release tag name for a bare version."""
    return f"v{version}"


def patch_pyproject(text: str, version: str) -> tuple[str, list[Edit]]:
    """Set ``[project].version``; every other key (and section) is untouched."""
    lines = text.splitlines(keepends=True)
    bodies = [_body(line) for line in lines]
    edit = _patch_table_version(
        lines, bodies, "[project]", version, "[project].version", label=str(PYPROJECT)
    )
    return ("".join(lines), [edit]) if edit else (text, [])


def patch_cargo_manifest(text: str, version: str) -> tuple[str, list[Edit]]:
    """Set the workspace version and the five internal dependency versions.

    ``rust-version``, ``edition`` and external dependencies (``serde_json``,
    ``base64``, a versioned ``half`` entry, ...) are deliberately left alone.
    """
    lines = text.splitlines(keepends=True)
    bodies = [_body(line) for line in lines]
    edits: list[Edit] = []

    workspace_edit = _patch_table_version(
        lines, bodies, "[workspace.package]", version,
        "[workspace.package].version", label=str(CARGO_MANIFEST),
    )
    if workspace_edit:
        edits.append(workspace_edit)

    dependency_re = re.compile(
        r'^(?P<lead>\s*(?P<name>[A-Za-z0-9_-]+)\s*=\s*\{\s*version\s*=\s*")'
        r'(?P<value>[^"]*)(?P<tail>".*)$'
    )
    start, end = _find_table(
        bodies, "[workspace.dependencies]", label=str(CARGO_MANIFEST)
    )
    pending = set(CARGO_DEPENDENCIES)
    for index in range(start + 1, end):
        match = dependency_re.match(bodies[index])
        if match is None or match.group("name") not in pending:
            continue
        name = match.group("name")
        pending.discard(name)
        old = match.group("value")
        if old != version:
            new_body = f'{match.group("lead")}{version}{match.group("tail")}'
            lines[index] = new_body + _eol(lines[index])
            bodies[index] = new_body
            edits.append(Edit(f"[workspace.dependencies].{name}.version", old, version))
    if pending:
        raise BumpError(
            f"{CARGO_MANIFEST}: [workspace.dependencies] is missing "
            + ", ".join(sorted(pending))
        )
    return "".join(lines), edits


def patch_cargo_lock(text: str, version: str) -> tuple[str, list[Edit]]:
    """Set the version of the six workspace-member ``[[package]]`` blocks."""
    return _patch_lock_packages(text, version, CARGO_PACKAGES, label=str(CARGO_LOCK))


def patch_uv_lock(text: str, version: str) -> tuple[str, list[Edit]]:
    """Set the root ``evoxc`` project version; ``evoxc-hf`` keeps its own."""
    return _patch_lock_packages(text, version, ("evoxc",), label=str(UV_LOCK))


def _iso_day(today: date | str | None) -> str:
    """``YYYY-MM-DD`` for the CHANGELOG heading (``None`` means today)."""
    if today is None:
        return date.today().isoformat()
    if isinstance(today, date):
        return today.strftime("%Y-%m-%d")
    candidate = str(today).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", candidate) is None:
        raise BumpError(f"invalid date {today!r}: expected YYYY-MM-DD")
    return candidate


def patch_changelog(
    text: str, version: str, today: date | str | None = None
) -> tuple[str, list[Edit]]:
    """Release ``## [Unreleased]`` under ``version`` and add its link reference.

    The released section becomes ``## [X.Y.Z] - <today>`` with a fresh empty
    ``## [Unreleased]`` above it; re-applying the same version is a no-op.
    """
    day = _iso_day(today)
    heading = f"## [{version}] - {day}"
    lines = text.splitlines(keepends=True)
    bodies = [_body(line) for line in lines]
    edits: list[Edit] = []

    version_heading_re = re.compile(rf"^##\s*\[{re.escape(version)}\]")
    if not any(version_heading_re.match(body) for body in bodies):
        unreleased = next(
            (i for i, body in enumerate(bodies) if _UNRELEASED_RE.match(body)), None
        )
        if unreleased is not None:
            eol = _eol(lines[unreleased]) or "\n"
            old_heading = bodies[unreleased].strip()
            lines[unreleased] = heading + eol
            lines[unreleased:unreleased] = [f"## [Unreleased]{eol}", eol]
            edits.append(Edit("heading", old_heading, heading))
        else:
            first_section = next(
                (i for i, body in enumerate(bodies) if _SECTION_RE.match(body)), None
            )
            if first_section is not None:
                lines[first_section:first_section] = [heading + "\n", "\n"]
            else:
                if lines and not lines[-1].endswith(("\n", "\r")):
                    lines[-1] += "\n"
                if lines and bodies[-1].strip():
                    lines.append("\n")
                lines.append(heading + "\n")
            edits.append(Edit("heading", "(none)", heading))

    bodies = [_body(line) for line in lines]
    link = f"[{version}]: {CHANGELOG_RELEASES}/v{version}"
    version_link_re = re.compile(rf"^\[{re.escape(version)}\]:")
    if not any(version_link_re.match(body) for body in bodies):
        first_reference = next(
            (i for i, body in enumerate(bodies) if _VERSION_REF_RE.match(body)), None
        )
        if first_reference is not None:
            eol = _eol(lines[first_reference]) or "\n"
            lines[first_reference:first_reference] = [link + eol]
        else:
            if lines and not lines[-1].endswith(("\n", "\r")):
                lines[-1] += "\n"
            lines.append(link + "\n")
        edits.append(Edit("link reference", "(none)", link))

    return "".join(lines), edits


Patcher = Callable[[str, str], tuple[str, list[Edit]]]


def build_patches(
    root: Path,
    version: str,
    *,
    cargo: bool = True,
    changelog: bool = True,
    today: date | str | None = None,
) -> list[FilePatch]:
    """Compute every file's new text in memory; missing files are skipped loudly."""
    targets: list[tuple[Path, Patcher]] = [(PYPROJECT, patch_pyproject)]
    if cargo:
        targets.append((CARGO_MANIFEST, patch_cargo_manifest))
        targets.append((CARGO_LOCK, patch_cargo_lock))
    targets.append((UV_LOCK, patch_uv_lock))
    if changelog:
        targets.append(
            (CHANGELOG, lambda text, value: patch_changelog(text, value, today))
        )

    patches: list[FilePatch] = []
    for path, patcher in targets:
        file_path = root / path
        if not file_path.is_file():
            print(f"warning: {path} not found; skipping", file=sys.stderr)
            continue
        original = file_path.read_text(encoding="utf-8", newline="")
        updated, edits = patcher(original, version)
        patches.append(FilePatch(path, original, updated, edits))
    return patches


def write_patches(root: Path, patches: Sequence[FilePatch]) -> list[Path]:
    """Write the changed files with ``newline=""`` (bytes preserved outside values)."""
    written: list[Path] = []
    for patch in patches:
        if not patch.changed:
            continue
        (root / patch.path).write_text(patch.updated, encoding="utf-8", newline="")
        written.append(patch.path)
    return written


def read_project_version(root: Path) -> str:
    """The declared root ``[project].version``."""
    path = root / PYPROJECT
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise BumpError(f"{path} does not exist") from None
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise BumpError(f"{path} is not valid TOML: {exc}") from None
    value = data.get("project", {}).get("version")
    if not isinstance(value, str):
        raise BumpError(f"{path} has no [project].version")
    return value


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run git in ``root``; a non-zero status becomes a :class:`BumpError`."""
    command = ["git", *args]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise BumpError(f"`{shlex.join(command)}` failed ({result.returncode}):\n{detail}")
    return result


def repo_root(cwd: Path | None = None) -> Path:
    """Resolve the repository root with ``git rev-parse --show-toplevel``."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=cwd, capture_output=True, text=True
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "not a git repository"
        raise BumpError(
            "cannot find the repository root (`git rev-parse --show-toplevel` failed): "
            + detail
        )
    root = Path(result.stdout.strip())
    if not root.is_dir():
        raise BumpError(f"git reported a root that is not a directory: {root}")
    return root


def tag_exists(root: Path, tag: str) -> bool:
    """Whether ``refs/tags/<tag>`` exists locally (tags are never moved)."""
    result = git(root, "rev-parse", "-q", "--verify", f"refs/tags/{tag}", check=False)
    return result.returncode == 0


def current_branch(root: Path) -> str:
    """The short branch name, or ``""`` when HEAD is detached."""
    result = git(root, "symbolic-ref", "--short", "HEAD", check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def commit_paths(root: Path) -> list[str]:
    """The paths recorded in the HEAD commit (repo-root-relative, posix)."""
    result = git(root, "show", "--name-only", "--format=", "HEAD")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def stdin_isatty() -> bool:
    """True when stdin is an interactive terminal."""
    try:
        return bool(sys.stdin.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def ask(prompt: str, *, default: bool, assume_yes: bool) -> bool:
    """Ask a yes/no question; ``--yes`` answers yes without reading stdin."""
    if assume_yes:
        return True
    suffix = " [Y/n] " if default else " [y/N] "
    try:
        answer = input(prompt + suffix).strip().lower()
    except EOFError:
        raise BumpError(
            f"no answer on stdin for {prompt!r}; pass --yes for non-interactive use"
        ) from None
    if not answer:
        return default
    return answer in ("y", "yes")


def _unified_diff(patch: FilePatch) -> str:
    """A unified diff of one file patch, every line newline-terminated."""
    diff = difflib.unified_diff(
        patch.original.splitlines(keepends=True),
        patch.updated.splitlines(keepends=True),
        fromfile=f"a/{patch.path}",
        tofile=f"b/{patch.path}",
    )
    return "".join(line if line.endswith("\n") else line + "\n" for line in diff)


def build_parser() -> argparse.ArgumentParser:
    """The command line (the file rules live in the module docstring)."""
    parser = argparse.ArgumentParser(
        prog="bump_version.py",
        description="Bump the evoxc release version, then commit, tag and push.",
        epilog=("examples:\n  python scripts/bump_version.py 0.2.0 --dry-run\n"
                "  python scripts/bump_version.py 0.2.0 --yes\n"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("version", nargs="?", metavar="VERSION",
                        help="release version (0.2.0 or v0.2.0); prompted for when omitted")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and diffs; write, commit and push nothing")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="answer yes to every confirmation (needed when stdin is not a TTY)")
    parser.add_argument("--no-commit", action="store_true",
                        help="write the files but do not commit")
    parser.add_argument("--no-tag", action="store_true",
                        help="commit but do not create the git tag")
    parser.add_argument("--no-push", action="store_true",
                        help="commit (and tag) but never push")
    parser.add_argument("--no-cargo", action="store_true",
                        help="skip native/Cargo.toml and native/Cargo.lock")
    parser.add_argument("--no-changelog", action="store_true", help="skip CHANGELOG.md")
    parser.add_argument("--remote", default="origin", metavar="NAME",
                        help="git remote to push to")
    parser.add_argument("--message", default=None, metavar="TEXT",
                        help='commit message (default "chore(release): evoxc <version>")')
    parser.add_argument("--allow-branch", action="store_true",
                        help="skip the non-main branch safety prompt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bump; returns the process exit code (never calls ``sys.exit``)."""
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except BumpError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    """The documented flow: validate, plan, confirm, write, commit, tag, push."""
    root = repo_root()

    if args.version is None:
        if not stdin_isatty():
            raise BumpError(
                "VERSION is required when stdin is not a TTY; pass it explicitly, "
                "e.g. `python scripts/bump_version.py 0.2.0 --yes`"
            )
        args.version = input("Version to release (e.g. 0.2.0): ")
    version = normalize_version(args.version)
    tag = tag_for(version)
    message = args.message or f"chore(release): evoxc {version}"

    current = read_project_version(root)
    print(f"evoxc release bump {current} -> {version} (tag {tag})\nrepository: {root}")

    # Tags are immutable in this flow: refuse before writing anything.
    if not args.no_tag and tag_exists(root, tag):
        raise BumpError(
            f"tag {tag} already exists locally; tags are never moved. Delete it "
            f"deliberately (`git tag -d {tag}`) or pick a new version."
        )

    patches = build_patches(
        root, version, cargo=not args.no_cargo, changelog=not args.no_changelog
    )
    changed = [patch for patch in patches if patch.changed]

    if not changed:
        print(f"nothing to do: {PYPROJECT} already declares {version}; no commit or tag was made.")
        print(f'If that version is already committed, tag it manually:\n'
              f'  git tag -a {tag} -m "evoxc {version}"')
        if not args.no_push:
            print(f"  git push --atomic {args.remote} <branch>:refs/heads/<branch> "
                  f"refs/tags/{tag}")
        return 0

    for patch in changed:
        if not patch.edits:
            print(f"  {patch.path}: (content changed)")
        for edit in patch.edits:
            print(f"  {patch.path}: {edit.key}: {edit.old} -> {edit.new}")

    if args.dry_run:
        print()
        for patch in changed:
            sys.stdout.write(_unified_diff(patch))
        print("dry run: no files written, nothing committed, nothing pushed")
        return 0

    # Fail closed before any write: never guess a confirmation.
    if not args.yes and not stdin_isatty():
        raise BumpError(
            "stdin is not a TTY: pass --yes to confirm the commit/tag/push "
            "non-interactively (or --dry-run to only inspect the plan)"
        )

    branch = current_branch(root)
    if branch != "main" and not args.allow_branch:
        print(f"warning: HEAD is on {branch or '(detached HEAD)'}, not main.")
        print("         the release gate requires the tagged commit to be on origin/main.")
        if not ask(
            f"Continue on {branch or 'a detached HEAD'}?",
            default=False,
            assume_yes=args.yes,
        ):
            print("aborted: nothing was written.")
            return 1

    written = write_patches(root, changed)
    print("updated: " + ", ".join(str(path) for path in written))
    if args.no_commit:
        print("--no-commit: the modified files are left in the working tree.")
        return 0
    if not ask("Commit these changes?", default=True, assume_yes=args.yes):
        print("no commit made; the modified files are left in the working tree:\n  "
              + "\n  ".join(str(patch.path) for patch in changed))
        return 0

    # `git commit -- <paths>` records ONLY these paths (git builds a temporary
    # index from them), so an unrelated staged or dirty file cannot ride along.
    paths = [patch.path.as_posix() for patch in changed]
    git(root, "add", "--", *paths)
    git(root, "commit", "-m", message, "--", *paths)

    committed = commit_paths(root)
    if sorted(committed) != sorted(paths):
        raise BumpError(
            "the commit does not contain exactly the bumped files "
            f"(expected {sorted(paths)}, got {sorted(committed)}); "
            "inspect `git show --stat HEAD` before pushing."
        )
    sha = git(root, "rev-parse", "HEAD").stdout.strip()
    print(f"committed {sha[:12]} on {branch or 'HEAD'} ({len(paths)} file(s))")
    if args.no_tag:
        print("--no-tag: no tag created")
    else:
        git(root, "tag", "-a", tag, "-m", f"evoxc {version}")
        print(f"tagged {tag} (annotated, on {sha[:12]})")

    refspecs: list[str] = []
    if branch:
        refspecs.append(f"{branch}:refs/heads/{branch}")
    if not args.no_tag:
        refspecs.append(f"refs/tags/{tag}")
    push_command = ["git", "push", "--atomic", args.remote, *refspecs]
    if args.no_push:
        hint = ("Push manually with:\n  " + shlex.join(push_command)) if refspecs else (
            "detached HEAD and --no-tag: nothing to push.")
        print("--no-push: nothing pushed. " + hint)
        return 0

    if not branch:
        raise BumpError(
            "HEAD is detached: check out the branch to publish (`git switch main`) "
            "or use --no-push to keep the commit and tag local"
        )
    if not ask("Push the commit and tag?", default=True, assume_yes=args.yes):
        print("nothing pushed. Push when ready with:\n  " + shlex.join(push_command))
        return 0

    print("$ " + shlex.join(push_command))
    pushed = git(root, *push_command[1:], check=False)
    if pushed.returncode != 0:
        detail = pushed.stderr.strip() or pushed.stdout.strip()
        raise BumpError(
            f"`{shlex.join(push_command)}` failed ({pushed.returncode}):\n{detail}"
        )
    published = f"branch {branch}" + ("" if args.no_tag else f" and tag {tag}")
    print(f"pushed {published} to {args.remote}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
